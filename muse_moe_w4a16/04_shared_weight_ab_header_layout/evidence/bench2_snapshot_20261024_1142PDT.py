"""Routing-diverse paired benchmark: per shape, many fresh routings (normal and skewed popularity,
seeds disjoint from bench.py's), each timed as a CUDA graph replay with a cold L2, ours and
SGLang's in alternating order on the same inputs. Prints the judge-style median of paired ratios
and the same statistic per routing stratum (experts touched, busiest expert's tokens), and saves
every sample so judge results can be correlated with the offline numbers later.

    python3 dev/bench2.py dev/abl.py [--reps 96] [--shapes 1,8] [--skewed 0.25] [--out file.json]
    python3 dev/bench2.py - --variants "a|dev/moe_v92.py|DEFS;b|dev/moe_v95.py|"   # A/B builds on the same routings
"""
import os, argparse, json, math, os, statistics, sys, time, importlib, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, PROB); sys.path.insert(0, HERE)
import torch
import reference as R

SHAPES = [(1, "decode 1"), (16, "decode 16"), (32, "decode 32"), (8, "verify 1"), (128, "verify 16"), (256, "verify 32")]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Graphed:
    def __init__(self, fn, state, data):
        self.data = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}
        for _ in range(2):
            fn(self.data, state)
        torch.cuda.synchronize()
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            self.out = fn(self.data, state)
        torch.cuda.synchronize()


def stats_of(data):
    ids = data["topk_ids"]
    loc = ids[ids < 32].long()
    c = torch.bincount(loc, minlength=32)
    return int((c > 0).sum()), int(loc.numel()), int(c.max()) if loc.numel() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sub")
    ap.add_argument("--reps", type=int, default=96)
    ap.add_argument("--shapes", default="")
    ap.add_argument("--skewed", type=float, default=0.0, help="fraction of routings drawn with dist=skewed")
    ap.add_argument("--variants", default="", help="label|module file|MOE_DEFINES (comma list);...")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-base", action="store_true")
    args = ap.parse_args()
    weights = R.generate_weights(R.WEIGHTS_SEED)
    variants = [v.split("|") for v in args.variants.split(";") if v] or [["sub", args.sub, os.environ.get("MOE_DEFINES", "")]]
    subs = []
    for i, v in enumerate(variants):
        label, path, defs = v[:3]
        mod = load(path, "sub%d" % i)  # a fresh module instance per variant
        if len(v) > 3 and v[3]:
            for k, val in json.loads(v[3]).items():  # module globals, e.g. {"NS_MAX": 3}
                setattr(mod, k, [tuple(x) if isinstance(x, list) else x for x in val] if isinstance(val, list) and k == "BUCKETS" else val)
            if "BUCKETS" in json.loads(v[3]):
                mod.BUCKETS = [(b, tuple(c)) for b, c in json.loads(v[3])["BUCKETS"]]
        if hasattr(mod, "CUDA_SRC"):
            mod.CUDA_SRC = "".join(f"#define {d.replace('=', ' ')}\n" for d in defs.split(",") if d) + mod.CUDA_SRC
        st = mod.prepare(weights)
        if subs and os.environ.get("B2_SHARE_EQ"):
            # different versions: share every tensor equal to the first variant's (the weights),
            # keep the version's own entries otherwise
            shared = []
            for k, v in list(st.items()):
                v0 = subs[0][2].get(k)
                if torch.is_tensor(v) and torch.is_tensor(v0) and v.shape == v0.shape and v.dtype == v0.dtype \
                        and v.numel() > 1 << 20 and torch.equal(v, v0):
                    st["_own_" + k] = v  # kept alive: tensor maps may still point at it
                    st[k] = v0
                    shared.append(k)
            print(f"  {label}: shares {shared} with {subs[0][0]}", flush=True)
            torch.cuda.empty_cache()
        elif subs and not os.environ.get("B2_OWN_STATE"):
            # one weight placement for every variant (separate copies measured a few % apart)
            st = subs[0][2]
            torch.cuda.empty_cache()
        subs.append((label, mod, st))
    base = None if args.no_base else load(os.path.join(PROB, "baseline.py"), "base")
    base_state = base.prepare(weights) if base else None
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    ge = torch.cuda.CUDAGraph()
    tmp = torch.zeros(1, device="cuda")
    with torch.cuda.graph(ge):
        tmp.add_(1)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def time_replay(g):
        flush.sum()
        ev[0].record()
        g.replay()
        ev[1].record()
        torch.cuda.synchronize()
        return ev[0].elapsed_time(ev[1]) * 1000.0
    fixed = statistics.median([time_replay(ge) for _ in range(30)])
    want = {int(x) for x in args.shapes.split(",") if x}
    record = {"fixed": fixed, "shapes": {}}
    gm = {label: [] for label, _, _ in subs}
    extra = [(b, "bs %d" % b) for b in sorted(want) if b not in dict(SHAPES)]
    for bs, name in SHAPES + extra:
        if want and bs not in want:
            continue
        data0 = R.generate_input(bs, "ep8", 77)
        gs = [(label, Graphed(mod.kernel_fn, st, data0)) for label, mod, st in subs]
        gb = Graphed(base.kernel_fn, base_state, data0) if base else None
        rows = []
        for r in range(args.reps):
            dist = "skewed" if (args.skewed > 0 and (r * args.skewed) % 1.0 + args.skewed > 1.0 - 1e-9) else "normal"
            fresh = R.generate_input(bs, "ep8", 50000 + bs * 1000 + r, dist=dist)
            ne, npairs, mx = stats_of(fresh)
            for _, g in gs:
                for k in ("hidden_states", "topk_ids", "topk_weights"):
                    g.data[k].copy_(fresh[k])
            if gb:
                for k in ("hidden_states", "topk_ids", "topk_weights"):
                    gb.data[k].copy_(fresh[k])
            order = list(range(len(gs))) + ([-1] if gb else [])
            if r % 2:
                order.reverse()
            t = {}
            for i in order:
                g = gb.g if i < 0 else gs[i][1].g
                t[i] = max(time_replay(g) - fixed, 0.1)
            if r < 2 and not os.environ.get("B2_NOVERIFY"):
                for label, g in gs:
                    ok, msg = R.check_implementation(fresh, g.out, weights)
                    if not ok:
                        print(f"  {label} bs {bs} rep {r}: FAIL {msg}", flush=True)
                        sys.exit(1)
            rows.append({"r": r, "dist": dist, "experts": ne, "pairs": npairs, "max_tok": mx,
                         "base": t.get(-1), **{label: t[i] for i, (label, _) in enumerate(gs)}})
        record["shapes"][bs] = rows
        med = statistics.median
        # strata: experts touched (terciles) and the busiest expert's tokens (above / below median)
        es = sorted(r_["experts"] for r_ in rows)
        lo, hi = es[len(es) // 3], es[2 * len(es) // 3]
        mt = med([r_["max_tok"] for r_ in rows])
        strata = [("e<=%d" % lo, lambda r_: r_["experts"] <= lo), ("e>%d" % hi, lambda r_: r_["experts"] > hi),
                  ("hot>%d" % mt, lambda r_: r_["max_tok"] > mt), ("skew", lambda r_: r_["dist"] == "skewed")]
        if bs <= 8:  # few experts: by exact count
            strata = [("e=%d" % k, (lambda k: lambda r_: r_["experts"] == k)(k)) for k in sorted(set(es))]
        line = f"{name:10s} bs {bs:3d} experts {statistics.mean(es):5.2f} (|{lo}..{hi}|) max_tok med {mt}"
        print(line, flush=True)
        for label, _ in gs:
            ts = [r_[label] for r_ in rows]
            s = f"   {label:16s} {med(ts):7.1f} us"
            if gb:
                ratios = [r_["base"] / r_[label] for r_ in rows]
                sp = med(ratios)
                gm[label].append(sp)
                s += f"  x{sp:.3f}"
                for sname, fn in strata:
                    sub = [r_["base"] / r_[label] for r_ in rows if fn(r_)]
                    if sub:
                        s += f"  {sname}:x{med(sub):.3f}({len(sub)})"
            if len(gs) > 1:
                s += f"  vs {gs[0][0]}: x{med([r_[gs[0][0]] / r_[label] for r_ in rows]):.4f}"
            print(s, flush=True)
        if gb:
            print(f"   {'sglang':16s} {med([r_['base'] for r_ in rows]):7.1f} us", flush=True)
        del gs, gb
        torch.cuda.synchronize()
    for label in gm:
        if gm[label]:
            print(f"geomean {label}: x{math.exp(sum(map(math.log, gm[label])) / len(gm[label])):.4f} over {len(gm[label])} shapes", flush=True)
    if args.out:
        json.dump(record, open(args.out, "w"))


if __name__ == "__main__":
    main()
