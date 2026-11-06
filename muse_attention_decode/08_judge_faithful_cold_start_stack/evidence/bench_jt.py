"""W4 judge-faithful bench = tools/bench.py + the judge's PAIR structure and its per-pair cold-TLB first rep.
   Differences vs tools/bench.py (dev/w4/NOTES.md jobs w4_j1-j3): reps come in PAIRS on one fresh input set (rep 2i one
   order, rep 2i+1 reversed); --tlbcold 1 (default) frees the fresh-input allocations + torch.cuda.empty_cache() before
   each pair (GPU TLB invalidation: reproduces the judge's even-rep penalty); --ovh judge subtracts the judge's fixed
   cost estimate 2*one - two (two = 2 trivial replays in one window) instead of one empty-graph replay.
   Same CLI as tools/bench.py plus --tlbcold 0/1 and --ovh judge|fixed. --joff defaults to 0 here (judge offset built in).

Judge-style bench for muse_attention_decode: CUDA-graph replays, cold L2 (read flush), paired with SGLang.

    python3 tools/bench.py --cand v1=dev/v1.py [--cand v2=dev/v2.py] [--loads "decode 1,verify 32"] [--layers sliding]
                           [--reps 30] [--check-reps 2] [--json /tmp/b.json] [--no-base] [--joff 1.0]

Per shape: every variant owns a private copy of the inputs captured in its graph; each rep generates a fresh input
set (new seed: new lengths, cache, queries), copies it into every variant's buffers, then times every variant's replay
after a cold-L2 read flush, order reversed on odd reps. The fixed cost of an empty graph replay is subtracted.
Shape speedup = median over rep pairs of sqrt(prod base/cand) (the judge's paired ratio); shape time = SGLang median
/ speedup; load time = 46 x sliding + 16 x full; score = geomean over loads of SGLang load time / ours.
--joff adds a per-call judge offset to both sides when projecting (calibrate against an okc check).
"""
import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROB)
import torch  # noqa: E402

import reference as R  # noqa: E402

# (load name, batch, q_len); every load = sliding (w 46, dist soft) + full (w 16, dist coherent), ctx 80000, page 1.
LOADS = [("decode 1", 1, 1), ("decode 16", 16, 1), ("decode 32", 32, 1),
         ("verify 1", 1, 8), ("verify 16", 16, 8), ("verify 32", 32, 8)]
LAYERS = {"sliding": (46, "soft"), "full": (16, "coherent")}
CTX = 80000


def shape_kwargs(batch, q_len, layer):
    kw = dict(batch=batch, ctx=CTX, heads="tp8", layer=layer, q_len=q_len, page=1, dist=LAYERS[layer][1])
    if batch > 1:
        kw["spread"] = "agentic"
    return kw


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path if os.path.isabs(path) else os.path.join(PROB, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def clone_data(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.clone()
        elif k == "config":
            out[k] = dict(v)
        else:
            out[k] = v
    return out


def copy_into(dst, src):
    for k, v in src.items():
        if isinstance(v, torch.Tensor):
            if dst[k].shape != v.shape:
                raise RuntimeError(f"input {k} changed shape {tuple(dst[k].shape)} -> {tuple(v.shape)}")
            dst[k].copy_(v)


class Graphed:
    def __init__(self, fn, data, warm=3):
        self.data = clone_data(data)
        for _ in range(warm):
            fn(self.data)
        torch.cuda.synchronize()
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            self.out = fn(self.data)
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", action="append", default=[])
    ap.add_argument("--loads", default="")
    ap.add_argument("--layers", default="")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--check-reps", type=int, default=2)
    ap.add_argument("--json", default="")
    ap.add_argument("--no-base", action="store_true")
    ap.add_argument("--flush-mb", type=int, default=512)
    ap.add_argument("--flush-mode", default="read", help="read (sum) | write (zero_) | rw (add_)")
    ap.add_argument("--joff", type=float, default=0.0)
    ap.add_argument("--tlbcold", type=int, default=1)
    ap.add_argument("--ovh", default="judge")
    ap.add_argument("--jscale", type=float, default=1.05,
                    help="judge factor on OUR time for big shapes (SGLang > 100 us); calibrated on okc k8: ours +4-6%%, SGLang +-0.5%%")
    ap.add_argument("--seed0", type=int, default=100000)
    args = ap.parse_args()

    cands = []
    for c in args.cand:
        label, path = c.split("=", 1)
        path, _, defs = path.partition("@")
        os.environ["K2_DEFS"] = defs
        t0 = time.time()
        mod = load_mod(path, "cand_" + label)
        os.environ["K2_DEFS"] = ""
        print(f"[{label}] import {time.time() - t0:.1f}s", flush=True)
        cands.append((label, mod))
    base = None if args.no_base else load_mod("baseline.py", "base")

    flush = torch.empty(args.flush_mb * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    flush.normal_()
    ge = torch.cuda.CUDAGraph()
    tmp = torch.zeros(1, device="cuda")
    with torch.cuda.graph(ge):
        tmp.add_(1)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def do_flush():
        if args.flush_mode == "write":
            flush.zero_()
        elif args.flush_mode == "rw":
            flush.add_(1.0)
        else:
            flush.sum()

    def time_replay(g):
        do_flush()
        ev[0].record()
        g.replay()
        ev[1].record()
        torch.cuda.synchronize()
        return ev[0].elapsed_time(ev[1]) * 1000.0

    def time_two(g):
        do_flush()
        ev[0].record()
        g.replay()
        g.replay()
        ev[1].record()
        torch.cuda.synchronize()
        return ev[0].elapsed_time(ev[1]) * 1000.0

    for _ in range(20):
        time_replay(ge)
    one = statistics.median([time_replay(ge) for _ in range(50)])
    two = statistics.median([time_two(ge) for _ in range(50)])
    fixed = 2 * one - two if args.ovh == "judge" else one
    print(f"one {one:.2f} two {two:.2f} us -> subtracted fixed cost {fixed:.2f} us ({args.ovh}); tlbcold {args.tlbcold}", flush=True)

    want_loads = {x.strip() for x in args.loads.split(",") if x.strip()}
    want_layers = [x for x in args.layers.split(",") if x] or list(LAYERS)
    rec = {"fixed": fixed, "shapes": {}}
    loadt = {}
    for lname, batch, q_len in LOADS:
        if want_loads and lname not in want_loads:
            continue
        for layer in want_layers:
            weight = LAYERS[layer][0]
            kw = shape_kwargs(batch, q_len, layer)
            data0 = R.generate_input(seed=1234, **kw)
            gs = [(label, Graphed(mod.kernel_fn, data0)) for label, mod in cands]
            gb = Graphed(base.kernel_fn, data0) if base else None
            del data0
            times = {label: [] for label, _ in cands}
            times["base"] = []
            for r in range(args.reps):
                if r % 2 == 0:   # judge: one fresh input set per PAIR of reps
                    fresh = R.generate_input(seed=args.seed0 + 1000 * batch + 100 * q_len + r, **kw)
                    for _, g in gs:
                        copy_into(g.data, fresh)
                    if gb:
                        copy_into(gb.data, fresh)
                    del fresh
                    if args.tlbcold:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()   # frees the fresh set: cudaFree -> GPU TLB invalidation (judge-like)
                order = [(label, g.g) for label, g in gs] + ([("base", gb.g)] if gb else [])
                if r % 2:
                    order.reverse()
                for label, g in order:
                    times[label].append(max(time_replay(g) - fixed, 0.05))
                if r < args.check_reps:
                    for label, g in gs:
                        ok, msg = R.check_implementation(g.data, R.collect(g.data, g.out))
                        if not ok:
                            print(f"FAIL {label} {lname} {layer} rep {r}: {msg}", flush=True)
                            sys.exit(1)
                        if r == 0:
                            print(f"  check {label} {lname} {layer}: {msg}", flush=True)
            row = {}
            med_b = statistics.median(times["base"]) if gb else None
            line = f"{lname:9s} {layer:7s} w{weight:2d}"
            if gb:
                line += f" sglang {med_b:8.2f}"
            for label, _ in cands:
                t = times[label]
                if gb:
                    pr = [math.sqrt(times["base"][i] * times["base"][i + 1] / (t[i] * t[i + 1]))
                          for i in range(0, len(t) - 1, 2)]
                    sp = statistics.median(pr)
                    tt = med_b / sp
                    line += f" | {label} {tt:8.2f} x{sp:5.3f}"
                else:
                    tt, sp = statistics.median(t), None
                    line += f" | {label} {tt:8.2f} (min {min(t):.2f})"
                row[label] = {"t": tt, "sp": sp, "raw": t}
                loadt.setdefault(lname, {}).setdefault(label, 0.0)
                loadt[lname][label] += weight * tt
            if gb:
                loadt.setdefault(lname, {}).setdefault("base", 0.0)
                loadt[lname]["base"] += weight * med_b
                row["base"] = {"t": med_b, "raw": times["base"]}
            sol = R.speed_of_light_ns(hw={"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}, **kw) / 1e3
            line += f"   sol {sol:7.2f}"
            print(line, flush=True)
            rec["shapes"][f"{lname}/{layer}"] = row
            del gs, gb
            torch.cuda.empty_cache()
    rec["loads"] = loadt
    if not args.no_base and cands:
        print("\nper-load (weighted sums, us):")
        gm = {label: [] for label, _ in cands}
        gj = {label: [] for label, _ in cands}
        for lname, d in loadt.items():
            nsh = sum(LAYERS[l][0] for l in want_layers)
            s = f"  {lname:9s} sglang {d['base']:9.1f}"
            for label, _ in cands:
                gm[label].append(d["base"] / d[label])
                # judge projection: ours on big shapes x jscale, every call + joff (both sides)
                ours_j = sum(LAYERS[l][0] * (rec["shapes"][f"{lname}/{l}"][label]["t"]
                             * (args.jscale if rec["shapes"][f"{lname}/{l}"]["base"]["t"] > 100 else 1.0) + args.joff)
                             for l in want_layers if f"{lname}/{l}" in rec["shapes"])
                gj[label].append((d["base"] + args.joff * nsh) / ours_j)
                s += f" | {label} {d[label]:9.1f} x{d['base'] / d[label]:.3f}"
            print(s)
        # The board shows SGLang's per-step time averaged over loads the same way as the speedup: a geomean.
        mean_base = math.exp(statistics.mean(math.log(d["base"]) for d in loadt.values()))
        mean_base_j = math.exp(statistics.mean(math.log(d["base"] + args.joff * 62) for d in loadt.values()))
        for label, _ in cands:
            g = math.exp(sum(map(math.log, gm[label])) / len(gm[label]))
            j = math.exp(sum(map(math.log, gj[label])) / len(gj[label]))
            print(f"GEOMEAN {label}: x{g:.4f} (board time {mean_base / g:.1f} us)   "
                  f"judge-projected (+{args.joff} us/call) x{j:.4f} (board {mean_base_j / j:.1f} us)")
            rec.setdefault("geomean", {})[label] = g
            rec.setdefault("geomean_judge", {})[label] = j
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rec, f)


if __name__ == "__main__":
    main()
