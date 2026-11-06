"""Judge-faithful local bench for muse_decode_layer (run on the GPU box from muse_decode_layer/).

    python cdl/H/bench.py --cand d23=cdl/d23_base.py [--cand x=cdl/X/cand_x.py ...]
                          [--shapes all|d1s,v32f|d|v16] [--reps judge|N] [--xclones 3] [--pool 8]
                          [--router-seed S] [--json out.json] [--nocheck]

What it reproduces (see common.py / NOTES.md for the evidence):
  * weights = reference.generate_weights(5353, router_seed=S): S is a router seed matched to the judge's
    per-load MoE load (router_scan.py); the judge picks its own per run with the same property.
  * each shape captured once per side (SGLang = ../baseline.py, and every --cand) and per input placement
    (--xclones copies of the input buffers, each with its own graph; rep i uses placement (i // 2) % xclones);
  * every rep: a fresh input (pool of --pool inputs: the case seed and its weights_fields derivations, each
    with its own local_start) copied into the side's buffers, then flush (512 MB read) -> 300 us spin gate ->
    event -> replay -> event; all sides run on the same input, order reversed every other rep
    (rep 0: candidates first, as the judge's "first": "submission");
  * fixed replay cost ovh = 2*median(one) - median(two) from graphs of one and two trivial kernels, timed
    the same way inside the same rep loop;
  * shape speedup = median_k sqrt(r_2k * r_2k+1), r_i = (sglang_i - ovh)/(cand_i - ovh); shape time =
    (median(sglang) - ovh)/speedup; load = sum(weight * time); score = geomean over loads.
  * judge-projected: the judge's own SGLang medians from the scored d09 run divided by the local speedups.
Candidate-vs-candidate: the first --cand is the control; every other cand gets a paired ratio vs it.
"""
import argparse
import gc
import json
import math
import random
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


class Side:
    def __init__(self, label, path, mod, state):
        self.label, self.path, self.mod, self.state = label, str(path), mod, state
        self.bufs, self.graphs, self.outs = [], [], []
        self.raw = []


def capture(side, data, warm=2):
    """A placement: input buffers + a graph of kernel_fn on them."""
    import torch
    buf = C.clone_data(data)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warm):
            C.copy_into(buf, data)
            side.mod.kernel_fn(buf, side.state)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    C.copy_into(buf, data)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = side.mod.kernel_fn(buf, side.state)
    torch.cuda.synchronize()
    side.bufs.append(buf)
    side.graphs.append(g)
    side.outs.append(out)


def poison(out):
    import torch
    for t in out:
        if t.dtype.is_floating_point:
            t.fill_(float("nan"))
        else:
            t.fill_(-1)
    torch.cuda.synchronize()


def check_side(R, side, data, weights, k=0):
    """Replay placement k on `data` with poisoned outputs and check it like the judge."""
    import torch
    C.copy_into(side.bufs[k], data)
    poison(side.outs[k])
    side.graphs[k].replay()
    torch.cuda.synchronize()
    raw = R.collect(side.bufs[k], side.outs[k])
    ok, msg = R.check_implementation(data, raw, weights)
    return bool(ok), msg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cand", action="append", default=[], help="label=path (repeatable)")
    p.add_argument("--shapes", default="all")
    p.add_argument("--reps", default="judge", help="'judge' (60 for decode 1, else 20), or N (even)")
    p.add_argument("--reps-mult", type=int, default=1, help="multiply judge reps")
    p.add_argument("--xclones", type=int, default=2, help="input placements per side")
    p.add_argument("--pool", default="auto", help="fresh inputs per shape: N, or auto = one per rep (as the judge)"
                   " capped at 20 for batch >= 16")
    p.add_argument("--router-seed", default="auto", help="int, 'none' (weights' own router), or auto")
    p.add_argument("--weights-seed", type=int, default=5353)
    p.add_argument("--json", default="")
    p.add_argument("--nocheck", action="store_true")
    p.add_argument("--no-sglang", action="store_true", help="cands only (ratios vs the first cand)")
    p.add_argument("--pad-mb", type=int, default=64, help="max random pad between placements")
    p.add_argument("--seed-offset", type=int, default=0, help="shift the fresh-input pool (j offset)")
    a = p.parse_args()

    import torch
    import reference as R
    torch.set_grad_enabled(False)
    rng = random.Random(1234)
    rs = {"auto": C.DEFAULT_ROUTER_SEED, "none": None}.get(a.router_seed, None)
    if a.router_seed not in ("auto", "none"):
        rs = int(a.router_seed)
    shapes = C.select_shapes(a.shapes)
    t0 = time.time()
    weights = R.generate_weights(a.weights_seed, router_seed=rs)
    out = {"args": vars(a), "router_seed": rs, "gpu": torch.cuda.get_device_name(), "shapes": {}, "sides": []}

    def save():
        if a.json:
            Path(a.json).parent.mkdir(parents=True, exist_ok=True)
            Path(a.json).write_text(json.dumps(out, indent=1))

    specs = [] if a.no_sglang else [("sglang", C.PROB / "baseline.py")]
    for s in a.cand:
        label, path = s.split("=", 1)
        specs.append((label, Path(path)))
    sides = []
    pads = []
    for i, (label, path) in enumerate(specs):
        tp = time.time()
        mod = C.load_module(path, f"cdlh_side_{i}_{label}")
        w = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in weights.items()}
        state = mod.prepare(w)
        torch.cuda.synchronize()
        pads.append(torch.empty(rng.randint(1, a.pad_mb) << 20, dtype=torch.uint8, device="cuda"))
        sides.append(Side(label, path, mod, state))
        print(f"[prep] {label}: {time.time()-tp:.1f}s  ({path})", flush=True)
        out["sides"].append({"label": label, "path": str(path), "prepare_s": time.time() - tp})
    cands = [s for s in sides if s.label != "sglang"]
    base = None if a.no_sglang else sides[0]
    timer = C.Timer()

    for si in shapes:
        case = C.SCORED[si]
        name = C.SHORT[si]
        ts = time.time()
        if a.reps == "judge":
            reps = C.JUDGE_REPS[si] * a.reps_mult
        else:
            reps = int(a.reps)
        reps += reps % 2
        npool = int(a.pool) if a.pool != "auto" else (reps if case["batch"] == 1 else min(reps, 20))
        pool, loads = [], []
        for j in range(npool):
            d, ld = C.make_input(R, case, C.fresh_seed(case["seed"], j + a.seed_offset), weights)
            pool.append(d)
            loads.append(ld)
        for s in sides:
            for k in range(a.xclones):
                pads.append(torch.empty(rng.randint(1, a.pad_mb) << 20, dtype=torch.uint8, device="cuda"))
                capture(s, pool[0])
        rec = {"case": case, "reps": reps, "pool_loads": loads, "checks": {}, "raw_ns": {}, "one_ns": [], "two_ns": []}
        if not a.nocheck:
            for s in sides:
                ok, msg = check_side(R, s, pool[1 % len(pool)], weights)
                rec["checks"][s.label] = {"ok": ok, "msg": msg}
                print(f"[check] {name} {s.label}: {'PASS' if ok else 'FAIL'} {msg}", flush=True)
        # warm-up replays (untimed)
        for s in sides:
            for k in range(a.xclones):
                C.copy_into(s.bufs[k], pool[0])
                timer.time(s.graphs[k])
        raw = {s.label: [] for s in sides}
        fwd = cands + ([base] if base else [])
        for i in range(reps):
            data = pool[i % len(pool)]
            k = (i // 2) % a.xclones
            order = fwd if i % 2 == 0 else fwd[::-1]
            for s in order:
                C.copy_into(s.bufs[k], data)
                raw[s.label].append(timer.time(s.graphs[k]))
            one, two = timer.overhead()
            rec["one_ns"].append(one)
            rec["two_ns"].append(two)
        ovh = 2 * st.median(rec["one_ns"]) - st.median(rec["two_ns"])
        rec["ovh_ns"] = ovh
        rec["raw_ns"] = raw
        res = {}
        if base:
            b_med = st.median(raw["sglang"]) - ovh
            rec["sglang_ns"] = b_med
            for s in cands:
                sp, r = C.paired_speedup(raw["sglang"], raw[s.label], ovh)
                res[s.label] = {"speedup": sp, "time_ns": b_med / sp, "median_ns": st.median(raw[s.label]) - ovh,
                                "proj_ns": C.JUDGE_D09_SGLANG_NS[si] / sp}
        if len(cands) > 1:
            ctl = cands[0]
            for s in cands[1:]:
                sp, _ = C.paired_speedup(raw[ctl.label], raw[s.label], ovh)
                res.setdefault(s.label, {})["vs_" + ctl.label] = sp
            if ctl.label in C.ANCHORS:  # projection through the judged control: its judge time / local ratio
                jt = C.ANCHORS[ctl.label][1][si]
                res.setdefault(ctl.label, {})["anchor_ns"] = jt
                for s in cands[1:]:
                    res[s.label]["anchor_ns"] = jt / res[s.label]["vs_" + ctl.label]
        rec["results"] = res
        out["shapes"][name] = rec
        mean_r = sum(x[0] for x in loads) / len(loads)
        mean_d = sum(x[1] for x in loads) / len(loads)
        line = (f"[{name:4s}] reps {reps} ovh {ovh/1e3:.2f}us (one {st.median(rec['one_ns'])/1e3:.2f} two "
                f"{st.median(rec['two_ns'])/1e3:.2f}) load r/d {mean_r:.2f}/{mean_d:.2f} (tgt d {C.TARGET_DISTINCT[si]})")
        if base:
            line += f" sglang {rec['sglang_ns']/1e3:.1f}us (judge {C.JUDGE_D09_SGLANG_NS[si]/1e3:.1f})"
        print(line, flush=True)
        for s in cands:
            r = res[s.label]
            extra = f" vs_{cands[0].label} {r['vs_' + cands[0].label]:.4f}" if ("vs_" + cands[0].label) in r else ""
            if base:
                print(f"    {s.label:12s} {r['time_ns']/1e3:8.2f}us  x{r['speedup']:.4f}  proj {r['proj_ns']/1e3:8.2f}us"
                      f"  floor {C.FLOOR_US[si]:.1f}{extra}", flush=True)
            else:
                print(f"    {s.label:12s} median {st.median(raw[s.label])/1e3 - ovh/1e3:8.2f}us{extra}", flush=True)
        print(f"       ({time.time()-ts:.1f}s)", flush=True)
        save()
        # free the shape
        for s in sides:
            s.bufs, s.graphs, s.outs = [], [], []
        del pool
        pads = pads[:len(sides)]
        gc.collect()
        torch.cuda.empty_cache()

    # loads and score
    summary = {}
    if base:
        for s in cands:
            loads_local, loads_proj = {}, {}
            for li, load in enumerate(C.LOADS):
                idx = [i for i in (2 * li, 2 * li + 1) if C.SHORT[i] in out["shapes"]]
                if len(idx) != 2:
                    continue
                w = [C.SCORED[i]["weight"] for i in idx]
                recs = [out["shapes"][C.SHORT[i]] for i in idx]
                bl = sum(wi * r["sglang_ns"] for wi, r in zip(w, recs))
                cl = sum(wi * r["results"][s.label]["time_ns"] for wi, r in zip(w, recs))
                jb = sum(wi * C.JUDGE_D09_SGLANG_NS[i] for wi, i in zip(w, idx))
                jc = sum(wi * r["results"][s.label]["proj_ns"] for wi, r in zip(w, recs))
                loads_local[load] = {"sglang_us": bl / 1e3, "cand_us": cl / 1e3, "speedup": bl / cl}
                loads_proj[load] = {"sglang_us": jb / 1e3, "cand_us": jc / 1e3, "speedup": jb / jc}
            summary[s.label] = {"loads_local": loads_local, "loads_proj": loads_proj}
            if len(loads_local) == 6:
                summary[s.label]["score_local"] = C.geomean([v["speedup"] for v in loads_local.values()])
                summary[s.label]["score_proj"] = C.geomean([v["speedup"] for v in loads_proj.values()])
            print(f"\n== {s.label} ==")
            print(f"{'load':10s} {'sglang us':>10s} {'cand us':>10s} {'speedup':>8s} | {'judgeSGL':>10s} {'proj us':>10s} {'proj x':>7s}")
            for load in loads_local:
                l, pj = loads_local[load], loads_proj[load]
                print(f"{load:10s} {l['sglang_us']:10.1f} {l['cand_us']:10.1f} {l['speedup']:8.4f} | "
                      f"{pj['sglang_us']:10.1f} {pj['cand_us']:10.1f} {pj['speedup']:7.4f}")
            if "score_local" in summary[s.label]:
                print(f"score local {summary[s.label]['score_local']:.4f}   judge-projected {summary[s.label]['score_proj']:.4f}")
    if len(cands) > 1:
        ctl = cands[0]
        for s in cands[1:]:
            rat = {C.SHORT[i]: out["shapes"][C.SHORT[i]]["results"][s.label]["vs_" + ctl.label]
                   for i in shapes}
            # load-level A/B: weight the control's times
            lr = {}
            for li, load in enumerate(C.LOADS):
                idx = [i for i in (2 * li, 2 * li + 1) if C.SHORT[i] in rat]
                if len(idx) != 2:
                    continue
                if base:
                    ct = [out["shapes"][C.SHORT[i]]["results"][ctl.label]["time_ns"] for i in idx]
                else:
                    ct = [st.median(out["shapes"][C.SHORT[i]]["raw_ns"][ctl.label]) - out["shapes"][C.SHORT[i]]["ovh_ns"]
                          for i in idx]
                w = [C.SCORED[i]["weight"] for i in idx]
                lr[load] = sum(wi * t for wi, t in zip(w, ct)) / sum(wi * t / rat[C.SHORT[i]] for wi, t, i in zip(w, ct, idx))
            summary.setdefault(s.label, {})["vs_" + ctl.label] = {"shapes": rat, "loads": lr}
            print(f"\n== {s.label} vs {ctl.label} (>1 = {s.label} faster) ==")
            print("  " + "  ".join(f"{k} {v:.4f}" for k, v in rat.items()))
            if lr:
                print("  loads: " + "  ".join(f"{k} {v:.4f}" for k, v in lr.items()))
                if len(lr) == 6:
                    print(f"  score ratio {C.geomean(list(lr.values())):.4f}")
    if cands and cands[0].label in C.ANCHORS:
        ctl = cands[0]
        jsgl = C.ANCHORS[ctl.label][0]
        print(f"\n== anchor-projected (judge {ctl.label} run: its SGLang medians and {ctl.label} times, local ratio vs {ctl.label}) ==")
        for s in cands:
            ld = {}
            for li, load in enumerate(C.LOADS):
                idx = [i for i in (2 * li, 2 * li + 1) if C.SHORT[i] in out["shapes"]]
                if len(idx) != 2:
                    continue
                w = [C.SCORED[i]["weight"] for i in idx]
                jb = sum(wi * jsgl[i] for wi, i in zip(w, idx))
                jc = sum(wi * out["shapes"][C.SHORT[i]]["results"][s.label]["anchor_ns"] for wi, i in zip(w, idx))
                ld[load] = {"sglang_us": jb / 1e3, "cand_us": jc / 1e3, "speedup": jb / jc}
            summary.setdefault(s.label, {})["loads_anchor"] = ld
            if len(ld) == 6:
                summary[s.label]["score_anchor"] = C.geomean([v["speedup"] for v in ld.values()])
            print(f"  {s.label:10s} " + "  ".join(f"{k} {v['speedup']:.4f}" for k, v in ld.items()) +
                  (f"   score {summary[s.label]['score_anchor']:.4f}" if len(ld) == 6 else ""))
    out["summary"] = summary
    out["elapsed_s"] = time.time() - t0
    save()
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
