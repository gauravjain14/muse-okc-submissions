"""Judge-style bench for muse_bf16_proj: CUDA-graph replays, cold L2 (read flush), paired with SGLang.

    python3 claude/bench.py --cand v1=claude/sub_v1.py [--cand cx=candidate_v2.py] [--shapes 1,8] [--layers o,router]
                            [--reps 30] [--json /tmp/b.json] [--no-base] [--check-reps 2]

Per shape: each rep times every variant on the same fresh input, order reversed on odd reps; the fixed cost of an
empty graph replay is subtracted. Speedup per shape = median over rep pairs of sqrt(prod base/cand) (judge's paired
ratio); a load's time = sum of its shapes' times; score = geomean over loads of base/cand load time.
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
PROB = HERE
sys.path.insert(0, PROB)
import torch  # noqa: E402

import reference as R  # noqa: E402

LOADS = [("decode 1", 1), ("decode 16", 16), ("decode 32", 32), ("verify 1", 8), ("verify 16", 128), ("verify 32", 256)]
LAYERS = ["qkv_gate", "o", "pooler", "unpooler", "router"]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path if os.path.isabs(path) else os.path.join(PROB, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Graphed:
    def __init__(self, fn, state, data, pad_bytes=0):
        # pad_bytes > 0: allocate a spacer first so this graph's x lands at a different address (x placement
        # moves small kernels by 0.1-0.5 us; the judge draws fresh inputs every call, so we average placements)
        self.pad = torch.empty(pad_bytes, dtype=torch.uint8, device="cuda") if pad_bytes else None
        self.data = {"x": data["x"].clone(), "config": dict(data["config"])}
        for _ in range(3):
            fn(self.data, state)
        torch.cuda.synchronize()
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            self.out = fn(self.data, state)
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", action="append", default=[])
    ap.add_argument("--shapes", default="")
    ap.add_argument("--layers", default="")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--check-reps", type=int, default=2)
    ap.add_argument("--wseed", type=int, default=R.WEIGHTS_SEED)
    ap.add_argument("--json", default="")
    ap.add_argument("--no-base", action="store_true")
    ap.add_argument("--flush-mb", type=int, default=512)
    ap.add_argument("--xclones", type=int, default=1,
                    help="graphs per candidate with x at randomised addresses; rep r uses clone r %% N (P5's method)")
    ap.add_argument("--joff", type=float, default=1.0, help="judge offset added to every shape time (judge ~= local + 1.0 us)")
    args = ap.parse_args()

    weights = R.generate_weights(args.wseed)
    cands = []
    for c in args.cand:
        label, path = c.split("=", 1)
        mod = load(path, "cand_" + label)
        t0 = time.time()
        st = mod.prepare(weights)
        torch.cuda.synchronize()
        print(f"[{label}] prepare {time.time() - t0:.1f}s", flush=True)
        cands.append((label, mod, st))
    base = None if args.no_base else load("baseline.py", "base")
    base_state = base.prepare(weights) if base else None

    flush = torch.empty(args.flush_mb * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    flush.normal_()
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

    for _ in range(20):
        time_replay(ge)
    fixed = statistics.median([time_replay(ge) for _ in range(50)])
    print(f"fixed graph cost {fixed:.2f} us", flush=True)

    want_m = {int(x) for x in args.shapes.split(",") if x}
    want_l = [x for x in args.layers.split(",") if x] or LAYERS
    rec = {"fixed": fixed, "shapes": {}}
    loadt = {}  # load -> label -> sum time
    for lname, m in LOADS:
        if want_m and m not in want_m:
            continue
        for layer in want_l:
            data0 = R.generate_input(m, layer, 1234)
            import random
            rng = random.Random(7 * m + len(layer))
            nx = max(1, args.xclones)
            pads = [0] + [rng.randrange(1, 1 << 16) * 512 for _ in range(nx - 1)]
            gsc = [[(label, Graphed(mod.kernel_fn, st, data0, pads[i])) for label, mod, st in cands] for i in range(nx)]
            gbc = [Graphed(base.kernel_fn, base_state, data0, pads[i]) if base else None for i in range(nx)]
            times = {label: [] for label, _, _ in cands}
            times["base"] = []
            for r in range(args.reps):
                gs, gb = gsc[r % nx], gbc[r % nx]
                fresh = R.generate_input(m, layer, 100000 + 1000 * m + r)
                for _, g in gs:
                    g.data["x"].copy_(fresh["x"])
                if gb:
                    gb.data["x"].copy_(fresh["x"])
                order = [(label, g.g) for label, g in gs] + ([("base", gb.g)] if gb else [])
                if r % 2:
                    order.reverse()
                for label, g in order:
                    times[label].append(max(time_replay(g) - fixed, 0.05))
                if r < args.check_reps:
                    for label, g in gs:
                        ok, msg = R.check_implementation(fresh, g.out, weights)
                        if not ok:
                            print(f"FAIL {label} m={m} {layer} rep {r}: {msg}", flush=True)
                            sys.exit(1)
            row = {}
            med_b = statistics.median(times["base"]) if gb else None
            line = f"{lname:9s} m={m:<4d} {layer:9s}"
            if gb:
                line += f" sglang {med_b:7.2f}"
            for label, _, _ in cands:
                t = times[label]
                if gb:
                    pr = [math.sqrt(times["base"][i] * times["base"][i + 1] / (t[i] * t[i + 1]))
                          for i in range(0, len(t) - 1, 2)]
                    sp = statistics.median(pr)
                    tt = med_b / sp
                    line += f" | {label} {tt:7.2f} x{sp:5.3f}"
                else:
                    tt = statistics.median(t)
                    sp = None
                    line += f" | {label} {tt:7.2f}"
                row[label] = {"t": tt, "sp": sp, "raw": t}
                loadt.setdefault(lname, {}).setdefault(label, 0.0)
                loadt[lname][label] += tt
            if gb:
                loadt.setdefault(lname, {}).setdefault("base", 0.0)
                loadt[lname]["base"] += med_b
                row["base"] = {"t": med_b, "raw": times["base"]}
            sol = R.speed_of_light_ns(m, layer, {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}) / 1e3
            line += f"   sol {sol:6.2f}"
            print(line, flush=True)
            rec["shapes"][f"{m}/{layer}"] = row
            del gs, gb
            torch.cuda.empty_cache()
    if not args.no_base:
        print("\nper-load speedups:")
        gm = {label: [] for label, _, _ in cands}
        for lname, d in loadt.items():
            s = f"  {lname:9s} sglang {d['base']:7.2f}"
            for label, _, _ in cands:
                gm[label].append(d["base"] / d[label])
                s += f" | {label} {d[label]:7.2f} x{d['base'] / d[label]:.3f}"
            print(s)
        for label, v in gm.items():
            g = math.exp(sum(map(math.log, v)) / len(v))
            nsh = {ln: sum(1 for k in rec["shapes"] if int(k.split("/")[0]) == mm) for ln, mm in LOADS}
            pj = [(d["base"] + args.joff * nsh[ln]) / (d[label] + args.joff * nsh[ln]) for ln, d in loadt.items()]
            gj = math.exp(sum(map(math.log, pj)) / len(pj))
            print(f"GEOMEAN {label}: x{g:.4f} over {len(v)} loads   judge-projected (+{args.joff} us/shape) x{gj:.4f}")
            rec.setdefault("geomean", {})[label] = g
            rec.setdefault("geomean_judge", {})[label] = gj
    rec["loads"] = loadt
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rec, f)


if __name__ == "__main__":
    main()
