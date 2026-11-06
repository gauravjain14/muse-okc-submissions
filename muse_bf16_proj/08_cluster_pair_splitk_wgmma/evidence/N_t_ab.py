"""Placement-averaged same-run A/B (judge-style timing) of full candidates (sub files with prepare/kernel_fn).

    python3 claude/t3/N/t_ab.py --cand a=claude/t3/bo6.py --cand b=claude/t3/N/sub_small.py --shapes 1,8 \
        --layers o,unpooler,router [--ng 6] [--reps 40]
Each candidate gets NG CUDA graphs per shape, each captured on its own freshly allocated input buffer (different x /
out placements; per-graph times differ by up to ~0.1 us). Every rep times every graph of every candidate after the
512 MB flush (order alternating); a candidate's time = median over all its graphs' samples; also prints the spread of
per-graph medians and the paired delta vs the first candidate (median over reps of the per-rep mean difference).
"""
import argparse
import importlib.util
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, PROB)
import torch  # noqa: E402

import reference as R  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path if os.path.isabs(path) else os.path.join(PROB, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", action="append", default=[])
    ap.add_argument("--shapes", default="1,8")
    ap.add_argument("--layers", default="o,unpooler,router")
    ap.add_argument("--ng", type=int, default=6)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--base", action="store_true", help="also SGLang baseline")
    a = ap.parse_args()
    w = R.generate_weights(R.WEIGHTS_SEED)
    cands = []
    for c in a.cand:
        label, path = c.split("=", 1)
        mod = load(path, "c_" + label)
        cands.append((label, mod, mod.prepare(w)))
    if a.base:
        b = load("baseline.py", "base")
        cands.append(("sglang", b, b.prepare(w)))
    torch.cuda.synchronize()
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    flush.normal_()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def capture(fn):
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        return g

    def tr(g):
        flush.sum()
        ev[0].record()
        g.replay()
        ev[1].record()
        torch.cuda.synchronize()
        return ev[0].elapsed_time(ev[1]) * 1000.0

    tmp = torch.zeros(1, device="cuda")
    ge = capture(lambda: tmp.add_(1))
    for _ in range(20):
        tr(ge)
    fixed = statistics.median([tr(ge) for _ in range(50)])
    print(f"fixed {fixed:.2f} us, {a.ng} graphs per candidate", flush=True)
    for m in [int(v) for v in a.shapes.split(",")]:
        for layer in a.layers.split(","):
            graphs = []   # (label, gi, data, graph)
            pad = []
            for label, mod, st in cands:
                for gi in range(a.ng):
                    d = R.generate_input(m, layer, 77 + gi)
                    data = {"x": d["x"].clone(), "config": dict(d["config"])}
                    pad.append(torch.empty(4096 * (gi + 1), dtype=torch.uint8, device="cuda"))   # shift placements
                    for _ in range(2):
                        mod.kernel_fn(data, st)
                    g = capture(lambda mod=mod, st=st, data=data: mod.kernel_fn(data, st))
                    graphs.append((label, gi, data, g))
            per = {(lb, gi): [] for lb, gi, _, _ in graphs}
            rep_mean = {lb: [] for lb, _, _ in cands}
            for r in range(a.reps):
                fresh = R.generate_input(m, layer, 5000 + r)["x"]
                for _, _, data, _ in graphs:
                    data["x"].copy_(fresh)
                order = graphs if r % 2 == 0 else graphs[::-1]
                cur = {lb: [] for lb, _, _ in cands}
                for lb, gi, _, g in order:
                    t = tr(g) - fixed
                    per[(lb, gi)].append(t)
                    cur[lb].append(t)
                for lb in cur:
                    rep_mean[lb].append(statistics.mean(cur[lb]))
            line = f"m={m:<3d} {layer:9s}"
            first = cands[0][0]
            for lb, _, _ in cands:
                allt = [t for (l2, gi), v in per.items() if l2 == lb for t in v]
                gm = [statistics.median(v) for (l2, gi), v in per.items() if l2 == lb]
                dl = statistics.median([x - y for x, y in zip(rep_mean[lb], rep_mean[first])])
                line += f" | {lb} {statistics.median(allt):5.2f} [{min(gm):.2f}-{max(gm):.2f}] d{dl:+.3f}"
            print(line, flush=True)
            del graphs, pad
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
