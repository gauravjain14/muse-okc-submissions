"""Whole-layer timeline of a candidate under nsys: CUDA graph replays with a cold L2, every kernel
node traced. Run under nsys, then summarize:

    nsys profile --cuda-graph-trace=node -t cuda,nvtx --capture-range=cudaProfilerApi -o /tmp/p \\
        python3 dev/layer_prof.py cand.py [--cases 0,1] [--replays 3]
    python3 dev/layer_prof.py --summarize /tmp/p.nsys-rep [--json out.json]

The summary gives, per case and replay, every kernel's start offset and duration (overlapping
kernels from side streams are kept), the idle gaps between kernels, and the span.
"""
import argparse, importlib.util, json, os, subprocess, sys, statistics, csv, io
HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, PROB); sys.path.insert(0, HERE)


def run(args):
    import torch
    import reference as R
    import bench as B
    weights = R.generate_weights(B.WEIGHTS_SEED)
    rs = B.router_seed(weights, B.scored_cases())
    weights = R.generate_weights(B.WEIGHTS_SEED, router_seed=rs)
    spec = importlib.util.spec_from_file_location("cand", os.path.join(PROB, args.cand))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    st = mod.prepare(weights)
    flush = torch.empty(1 << 28, dtype=torch.float32, device="cuda")
    cases = B.scored_cases()
    want = [int(x) for x in args.cases.split(",")] if args.cases else list(range(len(cases)))
    graphs = []
    for ci in want:
        case = cases[ci]
        d0 = B.fresh_input(case, case["seed"])
        ls = B.busiest(mod.kernel_fn(d0, st)[2])
        d0 = B.fresh_input(case, case["seed"], ls)
        g = B.Graphed(mod.kernel_fn, st, d0)
        graphs.append((ci, g))
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for ci, g in graphs:
        for r in range(args.replays):
            flush.sum()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(f"case{ci}/r{r}")
            g.g.replay()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()


def short(name):
    n = name.split("(")[0]
    for pre in ("void ", "cutlass::device_kernel<", "flash::"):
        n = n.replace(pre, "")
    return n[:60]


def summarize(args):
    rep = args.summarize
    out = subprocess.run(["nsys", "stats", "--report", "cuda_gpu_trace", "--report", "nvtx_pushpop_trace", "--format", "csv",
                          "--output", "-", rep], capture_output=True, text=True).stdout
    # nsys prints the reports one after another; split on header lines
    blocks, cur = [], []
    for line in out.splitlines():
        if line.startswith('"Start (ns)"') or line.startswith("Start (ns)") or line.startswith('"Start"'):
            if cur:
                blocks.append(cur)
            cur = [line]
        elif cur:
            cur.append(line)
    if cur:
        blocks.append(cur)
    kern, ranges = [], []
    for b in blocks:
        rows = list(csv.DictReader(io.StringIO("\n".join(b))))
        if not rows:
            continue
        keys = rows[0].keys()
        if "Name" in keys and any("Grid" in k for k in keys):
            for r in rows:
                kern.append((int(r["Start (ns)"]), int(r["Duration (ns)"]), r["Name"], r.get("Strm", r.get("Stream", ""))))
        elif "Range" in keys or "Name" in keys:
            for r in rows:
                nm = r.get("Range") or r.get("Name")
                st = int(r.get("Start (ns)") or r.get("Start"))
                en = int(r.get("End (ns)") or r.get("End"))
                ranges.append((nm, st, en))
    kern.sort()
    res = {}
    for nm, st, en in ranges:
        if not nm.startswith("case"):
            continue
        ks = [k for k in kern if st <= k[0] < en]
        if not ks:
            continue
        t0 = ks[0][0]
        span = max(k[0] + k[1] for k in ks) - t0
        # idle: time inside the span with no kernel running
        busy, cur_end = 0, t0
        for s, d, _, _ in ks:
            if s > cur_end:
                busy += 0
            e = s + d
            if e > cur_end:
                busy += e - max(s, cur_end)
                cur_end = e
        res.setdefault(nm.split("/")[0], []).append(dict(span=span, idle=span - busy,
                                                         kernels=[(short(n), (s - t0) / 1e3, d / 1e3, strm) for s, d, n, strm in ks]))
    for case, reps in sorted(res.items(), key=lambda x: int(x[0][4:])):
        spans = [r["span"] / 1e3 for r in reps]
        idles = [r["idle"] / 1e3 for r in reps]
        print(f"== {case}: span {statistics.median(spans):.1f} us (replays {', '.join('%.1f' % x for x in spans)}), idle {statistics.median(idles):.1f} us")
        r = reps[len(reps) // 2]
        for n, s, d, strm in r["kernels"]:
            print(f"   {s:9.1f} +{d:8.1f}  s{strm:>3}  {n}")
    if args.json:
        json.dump(res, open(args.json, "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cand", nargs="?")
    ap.add_argument("--cases", default="")
    ap.add_argument("--replays", type=int, default=3)
    ap.add_argument("--summarize", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    summarize(a) if a.summarize else run(a)
