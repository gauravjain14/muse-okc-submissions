"""Same-run A/B of single MODULE files (agent IC), placement-controlled (P5's trick, as LN/bench_ln.py): every candidate
gets XCL CUDA graphs per shape, graph i reading x at the SAME random pool address for every candidate; judge-style
512 MB read before every replay, empty-graph cost subtracted, replay order alternated per rep. Outputs of every
candidate are compared bit for bit with the first candidate's (layout-only patches must be identical).

    python3 claude/t3/IC/abm.py --cand A=claude/t3/LN/sk11_lean.py:sk11 --cand B=claude/t3/IC/sk11_ic.py:sk11 \
        --shapes 16 --layers o [--xcl 4] [--reps 40] [--rounds 1]
A candidate is LABEL=path:name (module exposes prepare_<name> / run_<name>); the same path may appear twice (separate
module instances = an A/A' control for weight-placement noise).
"""
import argparse
import importlib.util
import os
import statistics
import sys
import time

PROB = "/root/p" if os.path.exists("/root/p") else os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROB)
import torch  # noqa: E402

import reference as R  # noqa: E402


def load(path, modname):
    spec = importlib.util.spec_from_file_location(modname, path if os.path.isabs(path) else os.path.join(PROB, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", action="append", default=[])
    ap.add_argument("--shapes", default="16")
    ap.add_argument("--layers", default="o")
    ap.add_argument("--xcl", type=int, default=4)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--xseed", type=int, default=7)
    a = ap.parse_args()
    w = R.generate_weights(R.WEIGHTS_SEED)
    cands = []
    for i, c in enumerate(a.cand):
        label, spec = c.split("=", 1)
        path, name = spec.rsplit(":", 1)
        mod = load(path, f"abm_{i}_{label}")
        t0 = time.time()
        st = getattr(mod, "prepare_" + name)(w)
        torch.cuda.synchronize()
        print(f"[{label}] {path} prepare {time.time() - t0:.1f}s", flush=True)
        cands.append((label, getattr(mod, "run_" + name), st))
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    flush.normal_()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def t1(g):
        flush.sum()
        ev[0].record()
        g.replay()
        ev[1].record()
        torch.cuda.synchronize()
        return ev[0].elapsed_time(ev[1]) * 1000.0

    tmp = torch.zeros(1, device="cuda")
    ge = torch.cuda.CUDAGraph()
    with torch.cuda.graph(ge):
        tmp.add_(1)
    for _ in range(20):
        t1(ge)
    fixed = statistics.median([t1(ge) for _ in range(60)])
    print(f"fixed graph cost {fixed:.2f} us; xcl {a.xcl}, reps {a.reps}, rounds {a.rounds}", flush=True)
    pool = torch.empty(96 * 1024 * 1024 // 2, dtype=torch.bfloat16, device="cuda")
    rng = torch.Generator().manual_seed(a.xseed)
    for m in [int(v) for v in a.shapes.split(",")]:
        for layer in a.layers.split(","):
            # exactness: every candidate on a normal and an outliers input, bit-compared with the first candidate
            ref_out = []
            for seed, dist in ((1234, "normal"), (77, "outliers")):
                d = R.generate_input(m, layer, seed, dist=dist)
                outs = []
                for lab, run, st in cands:
                    o = run(d["x"], d["config"], st)
                    torch.cuda.synchronize()
                    assert o is not None, (lab, m, layer)
                    ok, msg = R.check_implementation(d, o, w)
                    assert ok, (lab, m, layer, msg)
                    outs.append(o.clone())
                same = ["=" if torch.equal(outs[0].view(torch.uint8) if outs[0].dtype != torch.float32 else outs[0],
                                           o.view(torch.uint8) if o.dtype != torch.float32 else o) else "DIFF"
                        for o in outs[1:]]
                ref_out.append(same)
            data = R.generate_input(m, layer, 1234)
            cfg = dict(data["config"])
            for rnd in range(a.rounds):
                graphs, keep = [], []
                for ci in range(a.xcl):
                    off = int(torch.randint(0, (pool.numel() - data["x"].numel()) // 2048, (1,), generator=rng)) * 2048
                    x = pool[off: off + data["x"].numel()].view_as(data["x"])
                    x.copy_(data["x"])
                    for lab, run, st in cands:
                        for _ in range(3):
                            out = run(x, cfg, st)
                        torch.cuda.synchronize()
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):
                            o = run(x, cfg, st)
                        torch.cuda.synchronize()
                        keep.append(o)
                        graphs.append((lab, ci, g))
                res = {(lab, ci): [] for lab, ci, _ in graphs}
                for _ in range(3):
                    for _, _, g in graphs:
                        t1(g)
                for rep in range(a.reps):
                    for lab, ci, g in (graphs if rep % 2 == 0 else graphs[::-1]):
                        res[(lab, ci)].append(t1(g) - fixed)
                med = {k: statistics.median(v) for k, v in res.items()}
                line = f"m={m:<4d} {layer:9s} r{rnd}"
                l0 = cands[0][0]
                for lab, _, _ in cands:
                    v = [med[(lab, ci)] for ci in range(a.xcl)]
                    line += f" | {lab} {statistics.mean(v):6.3f}"
                    if lab != l0:
                        dl = [med[(lab, ci)] - med[(l0, ci)] for ci in range(a.xcl)]
                        line += f" d {statistics.mean(dl):+.3f} [{min(dl):+.2f},{max(dl):+.2f}]"
                print(line + f"  bits {ref_out}", flush=True)
                del graphs, keep
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
