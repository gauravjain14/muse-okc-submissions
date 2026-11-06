"""Local judge-alike for muse_attention_decode.

python bench.py CAND.py [--tests] [--shapes all|d1s,d1f,...] [--reps N] [--seeds 0,1] [--nocheck]

Timing: one CUDA graph per shape, replayed after a cold-L2 read of a 512 MB buffer; median of reps.
Score: geomean over the 6 loads of (46 * sliding + 16 * full); with SGL times (from a judge run) also the
estimated board time = mean-like SGLang step time / geomean speedup (the judge's formula).
"""
import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference as R  # noqa: E402

SHAPES = {
    "d1s": dict(batch=1, layer="sliding", q_len=1, spread="fixed", dist="soft"),
    "d1f": dict(batch=1, layer="full", q_len=1, spread="fixed", dist="coherent"),
    "d16s": dict(batch=16, layer="sliding", q_len=1, spread="agentic", dist="soft"),
    "d16f": dict(batch=16, layer="full", q_len=1, spread="agentic", dist="coherent"),
    "d32s": dict(batch=32, layer="sliding", q_len=1, spread="agentic", dist="soft"),
    "d32f": dict(batch=32, layer="full", q_len=1, spread="agentic", dist="coherent"),
    "v1s": dict(batch=1, layer="sliding", q_len=8, spread="fixed", dist="soft"),
    "v1f": dict(batch=1, layer="full", q_len=8, spread="fixed", dist="coherent"),
    "v16s": dict(batch=16, layer="sliding", q_len=8, spread="agentic", dist="soft"),
    "v16f": dict(batch=16, layer="full", q_len=8, spread="agentic", dist="coherent"),
    "v32s": dict(batch=32, layer="sliding", q_len=8, spread="agentic", dist="soft"),
    "v32f": dict(batch=32, layer="full", q_len=8, spread="agentic", dist="coherent"),
}
LOADS = [("d1s", "d1f"), ("d16s", "d16f"), ("d32s", "d32f"), ("v1s", "v1f"), ("v16s", "v16f"), ("v32s", "v32f")]

TESTS = """batch 1, ctx 10, heads tp8, layer sliding
batch 3, ctx 2047, heads tp8, layer sliding
batch 5, ctx 5000, heads tp8, layer sliding, q_len 8, spread mixed, page 1
batch 4, ctx 3000, heads tp8, layer full
batch 8, ctx 8192, heads dp, layer full, spread mixed
batch 2, ctx 100000, heads tp8, layer full, q_len 8, page 1
batch 4, ctx 6000, heads tp8, layer sliding, dist edges, page 1
batch 4, ctx 6000, heads tp8, layer full, q_len 8, dist edges
batch 1, ctx 1048576, heads tp8, layer full, q_len 8
batch 3, ctx 4000, heads tp8, layer sliding, q_len 8, page 1, kv_scale 0.015625
batch 2, ctx 9000, heads tp8, layer full, kv_scale 0.015625
batch 4, ctx 80000, heads tp8, layer full, spread mixed, dist soft
batch 4, ctx 6000, heads tp8, layer sliding, q_len 8, dist soft, page 1
batch 2, ctx 20000, heads dp, layer full, q_len 8, dist soft
batch 4, ctx 80000, heads tp8, layer full, spread mixed, dist coherent, page 1
batch 2, ctx 20000, heads dp, layer full, q_len 8, dist coherent"""


def parse_test(line):
    kw = {}
    for part in line.split(","):
        k, v = part.strip().split(" ", 1)
        kw[k] = v if k in ("heads", "layer", "spread", "dist") else (float(v) if k == "kv_scale" else int(v))
    return kw


def gpu_lock():
    """No-op: cdl/sbx.py exec holds the box-exclusive GPU lock for the whole command."""
    return


def load(path):
    spec = importlib.util.spec_from_file_location("cand", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def check(fn, kw, seed):
    data = R.generate_input(seed=seed, **kw)
    out = fn(data)
    torch.cuda.synchronize()
    raw = R.collect(data, out)
    ok, msg = R.check_implementation(data, raw)
    return ok, msg


class Flush:
    def __init__(self):
        self.buf = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda").uniform_()
        self.acc = torch.empty((), dtype=torch.float32, device="cuda")

    def __call__(self):
        torch.sum(self.buf, dim=0, out=self.acc)


def time_shape(fn, kw, seed, reps, flush, check_too=True):
    data = R.generate_input(seed=seed, ctx=80000, heads="tp8", page=1, **kw)
    ok, msg = True, ""
    if check_too:
        d2 = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
        out = fn(d2)
        torch.cuda.synchronize()
        ok, msg = R.check_implementation(d2, R.collect(d2, out))
        del d2
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(data)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn(data)
    torch.cuda.synchronize()
    for _ in range(5):
        flush()
        g.replay()
    torch.cuda.synchronize()
    ts = []
    e0 = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    e1 = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        flush()
        e0[i].record()
        g.replay()
        e1[i].record()
    torch.cuda.synchronize()
    ts = [a.elapsed_time(b) * 1000 for a, b in zip(e0, e1)]
    del g, data
    return statistics.median(ts), ok, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cand")
    ap.add_argument("--tests", action="store_true")
    ap.add_argument("--shapes", default="all")
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--seeds", default="7")
    ap.add_argument("--nocheck", action="store_true")
    ap.add_argument("--sgl", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sgl_times.json"))
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    t0 = time.time()
    m = load(a.cand)
    print(f"[import+compile {time.time() - t0:.1f}s]", flush=True)
    gpu_lock()
    fn = m.kernel_fn
    if a.tests:
        bad = 0
        for i, line in enumerate(TESTS.splitlines()):
            kw = parse_test(line)
            try:
                ok, msg = check(fn, kw, 1000 + i)
            except Exception as e:  # noqa: BLE001
                ok, msg = False, repr(e)
            bad += not ok
            print(f"{'PASS' if ok else 'FAIL'} {line} :: {msg}", flush=True)
            torch.cuda.empty_cache()
        print(f"tests: {16 - bad}/16 pass", flush=True)
    names = list(SHAPES) if a.shapes == "all" else a.shapes.split(",")
    flush = Flush()
    seeds = [int(s) for s in a.seeds.split(",")]
    res = {}
    for n in names:
        tt = []
        for sd in seeds:
            t, ok, msg = time_shape(fn, SHAPES[n], sd, a.reps, flush, not a.nocheck)
            tt.append(t)
            if not ok:
                print(f"  !! {n} seed {sd} CHECK FAIL {msg}", flush=True)
            torch.cuda.empty_cache()
        res[n] = statistics.mean(tt)
        print(f"{n:5s} {res[n]:8.2f} us   " + " ".join(f"{x:.2f}" for x in tt) + (f"   {msg}" if not a.nocheck else ""), flush=True)
    if all(n in res for n in SHAPES):
        loads = [46 * res[s] + 16 * res[f] for s, f in LOADS]
        gm = math.exp(sum(math.log(x) for x in loads) / 6)
        print(f"geomean load (46s+16f): {gm:.1f} us  loads: " + " ".join(f"{x:.0f}" for x in loads))
        if os.path.exists(a.sgl):
            sg = json.load(open(a.sgl))
            ov = sg.get("_overhead", 0.0)
            sl = [46 * sg[s] + 16 * sg[f] for s, f in LOADS]
            ol = [46 * max(res[s] - ov, 0.1) + 16 * max(res[f] - ov, 0.1) for s, f in LOADS]
            sp = math.exp(sum(math.log(x / y) for x, y in zip(sl, ol)) / 6)
            print(f"est board time: {sg['_step'] / sp:.1f} us (speedup {sp:.3f}, overhead {ov})")
    if a.json:
        json.dump(res, open(a.json, "w"))


if __name__ == "__main__":
    main()
