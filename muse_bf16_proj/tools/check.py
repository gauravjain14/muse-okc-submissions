"""Correctness gate: the judge's 9 tests + every scored shape + nearby M, several seeds, normal and outliers.

    python3 claude/t3/check.py <cand.py> [--layers o,router] [--ms 1,8,16] [--quick] [--graph]

--ms restricts M (default: tests + scored + near-scored M). --graph also checks outputs produced by CUDA-graph
replay with fresh inputs copied in (what the judge times). Exit code 1 on the first failure.
"""
import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = HERE
sys.path.insert(0, PROB)
import torch  # noqa: E402

import reference as R  # noqa: E402

TESTS = [(1, "qkv_gate", "normal"), (7, "pooler", "normal"), (33, "o", "normal"), (130, "unpooler", "normal"),
         (5, "unpooler", "normal"), (8, "router", "normal"), (9, "router", "outliers"), (512, "qkv_gate", "outliers"),
         (1000, "unpooler", "outliers")]
SCORED_M = [1, 8, 16, 32, 128, 256]
NEAR_M = [2, 3, 4, 6, 9, 12, 15, 17, 24, 31, 33, 48, 64, 96, 127, 129, 192, 255, 257]
LAYERS = ["qkv_gate", "o", "pooler", "unpooler", "router"]


def _poison():
    """Fill freed caching-allocator memory with NaN so a kernel that leaves output elements unwritten fails the check
    instead of passing on stale correct values (agent Q2 found a 64-of-72-columns store bug hidden this way)."""
    try:
        t = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
        t.fill_(0xFF)              # 0xFFFF bf16 / 0xFFFFFFFF fp32 = NaN
        del t
    except RuntimeError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cand")
    ap.add_argument("--layers", default="")
    ap.add_argument("--ms", default="")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("cand", a.cand if os.path.isabs(a.cand) else os.path.join(PROB, a.cand))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    w = R.generate_weights(R.WEIGHTS_SEED)
    st = mod.prepare(w)
    layers = [x for x in a.layers.split(",") if x] or LAYERS
    if a.ms:
        cases = [(int(m), l, d) for m in a.ms.split(",") for l in layers for d in ("normal", "outliers")]
    else:
        cases = [c for c in TESTS if c[1] in layers]
        ms = SCORED_M + ([] if a.quick else NEAR_M)
        cases += [(m, l, d) for m in ms for l in layers for d in ("normal", "outliers")]
    nfail = 0
    worst = {}
    for m, layer, dist in cases:
        for s in range(a.seeds):
            data = R.generate_input(m, layer, 9000 + 31 * s + m, dist=dist)
            _poison()
            out = mod.kernel_fn(data, st)
            torch.cuda.synchronize()
            ok, msg = R.check_implementation(data, out, w)
            if not ok:
                print(f"FAIL m={m} {layer} {dist} seed {s}: {msg}", flush=True)
                nfail += 1
                break
        if a.graph and ok:
            d = {"x": data["x"].clone(), "config": dict(data["config"])}
            _poison()
            mod.kernel_fn(d, st)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                gout = mod.kernel_fn(d, st)
            for s in range(2):
                fresh = R.generate_input(m, layer, 777 + s, dist=dist)
                d["x"].copy_(fresh["x"])
                gout.fill_(float("nan"))   # graph-pool output: unwritten elements stay NaN and fail the check
                g.replay()
                torch.cuda.synchronize()
                ok, msg = R.check_implementation(fresh, gout, w)
                if not ok:
                    print(f"FAIL(graph) m={m} {layer} {dist} rep {s}: {msg}", flush=True)
                    nfail += 1
                    break
            del g
        worst[(m, layer)] = msg
    print(f"checked {len(cases)} cases x {a.seeds} seeds{' + graph' if a.graph else ''}: {'ALL OK' if not nfail else f'{nfail} FAILED'}")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
