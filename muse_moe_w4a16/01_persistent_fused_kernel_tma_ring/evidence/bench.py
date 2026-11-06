"""Local judge stand-in: correctness tests, then CUDA-graph timing against SGLang's baseline.

Mirrors TASK.md: each shape is captured as a CUDA graph, each replay starts with a cold L2,
inputs are fresh per repetition, and a shape's speedup is the median of paired ratios.

  python bench.py submission.py [--reps 40] [--no-base] [--shapes 1,16] [--tests-only]
"""

import argparse
import importlib.util
import math
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, PROB)

import reference as R  # noqa: E402

SHAPES = [(1, "decode 1", 401), (16, "decode 16", 402), (32, "decode 32", 403),
          (8, "verify 1", 404), (128, "verify 16", 405), (256, "verify 32", 406)]
TESTS = [dict(bs=4), dict(bs=33), dict(bs=257), dict(bs=128, dist="skewed"), dict(bs=1)]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def copy_into(dst, src):
    for k in ("hidden_states", "topk_ids", "topk_weights"):
        dst[k].copy_(src[k])


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


def touched_bytes(data):
    ids = data["topk_ids"]
    loc = ids[ids < 32]
    e = int(torch.unique(loc).numel())
    return e, int(loc.numel()), e * 3 * 4096 * 4096 * (0.5 + 1 / 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sub")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--no-base", action="store_true")
    ap.add_argument("--tests-only", action="store_true")
    ap.add_argument("--no-tests", action="store_true")
    ap.add_argument("--shapes", default="")
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    t0 = time.time()
    weights = R.generate_weights(R.WEIGHTS_SEED)
    torch.cuda.synchronize()
    print(f"weights generated in {time.time() - t0:.1f}s", flush=True)

    sub = load(os.path.abspath(args.sub), "submission")
    t0 = time.time()
    sub_state = sub.prepare(weights)
    torch.cuda.synchronize()
    print(f"submission prepare {time.time() - t0:.1f}s", flush=True)
    base = base_state = None
    if not args.no_base:
        base = load(os.path.join(PROB, "baseline.py"), "baseline")
        base_state = base.prepare(weights)
        torch.cuda.synchronize()

    if not args.no_tests:
        for i, t in enumerate(TESTS):
            data = R.generate_input(sharding="ep8", seed=1000 + i, **t)
            out = sub.kernel_fn(data, sub_state)
            torch.cuda.synchronize()
            ok, msg = R.check_implementation(data, out, weights)
            print(f"test {t}: {'PASS' if ok else 'FAIL'} {msg}", flush=True)
            if not ok:
                sys.exit(1)
    if args.tests_only:
        return

    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    # Fixed replay cost: an empty-ish graph.
    tiny = torch.zeros(1, device="cuda")
    ge = torch.cuda.CUDAGraph()
    with torch.cuda.graph(ge):
        tiny.add_(1)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def time_replay(g):
        flush.sum()
        s.record()
        g.replay()
        e.record()
        e.synchronize()
        return s.elapsed_time(e) * 1000.0

    fixed = statistics.median([time_replay(ge) for _ in range(30)])
    print(f"fixed replay cost {fixed:.2f} us", flush=True)

    want = {int(x) for x in args.shapes.split(",") if x}
    speedups = []
    for bs, load_name, seed in SHAPES:
        if want and bs not in want:
            continue
        data = R.generate_input(bs, "ep8", seed)
        gs = Graphed(sub.kernel_fn, sub_state, data)
        gb = Graphed(base.kernel_fn, base_state, data) if base else None
        ratios, ts, tb, nexp, npairs = [], [], [], [], []
        for r in range(args.reps):
            fresh = R.generate_input(bs, "ep8", seed * 1000 + r)
            ne, npr, _ = touched_bytes(fresh)
            nexp.append(ne)
            npairs.append(npr)
            copy_into(gs.data, fresh)
            if gb:
                copy_into(gb.data, fresh)
            order = (gs, gb) if r % 2 == 0 else (gb, gs)
            times = {}
            for g in order:
                if g is None:
                    continue
                times[id(g)] = max(time_replay(g.g) - fixed, 0.1)
            ts.append(times[id(gs)])
            if gb:
                tb.append(times[id(gb)])
                ratios.append(times[id(gb)] / times[id(gs)])
            if r < 3 and not args.no_check:
                ok, msg = R.check_implementation(fresh, gs.out, weights)
                if not ok:
                    print(f"  bs {bs} rep {r}: FAIL {msg}")
                    sys.exit(1)
        med_s = statistics.median(ts)
        mean_e = statistics.mean(nexp)
        gbps = mean_e * 3 * 4096 * 4096 * (0.5 + 1 / 16) / (med_s * 1e-6) / 1e9
        line = (f"{load_name:10s} bs {bs:3d}  experts {mean_e:5.1f} pairs {statistics.mean(npairs):6.1f}  "
                f"sub {med_s:8.1f} us ({gbps:6.0f} GB/s)")
        if gb:
            sp = statistics.median(ratios)
            speedups.append(sp)
            line += f"  base {statistics.median(tb):8.1f} us  speedup {sp:5.3f}x"
        print(line, flush=True)
        del gs, gb
    if speedups:
        gm = math.exp(sum(math.log(x) for x in speedups) / len(speedups))
        print(f"geomean speedup {gm:.3f}x over {len(speedups)} loads")


if __name__ == "__main__":
    main()
