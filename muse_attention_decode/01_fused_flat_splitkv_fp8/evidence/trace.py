"""Phase timeline of dev/k2.py built with K2_TRACE (globaltimer stamps per CTA), under graph replay + cold L2.

    python3 dev/trace.py [--shape "1,1,sliding"] [--reps 10]   shape = batch,q_len,layer
Prints, per phase, the median over reps of (min / median / max over CTAs) of the stamp minus the earliest CTA start.
Phases: 0 start, 1 seq loaded, 2 Q/new tile done, 3 first tile ready, 4 tiles done, 5 merged, 6 published,
7 combiner released, 8 combine done.
"""
import argparse
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
os.environ["K2_DEFS"] = "K2_TRACE"
import torch  # noqa: E402

import reference as R  # noqa: E402
from bench import Graphed, copy_into, shape_kwargs  # noqa: E402
import importlib.util  # noqa: E402

NAMES = ["start", "seq", "q/new", "tile0", "tiles", "merged", "publish", "released", "combined"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", action="append", default=[])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--file", default="dev/k2.py")
    ap.add_argument("--mode", default="flush", help="flush | noflush | warm (replay twice, trace the 2nd, no flush)")
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("k", os.path.join(HERE, "..", args.file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    flush = torch.empty(128 * 1024 * 1024, dtype=torch.float32, device="cuda").normal_()
    ws, _ = mod._workspace(torch.device("cuda", 0))
    tr = ws[-1048576:].view(torch.int64)
    for sh in args.shape or ["1,1,sliding", "1,1,full", "32,1,sliding", "32,1,full"]:
        b, L, layer = sh.split(",")
        kw = shape_kwargs(int(b), int(L), layer)
        data0 = R.generate_input(seed=1234, **kw)
        g = Graphed(mod.kernel_fn, data0)
        aligned, G, cpr = mod.plan(data0["config"], data0["page_table"].shape[1])
        hp = data0["config"]["num_kv_heads"] // 2
        per_phase = [[] for _ in NAMES]
        for r in range(args.reps):
            fresh = R.generate_input(seed=500 + r, **kw)
            copy_into(g.data, fresh)
            if args.mode == "warm":
                g.g.replay()
                torch.cuda.synchronize()
            tr.zero_()
            if args.mode == "flush":
                flush.sum()
            torch.cuda.synchronize()
            g.g.replay()
            torch.cuda.synchronize()
            t = tr[:8 * 1024 * 16].view(8, 1024, 16)[:hp, :G, :].reshape(-1, 16).cpu()
            t0 = int(t[:, 0][t[:, 0] > 0].min())
            for k in range(len(NAMES)):
                v = [int(x) - t0 for x in t[:, k] if int(x) > 0]
                if v:
                    per_phase[k].append((min(v), statistics.median(v), max(v), len(v)))
        print(f"\n[{args.mode}] {sh}: G={G} aligned={aligned} cpr={cpr}  (us from first CTA start; median over {args.reps} reps)")
        for k, name in enumerate(NAMES):
            if per_phase[k]:
                mn = statistics.median(x[0] for x in per_phase[k]) / 1e3
                md = statistics.median(x[1] for x in per_phase[k]) / 1e3
                mx = statistics.median(x[2] for x in per_phase[k]) / 1e3
                n = statistics.median(x[3] for x in per_phase[k])
                print(f"  {k} {name:9s} min {mn:7.2f}  med {md:7.2f}  max {mx:7.2f}   (CTAs {n:.0f})")


if __name__ == "__main__":
    main()
