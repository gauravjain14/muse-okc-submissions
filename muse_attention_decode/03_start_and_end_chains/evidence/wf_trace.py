"""WF per-CTA phase timeline for FLAT kernels built with K2_TRACE (globaltimer stamps, slot k of 32 per CTA).

    python3 dev/wf/trace.py --file dev/wf/k8a.py --shape 32,1,full [--shape 32,8,full] [--reps 10] [--defs X,Y]
        [--mode flush|warm]
Stamps (run1): 0 start, 9 prefix, 16 pt-LDG issued, 17 first row back, 1 prologue issued, 2 q/new, 3 tile0,
4 tiles done, 11 mO stored, 12 merged, 5 seg end, 6 published, 7 combine start, 8 combined; 14 nseg, 15 smid.
Prints min/med/max per phase (median over reps), tile-finish percentiles, per-nseg means, and the critical CTA.
"""
import argparse
import importlib.util
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
ap = argparse.ArgumentParser()
ap.add_argument("--shape", action="append", default=[])
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--file", default="dev/wf/k8a.py")
ap.add_argument("--defs", default="")
ap.add_argument("--mode", default="flush")
ap.add_argument("--ctx", type=int, default=0, help="override the context length (e.g. 8000: K/V fit in L2)")
args = ap.parse_args()
os.environ["K2_DEFS"] = ",".join(["K2_TRACE"] + [d for d in args.defs.split(",") if d])
import torch  # noqa: E402

import reference as R  # noqa: E402
from bench import Graphed, copy_into, shape_kwargs  # noqa: E402

NAMES = {0: "start", 9: "prefix", 16: "pt-iss", 17: "row-back", 1: "issued", 2: "q/new", 3: "tile0", 4: "tiles",
         11: "mstore", 12: "merged", 5: "segend", 6: "publish", 7: "comb0", 8: "combined", 18: "mstore1", 19: "s19",
         20: "comb1", 21: "tile0done", 22: "loopexit", 23: "cpwaitall"}
ORDER = [0, 9, 16, 17, 1, 2, 3, 21, 22, 23, 4, 18, 11, 12, 5, 6, 7, 19, 20, 8]


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]


def main():
    spec = importlib.util.spec_from_file_location("k", os.path.join(HERE, "..", "..", args.file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    flush = torch.empty(128 * 1024 * 1024, dtype=torch.float32, device="cuda").normal_()
    ws, _ = mod._workspace(torch.device("cuda", 0))
    tr = ws[-1048576:].view(torch.int64)
    for sh in args.shape or ["32,1,full"]:
        b, L, layer = sh.split(",")
        kw = shape_kwargs(int(b), int(L), layer)
        if args.ctx:
            kw["ctx"] = args.ctx
        data0 = R.generate_input(seed=1234, **kw)
        g = Graphed(mod.kernel_fn, data0)
        per = {k: [] for k in NAMES}
        tq = []
        nsegs = {}
        crit = []
        ends = []
        smdev = {}
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
            t = tr[:1024 * 32].view(1024, 32).cpu()
            rows = [i for i in range(1024) if int(t[i, 0]) > 0]
            t0 = min(int(t[i, 0]) for i in rows)
            for k in NAMES:
                v = [(int(t[i, k]) - t0) / 1e3 for i in rows if int(t[i, k]) > 0]
                if v:
                    per[k].append((min(v), statistics.median(v), max(v), len(v)))
            stamps = [(int(t[i, k]) - t0) / 1e3 for i in rows for k in ORDER if int(t[i, k]) > 0]
            ends.append(max(stamps))
            t4 = [(int(t[i, 4]) - t0) / 1e3 for i in rows if int(t[i, 4]) > 0]
            tq.append([pct(t4, q) for q in (0.0, 0.1, 0.5, 0.9, 1.0)] + [statistics.mean(t4)])
            for i in rows:
                ns = int(t[i, 14])
                if int(t[i, 4]):
                    nsegs.setdefault(ns, []).append((int(t[i, 4]) - t0) / 1e3)
                    smdev.setdefault(int(t[i, 15]), []).append((int(t[i, 4]) - t0) / 1e3 - statistics.median(t4))
            # critical CTA: max combined (8) else max publish (6)
            key = 8 if any(int(t[i, 8]) for i in rows) else 6
            ci = max(rows, key=lambda i: int(t[i, key]))
            crit.append([(int(t[ci, k]) - t0) / 1e3 if int(t[ci, k]) else float("nan") for k in ORDER])
        print(f"\n[{args.mode}] {sh} defs={args.defs}: (us from first CTA start; median over {args.reps} reps)")
        for k in ORDER:
            if per[k]:
                mn = statistics.median(x[0] for x in per[k])
                md = statistics.median(x[1] for x in per[k])
                mx = statistics.median(x[2] for x in per[k])
                n = statistics.median(x[3] for x in per[k])
                print(f"  {k:2d} {NAMES[k]:9s} min {mn:7.2f}  med {md:7.2f}  max {mx:7.2f}   (CTAs {n:.0f})")
        print(f"  last stamp med {statistics.median(ends):.2f} max {max(ends):.2f}")
        print("  tiles-done p0/p10/p50/p90/max/mean: " + " ".join(f"{statistics.median(x[j] for x in tq):7.2f}" for j in range(6)))
        for n, v in sorted(nsegs.items()):
            print(f"  nseg {n}: n {len(v):5d} tiles mean {statistics.mean(v):7.2f} max {max(v):7.2f}")
        cm = [statistics.median(x[j] for x in crit if x[j] == x[j]) if any(x[j] == x[j] for x in crit) else float('nan')
              for j in range(len(ORDER))]
        print("  critical CTA (med): " + " ".join(f"{NAMES[k]}={cm[j]:.2f}" for j, k in enumerate(ORDER) if cm[j] == cm[j]))
        dev = sorted((statistics.mean(v), sm) for sm, v in smdev.items() if len(v) >= max(2, args.reps // 2))
        if dev:
            print(f"  per-SM mean deviation of tiles-done (us): min {dev[0][0]:.2f} (sm {dev[0][1]}) max {dev[-1][0]:.2f} "
                  f"(sm {dev[-1][1]}) stdev {statistics.pstdev([d for d, _ in dev]):.2f}")


if __name__ == "__main__":
    main()
