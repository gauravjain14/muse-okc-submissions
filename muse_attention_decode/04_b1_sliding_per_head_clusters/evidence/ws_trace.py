"""WS per-CTA phase timeline for runA1 (decode sliding b1) built with K2_TRACE (from dev/wd/trace3.py).

    python3 dev/ws/trace.py --file dev/ws/k15s.py --shape 1,1,sliding [--reps 20] [--defs X,Y] [--mode flush|warm]
Prints min / med / max over CTAs (median over reps) of every stamped slot 0..31, sorted by the median time.
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
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--file", default="dev/ws/k15s.py")
ap.add_argument("--defs", default="")
ap.add_argument("--mode", default="flush", help="flush (cold L2, judge-like) | warm (replay once, then trace)")
ap.add_argument("--percta", action="store_true", help="per-CTA medians of every slot")
ap.add_argument("--G", type=int, default=0, help="CTAs per grid row to read (default: the host plan's)")
ap.add_argument("--raw", default="", help="slots holding raw values (TRV), printed as medians, not times")
args = ap.parse_args()
os.environ["K2_DEFS"] = ",".join(["K2_TRACE"] + [d for d in args.defs.split(",") if d])
import torch  # noqa: E402

import reference as R  # noqa: E402
from bench import Graphed, copy_into, shape_kwargs  # noqa: E402

NAMES = {0: "start", 1: "len-bar", 2: "q/new", 3: "kv-iss", 4: "tile0", 5: "tiles", 6: "pushed", 7: "recv",
         8: "s8", 9: "out", 10: "mstore", 11: "clwait", 12: "comb1", 13: "s13", 14: "s14", 15: "s15",
         16: "pt-iss", 17: "pt-in", 18: "len-in", 19: "verified", 20: "pre-ld", 21: "bar-w15", 22: "pre-iss",
         23: "row0-in", 24: "lane0", 25: "lane1", 26: "lane2", 27: "lane3", 28: "s28", 29: "s29", 30: "s30", 31: "s31"}


def main():
    spec = importlib.util.spec_from_file_location("k", os.path.join(HERE, "..", "..", args.file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    flush = torch.empty(128 * 1024 * 1024, dtype=torch.float32, device="cuda").normal_()
    ws, _ = mod._workspace(torch.device("cuda", 0))
    tr = ws[-1048576:].view(torch.int64)
    for sh in args.shape or ["1,1,sliding"]:
        b, L, layer = sh.split(",")
        kw = shape_kwargs(int(b), int(L), layer)
        data0 = R.generate_input(seed=1234, **kw)
        g = Graphed(mod.kernel_fn, data0)
        aligned, G, cpr = mod.plan(data0["config"], data0["page_table"].shape[1])
        if hasattr(mod, "_wc_plan"):
            G, cpr = mod._wc_plan(data0["config"], data0["page_table"].shape[1], aligned, G, cpr)[:2]
        G = min(args.G or G, 1024)
        per = {k: [] for k in range(32)}
        cta = {k: [[] for _ in range(G)] for k in range(32)}
        ends = []
        raw = [int(x) for x in args.raw.split(",") if x]
        rawv = {k: [] for k in raw}
        rawmx = {k: [] for k in raw}
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
            t = tr[:1024 * 32].view(1024, 32)[:G, :].cpu()
            t0 = int(t[:, 0][t[:, 0] > 0].min())
            for k in raw:
                v = [int(x) for x in t[:, k] if int(x) > 0]
                if v:
                    rawv[k].append(statistics.median(v))
                    rawmx[k].append(max(v))
                t[:, k] = 0
            for k in range(32):
                v = [int(x) - t0 for x in t[:, k] if int(x) > 0]
                if v:
                    per[k].append((min(v), statistics.median(v), max(v), len(v)))
                for c in range(G):
                    if int(t[c, k]) > 0:
                        cta[k][c].append(int(t[c, k]) - t0)
            ends.append(max(int(x) - t0 for x in t.reshape(-1) if int(x) > 0) / 1e3)
        print(f"\n[{args.mode}] {sh} {args.file} defs={args.defs}: G={G} cpr={cpr} (us from first CTA start; median over {args.reps} reps)")
        rows = []
        for k in range(32):
            if per[k]:
                mn = statistics.median(x[0] for x in per[k]) / 1e3
                md = statistics.median(x[1] for x in per[k]) / 1e3
                mx = statistics.median(x[2] for x in per[k]) / 1e3
                n = statistics.median(x[3] for x in per[k])
                rows.append((md, k, mn, mx, n))
        for md, k, mn, mx, n in sorted(rows):
            print(f"  {k:2d} {NAMES[k]:9s} min {mn:6.2f}  med {md:6.2f}  max {mx:6.2f}   (CTAs {n:.0f})")
        print(f"  last stamp: med {statistics.median(ends):.2f} max {max(ends):.2f}")
        for k in raw:
            if rawv[k]:
                print(f"  raw slot {k}: median over CTAs (median over reps) {statistics.median(rawv[k]):.0f}, max {statistics.median(rawmx[k]):.0f}")
        if args.percta:
            ks = [k for _, k, _, _, _ in sorted(rows)]
            print("  per-CTA medians: cta " + " ".join(f"{NAMES[k][:7]:>7s}" for k in ks))
            for c in range(G):
                vals = []
                for k in ks:
                    v = cta[k][c]
                    vals.append(f"{statistics.median(v) / 1e3:7.2f}" if v else "      -")
                print(f"   {c:3d} " + " ".join(vals))


if __name__ == "__main__":
    main()
