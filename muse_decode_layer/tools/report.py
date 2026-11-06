"""Tables from bench.py JSON (one or more runs of the same candidates; per-shape values are averaged over runs).

    python3 cdl/H/report.py run1.json [run2.json ...] [--cand d23] [--md]

Per shape: weight, floor, local SGLang, judge SGLang (scored d09 run), candidate time (local), local ratio,
judge-projected time two ways (judge SGLang / local ratio; and via the judged anchor d09: its judge time /
local ratio vs d09), time over floor, and the score impact of 1 us (weight / load time / 6).
"""
import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--cand", default=None)
    p.add_argument("--md", action="store_true")
    a = p.parse_args()
    runs = [json.loads(Path(f).read_text()) for f in a.runs]
    labels = [s["label"] for s in runs[0]["sides"] if s["label"] != "sglang"]
    cands = [a.cand] if a.cand else labels
    sep = " | " if a.md else "  "
    for cand in cands:
        rows = []
        for si, name in enumerate(C.SHORT):
            recs = [r["shapes"][name] for r in runs if name in r["shapes"]]
            if not recs:
                continue
            res = [x["results"][cand] for x in recs]
            row = {
                "shape": name, "w": C.SCORED[si]["weight"], "floor": C.FLOOR_US[si],
                "sgl": mean(x.get("sglang_ns") for x in recs) / 1e3,
                "jsgl": C.JUDGE_D09_SGLANG_NS[si] / 1e3,
                "t": mean(x.get("time_ns") for x in res) / 1e3,
                "ratio": math.exp(mean(math.log(x["speedup"]) for x in res)),
                "proj": mean(x.get("proj_ns") for x in res) / 1e3,
                "anch": (mean(x.get("anchor_ns") for x in res) or 0) / 1e3 or None,
                "ovh": mean(x["ovh_ns"] for x in recs) / 1e3,
            }
            row["jratio"] = C.JUDGE_D09_RATIO[si] if cand == "d09" else (C.JUDGE_D01_RATIO[si] if cand == "d01" else None)
            rows.append(row)
        print(f"\n### {cand} ({len(runs)} run(s): router seeds {[r.get('router_seed') for r in runs]})\n")
        hdr = ["shape", "w", "floor", "SGL loc", "SGL jdg", "cand loc", "x loc", "proj(SGL)", "proj(d09)", "x proj",
               "d09 x jdg", "/floor", "score%/us"]
        if a.md:
            print("| " + " | ".join(hdr) + " |")
            print("|" + "---|" * len(hdr))
        else:
            print("  ".join(f"{h:>9s}" for h in hdr))
        loads = {}
        for li, load in enumerate(C.LOADS):
            idx = [r for r in rows if C.SHORT.index(r["shape"]) in (2 * li, 2 * li + 1)]
            if len(idx) == 2:
                tl = sum(r["w"] * (r["anch"] or r["proj"]) for r in idx)
                loads[load] = tl
        for r in rows:
            li = C.SHORT.index(r["shape"]) // 2
            tl = loads.get(C.LOADS[li])
            imp = (r["w"] / tl / 6 * 100) if tl else None  # score % per us of this shape
            best = r["anch"] or r["proj"]
            si = C.SHORT.index(r["shape"])
            cells = [r["shape"], f"{r['w']}", f"{r['floor']:.1f}", f"{r['sgl']:.1f}", f"{r['jsgl']:.1f}", f"{r['t']:.1f}",
                     f"{r['ratio']:.3f}", f"{r['proj']:.1f}", "" if r["anch"] is None else f"{r['anch']:.1f}",
                     f"{r['jsgl'] / best:.3f}", f"{C.JUDGE_D09_RATIO[si]:.3f}", f"{best / r['floor']:.2f}",
                     "" if imp is None else f"{imp:.3f}"]
            if a.md:
                print("| " + " | ".join(cells) + " |")
            else:
                print("  ".join(f"{c:>9s}" for c in cells))
        # loads
        print()
        lh = ["load", "SGL jdg us", "proj(SGL) us", "x", "proj(d09) us", "x", "floor us", "x@floor"]
        if a.md:
            print("| " + " | ".join(lh) + " |")
            print("|" + "---|" * len(lh))
        sp_s, sp_a, sp_f = [], [], []
        for li, load in enumerate(C.LOADS):
            idx = [r for r in rows if C.SHORT.index(r["shape"]) in (2 * li, 2 * li + 1)]
            if len(idx) != 2:
                continue
            jb = sum(r["w"] * r["jsgl"] for r in idx)
            ps = sum(r["w"] * r["proj"] for r in idx)
            pa = sum(r["w"] * r["anch"] for r in idx) if all(r["anch"] for r in idx) else None
            fl = sum(r["w"] * r["floor"] for r in idx)
            sp_s.append(jb / ps)
            sp_f.append(jb / fl)
            if pa:
                sp_a.append(jb / pa)
            cells = [load, f"{jb:.0f}", f"{ps:.0f}", f"{jb/ps:.4f}", "" if pa is None else f"{pa:.0f}",
                     "" if pa is None else f"{jb/pa:.4f}", f"{fl:.0f}", f"{jb/fl:.3f}"]
            print(("| " + " | ".join(cells) + " |") if a.md else "  ".join(f"{c:>12s}" for c in cells))
        if len(sp_s) == 6:
            print(f"\nscore: proj(SGL) {C.geomean(sp_s):.4f}" + (f"  proj(d09) {C.geomean(sp_a):.4f}" if len(sp_a) == 6 else "")
                  + f"  floor {C.geomean(sp_f):.4f}   (judge d09 scored 2.1817, d01 check 2.0568)")


if __name__ == "__main__":
    main()
