"""Re-derive the judge score from a judge report's per-case table, and attribute a score change to shapes.

    python judge_score.py RUN.txt                 # per-load speedups and the geomean (matches the report's speedup)
    python judge_score.py BEFORE.txt AFTER.txt    # + each shape's share: AFTER's time put into BEFORE, one shape at a time

Load = 46 x sliding + 16 x full; load speedup = SGLang load time / ours; score = geomean over the 6 loads.
Written for this record (not part of the original lab tree); input = the okc report .txt files in this folder.
"""
import math
import re
import sys

NAMES = ["d1s", "d1f", "d16s", "d16f", "d32s", "d32f", "v1s", "v1f", "v16s", "v16f", "v32s", "v32f"]
LOADS = [("d1s", "d1f"), ("d16s", "d16f"), ("d32s", "d32f"), ("v1s", "v1f"), ("v16s", "v16f"), ("v32s", "v32f")]


def read(path):
    rows = re.findall(r"^batch=.*?\s+([\d.]+)\s+([\d.]+)\s+[\d.]+x\s+[\d.]+\s*$", open(path).read(), re.M)
    assert len(rows) == 12, (path, len(rows))
    return {n: (float(b), float(o)) for n, (b, o) in zip(NAMES, rows)}


def score(r):
    sp = [(46 * r[s][0] + 16 * r[f][0]) / (46 * r[s][1] + 16 * r[f][1]) for s, f in LOADS]
    return math.exp(sum(map(math.log, sp)) / len(sp)), sp


def main():
    runs = [read(p) for p in sys.argv[1:]]
    for p, r in zip(sys.argv[1:], runs):
        g, sp = score(r)
        print(f"{p}: score {g:.4f} | loads " + " ".join(f"{s[0][:-1]} {x:.3f}" for s, x in zip(LOADS, sp)))
    if len(runs) == 2:
        a, b = runs
        g0, _ = score(a)
        print("shape   before  after   delta   score share (after's time alone)")
        for n in NAMES:
            r = dict(a)
            r[n] = (a[n][0], b[n][1])
            print(f"{n:6s} {a[n][1]:7.2f} {b[n][1]:7.2f} {b[n][1] - a[n][1]:+7.2f}   {100 * (score(r)[0] / g0 - 1):+6.2f} %")


if __name__ == "__main__":
    main()
