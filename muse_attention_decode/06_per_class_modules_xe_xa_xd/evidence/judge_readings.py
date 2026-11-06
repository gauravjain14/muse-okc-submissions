"""Read okc judge results honestly: per-case before -> after, one-shape-at-a-time score attribution, and the
SGLang-bimodality check on a judge JSON.

python judge_readings.py BEFORE.txt AFTER.txt [MORE.txt ...] [--json RUN.json ...]

- The .txt files are okc profile reports (`okc run <job> ...`, "scored shapes" table: sglang / yours / speedup).
- Score = geomean over the 6 loads of (46 x sliding + 16 x full) SGLang time / ours (TASK.md). The attribution
  swaps ONE shape's "yours" from BEFORE to AFTER with BEFORE's SGLang times held fixed, so it isolates our side.
- --json: per case, our raw median vs the judged "yours" (= SGLang median / paired-ratio speedup) and the number of
  SGLang reps > 1.2 x its first quartile. A judged value far below raw - ~4.4 us (the replay offset) means SGLang's
  bimodal reps flipped the paired median: the reading is luck, not our kernel.
"""
import json
import math
import re
import sys

NAMES = ["d1s", "d1f", "d16s", "d16f", "d32s", "d32f", "v1s", "v1f", "v16s", "v16f", "v32s", "v32f"]


def short(case):
    b = re.search(r"batch=(\d+)", case).group(1)
    lay = "s" if "layer=sliding" in case else "f"
    kind = "v" if "load=verify" in case else "d"
    return f"{kind}{b}{lay}"


def read_txt(path):
    rows = {}
    head = open(path).readline().strip()
    for line in open(path):
        m = re.match(r"(batch=\S+.*?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x\s+([\d.]+)\s*$", line)
        if m:
            rows[short(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    sp = re.search(r"^speedup\s+([\d.]+)", open(path).read(), re.M)
    return head, [rows[n][0] for n in NAMES], [rows[n][1] for n in NAMES], float(sp.group(1))


def score(ours, base):
    s = 0.0
    for i in range(6):
        sl, fu = 2 * i, 2 * i + 1
        s += math.log((46 * base[sl] + 16 * base[fu]) / (46 * ours[sl] + 16 * ours[fu]))
    return math.exp(s / 6)


def main(argv):
    txts = [a for a in argv if a.endswith(".txt")]
    jsons = [a for a in argv if a.endswith(".json")]
    runs = [read_txt(t) for t in txts]
    for (h, b, o, sp) in runs:
        print(f"{h}: speedup {sp:.4f}")
    if len(runs) >= 2:
        (h0, b0, o0, s0), (h1, b1, o1, s1) = runs[0], runs[-1]
        f0, f1 = score(o0, b0), score(o1, b0)
        print(f"\nfirst -> last, SGLang times of the first run held fixed: {f0:.4f} -> {f1:.4f} ({(f1 / f0 - 1) * 100:+.2f} %)")
        print("shape   " + "  ".join(f"{r[0].split()[2][:8]:>9s}" for r in runs) + "   delta    alone")
        for i, n in enumerate(NAMES):
            o = list(o0)
            o[i] = o1[i]
            vals = "  ".join(f"{r[2][i]:9.2f}" for r in runs)
            print(f"{n:6s}  {vals}  {o1[i] - o0[i]:+7.2f}  {(score(o, b0) / f0 - 1) * 100:+6.2f} %")
    for j in jsons:
        d = json.load(open(j))
        print(f"\n{j}: job {d.get('job_id')}  speedup {d['summary']['speedup']:.4f}")
        print("shape   raw med   judged  raw-judged  paired   of-medians  SGLang slow reps")
        for i, bm in enumerate(d["benchmarks"]):
            runs_ns = sorted(bm["baseline_runs_ns"])
            q1 = runs_ns[len(runs_ns) // 4]
            slow = sum(1 for x in bm["baseline_runs_ns"] if x > 1.2 * q1)
            raw, jud = bm["median_raw_ns"] / 1e3, bm["median_ns"] / 1e3
            print(f"{NAMES[i]:6s} {raw:8.2f} {jud:8.2f} {raw - jud:9.2f}  {bm['speedup']:7.3f}  {bm['speedup_of_medians']:9.3f}"
                  f"  {slow:3d} / {len(runs_ns)}")
        # what the run would have read with every case at raw - (median offset of the other cases)
        offs = sorted(bm["median_raw_ns"] / 1e3 - bm["median_ns"] / 1e3 for bm in d["benchmarks"])
        med_off = offs[len(offs) // 2]
        ours = [bm["median_ns"] / 1e3 for bm in d["benchmarks"]]
        base = [bm["baseline_ns"] / 1e3 for bm in d["benchmarks"]]
        fixed = [bm["median_raw_ns"] / 1e3 - med_off if bm["median_raw_ns"] / 1e3 - o > med_off + 5 else o
                 for o, bm in zip(ours, d["benchmarks"])]
        print(f"median raw-judged offset {med_off:.2f} us; score as read {score(ours, base):.4f}, with outlier cases"
              f" set to raw - offset {score(fixed, base):.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
