"""Same-job A/B deltas from the era-0 stacking verifies (out/verify_kNN.txt, dev/coord/verify_stack.sh style).

python verify_chain.py verify_k20.txt verify_k21.txt ...

Each verify job ran check.py and then tools/bench.py with three candidates in ONE container: the previous build,
the new build kN and an identical copy of the previous build (label ending in 'c'). For every file this prints the new
build's per-shape time minus the mean of the two control copies, and its local geomean (and judge-projected geomean,
+1 us per call) relative to the controls. The chained product over the files is the local gain of the stack.
verify_k25.txt: its first control copy (k23) read d1s 5.66 us against 3.49 for the identical k23c copy (a one-off
outlier), so only k23c is used there.
"""
import math
import re
import sys

SHAPE = {("decode", "1", "sliding"): "d1s", ("decode", "1", "full"): "d1f", ("decode", "16", "sliding"): "d16s",
         ("decode", "16", "full"): "d16f", ("decode", "32", "sliding"): "d32s", ("decode", "32", "full"): "d32f",
         ("verify", "1", "sliding"): "v1s", ("verify", "1", "full"): "v1f", ("verify", "16", "sliding"): "v16s",
         ("verify", "16", "full"): "v16f", ("verify", "32", "sliding"): "v32s", ("verify", "32", "full"): "v32f"}
ORDER = list(SHAPE.values())


def parse(path):
    times, geo, labels = {}, {}, []
    for line in open(path, errors="replace"):
        m = re.match(r"(decode|verify) (\d+)\s+(sliding|full)\s+w\d+ sglang\s+[\d.]+ \|(.*)sol", line)
        if m:
            sh = SHAPE[(m.group(1), m.group(2), m.group(3))]
            for lab, t in re.findall(r"(\S+)\s+([\d.]+) x[\d.]+", m.group(4)):
                times.setdefault(lab, {})[sh] = float(t)
                if lab not in labels:
                    labels.append(lab)
        m = re.match(r"GEOMEAN (\S+): x([\d.]+).*judge-projected \(\+1.0 us/call\) x([\d.]+)", line)
        if m:
            geo[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return labels, times, geo


def main(files):
    tot = totj = 0.0
    shape_sum = {sh: 0.0 for sh in ORDER}
    for f in files:
        labels, times, geo = parse(f)
        cand = re.search(r"verify_(k\d+[a-z]?)", f).group(1)
        ctl = [l for l in labels if l != cand]
        if cand == "k25":
            ctl = [l for l in ctl if l.endswith("c")]
        g = math.log(geo[cand][0] / (sum(geo[c][0] for c in ctl) / len(ctl)))
        gj = math.log(geo[cand][1] / (sum(geo[c][1] for c in ctl) / len(ctl)))
        tot += g
        totj += gj
        deltas = []
        for sh in ORDER:
            d = times[cand][sh] - sum(times[c][sh] for c in ctl) / len(ctl)
            shape_sum[sh] += d
            if abs(d) >= 0.25:
                deltas.append(f"{sh} {d:+.2f}")
        print(f"{cand} vs {'/'.join(ctl)}: local geomean {g * 100:+.2f} %, judge-projected {gj * 100:+.2f} % | "
              + ", ".join(deltas))
    if len(files) > 1:
        print(f"chained: local {(math.exp(tot) - 1) * 100:+.2f} %, judge-projected {(math.exp(totj) - 1) * 100:+.2f} %")
        print("sum of same-job deltas per shape (us): " + ", ".join(f"{sh} {v:+.2f}" for sh, v in shape_sum.items()))


if __name__ == "__main__":
    main(sys.argv[1:])
