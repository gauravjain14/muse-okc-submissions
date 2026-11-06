"""Per-shape raw judge numbers from era-0 okc result JSONs (out/judge_kNN.json).

python3 judge_raw.py <judge_kA.json> [<judge_kB.json> ...] [--shapes d16f,d32f]

For each shape: our raw median (graph replay, before the fixed-cost subtraction), the judge's reported time
("yours" = SGLang median / paired-ratio median), SGLang's raw median, how many of SGLang's reps were slow
(> 1.15 x its best rep), the paired-ratio speedup and the speedup of medians. When "yours" moves while our raw
median does not, the reading moved because of SGLang's rep distribution, not our kernel.
"""
import json
import sys

NAMES = ["d1s", "d1f", "d16s", "d16f", "d32s", "d32f", "v1s", "v1f", "v16s", "v16f", "v32s", "v32f"]
args = [a for a in sys.argv[1:] if not a.startswith("--")]
shapes = NAMES
for a in sys.argv[1:]:
    if a.startswith("--shapes="):
        shapes = a.split("=", 1)[1].split(",")
for path in args:
    j = json.load(open(path))
    recs = [r for r in j["result"]["records"] if r.get("event") == "benchmark"]
    print(f"{path}: job {j['id']} speedup {j['result']['speedup']:.4f}")
    print("  shape   ours_raw  yours(judge)  sgl_raw  sgl_slow/reps  paired  of_medians")
    for r in recs:
        n = NAMES[r["index"]]
        if n not in shapes:
            continue
        b = r["baseline_runs_ns"]
        best = min(b)
        slow = sum(1 for x in b if x > 1.15 * best)
        print(f"  {n:5s} {r['median_raw_ns']/1e3:9.2f} {r['median_ns']/1e3:12.2f} {r['baseline_raw_ns']/1e3:9.2f}"
              f"  {slow:4d}/{len(b):<4d}    {r['paired_ratio']:.4f}  {r['speedup_of_medians']:.4f}")
