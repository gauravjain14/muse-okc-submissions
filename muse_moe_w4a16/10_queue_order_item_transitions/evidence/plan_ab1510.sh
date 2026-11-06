# planner 2026-11-01: high-replication local A/B v1504 vs v1510 (4 copies each, fillers at both ends), all six shapes
set -o pipefail
Q="--query-gpu=name,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader"
echo "GPU_BEFORE $(nvidia-smi $Q)"
V="fA|dev/moe_v1510.py|;fA2|dev/moe_v1504.py|;a1|dev/moe_v1504.py|;b1|dev/moe_v1510.py|;a2|dev/moe_v1504.py|;b2|dev/moe_v1510.py|;a3|dev/moe_v1504.py|;b3|dev/moe_v1510.py|;a4|dev/moe_v1504.py|;b4|dev/moe_v1510.py|;fB2|dev/moe_v1510.py|;fB|dev/moe_v1504.py|"
echo "STEP bench all shapes, 100 reps: $V"
B2_SHARE_EQ=1 B2_SEEDS=dev/jm_seeds.jseeds timeout 5400 python3 dev/bench2.py - --variants "$V" --reps 100 --shapes 1,8,16,32,128,256 --out /tmp/ab1510.json 2>&1 | grep -a -E "geomean" | head -14
python3 - <<'PY'
import json,statistics,math
r=json.load(open('/tmp/ab1510.json'))['shapes']
A=['a1','a2','a3','a4']; B=['b1','b2','b3','b4']
tot=0
for bs,rows in sorted(r.items(),key=lambda kv:int(kv[0])):
    # per rep: mean over A copies vs mean over B copies; median of paired ratios
    rat=statistics.median([statistics.mean(row[a] for a in A)/statistics.mean(row[b] for b in B) for row in rows])
    per=[statistics.median([row['a1']/row[b] for row in rows]) for b in B]
    tot+=math.log(rat)
    print(f"bs{bs}: v1510 vs v1504 (mean of 4 copies each) {100*(rat-1):+.2f}%   (b_i vs a1: "+" ".join(f"{100*(p-1):+.2f}" for p in per)+")")
print(f"geomean v1510 vs v1504: {100*(math.exp(tot/6)-1):+.2f}%")
PY
echo "GPU_AFTER $(nvidia-smi $Q)"
echo "STEP done"
