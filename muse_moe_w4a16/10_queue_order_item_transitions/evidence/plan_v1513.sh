# planner 2026-11-01: v1513 (= v1512 + WP-MIX mix5) validation + stacked line-B vs scored v1504, all six shapes
set -o pipefail
Q="--query-gpu=name,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader"
idle() { echo "IDLE $1: [$(timeout 4 nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader -lms 250 | head -12 | awk -F, '{print $1}' | tr -d ' ' | tr '\n' ' ')]"; }
echo "GPU_BEFORE $(nvidia-smi $Q)"
echo "STEP smoke"
o=$(timeout 300 python3 dev/hang1.py dev/moe_v1513.py 1,2,4,8,9,16,17,32,33,64,65,128,129,256,257,512 2>&1 | grep -a -E 'bs |HUNG|FAIL|Error|Traceback'); echo "ok=$(echo "$o" | grep -a -c 'ok rel') of 16"
echo "$o" | grep -q -E "HUNG|FAIL|Traceback" && { echo "$o"; echo "STOP smoke"; echo "STEP done"; exit 3; }
echo "STEP check_mt v1513"
timeout 1800 python3 dev/check_mt.py dev/moe_v1513.py 1 2 8 9 16 17 32 33 64 65 100 128 129 160 200 256 257 300 512 2>&1 | grep -a -E 'FAILURES|FAIL |Error|Traceback' | tail -4
echo "STEP interleaved calls"
timeout 900 python3 dev/wpmix_calls.py dev/moe_v1513.py 2>&1 | tail -2
idle bench
V="fA|dev/moe_v1513.py|;fA2|dev/moe_v1504.py|;a1|dev/moe_v1504.py|;b1|dev/moe_v1513.py|;a2|dev/moe_v1504.py|;b2|dev/moe_v1513.py|;a3|dev/moe_v1504.py|;b3|dev/moe_v1513.py|;a4|dev/moe_v1504.py|;b4|dev/moe_v1513.py|;fB2|dev/moe_v1513.py|;fB|dev/moe_v1504.py|"
echo "STEP bench all shapes, 100 reps: $V"
B2_SHARE_EQ=1 B2_SEEDS=dev/jm_seeds.jseeds timeout 5400 python3 dev/bench2.py - --variants "$V" --reps 100 --shapes 1,8,16,32,128,256 --out /tmp/v1513.json 2>&1 | grep -a -E "geomean" | head -14
python3 - <<'PY'
import json,statistics,math
r=json.load(open('/tmp/v1513.json'))['shapes']
A=['a1','a2','a3','a4']; B=['b1','b2','b3','b4']; tot=0
for bs,rows in sorted(r.items(),key=lambda kv:int(kv[0])):
    rat=statistics.median([statistics.mean(row[a] for a in A)/statistics.mean(row[b] for b in B) for row in rows]); tot+=math.log(rat)
    per=[statistics.median([statistics.mean(row[a] for a in A)/row[b] for row in rows]) for b in B]
    ctl=[statistics.median([statistics.mean(row[a] for a in A)/row[a2] for row in rows]) for a2 in A]
    print(f"bs{bs}: v1513 vs v1504 {100*(rat-1):+.2f}%  copies "+" ".join(f"{100*(p-1):+.2f}" for p in per)+" | v1504 controls "+" ".join(f"{100*(p-1):+.2f}" for p in ctl))
print(f"geomean v1513 vs v1504: {100*(math.exp(tot/6)-1):+.2f}%")
PY
PJ_ANCHOR=v1504 python3 dev/project2.py /tmp/v1513.json a1 2>&1 | tail -4
echo "GPU_AFTER $(nvidia-smi $Q)"
echo "STEP done"
