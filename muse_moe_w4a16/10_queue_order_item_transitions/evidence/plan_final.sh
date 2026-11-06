# planner 2026-11-01: FINAL stacked A/B: v4111 (line B = v1514 + WP-FAST GSD) and v4110 (+ bs8 start set) vs scored v1504
set -o pipefail
Q="--query-gpu=name,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader"
idle() { echo "IDLE $1: [$(timeout 4 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -lms 250 | head -12 | tr -d ' %' | tr '\n' ' ')]"; }
echo "GPU_BEFORE $(nvidia-smi $Q)"
for V in v4111 v4110; do
  echo "STEP smoke $V"
  o=$(timeout 300 python3 dev/hang1.py dev/moe_$V.py 1,2,4,8,9,16,17,32,33,64,65,128,129,256,257,512 2>&1 | grep -a -E 'bs |HUNG|FAIL|Error|Traceback'); echo "ok=$(echo "$o" | grep -a -c 'ok rel') of 16"
  echo "$o" | grep -q -E "HUNG|FAIL|Traceback" && { echo "$o"; echo "STOP smoke $V"; echo "STEP done"; exit 3; }
  echo "STEP check_mt $V"
  timeout 1800 python3 dev/check_mt.py dev/moe_$V.py 1 2 8 9 16 17 32 33 64 65 128 129 200 256 257 512 2>&1 | grep -a -E 'FAILURES|FAIL |Error|Traceback' | tail -3
done
idle bench
V="fA|dev/moe_v4111.py|;fA2|dev/moe_v1504.py|;a1|dev/moe_v1504.py|;b1|dev/moe_v4111.py|;c1|dev/moe_v4110.py|;a2|dev/moe_v1504.py|;b2|dev/moe_v4111.py|;c2|dev/moe_v4110.py|;a3|dev/moe_v1504.py|;b3|dev/moe_v4111.py|;c3|dev/moe_v4110.py|;a4|dev/moe_v1504.py|;b4|dev/moe_v4111.py|;c4|dev/moe_v4110.py|;fB2|dev/moe_v4110.py|;fB|dev/moe_v1504.py|"
echo "STEP bench all shapes, 100 reps: $V"
B2_SHARE_EQ=1 B2_SEEDS=dev/jm_seeds.jseeds timeout 6000 python3 dev/bench2.py - --variants "$V" --reps 100 --shapes 1,8,16,32,128,256 --out /tmp/final.json 2>&1 | grep -a -E "geomean" | head -18
python3 - <<'PY'
import json,statistics,math
r=json.load(open('/tmp/final.json'))['shapes']
A=['a1','a2','a3','a4']
for lab,B in (("v4111",['b1','b2','b3','b4']),("v4110",['c1','c2','c3','c4'])):
    tot=0
    for bs,rows in sorted(r.items(),key=lambda kv:int(kv[0])):
        rat=statistics.median([statistics.mean(row[a] for a in A)/statistics.mean(row[b] for b in B) for row in rows]); tot+=math.log(rat)
        per=[statistics.median([statistics.mean(row[a] for a in A)/row[b] for row in rows]) for b in B]
        ctl=[statistics.median([statistics.mean(row[a] for a in A)/row[a2] for row in rows]) for a2 in A]
        print(f"{lab} bs{bs}: vs v1504 {100*(rat-1):+.2f}%  copies "+" ".join(f"{100*(p-1):+.2f}" for p in per)+" | v1504 controls "+" ".join(f"{100*(p-1):+.2f}" for p in ctl))
    print(f"{lab} geomean vs v1504: {100*(math.exp(tot/6)-1):+.2f}%")
PY
echo "GPU_AFTER $(nvidia-smi $Q)"
echo "STEP done"
