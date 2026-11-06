# planner 2026-11-01: WU0 x WP-IT interaction at bs128/256 (v1503 = WU0, v1504 = WU0 + IT, v1505 = IT without WU0)
set -o pipefail
Q="--query-gpu=name,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader"
echo "GPU_BEFORE $(nvidia-smi $Q)"
echo "STEP smoke"
for f in dev/moe_v1504.py dev/moe_v1505.py; do o=$(timeout 300 python3 dev/hang1.py $f 1,8,16,32,64,65,128,129,256,257 2>&1 | grep -a -E 'bs |HUNG|FAIL|Error|Traceback'); echo "$f ok=$(echo "$o" | grep -a -c 'ok rel')"; echo "$o" | grep -q -E "HUNG|FAIL|Traceback" && { echo "STOP smoke $f"; echo "STEP done"; exit 3; }; done
echo "STEP check_mt"
for f in dev/moe_v1504.py dev/moe_v1505.py; do timeout 900 python3 dev/check_mt.py $f 64 65 100 128 129 200 256 257 2>&1 | grep -a -E 'FAILURES|FAIL |Error|Traceback' | tail -2; done
V="fA|dev/moe_v1505.py|;v1503|dev/moe_v1503.py|;v1504|dev/moe_v1504.py|;v1505|dev/moe_v1505.py|;v1503b|dev/moe_v1503.py|;v1504b|dev/moe_v1504.py|;v1505b|dev/moe_v1505.py|;fB|dev/moe_v1503.py|"
echo "STEP bench bs128/256 (+bs32 placebo), 100 reps: $V"
B2_SHARE_EQ=1 B2_SEEDS=dev/jm_seeds.jseeds timeout 2700 python3 dev/bench2.py - --variants "$V" --reps 100 --shapes 128,256,32 --out /tmp/v1505.json 2>&1 | grep -a -E "geomean" | head -12
python3 - <<'PY'
import json,statistics
r=json.load(open('/tmp/v1505.json'))['shapes']
for bs,rows in sorted(r.items(),key=lambda kv:int(kv[0])):
    labs=['v1503','v1504','v1505','v1503b','v1504b','v1505b']
    ref=[statistics.median([row['v1503']/row[l] for row in rows]) for l in labs]
    print(f"bs{bs}: "+"  ".join(f"{l}:{x:.4f}" for l,x in zip(labs,ref)))
PY
echo "GPU_AFTER $(nvidia-smi $Q)"
echo "STEP done"
