# coordinator 2026-11-03: sandbox calibration: v4111 (best local) vs v1504 (scored 5157.2), fillers both ends
set -o pipefail
Q="--query-gpu=name,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader"
echo "GPU_BEFORE $(nvidia-smi $Q)"
timeout 300 python3 dev/hang1.py dev/moe_v4111.py 1,8,16,32,128,256 2>&1 | grep -a -E 'bs |HUNG|FAIL|Error|Traceback' | head -8
V="fA|dev/moe_v4111.py|;a1|dev/moe_v1504.py|;b1|dev/moe_v4111.py|;a2|dev/moe_v1504.py|;b2|dev/moe_v4111.py|;fB|dev/moe_v1504.py|"
B2_SHARE_EQ=1 B2_SEEDS=dev/jm_seeds.jseeds timeout 3000 python3 dev/bench2.py - --variants "$V" --reps 60 --shapes 1,8,16,32,128,256 --out /tmp/calib.json 2>&1 | grep -a -E "geomean" | head -8
python3 - <<'PY'
import json,statistics,math
r=json.load(open('/tmp/calib.json'))['shapes']
A=['a1','a2']; B=['b1','b2']; tot=0
for bs,rows in sorted(r.items(),key=lambda kv:int(kv[0])):
    rat=statistics.median([statistics.mean(row[a] for a in A)/statistics.mean(row[b] for b in B) for row in rows]); tot+=math.log(rat)
    t=statistics.median([statistics.mean(row[b] for b in B) for row in rows])
    print(f"bs{bs}: v4111 vs v1504 {100*(rat-1):+.2f}%  v4111 median {t:.2f} us")
print(f"geomean vs v1504: {100*(math.exp(tot/6)-1):+.2f}%")
PY
PJ_ANCHOR=v1504 python3 dev/project2.py /tmp/calib.json a1 2>&1 | tail -8
echo "GPU_AFTER $(nvidia-smi $Q)"
