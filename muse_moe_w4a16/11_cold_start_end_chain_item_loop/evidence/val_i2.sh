# coordinator: validate t3/INT/moe_i2.py (= v4111 + SU prologue (bs8/16) + B1 (bs1)); A/B vs v4111 and v1504
set -o pipefail
Q="--query-gpu=name,clocks.sm,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader"
echo "GPU_BEFORE $(nvidia-smi $Q)"
o=$(timeout 300 python3 dev/hang1.py t3/INT/moe_i2.py 1,2,4,8,9,16,17,32,33,64,65,128,129,256,257,512 2>&1 | grep -a -E 'bs |HUNG|FAIL|Error|Traceback'); echo "hang1 ok=$(echo "$o" | grep -a -c 'ok rel') of 16"
echo "$o" | grep -q -E "HUNG|FAIL|Traceback" && { echo "$o"; echo STOP; exit 3; }
timeout 1800 python3 dev/check_mt.py t3/INT/moe_i2.py 1 2 4 8 9 16 17 32 33 64 65 128 129 200 256 257 512 2>&1 | grep -a -E 'FAILURES|FAIL |Error|Traceback|worst' | tail -4
V="fA|t3/INT/moe_i2.py|;a1|dev/moe_v1504.py|;b1|dev/moe_v4111.py|;c1|t3/INT/moe_i2.py|;b2|dev/moe_v4111.py|;c2|t3/INT/moe_i2.py|;a2|dev/moe_v1504.py|;fB|dev/moe_v4111.py|"
B2_SHARE_EQ=1 B2_SEEDS=dev/jm_seeds.jseeds timeout 3000 python3 dev/bench2.py - --variants "$V" --reps 60 --shapes 1,8,16,32,128,256 --out /tmp/val_i2.json 2>&1 | grep -a -E "geomean" | head -10
python3 - <<'PY'
import json,statistics,math
r=json.load(open('/tmp/val_i2.json'))['shapes']
B=['b1','b2']; C=['c1','c2']; tot=0
for bs,rows in sorted(r.items(),key=lambda kv:int(kv[0])):
    rat=statistics.median([statistics.mean(row[b] for b in B)/statistics.mean(row[c] for c in C) for row in rows]); tot+=math.log(rat)
    per=[statistics.median([statistics.mean(row[b] for b in B)/row[c] for row in rows]) for c in C]
    ctl=[statistics.median([statistics.mean(row[b] for b in B)/row[b2] for row in rows]) for b2 in B]
    print(f"bs{bs}: i2 vs v4111 {100*(rat-1):+.2f}%  copies "+" ".join(f"{100*(p-1):+.2f}" for p in per)+" | v4111 ctl "+" ".join(f"{100*(p-1):+.2f}" for p in ctl))
print(f"geomean i2 vs v4111: {100*(math.exp(tot/6)-1):+.2f}%")
PY
PJ_ANCHOR=v1504 python3 dev/project2.py /tmp/val_i2.json a1 2>&1 | tail -9
echo "GPU_AFTER $(nvidia-smi $Q)"
