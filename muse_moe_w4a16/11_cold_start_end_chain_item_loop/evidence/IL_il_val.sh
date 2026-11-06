#!/bin/bash
# IL final validation: candidate C (moe_il_6 = K53248) and W (moe_il_5 = K20480) vs moe_i3, ALL shapes: hang1 16 sizes,
# check_mt full list (C) / bs129-256 sizes (W), bench2 jm 60 reps, C x3 + W x2 + base x3 interleaved, fillers both ends,
# one v1504 copy for the judge-scale projection.
set -o pipefail
cd /root/okc
Q="--query-gpu=name,clocks.sm,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader"
echo "GPU_BEFORE $(nvidia-smi $Q)"
B=t3/INT/moe_i3.py; C=t3/IL/moe_il_6.py; W=t3/IL/moe_il_5.py
for F in $C $W; do
  o=$(timeout 300 python3 dev/hang1.py $F 1,2,4,8,9,16,17,32,33,64,65,128,129,256,257,512 2>&1 | grep -a -E 'bs |HUNG|FAIL|Error|Traceback')
  echo "hang1 $F: $(echo "$o" | grep -a -c finished) finished, $(echo "$o" | grep -a -c ' ok ') ok"
  echo "$o" | grep -a -E "HUNG|FAIL|Error|Traceback" && { echo "STOP smoke $F"; exit 3; }
done
echo "check_mt $C (full): $(timeout 1500 python3 dev/check_mt.py $C 1 2 4 8 9 16 17 32 33 64 65 128 129 200 256 257 512 2>&1 | grep -a -E 'FAILURES|Traceback|Error' | tail -2)"
echo "check_mt $W (129 200 256): $(timeout 600 python3 dev/check_mt.py $W 129 200 256 2>&1 | grep -a -E 'FAILURES|Traceback|Error' | tail -2)"
V="fA|$B|;a1|$B|;c1|$C|;w1|$W|;a2|$B|;c2|$C|;w2|$W|;a3|$B|;c3|$C|;p1|dev/moe_v1504.py|;fB|$B|"
B2_SHARE_EQ=1 B2_SEEDS=dev/jm_seeds.jseeds timeout 1800 python3 dev/bench2.py - --variants "$V" --reps 60 --shapes 1,8,16,32,128,256 --out /tmp/il_val.json 2>&1 | grep -a -E "FAIL|Error|Traceback" | head -10
python3 t3/IL/il_tab.py /tmp/il_val.json a1,a2,a3
PJ_ANCHOR=v1504 python3 dev/project2.py /tmp/il_val.json p1 2>&1 | tail -12
echo "GPU_AFTER $(nvidia-smi $Q)"
