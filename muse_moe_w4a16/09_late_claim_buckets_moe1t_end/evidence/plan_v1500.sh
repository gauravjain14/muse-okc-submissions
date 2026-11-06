# planner 2026-11-01: v1500 (= v1033 + v1111 INC2 + moe1t FXSS) validation + projection
set -o pipefail
Q="--query-gpu=name,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader"
echo "GPU_BEFORE $(nvidia-smi $Q)"
cp submission.py /tmp/v969s.py
echo "STEP smoke"
o=$(timeout 300 python3 dev/hang1.py dev/moe_v1500.py 1,2,4,8,9,16,17,32,33,64,65,128,129,256,257 2>&1 | grep -a -E 'bs |HUNG|FAIL|Error|Traceback'); echo "$o" | tail -16
echo "$o" | grep -q -E "HUNG|FAIL|Traceback" && { echo "STOP smoke"; echo "STEP done"; exit 3; }
echo "STEP check_mt"
timeout 1200 python3 dev/check_mt.py dev/moe_v1500.py 1 2 4 8 9 12 16 17 32 33 64 65 128 129 256 257 2>&1 | grep -a -E 'FAILURES|FAIL |worst|max|Error|Traceback' | tail -6
V="v969|/tmp/v969s.py|;v969b|/tmp/v969s.py|;v1500|dev/moe_v1500.py|;v1500b|dev/moe_v1500.py|"
echo "STEP bench all shapes, 80 reps: $V"
B2_SHARE_EQ=1 B2_SEEDS=dev/jm_seeds.jseeds timeout 2400 python3 dev/bench2.py - --variants "$V" --reps 80 --shapes 1,8,16,32,128,256 --out /tmp/v1500.json 2>&1 | grep -a -v 'gpu-health\|shares' | grep -a -E "geomean|bs |decode|verify|v969|v1500|sglang" | head -40
python3 dev/project2.py /tmp/v1500.json v969
echo "GPU_AFTER $(nvidia-smi $Q)"
echo "STEP done"
