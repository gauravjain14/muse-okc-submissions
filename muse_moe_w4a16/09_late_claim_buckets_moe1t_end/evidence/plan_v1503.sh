# planner 2026-11-01: v1503 (= v1502 + WP-TUNE W075 + WU0) validation; position-bias-safe (fillers at both ends)
set -o pipefail
Q="--query-gpu=name,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader"
echo "GPU_BEFORE $(nvidia-smi $Q)"
cp submission.py /tmp/v969s.py
echo "STEP smoke"
o=$(timeout 300 python3 dev/hang1.py dev/moe_v1503.py 1,2,4,6,8,9,16,17,32,33,64,65,128,129,256,257 2>&1 | grep -a -E 'bs |HUNG|FAIL|Error|Traceback'); echo "$o" | grep -a -c "ok rel"
echo "$o" | grep -q -E "HUNG|FAIL|Traceback" && { echo "$o" | tail -5; echo "STOP smoke"; echo "STEP done"; exit 3; }
echo "STEP check_mt v1503"
timeout 1500 python3 dev/check_mt.py dev/moe_v1503.py 1 2 3 4 5 6 7 8 9 12 16 17 32 33 64 65 100 128 129 200 256 257 2>&1 | grep -a -E 'FAILURES|FAIL |Error|Traceback' | tail -4
V="fillA|dev/moe_v1503.py|;v969|/tmp/v969s.py|;v1502|dev/moe_v1502.py|;v1503|dev/moe_v1503.py|;v1502b|dev/moe_v1502.py|;v1503b|dev/moe_v1503.py|;fillB|dev/moe_v1502.py|"
echo "STEP bench all shapes, 60 reps (controls in the middle): $V"
B2_SHARE_EQ=1 B2_SEEDS=dev/jm_seeds.jseeds timeout 3000 python3 dev/bench2.py - --variants "$V" --reps 60 --shapes 1,8,16,32,128,256 --out /tmp/v1503.json 2>&1 | grep -a -E "geomean" | head -10
python3 dev/project2.py /tmp/v1503.json v969
echo "GPU_AFTER $(nvidia-smi $Q)"
echo "STEP done"
