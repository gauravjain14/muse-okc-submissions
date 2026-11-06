# planner 2026-11-01: v1507 (= v1506 + WP-S5R bs8) validation; anchor = v1504 scored (5157.2)
set -o pipefail
Q="--query-gpu=name,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader"
echo "GPU_BEFORE $(nvidia-smi $Q)"
echo "STEP smoke"
o=$(timeout 300 python3 dev/hang1.py dev/moe_v1507.py 1,2,4,6,8,9,16,17,32,33,64,65,128,129,256,257 2>&1 | grep -a -E 'bs |HUNG|FAIL|Error|Traceback|S5R|fit'); echo "ok=$(echo "$o" | grep -a -c 'ok rel')"; echo "$o" | grep -a -i -E "s5r|fit" | head -3
echo "$o" | grep -q -E "HUNG|FAIL|Traceback" && { echo "$o" | tail -5; echo "STOP smoke"; echo "STEP done"; exit 3; }
echo "STEP check_mt v1507"
timeout 1500 python3 dev/check_mt.py dev/moe_v1507.py 1 2 3 4 5 6 7 8 9 12 16 17 32 33 64 65 100 128 129 200 256 257 2>&1 | grep -a -E 'FAILURES|FAIL |Error|Traceback' | tail -4
V="fA|dev/moe_v1507.py|;v1504|dev/moe_v1504.py|;v1506|dev/moe_v1506.py|;v1507|dev/moe_v1507.py|;v1504b|dev/moe_v1504.py|;v1506b|dev/moe_v1506.py|;v1507b|dev/moe_v1507.py|;fB|dev/moe_v1506.py|"
echo "STEP bench all shapes, 60 reps (fillers at ends): $V"
B2_SHARE_EQ=1 B2_SEEDS=dev/jm_seeds.jseeds timeout 3300 python3 dev/bench2.py - --variants "$V" --reps 60 --shapes 1,8,16,32,128,256 --out /tmp/v1507.json 2>&1 | grep -a -E "geomean" | head -10
PJ_ANCHOR=v1504 python3 dev/project2.py /tmp/v1507.json v1504
echo "GPU_AFTER $(nvidia-smi $Q)"
echo "STEP done"
