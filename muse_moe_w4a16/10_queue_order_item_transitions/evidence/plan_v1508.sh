# planner 2026-11-01: v1508 (= v1507 + WP-CGS RO bs32) validation + LATE9 (bs8) candidate; anchor = v1504 scored
set -o pipefail
Q="--query-gpu=name,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader"
echo "GPU_BEFORE $(nvidia-smi $Q)"
echo "STEP smoke"
for d in "" "MOE_LATE9=1500,MOE_LATE9_CFG=1"; do o=$(MOE_DEFINES=$d timeout 300 python3 dev/hang1.py dev/moe_v1508.py 1,2,4,8,9,16,17,32,33,64,65,128,129,256,257 2>&1 | grep -a -E 'bs |HUNG|FAIL|Error|Traceback'); echo "[$d] ok=$(echo "$o" | grep -a -c 'ok rel')"; echo "$o" | grep -q -E "HUNG|FAIL|Traceback" && { echo "STOP smoke"; echo "STEP done"; exit 3; }; done
echo "STEP check_mt"
timeout 1500 python3 dev/check_mt.py dev/moe_v1508.py 1 2 4 8 9 16 17 20 24 32 33 64 65 128 129 256 257 2>&1 | grep -a -E 'FAILURES|FAIL |Error|Traceback' | tail -3
MOE_DEFINES=MOE_LATE9=1500,MOE_LATE9_CFG=1 timeout 900 python3 dev/check_mt.py dev/moe_v1508.py 2 3 4 5 6 7 8 2>&1 | grep -a -E 'FAILURES|FAIL |Error|Traceback' | tail -3
L9="MOE_LATE9=1500,MOE_LATE9_CFG=1"
V="fA|dev/moe_v1508.py|;v1507|dev/moe_v1507.py|;v1508|dev/moe_v1508.py|;v1508l9|dev/moe_v1508.py|$L9;v1507b|dev/moe_v1507.py|;v1508b|dev/moe_v1508.py|;v1508l9b|dev/moe_v1508.py|$L9;fB|dev/moe_v1507.py|"
echo "STEP bench all shapes, 60 reps: $V"
B2_SHARE_EQ=1 B2_SEEDS=dev/jm_seeds.jseeds timeout 3300 python3 dev/bench2.py - --variants "$V" --reps 60 --shapes 1,8,16,32,128,256 --out /tmp/v1508.json 2>&1 | grep -a -E "geomean" | head -10
PJ_ANCHOR=v1504 python3 dev/project2.py /tmp/v1508.json v1507
echo "GPU_AFTER $(nvidia-smi $Q)"
echo "STEP done"
