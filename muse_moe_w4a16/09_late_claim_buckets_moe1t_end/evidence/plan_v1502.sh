# planner 2026-11-01: v1502 (= v1501 + WP-L2 CLD3+CLDI) validation + projection; v1503u bs16 split-rule candidates
set -o pipefail
Q="--query-gpu=name,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader"
echo "GPU_BEFORE $(nvidia-smi $Q)"
cp submission.py /tmp/v969s.py
echo "STEP smoke"
o=$(timeout 300 python3 dev/hang1.py dev/moe_v1502.py 1,2,4,8,9,16,17,32,33,64,65,128,129,256,257 2>&1 | grep -a -E 'bs |HUNG|FAIL|Error|Traceback'); echo "$o" | grep -a -E "ok|HUNG|FAIL" | tail -15
echo "$o" | grep -q -E "HUNG|FAIL|Traceback" && { echo "STOP smoke"; echo "STEP done"; exit 3; }
o=$(MOE_DEFINES=MOE_E2_KSXNT2=2,MOE_S2_SPLW2=1.0f timeout 300 python3 dev/hang1.py dev/moe_v1503u.py 9,12,16 2>&1 | grep -a -E 'bs |HUNG|FAIL|Error|Traceback'); echo "$o" | grep -a -E "ok|HUNG|FAIL" | tail -3
echo "STEP check_mt v1502"
timeout 1200 python3 dev/check_mt.py dev/moe_v1502.py 1 2 4 8 9 12 16 17 32 33 64 65 100 128 129 200 256 257 2>&1 | grep -a -E 'FAILURES|FAIL |Error|Traceback' | tail -4
echo "STEP check_mt v1503u K2 w1.0 / K4 w0.5"
MOE_DEFINES=MOE_E2_KSXNT2=2,MOE_S2_SPLW2=1.0f timeout 600 python3 dev/check_mt.py dev/moe_v1503u.py 9 10 12 13 15 16 2>&1 | grep -a -E 'FAILURES|FAIL |Error|Traceback' | tail -2
MOE_DEFINES=MOE_E2_KSXNT2=4,MOE_S2_SPLW2=0.5f timeout 600 python3 dev/check_mt.py dev/moe_v1503u.py 9 10 12 13 15 16 2>&1 | grep -a -E 'FAILURES|FAIL |Error|Traceback' | tail -2
V="v969|/tmp/v969s.py|;v969b|/tmp/v969s.py|;v1501|dev/moe_v1501.py|;v1502|dev/moe_v1502.py|;v1502b|dev/moe_v1502.py|"
echo "STEP bench all shapes, 70 reps: $V"
B2_SHARE_EQ=1 B2_SEEDS=dev/jm_seeds.jseeds timeout 2700 python3 dev/bench2.py - --variants "$V" --reps 70 --shapes 1,8,16,32,128,256 --out /tmp/v1502.json 2>&1 | grep -a -E "geomean" | head -10
python3 dev/project2.py /tmp/v1502.json v969
V2="v1502|dev/moe_v1502.py|;v1502b|dev/moe_v1502.py|;k2w1|dev/moe_v1503u.py|MOE_E2_KSXNT2=2,MOE_S2_SPLW2=1.0f;k4w05|dev/moe_v1503u.py|MOE_E2_KSXNT2=4,MOE_S2_SPLW2=0.5f;k2w1b|dev/moe_v1503u.py|MOE_E2_KSXNT2=2,MOE_S2_SPLW2=1.0f"
echo "STEP bench bs16 split rules (timing; outputs of KRS variants after rep 2 invalid with B2_SHARE_EQ), 100 reps: $V2"
B2_SHARE_EQ=1 B2_NOVERIFY=1 B2_SEEDS=dev/jm_seeds.jseeds timeout 1500 python3 dev/bench2.py - --variants "$V2" --reps 100 --shapes 16,8 --out /tmp/v1503s.json 2>&1 | grep -a -E "geomean|^\s+(v1502|k2|k4)" | head -20
echo "GPU_AFTER $(nvidia-smi $Q)"
echo "STEP done"
