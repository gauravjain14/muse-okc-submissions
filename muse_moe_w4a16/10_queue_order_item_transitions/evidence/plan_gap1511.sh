# planner 2026-11-01: gap decomposition of v1511 on jm routings: full (x2) vs data-only vs compute-only (timing only)
set -o pipefail
Q="--query-gpu=name,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader"
echo "GPU_BEFORE $(nvidia-smi $Q)"
echo "STEP hang1 smoke"
timeout 300 python3 dev/hang1.py dev/moe_v1511.py 2>&1 | tail -3
G=dev/moe_v1511.py
V="f0|$G|;full|$G|;nc|$G|MOE_NO_COMPUTE;nl|$G|MOE_NO_LOAD;fullb|$G|;f1|$G|"
echo "STEP bench2 60 reps"
B2_NOVERIFY=1 B2_SHARE_EQ=1 B2_SEEDS=dev/jm_seeds.jseeds timeout 4000 python3 dev/bench2.py - --variants "$V" --reps 60 --shapes 1,8,16,32,128,256 --no-base --out /tmp/gap1511.json 2>&1 | grep -a -v 'gpu-health\|shares'
echo "GPU_AFTER $(nvidia-smi $Q)"
echo "STEP done"
