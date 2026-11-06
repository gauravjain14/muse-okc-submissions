# planner: gap decomposition on judge-matched routings: full vs data-only vs compute-only (v200)
Q="--query-gpu=name,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader"
echo "GPU_BEFORE $(nvidia-smi $Q)"
G=dev/moe_v200.py
V="v200|$G|;v200b|$G|;nc|$G|MOE_NO_COMPUTE;nl|$G|MOE_NO_LOAD"
B2_NOVERIFY=1 B2_SHARE_EQ=1 B2_SEEDS=dev/jm_seeds.jseeds timeout 3000 python3 dev/bench2.py - --variants "$V" --reps 60 --shapes 1,8,16,32,128,256 --no-base --out /tmp/plan_gap.json 2>&1 | grep -a -v 'gpu-health\|shares'
echo "GPU_AFTER $(nvidia-smi $Q)"
