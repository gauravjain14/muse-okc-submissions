# planner 2026-11-01: nsys GPU-metric timelines on JUDGE-MATCHED routings (first 8 jm seeds per shape) for v1512 + v1504
set -o pipefail
Q="--query-gpu=name,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader"
idle() { echo "IDLE $1: apps=[$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')] samples=[$(timeout 4 nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader -lms 250 | head -12 | tr '\n' ';')]"; }
echo "GPU_BEFORE $(nvidia-smi $Q)"
for V in v1512 v1504; do
  for bs in 1 8 16 32 128 256; do
    SEEDS=$(python3 -c "import json;print(','.join(str(s) for s in json.load(open('dev/jm_seeds.jseeds'))['$bs'][:8]))")
    idle "$V bs$bs"
    echo "STEP capture $V bs$bs seeds $SEEDS"
    timeout 900 python3 dev/sprint_nsys_capture.py --output /tmp/nsysjm_${V}_bs$bs --source dev/moe_$V.py --shapes $bs --seeds $SEEDS --reps 24 2>&1 | tail -1 | cut -c1-200
    python3 dev/plan_nsys_an2.py /tmp/nsysjm_${V}_bs$bs.sqlite ${V}_bs$bs 10 2>&1
  done
done
echo "GPU_AFTER $(nvidia-smi $Q)"
echo "STEP done"
