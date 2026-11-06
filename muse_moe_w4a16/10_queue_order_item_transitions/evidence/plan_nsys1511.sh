# planner 2026-11-01: nsys GPU-metric timelines (cold-L2 graph replays, 75 kHz) of v1511 and v1504, all six shapes
set -o pipefail
Q="--query-gpu=name,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader"
gc() { echo "GPUCHECK $1: apps=[$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')] $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; }
echo "GPU_BEFORE $(nvidia-smi $Q)"; nsys --version
for V in v1511 v1504; do
  gc "before $V"
  echo "STEP capture $V"
  timeout 1500 python3 dev/sprint_nsys_capture.py --output /tmp/nsys_$V --source dev/moe_$V.py --reps 48 2>&1 | tail -3
  echo "STEP analyze $V"
  python3 dev/plan_nsys_an.py /tmp/nsys_$V.sqlite $V 10 2>&1
done
echo "GPU_AFTER $(nvidia-smi $Q)"
echo "STEP done"
