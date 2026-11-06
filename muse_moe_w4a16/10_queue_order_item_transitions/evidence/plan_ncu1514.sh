# planner 2026-11-01: ncu --set full on v1514 at the six judge shapes (judge-matched median seeds): smem / pipe / DRAM limits
set -o pipefail
Q="--query-gpu=name,clocks.sm,power.draw,temperature.gpu,utilization.gpu,memory.used,driver_version --format=csv,noheader"
echo "GPU_BEFORE $(nvidia-smi $Q)"; ncu --version | tail -1
echo "IDLE [$(timeout 4 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -lms 250 | head -12 | tr -d ' %' | tr '\n' ' ')]"
O=/tmp/ncu1514; mkdir -p $O
export ABL_SUB=dev/moe_v1514.py MOE_LINEINFO=$PWD/moe.cu
for p in "256 73011" "128 71637" "32 73736" "16 73249" "8 70261" "1 73235"; do
  set -- $p
  echo "STEP bs$1 seed $2"
  timeout 1500 ncu --set full --import-source yes --clock-control none -k regex:"^moe" --launch-skip 5 --launch-count 1 -f -o $O/full_bs$1 python3 dev/ncu_one.py $1 $2 > $O/log_bs$1.txt 2>&1; echo "ncu rc $?"
  ncu --import $O/full_bs$1.ncu-rep --page details > $O/details_bs$1.txt 2>&1
  ncu --import $O/full_bs$1.ncu-rep --page raw --csv > $O/raw_bs$1.csv 2>&1
  ncu --import $O/full_bs$1.ncu-rep --page source --csv --print-source sass > $O/sass_bs$1.csv 2>&1
  grep -a -E "Duration|DRAM Throughput|Memory Throughput|L1/TEX Hit|Shared|Compute \(SM\) Throughput|Issue Slots Busy|Executed Ipc|Pipe|SM Frequency|L2 Cache Throughput|Registers Per|Achieved Occupancy" $O/details_bs$1.txt | head -40
done
cd $O && tar czf /tmp/ncu1514.tgz details_*.txt raw_*.csv sass_*.csv log_*.txt full_*.ncu-rep && ls -la /tmp/ncu1514.tgz
echo "GPU_AFTER $(nvidia-smi $Q)"
echo "STEP done"
