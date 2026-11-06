#!/bin/bash
# ncu full set (flushed caches, no clock control) of one skv / sk11 launch; prints the key metric table.
#   bash claude/t3/V2/ncuv.sh tag kind m cfg [tag kind m cfg ...]      (run from /root/p on the sandbox)
set -u
R=/tmp/ncuv; mkdir -p $R
while [ $# -ge 4 ]; do
  tag=$1; kind=$2; m=$3; cfg=$4; shift 4
  ncu --set full --clock-control none --cache-control all -k "regex:sk(v|11)_" -c 1 -f -o $R/$tag \
      python3 claude/t3/V2/one.py $kind $m $cfg > $R/$tag.log 2>&1 || { tail -5 $R/$tag.log; continue; }
  ncu --import $R/$tag.ncu-rep --page raw --csv 2>/dev/null | python3 -c "
import csv, sys
r = list(csv.reader(sys.stdin)); h, u, v = r[0], r[1], r[2]
want = [('gpu__time_duration.sum','dur'), ('dram__bytes_read.sum','dramRd'), ('dram__throughput.avg.pct_of_peak_sustained_elapsed','dram%'),
 ('lts__throughput.avg.pct_of_peak_sustained_elapsed','L2%'), ('l1tex__throughput.avg.pct_of_peak_sustained_active','L1/smem%'),
 ('sm__throughput.avg.pct_of_peak_sustained_elapsed','sm%'), ('smsp__issue_active.avg.pct_of_peak_sustained_active','issue%'),
 ('sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active','hmma%'), ('sm__warps_active.avg.pct_of_peak_sustained_active','occ%'),
 ('smsp__warps_eligible.avg.per_cycle_active','elig'), ('launch__registers_per_thread','regs'), ('smsp__inst_executed.sum','inst'),
 ('lts__t_sectors_srcunit_tex_op_read.sum','L2rdSectTex'), ('sm__cycles_elapsed.avg.per_second','clk'), ('launch__grid_size','grid')]
idx = {n: i for i, n in enumerate(h)}
print('  ' + '  '.join(f'{k}={v[idx[n]]}' for n, k in want if n in idx))
st = []
for i, n in enumerate(h):
  if n.startswith('smsp__pcsamp_warps_issue_stalled_') and not n.endswith('_not_issued'):
    try: st.append((float(v[i].replace(',', '')), n.replace('smsp__pcsamp_warps_issue_stalled_', '')))
    except ValueError: pass
tot = sum(s for s, _ in st) or 1
print('  stalls: ' + ', '.join(f'{n} {100*s/tot:.0f}%' for s, n in sorted(st, reverse=True)[:5]))
" | sed "s/^/[$tag] /"
done
