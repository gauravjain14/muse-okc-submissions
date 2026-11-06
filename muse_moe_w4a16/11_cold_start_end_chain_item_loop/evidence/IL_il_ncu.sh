#!/bin/bash
# IL ncu: per-instruction stall samples (SourceCounters, zero perturbation) of moe_i3 and IL forms at bs256 / bs128
# (jm seeds 73011 / 71637, ST's st_ncu3 routings), then t3/IL/il_ncu.py's item-region decomposition.
# usage: bash t3/IL/il_ncu.sh "label:file:defs ..."
set -o pipefail
cd /root/okc
TAG=${TAG:-il_ncu}
mkdir -p /tmp/$TAG
M="smsp__inst_executed.sum,smsp__issue_active.sum,smsp__cycles_active.sum,sm__cycles_elapsed.avg.per_second,gpu__time_duration.sum"
for v in $1; do
  IFS=: read -r lab f d <<< "$v"
  for r in bs256:73011 bs128:71637; do
    t=${lab}_${r/:/_}
    ABL_SUB=$f MOE_DEFINES=$d timeout 600 ncu --clock-control none -k regex:"^moe" --launch-skip 5 --launch-count 1 \
        --section SourceCounters --section WarpStateStats --metrics $M \
        --warp-sampling-interval 3 --import-source no -o /tmp/$TAG/src_$t -f python3 dev/wpw_ncu_one.py $r > /tmp/$TAG/run_$t.txt 2>&1
    echo "ncu $t rc $? $(grep -a done /tmp/$TAG/run_$t.txt | tail -1)"
    ncu --import /tmp/$TAG/src_$t.ncu-rep --page source --csv --print-source sass > /tmp/$TAG/src_$t.csv 2>/dev/null
    ncu --import /tmp/$TAG/src_$t.ncu-rep --page details --csv > /tmp/$TAG/det_$t.csv 2>/dev/null
    python3 t3/IL/il_ncu.py /tmp/$TAG/src_$t.csv /tmp/$TAG/det_$t.csv 2>&1 | tail -7
  done
done
cd /tmp/$TAG && tar czf /tmp/$TAG.tgz src_*.csv det_*.csv run_*.txt && ls -la /tmp/$TAG.tgz
