#!/bin/bash
# val.sh: gate + replica (two panels, cases 0-3, 8 reps) for the stack s3 = rw_s16LnZ (MoE riders) + kv_t1k (attention)
L=research/claude/stack3/logs; mkdir -p $L
python3 research/modal_h200.py --timeout 3000 exec "timeout 2800 bash research/claude/gate.sh research/claude/stack3/s3.py /tmp/gate_s3.json 2>&1 | tail -3" > $L/gate.log 2>&1
for P in 11,22,33 44,55,66; do
  T=${P//,/_}
  python3 research/modal_h200.py --timeout 3000 exec "bash research/judge/run_exp.sh $L/rep_$T.json $P 0,1,2,3 8 base=research/claude/stk/s_b.py s3=research/claude/stack3/s3.py rw=research/claude/rw/rw_s16LnZ.py" > $L/rep_$T.log 2>&1
done
echo VAL_DONE
