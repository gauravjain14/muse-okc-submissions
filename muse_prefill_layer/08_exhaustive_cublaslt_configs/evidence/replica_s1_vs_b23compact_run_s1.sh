#!/bin/bash
# s1 (lt_v2 + bnd_tail) vs b23_compact: gate, then the fixed (per-candidate AB/BA) replica on 3 router draws.
cd /root/okc
bash research/claude/gate.sh research/claude/stack/s1.py /tmp/gate_s1.json
bash research/judge/run_exp.sh /tmp/s1_panel.json 11,22,33 0,1,2,3,4,5,6,7 8 base=research/claude/b23_compact.py s1=research/claude/stack/s1.py
