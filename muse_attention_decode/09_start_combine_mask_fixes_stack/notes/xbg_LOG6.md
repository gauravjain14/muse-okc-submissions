# xbg round 6: d32s (attn_1_1_1_1e3, runE1, cluster of 3 CTAs per request, DSMEM combine)

Base runs/k80.py (cand6.py). Slim: slim6_k80.py (work/base/t/mkslim.py). A/B: ab.sh d32s 7,8,9 (multi-process interleaved).
Probe builds: patch_e6prof.py (W6P: per-WG step data/end globaltimer stamps, combine stamps), prof_e6.py (timeline +
slowest clusters), W6Q / W6R (clock64 phases of we_tilex), prof_e6c.py (WG2 tail clocks).

## Where the time goes (k80, median over CTAs, ns from the first CTA start)
- page-table entries land 1440; producer copies issued: warp 15 4060, warp 8 5790 (DRAM backpressure).
- WG0/WG1 step 0 (tiles 0-1 / 2-3): data 4190 -> end 5380; step 1 (4-5 / 6-7): data 5790 -> end 6880.
- WG2 (tiles 8-9) / WG3 (tile 10 + new token): data 6620 -> end 7620 / 7520.
- merge done (WG0) 8060, push done 8220, owner receive 8510, owner done 8800; kernel end (max owner done) ~9.0-9.2 us.
- Data phase: 17.3 MB in ~5.3 us = 3.3 TB/s: it ends where the base worker's pure-gather floor ends. The ~2.2 us
  after the last data is the tail: last step ~0.95, WG merge + push ~0.57, receive ~0.25, owner ~0.3.
- we_tilex pair step (clock64, WG0 step 0): QK + V ldmatrix 600, mask + 12-shuffle max 450, bar 80, smem max
  gather + rescale 300, P (16 ex2 + cvt + sts) 430, bar 60, PV 300 -> ~2200 cycles (~1.1 us). WG2's step: same shape.
- ncu SourceCounters: no_inst (instruction fetch) stalls are small (185 of 6025 samples, prologue); producer copy
  loop: wait / short_sb (SHFL) / branch_resolving; owner + exit region dominated by the cluster barrier (waiting for peers).
- The tile math is already the efficient q_len-1 form: S^T = K Q^T with m64n8k32 (tokens as M, the 8 heads of a KV
  head as N), O^T = V^T P^T with V^T fragments in registers.

## Experiments (all vs k80 in the same ab.sh session)
| # | change | k80 | cand | verdict |
|---|---|---|---|---|
| 1 | W6S: lazy-reference steps (one barrier, exchange only when a score passes m + 0.75) for the producer WGs (seeded with WG li-2's max after its first step, + 2) and later steps | 13.17 13.12 13.38 | 13.86 13.67 13.80 | +0.55: with 16 rows per WG the vote almost always fires (consumer step 1: 1.05 -> 1.53 us) |
| 2 | W6V: V-fragment ldmatrix after the max exchange | 13.15 13.34 13.41 13.39 | 14.17 14.28 14.05 14.28 | +0.87: the producers' last copy round is issued ~1.5 us later (see note) |
| 3 | W6E: V-fragment ldmatrix before the QK wgmma | 13.38 13.36 13.64; rerun 13.21 13.32 13.34 13.27 13.38 | 13.23 13.62 13.28; rerun 13.38 13.29 13.32 13.26 13.33 | noise (13.32 vs 13.30) |
| 4 | W6CT: WG0 folds WG1 early, WG2 folds WG0's result + WG3 and pushes | 13.30 13.32 13.41 13.46 13.34 | 13.38 13.39 13.37 13.23 13.40 | 0 |
| 5 | W6W: merge/push (WG0) + owner epilogue run once dry at the end (i-cache warm-up) | 13.44 13.24 13.63 | 13.72 13.85 13.85 | +0.35 |
| 6 | W6W2: same, dry pass before WG0's bar 7 | 13.17 13.17 13.38 | 13.96 13.96 14.00 | +0.75 |
| 7 | W6W3: tail dry at kernel start (2-pass loop around the main body) | 13.33 13.24 13.26 | 15.63 15.70 15.56 | +2.3 (stack 32 -> 88) |
| 8 | W6PC: branch-free producer copies (16 shuffles grouped, new-token slot -> scratch sink) | 13.33 13.31 13.35 13.34 | 13.67 13.80 13.87 13.92 | +0.48 |
| 9 | W6PC=2: same, unroll-4 shape | 13.30 13.21 13.35 | 13.86 13.69 13.74 | +0.48 |
| 10 | W6NS: __nanosleep between copy rounds (100 / 400 ns) | 13.40 13.38 13.42 | 14.05 13.34 13.33 / 14.24 14.30 14.05 | no |

Note (sensitivity): small code changes anywhere in runE1 make ptxas reschedule the whole kernel (W6V: 986 of 2600
SASS lines differ after register renaming), and the copy phase reacts strongly: in the W6V build the producer loop
got BSSY/BSYNC regions + dummy LDS around each cp.async and the last round was issued 1.5 us later. The branch-free
copy loops (8, 9) are slower too, so it is not just instruction count. Any tail micro-change of 0.1-0.2 us is
swamped by these swings.

Result: no change beats k80 on d32s; k80 stays.
