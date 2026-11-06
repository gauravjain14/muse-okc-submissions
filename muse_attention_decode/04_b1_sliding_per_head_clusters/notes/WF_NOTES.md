# WF: structural floor of the FLAT full-attention layers (b16/b32, q_len 1 and 8)

Files: k8a.py (dev/k8.py + ablation switches WF_NOEND / WF_ONLYFLAT + prologue stamps 16/17), trace.py (per-CTA
timeline + tile-finish percentiles + critical CTA), lc.py (local NVRTC 12.4 compile + SASS), run1f.cu (new FLAT
q_len-1 function), mk_k9f.py (builds k9f.py = <base, default dev/k10.py> + run1f), regs.py (copy of dev/wd/regs.py),
job*.sh (Modal jobs), job*.log.

## Job 1 (k8a = k8; same container; probe floor of THIS GPU 80.0 / 159.0 us for 328 / 655 MB, 4.1 TB/s)
bench (local us, 30 reps): c1/c2 = k8 controls, nc = ABL_NOCOMPUTE, ne = WF_NOEND (no merge/publish/combine),
ncne = both.
  decode 16 full  90.19 / 90.23 | nc 88.54 | ne 83.17 | ncne 81.48     (probe 80.0)
  decode 32 full 169.54 /169.24 | nc 167.40 | ne 161.84 | ncne 161.18   (probe 159.0)
  verify 16 full 124.37 /124.13 | nc 96.30 | ne 83.84* | ncne 82.26
  verify 32 full 224.68 /223.89 | nc 178.43 | ne 163.30* | ncne 161.42
  (* run8 NOEND lets the compiler drop the dead softmax: not a valid "end only" number for q_len 8)
=> q_len 1: the lockstep-lane cp.async skeleton is only +1.5 / +2.2 us over the probe; compute +1.7 / +2.1;
   END CHAIN (merge + publish + combine + its effect) +7.1 / +6.2 us. q_len 8: end chain nc - ncne = 14 / 17 us.
Traces (flush, med over 10 reps, us from first CTA start):
- decode b32: prefix 1.04 | pt-LDG issued 2.95 | first row back 3.40 | prologue issued 6.54 | tile0 7.13 |
  tiles p0/p10/p50/p90/max/mean 149.4/155.3/161.1/163.4/165.1/160.0 | critical CTA: tiles 164.9 -> mstore 165.7 ->
  merged 166.1 -> publish 167.65 -> combined 169.6. nseg1 mean 161.1 vs nseg2 156.4 (OVH 1536 over-compensates).
  SM 124 is ~10 us faster than the median in every rep (per-SM stdev 1.7 us).
- decode b16: pt-LDG 2.53 | row back 2.94 | issued 5.93 | tile0 6.68 | tiles 69.6/77.4/82.6/84.1/85.4/81.7 |
  critical: tiles 85.1 -> publish 87.8 -> combined 90.4. nseg1 mean 82.4 vs nseg2 76.0.
- verify b32: issued 6.73 | tile0 8.70 | tiles 208.0/213.3/216.3/218.8/222.6/216.3 | max tiles 222.6 -> last
  stamp 234.2 (11.6 us end chain: K = P combiners for R = 128 rows all spin until the last partial lands).
- START: row back -> prologue issued = 3.1 us for 3 stages = cold instruction fetch: the 3 unrolled prologue stages
  are ~8.8 KB of SASS (fast + slow path each); prefix -> pt-LDG 1.9 us = ~7 KB of SASS (64-bit divisions, linear
  b_begin search, gate/Q/new-token loads before the page-table LDGs).

## Job 2 (k9f = dev/k10.py + run1f v1; same container; local us, 40 reps)
check.py 0 failures (also --match "layer full" with WF_POOLALL (pools on every FLAT problem) and WF_POOLALL,WF_X=1).
  decode 16 full: k10 87.26 / 87.13 | f (pools X=16) 91.10 / 90.41 | fx0 (WF_X=0: compact start only) 87.19 |
                  fx8 90.75 | fx32 91.13 | f+OVH1024 90.47
  decode 32 full: k10 162.57 / 162.43 | f 165.86 / 166.00 | fx0 162.61 | fx8 165.83 | fx32 165.62 | f+OVH1024 165.44
=> compact start: NO GAIN; pools: +3.5 us WORSE.
Traces: (fx0, b32) pt-LDG 1.94, row back 2.38, prologue issued 6.02, tile0 6.74: the rolled prologue still "takes"
3.6 us -> the prologue time is NOT cold code: LDGSTS issue stalls on the SM's outstanding-request resources while
HBM serves the 25 MB start burst (192 KB per SM); HBM is busy from ~2.4 us. Real idle-HBM start = ~2.4 us
(prefix 0.75 + split / page-table LDG 1.2 + row latency 0.45).
Pools (f, b32): tiles p0/p10/p50/p90/max 154.0/155.7/157.4/158.4/160.2 (fx0: 145.0/149.8/154.6/156.3/157.7);
per-SM stdev 0.53 (fx0 1.24): the spread is gone, but max tiles is 2.5 us LATER and the end chain 1.7 us longer
(everyone ends at once: critical publish 2.5 us, b16 publish 3.6 + combine 3.5). Static FLAT's spread is largely
self-correcting (early finishers leave HBM to the rest; only max - p90 ~1.5 us is lost), so balancing can win
<= ~1.5 us, and the pool claims cost more than that here (claim RTT under load vs one iteration of lead?).
End chain (fx0 b32 critical CTA): tiles 156.35 -> mstore 158.40 (2.05: lane barrier + 4 float2 stores!) ->
merged 158.82 -> publish 160.11 -> combined 162.18 = cold instruction fetch (~0.5-1 us per KB of SASS).

## Job 3 (same container)
GATHER CEILING (dev/wf/probe2.py, 655 / 328 MB, cold L2): best 151.9 us = 4.31 TB/s (ring 8 warps x 2 x 4 KB, 3 CTAs/SM)
/ 76.75 us = 4.27 TB/s. All ring/LDG configs with >= 16 issuing warps per SM: 4.24-4.31 TB/s. No gain from
.L2::128B prefetch hint, L2::evict_first policy, 192 KB vs 128 KB in flight, 2-3 CTAs/SM. Fewer issuing warps
lose (8 warps x 6 stages 3.91, 4 warps 2.73). cp.async.bulk (TMA bulk, one 128-B row per instruction):
1.75-3.67 TB/s (16 warps best). LDG with evict_first + L1::no_allocate HURTS (3.83-4.15). => ~4.3 TB/s is the
random-128-B-row ceiling on this part; no 4.5.
BENCH decode full (40 reps; us): k10 86.94 / 87.18 | 163.08 / 163.12
  g0 = run1f v2 (compact end: no stack arrays, double-floor splits, __fdividef epilogue; no pools)  85.94 | 161.03
  gh1 = g0 + .L2::128B on K/V cp.async     85.94 | 160.81
  gh2 = g0 + L2::evict_first on K/V        85.29 | 159.97   (-1.7 / -3.1 vs k10)
  gh3 = both hints                         85.07 | 160.44
  g = v2 + pools (X=16) 89.05 | 163.31; gd2 = pools, 2 claims in flight 88.80 | 163.78 => pools stay slower
  (the claim latency is not the cause). Pools now default off (WF_X=0).
trace g0 b32: critical CTA tiles 157.04 -> mstore 157.66 -> merged 158.06 -> publish 159.31 -> combined 160.99.

## Job 4 (same container, slower GPU: k10 90.0 / 169.3)
image NVRTC: v3 (k9f3: smem-held post-loop state, staged rolled combine) attn_1_0_1_0 128 regs, stack 8, spill 12/16
(outside the tile loop); v2 (k9f2 @WF_X=0,WF_CPH=2) 126 regs, 0 spill.
  decode 16 full: k10 90.03 / 90.31 | v2 88.13 | v3 88.33 / 88.25 | v3+OVH1024 88.56 | v3+OVH768 90.20 | v3 no hint 89.18
  decode 32 full: k10 169.24 / 169.49 | v2 165.88 | v3 166.69 / 166.61 | v3+OVH1024 167.41 | v3+OVH768 168.27 | v3 no hint 167.27
=> v2 (-2.0 / -3.5 us) stays best; v3's extra compaction does not pay (spills); OVH 1536 stays; evict_first -0.9 / -0.6.
WF_TWICE diagnostic (each end-chain phase run twice; 2nd run warm), critical CTA, b16 / b32 (us):
  lane-merge stores: cold 1.82 / 1.43, warm 0.70 / 0.64; staged combine: cold 1.55 / 2.16, warm 1.46 / 1.53.
  => cold instruction fetch is ~1.1 us of the merge and 0.1-0.6 us of the combine; the rest of the ~4.5 us end
  chain (tiles max -> combined) is barriers + L2 round trips (publish ~1.1 us, staging ~0.6 us) + serial merge.
Deliverable = v2 source with the pool code removed (dev/wf/run1f_k7.cu; same SASS instruction count as the measured
k9f2 @WF_X=0,WF_CPH=2) -> dev/wf/k7f_flat.py (WF_SRC=dev/wf/run1f_k7.cu python3 dev/wf/mk_k9f.py dev/k10.py ...).

## Job 5: DELIVERABLE dev/wf/k7f_flat.py (= dev/k10.py + run1f from dev/wf/run1f_k7.cu; run8 / runA1 / run1 ALIGNED
untouched; host code identical to k10). Image NVRTC: attn_1_0_1_0 124 regs, 0 spill, tile loop 601 instrs.
check.py: 0 failures (all 16 tests + 12 scored, 2 calls each). Bench, all loads, 40 reps, local us:
  decode 16 full  k10 87.00 / 86.87 | f7 85.37 / 85.52     decode 32 full  k10 162.61 / 162.63 | f7 159.38 / 159.70
  other 10 shapes: within noise (same kernels).  GEOMEAN k10 x2.1783 / x2.1841 -> f7 x2.1968 / x2.1971
  (judge-projected x2.069 / 2.074 -> x2.086).
Trace f7 b16: critical CTA tiles 80.86 -> mstore 81.50 -> merged 81.89 -> publish 83.14 -> combined 85.76 (2.4 us
combine: P ~ 9 partials loaded in 3 batches of 4); b32: tiles 156.14 -> publish 158.42 -> combined 160.26.
B16 vs B32: the fixed cost over the probe is the same in absolute terms (~8 us: start ~1 us over the probe, end
chain ~4.7 us, spread max - p90 ~1.1 us), so it is 10% of b16 but 5% of b32; b16 also has P ~ 9 partials per
request (b32 ~ 5), so its combine is ~0.7 us longer -> k7f_flat_b: one load round (2 groups x 8 partials).

## Job 6: k7f_flat_b (one-round split combine for a single K == 1 request) vs k7f_flat (slower container)
image NVRTC 124 regs, 0 spill; check.py 0 failures. decode full, 40 reps (us):
  b16: k10 90.08 | f7 88.22 / 88.53 | f7w 88.00 / 87.87      b32: k10 168.60 | f7 166.50 / 165.85 | f7w 166.39 / 166.15
  trace f7w b16 critical: publish 86.48 -> combined 88.42 (1.9 us; job 5 f7: 2.6 us).
=> FINAL dev/wf/k7f_flat.py = k7f_flat_b (WF_SRC=dev/wf/run1f_k7b.cu python3 dev/wf/mk_k9f.py dev/k10.py dev/wf/k7f_flat.py);
   the job-5 version is kept as dev/wf/k7f_flat_j5.py.

## Next levers (not done)
1. Lane-level publish on the last segment: each warpgroup stores its own partial as soon as its last tile is done and
   the combiner folds 4P lane partials (one load round). Removes the CTA lane barrier + smem merge from the critical
   path (~0.6 + 0.4 us warm, +1.1 us cold fetch, per WF_TWICE) for ~+0.3 us of extra staging. Est. -1 us b16/b32.
2. q_len 8 (run8 FLAT) end chain: max tiles -> last stamp 11.6 us at b32 (job 1): K = P combiners for R = 128 rows all
   spin until the last partial lands, then a ~3.7 us gap; a single last-arrival combiner with one load round over
   2-4 thread groups (as k7f_flat_b) should cut several us. Note W8: evict_first HURTS run8 (+2.5 / +5.8 us).
3. Start: HBM idle ~2.4 us (seq_lens 0.6 + split 1.0 + page-table row 0.5); anchored/speculative split could save ~1 us.

## Job 7: deliverable REBASED on dev/wd/k11d.py (coordinator) + WD_MOPAD (stride 68) ported into run1f's lane merge
Build: WF_SRC=dev/wf/run1f_k11.cu python3 dev/wf/mk_k9f.py dev/wd/k11d.py dev/wf/k7f_flat.py (header docstring edited).
Image NVRTC: attn_1_0_1_0 124 regs, 0 spill, tile loop 602 instrs. check.py 0 failures. 40 reps, two controls:
  decode 16 full  k11d 86.35 / 86.44 | k7f_flat 84.36 / 84.19     decode 32 full  k11d 162.10 / 161.42 | 158.96 / 159.01
  decode sliding (unchanged kernels, 20 reps): identical errors, times within noise.

## Job 8: run8 FLAT end chain on dev/k11.py (dev/wf/k11w.py = k11 + WF_NOEND8: no merge/publish/combine, accumulators
kept live so the compute is not dead-code eliminated), 30 reps, local us:
  verify 16 full: k11 120.66 / 120.78 | NOEND8 107.25 | NOCOMPUTE 95.41 | both 82.59
  verify 32 full: k11 219.99 / 219.89 | NOEND8 202.13 | NOCOMPUTE 177.35 | both 161.53
=> the run8 FLAT end chain costs 13.4 / 17.8 us. Trace b32 critical CTA: tiles 216.88 -> segend 218.08 -> publish
221.25 (3.2 us: 64-bit divisions + serialized acq_rel atomics for 2 partials) -> comb0 225.42 -> combined 228.22
(2.8 us: FLAT combine loads one partial at a time, CPF is ALIGNED-only) ; tiles max 217.8 -> last stamp 228.2.
k12f8 (dev/wf/mk_k12f8.py): params precomputed by an idle lane-1 thread during the register merge, one fence +
relaxed atomics for both partials, ready request combined first, 8 partials' loads in flight, fast epilogue.

## Job 9: DELIVERABLE dev/wf/k12f8.py (= dev/k11.py + WF run8 FLAT end chain; tile loop, ALIGNED and cluster paths
untouched, ALIGNED kernels compile to the same regs/stack). Image NVRTC attn_8_0_1_0: 128 regs, stack 8, 0 spills
(k11 / WF8_OLD: stack 40, 64/64 B spilled). check.py 0 failures. 40 reps, two controls (local us):
  verify 16 full  k11 119.69 / 119.32 | k12f8 112.29 / 112.55   (-7.1 us)
  verify 32 full  k11 219.31 / 219.49 | k12f8 209.29 / 209.41   (-10.1 us)   geomean of the 2 loads x1.745 -> x1.843
Trace b32: tiles max 214.53 -> last stamp 220.42 (5.9 us; k11 10.4). Remaining: 2-partial CTAs combine their two
requests one after the other (~1.6 us each).

## Job 10: evict_first on run8 FLAT K/V (opt-in -DWF8_EVF) on top of the WF end chain, 40 reps (local us):
  verify 16 full  k11 119.31 | k12f8 112.41 / 112.71 | +EVF 112.55 / 112.94
  verify 32 full  k11 219.81 | k12f8 208.99 / 209.71 | +EVF 209.89 / 211.01
=> EVF stays OFF in run8 (neutral / slightly worse even with the compact end chain). k12f8 reproduced (-6.7 / -10.5).
Next on run8 FLAT: the 2-partial CTAs combine their two requests serially (~1.6 us each); K < P combiners (larger
comb_bytes, python env K3_COMB_BYTES: not A/B-able inside one bench process) would take boundary CTAs off the
critical path; lane-level publish does not fit L = 8 (2P partials of 33 KB double the combine traffic).

## Round on dev/k13.py -> dev/wf/k14f.py (mk_k14f.py): K2_DEFS hooks COMB / COMB1 / COMB8 / COMBF1 / COMBF8 / CPR / MINU
(each falls back to the old env variable K3_COMB_BYTES / K3_CPR / K3_MIN_UNITS, then the old default).
Job 11 sweep (40 reps, local us; f = k13 kernels):
  verify 16 full: k13 113.28 | f 113.15 / 113.10 | COMBF8=64K 112.20 | 128K 114.79 | 256K 118.71 | 1M (K=1) 123.38
  verify 32 full: k13 208.75 | f 208.92 / 208.83 | COMBF8=64K 209.01 | 128K 212.64 | 256K 215.94 | 1M 216.44
  decode 16 full: f 87.98 / 87.84 | COMBF1=16K 88.75 | 8K 88.95;  decode 32 full: 166.51 / 166.41 | 167.58 | 167.57
=> q_len-8 FLAT: K ~ P/2 combiners (64 KB) -1.0 us at b16, neutral at b32; fewer combiners lose (one CTA cannot pull
   P x 33 KB fast enough). q_len-1 FLAT: K = 1 (131072) stays. check "layer full" 0 failures at COMBF8 64K / 1M + COMBF1 8K.
(2) overlapping the two combines of a 2-partial CTA: not built: with 512 threads each request is one load round, with
   256 threads per request each needs two -> same number of L2 rounds; the ready-first order already overlaps the wait.
(3) anchored FLAT start: fixed CTAs per request cannot balance +-7.5 % lengths (~+7 % on the slowest request), and
   anchoring only the first tiles creates an extra partial per CTA (a mid-run drain/refill ~3-4 us) for <= ~1.4 us of
   start: not built.
Job 12 (k14f with q_len-8 FLAT comb 64K default): verify 16 k13 113.40 / 113.05 | f14 112.82 / 112.14 | 48K 112.54 |
  96K 113.89; verify 32 k13 209.98 / 209.26 | f14 210.19 / 209.79 | 48K 211.16 | 96K 211.93; decode full unchanged.
  check.py 0 failures. => K ~ P/2 helps b16 (P ~ 9), not b32 (P ~ 5): replaced by a cap on the combiner count.
Job 13: FINAL dev/wf/k14f.py = k13 + hooks + WF8_KMAX = 5 (at most 5 combiners per q_len-8 FLAT request; K = P below);
  host defaults unchanged. Image NVRTC attn_8_0_1_0 128 regs, stack 8, 0 spill. check.py 0 failures. 40 reps:
  verify 16 full  k13 113.83 / 114.57 | k14f 113.67 / 113.41 | KMAX4 114.22 | KMAX6 113.70
  verify 32 full  k13 210.74 / 211.08 | k14f 210.56 / 211.24 | KMAX4 210.17 | KMAX6 210.88
  => -0.7 us at b16, neutral at b32 (the combiner count is a <= 1 us lever).

## Batch-1 full layers (coordinator round on dev/k13.py): decode full b1 = run1 ALIGNED, 132 CTAs, P = 132
dev/wf/run1b.cu (dispatched for ALIGNED && !ROPE && window == 0), builder dev/wf/mk_k15b1.py on dev/wf/k14f.py.
k13 trace (job 14, flush, med): issued 3.71 (max 5.49), tile0 4.48, tiles 8.20 (max 9.22), merged 9.02, publish
10.03 (max 10.98), combiners released 11.54, COMB2 loads+smem 12.73, fold 13.33, epilogue 14.18, combined 14.20.
Job 14 (run1b v1: anchored page-table loads + K/V burst FIRST, then Q / new-token loads, length barrier after):
  decode 1 full k13 13.88 / 13.78 vs run1b 18.14 / 18.36 (!). Trace: page-table LDG 0.38, rows back 1.09, issued
  3.94 -> length barrier 5.78, Q^T 7.12, tile0 7.94: the Q load queued behind the 20 MB burst.
Job 15 (v2: Q / new-token loads before the burst, length barrier moved to just before the loop): 18.37 / 18.43.
  Trace: issued 4.10 -> length barrier 7.16 (all CTAs within 0.25 us) -> tile0 8.03.
=> FINDING: a CTA-wide barrier AFTER the K/V burst waits for every warp's LDGSTS issue, which stalls on the SM's
   outstanding-request resources until most of the burst has been served (~7 us at b1 = 20.5 MB / 4.4 TB/s + start).
   This is also why WD's anchored k9a (length barrier moved after the issue) gained nothing. Any barrier, length
   or new-token, must come BEFORE the burst; after it only lane (named) barriers.
v3: loads (length, anchored page-table rows, Q, scales, new-token inputs) -> one CTA barrier (length) -> new token
(owner, its barrier) -> K/V burst with tokens beyond the length zero-filled -> Q^T -> loop.
Job 16 (v3: length barrier before the burst): 18.36 / 18.11 vs k13 13.67 / 13.59. Job 17 (v4: exact C (C rounded to
64 had left 7 CTAs empty), Q^T before the burst, new-tile fence + lane barrier before the new tile): 19.23 / 19.06
vs k13 14.27 / 14.30; check 0 failures. Trace v4: length barrier 1.38, issued 5.10, tile0 8.19 (stage 0 lands only
at the END of the all-at-once 20 MB burst), first tile 8.19 -> 10.58 (2.4 us: the tile-loop code runs cold AFTER
the burst), loop exit 12.75. k13 (staggered issue 2.8..5.5 us): tile0 4.48, the cold loop code overlaps the gather.
=> DEAD END (as WD's k9a): issuing every CTA's whole range at once delays every first tile to the end of the burst
   and serializes the cold tile-loop code after it. run1b kept only as a record (dev/wf/run1b*.cu).
Job 18 (fewer, fatter partials via the CPR hook on k14f; 40 reps, local us):
  decode 1 full: k13 14.14 | 132 (default) 14.10 | 120 14.03 | 110 14.25 | 99 15.42 | 88 15.63 | 66 16.43
  verify 1 full: k13 18.94 | 132 18.46 | 120 19.75 | 110 19.61 | 99 20.53 | 88 21.16 | 66 23.22
=> the gather needs every SM (per-SM gather ~33 GB/s); fewer partials never pay back. 132 CTAs stay.
Job 19: decode full b1 EARLY combiners (dev/wf/b1end.cu, inserted before run1's publish for ALIGNED && !ROPE,
cpr >= 16): the first 16 arrivals (arrival order = row) publish, then poll per-partial ready flags (call epoch,
release after the partial) and fold partials as they become visible; the last combiner advances the epoch and
resets the arrival counter (no flag resets). NOTE: requires all CTAs co-resident (true for hp = 1, 132 CTAs).
Job 19 result: check.py 0 failures; decode 1 full k13 14.25 / 14.21 vs early combiners 14.50 / 14.57 (+0.3 us).
  Trace: combiners start polling at 9.34 (they publish early), last publish 11.10, all partials folded 12.31 (+1.2 us:
  each pass of the poll loop is a flag round trip + a data round trip), fold + RMS 13.16, epilogue 13.94.
  => same conclusion as WS's per-source mbarrier polling: folding partials "as they land" does not shorten the
  chain after the last publish (still poll RTT + load RTT + fold + epilogue). NOT recommended; dev/wf/k15b1.py keeps
  it only as a record (-DWF_B1OLD restores run1's end chain).
Batch-1 summary: no change beats k13 on decode full b1. Next levers (not built): WS's tail recipe (sigmoid of the
gates precomputed while waiting, both reductions in one pass, one barrier, fast reciprocals: ~0.3-0.5 us of the
~1.6 us fold+epilogue), and a start that issues only stage 0 up front (smaller burst: first tiles land early) while
warming the tile-loop code during the burst.
Job 20 (run1b v5 on dev/k18.py: WS-style predicated prefix loads + stage 0 only before the loop, stages 1-2 at the
loop's first barrier): 21.4 / 21.6 (lazy 1), 20.9 (2), 20.8 (3) vs k18 14.14 / 14.05; check 0 failures. Trace:
length barrier 1.68, stage 0 issued 3.92, tile0 8.54, first tile done 12.54 (4 us), loop exit 14.8. Every run1b
variant (anchored split + everything before the first copy) lands its first tile at ~8 us and runs the first tile for
2.4-4 us, whatever the stage count; k18's run1 reaches tile0 at ~4.5 us. Cause not isolated (no spills in run1b;
same tile code). run1b abandoned.
Job 21: C2 = clusters of 2 (66 x 2 = 132 CTAs): pair halves exchanged over DSMEM, global one-row combine over 66
pair partials (dev/wf/mk_k19c2.py on dev/k18.py, kernels attn_1_0_P1_1q).
Job 21 result (40 reps, local us): k18 13.94 / 13.94 | C2 14.32 / 14.00 | same file NOC2 13.87. Smoke (scored b1
full) 0 failures; full check.py rc=1 with no failure line (not investigated; C2 not delivered). Trace C2: tiles max
9.07, merged 9.10 (pair exchange), publish max 10.99, released 11.52, COMB2 loads + smem 12.80 (1.28 us for 66 pair
partials vs 1.2 us for 132 in k18), fold 13.51, epilogue 14.46 (0.95 us).
=> the global one-row combine is LATENCY-bound (one load round, then barriers + fold + epilogue); halving the
   partials saves nothing and the pair exchange costs ~0.1-0.2 us. Shared with WB (verify-1 full owner).
Decode full b1 round summary: no improvement over k18 found (anchored / speculative starts, early combiners,
fewer CTAs, cluster pairs all measured worse or equal). Remaining small lever: the COMB2 epilogue (~0.9 us,
lane-0 16-sum + expf / division + barrier) with WS's tail recipe (precomputed sigmoid, one-pass reductions).
