# W9 (round 4, 2026-10-29/30): batch-1 sliding END of kernel - d1s (xe_r1_1) and v1s (xe_b1s_8c6)
Base dev/coord/k65.py (md5 b76344ec). Write only dev/w9/ (older files k13w.py ... t_fold.py here are a PREVIOUS W9's, untouched).
GPU: one slot via tools/gate.sh, tag w9_jN, cap 45 GPU-min. Local NVRTC 13 venv in the scratchpad (dev/p1/sass.py).

## Resume here
- DONE (02:25, 2026-10-30). 5 jobs, ~20 GPU-min of the 45 (plus container overheads). FROZEN dev/w9/deliver_r1/ = NS only
  (k65ns.py a52025e5 = deliver_r1/build.py on dev/coord/k65.py; check.py 56/0, t_xe full 0; nsys d1s -0.026, v1s -0.031
  over 11 runs each; bench_jt no visible difference). Everything else measured dead (list at the end).
- Open leads (not built): (1) the d1s owner pre-warm is worth +0.088 (P0 = no pre-warm) but its pass overlaps compute warp
  0 (owner CTAs push ~0.1 us late and are the critical sources) -> a pre-warm that is neither on scheduler 0 in the compute
  window nor a cold call path (PX failed: +0.1); (2) post-wait owner cost is i-fetch of the fold code (probe below):
  smaller post-landing code (partly rolled folds) may pay where no pre-warm exists (v1s owner, both readers).

## Facts gathered before job 1 (no GPU)
- ncu line attribution (coordinator's table) used the wrong source module: XE_SRC is compiled as xe_b1s.cu, module
  line N = k65.py line 10583 + N. Correct mapping: d1s 19.7% of samples at the xe_r1_1 ENTRY line (prologue i-fetch /
  constants, not the owner merge), 21% at the first K use (K/V wait); owner merge+epilogue 9% (4% = mb_wait itself);
  v1s entry 12.7%, first K use 15%, owner try_wait 9%, owner merge 4%, reader 5%. Sent to the coordinator.
- Scored sliding layers are dist "soft" (keys k = sqrt(c2) u + sqrt(1-c2) z: unit RMS, a common direction per head).
  q is RMSNorm'd x QK_SCALE_FACTOR 5.473, keys unit RMS -> |c s| <= 64 * 5.473 / 8 * log2(e) = 63 (log2 units) for
  every harness input. => a FIXED reference R = 0 keeps every fp32 partial finite (2^+-63 x 2048 x 448 < 2^128).
  FP16 P still needs the per-warp max (P relative to m_w), but every merge above the warp can be a plain sum.
- SASS (local NVRTC 13): v1s compute path Q-ready -> merge barrier 293 instrs/warp (F2FP 84, IMAD 30, HMMA 27,
  PRMT 24, ISETP 16, FMNMX 16, FSEL 14, MUFU 12 ...), reader 137 instrs (4 warps); per CTA ~2900 warp-instrs ~0.41 us
  of issue at 1.75 GHz on 4 schedulers: v1s compute+reader is issue-bound. d1s compute 190 instrs/warp, reader 67.
- shared::cluster f32 add = CAS spin loop (ATOM.E.CAST.SPIN); red.async.add.u32/u64 = native REDAS (integer only).
- Prior DSMEM numbers: P1 H4 probe D1S calibration (16 src x 8 owners x 272 B, simultaneous): all bytes arrived 614 cyc,
  classic 16-source fold + 5-shfl RMS + store 571 cyc (1.9 GHz). WS: 8.8 KB/owner +0.15 us vs 4.4 KB.

## Builder dev/w9/build.py <base> <out> [knobs]  (no knobs = base byte for byte)
R0 (correct): CTA partials pushed relative to R = 0 when the CTA row max is in [-100, 100] (else R = max, classic);
owners: plain sums when every source carries R = 0 (OR of the reference bits) and the new-token max is in range, else the
classic max merge; RMSNorm on y, L pre-scaled by 2^-exp(L) (exact). xe_r1_1 pre-warm pass always takes the fast path.
R0P (v1s) / R0P1 (d1s): compute warps scale slice partials to 0 (2 ex2 + 18 FMUL / thread) -> reader plain sums.
THR=0: test build, every partial takes the fallback. BN / BR: timing bounds (wrong results).
Local SASS: xe_r1_1 1272 -> 1336 instrs (R0), 82 -> 95 regs; xe_b1s_8c6 1800 -> 1832 (R0) / 1888 (R0P), 92 regs; 0 spills.

## Job 1 decision rules (written before the job)
- R0 (2 copies x 2 passes vs k65a/k65b): candidate if the mean nsys delta <= -0.03 us on d1s or v1s and neither shape
  is > +0.02 worse; then t_xe quick (r0all + r0t fallback build) must be 0 failures; confirmation = job 2 bench_jt
  --tlbcold 1 (2 copies vs 2 controls, extra control last) + check.py.
- R0P / R0P1 (1 copy): promote to 2 copies in job 2 if <= -0.03 vs R0 on its shape in both passes.
- BN: norm worth a structural attack only if its bound is <= -0.10 us on a shape; > -0.05 -> drop norm ideas.
- BR: v1s CTA-merge restructure (fewer slices / d-split PV) only if the bound is <= -0.15 us; > -0.08 -> drop.
- DSMEM probe: integer red.async fixed-point owner-free merge only if its all-to-all time <= 1.2 x the v4 st.async
  time and the late-source latency is within +30%; else drop that direction.

## Job 1 results (2026-10-30 00:20, ~3.5 GPU-min; dev/w9/out_j1.txt; raw traces out/20261030T002332_w9_j1/)
nsys medians (2 passes; controls k65a/k65b d1s 3.312 / 3.312, v1s 4.064 / 4.080): mean deltas (us):
  r0 (2 copies) d1s +0.024 v1s +0.072 | r0p (v1s prescale) v1s +0.168 | r0p1 d1s +0.016 (v1s +0.056 = R0's) |
  BN bound (no norm shuffles) d1s -0.056 v1s -0.024 | BR bound (v1s no CTA merge) v1s -0.056, d1s +0.016.
  => R0 DEAD: the fold's max / ex2 math is not on the critical path (trace: owner LDS+fold phase 504 -> 494 cycles);
  extra code on the cold v1s owner / reader path costs (R0P +0.17). Norm and v1s CTA merge are worth <= 0.06 each
  (rules: drop norm-structural and v1s-merge-restructure ideas; NS kept only as a cheap knob).
d1s trace (k65, representative; us from first CTA start, median over 16 reps x 2 clusters):
  land med 1.51 / max 1.65 | critical source: land 1.65 -> PV 2.02 -> bar 2.11 -> reader 2.24 -> pushed 2.30 ->
  owner recv 2.50 (recv - last push 0.19) -> out 2.94 (out - recv 0.45); critical source ranks spread (0, 1, 3, 5 most).
  Owner cycles: waited->LDS 224, LDS->sum 280, sum->shfl (norm) 177, shfl->stored 173 (854 total, trace build).
v1s trace was NOT representative: in the XE_TRACE build the pt-land stamp (slot 20) inside issueK makes ptxas issue K
  first with destination-register reuse (2nd K load waits for the 1st): K/V issued 1.86 us, out 4.06 = nsys. Fixed by
  knob TRX (stamps 19 / 20 dropped; SASS load order = production: 12 LDG.64 V, then 6 LDG.128 K, distinct registers).
DSMEM probe (dev/w9/dsprobe.py, cycles at 1.98 GHz):
  one-way hop (ping-pong) st.async v4 216-235, b32 190-195, red.async.add.u32 190-195 (peer distance irrelevant).
  d1s all-to-all (16 src x 8 owners, simultaneous): v4x17 (272 B) 591 | v2x34 480 | b32x68 602 | red.u32x65 601 |
  red.u64x33 628 | v4x9 FP16 (144 B) 521. Late single source (the critical case): v4x17 532 (sender issue 268) |
  red.u32 577 | red.u64 562 | v4x9 FP16 476 (issue 170). => DSMEM injection ~17 B/clk/SM; integer red.async no
  cheaper than st.async (fixed-point owner-free merge: no gain, DEAD); FP16 push saves ~56 cycles on the critical source.

## Job 2 results (2026-10-30 00:38, ~3 GPU-min; dev/w9/out_j2.txt; raw traces out/20261030T004138_w9_j2/)
nsys mean deltas vs k65a/k65b (2 passes; identical control copies differ by 0.03 on v1s -> noise floor ~+-0.03-0.05):
  PF d1s +0.272 (v1s -0.016) | H16 (FP16 pushes both) d1s +0.048 v1s 0.000 | NS d1s -0.016 v1s -0.064 (t_xe quick 0 fail) |
  QS (d1s-only change) d1s +0.008 v1s -0.048 (!: v1s moves +-0.05 with ANY code change -> layout / process noise) |
  EXH d1s -0.016 v1s 0.000 | XH d1s -0.016 v1s +0.016 | PW2 (v1s owner pre-warm) v1s +0.016.
v1s representative trace (TRX build of k65; us from the first CTA start, critical source of each cluster):
  K/V issued 0.93 (med) | landed med 1.95 / max 2.02 (K/V trip ~1.0 us: 44 KB / SM vs d1s 16 KB / SM -> 0.75) |
  critical src: land 1.98 -> PV 2.59 (compute 0.61: issue-bound, 2 warps x 293 instrs / scheduler) -> bar 2.66 ->
  reader 2.88 (0.22) -> pushed 2.94 -> owner recv 3.17 (0.23) -> out 3.39 (owner 0.22: fold 0.10 + norm/store 0.13).
  nsys 4.06 vs trace out 3.39 (d1s 3.30 vs 2.94).
PF trace (d1s): the pre-warm CALL of the __noinline__ fold (code placed after the kernel = a COLD far jump) delayed the
  compute warps: Q-ready 0.80 -> 1.14, K land max 1.65 -> 1.81; and the 4-warp fold after the wait took 0.16 us for 8 LDS
  (k65's 1-warp 16-source fold 0.24): the post-wait phase is dominated by a fixed latency, not by LDS / ex2 count.
  LESSON: a jump / call into not-yet-fetched code costs ~0.2-0.3 us (DRAM i-fetch after the flush) on whatever path runs it.
C12 (v1s on 8 clusters of 12, 2 tokens / CTA) built but NOT run: SASS 334 compute instrs / warp (C6 293) + reader on all 8
  warps (123) -> ~26% more issue per SM than C6 (key-rows per SM equal, 2 of 8 warps' slots empty); the gather halving
  (22 KB / SM) would be eaten by compute + 2x DSMEM bytes per owner. Dropped on SASS evidence.

## Job 3 results (2026-10-30 00:55, ~4 GPU-min; dev/w9/out_j3.txt)
nsys 3 passes (ns 2 copies = 6 runs): NS d1s -0.027 [-0.032 -0.048 -0.048 +0.032 -0.016 -0.048], v1s -0.016 [-0.016 0 -0.032
  -0.016 0 -0.032] -> below the -0.03 rule: NOT a win (job 2's v1s -0.064 did not replicate). RW +0.117 / +0.144, NS+RW
  +0.091 / +0.123 (the 2-pass loop around the compute path: v1s regs 92 -> 134, compute code becomes a cold jump target).
bench_jt --tlbcold 1 (80 reps, slots k65a ns k65b nsb k65c): d1s 4.23 | 4.23 | 4.28 | 4.29 | 4.32 (NS = controls);
  v1s 5.44 | 5.15 | 5.15 | 5.15 | 5.14 (first-slot k65a outlier; NS = controls).
## Owner-warp contention (from the job-1 d1s trace, owner CTAs = ranks 0-7 vs 8-15, medians)
  d1s: K-land 1.536 / 1.504 | warp-0 PV 1.920 / 1.792 (!) | warp-3 PV 1.792 / 1.792 | merge bar 2.048 / 1.952 | pushed
  2.240 / 2.144: in owner CTAs compute warp 0 shares scheduler 0 with the owner warp, whose PREW pass (pre-warm of the
  fold: stamps d-lds0 1.63 .. d-end0 2.11) runs exactly in the compute window -> owner CTAs push ~0.1 us late, and the
  critical source is almost always an owner (ranks 0-5). v1s (TRX trace): owner CTAs warp-0 PV +0.064, pushed +0.032 (owner
  warps' 8-token new-token partial, 40 shuffles, runs 1.4 -> 2.14 = in the compute window).
  -> PX (d1s): the pre-warm moves to an extra 6th warp at kernel entry (calls the __noinline__ fold on garbage); MRV (v1s).

## Job 4 results (2026-10-30 01:04, ~5 GPU-min; dev/w9/out_j4.txt)
nsys 3 passes: PX (2 copies) d1s +0.099 [+0.064 +0.072 +0.136 +0.128 +0.088 +0.104], v1s -0.016 | MRV d1s +0.005 v1s +0.005.
  PX = 6th warp pre-warms a __noinline__ owner fold at entry; owner calls it after the wait. The start phase / the call
  path cost more than the removed compute-window contention (~0.1 us for owner CTAs). Dead as built.

## Job 5 results (2026-10-30 02:19-02:24 after ~1 h in the gate queue; dev/w9/out_j5.txt)
Owner post-wait probe (dsprobe --only fold; d1s pattern, data received by st.async, cycles; L2-warm code, no flush):
  classic 16-source fold: 1st pass right after the wait 286, 2nd pass (same data) 57 | plain-sum fold 187 / 57 |
  one source (2 LDS) 70 / 33. => the post-wait cost is the first execution (instruction fetch of the fold code, ~10 B/cycle
  even from L2; DRAM after the judge's flush), not smem access to DSMEM-written data and not the max / ex2 math.
nsys 3 passes vs k65 x2: NS d1s -0.030 v1s -0.037 | P0 (XE_PREW=0, no owner pre-warm) d1s +0.088 v1s +0.011 |
  NS+P0 d1s +0.013 v1s -0.027. => the d1s pre-warm is worth ~0.09 us; keep it.
NS freeze checks on dev/w9/ns.py (a52025e5): tools/check.py CHECK_RC=0, 56 PASS, 0 failures; t_xe full rc 0, 0 failures.
=> FROZEN deliver_r1 (NS).

## Dead list (W9, numbers above; do not retry without a new twist)
R0 fixed-reference merges / plain-sum owner (+0.02 / +0.07), R0P / R0P1 prescaled slices (+0.17 v1s), integer red.async
fixed-point owner-free merge (transport no cheaper: probe), PF 4-warp parallel owner fold (+0.27: cold far CALL in the
compute path; the 4-warp fold still 0.16 us), H16 FP16 pushes (d1s +0.05, v1s 0), QS split QK chains (0), EXH / XH / PW2 (0),
RW reader warm-up loop (+0.12 / +0.14), PX extra pre-warm warp + __noinline__ owner fold (+0.10), MRV (0), P0 (+0.09),
C12 v1s geometry (SASS +26% issue per SM), BN / BR bounds: norm <= 0.056 us, v1s CTA merge <= 0.056 us.
