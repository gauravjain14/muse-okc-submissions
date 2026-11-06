# W10 NOTES: DRAM efficiency of the full-layer B16/32 DECODE kernels (d16f / d32f = run1f attn_1_0_1_0)
Owner W10 (round 4), dir dev/w10 only. Base dev/coord/k65.py (md5 b76344ec; k67 = BEST has the same run1f). GPU cap 40 min.
SCOPE (coordinator 00:20): W10 = DECODE full only + the shared timeline tools; VERIFY full (v16f / v32f, _XD_SRC) moved to
W11 (dev/w11), which reads dev/w10/tl_an.py + these notes (keep them readable; W11 never writes here).
GPU only via: bash tools/gate.sh tools/modal_run.py --tag w10_jN --cmd "cd /root/p && ls dev/coord dev/w4 dev/w10 tools >/dev/null && bash dev/w10/jN.sh"
(only dev/coord, dev/w4, dev/w10 uploaded: every helper tool used by a job is COPIED into dev/w10).

## Resume here
- FINAL (02:30): FROZEN dev/w10/deliver_r1/ (USE THIS): `python3 dev/w10/deliver_r1/build.py dev/coord/k69.py <out>`
  (defaults W10_GK=1 W10_BMAX=16) -> k69w10.py md5 5afe97d1 (== the J8-measured gka.py); stripped k69w10s.py md5 17ac0d04,
  869,538 B, cubin unchanged by stripping (mono bc6a8038). Change: CT2 (warp 1 touches the post-barrier constant-bank lines
  + loads k_scale / v_scale at entry while warp 0 waits for the lengths) compiled ONLY into a new kernel attn_1_0_1_0g used
  for decode full page-1 B <= 16; attn_1_0_1_0 (d32f, all tests with B > 16) is k69's code instruction for instruction.
  Gates: check.py 56 PASS / 0 failures, edge2 --quick 16 / 0. nsys (J8, vs k69): d16f -0.17 (all 4 runs -0.25..-0.13),
  d32f -0.13 (identical kernel: placement noise). bench_jt tlbcold 1: d16f +0.13 (+0.14 %), d32f -0.35 -> JTKEEP yes.
  Expected judge: ~+0.02 % (d16f -0.17 us x 0.158 %/us). GPU total ~63 of 65.
- Decode start dissected (J5 fine clock64 stamps, k67 thread 0, d16f | d32f): lens scanned 0.54 | barrier 0.61 | FP64 split
  0.96 | b_begin 1.08 | segment head 1.94 | 2.44 (the p.ovh constant-bank line 0x2c0 misses; slower at B 32 because the
  11 MB page-table superset prefetch floods DRAM) | rows issued 2.19 | 2.66. CT2 (warp 1 touches the constant lines + scale
  values at entry) moves rows to 1.86 / 1.87 and d32f's t_s 8.8 -> 7.8, but at B 32 the start window is NOT idle DRAM
  (the superset prefetch fills it): an earlier K/V start is zero-sum there, and every variant's re-scheduled hot loop
  costs ~0.1-0.5 us. Pattern over 7 jobs: start fixes give d16f -0.15..-0.25 (8/8 runs) and d32f +0.06..+0.13.
- Best piece cp = CT2 + PB2 (k67 + `build.py W10_CT=2 W10_PB=2`; check.py 56 / 0; edge2 quick 16 / 0): nsys pooled J5 + J7
  d16f -0.21 (8/8 < 0), d32f +0.112 (rule <= +0.10 fails); bench_jt tlbcold 1 -0.16 / -0.44 (JTKEEP yes) but tlbcold 0
  sanity d32f +1.13 -> not frozen by the J7 rule. A B <= 16 gate of CT2 alone (W10_CT=2 W10_BMAX=16) is the only untested
  follow-up (expected d16f ~-0.15 = +0.02 % score).
- FS (rows of the first segment issued before the prefix barrier, every warp scanning the lengths): DEAD (+0.5 / +0.7; the
  4 extra live registers re-schedule the hot loop: f6 = FS off at B 32 still +0.34 there).
- The coordinator's evict_first lever is already in run1f (WF_CPH 2 on every full-tile K/V copy; SASS checked).
- Round 1 (k65 base) summary: nothing frozen; start = the pool (7.4 / 8.2 us); RG / PB / PR1 / PF / OVH dead; KS1 / KS2
  d16f -0.25 / -0.12 but d32f +0.13 / +0.11 (bench_jt JTKEEP no).
- J5 done (base k67): CT2 + PB2 (cp) KEEP by rule but marginal (d16f -0.15, d32f +0.06); CT1 / CT2 alone fail (d32f +0.13).
  Fine-stamp dissection of k67 (thread 0, from CTA entry, d16f | d32f): lens scanned 0.54 | barrier 0.61 | split (FP64 divs)
  0.96 (+0.35) | b_begin 1.08 | segment head 1.94 | 2.44 (+0.8 | +1.3: the 0x2c0 constant-bank line (p.ovh) misses, slower
  at d32f because the page-table flood fills DRAM) | rows issued 2.19 | 2.66. CT2 cuts the segment-head step to +0.44 / +0.46
  (rows 1.86 / 1.87; d32f t_s 8.83 -> 7.81) but d32f's kernel does not get faster: at B 32 the start window is NOT idle DRAM
  (the 11 MB page-table superset prefetch fills it), so an earlier K/V start is zero-sum there; d16f (5.5 MB) gains.
- Coordinator's evict_first lever: ALREADY IN run1f (WF_CPH 2 = L2::evict_first createpolicy on every full-tile K/V
  cp.async since k9f; SASS: the createpolicy pair R98/R99 -> desc[UR4/UR22..UR32] on the fast-path LDGSTS). Only the
  segment-end partial tiles (<= 1 per lane per segment) use plain cp.async. The end chain's lines (accumulators, gates,
  page table) are normal priority and survive. Nothing to port.
- J6 prepared (dev/w10/j6.sh): FS (every warp scans the lengths itself and issues its first-segment rows before the prefix
  barrier and before the page-table flood), FS + CT2 + PB2, CT2 + PB2 again; gates (edge2 full + edge_sw) first.

## GPU ledger
- J1 w10_j1 00:12:42-00:20:26 (~8 GPU-min wall incl. start; job 446 s).
- J2 w10_j2 00:29:30-00:38:42 (~9.5 GPU-min; job 518 s).
- J3 w10_j3 00:48:14-00:55:24 (~7.2 GPU-min; job 418 s).
- J4 w10_j4 ~00:59-01:03:50 (~5 GPU-min; job 230 s). Total ~30 (round 2 cap: 65).
- J5 w10_j5 01:15:5x-01:22:24 (~7 GPU-min; job 373 s). Total ~37.
- J6 attempt 1 LOST: my local `timeout 1500` wrapper expired while gate.sh queued ~23 min behind another worker; the app
  (ap-kYv5p58) kept running headless (~9 GPU-min, output not retrievable). Never wrap gate.sh in a short timeout. Total ~46.
- J6 w10_j6 (rerun) 01:53:32-01:59:55 (~7 GPU-min; job 372 s). Total ~53.
- J7 w10_j7 02:10:21-02:14:43 (~5 GPU-min; job 252 s). Total ~58.
- J8 w10_j8 02:22:28-02:26:33 (~5 GPU-min; job 236 s). Total ~63.

## Files
- build.py <base> <out> [W10_PR=N W10_RG=1 W10_PB=1 W10_XQ=1 W10_KS=1 W10_OVH=N]: knobs default 0 => run1f / xd0 / xd1
  cubins == k65's (local NVRTC 13: mono 3578cad0, xd0 d5806e8f, xd1 a07f4682). Always inserts the W10 run1f stamps (inert
  unless -DW10TR). PB / KS restructure the source text only when on.
- trace_dec.py (run1f per-tile stamps -> npz), trace_xd.py (= dev/w2/trace.py: XD W2 stamps, pair tops), probe.py (lane3h
  gather probe on the real rows), tl_an.py (local: landing TB/s per 0.5-us bin + loss split start / steady / tail / end vs
  4.39 TB/s; modes dec / probe / xd), decide.py (nsys table + rule), loopdiff.py (hot-loop SASS diff), sass_r1f.py /
  sass_xd.py (local NVRTC 13), copies of nsys_go / nsys_multi / edge2 / edge_sw / xcheck.
- out_jN_*.txt (nsys / decide / timelines / bench_jt per job); raw npz in out/20261030T002026_w10_j1/x/ (k65 + RG + XQ traces,
  probe), out/20261030T003842_w10_j2/x/ (KS1 trace), out/20261030T005524_w10_j3/x/ (PF trace). pool.py pools deltas over jobs.
- STAMP CAVEATS: globaltimer ticks are 256 ns here; the per-tile "land" stamp is when the lane passes its cp.async wait +
  lane barrier, i.e. CONSUMPTION time (the median data wait is 0: the data was already there), not DRAM arrival.
  Drain bins can read > 4.8 TB/s for that reason. The steady slope (25-75 % of bytes) and t_s are still valid.

## Loss split (tl_an.py, J1, k65 traced; vs 4.39 TB/s; us)
| shape | kend | lost | start t_s | steady (BW_s) | tail | end | notes |
| d16f | 85.8 | 11.6 | 7.36 | 0.06 (4.387) | 0.38 | 3.20 | rows issued 2.30 (prefix 0.77), burst issue 3.0-6.1, first consumption 5.25, tile0 7.17 |
| d32f | 164.6 | 15.6 | 8.19 | 3.16 (4.299) | 1.41 | 3.07 | 26 two-segment CTAs |
| v16f | 94.1 | 19.2 | 5.70 | 3.59 (4.187) | 3.86 | 5.65 | (W11) copy0 3.41, prologue 4.03, tile0 5.90 |
| v32f | 174.9 | 25.2 | 5.50 | 10.47 (4.102) | 4.59 | 5.58 | (W11) |
- Decode start anatomy (d16f medians): prefix barrier 0.77 | rows issued 2.30 | tile k = 0, 1, 2 issued 3.33 / 4.10 / 5.63
  (24 LDGSTS per warp > the ~16 in-flight cap: the 3rd stage waits for credits) | prologue issued 6.14 | Q / new done 6.66 |
  tile 0 consumed 7.17, tile 1 10.62 (iteration 0 = 3.45 us: its refill waits 2.30 us for row(3) queued behind the burst;
  steady refill delay 1.02) | landing hole 7.5-10 us at 1.2-1.9 TB/s | steady from ~11 us.
- Decode 2-segment CTAs finish 4-5 us BEFORE 1-segment CTAs (d16f end med 77.8 vs 82.9, max 79.5 vs 85.8; d32f 156.9 vs
  161.2, max 160.3 vs 164.6); seg-0 loop exit -> seg-1 first consumption 6.7-6.9 us. OVH (1536) over-compensates, but the
  tail term is only 0.4 / 1.4 us (DRAM stays saturated while the 1-seg CTAs finish): OVH tuning is bounded by that.
- Probe (lane3h, rows from host) ran at only 3.7-3.9 TB/s steady here (its refill waits 2.3 us for an uncached page-table
  LDG): NOT a ceiling reference; the decode kernel's own steady slope (4.30-4.39) is the best measured gather.
- Verify (for W11): pair-top stamps; steady 4.10-4.19 TB/s (below decode's 4.30-4.39); tail ~4 us; end ~5.6 us.

## J1 RESULT (nsys medians, 2 copies x 2 passes, controls k65: d16f 87.10, d32f 164.98, v16f 92.12, v32f 174.74)
| arm | d16f D | d32f D | v16f D | v32f D | verdict |
| b = PB (dup issue code, +496 SASS) | +0.00 | +0.02 | (= k65 XD) +0.06 | +0.26 | no |
| a = RG + XQ | +0.82 (+0.66..+0.91) | +0.74 | +0.48 (+0.39..+0.57) | +0.51 | DEAD both |
| c = RG + PB + XQ | +0.89 | +0.73 | +0.70 | +0.47 | DEAD |
- RG trace (c): start t_s 7.36 -> 6.53 (the post-burst hole is filled: I(3)-L(0) 2.30 -> 1.15) but the lanes' finish spread
  widens (last consumption p0 / p10 67.2 / 75.0 -> 61.6 / 68.4, max 82.6 -> 84.9): tail 0.38 -> 3.26 us. Cause not isolated
  (+1 LDGSTS per tile per warp under the per-warp in-flight cap, or the re-scheduled 603-instr loop).
- XQ trace (W11): copy0 3.41 -> 3.07, prologue 4.03 -> 3.74, t_s 5.70 -> 5.00, but tile0 unchanged and steady BW lower;
  nsys +0.48 / +0.51 -> dead as built.

## J2 (w10_j2) - DECISION RULES (written before the job; decode only, shapes d16f + d32f)
Arms (2 copies x 2 passes, controls k65a/b first): ks = W10_KS; kp = KS + PB (single-copy PB: the new-token + Q^T work
inside the rolled prologue loop before stage 2, SASS 7448, loop 579); kr = KS + PR=1 (row of the first refill from smem,
loop 602); o10 / o06 = k65 with host OVH 1024 / 640 (cubins == k65).
Smoke: edge2 --quick on every arm (page 64 + segment cases); a failing arm is not timed.
- KEEP a piece iff D <= -0.10 on >= 1 decode shape with all 4 runs below the same-pass control mean and the other shape
  D <= +0.10 (stack-every-small-win rule).
- KS family pick: ks vs kp vs kr by score 0.158 (-D d16f) + 0.087 (-D d32f); kp / kr only if they beat ks by >= 0.01 %.
- OVH pick: the kept value with the larger score; then J3 stacks it on the KS pick (host + kernel are independent).
- Trace of ks (8 reps): explains (rows issued expected ~1.5 vs 2.30), never decides.
- J3 = the stack: check.py 0 failures + edge2 full + edge_sw + bench_jt --tlbcold 1 (2 copies vs 2 k65 + unjudged k65 last)
  -> freeze deliver_r1.

## J2 RESULT (nsys medians, 2 copies x 2 passes; controls k65: d16f 86.95 spread 0.19, d32f 164.76 spread 0.13)
| arm | d16f D (runs) | d32f D (runs) | verdict |
| ks = KS1 (scales through smem) | -0.26 (-0.32..-0.15) | +0.20 (+0.00..+0.31) | no (d32f) - re-measured in J3 |
| kp = KS1 + PB (single copy) | -0.27 | +0.28 | no: PB adds nothing |
| kr = KS1 + PR1 | +3.14 | +6.81 | DEAD: the predicated in-loop LDS made ptxas move the row LDG (steady-state slowdown) |
| o10 = OVH 1024 | +0.10 | -0.00 | no |
| o06 = OVH 640 | +0.32 | +0.74 | no: the early 2-seg CTAs cost nothing (DRAM-bound); OVH 1536 stays |
- KS1 trace: rows issued 2.30 -> 2.05 (d16f first tile issue 3.33 -> 2.82, t_s 7.36 -> 6.66) but d32f's first issue stays
  3.84 (rows trip 1.8 us vs 0.8 at d16f: they queue behind WS_PTP 10's 5/8-slice bulk page-table prefetch, ~6.9 MB at
  B 32 issued at the length barrier: WS round 8's known d32f start delay). KS1's loop is re-scheduled (152 diff lines).
- The scale load was a partial trip (~0.25 us), not the full 0.8 assumed.

## J3 (w10_j3) - DECISION RULES (written before the job)
Arms: ks (KS1, again), k2 (KS2: warp 0 lane 0 prefetch.global.L2 of the k_scale / v_scale lines at entry; tile loop SASS
identical to k65's, +8 SASS), p2 (KS2 + PF2: lanes 0-1 of each warp prefetch.global.L1 the line(s) of their first refill
row right after the first rows; loop same size, re-scheduled 266 lines). Smoke edge2 --quick on k2 / p2.
- KEEP rule as J2 (D <= -0.10 on >= 1 decode shape, all runs below the same-pass control mean, other shape <= +0.10).
  ks: the rule on the POOLED J2 + J3 deltas (8 runs per shape, each vs its own job's controls).
- Pick = the kept arm with the largest score (0.158 d16f + 0.087 d32f per us); p2 beats k2 only by >= 0.01 %.
- Gates in-job for k2 and p2: tools/check.py 56 PASS / 0 failures, edge2 full + edge_sw 0 failures.
- J4 = bench_jt --tlbcold 1 on the pick (2 copies vs 2 k65 copies + an unjudged k65 last): JTKEEP iff mean(pick) <
  mean(k65) on >= 1 decode shape and on no shape mean(pick) > mean(k65) x 1.003; then freeze deliver_r1.

## J3 RESULT (fast container: controls k65 d16f 83.74 spread 0.08, d32f 158.04 spread 0.27)
| arm | d16f D (runs) | d32f D (runs) | verdict |
| ks (KS1) | -0.25 (-0.30..-0.20) | +0.05 (-0.03..+0.18) | J3 alone KEEP; pooled with J2: d16f -0.252 (8/8 < 0), d32f +0.126 (2/8 < 0) -> NOT kept (rule) |
| k2 (KS2) | -0.14 (-0.17..-0.12) | +0.04 (-0.15..+0.25) | KEEP (pick); loop SASS == k65's, so d32f is noise |
| p2 (KS2 + PF2) | +0.05 | -0.00 | no: PF's loop re-schedule eats KS2's gain; trace: first refill wait 2.30 -> 2.05 only |
Gates on k2a.py (md5 1aeb95e1): check.py 56 PASS / 0 failures; edge2 full 26 / 0; edge_sw 42 / 0 (p2 also all pass).

## J4 (w10_j4) - DECISION RULE (written before the job): bench_jt on k2 (PICK=k2)
bench_jt --tlbcold 1, 40 reps, order k65, k2, k65b, k2b, k65c (k65c unjudged, last-slot sink). JTKEEP iff (jt_decide.py)
mean(k2 copies) < mean(k65 copies) on >= 1 decode shape and on no shape mean(k2) > mean(k65) x 1.003. --tlbcold 0 = sanity.
nsys 2 x 2 repeat (k65 vs k2) is pooled with J3 and reported. JTKEEP yes -> freeze dev/w10/deliver_r1 (mk_deliver.py).

## J4 RESULT (bench_jt judge pairs, 40 reps; order k65, k2, k65b, k2b, k65c (sink))
| | k65 / k65b | k2 / k2b | diff of means |
| d16f tlbcold 1 | 88.44 / 88.64 | 88.42 / 89.51 | +0.43 (+0.48 %) |
| d32f tlbcold 1 | 167.17 / 165.39 | 165.87 / 164.98 | -0.85 (-0.51 %) |
| d16f tlbcold 0 | 84.18 / 84.13 | 84.60 / 83.91 | +0.10 |
| d32f tlbcold 0 | 158.36 / 158.26 | 158.28 / 158.70 | +0.18 |
-> JTKEEP no (d16f > +0.3 %). nsys repeat (fast container, k65 83.70 / 157.94): k2 d16f -0.10 (-0.21..-0.06), d32f +0.18;
pooled J3 + J4 (8 runs): d16f -0.120 (8 / 8 < 0), d32f +0.112. KS1 pooled J2 + J3: d16f -0.252 (8 / 8), d32f +0.126.
Reading: both KS forms move the row loads ~0.25 us earlier; at B 16 that is a real start gain, at B 32 the rows then queue
behind WS_PTP 10's page-table flood and the earlier start re-orders the flood vs the burst for a small loss (same sign with
an identical tile loop, so not codegen). Not frozen.

## J5 (w10_j5, base k67) - DECISION RULES (written before the job)
Arms (2 copies x 2 passes, controls k67a/b first): c1 = CT1, c2 = CT2, cp = CT2 + PB2 (PB2: new-token + Q^T work before stage
1, i.e. only stage 0's 8 LDGSTS / warp ahead of it). Local SASS: 128 regs / 0 spills; loops 582 / 582 / 579 (re-scheduled
222 / 152 / 363 lines vs k67). No new waits / polls (no hang test needed). Smoke edge2 --quick on each.
- KEEP iff D <= -0.10 on >= 1 decode shape with all 4 runs below the same-pass control mean and the other shape <= +0.10.
- Pick = largest score; cp over c2 only if >= +0.01 %. Keeper -> J6: bench_jt --tlbcold 1 (balanced: k67, X, k67b, Xb, k67c
  dummy last) + check.py / edge2 full / edge_sw -> freeze deliver_r1 (builder on k67).
- Traces k67t / c2 with the fine stamps: dissect the barrier -> rows window (explain, never decide).

## J5 RESULT (base k67; controls d16f 83.67 spread 0.19, d32f 157.99 spread 0.05)
| arm | d16f D (runs) | d32f D (runs) | verdict |
| c1 = CT1 | +0.18 | +0.13 | no |
| c2 = CT2 | -0.17 (-0.30..-0.02) | +0.13 (+0.03..+0.34) | no |
| cp = CT2 + PB2 | -0.15 (-0.26..-0.08) | +0.06 (-0.03..+0.13) | KEEP (marginal) |

## J6 (w10_j6) - DECISION RULES (written before the job)
Arms (revised after the lost first attempt): fs = FS (B <= 32), fc = FS + CT2 + PB2, f6 = FS for B <= 16 only (W10_BMAX=16:
d32f keeps k67's start path, same hot loop as fs). Gates: fs edge2 FULL + edge_sw; fc / f6 edge2 --quick. FS local SASS: 128 regs / 0 spills, loop 587 (535
diff lines); rows LDG at SASS 0x0b60 before warp 0's prefix (0x0d70) and the barrier (0x1130). No new waits.
Gates before timing: edge2 FULL + edge_sw 0 failures (FS changes which code path loads the first rows).
- KEEP rule as J5. cp: on the pooled J5 + J6 deltas. Pick = largest score among kept {fs, fc, cp}.
- If fs / fc help d16f but break the d32f rule, the B <= 16 gate (W10_BMAX=16; d32f takes k67's start path) is the J7
  candidate. J7 = bench_jt --tlbcold 1 of the pick (balanced, dummy last) + check.py, then freeze deliver_r1 (builder on k67).

## J6 RESULT (base k67; controls d16f 86.94 spread 0.08, d32f 164.88 spread 0.48; gates fs edge2 full 26/0 + edge_sw 42/0,
## fc / f6 edge2 quick 16/0)
| arm | d16f D (runs) | d32f D (runs) | verdict |
| fs = FS | +0.52 (+0.43..+0.62) | +0.72 (+0.60..+0.81) | DEAD |
| fc = FS + CT2 + PB2 | +0.70 | +0.82 | DEAD |
| f6 = FS for B <= 16 | +0.54 | +0.34 (d32f runs k67's start path: the +0.3..0.5 is the re-scheduled hot loop, 535 diff lines) | DEAD |
- FS's hot-loop re-schedule (4 more live registers through the start) costs ~0.3-0.5 us of steady state, more than the start
  can give; the traced FS build also spilled (w10f rows stored to local -> the row trip serialized before warp 0's prefix:
  barrier 0.61 -> 1.68, burst later). Not pursued further (every row-earlier variant changes the loop schedule).

## J7 (w10_j7) - DECISION RULE (written before the job): cp = CT2 + PB2 (the only J5 / J6 rule-passing piece)
check.py on cpa.py: 56 PASS / 0 failures required. bench_jt --tlbcold 1, 40 reps, order k67, cp, k67b, cpb, k67c (unjudged,
last): JTKEEP iff mean(cp) < mean(k67) on >= 1 decode shape and on no shape mean(cp) > mean(k67) x 1.003 (jt_decide.py);
--tlbcold 0 sanity. nsys 2 x 2 repeat pooled with J5: pooled D <= -0.10 on d16f with all 8 runs < 0 and pooled d32f <= +0.10.
All three -> freeze dev/w10/deliver_r1 (mk_deliver.py, builder on k67, knobs W10_CT=2 W10_PB=2); else nothing frozen.

## J7 RESULT (cp = CT2 + PB2 vs k67)
- check.py 56 PASS / 0 failures.
- bench_jt tlbcold 1 (k67, cp, k67b, cpb, k67c sink): d16f k67 93.05 / 98.14 vs cp 92.62 / 98.26 (-0.16); d32f 168.77 / 167.68
  vs 167.99 / 167.58 (-0.44) -> JTKEEP yes. tlbcold 0: d16f +0.05, d32f +1.13 (cp 167.00 / 167.21 vs 166.15 / 165.80).
- nsys repeat (controls 86.92 / 164.78): d16f -0.27 (-0.40..-0.11), d32f +0.17 (-0.09..+0.45). Pooled with J5: d16f -0.212
  (8/8), d32f +0.112 (2/8 < 0) -> rule fails (d32f > +0.10). NOT frozen.

## J8 (w10_j8, base dev/coord/k69.py md5 df7dad27; run1f text == k65 / k67's) - DECISION RULE (written before the job)
gk = `build.py dev/coord/k69.py <out> W10_GK=1 W10_BMAX=16`: run1f<PAGE1, DST, W10G> template flag; CT2's warp-1 entry block
only in a NEW kernel attn_1_0_1_0g (= run1f<true, 3, true>), launched for decode full page-1 B <= 16. Local NVRTC 13:
attn_1_0_1_0 in gk == k69's instruction for instruction at the same addresses (only global label / internal-function
numbers differ: 7416 SASS, normalized md5 18cce3f2); attn_1_0_1_0g == the J5-measured CT2 kernel (loop diff 0, SASS 7472).
Gates: check.py 56 / 0, edge2 --quick 0 failures. FREEZE deliver_r1 iff d16f D < 0 in all 4 nsys runs (vs the same-pass
k69 mean; coordinator rule) and bench_jt --tlbcold 1 JTKEEP yes (gk vs k69, copies interleaved, k69c dummy last).

## J8 RESULT: FREEZE (rule met: d16f 4/4 runs below control, d32f kernel SASS identical, JTKEEP yes)
| | k69 | gk | D |
| nsys d16f (controls 86.79, spread 0.21) | | | -0.17 (-0.25..-0.13) |
| nsys d32f (controls 164.86, spread 0.48) | | | -0.13 (-0.40..+0.09), same kernel code |
| bench_jt tlbcold 1 d16f | 93.12 / 97.99 | 93.28 / 98.09 | +0.13 (+0.14 %) |
| bench_jt tlbcold 1 d32f | 170.35 / 169.83 | 169.75 / 169.73 | -0.35 (-0.21 %) |
