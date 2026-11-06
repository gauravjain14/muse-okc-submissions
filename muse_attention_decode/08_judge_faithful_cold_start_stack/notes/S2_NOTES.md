# S2: v1f (xg_v8f in k56's _XG5_SRC) tile-phase de-convoy (worker S2, started 2026-10-29)

## Resume here (ROUND 4 DONE: combine knobs within noise, nothing stacked; deliver_r2 stays the hand-back; ~33.3 GPU-min used of the 39 lineage cap)
- HAND-BACK = dev/s2/deliver_r2/ (build.py <base> <out>, defaults --mfree 1 --alb 1 --kf 1; k56_s2r2.py md5 1c92c15f5a70108689d96214661b12a0 =
  build(dev/coord/k56.py md5 427cc112); nsys_table.txt, out_j6.txt, md5s.txt). v1f nsys -0.29 us vs k56 controls (4/4 runs 11.90-11.98 vs 12.19-12.27);
  t_xg full 60 PASS, t_s2 75 PASS, tools/check.py 56 PASS / 0 failures; 127 regs, 0 spills, same pre-existing C7511.  deliver_r1 (MFREE+ALB, -0.165) is the
  fallback.  Knobs at 0 (--mfree 0 --alb 0 --kf 0) reproduce k56's SASS byte for byte (cubin md5 5fe1e05c68ed).
- Round 1 (dead): ping-pong token ring, BD, MF3, PF1, MM; see below.  Remaining bound: NOMASK 11.74-11.87 in j5 (r2 is already at 11.90-11.98).
- Next lever candidates: other skip-branches in the hot instances (ASK vote branch, the `if (lo_...)`-style guards are gone; check BSSY/BSYNC counts in
  r2.sass), M's cold fetch, rest of the 3 x issue_def copies is now 1.

## Local facts (NVRTC 13, no GPU)
- k56 _XG5_SRC: 127 regs, 0 spills, C7511 (wgmma serialized), cubin md5 5fe1e05c68ed. SASS: 16 QGMMA and 16
  WARPGROUP.DEPBAR.LE (one after EVERY wgmma): each of the 4 QK and the 4 PV wgmmas of a pair step completes before the next
  one issues, and V^T runs after QK completed (not under it). Non-serialized builds (XG_MF=0 XG_CP=1): 10 DEPBAR.
- build.py defaults -> cubin md5 identical to k56 (cmp of the nvdisasm output: SASS_IDENTICAL).
- Knob builds: all 123..128 regs, 0 spills, C7511 unchanged, 16 QGMMA / 16 DEPBAR, new BAR.SYNC/BAR.ARV with count 0x100.
- Step counts per WG: steps(m tiles) = m ? m/2 + 1 : 0 (verified against the loops); parity 0 has m = (nt + 1) / 2 tiles,
  parity 1 m = nt / 2, so SA - SB in {0, 1}.

## Design (what was built and why it differs from brief_s2's one-line sketch)
Ring tokens (named barriers 5..9, count 256 = one WG arrives, one WG syncs). A WG, right after its tile data wait, does
bar.sync on its own edge (the first A step does not wait), then runs QK -> V^T -> softmax -> PV, and bar.arrives on its
successor's edge at the release anchor XG_PPR (1 after the QK commit, 2 after V^T + QK wait, 3 after row max/bias, 4 before
the PV issue, 5 after the PV commit). Releasing only after the PV commit (the brief's wording) makes the ring a strict
A-chain then B-chain (period = 2 x chain, no overlap of anything but the PV tail), so the anchors are a knob; the useful
anchor is mid-chain: period ~ ring size x anchor offset, stagger = offset. PPN 4: A0 -> B0 -> A1 -> B1 (quarter-step
stagger), PPN 2: A_h -> B_h per head (half-step stagger between the parity groups, two independent rings). Exact arrive
counts: the last round when SA = SB + 1 has only the A WGs -> in PPN 4 A0 releases A1 directly (edge 9, skipping B0), A1
releases nothing; B's last release is skipped when A has no next step.
Deferred wave (brief: parity-0 WGs issue the whole wave): implemented as XG_BD instead = every WG still issues its own 2
rows per deferred tile (mbarrier counts unchanged, no page-table entry sharing between warps, no extra registers), but the
parity-1 WGs do it at their step-0 START (after the tile data wait, before the token wait / stagger sleep) and the parity-0
WGs at their step-0 PV commit as in k56. Same purpose (a delayed parity-1 chain does not delay half of every deferred
tile) and the parity-1 half goes out earlier than k56's, not later. A faithful parity-0-issues-all (PD) needs the partner
warps' page-table entries in the A warps' registers (an e1 lane range + 2 more loads); not built unless BD shows the
deferred timing matters.
Known risk: CTA 131 (new block): WG 0 (= A0, first in the ring) runs the RMSNorm / e4m3 new-block code before step 0, so in
PPN 4 the other three WGs wait for it at step 0; PPN 2 only delays ring h=0.

## Decision rules (written before each job)
### Job s2_j1 (exploration; 1 pass x 2 copies; controls k56a, k56b first; 11 variants; shape 1,8,full; nsys 120 replays)
Per variant: smoke = t_xg --quick (6 PASS) + t_s2 --nts 3,7,10 --extra 0 --calls 2 on s2a under timeout 120; a variant with
rc != 0 / FAIL / timeout is dropped (its nsys is skipped).  Delta = variant copy median - mean(k56a, k56b medians).
- advance to job 2 (2 passes x 2 copies, plus t_s2 all-nt full + t_xg full): variants with mean delta <= -0.08 and both
  copies <= -0.05, best 3 at most (plus the best BD-only/STAG row as a reference for the deferred change).
- a PP anchor/size whose mean delta >= +0.10 is dead (do not refine its neighbours in the same direction).
- if no PP variant has a mean delta <= -0.05: one refinement job at most (the neighbouring anchor of the best PP variant,
  PD true, STAG with PP), then stop and report per-step timings (trace) per the brief.
- keep rule (freeze): >= 0.10 us faster in both copies and both passes + t_s2 + t_xg full 0 failures + check.py 56 PASS.

## Job s2_j1 RESULT (2026-10-29 13:24-13:30, ~6.5 GPU-min incl. startup; dev/s2/out_j1.txt): PP DEAD, BD loses
Smoke: all 11 variants t_xg --quick 6 PASS + t_s2 nt 3,7,10 x 2 calls 6 PASS, 0 failures, xg path taken. nsys v1f medians (us,
shape 1,8,full, 120 replays; copy a / copy b):  controls k56a 12.40, k56b 12.35 (mean 12.375)
  BD                       12.67 / 12.67 (+0.29)   | BD + STAG 400            12.93 / 12.77 (+0.48)
  PP4 R1 BD  14.30 / 14.27 | PP4 R2 BD  14.91 / 14.98 | PP4 R3 BD  16.42 / 16.40 | PP4 R4 BD  16.74 / 16.83
  PP2 R2 BD  13.70 / 13.70 | PP2 R3 BD  13.79 / 13.79 | PP2 R4 BD  13.94 / 14.03 | PP2 R5 BD  14.10 / 14.14
  PP4 R2 (no BD) 14.53 / 14.56  (so BD costs ~+0.37 on top of PP too)
Verdict: every ring variant is +1.3 .. +4.4 slower (kill rule: >= +0.10 = dead). Ordering the WG chains with named barriers makes
the step period ~ ring size x anchor offset: the anchor offsets (QK commit/wait ~0.55 us, V^T+wait ~1.0 us at 4 WGs with
serialized wgmma) are far larger than the quarter/half-step stagger the ring is supposed to buy, so tensor and ALU phases of
different WGs are not overlapped, they are queued. BD alone +0.29: moving the parity-1 half of the deferred wave ~0.5 us EARLIER
(step start instead of PV commit) delays the first wave (the DEFER 4 lesson again; later issue was flat in P2: STAG 400/800).
Throughput view (XB job 7): 1 / 2 / 4 WGs per SM = 537 / 566 / 994 cycles per iteration per WG = 537 / 283 / 248 cycles per unit,
so the 4-WG convoy is the MOST throughput-efficient concurrency; two-at-a-time ping-pong (2 WGs) moves toward 283/unit.

## Finding from the SASS (local, before job 2): the production tile loop is wgmma-SERIALIZED (C7511), caused by the MF dump
k56 SASS: WARPGROUP.DEPBAR.LE gsb0,0 after EVERY QGMMA (16 of 16): the 4 QK and 4 PV wgmmas of a pair step run one at a time
and V^T cannot overlap the QK. Bisect (lab.py, NVRTC 13): the fold half of XG_MF is clean; the DUMP half (`if (par)` block of
MF: il = 1/l, f16 cvt, xv[20], uint4 stores) alone triggers C7511; even a dump of xor'ed raw accumulators does; a dump that stores
the accumulators directly (float4, like XG_CP 1 / the old parity merge) does not. XG_MF 3 (mf3.txt, added to build.py): parity-1
WGs store raw o[0..35] + mr as float4 (36 KB total vs MF's 20 KB), the parity-0 WG folds (o e0 + partner e1) / l in registers
(same arithmetic as MF's fold, partner O unnormalized) and stores the same tagged f16 words -> 126 regs, NO C7511, 10 DEPBAR / 6
ARRIVE: QK x4 back to back, wait<1>, V^T under the QK, PV x4 back to back (see s2a_mf3.sass). Prior XG evidence that C7511 is
harmless (TB1 MF 11.30 vs CP 11.38) compared builds with DIFFERENT publish code (MF publish was ~0.8 us shorter per CTA in the
traces), so it does not isolate the tile loop. Expected cost of MF3: +~0.15 us smem traffic in the publish.
### Job s2_j2 decision rule (written before)
nsys v1f 1,8,full, controls k56a/k56b first, pass 1 and pass 2: MF3 (2 copies x 2 passes) + PP variants on top of MF3 (1 pass x 2
copies: PP2 R2, PP2 R3, PP4 R1, PP4 R2; no BD) + trace_s2.py of control and MF3 (step timeline).
- MF3 is a candidate iff its mean over 4 runs is <= control mean - 0.10 and each run < both controls' same-pass values; then job 3 =
  t_s2 all-nt + t_xg full + check.py (56 PASS) [+ bench] and freeze deliver_r1.
- MF3 within +-0.05 of control: serialization is harmless in situ (XG's conclusion confirmed); PP-on-MF3 rows say whether the
  de-convoy needed the pipelined chain. Then stop and report.
- MF3 slower by >= 0.05: its publish volume (36 KB vs 20 KB) eats the tile gain -> only then try a lower-volume unserialized dump.
- A PP-on-MF3 variant beating its MF3 row by >= 0.10 is a PP win (confirm in job 3).

## Job s2_j2 RESULT (2026-10-29 13:35-13:39, ~4.5 GPU-min; dev/s2/out_j2.txt): MF3 slower; step traces taken
Cumulative GPU: j1 6.5 + j2 4.5 = ~11 min.  nsys v1f medians (copy a / copy b), pass 1 | pass 2:
  controls k56a/k56b 12.32 / 12.32 | 12.38 / 12.35 ;  XG_MF=3 (unserialized: 126 regs, no C7511 on the image either) 12.86 / 12.90 | 12.86 / 12.90 (+0.52)
  MF3 + PP2 R2 13.15 / 13.18 | PP2 R3 13.46 / 13.50 | PP4 R1 13.95 / 13.92 | PP4 R2 14.38 / 14.40 (pass 1 only): PP is dead on the pipelined chain too.
Smoke: MF3 t_s2 (all nt 1..10 + extras, 3 calls) 60 PASS, 0 failures; PP-on-MF3 quick/odd-nt 0 failures.
Step traces (XG_TRACE, us from the first warp, median over CTAs / max; WGs A0 A1 B0 B1 = wg 0 1 2 3; k56 | MF3):
  step 0 go 3.04 3.03 3.18 3.18 | 3.08 3.06 3.23 3.22 ; deferred issued 4.38 4.35 4.51 4.48 | 4.43 4.42 4.54 4.52
  step 1 go 4.67 4.64 4.80 4.80 | 4.74 4.70 4.90 4.86 ; step 2 wait 6.24 6.21 6.34 6.30 | 6.22 6.11 6.40 6.35 ; step 2 go 6.37 6.35 6.46 6.42 | 6.42 6.30 6.54 6.50
  loop end 7.87 7.82 8.22 8.19 (max 8.72 8.24 8.58 8.54) | 7.81 7.74 8.17 8.13 ; publish 9.15 9.10 8.61 8.61 | 9.79 9.77 8.62 8.62 ; out 11.52 | 12.22
  => MF3's unserialized tile loop is only ~0.05 faster per step (loop end -0.05) and its fp32 exchange costs +0.64 in the publish:
  the wgmma serialization (C7511) is NOT a tile-phase lever (XG's "harmless" holds). Reverted (kept as knob XG_MF=3, default off).
  FACTS from the k56 trace: every step is ~1.5 us warm: step 0 (single tile + the deferred issue) 1.49, step 1 (pair) 1.57, step 2
  (pair) 1.50 (A WGs) / 1.76 (B WGs). Data wait per step 0.06 / 0.13 / 0.13: compute-bound and landing (6.3-6.4) coincide, so faster
  steps 0-1 can give at most ~0.13 (the step-2 data wait). The B WGs (parity 1) finish 0.35 later than A: +0.14 later landing of their
  tiles (go 3.18 vs 3.04) and +0.25 in the LAST step: B owns the partial last tile (tile 9: 606 mod 64 = 30 valid tokens), so only
  B runs the high-side -inf mask (~126 SASS per warp per tile: 504 static instructions removed when the mask is deleted).
## Job s2_j3 plan: XG_PF1 = partial tile FIRST (token shift sft so tile 0 holds the 64 - sft valid tokens; low-side mask on tile 0 = A's step 0)
Expected: B's last step -0.2 (mask gone), A's step 0 +~0.12 -> loop end max(A, B) ~ 8.0 instead of 8.22 (-0.2). Also XG_NOMASK (timing-only
upper bound: delete the mask, wrong output). Decision rule: PF1 kept iff mean of 4 runs <= control mean - 0.08 with every run below both
controls of its pass AND t_s2 (all nt + shift edge cases) / t_xg quick 0 failures; then the same job runs t_xg full + check.py on a
built file (--pf1 1) for the freeze. If PF1 is within +-0.05: stop and report (trace explains). Cap: stop at 20 GPU-min.

## Job s2_j3 RESULT (2026-10-29 13:46-13:50, ~4 GPU-min; dev/s2/out_j3.txt): PF1 = no change; NOMASK upper bound -0.37
Cumulative GPU ~15 min.  Smoke PF1: t_xg quick 6 PASS, t_s2 full (nt 1..10, nc % 64 edge cases 0/55/56/57, lastc) 75 PASS, 0 failures.
nsys medians (copy a / copy b), pass 1 | pass 2:  controls 12.35 / 12.32 | 12.32 / 12.32 ;  PF1 12.35 / 12.34 | 12.32 / 12.35 (mean +0.012 -> dead as a lever)
  NOMASK (timing-only: no mask code at all) 11.94 / 12.00 | 11.94 / 11.97  (-0.37, all four runs)
NOMASK trace: loop end A0 7.51 A1 7.49 B0 7.58 B1 7.56 (k56: 7.87 7.82 8.22 8.19); steps A: 1.41 / 1.37 / 1.37 us (k56 1.49 / 1.57 / 1.50): the
A WGs get ~0.36 faster although they NEVER execute a mask (their tiles are full): the mask blocks are skipped by a forward `@P BRA` over
~2 KB (one block per tile per step instance: 504 static instructions = 8 KB), so the gain is code layout (a taken forward branch into
cold, never-prefetched code = an instruction-fetch miss per skip per step instance, ~0.12 us each) and basic-block splitting of the
softmax chain, not mask arithmetic. B gets the remaining ~0.28 (it also EXECUTES the mask in its last step). PF1 moves the executed mask to
step 0 of A and keeps the same skip-branches in every instance -> same total.
### Job s2_j4 plan (last job; cap): where do the skip-misses sit?  XG_MM = ONE skip-branch per step (pair tiles merged into one block; correct),
XG_NOMASK=2 (timing-only: pair-instance mask removed), XG_NOMASK=3 (single-instance masks removed).  Decision: MM kept as a CANDIDATE iff
mean of its 4 runs <= control mean - 0.10 with every run below both controls of its pass (needs check.py: not affordable inside the cap ->
delivered as an unverified candidate).  NOMASK 2/3 are explanations only.

## Job s2_j4 RESULT (2026-10-29 13:56-13:59, ~2.8 GPU-min; dev/s2/out_j4.txt): MM loses; NOMASK 2/3 explain nothing robust
Smoke MM: t_xg quick 6 PASS, t_s2 full 75 PASS, 0 failures.  nsys medians (copy a / copy b), pass 1 | pass 2:
  controls 12.29 / 12.29 | 12.32 / 12.26 ;  MM 12.54 / 12.51 | 12.51 / 12.54 (+0.24, dead)
  NOMASK=2 (timing-only: mask removed from the pair-step instance) 12.32 / 12.32 | 12.32 / 12.32 but BIMODAL (mean 12.29 vs controls 12.46,
  p10 11.71 vs 12.19: half of the replays ~11.7, half ~12.9; the median sits between the modes) ; NOMASK=3 (single instances) 12.35 / 12.29 | 12.35 / 12.35 (0).
So the -0.37 of full NOMASK needs the masks out of the pair AND single instances; the pair-instance block is the one that matters (it holds 2 of the
4 blocks and the loop body that runs twice).  Nothing more affordable inside the cap.
## Next lever (for whoever continues): mask-free hot step
Full NOMASK = -0.37 us (v1f = 0.28 % of score) timing-only.  A real implementation must keep the partial-tile (and CTA 131's causal) -inf
masking but keep it out of the three step instances: (1) data-side: make the invalid tail rows give -inf scores (no mechanism found: a
zero K row gives score 0 and corrupts the row max / row sum when all real scores are negative), (2) out-of-line block: the mask as a rarely
called code region placed after the hot path (ptxas keeps source block order; __builtin_expect did not move it; a __noinline__ function
with a 16-word local array would, at the price of a call ABI + C7511 risk), (3) one mask application per WG outside the step (e.g. a
dedicated masked step instance used only for the owner's last step: +13 KB of code that only the owner fetches).  PF1 / MM show that
moving or merging the SAME inline blocks does nothing; the code has to leave the hot instances.  Also check the other skip-branches
in the step (alpha rescale `if (!__all_sync(al == 1))`, ASK vote, issue_def x3 instances = 3 x 92 instrs) with the same ablation.

# ROUND 2 (funded by the orchestrator after the round-1 report: build the mask-free hot step; cap +8 GPU-min, lineage 28, used 17.8 before)
## Design XG_MFREE (written before job s2_j5)
- step() gets a MASKED flag through the TW type (XB8M<two> has ::m = true). Unmasked instances (A first single, B pair loop, C last single)
  contain NO mask code. ONE masked instance M (pair, `step(XB8M<true>{}, ms, lt, k, dmyA)`) exists and is called once per WG, only for the
  WG's last step when its last tile `lt` can need a mask: ml = (par < nt) && (nc + (lastc ? 1 : 0)) - 64 lt < 64 (smallest hi over the rows;
  only the WG's LAST tile can ever need it, proof in the build.py comments / checked in Python for nc 0..700 x lastc).
  Layout: the unmasked part runs the original single/pair/single sequence on ntu = ml ? ms : nt (ms = first tile of the last step;
  steps(m') = steps(m) - 1 holds for every m, so the unmasked part is exactly the first n-1 steps); the last step is M:
  pair (lt-2, lt) when the WG's tile count m is odd and >= 3, else a "dummy-A" pair (lt, lt) whose first half is forced to -inf (P = 0), so
  M only ever masks its second tile (126 instrs, unconditional, no skip branch). nt = 10 (the scored shape): parity-1 WGs [s, p, M(7,9)],
  parity-0 WGs [s, p, p] fully unmasked.
- Arm 2 XG_ALB: alpha-rescale of o made branch-free (36 FMUL always; removes the skip-branch + vote) on top of MFREE.
- Bound reference: NOMASK=1 (timing-only).
## Decision rule (job s2_j5): variants MFREE, MFREE+ALB (2 copies x 2 passes, controls k56a/k56b first, shape 1,8,full) + NOMASK bound (2 copies, 1 pass).
Smoke before nsys: t_xg --quick + t_s2 full (nt 1..10, nc % 64 edge cases, last CTA, 3 calls) 0 failures, else skipped.
Keep iff mean of 4 runs <= control mean - 0.10 and every run < both controls of its pass; then the same job runs t_xg full + tools/check.py on a
built --mfree 1 file (freeze: dev/s2/deliver_r1). Kill if within +-0.05 after this job unless the SASS shows mask code still inside a hot instance.

## Job s2_j5 RESULT (2026-10-29 14:05-14:09, ~3.8 GPU-min incl. check; dev/s2/out_j5.txt): MFREE+ALB = WIN, FROZEN as dev/s2/deliver_r1/
Cumulative GPU ~21.6 of the 28-min lineage cap.  Smoke (MFREE, MFREE+ALB, ALB): t_xg quick 6 PASS, t_s2 full 75 PASS, 0 failures.
nsys medians (copy a / copy b), pass 1 | pass 2: controls 12.22 / 12.26 | 12.22 / 12.19
  MFREE 12.10 / 12.13 | 12.06 / 12.19 (mean -0.102, but copy b pass 2 only -0.015 -> fails the every-run rule)
  MFREE+ALB 12.14 / 12.06 | 12.00 / 12.03 (mean -0.165; per run vs the control mean of its pass: -0.10 -0.18 | -0.205 -0.175)  <- KEPT
  ALB alone (pass 1) 12.19 / 12.16 (-0.06) ; NOMASK bound (pass 1) 11.87 / 11.74 (-0.37 .. -0.50)
Freeze checks on the built file (md5 8c4fa2f571ee7a4ecfd7a0c67b5744a5 = deliver_r1/k56_s2r1.py): t_xg full 60 PASS, 0 failures; tools/check.py
CHECK_RC=0, 56 PASS, 0 failures.  124 regs, 0 spills, same pre-existing C7511 as k56.
Static size (approx, SASS bytes from first QK wgmma of an instance to the next): k56 A 7.7 KB / pair 11.2 / last single 6.4 (1583 instrs hot);
r1 A 5.3 / pair 7.6 / last single 5.8 (1178 instrs hot) + masked pair instance M ~6.9 KB (430 instrs, executed once by the owner WG only).
## Job s2_j6 plan (XG_KF, one more step): the k == 0 deferred-issue block (92 instrs + skip branch) still sits in the loop and last-single
instances (k is a runtime value there); XG_KF gives the instances their step-0-ness through the TW type (first single = known 0, loop and last
single = known >= 1, M = runtime).  Arms vs the frozen r1 file (r1a/r1b = deliver_r1/k56_s2r1.py, same job): MFREE+ALB+KF and MFREE+KF.
Decision: deliver_r2 iff an arm's mean over 4 runs <= r1 mean - 0.05 and every run below the r1 runs of its pass minus 0.02 (and thus
>= 0.10 under the controls); the same job runs t_xg full + check.py on the built file. Otherwise r1 stays the hand-back.

## Job s2_j6 RESULT (2026-10-29 14:11-14:15, ~3.5 GPU-min; dev/s2/out_j6.txt): XG_KF = WIN, FROZEN as dev/s2/deliver_r2/
Smoke (MFREE+ALB+KF, MFREE+KF): t_xg quick 6 PASS, t_s2 75 PASS, 0 failures.  nsys (copy a / copy b), pass 1 | pass 2:
  controls 12.19 / 12.26 | 12.19 / 12.27 ; deliver_r1 file 12.13 / 12.16 | 12.19 / 12.27 ; MFREE+ALB+KF 11.90 / 11.90 | 11.97 / 11.98 (mean -0.25 vs r1, -0.29 vs controls)
  MFREE+KF (no ALB) 12.02 / 11.97 | 12.24 / 11.97 (-0.18 vs controls, one noisy run) -> ALB stays on.
Freeze checks on the built file (md5 1c92c15f5a70108689d96214661b12a0): t_xg full 60 PASS 0 failures, t_s2 75 PASS 0 failures, tools/check.py CHECK_RC=0, 56 PASS, 0 failures.
Static hot-instance size (instructions from the first QK wgmma of an instance to the next): k56 first single 482 / pair loop 698 / last single 401 = 1581;
r2 363 / 306 / 230 = 899 (-43 %), plus the masked instance M (~430 instrs, executed once per CTA by the owner WG of the partial tail tile only).

# ROUND 3 (funded: ONE last job s2_j7, cap +6 GPU-min, lineage 34; used 25.1): audit of the remaining skip-branches in the r2 hot instances
## Audit of deliver_r2's SASS (dev/s2/deliver_r2, r2.sass)
First single (A): TRYWAIT loops (BSSY/BSYNC + retry branch, out of line), ASK `VOTE.ANY` + `@!P0 BRA` over ~30 instrs, and the deferred issue block (four
tiny `@!P BRA` over one LDGSTS + 3 pads = 4 instrs: not into cold code). Pair loop (B, 306 instrs): 2 TRYWAIT regions, the same ASK skip (30 instrs), the back-edge,
no other forward branch. Last single (C, 230): same pattern. So the only forward skip of more than a few instructions in a hot instance is the ASK raise path; the
other branch-like structures are the mbarrier waits (asm-internal spin loop => BSSY/BSYNC convergence region + YIELD, 2 per pair step).
Knobs for job 7 (on top of r2 = deliver_r2/k56_s2r2.py; variants chosen with XG_DEFS on s2a/s2b built from the same build.py, r2 cubin identical md5 d6eddeed651b):
  ASK0 = XG_ASK=0 (existing knob: branch-free bias path, always recompute, no vote) | MBW1 = XG_MBW=1 (mbarrier wait as a compiler-visible C++ spin on try_wait,
  no BSSY/BSYNC region) | MBW2 = XG_MBW=2 (same with a suspend-time hint 1 ms) | ALL = ASK0 + MBW1 (screen, 1 copy).
## Decision rule (job s2_j7)
nsys 1,8,full, 2 passes, controls k56a/k56b first, r2 reference x2 (r2a/r2b), each knob x2 copies, ALL x1. A knob STACKS iff each of its 4 runs is >= 0.05 below the mean of the
r2 reference runs of the same pass AND none is above either control run of its pass. Smoke per knob first: t_xg --quick + t_s2 full 0 failures. If anything stacks the same job builds
the file (ALL if ASK0 and MBW1 both stack and the ALL runs are also >= 0.05 below r2; else the best stacking single knob) and runs t_xg FULL + t_s2 FULL + tools/check.py
(56 PASS / 0 failures) on that exact file -> freeze dev/s2/deliver_r3/ (build.py defaults = r2 + riders, built file, md5, nsys table). Otherwise report and stop.

## Job s2_j7 RESULT (2026-10-29 14:23-14:29, ~5.3 GPU-min; dev/s2/out_j7.txt): no rider stacks; deliver_r2 stays
Smoke (ASK0, MBW1, MBW2, ASK0+MBW1): t_xg quick 6 PASS, t_s2 full 75 PASS, 0 failures.  nsys (copy a / copy b), pass 1 | pass 2 (a slower container: controls 12.29-12.35):
  controls 12.35/12.35 | 12.29/12.35 ; r2 (deliver_r2 file) 12.06/12.18 | 12.16/12.03 (mean -0.19 vs controls here; -0.29 in j6)
  ASK0 12.00/12.00 | 12.03/12.11 (-0.072 vs r2; 3 of 4 runs only 0.03-0.08 below the r2 pass mean -> fails the >= 0.05 every-run rule)
  MBW1 12.10/12.29 | 12.29/12.06 (+0.077) ; MBW2 12.06/12.11 | 12.05/12.06 (-0.037) ; ALL (ASK0+MBW1, copy a) 12.13 | 12.06 (-0.012)
Static hot instances (A / pair loop / last single instrs): k56 482/698/401; r2 363/306/230; r2+ASK0 356/307/207; r2+MBW1 366/308/244; r2+ASK0+MBW1 336/308/202 (cubin sizes 3847/3799/3847/3775 instrs).
Audit conclusion: after MFREE+KF the pair loop has no forward skip of more than a few instructions except the ASK raise path (30 instrs); the mbarrier waits
are out-of-line retry loops. Both riders are inside the noise (+-0.08). The remaining headroom is not skip-branches in the hot instances.

# ROUND 4 (funded: ONE knob-sweep job s2_j8, cap +5 GPU-min, lineage 39; used 30.4): combine-phase knobs on the r2 build
Target: the v1f combine after the last publish (~1.8 us vs ~0.8 floor, XG notes lever 1: gate nearer G with throttled stale polls). No new code: existing knobs via
XG_DEFS on s2a/s2b (r2 = XG_MFREE=1,XG_ALB=1,XG_KF=1): XG_GATE in {4, 16, 32} (default 8), XG_PNS in {0, 64, 128} (default 32). (XG's own earlier sweeps, r2g on k46-era code:
GATE 4 +0.04, GATE 16 +0.12, PNS 0 +0.04 vs base = noise or worse; re-measured here because the tile phase changed.)
Job layout: pass 1 = controls k56a/k56b, r2a (reference), the 6 variants x 1 (s2a); pass 2 = controls, r2a/r2b, the two pass-1-best variants x 2 copies.
Smoke: t_xg --quick per variant. Decision rule: a knob STACKS iff its pass-1 run and both pass-2 runs are each >= 0.05 below the r2 reference mean of the same pass and none is above either control
of its pass. If one stacks the job builds the file with the knob baked (build.py --gate/--pns) and runs t_xg FULL + t_s2 FULL + tools/check.py (56 PASS / 0 failures) -> freeze dev/s2/deliver_r3/.
Otherwise report and stop.

## Job s2_j8 RESULT (2026-10-29 15:03-15:06, ~2.9 GPU-min; dev/s2/out_j8.txt): XG_GATE / XG_PNS sweep on r2 = noise, nothing stacks
Smoke: t_xg --quick 6 PASS, 0 failures for all 6 variants.  nsys v1f 1,8,full medians (us), pass 1 (1 copy): controls 12.13/12.13, r2 11.81 (-0.32 vs controls);
  GATE 4 11.81 | GATE 16 11.84 | GATE 32 11.84 | PNS 0 11.84 | PNS 64 11.78 | PNS 128 11.84.  Pass 2 (two best, 2 copies; controls 12.10/12.06, r2 11.84/11.81):
  PNS 64 11.78/11.81 | GATE 4 11.81/11.87.  Best case -0.03..-0.06 vs the r2 pass mean (needs >= 0.05 in all 3 runs): no knob stacks, no freeze. deliver_r2 stands.
