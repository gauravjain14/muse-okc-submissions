# W21 (round 5, 2026-10-30): native FP8 wgmma for Q.K^T on batch-1 sliding v1s (xe_b1s_8c6) and d1s (xe_r1_1) -- k70 section 2 XE_SRC
Base dev/coord/k70.py (md5 13d5ca11). Write only dev/w21/. Tags w21_jN. Rules: docs/ROUND5.md. GPU cap 25 min (this worker).
XE_SRC extract for reading: <scratchpad>/w21/xe_src.cu (python split of k70 on '"""', part 21).

## Resume here (updated 2026-10-30 09:50) -- STATE: DONE, deliverable frozen: dev/w21/deliver_r1/
- Deliverable: dev/w21/deliver_r1/build.py <base> <out> (default QG=3) on dev/coord/k70.py (md5 13d5ca11) = v1s (xe_b1s_8c6) S = K Q^T by native FP8 wgmma m64n16k32; output == dev/w21/qg3a.py
  (md5 3ec24ccd), stripped size 872,475 B (< 1,000,000), NVRTC-13 cubin identical to the unstripped one.  d1s (QG=2) and d16s (QG=4) are NOT in it: no gain / +1 us.
- Numbers (pooled over j2-j5, 4 containers): v1s kernel span -0.11 cold / -0.11 warm, judge-us events -0.15 (j3 -0.16, j4 -0.11, j5 -0.18; bench_jt verify 1 sliding -0.14, decode 1 control +0.015) ~ +0.2..0.3 % score,
  judge realizes ~1/3.  Costs: Q is rounded to E4M3 (v1s error 4.5e-3 -> 7.0e-2 on the scored shape, bound 0.150 = the reference's own E4M3-query error; worst err/bound 0.585).
- Gates on qg3a: t_xe full 0 failures, tools/check.py CHECK_RC=0 PASS=56 0 failures, t_rep 60 + 60 (v1s, d1s) 0 failures, edge2 (dev/w10) 0, layout probe OK, ncu A/B done.
- GPU used: j1-j5 ~14 min of 25.  If continuing: (1) stack with other section-2 patches (anchors in build.py qg(): the namespace xer2 strings); (2) hiding the V F2FP under the wgmma flight needs a PTX-level
  dependency (ptxas sinks the F2FP below WARPGROUP.DEPBAR, QGV=1 no effect); (3) d16s needs a non-serialized wgmma schedule at 96 regs (C7512) before it can win.
- Earlier plan items (history): (1) NOCV bound j1 (zero), (2) j2 diagnostic + real QG in timing mode (-0.2), (3) j3 gates, (4) j4 d16s, (5) j5 final.

## Mechanism (read from XE_SRC)
Both kernels hold K in registers straight from LDG: thread (g, q) of a warp has K[b][0] = key 16b+g, K[b][1] = key 16b+g+8, 16 B each = dims 16q..16q+15 (d permuted,
consistently with the Q^T fragments).  Today: e2h (2 x F2FP.F16.E4M3.UNPACK_B per word) + HMMA m16n8k16, per block 16 F2FP + 4 HMMA.
v1s xe_b1s_8c6: 16 clusters of 6 CTAs = one (token, KV head) per cluster, N = 8 rows (the 8 q-heads), 342 keys / CTA, 8 compute warps x 3 blocks of 16 keys (= 2 warpgroups),
48 K F2FP + 48 V F2FP per warp.  (The brief's "64 rows per KV head" is the whole 8-token x 8-head set; this kernel splits it into 8 clusters per head, each token separate.)
d1s xe_r1_1: 2 clusters of 16 CTAs (1 per KV head), N = 8 rows, 128 keys / CTA, 4 compute warps (= exactly one warpgroup) x 2 blocks, 32 K F2FP + 32 V F2FP per warp.
FP8 wgmma RS: A = K regs (m64 = 4 warps x 16 keys; per k32 step a0 = row g, a1 = row g+8, a2 = row g (k+16), a3 = row g+8 (k+16)): the existing K registers ARE the A fragments
(step s: {K[b][0].{x,z}, K[b][1].{x,z}, K[b][0].{y,w}, K[b][1].{y,w}}), D (m64n8: row g cols 2q,2q+1 | row g+8) = the HMMA C layout, so everything after QK is unchanged.
B = Q^T E4M3 [8 rows][k] K-major in smem, no swizzle, core matrix (8 rows x 16 B) = 128 B, desc gdesc(addr, 128, 256, 0), k-step s at +256 s (as dev/wd dec_tile `qd`).
Prep warps write Q8: thread (g, q) word w_j (dims 16q+4j..+3, j = 0..3) -> core matrix j at byte 128 j + 16 g + 4 q.  Q8 = e4m3(rope(rms(q))) = the reference's own FP8 query rounding
(reference.py: q rounded BEFORE qk_scale_factor); c keeps the qk factor.  TASK.md allows FP8 E4M3 queries.

## Step 1: bound (timing-only; decision rule written BEFORE the job)
Variants (dev/w21/build.py): NOCV=1: K words fed to HMMA unconverted (a0=a2=kw0[ks], a1=a3=kw1[ks]; 12 HMMA kept, 0 K F2FP; wrong numbers, NaN-safe for timing),
NOCV=2: the same with only ks = 0, 2 (6 HMMA = the instruction count of 6 QGMMA).  Both shapes patched in one file (xer1 d1s, xer2 v1s).
Harness: dev/roof/rc_ctx.py (graph replay, 512 MB flush, fresh inputs per pair, cold-TLB first rep, order swap), events (80+ reps) then nsys kernel span (40+ reps);
candidates k70a k70b k70c (3 identical copies of k70) vs nc1a nc1b nc1c (+ nc2): comparison = mean of copies, per (cold, warm) rep, kernel span AND judge-us.
RULE: build the real QGMMA QK only if, for v1s, nc1 mean is >= 0.15 us below the k70 mean in the nsys kernel span, in BOTH the cold and the warm rep, and the judge-us mean is
not worse than -0.10 (event-us = same sign); for d1s the threshold is 0.08.  A shape that does not clear is dropped.  (nc1 is an upper bound on the gain: the real build adds the
QGMMA fence/commit/wait latency and the Q8 smem writes.)

## Job w21_j1 (dev/w21/j1.sh, out_j1.txt): the bound.  k70a/b/c (identical copies of k70) vs nc1a/b/c (NOCV=1) + nc2a (NOCV=2); SASS (local NVRTC 13):
d1s xe_r1_1 F2FP 83 -> 51 (-32 K), v1s xe_b1s_8c6 131 -> 83 (-48 K); registers d1s 92 -> 96, v1s 118 -> 117 (nc1); no spills; K/V LDG order unchanged (V first, then K).
Events 200 reps, nsys 120 reps, shapes 1,1,sliding and 1,8,sliding.

## QG design as built (dev/w21/build.py QG=1; local NVRTC-13 SASS: <scratchpad>/w21/lc_qg1)
First attempt (m64n8k32, two k-steps) needed ~40 predicated MOV / CS2R to assemble the A quad {a0, a1, a2, a3} = {K0.x, K1.x, K0.y, K1.y} from two LDG.128 results (A regs must be 4 consecutive
registers) and made ptxas interleave K and V loads.  Final: ONE wgmma m64n16k32 per 8 keys of a warp whose A quad is the LDG.128 destination itself: thread (g, q)'s 16 B of key X are
a0 = bytes 0-3 (row g, k 4q..), a1 = bytes 4-7 (row g + 8), a2 = bytes 8-11 (row g, k 16 + 4q..), a3 = bytes 12-15 (row g + 8): rows g and g + 8 are the SAME key, two dim halves.
B (Q^T E4M3, [16 cols][k32]): cols 0-7 = q-head n with dims {16q + 0..3, 16q + 8..11} (row-g dims), cols 8-15 = the same q-heads with dims {16q + 4..7, 16q + 12..15}.
S[X][n] = D[row g][n] + D[row g + 8][8 + n] = d0 + d6, d1 + d7 of one thread (4 FADD per 16 keys).  Core matrices (8 n x 16 B) at nb * 256 + kb * 128, desc LBO 128 / SBO 256, 512 B per cluster-head.
SASS v1s: 6 QGMMA.64x16x32 back to back + 1 WARPGROUP.DEPBAR, A = K registers directly (no MOV), V-first / K load order as k70, 117 -> 118 regs; d1s 4 QGMMA, 92 -> ~97 regs.
Prep warps: Q8 words from the same floats (e4x4), the owner's FP16 fragment = e2h(Q8) (exact image: every partial of a row uses the same rounded q).

## Result w21_j1 (out_j1.txt, out/20261030T092735_w21_j1; 1.5 min GPU).  nsys kernel span, us, cold-pair rep / warm rep (3 identical k70 copies vs 3 nc1 copies)
| shape | k70a | k70b | k70c | k70 mean | nc1a | nc1b | nc1c | nc1 mean | delta cold / warm | nc2a (1 copy, half the K HMMA) |
| d1s | 3.52 / 3.46 | 3.49 / 3.49 | 3.49 / 3.49 | 3.50 / 3.48 | 3.49 / 3.50 | 3.49 / 3.52 | 3.49 / 3.46 | 3.49 / 3.49 | -0.01 / +0.01 | 3.55 / 3.36 |
| v1s | 4.27 / 4.21 | 4.22 / 4.22 | 4.16 / 4.19 | 4.22 / 4.21 | 4.22 / 4.19 | 4.16 / 4.16 | 4.26 / 4.22 | 4.21 / 4.19 | -0.01 / -0.02 | 4.29 / 4.11 |
judge-us (events, 200 reps) mean of cold+warm: d1s k70 4.32 / 4.34 / 4.32 vs nc1 4.33 / 4.32 / 4.34; v1s k70 5.34 / 5.17 / 5.23 vs nc1 5.20 / 5.23 / 4.95 (one outlier copy; its kernel span is 4.26 / 4.22).
RULE (written before): v1s needs >= 0.15 and d1s >= 0.08 in the nsys kernel span, cold AND warm.  Measured -0.01 / -0.02 (v1s) and ~0.00 (d1s): the bound is ZERO.
Removing 48 (v1s) / 32 (d1s) of the K F2FP per warp does not shorten the kernel => the K-landed -> PV chain is NOT bound by the K conversion pipe.  -> the real QK build is not
justified by the rule.  Follow-ups run anyway because they are cheap (1.5 min): j2 = (a) the coordinator's d16s bound (NOCVM), (b) NOCVV diagnostic (K and V both unconverted: is the conversion pipe
on the critical path at all?), (c) the real QG build in timing mode (compiles clean, 6 / 4 QGMMA, no MOV): if it also shows 0, the lever is dead.  Rule for (a): d16s candidate only if the
3-copy mean kernel span is >= 0.10 below k70 in cold and warm.  Rule for (b)/(c): reported only.

## Result w21_j2 (out_j2.txt, out/20261030T093015_w21_j2; 1.5 min GPU): d16s bound, conversion-pipe diagnostic, and the REAL QG build in timing mode
nsys kernel span cold / warm (3 copies of k70 vs 3 copies of the variant; qg1a = 1 copy):
| shape | k70 mean | nm1 (K raw; HMMA kept) | nv1 (K and V raw) | qg1a (real wgmma QK, 1 copy) |
| d1s  | 3.51 / 3.49 | 3.50 / 3.52 | 3.47 / 3.43 | 3.55 / 3.26 |
| v1s  | 4.24 / 4.18 | 4.21 / 4.19 | 4.13 / 4.13 | 4.06 / 3.97 |
| d16s | 7.96 / 6.47 | 7.83 / 6.43 | 7.89 / 6.57 | (= k70 code, noise) 7.84 / 6.35 |
judge-us (events 160 reps, mean of cold + warm): d1s k70 4.27 | nm1 4.16 | nv1 4.11 | qg1a 4.04;  v1s k70 5.12 | nm1 4.94 | nv1 4.91 | qg1a 4.85;  d16s k70 8.06 | nm1 7.85 | nv1 7.95 | (qg1a = k70 code) 7.86.
READING: (1) the d16s bound (NOCVM) is -0.13 / -0.04 kernel span, -0.2 judge-us: below the 0.10-both rule in the warm rep => d16s does NOT clear.  (2) NOCV (HMMA on unconverted bits) = 0, and even
removing ALL conversions (nv1) gives only -0.07 / -0.05: the E4M3 -> FP16 conversion pipe is NOT the critical resource of the K-landed -> PV chain.  (3) BUT the real QGMMA build is -0.18 / -0.21 (v1s) and
-0.23 warm (d1s) in the kernel span: the gain comes from replacing the DEPENDENT HMMA chain (4 chained m16n8k16 per block + Q fragment LDS) by 6 / 4 independent wgmma and one wait, not from the F2FP
count.  The written rule (bound variant NOCV) therefore under-predicted the real build: DEVIATION, stated here: the rule said stop; the real build was already compile-clean and cheap to time, and it
beats the rule's threshold.  Single copy => needs 3 copies (A/A spread of identical copies 0.1-0.2 us): j3 = correctness + 3 x qg1 vs 3 x k70 + gates + ncu A/B.

## Result w21_j3 (out_j3.txt, out/20261030T093438_w21_j3; ~3.5 min GPU): real QG build (qg1a/b/c = build.py QG=1), correctness + timing + HARDWARE CHECK
Layout probe (dev/w21/probe_qk.py, 1 warpgroup, the exact K-fragment / Q8-core-matrix / wgmma m64n16k32 scheme vs torch fp32 of the E4M3 values): max rel err 4.7e-5 .. 9.2e-5 (8 seeds) -> PROBE OK.
Gates on qg1a: dev/xe/t_xe.py --quick 0 failures (worst err/bound 0.627: d1s ctx 80000 dist normal err 9.41e-2 vs bound 0.150; v1s 7.0e-2 .. 8.0e-2 vs 0.150 = the reference's own
E4M3-query error: Q is now rounded exactly as the reference's FP8 query path), tools/check.py CHECK_RC=0 CHECK_PASS=56 0 failures, dev/w7/t_rep.py 60 + 60 replays (d1s, v1s) 0 failures / no hang,
dev/w10/edge2.py --quick 0 failures (dev/xe/edge2.py does not exist; w10's is the live one; it targets run1f, untouched by this change).
nsys kernel span, cold / warm, 3 copies each: d1s k70 3.63/3.52, 3.58/3.46, 3.62/3.41 (mean 3.61/3.46) vs qg1 3.49/3.46, 3.52/3.42, 3.58/3.33 (3.53/3.40) = -0.08 / -0.06;
v1s k70 4.67/4.13, 4.58/4.19, 4.58/4.19 (4.61/4.17) vs qg1 4.48/4.06, 4.48/4.10, 4.38/3.94 (4.45/4.03) = -0.16 / -0.14.  judge-us (events 240 reps, cold+warm mean): d1s k70 4.37, 4.42, 4.18 vs qg1 4.38, 4.26, 4.37 (+0.02);
v1s k70 5.15, 5.04, 5.10 vs qg1 5.02, 4.94, 4.86 (-0.16).  (j2, 1 copy: v1s -0.18 / -0.21, d1s +0.04 / -0.23.)
ncu A/B (dev/roof/ab_one.py pattern, one profiled cold-L2 launch, k70 -> qg1): inst executed d1s 78760 -> 76520 (-2.8%), v1s 644496 -> 624264 (-3.1%); gmma inst 0 -> 512 / 4608; tensor-pipe active
2.30 -> 1.95 % / 5.32 -> 4.66 % (HMMA chain gone); long_scoreboard stall/issue 2.83 -> 2.00 (d1s), 2.82 -> 2.26 (v1s); barrier 0.25 -> 1.21 (d1s), 0.31 -> 0.74 (v1s) (the wait moved to the wgmma / CTA barrier);
ncu launch time d1s 4.54 -> 4.61 (+0.06, single launch noise), v1s 5.18 -> 5.09 (-0.10).  NOTE sm__inst_executed_pipe_xu does NOT change (2304 / 17504 both): F2FP.E4M3 is not an XU-pipe instruction,
the conversion removal shows only in the total instruction count and the missing HMMA, not in a pipe counter.
QGV (build.py QG=1 QGV=1: issue the V -> FP16 conversion between the wgmma commit and its wait): ptxas sinks the F2FP below WARPGROUP.DEPBAR anyway (SASS checked) -> no effect, not pursued.
d16s QG (QG=4): xe_dm_8_4_2 compiles but ptxas reports C7512 (wgmma serialized, insufficient registers at launch_bounds(160, 4) = 96 regs): tested in j4 (3 copies) only as a timing candidate.

## Result w21_j4 (out_j4.txt, out/20261030T094149_w21_j4; ~4 min GPU; fresh container): 4 x k70, 3 x qg1, 3 x qg5 (= qg1 + d16s QG)
nsys kernel span cold / warm, mean of the copies' medians: d1s k70 3.512/3.432 | qg1 3.509/3.477 | qg5 3.499/3.419 (=> d1s: NO gain: -0.00 / +0.02 / -0.01; judge-us k70 4.08 vs qg 4.11 / 4.12);
v1s k70 4.240/4.160 | qg1 4.075/4.075 | qg5 4.096/4.107 (identical v1s code in qg1 / qg5: pooled 6 copies 4.086/4.091 => -0.155 / -0.07; judge-us k70 4.80 vs qg1 4.71, qg5 4.67 => -0.11);
d16s k70 7.896/6.328 | qg1 (k70 code) 7.755/6.453 | qg5 9.008/7.280 (+1.1 / +0.9: C7512 serialized wgmma at 96 regs, 4 CTAs / SM) => D16S_RULE no, d16s QG DROPPED.
t_xe full on qg1a: 0 failures (worst err/bound 0.681, d1s ctx 2050 dist normal, 0.102 vs 0.150).  [exit 2] of the job = the final tar of absent *.txt files, harmless.
Pooled d1s over j3 + j4 (kernel span): -0.04 cold / -0.02 warm, events +0.02 / +0.03 => noise => d1s stays on k70 code (also keeps its accuracy).  v1s pooled over j2 (1 copy) + j3 + j4:
kernel span -0.17 / -0.14 (j3: -0.16 / -0.14; j4: -0.155 / -0.07 incl. the spread of copies), judge-us (events) -0.16 (j3), -0.11 (j4).
=> DELIVERABLE = QG=3 (v1s only): j5 = its gates + timing vs 3 x k70 + bench_jt + ncu A/B.

## Result w21_j5 (out_j5.txt, out/20261030T094642_w21_j5; ~3.5 min GPU): the deliverable candidate qg3 (QG=3, v1s only) vs 3 x k70, same container
nsys kernel span cold / warm: v1s k70 4.19/4.16, 4.19/4.19, 4.19/4.19 | qg3 4.13/4.13, 4.10/4.10, 4.46/3.97 (copy c has a large cold-TLB rep, the v1s +0.36 effect of W18; its event cold rep is 5.25 too).
Medians over copies -0.06 cold / -0.12 warm; mean of copies +0.04 cold (outlier) / -0.11 warm.  judge-us events (cold+warm mean): k70 5.17, 5.20, 5.04 | qg3 4.91, 5.02, 4.96 => -0.18.
d1s (identical code to k70, control): kernel span within +-0.05, events 4.34 / 4.30 / 4.28 vs 4.26 / 4.91 (copy b: +0.6 outside the kernel, placement) / 4.18.
bench_jt --tlbcold 1 (80 reps, slots k70a qg3a k70b qg3b k70c-sink): verify 1 sliding k70 4.91 / 5.14 vs qg3 4.78 / 5.00 (x4.884 / x4.667 vs x5.019 / x4.796); decode 1 sliding (control) 4.15 / 4.16 vs 4.17 / 4.17.
Strict j5 rule (b) (>= 0.08 below in BOTH reps by the 3-copy mean) is missed in the cold rep by the single outlier copy; pooled over the four independent containers it is met (-0.11 / -0.11 kernel span, -0.15 events,
4 of 4 jobs negative).  Reported as such; the integrator should treat the gain as ~0.1-0.15 us on v1s (1.83 %/us => 0.2-0.3 %) with a judge realization of ~1/3.
Gates (qg3a): t_xe full 0 failures (worst err/bound 0.585: v1s ctx 3000 dist normal, 0.0877 vs 0.150); check.py CHECK_RC=0 PASS=56 0 failures; t_rep 60 + 60 0 failures; edge2 --quick 0 failures.
Precision change (k70 -> qg3, same cases, t_xe --quick): v1s (q_len 8 batch 1) 4.5e-3 -> 7.0e-2 (ctx 80000 soft), 5.1e-3 -> 8.0e-2 (normal), 5.0e-3 -> 7.9e-2 (edges); d1s unchanged (3.7e-3).  The bound there is
0.150 (2.5 x the reference's E4M3-query error 0.07 + 1.5 %, capped): Q is now rounded exactly like the reference's FP8 query (reference.py rounds before qk_scale_factor), which TASK.md allows ("queries may be rounded to FP8 E4M3").
ncu A/B v1s (k70 -> qg3, one cold-L2 launch): gpu time 5.34 -> 5.22 us; inst executed 644496 -> 624264 (-3.1%); gmma inst 0 -> 4608; tensor-pipe active 5.25 -> 4.54 %; ALU 25.5 -> 24.6 %;
long_scoreboard stall/issue 3.22 -> 2.50; barrier 0.31 -> 0.66; regs 92 -> 117.  (XU pipe count unchanged: F2FP.E4M3 is not an XU-pipe instruction; the removal shows as -48 instr/warp in the total.)

## Verdict
The brief's mechanism (no K conversion) is real but is NOT where the time goes: removing the K F2FP alone (NOCV, even all F2FP: NOCVV) moves the kernel span by 0 / -0.05.  The gain of the real build (-0.11 .. -0.15 us on
v1s) comes from replacing the DEPENDENT HMMA chain (4 chained m16n8k16 per block, Q fragment LDS) by 6 independent wgmma and ONE wait.  d1s gains nothing (4 compute warps, no conversion contention), d16s loses (serialized wgmma).
