# WP-B1c findings: bs1 (moe1t) buckets with new mechanisms, base dev/moe_v1002.py

Worker WP-B1c, 2026-10-31. Namespace v1300-v1399. Files: dev/moe_v1300.py (switches, all off by default: the default
build's moe1t_4 is v1002's, judge NVRTC 13.0 sha d0da71311c80, 2304 instructions, 128 registers), dev/wpb1c_sass.py
(moe1t SASS summary with the judge's NVRTC 13 + cuobjdump 13, registers via -res-usage), dev/wpb1c_check.py (several
builds' correctness in one process: watchdog per shape, check_mt's M 1 2 cases, 40 sequential bs1 calls, outputs vs the
first build), dev/wpb1c_run*.sh (Modal jobs). Logs: scratchpad wpb1c/logs/run*.log.

## 0. Scoring note that sets the priorities

bs1's shape score is the MEDIAN of per-rep paired ratios. The jm routings give ne 2 (35%), ne 3 (47%), ne 4 (14%),
ne 5 (4%); ratios sort by ne (ne 5 lowest, ne 2 highest), so the 50th percentile falls inside the ne-3 block (at about
its 68th percentile). Gains at ne 2 alone (e.g. the last-expert dependency stall, ~1.1 us at ne 2 but ~0.45 incl. the
fragment reload at ne 3) barely move the score; gains at ne 3 move it 1:1. So the work targets ne 3's critical path:
the end chain, the last expert's B stages, the fragment reloads.

## 1. Switches in dev/moe_v1300.py

| switch | mechanism |
|---|---|
| MOE1T_BL4 | B fragments (token row at it 0, an expert's intermediate at a phase-B expert change) loaded by lanes 0-3 only. mma m16n8k16's B column n is held by lanes 4n..4n+3, and only column 0 is read back (lanes tig 0, c[0] / c[2]); lanes 4-31's fragments are dead. The token-row LDS.128 (4-way bank conflict: 16 wavefronts per instruction with 32 lanes) become 4 wavefronts, the swizzled intermediate reloads 1 instead of 4 (smem cycles at it 0: ~2048 -> ~512; per expert change ~512 -> ~128). Output bit-identical by construction. |
| MOE1T_FXSS | Fence-free end of call. Per expert one 64-bit word {count:8 \| fixed-point sum of squares:56}; each CTA adds (1 << 56) + round(ssp 2^S_q) with one relaxed red.add.u64 (no fence); lane q of warp 9 polls word q with relaxed loads until count = 128. Formal argument: the word is single-copy atomic and every add is an RMW on it, so the value whose count field is 128 is the sum of all 128 contributions (no cross-location ordering is needed at all). The call index stays ctr / 128 (now a relaxed +1 per CTA; a CTA reads it before its own increment, so at most 127 of this call's increments are visible). S_q is identical in every CTA: U_q = 2 * 4096 * G_e * max\|h\|^2 >= 2 * sum_r v_r^2 (Cauchy-Schwarz per row; G_e = sum_r (1 + w_r)^2 \|W2_e[r]\|^2 from the dequantized weights in prepare, float64; max\|h\| over the fetched intermediate, accumulated with VHMNMX on the fetchers' fresh words: a max is exact, so all CTAs agree), 2^S U < 2^53: no overflow can reach the count field. Judge data (local check, 101 expert outputs, normal + skewed): the fixed-point total is 2^31.4 .. 2^33.9 units, i.e. relative rounding error < 3e-8 (FP32-level). Integer sums are also order-independent (deterministic output). |
| MOE1T_S6 | (existing since v561, planner request) 6-bit block scales, 35 KB stages; tested again because phase A is now data-bound. |
| MOE1T_MAXREG=n | (diagnostic) __maxnreg__ instead of __launch_bounds__; does not pin the stage-loop schedule (not used). |

SASS (judge NVRTC 13.0, moe1t_4): default 2304 instr / 128 regs (= v1002). BL4 2232 / 116 regs; FXSS 2552 / 136 (no
MEMBAR, one REDG.ADD.64, 16 VHMNMX in the fetcher, LDG.E.64 poll); BL4+FXSS 2488 / 116. No spills. The consumer stage
loop is re-scheduled by ptxas in every variant (same span 437 instructions and opcode mix +-3, opcode-order similarity
0.3-0.6 vs v1002); BL4 and BL4+FXSS have the identical stage loop.

## 2. Run 1 (dev/wpb1c_run1.sh, Modal H200, 7.5 min; jm seeds, 100 reps, shared weights except S6's own copy)

Correctness (dev/wpb1c_check.py): v1002, BL4, FXSS, BL4+FXSS, S6: check_mt M 1 2 0 failures, max rel-L2 identical to
v1002 (M1 2.73e-3 normal / 2.58e-3 skewed, M2 3.23e-3 / 3.27e-3), 40 sequential bs1 calls 0 failures (max 2.63e-3);
outputs vs v1002 max |diff| 4.9e-4 (one bf16 ulp; v1002's float red.add order is itself non-deterministic).

| bs1 vs v1002 copy a | all | ne 2 (35) | ne 3 (47) | ne 4 (14) | ne 5 (4) | bs8 placebo |
|---|---|---|---|---|---|---|
| v1002b (control) | 1.0053 | 0.9965 | 1.0065 | 1.0053 | 1.0074 | 0.9981 |
| BL4 | 0.9719 | 0.9238 | 0.9908 | 0.9881 | 0.9842 | 0.9931 |
| **FXSS** | **1.0207** | 1.0127 | **1.0279** (24.70 -> 24.03 us) | 1.0165 | 1.0243 | 0.9932 |
| BL4 + FXSS | 0.9930 | 0.9468 | 1.0093 | 1.0037 | 1.0114 | 0.9981 |
| S6 (own weight copy) | 0.9890 | 0.9775 | 0.9948 | 0.9805 | 0.9826 | 0.9945 |
| S6 + BL4 + FXSS (own copy) | 0.9935 | 0.9558 | 1.0093 | 1.0100 | 1.0026 | 0.9942 |

- FXSS: +2.1% at bs1 (ne 3 +2.8%). Trace (v1300 FXSS, seed 73235): the last arrival -> all seen 0.51 us (v1002: 0.76
  incl. the release fence) and no ss reload; but warp 9's final reduce got 0.27 us slower (h-last 23.02 vs 22.75: the
  scale computation sat on the end chain) -> v1301 computes each expert's scale after its first B stage.
- BL4 loses 2.8% (ne 2 -7.6%) for the same reason as XN/XNX: stage 0 ready 2.62 vs 3.33 us (the conflict-free 4-lane
  loads work), the CTAs then run data-bound from the start and their A(0) stores spread (median 7.17 vs 7.68, max the
  same 7.97), the q = 0 fetch starts before the slow CTAs' items are out and needs 6 rounds instead of 4 (f0ok 11.58
  vs 9.63 us); ne 2 then stalls ~1 us at B(0) and B(1). The token-row load's lateness is load-bearing (it keeps the CTAs
  in step before the first exchange).
- The phase-B expert-change gap is NOT the fragment LDS: 0.40-0.45 us with BL4's 4x fewer wavefronts vs 0.42-0.48 us
  (normal B gaps 0.16-0.19). In phase B the stage intervals alternate 1.22-1.31 / 0.99 us (consumer-bound; phase A
  1.03-1.05 data-bound), so the ~0.25 us per expert change is on the ne-3 critical path (3 x).
- S6: -1.1% alone (own weight copy: placement noise up to 2%), +0.05% on top of BL4+FXSS: does not convert (as WP-S5's
  byte elasticity 0.08 predicts); re-measured on FXSS in run 2.

## 3. Switches added after run 1 (dev/moe_v1301.py .. dev/moe_v1303.py; default builds all = v1002's moe1t_4, sha d0da71311c80)

| version / switch | what |
|---|---|
| v1301 MOE1T_FXSS | each expert's fixed-point scale 2^S_q computed right after its first B stage (off the end chain); trace-only: SM id (slot 31), producer copy-issue stamps per stage |
| v1302 MOE1T_COLS | per-expert mma B columns: expert q's fragments go to column q + 1 (q 7 -> column 0 after phase A), loaded by that column's 4 lanes at the first stage boundary after its fetch completes (one non-blocking mbarrier test per boundary; all lanes must see the phase); the stage's partials come from the expert's column (lanes tig = n / 2, accumulator n & 1). No fragment reload at a phase-B expert change; the token row is still loaded by all lanes (BL4's early start is harmful) |
| v1303 MOE1T_FLATE=n, MOE1T_F0DLY=ns | v818's later first polls (to pair with BL4) |
| v1303 MOE1T_WBL=1/2 | phase-B blocks moved (lossless): 1 = CTA b's blocks at 4 b + t of a 512-block expert (the CTAs' concurrent copies 147 KB apart as in phase A, 74 KB in v1002), 2 = [expert][t][CTA] (contiguous). Phase B's copy latency is ~0.3 us higher than phase A's (run 1 trace: release -> ready 3.55-3.75 vs 3.30-3.33 us) |
| v1303 MOE1T_OWARM | warp 9 writes zeros to its 32 output rows at entry (the final store overwrites them, same thread): tests whether the output page's first touch (seen -> out 0.4 us even without the reload in FXSS) is on the end chain |
| v1303 MOE1T_TOBS | (trace only) observer warp 12 (416 threads) sleeping on each stage's full barrier: stamps when the stage's bytes have landed, separating data waits from consumer boundary costs |

## 4. Run 2 (dev/wpb1c_run2.sh; v1300 / v1303; jm seeds, 100 reps, two passes A / B in opposite variant order)

Correctness: v1302 FXSS, COLS+FXSS, WBL 1 / 2, OWARM, BL4+FXSS+FLATE1, S6+FXSS: check_mt M 1 2 0 failures, errors =
v1002's, 40 sequential bs1 calls 0 failures. hang1 smoke ok (incl. the TOBS trace build, 416 threads).

| bs1 vs v1002 (pass A / pass B) | all | ne 3 |
|---|---|---|
| v1002b (control) | 0.9962 / 1.0099 | 1.0000 / 1.0092 |
| FXSS (v1300) | 1.0127 / 1.0314 | 1.0132 / 1.0295 |
| FXSS (v1301, scale off the chain) | 1.0146 / 1.0214 | 1.0221 / 1.0212 |
| FXSS + OWARM | 1.0165 / 1.0220 | 1.0184 / 1.0238 |
| FXSS + WBL 1 | 1.0069 / 1.0212 | 1.0144 / 1.0215 |
| FXSS + WBL 2 | 1.0037 / 1.0216 | 1.0086 / 1.0196 |
| BL4 + FXSS + FLATE 1 | 1.0119 / 1.0100 | 1.0228 / 1.0168 |
| BL4 + FXSS + FLATE 2 | 1.0213 / 1.0260 | 1.0266 / 1.0338 |
| COLS + FXSS (v1302) | 0.35 (72 us) | |

- Pass-to-pass spread ~1% (v1002b 0.996 vs 1.010): sub-1% differences are not resolvable from two passes. All FXSS
  variants are +1.5..+2.5%; WBL / OWARM / v1301's refinement are within noise of plain FXSS (not banked separately);
  BL4 + FLATE2 is the best mean (+2.4%, ne 3 +3.0%) but within ~0.5% of FXSS.
- COLS (v1302) used mbarrier.try_wait for its non-blocking test; try_wait suspends up to ~5 us when the phase is not
  complete (trace: every phase-A boundary +4.5-5 us). Fixed in v1305 with mbarrier.test_wait.

### The observer trace (TOBS): moe1t is CONSUMER-bound everywhere, not data-bound

Per stage (v1002 code, seeds 73235 / 72563 ne 3, 70055 ne 2; 5 reps; medians over CTAs):
- copy latency (producer issue -> bytes landed) 1.46-1.62 us, NOT the ~3.3 us the ready stamps suggested (ready(it) =
  done(it - 4) + 3.3 was the consumer period times 3, an artifact);
- the data lands 0.6 us before warp 0 is ready at stages 0-2, 1.2-1.6 us before in phase A, 1.8-2.0 us before in
  phase B; even the minimum over CTAs of (ready - landed) is 0.6-1.6 us from stage 3 on;
- consumer period 1.088 us/stage in phase A (compute 0.86-0.90 + boundary 0.2), 1.14-1.23 in phase B (expert changes);
  the HBM floor for 128 x 36 KB at 4.55 TB/s is 1.037 us/stage.
So every consumer-side cost converts 1:1: the ~0.2 us stage boundary (x 18 at ne 3), the phase-B expert changes, the
0.6 us late start; the earlier "phase A is data-bound" reading (wpb1.md) came from ready stamps without landing stamps.
- Start: stage 0 lands between 1.9 and 3.2 us across CTAs (median 2.66; all 4 initial stages issued 0.77-1.28 us and
  interleaved in HBM). v1002's slow token-row load (~1.1 us) ends at ~3.3 for every CTA, after the latest landing, so
  all CTAs start in step; BL4 starts each CTA at its own landing time, and the 1.3 us spread persists (consumer-bound,
  no catch-up) and slows the first exchanges: that is why BL4 / XN / XNX never converted.

## 5. Consumer-side switches (v1305-v1307; default builds = v1002's moe1t_4, sha d0da71311c80)

| version / switch | what | SASS (judge NVRTC 13, moe1t_4, with FXSS) |
|---|---|---|
| v1305 MOE1T_PIPE=b | software-pipelined stage boundary: a stage's 4 partial accumulators (c0[0], c1[0], c0[2], c1[2]) are carried into the next stage; its body writes them (predicated asm st.shared, tig 0) and arrives on the slot's named barrier (bar.arrive without a memory clobber) after block b, so the HMMA drain, the partial store and the barrier overlap the next stage's decode; the CTA's last stage's partials right after the loop. The stage's loads stay schedulable (all 8 LDS.128 hoisted ahead of the epilogue) | b 0: 168 regs; b 1: 132; b 3: 137; no spills |
| v1305 MOE1T_COLS (fixed) | per-expert B columns with mbarrier.test_wait for the non-blocking test | 116 regs |
| v1306 MOE1T_S0DLY=ns | the producer issues stage 0, waits ns, then stages 1-3: the 128 CTAs' stage-0 copies are served ahead of the rest of the 18.9 MB initial burst, so with BL4 every CTA can start at ~2.5 us in step (v1002: all start at ~3.3 after the conflicted token-row load) | 116 regs (BL4) |
| v1307 MOE1T_EPRE | (FXSS) experts 0 .. ne-2 folded into the output while the last expert's word is pending; after it only one rsqrt + shuffle + FFMA | 136 regs, stage loop = FXSS's |
| v1307 PIPE + COLS | PIPE's carried partials from the stage's B column | 134 regs |

## 6. Run 3 (dev/wpb1c_run3.sh, v1307; jm, 100 reps, passes A / B; correctness: all 11 builds check_mt M 1 2 0 failures)

| bs1 vs v1002 (A / B) | all | ne 3 |
|---|---|---|
| v1002b | 1.0006 / 0.9911 | 1.0052 / 0.9910 |
| FXSS | 1.0164 / 1.0080 | 1.0171 / 1.0092 |
| FXSS + PIPE 1 | 1.0068 / 1.0020 | 1.0103 / 1.0053 |
| FXSS + PIPE 3 | 0.9528 / 0.9542 | |
| COLS + FXSS (test_wait) | 0.9489 / 0.9410 | |
| COLS + FXSS + PIPE 1 | 0.9346 / 0.9217 | |
| BL4 + FXSS + S0DLY 400 | 0.9983 / 1.0026 | 1.0078 / 1.0079 |
| **BL4 + FXSS + S0DLY 400 + FLATE 2** | **1.0203 / 1.0160** | **1.0238 / 1.0214** |
| BL4 + FXSS + S0DLY 400 + PIPE 1 | 1.0125 / 1.0041 | |
| FXSS + EPRE | 1.0000 / 0.9917 | |
| all (BL4, S0DLY, COLS, PIPE 1, EPRE) | 0.9334 / 0.9312 | |

bs8 placebo 0.989-1.000. Readings:
- PIPE loses 0.6-1% (PIPE 3 -5%): in the trace the stage body gets ~0.06-0.1 us slower (all 8 LDS hoisted, more
  live registers) while the ~0.19 us gap between the "computed" and "ready" stamps does NOT shrink, so that gap is not
  the partial store / barrier / HMMA drain.
- COLS loses 5-6%: its per-boundary mbarrier test + vote adds ~0.2 us per stage (trace gap 0.35-0.4 vs 0.19).
  mbarrier ops on the consumer path are expensive; COLS is dead in this form.
- EPRE loses ~1.5%, unexplained (its output loop runs before the last word instead of after; maybe the extra polling
  lanes / instructions at the end), not pursued.
- BL4 + S0DLY + FLATE2 is the best so far (+2.0 / +1.6% vs FXSS's +1.6 / +0.8% in the same job).
- The trace builds (per-stage stamps in warps 0 and 5) run ~0.05 us per stage slower than the bench build, so their
  "consumer-bound" picture can be an artifact of the stamps: run 4 uses MOE1T_TRLITE (no consumer stamps).

## 7. Switches for run 4 (dev/moe_v1308.py, dev/moe_v1309.py; default builds = v1002's moe1t_4)

Working hypothesis after run 3: the consumer-path costs that matter are mbarrier (SYNCS-unit) operations and global
stores / reds whose address translation misses (the weight stream evicts TLB entries); the per-stage gap (one try_wait),
the phase-B expert change (+ one try_wait, + one arrive) and COLS's extra test_wait (+0.2 us per stage) all fit the
first; the trace's unexplained 0.4-0.6 us intervals that each follow a global store (warp 9's partial sum after a
stamp, seen -> out after the output store) fit the second. dev/wpb1c_ubench.py measures the sync-op latencies
(try_wait / test_wait on a completed mbarrier, arrive, ld.volatile / ld.acquire.cta flag, bar.sync, LDS), idle and
with 4 x 36 KB bulk copies in flight per CTA.

| switch | what |
|---|---|
| v1308 MOE1T_PIPE2=b | (with PIPE) the next stage's full-barrier wait and its first 2 LDS.128 + scale word issued in the current stage's tail (after block b), carried into the next stage (one body copy); 145 regs |
| v1308 MOE1T_NS5 | the 5-slot ring kernel moe1t_5 (208 KB smem): does the data front move? |
| v1308 MOE1T_TRLITE | trace builds without the per-stage consumer stamps (the consumers run bench code; producer issue + observer landing + end stamps kept) |
| v1309 MOE1T_MPOLL=ns | (FXSS) the last expert's word polled with two relaxed loads in flight, staggered by ns |
| v1309 MOE1T_SIGW | signaler warp 12 (416 threads) sleeps on each stage's full mbarrier, then st.release.cta of the stage index into a per-slot shared word; consumers wait with ld.acquire.cta (compiles to a plain LDS poll loop: no SYNCS op, no fence on the consumer path) |
| v1309 MOE1T_SIGB | phase-B intermediate hand-offs (fetchers -> consumers 'full', consumers -> fetchers 'loaded') through release / acquire shared words instead of mbarrier arrive / wait (2 fewer SYNCS ops per consumer warp per expert change) |
| v1309 MOE1T_CTRE | (FXSS) the call-index +1 issued right after the CTA reads the counter at entry (floor(ctr / 128) still gives the call: the reader has not added its own 1), so no global red precedes the end polls |
| v1309 MOE1T_OLATE | zeros stored to the CTA's output rows at the last expert's last stage (same thread as the final store), so its translation is warm |

### Pooled estimates after runs 1-3 (bs1, vs same-job v1002 copy a; "judge-style" = median(SGLang / variant) / median(SGLang / v1002))

| build | passes | paired median ratio (mean) | ne 3 | judge-style |
|---|---|---|---|---|
| v1002b (control) | 5 | 1.0006 (0.991..1.010) | 1.0024 | 0.9999 |
| FXSS (v1300 / v1301 / v1307) | 7 | **1.0179** (1.008..1.031) | 1.0200 | **1.0184** |
| BL4 + FXSS + FLATE2 | 2 | 1.0236 | 1.0302 | 1.0312 |
| BL4 + FXSS + S0DLY 400 + FLATE2 | 2 | 1.0181 | 1.0226 | 1.0196 |
| FXSS + OWARM / WBL 1 / WBL 2 | 2 each | 1.0192 / 1.0141 / 1.0127 | | |

BL4 + FLATE2 (+/- S0DLY) vs FXSS in the same pass: +0.75 / -0.04 / +0.39 / +0.79% (mean +0.47%).

### Status 2026-10-31 22:25 PT
Banked: MOE1T_FXSS (dev/moe_v1300.py or later, define MOE1T_FXSS): bs1 +1.8% judge-style over 7 passes, check_mt clean,
moe1t-only change. Candidate on top (+~0.5%, 4 passes): MOE1T_BL4 + MOE1T_FLATE=2 (+ MOE1T_S0DLY=400 neutral).
Run 4 (queued: sync-op microbenchmark, TRLITE traces, SIGW / SIGB / CTRE / OLATE / MPOLL / NS5 / PIPE2 / S6) decides
the rest; dev/wpb1c_bake.py bakes a define set into a version file (baked default = tested SASS, all-escape build =
v1002's moe1t SASS, checked on a dry run).

## 8. Run 4 (dev/wpb1c_run4.sh, v1309; 2026-10-31 22:23 PT)

Sync-op microbenchmark (dev/wpb1c_ubench.py, H200, cycles per op in a dependent chain, median over 512 warps; with 4 x
36 KB bulk copies in flight per CTA the numbers are the same within 1 cycle): mbarrier try_wait / test_wait on a completed
phase 65, mbarrier.arrive 8 (issue), ld.acquire.cta.shared 43, plain LDS 42, bar.sync (4 warps) 21. So an mbarrier wait
is ~0.035 us: the SYNCS hypothesis of section 7 is wrong, and the ~0.19 us stage "gap" is not a latency of the boundary.

TRLITE traces (consumers run bench code; producer issue + observer landing + end stamps): the bench-like build is ALSO
consumer-paced: each stage's bytes land ~1.5 us before use (landing -> slot re-issue 2.2-2.9 us vs ~1.1 if the
consumers waited for data), landings ~1.0-1.05 us apart, copy latency 1.3-1.9 us. With a try_wait at 65 cycles the pace
(~1.04 us/stage) is SMSP dispatch throughput of the stage body (2 consumer warps x ~1408 decode dispatch cycles per
stage at the ~1.75-1.8 GHz MMA clock) plus the boundary's instructions: any switch that adds consumer-side instructions
or an extra warp on an SMSP loses (PIPE, PIPE2, COLS, SIGW/SIGB below); fewer instructions per stage is the lever
(WP-BCPS's area).
- End chain, FXSS (seed 73235, median / slowest CTA): last consumer warp 22.43 / 22.98 us, warp 9's red 22.62 / 23.17,
  all words seen 23.58-23.68, output 24.03. The slowest CTAs are ~0.55 us behind the median at the end and fall behind
  gradually through the call (issue lag +0.06 at stage 4, +0.17 at 8, +0.30 at 16; landing lag equal, so consumer-
  limited, ~2% slower); they recur on the same SMs, mostly TPC pairs (14/15, 30/31, 46/47, 62/63, 104/105, 118, 122,
  127): a placement / hardware property, not addressed.
- Trace caveat: every trace stamp is a global store; a stamp followed by shared-memory loads in the same warp shows
  0.4-0.7 us gaps (warp 9's partial sum after its stamp, the output loop after the 'seen' stamp) that the bench build does
  not have; the same mechanism (a global store / red whose translation or line is cold, ahead of loads in the warp's
  memory queue) is what CTRE and OLATE remove from the real end chain.
- BL4 + S0DLY + FLATE2: stage 0 starts at its landing (hold 1.0 us) and the median CTA is 0.2-0.6 us ahead, but the
  slowest CTA ends later (last consumer warp max 23.10 vs 22.98, spread p50 -> max 0.9 vs 0.55 us): its bench gain
  is small and noisy.

bench2 bs1 (jm, 100 reps; A / B; vs v1002 copy a; same-pass FXSS in brackets):

| build | A | B | ne 3 (A / B) |
|---|---|---|---|
| v1002b | 0.9994 | 0.9869 | 1.0026 / 0.9871 |
| FXSS | 1.0204 | 1.0032 | 1.0213 / 1.0039 |
| **FXSS + CTRE + OLATE** | **1.0272** | **1.0177** | **1.0319 / 1.0133** |
| FXSS + CTRE | 1.0204 | 1.0139 | 1.0253 / 1.0133 |
| FXSS + CTRE + OLATE + MPOLL 200 | 1.0013 | 0.9965 | |
| FXSS + MPOLL 200 | 1.0087 | 0.9947 | |
| BL4 + FXSS + S0DLY + FLATE2 | 1.0140 | 1.0089 | 1.0237 / 1.0173 |
| ... + MPOLL / + SIGW + SIGB | 1.0065 / 0.9884 | 0.9889 / 0.9732 | |
| FXSS + SIGW / SIGB / both | 0.9955 / 1.0079 / 0.9787 | 0.9837 / 0.9909 / 0.9665 | |
| FXSS + NS5 | 0.9915 | 0.9799 | (ne 2 0.937 / 0.925) |
| S6 + FXSS (own weight copy) | 0.9910 | 0.9830 | |
| FXSS + PIPE 1 + PIPE2 5 | 0.9512 | 0.9491 | |
| all of BL4, S0DLY, FLATE2, SIGW, SIGB, CTRE, OLATE, MPOLL | 0.9684 | 0.9604 | |

bs8 placebo 0.992-1.000. Correctness: all 16 builds check_mt M 1 2 0 failures (errors = v1002's), 40 sequential bs1 calls
0 failures; hang1 smoke ok.
- CTRE + OLATE is +0.7 / +1.4% over FXSS in the same pass (CTRE alone +0.0 / +1.1%): moving the call-index red off the
  end chain and warming the output row's translation one stage early both pay.
- MPOLL loses ~1% (two polls per CTA in flight on the one hot line slow the line's atomic updates); SIGW / SIGB lose
  1-3.5% (an extra warp on SMSP 0 / fences and flag polls in place of 65-cycle mbarrier ops); NS5 loses (ne 2 -6..-7%:
  208 KB of shared memory per CTA; also more bytes in flight against the same consumer pace); S6 again -1..-2%.

## 9. Candidate dev/moe_v1310.py and run 5 (queued 22:40 PT)

dev/moe_v1310.py = dev/wpb1c_bake.py(dev/moe_v1309.py, MOE1T_FXSS, MOE1T_CTRE, MOE1T_OLATE): baked on by default with
escapes MOE1T_NO_FXSS / MOE1T_NO_CTRE / MOE1T_NO_OLATE. Judge NVRTC 13.0, whole module (INST lines as _module()): every
main kernel (moe_8_8_1_3_16 .. moe_2_1_1_3_1) and moe1_3/4/5 byte-identical to v1002; moe1t_4 = b3ca7bbf93c3 (2552
instructions, 136 registers, no spills) = the tested FXSS+CTRE+OLATE build; with the three escapes moe1t_4 = v1002's
d0da71311c80. Run 5: hang1 all shapes, check_mt M 1..257, bench2 all six shapes (v1002 x2, v1310 x2), two bs1 decision
passes (+ WARMF, + BL4 + FLATE2, + BL4 + S0DLY + T0 2600 / 2900) and two TRLITE traces.
New switches in v1309 (off by default): MOE1T_WARMF (FXSS: lane 0 loads the last expert's word at its last stage, result
unused); MOE1T_T0=ns (with BL4: every consumer warp starts stage 0 no earlier than ns after its own entry, a common start
across CTAs, ~0.3 us before v1002's implicit one, with S0DLY bringing the latest stage-0 landing to ~3.0 us).

## 10. Run 5 (dev/wpb1c_run5.sh, 2026-10-31 23:01 PT): v1310 validated

- hang1 bs 1, 8, 16, 32, 128, 256, 257: all finish and pass. check_mt v1310 M 1 2 4 8 9 16 17 32 33 64 65 128 129 256 257
  (normal + skewed, 3 seeds, 2 calls): **0 failures, max rel-L2 3.30e-3** (= v1002's). dev/wpb1c_check.py: v1310, its
  escape build (MOE1T_NO_FXSS,MOE1T_NO_CTRE,MOE1T_NO_OLATE), WARMF, BL4+FLATE2+CTRE+OLATE, T0 2900: check_mt M 1 2 0
  failures, 40 sequential bs1 calls 0 failures.
- bench2 all six shapes (jm, 100 reps; v1002 x2 vs v1310 x2):

| shape | v1002 | v1002b | v1310 | v1310b |
|---|---|---|---|---|
| bs1 | 24.6 us x3.258 | 1.0030 | **1.0189 (x3.330)** | **1.0268 (x3.347)** |
| bs16 | 85.7 | 1.0009 | 0.9988 | 1.0046 |
| bs32 | 112.6 | 0.9991 | 0.9980 | 0.9995 |
| bs8 | 63.7 | 0.9995 | 0.9990 | 0.9961 |
| bs128 | 166.7 | 1.0003 | 0.9982 | 1.0005 |
| bs256 | 200.3 | 0.9979 | 0.9966 | 1.0010 |
| geomean | 2.5659 | 2.5646 | 2.5722 | 2.5795 |

  (main kernels byte-identical: the other shapes are placement noise.)
- bs1 decision passes (vs v1002, A / B): v1002b 1.0026 / 1.0009; FXSS 1.0158 / 1.0157; **v1310 1.0259 / 1.0233** (ne 3
  1.0283 / 1.0277); + WARMF 1.0233 / 1.0231; BL4 + FLATE2 + CTRE + OLATE 1.0176 / 1.0206 (ne 3 better, ne 2 worse);
  v1310 escape build 0.9973 / 0.9996; BL4 + S0DLY + CTRE + OLATE + T0 2600 / 2900: 1.0087 / 1.0071, 0.9965 / 0.9940
  (T0's sleep overshoots: the TRLITE trace has the token row consumed at 3.58 us).

## 11. Conclusion

**Bank dev/moe_v1310.py** (= dev/moe_v1309.py with MOE1T_FXSS,MOE1T_CTRE,MOE1T_OLATE; baked, escapes MOE1T_NO_FXSS /
MOE1T_NO_CTRE / MOE1T_NO_OLATE): bs1 +2.3% judge-style over v1002 (6 passes in runs 4-5: 1.0272, 1.0177, 1.0259,
1.0233, 1.0189, 1.0268; FXSS alone +1.8% over 9 passes), every other kernel byte-identical to v1002 -> ~+0.4% geomean.
What the measurements say about bs1 now (ne 3, ~24.0 us): phase A streams at ~4.5 TB/s with consumers and HBM at the same
pace (~1.04 us per 36 KB stage); the consumers are dispatch-bound on the stage body (any added consumer instruction or
warp loses), the data lands ~1.5 us ahead; the remaining tail after the last byte (~19.8 us) is the consumers' lag +
the last stage + warp 9 + the FXSS exchange (~0.4 us) + the slowest CTAs (~0.55 us behind the median, recurring on the
same SMs). Further bs1 time needs fewer instructions per stage (WP-BCPS) rather than boundary / end mechanisms.
Closed by measurement here: BL4 (alone), COLS, PIPE, PIPE2, SIGW, SIGB, MPOLL, EPRE, NS5, S6 (again), WBL 1/2, OWARM,
WARMF, T0, S0DLY (neutral); BL4 + FLATE2 is +0..0.5% and widens the CTA spread (not banked).

Notes for whoever continues:
- Trace stamps are global stores and delay that warp's following shared-memory loads by 0.4-0.7 us (seen in every end
  chain: the consumers' own stamps land after warp 9 has already passed their barrier, warp 9's 'sum' stamp 0.7 us after
  its 'bar' stamp). Stamp values taken before the first store of a sequence are reliable; for a clean end-chain
  timeline, write stamps to shared memory and flush them after the output store (not built).
- Tools: dev/wpb1c_trace.py (raw dumps: per-CTA stamps, producer issue, observer landing, end stamps), dev/wpb1c_tlite.py
  (issue / landing / hold per stage from TRLITE dumps), dev/wpb1c_ubench.py, dev/wpb1c_check.py, dev/wpb1c_sum.py,
  dev/wpb1c_sass.py (WPB1C_ALL=1: every kernel), dev/wpb1c_bake.py. Dumps / bench JSON: dev/remote_out/wpb1c_*.
- Housekeeping: early in the session my first SASS dump wrote v1002.cubin / .sass / .log into the shared scratchpad
  directory 'sass/' (another worker had v1002.* files there; if they had a v1002.cubin it was overwritten with a no-define
  build). All later files of this worker are in the scratchpad's wpb1c/ subdirectory; nothing was deleted.
