# WP-B1 findings: two-phase balanced one-token (bs1) kernel, BF16 operands / FP32 accumulation

Files: dev/moe_v560.py (= v610 + moe1t, on by default; `MOE1_NO_TWO` = v610's moe1 path), dev/wpb1_cc.py (local nvcc
compile), dev/wpb1_layout.py (repack vs kernel addressing, inverse permutation), dev/wpb1_trace.py (MOE1T_TRACE
timeline), dev/wpb1_run*.sh (Modal jobs).

## 1. Design (v560 moe1t)

- 128 CTAs x 352 threads: warps 0-7 consumers, 8 producer (one 36 KB bulk copy per stage), 9 epilogue + combine,
  10 intermediate fetcher. Every stage = 16 weight rows x the full K = 4096: 32 KB of E2M1 words + 4 KB E4M3 scales.
  Consumer warp w takes k [512 w, 512 w + 512): 32 words per lane (8 LDS.128, each one 16-k scale block of rows g and
  g + 8 -> one BF16 scale-pair register, dequant_o with the HMUL2 half selectors), 32 mma.sync m16n8k16 bf16 -> f32
  into two accumulators; the B fragments (token's 512 x values, later the expert's 512 intermediate values) stay in 64
  registers for the whole phase. No x LDS in the stage loop. Per stage the warp leaves 16 FP32 row partials in a
  4-slot shared buffer (mbarriers pfull/pempty), warp 9 reduces them: consumers never wait on a CTA barrier.
- Phase A (local experts in order): CTA b computes items j = 4 b .. 4 b + 3 of each expert (item = gate rows 8 j .. +7
  and the same up rows, one stage); warp 9 applies SwiGLU and stores the 8 BF16 intermediates as four 8-byte
  relaxed stores (bf16 pair, call epoch): freshness is checked per word by the reader, no fence, no counter.
- Phase B (same order): CTA b owns output rows [32 b, 32 b + 32) of every expert (2 stages); warp 10 polls the
  expert's 16 KB of tagged words (32 LDG.128 per lane in flight, retry only stale ones) into a swizzled 8 KB
  buffer (double-buffered, mbarriers ifull/iempty), the consumers reload their 64 B-fragment registers from it.
- Combine: per expert one red.add of the CTA's sum of squares into ss[call parity][q]; warp 9 lane 0 does one
  red.release.add on a monotonic arrival counter (call c waits for 128 (c + 1); epoch = c + 1, so nothing is reset)
  and every CTA writes its own 32 output rows. CTA 0 zeroes the other parity's sums for the next call.
- Repack (prepare, lossless): wa [32][512][36864], wb [32][256][36864] (906 MB): same codes (zero-scale blocks zeroed
  as in _repack) and the same scale bytes; dev/wpb1_layout.py: 0 mismatches against the kernel's addressing.
- SASS (local nvcc 12.1): moe1t_4 168 registers, no spills, 2576 instructions; stage body 32 HMMA, 128 HMUL2,
  164 LOP3, 65 SHF, 87 IMAD, 9 LDS, 8 PRMT (= 7 ALU + 2 IMAD + 4 HMUL2 per word, no x loads).

## 2. Run 1 (dev/wpb1_run1.sh, Modal H200, jm seeds, 100 reps, two passes): v560 vs v610

| variant | pass 1 | pass 2 |
|---|---|---|
| v610 / v610b (us) | 27.4 / 27.5 (x0.9965) | 27.3 / 27.6 (x1.0000) |
| v560 / v560b | 26.0 / 25.9: **x1.0447 / x1.0479** | 26.1 / 25.9: **x1.0421 / x1.0491** |

Per ne (speedup over SGLang, pass 1): ne=2 3.56 -> 3.56/3.61, ne=3 2.90 -> 3.05, ne=4 2.65 -> 2.81, ne=5 2.40 -> 2.59.
Correct: check_mt M=1 0 failures, rel-L2 2.39-2.73e-3 (v610's errors on the same cases are 2.39-2.73e-3 too). Judge
NVRTC: 168 registers, no spills, 32 HMMA.

Timeline (dev/wpb1_trace.py, us, median / max over CTAs; seed 73235, ne 3):
- x 0.90, stage 0 ready 2.53, **stage 0 computes 2.94 us** (steady 0.83): cold code on the critical path (~2.1 us).
- stages 1-11 (phase A): ready every 1.00-1.04 us = the HBM rate (36 KB x 128 per 1.02 us = 4.6 TB/s), compute
  0.83, gap 0.16-0.19: phase A is data-bound.
- phase B: the B-fragment reload adds ~0.2 us per expert (gap 0.37 vs 0.18); last expert (q = 2): fetch done 20.38
  med / 21.89 max vs needed ~21.1: ~0.3-0.4 us stall. ne = 2 (seeds 70055, 73944): last-expert fetch done 15.9-16.7
  (it starts at 13.55, 2-2.5 poll rounds of ~1.1 us), needed at ~15.0: **stalls 1.3-2.1 us**.
- end: last stage ready 22.56, consumers done 23.46 / 24.13, helper arrival 24.43, all arrivals seen 25.66 (+1.2),
  output 26.11 (+0.45, the ss reload): **3.5 us from the last stage's compute to the output**.

## 3. Structural lessons (for WP-B8's M <= 8 generalization)

- The stage loop is the bs1 floor already: 16 rows x 512 k per warp with resident B fragments costs 0.83 us per 36 KB
  stage (7 ALU + 2 IMAD + 4 HMUL2 + 1 HMMA per word, 32 words per lane, 9 LDS). With T <= 8 tokens as the n8 columns
  the MMA count and the decode are identical; only the partials grow (16 rows x T columns per warp: every lane holds
  4 useful accumulators, 4 STS per lane instead of 2 on tig 0 lanes) and the helper's reduce (8 x 128 floats per item).
- The phase-B operand is the expert's whole intermediate for the CTA's rows (full K): at bs1 8 KB (16 KB tagged) per
  expert per CTA, read by 128 CTAs = 2 MB of L2 reads per expert; at T = 8 it is 64 KB (128 KB tagged) per expert per
  CTA (~8-16 MB per expert, ~25-50% of the weight bytes in extra L2 traffic) and a 64 KB shared buffer (the ring is
  147 KB): budget smem and L2 bandwidth before writing it.
- Expert-ordered phases leave the expert finished last in phase A only (ne - 1) x 2 stages of slack before its
  down rows; the poll latency (own last store -> all 512 items seen) measured 2.3-2.5 us under the stream, so ne = 2
  stalls 1.3-2.1 us. At bs8 (~9 experts) the slack is ~16 stages: no stall expected.
- Every cross-CTA word is epoch-tagged (8-byte (value, epoch) stores, call epoch from a monotonic counter): no fences,
  no resets, no atomics on the data path.

## 4. Run 2 (v561 switches; jm seeds, 100 reps, two passes; ratios vs v610, v610b 0.988/0.991)

| variant | pass 1 | pass 2 |
|---|---|---|
| v560 | 1.0279 | 1.0316 |
| **v561 MOE1T_WARM** | **1.0428** | **1.0434** |
| MOE1T_TAGSS | 1.0066 | 1.0024 |
| WARM + TAGSS | 1.0144 | 1.0129 |
| S6 + TAGSS + WARM (x2) | 1.0026 / 0.9953 | 1.0020 / 1.0000 |

- WARM: stage 0 compute 2.94 -> 1.09 us, but the dry pass at entry takes ~2.2 us itself (x consumed at 2.30, stage 0
  ready 3.33 vs 2.53): the first-stage cost moved, net ~1 us earlier: +1.2-1.5% over v560. check_mt identical.
- TAGSS loses 1.5-3%: arrival -> every CTA's tagged sum seen 1.8 us (counter path: 1.2 + 0.45 reload).
- S6 loses ~1%: same 0.86 us stage compute and the SAME 1.09 us stage interval with 2.8% fewer bytes.
- Per-stage picture in this job: warp 0 computes 0.86 us and waits 0.22 us at every boundary, a constant: the stage
  interval 1.08-1.09 us is consumer-side (compute + boundary), not the 1.02 us HBM interval. Also the consumers finish
  ~0.9 us apart (warp 0's last partial 23.65, warp 9 sees all 8 at 24.58). Suspect: warps 9 (epilogue) and 10
  (fetcher) spin hot on their mbarriers on SMSPs 1 and 2, taking issue slots from consumer warps 1, 5, 2, 6.
  -> v563: HSUS (200 ns suspend hints on warps 9/10) and REL1 (partial slots = ring slots; one arrive per consumer
  warp per stage; warp 9 releases the ring slot after reading the partials; no pempty wait).

## 5. Switches written after run 2 (v562-v564; all bit-identical arithmetic)

| version / switch | what |
|---|---|
| v561 MOE1T_TOUCH | each CTA writes its own lines of the tagged buffers (zero = stale tag) at entry: after the judge's flush a poll of a line nobody wrote yet misses to DRAM behind the weight stream |
| v562 MOE1T_TAIL2 | stage blocks laid out [scales][k half 0][k half 1] (consumer addresses change, data identical); each CTA's last stage copied in two halves on two mbarriers, so its first half is computed while the second lands |
| v563 HSUS (default) | warps 9/10 wait with a 200 ns suspend hint (no hot spin beside consumer warps 1, 2, 5, 6) |
| v563 REL1 (default) | partial slots = ring slots: one arrive per consumer warp per stage, warp 9 releases the ring slot after reading the partials (no pempty barrier, one syncwarp) |
| v564 MOE1T_WARM2 | the dry iteration runs before the consumers read the ids (every role derives the expert list itself; an empty asm on the id copies keeps the compiler from hoisting it), so its code fetch is ahead of the weights |
| v564 MOE1T_BSPLIT | phase-B fragment change in halves inside the neighbouring stages (next expert's xb[0..31] after block 3 of the previous stage, xb[32..63] at the start of the new one) |

## 6. Run 3 (v563 = HSUS + REL1 defaults; jm, 100 reps, two passes; ratios vs v610, v610b 0.993/1.000)

| variant | pass 1 | pass 2 |
|---|---|---|
| v560 | 1.0422 | 1.0487 |
| v561 WARM | 1.0445 | 1.0456 |
| v563 | 1.0468 | 1.0585 |
| **v563 + WARM** | **1.0709 (25.2 us)** | **1.0710** |
| + TOUCH + TAGSS | 1.0192 | 1.0274 |
| + TAIL2 | 1.0213 | 1.0237 |
| + TOUCH + TAGSS + TAIL2 | 1.0009 | 0.9967 |

Per ne (pass 1, speedup over SGLang): ne=2 v610 3.62 -> 3.89, ne=3 2.93 -> 3.15, ne=4 2.68 -> 2.86.
- HSUS + REL1 (+0.5-1% alone) and WARM stack: +2.5% over v561 WARM. check_mt identical errors.
- TAIL2 makes every stage 0.90 us (from 0.83): the mid-stage wait costs the steady state; rejected.
- TOUCH does not shorten the poll rounds (~1.1 us each, 2-3 rounds): the round trip under the stream is not a DRAM
  miss; TAGSS stays slower than the counter (1.9 us arrival -> all sums seen). Both rejected.
- Trace (v563 WARM, seed 73235, ne 3): stage 0 ready 3.18 (dry pass), steady 0.83 + 0.19-0.22 gap (both warps 0 and
  5), phase-B expert changes +0.2-0.3 us, consumers done 22.27 med / 23.52 max, output 24.99. ne=2: last-expert fetch
  done 14.7-15.5 (own items stored 12.2) -> 1.3-2.1 us stall before its first down stage.

## 7. Run 4 (v564) and a methodology problem

| variant (separate weight copies!) | pass 1 | pass 2 |
|---|---|---|
| v563 WARM (run 3 best) | 1.0858 | 1.0849 |
| v564 WARM | 1.0606 | 1.0531 |
| v564 WARM2 | 1.0609 | 1.0681 |
| v564 WARM + BSPLIT | 1.0211 | 1.0199 |
| v564 WARM2 + BSPLIT | 1.0026 | 1.0027 |
| v564 WARM2 + BSPLIT + PF | 0.9081 | 0.9096 |
| v564 WARM + PF | 0.9619 | 0.9432 |

- PF is clearly bad (consumer stages 2.2-4.0 us while the cp.async rounds run; the fetch itself slower): rejected.
- v564 WARM and v563 WARM have the same stage-loop SASS (local nvcc: identical opcode mix) yet differ by 2.5-3%:
  every moe1t variant here had its OWN 906 MB weight copy (bench2 shares tensors only with the first variant, v610,
  which has no wa1t/wb1t). Separate copies were measured ~3.6% apart in the main kernel (placement) -> all
  cross-variant moe1t A/Bs so far carry placement noise of that order. Run 5 measures it (3 own copies of one build)
  and repeats the A/B with the moe1t reference first, so equal-layout variants share wa1t/wb1t.

## 8. Run 5: placement noise measured; shared-weight A/B

- 3 own copies of v563 WARM: x3.258 / 3.214 / 3.187 over SGLang (24.8 / 25.2 / 25.3 us): **2.2% spread from the
  weight placement alone** (v610 x2.953 in the same pass). Shared copies agree within 0.1-0.3%.
- Shared A/B (v563 WARM first; every moe1t variant shares wa1t/wb1t), speedup over SGLang, two passes:
  v563 WARM 3.192 / 3.218 (refb 3.190 / 3.207), v564 WARM 3.164 / 3.174, WARM2 3.130 / 3.159, WARM+BSPLIT 3.052 /
  3.096, WARM2+BSPLIT 3.004 / 3.025, v610 2.923 / 2.967 (v610b 2.933 / 2.953).
  -> v563 WARM = **x1.092 / x1.085 over v610** (25.1 / 24.8 us vs 27.3 / 27.0). v564's refactor costs ~1%, WARM2 and
  BSPLIT lose (BSPLIT: stages 0.96-1.02 us, a mid-stage mbarrier wait splits the stage schedule).
- Judge-NVRTC SASS: v563 WARM 2384 instructions, 168 registers, no spills; stage loop 32 HMMA, 128 HMUL2, 163 LOP3,
  82 IMAD, 66 SHF, 9 LDS, 8 PRMT.
- Trace (v563 WARM, seed 73235, ne 3): x 2.18, stage 0 ready 3.33 (+0.98 compute), steady 0.83 + 0.19 gap,
  expert changes in phase B +0.15-0.3, the last expert's fetch 21.26 (needed ~20.3: 1.3 us stall at stage 16),
  consumers done 23.52, arrival 23.78, seen 24.77, output 25.23. ne 2 (70055): stall 1.34 us before stage 10,
  output 18.50.

## 9. Run 6 (v566 WARMW / SMAJ)

| shared A/B (v563 WARM first) | pass 1 | pass 2 |
|---|---|---|
| v563 WARM / copy | 3.150 / 3.159 | 3.160 / 3.163 |
| v566 WARM (runtime warp mask code, all warps) | 3.119 | 3.128 |
| v566 WARMW 0x0E (warps 1-3) | 3.163 | 3.167 |
| v566 WARMW 0xE0 (warps 5-7) | 3.161 | 3.165 |
| v566 WARMW 0x66 | 3.118 | 3.109 |
| v610 / v610b | 2.904 / 2.908 | 2.910 / 2.907 |

- v563 WARM vs v610 again x1.085-1.086. WARMW (dry pass on warps 1-3 only) moves stage 0 back to 2.56 us ready and
  0.93 us compute (the dry pass no longer delays the producer's SMSP), but the bench gain is only +0.1-0.4%: in the
  same trace the q = 0 intermediate fetch took ~5 us (2 rounds of ~2.5 us) and stalled ne = 2 at B(0) for 1.6 us.
- SMAJ (own copies, 3 each): means 3.150 vs 3.154: neutral; rejected.
- The poll round trip is the variable: 1.1-2.5 us per round. All 128 CTAs poll the same 16 KB (128 lines): ~1000
  requests per line per round concentrated on a few L2 slices. Next: replicate the intermediate (each item stored to
  R copies, CTA b polls copy b % R).

Per local-expert count (run 6, medians us; v563 WARM / v610): ne 2: 19.1-19.8 / 20.5-20.6 (only -4..-7%), ne 3:
25.1-25.3 / 27.4-27.5 (-8.5%), ne 4: 31.6-32.0 / 33.9-34.3 (-7%), ne 5: 37.8-38.3 / 41.7-42.1 (-9%). A stream at
4.55 TB/s + 2.4 us of end would give ~15.5 / 21.8 / 28.0 / 34.2 us: moe1t is ~3.5 us over at every ne (start ~0.8 with
the all-warp dry pass, phase-B fragment changes ~0.2 each, last-expert fetch stalls 1.2-1.6 us at ne 2-3, end ~1.4).

## 10. Baked file for the judge check: dev/moe_v567.py (= v563 + MOE1T_WARM on by default)

- moe1t defaults: MOE1_TWO (escape MOE1_NO_TWO), HSUS (MOE1T_NO_HSUS), REL1 (MOE1T_NO_REL1), WARM (MOE1T_NO_WARM);
  S6 / TAGSS / TOUCH / TAIL2 stay off. v610's CUDA source is a strict subset of v567's (0 lines removed: the main
  kernels are untouched); no "#define MOE_F16" text in CUDA_SRC. Validation job: dev/wpb1_run7.sh.
- Next switches (v568 / v569, all off by default): MOE1T_REPL (intermediate in R copies), MOE1T_FDLY (delay before the
  last expert's first poll), MOE1T_WARMW (dry pass on a warp mask), MOE1T_BSPLIT (v569: wait-free B-fragment halves).

## 11. Run 7: validation of the baked dev/moe_v567.py (READY FOR A JUDGE CHECK)

- Judge NVRTC: moe1t_4 168 registers, no spills, 2384 instructions.
- hang1 bs 1, 8, 16, 32, 128, 256, 257: all finish and pass.
- check_mt M 1 2 4 8 9 16 17 32 33 64 65 128 129 256 257 (normal + skewed, 3 seeds, 2 calls): **0 failures**, max
  rel-L2 3.30e-3 (M = 1: 2.73e-3 normal, 2.58e-3 skewed).
- bench2 jm, 100 reps, all six shapes (v610 x2, v567 x2, ratios vs v610 copy a):

| shape | v610 us | v610b | v567 | v567b |
|---|---|---|---|---|
| bs1 | 26.9 | 1.0012 | **1.0766 (25.0 us)** | **1.0715 (25.0 us)** |
| bs8 | 65.1 | 0.9972 | 0.9998 | 1.0008 |
| bs16 | 85.8 | 1.0026 | 1.0023 | 1.0018 |
| bs32 | 117.6 | 0.9974 | 0.9994 | 1.0000 |
| bs128 | 175.6 | 0.9994 | 0.9982 | 1.0002 |
| bs256 | 214.4 | 0.9952 | 0.9967 | 0.9983 |
| geomean (vs SGLang) | 2.4255 | 2.4199 | **2.4524** | **2.4515** |

  bs1 per ne (speedup over SGLang, copy a): ne 2 3.63 -> 3.86, ne 3 2.92 -> 3.15, ne 4 2.68 -> 2.85, ne 5 2.40 -> 2.61.
  The two v567 copies hold separate 906 MB weight copies (placement spread 0.5% here, up to 2.2% measured in run 5).

## 12. Run 8 (v568 REPL / FDLY / WARMW, v569 BSPLIT; shared A/B, speedup over SGLang)

| variant | pass 1 | pass 2 |
|---|---|---|
| v567 / copy | 3.163 / 3.208 | 3.176 / 3.202 |
| REPL=2 | 3.139 | 3.129 |
| REPL=4 | 3.118 | 3.092 |
| REPL=4 + FDLY=600 | 3.154 | 3.133 |
| REPL=4 + WARMW=0x0E | 3.166 | 3.168 |
| REPL=4 + FDLY + WARMW | 3.171 | 3.150 |
| v569 BSPLIT | 2.973 | 2.975 |
| v569 BSPLIT + REPL4 + WARMW | 3.029 | 3.027 |
| v610 / v610b | 2.954 / 2.930 | 2.940 / 2.929 |

- Control spread this job 0.6-0.9% (ref vs refb share wa1t/wb1t but each has its own small buffers: ib1t, counters).
- REPL loses 1-2.5% and does not shorten the poll rounds (still 1.1-1.5 us): not L2 hot-spotting. Rejected.
- BSPLIT in any form makes every stage slower (0.90-1.06 us): fragment loads must stay out of the stage body.
- On top of REPL4, FDLY=600 (+1.2%) and WARMW (+1.5-2.5%) help -> run 9 tests them without REPL.
- Helper latency is visible: the last phase-A item is computed at ~11.5 (warp 0) but stored at 12.2 (A1st): ~0.5 us
  of warp-9 epilogue on an SMSP shared with two computing consumer warps. v570 MOE1T_HFAST shortens that chain.

## 13. What bounds moe1t now (for whoever continues)

- Inherent: with balanced CTAs and expert-ordered phases the expert finished last in phase A has only (ne - 1) x 2
  stages of slack before its down rows (any reordering, K-split or uneven row split leaves at most that much work that
  does not depend on the last A items). The measured latency from a CTA's last own store to all 512 items fetched is
  2.3-4.4 us (helper epilogue ~0.5 us on an SMSP shared with two computing consumers, store visibility, 2-4 poll rounds
  of 1.1-2.5 us each): ne 2 stalls ~1.0-1.4 us, ne 3 ~0.4-1.3 us at the slowest CTA. Things that did NOT shorten it:
  touching the lines first (TOUCH), replicated copies (REPL), pipelined cp.async polls (PF: kills the consumers).
- End: last data -> output ~2.4 us = last stage 0.8 + warp-9 reduce 0.1-0.6 + arrival/poll 1.0 + ss reload 0.45. Only
  formally unsafe tricks (a 16-byte {count, ss} vector acquire) would remove the reload; tagged per-CTA sums (TAGSS)
  are slower than the counter.
- Stage loop: 0.83 us per 36 KB stage (judge clock), data interval 1.03-1.05 us: phase A is data-bound; anything put
  inside the stage body (mid-stage waits TAIL2, fragment loads BSPLIT) costs 8-25% per stage.
- Weight placement alone moves bs1 by up to 2.2% between copies (run 5): judge runs will scatter by about that much.

## 14. Run 9 (v568 FDLY / WARMW without REPL, v570 HFAST; shared A/B, speedup over SGLang)

| variant | pass 1 | pass 2 |
|---|---|---|
| v567 / copy | 3.143 / 3.175 | 3.155 / 3.160 |
| FDLY=600 | 3.069 | 3.081 |
| FDLY=1000 | 3.108 | 3.118 |
| WARMW=0x0E | 3.154 | 3.139 |
| FDLY=600 + WARMW | 3.137 | 3.150 |
| FDLY=1000 + WARMW | 3.115 | 3.138 |
| v570 HFAST | 3.084 | 3.109 |
| v570 HFAST + FDLY + WARMW | 3.157 | 3.161 |
| v610 / v610b | 2.924 / 2.904 | 2.924 / 2.926 |

Nothing beats v567 beyond the ~1% copy-to-copy band (ref vs refb differ by up to 1.0% here, ne = 2 medians by
0.3-0.6 us). FDLY loses. v567 stays the baked candidate.
Observation for the next attempt: one poll round costs ~1.1 us of round trip PLUS ~0.75 us of processing on the
fetcher warp (judge SASS: ~460 instructions to check 32 words per lane and store the fresh ones, on an SMSP shared
with two computing consumer warps); two fetcher warps (16 words each, SMSPs 2 and 3) would halve the processing.

## 15. Final picture (2026-10-27): v567 judge-checked at bs1 x3.200 (v496/v610 line x3.050)

bench2 (jm seeds, 100 reps): bs1 v610 26.9-27.4 us -> v567 24.8-25.2 us over 5 jobs (x1.072-1.092; validation job
x1.077 / x1.072 for two weight copies), other shapes unchanged; geomean +1.1-1.3%. check_mt M 1..257 clean.

Where bs1's remaining ~5 us over the stream floor sits (ne = 3, 85 MB, v567 trace seed 73235; floor = first request
0.7 us + 85 MB at 4.55 TB/s incl. the ~1.8 us queue ramp, ~19.4 us; observed output ~25.2 us):

| bucket | cost | what would remove it |
|---|---|---|
| start: the all-warp dry pass slows the producer's SMSP, stage 0 ready 3.33 vs 2.5 us | ~0.8 us | dry pass off SMSP0 (WARMW fixes it in the trace, bench neutral because the first fetch then varies more) |
| phase-B expert changes (64-register B-fragment reload + handshake at a stage boundary) | 0.15-0.3 us x ne (~0.6) | double-buffered fragments need 64 more registers (168 used); any reload inside the stage body costs 8-25% per stage (BSPLIT, TAIL2) |
| last expert's dependency: slack (ne-1) x 2 stages = 4.1 us vs ~4.4 us from the last own store to all 512 items fetched | 1.2-1.4 us (ne 2: 1.0-1.4 every call) | shorter poll rounds: ~1.1 us round trip under the stream + ~0.75 us of tag checks per round on a contended SMSP; v571 MOE1T_F2 splits the checks over two warps (untested, <= ~1%). REPL, TOUCH, PF, FDLY did not help |
| last stage after the last byte | 0.8 us | splitting the last stage (TAIL2) costs every stage more than it saves |
| end: warp-9 reduce 0.1-0.3 + arrival -> all seen 1.0 + ss reload 0.45 | ~1.6 us | the reload only with a {count, ss} 16-byte vector acquire (practically atomic, formally not); a tag-based exchange (TAGSS) is slower |

ne = 2 (35% of judge reps) is the weakest case (-6% vs v610, against -7.5% at ne 3-5): the dependency slack is only
2 stages there. Weight placement alone moves bs1 by up to 2.2% between copies (judge runs scatter accordingly).

## 16. Reopened (2026-10-27 morning) on base v801 (moe1t = v764: WP-N named-barrier slot hand-offs)

ncu per-line view (profile/v764_ncu, bs1): the "fetch" line the planner flagged, `*(xp + 16 * m)` with 77% MIO-throttle
stalls (3.0% of all samples), is NOT warp 10's fetch: it is the 8 consumer warps loading the token row's B fragments
from shared memory at it == 0. The old k order (k = 512 w + 128 tig + 16 i + ...) puts the 4 tig lanes' regions
256 B apart: every LDS.128 is a 4-way bank conflict, 16 per lane x 8 warps ~ 2000 shared-memory cycles (~1.1 us) right
before stage 0. That is the "dry pass delays stage 0 to 3.3 us" seen in every WARM trace (x consumed 2.2, stage 0
ready 3.3), not producer starvation. (Warp 10's poll lines show long_sb: round-trip latency, not MIO.)

- v810 MOE1T_XN: k = 512 w + 64 i + 16 tig + 4 st + 2 ps + h inside a warp's slice: each lane's 16-k scale blocks are
  still whole blocks (b = 32 w + 4 i + tig), the 4 tig lanes' blocks adjacent -> conflict-free LDS.128 from the
  natural-order token row; the intermediate buffer becomes natural order too (no swizzle; the fetcher stores at
  dst + 256 m + 8 lane). Same codes / scale bytes per lane (dev/wpb1_layout.py with MOE_DEFINES=MOE1T_XN: 0 mismatches;
  the v801 repack fails the XN check, as it should). Judge NVRTC: 166 registers, no spills, 2568 instructions (v801
  2648); the default build of v810 is byte-identical to v801's moe1t_4 (SASS hash).
- v812 MOE1T_FTMA (with XN): a poll round = one elected-lane 16 KB cp.async.bulk of the tagged words into a staging
  buffer past the ring (mbarrier 30), tag checks from shared memory, fresh words to the fragment buffer, the whole copy
  repeated while any word is stale. 118 registers (the fetcher no longer holds 32 uint4).

### Run 11 (v810 XN vs v801): XN loses 0.9-1.2%, and why

| bench2 bs1 | result |
|---|---|
| own copies x3 (B2_OWN_STATE) | v801 3.180 / 3.203 / 3.221 (mean 3.201) vs XN 3.151 / 3.177 / 3.163 (3.164): -1.2% |
| shared | v801 / v801b 3.184 / 3.193, XN / XNb 3.162 / 3.167: -0.9%; bs8 placebo +0.6% (noise) |

Trace: the conflict fix works (stage 0 ready 2.59 us vs 3.33, x loads no longer 1.1 us), but the q = 0 intermediate
fetch takes 6.1 us instead of 3.0 (A0 stored 7.2, fetched 13.4), stalling ne = 2 at B(0) by 2.3 us; ne = 2 medians
+1.2 us. Judge SASS of the fetcher (dev ldguse check): v801 issues 13 of its 32 poll LDG.128 AFTER the first tag check,
v810 XN 11: the warp stalls on the first check's load (~1.1 us) before issuing the rest, so every poll round is two
round trips, and which loads land late depends on the schedule. That is the "1.1-2.5 us per round" seen in every
trace since run 1. Fixes (v813): the F2 refactor (runtime word offset) already makes ptxas issue all 32 loads first;
MOE1T_FREV (checks in reverse load order) does it explicitly; MOE1T_F2 (two fetcher warps x 16 loads). Measured in
run 14; v812 FTMA (one bulk copy per round, no LDG at all) in run 13.

### Run 13 (v812 FTMA on XN): rejected, -6%

TMA bulk-copy poll rounds are far slower under the stream: q = 0 fetch 7.5 us (own items 7.2 -> fetched 14.7), ne 2
stall 3.4 us at B(0). The SM's TMA unit serves the poll copy behind the ring's 4 x 36 KB weight copies. Shared A/B
(XN first): FTMA 2.991/3.025, 3.005/3.040 vs XN 3.207/3.142, 3.196/3.161; v801 3.217/3.221, 3.205/3.197.

### Run 14 (v813 poll-load ordering): the fetcher's instruction count dominates

| shared A/B, bs1 (v801 first) | pass 1 | pass 2 |
|---|---|---|
| v801 / v801b | 3.172 / 3.211 | 3.179 / 3.201 |
| v813 default (all loads first, ~940 instr per round) | 2.977 | 2.968 |
| FREV | 2.970 | 2.981 |
| F2 (two warps) | 3.157 | 3.149 |
| F2 + FREV | 3.184 | 3.177 |
| F2 + FREV + FDLY 500 | 3.202 | 3.202 |
| XN group: v810 XN / F2+FREV+XN / v801 | 3.195 / 3.224 / 3.228 | |

v813's generic code (a runtime word offset) made ptxas recompute every poll load's 64-bit address (~10 instructions
per load, ~940 per round on an SMSP shared with two computing warps): -7% even though every load now issues before
the first check. So the per-round instruction count matters as much as the load ordering. v815 MOE1T_FLEAN: base
pointers once per expert, immediate offsets: ~464 instructions per round (single warp), ~215 per warp with F2, all
loads before the first check (judge SASS). Run 16.

### Run 16 (v815 lean fetcher): +2.2-2.7% at bs1 -> baked as dev/moe_v814.py

| shared A/B bs1 (v801 first) | pass 1 | pass 2 |
|---|---|---|
| v801 / v801b | 3.210 / 3.208 | 3.211 / 3.217 |
| FLEAN (one warp) | 3.188 | 3.208 |
| FLEAN + F2 | 3.239 | 3.260 |
| **FLEAN + F2 + FDLY 500** | **3.282** | **3.297** |
| XN group: FLEAN+XN / FLEAN+F2+XN x2 / v801 | 3.247 / 3.241, 3.242 / 3.240 | 3.223 / 3.233, 3.257 / 3.243 |

Per ne (us, pass 1): ne 2 18.16 vs v801 18.70, ne 3 24.40 vs 25.04, ne 4 30.66 vs 30.91, ne 5 37.02 vs 37.31.
Trace (FLEAN+F2): q = 0 fetched 1.9 us after own items (v801 3.0, 3 rounds -> 1), last expert 2.0-2.2 us; ne 2 stall
before the last expert 1.09 us (v801 1.15-1.34), ne 3 0.45 (v801 0.48). With XN the q = 0 fetch stays slow (4.3 us),
so XN gives nothing on top; FTMA, XN, v813's generic fetcher all rejected.
dev/moe_v814.py = v815 with MOE1T_FLEAN, MOE1T_F2, MOE1T_FDLY=500 baked (escapes MOE1T_NO_*; host launches 384 threads
unless MOE1T_NO_F2): judge NVRTC moe1t_4 2304 instructions, 128 registers, no spills, SASS hash identical to the
tested v815 build; every other kernel's SASS identical to v801's. Validation job dev/wpb1_run15.sh.

### Run 15: validation of dev/moe_v814.py (READY FOR A JUDGE CHECK)

- Judge NVRTC: moe1t_4 128 registers, no spills, 2304 instructions. hang1 bs 1, 8, 16, 32, 128, 256, 257 ok.
- check_mt M 1 2 4 8 9 16 17 32 33 64 65 128 129 256 257 (normal + skewed, 3 seeds, 2 calls): 0 failures, max rel-L2
  3.30e-3 (M = 1: 2.73e-3 normal, 2.58e-3 skewed).
- bench2 jm 100 reps (ratios vs v801 copy a): bs1 **x1.0288 / x1.0293** (24.5 / 24.4 us vs 25.1; v801b x1.0035);
  bs8 0.997 / 0.997, bs16 0.997 / 0.999, bs32 0.998 / 0.998, bs128 1.000 / 1.000, bs256 0.996 / 1.000 (v801b
  0.995-1.001): placebo shapes within noise; geomean 2.5413 / 2.5440 vs 2.5356 / 2.5335.
  Per ne (us): ne 2 18.35 / 18.51 vs 19.54 / 19.60, ne 3 24.62 / 24.53 vs 25.36 / 25.01, ne 4 30.8 vs 31.2, ne 5 37.7 /
  37.9 vs 38.5.
- Sweep (v814 first, shared): FDLY 300 x0.998, FDLY 800 x0.993, FLATE 2 x0.983 (v801 x0.985).

### Updated bucket picture (v814, ne = 3)

The fetch fix took ~0.6 us off the last-expert stall at ne 2 and ~0.4 at ne 3 (q = 0 fetched in one round). Left:
start ~0.8 us (the 4-way conflicted token-row loads before stage 0; XN removes them but XN's layout makes the early
fetches slow for reasons not yet understood, so it nets zero), phase-B fragment changes ~0.2 us x ne, the last expert's
dependency ~0.5-1.0 us (structural: 2 stages of slack at ne 2 against store visibility + one ~1.1 us poll round trip),
last stage 0.8 us, end ~1.6 us (arrival -> seen 1.0 with its release fence, ss reload 0.45).

### v817 MOE1T_XNX (coordinator follow-up): XN's conflict-free order for phase A only

- wa (gate/up blocks) in the XN k order, the consumers' token-row B-fragment loads conflict-free (1024 w + 32 tig +
  128 (m >> 1) + 16 (m & 1)); wb, the intermediate buffer (swizzled), the fetcher and every phase-B load are v814's.
  The phase-A and phase-B stages run the same stage code: only the B-fragment source and the weights' k order change
  together, per phase.
- dev/wpb1_layout.py (now per-matrix: MOE1T_XN -> both, MOE1T_XNX -> wa only): 0 mismatches for default / XNX / XN.
  Also fixed: v814's repack test `"#define MOE1T_XN" in CUDA_SRC` matched "#define MOE1T_XNX" as a substring (v817
  checks "#define MOE1T_XN\n").
- Judge SASS (local NVRTC 13): XNX moe1t_4 2304 instructions; the stage loop (loop:0) identical to v814; the fetcher's
  poll loop (first poll LDG .. last STS.64, 213 instructions) byte-identical to v814, exact registers; only the
  consumer x-load address math differs; every other kernel identical. Default build of v817 = v814 (same hash).

### Run 17 (v817 XNX vs v814): rejected, -0.8..-2.3%

| shared A/B bs1 | XNX | XNX b | v814 | v814 b |
|---|---|---|---|---|
| pass A (v814 first) | 3.241 | 3.211 | 3.282 | 3.289 |
| pass B (XNX first) | 3.241 | 3.247 | 3.270 | 3.272 |

bs8 / bs16 placebo: within noise. check_mt M 1 2: 0 failures. Per ne: ne 2 19.3-19.5 vs 18.3-18.4 us, ne 3 24.8-25.1 vs
24.4-24.5. Trace: stage 0 ready 2.62 vs 3.36 (the conflict fix works), but the q = 0 fetch takes 4.2 us vs 1.8 with
byte-identical fetcher code. Cause: with the earlier, data-bound start the CTAs' A(0) stores spread 0.7 us (median
7.17, max 7.87; v814 7.74 / 7.90), so the first poll rounds see stale words from the slowest CTAs; ne 2 then stalls
~0.9 us at B(0). v814's late start (consumers behind the data) re-synchronizes the CTAs before the first exchange. So
the XN loss was never the natural-order intermediate buffer. Untested follow-up: XNX + delaying the polls of experts
other than the last until their items are surely visible (v816's FLATE: -1.7% on v814, where nothing is lost early).

### v818 (coordinator follow-up): XNX + a later first poll for the experts other than the last

- MOE1T_F0DLY=<ns>: the first expert's (q = 0, ne > 1) first poll waits that long after the CTA's own A(0) items (meant
  to step over XNX's 0.7 us A(0) store spread); MOE1T_FLATE=<items> (v816's): an expert other than the last is first
  polled once that many items of the next expert are stored here. Both off by default: v818's default build is v814's
  moe1t_4 (same SASS hash), v818 XNX = v817 XNX (same hash). Run 18: F0DLY 300 / 500 / 700, FLATE 1 / 2 on XNX vs v814.

### Run 18 (v818: XNX + later first polls): best +0.3/+0.6%, below the 1% bar, not baked

| bs1, speedup over SGLang | pass A (v814 first) | pass B (XNX first) |
|---|---|---|
| v814 / v814b | 3.262 / 3.246 | 3.260 / 3.266 |
| XNX | 3.219 | 3.236 |
| XNX + F0DLY 300 / 500 / 700 | 3.240 / 3.232 / 3.242 | 3.241 / 3.252 / 3.251 |
| XNX + FLATE 1 | 3.263 | 3.284 |
| XNX + FLATE 2 | 3.233 | 3.269 |

The delays win back XNX's loss but not more: ne 2 stays 0.3-0.5 us slower than v814 (XNX+FLATE1 18.7 us vs 18.2-18.5),
ne 3-4 equal, ne 5 ~1 us faster. Trace: a q = 0 poll round takes ~1.9 us during phase-A streaming (go 9.47 -> done 11.36
with FLATE 2), so between the slowest CTA's last A(0) store (~8.0) and B(0) (~11.1 with the early start) there is room
for barely one round. check_mt M 1 2: 0 failures (F0DLY 500, FLATE 2). v814 stays the base; XNX+FLATE1 (v818,
MOE1T_XNX,MOE1T_FLATE=1) is the best measured alternative (+0.45% mean, not significant).

## 17. End of call on base v894 (moe1t = v814): v920 MOE1T_TAGS2

The counter end costs, after the last CTA's final reduce: red.add ss -> release fence (MEMBAR waits for that red's ack)
-> counter red -> 128 pollers' next acquire -> one more dependent round trip for the ss values (an acquire load holds
back the later loads). MOE1T_TAGSS (v561) removed all of that but measured slower (run 2/3); its judge SASS shows why:
the poll issues 9 of its 32 loads after the first tag check (the same two-round-trip pattern as the old fetcher).
v920 MOE1T_TAGS2 = TAGSS with the checks in reverse load order and immediate offsets from one per-lane base: all 32
poll loads issue before the first check (judge NVRTC: 0 after; TAGSS 9). The data is the signal: no fence, no counter
atomics, no ss reload; CTA 0 advances the call index (the epoch) after it has seen this call's words. Default build of
v920 = v894 (same moe1t_4 hash). Run 20.

### Run 20 (v920 TAGS2 vs v894): neutral, rejected

| shared A/B bs1 | pass 1 | pass 2 |
|---|---|---|
| v894 / v894b | 3.234 / 3.250 | 3.217 / 3.258 |
| TAGS2 / copy | 3.240 / 3.243 | 3.225 / 3.242 |
| old TAGSS (forward checks) | 3.228 | 3.214 |

bs8 placebo within noise; check_mt M 1 2 clean. End timing (seed 73235, ne 3), from the last CTA's arrival:
- counter path (v894): red.release -> all 128 acquire polls see it 0.77 us -> ss reload 0.42 us -> output.
- TAGS2: the last CTA's tagged word -> every CTA has all 384 words 1.54 us -> output (no reload).
So the tagged exchange costs more than the fence + reload it removes: every CTA reading every partial puts ~2048
requests per line per round on the 24 tagged lines, versus 128 per round on the single counter line.

What this says about the end bucket (~1.2 us after the last arrival): one release + one acquire round trip + the
dependent ss load is the minimum for a formally ordered grid reduction here. A replicated done flag or a 2-level tree
adds a hop (the last arriver's atom return, then its flag store) before the pollers' round trip, so it cannot beat 0.77
+ 0.42 us. Backoff only lowers traffic, which is not the limit at 128 requests per line. The cheapest remaining lever
is the release fence itself: it waits for the preceding ss red's acknowledgement, about one round trip; issuing the
first half of the last expert's sum right after stage t = 0 does not help, because the fence still waits for the second
half's red.

## 18. Last stage after the last byte on base v969: v921 MOE1T_TAILX (rejected, -5..-6% bs1)

v921 = v969 + MOE1T_TAILX (default build = v969, moe1t_4 hash d0da71311c80). The stage loop runs every stage except the CTA's
final one. The final stage (the last expert's second down block) runs in a `__noinline__` `moe1t_tailx<NS>(int nt)`,
with every other value derived inside so the caller keeps its registers. The producer copies that stage as part 0 (the
4 KB of scales + each warp's first k half, 8 x 2 KB, on barrier 29) and part 1 (the second halves, 8 x 2 KB, on the slot's
barrier) out of the normal layout, so the weights are shared with v969. The tail reloads the B fragments from ibuf (16 LDS),
computes blocks 0-3, waits for part 1, computes blocks 4-7, then writes the partials and arrives on the named barrier;
the accumulation order is v969's. Warp 11 calls the same function dry after `bar.sync 2`, so the tail's code is fetched
at entry (MOE1T_TXNW: no such call).
SASS (judge NVRTC): the 477-instruction compute core is the same as v969's except one address IMAD immediate; the loop top
gains a YIELD + a LOP3 (from the CALL). A pure `nt - 1` bound without the call only renames address registers.
Kernel code grows from 2304 to 3408 instructions (+18 KB).

### Run 21 (v921 vs v969, shared weights): rejected
| bs1 (jm 100 reps) | pass 1 | pass 2 |
|---|---|---|
| v969b | 1.0013 | 0.9993 |
| TAILX / copy | 0.9440 / 0.9393 | 0.9492 / 0.9362 |
| TAILX + TXNW | 0.9298 | 0.9305 |
bs8 placebo within noise (0.996-0.999). check_mt M 1 2: 0 failures (both builds).

Trace (seed 73235, ne 3, v969 -> TAILX): the split works locally. Only half a stage is computed after the last part lands
(22.43 + 0.35 vs v969's 21.79 + 0.83). But part 0 lands later than v969's whole stage: 21.98, which is 0.31 us after stage
16 is done, versus 0.16 us for the full stage in v969. The 17 small copies deliver more slowly than one 36 KB copy. So the
time from stage 16 done to the last stage done is 1.11 us vs 0.99. The rest of the call is slower too, with or without
the warm call:
- x lands at 2.30 vs 2.06 us.
- The q = 0 fetch finishes at 10.78 vs 9.66 (rounds 4.5 vs 3.0).
- Stage gaps are 0.22 vs 0.19 us.
- The slowest CTA finishes its consumers 0.73 us later (c-done max 23.71 vs 22.98).
- seen -> out takes 0.70 vs 0.41 us.
Output 25.60 vs 24.54 (ne 2: 19.30 vs 18.11). The call version costs about 1 us more than the 0.25 us it saves. The likely
causes are the larger code footprint plus the call boundary and the piecewise copy; one job does not separate them. TXNW is
worse again (-1.5%), so the warm call itself helps.
Closing the bucket: computing the last stage's first half while the second lands needs either a wait inside the stage
body (TAIL2: every stage +8%) or a second code path (TAILX: the whole call slower). Both lose; the 0.8 us stays.
