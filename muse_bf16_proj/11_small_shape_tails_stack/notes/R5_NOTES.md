# Agent R5 — router m = 32 / 128 / 256 (base claude/t3/bo16L.py = bo17L for the router: rt3 ya at m = 32, rt4 at m = 128 / 256)

## RESULT (read first)
DELIVERABLE claude/t3/R5/r5.py (17 KB, readable, one config; prepare_r5 / run_r5): router 97 <= m <= 128 only (None for
everything else: m = 32 stays rt3 ya, m = 256 stays rt4). Candidates: sub_r5.py (claude/t3/bo17L.py + r5 first),
sub_r5_only.py (r5 + torch). Integration: `if layer == 'router' and 97 <= m <= 128: out = run_r5(x, cfg, st)` before
R4's rt4 route (run_r5 returns None elsewhere). prepare ~0.3 s (pack 8.4 MB + one NVRTC kernel + self-test).
Kernel: R4's x-as-A / [hi; lo]-stacked-in-N SS-wgmma, retiled to 32 W rows x 64 tokens x 1024 k (8 K splits): partial
egress 8 KB per CTA instead of 16 KB; stages of 2 k-blocks = ONE W bulk copy + ONE 3D x TMA (x first), 5-slot ring.
| router m = 128, local us (x vs SGLang) | SGLang | base (bo17L rt4) | r5 | delta |
|---|---|---|---|---|
| t5.py same process, x clones (XCL 4, 30 reps) | 15.22 | 4.64 (x3.280) | **4.40 (x3.456)** | -0.24 |
| bench.py same process (bo17L vs sub_r5) | 15.55 | 4.79 (x3.248) | **4.48 (x3.471)** | -0.31 |
| alternating single-candidate bench.py processes, pair 1 | 15.58 / 15.62 | 4.66 (x3.341) | **4.43 (x3.526)** | -0.23 |
| pair 2 | 15.62 / 15.52 | 4.68 (x3.335) | **4.33 (x3.587)** | -0.35 |
| pair 3 | 15.49 / 15.62 | 4.68 (x3.309) | **4.45 (x3.512)** | -0.23 |
| nsys in-graph kernel (same process) | | r4_64_1_16 5.65 | **r5_32_8_2_5 5.33** | -0.32 |
| ncu (full set, flushed, NVTX-filtered single launch) | | 6.66 | **6.27** | -0.39 |
Mean of the alternating pairs: x3.328 -> x3.542 (+6.4 % on this shape, -0.27 us); int_v7 (base_v7.log) x2.328.
Judge-projected (+1 us): x2.86 -> x3.02; ~+0.07 % of the final score (0.26 %/us).
Gates (box 0): check.py sub_r5.py --layers router --graph (NaN-poisoned): "checked 52 cases x 2 seeds + graph: ALL OK";
Y/exact_router.py sub_r5.py --nesc 200: EXACT ROUTER ALL OK (r5 at m 127 / 128: 0 non-tiny one-hot mismatches, dense rel
err 2.0e-6); check.py sub_r5_only.py --ms 128,97: ALL OK. Same run at m = 100 (not scored): tie (4.16 / 4.22 vs 4.20 / 4.28).
Build integration: LN/build_t3.prefix_names(r5.py, '_r5_') -> R5/r5p.py; check.py sub_r5p.py --layers router
--ms 97,100,113,127,128 --graph: ALL OK. Not taken: m = 32 (ya stays; no design here beat it) and m = 256 (rt4 stays).

All times local us (graph replay after a 512 MB read flush, empty-graph cost subtracted; judge ~= local + 1.0). Same-run
A/Bs only, x at 3-4 random 4 KB-aligned addresses per candidate (P5's XCL method, t5.py), box 0 only.

## Files
- t5.py: same-process harness (correctness: normal + outliers via reference.check_implementation on a NaN-poisoned
  allocator + one-hot rows on random columns, exact vs fl32(hi+lo)); candidates sg / rt3 / rt4 (LN lean modules = the
  base) / r4:<cfg> (R4's lab, copied, + R = 32 support) / r5:<cfg> (r5lab) / r6:<cfg> (r6lab); --timefail for ablations.
- t5trace.py: per-CTA clock64 timelines (thread 0 of the consumer warpgroup, globaltimer start, graph replay after the
  flush): prologue issued, first / last stage seen, wgmma drained, staged, zero barrier, reds issued, per-stage arrival.
- t5exact.py: injected-escape exactness gate for r5lab (Y/exact_router.py method). t5one.py + ncu5.sh + ncusrc5.py: ncu
  full set + per-SASS-line stall sampling. gensrc.py / gensrc6.py: local nvcc 12.1 sm_90a compile (regs / spills / SASS).
- r5lab.py: "W as A" split-K wgmma (64 W rows = M, tokens = N; hi then lo into one accumulator); MODE 0 SS raw BF16,
  MODE 1 RS with agent S5's lossless 24-bit hi/lo pairs decoded in registers.
- r6lab.py: SS wgmma, x as A (R4 stacking, any R) or W as A, KS k-blocks per stage (one W bulk copy + one 3D x TMA).
- Copies of R4/r4lab.py + t_r4.py, Y/yr.py (+ t_trace_ya.py) for base traces.

## iteration 0 — base timelines (t_r4.py / t_trace_ya.py traces, box 0)
| m | kernel | prologue issued | first stage | last stage | mma done | zbar ok | reds issued |
|---|---|---|---|---|---|---|---|
| 32 | rt3 ya (LDG all up front, 128 CTAs x 16 rows x 1024 k) | - | w0 2.82-3.07 | (same) | 3.33 | 3.58-3.84 | 3.84 (max 4.10) |
| 128 | rt4 (64 rows x 64 tok x 512 k, 8 x 24 KB, all resident) | 1.12 | 1.34 | 3.65 | 3.90 | 4.10 | 4.51 |
| 256 | rt4 (64 x 64 x 1024 k, 8 x 24 KB ring) | 1.12 (prod done 3.84) | 1.25 | 5.31 | 5.57 | 5.76 | 6.18 |
(us from the first CTA start; trace builds are slower than the timed ones; ya's globaltimer is quantised at 0.26 us.)
Reading: m = 128 streams 10.4 MB of DRAM in ~2.3 us (DRAM-bound) between a ~1.3 us start and a ~1.1 us tail
(last-stage MMA 0.25, staging + zbar 0.2, 16 KB of reds 0.4, drain); m = 256 streams 50 MB L2->SM in 4.1 us (~95 GB/s per
SM with 192 KB in flight: latency x depth bound), same tail; m = 32: the LDG burst lands at ~2.8-3.0, tail ~0.8-1.0
(MMA 0.3-0.5, partial sum / staging 0.25, reds 0.25).

## iteration 1 — tile shapes that cut egress (r4lab, x as A) and a TMA ring at m = 32
Same run (t5.py, 30 reps, XCL 4): m = 32: rt3 3.31 | r4 (64 rows x 32 tok (M 64, half wasted) x 256 k, 4 stages) 3.41;
m = 128: rt4 4.52 | R = 32 rows x 64 tok x 1024 k (egress 8 KB) NS 8 4.53 / NS 10 4.85 / NS 12 4.98; m = 256: rt4 5.42 |
NS 9 5.76 | R = 32 S = 4 7.56. Deeper rings are slower (P5's in-flight sweet spot); halving the egress alone ties.
Trace of R = 32 (m = 128): 16 stages x 2 copies; producer done 3.62, last stage 4.93 (trace build) — the extra L2->SM
(256 KB per SM) and twice the copies cancel the egress saving.

## iteration 2 — r5lab: W as A, SS raw vs RS 24-bit (S5's format), m = 32 / 128
- SS (MODE 0): m = 32 (64 rows x 32 tok x 256 k, 4 stages) 3.47-3.59 vs rt3 3.48-3.68 (tie); m = 128 (64 x 64 x 512)
  4.48-4.79 vs rt4 4.52-4.77 (tie); m = 256 (S 8) 5.53 vs rt4 5.51 (tie).
- RS 24-bit (MODE 1): exact (t5exact.py, 537 escapes incl. 200 injected: 0 one-hot mismatches at m 17/31/32/97/100/128)
  but slower: first build m = 32 5.7 / m = 128 8.6 (escape fix-up), final build 3.83 / 5.82.
- Escape fix-up anatomy (trace + ablations): the per-float4 fix-up loop in the epilogue (S5's per-row cumulative index)
  cost +1.8 us (m = 32) / +3.7 us (m = 128): without the 512 MB flush it still costs +0.57 us (divergent serial LDS
  chains), the rest is cold-code fetch after the flush (warming it from the helper warp did not help). Fix: escapes are
  precomputed by the idle helper warp DURING the stream into two smem tiles per (token, row): X1 = sum of x over the
  row's escapes, C = sum of fma(x, hi16, x*lo16); the epilogue is branch-free v = fmaf(-X1, CE, v) + C (exact for one-hot
  rows since v = x*CE there). Escape cost now +0.2 us (m = 32) / +0.5 (m = 128).
- The RS main loop itself loses (ablations at m = 128, same run): full 5.31 | no decode 4.92 | no wgmma 4.67 | neither 4.35
  | SS raw 4.66. Per-stage trace: RS consumer 0.33 us per stage (570 clk) vs the 0.24 us DRAM pace => consumer-bound;
  8 RS wgmma m64n64k16 per stage run at ~half rate (~67 clk each vs 32) and the decode does not overlap them.
  => 24-bit via RS wgmma at N <= 64 is a dead end for the router (hi + lo doubles the MMA work).

### iteration 1-2 analysis block
1. nsys: not run (no candidate beats the base yet); bench + per-CTA traces above.
2. ncu (ncu5.sh, full set, flushed, m = 32): RS r5 7.07 us, DRAM 6.91 MB, L2 28 %, SM 9 %, issue 16 %, hmma-pipe 0.8 %,
   occ 7.8 %, elig 0.17, stalls wait 26 / long_sb 20 / selected 15 / barrier 13 / imc_miss 8 %; SS r5 5.60 us, DRAM 8.92 MB,
   L2 42 %, SM 6.4 %, issue 10.7 %, occ 8.1 %, elig 0.11, stalls long_sb 37 / barrier 16 / wait 12 / no_instruction 12 %.
3. SASS (nvcc 12.1 sm_90a on the NVRTC source; NVRTC on the box: same regs, LOCAL 0, empty log): SS r5 42 / 58 regs,
   RS 106-127 regs, 0 spills; RS loop per stage: 6 LDS.128 + 72 decode ALU (32 PRMT) + WARPGROUP.ARRIVE + 8 HGMMA.64xNx16
   (RS) + DEPBAR.LE gsb0 1; no LDG; producer 1 UBLKCP + 1 UTMALDG per stage.
4. Binding unit: m = 128 DRAM (stream at ~4.3-5 TB/s between the first and last stage) + fixed start (~1.3-1.7 us to the
   first landing) + tail (~0.85 us); m = 32 DRAM burst + tail; m = 256 per-SM ingress (latency x 192 KB in flight).
   Tensor 1-30 % everywhere. TMA issue: ~90-110 ns per copy (prologue of 16 copies done at 1.42 us, 8 copies 0.85-0.93,
   9 copies 0.91-1.22) — per copy, not per byte.

## iteration 3 — r6lab: multi-k-block stages (fewer TMA copies), x as A with R = 32 / 16, 3D x map
- Same run (XCL 3, 20 reps): m = 32: rt3 3.48 | W-as-A 64x32x256 3.57 | R = 16 x-as-A (M 64 with 32 valid) 4.38-4.40 |
  R = 32 S 16 KS 2 3.59; m = 128: rt4 4.66 | rt4 geometry in r6 4.87 | R = 32 S 8 KS 2 NS 6 4.51; m = 256: rt4 5.58 |
  r6 KS 1 5.80 / KS 2 5.74.
- The r6 implementation of rt4's geometry was 0.2 us slower than rt4: 2D x map instead of 3D (-0.09) and rt4's epilogue
  order (zero-barrier wait first, LDS / red interleaved; -0.13; loading every float4 before the first red is slower)
  close the gap (4.60 vs 4.58). With the rt4 epilogue R = 32 S 8 KS 2 NS 6: 4.29-4.35 vs rt4 4.57-4.58.
- Sweep (two seeds, XCL 4, 30 reps, m = 128 / m = 100): rt4 4.43 / 4.49 | NS 6 4.21 / 4.26 | NS 6 x-first 4.24 / 4.33 |
  **NS 5 x-first 4.11 / 4.23** | NS 7 4.74 | KS 4 NS 3 4.80. m = 100 (not scored): NS 5 x-first 4.16 / 4.22 vs 4.20 / 4.28.
- m = 256 aggregate-vs-per-SM test (FL 8: only every 4th compute CTA streams; per-stage arrival trace): stage cadence
  0.224 us (all CTAs) vs 0.208 us (a quarter) = 107 vs 115 GB/s per SM => per-SM bound (TMA engine / latency x 192 KB),
  not aggregate L2: cluster multicast cannot raise it unless the limit is TMA request generation.
### iteration 3 analysis block (shipped r5 kernel vs rt4, m = 128)
1. nsys in-graph (nsys_shapes.py, same process): r5_32_8_2_5 5.33 us vs r4_64_1_16_8 (bo17L) 5.65 us.
2. ncu (ncu5.sh: full set, flushed, NVTX-filtered, clock-control none): r5 6.27 us, DRAM 10.50 MB (1.67 TB/s over the ncu
   duration = 35 % of 4.8; ~4.3-5 TB/s inside the stream window per the stage trace), L2 49.5 %, SM 17.2 %, issue 18.0 %,
   hmma-pipe 1.6 %, occ 8.6 %, elig 0.19, stalls long_sb 29 / wait 23 / selected 17 / barrier 16 %; rt4: 6.66 us, DRAM
   10.50 MB, L2 49.8 %, SM 15.9 %, issue 14.4 %, occ 8.1 %, elig 0.15, stalls long_sb 26 / barrier 23 / wait 18 /
   selected 14 %. smem wavefronts 291 K vs 251 K (x read by 8 row tiles instead of 4), bank conflicts 9 K vs 17.5 K.
3. SASS (local nvcc 12.1 sm_90a on the NVRTC source; NVRTC on the box: 58 regs, LOCAL 0): 688 SASS lines, 0 spills;
   stage loop: SYNCS.PHASECHK.TRYWAIT + WARPGROUP.ARRIVE + 8 HGMMA.64x64x16.F32.BF16 (2 k-blocks x 4 k16) + DEPBAR.LE gsb0 1
   + 1 SYNCS.ARRIVE (release) + ~10 uniform ALU (slot = i % 5); producer per stage 1 UTMALDG.3D (x, 16 KB) + 1 UBLKCP
   (W, 16 KB); epilogue 4 STS.64 per thread + BAR + LDS.128 / REDG.E.ADD.F32x4 interleaved; no LDG anywhere.
4. Binding unit: DRAM during the stream (10.5 MB, stages land at the DRAM pace) + fixed start (~1.3-1.5 us to the first
   stage) + tail; the win is the tail: 8 KB instead of 16 KB of reds per SM (~30 GB/s per-SM egress) and a 2-k-block last
   stage, paid for with 256 KB instead of 192 KB of L2->SM per SM (~107 GB/s, at the per-SM TMA rate). Tensor 1.6 %.
   Next: the start (1.3-1.5 us to the first landing) is the largest remaining fixed cost.

## iteration 4 — r7lab: m = 32 with 24-bit W via a TMA ring + mma.sync (decode in registers, no RS-wgmma penalty)
- 16 W rows x 32 tokens x 1024 k (128 CTAs, 2 KB egress like ya), 4 stages of 4 k-blocks (W 12 KB 24-bit + x 16 KB 3D
  box), 4 consumer warps (one k-block each per stage, mma.sync m16n8k16, B by ldmatrix.x4 from the swizzled x tile), the
  helper warp's X1 / C escape tiles. Exact (t5exact.py --lab r7: 537 escapes, 0 one-hot mismatches at m 17 / 20 / 31 / 32).
- Same run (XCL 4, 24 reps): rt3 3.58 | r7 (1,4,8,4,4) 3.85 | 8 warps 4.02 | KS 2 3.95 | RT 2 S 16 3.95 / KS 2 3.84.
- Trace (1,4,8,4,4): prologue 0.71, stages land 1.68 / 2.05 / 2.42 / 2.81 (0.37-0.39 us apart = DRAM pace for 6.8 MB),
  last-stage MMA +0.30, staged +0.06, zbar +0.07, reds +0.27. A TMA ring at m = 32 starts landing ~0.4 us later than
  ya's LDG burst (first stage 1.3-1.7 us after the CTA start vs LDG ~0.9-1.0) and ends at the same ~2.8 us: the 24-bit
  byte saving is eaten by the later start. With R = 16 x-as-A (M 64, half wasted) the 4 x 48 KB stages land at
  1.90 / 2.62 / 3.14 / 3.54 (per-SM TMA fill rate). Dead end for m = 32: ya (LDG, 2 KB egress) stays.
### iteration 4 analysis block
1. nsys: not run (no candidate). 2. ncu: not run (no candidate); trace above. 3. SASS (NVRTC on the box): r7 59-77 regs,
   LOCAL 0, empty NVRTC log; per k16 per warp: 1 LDSM.x4 per 2 n-tiles, 2 LDS.128 (W raw + codes), 18 decode ALU, 8 HMMA.
4. Binding: DRAM pace during the stream + the TMA start latency; tensor < 5 %.

## iteration 5 — m = 256 retiling, prologue order, m = 128 stage granularity (r6lab, same runs, XCL 4, 24-30 reps)
- m = 256, 64 rows x 128 tokens (two consumer warpgroups sharing the W tile, 1/3 less ingress, 32 KB egress): NS 6 6.19 |
  NS 5 6.08 | KS 2 NS 3 6.58 vs rt4 5.48 (R4's finding holds in this pipeline too: the 2.4 us of MMA per SM can no longer
  hide under the shorter stream and the 32 KB egress lands in the tail); 32 x 128 x 1024 (same ingress / egress as rt4)
  5.61; rt4 geometry with 2-k-block stages + 3D x + x first 5.50 (tie). => m = 256 stays rt4.
- Holding back the prologue (only stage 0, or stages 0-1, issued before the first landing; the producer issues the rest
  when stage 0 lands): m = 256 6.26 / 5.97 vs 5.65, m = 128 5.01 / 4.70 vs 4.42. The full up-front burst is better.
- m = 128 stage granularity (seed 31): rt4 4.75 | r5.py 4.48 | 1-k-block stages NS 8 4.92 / NS 10 5.28 | KS 2 NS 4 4.83 |
  KS 2 NS 5 W first 4.49. The shipped KS 2 NS 5 x-first point is the optimum of this family (4 x-clone runs: -0.27 / -0.24
  / -0.32 / -0.26 us vs rt4).
### iteration 5 analysis block
1. nsys / 2. ncu: no new candidate (variants lose in bench); shipped kernel numbers in iteration 3.
3. SASS (local nvcc 12.1 + NVRTC attrs on the box): 2-warpgroup variant 58 / 90 regs, 0 spills, LOCAL 0; 8 HGMMA per stage
   per warpgroup at N 128; delayed-prologue variant adds one SYNCS.PHASECHK in the producer.
4. Binding: m = 256 per-SM ingress (~110 GB/s at 192 KB in flight; quarter-CTA ablation 115 GB/s) + 2.4 us of MMA per SM
   (tensor 29 % of the kernel); m = 128 DRAM during the stream + the tail; m = 32 the LDG burst + the MMA / reduction tail.

## Dead ends (measured, this agent)
- 24-bit W via RS wgmma (A from registers) at N <= 64: RS m64n64k16 runs at ~half rate and the decode does not overlap it
  (m = 128 5.31 vs SS 4.66; ablations no-decode 4.92 / no-wgmma 4.67 / neither 4.35).
- Escape fix-up in the epilogue (per-row cumulative index, S5 style): +1.8 / +3.7 us (divergent serial LDS chains +
  cold code after the flush); helper-precomputed X1 / C tiles bring it to +0.2 / +0.5 (kept in r5lab / r7lab).
- m = 32 TMA rings (W-as-A SS 3.47-3.59, x-as-A R 16 4.38, 24-bit mma.sync r7 3.84-4.02) vs ya 3.31-3.68: the TMA ring's
  first landing is ~0.4 us later than an LDG burst and the stream ends at the same ~2.8 us.
- m = 256: 64 x 128 tiles (2 warpgroups) 6.1-6.6; R = 32 S = 4 7.56; deeper rings (NS 9) 5.76; delayed prologue 5.97-6.26.
- Ring depth beyond the sweet spot (m = 128 R 32: NS 7 4.74, NS 10 4.85-5.28, NS 12 4.98) and 1-k-block stages (2x copies).
