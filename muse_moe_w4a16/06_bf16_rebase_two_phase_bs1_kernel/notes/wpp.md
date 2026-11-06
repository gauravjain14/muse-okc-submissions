# WP-P findings: CUDA-core (tensor-free) stages for low-token experts at bs8 / bs16

Worker P, started 2026-10-27. Base: dev/moe_v200.py (== submission.py). Files: dev/moe_v50x.py, dev/wpp_*.

## Tools
- dev/wpp_sass.py: compile a version (NVRTC on Modal, local nvcc 12.1 otherwise) and count SASS opcodes per kernel
  (HMMA / HGMMA must be 0 in the CUDA-core kernel), registers, spills.
- dev/wpp_clk.py: in-kernel SM clock per CTA (MOE_CC_CLK: clock64 vs %globaltimer over the consumers' item loop),
  judge protocol, judge-matched seeds; split into CTAs that ran CUDA-core-only vs tensor items.
- dev/wpp_timn.py: per (phase, token count) item cycle in the real kernel (MOE_CC_TIMN), first items apart.
- dev/wpp_err.py: rel-L2 vs the FP32 reference of several versions on the same inputs.

## v500 design (MOE_CC, default on; MOE_NO_CC = v200's tensor path)
Scope: the bs <= 8 config (4, 1, 1, 4, 1) only; every other config unchanged.

- Weights: the existing _POSH words (P.w13h / P.w2h, built for MOE_F16_NT4: same mma-fragment word positions as
  w13 / w2, bit layout for an FP16 decode: E2M1 * 2^-14 per half, dec16 = 7 ALU + 3 IMAD per word). Same scales
  (S6 6-bit or 8-bit). So the ring slot layout is unchanged: WP-S's WBLK block re-layout (byte-for-byte the slot)
  merges by building its blocks from w13h / w2h instead of w13p / w2p.
- Per lane (g, tig) and 64-k chunk: its 2 row groups x rows (g, g+8) x its 16-k block (k = 16 tig + 0..15) =
  4 words; dec16 gives (row g, k 4jj+0..1), (g+8, same), (g, 4jj+2..3), (g+8, same). Per token: 8 HFMA2 per row
  block into an f16x2 partial (FMA chain, rounding at the partial's magnitude), then 1 HFMA2 with the block scale
  pair (s * 2^-4) into an f16x2 item accumulator per (row, token). x is read as the lane's 32-byte block per
  token (2 LDS.128, broadcast across the 8 lanes of a tig).
- Item end: (lo + hi) * 2^6 in FP32 (the tensor path's units), 2 butterfly steps over the 4 tig lanes (lane keeps
  row group tig & 1, half tig >> 1), staged as [kw][token][row] floats: the existing epilogue runs unchanged.
- Activations: gate/up x converted in place per stage, BF16 -> FP16 * 2^4 (exponent add + cvt.rn.f16x2.f32), by the
  4 warps of each k-warp group (one word per token per thread), then bar.sync 2 + kw, 128. Down: the gate/up
  epilogue writes the intermediate BF16-valued (f2bf of h * 16) as FP16 (bf2h2), so no conversion.
- Range: block partial |p| <= 96 |x|max 2^-10 (x * 2^4 representable to |x| < 4094); accumulators typical ~0.5;
  scales s * 2^-4 normal in FP16 for every E4M3 value (subnormal E4M3 2^-9 -> 2^-13).
- Producer: f16t (the _POSH word source) for every item of the config, incl. the early-start stages.

## Log
(see below)

### P1 (2026-10-27) phase 1: all-CUDA-core bs8 (v500), dev/wpp_run1.sh, Modal H200
- SASS (judge NVRTC): moe_4_1_1_4_1 HMMA 0, HGMMA 0; 15.9k instructions; no new spills (LDL/STL as v200).
  Stage bodies per token count n (local nvcc): 538 / 690 / 842 / 994 / 1146 / 1302 / 1454 / 1607 instructions
  for n = 1..8 = 394 + 144 n (decode 10/word, scales 48/stage, HFMA2 4.5 n/word; ptxas puts ~half of the
  HFMA2 on HFMA2.MMA).
- Correctness: check_mt M 1..16 normal + skewed, 0 failures. rel-L2 (wpp_err, M 2/4/8 x normal/skewed x 3 seeds):
  v500 mean 3.65e-3 (max 3.77e-3) vs v200 3.22e-3 (3.32e-3) on the same inputs.
- In-kernel clock (wpp_clk, bs8 jm seeds, judge protocol): CUDA-core build 1.953 GHz median (p10 1.939, p90
  1.964) vs the same build with MOE_NO_CC (tensor) 1.749 GHz (1.732 / 1.766): +11.7% clock, as P6 predicts.
- Speed (bench2 jm 100 reps, 2 copies each, same job): bs8 v200 65.4 / 65.2 us, CC 86.3 / 86.4 us (x0.79-0.80);
  bs16 unchanged (87.5-88.1). Every expert-count stratum loses (e=5-8 -12..-15%, e=10-12 -25%): far worse than the
  stage model (CC stage at n <= 2 should hide under the 1.07 us memory stage). Diagnosis: wpp_run2.

### P2 (2026-10-27) why the CUDA-core bs8 is slow (dev/wpp_run2.sh, v500, bs8 jm seeds)
Per-(phase, n) item cycle in the real kernel (wpp_timn, items after each CTA's first, us):

| n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| tensor G1 | 8.10 | 8.19 | 8.23 | 8.33 | 8.22 | 8.26 | 8.49 | 8.60 |
| tensor G2 | 8.20 | 8.06 | 8.22 | 7.99 | 8.32 | 8.83 | 8.03 | 8.12 |
| CUDA-core G1 | 8.16 | 9.68 | 11.83 | 17.10 | 19.03 | 22.23 | 29.28 | 30.74 |
| CUDA-core G2 | 7.51 | 9.01 | 10.86 | 14.00 | 14.32 | 19.00 | 19.28 | 23.88 |

- n = 1 down items are 8.4% FASTER than the tensor ones (clock + no coupling); n = 1 gate/up items only tie (the
  in-place x conversion + group barrier costs ~0.65 us per item); every n >= 2 item loses.
- Slope ~0.31 us per stage per extra token = ~2.1 cycles per HFMA2 warp-instruction (the issue model assumed ~1):
  ptxas puts half of the plain-register HFMA2 on HFMA2.MMA, and those do not add throughput here.
- Ablations (bench2 40 reps, no verify): tensor 65.0 / 64.9, CC 88.3 / 86.5, no conversion 100.5 (slower: x as
  raw BF16 bits gives more subnormal FP16 products -> data-dependent cost suspected), no barrier 85.8 (barrier ~1-2%),
  data-only CC 61.0 vs tensor 60.6 (same streaming), compute-only CC 82.4 vs tensor 60.5 (CC is compute-bound).
- Next (v502 / v503): FTZ timing test (subnormal hypothesis), pre-scaled words (PRES), private per-warp x conversion
  (XPRIV), and WP-1C's pair layout (dec9 + operand-selector broadcast: inner products stay plain HFMA2).

## Interface for WP-S (scheduler host) -- v504 (dev/moe_v504.py), all behind #ifdef MOE_CC (MOE_NO_CC = v200 path)
Functions touched (v200 -> v504): moe_body (constexpr CC / CCMAX / CCPOOL; routing rank + pool split; early-start
producer: rank-order tiles + pool + first item; producer f16t / esrc word source per tile, claim from the pool's
head; consumer dispatch per item; epilogue staging skip + FP16 intermediate for CUDA-core tiles), new device
functions cc_x16, cc_scp6/cc_scp8, dec9, cc_bc, cc_kk9, cc_stage9, cc_item, cc_pool_split, cc_pool_item (all new),
python: _POS9, _repack_p9, _cc_p9_words, prepare (w13c / w2c), kernel_fn (cc9 word pointers), _smem_bytes.
- A tile is CUDA-core iff its token count n <= MOE_CC_MAX (default 2); everything else about it follows from n:
  producer word source (P.w13h / P.w2h slots carry w13c / w2c in these configs), consumer path (cc_item<n>),
  gate/up epilogue output format (FP16 * 2^4 intermediate for CUDA-core tiles, BF16 otherwise), down input.
- Pools (MOE_CC_POOL): routing ranks tiles largest first (also in the early producer), tB = #tiles with n >
  MOE_CC_MAX = the tensor pool's tiles [0, tB); PA = cc_pool_split(ntiles, tB, #2-group tiles, 96, grid, lane)
  (scal[12] = tB, scal[13] = PA); CTAs [0, PA) claim from CB + 0, [PA, grid) from CB + 7; pool-local index ->
  queue item by cc_pool_item (each pool's G1 items, then its G2 items). Costs MOE_CC_CA / CB1 / CB2 (10 ns units).
- A CTA in the CUDA-core pool must never run a tensor item (the clock cap is sticky within the launch).

### P3 (2026-10-27) v502 / v503 stage variants (dev/wpp_run4.sh, bs8 jm seeds; item cycles us, items after the first)

| variant | G1 n1 | G1 n2 | G1 n4 | G2 n1 | G2 n2 | G2 n4 | bs8 bench (60 reps, no verify) |
|---|---|---|---|---|---|---|---|
| tensor (MOE_NO_CC) | 8.17 | 8.28 | 8.34 | 8.16 | 8.06 | 7.95 | 64.2 / 64.3 us |
| v502 base (= v500) | 8.20 | 9.75 | 17.76 | 7.40 | 9.03 | 14.02 | (v500 86-88) |
| v502 FTZ (timing only) | 8.18 | 9.71 | 16.98 | 7.35 | 9.06 | 13.92 | - |
| v502 PRES (pre-scaled words) | 9.15 | 11.42 | 19.48 | 8.30 | 10.64 | 15.98 | 89.9 |
| v503 P9 (pair layout, plain HFMA2) | 8.30 | 9.07 | 17.01 | 7.36 | 8.46 | 13.49 | 83.8 / 82.3 |
| v503 P9 + XPRIV | 7.79 | 11.15 | 18.84 | 7.59 | 10.68 | 17.31 | 86.3 / 88.5 |
| v503 P9 FTZ (timing only) | 8.29 | 9.07 | 16.22 | 7.40 | 8.52 | 13.22 | 83.3 |

- Subnormals are NOT the cost (FTZ identical). The pair layout's plain HFMA2 (no HFMA2.MMA) helps n = 2 by ~0.6 us
  per item but the per-token cost stays ~2 cycles per HFMA2 warp-instruction (~0.43 us per stage per token at n 3-4),
  2x the planner's assumption: CUDA cores pay only for n = 1 tiles (G2 -10%, G1 -5% with the private conversion).
- XPRIV (per-warp private conversion, no group barrier) makes n = 1 gate/up 7.79 (from 8.30) but its duplicated
  stage body hurt n >= 2 (fixed in v504: one stage body, run-time source and stride).
- Correctness: check_mt M 1..8 (normal + skewed) 0 failures for PRES, P9, P9 + XPRIV; rel-L2 (18 cases) v200 3.22e-3,
  PRES 3.67e-3, P9 3.71e-3 (max 3.81e-3).
- Clock: P9 1.954 GHz, P9 + XPRIV 1.957 (CUDA-core-only CTAs).
=> v504: two pools (CUDA-core pool for tiles of <= MOE_CC_MAX tokens, tensor pool for the rest), MOE_CC_MAX 1 first.

### P4 (2026-10-27) v504 two pools (dev/wpp_run5.sh, jm seeds, 100 reps, 2 copies each, same job)
| build | bs8 us | bs16 us |
|---|---|---|
| v200 / v200b | 65.6 / 65.5 | 88.9 / 88.6 |
| a: MOE_CC_MAX=1 + XPRIV (pools) | 70.5 / 70.0 (x0.914 / 0.919) | 96.0 / 96.4 (x0.926 / 0.919) |
| b: MOE_CC_MAX=2 + XPRIV + ILP | 70.7 / 71.3 (x0.907 / 0.901) | 96.4 / 96.5 (x0.925 / 0.929) |
- Item cycles (a, bs8): CUDA-core n=1 G1 7.68 / G2 7.37 us (tensor-build 8.17 / 8.16); tensor-pool items 8.2-8.5
  (3-5% slower than in the all-tensor kernel). bs16: CC n1 7.75 / 7.43, 2-group tensor items 10.3-11.0.
  ILP (b) did not make n = 2 pay: CC n2 8.78-8.85 vs tensor ~8.1-8.3.
- Sum of item time / 132 + startup + end ~= v200's time, so the 5-8 us loss is the two pools' separate wave
  quantization (each pool ~6 waves of ~8 us items: the makespan is the later pool) -- format independent.
- Clock (wpp_clk): CC-only CTAs 1.946 GHz median but p10 1.801 (bs8), 1.930 / p10 1.776 (bs16); tensor CTAs
  1.782 / 1.764. Some MMA-free CTAs ran capped: the cap may be per TPC/GPC rather than per SM when pools are
  interleaved by blockIdx (untested: record %smid with the clock and place pools by TPC).
- Correctness: check_mt M 1..16 normal + skewed 0 failures; rel-L2 (24 cases) v200 3.22e-3, a 3.34e-3.
- MOE_NO_CC of v504 does not compile (cc_pool_item referenced outside #ifdef MOE_CC; not fixed: work stopped).

## STATUS 2026-10-27: STOPPED. User rule: BF16 wherever SGLang (Marlin W4A16) uses BF16, no FP16 operands, no
FP16/BF16 accumulation. Every CUDA-core variant here (v500-v504) uses FP16 HFMA2 operands and f16x2 partials /
accumulators -> invalid. BF16x2 FMA would accumulate in BF16 (also invalid); FP32 FFMA needs one value per register
(~2x the decode work, and FFMA is the same ~2-cycle FMA-pipe instruction per value pair as measured for HFMA2).

## Format-independent lessons
1. Tensor clock cap confirmed in the real kernel: MMA-free build 1.953 GHz vs 1.749 GHz (bs8, judge protocol).
2. CUDA-core multiply-add throughput in this decode mix is ~2 cycles per packed warp-instruction (HFMA2 plain or
   split onto HFMA2.MMA alike; FTZ / subnormals irrelevant): per token ~8.5 cycles per 32-bit weight word vs the
   tensor stage's ~24.5 cycles per word for up to 8 tokens -> CUDA cores win only for 1-token tiles (-5..-10% item
   time), lose from 2 tokens on. Planner's (18 + 4T) model is ~2x optimistic on the per-token term.
3. Two static pools lose ~8% at bs8/16 from per-pool wave quantization even when every item gets faster; any
   pool scheduler needs tail work stealing (a pool's CTAs take the other pool's items once their queue is empty)
   or much finer items. Pool placement by blockIdx left >= 10% of MMA-free CTAs at ~1.80 GHz: check TPC/GPC
   granularity (%smid) before relying on per-SM clocks.
4. In-place per-stage activation conversion with a 4-warp named barrier costs ~0.5-0.65 us per 8-stage item; a
   per-warp private conversion removes it (7.79 vs 8.30 us) -- relevant to any format change of gathered x.
5. Duplicated stage bodies (two call sites of the same unrolled stage) cost 10-25% at n >= 2 (i-cache / reg alloc):
   pass the source pointer and stride at run time into one body.
