# WP-D: consumer stage loop (decode + mma.sync), microbenchmarks on Modal H200

Worker D, 2026-10-26/27. All numbers: Modal H200, NVRTC 13.0, 1 CTA per SM (132), SMEM-resident data, cycles
from clock64 per CTA (median over CTAs), SM clock from clock64 vs %globaltimer in the same region. Tools:
`dev/wpd_util.py` (compile, SASS histogram of the hot loop, timing), `dev/wpd_pipe.py` (Q1), `dev/wpd_stage.py`
(Q2/Q3 stage variants + real-data error).

## Q1. Issue cost per SASS instruction and which pipes co-issue

Single ops: 16 independent chains per thread per op, `#pragma unroll 1`, SASS loop histogram checked (ptxas folds
some chains: IADD3 pairs fuse into 3-input IADD3, f16/bf16 HMUL2/HFMA2/HADD2 are split half onto `HFMA2.MMA`).
W8 = 8 warps/CTA (2 per SMSP, the kernel's consumer count), W16 = 4 per SMSP. Cycles per warp-instruction per SMSP.

| SASS op | W8 | W16 | pipe (from mixes) |
|---|---|---|---|
| LOP3.LUT (imm or 3 regs) | 2.06 | 2.06 | ALU |
| SHF.L.W / SHF.R | 2.06 | 2.06 | ALU |
| PRMT | 2.06 | 2.07 | ALU |
| SEL | 2.06 | 2.06 | ALU |
| IADD3 | 2.13 | 2.13 | ALU |
| SGXT (bfe -> SHF+SGXT) | 2.03 | 2.03 | ALU |
| I2FP.F32.U32 | 2.06 | 2.06 | ALU (additive with LOP3) |
| F2FP.F16.F32.PACK_AB, F2FP.F16.E4M3.UNPACK_B | 2.06 | 2.06 | ALU (additive with LOP3, co-issues with IMAD) |
| HADD2.F32 (f16 -> f32) | 2.05 | 2.00 | FMA-heavy class |
| F2I.U32.TRUNC | 8.0 | 8.0 | XU |
| IMAD (reg or c[] multiplier) | 2.05 | 2.00 | FMA-heavy |
| IMAD.HI.U32 | 4.11 | 4.01 | FMA-heavy, 2 passes |
| IMAD.WIDE.U32 | ~4.4 | ~4.3 | FMA-heavy, 2 passes |
| add x,x / shl+add (ptxas: IMAD.IADD + IADD3, IMAD + LEA mix) | 1.25-1.30 | 1.23-1.25 | ptxas balances ALU + FMA |
| FFMA / FMUL / FADD | 1.10 | 1.15 | FMA (heavy + lite) |
| HMUL2 / HFMA2 / HADD2 (f16 or bf16), ptxas half on HFMA2.MMA | 1.16-1.22 | 1.16-1.18 | FMA + MMA pipe |
| HMUL2 / HFMA2 all on the FMA pipe (next to HMMA ptxas does not split) | ~2.0 | ~2.0 | FMA-heavy |
| HMMA.16816.F32.BF16 / .F32 (f16 in) | 4.42 | 4.42 | tensor; clock drops to 1.80 GHz |
| HMMA.16816.F16 (f16 accumulators) | 4.34 | 4.34 | tensor |
| HMMA.1688.F32.BF16 | 4.01 | 4.11 | tensor |

Clock: 1.97-1.98 GHz for every non-tensor kernel, 1.79-1.80 GHz as soon as HMMA runs (confirms the tensor clock cap).

Mixes (cycles per loop iteration of one warp per SMSP, W8; "sum" = sum of the single-op costs):

| mix (per iteration) | measured | sum of singles | reading |
|---|---|---|---|
| 32 LOP3 + 32 IMAD(c[]) | 67.1 | 131.7 | ALU and IMAD co-issue fully |
| 16 LOP3 + 32 IMAD / 32 LOP3 + 16 IMAD | 67.1 / 67.6 | 98.7 / 98.9 | full overlap, bound by the busier pipe |
| 32 LOP3 + 32 IMAD (3 reg sources) | 72.1 | 131.7 | almost full |
| 32 LOP3 + 32 FMUL | 73.0 | 101.1 | almost full |
| 32 LOP3 + 32 FFMA (3 reg sources) | 101.0 | 101.1 | none |
| 32 LOP3 + 32 HMUL2 / HFMA2 | 80.0 / 85.1 | 105 | partial |
| 32 IMAD + 32 FFMA / + 32 HMUL2 | 97.5 / 100.0 | 101 / 105 | none (same FMA-heavy pipe) |
| 32 LOP3 + 32 SHF / PRMT / I2F / E4M3 cvt | 130 | 132 | one pipe (ALU) |
| 32 IMAD + 32 E4M3 cvt | 67.1 | 131.6 | cvt is ALU, co-issues with IMAD |
| 32 LOP3 + 8 HMMA (F32 acc) | 101.0 | 119.1 | HMMA.F32 takes ~4.6 cycles of exclusive issue |
| 32 LOP3 + 8 HMMA (F16 acc) | 85.0 | 118.1 | HMMA.F16 only ~2.6 |
| 64 FFMA + 8 HMMA.F32 / HMMA.F16 | 101.1 / 90.1 | 123 / 122 | |
| 32 HMUL2 + 8 HMMA.F32 | 98.5 | 92.1 | HMUL2 back at 2 cycles (no MMA-pipe split) |
| decode 56 LOP3 + 16 IMAD + 32 HMUL2.BF16 (= 8 words) | 136.6 | 187.5 | 17.1 cycles/word without MMA |
| same + 8 HMMA.F32 (today's mix) | 168.6 | 240.5 | **21.1 cycles/word** |
| same + 8 HMMA.F16 | 148.6 | 239.5 | 18.6 (-12%) |
| FP16 decode 56 LOP3 + 24 IMAD + 32 HMUL2 + 8 HMMA.F16 | 157.1 | 255.9 | 19.6 (-7%) |
| post-scale 64 LOP3 + 32 IMAD + 16 HFMA2 + 8 HMMA.F16 | 150.1 | 269 | 18.8 (-11%) |
| 2 groups: 56 L + 16 I + 32 HMUL2.BF16 + 16 HMMA.F32 | 226.6 | 293.6 | 28.3/word |
| 2 groups, FP16: 56 L + 24 I + 32 HMUL2 + 16 HMMA.F16 | 197.9 | 308.0 | 24.7/word (-13%) |

Corrections to the old dispatch model ("every half-rate INT op holds dispatch for 2 cycles, nothing co-issues"):
- The ALU pipe (LOP3/SHF/PRMT/SEL/IADD3/I2FP/F2FP) and the FMA-heavy pipe (IMAD, IMAD.HI, HMUL2/HFMA2 when not on
  the MMA pipe) DO co-issue: 32 LOP3 + 32 IMAD costs the same 67 cycles as 32 LOP3 alone. The decode is bound by
  max(ALU, FMA-heavy) work, not by their sum, as long as instructions have few register sources.
- Instructions with 3 register sources (FFMA, 3-reg IMAD, HFMA2) do not overlap with ALU work (register-file read
  bandwidth looks like the shared limit).
- HMMA overlaps with nothing: each HMMA.16816 with FP32 accumulators removes ~4.6 issue cycles from the ALU/FMA
  work around it, with FP16 accumulators ~2.6. That is the largest single lever found in Q1.
- f16x2/bf16x2 HMUL2 costs 2 cycles on the FMA pipe; ptxas only halves it (HFMA2.MMA) when no HMMA is around.

## Q2/Q3. Stage-loop variants on one 32 KB stage (dev/wpd_stage.py)

8 consumer warps x (2 row groups x 256 k), 8-bit scales, SMEM-resident (two alternating stage buffers), 132 CTAs,
1.79-1.81 GHz in every variant. Error: the variant's own data layout built from the problem's weights (expert 3,
256 rows, k = 4096 = 16 stages) and activations (w13: generate_input hidden states; w2: the reference's
SwiGLU intermediate), rel-L2 of the 256 x (8 NTL) outputs vs FP64.

| V | stage loop | NTL1 cyc/stage | NTL2 | rel-L2 (w13 / w2) |
|---|---|---|---|---|
| 0 | v200: BF16 decode, 4 HMUL2.BF16/word, HMMA.F32.BF16 | **1450** | **1804** | 1.0e-6 / 2.9e-6 |
| 1 | FP16 decode (_POSH), 4 HMUL2/word, HMMA.F16 accumulators | 1411 (-2.7%) | 1634 (-9.4%) | 2.3e-3 / 2.4e-3 |
| 2 | as 1 with HMMA.F32 (FP16 in, FP32 acc) | 1546 | 1855 | 1.0e-6 / 2.9e-6 |
| 3 | post-scale: unscaled FP16 codes, MMA k16 = 1 block, HMMA.F16 C=0, 2 HFMA2 (.H0_H0/.H1_H1) by cvt.e4m3x2 scale pair, FP16 acc | **1301 (-10.3%)** | 1650 (-8.6%) | 2.4e-3 / 2.3e-3 |
| 4 | as 3, scale pair by PRMT + IMAD | 1390 | 1676 | same as 3 |
| 5 | as 3, activations permuted for 2 LDS.128 per chunk (timing) | 1367 | 1621 (-10.1%) | - |
| 6 | V1 + FP16 acc added into FP32 every stage (16 blocks) | 1468 | 1708 | 6.1e-4 / 6.2e-4 |
| 7 | V3 + same flush | 1332 (-8.1%) | 1740 | 6.5e-4 / 6.4e-4 |
| 9 | V3, scales pre-converted once per warp into shared scratch | 1349 | 1672 | as 3 |
| 3s | V3 with _POSS (6 ALU + 3 IMAD + IMAD.HI) | 1373 | 1800 | as 3 |
| 1s | V1 with _POSS | 1618 | 1871 | as 1 |

SASS per stage per warp (32 words): V0 161 LOP3 + 64 SHF + 16 PRMT (ALU 241), 83 IMAD + 128 HMUL2 (FMA-heavy),
32 HMMA.F32; V3 193 LOP3 + 32 SHF + 32 F2FP (ALU 257), 99 IMAD + 64 HFMA2, 32 HMMA.F16.

Readings:
- The accumulator type matters most: the same FP16 decode costs 1546 with FP32 accumulators and 1411 with FP16
  ones (-9%), at 2 groups 1855 vs 1634 (-12%).
- Post-scale (V3) is the best 1-group stage: -10.3%, although it has 16 more ALU ops per stage (the cvt per MMA)
  than v200; it drops 64 FMA-pipe multiplies and uses FP16 accumulators. Scale conversion cost: cvt.e4m3x2 (1 ALU
  op) beats PRMT + IMAD; removing the 4x quad-redundant cvts through shared memory costs more than it saves.
- FP16 accumulation over k = 4096: rel-L2 2.3-2.4e-3 per GEMM (inside the 1.5e-2 budget, comparable to the FP16
  wgmma path already in production). A per-stage FP32 flush brings it to 6.5e-4 for ~30 cycles/stage at 1 group
  (more at 2: the HADD2.F32 conversions are 2-cycle ops).
- Magnitudes (expert 3): x rms 1.0, max 4.7; intermediate rms 0.59, max 8.3. Post-scale holds x as FP16 x * 2^8
  and the intermediate as v * 2^4 (FP16 overflow at |x| > 255 / |v| > 4095), block partials D ~ 0.05-0.2 (normal
  FP16), accumulators ~100-2000 (no overflow seen; nonfinite 0).
- Q3: moving int ops to the FMA-heavy pipe does not pay once it carries the multiplies: _POSS (IMAD.HI at 4
  cycles) loses 5-15%. The ALU pipe stays the binding unit of the post-scale stage (257 ALU ops = 514 cycles of
  1301 per warp-stage x 2 warps/SMSP).

## Q4. Stage boundary (full-barrier wait + lane-0 arrive), isolation (dev/sc_bench.py kloop on v200)

| loop | NTL1 | NTL2 |
|---|---|---|
| no ring (plain loop) | 1451 | 1791 |
| ring: wait + arrive per stage (kernel's form) | 1615 (+164, 11%) | 1922 (+131, 7%) |
| ring, 2 stages per wait | 1512 (+61) | 1812 (+21) |
| test next barrier early + prefetch (MODE 4) | 1658 | 1969 |
| all operands up front, release before math (MODE 5) | 1632 | 1887 |
| warps 4-7 offset by half a stage (MODE 6) | 1540 | - |

The boundary costs 164 cycles at 1 group in isolation; waiting once per 2 stages removes 100 of them (and 110 at
2 groups). Hand prefetch across the boundary loses (it breaks the compiler's 2x unrolled overlap). In the kernel
this means 64 KB consumer steps (two slots per wait), i.e. half the ring granularity.

## Real-kernel A/B (dev/moe_v450.py)

| switch | change | result |
|---|---|---|
| MOE_PS1 | one-token kernel (bs1): post-scale gate/up (new unit layout) and down (_POSH w2 slices), x and intermediate FP16 | bs1 26.6 / 26.6 us vs v200 27.4 / 27.7 (x1.034-1.035), check_mt M=1 0 failures (run wpd_run1) |
| MOE_PS | one-k-warp configs (bs17-512): 1-group tiles on stage_ps (own words/scales w13ps/s13ps/w2ps/s2ps, wc13ps/wc2ps for the 64-token config's combined blocks), gathered x FP16 * 2^8, intermediate FP16 * 2^4 (staged and async epilogues) | bs32 x1.012/1.014, bs128 x1.007/1.008, bs256 x0.999/1.004 (controls 0.2-0.3% apart) |
| MOE_PS + MOE_PS2 | + 2-group tiles on stage_h (FP16 decode, A-side HMUL2, FP16 accumulators; the existing _POSH words), FP16 x / intermediate | bs32 x1.0135/1.0129, bs128 x1.0128/1.0134, bs256 x1.0142/1.0199 (controls within 0.3%) |

| MOE_PS + MOE_PS2 + MOE_PS1 (confirmation, one job) | all three | bs1 x1.039/1.037, bs32 x1.011/1.011, bs128 x1.011/1.012, bs256 x1.012/1.016 (controls within 0.2%): **kept** |
| MOE_PAIR + MOE_PAIR1 | Q4 in the kernel: two stages per basic block when the next slot is already full | bs1 x0.92, bs8 x0.93-0.94, bs16 x0.91, and on top of PS+PS2 bs32 x0.978, bs128 x0.981, bs256 x0.971: rejected (the isolation gain 1615 -> 1512 does not carry; doubled bodies + per-stage test_wait/vote) |
| MOE_PSK2 | bs<=8 config on post-scale (8-bit scales, per-CTA FP16 x copy at start by warps 10-11) | bs8 x0.85; timing-only no-wait variant still x0.94: rejected |

Correctness: check_mt (M 1..257, normal + skewed, 2 calls each) 0 failures for every combination; hang1 seed-11
rel-L2 (v200 -> MOE_PS+PS2+PS1): bs1 2.5e-3 -> 3.3e-3, bs8/16 unchanged 3.2e-3, bs32 3.1e-3 -> 5.2e-3, bs128 4.1e-3 ->
5.35e-3, bs256 4.6e-3 -> 5.4e-3 (tolerance 1.5e-2).

Why bs8/16 need more than a stage swap: their gate/up activations are raw BF16 rows bulk-copied from P.x per stage
(no gather), and an FP16-accumulator MMA needs FP16 B. Converting per stage in shared memory costs ~2 ALU ops per
pair (8 pairs per thread per stage, ~5% of the stage) plus a consumer barrier, about what post-scale saves. MOE_PSK2
(bs<=8 config) instead has warps 10-11 (idle there) write one per-CTA FP16 * 2^4 copy of x at kernel start (no
grid-wide dependency; fence.proxy.async + an mbarrier to the producer), drops the 6-bit scales (post-scale needs the
8-bit layout; the 8-token config keeps NS = 4, the 16-token config would drop to NS = 3, so it is excluded) and runs
every tile on stage_ps.

## Notes for other packages

- S6O (WP-E's 6-bit register format) and post-scale: S6O gives one register per (row group, 64-k chunk) with rows
  g / g + 8 in the halves, which fits the kperm layout (one scale block per lane per chunk). Post-scale needs, per
  lane, 16 pairs per row group per stage (one per MMA block, the same for the 4 lanes of a quad). With 6-bit fields
  the (w & M) | C trick gives a pair in 1 LOP3 only for fields already at bits 7-12 / 23-28; a 32-bit word then holds
  2 pairs (the second via an IMAD x128 before the LOP3), i.e. 32 bytes per row group per lane, no smaller than 8-bit.
  Packing 32 six-bit fields into 6 words needs funnel shifts (extra ALU) on the binding ALU pipe, so a 6-bit
  post-scale format costs ~+0.3-0.5 ALU op per word (~2-3% of stage cycles) to save 2.8% of bytes: about even.
- bs8/16: the remaining blocker for FP16 paths is the raw BF16 x of gate/up items. The 16-token config misses NS=4
  with 8-bit scales by exactly 1 KB of shared memory (233472 vs 232448 B).
- Why the kernel gains are smaller than the stage gains (-10% cycles -> +1-2% time at bs32-256): only 1-2 group
  tiles use these stages (larger ones are on wgmma), and bs32+ items are partly data-bound.

## BF16-only rule (2026-10-27): BF16 post-scale (wpd_stage.py V12, V13)

Same harness (V0 = v200 1450 / 1804 cycles per stage at NTL1 / NTL2, 1.80 GHz).

| V | stage | NTL1 | NTL2 | rel-L2 (w13 / w2) |
|---|---|---|---|---|
| 12 | _POS codes unscaled, MMA k16 = one block, BF16 x BF16 -> FP32 with C = 0, 4 FFMA by FP32 s * 2^104 (E4M3 bits at [26:20] OR 0x70000000; IMAD shift + LOP3 per scale), both row groups' MMAs before their FFMAs | 1631 (+12.5%) | 2221 (+23%) | 1.7e-7 / 2.6e-7 |
| 13 | as 12, scales converted once per warp-stage into FP32 shared scratch | 1892 (+30%) | 2375 (+32%) | same |

Not ported (threshold was >= 4% faster). The 128 FFMA per warp-stage have 3 register sources and do not co-issue
with the ALU decode (Q1: LOP3 + FFMA = the sum of both), the FP32 accumulators keep HMMA at ~4.6 issue cycles, and
each MMA needs two FP32 scales (64 conversions per lane-stage vs 16 PRMT today). The numerics are exact: subnormal
BF16 codes (0.5 * 2^-126) and FP32 block partials near 2^-122 are handled without flush by the tensor core, so a
BF16 post-scale is valid, just slower. Under the BF16 rule the FP16 wins above (MOE_PS / PS2 / PS1) do not apply.
