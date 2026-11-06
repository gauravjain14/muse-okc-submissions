# base worker, round 3 (tile cost + cluster paths), start = runs/k80.py (work/base/cand3.py)

## Per-tile cost analysis (k77/k80 w8_tile, ~190 SASS instr per warp per 64-token tile)
Biggest items per tile per warp: 36 FFMA (O update), ~29 HMNMX2/VHMNMX (max + clamp), 16 HFMA2, 18 PRMT, ~40
integer/address/control (descriptors, need_mask 64-bit compares, mbarrier wait, loop), ~20 scalar softmax (alpha,
bias), V^T transpose 2 LDSM + 8 PRMT + 2 STSM + fence.
FP16-PV idea (coordinator): does not cut ALU work here - V would need an FP8 -> FP16 convert into smem (2 LDS.128 +
16 cvt + 4 STS.128 per thread per tile vs today's 2 LDSM + 8 PRMT + 2 STSM), and FP16 P needs either MUFU ex2 (16
f16x2 per thread per tile; MUFU-bound at ~512 clk per tile-step for 4 WGs) or the E4M3 code + 16 cvt.f16x2.e4m3x2 on
top of today's path. Not pursued.
Instead: amortize the per-tile fixed costs over 2 tiles (w8_pair): one max/alpha/bias, one O rescale (36 FMUL per
128 tokens, PV accumulates in place -> no t registers), shared loop/barrier overhead.

## w8_pair results (kept as work/base/t/cand3_pairs.py, not in cand3.py)
- Correct (16/16 tests). Always-pair (BZ_PR=2-like first version): v32s +2.08, v16s +0.20 vs k80 (ab, 6 seeds).
- Adaptive (pair only if tile k + 2 has already landed; warpgroup-uniform decision via bar.red.or + mbarrier
  test_wait): v32s +2.11, v16s +0.16.
- clock64 inside run8w (v32s, WG0): a pair costs 1650 clk = 825 per tile vs 1098 per tile for w8_tile (-25 %), yet the
  streams finish 0.4-0.6 us later (trace) and the kernel is 1.3-2 us slower. The pair code raises register pressure
  in the global-path kernel (STACK 32 -> 48 B, 5 -> 12 local loads/stores, including 64-bit reloads at the combine
  entry after the loop); v16s (no new spills) still loses 0.16-0.2 (one pair + one single tile per stream, extra
  barriers and the per-tile pair check). Not resolved; reverted.

## Round 3 result
No kept change: work/base/cand3.py == runs/k80.py.
