# xbg round 3 (XE section: d1s xe_r1_1, v1s xe_b1s_8c6), base runs/k77.py (work/xbg/cand3.py == k77, unchanged)

## The proposed end-phase distribution is already the current design
- d1s (xe_r1_1, xer1::xe_b1s<1>): XC<1>::NOW = 1; owners = CTAs rank 0..7 of each 16-CTA cluster (`own = rank < 8`),
  owner rank r owns ONE row = q head 8h + r (`rho = rank`). Every source CTA's CTA-combine pushes row rho's 64-value
  slice + (M, l) with st.async to CTA `ow = rho` (recv[16 sources][1 row][17 x float4]). So each of the 8 owners merges
  16 x 64 values for one head, does that head's RMSNorm locally (W9 NS smem broadcast inside the warp), gate and a
  128-B store; the 8 owners run in parallel; ranks 8..15 only compute + push.
- v1s (xe_b1s_8c6, xer2::xe_b1s<8, 6>): NT = 1 (one token per cluster: 16 clusters = 8 tokens x 2 KV heads), 8 rows per
  cluster; owners = ranks 0..3 with NOW = 2 owner warps each (`2 rank + pw < 8`), `ow = rho >> 1`, slot `rho & 1`:
  each owner WARP merges 6 x 64 values for one head row and normalizes it locally. Again one row per warp, all in parallel.
- Epilogue gates: loaded at the owner prep warp's entry (`gt2`, .nc, before the wait) and stashed in smem
  (sm.stash with the k/v scales): already early.

## Why the variants were not built
- The remaining owner tail (XE worker's ablation: 0.36 us d1s / 0.25 us v1s) is one warp's merge + norm for one row
  (2 values per lane). Splitting a row over 2 warps (d halves) keeps the merge's dependency chain (16 sources, OLP
  weights + 2 FMA chains of 8) and only halves the per-lane instruction count, then adds a cross-warp sum-of-squares
  (smem + named barrier): expected <= ~30 ns, below the A/B floor (+-0.02-0.03 with 64+ instances).
- 16 owners (half a head each): same chain length per lane, plus a cross-CTA (DSMEM) sum-of-squares exchange
  (st.async + mbarrier, ~0.15-0.3 us): worse by construction.
- v1s: owners already one warp per row; spreading 8 rows over all 6 CTAs leaves the per-warp work unchanged.
