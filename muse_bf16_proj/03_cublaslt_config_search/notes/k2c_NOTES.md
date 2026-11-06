# K2c notes: M = 128 / 256, four BF16 layers. Result: does NOT beat cuBLAS. Do not integrate.

## Design built (k2c.py)
- Swap-AB wgmma. A = W, with 64 rows per consumer warpgroup and two consumer warpgroups, so 128 rows per CTA. B = x^T, TN = 64/128/256 tokens, loaded by TMA with 128B swizzle and K-major layout.
- 12-bit weights (K1s format, `decode12`) decoded straight into the wgmma A registers. Decode of chunk s+1 overlaps the wgmma of chunk s (double-buffered A, `commit_group`/`wait_group 1`).
- Raw BF16 fallback works per (warp, k16 step) unit, about 3% of units: a per-chunk bitmap plus a per-CTA overflow area copied into smem at start.
- BF16 SS variant: pre-swizzled 64x64 tiles.
- Producer: one thread issuing one bulk copy (W) and one TMA box (x) per 64-k chunk; `setmaxnreg` 40/232.
- Split-K, three modes:
  - S=1: direct BF16 store.
  - S=2: cluster pair, with remote shared-memory (`ld.shared::cluster`) reduction.
  - S>2: `red.add.v4.f32` into a self-zeroing fp32 workspace, then a PDL convert kernel.
- Correct on every test, all four layers: M 33, 64, 65, 100, 127, 128, 129, 130, 200, 255, 256, normal and outlier inputs, both COMP and BF16 (`t_k2c.py`).

## Final bench (run8.log, local µs)
| M | layer | SGLang | K2c 12-bit | K2c BF16 | Codex v2 |
|---|---|---|---|---|---|
| 128 | qkv | 16.05 | 23.30 | 21.67 | 15.86 |
| 128 | o | 6.24 | 15.60 | 14.49 | 6.62 |
| 128 | pooler | 24.72 | 30.60 | 28.37 | 24.46 |
| 128 | unpooler | 18.62 | 29.24 | 25.81 | 18.48 |
| 256 | qkv | 18.59 | 28.78 | 28.28 | 19.88 |
| 256 | o | 8.19 | 20.82 | 20.21 | 8.17 |
| 256 | pooler | 30.53 | 38.88 | 37.89 | 29.61 |
| 256 | unpooler | 23.46 | 38.56 | 36.58 | 23.34 |

## Measured facts (trace_k2c.py, in-graph globaltimer stamps)
- **Main loops are L2->SM ingress bound.** Each SM pulls W plus x for its k-range at about 54-70 GB/s per SM, about 7-9 TB/s aggregate. With 128-row tiles, x re-reads (N/128 x M x K x 2 bytes) are as large as or larger than W.
  - Loop end with BF16 weights, which is HBM-bound at ~4.5 TB/s: o m128 3.6 us after first data; unpooler m128 16.5 us.
  - The 12-bit loop is not faster. It reads fewer HBM bytes, but per-SM ingress plus decode keep it at about the same time.
  - The loop alone is already at or above cuBLAS's total time on 6 of 8 shapes.
- **Staggering chunk order across row blocks** (to avoid L2 hot lines): no gain.
- **Global split-K reduction is expensive.**
  - `red.add.v4.f32` drains at only ~2 TB/s: o m256 (16.8 MB of reds) costs ~9 us after the last red is issued; qkv m128 (8.3 MB) ~4 us.
  - A naive PDL convert kernel was latency-bound; 4-way unrolling helped but did not fix it.
- **Cluster-pair reduction** (remote shared-memory loads, after a cluster barrier): ~6 us for 32 KB per CTA under load, even with 8 loads in flight per thread. Pathologically slow; not understood. K1's push variant (`st.shared::cluster`) was ~0.5 us, so use push if this is revisited.
- **setmaxnreg** 40/232 vs none: no difference.
- **Startup:** CTAs start ~1.4-1.5 us after the preceding kernel; the first chunk lands ~1.6 us after CTA start.

## Why it cannot win as designed
- For M=128 on 132 SMs, every decomposition pays two costs, and their sum is ~50 MB or more of extra on-chip traffic for qkv:
  - x re-read traffic: (N/R) x 2 MB.
  - split-K reduction traffic: 132 x R x M x 4 bytes.
- 256-row tiles at M=256 do not fit in registers.
- cuBLAS is within ~20-40% of the HBM floor at m128 and close to the compute floor at m256.
- A win would need the split-K reduction kept on-chip at near-zero cost together with x multicast. That requires clusters of 4 or more, which only reach 120 SMs.

## Files
- `k2c.py`: kernels, prepare, launch. `prepare_k2c(weights, comp)`, `run_k2c(x, layer, n, st)`.
- `sub_k2c.py`, `sub_k2c_bf.py`, `sub_k2c_ns.py`, `sub_k2c_bfnosmr.py`: dev candidates for the variants.
- `t_k2c.py`: correctness sweep.
- `trace_k2c.py`: timeline with in-graph stamps.
- `run*.log`: raw logs.
