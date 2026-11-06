---
name: okc-prefill-moe-worker
description: "Prefill expert-path (MoE) worker results 2026-10-25 — shipped vN chain, measured limits, dead ends"
metadata:
  node_type: memory
  type: project
  originSessionId: <session>
  modified: 2026-10-25T03:26:25.015Z
---

Expert-path files: muse_prefill_layer/cand_dense2_moe_vN.py (each = cand_dense2 + expert diff, build_cand/patch_pdl compatible). Best chain (layer bench --check, same-run gains): v6p bucketing+PDL+16K pnr2 (+1.56%) → v7 ldmatrix mma.sync on wgmma layout (+0.96%) → v9 MOE_XPU=1 for mw<512 (instruction cache; +0.67%) → v14 pnr9 (routing on a 9th warp, REDUX top-k; +0.65%) → v15 row-major pre-gathered slots + TMA maps at 16K (+0.78%) → v16 pre-gather at 3012 → v21 two-pass in-kernel gather at 753 → v23 160-thread routers (+0.34%). v23 = best as of 2026-10-25.

Measured limits (dev/wgmma_probe.py, logs/wgmma_probe3.log): RS-wgmma alone ~95% of peak for N>=48; with the dequant chain the per-stage fixed cost ~0.9 us is the dequant issue stream (small tiles in-kernel = standalone). Mid tiles in-kernel ~25-35% over standalone (unexplained; not drains, not latency).

Dead ends (do not retry): hybrid hot/cold two-kernel split; variable ring units (VU); async epilogue via bulk S2G (AEPI); IMAD.WIDE rotations in dequant (+30% slower); drain overlap of next stage lead-in (WGPRO); L2 prefetch-ahead (MOE_PD); XPU=4; NTL padding; WGMIN 3/4; KSX8=2 tail split; finishes: all-slots 2D Triton, unrolled masked Triton, ssq-from-epilogue Triton (2x slower), cluster-of-4 DSMEM CUDA finish, in-kernel combine (753 only +1% in moe_bench, 0 in layer); pre-gather at 753; vectorized / smem-staged in-kernel gather; warp-per-token router (pnrw).

**Why:** the coordinator (main session) asks for vN deliverables; these results avoid repeating failed ideas. **How to apply:** start from cand_dense2_moe_v23.py; see [[okc-prefill-start]] for the layer-level state.

2026-10-26 session (real routing, dev/moe_item_tl.py per-item tracer, dev/hot_probe.py, dev/corun_probe.py):
- Per-item cost at 753 ~= 16.6 us + 0.218 us/token (1M weights per item); small items (n<=32) = 34% of CTA time for ~4% of routes; hot n128 items 44-45 us at 753/3012/16K alike (84% of MMA).
- hot_probe: the full-width stage loop (stage_compute_wg2, resident smem) runs at 98.9-99.9% of 4096 clk, drain every 1/2/4 stages or never; + a concurrent 102 KB/stage bulk-copy writer: 99.1%. So hot-item losses are the pipeline: epilogue ~6%, stage waits ~4.5%, misc ~4% -- smem-capacity bound (2 x 103 KB stages, 23 KB free).
- Small items are compute-bound (same 3450 clk per double-k stage on 16/40/132 SMs; waits 3-4%): ~15 instr per 8-code word (9 decode + 4 HMUL2 + HMMA + misc), loop at ~64% of issue/ALU bound; ncu stalls selected/not_selected/math/dispatch. Best possible lossless decode = 6 ALU + 3 IMAD (9 instr, same as now; counting argument: 12 low targets vs 9 low sources).
- Co-run (4th-WG tiny lane beside the main lane): dequant-only lane slows main 12%; mma.sync lane 3x slower than alone. Register file caps it. Dead.
- DEAD (measured): NT=18 / 144-token tiles (v42: compute -19 us/CTA on bench routing but idle +9, A/B -0.8/-2.8%); queue reorder big-p1 / big-p2+small-p1 / small-p2 (v43/v44: -1.4..-2.5%); router pre-gather at 753 (PG_MIN_M 512: -1%).
- KEPT: MOE_STSM (stmatrix.x4.trans staging of the whole-k down epilogue, bit-identical): +0.1..0.4% MoE (v45), layer cand_full29 ~+0.15%.
