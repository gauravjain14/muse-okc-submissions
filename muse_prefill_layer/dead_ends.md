# muse_prefill_layer: dead ends

Failures that changed direction, from the start through milestone 10. Each row gives the idea, what it measured, why it failed, which milestone it informed, when it was found, and the source.

Path keys (all relative to ``):

| key | path |
|---|---|
| MO | `muse-opt/muse_prefill_layer/` |
| MEM | `<agent-state>/memory/okc-prefill-start.md` (line numbers) |
| R | `muse-opt-codex/muse_prefill_layer/research/` |
| RC | `muse-opt-codex/muse_prefill_layer/research/claude/` |

## Attention

| idea | number | why | informed | found | source |
|---|---|---|---|---|---|
| FMA-pipe polynomial exp2 (`ATT_POLY` 1/2/3) | case-1 attention 435-481 vs 386 µs; POLY1 592.7 vs 392.5 µs | ALU and issue already ~half busy; offload only adds instructions | 01, 03 | 10-24 | MO/logs/a8_01.log, a8_03.log, ncu_poly.log |
| 2 consumer warpgroups (`ATT_CW 2`), any ping-pong mode | 557-666 vs 381-394 µs | too few warps to hide the serial softmax chain | 01 | 10-24 | MO/logs/a8_01.log, a8_04.log |
| bias-wgmma for −m; split-QK halves; lazy max | +23 %; +22 %; +9 % | more serial work in the softmax window | 02 | 10-24 | MO/logs/a8_bias.log, a8_split.log, a8_lazy.log |
| stream-K for sliding layers (`UNIT_SLIDING 0`) | case 2 154.4 vs 67.9 µs (+18..+76 % across sliding cases) | too little work per query block; units win | 02 | 10-24 | MO/logs/a8_unit0.log, a8_unit1.log; MEM:63 |
| direct new-key packing above 1024 tokens | restricting it gained +0.8 % (16K +1..+2.3 %) | byte scatter loses on large blocks | 02 | 10-24 | MO/logs/a8_dn0.log, a8_dn1.log, bench_full7.log |
| double-buffered f16 S (`ATT_F16S 2`) | −25 % with CW2 | wgmma serialization + spills | 02 | 10-24 | MEM:27, 40 |
| runtime-only "no combine" switch (`ATT_NOCOMB_RT`) | recovers ¼ of the 11 % | the cost is the presence of the combine code in the tile loop | 05 (stream-K-only module) | 10-25 | MO/logs/attn_nocomb_rt.log |
| finishing split blocks inside attn_fwd (`ATT_INCOMB`) | attn_fwd +5..+9 % | the same codegen penalty returns | 05 | 10-25 | MEM:57 |
| per-CTA partial slots at 16K | cases 6/7 +3.1 / +3.9 % slower | main loop of the slot build is slower at 16K | 05 (slots only ≤ 4096 tokens) | 10-25 | MO/logs/attn_comb2.log |
| early K/V loads before the PDL wait (`ATT_EARLYKV`); gate-row L1 prefetch (`ATT_GPF`) | −0.4 %; −0.3..−1.2 % | SM contention at kernel start | 05 | 10-25 | MEM:51, 69 |
| public okc-2026 attention | case 1 368.5 vs 243.4 µs; 16K full 6848 vs 3862 µs | FP32 scores, SFU exp2 | 06 (only the exact early vote was ported) | 10-29 | R/public_attention.log |
| lazy EXPQ max with an OR guard | −7..−20 % (bit-identical) | | 07 | 10-29 | R/RUN_STATE.md |
| 16-query / CW2 attention variants | −21..−24 % | | 07 | 10-29 | R/RUN_STATE.md |
| larger rescale headroom (D2-D6) | case 3 +30..+50 µs, but outputs move 1.3-2.6 % | a precision reduction beyond what was accepted | 09 | 10-30 | RC/attn/NOTES.md |
| run-time split search; split units on full layers | −14 / −45 µs; +2.0 vs +4.4 µs | dependent loads; idle SMs | 09 | 10-30 | RC/attn/NOTES.md |
| fused multi-unit path for case 2 (lat v2-v5); balanced key split | case 2 −8.8..+2.3 µs; outputs move 2.2 % (routing flips) | per-unit production overhead; changes numerics | 09 | 10-30 | RC/lat/NOTES.md |
| direct K/V gather from the paged cache inside attn_fwd (dkv) | front −166 µs (case 1), −658 µs (case 3) | the gather producer is 1.8× slower than the tensor work | 10 | 10-31 | RC/kv/NOTES.md |
| speculative codes, half votes, hoisted probes, prefetches in attn_fwd (kv lane) | −1.7..−117 µs | loop-body growth, i-cache pressure, ptxas hoists the wait | 10 | 10-31..11-01 | RC/kv/NOTES.md |

## Routed experts (MoE)

| idea | number | why | informed | found | source |
|---|---|---|---|---|---|
| async bulk-S2G expert epilogue (AEPI, MOE_BSTORE, AEPI2, WSPLIT) | ×1.003-1.010 vs v9's ×1.029-1.036; −0.6..−2 % | store path shared with LSU/MIO; slot held longer | 04, 05 | 10-24..25 | MO/logs/moe_v9v10b.log; MEM:57, 65 |
| combine inside the expert kernel (v22, later mf) | MoE ×0.952 at 3012, ×0.930 at 16K; mf spills (LDL 409-643), +54 µs | registers | 04, 10 | 10-24, 10-31 | MO/logs/moe_v22.log; RC/mf/NOTES.md |
| Triton / DSMEM / sum-of-squares finish variants | ×0.949-0.993 | | 04 | 10-24 | MO/logs/moe_v17.log, moe_v20.log, moe_fin2.log |
| L2 prefetch-ahead (MOE_PD 2/4/8); small-tile prefetch (v26) | 753 ×0.953 / 0.948 / 0.927; −6..−10 % | prefetch-issue cost; the kernel is not latency-bound | 05 | 10-25 | MO/logs/moe_real_pd.log; MEM:39 |
| 128-k stages × 4 slots (v27); XPU = 2; all-wgmma (v31); MOE_A3 | −3.5..−6.7 %; −3 %; −1..−2 %; −11 % | per-stage overhead | 05 | 10-25 | MO/logs/moe_real_v27.log; MEM:39, 43 |
| row-block-major tile groups (MOE_GROUP) | 16K DRAM reads 4.85 → 2.46 GB, time flat / −1 % | the expert kernel is not DRAM-bound | 05 | 10-25 | MEM:60 |
| tiny lane: a 4th warpgroup running z ≤ 8 tiles beside the main lane (MOE_TINY, v34) | −1.1..−16 % at every threshold | its mma.sync waits on the tensor pipe the main wgmma saturates | 10 | 10-25..26 | MEM:72 |
| public expert kernel at T = 753 | ×1.005 / ×0.963 | the tiny-expert tail dominates | 06 (2048 threshold) | 10-29 | R/moe_public_bound.log |
| native tiny-WGMMA groups; deq6 decode; concurrent hot/tiny partition | −8..−18 %; +0.5..+1 % at 753 but −0.6..−0.9 % above; 339.6 vs 321.8 µs | ptxas C7512 serialization | 06 | 10-29 | R/moe/NOTES.md |
| fresh BF16 weight expansion + cuBLAS | 5428 vs 4411 µs | the dense GEMM alone already loses | 06 | 10-29 | R/moe/NOTES.md |
| SS-wgmma decoder-warp kernel (WG192, later moe3) | 745-885 vs 300-311 µs; 1.75-2.6× slower at N ≤ 128 | the 128 B/clk smem port binds; FP4-through-RS is smem-optimal on sm_90 | 07, 09 | 10-29, 10-30 | R/RUN_STATE.md; RC/moe3/NOTES.md |
| full mixed-queue interleave | +15..+32 µs | CTA finish imbalance, delayed down-projection dependencies | 07 (gate-only queue) | 10-29 | R/queue_item_trace_summary.json |
| co-resident tiny warpgroup / riders in the single kernel / split hot+tiny launch | 258 µs per tiny item; +10.8-12.2 µs per host; +30-50 µs | consumers need 224-232 registers; tensor-pipe queueing | 09 → 10 | 10-30 | RC/moe/NOTES.md |
| 256-row items, 2-CTA multicast of x, 5 ring slots, longer prefetch, L2 hints | +27 / +31 µs (C7512); +17.5 / +42 µs; +1.6..+13.8 µs; ±0.3 % | **root cause found here: a running wgmma halves the int/fp16 pipes**, so the NVFP4 decode is not hidden | 09, 10 | 10-30 | RC/moe2/NOTES.md |
| concurrent wide and narrow kernels (moe4) | 375-555 vs 264-319 µs | narrow items are decode-issue bound in any kernel | 09 | 10-30 | RC/moe4/NOTES.md |
| gather folded into routing at 753/3012 | 0.9900 / 0.9926 replica | remainder tiles need width-dependent layouts | 09 | 10-30 | RC/NOTES.md:83-85 |
| staggered warpgroup epilogues at 16K | −1.5..−2.6 % | one WG alone drives n256 at ~65 % of the pipe | 09 | 10-30 | RC/moe16/NOTES.md |
| lookup-table FP4 decode | +10..+19 % cycles per k16 | a lane-replicated table: one 128-B wavefront per lookup | 10 | 10-31 | RC/lut/NOTES.md |
| scale applied after the MMA for narrow items | N=16 +24 %, N=32 +43 % | the narrow loop is issue-bound | 10 | 10-31 | RC/ms/NOTES.md |
| 3-4 consumer warpgroups for narrow items | +2..+7 % per item at N ≥ 32 | pipe throttle, not warp starvation | 10 | 10-31 | RC/mn/NOTES.md |
| mma.sync riders; single-kernel wgmma riders | +11.4 µs per rider; −6.4 / −2.9 % | break wgmma pipelining; the register limit is function-wide | 10 (split into W and N) | 10-31 | RC/mx/NOTES.md |
| W and N kernels running concurrently | −1.8..−3.1 % | idle time at the end | 10 | 10-31 | RC/NOTES.md:245-247 |
| wide-only kernel W alone | parity | the split pays only with riders | 10 | 10-31 | RC/ww/NOTES.md |

## Dense GEMMs, boundaries, glue

| idea | number | why | informed | found | source |
|---|---|---|---|---|---|
| own stream-K / split-K BF16 GEMMs; 2-CTA multicast; ping-pong | 0.53-0.88×; −1.5..−4 % | cuBLASLt mainloops already at 96-99 % of the power-capped wgmma rate | 05 | 10-25 | MO/logs/gemm2_notes.md |
| L2 weight prefetch before the PDL wait | o_proj 27.4 vs 21.2 µs | floods DRAM | 05 | 10-25 | MO/logs/gemm2_notes.md |
| a memory "lane" on the same SM beside a GEMM; megakernel | GEMM +10..+23 % | the GEMM uses ~95 % of the L1/smem port | 05 | 10-25..26 | MO/logs/mlane_probe.log; MEM:75, 78 |
| fused unpool + boundary (ub2) at 753/3012 | ×0.865 / ×0.858; later replica 0.9929 / 0.9956 / 0.9987 / 0.9967 | few waves; our GEMM 3-6 % behind cuBLAS | 05, 10 | 10-25, 10-31 | MO/logs/ub2_v7.log; RC/bf/NOTES.md |
| kv_pack before the QKV GEMM | −1.2..−1.5 % | its blocks take the SMs first | 05 | 10-25 | MO/logs/bench_full26.log |
| FFN token chunking / request pipelining / 16K chunk pipeline | −2..−6 %; −11..−15 %; −1 % | lost batch efficiency and SM competition | 06, 07 | 10-29 | R/RUN_STATE.md |
| pooler / router cuBLAS overlap | 1552 → 1569-1795 µs | | 07 | 10-29 | R/RUN_STATE.md |
| cuBLASLt configs with a workspace | −0.2..−0.5 % of the layer at 3012 | each adds a memset graph node | 08 | 10-30 | RC/lt/NOTES.md |
| 16K boundary row sums in the epilogue, async bulk stores, ping-pong WGs | UB tiles +1.2-1.5 µs each; fused ping-pong +3..+47 µs | per-SM write port ~63 GB/s; joint items cost ~6 µs each | 08 | 10-30 | RC/bnd/NOTES.md |
| removing inter-kernel gaps | case 0 idle only 2.2-5.4 µs (≤ 0.22 %) | a cross-stream join costs ~2 µs wherever it sits | 09 | 10-30 | RC/gap/NOTES.md |
| custom GEMMs at 753 (DG, CUTLASS, per-shape tournament) | no route beats the current plan at any scored shape | cuBLASLt at 96-99 % | 09, 10 | 10-30, 10-31 | RC/gemm/NOTES.md; RC/g7/NOTES.md |
| L2 prefetch of the next GEMM's weights from the preceding kernel | turn 1 +0.14 % (1.5σ), turn 4 −0.10 % | the host kernel pays as much as it saves | 10 | 10-31 | RC/stk/NOTES.md |
| bnd3 (753) rewrites toward the HBM peak | best −0.40..−0.58 µs; in-layer headroom to the floor 2.7 µs (b1) / 1.5 µs (b2) | latency-bound per-row chain | 10 | 10-31 | RC/b3/NOTES.md |

## Process

| idea | number | why | informed | found | source |
|---|---|---|---|---|---|
| absolute local latency as the metric ("197.4 ms local") | never comparable with the judge's score | the judge scores a paired speedup over loads with a fresh router per run | 08 (replica.py) | 10-30 | RC/NOTES.md:3-13; `.../memory/prefill-judge-metric.md` |
| tuning on one fixed router seed | b23 case 5: 2.054 on the fixed seed vs 1.935 / 2.013 on judge-style draws | overfit to one routing | 08 | 10-30 | R/judge/router_seed_test.log |
| 3-candidate bench order [SGLang, c1, c2] / reversed | 0.4-1.1 % bias for the middle candidate | the last of three back-to-back long graphs runs slow | 08 | 10-30 | RC/NOTES.md:59-63; RC/bnd/NOTES.md |
| fixed-order bench at power-capped 16K | ~2-3 % bias | 16K shapes run at the power cap | 05 | 10-25 | MEM:52 |
