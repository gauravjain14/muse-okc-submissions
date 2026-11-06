# Dense GEMM worker (research/claude/gemm), 2026-10-30

## DELIVERABLE: research/claude/gemm/gemm_c2.py (862,185 bytes) = s1.py + DG overlay (build.py)
Rebuild: `python3 research/claude/gemm/build.py gemm_c2.py --dense '753,2304,8192=2,1,0,112,112,6,0,1,,,0,1,96'
--post3 753 --oslide '753=66,33,1,0,0,1,35,0,2'`. Changes only T=753 sliding layers (case 0 and nearby T=753 sliding):
1. QKV projection on a DG kernel (128x112 tiles, 126 CTAs, launch bound capping the CTA at 96 regs/thread, TRIG 1 =
   launch_dependents at start, stagger 1) so that qkv_post3 (qkv_post2 with the projection-independent loads hoisted
   above griddepcontrol.wait) sits resident beside it and starts the moment the projection completes.
2. o_proj on sliding layers: cuBLASLt nvjet 128x96 ssched, workspace 0 (attn_fwd does not trigger PDL there, so sk2's
   overlap was unused). Full layers keep sk2.
Numerics: QKV GEMM FP32 accumulation, BF16 out (same as cuBLAS up to accumulation order); qkv_post3 = qkv_post2 math.
Deterministic (no atomics). If the DG module cannot get its register allocation, prepare() keeps s1's paths.
- Gate: CHECK_RC=0 PASS=42 CASES=21 (fin.log).
- Replica, all 8 cases, panel 11,22, 6 reps (fin.log): RELATIVE 1.0095 / 1.0129 (mean 1.0112) -- but that headline is
  first-turn (16K) noise: the 16K paths are byte-identical to s1. Per load: turn 1 2.0239->2.0228, 1.9803->1.9851;
  turn 4 (untouched) 1.9163->1.9145, 1.9118->1.9113.
- What it really does (case 0, us, base -> new, every replica run): 543.2->541.3, 545.9->543.5 (r1), 534.5->533.1,
  511.4->511.4 (r2), 537.1->536.6, 542.7->540.7 (fin): mean -1.4 us (-0.26% of case 0, ~+0.06% of score). Case 1:
  unchanged within noise (+0.8, +0.3, -6.1, +1.2, +1.9, 0.0). In-layer node traces: bnd3_b1 starts 1.8-2.6 us earlier.
- ncu (isolated kernels, case 0, n1.log): DG QKV 40.8 us TC 80.8% (nvjet 39.8 / 82.1%); o_proj nvjet ws0 19.9 us TC 71.3%
  (sk2 22.0 / 62.7%). qkv_post3's gain is its co-residency, which ncu's serialized replay does not show.

Goal: beat s1's dense GEMM plans at T=753 (case 0/1) and T=3012 (case 2) in-layer.

## Files
- `dg_src.py`: DG kernel family (CUDA source + DgCfg). TMA + wgmma, persistent data-parallel tiles, NWG consumer WGs
  (M-split like sk2, or SPLITN: WGs split the tile's columns so a 64-row tile can be ~400 wide), overlapped last
  n-tile (stores only owned columns), OUT 0 BF16 / 1 FP32 / 2 mixed (columns < nsplit BF16, rest FP32: the fused
  pool+router GEMM writes pooled BF16 + router hi|lo FP32), PDL (npre W stages before griddepcontrol.wait, TRIG),
  stagger (wait for stage 0 before issuing the rest of the ring), DG_TRACE (per-CTA globaltimer/clock64 trace).
- `dg_host.py`, `probe.py`: standalone cold-L2 CUDA-graph probe vs s1's exact plan for a shape (paired, + trace).
- `build.py`: single-file candidate = s1.py + DG overlay; routes keyed by exact (M, N, K) / token count; the fused
  route switches pnr5 -> pnr2y (pooled BF16 + FP32 logits, the 16K path's router kernel).
- `lcompile.py`: local nvcc 12.1 sm_90a compile check (registers, spills, C751x serialization warnings).
- `t1.sh` (+ research/claude/gap/trace.sh): in-layer nsys node trace.

## Standalone results (random data, cold L2, CUDA graph, paired vs s1's plan; times include ~4-5 us graph overhead)
T=753 (p1-p3 logs):
| shape | s1 plan | best DG | DG - s1 | DG timeline (fill / mainloop / epilogue, clock) |
|---|---|---|---|---|
| fused f (N 4608, K 8192) | cuBLASLt 144x192, 128 CTAs: 78.8 | 192x144 3WG mixed-out, 128 CTAs: 78.7-79.1 | -0.3..+0.2 | 1.6-2.2 / 69.2 / 1.6 (BF16) vs 3.4 (FP32), 1.60 GHz |
| unpool u (N 8192, K 4096) | cuBLASLt nn 192x128 1x2: 70.5 | 128x192 2WG 132 CTAs: 71.7 | +1.4 | 1.9 / 30.9+31.4 / 1.4+1.4, 1.61 GHz |
| qkv q (N 2304, K 8192) | SGLang dispatch 112x128: 43.1-43.3 | 128x112 2WG 126 CTAs: 44.0 | +0.6..+0.9 | 2.0 / 35.8 / 0.9, 1.64 GHz |
| o_proj o (N 8192, K 1024) | sk2 3WG 192x128 epi2: 24.9-25.1 | 192x128 3WG direct epi: 25.1-25.4 | +0.2 | 1.9 / 7.7+7.5 / 1.4+1.0, 1.68 GHz |

Key facts:
- Mainloops are at 97-99.5% of the wgmma rate at the measured clock (e.g. fused 192x144: 110.6K clk / 1.601 GHz =
  69.1 us expected, 69.2-69.4 measured). The clock is power-limited at 1.58-1.70 GHz and FALLS when more SMs are
  busy: QKV 126 CTAs 1.638 GHz vs 132 CTAs 1.584 GHz; fused 128 CTAs 1.606 vs 132 CTAs 1.592. A 132-SM tiling that
  cuts per-CTA work by 3.6% (QKV 64x216) gains ~0.3% net. Wave quantization is not the lever: time ~ energy / power.
- SPLITN (64-row tiles, WGs split columns): L2->SM traffic per FLOP +45-70% (fused 64x424: 73 B/clk/SM): mainloop
  78.6 vs 69.2 us (fabric/latency bound). QKV 64x216: 2 inlined consumer copies -> ptxas C7516 serialization;
  one copy with a uniform width branch -> no warning but mainloop 49.8 vs 35.8 us. Dead end.
- 3 ring stages are too few for 192x144 (mainloop 82 vs 69 us); 4-5 needed.
- Fill (CTA start -> first stage landed) 1.8-2.8 us = the whole initial ring requested at once; stagger (stage 0
  first) cuts it to 1.4-1.6 us but end-to-end within noise.
- BF16 pooled epilogue (fused) 1.6 us vs FP32 3.4 us, but DG's fill+mainloop is ~1.5 us behind nvjet's -> parity.

## In-layer (nsys node trace, fields seed 11, 8 reps; t1.log)
Fused GEMM -> DG 192x144 mixed out + pnr2y (g1a: PDL npre1 stagger; g1b: PDL npre 4; g1c: no PDL, stagger):
| | bnd1 end -> MoE start, case 0 | case 1 |
|---|---|---|
| s1 (nvjet + pnr5) | 86.2 us | 86.4 |
| g1a | 86.0 | 85.6 |
| g1b | 86.0 | 85.4 |
| g1c | 85.6 | 84.3 |
- pnr2y (reads BF16 pooled, 7.7 MB) = pnr5 (reads FP32, 14 MB): 5.4-5.6 us both: latency-bound at 753.
- PDL launch behind bnd3_b1 does not help (the GEMM CTA cannot co-reside with bnd3's 2x512-thread CTAs).
- Net <= ~1 us: within noise. Not worth a candidate on its own.

## T=3012 standalone (p4.log): DG loses
fused 305 vs 290 (cuBLASLt nn 128x168), unpool 270 vs 257, qkv 203 vs 148 (round-robin m-fastest raster thrashes L2
with 24 m-tiles; cuBLAS uses swizzled rasters + overlapped epilogues). Not pursued: DG is at parity at best at 753.

## QKV + qkv_post2 (case 0, sliding, T=753): the one structural lever found
s1 case 0: QKV 0 -> 39.4, qkv_post2 39.7 -> 46.2 (latency-bound, 6.2-6.6 us), attn_fwd starts 48.2 (join gap ~2 us).
(a) co-resident qkv_post3 (q1/q2/c2): DG QKV 128x112 (126 CTAs) with a launch bound that caps the CTA at 96 regs/thread
    (registers are allocated per SM sub-partition: lbt = 4*32*w threads -> 16384/(32 w) regs), TRIG 1 (launch_dependents at
    start), stagger 1; qkv_post3 = qkv_post2 with every load that does not depend on the projection (seq_lens, cache_loc,
    scales, RoPE table) hoisted above griddepcontrol.wait (48 regs x 256 threads: fits beside the GEMM CTA). qkv_post3's
    CTAs sit resident from t=1.4 us and finish 4.4-4.7 us after the GEMM (vs 6.5-6.8 for qkv_post2 after nvjet).
    attn_fwd start: -2.2 us (t2), -0.5 (t11), -0.7 (t13): noisy, ~-1 us.
(b) fused epilogue (OUT 3, qp*/c3): qkv_post2's whole job in the QKV GEMM epilogue (rbf of the accumulators, per-head RMS
    with the partial sums of heads cut by a 112-column tile edge swapped with the neighbour tile through {epoch, value}
    64-bit words, RoPE, FP8, staged in the drained ring, 16-byte stores to q8 / cache rows / packed kp, byte scatter vp).
    Bit-identical to s1 (1 FP8 byte of 771072 differs: RMS summation order). But the epilogue costs 5-8 us (max CTA):
    first version 18 us (I-cache: 28 inlined sincosf + per-element branches, 18K SASS), then loads serialized behind
    possibly-aliasing stores (rope loads after STS/STG), then divergence barriers (BSSY per class branch). Final: epilogue
    median 5.1 / max 7.8 us, kernel ends 47.8 vs qkv_post2's 46.2 -> attn_fwd +2.5 us later than s1. Rejected.

## o_proj on sliding layers (c1): cuBLASLt instead of sk2
attn_fwd does not trigger PDL on sliding layers (grid 128 < 132), so sk2's PDL overlap is unused at case 0; the lt
worker's best nvjet ws0 config (tn [66,33,1,0,0,1,35,0,2], 128x96 ssched, 132 CTAs, no workspace memset) is ~1.4 us
faster in the layer (case 0: bnd3_b1 starts 89.1 vs 90.4). Full layers (case 1) keep sk2 (attn_fwd triggers there).
c2 = c1 + (a): case 0 bnd3_b1 starts 87.8 vs 90.4 in s1 (-2.6 us), t13.

## Judge replica (run_exp.sh, AB/BA pairs per candidate vs SGLang, 8 reps, turn-1 cases 0,1 only)
| draws | c1 (o_proj sliding -> cuBLASLt) | c2 (c1 + DG QKV r96 TRIG1 + qkv_post3) | c4 (c2 + DG fused mixed-out + pnr2y) |
|---|---|---|---|
| 11, 22 (r1.log) | 1.0030, 0.9990 | 1.0027, 1.0038 | - |
| 33, 44 (r2.log) | - | 1.0036, 1.0017 | 1.0017, 1.0043 |
c2: +0.30% on the turn-1 load over 4 draws (case 0 -1.4..-2.4 us; case 1 = s1 within noise), i.e. ~+0.1% of score.
c4 adds nothing measurable over c2 (and changes pooled numerics to BF16): dropped.

## Attention trigger alternative (t14): attn_fwd triggers PDL also for grid >= nsm - 8 (T=753 sliding)
c6 = c2's QKV path + sk2 o_proj launched early on the 4 idle SMs: case 0 bnd3_b1 starts 87.2 (c2 87.7, s1 89.5) = same
as cuBLASLt o_proj; c5 (+ sk2 L2 prefetch of all 16 W k-blocks) slows attn_fwd (22.0 vs 19.7 us): worse; and its sk2
route override hit case 1 too (+4 us). Kept c2 (no attention-kernel change).

## Why the T=753 GEMMs stay at 82-87% tensor-active (root's ncu table) and what is left
- Mainloops run at 97-99.5% of the wgmma rate at the measured clock (fused 192x144: 69.2 us vs 69.1 ideal at 1.60 GHz).
  The ncu TC% gap is the fixed per-kernel cost: fill (first ring from cold DRAM / the whole ring requested at once,
  1.6-2.8 us), epilogue (0.9 us QKV BF16 .. 3.4 us fused FP32), end spread (~1 us) and idle SMs (126 / 128 of 132).
  At T=3012 the same kernels reach 93-95% because those costs are 4x smaller relative to the mainloop.
- Idle SMs are not free: the clock is power-limited and falls when more SMs are busy (QKV 126 CTAs 1.638 GHz,
  132 CTAs 1.584 GHz; fused 128 -> 132: 1.606 -> 1.592), so a 132-CTA tiling with 3.6% less work per CTA gains ~0.3%.
  Finer 132-way tilings need 64-row tiles, which raise L2->SM traffic 45-70% and become fabric/latency bound.
- Stream-K / split-K: partial tiles are 48-110 KB of FP32 per hand-off (1.5-3.4K clk at the 32 B/clk write port),
  i.e. 3% (K 8192) to 25% (K 1024) of a CTA's work, more than the 1-3% balance they buy (Codex measured the same).
- Fill and epilogue can only be hidden by overlapping a neighbour kernel. bnd3_b1/b2 (2 x 512 threads x 64 regs = the
  whole register file per SM) and the GEMM CTAs cannot co-reside; their DRAM streams would compete with any W prefetch
  anyway (bnd1/bnd2/combine are DRAM-bound). The two latency-bound neighbours were the levers: qkv_post2 (done, via
  co-residency) and the o_proj's fill under attention (cuBLASLt ws0 or an attention trigger: ~-1.4 us).

## Not done (ideas with numbers, for whoever continues)
- Case 0 cross-stream join (~2 us between qkv_post and attn_fwd): with the DG QKV capped at 96 regs, kv_pack (part 1,
  44 regs x 256 threads, 8.7 KB smem) also fits beside the GEMM CTA (+ qkv_post3: 15.1K of 16.4K regs per SM
  sub-partition). A main-stream PDL chain DG QKV (TRIG 1) -> kv_pack (launch_dependents at start, griddepcontrol.wait
  at its end, so its completion implies the projection's) -> qkv_post3 -> attn_fwd would drop the side stream and its
  join: est. -1.5..-2 us at case 0. Needs a kv_pack variant (attention worker's kernel; attn1 also edits kv_pack).
- attn_fwd PDL trigger on sliding grids >= nsm - 8 (c6, one-line ATT change) is worth the same as the cuBLASLt o_proj
  (-1.4 us); not additive with it.
- Fused pool+router with BF16 pooled output + pnr2y (g1c / c4): -0.6..-2 us in traces, nothing in the replica.
- Everything else at T=753 (wave quantization, stream-K, 64-row split-N tiles, multicast, stagger) is within noise or
  worse; see the tables above.
