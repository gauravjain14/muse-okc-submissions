# Router workstream (worker R) — notes

All times: claude/bench.py style (CUDA graph replay, cold L2 via 512 MB read, fixed graph cost subtracted), local us,
H200 on Modal. NOTE: different Modal containers differ by up to ~25% on these small kernels; only compare numbers taken
in the same run.

## Default dispatch (router.py, DISPATCH=True)
- M <= 4: `rs<M>` — no split-K: CTA = 2 rows x full K (128 CTAs), exact FP32 weights (SIMT FMA), W 64 KB per CTA in
  4 bulk copies, x via 3D TMA box [256 k, 8 k_outer, M rows] (no swizzle). Sits at the 8.4 MB streaming floor.
- 5 <= M <= 64: `rt16/rt32/rt64_f16` — mma.sync m16n8k16 (x = A, W hi/lo = n8 B), CTA tile (TT tokens x RR rows x KK k),
  128 adder CTAs + 4 "zero CTAs" (blockIdx 0..3, on the 4 otherwise idle SMs) in ONE kernel: zero CTAs zero `out`,
  then red.release a counter; adder CTAs red.add.v4.f32 their split-K partials into `out` after an acquire-poll of that
  counter. A private per-CTA generation word (zc[64 + b], one 4096-int block per (kernel, grid) key) makes the
  protocol call-independent with no reset and no grid barrier.
- M > 64: `rw64` (M <= 128) / `rw128b` (M > 128) — wgmma m64nNk16 (A = W hi/lo 64x64 128B-swizzled tiles prelaid in
  global, B = x via 2D TMA 128B swizzle, N = 64/128 tokens), 64 rows per CTA, K split 16, epilogue transposes through
  smem and red.adds into `out`, which a tiny `rzero` kernel zeroes first; the main kernel is a PDL launch
  (programmatic stream serialization) and executes griddepcontrol.wait before its reds. Correct for M up to 1000.

Validated (reference.check_implementation, normal + outliers, 3 seeds) for M in
{1,2,5,7,8,9,15,16,17,31,32,33,64,127,128,129,130,255,256,257,512,1000}; rel L2 ~1.4e-7 (rs) / ~2.5e-6 (hi+lo paths).

## Measured facts
- Global split-K fixup (red.add to workspace + threadfence + atomic counter + last-CTA copy/zero): +1.4 us at small M,
  +2.8 us at M=128, +4 us at M=256 over no reduction. Dead.
- Graph node costs: memset node +1.1 us, plain second kernel +0.9 us; rzero + PDL main kernel ~+0.65-0.85 us over
  "no reduction"; red.add itself is ~free at small M (red-only 2.75 vs no-reduction 2.70 us at M=8).
- In-kernel zero CTAs (single kernel): ~0.8 us better than PDL at M=8..64. Residual cost: the zero CTAs' release lands
  ~0.5-0.76 us after the adders reach their epilogue (stores/fence queued behind the read flood). At M=128/256 (128-256 KB
  to zero) it is no better than PDL, so rw keeps PDL.
- Clusters of 4 or 8 only ever occupy 120 distinct SMs (even 128 CTAs at 60 KB smem land on 120 SMs). Cluster 2: all 132.
- Cluster barrier ~0.5 us each (globaltimer trace, 0.26 us resolution). rc (cluster-8 DSMEM, 2 barriers) 4.3 us at M=1.
- TMA multicast of W across cluster pairs at M=256: worse (10.2 vs 8.7 us) — cluster start barrier + coupling.
- First weight bytes land ~1.5-1.8 us after kernel start (cold, after the flush).
- L2-hit bulk ingress: 256 KB per CTA in 2.5 us (~105-140 GB/s per SM, 13.4 TB/s total); HBM 256 KB/CTA 7.8 us (4.3 TB/s).
- No-split designs at M=8 (rs SIMT, rq mma) are limited by x re-reads (128 KB per CTA, 16 MB through L2): ~3.9-4.4 us.
- Lossless 24-bit hi/lo compression for rt (hi: sign+mantissa byte + 4-bit exponent code; lo: byte + 4-bit code of
  e_hi - 8 - e_lo; 172 escapes applied exactly from the x tiles in smem) is implemented and exact (FLAGS=48), but gives
  no gain: at M <= 64 the weight stream is latency-bound (last stage lands at the same time as uncompressed) and the
  extra prologue/correction work costs 0.5-3 us. Disabled.
- Compressed wgmma path (`rw64c/rw128bc`, RWC=True): W hi/lo 24-bit codes decoded into register A fragments,
  wgmma with A from registers, per-stage escape corrections from the smem x tile. Exact, but 10.1 / 12.9 us
  (vs 6.5 / 8.4) unpipelined and 12.3 / 16.2 us with double-buffered A registers: one consumer warpgroup = one warp per
  SMSP, so the dependent decode chain cannot hide behind the tensor pipe. Would need 2 consumer warpgroups. Disabled.
- Delaying the rt producers so the zero CTAs release first: +0.2 us per 200 ns of delay (the stream is the critical path).
- 3D x TMA box per stage in rt (1 op instead of 4): no change.
- M=256 rw128b timeline: first data 1.8, last stage 5.6, mma done 6.1, reds done 7.2-7.7 us.
- rw config sweep (M=128: rw64 / rw64n8 / rw128a; M=256: rw128b / rw128b7 / rw256): all within noise except rw256
  (8.4 MB of reds) which is worse.

## Integration
router.py is self-contained (torch + cuda.bindings NVRTC; compiles one module, ~a few s). API:
`prepare_router(weights) -> st` (computes hi/lo, all layouts, sync blocks) and `run_router(x, st) -> out [M,256] fp32`.
A call issues 1 kernel (M <= 64) or 2 kernels (rzero + PDL main, M > 64) on torch's current stream — graph-capturable.
x must be contiguous bf16 [M, 8192] (TMA descriptors encode its pointer at call time; capture-safe since graph inputs are
static). `st["zc"]` holds cross-call sync state (generation counters); it must persist across calls (it lives in st).

## Round 2 (coordinator targets judge m8/16/32 <= 4.2, m128 <= 5.5, m256 <= 6.5)
Same-run A/B (r28, bench.py, local us; r1 = round-1 router, r2 = current): m1 2.80/2.88, m8 4.09/4.34, m16 4.28/4.41,
m32 4.81/4.71, m128 6.88/6.87, m256 9.04/8.93 (r2/r1). Only change in effect: 5 <= M <= 8 uses FLAGS=272.
- FLAGS=272 (rt, M <= 8): per-warp epilogue — each consumer warp acquire-polls the zero counter itself and red.adds its
  mma fragments directly (no smem cross-warp reduction, no __syncthreads); zero-gen word read at kernel start.
  m8 -0.25..-0.3 us; m16 equal; m32 +1, m64 +4 us (8x the reds) -> M <= 8 only.
- cuBLASLt (claude/router/lt_router.py: exhaustive algo x tile x stages x cluster x split-K search, FP32 C/D,
  hi GEMM beta=0 then lo GEMM beta=1, exact): best pair 11.6 us (m128) / 12.7 us (m256), single GEMM 5.1-5.5 us,
  vs ours 6.5 / 8.3 in the same process. Dead.
- rw without the rzero kernel (`rw64za/rw128bza`, RWZA=True): a 6th warp in every adder CTA zeroes its 1/S slice of its
  own output tile at kernel start and releases a per-tile counter; adders acquire-wait on it before their reds.
  Correct for M up to 1000, but 6.62 vs 6.51 (m128) and 8.62 vs 8.26 (m256): the zeroing release lands with the end of
  the read stream (same effect as the zero-CTA wait), so the cost only moves. The rzero kernel's 1 us in the judge
  profile overlaps the PDL main kernel.
- Producer start delays (to let zero-CTA stores/release clear first) add exactly the delay: the zero release is ordered
  behind the read stream in the memory system, not by issue time.
Why the targets are not met: at M = 8..64 every split-K reduction (zero CTAs + flag, rzero + PDL, global fixup, cluster
DSMEM barriers) costs ~0.7-1.3 us after the stream ends, and no-split designs pay x re-reads (128 KB per SM at M=8,
>= 256 KB at M >= 16). At M >= 128 the stream phase is bound by per-CTA operand ingress (192-256 KB/CTA at ~70 GB/s
effective) plus ~1.5 us of tail (last wgmma stage, transpose, 2-4 MB of reds).
