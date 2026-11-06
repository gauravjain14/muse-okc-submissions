<!-- Excerpt, copied verbatim: lines 1-146 of muse-opt-codex/muse_bf16_proj/claude/t3/ROOFLINE.md (written 2026-11-02 01:00-07:55 PT). The full file continues past this milestone; it is kept as (private notes, not included). -->
# muse_bf16_proj — roofline, gap table, plan (top-3 push, 2026-11-02)

Board 2026-11-02 08:50 UTC: #1 1.6596, #2 1.6523, #3 1.6498 (screened 1.629), #10 1.62. Our best (okc check, int_v7) 1.3999.
Target: judge >= 1.70 (margin for screening/reruns). Needed: x1.21 over int_v7 => sum over 6 loads of ln(gain) >= 1.15.

## Model (claude/t3/roofline.py; all measured on the Modal H200, cold L2, CUDA-graph replay)
- Local bench (claude/bench.py) -> judge: judge = local + 1.0 us per shape (calibrated: v2 1.247/1.2765, v7 1.3999/1.4037).
- Cold streaming: local t = 0.81 us + B / 4.56 TB/s (least squares over 6.3..67 MB probes). Judge floor = 1.81 + B/4.56.
- Tensor: 132 SM x 4096 FLOP/clk x 1.755 GHz (clock under TC load) = 949 TF; assume 85% achievable.
- Weight entropy (measured, all 4 BF16 layers): 10.46 bit/weight (exponent 2.54 bit, sign 1, mantissa 7 incompressible).
  12-bit format (low byte + 4-bit s|exp>>1 code; 16-exponent window, escapes 7e-5) = 0.75x. 11-bit (8-exp window) escapes 1.3%.
  Router hi: 10.46 bit, lo: 10.98 bit (pair 21.4 of 32).

## Ceilings (judge geomean, every shape at its floor)
| weights at floor | geomean |
|---|---|
| BF16 | 1.551 |
| 12-bit | **1.865** |
| 11-bit | 1.968 |
| entropy (10.46 b) | 2.021 |
| int_v7 now | 1.404 |

=> the gap is kernel efficiency, not the format. 12-bit at floor beats #1 by 12%.

## Per-shape gap table, judge us (int_v7 = local bench + 1.0; floor12 = 12-bit floor)
| load | layer | m | floor12 | sglang | v7 | gap |
|---|---|---|---|---|---|---|
| decode 1 | qkv | 1 | 8.02 | 14.41 | 8.83 | +0.81 |
| decode 1 | o | 1 | 4.57 | 6.68 | 5.51 | +0.94 |
| decode 1 | pooler | 1 | 12.85 | 21.00 | 13.67 | +0.82 |
| decode 1 | unp(slice) | 1 | 3.19 | 7.14 | 3.58 | +0.39 |
| decode 1 | router | 1 | 3.19 | 10.39 | 3.93 | +0.73 |
| decode 16 | qkv | 16 | 8.09 | 14.46 | 11.70 | **+3.61** |
| decode 16 | o | 16 | 4.63 | 6.65 | 6.29 | +1.66 |
| decode 16 | pooler | 16 | 12.93 | 21.48 | 16.79 | **+3.86** |
| decode 16 | unpooler | 16 | 12.93 | 18.60 | 16.09 | **+3.16** |
| decode 16 | router | 16 | 3.25 | 15.26 | 5.36 | +2.11 |
| decode 32 | qkv | 32 | 8.16 | 14.79 | 13.01 | **+4.85** |
| decode 32 | o | 32 | 4.70 | 6.86 | 6.14 | +1.44 |
| decode 32 | pooler | 32 | 13.02 | 21.93 | 18.27 | **+5.26** |
| decode 32 | unpooler | 32 | 13.02 | 18.47 | 18.08 | **+5.06** |
| decode 32 | router | 32 | 3.31 | 15.67 | 5.75 | +2.44 |
| verify 1 | qkv | 8 | 8.05 | 14.66 | 11.52 | **+3.47** |
| verify 1 | o | 8 | 4.60 | 6.65 | 5.33 | +0.73 |
| verify 1 | pooler | 8 | 12.89 | 21.37 | 16.09 | **+3.20** |
| verify 1 | unp(slice) | 8 | 3.21 | 7.42 | 4.94 | +1.74 |
| verify 1 | router | 8 | 3.22 | 26.44 | 5.16 | +1.94 |
| verify 16 | qkv | 128 | 8.61 | 17.38 | 15.75 | **+7.14** |
| verify 16 | o | 128 | 5.08 | 7.43 | 7.41 | +2.33 |
| verify 16 | pooler | 128 | 13.54 | 25.94 | 20.68 | **+7.15** |
| verify 16 | unpooler | 128 | 13.54 | 19.91 | 18.63 | **+5.10** |
| verify 16 | router | 128 | 3.68 | 16.70 | 7.74 | **+4.07** |
| verify 32 | qkv | 256 | 13.79 (mma) | 19.61 | 17.45 | +3.66 |
| verify 32 | o | 256 | 7.13 (mma) | 9.38 | 9.18 | +2.04 |
| verify 32 | pooler | 256 | 23.11 (mma) | 29.78 | 24.14 | +1.03 |
| verify 32 | unpooler | 256 | 23.11 (mma) | 24.92 | 24.25 | +1.15 |
| verify 32 | router | 256 | 4.16 | 17.54 | 9.75 | **+5.58** |

Value of 1 us saved (geomean %): decode1 0.47, verify1 0.39, decode16 0.30, decode32 0.27, verify16 0.24, verify32 0.20.

## What bounds the current kernels (nsys in-graph + ncu full set, claude/t3/nsys_v7.log, ncu_v7.log)
- GV3 (m=1): 2048 CTAs x 512 thr, 30 regs, 88% occupancy, 2.4 eligible warps/clk, issue 58% — near floor.
- K1s/K1t (m=8..32 qkv/pooler/unpooler): 1 CTA x 16 warps/SM, 122-128 regs, 25% occupancy, 0.50-0.53 eligible
  warps/clk, issue 32-35%, stalls long_scoreboard 24-43% + wait 17-23% + barrier 8-11%; 2.3-2.9 TB/s under ncu.
  => latency-bound: too few bytes in flight per SM (register streaming D=3) + serial prologue (x TMA) and split-K
  epilogue that nothing overlaps.
- cuBLASLt m=128 pooler (nvjet 128x32): 14% occupancy, issue 24%, long_scoreboard 60%; BF16 bytes (69 MB).
- Router rw64 (m=128): 1.27 TB/s, 7.8% occupancy, issue 15%; only 64 of 132 SMs stream; + separate rzero kernel.
  rt16 (m=16): 1.38 TB/s, issue 11%, long_scoreboard 50%.
- MoE sibling kernel (muse-opt/muse_moe_w4a16, read-only): persistent 132 CTAs, producer warp + 3-4 x 32 KB
  cp.async.bulk ring (weights + activations per stage), 8 consumer warps setmaxnreg 216: within ~4% of its HBM
  ceiling (4.13-4.18 TB/s) at bs8-32. Per 32 KB stage of 12-bit weights (21.8K weights) at m=32 the compute is
  ~256 issue cycles of decode + ~680 cycles of mma.sync per SMSP vs a memory budget of ~1830 cycles: memory-bound.

## Workstreams (agents; one shared GPU via claude/sbx.py)
- S  : m in [2,32] for qkv/o/pooler/unpooler (scored m = 8, 16, 32). Gap 38 us judge across 12 shapes.
- W  : m > 32 (scored 128, 256) for qkv/o/pooler/unpooler: compressed weights decoded into wgmma A registers.
- R  : router, all m. Gap 17 us judge across 6 shapes.
- G  : m = 1 GEMV polish + sub-12-bit fixed-length format (ZipServ TCA-TBE style 3-bit code + fallback, ~11.2 b).

## Launch / streaming probe (claude/t3/launchprobe.py, 2026-11-02, local us after flush, empty-graph subtracted)
- Empty kernel node: 132 CTAs with 0..227 KB dynamic smem or cluster-2: ~0 extra. 1056x128 CTAs +0.64, 2048x512 +1.09 us
  (CTA launch throughput).
- Pure LDG streaming (4 x 16 B in flight per thread, ld.global.nc.L1::no_allocate):
  50 MB: 2048x512 12.00 | 1056x256 12.74 | 264x512 12.91 | 132x512 13.42 | 132x512 + 200 KB dyn smem 14.34
  6 MB : 1056x256 2.11 | 264x512 2.21 | 132x512 2.51 | 132x512 + 200 KB smem 2.64 | 2048x512 4.00
  => LDG streaming loses ~7% when the smem carveout leaves a small L1 (in-flight buffer); very large grids cost
  ~1 us of CTA launch on small shapes. Bulk-copy (TMA) rings are not limited by L1 (measure per design).

## Tighter streaming floors (agent S probe_ring.py, 12-bit bytes, local us, no compute)
| pattern | pooler/unp 50.3 MB | qkv 28.3 | o 12.6 | slice 6.3 |
|---|---|---|---|---|
| hw-scheduled LDG 2048x512 (1056x256 small) | 11.17-11.41 | 6.75-6.82 | 3.31-3.44 | 1.95-2.13 |
| persistent LDG 132x1024 (128 KB/SM in flight) | 12.02 | | 4.00 | 2.54 |
| bulk-copy ring 132 CTAs 6x32 KB | 12.16-12.62 | | 3.97-4.11 | 2.59-2.86 |
=> real floors are ~0.5 us below the fitted model for pooler/qkv; TMA rings cost +1.0-1.3 us (bursty last stages).

## Cross-cutting facts from agents (2026-11-02 ~02:00)
- (R) Kernel-param constant bank is a cold miss after the flush: param-dependent loads wait 0.2-0.45 us. Baking weight
  pointers as NVRTC compile-time constants: 8.4 MB stream 2.70 -> 2.50 us (128 CTAs), 2.80 -> 2.35 (256 CTAs),
  6.3 MB 2.35 -> 2.03. NVRTC trap: cast of a 64-bit constant dropped the high half; use asm mov.u64 "n" constraint.
- (R) Per-SM bulk-copy queue is FIFO; many CTAs reading the same x bytes at once = L2 hot spot (x lands late).
- (G) o at M=1: one-wave tensor-core tile GEMV tc(1,8) 3.76-4.07 vs GV3 4.30-4.54. 11-bit F11f format exact,
  11.45-11.54 bit/weight incl. escape slots (escapes ~1.3%) vs 12.03-12.13 for 12-bit (-4.5% bytes).
- (G, SASS) ptxas load sinking: `__launch_bounds__(NT)` without min-blocks -> low register target -> LDGs sunk below the
  first FFMA/HMMA (serialized DRAM round trips). GV3 o: 6 of 19 LDGs late. Fix (min-blocks / hoisting) at M=1:
  qkv 7.16 (11-bit g311) vs GV3 7.81; pooler 11.71 vs 12.54; o 3.72-3.85 (tc 12-bit) vs 4.40; unpslice 2.48 vs 2.60.
- (G) Baked W pointer + multi-wave kernel: x (param-dependent) queues behind the W flood (+1.2-1.4 us): issue x first.

## M=128/256 L2 model (2026-11-02 03:10)
cuBLASLt unpooler m128: 199 MB L2->SM in 17.06 us => ~12 TB/s effective L2 (bf16 W 67 + x re-reads 132 MB). It is
L2-bound. Our 12-bit at R=64/S=1: 50 + 134 = 185 MB -> >= 15.4 us. Cluster-pair x multicast (R=64, S=1): 117 MB ->
HBM-bound ~12.1 us. Pooler/qkv (K=8192) need R=128 S=4 mc=2 (x 34 MB) and a split-K reduction (measured 2-4 us tail
when partners finish together). W round 1 (claude/t3/W/NOTES.md): RS-wgmma loop 87% with no traffic; delivery-bound
with traffic (W 3.3 TB/s in-kernel); slower than cuBLASLt at all 8 shapes. Round 2: multicast x + LDGSTS W.
- W round 2 (03:40): cluster-pair x multicast and LDGSTS W both slower (unpooler m128 21.2 / o 8.7 vs v7 17.8 / 6.5).
  x traffic is not the binding limit. Use `mbarrier.arrive.shared::cluster.b64` (CUTLASS form) for remote releases;
  `.release.cluster` emits MEMBAR.ALL.GPU.
- Hypothesis (W round 3): RS-wgmma m128 loop is SMEM-bandwidth bound. Per WG step (64 rows x 128 tok x k16 = 64 clk at
  2048 FMA/clk/SM): B read 4 KB + x TMA write 4 KB + W write 1.5 + W LDS 1.5 = 11 KB = 172 B/clk vs 128 B/clk.
  Fix: RG2 (x shared by 2 WGs) + W via LDG straight to registers -> ~96 B/clk.

## Banked builds
- int_t3a (03:25): int_v7 + G gemv1 at M=1 (F11 g311 for qkv/pooler/slice, F12 tc for o). Same run vs v7 (local):
  qkv 7.19/7.85, o 3.91/4.37, pooler 11.70/12.67, slice 2.53/2.65 -> decode-1 load judge-proj 1.806 vs 1.694
  (+1.1% geomean). ncu g311: occ 64%, issue 50%, elig 1.24, long_scoreboard 47-54% (streaming-bound). check ALL OK.
- R (router) final: no design beat int_v7 (reduction >= 0.6-0.8 us after the stream at M 8-32; M=128/256 trades
  ingress vs atomics at equal cost). Router stays int_v7.
- W round 3: smem hypothesis false (smem pipe ~33%); loop is latency-bound (27% of samples in mbarrier waits; smem
  can't hold W+x in flight). Round 4: W in a register ring via LDG, smem only for x.
- okc check int_t3b (job a5086e19abb541ba): judge 1.4168 [1.3955, 1.4380] vs int_v7 1.3999 (+1.2%), projection held.
- S final (hs split-K units + PDL converter): only pooler m=8 wins robustly (14.20 vs 15.01 same run); pooler m16 tie,
  unpooler m16 loses. Main kernel without reduction 13.84 vs pure stream 11.5: x delivery + CTA barrier; converter
  0.6 launch + 1.2 work. NVRTC-only pitfall: runtime-indexed acc -> local memory (+6 us); generic stores +6 us.
- Banked int_t3c = int_v7 + G (m=1, unp slice m<=8) + S (pooler m<=8): check 259 cases ALL OK.
- Per-SM TMA/bulk drain ~55-90 GB/s is the m=128 limiter (cuBLASLt 88 GB/s/SM; W RG1 needs ~115). Agent M: x via
  TMA + W via LDGSTS from 8 loader warps, R=128 (2 WGs share x), split-K 2 + PDL converter.
- Agents now: Q (qkv m2-32), O (o m2-32), P (pooler/unpooler m9-32 barrier-free), M (m=128).
- Banked int_t3e = int_v7 + G (m=1, slice m<=8) + S (pooler m<=8) + Q (qkv m 2..16): 259-case check ALL OK; qkv EXACT.
  Full bench (04:50): loads decode1 1.962 / d16 1.397 / d32 1.314 / v1 2.082 / v16 1.271 / v32 1.203 (local);
  judge-projected 1.4474 (base int_v7 1.4037).
- Exactness gate (claude/t3/exact_all.py, injected out-of-window weights + one-hot rows): Q's kernels bit-exact; G's
  g311/tcm, S's smallm and int_v7's GM give 0 instead of escapes < ~2^-39 (correction delta true*s - wdec rounded in
  fp32; equivalent to fp32 accumulation rounding; never occurs for normal weights). Stored formats are bit-exact.
  int_v7's K1s prepare raises on >~3000 injected escapes per layer (capacity) -> final build needs try/except fallbacks.

