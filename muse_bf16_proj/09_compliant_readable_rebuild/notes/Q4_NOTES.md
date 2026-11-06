# Agent Q4 — qkv_gate W[2304, 8192] at m = 32 (17..32): beat bo12 (F's sk11, R = 24, 96 CTAs)

## RESULT (read first): no win at m = 32 — bo12 / bo13 keep qkv_gate m = 17..32
- Built and measured q32 (RS-wgmma on 12-bit F12 weights, 64 W rows x 1/3 of K per CTA, 108 CTAs in clusters of 3,
  raw-unit escapes, DSMEM st.async reduce-scatter). Correct and exact (check.py --graph 52 cases x 2 seeds ALL OK,
  exact_all --nesc 200 and --nesc 3000 EXACT ALL OK, with Q32_FORCE=1), but slower than bo12's sk11 in every run:
| run (bench.py, 40 reps, local us; judge ~= local + 1.0) | SGLang | q32 (x vs SGLang) | sk11 = bo12 / bo13 (x vs SGLang) | int_v7 |
|---|---|---|---|---|
| box 0 run 1 | 13.73 | 9.79 (x1.403) | **8.79 (x1.562)** | x1.148 |
| box 0 run 2 | 13.87 | 9.66 (x1.436) | **8.69 (x1.596)** | |
| box 1 run 1 | 13.58 | 9.54 (x1.424) | **8.56 (x1.586)** | |
| box 1 run 2 | 13.47 | 9.46 (x1.425) | **8.55 (x1.575)** | |
  nsys in-graph (box 0, 20 reps): q32 10.40 us vs sk11 9.63 us.
- Deliverable: q32.py (prepare_q32 / run_q32; run_q32 returns None for every shape unless Q32_FORCE=1 or a tag is
  given, because bo12 wins), sub_q32.py (= bo13 + q32 -> identical to bo13), sub_q32_only.py (q32 + torch, for forced
  tests), q4t.py (same-run harness vs F's sk11 with per-CTA timelines and ablation flags), sass.py, one.py + ncuq.sh,
  dbg.py. q32_it1/2/3.py = frozen earlier iterations.
- Why it cannot win (all measured, iterations 1-5):
  1. wgmma at N = 32 tokens runs at ~45 % of the tensor peak (35-43 clk per HGMMA.64x32x16 RS vs 16 nominal;
     agent W: RS n64 73 %, SS n64 66 %). Per chunk of 4096 weights the tensor pipe needs ~140 clk, more than
     mma.sync's ~99 clk-equivalent (16 HMMA x 6.2 clk per SMSP). At m = 32 the mma.sync primitive of sk11 is the
     more tensor-efficient one; 4 accumulators, 2-4 groups in flight and 2 consumer warpgroups do not change it.
  2. The data ring of this geometry is not faster than sk11's: W streams at 4.7 TB/s (108 CTAs, 12 KB copies) vs
     ~5.1 TB/s in sk11's ring; cutting per-SM ingress from 807 to 436 KB only cuts the x cost (+0.5 vs +1.2 us at the
     ring max). The EMPTY ring + cluster exchange (8.58 local) already ties sk11's full kernel (8.61).
  3. Under traffic the consumer needs 0.165 us per chunk vs the ring's 0.144 -> it finishes ~1.1 us after the ring.
  4. The 3-way DSMEM exchange costs ~0.5 us after the loop (no-cluster ablation: 9.22 vs 9.76, wrong results).
  A win would need BOTH a consumer that adds < 0.1 us over the ring AND a hidden exchange; the headroom even then is
  ~0.35 us (no-exchange ring 8.26 vs sk11 8.61).
- Coordinator's suggestion (K-split 2 + Q3's hidden exchange): the exchange is item 4 (0.5 us, hideable at the cost of
  ~1 extra chunk per rank), but items 1-3 remain; for an sk11-style mma.sync consumer at K-split 2 (R = 36, 128 CTAs,
  5 n8 tiles with a half-empty one) V2 measured an unchanged loop-end max (8.32 vs 8.32), so a free exchange gives
  at most ~0.2-0.3 us there. Not built.


Files (claude/t3/Q4/): q32.py (kernel module: prepare_q32 / run_q32), q4t.py (same-run harness: `q:` q32 variants vs
`f:` F's sk11 = bo12's kernel; judge-style graph replay after a 512 MB flush, empty graph subtracted, interleaved;
CHECK=1 normal + outliers + one-hot exact on NaN-prefilled outputs; TRC & 1 per-CTA timeline), sass.py (local NVRTC
sm_90a build: regs / spills / hot-loop mix).
Times: local us (judge ~= local + 1.0). Only same-run deltas count (per-instance offsets 0.3-0.5 us, agent H).

## First-principles budget (why a new kernel, not another sk11 knob)
Per SM, sk11 m32 (96 CTAs x 24 rows x full K): 196 K weights, 768 HMMA per SMSP (m16n8k16 at ~6.2 clk = 2.8 us of
tensor pipe at 1.72 GHz) + ~1350 half-rate decode ALU per SMSP (1.6 us) + 16 LDSM + 12 LDS per 48 HMMA, ONE warp per
SMSP (in-order, every LDS -> decode -> HMMA latency exposed: measured ~13 clk/HMMA in T's loopbench -> 5.8 us); ingress
295 KB W + 512 KB x = 807 KB per SM (x = 63 %), above the ~100-110 GB/s per-SM (per-GPC/16) delivery V2/H measured.
=> two binding limits at once: consumer latency and x ingress. Any 132-SM n8 layout keeps 3 tiles/SM (288 tiles).

q32 design: W rows as the wgmma M dimension.
- CTA = 64 W rows x 1/3 of K (43/43/42 or 44/42/42 chunks of 64 k) -> 36 row groups x 3 = 108 CTAs, cluster of 3.
- consumer = ONE warpgroup: per 64-k chunk 3 LDS.128 + 1 LDS.64 header, 28 decode ALU (agent G's F12: 1 LOP3 + 1/2
  SHF per 4 weights' hi bytes, 1 PRMT per bf16x2), 4 async HGMMA.64x32x16 (A = decoded W registers, B = x^T straight
  from the TMA 128B-swizzled tile: no ldmatrix, no cross-warp k reduction). Decode of chunk u+1 overlaps the async
  HGMMAs of chunk u (agent M / T's LOOP7 schedule). Budget per SM: 44 chunks x (64 tensor clk at 100 % + 56 ALU clk
  per SMSP, overlapped) ~ 2.0 us of a ~6 us stream -> the consumer has ~3x slack (sk11: ~1.0x).
- ingress per SM: 264 KB W + 172 KB x = 436 KB (x read 108 x 172 KB = 18.6 MB from L2 vs 49 MB).
- escapes: raw units (agent M / T): a (chunk, warp, k16) unit with any out-of-window weight is stored as exact fake
  BF16 (true x 2^(112 - elo); lo byte in the lo plane, hi bytes of weights 0-3 in the nibble word, 4-7 in a per-CTA
  overflow list). qkv: elo 107, 1335 raw units (1.81 % of units), overflow <= 2.8 KB per CTA; no escape warps, no
  correction pass, bit-exact (validated at prepare: fake values must be exact normal BF16 or zero, else fallback).
- reduction: cluster DSMEM reduce-scatter by 8-token groups (group i owned by rank (i + 2) % 3, so the 44-chunk rank
  owns the fewest), st.async.v4 pushes from registers with complete_tx on the owner's mbarrier (iteration 1 used bulk
  copies from smem staging), fixed summation order (deterministic); bf16 via stmatrix.trans + 16-B stores.
- (measured later: the "~3x consumer slack" estimate above was wrong; HGMMA.64x32x16 runs at ~45 % of nominal and the
  consumer under traffic is slower than the ring, see iterations 3-5.)

## iteration 1 — first build (q32_it1.py): S 3, UPC 2 (12 KB W + 8 KB x per stage), NSL 12 chunks (120 KB ring)
Correct on the first build (normal, outliers, one-hot exact; m 32 / 25 / 17). But slow, same run (box 1/0, 10-20 reps):
q32 10.4-11.1 vs sk11 (bo12) 9.14-9.15 local.
Per-CTA timeline (flushed, median of 7 replays, us from first CTA start; ablations same process):
| variant | first-W med | loop-end min/med/max | end max | per chunk |
|---|---|---|---|---|
| full | 1.15 | 8.45 / 9.06 / 9.82 (rank 0 9.54, ranks 1-2 8.9) | 10.53 | 0.18-0.19 us |
| no MMA (decode dead-code eliminated) = data ring | 1.28 | 7.39 / 7.62 / 7.94 | 8.67 | 0.144 (W 4.5 TB/s) |
| no W traffic | 0.42 | 5.31 / 5.41 / 5.79 | 6.53 | |
| no x traffic | 1.18 | 7.62 / 8.16 / 8.99 | 9.70 | 0.162 |
| no W, no x (pure consumer) | 0.38 | 4.58 / 4.67 / 5.02 | 5.76 | 0.104 = 177 clk |
| no traffic, no MMA | 0.32 | 1.06 / 1.12 / 1.22 | 1.98 | |
| NSL 18 (198 KB ring) | 1.25 | 9.09 / 9.82 / 13.82 (rank 0 13.7) | 14.88 | much worse |
Reading: the data ring is DRAM-bound at ~4.5 TB/s (W) -> floor ~1.2 + 6.3 us; the consumer alone needs 177 clk per
chunk (expected ~70: 4 x 16 clk of HGMMA.64x32x16 + 28 decode ALU) and does not hide behind the data (9.06 vs 7.62).
Deeper rings hurt (more DRAM concurrency, as V2 / H found). Tail loop-end -> end 0.7-1.1 us (bulk-copy DSMEM push
from smem staging + rank skew: rank 0 has 44 chunks vs 42).
The first 5-variant run died with CUDA_ERROR_ILLEGAL_INSTRUCTION: bisected (dbg.py) to the no-W ablation only: garbage
raw-unit headers sent rawfix to out-of-range overflow addresses (fixed: mine = 0 in no-W ablations). Real kernel clean.
### iteration-1 analysis block
1. nsys: not run (bench + traces only; superseded within the hour).
2. ncu (full set, flushed, box 0, ncu-inflated): q32 dur 12.77 us, DRAM rd 29.30 MB, L2 53.7 %, L1/smem 20.2 %,
   SM 12.1 %, issue 18.6 %, tensor pipe (hgmma) active 9.7 % of elapsed, occ 8.7 % (6 warps), elig 0.19 warps/clk,
   clk 1.81 GHz, 1.47 M instructions; stalls long_scoreboard 30 %, wait 18 %, **no_instructions 17 %**, selected 12 %,
   warpgroup_arrive 5 %. sk11 same run: 10.88 us, 28.89 MB, L2 59 %, L1/smem 47.7 %, issue 25.7 %, hmma 10.4 %,
   occ 11.9 %, elig 0.28, 1.51 M inst, long_sb 42 %, wait 20 %.
3. SASS (NVRTC sm_90a local): 117-121 regs, 0 spills, LOCAL 0; per chunk 4 HGMMA.64x32x16 (all into ONE accumulator),
   16 PRMT, 8 LOP3 (hi bytes), 4 SHF, LDS.64 + 3 LDS.128, ~10 uniform descriptor ops, 2-3 SYNCS, WARPGROUP.ARRIVE +
   DEPBAR.LE gsb0 1; no LDG. Code: two fully unrolled 12-chunk loops + tails (~8 K instructions, ~128 KB of SASS)
   -> the no_instructions stalls.
4. Binding unit: none saturated (tensor 9.7 %, smem 20 %, issue 19 %, DRAM 4.5 TB/s only inside the data ring); the
   consumer is latency-bound: 4 dependent HGMMAs on one accumulator (N = 32 = 16 clk of work vs ~40 clk latency) +
   i-cache misses. Next: 4 independent accumulators, compact loop, st.async push, rank balance.

## iteration 2 — 4 accumulators (one per k16 step), compact runtime-stage loop, st.async push, rank balance
(q32_it2.py) Loop = 1 stage (2 chunks) per iteration, same code for both rank sizes: 1160 SASS instructions total (was
~8 K). Epilogue: partner partials pushed with st.async.v4 from registers (complete_tx on the owner's mbarrier), the
44-chunk rank owns the fewest token groups. Correct (m 32 / 25 / 17, one-hot exact). Same run (box 1, 20 reps):
NACC 4: 10.78-10.94 | NACC 1: 10.08 | NSL 16: 11.31 | NSL 8: 10.59 | sk11 8.59.
Pure consumer (no traffic): 0.105 us/chunk, unchanged -> the HGMMA->HGMMA accumulator chain is not the limit.

## iteration 3 — D wgmma groups in flight (4 A-register buffers, wait_group D-1), per-chunk accumulator alternation
(q32_it3.py) Correct (m 32 / 21). Same run (box 1, 25 reps): D 2 / 3 / 4 (NACC 1) 10.14 / 10.82 / 10.21, NACC 2 D 4
10.75, sk11 8.58. Pure consumer 0.115 us/chunk for every D / NACC -> not the group latency either.
Component ablations (pure consumer, no traffic; loop us for 42-44 chunks; per chunk at 1.72 GHz):
| consumer content | loop | per chunk |
|---|---|---|
| decode + LDS + 4 HGMMA (full) | 5.0 | 200 clk |
| 4 HGMMA only (no decode, no LDS) | 3.5 | 140 clk (~35 clk per HGMMA.64x32x16 RS: ~46 % of the tensor peak) |
| decode + LDS only (XOR sink, no HGMMA) | 3.07 | 123 clk |
| loop skeleton (barrier arrives, no work) | 0.96 | 38 clk |
With traffic (W + x): data ring 6.3 | + HGMMA 6.6 | + decode 7.1 | both 7.8 us.
=> one warp per SMSP serialises decode (ALU pipe, ~2 clk per op), HGMMA issue and barrier waits; neither the tensor
pipe (140 clk) nor the decode (123 clk) alone would bind under the ~250 clk/chunk data rate, but their sum does.
### analysis block (iterations 2-3)
1. nsys: not run (trace + bench decided; kernels not at par yet).
2. ncu: iteration 1 numbers still describe the loop (tensor 9.7 %, issue 18.6 %, smem 20 %); the i-cache share
   (no_instructions 17 %) is removed by the compact loop (SASS 1.1-1.6 K instructions).
3. SASS: 108-156 regs (NACC 4: 149-156), 0 spills, LOCAL 0; per chunk 4 HGMMA (+ 4 / 2 accumulators), 16 PRMT,
   8-12 LOP3, 4 SHF, 4 LDS, ~18 R2UR / IMAD for runtime-slot descriptors, 1 DEPBAR.LE (no wgmma serialisation).
4. Binding unit: consumer issue/latency on one warp per SMSP (tensor 140 + decode 123 clk per chunk do not overlap).
   Next: 2 consumer warpgroups (2 warps / SMSP), each owning one chunk of every stage.

## iteration 4 — two consumer warpgroups (KWG 2: WG g owns chunk g of every stage, accumulators folded via smem)
(current q32.py, cfg (S, UPC, NSL, TRC, NACC, D, KWG) = (3, 2, 12, 0, 1, 2, 2)) Correct (m 32 / 19 / 28, one-hot exact).
Same run (box 1, 25 reps): KWG 2 D 2 **9.57** | D 3 9.86 | D 4 11.17 | NSL 8 16.45 | NSL 16 10.69 | KWG 1 D 2 10.18 |
sk11 (bo12) **8.74**. Pure consumer (no traffic) 4.4 us vs 5.0 with one WG: the tensor pipe is shared, so a second
warpgroup buys little. Trace KWG 2 D 3: first-W 1.31, loop-end 8.58 / 8.90 / 9.15 (min / med / max), end max 9.79.
### iteration-4 analysis block
1. nsys: see the final block.
2. ncu (full set, flushed, box 1): KWG 2 with traffic: dur 11.65 us, DRAM rd 29.27 MB, L2 55.7 %, L1/smem 24.5 %, issue
   29.8 %, tensor pipe active 10.7 % of elapsed, ALU pipe 29.5 %, occ 14.4 % (10 warps), elig 0.39, 2.08 M inst; stalls
   long_scoreboard 33 % (full-barrier data waits), wait 22 %, selected 12 %, warpgroup_arrive 6 %, no_instructions 5 %.
   Same kernel without traffic: 7.49 us, tensor 16.4 %, ALU 39.4 %, issue 36.8 %, stalls wait 26 %, selected 17 %,
   short_scoreboard 10 %, math_pipe_throttle 10 %.
3. SASS: 140-146 regs (320 threads), 0 spills, LOCAL 0, ~2 K instructions; per chunk per WG 4 HGMMA.64x32x16, 16 PRMT,
   12 LOP3 / SHF, 4 LDS, 1 DEPBAR.LE, slot / parity math ~20 IMAD / R2UR.
4. Binding unit: the tensor pipe at N = 32: 35-43 clk per HGMMA.64x32x16 RS (nominal 16; agent W measured RS n64 at
   73 %, SS n64 66 %), i.e. ~3.6 us of tensor time per SM for 44 chunks, plus decode on the ALU pipe; under traffic the
   consumer needs 0.165 us / chunk vs the ring's 0.144 -> it ends ~1 us after the ring.

## iteration 5 — what the ring itself costs (consumer emptied: no MMA / decode / LDS) and the cluster exchange
Same run (box 0, 25 reps), local us; trace loop-end med / max:
| ring variant | local | first data | loop-end med / max | end max |
|---|---|---|---|---|
| W + x (cluster of 3, exchange) | 8.58-8.61 | 1.18 | 7.52 / 7.84 | 8.48 |
| W + x, no cluster launch, no exchange | 8.26 | 1.18 | 7.42 / 8.03 | 8.29 |
| W only (cluster, exchange) | 8.13-8.22 | 1.12 | 7.17 / 7.36 | 8.00 |
| W only, no cluster, no exchange | 7.62 | 1.15 | 7.17 / 7.42 | 7.65 |
| x only | 4.54 | | 3.78 / 4.00 | |
| W + x, ring 10 / 14 chunks | 8.45 / 8.77 | | 7.52 / 7.55 | |
| full kernel KWG 2 D 2: cluster / no exchange (wrong results) | 9.76 / 9.22 | 1.22 | 8.61 / 9.12 | 9.70 |
| sk11 (bo12) | 8.61 | | | |
Reading:
- The W stream of this geometry (108 CTAs, 12 KB copies) runs at 4.7 TB/s (W-only: 6.05 us after first data),
  sk11's W-only ring at ~5.1 TB/s (V2 iteration 1). Per-SM ingress (436 vs 807 KB) is not what limits either ring.
- x costs only +0.35 (med) / +0.5 (max) here vs +0.5 / +1.2 in sk11's ring -> the ring max of this geometry is
  ~0.5 us better than sk11's, the only real gain this design has.
- The 3-way cluster exchange costs ~0.5 us after the loop (cluster placement itself does not slow the ring: loop ends
  are identical with and without the cluster launch).
- Even the EMPTY ring + exchange (8.58) ties sk11's full kernel (8.61): to win, the consumer would have to add < 0.1 us
  over the ring and the exchange be hidden (Q3's solo phase); the measured consumer adds ~1.1 us (iteration 4).

## final analysis block (shipped config (3, 2, 12, 0, 1, 2, 2), forced on)
1. nsys in-graph (box 0, 20 reps, flushed): q32_3_2_12_0_1_2_2 10.40 us vs sk11_qkv_gate_32 (bo12) 9.63 us.
2. ncu (iteration 4, box 1, flushed, full set): dur 11.65 us (ncu-inflated), DRAM 29.27 MB read (2.5 TB/s averaged over
   the ncu duration; 4.5-4.7 TB/s inside the ring), L2 55.7 %, L1/smem 24.5 %, issue 29.8 %, tensor pipe 10.7 % of
   elapsed, ALU 29.5 %, occupancy 14.4 % (10 warps), eligible 0.39 warps/clk; stalls long_scoreboard 33 %, wait 22 %,
   selected 12 %, warpgroup_arrive 6 %.
3. SASS (NVRTC sm_90a): 140-146 regs, 0 spills, LOCAL 0, ~2 K instructions; per 64-k chunk per warpgroup: 4
   HGMMA.64x32x16 (RS), 16 PRMT + 8 LOP3 + 4 SHF decode (7 ALU per 8 weights), LDS.64 + 3 LDS.128, 1 DEPBAR.LE,
   1 WARPGROUP.ARRIVE, 2-3 SYNCS; no LDG anywhere (W bulk copies, x 3D TMA); epilogue 1-2 STAS.128 (st.async) per
   partner group, stmatrix.trans, 16-B STG.
4. Binding unit: DRAM for the ring (4.7 TB/s W = 98 % of the practical 4.8, but only 6.05 us of the 9.5) plus the
   consumer's tensor pipe at N = 32 (~45 % efficiency, 3.6 us of tensor time per SM) that cannot overlap with the
   bursty ring, plus a 0.5 us exchange tail. Not a path to beat sk11 at m = 32; see RESULT for what would be needed.
