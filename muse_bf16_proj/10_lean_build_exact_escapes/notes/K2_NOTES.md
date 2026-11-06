# Agent K2 — o / pooler / unpooler at m = 256 (verify-32), base claude/t3/LN/bo14_lean.py (2026-11-02)

## RESULT (read first): NO WIN — bo14_lean's m=256 kernels (agent K's g256k) stay; nothing to integrate
- Best same-run deltas vs bo14 (box 0, XCL 4 x clones, 16 reps; byte-identical K instances spread 0.06-0.16 us):
  o +0.18 (k2.py) / +0.26..+0.35 (k2dev), unpooler -0.04..+0.12 (k2.py) / +0.32..+0.39 (k2dev), pooler +0.07..+0.10
  (k2dev) / +0.29..+0.36 (k2.py). nsys in-graph: o 8.34 vs 8.13, pooler 23.33 vs 23.42, unpooler 23.28 vs 23.06.
- k2.py (15 KB, lean, readable) is K's kernel re-expressed (m64n128 / m64n64, LASTD 3, packed baked W, x-first issue,
  register epilogue) + test_wait spin consumer waits. It is NOT a win: do not integrate. Correctness: kt2 checks
  (random, outliers, one-hot bit-exact, NaN-filled graph replay) at m=256 passed; check.py / exact_all not run (not shipped).
- Why (measured): the loops already run at 98-99 % of the tensor rate at the 700 W-capped clock; the rest is the cold
  start (~1.3 us first-stage latency) and the per-SM store drain (~0.6 us). Ideas measured and why they fail:
  * 132 SMs by stream-K: exchange-free it saves only 0.22-0.28 us (unpooler) and is +1.8 us for the pooler; moving
    the 32-64 KB fp32 partial per CTA costs +2.5..+7 us (spread STGs +0.13 us per stage, LDG +0.8, fence +0.2).
  * Epilogue overlap (XY column-split tail): the last stage lands just in time, so half 0 cannot finish early; and the
    two-n64-accumulator layout it needs costs +0.3..+0.4 us vs m64n128.
  * Decoupled x / W rings: much worse (x is the L2 hot spot, not the cheap stream); W-before-x issue +1.8 us (pooler).
  * Smaller ring for o (NS 4/5): worse (9.33 / 7.64 vs 7.47).
  * Spin waits: +-0.1 (below instance spread). Best-of-N W placement: spread <= 0.16 us, not worth it.

Goal: beat bo14_lean's m=256 kernels (agent K's g256k: o / unpooler 128x128 tiles NS 6 LASTD 3, pooler 64x128 LAY1 NS 9;
LN/g256k_lean.py is SASS-identical) on pooler W[4096,8192], unpooler W[8192,4096], o W[8192,1024].
Files: k2dev.py (dev kernel, options below), kt2.py (same-process harness: x clones at random addresses (P5's method),
alternating order, K's lean g256k as in-process baseline, NaN-prefilled graph output check, one-hot exactness check,
--trace per-CTA globaltimer/clock64 timelines, --dcfg diagnostic builds), gensrc.py (local nvcc / SASS source dump).
All times local us (graph replay after a 512 MB flush, empty-graph cost subtracted), same process unless noted.

k2dev cfg = (LAY, NS, D, SK, MK, SS, LP, LD):
- LAY 0: 128 tok x 128 col tile, warpgroup g = tokens 64g.. as TWO n64 accumulator arrays (halves); LAY 1: 64 x 128, WG g
  = cols 64g.. (pooler, = K's LAY 1). NS ring stages of [x | W]; W prepacked + baked address (K's WPK 2), fast prologue,
  register epilogue (K's EPI 2).
- D (XY tail): the last D k-blocks run half 0 (cols 0-63) of both warpgroups, then half 1; half 0 is stored while half
  1's MMAs run (no lone-warpgroup phase).
- SK (stream-K on 132 SMs): chains of 33 CTAs over 32 tiles; position q: helper = tail k-blocks [NKB - q d, NKB) of tile
  q-1 (fp32 partial to global by spread STGs, flag published by a producer-WG warp after fence.acq_rel.gpu), then
  finisher = head [0, MK - q d) of tile q (partner's partial LDG'd into registers LP stages before the end, added in
  the epilogue). Makespan MK k-blocks instead of NKB. Descending k in every segment (see iteration 2).
- LD: agent K's LASTD tail priority (for parity on o / unpooler).

## Iteration 1 — XY tail and stream-K, first build
- Correct first time (kt2 checks: random, outliers, one-hot exact, NaN-filled graph replay).
- ptxas (local nvcc 12.1 sm_90a + NVRTC on the box): stream-K needed two fixes to avoid wgmma serialisation:
  C7515 (the finisher accumulator's zero-init was sunk next to the helper's in-flight wgmma group -> pin zeros with an
  asm register fence at kernel start; add the partial into a temporary, never into accumulator registers) and C7516
  (ANY CTA-uniform branch that lets some CTAs skip the finisher's wgmma region, including an early `return` or an
  if/else around it, serialises every wgmma of the kernel -> the pure-helper chain end runs a dummy finisher segment
  with the same code shape and skips only its stores). Final: 168 regs, 0 spills, LOCAL 0, no C751x.
- Same run (box 1, XCL 2, 10 reps): o K 7.36 | XY D2 7.42 | plain 7.38; unpooler K 22.13 | XY D2 22.13 | plain 22.06;
  pooler K 22.07 | my LAY1 copy 22.14.
- Stream-K first build: pooler 45 us, unpooler 30.8 us (!). Trace: the LOOP itself ran at ~2x the k-block time
  (helper segments of ~48 kb took 17 us). Cause: with ascending k, chain position q starts its finisher tile q*d
  k-blocks late, so the 32-64 CTAs that share an x token tile read the same x k-block up to ~14 us apart; 67 MB of W
  streaming through L2 in between evicts x -> x re-read from DRAM (~128 MB extra).

## Iteration 2 — descending k, partial into registers, XY ordering
- Descending k in both segments: CTA q works on k-block 127-tau (helper) / 124-tau (finisher) at time tau whatever q,
  so all CTAs stay within d k-blocks: pooler SK 45 -> 26.2, unpooler 30.8 -> 25.1 us. Still +3..+4 us.
- Pooler SK used NS 8 (partial buffer): loop 0.188 us/kb vs 0.162 at NS 9: the pooler loop needs every byte of ring
  depth (it streams 24 KB per 0.155 us k-block = ~150 GB/s per SM, the per-SM ingress limit Q3 measured).
- XY tail D2/D3/D4 (unpooler): 22.31 / 22.38 / 22.38 vs plain 22.34 (no gain). Trace: the last stage is issued ~1.6 us
  before the loop end and lands just in time (6-stage ring, ~1.3 us W latency): half 0 cannot finish before the last
  stage lands, so the X phase starts late (main end -> X done 0.84-1.05 us), and the 32 KB/CTA store drain
  (~30 B/clk/SM) is then the same length. A named barrier so both warpgroups issue half 0 before half 1 did not help.

## Iteration 3 — what the stream-K exchange costs (diagnostic builds, box 0, XCL 2, 16 reps)
| unpooler build | us | vs K 22.38 |
|---|---|---|
| plain 128 CTAs | 22.37 | -0.01 |
| SK (MK 63) full exchange | 25.02 | +2.64 |
| SK, no fence | 24.82 | +2.44 |
| SK, no partial stores (STG) | 23.30 | +0.91 |
| SK, no partial loads (LDG) | 24.24 | +1.86 |
| SK, no stores / loads / fence (wrong result, timing only) | 22.16 | **-0.22** |
- The 132-SM gain itself is real but small (unpooler 63 vs 64 k-blocks: -0.22..-0.28 us); the exchange of a 64 KB fp32
  partial per CTA costs ~3 us: the spread STGs add ~0.13 us per stage they are interleaved with (+1.7 us), the LDG burst
  ~0.8 us, the fence ~0.2 us (published 1.9 us after the last partial store was issued).
- Moving the STG / LDG slices between each stage's commit and wait (iteration 4 build) did not change this
  (unpooler SK 24.84-25.28 full, 22.26-22.90 exchange-free).
- Pooler SK (NS 9, MK 125): +7 us full, **+1.8 us even with no exchange at all** (loop 0.188 us/kb at 132 CTAs vs 0.162
  at 128): the pooler loop does not get faster with more SMs. Stream-K is a dead end for these shapes.

## Iteration 4 — decoupled x / W rings, issue order, producer leanness
- Decoupled rings (W NS slots issued NS stages ahead, x NX < NS slots): unpooler NX 5/4: 23.2 / 28.3 vs 22.4; pooler
  NX 4/5/7: 39.1 / 30.8 / 27.5 vs 22.4: much worse. x is NOT the cheap stream: 32-64 CTAs read the same x tile at once
  (L2 hot spot); it needs the full lead.
- Issue order inside a stage: W bulk before the x TMA: pooler +1.8..+2.1 us. x first (K's order) is required.
- PRODUCER LEANNESS: a producer issue path that recomputed slot / tile / k-block with %, / and lambdas per issue cost
  the pooler +2.1 us and the unpooler +0.3 us (trace: last issue 0.8 us before loop end instead of 1.24, loop 0.174 us/kb
  at a HIGHER clock 1.647 GHz = delivery-starved). Thread 0 shares SMSP 0 with two consumer warps; with precomputed
  per-segment bases + slot/phase counters it is back to parity (pooler 22.52 vs K 22.55).

## Iteration 5 — start anatomy (o, trace, box 0)
| o build | CTA start | stages 0-1 issued | prologue (6 stages) issued | first stage landed | main end | end wg0 / wg1 |
|---|---|---|---|---|---|---|
| plain NS 6 | 0.19 | 0.43 | 0.86 | 1.64 | 6.79 | 7.15 / 7.58 |
| XY D2 | 0.22 | 0.46 | 0.89 | 1.66 | 6.14 | 7.21 / 7.57 |
- First stage lands ~1.3 us after its issue (x needs two dependent cold round trips: the kernel-param tensor map,
  then the data; W from a baked address needs one), later stages arrive ~0.3 us apart (1 MB of distinct W per stage
  index at the start burst). o's loop (16 kb) then runs at ~0.34 us/kb (92 % of the 0.313 MMA time at 1.641 GHz) because
  the consumer cannot get ahead of the DRAM-bound arrival curve.
- Spin-waits (test_wait loops instead of try_wait) for producer and/or consumer: o 7.46-7.51 vs 7.55 plain (noise level).
- Harness pitfall: `timeout N python3 claude/sbx.py exec ...` counts the GPU-lock wait; with 4 agents on one box a
  700 s outer timeout killed a run mid-measurement (its remote process then ran outside the lock). Use a long outer
  timeout (lock wait) and a short inner one.

## Floors at the measured (power-capped) clocks
| shape | GFLOP | per-CTA MMA time, 128 CTAs (132) | clock in loop | W bytes / DRAM time at 4.5 TB/s |
|---|---|---|---|---|
| o | 4.29 | 4.99 us (4.84) | 1.641 GHz | 16.8 MB / 3.73 us |
| unpooler | 17.18 | 20.07 us (19.46) | 1.633 GHz | 67.1 MB / 14.91 us |
| pooler | 17.18 | 20.48 us (19.86) | 1.600 GHz (n64 per WG: more smem traffic per FLOP) | 67.1 MB / 14.91 us |
Measured (K's kernels = mine at parity): unpooler / pooler loops run at 98-99 % of these MMA times; o's loop at ~92 %
(arrival-bound start burst). Everything above the MMA time is start (~1.5-1.7 us to the first stage) + tail (unpooler
/ o ~0.6-0.7 us store drain of 32 KB/CTA at ~30 B/clk/SM, partly hidden by LASTD; pooler ~0.35 us for 16 KB).

## Iteration 6 — K's tail priority (LD) in k2dev, ring depth for o, spin waits
- Incident: the first LD build consumed LD extra stages (loop bound bug) and hung until its 330 s inner timeout on the
  shared box (my mistake: not smoke-tested under a short timeout first). Fixed; smoke test (150 s) passed.
- o ring depth: NS 4 / LD 2 = 9.33 vs NS 6 / LD 3 = 7.47 (smoke, XCL 1): fewer stages in flight starve o badly (P5's
  "60-72 KB of W in flight" sweet spot does not transfer: this ring also carries x, and o has only 16 k-blocks).
- Same run (box 0, XCL 3, 16 reps), vs bo14_lean's K kernels:
  | shape | K (bo14) | k2dev LD3 / pooler LAY1 | + consumer spin-wait (test_wait) | o NS 5 LD 3 |
  |---|---|---|---|---|
  | o | 7.36 | 7.69 (+0.33) | 7.56 (+0.20) | 7.64 (+0.28) |
  | unpooler | 22.09 | 22.48 (+0.39) | 22.41 (+0.32) | |
  | pooler | 22.51 | 22.60 (+0.09) | 22.52 (+0.01) | |
  => the TWO-n64-halves accumulator (needed for the XY tail) costs o / unpooler +0.3..+0.4 us vs K's single m64n128
  (8 HGMMA.64x64 per stage instead of 4 HGMMA.64x128: A is read from smem twice; smem traffic per FLOP +33 %).
  The spin-wait looked like -0.07..-0.13 on the same kernel -> rebuilt as k2.py = K's kernel shape (n128) + spin
  consumer waits, A/B'd against K below.
- k2.py (lean module = K's kernel shape: m64n128 for 128x128 tiles, m64n64 LAY1 pooler, LASTD 3, x-first issue, baked
  packed W, register epilogue; only difference: consumer full waits by test_wait spin, `spin=False` = try_wait), same
  run (box 0, XCL 4, 16 reps):
  | shape | K (bo14) | k2 spin | k2 no spin | k2dev (2 x n64 + LD3 / LAY1) |
  |---|---|---|---|---|
  | o | 7.25 | 7.42 (+0.18) | 7.45 (+0.20) | 7.60 (+0.35) |
  | unpooler | 22.26 | 22.23 (-0.04) | 22.39 (+0.12) | 22.59 (+0.32) |
  | pooler | 22.33 | 22.69 (+0.36) | 22.62 (+0.29) | 22.40 (+0.07) |
  k2 no-spin should equal K but sits +0.12..+0.29 above it -> instance (W copy) effects of that size exist; the spin
  effect (+-0.1) is below them. No shape wins.

## Iteration 7 — W placement (3 byte-identical instances of bo14's kernels, each with its own packed W, one process)
| shape | instance 0 | instance 1 | instance 2 | spread | k2dev (same run) |
|---|---|---|---|---|---|
| o | 7.36 | 7.44 | 7.36 | 0.08 | 7.63 |
| unpooler | 22.16 | 22.24 | 22.32 | 0.16 | 22.52 |
| pooler | 22.52 | 22.51 | 22.57 | 0.06 | 22.61 |
(box 0, XCL 4 x clones each, 16 reps.) W placement moves these kernels by <= 0.16 us: a best-of-N W copy chosen by
timing at prepare could gain <= ~0.1 us (and costs 67 MB per copy + N kernels to compile): not worth it. The +0.1..+0.4
of my variants over K is larger than this spread, i.e. real.

## Final analysis block (k2dev best variants vs bo14's K kernels, box 0)
1. nsys in-graph (claude/t3/nsys_shapes.py, 20 reps, flushed; K2/prof.sh): o k2dev(LD3) 8.34 vs K 8.13; pooler k2dev
   23.33 vs K 23.42; unpooler k2dev(LD3) 23.28 vs K 23.06 us.
2. ncu (claude/t3/ncu_run.sh, full set, flushed, unpooler): k2dev 24.74 us @ 1.70 GHz vs K 25.12 @ 1.65 (ncu clocks
   differ: compare utilisations); DRAM read 69.2 MB both (~2.8 TB/s = 58 % of 4.8 over the ncu duration; not binding);
   SM throughput 75.5 vs 76.7 %; issue active 23.0 vs 22.0 %; eligible 0.27 vs 0.24 warps/clk; occupancy 14 % (12 warps);
   stalls k2dev long_scoreboard 26 % (mbarrier waits), warpgroup_arrive 23 % (two m64n64 groups per stage), wait 15 %;
   K barrier 30 % (LASTD named barrier), wait 20 %, long_scoreboard 18 %. (K's iteration-3 ncu: tensor 77 % of elapsed /
   89 % of SM-active cycles.)
3. SASS (NVRTC/nvcc 12.1 sm_90a): all kernels 168 regs, 0 spills, LOCAL 0, no C751x serialisation, no LDG anywhere
   (stream-K builds: 2 LDG = flag polls + the partial loads); per stage per warpgroup: K 4 HGMMA.64x128x16 (o / unp) /
   4 HGMMA.64x64x16 (pooler), k2dev 8 HGMMA.64x64x16 (o / unp); producer loop ~40 uniform-datapath instructions per
   stage (1 UTMALDG + 1 UBLKCP + SYNCS); epilogue 8 STG.128 + 32 SHFL per thread (o / unp), 4 + 16 (pooler).
4. Binding units: inside the loop the tensor pipe at 98-99 % of its rate at the power-capped clock (pooler 1.60 GHz,
   unpooler 1.633, o 1.641; o's loop 92 %: DRAM-arrival-bound start burst). Outside it: the first stage lands ~1.3 us
   after issue (x = two dependent cold round trips: param tensor map then data) and the 32 KB/CTA bf16 store drains at
   ~30 B/clk/SM (~0.6 us; LASTD hides part). Neither the 132-SM split (exchange of fp32 partials costs more than the 3 %
   it saves), deeper / decoupled / shallower rings, an XY column-split tail, spin-waits, nor W placement moved them.
