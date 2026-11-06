# 16K boundary-fused GEMMs (UB unpooler+post-FFN, D6 o_proj+post-attention): notes

Tools (all in this dir): probe.py (kernel-level: full / gemm / bnd-only / cublas modes, UB_TRACE summaries,
output checks), stream_probe.py (per-SM streaming ceilings), make_cand.py (builds single-file candidates from
b23_compact.py + _BN_BODY.cu), lcompile.sh (local sm_90a nvcc compile/spill check), sass.sh (remote SASS dump).

## Step 1 measurements (m1/m2 logs; probe inputs random, M=16384, cold L2, CUDA-graph replays)
| kernel | full | GEMM-only | bnd-only (132 CTAs) | cuBLAS GEMM |
|---|---|---|---|---|
| UB ub2_2_192_3 (K 4096) | 1757 us | 1434 | 481 (HBM-bound, 3.9 TB/s) | 1455 |
| D6 ub2_2_192_4_4 (K 1024) | 617 | 418 (min 390) | 261 | 411 |

Trace (UB_TRACE, per-CTA items):
- UB: GEMM tile 69.3 us (unchanged vs GEMM-only 68.8), boundary 1.93 us/row (16-row items 30.9 us);
  SM-time/CTA: GEMM 1445 us, boundary 240 us, idle 2.7%; tail after last GEMM tile 29 us; ~18 CTAs on boundary.
- D6: tile 18.8 us, boundary 1.33 us/row; SM-time/CTA GEMM 392, boundary 165, idle 4%; tail 29 us (last group
  is 4 m-tiles = 1024 rows after the last tile).
- => fused = GEMM SM-time + boundary SM-time + tail. Bound if boundary were free: UB ~320 us, D6 ~200 us per call.

Boundary-only at 16 CTAs (no HBM contention) with ablations (us/row): UB full 1.48, no-store 1.26, no-load 1.39,
neither 1.24; D6 0.94 / 0.88 / 0.92 / 0.87. Fused is slower (1.93 / 1.33): lower clock under GEMM load (~1.6 vs
~1.98 GHz) and L2 contention.

## Per-SM ceilings (stream_probe.py)
- Writes: 63 GB/s per SM (32 B/clk) for STG.128 (256/512 threads, with or without .cs) AND for bulk S2G, at grid
  1 and 16; 31.8 GB/s/SM at 132 CTAs (HBM 4.19 TB/s).
- Bulk G2S loads: 99 / 141 / 177 / 199 GB/s per SM with 2 / 3 / 4 / 6 32 KB stages (latency-bound).
- => UB row writes 48 KB: floor 0.78 us/row at 1.98 GHz, ~0.94 at fused clock. 16384 rows / 132 SMs -> ~117 us
  SM-time per call is the hard floor for the UB boundary (vs 240 now). D6 row writes 16 KB: 0.31 us/row floor.

## c1/c2/c3: epilogue row sums + one block reduction per row (_BN_BODY.cu, BnPlan)
- GEMM tile epilogues red.add each row's sum of t^2 (UB: t = f (1 + wpost)) into ssq[M]; boundary reads it (rt
  known up front), computes y = A s + Bv ra a + C' rt f in FP32 (no bf16 round trips), one reduction (sum y^2),
  zeroes ssq[row]. Instructions per row ~-50%. Outputs vs base: rel 2.6e-3 (x), 1.7e-3 (s), within bf16 noise.
- c1 (2-row software pipeline): register spills (CUDA 13: STACK 168-184) inside the GEMM mainloop: tiles 69 ->
  76 us (UB), 18.8 -> 22.4 (D6). Rejected.
- c2/c3 (serial rows): no spills, tiles back to ~70.5 / 18.6 us; boundary-only compute (no load/store) UB 1.24 ->
  0.96 us/row, D6 0.87 -> 0.69, BUT fused per-row unchanged (UB 1.85-1.88, D6 1.36-1.40): stores do not overlap
  compute (c2 fused: UB none 1.19 / no-store 1.31 / no-load 1.68 / full 1.875; D6 0.80 / 0.98 / 1.15 / 1.36).
  Removing the "memory" clobber from the stores changed nothing (c3, c0n).
- Next: async bulk S2G stores from smem (outputs staged over the consumed stage, a storer thread in the producer
  warpgroup releases the stage after the bulk read) so the write port drains while the warps compute.

## c4: async bulk S2G stores (BN_ASYNC, storer thread in the producer WG) (m7)
- Outputs staged over the consumed stage (y over s, x over g), fence.proxy.async + per-stage obar, storer
  issues cp.async.bulk S2G, waits .read, releases the stage (arrive count 8). Bit-identical to c2/c3 outputs.
- Fused per-row: UB 1.78 us (base 1.93), D6 1.26 (1.33). But UB GEMM tiles +1.4 us (70.8 vs 69.3: the epilogue
  wp1 loads for the row sums) -> UB kernel 1821 vs base 1777 (worse); D6 613 vs 616 (noise). Stage held through
  the store => load latency exposed (c4 UB no-store 1.63 vs c2 1.31): only 3 x 64 KB stages.

## m8: knob sweep (base kernels, 30 paired reps, noise ~ +-5 us D6 / +-15 us UB)
- D6 (base 605.5): tail 2.1.1 605.7, tail 1.1.1.1 605.5, R8 612, tail+R8 614, bcap16 604, bcap64 620, G2 598.9,
  c4+tail 603. UB (base 1891, hot GPU run): R8 1911, bcap12 1898, bcap24 1903, bcap32 1881, tail1111 1900, G2 1918.
  => the existing knobs are at a flat optimum; nothing outside noise.
- L2 prefetch of an item's later rows (cp.async.bulk.prefetch.L2, c0p / c4p): UB +100 us, D6 +53 us. Rejected.

## Probe data note
- Random N(0,.5) x N(0,.02) data draws more power than real activations: cuBLAS pooler shape 1574 us here vs 1337
  in situ; UB shape 1438 (765 TF/s). Our ub2 GEMM-only (1434) = cuBLAS on this shape: no GEMM-efficiency lever.
- Late runs got noisy (hot GPU / shared): probe now prints the median paired difference vs the first variant.

## c5/c6 (m9-m11): cw in registers, 1 + wpost in shared memory for the UB epilogue row sums
- c5: UB rows 1.73 us (base 1.93-1.96) but GEMM tiles still +1.2-1.5 us (70.6-70.9 vs 69.3); trace ablation:
  no sums 69.7, sums without the wp1 multiply 70.2. Net UB kernel = base (1841 vs 1832 m9; 1920 vs 1919 m10).
  D6 c5: 648 vs 659 (noisy run), rows 1.28.
- c6 (sums interleaved into the store loop): tiles slower still (UB 71.1, D6 19.4): paired UB +38 us, D6 +13.5.
- Variable-R tail items (BN_TAILR rows for the last BN_TAILK m-tiles to complete): on c6 UB tail 26 -> 14 us,
  paired -13 us vs c6; D6 (R4, K4) CTA end spread 41 -> 8 us, idle 5.4 -> 2.6%.
- => The BnPlan path (epilogue row sums) costs the GEMM tiles about what it saves on the boundary rows. Keep the
  incumbent row math; try the tail items alone on the incumbent kernels (c7, tail_patch.py: bit-identical rows).

## c7: incumbent kernels + variable-size tail items (tail_patch.py), bit-identical outputs (m12-m14, 30-40 pairs)
- UB, last m-tile (K1) as 4-row items (R4): paired -25 / -16 / -20 us (29/30, 33/40, 34/40 pairs faster); tail
  after the last GEMM tile 29 -> 13.6 us, idle 3.1 -> 1.9%. R4K2 -22, R8K2 -14, R2K1 -18; with tail 2.1 -23,
  bcap 24 -22, R32+R8K2 -23: all within noise of R4K1. Keep the incumbent knobs + R4K1.
- D6: R4K4 -3, R8K4 -5, R4K8 -6, tail 2.1.1 + R4K2 -8 / -6; G2 + tail 2.1.1 + R4K2 -14.7 (34/40), -19.0 (38/40);
  G2 + R4K2 -19.5, G1 -8.6, R32 same as R16.
- Chosen (bnd_tail.py via make_tail.py): UB R4K1 (G4 R16 tail 2.1.1 bcap16 unchanged), D6 G2 R16 bcap32 tail
  2.1.1 + R4K2. Kernel-level expectation ~ -20 (UB) -19 (D6) = ~ -39 us per 16K call.

## Replica rep1 (cases 4-7, panel 11,22, 6 reps): bnd_tail vs base
- base x1.9787 / 1.9759, tail x1.9762 / 1.9633 -> RELATIVE 0.9987 / 0.9936 (mean 0.9962): WORSE in situ.
  Per case (us): c4 -19 / -15, c5 +4 / +60, c6 +58 / +78, c7 +68 / +112. Contradicts the kernel-level probe
  (random data, -39 us). Next: nsys node trace in situ (prof.sh) to see the two kernels' real durations.

## In situ (insitu.py: nsys node trace of the whole-layer graph, real inputs, fields seed 11, 8 reps, base/tail
## alternating first/second)
- case 5: ub2_2_192_3 1730.6 -> 1716.9 us, ub2_2_192_4_4 574.7 -> 573.4; span 8921 -> 8899.
- case 7: ub2_2_192_3 1724.0 -> 1716.4, ub2_2_192_4_4 586.4 -> 574.8; span 12115 -> 12115.
- => the tail items help in situ too (-9 to -25 us per call), smaller than on random data.

## REPLICA POSITION BIAS (important for everyone using run_exp.sh with >1 candidate)
- replica.py orders graphs [sglang, cand1, cand2] on even reps and reversed on odd reps: cand1 is ALWAYS in the
  middle, cand2 alternates first / LAST. The last of three back-to-back 10-12 ms graphs runs slow (power/thermal):
  rep1 raw data, case 7: tail-base = +213/+367/+426 us when tail ran last (reps 0,2,4) and -75/-58/-78 when it ran
  first; base (always middle) was order-insensitive. Same pattern in cases 5, 6 (smaller in 4).
- So rep1's "tail 0.9962 vs base" is an artefact of listing tail second. Fair A/B: swap the order in a second
  run and average, or run each candidate alone against sglang.

## rep2 (order swapped: tail listed first) and the order-averaged estimate
- rep2: tail x2.0111 / 2.0047, base x1.9928 / 1.9775 -> base/tail 0.9909 / 0.9865 (now biased the other way).
- Order-averaged (geomean of rep1 tail/base and rep2 tail/base per case): +0.39% on the first-turn load
  (per case -30..+109 us; case 5 +30 / +36 us), i.e. ~+0.13% on the score. Consistent with nsys in situ.

## Gate: gate.sh research/claude/bnd/bnd_tail.py -> CHECK_RC=0 PASS=42 CASES=21 (gate_tail.log)

## DELIVERABLE: research/claude/bnd/bnd_tail.py (801,474 bytes; built by make_tail.py from b23_compact.py)
- UB (unpooler + post-FFN boundary): incumbent knobs, the last m-tile to complete as 4-row items.
- D6 (o_proj + post-attention boundary): G=2, tail=(2,1,1), the last 2 m-tiles as 4-row items.
- Row math untouched: outputs bit-identical to b23 at kernel level (probe checks rel 0.0).

## Bound summary / why the boundary itself was not made cheaper
- Fused minus GEMM-only: UB ~+320 us, D6 ~+200 us per call (random data). Boundary SM-time 240 / 165 us per CTA.
- Hard floors per SM: 32 B/clk write port (63 GB/s at boost, ~51 GB/s at the fused clock) -> UB rows (48 KB out)
  >= ~0.94 us, total >= ~117 us SM-time per call; loads need 3 x 64 KB of smem in flight with ~3 us latency under
  the GEMM's L2/HBM load. Lean FP32 math with epilogue row sums halves instructions but the rows are memory-
  pipeline bound (1.93 -> 1.73 us/row best), and the epilogue sums cost the UB GEMM tiles ~1.2 us each (~+25 us),
  cancelling the gain. Async bulk stores lengthen stage residency (load latency exposed). L2 prefetch hurts.
- Remaining ideas (not done): ping-pong consumer WGs so tile epilogues (96 KB stores, ~1.9 us/tile at the write
  port; ~10% of a D6 tile) overlap the other WG's mainloop; a D6-only BnPlan (its epilogue sums were free) for
  maybe -10 us more.

# Follow-up (coordinator): ping-pong consumer warpgroups, D6-only row sums. Base is now research/claude/stack/s1.py.

## Epilogue cost and D6-only BnPlan (m15)
- GEMM-only with the tile epilogue stores removed (s1e.py, UB_NO_EPI_STORE): UB tile 66.5 -> 64.5 us (kernel -24
  us paired), D6 tile 17.97 -> 16.85 (-19 us). This is the most a hidden epilogue can give.
- D6-only BnPlan on s1 (d6bn.py: epilogue row sums, one reduction per row, async stores): 643.6 vs 622.3 us
  (+21): with s1's G=2 / tail D6 rows are already 1.19 us/row, BnPlan 1.28; tiles 19.2 vs 18.8. Rejected.

## Ping-pong GEMM-only prototype (pp_probe.py; 128 x 192 tiles, one consumer WG each, static schedule)
- Needs an order barrier: the WG of tile i+1 must not wait on ring stages until tile i's k-blocks have landed,
  else mbarrier parity waits alias (first version: launch failure at the producer's arrive.expect_tx).
- Bit-identical to cuBLAS. UB: 1409 vs cooperative 1418 us (-11; NS 4/5 the same). D6: 360 vs 392 (-32).
  Register budget is not the issue (each WG keeps its 192 accumulators). Smem traffic per FLOP rises (B loaded
  for 128 rows instead of 256: 135 vs 120 B/clk), so the hidden epilogue barely pays at K=4096.

## Fused ping-pong (pp_main.cu + gen_pp.py -> _PP_BODY; make_pp.py -> pp1.py; PpPlan)
- 128-row GEMM tiles alternate WGs; boundary items stay joint (incumbent row code, 8 warps); both WGs read every
  queue item and track the ring; empty barrier count 8 (a GEMM warp arrives with 2); a fetcher thread claims
  items (the TMA thread never waits on claim atomics: a 128-row tile drains the ring twice as fast, so the
  incumbent's in-line claim exposed ~1-2 us per tile, GEMM-only 408 -> 379 us after the split); a publisher
  thread takes tile-done posts from both WGs (slot by shared atomic). D6 outputs bit-identical to s1.
- D6 result (30 paired reps, vs s1): NQ 4 +22..+28 us, NQ 2 +3..+7, bcap 16/48/64/96 no better.
- Trace accounting (per CTA, pp_trace_an.py): s1 = mainloop 391 + boundary 157 + rest 18 = 566;
  pp = mainloop-union 374 + boundary 209 + rest 13 = 596; pp GEMM-only = 355 + 6 = 361 (s1 GEMM-only 384).
  => the GEMM saves ~20-25 us, but every joint boundary item costs ~6 us more (the second WG must finish its
  in-flight tile and epilogue before the rows start, and the ping-pong restarts cold after each item):
  ~8.5 items per CTA ~ +50 us. A single WG cannot take boundary rows alone: 64 columns per thread x 4
  coefficient vectors = 256 registers > 232, and there is no shared memory left for coefficients.
- Bigger joint items to amortise the join (m22/m23): R 32 cuts the boundary time (209 -> 169 us per CTA) but the
  coarse items lengthen the end (rest 13 -> 85 us); with 4-row tail items over the last 8/16 m-tiles: +47 / +21 us
  vs s1; NQ 2 R16 +22 (this run). No ping-pong variant beat s1 (best +3..+7 in one run, +21..+28 in others).
  Caution: items encode rows in 6 bits (rows << 16, & 63): R must stay <= 63 (R 64 hung the probe).
- Conclusion: ping-pong pays only in GEMM-only form (D6 -24..-32 us, UB -11 us). In the fused kernels the boundary
  items must be joint (both WGs: register budget), and each join costs about as much as the hidden epilogues
  save. UB's GEMM-only gain is ~10 us, so its fused version would lose more. Not delivered; s1 stays.
