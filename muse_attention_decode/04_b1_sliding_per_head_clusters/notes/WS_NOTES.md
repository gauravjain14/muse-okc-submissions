# WS: decode sliding batch 1 (runA1, cluster of 16) — base dev/k11.py

Files: k11_base.py (= dev/k11.py, md5 a23992ec), k15s.py (work file / deliverable), trace.py (all 32 stamp slots,
--percta, --raw for TRV slots, --mode warm), t_c16.py (extra B=1 sliding cases: speculation misses via spread
mixed / agentic, short contexts, page 64, kv_scale, 3 calls each), job*.sh (Modal jobs), k11_diag*.py (diagnostics).

## Deliverable: dev/ws/k15s.py = dev/k11.py + runA1h (dev/ws/runa1h.cu), built by dev/ws/build_k15s.py
(runA1 / run1 / run8 / FLAT byte-identical to k11; only new kernels attn_1_RP_P1_1s + their host selection).
Development file with all variants: dev/ws/k15s_dev.py (runA1<..., C16> tails T1/T2, runA1s, runA1h knobs).
runA1h (decode sliding batch 1, window <= 2048, B * hkv clusters of 16 co-resident; else k11's _1c):
- one cluster of 16 CTAs per KV head (grid y = hkv): 64 B of each K/V row per CTA (16 KB per SM; a CTA's gather
  is bound by its SM's memory parallelism ~30 GB/s), one-head tiles (head's 64 B in the first half of the 128-B
  swizzled smem rows: head-0 descriptors);
- split C = ceil((past + 1) / 16) per CTA from the window start: <= 128 tokens = 2 tiles per CTA, the new token folded
  into the last CTA's tile slot after its past tokens (no extra tile, no straggler);
- warp specialization: warps 8-15 = producers (length + guessed page-table entries loaded in the straight-line prefix
  by predicated asm loads, K/V cp.async issued as soon as the entries land, the per-tile mbarrier completed only after
  the guess is confirmed; miss: drain own copies, producer barrier, true entries, re-issue); warp 12 of the last CTA
  writes the new token's e4m3 K/V (cache row + tile slot); warp 14 inits the mbarriers before any load;
- lane 0 (warps 0-3) attends both tiles (dec_tile_h2: one max exchange, one PV chain); warp 0 builds the RoPE table
  from its own length load and publishes it with one barrier; cluster phase-1 wait right after that barrier;
- tail: lane 0 pushes the CTA's 8 partial rows from registers by st.async to owners CTA 0..7 (16 x 288 B each), owner
  warp 0: M by 4 shuffles, weights by shuffle broadcast, L and sum(y^2) reduced side by side
  (rmsnorm(y vsc / L) = y rsqrt(mean(y^2) + eps (L / vsc)^2)), precomputed sigmoid(gate), bf16x2 store.

## Log
- job1 (baseline k11, flush, median us): seq-iss 0.35, spec-iss 0.77, len-bar 1.35, kv-iss 1.96, q/new 2.48,
  tile0 2.69, tiles 3.62, mstore 4.07, clwait 4.17, pushed 4.54, recv 4.83, comb1 5.30, out 6.02. warm: tail
  (tiles -> out) 2.52 us vs flush 2.40: not L2-bound.
- job2/3 diagnostics: one TR stamp = 80 cycles; k11's mO phase (60 instrs) = 686 cycles; executing the mO phase
  twice: pass 1 658 / pass 2 644 cycles, combine+epilogue pass 1034 / 919 cycles -> NOT instruction-fetch bound
  (latency chains: S2R, dependent shuffles, barriers, divisions).
- job3: k15s v1 (C16 tail + SPECTOP, tid-0 init): t_c16 0 failures, check b1 sliding 0 failures.
  bench decode-1 sliding 2 x 40 reps: k11 5.65 / 5.68 | k15s 4.53 / 4.52 | SPECTOP=0 4.60 / 4.60.
  trace: len-bar 1.28, kv-iss 1.89, tile0 2.66, tiles 3.65, merged 3.84, pushed 4.03, recv 4.38, out 4.90
  (tail 1.25 us, was 2.40). CTA 15 (new-token part) is the straggler (tiles 3.81, pushed 4.26).
- job4 (v2: init by tid 448 + length first + cluster arrive after the loads, WS_FRESH, WS_DST=3): 2 x 40 reps:
  k11 5.65 / 5.69 | k15s 4.62 / 4.59 | FRESH=0 4.62 / 4.56 | SPECTOP=0 4.51 / 4.50 (v1 job3 4.53 / 4.52).
  FRESH neutral. trace: len-bar 1.15, pre-iss 1.41 (hit check etc. 0.26 us), row0 landed 1.47, kv-iss 1.82 (issue
  code 0.35 us), q/new 2.40, tile0 2.67 (K/V trip ~0.85 us), lanes done 3.55, out 4.90. The new tile (lane 2 of
  CTA 15) also takes ~1.0 us -> one dec_tile ~0.9-1.0 us.
- job5 (T2 = lanes 0/1 push their own partials, 32 per owner, no CTA barrier; I32 issue; WS_DST=2): 2 x 40 reps:
  k11 5.75 / 5.83 | T2 4.76 / 4.74 | T1 4.48 / 4.53 | SPECTOP=0 4.81 / 4.64 | I32=0 4.74 / 4.81.
  T2 LOSES 0.25 us: pushed -> received 0.51 us (T1 0.36): 2x the st.async bytes (8.8 KB per owner) is slow.
  Tile sub-stamps (cycles from dec_tile_ws entry, trace build): QK done 342, max exchange 616, make_p + fence 1184,
  PV done 1412 (the proxy fence after P^T looks expensive; trace stores may inflate it).
- job6: runA1s (WS_WSP: warp-specialized start + fold split + T1 tail): t_c16 0 failures. 2 x 40 reps:
  k11 5.84 / 5.77 | wsp 4.09 / 4.10 | runA1-C16-T1 4.60 / 4.60. trace: len-bar 0.99, kv-iss 1.60, tile0 2.19,
  lanes done 2.95, merged 3.14, pushed 3.34, recv 3.74, out 4.16.
- job7: check.py k15s (wsp default) 0 failures (all 28 cases x 2 calls). WS_INC (online per-source mbarrier fold)
  passes but LOSES badly (5.34 / 5.53 vs 4.09 / 4.17): polling 16 mbarriers with test_wait.acquire.cluster is slow
  (pushed 3.62 -> recv 5.44). DEAD END. Producer stamps: page-table load issued 0.61 us (cold branch target +
  constant lines), landed 0.93; length 0.99; verified 1.13; K/V issue loop done 1.70 (0.57 us for 64 LDGSTS.128 =
  32 KB per SM: the per-SM gather rate ~30 GB/s = 4.3 TB/s / 132 -> per-SM memory parallelism bound).
  => next: per-KV-head clusters (2 x 16 CTAs, 16 KB per SM) - runA1h.
- job8: runA1h (WS_PH: one cluster of 16 per KV head, grid y = hkv, one-head tiles dec_tile_h1, 8 owner CTAs per
  cluster, 16 KB of K/V per SM) = dev/ws/k15s_j8.py: t_c16 0 failures, check.py 0 failures. 2 x 40 reps:
  k11 5.62 / 5.60 | ph 3.21 / 3.13 | runA1s 3.82 / 3.79 | runA1s+V4 3.85 / 3.90 | runA1s+HALF 3.87 / 3.83.
  (V4 = 16-B pushes via lane-pair swap, HALF = lanes 0/1 each fold+push half the rows: both neutral, dropped.)
  trace (ph): pt-iss 0.67, len-bar 0.92, verified 0.99, kv-iss 1.38, tile0 1.98, lanes done 2.43, merged 2.62,
  pushed 2.78, recv 3.01, out 3.39.
- job9: (2 x 40) j8 3.42 / 3.46 | + hoisted first loads (predicated asm loads in the straight-line prefix: plain
  loads were sunk into the role branches, whose code is a cold fetch) 3.39 / 3.41 | + WS_KVS (producers issue the
  speculative K/V as soon as the guessed page-table entries land, verify after; miss: drain own copies, producer
  barrier, re-issue) 3.28 / 3.36. t_c16 0 failures for both.
- job10: (2 x 40, 2nd repeat noisy +0.3 for all) hoist 3.40 / 3.72 | kvs 3.25 / 3.55 | WS_ONE (lane 0 attends both
  tiles: dec_tile_h2, one max exchange, PV chain over both tiles, no lane merge) 3.16 / 3.46 | ONE+KVS 3.21 / 3.40.
  trace ONE+KVS: kv-iss 1.26, tile0 1.82, tiles done 2.34, pushed 2.58, recv 2.78, out 3.17.
- job11 (3 x 40 reps, same job): k11 5.77 / 5.75 / 5.77 | ONE 3.50 / 3.43 / 3.38 | ONE+KVS 3.34 / 3.28 / 3.31 |
  ONE+KVS+ECW (consumers' cluster phase-1 wait moved to right after the length barrier: off the post-tile path)
  3.16 / 3.13 / 3.16 | + EVF (L2::evict_first on the K/V cp.async) 3.32 / 3.26 / 3.35 (EVF LOSES +0.17).
  -> defaults WS_ONE=1, WS_KVS=1, WS_ECW=1, WS_EVF=0.
- job12 (dev defaults ONE+KVS+ECW): check.py 0 failures. trace: prefix loads issued 0.42 (constant-line misses
  after the flush), guessed entries land ~1.05, K/V issued 1.26, tile0 1.79, tiles 2.30, pushed 2.54, recv 2.75,
  out 3.12. Start chain = constant trip + page-table trip + K/V trip (3 DRAM trips ~1.7 us) is near its floor.
- job13 (FINAL dev/ws/k15s.py built from dev/k11.py, md5 below): image SASS: 30 kernels, HMMA 0, F2FP.F16 0, QGMMA 344
  (attn_1_*_1s: 1624-2008 instrs, 12 QGMMA; run1 attn_1_1_1_1 SASS byte-identical to k11). t_c16 0 failures,
  check.py 0 failures. decode-1 sliding 2 x 40 reps: k11 5.73 / 5.77 | k15s 3.15 / 3.28 | dev 3.23 / 3.33.
  decode loads (20 reps, 1 sample): d1 sliding 6.05 -> 3.36 (noisy sample), d1 full 14.40 / 14.14, d16 sliding
  8.91 / 9.15, d16 full 87.95 / 88.19, d32 sliding 10.92 / 11.10, d32 full 166.52 / 166.91 (non-b1 shapes run
  identical code: noise); decode geomean x2.5245 -> x2.7691 (judge-proj x2.377 -> x2.577).

## Dead ends / did not help (numbers above)
- T2 (both lanes push their own partials, 32 per owner): +0.25 us (DSMEM bytes x2 -> receive 0.51 vs 0.36 us).
- WS_INC (owner folds per-source mbarriers online, polling test_wait.acquire.cluster): +1.2 us.
- 16-B pushes via lane-pair swap (V4), split-row push by both lanes (HALF): neutral.
- FRESH first tile in the old runA1 path: neutral there (kept implicitly: runA1h tiles are always fresh).
- L2::evict_first on the K/V cp.async: +0.17 us.
- Instruction-fetch hypothesis for the old tail: running a phase twice gave the same cycles (latency chains instead).
2f66acdc0639d5e7120f15dde4ae55ed  dev/ws/k15s.py
a23992ec5c7f8a24bec39c164374d0c6  dev/k11.py
99580e7368cb5fb9782e6dc889a86f2c  dev/ws/runa1h.cu

## Round 2 (coordinator): verify sliding batch 1 (q_len 8) -> dev/ws/k17s.py = dev/k17.py + dev/ws/runa8h.cu
Builder: `python3 dev/ws/build_k17s.py [dev/k17.py] [dev/ws/runa8h.cu] [dev/ws/k17s.py]`; kernels attn_8_1_P1_1h
(runA8h), used for B == 1, q_len 8, sliding window <= 2048 when the q_len-8 plan picks a 16-CTA cluster and hkv
clusters of 16 are co-resident (else WB's _1r / _1c). K2_DEFS WS_V8=0 restores k17. Own tile function (ws8_tile; not
run8's wg_tile). Extra tests: dev/ws/t_v8.py (13 q_len-8 B=1 sliding cases incl. misses, page 64, edge windows).
Design: one cluster of 16 per KV head; split C = ceil(past / 16) -> <= 2 tiles per CTA, WG0 tile 0 / WG1 tile 1
(fresh tiles: the row softmax stays inside a quad, no max exchange); WG2 + WG3 producers (prefix loads, speculative
K/V issue, mbarrier per tile); WG3 of the last CTA builds + attends the 8-token new block; WG0 / WG1 swap row halves
(+ the new block's) and push their 32 rows each from registers (owner CTA o: token o >> 1, q-heads 4 (o & 1)..+3);
owners = WG2's 4 warps (one row each, runA1h's one-warp combine).
- job14: first build NaN in some calls: WG3's 20 KB exchange area ran into WG0's V^T buffer (last CTA only).
- job15 (fixed: 4 stages): t_v8 0 failures (13 cases x 2 seeds x 3 calls), check.py --match "q_len 8" 0 failures.
  verify-1 sliding 2 x 40: k17 8.47 / 8.40 | k17s 4.87 / 4.96 | WS_V8=0 8.54 / 8.64.
  trace: tile0 2.25, tiles done ~3.1 (one tile 0.85 us), folded 3.47, pushed 4.05 (16 KB per CTA by st.async).
- job16 (2 x 40, same job): k17 8.53 / 8.49 | v8 (st.async push) 4.88 / 4.90 | WS8_BULK (rows staged per owner,
  one 1088-B cp.async.bulk per owner) 4.76 / 4.77 | WS8_EXQ (W9's FFMA exp-to-E4M3 in ws8_tile) 4.90 / 4.89 (neutral)
  | BULK + EXQ 4.77 / 4.75. -> WS8_BULK=1 default, EXQ off. smoke 0 failures for all three.
  trace (BULK): tile0 2.14, tiles done 2.8-3.0, folded 3.30, pushes issued 3.81, owners received 4.59 (DSMEM 17 KB per
  owner ~0.8 us), out 5.01. Remaining levers: DSMEM volume (fp32 partials, 16 KB per CTA), tile 0.85 us, start 2.1 us.
e4de6d0b574f2a45d3548342f8dd34ef  dev/ws/k17s.py
a176b5c6d76aafded4a8bcd9a0eb96a8  dev/ws/runa8h.cu
402ab5ff05c6c5b5f2b2b16def5cc5f4  dev/k17.py
- job17 (FINAL round 2, dev/ws/k17s.py md5 e4de6d0b): check.py 0 failures, t_v8 / t_c16 0 failures, HMMA 0.
  verify-1 sliding 2 x 40: k17 8.64 / 8.75 | k17s 5.01 / 4.99 | k17 dup 8.71 / 8.71. Verify loads (20 reps):
  geomean x2.039 -> x2.262; b16/b32 kernels SASS-identical (one constant-bank offset). Coordinator: integrated as
  dev/k19.py; RULING: exchanged partials stay FP32 (no bf16).

## Round 3 (coordinator): verify sliding B > 1 (b16 / b32, run8 ALIGNED global path) on dev/k19.py
- WB (msg): no duplicate K/V bytes (b32 16.8 MB, b16 8.4 MB; floors 3.9 / 2.0 us); part 0 gets the extra tile
  (b16: 5 tiles = 3 lane steps, b32: 9 = 5 steps); tile phase ~1.25 us per lane step at b32; per-warp cp.async
  in-flight cap ~13 KB (WE).
- job18 (k18 traces, flush, med / max us): b16 issued 2.40, q/new 3.42, tile0 4.26, tiles 6.51 / 7.92, seg end 7.45,
  publish 8.12, first landed 8.87, combined 11.25 / 11.89. b32: issued 2.69, tile0 4.52, tiles 9.46 / 10.78,
  combined 13.62 / 14.37.
- run8g (dev/ws/build_k20s.py: verbatim copy of run8 + 2 edits, kernels attn_8_1_P1_1g): (1) the past tiles split
  evenly in whole tiles (b16 4 / part, b32 8 / part), the new block extra in the last part; (2) the new block's tile
  attended first by the lane with fewer ring tiles (it needs no gather). Expected: one lane step less on the
  slowest CTA (~1 us).
- job19 (run8g on k19): t_v8b (11 B>1 q_len-8 sliding cases incl. mixed / agentic / page 64 / small contexts / dp)
  0 failures, check.py --match "q_len 8" 0 failures. 2 x 40 reps (k19 | k20s split+nfirst | split only | k19 dup):
    v16 sliding 11.91 | 11.90 | 11.76 | 11.93  and  11.83 | 12.04 | 12.02 | 11.94
    v32 sliding 14.38 | 14.35 | 14.21 | 14.34  and  14.75 | 14.86 | 14.77 | 14.86
  -> NO GAIN (within noise). Trace b16: tiles done med 6.50 / max 7.79 (k18 6.51 / 7.92), combined 11.36 / 11.87:
  removing part 0's 5th tile step does not move the end: the tile phase is bound by the gather stream (all CTAs
  finish within ~1.3 us of each other), and the end chain (~4.5 us: merge, publish, release, MAL fold) dominates.
  Coordinator then moved b16/b32 to another worker. Files: dev/ws/build_k20s.py, dev/ws/t_v8b.py (reusable tests).
  Note for the next owner: dev/wb/regs.py is gone from the tree (spill checks: dev/wd/regs.py).

## Round 4 (coordinator): decode full batch 1 (run1 ALIGNED, 132 CTAs, P = 132) on dev/k19.py -> dev/ws/k20d.py
Builder dev/ws/build_k20d.py: run1w = verbatim copy of run1 (kernels attn_1_0_P1_1w, host: ALIGNED no-RoPE q_len-1
launch; K2_DEFS WS_B1F=0 restores k19). Extra tests: dev/ws/t_b1f.py (10 full b1 cases: coherent / soft, page 64,
mixed / agentic spreads (speculation misses), kv_scale, ctx 10 / 3000 / 20000 / 40000 / 100000, dp heads).
- job20 (WS_B1M: register lane merge (lanes 1-3 fragments -> lane 0 folds 3, partial written from registers) +
  compact one-row combine (max by 1 shuffle + smem, weights from the final max, L and sum(y^2) side by side,
  precomputed sigmoid)): t_b1f / check 0 failures, but 15.02 / 15.14 vs k19 14.07 / 13.97 (+1 us). Trace: tiles 8.35
  -> merged 9.57 (lane-0 fold of 3 lanes + scattered 8-B partial stores 0.9 us vs k19's 2-pass 0.35 + 0.35),
  released 11.80 -> loads 13.59 (1.8 vs 1.28), epilogue 15.06. k19 trace: issued 3.67 (2.77..4.75), Q/new 4.06,
  tile0 4.46, tiles 8.20 / 8.91, merged 9.02, publish 10.02 / 10.70, released 11.34, loads 12.62, fold 13.26,
  combined 14.02. -> WS_B1M off.
- WS_B1A (fixed-reference accumulation, testing in job21): every CTA scales its merged partial by 2^(m - 64) (scores
  of RMS-normalized q / keys are bounded by 63.2 in these log2 units) and red.add.v4.f32's it into a zeroed fp32 sum
  (counters buffer extended by 8 x 1088 floats); the last arrival (acq_rel counter) reads 4 KB and runs the epilogue
  (one warp per row), then clears the sum. A partial max outside [REF - 90, REF + 40] sets a flag -> the classic
  single-combiner path (correct for any input). Replaces 16 combiners x 132 partial loads + fold.
- job21 (WS_B1A, merged partial red.add'ed, flag -> classic single combiner): t_b1f 0 failures (2 seeds).
  2 x 40: k19 14.26 / 14.10 | k20d 13.27 / 13.49 | k19 dup 14.05 / 13.98  -> about -0.75 us.
- WS_B1R (job22): no lane merge at all: right after the tile loop one CTA barrier votes (bar.red.or) whether every
  lane's max is inside [REF - 90, REF + 40]; if so each thread red.add's its scaled O pairs and its warp's l
  (3 shuffles) into L slots [row][lane, warp] (16 slots: low contention); otherwise the CTA runs the classic merge,
  writes its ws partial, sets the flag and appends itself to a list; the last arrival folds the listed partials
  into the in-range sum (m = REF) - correct for any input (forced with WS_REF=500 in the smoke).
- job22: t_b1f 0 failures (B1R, 2 seeds), forced fallback (WS_REF=500) 0 failures for B1R and B1A. 2 x 40:
  k19 14.08 / 14.23 | B1R 23.42 / 23.58 (!) | B1A (merged accumulation) 13.07 / 13.20 | k19 dup 14.11 / 14.05.
  B1R trace: tiles med 12.7 / max 15.95 (k19 8.2 / 8.9): 132 x 512 threads of red atomics from registers collide
  with the gather stream -> DEAD END. B1A (one 256-thread burst of v4 reds from the merged partial) = -0.9 us.
- Coordinator split (after job22): the END chain belongs to WB (unified reference + fp32 accumulation, porting to
  run1); WS keeps the START chain / tile phase. B1A results + code handed to WB (dev/ws/build_k20d.py, WS_B1A).
- START chain (my part after the split), dev/ws/build_k21d.py (run1x, kernels attn_1_0_P1_1x, base dev/k20.py):
  WS_XSPEC = page-table rows of the prologue stages for the guessed length loaded before the length barrier
  (burst right after it on a hit); WS_XPRO1 = only stage 0 at the barrier, stages 1-2 after Q^T / new token;
  WS_XWARM (dry dec_tile x2 on a zeroed stage before the barrier) is USELESS as built: the dry call is a second
  inlined copy of dec_tile (6496 vs 5960 instrs), so it warms other instruction addresses -> off.
- job23 (2 x 40): k20 14.31 / 14.18 | spec 14.43 / 14.48 | spec + PRO1 14.56 / 14.46 | k20 dup 14.11 / 14.20 -> no gain.
  Traces: spec "issued" (after all 3 stages) 3.87 (k19 3.67): issuing 12 stages / CTA at once stalls the issuing
  warps ~1.5 us (LDGSTS throttled by outstanding requests). PRO1: stage 0 issued 2.40 (2.27..2.50, tight), tile0
  4.60, yet tiles end 8.58 / 9.10 as in k20 (8.20 / 8.91): the tile phase END is set by the 20.5 MB gather stream
  (~3.1 TB/s effective from the first issue at ~2.4 to ~9.0), not by when the first copy is issued. A start-chain
  change can only pay if it raises the effective gather rate (probe: 20.5 MB in 5.44 us = 3.8 TB/s).
fff93539cb71b863a42b122fc7ebc73c  dev/ws/k20d.py
0f3b970bd66243d98a6d0230a7d57ff7  dev/ws/build_k20d.py
- job24 (dev/ws/k20d.py = k20 + run1w WS_B1A, built by `python3 dev/ws/build_k20d.py dev/k20.py dev/ws/k20d.py`):
  check.py 0 failures, t_b1f 0 failures (2 seeds), forced fallback (WS_REF=500) 0 failures. 2 x 40 reps:
  k20 14.11 / 13.94 | k20d 13.28 / 13.35 | k20 dup 14.04 / 13.93  -> -0.7 us on decode full b1. (Local SASS of
  attn_1_0_1_1w: 0 HMMA, 16 QGMMA, 128 regs, stack 24 as run1.) End chain is WB's now: offered as a candidate.

## Round 5 (coordinator): copy-mechanism probe -> dev/ws/probe_copy.py (job25)
Real launch shape: 132 CTAs x 512 threads, 1 CTA/SM (198 KB smem), 4 lanes x 4 warps, 64-token tiles (K [64][128 B],
V at +8 KB), DST-stage ring per lane, page-table indirection (randperm slots), XOR-fold consumer, cold L2 (512 MB
flush) before every CUDA-graph replay, empty graph subtracted (5.5 us). Stamps: first copy issued (CTA min), prologue
stages issued (per-SM median), last stage landed (max). Variants: a3 / a2 = run1's cp.async 16-B mapping, 3 / 2
stages (4 x 4 lanes x 16 KB does not fit); b16 / b8 / b4 = cp.async.bulk 128-B rows + full/empty mbarriers, issued
by 16 / 8 / 4 warps; t16 / t8 / t4 = TMA 2D 1 x 128-B boxes with SWIZZLE_128B (rows land in exactly the cp.async
swizzled layout: checked equal for 14..64 rows); m16 = cp.async K + bulk V. Graph time us (TB/s); "issue" = per-SM
prologue issue time (us):
  size          a3            a2            b16           b8            t16           m16
  0.5 MB/132    2.13 (0.23)   2.13 (0.23)   1.94 (0.26)   2.26          2.03          2.51
  0.5 MB/32     1.39 (0.36)   1.47          2.34 (0.21)   3.17          2.34          2.48     (b1 sliding shape)
  8.4 MB        3.38 (2.49)   3.39          3.95 (2.13)   4.00          3.95          3.74
  16.8 MB       5.26 (3.19)   5.20 (3.23)   6.37 (2.64)   6.78          6.42          6.06
  20.5 MB       6.43-7.10     6.43-6.50     7.90-8.02     9.54          8.10-8.19     7.46-7.70
                (3.2 TB/s, issue 4.8)       (2.6, issue 6.9)            (2.5, issue 7.0)
  328 MB        80.2 (4.09)   80.5 (4.07)   97.5 (3.36)   112.7 (2.91)  99.9 (3.28)   90.3 (3.63)
  (b4 / t4: 1.0-1.6 TB/s everywhere.) First-issue -> last-landed spans equal the graph times within ~0.1 us.
Verdict: the cp.async 16-B ring wins at every size that matters; bulk / TMA per 128-B row are issue-bound (one row
per instruction: prologue issue 6.9-7.0 us vs 4.8 at 20.5 MB) and 20-30% slower; more issuing warps help them but
never reach cp.async. Only 0.5 MB on 132 CTAs (not a real shape) favours b16 by 0.15 us; on the real b1-sliding
grid (32 CTAs) cp.async is 1.4 vs 2.3 us. => NO tile-layout change. Also: run1's decode-full-b1 tile phase (first
issue ~2.4 -> tiles end ~8.9 = 6.5 us) already equals this launch shape's gather floor (a3 6.4-7.1 us at 20.5 MB);
the older 5.44 us / 3.77 TB/s figure (dev/probe_gather.py) came from different CTA / warp configurations, so a
faster full-b1 gather needs a different launch shape (more resident warps or CTAs per SM), not a different copy.
- dev/ws/probe_wsp.py (FA3-style dedicated-producer probe: 1 thread / 1 warp / 1 warpgroup issuing bulk or TMA rows
  for 3 busy consumer warpgroups, 3-6 stage mbarrier rings, producer-read page-table ring) was written but NOT run:
  the coordinator stood the extra cases down (Round 5's b4 = 1 warp per lane at 1.0-1.6 TB/s already answers it).
  Unvalidated draft; compile-check before use.

## Round 6 (coordinator): launch-shape sweep for the cp.async ring -> dev/ws/probe_shape.py (jobs 26-29)
`python3 dev/ws/probe_shape.py --mb 8.4,16.8,20.5 --reps 30 --cta --only a,a0,a2,e,d,b768,b768w,b1024,c,c6`
(--only also sets the run order; --cta adds per-CTA completion median / max and the CTA start spread). Same workload
as probe_copy.py (cold L2, graph replay minus the empty graph, page-size-1 randperm rows, XOR-fold consumer) but made
faithful to run1 in two places probe_copy.py was NOT:
 1. Tail tiles: probe_copy copied row 0 for tokens past the CTA's range, so every SM hammered one K and one V 128-B
    line (one L2 slice): +0.5-1.1 us at 20.5-24.6 MB, bimodal per container (a: 6.45 / 7.0-7.1 us). That artifact is
    Round 5's "a3 6.43-7.10" and the whole of jobs 26/27's apparent b1024 / b768 win (-9..-16% at 20.5-24.6 MB).
    run1 uses src-size 0 (`2*jj < rem ? 16 : 0`, `ok ? 16 : 0`), and so does probe_shape now. (Real kernels: never
    point padding tokens at a shared row with a non-zero src-size.)
 2. Page-table rows: probe_copy loaded each stage's row behind the previous stage's cp.async (memory clobber ->
    LDG, 16 LDGSTS, LDG, ...). run1 loads r0..r2 before any copy and the refill row before cp_wait; probe_shape does
    the same (SASS: 3 LDG then 48 LDGSTS) and keeps the old order as `a0` (prologue issue 4.74 vs 2.27 us at 20.5 MB,
    span +0.2 us).
Configs (all 4-warp lanes, run1's (K/V, chunk) x token mapping and 128-B swizzle):
  a  132x512, 4 lanes, 64-tok (16 KB) x 3   | a2 same x 2 | e = a, stage 0 of every lane, __syncthreads, stages 1-2
  b768 132x768, 6 lanes, 32-tok x 3 | b768w 132x768, 6 lanes, 64-tok x 2 | b1024 132x1024, 8 lanes, 32-tok x 3
  c 264x256 (2/SM), 2 lanes, 64-tok x 3 | c6 264x256, 32-tok x 6 | d 132x512, 4 lanes, 32-tok (8 KB) x 6
job29 (faithful; one container, 2 passes fwd / rev, 30 reps each). Graph us pass1 / pass2 (globaltimer span
first issue -> last landed), then per-CTA time-to-first-landing (CTA start -> first stage landed, median) at 8.4 /
16.8 / 20.5 MB:
  cfg    8.4 MB             16.8 MB            20.5 MB            first landing      issue@20.5  done med/max@20.5
  a      3.12/3.39 (3.30)   5.20/5.14 (5.12)   5.90/6.16 (5.98)   2.59 / 2.83 / 2.85   2.27        5.60 / 5.89
  a0     3.25/3.36 (3.30)   5.25/5.12 (5.20)   6.14/6.27 (6.17)   2.36 / 2.94 / 2.99   4.74        5.70 / 6.06
  a2     3.18/3.42 (3.31)   5.10/5.01 (5.13)   6.22/6.10 (6.16)   2.59 / 2.85 / 2.87   1.31        5.83 / 6.07
  e      3.15/3.42 (3.31)   5.07/5.06 (5.13)   6.06/5.95 (5.98)   2.57 / 2.80 / 2.81   2.30        5.60 / 5.89
  d      3.18/3.39 (3.26)   5.17/5.12 (5.15)   6.02/5.87 (5.98)   1.95 / 2.02 / 2.16   2.46        5.63 / 5.88
  b768   3.39/3.26 (3.30)   5.07/5.15 (5.10)   6.38/6.24 (6.30)   2.01 / 2.08 / 2.17   2.18        5.88 / 6.16
  b768w  3.44/3.34 (3.34)   5.20/5.22 (5.18)   6.03/5.92 (5.88)   2.59 / 3.46 / 3.32   2.24        5.59 / 5.76
  b1024  3.66/3.49 (3.36)   5.23/5.39 (5.15)   6.18/6.02 (5.95)   2.24 / 2.28 / 2.28   2.37        5.54 / 5.79
  c      3.25/3.17 (3.23)   4.93/5.20 (5.06)   5.89/5.89 (5.92)   2.56 / 3.02 / 3.10   1.93        5.46 / 5.77
  c6     3.25/3.10 (3.23)   5.01/5.23 (5.13)   6.08/5.95 (6.00)   2.06 / 2.15 / 2.38   2.12        5.42 / 5.82
  24.6 MB: all 6.72-7.20 graph, spans 6.82-6.98 (a 6.83, c 6.82). Graph times jitter +-0.25 us pass to pass at
  8.4 MB in this container; spans are the steadier metric. Regs (NVRTC 13): 46-72, 0 local, for every config.
Same-job dev/probe_gather.py (16-token per-warp private rings, no lane barriers): ring_8_2 x396 (3 CTAs/SM, 24
warps) 2.88 / 4.85 / 5.68, ring_16_3 x132 3.12 / 4.86 / 6.35, ldg_8 x264 2.99 / 4.64 / 5.92. Its numbers swing
10-15% between containers (ring_16_3 @ 20.5 MB: 5.36 in job28, 6.35 in job29) while `a` holds 5.90-6.16; the old
"5.44 us / 3.77 TB/s" floor was a fast container. Same-job best (ring_8_2) is 4-8% under `a` at 20.5 MB.
Time-to-first-landing: a 64-token first stage lands 2.6 / 2.8 / 2.85 us after CTA start (that includes the cold
page-table read: CTA issue 0.9 us at 8.4 MB); 32-token stages land 0.6-0.7 us earlier (d 1.95-2.16, b768 2.0-2.2,
c6 2.06-2.38). The first-wave split (e) does NOT move it (2.57-2.81). Fewer prologue bytes matter only above 20 MB
(a2 2.87 vs a 3.11 at 24.6 MB). The stream end does not move with any of it.
Budgets for adoption: run1 (attn_1_0_*) 512 thr x 128 regs = the whole RF (stack 24), smem 221.6 KB of 227
(runA1 219.6, runA1h 224.6); run8 full (L=8, DST 4, 2 lanes) 512 thr x <=128 regs, 194.4 KB. b768 / b1024 cap
regs at 80 / 64 (setmaxnreg producer/consumer split needed); c (2/SM) needs <= 113 KB per CTA (run1's layout
does not fit); 32-token tiles (d, b768, b1024, c6) waste half of run1's swap-AB m64 QGMMA (M = tokens = 64).
VERDICT: no configuration is >= 15% faster anywhere in 8.4-20.5 MB. Best spans vs a: c -2% / -1% / -1%, d -1% / 0 /
0, b768w +1% / +1% / -2%. => no follow-up kernel round from launch shape. Round 5's tile-layout verdict stands, and
more strongly: its bulk / TMA modes already skipped tail tokens, so only the cp.async a3 / a2 paid artifact 1
(faithful a: 5.9-6.2 us = 3.3-3.5 TB/s at 20.5 MB). run1's full-b1 tile phase
(~6.5 us in traces, traced builds +0.35-0.4) sits at this floor.

## Round 7 (coordinator): page-table L2 prefetch on FULL FLAT layers (RULING (c)) -> dev/ws/k26p.py (jobs 30-35)
39d0eb62089b8686b99a836f1e14b8fe  dev/ws/k26p.py   (`python3 dev/ws/build_k26p.py dev/k25.py dev/ws/k26p.py`)
8d83f77af0be28584485f8d28a75b871  dev/ws/build_k26p.py   (extra NAME=VALUE args bake variant defaults)
Delivered: run1f (decode full B > 1) entry: warps 1..15 issue prefetch.global.L2 for a 1/(gridDim.x*gridDim.y) slice
of the whole [B][max_pages] int32 page table (b32 ~11 MB = 86K lines, ~650 per CTA), warp 0's length load unaffected
in issue order; nothing waits. run8 FLAT (verify full) is left byte-identical to k25 (WS_PTPK=1; SASS 4624 instrs both).
attn_1_0_1_0: 124 regs, 0 spill, tile loop 598 instrs (unchanged), +48 instrs total; module HMMA 0.
job35 (final, 40 reps x2, us; controls k25 / k25dup): check.py 0 failures.
  decode 16 full  k25 87.40 / 87.27, dup 87.44 / 87.39 -> k26p 86.12 / 86.32   (-1.1..-1.2)
  decode 32 full  k25 166.10 / 166.36, dup 165.81 / 166.22 -> k26p 164.35 / 165.36   (-1.0..-1.7)
  verify 16 / 32 full: run8 unchanged (k26p within the controls' +-0.6 / +-0.6 spread).
Same run1f-only build in jobs 33/34: d16f -0.95..-1.0, d32f -1.0..-1.15; both-kernels build in jobs 30/32: d16f -1.0..-1.05,
d32f -0.9..-1.4. Variants (WS_PTP, all measured with k25 controls in the same job):
  1 t=0 per-line (superset slice)            d16f -1.0  d32f -1.1  v16f / v32f  0 +-0.3   <- delivered (run1f only)
  2 t=0 bulk (warp 1, a chunk per lane)      d16f -0.85 d32f -0.8  v 0
  3 exact own range, bulk, after the split   d16f -0.5  d32f -0.9  v16f +2.4  v32f +3.6
  4 in-loop distance prefetch (2 / 4 tiles) + next segment's first rows (run1f)   d +0.1..+0.6 (loop 601 -> 695 instrs)
  5 / 6 = 1 / 2 issued after the first prologue  d16f -0.1 / -0.3, d32f -0.2 / -0.1; v32f +5.4 / +0.5
Mechanism (K2_TRACE, job31, d32f, us from first CTA start; off / mode 1 / mode 2):
  length+prefix barrier 0.83 / 2.13 / 1.31 (mode 1: prefetch issue stalls warps 1-15 before __syncthreads; mode 2:
  HBM contention on the length line), first rows in 2.50 / 3.73 / 3.65, prologue issued 6.18 / 7.68 / 7.30, tiles med
  155.2 / 154.0 / 153.6, last stamp 159.8 / 159.0 / 158.1. So the start gets LATER by 1.1-1.5 us, and the tile phase
  (prologue -> tiles) gets ~2.6 us SHORTER: the prefetch moves the page-table traffic (~1.7% of the bytes) out of the
  bandwidth-bound K/V phase into the idle start (run1f's K/V copies are evict_first, so the lines survive). Every later
  placement (3-6) leaves that traffic inside the bandwidth-bound phase and gains nothing; the in-loop page-table
  latency is NOT the lever (mode 4). The first-row trip itself only drops 0.58 -> 0.43 us.
Verify full (run8 FLAT): no gain at any placement. Its tile is issue-bound (~3.1 TB/s), so removing 1.7% of the
traffic does not shorten it; it already prefetches page-table rows through its own cp.async ring; its K/V copies
use the normal L2 policy (prefetched lines get evicted); the start flood only delays its length / K/V issue (mode 1:
len-bar 5.09 -> 6.45 us at v32f). The end is set by random stragglers (tiles max - med ~7 us).
Headroom left on decode: the 1.1-1.5 us start delay. A prefetch that leaves the length load and the first-row trip
alone (e.g. issued only once the length line is in flight, or one that does not stall the barrier warps) could reach
~-2.5 us; not built.

## Round 8 (coordinator): page-table prefetch off the start chain's critical path -> dev/ws/k29p.py (jobs 36-39)
2d8788b07118d70e26e81f5d47fbda48  dev/ws/k29p.py   (`python3 dev/ws/build_k26p.py dev/k28.py dev/ws/k29p.py WS_PTP=10 WS_PTL=4 WS_PTF=5`)
cffed175d27207d896625179ae5cee79  dev/ws/build_k26p.py   (base k28 = k29 without the k26p prefetch; run1f identical in k25 / k28)
Delivered mode 10: no entry prefetch; right after the length barrier lanes 0-3 of warp 1 issue one bulk
cp.async.bulk.prefetch.L2 each for 5/8 of the CTA's superset slice; the remaining 3/8 goes out right after the first
segment's rows are requested. run8 untouched (SASS 4624 instrs as k29). attn_1_0_1_0: 127 regs, 0 spill, tile loop
588 instrs, HMMA 0. check.py 0 failures (job38, byte-identical build).
Bench vs k29 control copies (40 reps x2, us):
  job37  d16f k29 86.03 / 86.22, dup 86.47 / 86.26 -> 85.82 / 86.10 (-0.3);  d32f 164.78 / 165.08, 165.12 / 165.23 -> 164.58 / 164.40 (-0.55)
  job38  d16f k29 83.17 / 83.55, dup 83.60 / 83.26 -> 82.92 / 83.05 (-0.4);  d32f 158.82 / 158.41, 158.58 / 157.94 -> 158.27 / 157.38 (-0.6)
  (job38's container is the fast kind: d16f 83, d32f 158, like the judge's.)
Variants (traces d32f / d16f, job36-37; starts in us from the first CTA: barrier / first rows in / prologue issued):
  off               0.77 / 2.37 / 6.08 (d16f 0.77 / 2.34 / 5.82)
  1 (k29)           2.34 / 3.82 / 7.65 (1.10 / 2.56 / 6.21)    per-line issue stalls warps 1-15 before the barrier
  2 bulk at entry   1.31 / 3.68 / 7.39                         32 bulk instrs cost warp 1 ~0.5 us of issue
  7 sleep + bulk    1.47-1.92 / 3.9-4.4 / 7.6-8.1              worse: the sleep only delays warp 1 at the barrier
  8 bulk after bar  0.77 / 3.31 / 7.12 (0.74 / 2.37 / 5.66)    d16f start free; d32f rows wait behind the 11 MB flood
  8, 1 or 4 lanes   0.77 / 3.39 / 7.1                          same as 32 lanes
  10 3/8 | 5/8      0.75 / 2.54 / 7.01 | 0.77 / 3.06 / 6.59    rows protected, the deferred part delays the prologue
  9 thirds          1.54 / 3.28 / 6.80
  Bench vs k28 (job36): m1 -0.96 / -1.0, m7 -0.8 / -0.55, m8 -1.37 / -1.68, m9 -0.2 / -1.15. Vs k29 (jobs 37-38): m8
  -0.2 / -0.1 and -0.1 / -0.35; m10 (5/8, 4 lanes) -0.3 / -0.55 and -0.4 / -0.6; 3/8 +0.2 / -0.35; 5/8 x 32 lanes +0.2 / -0.75.
Why not -2.5: the idle window between the length barrier (0.77) and the first rows (~2.4) is ~1.6 us of HBM, while
the b32 superset is 11 MB (~3 us at the gather rate). Whatever does not fit lands either in front of the critical
row loads (+0.9 us rows trip, mode 8) or inside the HBM-saturated prologue / tile phase, where it costs about what
it saves (modes 3, 5, 6, 10's tail). b16 (5.5 MB) fits: mode 8 / 10 start as fast as off and the whole ~1.3 us
page-table share leaves the tile phase (k28 -> m8 -1.37), so d16f is at its limit for this lever; d32f keeps ~1 us
of start delay. Remaining options (not built): shrink the start-chain dependence (rows known before the length
barrier = speculation, ruled out at B > 1) or a smaller superset (only [0, max seq) instead of [0, max_pages):
~7% less, needs the lengths, i.e. after the barrier, which mode 10 already is).

## Round 9 (coordinator): branch-free producer K/V issue in the sliding kernels (WI CENSUS #1) -> dev/ws/k38w.py (jobs 40-45)
8914e6e01d8e52ff7d8badc7e5c4915e  dev/ws/k38w.py   (`python3 dev/ws/build_k38w.py dev/k33.py dev/ws/k38w.py`, default WS_PD=1)
ccf9e03c68186dba09e73dd1b3c15c2e  dev/ws/build_k38w.py   (WS_PD kernel mask: 1 runA1h, 2 runA8h, 4 runE1, 8 run8w; WS_E1U)
Tool: dev/ws/issue_sass.py FILE KERNEL:FUNC[:MARKER] ... -> SASS instructions whose inline-frame chain touches the issue
block's source lines / LDGSTS among them (image NVRTC -lineinfo + nvdisasm -gi; --nvcc = local 12.1 for structure).
Rewrite (each under #if WS_PD & bit, old code in #else): row shuffles first, 32-bit row clamp (invalid tokens -> row 0,
src-size 0), one IMAD.WIDE per source, destinations as compile-time offsets from one per-thread base (runA1h / A8h: the
swizzle of token tl0 + 4 jj is (ch ^ tl0) ^ 4 (jj & 1) -> d0 + 512 jj + 64 (jj & 1); runE1: cq[jj & 3] + 1024 (jj >> 2)),
the new-token slot skip as a predicated cp.async (ws_cp16q: setp + @q cp.async, no BSSY).
Image SASS, static issue block (instr / LDGSTS): runA1h 261 / 8 = 32.6 -> 165 / 8 = 20.6 (kernel 1728 -> 1632);
runA8h 201 / 8 = 25.1 -> 134 / 8 = 16.8 (2552 -> 2488); runE1 205 / 4 (unroll-4 loop) -> 353 / 16 (full unroll, 16
WARPSYNC + ENDCOLLECTIVE around the grouped shuffles) or 201 / 7 (WS_E1U 2, unroll 4); run8w first-tile block ~7 / LDGSTS.
Bench (40 reps, two k33 controls each job; us):
  d1s  runA1h   job40 -0.08 / -0.10, job43 -0.07 / -0.11, job44 -0.115 / -0.105 / -0.09, job45 -0.09 (3.01 -> 2.92)
  v1s  runA8h   job40 +0.20 / +0.26, job43 +0.31 / +0.29 (k38w minus runA8h: -0.04)       -> off
  d16s runE1    job43 -0.20 / -0.26, job44 full-unroll +0.13, unroll-4 +0.03                -> noise-level, off
  d32s runE1    job43 +0.28 / +0.50, job44 full-unroll +0.44, unroll-4 +0.36                 -> off
  v16s run8w    job43 +0.15 / +0.25; v32s 0                                                   -> off
Traced v1s build (job41) is FASTER with the rewrite (K/V issued 1.56 -> 1.42, tile0 2.18 -> 2.01, last stamp 4.86 ->
4.74) while the untraced one is +0.3, and registers / stack are unchanged (80 / 32, 60 / 32): the regressions are not
in the producer (unroll-4 runE1 has the same static size as k33 and still +0.36 on d32s); they look like whole-kernel
ptxas scheduling / layout side effects of the same size as the savings. Instruction count is not time here either.
Delivered k38w = k33 + runA1h only (d1s -0.09..-0.11, below the 0.15 us kill line but consistent over 9 passes);
check.py 0 failures, t_c16 (3 seeds) 0; the WS_PD 7 / 15 builds also passed check, t_c16, t_v8, t_e1, t_v8w (job43).

## Round 10 (coordinator, final): decode-1 full on clusters with a DSMEM row-owner reduce -> dev/ws/k40c*.py (jobs 46-49) KILLED
dev/ws/build_k40c.py (md5 below) on dev/k34.py: `python3 dev/ws/build_k40c.py dev/k34.py dev/ws/k40c.py [WS_CLW=16]`.
New kernel attn_1_0_{P1}_1k = run1w<.., CLW>, host-selected only when the plan wants >= NCL x CLW CTAs and all NCL
clusters are co-resident (else the unchanged _1w). Tail: merged 16-row partial staged in the ring -> row r pushed by
st.shared::cluster to owner rank r % CLW (receive area past the RoPE table, +4352 B smem) -> one cluster barrier ->
owner warp merges its row over CLW sources, stores it in its own p.ws slot, acq_rel per-row counter (p.counters past
W9E_PHO) -> the last of the NCL owners of the row loads NCL row partials, merges, RMSNorm x gate, stores. Exact (no
fixed reference). check.py 0 failures and t_b1f 0 failures for both C8 and C16.
(Host bug found and fixed: non-portable cluster sizes are enabled only for listed kernel-name suffixes; `_1k` added,
otherwise C16 has 0 co-resident clusters and silently falls back. Co-resident: C16 7, C8 15.)
Cost-side probe first (job46: run1w on fewer CTAs via the CPR override, dev/ws/k34_c112.py / _c120.py): bench 132 CTAs
12.14 / 12.20 (dup 12.26 / 12.23), 120 CTAs 12.05 / 12.09 (-0.14), 112 CTAs 12.48 / 12.49 (+0.28). Traced tiles-end
max 10.34 / 10.14 / 10.69: the gather phase barely depends on the CTA count (the per-SM rate rises).
Bench (40 reps x2, two k34 controls, us):
  job47  k34 12.17 / 12.15, dup 12.27 / 12.29 | 15 x C8 (120) 12.50 / 12.35 (+0.2)
  job49  k34 11.91 / 12.03, dup 11.90 / 12.02 | 7 x C16 (112) 12.33 / 12.20 (+0.30) | 15 x C8 12.07 / 12.08 (+0.11)
Traces (flush, us from the first CTA start; k34 | C8 | C16):
  prologue issued (len-bar) med 3.86 | 4.13 | 4.45 (cluster launch / start cost), tiles end max 10.24 | 9.97 | 10.34
  k34: publish max 11.10 -> last arrival 12.13 -> output 12.62
  C8:  cluster barrier passed 10.35 / 10.64 (med / max) -> owners published 11.02 / 11.38 -> row outputs 12.43
  C16: cluster barrier passed 10.78 / 11.01 -> owners published 11.55 / 11.78 -> row outputs 12.61
Why it does not pay: tiles-end -> output stays ~2.3 us. The DSMEM reduce replaces the red.add drain (~0.7 us) but
adds a cluster round (push + barrier ~0.65 us) and an owner merge; two dependent global latencies remain whatever
the partial count (owner store + release ~0.75 us, then the last owner's acquire + load of NCL rows + epilogue ~0.9
us), and the cluster launch delays the start by 0.3-0.6 us. Net +0.1..+0.3 us vs the 0.5 us kill rule: KILLED; no
verify-1 full port (coordinator: final round).
