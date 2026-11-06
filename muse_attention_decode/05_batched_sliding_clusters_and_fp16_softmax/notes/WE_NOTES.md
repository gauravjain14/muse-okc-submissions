# WE: decode sliding batch 16 / 32 (q_len 1, window 2048) — cluster per request with DSMEM combine

Base dev/k16.py (md5 49380899). Deliverable dev/we/k17e.py built by dev/we/mk_k17e.py from dev/k16.py + dev/we/rune1.cu.
Tools: dev/we/probe.py (co-residency + gather floors), dev/we/t_e1.py (extra correctness cases, every C),
dev/we/trace.py (per-CTA stamps, from WS's), dev/we/regs.py (image NVRTC ptxas / C75xx / SASS size), job*.sh.

## Log
- job1 (probe): cuOccupancyMaxActiveClusters, 512 thr 1 CTA/SM (100/150/200 KB): C2 66, C3 39, C4 30, C5 22, C6 17,
  C7 15, C8 15, C10 7, C12 7, C16 7. 256 thr 100 KB (2/SM): C2 132, C3 79, C4 62, C5 47, C6 39, C7 32, C8 30, C16 14;
  60 KB (3/SM): C4 92, C8 45, C16 21. => at 1/SM: b16 per-request C8 (16 clusters) and per-head C4 (32) do NOT fit;
  b32 per-request C4 (32) does not fit; launching them anyway = second wave (32 x C4: 11.9 us vs 8.0, start spread 6.9).
  Gather floor with the real chain (seq -> pt -> K/V, real generate_input data, each 512-thr CTA issues ALL its
  cp.async at once, cold L2, graph replay - empty graph):
  b32: full 128-B rows C4 128 CTAs 7.97 us | half 64-B rows per head C2 128 CTAs 9.02 (cl2 8.90) | full C3 cl3 96
  CTAs 8.26. b16: full C8 128 CTAs 5.66 | half C4 128 6.02 | half C3 cl3 96 6.13 | full C6 cl6 96 5.70.
  => 64-B half rows lose 6-13%: per-REQUEST clusters with full rows: b16 C6, b32 C3 (96 CTAs, ~= 128-CTA gather).
  Stamps (b16 full C8): thread 0 issue done 3.04 us (cp.async backpressure), landed med 4.42 max 5.20. The marginal
  rate b16 -> b32 is ~4.7 TB/s; the fixed start (chain + first trip) ~2.6 us dominates.
- job2: k17e v1 (runE1: producers warps 8-15 issue tiles pw / pw + 8, consumers WG0 / WG1 attend alternate tiles with
  dec_tile, WG1 -> WG0 register merge, st.async push to row owners, owner warp per row): image NVRTC 96 regs, 32 B
  stack (sincosf slow path), no C75xx, 2152 SASS instrs. t_e1 (all C: 16, 8, 6, 5, 4, 3; mixed / short / page 64 /
  kv_scale / dp / edges, 3 calls each) 0 failures.
- job3 (2 x 40 reps, k16 twice as A/A): b16 k16 8.53 / 8.50, k16b 8.62 / 8.55, k17e 8.13 / 8.10 (-0.4);
  b32 k16 10.60 / 10.84, k16b 10.74 / 10.52, k17e 11.34 / 11.29 (+0.6: LOSES).
- job4 trace (us from first CTA start, median CTA): b16 pre-ld 0.45, bar10 (seq + RoPE table) 1.34, warp 8's tile-0
  issue done 3.34, tile0 4.00, t2 4.96, t4 5.79, tiles 6.53, merged 6.69, pushed 7.07, recv 7.30, out 8.03.
  b32: bar10 1.51, tile-0 issue 3.36, tile-8 issue 5.63, t0 4.94, t2 5.89 ... t10 9.06 (0.8 us apart), tiles 9.68,
  merged 9.86, pushed 10.21, recv 10.52, out 11.15. => dec_tile ~0.6-0.8 us per 64-token tile per warpgroup
  (t10 -> tiles = 0.62 us for one landed tile): 2 consumer WGs x 5-6 tiles are COMPUTE-bound at b32. Tail tiles ->
  out ~1.5 us (push 0.35, slowest peer 0.3, owner combine 0.6).
- job5 (v2: NCW consumer WGs + we_tile2 = 2 tiles per step: one QK batch, one vote, one P^T fence, one PV chain):
  2 x 40: b16 k16 8.49 / 8.41 | NCW3+pair 9.33 / 9.21 | NCW2+pair 7.79 / 7.77 | NCW3 single 8.88 / 8.81;
  b32 k16 10.52 / 10.47 | 11.77 / 11.72 | 10.39 / 10.34 | 11.65 / 11.61. NCW=3 (only 4 producer warps) LOSES: trace
  warp 12 needs 2.5 us to issue its 2nd tile -> cp.async issue is capped PER WARP (~6.5 GB/s, ~13 KB in flight / warp).
- job6 (v3: half-tile issue units (32 tokens, one page-table entry per lane), CI = consumer warps issue tiles 4..7):
  b16 k16 8.59 / 8.56 | v3 CI=1 7.69 / 7.56 | v3 CI=0 7.54 / 7.43 | v2 7.79 / 7.83; b32 10.56 / 10.64 | 9.82 / 9.90 |
  9.42 / 9.36 | 10.34 / 10.30. CI=1 delays the consumers' Q prep (q/new 1.8 -> 3.1 us) and competes with tiles 0-3.
- job7 trace (v3 CI=0): b16 pt entries landed 1.34, last tile landed 4.24, WG0 2 steps 4.06 -> 5.38 -> 6.27 (a 2-tile
  step costs ~0.9-1.3 us), out 7.79; b32 landL 6.27, tiles 8.03, out 9.45 => compute-bound after landing, tail 1.4 us.
- job8 (v4: producer WGs attend the last 1-2 steps after issuing (WE_NP), WG0's Q^T shared via an mbarrier):
  b16 k16 8.76 / 8.54 | NP2 7.99 / 7.96 | NP1 7.87 / 7.81 | NP0 8.09 / 7.99; b32 10.82 / 10.93 | 9.95 / 10.09 |
  10.36 / 10.53 | 10.15 / 10.27 (slow machine / noisy job; v3 not in the job).
- job9 (v5: we_tilex = EXACT step max instead of dec_tile's lazy vote (logit std ~6 log2 units: the vote's rescale
  path fired on most steps)): b16 k16 8.47 / 8.36 | v3 7.48 / 7.41 | x NP2 7.26 / 7.18 | x NP1 7.31 / 7.17 | lazy NP2
  7.43 / 7.43; b32 10.46 / 10.39 | 9.29 / 9.16 | 9.33 / 9.17 | 9.40 / 9.37 | 9.42 / 9.38. Pair-step phases (cycles,
  first step): QK 500-575, max exchange 470-560, P + fence 325-376, PV 280.
- job10: flush vs warm traces: tail phases identical (s11 -> merged 0.5, push 0.33, recv 0.3, combine 0.73 us) ->
  latency-bound, not instruction fetch (as WS / WC found).
- job11 (v6: phase-1 cluster wait right after barrier 10 (it was after the steps), owner table p.gpos copied to smem
  at entry): b16 k16 8.40 / 8.39 | v3 7.51 / 7.61 | v6 7.29 / 7.35; b32 10.44 / 10.53 | 9.26 / 9.45 | 9.26 / 9.24.
  Tail cycles (thread 0): fragment wait 31, merge (3 sequential we_mrg) 575-732, push 482, owner combine to the
  shuffles 1008-1140 (!: runtime C -> 16-way predicated unroll with per-source address math, ~470 SASS instrs).
- job12 (v7: kernels _1e3 / _1e6 with compile-time C (constant split division, exact owner loops), one-level
  4-fragment merge, cluster arrive after the store): b16 k16 8.55 / 8.46, k16b 8.59 / 8.52 | v6 7.38 / 7.39 |
  v7 6.59 / 6.59 | v7 runtime-C 7.26 / 7.18; b32 10.63 / 10.55, 10.57 / 10.41 | 9.38 / 9.14 | v7 8.54 / 8.60 |
  9.25 / 9.01. => -1.9 us at b16 and b32. Tail cycles: merge 317-336, push 483, owner combine 131-173 + 268.
- job13 (v8: WE_NS = trailing single-tile steps for the producer WGs, push targets computed before the fragment
  barrier, 16-B pushes WE_V4): b16 k16 8.47 / 8.28 | v7 6.53 / 6.47 | v8 NS2 6.58 / 6.48 | NS2 noV4 6.37 / 6.59 |
  NS0 6.52 / 6.44 | NS1 6.72 / 6.63; b32 10.55 / 10.47 | 8.47 / 8.30 | 8.71 / 8.76 | 8.61 / 8.62 | 8.36 / 8.28 |
  8.58 / 8.48. NS2 gives WG0 3 steps at b32 (worse); NS0 + V4 + precomputed targets ~= v7 (push 483 -> 213 cycles).
- job14 (v9 WE_CI2: round-0 half-tiles 8..15 issued by the consumer warps right after the entries land): b32
  9.57 / 9.71 vs 8.33 / 8.35 without: the per-SM issue rate does not grow with 16 issuing warps (warp 8's first
  half-tile done 3.1-3.3 us vs 2.5) and the consumers' Q prep is delayed (bar10 1.6 -> 3.6-4.1 us). DEAD END.
  => the K/V issue is backpressured by the memory system (~110-135 LDGSTS / us / SM ~ 60 GB/s per SM); with only 4
  issuing warps it is per-warp limited (job5), 8 warps suffice.
- job15 DELIVERABLE dev/we/k17e.py (v9 defaults: CI2 0, NS 0, NP 2, V4 1, EXACT 1, PAIR 1; md5 below), built by
  dev/we/mk_k17e.py from dev/k16.py (md5 49380899) + dev/we/rune1.cu. image NVRTC: _1e3 / _1e6 127 regs, 32 B stack
  (sincosf slow path), no spills, no C75xx; 2440 / 2472 SASS instrs. t_e1 0 failures, tools/check.py 0 failures.
  2 x 40 reps: b16 k16 8.72 / 8.48, k16b 8.92 / 8.58, k17e 6.98 / 6.69; b32 10.63 / 10.74, 10.82 / 10.64,
  k17e 8.64 / 8.67. All loads (20 reps): d16 sliding 8.59 -> 6.92, d32 sliding 10.68 -> 8.67, every other shape runs
  identical code (within noise); GEOMEAN x2.3644 -> x2.3799 (judge-proj x2.2270 -> x2.2413).

## Where b16 / b32 time goes now (trace, us from first CTA start)
- chain: param / i-cache cold start 0.42 (prefix loads issued), length lands ~0.8, page-table entries ~1.25,
  first K/V copies done issuing ~2.5, first tile lands ~3.9; gather is memory-bound after that (b16 last tile ~4.1,
  b32 ~6.2: ~4 TB/s on 96 SMs, same as the probe).
- after the last tile lands: last pair step ~0.9 us (cycles: QK ~500, max exchange ~470, P + fence ~330, PV ~280),
  fragment merge ~0.17, push ~0.11, slowest peer ~0.25, owner combine ~0.2.
- Floors with this structure: b16 ~5.5, b32 ~7.6 local; targets 5.5 / 6.5 are below the measured gather probe floors
  at b32 (8.0 us for 128 CTAs, 8.3 for 96, with the real chain and data).

## Dead ends (numbers above)
- 64-B half rows (per-(request, KV head) clusters): gather 6-13% slower; C4 per head at b16 not co-resident (30 < 32).
- 3 consumer WGs with 4 producer warps (issue rate per warp), consumer warps issuing K/V (CI, CI2), lazy-vote softmax
  (dec_tile) on these logits, runtime-C owner loops (16-way predicated unroll), trailing single-tile steps (NS).
e1b2964d1f7058a2b73a6cf6db63ff4d  dev/we/k17e.py
3655789448da1f75c6fee5084ae89445  dev/we/rune1.cu
dabd4fa60747f2d66c9ea49f6547572f  dev/we/mk_k17e.py

## Round 2 (coordinator: the ~2 us after the last tile lands), base dev/k23.py (md5 a993af41), deliverable k24e
Builder dev/we/mk_k24e.py [base] [out] [src]: replaces the runE1 device section (content anchors: the runE1 banner up to
the WF FLAT banner) and the NVRTC program name (regex) -> k24e.cu; host / instantiations are the base's.
- job16: WE_SCH=1 (WG j attends pair j for j < 4, the last-landing tiles as single steps for WG 0, 1, 3, 2):
  b16 k23 6.57 / 6.57, k23b 6.62 / 6.61 | SCH1 6.62 / 6.62 | SCH0 6.64 / 6.52; b32 8.59 / 8.45, 8.58 / 8.49 |
  SCH1 9.07 / 8.96 | SCH0 8.48 / 8.39. SCH1 LOSES at b32: WG3 = pair (6, 7) then single 10 ends 8.11 (WG0 7.46).
  Step phases of WG2's first step (cycles; b16 / b32): QK 566 / 1104, local max + 3 shuffle rounds 296 / 328,
  barrier wait (warp skew) 277 / 273, max table + rescale 171 / 128, P 247 / 309, fence 26 / 47, barrier 20 / 28,
  PV 248 / 323. => the fence is not on the path (item 4: nothing to move); QK and the exchange dominate, inflated
  by 3-4 warpgroups computing at once (QK 1104 at b32).
- job17: WE_SPL=1 (3 sources per CTA: WG0 + WG1 merged, WG2, WG3 push their own partials right after their steps,
  no fragment hand-off): b16 k23 6.84 / 6.73, k23b 6.92 / 6.92 | SPL 7.19 / 7.10 | SPL0 6.86 / 6.85; b32 8.75 / 8.67,
  8.82 / 8.76 | 9.08 / 8.98 | 8.88 / 8.69. LOSES: pushed -> recv 0.25 -> 0.7-0.85 us (3x the DSMEM bytes; the
  st.async exchange runs at ~15-20 GB/s per SM) and the owner combine 130-170 -> 330-480 cycles.
  => DSMEM bytes are on the critical path: fewer bytes per CTA, not more sources.
- job18: WE_SELF=1 (a CTA's own rows stored locally, expect_tx (C - 1) sources, named barrier to the owner warps:
  1 / C fewer DSMEM bytes): b16 k23 6.48 / 6.55, k23b 6.57 / 6.50 | SELF 6.64 / 6.56 | SELF0 6.53 / 6.53; b32
  8.71 / 8.30, 8.58 / 8.43 | 8.61 / 8.34 | 8.49 / 8.40 -> neutral.
- Not built (reasons measured above): (1) QK(i+1) under softmax/PV(i): the critical last steps (WG2: tiles 8-9,
  WG3: tile 10 at b32; WG2: 4-5 at b16) are those warpgroups' ONLY steps (their tiles land last), nothing to overlap;
  WG0 / WG1's second steps end before them. (2) stale / lazy max: each warpgroup's first step needs the exchange
  (m = -inf), and with logit std ~6 log2 units a new 128-token step exceeds a 128-token reference by > 0.75 in ~40%
  of steps (rescale path ~1000 cycles) -> expected cost ~= the exchange (~725 cycles); a larger threshold needs a
  lower P offset, which pushes the bulk of the tokens (~14 log2 below the max) into E4M3 subnormals. (3) EXQ: at
  q_len 1 the FP32 row sums are taken from the P values (no ones column in the PV wgmma): decoding 16 codes per thread
  costs more than the 16 MUFU + 8 F2FP it removes (the P phase is ~250-300 of ~2000 cycles); W9 measured EXQ in
  run1's dec_tile slower on sliding. (4) fence placement: the fence is 26-47 cycles (stamps), not on the path.
- Round 2 result: no lever beat k23 beyond the A/A noise; k24e (= k23's behaviour with the knobs default off)
  is a tool build, not for integration.
c3ea976a51a02c77fa66b272949eac74  dev/we/k24e.py
dbd655984c26d8317eff51c021326a79  dev/we/rune1.cu
a40114731c6549a1691ccb3430367e60  dev/we/mk_k24e.py
  (job19 on k24e, all round-2 knobs off: t_e1 0 failures, tools/check.py 0 failures; the md5 above differs from the
  tested build by two comment lines only)

## Round 3 (coordinator IDEA 3: 32-token first stage), base dev/k25.py (md5 6236c117, runE1 = k23's)
Builder dev/we/mk_k26e.py [base = dev/k25.py] [out = dev/we/k26e.py] [src = dev/we/rune1.cu] (program name = out
basename). WE_HS = 1: WG0's first pair (tiles 0-1) starts with a 32-token step on h 0 (its own mbarrier, warp 8's
copies; we_tilex<false, HALF>: PV k-step 0 only, the still-landing tokens 32..63 masked and never multiplied), then the
pair with tile 0's tokens < 32 masked (lo0); WE_HS = 2 also WG1 on h 4 (tile 2).
- job20 / job21 (2 x 40 reps each, k25 twice): b16 k25 6.62 / 6.78 / 6.63, k25b 6.69 / 6.84 / 6.70 | HS1 6.98 /
  7.05 / 6.94 | HS2 7.02 | HS0 6.68 / 6.91 / 6.81; b32 k25 8.56 / 8.67 / 8.59, k25b 8.79 / 8.62 / 8.67 | HS1 8.74 /
  8.69 / 8.77 | HS2 9.05 | HS0 8.67 / 8.71 / 8.47. => HS LOSES (+0.3 us at b16, +0.1..+0.5 at b32).
- Which warpgroup ends last (trace, median CTA, us): HS0 b16 WG0 5.25, WG1 5.22, WG2 5.68 (last; WG3 has no step),
  b32 WG1 7.09, WG0 7.14, WG3 7.41, WG2 7.73 (last). HS1: b16 WG0 5.73 (+0.5), WG2 6.05; b32 WG0 7.57, WG2 8.14.
  The extra 32-token step costs a full step's latency (~0.6 us) in WG0's chain and adds contention, while the
  first landing is not on the critical path: the end is set by the producer warpgroup WG2's step on the
  last-landing pair (tiles 4-5 at b16, 8-9 at b32), started once its own copies are issued and its tiles land.
- k26e default WE_HS = 0 (= k25 behaviour): t_e1 0 failures (default and WE_HS=1), tools/check.py 0 failures.
a11b9ccfa119d854494c1bf4a536665e  dev/we/k26e.py
6e9963a095122c881857331b2728da02  dev/we/mk_k26e.py
ef42e4b0af8e8ffc5f9eee45e4c8d272  dev/we/rune1.cu
