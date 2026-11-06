# Attention / KV path worker notes (2026-10-30), base research/claude/stack/s1.py

## Tools (this directory)
- `tl.py` / `tl.sh`: nsys node timeline of the whole-layer graph replay (per kernel start/end/stream/grid, median
  over reps). `bash research/claude/attn/tl.sh OUT 'base=research/claude/stack/s1.py' --cases 1,2,3 --reps 6`
- `front.py`: front-of-layer graph [fork, QKV GEMM, kv_pack, qkv_post2, attn_fwd, attn_comb, o_proj], cold L2 before
  every replay, candidates interleaved AB/BA, paired gain vs the first candidate, `--check` = o_proj output diff.
  Noise ~1 us. A candidate may define `_front(data, state)`; else s1's kernel_fn front is replicated.
- `trace.py` / `trace_tiles.py` / `trace_unit.py` + `v_trace.py`: per-CTA phase stamps (ATT_TRACE, %globaltimer),
  per-segment stamps and per-tile release stamps inside attn_fwd (stamps go to the otherwise unused hacc tail / oacc).
- builders (same-line patches of ATT_CUDA_SRC: the derived attention sources slice it by line number, so a patch
  must never add or remove lines): `mk_resc.py` (rescale trigger / headroom, `full` = full-attention modules only),
  `mk_headroom.py`, `mk_sk.py` (split, run-time search: dead), `mk_sk2.py` + `mk_sk3.py` (split with tables + trims),
  `mk_post.py` (qkv_post2 load hoist), `mk_comb.py` (dead), `mk_u2.py` (dead), `mk_final.sh OUT BETA [post]`.
- `sk_check.py` / `sk2_check.py`: CPU proofs of the cost-aware split (partition, combine ranges, slot kinds, tables).
- `front.py --spec '[{...generate_input params...}]'` times nearby shapes too (cases 100+).

## Critical path (s1, nsys node timeline, case medians)
- case 0 (753 sliding): QKV 0-42.9 | kv_pack side 20.9-29.9 (hidden) | qkv_post2 42.9-49.2 | attn_fwd 50.6-71.7 | o_proj.
- case 1 (753 full 80K): QKV (108 CTAs, qkv_smt) 0-43.0 | side kv_pack starts only 24-37 us in, ends 54.4 |
  qkv_post2 44.2-56.4 | attn_fwd 58.1-263.1 | attn_comb ends 273.9 | o_proj ends 292.6.
  Pre-attention costs 15 us after the GEMM (case 0: 7.7): ~7 us of kv_pack is not hidden.
  attn_comb adds ~11 us after attn_fwd (o_proj gets ~2 us back by starting early).
- case 2 (4x753 sliding): QKV 0-144.9 (132 SMs) | side kv_pack 72.5-126.4 (hidden; its 46 us profile time is SM
  starvation under the GEMM, not work) | qkv_post2 5.7 | kv_pack part 2 3.7 | attn_fwd 43.6.
- case 3 (4x753 full): QKV on 108 SMs 171.6 (vs 144.9 at 132 SMs: the price of the side kv_pack) | side kv_pack
  26-187 | qkv_post2 ~10 | kv_pack part 2 7.9 | attn_fwd 770 | attn_comb +17.8.
- cases 5-7 (16K): side kv_pack sits in the QKV GEMM tail (hidden); kv_pack part 2 4.4-5.1 and attn_comb +12 on
  the path; weights 0.0025%/us (5) and 0.0003%/us (6, 7): not worth tuning.

## attn_fwd imbalance (the big finding)
Per-CTA end times in full attention spread 188.7-206.1 us (case 1) and 686-748 us (case 3) for equal tile counts.
The spread is work-dependent (per-CTA finish correlates 0.96-0.99 across replays, per-SM does not) and is all in
the number of segments: case 1 CTAs with 2 segments end 7.7 us after 1-segment CTAs; case 3 8 vs 4 segments: +30 us.
Per-tile stamps: a segment switch itself costs ~2 us (drain, partial store, Q load, refill; last causal tile
+0.6 us), and the new segment's tiles then run slow for ~100+ tiles (+0.10 us/tile at first, decaying ~1/j):
the online-softmax running max starts over, and a tile rescales whenever any of a warp's 16 rows sets a new
record (probability ~16/j at tile j). A rescale costs ~0.1 us of a 0.62 us tile (the shuffles, two EX2, bias
refresh and 36 O multiplies, between the PV wait and the next issue). Sum ~6-8 us per segment.

## Rescale policy experiments (front probe, paired gain vs s1, 20 reps, cold L2)
EXPQ codes k = 8 (S - m) + 56 must stay <= 126 (E4M3 448): rescale iff tile max > m + 8 + TRIG (s1: TRIG 0), then
m = max - 8 + D (s1: D 0). FP8 underflow cutoff: S < m - 7, i.e. max - 15 + D.
- headroom D (P_max 2^(8-D), cutoff max - 15 + D; fewer rescales):
  case1 D2 +7.1 D3 +9.4 D4 +9.8 D6 +12.3 us; case3 D2 +29.6 D3 +37.9 D4 +43.9 D6 +49.8 us (20/20 wins each);
  sliding cases 0/2: no change (+-1 us). Output moves 1.3-2.6% vs s1.
- full-layer softmax spread: c2 in [0.5, 0.62] -> sigma_S ~5 log2 units; mass lost below max-15 (s1) ~1-2%,
  below max-12 (D3) ~4-8% (numpy model). SGLang's FA3 FP8 uses P = exp2(S - max + 8) in E4M3 (max offset 8, like
  s1's D0), so D > 0 is a precision reduction vs the baseline: kept only as a measured option, not proposed.
- gate (42/42 all): worst error/bound s1 0.500, D3 0.504, D6 0.579; D3 raises errors ~0.5% relative (16K full
  cases ~4%: 0.0536 -> 0.0557), D6 a lot on 16K full (0.0536 -> 0.0711).
- range-neutral trigger (rescale only near the E4M3 overflow limit, TRIG 0.75, D 0: m stays <= max - 8, so the
  FP8 cutoff is never coarser than s1's; the codes stay <= 126; NOT error-neutral, see the gate comparison below): v_n75. front: case1 +3.4 us (15/16), case3
  +15.7 us (15/16); TRIG 0.5: +2.6 / +11.8; sliding 0/2: tie. TRIG 0.75 + D 1: +6.3 / +26.2. Gate 42/42.
  n0 (builder control, TRIG 0 D 0): bit-identical to s1, tie.

## Dead ends (measured, front probe unless noted)
- INCOMB (last-arriving part combines inside attn_fwd): case1 -11.0 us, case3 -66.6 us (each combine's L2/HBM
  latency lands on that CTA's next segment).
- DIRECT_NEW_MAX 4096 (qkv_post2 packs new keys for B=4, no main-stream kv_pack part 2): case2 -3.2, case3 -2.2.
- cost-aware stream-K (mk_sk, owner() by binary search over ctab at run time): beta 0 -1.0/-2.0, beta 6
  -13.7/-44.9, beta 12 -12.7/-37.2 (case1/case3, on top of n75). The searches are dependent global loads, and
  attn_comb runs three of them on the critical path; beta 0 compiles the search away. Would need precomputed tables.
- comb rewrite (gate loads hoisted, slot loop unrolled 4): case1 -1.0, case3 -5.7.

## attn_comb
nsys with attn_comb launched without PDL: standalone 7.0 us (case 1), 18.5 us (case 3); with PDL the case-1 tail
after attn_fwd is 9.7 us. It re-reads every stream-K partial slot (~14 MB per request, 17.4 KB per (CTA, kind, WG))
after they have mostly left L2 (case 3 streams 82 MB of KV): ~2 TB/s. PDL itself does not help it here.
Codex's full-units (unit 1 for B*2*nqb >= grid) lost 26 us on case 3 (requests of different lengths share CTAs).

## More dead ends / side results
- full-layer split units (mk_u2: B=1, 2*units within 93-100% of the grid -> p.unit 2 with the existing hacc handoff,
  cached+vote unit module): case1 u2 alone -1.6 us, n75+u2 +2.0 vs n75 +4.4 (9 extra tiles per CTA, 4 idle SMs and
  the handoff outweigh the comb + second segment). Dead end.
- n75 costs ~1 us on sliding case 0 (3/24 and 3/16 wins) -> full-only build (mk_resc ... full: only mod_sk_vote /
  mod_sk_empty get TRIG 0.75; v_f75): sliding bit-identical to s1.
- cached+vote module for sliding unit2 (u2s): bit-identical to the plain one, no gain on case 0 (-0.9 vs -1.07).

## Cost-aware stream-K with tables (mk_sk2 + mk_sk3), on f75
kv_pack's ctab block (one per request) also writes sktab: each CTA's first tile and each unit's (c0, c1, starts-
inside-c0 flag), with ctab staged in its idle shared buffer (binary searches in shared memory). attn_fwd reads its
range (2 loads), attn_comb its part range (CPU proof: sk_check.py / sk2_check.py).
front (20 reps, vs s1): f75 +4.1 / +13.6; t0 +1.7 / +15.6; t4 +4.0 / +19.0; t8 +5.2 / +24.2 (case1 / case3, all
20/20). t0 < f75 on case 1 = table plumbing latency (ctab blocks ran last in kv_pack; kind / c0,c1 became dependent
loads) -> mk_sk3: ctab blocks first, kind loaded at segment start, attn_comb indexes before griddepcontrol.wait.
After mk_sk3 (front, 20 reps, vs s1): s8 +6.4 / +24.4, s12 +6.8 / +27.7, s16 +6.8 / +28.2 (case1 / case3).
qkv_post2 loads before the PDL wait (mk_post, bit-identical): case0 +0.1 (12/20), case1 +0.6, case2 +1.2 (15/20),
case3 -0.4: marginal (its CTAs only become resident as the GEMM drains), kept because it is free and exact.

## Candidate: attn1.py = mk_final.sh attn1.py 12 post
s1 -> full-only rescale trigger at the overflow limit (TRIG 0.75, D 0) -> cost-aware stream-K tables (BETA 12) ->
latency trims -> qkv_post2 load hoist. Sliding layers: bit-identical attention output to s1.
Gate: CHECK_RC=0 PASS=42 CASES=21. File 811,641 bytes. mk_final.sh reproduces the measured s12 byte for byte.
Nearby shapes (front, 12 reps, vs s1): B1 ctx60000 full L760 +4.8 us (12/12); B2 ctx40000 full L1000 +8.5 (12/12);
B5 ctx20000 full L600 +8.4 (12/12); B1 ctx3000 full L753 -0.5 (5/12, noise); B4 ctx70000 sliding L760 +0.6
(bit-identical); B1 ctx8000 full L3000 -1.5 (2/12: short causal units, ~3 segments per CTA everywhere, so the
split has nothing to balance; <0.1% of that layer).

## Judge replica (panel 11,22, all 8 cases, 6 reps; s1 and attn1 each in their own AB/BA pairs with SGLang)
DRAW_SCORE d11 base x1.9719 new x2.0024; d22 base x1.9531 new x1.9784; PANEL base x1.9625 new x1.9904;
RELATIVE new/base 1.0155 1.0130 mean 1.0142. Per load: turn1 2.0343->2.0380, 1.9766->1.9867; turn4 1.9222->1.9330,
1.9177->1.9238; first turn 1.9608->2.0381, 1.9655->2.0261.
The headline overstates it: first-turn per-case times move +-150-200 us between draws (case 5 is sliding with
bit-identical attention and reads 224 us faster; case 4 reads +190 us and -115 us). Trust cases 1 and 3, which
match the front probe: case1 713.3->708.0, 779.4->770.8 us; case3 2519.8->2492.1, 2617.4->2593.7 us; cases 0, 2
tie. Expected value from the front numbers and the score sensitivities: ~+0.25% (case1 ~7 us x 0.014%,
case3 ~27 us x 0.0042%, cases 4/6/7 small).

16K cases (front, 8 reps, vs s1): case4 +10.5 us (8/8), case5 tie (bit-identical), case6 +45.8 (8/8), case7 +71.1
(7/8): ~+0.04% at their weights.

## Precision note (gate JSON comparison attn1 vs s1)
Full-layer errors rise 0.4-0.8% relative (scored/01 update 0.0646 -> 0.0651, bound 0.150; scored/07 0.0611 ->
0.0614; worst error/bound over the gate 0.5003 -> 0.5023); sliding unchanged. Cause: EXPQ's code has a linear
mantissa (1 + mant/8), exact only at mant 0, and s1 puts each new row max exactly there (m = round8(max - 8),
code 120). With TRIG 0.75 the top keys can sit up to 0.75 above the last rescale point (codes 121-126, up to ~+4%
weight error on the dominant keys). FP8 range/cutoff unchanged; numerics differ from s1 only by this.
Split-only alternative (no trigger change, s1's softmax numerics): attn1s.py = mk_sk2 12 + mk_sk3 + mk_post on s1.
front (20 reps, vs s1): attn1 case1 +8.0 us, case3 +26.9 (20/20); attn1s case1 +3.7, case3 +14.3 (20/20); case0 both
+0.4-0.5 (15-17/20, the qkv_post2 hoist). attn1s output differs from s1 by 0.6% (the stream-K partition moves the
segments' running maxima, so the EXPQ rounding lands differently: s1's own numerics are partition-dependent).
attn1s gate: CHECK_RC=0 PASS=42 CASES=21 (811,159 bytes).

## Ideas not done (for whoever continues)
- Sliding case 0 (highest weight, 0.040%/us): attn_fwd 21 us = first tile ~3.5 us after Q, 0.96 us/tile steady (full
  layers 0.62; headroom/fewer rescales did not help, cached+vote module did not help), merge-half handoff ~2.8 us
  (threadfence + flag poll + hacc reload) and epilogue ~2.6 us. A DSMEM handoff (the two halves of a unit in one
  2-CTA cluster) could cut ~1.5-2 us; prefetching the gate values before the handoff wait ~0.7 us.
- attn_comb re-reads ~14 MB of partial slots per request from HBM (7 us case 1, 18.5 us case 3); keeping them
  L2-resident (evict_last stores) competes with the KV stream.
- B=4 full (case 3) still pays 4-8 segments per CTA; a phased split (one request per phase, unit halves with the
  hacc handoff, grid 128) was estimated at best +15-20 us and not built.
attn1s replica (panel 11,22, cases 1,3, 6 reps): case1 714.3->709.7, 779.1->773.9 us; case3 2528.9->2509.8,
2606.5->2598.9 us; RELATIVE (turn1+turn4 loads only) 1.0067 1.0036 mean 1.0051.
