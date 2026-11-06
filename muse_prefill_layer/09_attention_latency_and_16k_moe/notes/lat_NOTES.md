# lat worker: latency-bound small kernels at T=753 (2026-10-30)

Base: research/claude/attn/attn1.py (root's instruction mid-session; the first experiments below were on s1.py, whose
sliding path is bit-identical to attn1's). Replica comparisons: base=research/claude/attn/attn1.py.

## Files
- `sla.cu`: new sliding-window attention for short blocks (`4 * B * ceil(L / 24) <= 128` CTAs, e.g. case 0):
  one 2-CTA cluster per (request, kv head, 24-token block); rank r attends to its half of the unit's key tiles
  (same split as s1's unit-2 mode), the halves meet at the absolute scale 2^m through DSMEM (`st.async` +
  mbarrier tx count), and each rank finishes 12 of the 24 tokens (gates TMA-prefetched into shared memory).
  `fused=1`: the CTAs also do qkv_post2's work (QK-norm, RoPE, FP8 Q into shared memory, the new keys' cache rows
  and packed tiles); new-key tiles are loaded once all producers of that head have published (ready counter).
  `flagjoin=1`: the cached-key packing (`sla_pack`, side stream) is awaited through a done counter (no graph join
  before attention; the side stream joins at the end of kernel_fn); the last attention CTA resets the counter.
- `mk_sla.py BASE OUT [--fused] [--flagjoin] [--split] [--trace] [--define X]`: single-file candidate builder.
- `trace_sla.py CAND --cases 0`: per-CTA phase stamps + SM clock (SLA_TRACE build).
- `gate_lat.py` / `gate_lat.sh`: gate + extra sla-path shapes.

## Baseline (s1) case 0 attention phases (trace_unit, us since attn_fwd start; attn_fwd starts ~1.5 us after
qkv_post2 ends because of the side-stream join)
Q loaded 1.82 | first tile done 5.2-5.9 | 0.96 us/tile | main loop done 12.0 / 13.3 (half 0 / 1) | handoff done
14.1 / 16.0 | epilogue done 18.6. nsys: qkv_post2 39.2-46.2, attn_fwd 47.7-68.2.

## Results (front probe = research/claude/attn/front.py, case 0, cold L2, paired gain vs base, 20-24 reps)
All sla variants with the s1 split are bit-identical to s1/attn1 on case 0 (FRONT_CHECK rel 0.000).
| variant | gain vs s1 | notes |
|---|---|---|
| nf: cluster + DSMEM exchange + split epilogue + TMA gates, q8 from qkv_post2 | +4.2 us | 20/20 |
| nfj: + flag join (no graph join before attention) | +5.5 us | 20/20 |
| f: + fused qkv_post2 | +7.3 / +8.0 us | Q ready 5.7 us after the GEMM wait |
| f1: scales loaded before the PDL wait, publication off the Q path | +8.2 us | Q ready 3.1 us after the GEMM wait |
| f1t: + griddepcontrol.launch_dependents after Q (o_proj launches onto idle SMs) | +9.9 us | 24/24 |
| f1s1 / f1s2: balanced split nt/2, (nt+1)/2 | +8.5 / +9.1 (no trig) | o_proj output moves 0.22% / 0.63% (partition) |
| g0: + sigmoid gates precomputed by idle producer warps, f16 masks, cached EXPQ bias | +9.5 | no gain end-to-end |
| g2pp: + bias from rint(8 ta) (exact), ping-pong issue order | +10.0 | kernel end -0.6 us in trace |

Per-tile ablations (trace, steady state, SM clock ~1.7 GHz): full softmax 0.82 us/tile; rescale only on tile 0
0.62; no softmax 0.50 (= tensor floor: 3 WG x (QK 128 + PV 144 clk)). Short sliding segments rescale on ~90% of
tiles (any of 16 rows sets a new running max); headroom/trigger changes are precision changes (not allowed on
sliding). A masked tile costs ~0.7 us more (integer selects); the f16 mask path cuts it to ~0.4.
Prologue (fused, trace): GEMM wait returns 2.2 us after the first CTA starts (PDL launch overlaps the GEMM tail);
production + Q 3.1 us more (seq_lens -> rope_tab HBM chain, 2.1 us of it in the production pass).

## Checkpoint c1 = attn1 + sla (fused, flagjoin, trigger, CACHE_C 2, ping-pong, s1 split; bit-identical on case 0)
- gate_lat (21 standard + 6 sla-path shapes): CHECK_RC=0 PASS=54 CASES=27.
- replica vs attn1 (panel 11,22, cases 0-3, 6 reps): case 0 539.1 -> 527.5 / 532.6 us; RELATIVE 1.0057 / 1.0040.

## Multi-unit generalization (v2-v4, sla.cu now; c1's single-unit source kept as sla_v1.cu)
Goal: case 2 (B=4, 256 units) through the fused path (no qkv_post2 / kv_pack part 2 / join).
- v2: cluster cid takes units cid + k C (4 per cluster for case 2), K/V of all units produced in the prologue, per-unit
  Q, exchange-buffer handshake (remote mbarrier arrive), per-unit gate barriers. Case 0 +6.8 us (Q ready 7.9 us: the
  seq_lens -> rope_tab chain now sits after the production pass), case 2 -8.8 us.
- v3: sla_pack copies each new token's RoPE row into rope_new (bit-exact copy, ceil(L/128) blocks per request) so RoPE
  addresses no longer depend on seq_lens; the producer TMA-stages each unit's Q rows + RoPE rows one unit ahead.
  First version raced (consumers read rope_new before sla_pack was done: 3.8% drift on case 2) -> consumers acquire
  the pack-done counter. Still case 0 +9.3, case 2 -10.6: the extra pack blocks delay sla_pack's completion
  (pack done ~1.5 us after the attention starts instead of ~25 us before it, c1) and production stays slow.
- v4: production moved to producer-warpgroup warps 1-3 (PRODW, software pipelined), whole-unit mode (no split, no
  exchange, s1's unscaled epilogue) for U > 64 when 2 ceil(U/128) <= ceil(U/64).
  Whole mode is BIT-IDENTICAL to attn1 on case 2 and on B1 ctx80000 L1536 (validates the multi-unit machinery; the
  split-mode differences on those shapes, 1.9-4%, are the half-unit numerics: each half quantizes P against its own
  running max). Front: case 0 +8.9/+9.3 (c1 +11.2), case 2 -1.4 (whole), B1 L1536 +4.5 (whole).
  Bottleneck: production passes take ~2.3 us each (cold cache_loc/seq_lens round trips per pass, V byte scatter
  ~0.6 us/pass); case 2 whole mode finishes production at ~25 us, stalling the new-key tiles. Not pursued further
  (case 2 weight 0.012%/us); c1 (single unit, U <= 64) stays the attention path.

## Layer trace (gap/trace.sh, case 0, judge-style gate, 6 reps)
c1: QKV 0-39.7 | sla_pack (side) 2.6-14.7 | sla_fwd 39.4-57.8 (18.3) | sk2 o_proj launched at 46.5 (trigger), ends 77.9 |
bnd3_b1 77.7-90.7 (13.0) | ... | bnd3_b2 509.4-532.0 (22.0) | span 532.0 (s1 ~540-545 in the gap worker's runs).
bnd4 (persistent, 2 stages) at 753 rows: b1 12.7 (-0.3), b2 24.0 (+2.0): not enough rows in flight.
- v5 (no rope_new; RoPE rows TMA-staged straight from rope_tab, consumer production with all loads up front): whole
  mode bit-identical on case 2, B1 L1536, B2 L753; front: case 2 -1.2, B1 L1536 +0.5, B2 L753 +2.3, but case 0 only
  +4.2 (c1 +10.5): the staged Q rows land 7.9 us after the start because the small staging TMAs queue behind the
  ring's 128 KB of tile loads on the same SM. Lesson: latency-critical small loads must not share the bulk-copy queue.
  Multi-unit fused path closed: its per-unit overheads eat the ~11 us of removed pre-attention kernels.
- c1's prologue: GEMM wait returns ~2.2 us after the first CTA; production + Q take 3.1 us more, ~1 us of it the
  seq_lens -> rope_tab HBM chain (with L2-resident RoPE rows the production pass drops from 2.1 to 1.05 us).

## bnd5 (deeper-pipelined persistent b1/b2, 4/3 or 3/2 bulk-copy stages, s prefetched before the PDL wait): rejected
Layer trace (case 0 / case 1): b1 15.2 / 13.8 (4 / 3 stages) vs bnd3 12.8; b2 26.0 / 24.1 vs 21.7. Not bit-exact
either (1e-4: bnd4-style arithmetic differs from bnd3's in the last bits). The 753-row boundaries are bound by the
per-row latency chain (row loads -> block reduction -> coefficient loads -> reduction -> w loads -> stores), with two
rows in flight per SM in bnd3 (two 512-thread CTAs); a persistent CTA serializes rows and loses. ncu's 55% DRAM is
latency, not bandwidth headroom that more bytes in flight could take.

## Micro-overhead pass (root's list) on c1, judge-gated layer trace (gap/trace.sh), cases 0 and 1, 6-8 reps x 2 draws
- Side-stream join before attention: removed on case 0 by the sla flag join (sla_pack done counter). Not applied to the
  attn1 path (case 2 / 5): there the attention grid fills all 132 SMs, so a spin on the side-stream pack could
  deadlock if the pack were ever starved; a graph-edge join must stay.
- Attention early trigger: in sla_fwd (SLA_TRIG): sk2 o_proj launches at 47 us during attention (sla_fwd 39.9-57.7).
- sk2 TRIG at 753 (o_proj config string trig=1 / trig=2): trig=1 lets bnd3_b1 squat the SMs for 35 us (neutral to
  worse); trig=2 moves bnd3_b1 1 us earlier but it runs 1-2 us longer: EVENTS case 0 537.8 / 544.6 vs 540.1 / 541.1,
  case 1 710.3 / 717.7 vs 712.7 / 714.2 (two draws): noise. Not kept.
- bnd3e (bnd3 b1/b2 with the residual-row loads issued before griddepcontrol.wait): worse, b1 14.4 vs 12.9, b2 23.4
  vs 21.4 (the wait appears to drain the outstanding loads, serializing them with the a/f loads).
- Balanced attention split (s1/s2): sla_fwd 17.3-17.6 vs 17.6 us, no layer gain; moves the layer output 2.2%
  (149 routing near-tie flips downstream). Rejected.
- Combine without PDL (0.29 us gap): not attempted (Triton launch; at most ~0.3 us).
- The layer itself is nondeterministic at ~1.4e-4 relative (c1 vs an identical copy, cmp_out.py): layer-level
  "bit-exactness" checks need that floor; attention-level FRONT_CHECK (o_proj output) is exactly 0.

## Final candidate: research/claude/lat/c2.py (= c1 + co-residency guard), base attn1
- attn1 + the sla path for short sliding blocks (4 B ceil(L/24) <= 128 CTAs: case 0 and nearby short sliding
  shapes; everything else runs attn1 unchanged): 2-CTA clusters with a DSMEM handoff, fused qkv_post2 work,
  flag join with the side-stream sla_pack, early PDL trigger for the o projection, cached EXPQ bias, ping-pong
  issue, f16 masks, s1's split (case-0 attention output bit-identical to s1/attn1).
- Guard: the sla path is taken only if all 2-CTA clusters fit co-resident (cuOccupancyMaxActiveClusters: 66 on the
  H200) and >= 4 SMs stay free for the side-stream pack; otherwise attn1's path (no spin-wait deadlock possible).
- Front probe (case 0, vs attn1): +9.9 to +11.6 us across runs (24/24, 16/16, 12/12 wins), FRONT_CHECK exactly 0.
- Layer trace (case 0): attention node 39.4-57.8 us (s1: qkv_post2 39.2-46.2 + join 1.5 + attn_fwd 47.7-68.2).
- Gate: CHECK_RC=0 PASS=42 CASES=21 (gate.sh); c1 also passed the extended gate_lat (27 cases incl. 6 extra sla-path
  shapes) 54/54. Error comparison vs s1's gate JSON (research/claude/lat/gate_c2.json vs stack/gate_s1.json): every
  sliding case identical to s1 to 4 digits (tests/00, tests/02, tests/04, scored/00, /02, /05, edges/00, /02, /04,
  /06); full-layer cases carry attn1's accepted +0.4-0.8% relative.
- ncu (research/claude/prof/ncu.sh, cold caches per kernel, case 0): sla_fwd 19.0 us (tensor 19.6%, issue 30.8%,
  1.82 GHz, grid 128) + sla_pack 6.5 (side stream) replace attn1's qkv_post2 7.3 + attn_fwd 19.5 (+ kv_pack 6.0 side);
  all other kernels unchanged (bnd3_b1 15.1, pnr5 8.5, gather 8.2, combine 10.1, bnd3_b2 22.6).

## Files kept (generated variants pruned; each experiment above is described by its defines)
- c2.py: final candidate (self-contained, 860 KB). c1.py: same without the occupancy guard (gated 54/54, replica'd).
- sla_v1.cu: the kernel source embedded in c1/c2 (SLA_SPLIT 0, SLA_TRIG, SLA_CACHE_C 2, SLA_PP 1, SLA_GPRE 0;
  FMASK on by default). mk_sla.py has since evolved with the multi-unit source (sla.cu = v5; sla_v2-v4.cu history):
  its glue matches sla.cu, not sla_v1.cu, so c1/c2 are the reference artifacts for the v1 path.
- bnd5.cu / mk_bnd5.py, mk_bnd3e.py: rejected boundary experiments. cmp_out.py: whole-layer output comparison.
- trace_sla.py: sla_fwd phase stamps (SLA_TRACE builds). gate_lat.py/.sh: gate + 6 sla-path shapes.

## Final replica: c2 vs attn1, all 8 cases, panel 11,22, 6 reps (research/judge/run_exp.sh)
| case | draw 11 base -> new (us) | draw 22 base -> new (us) |
|---|---|---|
| 0 sliding B1 753 | 537.8 -> 530.0 (x2.0388 -> x2.0754) | 540.8 -> 530.8 (x1.9768 -> x2.0141) |
| 1 full B1 753 | 703.0 -> 705.9 | 770.3 -> 769.6 |
| 2 sliding B4 753 | 1765.1 -> 1769.7 | 1775.5 -> 1774.4 |
| 3 full B4 753 | 2494.7 -> 2492.3 | 2597.8 -> 2598.9 |
| 4-7 (16K, unchanged code) | +/-150-330 us both ways | +/-50-300 us both ways |
DRAW_SCORE base 1.9717 / 1.9562, new 1.9900 / 1.9840; RELATIVE 1.0093 / 1.0142, mean 1.0118.
The headline is inflated by the 16K first-turn draw noise (cases 4-7 run attn1's code unchanged; their per-case
ratios move with the paired SGLang timings). Trust case 0 (-7.8 / -10.0 us; turn-1 load +1.17% / +1.11%):
expected score gain ~ +0.37% (9 us x 0.040%/us). Cases 1-3 (identical code paths) within noise.

## Open ideas (for whoever continues)
- Per-tile cost of the sliding segments: 0.82 us vs a 0.50 us tensor floor. The online-softmax rescale (~0.2 us/tile,
  ~90% of tiles in 8-9 tile segments) dominates; headroom/trigger changes are precision changes (not allowed on
  sliding). A structural option is QK(j+1) overlapped with softmax(j) (double-buffered S, +32 registers per thread).
- Prologue: ~1 us of c1's Q-ready is the seq_lens -> rope_tab HBM chain. An L2-resident copy of the new tokens' RoPE
  rows written by a side-stream kernel removes it (production pass 2.1 -> 1.05 us), but adding those blocks to sla_pack
  delayed its completion from ~25 us before the attention to ~1.5 us after it (block scheduling under the QKV GEMM:
  not understood). sincosf instead of the table is bit-identical but costs more than it saves (+0.6 us).
- Epilogue: 32 MUFU per thread for the sigmoid gate (~0.45 us at 3 warps/SMSP). Precomputing them on idle warps
  (GPRE) slowed rank 1 for reasons not found.
- Case 2 / B>=2 short sliding: the whole-unit fused path (sla.cu v5, mk_sla.py --whole 1) is bit-identical to attn1
  but only -1.2..+2.3 us; it needs fast K/V production (per-pass cold loads, V byte scatter ~0.6 us per pass).
