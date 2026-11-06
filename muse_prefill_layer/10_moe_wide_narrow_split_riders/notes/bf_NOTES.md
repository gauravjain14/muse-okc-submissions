# bf: boundary fusion (UB / D6 fused GEMM + residual/RMSNorm) at T = 753 / 3012 (2026-10-31)

Base: research/claude/stk/s_b.py (submitted, judge x1.9961). Goal: replace cuBLASLt/sk2 GEMM + bnd3/bnd4 boundary
kernels at T=753 / 3012 by the 16K fused kernels (UB = unpool + post-FFN, D6 = o_proj + post-attention).
s_b times (nsys, research/claude/prof/nsys/s_b.summary.txt):
- case 0 (753 sliding): o_proj sk2 (early PDL launch from sla_fwd) ends 19.9 us after attention, bnd3_b1 13.0;
  unpool nvjet 65.0 + bnd3_b2 21.5 = 86.5 us.
- case 2 (3012 sliding): o_proj nvjet 63.3 + bnd4_b1 51.5 = 114.8; unpool nvjet 249.5 + bnd4_b2 88.2 = 337.7 us.
Score sensitivity: case 0 0.040 %/us, case 2 0.012 %/us (root NOTES).

## 1. What gates the fused path in s_b.py (line numbers of research/claude/stk/s_b.py)
- kernel_fn (s_b.py:2550). D6: s_b.py:2563 `d6 = state.get('d6_plans', {}).get(tokens) if o.is_contiguous() and
  data['stream'].is_contiguous()` -> s_b.py:2564-2570 (d6.run(..., pdl=False), attn_output = d6.f, ra = d6_ra) else
  s_b.py:2571 `_gemm(o, dense['o'])` (sk2 at 753 via SK2_ROUTE s_b.py:2191, cuBLASLt at 3012) + s_b.py:2578 `_bnd1`.
- UB: s_b.py:2575 `ub = state.get('ub_plans', {}).get(tokens) if fast_bnd` (fast_bnd: tokens > 8 + contiguity,
  s_b.py:2574) -> s_b.py:2631-2636 ub.run(...) returns (xout, sout); else unpool `_gemm` (s_b.py:2641) + `_bnd2`
  (s_b.py:2643).
- The tables: s_b.py:2650 `UB_ROUTE = {16384: {'G': 4, 'R': 16, 'tail': (2, 1, 1), 'bcap': 16}}` (built in prepare
  s_b.py:2653-2657), s_b.py:2660-2665 `state['d6_plans'] = {16384: D6Plan(16384, bk=4, G=2, R=16, bcap=32,
  tail=(2, 1, 1))}`. I.e. an exact-T dict lookup: only T == 16384 takes the fused kernels.
- Tail items: s_b.py:2868-2928 patch_tail; s_b.py:2929-2930 `_UB_BODY = patch_tail(_UB_BODY, 4, 1)`,
  `_D6_BODY = patch_tail(_D6_BODY, 4, 2)` (BN_TAILR / BN_TAILK compile-time, overridable by plan defines).
- Boundary kernels otherwise: s_b.py:1529-1537 `_bnd_launch`: rows >= BND4_MIN_ROWS (2048) -> bnd4_* (persistent,
  132 CTAs), else bnd3_* (one CTA per row).
- No T >= 8192 assumption in the plan code: UbPlan / D6Plan take M; tiles = ceil(M/256) x 43; grid = min(132,
  ntiles); counters zeroed by the last CTA. Constraints: R must divide 256 (IPM = BM / R) and R <= 63 (6-bit row count
  in the item word); row0 is 16 bits (M <= 65535). The D6 launch mode is hard-coded pdl=False at s_b.py:2569.

## 2. Files
- ov_bf.py: overlay appended to s_b.py. BF_UB / BF_D6 tables (T -> plan kw); _BfUbPlan / _BfD6Plan (launch mode per
  plan); bf plans compile their own copy of the bodies (marker define BF_BODY) so the 16K kernels' source is byte-
  identical to s_b's; BN_PREB (D6): the producer fetches its first item before griddepcontrol.wait and issues that
  tile's first BN_PREB weight stages before the wait (like sk2's npre at 753), activations after it.
- mk_bf.py: builds the candidates (s_b.py + ov_bf.py + tables) and holds the preset tables (UB753, D6753, UB3012,
  D63012). Candidates: ub753.py d6753.py both753.py ub3012.py d63012.py both3012.py bothall.py.
- sim.py: discrete-event model of ub_fetch scheduling (G / R / bcap / tail) -> knob choice before measuring.
- lcompile.py: local nvcc 12.1 sm_90a compile of every bf plan kernel (+ --variants, --trace, --ref16k).
- unit_bf.py: kernel-level identity vs s_b's path on identical inputs. probe_bf.py: in-situ A/B of plan variants
  (whole-layer graphs, judge-style cold L2, one process) + output check vs A/A + per-CTA UB_TRACE summaries.
  variants.py: the variant tables. run1.sh: unit + cmp + gate + replica + nsys trace (on the sandbox).

## 3. Plan parameters for M = 753 / 3012 (132 SMs; tiles 256 x 192, K 4096 UB / 1024 D6)
- What the 16K plans assume: 64 m-tiles x 43 n-tiles = 2752 tiles = 20.8 waves, so groups of G m-tiles complete
  throughout the GEMM and their rows stream on <= bcap CTAs while tiles remain; R 16-row items; the last m-tiles as
  4-row items (tail patch). Nothing in the code needs T >= 8192.
- M = 753: 3 m-tiles x 43 = 129 tiles = ONE wave (grid 129). All m-tiles complete at about the same time, so every
  boundary row follows the GEMM; G, bcap, tail groups are inert. What matters: R (balance 753 rows over 129 CTAs:
  5.8 rows each). R 4 everywhere (189 items, busiest CTA 2 items = 8 rows); R 2 (377 items, 3 items = 6 rows but 2x
  per-item overhead). D6: PDL launch + BN_PREB=4 (sk2's early weight stages).
- M = 3012: 12 m-tiles (last 196 rows) x 43 = 516 tiles = 3.91 waves. sim.py: the old 3012 knobs (G 4, tail 2.1.1,
  R 8, bcap 16) model at UB 355 us vs ~314 best: with 69 us tiles and only 12 spare tile slots in 4 waves, boundary
  rows streamed mid-GEMM push CTAs into a 5th tile; G 3 makes each group ~one wave (129 tiles) so rows of group g
  stream during group g+1. (Model caveat: rows after the GEMM on all CTAs are HBM-bound: 112 KB/row UB, 3012 rows =
  337 MB = the 88 us of bnd4_b2; the model's 1.5 us/row there is optimistic.)
- Presets: UB753 G1 R4 (TAILK 0), D6753 G1 R4 PDL+PREB4, UB3012 G3 R8 bcap16 TAILR4 TAILK2, D63012 G3 R8 bcap32
  TAILR4 TAILK2 PDL+PREB4. Sweeps in variants.py.

## 4. Compile (local nvcc 12.1, -gencode=arch=compute_90a,code=sm_90a, lcompile.py --variants --ref16k)
- Every bf production kernel (ub2_2_192_3 / ub2_1_192_3 / ub2_2_192_4_4 with all variant defines): 168 regs (the
  __launch_bounds__(384, 1) cap; consumers raise to 232 by setmaxnreg), 0 stack, 0 spill, no C75xx wgmma warnings,
  8 HGMMA. Same as the 16K kernels. UB_TRACE builds of rm=2 UB: 8 B stack / spill (diagnostic builds only).

## 5. Measurements (sandbox H200, fields seed 11; probe_bf.py = whole-layer graph replays, 256 MiB flush before each)
### u1 (unit_bf.py, u1.log): kernel-level vs s_b's path, identical inputs, T 753 / 3012 / 760 / 3040 / 2048
- GEMM halves bit-exact (D6 vs sk2 at 753 and cuBLASLt at 3012; UB vs cuBLASLt) at every T.
- Boundary outputs: 2-20 elements per million differ by <= 1 BF16 ulp (rel 3e-6 .. 2e-5; UB stream rel <= 1.4e-5;
  ra 1 FP32 ulp in 17-19 % of rows): summation order of the two RMSNorm reductions (bnd3/bnd4: 512 threads, per-thread
  16-element chains + 2-level shuffle tree; UB/D6: 256 threads, 4-chain FMAs + sum8). Same numerics class as s_b's 16K
  path. UNIT_RC=0.
### p1 (p1.log): first sweep, cases 0-3, 16 reps, base = s_b path, d = median paired difference (+ = slower)
- Output check vs base (A/A of base in brackets): UB variants inside the A/A envelope (case 0 1.7-2.2e-4 [2.0e-4],
  case 2 1.1-1.2e-4 [1.2e-4]); D6 variants 2.9-3.9e-4 (2-5x A/A: the 1-ulp h / ra differences of a few rows are
  amplified through router -> MoE -> unpool; topk / caches exact). Far inside the judge bounds, but above the A/A
  envelope -> a bit-identical D6 reduction order would be needed to claim "within the envelope".
- Every fused variant LOSES: case 0 UB +7.6..+8.8 us (rm=1 tiles +23..26), D6 +5.5..+9.8 (PDL/PREB +1..2 us better
  than without), both +16.4; case 1 similar (+5..+9 each); case 2 UB +15..+34 (old knobs ub_c +32.7, G12 ub_b +15.2),
  D6 +10..+20; case 3 UB +11..+32, D6 +4..+16.
### t1 (t1.log): per-CTA UB_TRACE in situ + gated nsys node trace (s_b vs bothall)
- case 0 nsys: s_b o_proj sk2 ends 20.0 us after sla_fwd + bnd3_b1 13.2 = 33.0 us; D6 ends 40.6 us after sla_fwd
  (+7.6). s_b unpool 64.8 + bnd3_b2 21.4 = 86.2 us; UB 96.1 (+9.9). case 2: s_b o_proj 62.6 + bnd4_b1 51.4 (ends
  115.6 after attention); D6 ends 126.8 after attention (+11); s_b unpool 251.6 + bnd4_b2 87.0 = 338.6, UB 369.5 (+31).
- WHY at 753 (one wave of 129 tiles): the GEMM halves are on par (D6 tile 19.1 us ~ sk2's 20; UB tile 63.4 ~ nvjet
  64.8), but every boundary row waits for the single wave and then runs serialized per CTA: D6 1.9 us/row, UB 2.9-3.1
  us/row (<= 8 rows per CTA at R 4) -> tails 17-19 us (D6) / 26-27 us (UB) vs the standalone bnd3_b1 13.2 / bnd3_b2
  21.4, which keep ~4 rows in flight per SM (753 one-row CTAs, 2 per SM). The tail traffic (~85 MB UB, s/a from HBM)
  is HBM-bound either way, so fusion has nothing to overlap it with.
- WHY at 3012: UB with rows streamed during the GEMM (G3 bcap16): tiles 66.5 us (vs 63.3 when no rows stream: HBM /
  L2 contention), rows 2.2-2.5 us each during the GEMM = 55 us of SM-time per CTA, 12-13 % idle and some CTAs doing 5
  tiles (3.91 waves of 66 us tiles cannot absorb the stolen SM-time): 360-367 us. UB with every row after the GEMM
  (G12, ub_b): GEMM 255.7 us (~ nvjet 251.6) + tail 92 us at 3.5 us/row (= bnd4_b2's 87, HBM-bound): 348 us. D6: GEMM
  SM-time 72 us vs nvjet 62.6 (tile epilogue stores not overlapped: ~1.9 us of each 18.6 us tile) + rows 34 us/CTA.
### p2 (p2.log): 753, L2 prefetch of s (+ a) in the last wave (BN_PFS_KB), R 2 vs 4 (cases 0, 1)
- Prefetch: no effect on the tail (UB rows 3.5 us, D6 2.2 us) and UB tiles 63 -> 67 us: UB +12..+13 (vs +8.8 without),
  D6 +5..+8. R 2 (377 items, <= 6 rows per CTA): UB +6.3..+8.6, D6 +5.3..+6.0 (tail still 26 / 17 us).
### p3 (p3.log): what bounds the tail (UB ablations, timing only) + W evict_first (cases 0, 2)
- 753 UB with NO row loads (UB_BND_NOLOAD): tail 26 -> 18 us, layer +0.5 us vs s_b = a TIE at best; no stores +2.7.
  => UB at 753 cannot win: fill + tile + a load-free tail = nvjet + bnd3_b2.
- 3012 UB, all rows after the GEMM (G 12): real +16 us; NO loads -32.5 (tail 94 -> 48); no stores -19.5. The 3012
  tail is HBM-bound (s + a = 145 MB of row reads cannot be staged in a 50 MB L2).
- W tile loads with an L2 evict_first policy (BN_WEF) +/- the prefetch: tiles slower (+4..+15 us GEMM), tail unchanged
  -> rejected. (evict_last on the prefetch not used: such lines could outlive the judge's L2 flush.)
### p4 (p4.log): 3012 schedule: rows interleaved on every CTA after each tile (bcap 64 / 132), cases 2, 3
- UB G3 R8 bcap 132 / 64: +4.2 / +3.0 (case 2), +5.6 / +4.6 (case 3) (vs bcap 16 +32.7, bcap 32 +22): no 5-tile CTAs,
  idle 5 %; tiles 65.7 us (63.1 with no rows streaming), rows 2.46 us during the GEMM, tail 27 us. R4 +12..+15, R16
  +22..+24, G4 +13, G2 +6..+8.5. D6 G3 R8 bcap 132: +9.8 / +14.3; R4 +9..+17; G2 +8..+11; G4 +9..+23.
### p5 (p5.log): 3012 per-m-tile L2 prefetch (BN_PFT_KB: each tile prefetches 1/43 of its m-tile's rows)
- s only +14 / +15.5, s + a +22 / +22 (tiles 65.9 -> 70 us, rows not cheaper: 2.33 -> 2.44 us), with W evict_first
  +30 / +31, BN_WEF alone +22 / +10. D6 + prefetch +16..+22. => prefetch moves the boundary's HBM traffic into the
  GEMM, slowing the tiles, without making the rows cheaper. Direction rejected at both sizes.
### p6 (p6.log): 3012 smaller last groups (16K-style tail groups), D6 753 ablations
- UB G3 R8 bcap 64/132 tail (2,1): +6.8 / +3.8 (bcap 64), +3.5 / -0.0 (bcap 132); tail (1,1,1) +7.3 / +1.3; (1,1)
  +4.7 / +5.8; (2,1,1) +34.7 / +6.6; (2,1) TAILR 2 +3.4 / +3.5. Tail after the last tile 27 -> 15-18 us, but the
  same-run reference u3_i8b moved +14.7 / +5.3 (vs +3.0 / +4.6 in p4): run-to-run noise ~ +-5-10 us at 3012.
  => UB at 3012 is at break-even (0..+7 us); no consistent win.
- D6 3012 tail (2,1) / (2,1,1): +14.5 / +8.8, +11.8 / +5.8.
- D6 753 tail ablations (trace): real tail 17.8 us, NO loads 11.8 us, no stores 16.2 us; D6 span from its first item
  32 us even load-free, i.e. ~34 us after attention vs s_b's 33 => D6 at 753 cannot win either.

### run1a / run1b (run1.sh unit, cmp, gate) with the final presets (mk_bf.py: UB753 R2; D6753 R2 PDL PREB4; UB3012
### G3 R8 bcap132 tail (2,1) TAILK2; D63012 G3 R8 bcap132 tail (2,1,1) PDL PREB4)
- cmp vs s_b (cases 0-3, 2 fresh inputs each; s_b A/A first): UB-only candidates inside the A/A envelope (e.g. case 0
  ub753 2.06e-4 vs A/A 2.0e-4; case 1 rep 1, where s_b's A/A was exact: ub753 2.7e-5 = UB's own diff); D6 candidates
  3.2-4.4e-4 (A/A 0.5-2e-4); topk_ids, k_cache, v_cache exact everywhere.
- gate (research/claude/gate_check.py had to be synced to the sandbox): ub3012 CHECK_RC=0 PASS=42 CASES=21; bothall
  CHECK_RC=0 PASS=42 CASES=21.

### run1c (run1.sh rep): judge replica, panel 11,22,33, cases 0-3, 8 reps, every candidate in its own AB/BA pair
### (full data: run1c_rep.json; run1c.log keeps the last 40 lines)
- RELATIVE vs s_b (draws 11 / 22 / 33): ub753 0.9930 0.9928 0.9931 (mean 0.9929); d6753 0.9929 0.9981 0.9957
  (0.9956); ub3012 0.9970 0.9994 0.9997 (0.9987; turn-4 load lower on all 3 draws: -0.44 / -0.16 / -0.29 %); d63012
  0.9959 0.9990 0.9953 (0.9967); bothall 0.9830 0.9837 0.9830 (0.9833). => the old b23-era losses (0.9877 / 0.9960)
  were real, not the replica's position bias. Panel 44,55,66 not run: nothing looked positive.
### p7 (p7.log): 3012 finer tiles, rows interleaved
- UB 256x128 tiles (768 = 5.8 waves): +55..+78 us; 128x192 (1032 = 7.8 waves): +57..+71: GEMM SM-time 321-339 us vs
  262 for 256x192 (tile 55 us for 2/3 of the work, 42.8 us for 1/2). D6 256x128 +3..+13, 128x192 +7..+14 (= 256x192).
### run1d (run1.sh trace, gated nsys node trace, final presets): node durations, us
- case 0: both753 D6 node ends 96.4 (s_b bnd3_b1 ends 90.8: +5.6); UB 94.7 vs nvjet 64.5 + bnd3_b2 21.2 = 85.7 (+9.0).
- case 2 / 3: ub3012 UB 337.9 / 339.1 vs nvjet 249.8 + bnd4_b2 87.5 = 337.3 / 337.2 (+0.6 / +1.9: break-even);
  d63012 D6 ends 5.6 / 3.3 us after s_b's bnd4_b1.

## 6. Verdicts (coordinator directions)
1. Overlap boundary HBM traffic with the GEMM (L2 prefetch of s / a): MEASURED LOSS at 753 (p2, p3) and 3012 (p5):
   the prefetch does not shorten rows (753: rows are not load-latency bound; even zero-cost loads only tie) and its
   traffic slows the tiles 4-7 %. Unfused variant (prefetch s for bnd3_b2 / bnd4_b2 during the cuBLASLt GEMM): needs
   either an L2 persisting access-policy window (persisting lines can survive the judge's cold-L2 flush, i.e. carry
   data across replays: rejected on rules) or a prefetch from the kernel before the GEMM (the 64 MB weight stream of
   the unpool GEMM evicts it before bnd*_b2 runs: same class as stk lever A, measured <= 0.7 us). Not pursued.
2. 753 tail parallelism: bounded by the load-free ablations (UB ties, D6 ~ +1 us): more rows in flight attacks load
   latency, which is not what bounds the 753 tail. The remaining floor is compute + 2 block reductions + stores of
   one row at a time per CTA (1.4 us / row D6, 2.3 us / row UB with all CTAs); a per-WG row pipeline would need 64
   columns x (3-4 coefficient vectors + y) per thread = 256-320 registers > 232. Not built.
3. D6 epilogue overlap at 3012: bounded: D6 GEMM SM-time 73-75 us vs nvjet 62.6; hiding the 1.9 us epilogue of each
   of ~3.9 tiles per CTA gives <= 7.4 us -> D6 3012 from +6..+15 to ~ -1..+8: no consistent win reachable (and the
   previous bnd worker's fused ping-pong lost at 16K to the boundary joins). Not built.
4. D6 bit-identical reduction order: moot (D6 loses at both sizes). Note for any future D6 use below 16K: the
   whole-layer diff is 2-5x the A/A envelope (kernel-level 1-ulp h / ra differences amplified by the MoE path).

## 7. Conclusion
No consistent win: s_b stays the incumbent, nothing to stack. The fused GEMM + boundary kernels pay at 16K because the
boundary rows (1.9 us of SM time each) hide inside a 20-wave GEMM and replace an HBM-bound standalone pass (3.9 us of
SM time per row). At 3012 (3.9 waves) the same exchange is break-even: fused rows cost 2.3-2.5 us of SM time plus
4-7 % slower tiles while they stream, against bnd4's 3.8 us per row; at 753 (one wave) the rows cannot overlap the
GEMM at all and the fused tail moves rows no faster per SM than bnd3 (even with zero-cost loads it only ties).
Built and kept for reference: ov_bf.py (BN_PREB, BN_PFS, BN_PFT, BN_WEF, _BfUbPlan / _BfD6Plan, separate bf body
copies), the 7 candidates (mk_bf.py presets = best measured settings; ub3012 and bothall gated PASS=42), run1.sh.
GPU released (no sandbox start / stop issued).
