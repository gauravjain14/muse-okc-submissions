# WP-MIX: queue order for bs128 / bs256 (heavy-first / light-later phase split) (2026-11-01)

Worker WP-MIX. Base dev/moe_v1512.py (line B); every patch also applies to dev/moe_v1504.py (line A). Namespace
dev/moe_v3700-v3799.py, scripts dev/wpmix_*, logs kda_ws/runs/wpmix_runN.log. Target kernels moe_8_8_1_3_4 (bs65-128)
and moe_8_8_1_3_8 (bs129-256).

## Files
- dev/wpmix_mk3700.py BASE OUT (first form: orders + trace) -> dev/moe_v3700.py (B) / dev/moe_v3750.py (A).
- dev/wpmix_mk3710.py BASE OUT (= mk3700 + MOE_WPMIX=3 + MOE_WPMIX_PIPE) -> dev/moe_v3710.py (B) / dev/moe_v3760.py (A).
  Default builds = the base's SASS in all 16 kernels (NVRTC 13, MOE_S5R_FIT; checked for v3700, v3710 vs v1512 and
  v3750, v3760 vs v1504). No ptxas C7511/C7512 in any build below.
- Baked: dev/moe_v3701.py (v3700 + MOE_WPMIX=2), v3702 (+ MOE_WPMIX=1 = mix5), v3703 (+ MOE_WPMIX=1,MOE_WPMIX_K=0),
  v3711 (v3710 + MOE_WPMIX=1,MOE_WPMIX_PIPE=6); FINAL: dev/moe_v3720.py (line B, v3710 + MOE_WPMIX=1), dev/moe_v3770.py
  (line A, v3760 + MOE_WPMIX=1); K 8/5 twins dev/moe_v3721.py / v3771.py (not kept).
- dev/wpmix_trace.py (per-item trace of MOE_WPMIX_TR builds: classes, code switches, gaps, tail, 10-us occupancy),
  dev/wpmix_b2an.py (bench2 json -> per-copy ratios vs the middle controls), dev/wpmix_calls.py (interleaved calls of
  every config, bs 1-512), dev/wpmix_pipechk.py (PIPE decode invariants), dev/wpmix_bench2i.py (bench2 + an idle gap
  between flush and replay, env WPMIX_GAP_US), dev/wpmix_run1.sh, run2.sh, run2b.sh, run3.sh, run4.sh; logs
  kda_ws/runs/wpmix_run{1,2,2b,3,4}.log; data dev/remote_out/wpmix_*.json, nsys_v37*.sqlite, nsysjm_wpmix_*.sqlite.

## 1. Why MOE_RANK_MIX=4 (dev/moe_v1590.py) is wrong
hang1 (run 1, full output): bs17 / 24 / 32 FAIL "output contains NaN or Inf"; bs33, 48, 64, 65, 100, 128, 129, 200, 256,
257, 300, 512 ok. With MOE_NO_WP32 (the bs17-32 fast start off) bs17 / 24 / 32 are ok.
Cause: MOE_RANK_MIX=4 applies to every NT >= 4 kernel, which includes the (8,4,1) MW 1 kernel moe_8_4_1_4_1 (bs17-32).
That kernel runs wp32_fst (MOE_WP32_FST, v831-v837): warp 8 issues the CTA's first item's first NS stages (stage records,
weights, scales, x rows) right after the routing bitmaps, BEFORE the tile table exists, and derives that item's expert
itself as "the tile0-th expert in count-desc / index-asc order" (redux.sync.max over count << 5 | 31 - lane keys) with
the pair offset p0 = the larger experts' counts. Under any permuted rank (MIX, also MOE_RANK_SF) the tile table and the
queue disagree with wp32_fst: the first stages carry another expert's weights / x rows / record while the consumers and
the epilogue use the queue's item -> garbage partials -> NaN. The old "bs32 NaN (stale)" note is this bug. The (8,4,1)
MW 2 (bs33-64: hang1 ok under MIX=4) and (8,8,1) kernels (hang1 + check_mt 0 failures under every WP-MIX order) do not
depend on the descending order: tsplit / KSX / KRS / CLD3 / GATHER_MIN / buffer sizes are order-agnostic (the split
tiles are the queue's LAST nspl tiles whatever their size; pair offsets and the gather follow the tile table). The
suspected invariants (k-split of the last tiles, late claim, buffers sized for small split tiles) are NOT the cause; but
with MIX the split wave lands on median tiles (2-group pieces 4.5 us vs 3.9 us), which costs ~0.2% (mix0 vs mix5).
The WP-MIX switches therefore apply only to NT >= MOE_WPMIX_NT (8) and MW <= MOE_WPMIX_MW (8) kernels: bs65-256.

## 2. Measurements (run 1, kda_ws/runs/wpmix_run1.log)
Correctness (run 1): hang1 v3701 / v3702 / v3703 at 32, 64, 65, 100, 128, 129, 160, 200, 256, 257, 300, 512 all ok;
check_mt (17 24 32 33 48 64 65 100 128 129 160 200 256 257 300 512, normal + skewed, 2 calls each) 0 failures for all
three; dev/wpmix_calls.py (interleaved bs 1-512) v3701 120 calls / v3703 80 calls 0 failures (worst 3.29e-3);
dev/wps2_mix.py v3701 0/60. Note: with K = 5 the split tiles are the same 5 smallest experts as in the base, so the
outputs match the base's error values exactly (the KSX pieces' fp32 sums are the only order-dependent rounding).

Orders (expert rank -> queue position; nz = experts with tokens, the K smallest keep their place at the end):
- mix0 (MOE_WPMIX=1,K=0 = MOE_RANK_MIX's order): r0 r(nz-1) r1 r(nz-2) ... (the split tail lands on median tiles)
- mix5 (MOE_WPMIX=1): the same over the first nz - 5 ranks, the 5 smallest (the split wave) stay last
- spr5 (MOE_WPMIX=2): the wgmma experts (> 16 tokens) evenly among the light ones (both descending), 5 smallest last

bench2 run 1 (jm seeds, 100 reps, SGLang interleaved, fillers fA/fA2 + fB2/fB at the ends; middle a1 s1 m1 x1 a2 ...
a4: 4 v3700 controls, 3 copies of each candidate; % = median over reps of mean(controls) / candidate):

| | bs32 (placebo: kernel unchanged) | bs128 | bs256 |
|---|---|---|---|
| controls band, pass 1 / 2 | -0.07..+0.06 / -0.14..+0.20 | -0.11..+0.14 / -0.11..+0.14 | -0.04..+0.07 / -0.09..+0.12 |
| mix5 (v3702) | +0.08 / +0.11 | +0.49 (+0.61 +0.41 +0.38) / +0.48 (+0.49 +0.47 +0.48) | +0.65 (+0.63 +0.70 +0.72) / +0.60 (+0.68 +0.63 +0.67) |
| mix0 (v3703) | -0.06 / +0.02 | +0.19 / +0.28 | +0.45 / +0.42 |
| spr5 (v3701) | -0.08 / -0.04 | +0.02 / +0.13 | +0.21 / +0.15 |

mix5: every copy above every control at bs128 and bs256 in both passes (6/6 + 6/6). The gain is uniform over routing
strata (bs256: +0.49 with 20-23 experts, +0.64 with 24-27, +0.80..+0.97 with 28+; hot > 64 tokens or not: same).
Old note (plan_mix2 on v1033: MIX=8 bs256 -1.1%) does not reproduce on v1512 (mix0 is +0.4%).

Per-item trace (MOE_WPMIX_TR twins, 8 jm seeds per shape; the trace build's stage loops are re-scheduled vs the real
build (opcode similarity 0.33-0.38), so only relative statements; all traced variants share identical loops):
- span (first data -> last item end) bs128 148.8 / mix0 147.9 / mix5 148.5 / spr5 148.7 us; bs256 181.0 / 180.6 / 181.4 /
  181.2: in the traced build the order changes nothing measurable. Busy fraction 94-95% (items), gaps between a CTA's
  items ~6 us per CTA (0.57-0.66 us each, the same whether the next item has the same token-group count or not: no
  code-switch cost), tail idle 2.8-3.6 us.
- Item durations do not depend on the neighbours: bs256 G1 1-group 15.9 us (0.95-1.07 us per stage), 2-group 18.1,
  3-group 21.5, 8-group 28.1 (1.71-1.76 us per stage), down items the same, split pieces 3.9 (1-group) / 4.5 (2-group).
  1-group stages are 5-7% slower in the all-light window of the base order (1.01-1.04 us at 50-110 us vs 0.95-0.97 at
  120-170 us) -- the same window where the GPC clock dips (1.50-1.53 GHz vs 1.65-1.70), i.e. mostly the clock.
- Occupancy (CTAs in wgmma items per 10 us, bs256): base 129 132 111 78 58 16 12 0 2 6 12 32 56 65 46 31 9 ...;
  mix5 83 94 97 67 71 50 33 16 11 13 15 27 46 64 48 32 20 ...; spr5 78 85 77 39 48 45 46 42 36 30 20 26 37 51 47 41 27.

nsys run 1 (sprint seeds 70261/70449/70513, 48 reps, 75 kHz; corrected analysis dev/plan_nsys_an2.py: sample window
13.3 us, midpoint bins, DRAM calibrated on the 512 MiB flush = 4.89 TB/s; dev/plan_nsys_int.py per-kernel integrals):

| bs256, 10-us bins from kernel start | DRAM TB/s | tensor % | GPC GHz |
|---|---|---|---|
| base v3700 (= v1512) | 0.47 1.05 1.15 1.48 2.42 2.75 3.04 3.75 3.91 4.35 4.57 4.54 4.29 3.77 3.37 2.97 2.84 2.90 3.17 3.95 | 15 51 71 62 51 52 48 40 37 30 25 25 29 38 42 44 46 38 33 27 | 1.84 1.72 1.73 1.76 1.76 1.69 1.60 1.53 1.51 1.54 1.59 1.63 1.65 1.66 1.67 1.69 1.70 1.72 1.75 1.75 |
| mix0 v3703 | 0.48 1.05 1.26 1.70 2.60 2.78 3.26 3.85 4.06 4.16 4.32 4.21 4.10 3.78 3.31 3.28 3.05 2.97 3.19 3.78 | 12 47 69 60 47 48 41 33 34 34 34 37 36 36 41 41 46 42 35 29 | 1.83 1.72 1.73 1.77 1.77 1.69 1.60 1.53 1.51 1.54 1.59 1.62 1.64 1.66 1.66 1.67 1.69 1.71 1.75 1.75 |
| spr5 v3701 | 0.47 1.03 1.23 1.82 3.37 3.19 2.75 3.19 3.56 4.17 4.45 4.46 4.19 3.86 3.73 3.48 3.11 3.05 3.01 3.96 | 13 49 70 62 45 46 48 45 39 31 27 26 31 38 40 42 47 41 36 29 | 1.84 1.72 1.72 1.77 1.77 1.69 1.59 1.52 1.51 1.55 1.60 1.63 1.64 1.65 1.67 1.68 1.69 1.71 1.74 1.74 |
| mix5 v3702 (run 2) | 0.50 1.07 1.32 1.61 2.70 2.77 3.16 3.69 3.83 4.00 4.26 4.31 4.26 3.87 3.47 3.08 3.04 3.14 3.03 3.85 | 13 49 69 61 46 49 43 35 37 36 33 33 33 35 40 46 47 40 35 29 | 1.82 1.73 1.74 1.78 1.77 1.69 1.58 1.50 1.47 1.50 1.56 1.60 1.64 1.65 1.65 1.65 1.66 1.68 1.72 1.71 |

Per-kernel integrals bs256: base 627 MB, 2.99 TB/s avg, issue 51.9%, tensor 38.6%, 1.688 GHz; mix0 2.99 / 52.5 / 38.9 /
1.682; mix5 2.97 / 52.8 / 39.1 / 1.664 GHz. What the orders change: the light-only window (80-120 us) is flatter (DRAM
peak 4.57 -> 4.26-4.32 TB/s, tensor trough 25% -> 33-34%). What they do NOT change: the first ~40 us (DRAM 0.5-1.7 TB/s,
tensor 50-71% in EVERY order, i.e. startup + first-wave items, not the order) and the GPC clock dip (1.47-1.53 GHz at
70-90 us in every order and at bs128 / bs32 too: a power-limiter transient ~50-60 us after a high-power kernel starts,
recovering to 1.65-1.75 by 130 us; mixing makes it slightly deeper, 1.47 vs 1.51). A power-aware order cannot remove
it: the kernel is consumer-compute-paced in every phase (trace: light stages 0.95-1.04 us, heavy 1.71-1.76 us per stage,
both clock-bound), so the energy per call and the limiter's response are order-independent.

## 3. Run 2 / 2b (kda_ws/runs/wpmix_run2.log, wpmix_run2b.log): K, ratio, and the pipelined order
Run 2's bench (21 variants) ran out of memory (every B2_SHARE_EQ variant keeps its own ~6.5 GB of weights alive:
<= 17 variants per bench2 process); its correctness / trace / nsys steps are valid; run 2b repeats the bench (17 variants).
Correctness: hang1 64-512 ok for PIPE6, mix5+PIPE6, mix5+PIPE8, R2, K3, K8; check_mt 16 sizes 0 failures for PIPE6,
mix5+PIPE6, mix5+PIPE8, R2; wpmix_calls mix5+PIPE6 0/120.

bench2 run 2b (3 controls, 2 copies each, 100 reps x 2 passes; % vs control mean, per copy):

| | bs32 (placebo) | bs128 | bs256 |
|---|---|---|---|
| controls band p1 / p2 | -0.03..+0.04 / -0.06..+0.12 | -0.17..+0.14 / -0.19..+0.12 | -0.18..+0.15 / -0.15..+0.14 |
| mix5 (K 5) | -0.06 / -0.00 | +0.47 (+0.42 +0.51) / +0.51 (+0.51 +0.47) | +0.75 (+0.78 +0.69) / +0.73 (+0.79 +0.70) |
| K 3 | -0.14 / -0.16 | +0.39 / +0.43 | +0.66 / +0.87 |
| K 8 | -0.12 / +0.04 | +0.56 (+0.40 +0.53) / +0.72 (+0.61 +0.65) | +0.58 / +0.59 |
| R 2 (1 large : 2 small) | +0.05 / -0.01 | +0.38 / +0.34 | +0.59 / +0.74 |
| mix5 + PIPE 6 | -0.06 / -0.02 | -0.94 (-0.91 -0.87) / -0.78 | -0.75 (-0.76 -0.63) / -0.73 |

- Every mix variant with the split wave kept on the smallest experts lands at +0.4..+0.9%; mix5 is the most stable,
  K 8 is a little better at bs128 (+0.64 vs +0.49 mean) and worse at bs256 (+0.59 vs +0.74) -> v3721 = K 8 at MW <= 4,
  K 5 at MW 8 goes to the confirmation run.
- PIPE (MOE_WPMIX_PIPE=L: the planner's "heavy G2 alongside light G1"): correct (deadlock-free by claim order, checked
  bijective / G-before-D / first wave unchanged / split pieces last for ntiles 1..69, L 5..12: dev/wpmix_pipechk.py) and
  loop-exact in its remap form (MOE_WPMIX_PH 18: every stage + producer loop byte-identical, +1 / +2 LDL outside the
  loops), but it LOSES 0.7-0.9% (1.3-1.5% vs mix5). Trace: the gaps between items grow 0.60 -> 0.83-0.86 us at bs256,
  0.65 -> 0.82 / 1.18 us (G1 -> G2 transitions) at bs128: the down items claimed L * 48 = 288 items after their tile's
  last gate/up item wait in the producer's readiness poll; heavy down items take 29.2-29.7 us instead of 28.1-28.7.
  nsys (sprint seeds): the heavy work moves INTO 40-130 us (tensor 42-51%, DRAM 2.1-2.9 TB/s) and the end becomes an
  all-light DRAM-saturated window (4.2-4.5 TB/s for 60 us): the split is inverted, not removed. Rejected.
- MOE_WPMIX=2 (heavy spread evenly over the whole phase) is the flattest timeline but the weakest gain (+0.0..+0.2%):
  heavy tiles late in a phase put 28-us items close to the tail. What pays is LPT kept (heavy front-loaded) with the
  smallest experts interleaved next to them, and the split wave on the smallest experts.

## 4. Design (what is kept)
MOE_WPMIX=1 (routing warp only, once per call; +8..+16 instructions in moe_8_8_1_3_4 / _3_8; every stage loop and
producer loop byte-identical, no spill change, no C7511/C7512; moe_8_8_1_3_16 (bs257-512) and every other kernel
unchanged): after the count-descending rank, with nz = experts with tokens and m = nz - K,
  rank r < m  ->  position 2r if r < ceil(m/2), else 2(m-1-r)+1     (largest, smallest-of-m, 2nd largest, ...)
  rank r >= m ->  unchanged                                         (the K smallest experts stay last, descending)
so every wave of the gate/up phase and of the down phase pairs a compute-bound wgmma tile with a DRAM-bound 1-2 group
tile, the heavy tiles stay front-loaded (LPT), and the queue still ends on the smallest experts: the split wave
(tsplit = the last nspl = ceil(0.6 * 132 / 16) = 5 tiles, KSX 4) covers exactly the same tiles as before for K >= 5.
Queue structure, dependencies (every down item after all of its tile's gate/up items), the static first wave, the
late claim and the split rules are untouched; the tile table, pair offsets, gather and combine follow the permuted
ranks as they already did for any rank order.

## 5. Confirmation run 3 (kda_ws/runs/wpmix_run3.log; formal protocol, both lines)
Candidates: line B dev/moe_v3720.py (v3710 + MOE_WPMIX=1 = mix5; kernels moe_8_8_1_3_4 595ff80197a9, moe_8_8_1_3_8
ee25b69d8408, the 14 others = v1512's), dev/moe_v3721.py (MOE_WPMIX_K=(MW<8?8:5): bs65-128 kernel = K 8 eda1a10a4a8e);
line A dev/moe_v3770.py (v3760 + MOE_WPMIX=1: 786f39b7a053 / d8adaacb5a69, the rest = v1504's), dev/moe_v3771.py (K 8/5).
Correctness: hang1 17-512 (14 sizes) ok for all four; check_mt 22 sizes (1 2 4 8 9 16 17 24 32 33 48 64 65 100 128 129
160 200 256 257 300 512, normal + skewed, 2 calls each) 0 failures for all four; wpmix_calls 160 interleaved calls 0
failures (worst 3.29e-3); wps2_mix 120 calls 0 failures (worst 3.30e-3).
bench2 (jm seeds, 100 reps x 2 passes, SGLang interleaved, fillers fA fA2 .. fB2 fB, middle a1 c1 e1 a2 c2 e2 ... a4 c4
e4: 4 controls + 4 copies of each candidate; controls = dev/moe_v1512.py / dev/moe_v1504.py):

| line / candidate | bs32 (placebo) p1 / p2 | bs128 p1 / p2 | bs256 p1 / p2 |
|---|---|---|---|
| B controls band | -0.10..+0.15 / -0.10..+0.12 | -0.10..+0.16 / -0.14..+0.17 | -0.05..+0.07 / -0.07..+0.10 |
| B v3720 mix5 | +0.07 / -0.06 | +0.48 (+0.48 +0.53 +0.43 +0.47) / +0.49 (+0.50 +0.57 +0.33 +0.60) | +0.44 (+0.35 +0.50 +0.42 +0.55) / +0.39 (+0.42 +0.47 +0.39 +0.36) |
| B v3721 K8/5 | +0.02 / -0.10 | +0.53 / +0.43 | +0.51 / +0.41 |
| A controls band | +0.02..+0.04 / +0.01..+0.05 | -0.08..+0.04 / -0.07..+0.15 | -0.06..+0.07 / -0.05..+0.10 |
| A v3770 mix5 | +0.16 / +0.12 | +0.58 (+0.57 +0.59 +0.63 +0.47) / +0.59 (+0.54 +0.52 +0.51 +0.61) | +0.41 (+0.30 +0.31 +0.34 +0.45) / +0.37 (+0.39 +0.46 +0.30 +0.48) |
| A v3771 K8/5 | +0.04 / -0.02 | +0.56 / +0.61 | +0.38 / +0.41 |

Every mix5 copy is above every middle control at bs128 and bs256 in both passes on both lines (32/32). K 8 at bs128
does not replicate (2b: +0.15 over K 5; here -0.05..+0.05) -> the simpler mix5 (one K) is the pick. (Line A's bs32
copies sit +0.12..+0.16 over a very tight control band with the bs32 kernel unchanged: a module/placement offset of
that size is the noise floor for "placebo" shapes in this bench.)

nsys on judge-matched routings (first 8 jm seeds of the shape, 24 reps each, 192 kernels; separate captures per build;
dev/plan_nsys_an2.py; files dev/remote_out/nsysjm_wpmix_{v1512,v3720,v3721}_bs128.sqlite, _{v1512,v3720}_bs256.sqlite):

| bs256, 10-us bins | DRAM TB/s | tensor % | GPC GHz |
|---|---|---|---|
| v1512 | 0.89 2.25 2.82 3.28 4.08 3.96 3.96 4.01 4.14 4.33 4.45 4.40 4.12 3.82 3.41 3.58 3.99 4.06 3.42 3.21 | 14 44 58 48 40 38 38 33 31 27 26 27 31 38 40 38 35 27 20 17 | 1.81 1.72 1.76 1.79 1.77 1.66 1.51 1.43 1.42 1.48 1.54 1.58 1.60 1.62 1.64 1.66 1.67 1.69 1.71 1.73 |
| v3720 mix5 | 0.91 2.39 3.13 3.51 4.19 4.09 3.96 3.83 3.86 4.00 4.14 4.33 4.23 3.98 3.65 3.62 4.02 3.90 3.53 3.04 | 12 39 53 44 34 34 35 35 36 36 34 32 31 36 39 38 35 30 27 21 | 1.80 1.71 1.73 1.77 1.75 1.65 1.51 1.42 1.42 1.47 1.52 1.57 1.60 1.61 1.62 1.64 1.65 1.67 1.68 1.71 |

| bs128, 10-us bins | DRAM TB/s | tensor % | GPC GHz |
|---|---|---|---|
| v1512 | 1.79 3.44 3.84 4.23 4.56 4.60 4.51 4.26 4.27 4.22 4.25 4.06 3.95 4.31 4.14 3.36 3.11 | 11 37 45 38 27 24 25 26 27 28 30 33 30 28 23 18 17 | 1.87 1.73 1.74 1.78 1.74 1.61 1.47 1.41 1.43 1.48 1.53 1.57 1.61 1.62 1.64 1.67 1.68 |
| v3720 mix5 | 1.82 3.54 4.03 4.32 4.55 4.54 4.45 4.21 4.27 4.25 4.24 4.15 4.03 4.32 3.93 3.49 3.15 | 8 30 42 37 30 29 28 28 27 28 31 31 31 29 24 21 18 | 1.87 1.74 1.75 1.78 1.75 1.62 1.48 1.41 1.42 1.47 1.52 1.57 1.60 1.62 1.64 1.66 1.67 |

Integrals: bs256 v1512 701 MB, 3.58 TB/s, issue 55.7%, tensor 33.2%, 1.638 GHz; mix5 708 MB, 3.60 TB/s, 55.8%, 33.3%,
1.626 GHz. bs128 v1512 646 MB 3.93 TB/s; mix5 3.96 TB/s. On the jm routings the "phases" are much weaker than on the
sprint seeds (bs256 tensor 26-58% vs 25-71%); mix5 flattens what there is (bs256 tensor trough 26-27% -> 31-36% at
60-120 us, early DRAM +0.1..+0.3 TB/s at 10-40 us, DRAM peak 4.45 -> 4.33), and the clock dip (1.42 GHz at 70-90 us)
is identical. Paired per-(seed, rep) kernel durations of these captures: bs128 mix5 +0.49% (8/8 seeds positive), but
bs256 -0.33% (8/8 seeds negative, -0.18..-0.81) -- see run 4 (D) for the protocol difference (the nsys driver idles
>= 80 us between the flush and each replay; bench2 and the judge replay right after the flush).

## 6. Deliverable (files + defines, both lines)
- Patch: dev/wpmix_mk3710.py BASE OUT (applies to dev/moe_v1512.py and dev/moe_v1504.py; default build = the base's
  SASS in all 16 kernels). Kept switch: MOE_WPMIX=1 (K 5 default, NT >= 8, MW <= 8). Diagnostic / rejected switches in
  the same patch, all off: MOE_WPMIX=2 (spread), =3 (ratio R), MOE_WPMIX_PIPE (pipelined down items), MOE_WPMIX_TR.
- Line B: dev/moe_v3720.py = dev/moe_v3710.py (= wpmix_mk3710 on v1512) + "MOE_WPMIX=1" baked.
- Line A: dev/moe_v3770.py = dev/moe_v3760.py (= wpmix_mk3710 on v1504) + "MOE_WPMIX=1" baked.
- Equivalent: MOE_DEFINES=MOE_WPMIX=1 on dev/moe_v3710.py / dev/moe_v3760.py (or v3700 / v3750 from wpmix_mk3700).
- SASS (NVRTC 13, MOE_S5R_FIT): only moe_8_8_1_3_4 / moe_8_8_1_3_8 change (B: 595ff80197a9 / ee25b69d8408; A:
  786f39b7a053 / d8adaacb5a69); 8/8 stage loops and 2/2 producer loops byte-identical to the base in both, LDL/STL
  unchanged (10/3, 11/3), no C7511/C7512; moe_8_8_1_3_16 (bs257-512) identical to the base.

## 7. Second confirmation run 4 (kda_ws/runs/wpmix_run4.log): v3720 (B) / v3770 (A), + nearby shapes bs100 / bs200
Same protocol (fillers at both ends, 4 controls + 4 copies interleaved, 100 reps x 2 passes, SGLang interleaved; bs100 /
bs200 use bench2's own fresh seeds):

| line | bs32 (placebo) | bs100 | bs128 | bs200 | bs256 |
|---|---|---|---|---|---|
| B controls band p1 / p2 | -0.03..+0.13 / -0.07..+0.17 | -0.05..+0.19 / -0.07..+0.12 | -0.08..+0.12 / -0.08..+0.18 | -0.05..+0.11 / -0.06..+0.02 | -0.13..+0.10 / -0.20..+0.16 |
| B v3720 p1 | -0.01 | +0.26 (4/4) | +0.60 (+0.55 +0.45 +0.73 +0.69) | +0.40 (4/4) | +0.61 (+0.55 +0.64 +0.66 +0.58) |
| B v3720 p2 | -0.01 | +0.26 (3/4) | +0.52 (+0.53 +0.41 +0.59 +0.57) | +0.37 (4/4) | +0.59 (+0.63 +0.60 +0.67 +0.69) |
| A controls band p1 / p2 | -0.04..+0.08 / -0.09..+0.14 | -0.08..+0.08 / -0.22..+0.14 | +0.01..+0.09 / -0.02..+0.06 | -0.09..+0.06 / -0.05..+0.11 | -0.04..+0.04 / -0.07..+0.05 |
| A v3770 p1 | +0.09 | +0.48 (4/4) | +0.71 (+0.58 +0.79 +0.59 +0.73) | +0.50 (4/4) | +0.63 (+0.63 +0.65 +0.69 +0.63) |
| A v3770 p2 | +0.01 | +0.40 (4/4) | +0.73 (+0.59 +0.82 +0.68 +0.73) | +0.50 (4/4) | +0.64 (+0.59 +0.63 +0.69 +0.57) |

(D) power-state check, line B, bs128 / bs256, an idle gap (sync + host busy-wait) between the flush and every replay:
gap 150 us: +0.59 / +0.55%; gap 2000 us: +0.66 / +0.49% (every copy above every control). The gain does not depend on
the power state the flush leaves; the nsys captures' bs256 -0.33% is an artifact of comparing two separate,
non-interleaved capture processes (drift), not of the protocol.

GPU checks: every step's compute-apps list empty; idle samples 0% and 76-140 W decaying, except the first sample right
after a previous python process (bench2 / check_mt) exited (11-100% for one 250-ms sample, then 0%: the previous step's
tail in nvidia-smi's averaging window, not another tenant); memory.used 128 MiB at 0% during run 3's nsys steps (as the
planner saw before), 4 MiB elsewhere.

## 8. Summary
- MOE_RANK_MIX bug: wp32_fst (bs17-32 fast start) re-derives the first item's expert in count-desc order; any permuted
  order breaks it (NaN at bs17/24/32; MOE_NO_WP32 fixes it). Not a tsplit / KSX / KRS / CLD / buffer invariant.
- Kept: MOE_WPMIX=1 (mix5) on bs65-256: bs128 +0.47..+0.73%, bs256 +0.37..+0.75% over 4 jobs (runs 1, 2b, 3, 4; runs 3
  and 4 = the formal 4 + 4 protocol on both lines, every copy above every middle control at bs128 / bs256 in every
  pass; run 4 (D) also with 150 / 2000 us idle gaps), nearby bs100
  +0.26..+0.48%, bs200 +0.37..+0.50%, bs32 / bs257-512 unchanged kernels. Mean of the formal runs: line B bs128 +0.52,
  bs256 +0.51 (+0.17% geomean); line A bs128 +0.65, bs256 +0.51 (+0.19% geomean). On top of line B's +0.68% the local
  stack is ~+0.85% over the scored v1504 (still below the user's 1.2% bar by itself).
- What the order does and does not do: it moves light items next to the heavy ones (tensor trough +5-10 points, DRAM
  peak -0.1..-0.3 TB/s at bs256) but the kernel stays consumer-compute-paced in every phase, so the phase split itself
  was never worth the 11% coupling bound; the clock dip is a power-limiter transient that no order changes. Rejected:
  spread (MOE_WPMIX=2, +0.0..+0.2), mix0 / MOE_RANK_MIX order (split wave on median tiles, +0.2..+0.45), R2 / K3 / K8
  (same as mix5 within noise), PIPE (heavy G2 alongside light G1: -0.7..-0.9%, readiness waits + inverted phases).
