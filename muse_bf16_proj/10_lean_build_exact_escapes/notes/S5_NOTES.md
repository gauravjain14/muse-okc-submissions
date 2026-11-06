# S5 — small latency-bound shapes (router m=1/8/16/32, unpooler slice m=1/8)

## RESULT (read first) — deliverable claude/t3/S5/sm5.py (34 KB, readable, no lab knobs)
`prepare_sm5(weights)` / `run_sm5(x, config, state)` (None = base). Candidates: sub_sm5.py = LN/bo14_lean + sm5 (sm5
first), sub_sm5_only.py = sm5 + torch. Integration: `out = run_sm5(x, cfg, st)` before the base router / unpooler
routes (it returns None for router m > 16, unpooler m > 8 and anything malformed).
| shape | sm5 kernel | lean base kernel |
|---|---|---|
| router m=1..8 | sm5_rz1: 24-bit lossless hi/lo pairs (6.3 MB), mma.sync, split-K x16, red.add, exact escapes | R rs1_i (m=1), Y ys (m=8) |
| router m=9..16 | sm5_rz2 (same, 2 n8 token tiles) | Y ya |
| unpooler slice m=1 | sm5_cgemv: alygg77 cgemv 12-bit, 1 row x 128 thr, min-blocks 12, EXACT escapes | G g311 (inexact) |
| unpooler slice m=2..8 | unps_m: G's tcm with EXACT two-term escapes, compile-time M | G tcm (inexact) |
Same-run tables, bench.py (judge-style, paired with SGLang), local us, 3 runs on box 0 + 1 earlier run on box 1:
| shape | SGLang | lean (bo14_lean) | sm5 | delta | x vs SGLang lean -> sm5 (run d) |
|---|---|---|---|---|---|
| router m=1 | 9.33-9.41 | 2.79 / 2.80 / 2.84 (2.78) | 2.65 / 2.74 / 2.72 (2.70) | -0.10 | 3.312 -> 3.460 |
| unpooler m=1 | 5.97-5.98 | 2.49 / 2.54 / 2.54 (2.47) | 2.39 / 2.43 / 2.39 (2.44) | -0.10 | 2.353 -> 2.506 |
| router m=8 | 25.34-25.58 | 3.09 / 3.09 / 3.10 (3.02) | 2.79 / 2.80 / 2.81 (2.72) | **-0.30** | 8.223 -> 9.067 |
| unpooler m=8 | 6.16-6.26 | 2.85 / 2.96 / 2.90 (2.88) | 2.85 / 2.82 / 2.91 (2.85) | -0.04 (tie; exactness fix) | 2.138 -> 2.130 |
| router m=16 | 14.00-14.26 | 3.23 / 3.30 / 3.33 (3.30) | 2.95 / 3.06 / 3.07 (3.07) | **-0.25** | 4.279 -> 4.648 |
| router m=32 | — | not taken (Y's ya wins: 3.6-3.8 vs my best 4.1-4.2) | | | |
Logs: bench_final_box0a.log, bench_final_c.log, bench_final_d.log (box 0), bench_final_box1a.log (box 1).
Projected: router m1 -0.10 (x0.53 %/us) + unp m1 -0.10 (x0.53) + router m8 -0.30 (x0.46) + router m16 -0.25 (x0.36)
~= +0.33 % of the final score, plus exact one-hot rows on the unpooler slice (G's kernels flushed 50-75 of 200
injected columns).
Gates (gates_final.log, box 0, sub_sm5.py = lean base + sm5): check.py --layers router,unpooler --graph (NaN-poisoned):
"checked 105 cases x 2 seeds + graph: ALL OK"; Y/exact_router.py: EXACT ROUTER ALL OK; exact_all.py --layers unpooler
--nesc 200: EXACT ALL OK. Local: exact_all --nesc 3000 OK; t_unp.py --exact (200 columns, x = 1.0, strict, no tiny
exemption): 0 mismatches at m = 1/2/5/8 (G's kernels: 50-75).
prepare_sm5: ~1 s + one NVRTC compile (bench prepare of the candidate 3.4-3.8 s incl. the lean base).
Integration check: build_t3.prefix_names(sm5.py, '_s5_') runs and passes check_implementation on router m=1/8/16 and
unpooler m=1/8 locally (returns None for router m=32). Research code (not shipped): rz.py / t_rz.py (router lab,
trace + ablation bits), unps.py / t_unp.py / t_cg.py (unpooler labs, incl. the exact g311 variant), sass_rz.py /
sass_unp.py (NVRTC SASS stats), sub_rzlab.py (ncu wrapper), sm5_v4.py / sm5_lab_g311.py (earlier module states).

## Base kernels (bo12) for my shapes
| shape | kernel | nsys in-graph (dash_bo11) | local (same-run controls in my runs) |
|---|---|---|---|
| router m=1 | alygg77 rgemv (FP32 W 8.4 MB, 256 CTAs x 256 thr) | 3.54 | 2.62-2.88 |
| router m=8 | Y's ys (cluster 8, 64 CTAs, raw hi/lo 8.4 MB) | 3.87 | 3.04-3.33 |
| router m=16 | Y's ya (4 zeroing CTAs + 128 compute CTAs, red.add) | 4.00 | 2.98-3.36 |
| router m=32 | Y's ya | 4.45 | 3.46-3.84 |
| unpooler slice m=1 | alygg77 cgemv (12-bit, 512 CTAs x 160) | 3.39 | (not yet measured here) |
| unpooler slice m=8 | G's g1m_unpslice (tcm, 64 CTAs x 1024) | 3.74 | (not yet measured here) |

## Tools (this dir)
- `rz.py` router lab: lossless 24-bit hi/lo pair format (packer `codes`/`pack_units`/`pack_cfg`), kernel `rz_body`
  template (RT 16-row tiles x KC = 8192/S k per CTA, NW compute warps + 1 helper warp, MT n8 token tiles, XF issue
  order bits, SEP hi/lo accumulators, LDF load flavour, TRC trace/ablation bits), NVRTC module + launcher.
- `t_rz.py` same-run timing (NG graphs per candidate, interleaved, paired deltas), correctness (dense rel-L2 vs FP64,
  `--exact`: 200 injected special values + one-hot rows must equal fl32(hi+lo)*p2 bit-exactly like Y/exact_router.py),
  `--trace` per-warp / per-CTA timelines (globaltimer start + clock64 deltas), controls `--ctl` (Y's rt3 = bo12's
  ys/ya, N's NVRTC port of alygg77's rgemv).
- `sass_rz.py` NVRTC sm_90a SASS stats (regs, stack, LDG count, LDGs after the first HMMA, op histogram).
- `rz_v1.py` first version (kept for reference).

## 24-bit format (rz.py docstring)
Per weight 3 bytes: hi low byte, lo low byte, hi code (s | c = e7_h - 54), lo code (s | f = e7_l - e7_h + 11).
Escapes (hi e7 outside [54, 61], lo == +-0, e7 gap outside [4, 11]): 342-356 per weight set (0.017 %), stored as
all-zero fields = decoded (2^-19, 2^-41) exactly. Decode per 8 weights (one k16 step, hi+lo A fragments): 2 SHF.W
(rotate) + 6 LOP3 + 4 IADD3 + 8 PRMT = 20 ALU (Y's layout: 24); fake-exponent variant (XF & 8): 18.
Exactness of escapes: the warp owning the escape cancels the decoded contribution exactly on its accumulator
(c - x*2^-19 = 0 for a one-hot row) and adds x*hi, x*lo (SEP: separate hi / lo accumulators) -> fl32(hi+lo)*x.
Local (sm_89) and H200: --exact with 551 escapes: 0 non-tiny mismatches at m = 1..32.

## iteration 1 — rz v1 (CTA-level escape fix-up, merged / dedicated helper warp)
- First timing (H200, same run): m=1 4.13 vs rgemv 2.72; m=8 4.10 vs ys 3.10; m=16 4.51 vs ya 2.98; m=32 4.74 vs 3.46.
- Causes found (ablation bits, same-run): (a) helper duties merged into warp 0 put the zero-counter ticket atomic +
  acquire poll in the tail (+~1 us) -> dedicated helper warp (ticket at kernel start): m=1 3.42 / m=8 3.55.
  (b) escape fix-up via param-block list + x gathers (LDC chain + cp.async behind the W flood): +0.5..0.6 us
  (`a2` ablation). (c) CTA reduction with a 512-B stride smem layout: 16-way bank conflicts, +0.45 us
  (a110 2.56 vs no-reduction a366 2.11 vs exit-after-compute a622 1.98 vs pure LDG probe 1.90-1.94).
  (d) kernel parameter block size (3.4 KB vs 112 B): no effect (+-0.07).
## iteration 2 — rz v2 (column-sum reduction, per-warp escape trailers in the W stream)
- Reduction: thread o sums output o over the NW warp partials (conflict-free LDS.32), 4-way ILP.
- Escapes: 3 uint4 trailer (6 entries) per warp chunk (lands with the warp's W); correction in the owning warp.
  Pitfalls: runtime-indexed accumulator -> whole c[] in local memory (+1..2.5 us); fully unrolled 6-entry predicated
  correction -> ~1000 instructions in escape warps at m=32. Fixed: rolled loop over valid entries, constant-index
  branches (stack 0).
- Same run (box 0): c = (RT 2, S 16, NW 16): m=1 3.04 / m=8 3.01 / m=16 3.71 / m=32 4.48 vs rgemv 2.88 / ys 3.33 /
  ya 3.36 / ya 3.84. Escape cost still 0.19-0.54 (c2 ablation: 2.72 / 2.82 / 3.17 / 4.03): the correction sits at
  the end of the owning warp (shuffle chains, ~150-200 cycles latency per escape) and every CTA has 2-4 escape warps.
- Trace (c2, m=8): last W landing 2.39 us in-kernel, last MMA done 2.87 (+0.48), T ready +0.17, end +0.2.
  The decode (20 ALU / 8 weights) + 8 HMMA of all 16 warps of an SM run AFTER the data lands: ~0.37 us of ALU per
  SMSP + 32 HMMA x 6.7 clk per SMSP => the per-SM compute is not overlapped with the stream.

## 2026-11-02 ~15:50 — coordinator update: base = bo13r; G's unpooler-slice kernels are inexact -> fixed here
### DELIVERABLE v1 (unpooler slice m = 1..8, exact): claude/t3/S5/sm5.py (22 KB), sub_sm5.py (bo13r + sm5),
### sub_sm5_only.py (sm5 + torch)
- m = 1: `unps_g311` = agent G's g311 (F11, 1024 CTAs x 128 thr, 40 regs, min-blocks 12) with EXACT escapes: G's single
  term x * (true*2^(120-E_lo) - 2^-7) is kept only for escapes where that delta is exactly representable in FP32
  (decided at prepare with fractions); "far" escapes (slot bit 28) go through a two-term path: fake sum += -x*2^-7
  (cancels the decoded 2^-7 exactly), true sum += x*true, out = bf16(fma(fake_sum, 2^(E_lo-120), true_sum)). The far
  path is compiled in only if the weights have far escapes (natural weights: 0-1 per seed; injected tests: ~75).
  Anti-sink (C2): FMA chain starts from +-0 with the sign of x quarters ^ slot -> all 10 W/x/slot loads before the first
  FFMA (SASS: 40 regs, 0 stack, late 1 = the escape x gather).
- m = 2..8: G's tcm (F12 16-row tiles, 64 CTAs x 1024 thr, all x B fragments up front) with exact two-term escapes
  (escc holds -x*2^-15 in the fake domain and x*true; out = bf16(fma(sum, 2^(E_lo-112), sum_true))), compile-time M.
- Gates (H200, gates_unp1.log): check.py sub_sm5.py --layers unpooler,router --graph: "checked 105 cases x 2 seeds +
  graph: ALL OK"; exact_all.py sub_sm5.py --layers unpooler --nesc 200: EXACT ALL OK; strict 200-column one-hot test
  (t_unp.py --exact, x = 1.0, no tiny exemption): sm5 0 mismatches at m=1/2/5/8, G's gemv1 (bo13r) 50-75 mismatches.
  Local: exact_all --nesc 3000 EXACT ALL OK.
- bench.py same run (bench_sm5_v1a.log, box 1, 60 reps, local us): unpooler m=1 bo13r 2.33 / sm5 2.40 (x2.457 ->
  x2.386 vs SGLang), m=8 2.75 / 2.76 (x2.143 -> x2.131). Router lines (identical code in both) 2.62 / 2.73 and
  2.96 / 2.94 = the run's noise. t_unp.py with XCL x placements: m=1 exact 2.40-2.46 vs G 2.24-2.27, m=8 tie.
  => exactness costs ~0.07-0.15 us at m=1 and nothing at m=8.
- Ablations at m=1 (same run): my build of G's algorithm without anti-sink 2.45, with anti-sink 2.35, exact + anti-sink
  2.40-2.46, exact without anti-sink 2.80 (x load sunk below the first FFMA), runtime E/RB/scale params +0.05..0.1.

## iteration 3 — router: issue order / x via TMA / stagger / raw (same runs, box 0-1, local us)
- x bulk-copied to smem by the helper warp (XF 16): +0.2..+0.6 (one thread issues up to 32 copies; warps wait for all
  of x). Staggered warp issue: named-barrier chain serialised on DATA (bar.sync has acquire semantics, so ptxas waits
  for the previous warp's loads: warp k lands at 1.3 + 0.5k us); clock-delay stagger: +-0.0. Raw hi/lo (8.4 MB, no
  decode) in the same skeleton: m=1/8/16/32 3.01/3.17/3.23/4.06 vs 24-bit-no-escape 2.59/2.72/3.04/3.81 -> the 24-bit
  format is worth 0.2-0.4 us when its escapes are free.
- Trace (2,16,16, m=8): per-CTA landing spread 0.53-0.75 us; CTA last MMA - last landing 0.41 us = the decode ALU
  (20 ops / 8 weights; 0.37 us of ALU per SMSP at 48 KB/SM) runs after the data lands. Median landing by warp index is
  flat (all of a CTA's warps land together): LDG data is not returned in issue order, so overlap cannot be forced.
## iteration 4 — router escapes (the remaining 0.5-1.4 us)
- ncu (m=8, escapes on vs runtime-disabled, same binary): 5.34 vs 4.74 us; stall samples at the bar.sync after the
  partials (escape warps late). Causes found one by one: x gathers from global in the tail (each L2 round trip
  ~0.3-0.5 us, serialised per escape) -> x copied to smem by every warp (1 STS.128 per lane); then the per-CTA list scan
  by every output thread (ne/2 dependent LDS.128 iterations) -> per-row cumulative index in the CTA trailer (32 u8 row
  ends, entries sorted by row): a thread only touches its own row's escapes (usually none).
- Result (run_iter5a.log): escapes now cost 0.02-0.1 us (c vs runtime-disabled zc).
## iteration 5 — DELIVERABLE v2 = sm5.py (36 KB): router m=1 (alygg77 rgemv), router m=2..8 (24-bit rz), unpooler
## slice m=1 (exact cgemv), m=2..8 (exact tcm). Base bo14.
- Gates (H200, gates_v2.log): exact_router sub_sm5_only m=1,2,5,8 EXACT ROUTER ALL OK; check.py sub_sm5.py --layers
  router,unpooler --graph "checked 105 cases x 2 seeds + graph: ALL OK"; exact_router sub_sm5.py EXACT ROUTER ALL OK;
  exact_all sub_sm5.py --layers unpooler --nesc 200 EXACT ALL OK (local: --nesc 3000 OK).
- bench.py same run (bench_sm5_v2b.log, 100 reps, local us; x vs SGLang):
  | shape | SGLang | bo14 | sm5 | bo14 x -> sm5 x | kernel |
  |---|---|---|---|---|---|
  | router m=1 | 9.30 | 2.93 | **2.78** | 3.171 -> 3.344 | rgemv (FP32 W, alygg77 via N's NVRTC port) |
  | unpooler m=1 | 6.10 | 2.58 | **2.52** | 2.364 -> 2.421 | cgemv 12-bit, exact two-term escapes |
  | router m=8 | 25.36 | 3.21 | **2.86** | 7.896 -> 8.867 | sm5_rz (2,16,16) 24-bit hi/lo |
  | unpooler m=8 | 6.19 | 3.02 | 2.91 | 2.053 -> 2.130 | tcm exact (G's kernel + two-term escapes) |
  Earlier runs: router m=1 2.62-2.66 vs 2.84-2.86; unpooler m=1 2.46/2.48 vs 2.51-2.52; unpooler m=8 2.95 vs 2.89 (tie).
- nsys in-graph (prof_m1.log / this run): router m=1 rgemv 3.55 vs rs1_i 3.70; unpooler m=1 cgemv 3.38 vs g311 3.44;
  router m=8 sm5_rz 3.65 vs ys 3.87; unpooler m=8 tcm-exact 3.84 vs 3.71 (in-graph +0.1, bench -0.11..+0.06).
- ncu (flushed, single launch):
  | kernel | dur | DRAM MB | DRAM % (in-graph) | occ % | issue % | hmma % | elig | top stalls |
  |---|---|---|---|---|---|---|---|---|
  | sm5_rgemv (router 1) | 4.77 | 8.41 | 49 | 23 | 8.3 | 0 | 0.10 | long_sb 65, barrier 12, imc_miss 8 |
  | rs1 (bo14 router 1) | 4.70 | 8.41 | 47 | 12 | 9.5 | 0.1 | 0.11 | long_sb 55, no_instruction 21 |
  | sm5_cgemv (unp 1) | 4.48 | 6.33 | 39 | 28 | 30 | 0 | 0.47 | long_sb 54, barrier 11, imc_miss 6 |
  | sm5_rz (router 8) | 4.74 | 6.47 | 37 | 25 | 18 | 1.2 | 0.29 | long_sb 58, wait 9, barrier 7 |
  | ys (bo14 router 8) | 5.06 | 8.53 | 46 | 12 | 6.7 | 2.1 | 0.09 | long_sb 63, no_instruction 10, lg_throttle 9 |
  | unps_8 tcm exact | 4.83 | 6.38 | 36 | 47 | 29 | 2.2 | 0.82 | long_sb 26, lg_throttle 25, wait 9 |
  | g1m_unpslice (bo14) | 4.83 | 6.38 | 36 | 47 | 33 | 2.3 | 0.89 | long_sb 38, lg_throttle 13, wait 12 |
- SASS (NVRTC 12.1 sm_90a, local; 0 stack everywhere): sm5_rz 48 regs, 10 LDG (6 W + 1 x + trailer...), late 2 (counter
  polls), 8 HMMA + 4 x 18 decode ALU per warp; sm5_rgemv 36 regs, 6 of 12 LDGs sunk below the first FFMA (anti-sink
  and min-blocks 4 variants issue all 12 up front but measured equal/slower: 2.67 / 2.72 vs 2.65); sm5_cgemv 40 regs.
- Binding unit: all are latency-bound streams (DRAM 36-49 % of 4.8 TB/s averaged over the in-graph time; tensor
  <= 2 %, issue <= 30 %). sm5_rz moves 6.47 MB (-24 %) and its tail is now the decode ALU of the last-landing warps
  (~0.4 us) + reduction; rgemv/cgemv sit ~0.2-0.3 us above the hw-scheduled LDG floor for their bytes (8.4 / 6.3 MB).

## iteration 6 — router config sweep + min-blocks pitfall, router m=1 via rz (box 0)
- Sweep (run_iter6a/6b.log, local us, same run): m=16: (1,16,8) shared accumulator 3.04, separate 3.14, (2,16,16)
  3.30, (2,32,8) 3.23, rt3 3.14; m=8: (1,16,8) 2.75 / (2,16,16) 2.85 / rt3 3.10. m=32: every config 4.13-5.60 vs
  ya 3.71 (not taken). (1,16,8) = 256 compute CTAs (2 per SM) x 8 warps of 64 k.
- PITFALL: the deliverable's first build dropped the ", 1" min-blocks from __launch_bounds__: ptxas then targets
  fewer registers (36-52) and sinks 4-7 W/x loads below the first HMMA (SASS late 4-7 vs 2) -> router m=8 3.37-3.63
  instead of 2.86. Always keep min-blocks and check SASS of the shipped build.
- Router m=1 through the same rz kernel (n8 tile, 1 valid token) beats alygg77's rgemv in 3 runs (2.57/2.61/2.75 vs
  2.64/2.69/2.86) -> rgemv dropped (it also sinks 6 of 12 loads; anti-sink / min-blocks versions measured equal).
- Unpooler m=1 cgemv sweep (t_cg.py, run_cg1.log, XCL placements, exact): 1 row x 128 thr + min-blocks 12 2.18,
  2 rows x 128 2.21, 2 x 256 2.27, 1 x 128 no-mb 2.30, 2 x 128 mb6 2.37, 4 x 128 2.46, 2 x 64 3.10; G's g311 2.30.
## iteration 7 — final kernels, mandatory analysis (prof_v5.log, box 1; SASS NVRTC 12.1 sm_90a, 0 stack everywhere)
1. nsys in-graph (same process, lean-equivalent bo14 kernels): router m=1 sm5_rz1 3.54 vs rs1_i 3.68; unpooler m=1
   sm5_cgemv 3.28 vs g1_unpslice_g311 3.39; router m=8 sm5_rz1 3.60 vs ys 3.90; unpooler m=8 unps_8 3.81 vs g1m 3.70;
   router m=16 sm5_rz2 3.94 vs ya 4.00 (nsys in-graph +0.8-1.0 us vs bench local, as usual).
2. ncu (flushed, single launch):
   | kernel | dur us | DRAM MB | DRAM % in-graph | occ % | issue % | hmma % | elig/clk | top stalls |
   |---|---|---|---|---|---|---|---|---|
   | sm5_rz1 router m=1 | 4.64 | 6.38 | 38 | 26 | 18 | 1.3 | 0.27 | long_sb 48, barrier 14, wait 11 |
   | sm5_rz1 router m=8 | 4.74 | 6.47 | 37 | 25 | 18 | 1.2 | 0.29 | long_sb 58, wait 9, barrier 7 |
   | sm5_rz2 router m=16 | 5.12 | 6.63 | 35 | 26 | 18 | 2.2 | 0.28 | long_sb 51, wait 11, barrier 10 |
   | ya router m=16 (base) | 5.12 | 8.66 | 45 | 13 | 6.9 | 2.2 | 0.09 | long_sb 61, no_instruction 9, lg_throttle 7 |
   | ys router m=8 (base) | 5.06 | 8.53 | 46 | 12 | 6.7 | 2.1 | 0.09 | long_sb 63, no_instruction 10 |
   | sm5_cgemv unp m=1 | 4.22 | 6.34 | 40 | 54 | 39 | 0 | 0.76 | long_sb 52, short_sb 13, barrier 8 |
   | unps_8 unp m=8 | 4.83 | 6.38 | 35 | 47 | 29 | 2.2 | 0.82 | long_sb 26, lg_throttle 25, wait 9 |
3. SASS: sm5_rz1 55 regs / 11 LDG (6 W + 2 x + trailer...) / 8 HMMA; sm5_rz2 67 regs / 13 LDG / 16 HMMA; late 2 =
   the zero-counter polls only; decode 18 ALU (2 SHF + 6 LOP3 + 2 IADD3 + 8 PRMT) per 8 weights + 1 LOP3/4 for hi.
   sm5_cgemv 32 regs (the "late" count includes the escape warp's branch); unps_8 54 regs, late 8 = the escape x reads.
4. Binding unit: latency-bound DRAM streams (35-40 % of 4.8 TB/s averaged over the in-graph time; tensor <= 2 %,
   issue <= 39 %). Router: the 6.3 MB stream lands by ~2.5 us in-kernel; then ~0.8 us of tail = decode ALU of the
   last-landing warps (~0.38 us; all warps of a CTA land together, LDG data is not returned in issue order) + CTA
   reduction / zero barrier (0.24) + red.add issue (0.17). Next lever there: fewer ALU ops per weight or a tail that
   skips the T staging (scalar red.add). Unpooler m=8: per-SM ingress (98 KB W + 64 KB x per SM, lg_throttle 25 %) on
   64 SMs; restructurings tried (NW 16, 2 rows per CTA, min-blocks 2) are 0.5-1.4 us slower.

## Dead ends (measured)
- Router: per-warp escape trailers + correction on the accumulators (shuffle chains in the warp tail, runtime-indexed
  accumulator -> local memory), escape x gathers from global in the tail, CTA-wide list scans: 0.2-1.4 us each;
  x via TMA bulk copies (+0.2..0.6); warp issue staggering (barrier chain serialises on data; clock delay no effect);
  raw hi/lo in the same skeleton (+0.2..0.4 vs 24-bit); RT=4 tiles (escape row index is 5 bits -> illegal address).
- Unpooler m=1: G's g311 with exact two-term escapes: +0.1..0.2 vs G (compile-time constants and anti-sink needed to
  keep all loads ahead of the FFMAs); cgemv with exact escapes is faster than both.
- Router m=32: no 24-bit configuration beats ya (decode + 4x HMMA + 4x partials after a burst landing).
