# S7 — wave-quantization / occupancy audit of bo23L's small kernels (2026-11-03)

All times local us (judge ~= local + 1.0), H200 sandbox, judge-style (512 MB flush read before every replay, CUDA
graph, empty graph subtracted) unless noted. Base = claude/t3/bo23L.py (module per shape: LN/build_t3.py MODULES).

## RESULT (read first)
- **No wave / occupancy fix wins.** Only ONE small kernel has a partial second wave (nk_o1 = o m=1, 1.29 waves) and
  only the two n2 GEMVs are multi-wave (2.9 / 5.2 waves, steady-state streaming). Every other kernel is a single wave
  (all CTAs start within 0.10-0.38 us). Each fix that the audit suggested was built and measured; all lose:
  | kernel (shape) | waves / occupancy | fix tried | same-run result (control-corrected) |
  |---|---|---|---|
  | nk_o1 (o m=1) | 512 CTAs x 512 thr, 38 regs, 3/SM = 396 slots: 1.29 waves; 116 CTAs start 1.75-3.0 us | one wave: KC=8 (8 warps x 128 k, 56 regs, 4/SM, 528 slots, all loads early) | bench.py xclones 6, both orders: o m1 x **0.986 / 0.972** (control means 0.996 / 1.003) = +0.05..+0.10 us; ab.py +0.07/+0.06 net of A/A'; nsys in-graph 4.64 vs 4.58 |
  | " | " | one wave at 32 regs (MB 4, NACC 2 or 1) | +0.57..+0.62 (ab), nsys 5.09 vs 4.58: one W LDG sunk below the first HMMA (afterUse 2) |
  | n2_qkv_gate / n2_pooler (qkv / pooler m=1) | 2304 / 4096 CTAs x 256 thr, 40 regs, 6/SM: 2.9 / 5.2 waves, refilled continuously (DRAM-saturated: first wave = 9 MB in flight) | 8/SM (MB 8 -> 32 regs) | +1.6..+1.8 (qkv), +2.4 (pooler): 3 loads sunk (late 4 vs 1); the 28 data registers per lane cannot fit 32 |
  | " | " | fewer CTAs/SM (5/4/3 via dynamic-smem padding) | +3.3..+4.1 (qkv), +6.2..+8.4 (pooler): confounded - the smem carveout shrinks L1 and n2's x (16 KB, L1-cached, shared by the SM's CTAs) is re-read from L2 per row. Shows n2 DEPENDS on L1 caching of x |
  | unps_8 (unpooler m=8) | 64 CTAs x 1024 thr on 64 of 132 SMs (98 KB W + 64 KB x per SM, lg_throttle 28 %) | K split over a 2-CTA cluster (128 SMs, DSMEM reduction, barrier.cluster) | **+0.20..+0.33** (ab, 3 runs), nsys 4.00 vs 3.76. Ablations: same split with no reduction (wrong results) -0.05..-0.17 = the most the split can give; the cluster barriers cost +0.35..+0.5 (cluster launch alone is free) |
  | qkvb (qkv m=8/16) | 96 CTAs on 96 SMs | none (V2 iteration 1: 128-CTA R=18 byte-balanced ring = same W-only stream time; DRAM-bound) | - |
- Nothing to ship: no `<module>_s7.py`, no `sub_s7.py` (a sub identical to bo23L would be pointless); bo23L stays.
- The real (new) imbalance found is **per-SM, not per-wave**: in single-wave kernels the last CTA ends 0.2-0.75 us after
  the median one, and for m8_pooler (pooler m=8) the slow SMs are the SAME SMs run after run (end-time correlation by
  SM +0.87..+0.91, by blockIdx +0.02..+0.23; the blockIdx->SM placement changes every run). Projected time if work were
  spread in proportion to each SM's measured speed: m8_pooler 11.97 vs T_max 12.43 us (**-0.46 us**); every other
  kernel <= 0.1-0.2. Realising it needs dynamic, fine-grained work distribution across CTAs (stream-K style with a
  cross-CTA reduction); every cheap form costs more (E8: in-CTA dynamic claims +1.6..3.4; global red.add / ticket
  reduction ~1 us; cluster barriers +0.35..0.5 here). Left as a lead (table below).

## Tools (this dir; research only)
- `audit.py <build> [--shapes m:layer,...] [--trace --hook H --reps N --skew] [--savesrc d]`: patches NVRTC + driver
  (as LN/trace.py) to record every launch of each shape: kernel, grid, block, dyn + static smem, regs, local bytes,
  resident CTAs/SM (cuOccupancyMaxActiveBlocksPerMultiprocessor / MaxActiveClusters), waves, registers needed for one
  more CTA/SM, plus the flushed graph-replay time. `--trace` injects a per-CTA hook into every extern "C" kernel
  (buffer address baked as a constant, no parameter change): H bits 1 entry clock64, 2 exit hook, 4 exit clock64,
  8 no entry globaltimer, 16 per-warp exit store (instead of atomics), 32 exit globaltimer, 64 calibration-CTA mode.
  Prints start-time quantiles + 0.25 us histograms, CTA duration, slot reuse (a CTA starting after another CTA on the
  same SM ended), duration/end by CTAs-per-SM, and (`--skew`) end-time correlation across reps by blockIdx and by SM,
  per-SM finish times and the balanced-work projection.
- `ab.py` = IC/abm.py + launch attributes per candidate (same-process A/B, x at shared random placements, A/A' control);
  labels starting with `abl` skip the correctness assert (timing-only ablations).
- `varmod.py` (fresh module instance with patched globals) + `v_*.py` variant wrappers; `sub_*.py` research subs (ncu /
  nsys); `sub_bo23L_o1kc8.py` (bo23L + KC=8 o m=1, for bench.py); `unps_lab.py` (cluster K-split unpooler lab, CFG
  (NW, KS, ABL)); `sass_s7.py` (NVRTC SASS stats via N2/sass_n2.py: o1 / n2 / unp variants); `gpcmap.py` (SM -> GPC
  probe via cluster placement).

## Measurement caveat: the trace hook perturbs these kernels
Any code in a kernel can change ptxas's schedule (N2/IC load sinking) and %globaltimer reads are slow:
| hook | o m1 | pooler m1 | unpooler m8 | o m8 |
|---|---|---|---|---|
| clean | 3.42-3.63 | 11.34-11.50 | 2.62-2.69 | 3.62-3.74 |
| 0: entry %globaltimer + %smid (thread 0, one 24 B store) | +0.30..+0.45 | +0.2..+0.3 | +0.03..+0.10 | +0.04..+0.16 |
| 50: + every warp's exit %globaltimer (plain store) | +0.37 | +1.08 | +0.24 | +0.28 |
| 7: entry + exit atomics (globaltimer + clock64) | +1.7 | +2.8 | +0.8 | +0.8 |
=> start distributions below use hook 0; end/duration analyses use hook 50 (relative comparisons only).
%clock64 is NOT synchronized across SMs on this H200 (two groups of SMs 0.35-0.38 s apart); %globaltimer has 32 ns
resolution here. Calibration mode (clock64 everywhere, globaltimer only in the first 132 CTAs) costs as much as hook 50.

## iteration 0 — static audit + measured start-time distribution (audit_static.log, audit_trace1.log, audit_hook0.log)
| shape (weight %/us) | kernel (module) | grid x block | regs | smem dyn+static | CTA/SM | slots | waves | regs for +1 CTA/SM | starts (hook 0): p50 / p90 / max ns | local us |
|---|---|---|---|---|---|---|---|---|---|---|
| qkv m1 (0.53) | n2_qkv_gate (N2) | 2304 x 256 | 40 | 0+32 | 6 | 792 | 2.91 | 32 | 792 in 0-0.5 us, refill from 1.5 us, last 6.56 us | 7.09 |
| o m1 (0.53) | nk_o1 (N, small_lean) | 512 x 512 | 38 | 0+11264 | 3 | 396 | **1.29** | 32 | 396 in 0-0.5 us, 116 at 1.75-3.04 us | 3.42-3.63 |
| pooler m1 (0.53) | n2_pooler (N2) | 4096 x 256 | 40 | 0+32 | 6 | 792 | 5.17 | 32 | 792 in 0-0.5 us, refill from 1.75 us, last 10.66 us | 11.34-11.66 |
| unpooler m1 (0.53) | sm5_cgemv (S5) | 1024 x 160 | 32 | 0+32 | 12 | 1584 | 0.65 | (warp-limited) | 160 / 224 / 288 | 2.24-2.27 |
| router m1 (0.53) | sm5_rz1 (S5) | 260 x 288 | 55 | 13696 | 4 | 528 | 0.49 | 40 | 96 / 128 / 192 | 2.46-2.56 |
| qkv m8 (0.46) | qkvb_qkv_gate_8 (V2/IC) | 96 x 256 | 88 | 139776 | 1 | 132 | 0.73 (96 SMs) | 128 | 64 / 96 / 128 | 7.74-7.79 |
| o m8 (0.46) | o8_m8 (O8) | 512 x 512 | 32 | 0+10240 | 4 | 528 | 0.97 | - | 192 / 320 / 384 | 3.62-3.78 |
| pooler m8 (0.46) | m8_pooler (E8) | 256 x 544 | 56 | 0+12800 | 2 | 264 | 0.97 | 40 | 160 / 224 / 256 | 12.26-12.30 |
| unpooler m8 (0.46) | unps_8_1_32_1 (S5) | 64 x 1024 | 54 | 19456 | 1 | 132 | 0.48 (64 SMs) | 32 | 32 / 64 / 96 | 2.58-2.82 |
| router m8 (0.46) | sm5_rz1 (S5) | 260 x 288 | 55 | 13696 | 4 | 528 | 0.49 | 40 | 96 / 128 / 192 | 2.53-2.62 |
| qkv m16 (0.36) | qkvb_qkv_gate_16 | 96 x 256 | 88 | 180736 | 1 | 132 | 0.73 (96 SMs) | 128 | 64 / 96 / 128 | 7.92-7.97 |
| router m16 (0.36) | s6_rz2 (S6) | 257 x 544 | 45 | 35328 | 2 | 264 | 0.97 | 40 | 160 / 288 / 320 | 2.74-2.82 |
| router m32 (0.27) | ya_1_8_4_17_0 (Y) | 132 x 288 | 159 | 18960 | 1 | 132 | 1.00 | 112 | 64 / 128 / 160 | 3.41-3.42 |
| (also) o m16 | sk11_o_16 (F/O2) | 256 x 288 | 54 | 68176 | 3 | 396 | 0.65 | 56 | 64 / 128 / 160 | 4.14 |
| (also) pooler / unpooler m16 | pu5 (P5) | 128 / 256 x 288 | 54 / 50 | 129664 / 57968 | 1 / 3 | | 0.97 / 0.65 | | <= 192 | 12.26 / 12.45 |
| (also) m32 qkv / o / pooler / unpooler | sk11 96 / o3 128 / pu5 128 / skh 128 CTAs | | 96 / 96 / 80 / 115 | | 1 | 132 | 0.73-0.97 | | | |
Zeroing CTAs of the split-K routers are blockIdx 0..3 (rz1, ya) / 0 (s6): dispatched first, as intended.

## iteration 1 — o m=1 (nk_o1): make it one wave (ab_o1_v1.log, bench_o1kc8.log, ncu_iter1.log, nsys_iter1.log)
Variants = small_lean configs (pack_ta with the variant's KC): A (shipped: KC 16, 16 warps x 64 k, MB 2 -> 38 regs, 3/SM),
kc8 (KC 8: 8 warps x 128 k, 256 thr, MB 4 -> 56 regs, 4/SM, 528 slots), kc8n1 (NACC 1, 54 regs), mb4 / mb4n1 (KC 16, MB 4
-> 32 regs, 4/SM).
1. ab.py (box 0, 4 x-placements x 40 reps x 2 rounds): A 3.62/3.61 | A' +0.03/+0.07 | kc8 +0.10/+0.14 | kc8n1 +0.15/+0.18 |
   mb4 +0.57/+0.60 | mb4n1 +0.59/+0.62. bench.py --xclones 6 --reps 60 both orders (controls: pooler/router m1, o/pooler/
   router m8 = identical kernels): o m1 speedup ratio kc8/bo23L 0.982 and 0.975, control means 0.996 and 1.003 ->
   corrected **0.986 / 0.972** (kc8 loses 1.4-2.8 %).
2. nsys in-graph (30 reps): A 4.58 / A' 4.58 | kc8 4.64 | mb4 5.09.
3. ncu (full set, flushed, single launch): A 5.82 us, 12.71 MB DRAM, occ 61 %, issue 29 %, hmma 1.9 %, elig 0.85,
   long_sb 54 / barrier 13 / wait 8 %; kc8 5.60 us, occ 41 %, issue 21 %, elig 0.44, long_sb 49 / barrier 17 %;
   mb4 6.08 us, occ 85 %, barrier 24 %. (ncu's isolated launch favours kc8 by 0.2 us; in-graph and bench do not.)
4. SASS (NVRTC 13.0 = judge): A 38 regs, 7 LDG, afterUse 1 (= the escape x gather, by design); kc8 56 regs, 12 LDG,
   afterUse 1; mb4 32 regs, afterUse 2 (one W LDG at index 76 behind the first HMMA at 49). 0 stack everywhere.
5. Binding unit: DRAM stream. 12.7 MB in ~3.5 us local = 3.6 TB/s; the first wave alone puts 396 x 24.6 KB = 9.7 MB in
   flight (> BW x latency ~4.5 MB), so the 116 second-wave CTAs refill slots while DRAM is still busy with wave 1:
   the second wave is not on the critical path at m=1 (O8 iteration 5 saw the same; at m=8 the escape gathers made it
   matter). One-wave variants only add per-warp serialisation (kc8: 2x loads per warp) or sunk loads (32 regs).

## iteration 2 — n2 (qkv / pooler m=1) occupancy (ab_n2_v1.log, ab_n2_pad.log, ncu_iter1.log, nsys_iter1.log)
1. ab.py (3 placements x 30 reps x 2 rounds): qkv A 7.07/7.14, A' +0.05/-0.03, MB 8 (32 regs, 8/SM) +1.79/+1.62, MB 5 (40
   regs, 6/SM, different schedule) +0.13/+0.11; pooler A 11.58/11.61, A' +0.11/+0.09, MB 8 +2.42/+2.42, MB 5 +0.17/+0.13.
   smem padding (kernel unchanged, 40/50/70 KB dyn smem -> 5/4/3 CTAs/SM): qkv +3.3/+3.5/+4.1, pooler +6.2..+8.4.
2. nsys in-graph: qkv A 8.16 vs MB8 10.00; pooler A 12.70 vs MB8 15.23.
3. ncu: A (qkv) 8.99 us, 27.3 MB, occ 63 %, issue 52 %, elig 1.24, long_sb 48 / barrier 16 %; MB8 10.82 us, occ 86 %,
   issue 42 %, long_sb 66 %.
4. SASS (NVRTC 13): MB 6 40 regs, 11 LDG, afterUse 1; MB 7 and MB 8 both 32 regs (36 is not allocatable: 8-register
   granule), afterUse 4 (3 loads sunk: indices 53, 61, 82 vs first FFMA 37). Data per lane = 8 (lo) + 3 (codes) + 1
   (slot) + 16 (x) = 28 registers before addresses/accumulator: 32-register builds must sink.
5. Binding unit: DRAM stream with continuous refill (hook-0 trace: no wave gaps after the first; slot reuse 1512 / 3304
   CTAs); CTA duration under load ~1.7-1.9 us vs 68 KB in flight per SM = ~36 GB/s/SM >= the 34.5 GB/s each SM needs
   for 4.56 TB/s. The extra time vs the 0.81 + B/4.56 floor (qkv ~0.6, pooler ~0.4-0.7 us) is ramp + drain, not
   occupancy. The padding result is a warning for anyone touching n2: x reuse through L1 is essential (do not add smem).

## iteration 3 — unpooler m=8: spread the 64 CTAs over 128 SMs (unps_lab.py; ab_unp_*.log, prof_unp.log, sass_box.log)
unps (S5) = 64 CTAs x 32 warps x 128 k on 64 SMs. Lab = same F12 blob; tile split over a KS = 2 cluster (virtual warp
gw = rank * 32 + wid, 64 k per warp), rank 0 does the escapes; rank 1 writes its 128 sums into rank 0's smem
(mapa + st.shared::cluster) between barrier.cluster arrive.release / wait.acquire (phase 0 arrive right after the
loads so the start-up guarantee is free). Bit-identical outputs to S5 on normal + outliers inputs.
1. ab.py (3 placements x 30 reps, 2-3 runs, box 0/1), vs S5: cluster split **+0.19 / +0.20 / +0.27 / +0.27 / +0.33**; same with
   NW 16 (512 thr) +0.13; my K-unsplit rebuild of S5 +0.27 (one x LDG sunk: afterUse 9 vs 8) — so compare to S5 itself.
   Ablations (timing only, wrong results): no DSMEM transfer but barriers kept +0.32/+0.44; no cluster, no reduction
   (every CTA stores its partial) **-0.17 / +0.02 / -0.11 / -0.07**; cluster launch attribute without barriers -0.13 /
   -0.05; no x loads (cluster) +0.14 / +0.19 (x ~0.05).
2. nsys in-graph (30 reps): S5 3.76 / S5' 3.74 | cluster split 4.00 | no-reduction ablation 3.68.
3. ncu: S5 4.74 us, 6.38 MB, occ 47 %, issue 29 %, hmma 2.4 %, long_sb 29 / **lg_throttle 28** / wait 13 %; split 4.90
   us, issue 21 %, long_sb 51 / barrier 9 / lg_throttle 8 %; no-reduction 4.51 us, long_sb 57 / lg_throttle 8 %.
   The split removes the LSU throttle as predicted, but the kernel stays DRAM-latency bound.
4. SASS (NVRTC 13): S5 unps 54 regs, 19 LDG, afterUse 8 (= the 8 escape x gathers, by design); split 40 regs, 14 LDG,
   afterUse 8 (all 6 W/x LDG before the first HMMA). 0 stack.
5. Binding unit: DRAM latency (6.4 MB in ~2.7 us local = 2.4 TB/s, 49 % of peak) plus the LSU queue on the 64 busy SMs;
   halving the per-SM bytes buys at most ~0.1 us and every cross-CTA combine tried costs more (cluster barriers
   0.35-0.5 us at the end of a 3 us kernel). Dead end unless the combine is free (e.g. an 8-row swapped-operand tile
   with x as the A operand: 128 CTAs, no reduction, but full x per CTA; expected <= -0.1 us).

## iteration 4 — per-SM imbalance and end skew (audit_bysm.log, audit_skew.log, audit_balance*.log, gpcmap_box0.log)
Hook 50, 6-8 replays per shape. "by CTAs/SM": CTAs alone on an SM finish earlier when the SM is the limiter:
sk11_o_16 1/SM 3.89 vs 2/SM 4.13 us (-6 %), s6_rz2 2.56 vs 2.69 (-5 %), m8_pooler 11.63 vs 11.84, cgemv 7/SM 2.02 vs
8/SM 2.08, pu5 unpooler 12.26 vs 12.51. Balanced projections (T_max = last SM to finish; T_bal = total work / sum of
per-SM speeds):
| kernel | T_max | T_med | T_bal | end corr by blockIdx / by SM |
|---|---|---|---|---|
| **m8_pooler** (pooler m8) | 12.43 | 12.14 | **11.97** | +0.02..+0.23 / **+0.87..+0.91** |
| o8_m8 (o m8) | 4.03 | 3.82 | 3.82 | +0.64..+0.70 / +0.62..+0.67 |
| sk11_o_16 (o m16) | 4.43 | 4.22 | 4.23 | +0.18..+0.35 / +0.61..+0.73 |
| s6_rz2 (router m16) | 3.12 | 3.00 | 2.99 | +0.63..+0.66 / +0.62..+0.63 |
| pu5_pooler_16 / pu5_unpooler_16 | 12.59 / 12.66 | 12.48 / 12.54 | 12.48 / 12.53 | +0.02 / +0.17; +0.39 / +0.50 |
| qkvb 8 / 16 | 8.06 / 8.22 | 7.87 / 8.03 | 7.88 / 8.03 | +0.20..+0.31 / +0.11..+0.14 |
| unps_8 | 3.15 | 2.95 | 2.97 | +0.20 / +0.30 |
| cgemv, rz1 | 2.48, 2.82-2.90 | 2.38, 2.67-2.75 | 2.38, 2.67-2.75 | |
| n2 / nk_o1 (multi-wave) | | | | +0.87..+0.99 by blockIdx (dispatch order), 0 by SM |
m8_pooler: the fast SMs (finish 10.85-11.10 us: smid 6, 7, 32-37, 42, 43, 54, 55, 81) and slow SMs (12.30-12.40:
4, 16, 49, 50, 52, 57, 60, 96, 124, 128-130) are stable across runs while blockIdx->SM placement changes (4-17 % same).
GPC map (gpcmap.py, 200 launches of an 8-CTA-cluster probe): two complete 18-SM groups ({12,13,28,29,44,45,60,61,74,
75,88,89,102,103,116,117,120,121} and the same set +2); the other SMs only merged into 8-SM groups ({2g,2g+1,2g+16,
2g+17,...}; 6 GPCs of 16 SMs inferred). The fast and slow sets contain SMs of the same 8-SM groups (e.g. fast 32,33 /
slow 16,49; fast 34,35 / slow 50), so GPC size does not explain the per-SM speed. E8's kernel is per-SM latency-bound (register ring, 2 CTAs/SM), so per-SM path latency
differences show up directly. Lead for whoever owns pooler m8: ~0.46 us (x0.46 %/us ~ +0.2 %) is available only with
dynamic work balancing at sub-tile granularity whose combine costs < 0.4 us.

## Dead ends (measured)
- One-wave o m1 (KC 8 at 4/SM: +0.05..+0.10; 32-reg 4/SM: +0.57). More occupancy for n2 (8/SM: +1.6..+2.4, loads
  sink at 32 regs). Less occupancy for n2 via smem (+3..+8: L1 carveout). unps K-split over a 2-CTA cluster (+0.2..+0.33).
- Tracing with %globaltimer at every CTA exit or with atomics perturbs multi-wave kernels by 1-3 us.
