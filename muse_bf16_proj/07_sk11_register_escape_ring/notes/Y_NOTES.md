# Agent Y — router (phase 2, base bo1), notes

Deliverable: `claude/t3/Y/rt3.py` (`prepare_rt3(weights)`, `run_rt3(x, config, state)` -> out or None),
`claude/t3/Y/sub_rt3.py` (bo1 + rt3, for bench/check). Lab module: `yr.py` (variants, trace), `exp_common.py`
(bo1-wrapping experiments with existing kernels).

## Current routes (rt3 v6, built by mk_rt3.py from yr.py + yk.py; prepare 0.4 s)
- 2 <= m <= 8: `ys` RT=2, S=8, x-first (64 CTAs, cluster 8 along K, DSMEM split-K reduce), raw hi/lo.
- 9 <= m <= 32: `ya` RT=1, S=8: 4 zeroing CTAs (blockIdx 0-3, dispatched first) + 128 compute CTAs (16 rows x 1024 k,
  no cluster), red.add epilogue behind a release counter; raw hi/lo. No co-residency requirement (compute CTAs only
  wait for the 4 zeroing CTAs).
- 97 <= m <= 128: `rk` port with K split 16 (grid 128; in-kernel grid counter, all CTAs resident, 0.2 s trap).
- Everything else (m=1, 33..96, 129..256): None (bo1).

## Same-run table, rt3 v6 (bench_rt3_v6.log, box 0, --reps 60; local us; judge ~= local + 1.0)
check.py --layers router --graph: ALL OK (52 cases x 2 seeds + graph); exact_router.py --nesc 200: EXACT ROUTER ALL OK;
exact_all.py --layers router --nesc 200: EXACT ALL OK (that harness casts the router weights to BF16, so rt3 declines
and bo1 runs; exact_router.py is the FP32-weight version that exercises rt3).
| M | SGLang | bo1 | rt3 | rt3 kernel | delta | speedup vs SGLang bo1 -> rt3 |
|---|---|---|---|---|---|---|
| 1 | 9.34 | 2.64 | 2.59 | bo1 rgemv | same kernel | x3.54 -> x3.60 |
| 8 | 25.22 | 3.28 | **3.09** | ys | -0.19 | x7.68 -> x8.17 |
| 16 | 14.26 | 3.66 | **3.15** | ya + zeroing CTAs | -0.51 | x3.89 -> x4.53 |
| 32 | 14.62 | 4.26 | **3.44** | ya + zeroing CTAs | -0.82 | x3.43 -> x4.26 |
| 128 | 15.62 | 5.13 | **5.05** | rk S=16 | -0.08 | x3.05 -> x3.09 |
| 256 | 16.80 | 6.08 | 6.06 | bo1 rk | same kernel | x2.76 -> x2.77 |
Router-only geomean x3.809 -> x4.116 (judge-projected x3.237 -> x3.440).
Across all runs (v4..v6, both boxes): m=8 -0.17..-0.23, m=16 -0.16..-0.56, m=32 -0.79..-0.89, m=128 -0.08..-0.28.
In-graph nsys (iter8_prof): m=8 3.90 vs 4.19, m=16 4.03 vs 4.45, m=32 4.45 vs 5.26, m=128 5.82 vs 5.84 (bo1).

## Baseline facts (bo1)
- Kernels actually run (nsys, nsys_base0.log): m=1 rgemv (FP32 W, 256 CTAs x 1 row), m=8/16 rr_small (Nathen, cluster
  16, 256 CTAs), m=32 sk2, m=128/256 rk. In-graph durations 3.58 / 4.16 / 4.42 / 5.22 / 5.79 / 6.90 us (nsys
  durations ~1.0-1.2 us above bench local).
- Existing-kernel quick variants (bench_e1/e2.log): rr_small at m=32 (MT=4) 4.48 vs sk2 4.27 (worse: 2 waves, see
  below); Nathen wgmma configs (unused in bo1): m=32 4.14 vs 4.27 (tie-ish), m=128 6.24, m=256 7.48 (worse); rk split
  S at m=128: S=16 4.98, S=12 (bo1) 5.08 vs bo1 route 5.34 in the same run (noise ~0.3), S=10/11/14 worse;
  m=256 S=8 is the max (grid 128); S=6/7 worse.
- NOTE for experiments: `torch.zeros(...)` inside kernel_fn is captured as a fill node (+0.9 us). Allocate in prepare.

## Measured facts
- Cluster co-residency on this H200 (cuOccupancyMaxActiveClusters): cluster 16 at 1 CTA/SM: 7 clusters; at 2/SM: 14;
  at 4/SM: 28. Cluster 8 at 1/SM: 15; at 2/SM: 30. => 128 CTAs in clusters of 8 at 1 CTA/SM (16 clusters) do NOT fit;
  rr_small at MT=4 (95 regs, 2/SM) runs 2 waves at m=32 (trace: start p90 3.6 us).
- Placement (trace %smid): cluster-16 grids of 128/256 CTAs land on 112-116 SMs with up to 2-3 CTAs per SM;
  ys RT=2 S=16 (128 CTAs, 2/SM packing) is much slower than RT=2 S=8 (64 CTAs, 1/SM): m=8 3.53 vs 2.90.
- In-kernel timeline (clock64, ys RT=2 S=8 m=8, 64 CTAs): all W landed 2.7-2.9 us after start, MMA done +0.35,
  DSMEM push/recv +0.3, final sum/store +0.18 => ~0.85 us tail after the stream. MMA order following W arrival
  (xf=3) does not help (arrivals are unordered). m=32 (MT=4, 64 CTAs): MMA phase 1.15 us after the last load
  (128 HMMA/warp, 2 warps/SMSP => ~1 us at 6.7 cyc/HMMA): MMA-bound tail.
- Router weight stats (3 seeds): hi exponent 97..122 (16-window [108,123] misses ~290 weights), lo==0 for 17-33
  weights (W exactly BF16), lo exponent offset d = Ehi-Elo-8 in 0..15 (d>=1 99.8%); delta(e7) = (Ehi>>1)-4-(Elo>>1)
  in 0..8 (8: ~30 weights).

## iteration 1 — ys raw hi/lo, RT=2 S=8 xf (m <= 8)   [iter1_prof.log, bench_rt3_v1.log]
- nsys in-graph (same run): m=8 ys 3.87 us vs bo1 rr_small 4.16 (-0.29); m=16 both rr_small 4.42/4.46.
- ncu (full set, flushed, m=8): ys dur 4.90 us, DRAM read 8.53 MB -> 1.74 TB/s = 36% of 4.8 TB/s over the ncu
  duration (2.2 TB/s = 46% over the nsys in-graph time); grid 64 x 256 thr, 157 regs, occupancy 12% (8 warps/SM),
  issue 5.4%, hmma 2.1%, eligible 0.07 warps/clk; stalls long_scoreboard 58%, lg_throttle 14%, wait 12%.
  bo1 rr_small: dur 5.34, same 8.53 MB (1.60 TB/s = 33%), 256 CTAs x 48 regs, occ 25%, issue 9.5%, hmma 1.4%,
  stalls long_scoreboard 45%, barrier 25%.
- SASS (nvcc 12.1 -cubin, same source): ys_2_8_1: 157 regs (NVRTC 157), 0 spill, 0 local; 36 LDG (32 W + 4 x),
  0 LDG after the first HMMA; 32 HMMA; no decode. ys_2_8_4: 246 regs, 48 LDG, 128 HMMA.
- Binding unit: HBM stream latency/bandwidth. In-kernel trace: the 8.4 MB lands 2.7-2.9 us after start (~3.0 TB/s =
  63% of peak inside the load window, 64 SMs), then a ~0.85 us latency tail (MMA 0.35, DSMEM 0.3, final 0.18) with
  DRAM idle. Tensor pipe idle 98%. Next: fewer bytes (24-bit pair format: 6.3 MB) and a shorter tail.

## iteration 2 — ys with the 24-bit pair format (hi/lo low bytes + 4-bit codes, 6.3 MB)   [bench_ysc1, iter2*.log]
- Format: per value hi low byte, lo low byte, hi code (sign | e7-54, window E 108..123), lo code (sign | delta,
  lo e7 = hi e7 - 4 - delta). Decode = SIMD byte ops on 4 high bytes + PRMT: SASS hot loop per warp/k-unit (RT=1):
  32 PRMT + 32 LOP3 + 16 IADD3 + ~11 SHF for 64 values = 1.4 ops/value; regs 54-62 (MT 1-2), 0 spill.
  Decode emulated in torch (ys_decode_codes) == packed layout (0 mismatches except escapes).
- Escapes (hi exponent < 108, lo == 0, delta == 8): 342-360 per seed (0.016%), ALWAYS present for random weights
  (lo == 0 ~30/seed), so they must be handled in-kernel.
- Timing (same run, local us, m=8 / m=16): bo1 3.29 / 3.60; raw ys(2,8) 3.07 / 3.59; 24-bit ys(1,16) without
  escape handling (wrong at 342 weights) 3.01-3.06 / 3.16-3.24 => the format is worth ~0.35-0.4 us at m=16, ~0 at m=8.
  Compressed RT=2 S=8 (64 CTAs): worse than raw (decode on 64 SMs); RT=1 S=8 (128 CTAs): worse.
- Escape handling variants (all correct, all too slow):
  * escape warp (9th warp, LDG entries + LDG x + smem fp32 atomics): 9-10 us (fp32 smem atomicAdd CAS-serializes on
    the shared padding address); skipping padding: still +0.5 (entries land at 2.3 us behind the W flood).
  * producer-side x tile in smem + atomics: +1.5-3 us.
  * per-warp register correction via shuffles, LDG entry table: +0.4 us (zero-filled table alone: +0.16-0.23).
  * same with a __constant__ table (LDC): +0.55 us (constant-cache misses after the flush).
  * generated per-CTA switch with literal positions (no table loads): +2.2 us (150 KB of SASS -> cold I-cache).
  => not shipped. Open idea: per-warp escape bitmask in the kernel param block (already fetched) so only the ~15%
  warps with escapes do work; or a 25-bit layout with a wider hi window.
- Binding unit for m=16: HBM stream (6.3 MB) + ~0.9 us latency tail; tensor 1-2%, issue ~9-10%.

## iteration 3 — rk (alygg77's router GEMM, Apache-2.0) ported to NVRTC (yk.py) + variants, m=128/256
- Trace (globaltimer per CTA, t_yk.py): m=128 S=12: first TMA stage lands 1.3-2.0 us after start, last stage 3.6-4.1,
  wgmma done +0.25, zero-barrier ok +0.25, red.adds issued +0.5 (5.1). m=256 S=8: first stage 1.0-1.5, last 4.9-5.4
  (384 KB per CTA in ~3.6 us = ~107 GB/s per SM), reds issued 5.6-6.1. Helper-warp grid counter is off the critical path.
- ncu (iter3_ncu2.log, flushed, NVTX target): m=128 rt3-rk S=16: 6.75 us (bo1 rk S=12: 7.17), DRAM 10.5 MB = 32.5% of
  peak, L2->SM 25.2 MB (both), L2 throughput 51%, issue 13.7%, hmma-pipe 1.5% (wgmma pipe metric n/a on this ncu),
  occ 8.6%, stalls long_scoreboard 49%, wait 21%. bo1 m=256 rk: L2->SM 50.3 MB, DRAM 12.6 MB = 33%, L2 50%.
- Binding unit: per-SM L2->SM ingress during the stream (~107 GB/s per SM ~ 90% of a 64 B/clk port), then a fixed
  ~1 us tail (last wgmma + zero barrier + 16 KB of red.add per CTA). DRAM only 33%, tensor ~1-2%.
- Variants (same-run local us, m=128 / m=256; bo1 5.07-5.11 / 6.04-6.27):
  * S=16 at m=128 (grid 128): 4.69-4.99 (-0.12..-0.28, shipped). m=256 S=8 port: 6.00-6.20 = tie (not shipped).
  * W TMA multicast across token tiles (cluster 2 at m=128): last stage 5.1 vs 4.1 (worse); cluster 4 at m=256 is
    not co-resident (30 clusters of 4 max at 1 CTA/SM < 32) -> the in-kernel grid barrier hangs (trap added).
  * BN=128 (m64n128, x re-read halved): m=256 stream ends 0.8 us earlier but wgmma tail +0.5 and 32 KB reds/CTA
    (+0.5): 6.12-6.58 vs 6.0 (worse); m=128 S=32: 5.69 (reds 1 us).
  * DSMEM pair pre-reduction (cluster 2 along K, half the reds): +3 us (stream itself slows to 5.4 us; not understood).
  * ring depth QS: 3 / 4 / 5 / 6 / 8 at m=128: 6.07 / 5.20 / 5.14 / 4.99 / 5.16 -> 6 is best.

## iteration 4 — `ya`: the ys main loop without clusters (m 9..32)   [iter4a.log, bench_rt3_v4.log, iter4_prof.log]
- Design: grid 128 = 16 row tiles (16 rows) x 8 K splits (1024 k), 8 compute warps (2 k-units each, all hi/lo A
  fragments + x B fragments LDG'd up front, mma.sync) + 1 helper warp that clears this CTA's slice of out, bumps an
  rk-style grid counter (release), spins (acquire, 0.2 s trap) and arrives on a CTA mbarrier. Compute warps sum their
  partials in smem, stage [token][row], wait on the mbarrier, red.add.v4 16 rows x m tokens into out. No cluster
  => no co-residency limit (cluster-8 at 1 CTA/SM caps at 120 CTAs; here 128 CTAs at 1/SM, 159 regs at MT=4).
- Same-run local us (bench_rt3_v4): m=16 3.24 vs bo1 3.58 (-0.34), m=32 3.31 vs 4.20 (-0.89). Variants: RT=2 S=16
  (32 rows x 512 k): m=16 3.28, m=32 3.87; m=8: ya 3.09 vs ys 2.88 (ys kept for m<=8).
- nsys in-graph (iter4_prof): m=16 ya 4.21 vs bo1 rr_small 4.50; m=32 ya 4.42 vs bo1 sk2 5.15.
- ncu (flushed): m=32 ya: 5.54 us, DRAM 8.92 MB = 33.7% of peak, L2->SM 16.8 MB (x re-read by 16 row tiles; bo1 sk2
  25.8 MB), L2 44%, issue 8.8%, hmma 4.0%, occ 13.7% (9 warps/SM), eligible 0.11; stalls long_scoreboard 56%,
  wait 10%, lg_throttle 9%. m=16: 5.98 us, 8.66 MB DRAM (30%), L2->SM 12.6 MB, hmma 2.1%, long_scoreboard 69%.
- SASS (nvcc 12.1, same source): ya_1_8_4: 159 regs (NVRTC 159), ya_1_8_2: 111, 0 spill, 0 local; all LDGs before the
  first HMMA.
- Binding unit: HBM latency/bandwidth of the 8.4 MB stream (all loads in flight at t=0; ~2.6 us to land), then the
  MMA + smem sum + red.add tail; DRAM 30-34% of peak averaged over the kernel, tensor <5%. Next: the 24-bit pair
  format (6.3 MB) with escapes handled by the idle helper warp (global red.add of x[tok][k]*delta, off the critical path).

## iteration 5 — ya + 24-bit format, escapes via the helper warp   [iter5a/b/c/d.log]
- Escapes moved off the compute warps: the idle helper warp (already waiting on the grid counter) adds
  x[tok][k] * delta for this CTA's escapes with global red.add after the zero barrier. Tables: (a) LDG table
  (entries land behind the W flood at ~2+ us, then dependent x loads): +1.6 us; (b) escape list in the kernel
  parameter block (PAE struct: eo[132] u16 + 472 x (row|k<<5, fp32 delta) = 4080 B; deltas needing >24 bits are
  split into 2 entries; 364 entries for 360 escapes): correct, m=8/16/32 3.48/3.71/4.16 vs raw ya 3.40/3.45/3.59.
  => the decode is not hidden: all of a warp's loads land at the end of the stream, so the 1.4 op/value decode
  (and the x2 HMMA at m=32) runs after the stream. Not shipped.
- In-kernel trace of raw ya(1,8) (t_trace_ya.py, globaltimer): helper releases its counter at 2.3-2.6 us (its zero
  stores are acked behind the load flood); zero barrier ok 3.07 (m=8) / 3.33 (m=16) / 3.58 (m=32) vs staged partials
  2.82 / 3.33 / 3.58: the helper is the critical path only at m=8 (+0.25 us). Zero-first (compute warps wait at a
  named barrier until the helper's stores are issued): worse (4.14/4.05/3.87).
- Geometry (same run, m=16/32): ya(1,8) 3.41/3.66, ya(1,16) 256 CTAs 3.76/4.00, ya(2,8) 64 CTAs 3.57/5.73,
  x-first vs W-first: equal.

## iteration 6 — W through cp.async (LDGSTS) into smem with one commit group per k16 half   [iter6a.log]
- Goal: let decode + MMA of early groups overlap the stream (cp.async.wait_group N per group; LDS to registers).
- 45-120 regs (vs 90-159), 54-85 KB smem. Correct (t_ya.py). Timing: m=8/16/32 raw 3.92/4.20/5.07, 24-bit
  3.87/4.11/4.90 vs LDG-to-register ya 3.29/3.47/3.82 => LDGSTS streaming is much slower here; dropped.
- Note: the same ya kernel measured 3.53 (rt3) and 3.82 (lab copy) at m=32 in one run -> candidate-to-candidate
  noise of ~0.1-0.3 us in a run; ship decisions use repeated same-run A/Bs.

## iteration 7 — rk epilogue as one TMA tensor reduce-add (cp.reduce.async.bulk.tensor .add, f32)   [iter7a.log]
- Correct (rows >= M clipped by the map), but 5.05 vs 4.81 (m=128) and 6.30 vs 6.09 (m=256): the proxy fence +
  extra barrier + bulk reduce completion cost more than 1024 red.v4 per CTA. Dropped.

## Dead ends / measured limits (summary)
- Escapes of a 24-bit pair format are unavoidable for random weights (lo == 0 in ~30 weights per seed); every way of
  applying their corrections cost >= +0.25 us, and the 1.4 op/value decode is not hidden under the LDG-all-up-front
  stream (all loads of a warp land at the end) -> raw hi/lo wins at every M.
- W multicast (rk, cluster along token tiles), cluster-2 DSMEM pre-reduction (rk), BN=128 tiles, TMA reduce,
  cp.async (LDGSTS) W, zero-first helper ordering, ring depths != 6: all slower.
- Helper warp's `red.release.gpu` (MEMBAR) completes only after the SM's LDG flood drains (~2.3 us): the in-kernel
  zero barrier is the critical path of ya at m=8 (so ys keeps m<=8).

## iteration 8 — ya with 4 dedicated zeroing CTAs (grid 132)   [iter8a/b.log, bench_rt3_v5.log, iter8_prof.log]
- Root cause found for the helper latency: a `red.release.gpu` (MEMBAR) on an SM with an LDG flood in flight completes
  only when the flood drains (~2.3 us). Fix: 4 extra CTAs (blockIdx >= 128, no loads) clear out and bump counter A
  with release (fast on an idle SM); compute CTAs take their generation from a relaxed atom.add on counter B at start
  and their helper warp spins on A >= 4*(gen+1). Trace: helper done 2.05-2.3 us (was 3.07-3.33), never critical.
- Same run: m=16 3.15 vs 3.46 (old ya) vs bo1 3.74; m=32 3.58 = 3.59; m=8 3.26 (ys 3.07 stays).
  ya(1,4)+zero CTAs (64 compute CTAs): m=8 3.21, m=16 5.02, m=32 8.54.
- ncu (flushed): m=16 ya 5.50 us, DRAM 8.66 MB = 32.9% of peak, L2->SM 12.6 MB, L2 39%, issue 6.7%, hmma 2.2%,
  occ 13.5%, elig 0.08, stalls long_scoreboard 63%, wait 10%. m=32: 5.44 us, DRAM 8.92 MB = 34.3%, L2->SM 16.8 MB,
  hmma 3.6%, issue 8.5%, long_scoreboard 59%, lg_throttle 10%.
- Binding unit: HBM stream (8.4-8.9 MB issued at t=0, lands by ~2.6-2.8 us in-kernel = ~3.2 TB/s inside the load
  window, 67% of 4.8) + an MMA/sum/red tail of ~0.5-1 us; DRAM averaged over the whole kernel ~33%. The remaining gap
  to the 8.4 MB floor is the first-byte latency (~1 us) and the tail.
- Also measured: launching rk with only a cluster attribute (dims 2, no code change) slows its TMA stream from 3.6 to
  4.6 us at m=128 (placement); cluster 4 at 1 CTA/SM is not co-resident (30 max) -> grid barrier trap.

## iteration 9 — rk with the 4 zeroing CTAs (grid 132)   [iter9a.log]
- Correct, but m=128 5.18 vs 4.91 (rk S=16) and m=256 6.23 vs 6.07: the compute CTAs' TMA stream ends ~0.75 us later
  with the 4 extra CTAs in the grid (zbar was only ~0-0.25 us behind the last wgmma in plain rk). Dropped.
- ya at m=8 with zeroing CTAs: (1,8) 3.10, W-first 3.04, (2,8) 3.12 vs ys 2.96 -> ys stays for m<=8.

## iteration 10 — more ya/ys variants   [iter10a/b/c.log]
- ya with 16 compute warps (KU=1, 544 threads): m=8/16/32 3.09/3.37/4.20 vs 8 warps 3.15/3.14/3.67 (more smem
  partials to sum) -> no.
- ys cluster-4 geometries at m=8: (1,4) 3.20, (1,4) W-first 3.25, (2,4) 4.42, (1,8) 3.19 vs (2,8) 3.06 -> no.
- ya two-batch W loads (k-unit 1 issued when k-unit 0's last load lands, via an asm data dependency): raw
  3.23/3.47/3.77 vs 3.21/3.22/3.66 -> the stream lengthens more than the MMA overlap saves.
- ya + 24-bit format + param-block escapes + zeroing CTAs (helper now also applies the escapes): m=8 3.00 (= ys 3.03,
  vs raw ya 3.21), m=16 3.32 (raw 3.22), m=32 3.86 (raw 3.66); with two-batch loads 3.02/3.38/3.78. Compression only
  pays where the MMA/decode tail is small (m=8) and there it only ties ys.

## iteration 11 — zeroing CTAs first (rt3 v6)   [bench_rt3_v6.log]
- The 4 zeroing CTAs are now blockIdx 0-3 (dispatched first); compute CTA index = blockIdx - 4. Compute CTAs only wait
  for counter A (bumped by the zeroing CTAs), so ya no longer needs all 132 CTAs co-resident (no deadlock if fewer
  SMs are free). Same speed (m=16 3.15, m=32 3.44 vs bo1 3.66 / 4.26). check ALL OK, EXACT ROUTER ALL OK.
- Integration check: rt3.py run through build_t3.prefix_names("_yr_") (rt3_prefixed_test.py) -> check.py --graph
  ALL OK (NVRTC sources and f-string generators survive the prefixing).

## Open ideas (not done)
- m=256 is bound by per-SM L2->SM ingress in rk (384 KB per CTA at ~107 GB/s) + 2 MB of red.add; every re-tiling
  trades ingress for reds at ~constant cost (BN=128: -0.8 us of stream, +1 us of MMA tail + reds), and cluster
  launches (multicast / DSMEM pre-reduction) slow the TMA stream on this H200. A cheaper split-K reduction is the
  missing piece.
- m=1: a 24-bit SIMT GEMV would save ~0.4 us of stream but adds ~0.4-0.55 us of decode after the stream
  (~7 ops/weight); estimated net <= -0.15 us, not attempted.
- m=8/16: the 24-bit format ties/loses because the LDG-all-up-front stream lands all of a warp's data at the end;
  a stream that lands progressively without lengthening (two-batch LDG, cp.async) was slower in every form tried.
