# PW: SM clock / power lever on v4111 (worker PW, 2026-11-03)

STATUS (2026-11-03 ~11:00 PT): CLOSED, no kernel lever converts. 7 GPU jobs (box 0 / box 1). Quantitative
report below; the final summary is at the end ("Conclusions"). No candidate moe_pw_*.py was built: every lever
measured here is either bounded below the 0.5% bar or loses on the producer side (L2-partition streaming).

## Tools (t3/PW/)
- pw_lim.py: synthetic time-programmed kernel (132 CTAs x 320 thr: 8 consumer warps running v4111's
  stage_compute<1,1,S6O> on ring slots, 1 producer warp streaming 36 KB bulk copies HBM->SMEM (4 in flight), 1 OBSERVER
  warp sampling (%globaltimer, %clock64) every ~1 us). Programs = segments (end us, ring on, compute warps, CTAs).
  Per program: 6 judge-style calls (512 MB flush.sum() before) -> clock by 5/10-us bin (cycle-weighted over CTAs);
  1 s loop -> NVML energy per call (flush-only loop subtracted), NVML POWER violation time and nvidia-smi
  clocks_event_reasons_counters per call, 2-ms SW-cap flags.
- pw_judge.py: a v4111 build (MOE_CLK stamps) under judge-like replays: j = flush; ours / js = flush; sglang; sync;
  flush; ours / jsn = no sync between; clock by 10 us of the call, mJ/call, violation us/call. Watchdog 240 s.
- pw_nsys_align.py: offline nsys GPU-metric alignment on kernel start (incl. the flush before).
- pw_l2map.py: L2 home-partition probe (latency per line from chosen SMs) + optional near/far 4 KB-chunk ring energy.
- pw_cc.py: local NVRTC sm_90a compile + SASS counts.

## Measured (Modal H200 box 0, driver 610.57, 700 W limit, EDPp multiplier 100%, idle 77 W, 25-50 C)

### J1 synthetic limiter (pw_lim1.json; flush before each call unless noted; GHz from the observer warps)
| program | in-call | 0-50 us | 50-150 | 150-250 | min (at us) | NVML POWER violation us/call |
|---|---|---|---|---|---|---|
| ring only (no MMA) | 650 W | 1.978 | 1.978 | 1.979 | 1.974 | 0 |
| compute 8 warps, no ring | 416 W | 1.772 | 1.797 | 1.797 | 1.715 @0 (MMA start transient) | 0 |
| ring + 1 compute warp | 649 W | 1.793 | 1.783 | 1.799 | 1.773 @70-100 | 0 |
| ring + 2 | ~765 W | 1.784 | 1.752 | 1.790 | 1.732 @80 | 90 |
| ring + 4 | ~788 W | 1.762 | 1.699 | 1.783 | 1.637 @65 | 836 |
| ring + 8 (full) | ~720-800 W | 1.754 | 1.668 | 1.765 | 1.59 @65-70 | 715-1000 |
| full, 1 ms idle before (no flush) | ~865 W | 1.716 | 1.563 | 1.718 | 1.46 @70 | 0 |
| full, flush, sync, 1 ms idle | ~950 W | | | | 1.486 @70 | 0 |
| full, 2 ms idle, flush (low duty) | | | | | 1.583 @70 | 0 |
| 130 us ring-only then full (same kernel) | | 1.974 | 1.916 | 1.731 | 1.688 @195 (65 us after the step) | 926 |
| full right after another full call | | 1.755 | 1.722 | 1.740 | ~1.69 @70-80 | 1454 |
| ramp 2/4/6/8 warps over 60 us | | 1.763 | 1.665 | 1.745 | 1.605 @75 | 764 |
| pulses 50 us full / 50 us idle | 535 W | bursts 1.70-1.78, no deep dip | | | | 0 |
| steps full 150 / idle 150 / full | | 2nd step dips to 1.465 @365 (deeper than 1st 1.59) | | | | 70 |
| NO_MMA build (ALU-only compute), full | ~813 W | 1.96 pre-dip | dip 1.768 @70 | 1.93 | ~-0.20 GHz like MMA | 882 |
| NO_MMA, idle before | ~974 W | 1.95 | dip 1.623 @70 | | | 0 |

Readings:
1. Two separate effects. (a) MMA start transient: the first ~30-40 us after the first MMAs run at 1.70-1.76 and
   climb to the 1.798 cap (also at 416 W with no ring: not power). (b) A power-step limiter: ~40-50 us after the load
   step the clock falls, bottoms at 65-80 us, recovers over ~100-150 us. Same depth with and without MMA (ALU-only
   build dips 1.96 -> 1.77): power-driven, not a tensor-core rule.
2. The dip depth depends on the PRE-STATE: idle before -0.32 GHz, judge flush before -0.20, 130 us at ring power or a
   previous heavy call before -0.10. The NVML SW-power-cap state does NOT change the dip (full vs full_sf: 1.581 vs
   1.583). NVML violation counters / 2-ms flags never see the dip itself (0 us for full_nf with the deepest dip):
   it is a sub-ms limiter invisible to NVML. Ramping the start over 60 us does not avoid it (1.605).
3. The SW power cap engages only when the loop duty is high (500-us calls); 200-us calls barely (full200 0.92 flags).

### J2 v4111 under judge-like replays (pw_judge1.json; MOE_CLK; 8 jm seeds x 16 reps)
| shape | us | mJ/call | in-call W | kclk j / js / jsn | clock by 10 us (j) | violation |
|---|---|---|---|---|---|---|
| bs1 | 28.0 | 18.5 | 661 | (moe1t: no stamps) | | 0 |
| bs8 | 63.5 | 47.0 | 740 | 1.786 / 1.782 / 1.777 | 1.84 1.76 1.78 1.79 1.79 1.74 1.72 | 0 |
| bs16 | 85.0 | 71.0 | 835 | 1.722 / 1.741 / 1.739 | 1.83 1.75 1.77 1.79 1.76 1.71 1.63 1.62 1.64 | 0 |
| bs32 | 111.2 | 93.1 | 837 | 1.719 / 1.718 / 1.710 | .. 1.79 1.77 1.71 1.66 1.63 1.62 1.65 1.67 1.70 | 0 |
| bs128 | 161.6 | 142.2 | 880 | 1.690 / 1.674 / 1.664 | .. 1.79 1.78 1.73 1.62 1.58 1.56 1.58 1.61 1.65 1.67 1.68 1.69 1.70 | 0 |
| bs256 | 195.2 | 175.7 | 900 | 1.679 / 1.653 / 1.663 | .. 1.79 1.73 1.64 1.57 1.55 1.57 .. 1.66 1.68 1.69 1.71 1.72 1.73 | 100 (j only) |
=> same step response as the synthetic kernel; SGLang alternation changes it by <= 0.03 GHz; the judge loop is not
SW-capped except marginally at bs256.

### J3 energy / clock decomposition of v4111 (pw_judge2.json, box 1, judge mode j, 8 jm seeds x 16 reps; timing-only
ablation switches; control = mean of two MOE_CLK copies a / a2 run first and last)
Energy per call has +-3% copy noise (a vs a2: 158 / 166 mJ at bs256; drift with GPU temperature 37 -> 42 C, ~+1%/C);
the in-kernel clock is reproducible to +-0.005 GHz between copies.

| bs | variant | us (vs ctl) | mJ/call | in-call W | kclk GHz (vs ctl) |
|---|---|---|---|---|---|
| 16 | control | 85.0 | 66.8 | 786 | 1.753 |
| 16 | data only (NO_COMPUTE) | 80.5 (-5.3%) | 50.6 | 628 | 1.980 |
| 16 | compute only (NO_LOAD) | 76.2 (-10.3%) | 27.3 | 359 | 1.806 |
| 16 | NO_X | 84.3 (-0.7%) | 65.0 | 771 | 1.758 |
| 16 | HMUL_NONE | 84.5 (-0.6%) | 62.9 | 745 | 1.758 |
| 16 | NO_MMA+NO_WGMMA | 84.3 (-0.8%) | 61.7 | 732 | 1.934 (+10%) |
| 16 | NO_DEQ | 82.4 (-3.0%) | 51.4 | 624 | 1.788 |
| 32 | control | 111.1 | 87.1 | 784 | 1.723 |
| 32 | data only | 104.2 (-6.2%) | 61.4 | 590 | 1.969 |
| 32 | compute only | 98.8 (-11.1%) | 41.9 | 425 | 1.798 |
| 32 | NO_X | 109.0 (-1.8%) | 81.6 | 748 | 1.737 |
| 32 | HMUL_NONE | 110.0 (-0.9%) | 76.6 | 696 | 1.746 |
| 32 | NO_MMA+NO_WGMMA | 109.5 (-1.4%) | 84.0 | 767 | 1.828 (+6%) |
| 32 | NO_DEQ | 109.8 (-1.1%) | 76.6 | 697 | 1.768 |
| 128 | control | 161.8 | 135.8 | 840 | 1.697 |
| 128 | data only | 152.2 (-5.9%) | 89.3 | 586 | 1.825 |
| 128 | compute only | 140.1 (-13.4%) | 41.0 | 293 | 1.803 |
| 128 | NO_X | 157.5 (-2.7%) | 126.4 | 803 | 1.735 (+2.3%) |
| 128 | HMUL_NONE | 160.1 (-1.0%) | 130.2 | 813 | 1.716 |
| 128 | NO_DEQ | 158.0 (-2.3%) | 134.1 | 848 | 1.701 |
| 256 | control | 195.0 | 162.3 | 833 | 1.684 |
| 256 | data only | 176.2 (-9.6%) | 112.2 | 637 | 1.841 |
| 256 | compute only | 172.8 (-11.4%) | 57.9 | 335 | 1.802 (no dip at all) |
| 256 | NO_X | 189.4 (-2.8%) | ~162 | ~860 | 1.743 (+3.5%) |
| 256 | HMUL_NONE | 194.5 (-0.3%) | ~164 | ~843 | 1.701 (+1.0%) |
| 256 | NO_DEQ | 186.7 (-4.3%) | ~168 | ~900 | 1.704 (+1.2%) |
(NO_MMA+NO_WGMMA does not remove the bs128/256 kernels' tensor paths: clock unchanged there.)

Readings:
1. Energy: the data side dominates. Data-only is 63-75% of the full call's energy (bs16 50.6 of 66.8 mJ, bs256 112 of
   162); compute-only 27-41%; they overlap in static power (~100 W x call time = 8-20 mJ). The decode ALU (NO_DEQ)
   is ~10-15 mJ at bs16/32, HMUL ~4-10 mJ, x copies ~2-9 mJ (energy deltas at bs128/256 are inside the copy noise).
2. Compute-only runs at 293-425 W and NEVER dips (flat 1.80); data-only (no MMA) runs 586-637 W and is limited to
   1.81-1.84 at bs128/256 (from 1.97): the limiter acts on the SUM.
3. Time is only weakly clock-sensitive: data-only is 90-95% of the full time at every shape; NO_MMA at bs16 / bs32
   lifts the clock +10% / +6% but time only -0.8% / -1.4%; HMUL_NONE at bs256: clock +1.0%, time -0.3%. The P-CLK
   "compute-paced, time ~ 1/clock" ceiling (+4% geomean) overstates the lever by >2x.

### J4 (box 1): power shaping, pre-states before the real kernel, L2 partition map
Synthetic work counts (warp-stages done in a 200-us call, flush before): full 232226 (kclk 1.719); 4 warps for the
first 40 us then 8: 227290 (-2.1%, kclk 1.735); 6 then 8: 231373 (-0.4%); 6 warps throughout 214647; 4: 199078.
=> lowering the first ~40-50 us of power does make the dip shallower (min 1.645 vs 1.603) and later high power does
NOT re-trigger it (lo50_hi), but the work lost early exceeds the clock gained: no "throttled start" lever.
(Same in 550-us calls: lo50_hi 657397 vs full 664476 warp-stages.)
Pre-state of v4111 itself (pw_judge3, 8 jm seeds x 40 reps, modes in order j, jr60, jf100, ji, jB):
| bs | j | flush + 60 us ring kernel | flush + 100 us ring+compute kernel | flush + 30 us idle kernel | j again |
|---|---|---|---|---|---|
| 256 | 196.21 us, kclk 1.688, min 1.557 | 196.58, 1.662, min 1.507 | 196.42, 1.665, 1.510 | 197.21, 1.650, 1.479 | 195.06, 1.691 |
| 128 | 161.69, 1.694, 1.564 | 163.24, 1.653, 1.493 | 162.24, 1.674, 1.533 | 163.58, 1.646, 1.475 | 161.48, 1.696 |
| 32 | 111.11, 1.727, 1.637 | 113.02, 1.676, 1.547 | 113.15, 1.686, 1.564 | 113.92, 1.667, 1.519 | 111.15, 1.732 |
| 16 | 84.95, 1.757, 1.660 | 86.52, 1.702, 1.555 | 85.95, 1.717, 1.582 | 86.85, 1.691, 1.530 | 84.76, 1.753 |
=> every extra kernel (even a high-power one) between the flush and ours DEEPENS the dip (the kernel boundary drops
the power; only an uninterrupted in-kernel pre-phase helps, as pre_r30/pre_r60 showed). Value of the clock:
kclk -0.04..-0.05 GHz (-2.5..-3%) costs +0.4..0.8% (bs256), +0.4..1.2% (bs128), +1.7..2.5% (bs32), +1.3..2.4% (bs16)
time. CORRECTED elasticity (refined from all three pre-states vs the mean of the j / jB controls, see the J5-J6
section): ~0.3 (bs256), ~0.4 (bs128), ~0.65 (bs16/32); the first-pass "0.1 at bs256" used one control copy.

L2 partition map (pw_l2map, box 1): L2-hit latency from SM 0 270-310 cycles for every line it warmed itself; from an
SM of the other partition half the 4 KB lines are ~480 cycles (far). SMs split 66/66 by smid (GPC pattern
"0011111100000011..."). Home partition of a 4 KB chunk = parity((addr >> 12) & 0xAB) ^ k(2 MB page): exact for
250/256 pages, 6 pages with a single noisy chunk. Probed at 4 KB granularity only (bits < 12 not tested; the energy
result below says 4 KB chunks are homogeneous).
4 KB-chunk ring, 9 chunks per 36 KB stage, 2 MB per SM (272 MB), flush before, no consumers:
| chunks | us | TB/s | mJ/call | pJ/B (call energy / bytes) | b2b NVML avg board / memory W |
|---|---|---|---|---|---|
| contiguous (mixed: half far) | 74.5 | 3.66 | 32.4 | 119 | 396 / 196 |
| all near (own partition) | 71.0 | 3.84 | 23.6 | 86.5 | 392 / 207 |
| all far | 118.2 | 2.31 | 40.1 | 147 | 333 / 152 |
=> the inter-partition fabric costs ~60 pJ per far byte and caps far traffic at ~2.3 TB/s. Near-only streaming:
-27% energy per byte, +5% rate at this size. L2-resident ring (no DRAM): 122 pJ/B vs DRAM ring 144 pJ/B:
DRAM itself is only ~15% of the per-byte energy; the on-chip path (L2 + fabric + crossbar + SMEM) dominates.

### J5-J6 L2-partition-local streaming in a COUPLED kernel (pw_near.py, box 0 / box 1, 300- and 200-us calls,
flush before, 2 interleaved rounds; hash re-probed in-process on both boxes: same SM split, same 0xAB mask,
33 noisy chunks of 523776)
| producer per 36 KB stage | ring only, no consumers | ring + 8 compute warps (full300) | ring + 4 warps (c4_300) |
|---|---|---|---|
| 1 x 36 KB contiguous (today) | 4.59 TB/s | 4.58 TB/s, kclk 1.72-1.74 | 4.58, 1.75-1.78 |
| 3 x 12 KB contiguous | | 4.58 | 4.58 |
| 4 x 8 KB + 4 KB contiguous | | 4.575 | 4.58 |
| 9 x 4 KB contiguous (lanes 0-8) | 4.58 | 3.20 | 3.85 |
| 9 x 4 KB contiguous, lane 0 serial | | 1.66 | 2.03 |
| 9 x 4 KB NEAR chunks | 4.58 | 3.49-3.74 | 4.23-4.36 |
| NEAR, adjacent pairs merged (~6 copies of 4-8 KB) | | 3.71 | 4.52-4.53 |
| 9 x 4 KB FAR chunks | 2.55 | 2.54 | 2.54 |
=> near vs mixed at the SAME copy structure: +9-11% rate under load (the fabric adds latency; far-only traffic caps at
~2.5 TB/s, and today's mixed stream puts ~2.3 TB/s on it), and -10..-27% energy per byte in the ring-only tests.
But near regions are at most 8 KB (bit 12 is in the hash, so an aligned 8 KB pair always holds one chunk of each
side; the contiguous near pairs are {4g+1, 4g+2} for half the 16 KB groups): a near stage needs ~6 bulk copies
of 4-8 KB, and 4 KB bulk copies are expensive when consumers load the SM (per-copy issue cost ~170 ns in parallel
lanes, ~320 ns serial: they queue behind the consumers' LDS traffic). Best near form (merged runs) = 3.71 TB/s
under full load vs 4.58 for one 36 KB copy (-19%), 4.53 vs 4.58 under half load (-1%). Steady-state DRAM rate is
NOT raised by near streaming when the producer keeps up (all modes 4.58 TB/s without consumers).
Verdict: the energy saving is real (~60 pJ per far byte) but every implementation needs >= 5-6 copies per stage
and would cost more producer bandwidth than it saves; plus two weight copies per layout and a per-page k table.
Not built. (Fact for later work: home(addr) = parity((addr >> 12) & 0xAB) ^ k(2 MB page); SM side = GPC pattern.)

### Time / clock elasticity of v4111 (from pw_judge3: the dip deepened by pre-states, same build, 8 jm seeds x 40 reps)
| bs | +60 us ring kernel before | +100 us ring+compute kernel | +30 us idle kernel | elasticity d ln t / d ln clk |
|---|---|---|---|---|
| 16 | +1.96% t, -3.02% clk | +1.29%, -2.14% | +2.35%, -3.63% | 0.60-0.65 |
| 32 | +1.70%, -3.10% | +1.82%, -2.53% | +2.51%, -3.61% | 0.55-0.72 |
| 128 | +1.03%, -2.46% | +0.41%, -1.27% | +1.24%, -2.87% | 0.32-0.43 |
| 256 | +0.48%, -1.63% | +0.40%, -1.42% | +0.81%, -2.33% | 0.28-0.35 |
(control j / jB copies differ by 0.6% at bs256, 0.1% elsewhere: bs256's elasticity is +-0.1.)
CEILING of any clock/power lever (clock held at the 1.798 cap for the whole call, from kclk j): bs8 0.8% x ~0.6 =
~0.5%, bs16 2.5% x 0.63 = 1.6%, bs32 4.0% x 0.66 = 2.6%, bs128 6.1% x 0.39 = 2.4%, bs256 6.4% x 0.30 = 1.9%, bs1 (no
stamps; ~1.77 GHz in the MMA-start transient) ~1% -> geomean ~+1.7% for a dip-free 1.80 GHz everywhere (P-CLK's
"time ~ 1/clock" +4% assumed elasticity 1). Realistic energy cuts reach a fraction of that: e.g. NO_X (all
activation copies gone, -164 MB on-chip traffic at bs256) lifts kclk +3.5% but is not implementable; HMUL_NONE (all
scale multiplies gone) +1.0% clk.

## For HV (partition-local streaming): exact facts and caveats (2026-11-03 11:10 PT)
- Hash: home(addr) = parity((addr >> 12) & 0xAB) ^ k(2 MB page), i.e. VA bits 12, 13, 15, 17, 19 within a 2 MB-
  aligned page plus one unknown bit per page (k, from PA bits >= 21; measured per page, structured but not
  derived). Assumes VA = PA below bit 21 (CUDA's 2 MB pages; held for every allocation tested). Verified on BOTH
  boxes: box 1 250/256 pages exact (6 pages with one noisy chunk, always the first chunk of a probe window), box 0
  1023 pages with 33 mismatching chunks of 523776 (noise level). Only the first word of each 4 KB chunk was probed:
  bits < 12 are NOT in the hash as far as the energy result shows (near 86.5 vs far 147 pJ/B), but untested by latency.
- SM sides by %smid (identical on box 0 and box 1, 66 / 66; '0' = side of SM 0):
  001111110000001100111111000000110011111100000011001111110000001111111100000011111111000000111111110000001111111100000011001100000000
- Near/far from the latency of a probing SM on the OTHER side after SM 0 warmed the lines: near ~270-310 cycles,
  far ~480 (L2 hit). SMs on SM 0's side see all lines warmed by SM 0 as near (SM 0's own reads do not separate).
- Probe gotcha: pw_l2map's dependent-load chain adds (loaded & 0x10000000) to the address: the probed buffer must be
  zeros while probing (random data -> illegal address).
- Contiguous near runs are at most 8 KB ({4g+1, 4g+2} in half of the 16 KB groups): a 36 KB stage = ~6 copies.
  4 KB bulk copies are expensive under consumer load (J5-J6 table): 9 x 4 KB in parallel lanes loses 30% of the
  ring rate under full load; merged near runs lose 19% (full load) / 1% (half load). 3 x 12 KB or 4 x 8 KB + 4 KB
  contiguous copies cost nothing. Far-only traffic caps at ~2.5 TB/s (fabric).
- Clock-side value of partition-local weights (my estimate from J3/J4, not measured): -60 pJ per far byte x half the
  weight bytes = -18 mJ at bs256 (-11%), -14 mJ at bs128 (-10%), -10 mJ at bs32 (-11%), -7 mJ at bs16 (-11%);
  energy -> clock ratio 0.15-0.3 (NO_X / HMUL_NONE points) -> clock +1.5..3% -> time ~-0.5..-1% from the clock
  alone, before any latency gain or copy-count loss.

## Conclusions (mechanism, what sets the clock)
1. Three layers: (a) the binary tensor cap 1.798 GHz per SM while it runs MMA (planner P2/P6); (b) a ~30-40 us
   start transient after the load step (1.70-1.76 -> cap), also with ALU-only work, worth ~0.5-0.8 us per call;
   (c) a power-step limiter: ~40-50 us after the step the clock falls, bottoms at 65-85 us, recovers over 100-150 us.
   Depth of (c) is set by the power of the first ~50 us after the step RELATIVE to the recent pre-state:
   idle before -0.32 GHz, judge flush before -0.20..-0.25, an in-kernel 60-130 us ring/compute pre-phase -0.10..-0.15,
   any extra kernel or idle gap between the flush and ours deepens it (-0.05..-0.08 more, +0.4..+2.5% time).
   Power added after ~50 us does not re-trigger it; lowering power for the first 40-50 us shallows it but loses more
   work than it saves (synthetic work counts -0.4..-2.1%). It acts on the SUM of data and compute power: compute-only
   (293-425 W) never dips, data-only (586-637 W, no MMA) is cut from 1.97 to 1.81-1.84, full (780-900 W) to 1.55-1.66.
   It is invisible to NVML (0 violation time, no reason flags, even for the deepest dip); NVML's SW power cap engages
   only for high-duty loops (500-us calls back to back with the flush) and does not change the dip.
2. Energy per call (judge mode): bs16 67 mJ, bs32 87, bs128 136, bs256 162 (780-850 W in-call); data side 63-75%,
   compute 27-41% (overlapping static ~100 W). Per byte the DRAM itself is ~15%; the on-chip path dominates and the
   inter-partition L2 fabric costs ~60 pJ per far byte (half of all bytes are far).
3. Levers measured: throttled start (no), smoother start / 60-us ramp (no: 1.605 vs 1.591), pre-states (all worse
   unless inside the same kernel, which the judge's flush rules out), L2-partition-local streaming (energy yes, but
   the >= 6 small copies per stage cost -1..-19% producer rate), ZPAD / suspend hints (WP-E: no power effect; not
   re-run). Remaining clock ceiling ~+1.7% geomean even if the dip vanished entirely.
