# WP-FAST: time to first copy / first data in the bs8 / bs16 / bs32 kernels (2026-11-01)

Worker WP-FAST. Base line B dev/moe_v1514.py, line A dev/moe_v3960.py. Kernels moe_4_1_1_4_1 (bs2-8), moe_4_2_1_4_1
(bs9-16), moe_8_4_1_4_1 (bs17-32, MW 1). Namespace dev/moe_v4100-v4199.py, defines MOE_FT_*, scripts dev/wpfast_*,
logs kda_ws/runs/wpfast_runN.log.

## Files
- dev/wpfast_mk4100.py BASE OUT: the patch (both lines; every switch off by default; default build = the base's SASS in
  all 16 kernels, local NVRTC 13 + MOE_S5R_FIT, checked after every change). Outputs: dev/moe_v4100.py (run 1),
  dev/moe_v4101.py (line B, run 2 on) / dev/moe_v4150.py, dev/moe_v4151.py (line A).
- dev/wpfast_probe.py: per-CTA startup chain from the MOE_FT_TS stamps (+ MOE_FT_POLL barrier poller), judge protocol
  (graph replay after a 512 MB flush, jm seeds); "cold" = after a replay of a second compile of the same build, "warm" =
  after its own replay. Slot map in its header.
- dev/wpfast_run1.sh, run2.sh (Modal jobs), logs kda_ws/runs/wpfast_run1.log, run2.log; bench JSON dev/remote_out/wpf1_p*.json,
  wpf2_p*.json.

## Switches (dev/wpfast_mk4100.py)
| switch | what | kernels | loops (local NVRTC 13) |
|---|---|---|---|
| MOE_FT_TS / MOE_FT_POLL | diagnostic stamps / warp-11 barrier poller (stage s landed, released) | all | trace build |
| MOE_FT_GSD | global scales' x 256 at the smem store after the bitmaps, not right after the hoisted load at entry (warps 0-1 waited there for the cold load before the init barrier) | bs8/16/32 | every stage + producer loop EXACT |
| MOE_FT_INIT | (4,*,1): the mbarrier inits on tid 352 (no loads in flight) instead of tid 0, whose init fence waited for its hoisted ids / parity / scale loads (MOE_RT_INIT did this for KW 1) | bs8/16 | stage + producer steady loop EXACT |
| MOE_FT_PAR (+PARM mask) | the parity-flip atomics (CB + 6 / + 7) by the signaler at its role start instead of in the routing (the tables barrier waited for their round trip) | bs8/16 (PARM=3) | EXACT at bs8/16; bs32: stage loops renamed (excluded) |
| MOE_FT_NW0 | (SEPI kernels) the consumers' 16 KB post-norm-weight copy into smem skipped by a never-true runtime guard: with MOE_WPS_SEPI nothing reads it (the epilogue warps read P.nw1) | bs8 | EXACT (form NW0K=0; NW0K=1 = compiled out re-schedules the stage loop, 0.595) |
| MOE_FT_EIDS | the early producer's ids loaded at entry | bs8/16 | EXACT on line B; line A bs16 renamed; bench bs16 -0.28%: dropped |
| MOE_FT_E4=n | the early path issues stages 2..n-1 of the first item too (eis still after 2), so they leave before the producer's cold loop code runs under the burst | bs8/16 | NOT exact (stage loops renamed / 0.69) in every form tried (unrolled, rolled, __noinline__ call) |
| MOE_FT_CDL=ns | warps 0-7 spin ns after the init barrier (the producer's cold code fetched first) | bs8/16 | |
| MOE_FT_TRP | producer bitmaps by a 5-round shuffle transpose (-2.5 KB code) | bs8/16 | producer steady loop renamed: not run |

## Run 1 (kda_ws/runs/wpfast_run1.log): the start of v1514 (stamp build, 6 jm seeds x 2 reps, per-CTA medians, us after the first CTA's entry)
(stamp build: consumer stage loops re-scheduled by the stamps, whole call +6 us; the startup order is what it shows;
"warm" differs from cold by only 0.0-0.3 us at these steps: the >85 KB kernels evict their own startup code, see
wpcold.md, so the cold - warm column does not isolate the i-fetch cost.)

| step (cold) | bs8 | bs16 | bs32 |
|---|---|---|---|
| init barrier | 0.91 | 0.97 | 0.62 |
| bitmaps barrier | 1.91 | 2.21 | 1.28 |
| parity atomic back (its thread) | 2.26 | 2.58 | 1.84 |
| tables | 2.80 | 3.08 | 2.08 |
| pair list | 2.98 | 3.27 | 2.37 |
| consumers' gather done (bs32) | - | - | 3.26 (p90 6.07) |
| setmaxnreg.inc | 3.10 | 3.39 | 4.36 |
| eis passed | 3.38 | 3.79 | - |
| NW1S copy done | 4.92 | (none) | - |
| consumers at the first wait | 4.99 | 3.89 | 4.41 |
| producer past setmaxnreg.dec | 1.28 | 1.29 | 4.00 (after FST) |
| producer role start | 1.53 | 1.58 | 4.87 |
| producer's first-item bitmap (bs32: wp32_fst entry) | 2.01 | 2.34 | 1.60 |
| first item worked out | 2.24 | 2.62 | 2.18 |
| stage 0 copies issued | 2.49 | 2.86 | 2.96 (111 FST CTAs) |
| stage 1 copies issued | 3.05 | 3.18 | 3.31 |
| eis arrived / FST returned | 3.07 | 3.65 | 3.76 |
| rtd seen | 3.67 | 3.99 | 3.94 |
| stage 2 copies issued (first loop stage) | 5.95 | 6.40 | (FST issues 4) |
| stage 0 landed (poller) | 4.30 | 4.61 | 4.52 (p90 7.75) |
| first stage passed (tid 0) | 5.09 | 4.56 | 4.97 (n<=16: 4.78, wgmma-first: 9.03) |
| first stage released by all 8 warps | 6.89 | 6.70 | 7.73 |
| stage 2 / 3 landed | 7.72 / 9.01 | 8.06 / 9.38 | 6.67 / 7.07 |
| first refill (stage NS) issued / landed | 8.20 / 10.14 | 8.08 / 10.44 | 8.00 / 9.72 |
| gate at the first wait (CTA count: data later than the consumers / consumers later) | 123 / 1461 | 1492 / 92 | 893 / 691 |
Durations: init barrier 0.83 / 0.89 / 0.55; issue -> land (stage 0) 1.78 / 1.65 / 1.47; first stage (passed -> released)
1.73 / 2.07 / 2.70 vs ~1.0 steady; release of slot 0 -> refill issued 1.35 / 1.42 / 0.42; rtd -> stage-2 issue 2.32 / 2.39.

Readings:
- bs8 is CONSUMER-gated (92% of CTAs): the consumers spend 1.50 us after eis copying the post-norm weights into shared memory
  (a cold global round trip before their first stage), and nothing reads that copy (SEPI epilogue warps read P.nw1).
- The init barrier (0.9 us at bs8/16, 0.6 at bs32) waits for cold loads: warps 0-1 multiply the hoisted gs load by 256
  right after issuing it (FMUL right after the LDG in SASS), and in the KW 2 kernels tid 0's mbarrier-init fence waits for
  its hoisted ids / parity loads.
- The parity atomic (issued after the bitmaps barrier) returns 0.35-0.57 us later; the tables barrier waits for it.
- bs16 is DATA-gated (94%): the producer's chain (role 1.58 -> first copy 2.86 -> landed 4.61) sets the start.
- At bs8 / bs16 the ring's stage 2 is issued only ~6 us (2.3-2.4 us after rtd): the producer's loop code is first
  executed under the initial weight burst (cold misses wait behind the HBM queue), so the ring holds 2 stages until ~7 us.
- bs32: the 32 wgmma-first CTAs see first data at 9.0 us (the all-CTA gather release xrel 6.80: warp 9's gpu-scope release
  waits for the SM's in-flight FST weights); the FST CTAs at 4.78.

## Run 1 candidates
Stamps, B = GSD + INIT + PAR(bs8/16) + NW0 + EIDS, bs8 (cold): init 0.43, bitmaps 1.42, tables 2.07, pair list 2.26,
inc 2.38, eis 3.11 (now waits for the producer), first wait 3.28 (was 4.99) -> DATA-gated (1560 / 24); but the producer's
first copy does not move (role 1.67, first copy 2.67 vs 2.49): released 0.5 us earlier, its cold code fetch overlaps the
consumers' routing fetch; stage 0 lands 4.33 (4.30), stage 2 issued 6.14 (5.95). bs16: init 0.41, tables 2.29, but the
producer's first copy 3.21 (2.86) -> landed 4.80 (4.61): later. bs32 (= GSD alone): init 0.33, bitmaps 1.12, tables
1.94, FST copy 2.82 (2.96), first stage passed: n <= 16 CTAs 4.62 (4.78), wgmma-first 8.93 (9.03).
bench2 jm 100 x 2 passes, 4 middle v4100 controls, fillers at both ends (% vs the control mean; band = control spread):

| | bs8 | bs16 | bs32 | bs1 (placebo) |
|---|---|---|---|---|
| controls p1 / p2 | -0.05..+0.11 / -0.29..+0.38 | -0.12..+0.07 / -0.10..-0.01 | -0.12..+0.15 / -0.06..+0.17 | +0.08..+0.78 / -0.12..+0.66 |
| A = GSD+INIT+PAR(3)+NW0, copies p1 | +0.25 -0.12 -0.11 -0.08 | -0.02 +0.10 -0.36 -0.02 | +0.31 +0.22 +0.24 +0.18 (4/4) | |
| A p2 | +0.28 +0.07 +0.23 +0.45 | +0.00 +0.08 -0.15 -0.06 | +0.30 +0.10 +0.44 +0.07 (2/4) | |
| B = A + EIDS p1 | +0.03 +0.23 +0.11 +0.06 | -0.25 -0.27 -0.24 -0.53 | +0.22 +0.02 +0.18 +0.26 (3/4) | |
| B p2 | -0.16 +0.17 +0.10 +0.50 | -0.49 -0.22 -0.28 -0.36 | +0.16 +0.18 +0.15 +0.26 (2/4) | |
(at bs32 A and B are the same kernel = GSD alone: 16 copy-passes, mean +0.19%, 11/16 above every control.)
Correctness (B): hang1 2-33 ok, check_mt M 2-9 12 15 16 17 20 24 31 32 33 normal + skewed 0 failures, wps2_mix 160 calls 0,
wpmix_calls 120 calls 0.

## Run 2 (kda_ws/runs/wpfast_run2.log): does anything move bs8 / bs16 DATA landing? (stamps, cold, medians, us)
A = GSD + INIT + PAR(3) + NW0; X = A + E4=4 (the early path issues stages 2-3 too); Y = X + CDL=400; ACDL = A + CDL=400.

| bs8 | base (run 1) | A | X | Y | ACDL |
|---|---|---|---|---|---|
| init barrier / tables | 0.91 / 2.80 | 0.41 / 2.04 | 0.41 / 2.01 | 0.42 / 2.50 | 0.43 / 2.53 |
| consumers at the first wait | 4.99 | 3.56 | 3.56 | 3.55 | 3.77 |
| producer role / stage-0 copy | 1.53 / 2.49 | 1.78 / 2.65 | 1.85 / 2.75 | 1.62 / 2.57 | 1.58 / 2.63 |
| stage 0 landed / first stage passed | 4.30 / 5.09 | 4.47 / 4.39 | 4.54 / 4.44 | 4.44 / 4.34 | 4.54 / 4.44 |
| stage 2 issued / landed | 5.95 / 7.72 | 6.51 / 8.34 | (early) / 7.31 | (early) / 7.52 | 6.58 / 8.35 |
| rtd seen by the producer | 3.67 | 4.30 | 6.47 | 6.27 | 4.47 |
| first refill (stage 4) issued / landed | 8.20 / 10.14 | 8.34 / 10.85 | 9.05 / 10.92 | 9.18 / - | 8.49 / 10.84 |
bs16: A first copy 3.28 (base 2.86), landed 5.03 (4.61): later; X stage 2/3 land 7.47 / 8.25 but the first refill is issued
at 9.17 (A 8.71); ACDL first copy 2.97, landed 4.90.
- The consumers' first wait moves 1.4-1.7 us earlier at bs8 (NW0 + INIT + GSD + PAR), the first stage is passed 0.7 us
  earlier (4.39 vs 5.09), but stage 0 lands no earlier (4.30-4.54 in every variant): the producer's first copy is not
  advanced by an earlier init barrier (its cold code fetch then overlaps the consumers' routing fetch; CDL (consumers
  0.4 us late) gives the producer back only 0.1-0.2 us).
- E4 (stages 2-3 from the early path) lands them 1.0 us earlier but only moves the producer's first cold pass through its
  loop code later (rtd seen 6.47 instead of 4.30, first refill 9.05 instead of 8.34): the first execution of the producer
  loop under the initial burst costs ~2.3-2.9 us wherever it sits. E4 / CDL also re-allocate the stage loops (not exact).
  Dropped. (A dry pass of the producer loop before the first copies is the untested lever left; not built per the
  planner's stop rule.)
- Correctness (Y): hang1 2-33 ok, check_mt M 2-10 12 14-17 32 normal + skewed 0 failures, wps2_mix 160 calls 0.
- The run-2 bench2 (16 variants incl. bs128) failed with CUDA OOM at the flush allocation (both passes): no A/B from run 2.

## Run 3 (kda_ws/runs/wpfast_run3.log): confirmation of the baked candidates (bench JSON dev/remote_out/wpf3{A,B}_p{1,2}.json)
Files (dev/bake.py on the patch outputs):
- dev/moe_v4110.py = dev/moe_v4101.py + MOE_FT_GSD,MOE_FT_INIT,MOE_FT_PAR,MOE_FT_PARM=1,MOE_FT_NW0,MOE_FT_CFG=5 (bs8 set + bs32
  GSD); line A twin dev/moe_v4160.py (= dev/moe_v4151.py + the same defines).
- dev/moe_v4111.py = dev/moe_v4101.py + MOE_FT_GSD,MOE_FT_CFG=4 (bs32 GSD only); line A twin dev/moe_v4161.py.
SASS (local NVRTC 13): only moe_4_1_1_4_1 (v4110: 24f5a6f1954a = run 1's A bs8 kernel) and moe_8_4_1_4_1 (b2b01079ea0e line B
= run 1's GSD kernel; 88a44a2a9592 line A) differ from the bases; every stage loop and the producer steady loop
byte-identical (control bits included), no spills; v4111 / v4161 change moe_8_4_1_4_1 only.
Correctness (v4110 and v4160; they hold every kernel of v4111 / v4161): hang1 bs 1 2 4 8 9 16 17 24 32 33 64 65 128 256 ok,
check_mt M 1..64 normal + skewed x 3 seeds x 2 calls 0 failures, wps2_mix 200 calls 0 (worst 3.32e-3), wpmix_calls 160
calls 0 (3.29e-3), both lines.
bench2 jm 100 x 2 passes, base file x 4 interleaved with candidate x 4, 2 base fillers at each end (% vs the mean of the 4 middle
base controls; * = copies above every control):

| | bs8 | bs16 (placebo, identical kernel) | bs32 |
|---|---|---|---|
| line B controls p1 / p2 | -0.54..+0.23 / -0.29..+0.14 | -0.08..+0.07 / -0.13..+0.08 | +0.04..+0.15 / -0.07..+0.12 |
| line B v4110 p1 | +0.47 (+0.40 +0.61 +0.86 +0.30, 4/4*) | +0.08 (1/4*) | +0.17 (+0.32 +0.12 +0.13 +0.21, 2/4*) |
| line B v4110 p2 | +0.41 (+0.30 +0.68 +0.88 +0.37, 4/4*) | -0.00 (1/4*) | +0.17 (+0.27 +0.22 +0.13 +0.32, 4/4*) |
| line A controls p1 / p2 | -0.33..+0.20 / -0.23..+0.04 | -0.10..+0.13 / -0.15..+0.14 | -0.10..+0.26 / -0.12..+0.16 |
| line A v4160 p1 | -0.23 (0/4*) | -0.00 | +0.24 (+0.38 +0.28 +0.25 +0.18, 2/4*) |
| line A v4160 p2 | -0.21 (0/4*) | -0.00 | +0.23 (+0.44 +0.14 +0.22 +0.30, 3/4*) |

## Summary (WP-FAST)
BANKED: bs32 MOE_FT_GSD (the global scales' x 256 moved from the entry, where warps 0-1 stalled on the cold load before the
init barrier, to the smem store after the bitmaps): line B dev/moe_v4111.py (v1514 + patch + MOE_FT_GSD,MOE_FT_CFG=4),
line A dev/moe_v4161.py (v3960 + patch + same defines). Only moe_8_4_1_4_1 changes, all its loops byte-identical.
bs32: line B +0.19% (run 1, 16 copy-passes, 11 above every control) and +0.17% (run 3, 8 copy-passes, 6 above every
control); line A +0.24 / +0.23% (run 3, 5/8). Geomean ~+0.03%. Mechanism (stamps): init barrier 0.62 -> 0.33 us, bitmaps
1.28 -> 1.12, FST first copy 2.96 -> 2.82, first stage passed 4.78 -> 4.62 (FST CTAs), 9.03 -> 8.93 (wgmma-first CTAs).
NOT BANKED (candidate for the planner): bs8 set (v4110's moe_4_1_1_4_1: GSD + INIT + PAR + NW0, loop-exact): line B +0.47 /
+0.41% in run 3 (8/8 above every control) but +0.08 / +0.28% in run 1 (2/8; same SASS), mean ~+0.31% over 16 copy-passes;
line A twin -0.23 / -0.21% (0/8): fails the two-job rule and does not port (line A has no LATE9 / S5R: the earlier start
lets warps 9-11's cold role code compete with the producer's). Mechanism: consumers' first wait 4.99 -> 3.56 us (dead NW1S
copy gone, init barrier 0.91 -> 0.41), first stage passed 5.09 -> 4.39, stage 0 landing unchanged (4.3-4.5).
Rejected: EIDS (bs16 -0.28%, renames line A's bs16 loops), PAR at bs32 (no loop-exact form: renamed / 0.99), E4 / CDL (move the
producer's cold first loop pass, not exact), TRP (renames the producer loop), bs16 anything (A -0.04 / -0.05%: bs16 is
gated by the producer's first copy, which an earlier init barrier makes later, 2.86 -> 3.28 us).
Open lever: the producer's FIRST pass through its steady-loop code runs under the initial 2-stage burst and takes 2.3-2.9 us
(rtd -> stage 2 at bs8 / 16), so the ring holds only 2 stages until ~6-7 us and the first refill (stage 4) is issued ~8.2
us. A dry pass of that code before the first copies (or a loop-exact form that issues stages 2-3 from the warm early
path without moving the cold pass) is the untested lever; not built (planner's stop rule).

## Run 4-5 (planner follow-up): MOE_FT_DRY, a dry pass of the early producer's steady-loop code -- STOPPED at the stamp step
Patch dev/wpfast_mk4120.py BASE OUT on top of a mk4100 output (both lines): dev/moe_v4121.py (from dev/moe_v4111.py, line
B), dev/moe_v4171.py (from dev/moe_v4161.py, line A); dev/moe_v4120.py / v4170.py = the first version (dry lane value
0x7fffffff: `lane + 32 < n` overflowed to true, so the dry pass issued garbage x-row copies: illegal address, run 4).
MOE_FT_DRY=1: before its early path, warp 8 runs one item of the real producer loop (item 0 of a fake 1-tile queue, 8
stages) with every lane test of the loop reading ft_ln = 0x3fffffff (no claim, slot wait, record, expect_tx, copy or
pair-list wait), then jumps back (goto) and runs the early path + loop for real. MOE_FT_DRY=2 = twin: the flag is
((u32)M >> 31) ^ imm, so the SASS is the same as DRY=1 except one immediate (checked: all loops identical between them).
Default builds = the bases' SASS in all 16 kernels. But both DRY builds re-schedule the bs8 / bs16 consumer stage loops
(similarity 0.54-0.57 / 0.45-0.47) and add 1-3 spill words: any form tried (all lane reads or only the predicates through
ft_ln) changes the kernel's allocation.
Stamps (run 5, cold, 8 jm seeds x 2 reps; "role" is stamped again at the real start, i.e. after the dry pass):

| bs8 | base | DRY=2 (twin) | DRY=1 |
|---|---|---|---|
| producer real start / stage-0 copy | 1.70 / 2.68 | 1.61 / 2.58 | 4.94 / 5.74 |
| rtd seen -> stage-2 copy issued | 3.58 -> 6.19 (2.61) | 3.13 -> 6.05 (2.92) | 6.04 -> 6.58 (0.54) |
| stage 0 landed / first stage passed | 4.48 / 5.28 | 4.39 / 5.13 | 7.81 / 8.71 |
| first refill (stage 4) issued | 8.49 | 8.09 | 11.70 |
bs16: twin stage-0 copy 2.84 / stage 2 6.02 / refill 7.96; DRY=1 5.91 / 6.75 / 11.71, first stage passed 7.80 (twin 4.54).
- The mechanism is confirmed: with the loop code fetched beforehand, rtd -> stage-2 issue drops from 2.6-2.9 us to 0.54 us.
  The 2.3-2.9 us gap is the producer loop's cold instruction fetch under the burst.
- But the dry pass itself takes 3.3 us at the start (the producer's real start moves 1.6 -> 4.9 us), so every absolute time
  moves later (stage 2 issued 6.58 vs 6.05, first refill 11.70 vs 8.09, first data +3.4 us). The cold fetch of that code
  costs about as much before the burst as under it, and it then sits in front of the first copy.
- The twin (code presence only) is within the probe's noise of the base at both shapes.
- Correctness: DRY=1 (v4121) is NOT clean: an illegal instruction at bs4 in hang1 and in check_mt / wps2_mix (the stamped
  build passed bs 2 / 8 / 16). Not debugged (stopped per the planner's rule: stage-2 / first refill did not move earlier).
Not benched. Untested variants: a 1-stage dry item (shorter pass), or the dry pass on the idle signaler warp 9 in parallel
with the producer's early path. WP-COLD's rule (cold misses of different code by different warps serialize through one
per-SM miss path) predicts the second would slow the producer's own early-path fetch.
