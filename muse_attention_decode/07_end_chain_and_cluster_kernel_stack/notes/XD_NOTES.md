# XD (round 2): verify full B > 1 (v16f / v32f) — new architecture

Owner XD, dir dev/xd. Scope (coordinator 2026-10-28): v16f + v32f only (decode full moved to XF). Targets are minimums:
v16f <= 88, v32f <= 168 local; aim past them.

Files
- xdcore.cu   helpers + the q_len-8 tile (copy of k34's W9H variant-A FP16 tile, `xd_tile`, hooks pre() / pvf())
- mb.py       tile-phase microbenchmark (equal flat token slices, real tile compute, no start / new / end chain)
- lc.py       local NVRTC 12.4 compile: ptxas regs/spills + loop SASS summary (dev/w9/sassloop.py)
- job_prof0.sh  profile job (nsys v16f/v32f, image SASS of attn_8_0_1_0, ncu v32f per line)

## 0. Profile of k34 (job xd_prof0, out/20261028T132112_xd_prof0)
nsys (40 reps, judge-visible kernel duration): v16f med 99.71 (p10 98.59, p90 101.95), v32f 182.66 (180.42 / 185.09);
flush -> kernel gap 4.1 / 4.3 us (common mode).
Image NVRTC 13 attn_8_0_1_0: 128 regs, 16-B stack, spill 4/8 B (outside the loop); loop 362 SASS, 4 QGMMA, PV waited
in its own iteration (DEPBAR before the back edge), 0 in-loop LDL/STL.
ncu v32f (clock-control none): 193.5 us, DRAM 73.1 %, issue slots busy 55.3 %, eligible warps / scheduler 1.22 of 4.03.
Stalls: not_selected 16.3, wait 14.9, selected 14.4, barrier 13.8, short_scoreboard 10.5, math_pipe_throttle 10.0,
long_scoreboard 8.0, dispatch 7.1 %. Top lines: K/V cp16p issue 8.6 % (barrier / long_sb), PV wg_wait 6.5 % (barrier =
the pre-PV WG barrier), O fold 6.5 %, prmt (V^T + P pack) 4.8 %, f16 cvt 4.0 %, f16 max 3.8 %, row-max shuffles 5.4 %.

## 1. Budget per 64-token stage per SM (v32f; one stage = 64 tokens x both heads = 16 KB = 2 warpgroup-tiles)
- Memory: 16 KB at the 4.3 TB/s gather ceiling = 32.6 GB/s/SM -> 503 ns = 882 cycles @ 1.755 GHz (995 @ 1.98).
- Issue: 272 warp-instr per warp-tile (WI census; 179 tile fn + 67 copy/page-table issue + 26 control) x 8 warps =
  2176 per stage / 4 schedulers = 544 issue cycles = 62 % of the memory time (ncu: 55 % busy).
- Tensor: QK 2 x m64n64k32 + PV 2 x m64n72k32 per warpgroup-tile = 1.11 M MAC per stage / 4096 MAC/clk = 272 cycles (31 %).
- Latency: each warpgroup runs a serial chain per tile (QK issue -> wait -> V^T -> f16 softmax -> WG barrier -> PV ->
  wait -> fold). W9's resident harness: 1402 cycles per lane-tile with 2 lanes interleaved (FP16 tile), i.e. the chain
  is ~79 % of the 1764-cycle period a lane has per tile (2 lanes share the SM's 882 cycles per stage). Little slack:
  any jitter between the two warpgroups of a lane (lockstep: one 256-thread barrier per tile at the loop top) or
  between compute and the memory stream shows up as time.
- k34's structure: refill of slot it-1 issued at the loop top only after BOTH warpgroups finished tile it-1 AND
  tile it landed AND the lane barrier passed -> ~3 stages (DST-1) in flight per lane = 96 KB per SM (6 KB / warp; WF:
  the gather wants >= 8 KB / warp); decode run1f keeps 4 lanes x 3 x 16 KB.

## 2. Architecture candidates (first principles)
- 1 WG per lane holding both heads: O 2 x 36 + t 36 + S 32 + Q 16 > 128 regs. Dead (registers), any variant.
- Swap-AB (tokens as M): row max over tokens spans warps (vote / smem): worse softmax. Dead.
- XD-M (built first): keep 2 WGs (one per head) sharing full-row 16-KB stages, but remove the lockstep:
  per-slot FULL mbarriers (cp.async.mbarrier.arrive.noinc, consumer-side proxy fence), no lane barrier, and each
  16-token quarter of a slot is refilled by whichever of the two same-index warps (head 0 / head 1) finishes the tile
  LAST (shared atomic parity per (slot, quarter)): no warp waits for the other head except when the ring runs dry
  (slack = DST - lead tiles), refill issued as soon as the slot is free (mid/end of tile), full 128-B rows only.
- X4 (fallback): cluster of 2, one KV head per CTA, 4 independent 1-WG lanes, the fetching CTA forwards the other
  head's halves over DSMEM. Same slack structure as XD-M (cross-SM), more machinery.

## 3. Microbenchmark (mb.py): see results below
### mb round 1 (job xd_mb1, 20 reps, cold L2, graph replay minus empty graph; flat token stream, 132 CTAs)
names attn_xdmb_MODE_DST_ISS_VAR (VAR 1 = no compute). B=32 (2.51 M tokens, 643 MB) | B=16 (1.29 M, 332 MB), median us:
  mode 0 (k34 lockstep) DST4: 170.22 | 90.80;  no-compute 150.29 | 78.59;  DST3 172.35 | 91.06
  XD-M DST4 ISS0 176.69 | 93.44 (no-compute 149.28 | 78.13); DST5 178.35 | 94.91; DST6 179.62 | 96.06;
  XD-M ISS1 (refill at the pre-PV barrier) DST4 176.29 | 93.71; ISS2 (PV window) 181.84 | 96.37 (16-B spill)
  k34 full kernel (same data) 177.66 | 99.36
Readings:
- The lockstep SKELETON is at the gather ceiling in the microbench (150.3 us = 4.28 TB/s; probe floor 152): the k34
  no-compute gap (166-169) is start / segment / end structure, not the loop. The loop's whole cost is COMPUTE
  coupling: +20 us (b32) / +12 (b16) over the skeleton.
- XD-M loses 6.5 / 2.6 us. Cause (SASS + reasoning): positive feedback in "last finisher issues": the warpgroup that
  is behind gets the issue work (~95 SASS per refill), stays behind, and ends up issuing every refill; the leading
  warpgroup runs into the empty ring. Decoupling does not pay when the issuer assignment follows the lag.
- DST 5/6 (and DST 3) results are confounded: `% DST` / `/ DST` on non-powers of two compile to division sequences
  in the loop (also true of k34's `it % DST`: WB's "DST 5 slower" may be partly this). Round 2 uses running counters.
### mb round 2 (job xd_mb2, logs/mb2.log): running slot counters, ablations, SM clock (clock64 / globaltimer, thread 0)
B=32 | B=16 median us; [SM clock MHz from the first (eager) launch]
  mode 0 DST4 170.02 | 90.85 [1656]   no-compute 156.70 | 81.79 [1931]   RESIDENT (no K/V copies) 143.98 | 76.19 [1806]
  mode 0 no PV wgmma 162.69 | 86.62 [1650]   no softmax (P from S bits) 160.99 | 85.97 [1603]
  mode 0 DST3 171.94, DST5 172.02, DST6 173.28 (no-compute DST6 158.00) | b16 90.85 / 92.14 / 93.66
  XD-M DST4 181.39 (ISS1 177.55), DST5 181.70, DST6 182.83 | b16 96.18 / 94.32 / 97.33 / 98.42;  k34 full 182.43 | 101.14
KEY FINDINGS
- The verify-full tile phase is COMPUTE-BOUND, not a coupling problem: compute alone (resident, no gather) takes 144 us
  for b32 (76 for b16) at 1806 MHz, the gather alone 150-157 us, together 170 (b16: 76 / 79-82 / 91).
- POWER: the SM clock drops to ~1650 MHz when tile compute and the HBM stream run together (1931 MHz gather only,
  1806 compute only). So compute at the real clock is ~157 us > the gather's 150: every instruction removed buys
  issue time AND clock. Decode full (light compute) does not pay this.
- More stages do not help (DST 5/6 slower even with counters): bytes in flight are not the limiter.
- XD-M is worse (issue concentration on the lagging warpgroup, see round 1).
=> The lever is the per-tile instruction / latency cost (and power), not the skeleton or decoupling.
### mb round 3 (job xd_mb3, logs/mb3.log): lean issue path (MODE 2) and f16-accumulated QK (VAR 5 = W9H variant B)
MODE 2 = k34 lockstep schedule with a lean K/V issue: thread tokens tq0 + 16 k all have t & 7 = warp, so the swizzled
destinations are base + 2048 k (immediates); page-table entries = 17 aligned 16-B chunks per tile copied by ONE warp
(lanes 0..16) into a ring slot; each thread reads its 4 entries with 4 LDS. ISS: 0 if-converted full/partial paths,
1 branch-free (zero-fill + row select per copy), 2 full-tile path only (timing bound).
B=32 | B=16 (us):
  mode 0 A 169.71 | 90.40 (resident 144.91 | 76.88)      mode 0 B 174.42 | 92.64
  MODE 2 ISS0 171.14 | 91.04  ISS1 168.16 | 88.86  ISS2 165.23 | 87.55  (resident ISS0 133.70 | 71.38, no-compute 156.48)
  MODE 2 B 173.54 | 91.42;  DST5 172.19, DST6 171.68;  k34 full 180.27 | 101.47
- Lean issue: resident compute -11 us (145 -> 134), end to end only -1.5 (branch-free) .. -4.5 (bound).
- Variant B loses again (+4.7 / +5.4), BUT per-CTA times say why it is interesting: B's median CTA finishes at
  163.9 us vs A's 178.4 (-14.5) while its slowest CTA takes 181.4 (A 182.6): a few straggler CTAs. Round 4 looks at them.
### mb round 4 (job xd_mb4, logs/mb4.log, B=32 only; a FASTER container: k34 full 175.98, clocks ~1750 MHz)
  MODE 2 ISS2 A 160.11 (CTA p10/50/90/max 155.0/155.9/157.0/158.7)   B 167.39 (150.9/157.6/163.4/165.4)
  resident A 135.66 (131.4/132.1/133.1/133.8)   resident B, zero data 131.92 (126.0/129.3/130.2/130.8),
  resident B, nonzero data 132.51 (123.2/127.6/130.4/131.1)
- Variant B saves only ~3 % of the resident compute, and its per-CTA time spreads ~3x wider than A's (even resident,
  even with identical smem data in every CTA; slow CTAs are on different SMs each run): the kernel waits for the
  slowest CTA, so B loses end to end. Mechanism not isolated (phase-locking of the 4 warpgroups?). B dropped.
- Container-to-container variance is large (same code: 165.2 vs 160.1, clocks 1636 vs 1751 MHz): only same-job A/Bs.
### mb round 5 (job xd_mb5, logs/mb5.log): 128-token step (MODE 3: two 64-token stages per iteration, DST 5 slots,
one max / alpha / fold per 128 tokens, PV = 4 chained k-steps). B=32 | B=16 (us):
  T64 lean (2_4_2) 166.72 | 88.27   resident 133.68 | 70.48
  T128 QV0 (f32 QK, serial halves) 165.38 | 89.10   resident 117.86 | 63.36   (8-B stack, 16-B spill outside the loop)
  T128 QV1 (f32 QK, both in flight) 216.14 (spills 56 B, in-loop LDL/STL 9)
  T128 QV2 (f16 QK, both in flight) 164.19 | 88.10   resident 109.65 | 59.04
  T128 no-compute skeleton 158.61 | 82.77 [1979 MHz];  T64 DST4 2_4_2 at 1713 MHz, T128 QV2 1748 MHz;  k34 full 180.93 | 100.82
- T128 cuts the resident compute by 12 % (QV0) / 18 % (QV2), but end to end only -1.3 / -2.5 us: with compute at
  ~70 % of the memory time the kernel still sits 5.6 us above its no-compute skeleton (CTA medians +4.2 us).
- Hypothesis under test (job xd_pw1): the HBM stream + SM work hit the power cap, the SM clock drops (1979 -> ~1750),
  and the gather ceiling itself depends on the SM/L2 clock.
### Power job (xd_pw2, logs/pw2.log): back-to-back replays (3 s soak each), nvidia-smi 50 ms samples
power limit 700 W (enforced). Mean power / SM clock / throttle reason:
  T64 lean (2_4_2_0)       685.6 W  1534 MHz  SW POWER CAP (0x4)      k34 full   685.3 W  1583 MHz  SW POWER CAP
  T128 f16 (3_5_2_0)       685.7 W  1390 MHz  SW POWER CAP            no-compute skeleton (3_5_0_1)  656.5 W  1980 MHz  none
  resident T64 / T128      399.9 / 396.3 W  1980 MHz  none
=> CONFIRMED: the HBM gather alone draws ~650 W; tile compute adds ~320 W; together they exceed the 700 W cap and the
   SM clock drops (1390-1580 MHz soak; ~1650-1750 MHz in the judge-like flush+replay timing). Verify full is
   POWER-bound: time ~ total energy. Every instruction / smem byte / tensor op removed buys clock. This is the
   "compute coupling" W9 could not explain (and why decode full, with ~1/3 of the SM work, runs at its gather time).

## 4. Real kernel attn_xd_v8f (dev/xd/xdk.cu + xdcore.cu; builder dev/xd/build.py; FROZEN in dev/xd/deliver_r1/)
NEW kernel (not a patch of run8): replaces k34's run8 FLAT path for q_len 8, full layer, page size 1, tp8, 1 < B <= 128
(everything else falls back to the base kernel_fn). Structure:
- prefix / FLAT split / segments / lane merge / publish / combine = compact copy of k34 run8 FLAT (WF end chain,
  NEWW-weighted split, <= 2 partials per CTA, self-resetting counters in XD's own buffers);
- tile loop = MODE 3 of the microbench: T128 pairs (xd_tile2, QV 2 = f16-accumulated QK by default), DST 5 slots of
  16 KB, lean issue (stride-8 tokens -> immediate swizzled destinations; page-table chunks by one warp per tile;
  zero-fill predicate past the segment end), lockstep refill at the pair boundary;
- new block = one causal 64-token xd_tile step of lane (ntp & 1) after its pairs (slot 0; V rows 8..63 zeroed);
  a missing 2nd tile of the last pair re-uses stage 0 fully masked (no ring zeroing anywhere);
- gates NOT in smem (to fit DST 5 + 4 V^T blocks): L2 evict_last prefetch of the request's gate rows at segment start,
  loaded at the START of each combine item (in flight with the partial loads);
- reductions inside the segment loop (Q RMS, new-k RMS, complete-segment output RMS) go through smem + __syncwarp
  (W9: shuffles there serialize the wgmmas, C7515); 128 regs, 16-B stack (outside the loop).
Bugs found on the way: red8 float4 misalignment (pre[257] long long before it); NaN from FP32 merge data left in the
ring being read as V of a masked half (fixed by stage-0 reuse + zeroing only the new tile's V rows).
Correctness: dev/xd/xcheck.py (12 edge cases inside the predicate, incl. complete segments, B 128 ctx 300 mixed,
ctx 1, kv_scale 1/64, dist edges; each called twice; asserts the XD path ran): 0 failures; tools/check.py: 0 failures
(logs/b3.log, defaults QV 2 / NEWW 1536). v16f err 2.77e-2 / 2.75e-2 (bound 0.082), v32f 2.68e-2 / 2.80e-2 (bound 0.080),
same as k34 (FP8-query error dominates).
Bench (tools/bench.py, 20 reps, two k34 controls in the same job; us):
  b2 (slow container, defaults then QV0/NEWW1024): v16f k34 101.37 / 101.37 | xd QV0 99.98 | QV2 98.97 | QV2 NEWW768 99.66 |
     QV2 NEWW1536 97.89;  v32f k34 186.04 / 186.04 | QV0 181.58 | QV2 180.33 | 768 181.03 | 1536 180.15
  b3 (defaults QV2 NEWW1536): v16f k34 99.74 / 99.78 | xd 96.79 | NEWW2048 96.89 | NEWW1280 97.29 | QV0 98.29;
     v32f k34 182.37 / 182.19 | xd 179.78 | 2048 180.64 | 1280 179.44 | QV0 181.30
  b1 (FIRST version: ring zeroed at start + after every merge, combine gate load after the reduction): no gain (= k34).
nsys (job xd_n1, 40 reps, two passes): v16f k34 100.02 / 100.78 -> xd 97.28 / 97.46; v32f k34 184.32 / 184.86 -> xd
180.05 / 179.81 (-2.9 / -4.7 us).
Trace (traced builds, v32f, logs/t1.log): xd start: prefix 0.72, pt chunks issued 2.38 (!), back 3.67, K/V prologue issued
4.87, tile0 5.97; tiles-done p10/50/max 169.9/171.6/175.5; end chain (tiles max -> combined max) ~7.5 us.

## 5. Next levers (resumable)
- Start: 1.7 us between the prefix and the first page-table chunk issue (64-bit divisions c*NU/G, segment math, cold
  code) + 1.3 us cold page-table DRAM trip: whole-table L2 prefetch at entry (ruling (c)), 32-bit split math.
- End chain ~7.5 us (decode ~4): XC now owns start/segment/end; XD's kernel is separate, so XC's run8 patches need porting.
- Tile energy: the f16 softmax core (~90 instr / 64 tokens / warp) and the fold; T128 last-pair waste (a lane with an odd
  tile count computes a fully masked half); per-row causal mask in xd_tile2 would let the new block ride in that half.
- XD-M (decoupled warpgroups) and DST > 5 are measured dead ends (sections 3 rounds 1-2).

## 6. Round r2 (tile energy; XC owns start / segment / end of xdk.cu from deliver_r1)
Interface kept: xd_tile2 / xd_tile in xdcore.cu are byte-identical to deliver_r1. Experiments use xd_tile2x (same math
+ ABL ablation bits, a pre-PV hook with a configurable barrier, per-row causal masks dh0/dh1). Note: deliver_r1 itself
compiles to 128 regs, 24-B stack, 36/28 B spills on local NVRTC 12.4, all OUTSIDE the pair loop (0 in-loop LDL/STL).
### mb6 (logs/mb6.log): T128 QV2 tile ablations, b32 | b16 (us; combined = gather + compute)
  full 163.50 | 87.57, no-compute skeleton 158.03 | 82.38, resident 110.67 | 59.63
  single cuts (combined): no fold 163.36, no clamp 163.66, no V^T transpose 162.82, no row-max tree 163.01,
  PV N=64 (no ones columns) 163.68, no per-row max/alpha/bias math 163.63  -> each <= 0.7 us (noise)
  ALL six cuts: combined 161.22 | 85.15 while the resident compute halves (110.7 -> 58.2 | 59.6 -> 32.6)
  => leverage ~0.045 us of kernel time per us of resident compute: CUDA-core work is ~2.3 of the ~5.5 us coupling;
     the remaining ~3.2 us exist with nothing but QK + PV wgmma + barriers. The tile-energy ceiling is ~1-2 us.
### mb7 (logs/mb7.log, fast container): skeletons are at the gather ceiling
  probe-like per-warp rings (16 warps x 3 or 2 stages, no barriers) 149.79 / 150.05 | 77.97 / 77.82;
  T128 lockstep skeleton 150.74 | 79.09; T64 lean skeleton 150.02 | 78.48; refill-before-wait (2 lane barriers per
  pair) skeleton 150.03 but with compute 161.18 (vs 157.10) | 86.24 (vs 84.24). k34 full 176.26 | 98.59.
### clock locking (logs/clk.log): nvidia-smi -lgc is not permitted in the Modal container (no clock control).
  Indirect: the gather alone at 1980 MHz is 150.96; with compute at 1735 MHz the kernel is 157.06, far below the
  172 a clock-proportional gather would give -> the gather is not strongly SM-clock bound.
### mb8 (logs/mb8.log): mid-pair refill (refill the pair's own slots after a 256-thread pre-PV barrier)
  165.49 vs 163.47 (resident 119.95 vs 110.42: the lane-wide mid-pair barrier lengthens the chain). DST 4: 163.97. Lost.
### Real kernel r2: new block + phantom halves FIRST (xdk_r2_newfirst.cu, build_r2_newfirst.py; xcheck 0 failures)
Each lane runs nv = my_n + voff virtual tiles in pairs; the first voff are the new block (owner lane, causal, from a
16-KB newt stage) and / or a fully masked phantom, so no pair at the END is half empty and the new block never runs
after the stream. bench b4 (fast container, 20 reps): v16f k34 97.84 | r1 94.05 | r1 copy 94.32 | r2 94.96 |
r2 NEWW1024 95.36 | NEWW768 96.53;  v32f k34 178.42 | r1 173.16 | r1 copy 174.43 | r2 173.68 | 1024 173.25 | 768 174.22.
=> neutral (r1 vs its own copy differs by 0.3 / 1.3 us). Not adopted; xdk.cu = deliver_r1.
### Where the rest of verify full's gap is (v32f, xd vs decode-like ~158)
  ~3 us structural coupling under the 700 W cap (QK/PV + barriers; not CUDA-core work, not the skeleton),
  ~1-2 us further CUDA-core cuts at most (ablation ceiling),
  start ~5-6 us to tile0 and end chain ~7.5 us (XC), CTA spread p50 -> max ~4 us with EQUAL work per CTA (microbench
  eager stats too): random per-SM speed under the cap -> only dynamic tail balancing (work stealing / tail chunk
  queue with a static partial count) can recover it.

## 7. Round r3: the START (owner XD; XC owns the end chain). FROZEN in dev/xd/deliver_r2/ (#if XD_START, default 1)
Changes vs deliver_r1 (all under XD_START; XD_START=0 = deliver_r1's start):
- split: 32-bit exact c0 / c1 when NU <= 2^32 / (G + 1) (64-bit fallback), b_begin by ballots over pre[] (no serial scan);
- the K/V prologue (tiles 0 .. DST-3) takes its page-table rows by DIRECT loads issued first after the split (no wait for
  the page-table chunks, no CTA barrier before the burst); the refill chunks (tiles 0 .. PD-1) ride in tile 0's group;
- the new block is computed by the lane that attends it (ntp & 1): without the CTA barrier, lane 0's writes of the new
  rows raced lane 1's read (found by xcheck: call 1 of B=2 ctx 100000 err 0.64; fixed);
- XD_SP=1 (burst before the Q / new-token work) is off: ptxas then sinks the Q loads below the burst (SASS checked).
Page-table L2 prefetch variants (XD_PTP) all LOSE here: 1 = 1/G slice at entry: the length loads arrive 2 us later
(prefix 3.10 vs 0.94); 2 = slice after the prefix: v16f +1.2, v32f +1.6; 3 = own rows after the split: pt back 4.92 vs
3.33; 4 = slice at entry by warps 1..15 with evict_last after warp 0's length loads: v16f +2.7, v32f +0.6..1.5 (prefix
still 1.79); 5 = 4 without evict_last: +3 / +3.4. Default XD_PTP 0.
Trace (v32f, traced build, us from the first CTA start, medians): r1-start (XD_START=0): prefix 0.99, pt chunks issued
2.68, pt back 3.99, K/V issued 5.28, tile0 6.24 | XD_START: prefix 1.08, row loads issued 2.11, K/V burst start 3.25,
issued 4.21, tile0 5.87. (XF's gather probe: first issue 1.54, first land ~4.2.)
check: xcheck --seeds 2 0 failures, tools/check.py 0 failures (logs/s4.log).
bench (20 reps, --no-base, same job; r1 = deliver_r1 build, two copies):
  s4: v16f r1 93.63 / 93.79 -> 92.74 / 92.26;   v32f r1 173.68 / 173.15 -> 172.82 / 172.94   (k34 98.54 / 180.46)
  s5: v16f r1 93.23 -> 92.85 / 93.04;           v32f r1 173.54 -> 172.67 / 172.30
nsys (job xd_n2, 40 reps, 2 passes): v16f r1 96.88 / 97.86 -> 96.75 / 97.12; v32f r1 180.50 / 179.97 -> 179.73 / 179.06;
k34 101.60 / 185.75 (same job).
=> -0.4..-1.2 us (v16f), -0.5..-1.4 us (v32f): small, consistent. What remains of the start: the burst starts at ~3.25 us
(probe 1.54): lengths (DRAM) -> ~380 SASS of split / setup (~1 us, cold code) -> row loads (DRAM ~1 us) -> burst; the
first pair then waits ~2.6 us for its data under the start burst (same as the probe's issue -> land).

## 8. Round r4: cold-code trim + work-distribution weights. FROZEN in dev/xd/deliver_r3/
- Finer start stamps (t3): prefix 1.00 -> c0/c1 1.34 -> b_begin 1.50 -> segment start 1.84 -> row loads issued 2.11
  (each small code region ~0.15-0.35 us: first-execution instruction fetch).
- Trim (host-provided 32-bit split bound p.nu32 instead of a device division; also removes ALL spills of deliver_r2:
  24-B stack, 0 B spilled before OVH, 24/16 B after): nsys neutral (n3: v16f 96.30 / 96.86 -> 96.48 / 96.27, v32f
  179.26 / 178.96 -> 179.41 / 178.83). Kept (spill-free start path). XD_DSPL=1 (double reciprocal split) spills 100 B:
  off.
- Per-class tile-done offsets (t4, traced, vs the CTA median): 2-segment CTAs (they own a new block and cross a
  request boundary) -2.9 us (b32) / -4.0 (b16); 1-segment CTAs owning a new block -9.6 / -11.6; plain CTAs +0.4 / +0.1.
  => NEWW 1536 over-weights the new block itself (~300 units of real cost) and lumps in the boundary restart.
- XD_OVH = virtual units at each request's START (the CTA that crosses into the request pays its restart), with NEWW at
  its end. nsys sweep (o2 / o3, 2 passes each, us):
    default (OVH 0, NEWW 1536)  v16f 96.37 / 96.43  v32f 178.46 / 178.48
    OVH 1024 NEWW 512           95.09 / 95.26 / 94.86 / 95.34      178.54 / 178.61 / 178.58 / 178.50
    OVH 512  NEWW 1024          95.62 / 95.76                      178.42 / 178.75
    OVH 1024 NEWW 256           96.27 / 96.66                      178.91 / 178.88
    OVH 768  NEWW 256           97.50 / 97.62                      179.28 / 178.85
    OVH 896  NEWW 128           97.38 / 97.36                      178.66 / 178.77
    OVH 640  NEWW 384           97.34 / 97.22                      178.74 / 179.15
  (lower total weight is worse for v16f even though the trace says 2-segment CTAs finish early: they also run the
  combine of 2 requests). Default -> OVH 1024 / NEWW 512.
- deliver_r3 vs deliver_r2 (job xd_r3, nsys 2 passes, fast container): v16f 93.55 / 93.52 -> 92.54 / 92.37 (-1.1),
  v32f 172.70 / 172.29 -> 172.37 / 172.61 (0); k34 same job 97.84 / 179.22. xcheck --seeds 2 and tools/check.py:
  0 failures (logs/r3.log).

## 9. Experiment: XD's kernel on verify-1 full (v1f), test build dev/xd/r1base/k34xd_b1.py (deliver_r3, predicate 1 <= B)
xcheck --b1 (7 batch-1 cases incl. ctx 1 / 300 / 200000, kv_scale 1/64, edges): 0 failures (logs/b1a.log).
nsys (40 reps, 2 passes): k34 17.01 | XD default 35.92 / 35.86 | OVH 0 35.70 / 35.74 | OVH 0 NEWW 128 35.90.
(XA's k34xa_r3 lines did not parse: dev/wp/nsys_parse.py only matches kernels named attn*; XA reports ~14.8.)
Trace (traced, OVH 0, medians / max): prefix 0.90, row loads 1.90, burst start 3.18, issued 3.78, tile0 5.75 (6.67),
tiles done 11.90 (12.70), publish 14.21 (14.75), combine start 15.20 -> combined 34.72 (34.98).
=> NOT competitive at B = 1: (1) the WF-style FLAT combine (<= 5 combiners, each 26 rows x 132 partials in rounds of 8
   dependent loads) takes ~20 us; (2) even with a 128-combiner fix the start (tile0 5.75 vs XA ~4.3-4.5: no verified
   length speculation, prefix -> split -> rows chain) and the tile phase (11.9 vs XA's loop end ~7.5-9; only ~2.5 pairs
   per lane at B = 1, so T128 buys little - XA measured XA8_T128 itself) are slower. Nothing to transplant into XA.

## 11. Round 4: verify sliding B <= 16 (run8w cluster path attn_8_1_1_1x6), deliverable dev/xd/deliver_r4/
(Coordinator split: XD owns v16s; XF owns v32s / the global path.) Research builder dev/xd/build_vs.py -> dev/xd/k34vs.py
(every knob XDV_* default off = base SASS); tools: trace_vs.py (per-CTA stamps + SM placement + per-rep slowest CTA),
sass_vs.py (stamp / instruction offsets), occ_vs.py (cluster co-residency), mbs_vs.py (XF's gather/tile microbench with a
CTA-count sweep). All numbers nsys medians, 2 passes, same-job controls (k34 9.95-10.08 v16s, 13.70-13.82 v32s).
- Phase budget of the C6 kernel by removing phases (timing-only, wrong output), v16s: start + gather only (no tile math,
  no combine) 6.02 | + tile math 7.71-7.78 | + stream merge 7.82-7.94 | + DSMEM push + receive wait 9.38-9.49 | + owner
  fold / output 9.84-9.89 (Q8). Without math the last tile lands at 4.8 med / 5.3 max; with math the WGs finish at
  ~6.7 med / 7.7 max: the tile phase is CHAIN-bound (3 serial ~1 us T64 chains per WG in the 6-tile parts), not gather-bound.
- Placement: 16 C6 clusters on 96 distinct SMs (1 CTA / SM), no doubling. Co-resident clusters (occupancy query):
  C4 30, C6 17, C7 15, C8 15 -> C8 / C7 at B = 16 and C4 at B = 32 do not fit (fall back to the global path: 11.8).
- WIN, frozen: XDV_Q8 = cluster push as int8 codes + per-row scale (80 B / row vs 136 B FP16): v16s k34 10.05 / 9.98 ->
  9.82 / 9.79; k48 10.00 / 9.95 -> 9.79 / 9.76 (-0.20). Error unchanged (t_v8w B <= 16 max rel L2 0.0846 vs 0.0844).
  Basis: halving the push bytes (timing-only probe XDV_DIAG) gave -0.5: the DSMEM push is bandwidth-bound (~25-30 GB/s
  per SM; 0.55 us to issue 17 KB of st.async per CTA).
- Dead (nsys): T128 pairs in the streams (+0.1 .. +0.5 even with the combine removed: my xdv_tile2 transplant is slower
  per pair than two T64 chains here); C4 / C7 / C8 clusters (co-residency); equal-token split (XDV_TSPL, +0.2: five parts
  get 3-chain streams); bulk-copy push (XDV_BK, neutral); per-WG pushes without the stream merge (XDV_NM, +0.75: early
  DSMEM traffic slows the other CTAs' tiles, WG-done max +1 us); early cluster wait (XDV_CLW, neutral); TLB-warm
  prefetch (v16s -0.1 .. -0.2, v32s +0.4, and nothing with Q8); whole page-table-row L2 prefetch (XDV_PTPF, +0.4: delays
  the length load itself); gate prefetch into smem by register (GP: the LDG got sunk below the burst), by cp.async
  before (GPA, +1 KB on the start path, +0.25) or after the burst (GPB, +0.3); compact Cody-Waite + MUFU RoPE angles
  (RSC, +0.2); dropping the early CTA barrier / hoisting the length load / late mbarrier inits (neutral; ptxas sinks
  .nc loads below bar.sync - the length LDG sat at instr ~124, now ~94). Code-size probe: +2.3 KB of executed IMADs on the
  end chain cost +0.3 (= their issue cost); +1 KB before the K/V burst costs ~+0.2 (start-path i-fetch is expensive).
- Floor for this decomposition: skeleton 6.0 + one chain after the last landing (~0.9) + a minimal combine (~0.8) ~ 7.7;
  current 9.8. Gap: tile chains beyond the gather (+1.7, of which ~0.8 is the extra chains of the 6-tile parts) and the
  push + receive (+1.5). Target 8.0 needs BOTH a cheaper tile chain (or fewer chains per WG) and a ~1 us cheaper combine.
- Coordinator: XDV_Q8 REJECTED (8-bit partials are coarser than SGLang's FP32 partials; FP16 is the approved gray area).
  deliver_r4/build_vs.py now defaults XDV_Q8 0 -> nothing to integrate from round 4.

## 12. Round 5: v16s combine = owner PULL (FP16 payload unchanged), deliverable dev/xd/deliver_r5/
- Microbench dev/xd/dsm.py (16 C6 clusters x 512 threads, 128 FP16 rows per CTA, owners fold 22 rows x 6 sources;
  clock64 from a common cluster sync to folded): st.async push med 1.70 / max 2.36 us | owner pull (ld.shared::cluster
  after a release / acquire cluster barrier) 1.38 / 1.64 | pull with per-warp arrive 1.33 / 1.64 | bulk-copy push 1.48 /
  2.24 | L2 scratch + cluster barrier 2.51 / 2.78. DSMEM moves only ~18-30 GB/s per SM either way; pull avoids the push's
  issue stalls.
- ptxas lowers barrier.cluster.arrive.release (and fence.acq_rel.cluster) to MEMBAR.ALL.GPU (dev/xd/fence_probe.py on
  Modal's NVRTC 13.0); fence.release.sync_restrict::shared::cta.cluster + arrive.relaxed is MEMBAR.ALL.CTA +
  FENCE.VIEW.ASYNC.S only. In the kernel: pull with the full release 9.76-9.89, with the restricted fence 9.54-9.62, an
  unsafe relaxed-only timing probe 9.44-9.54 -> the restricted fence recovers it all. The host passes -DXDV_RESTRICT=1
  when nvrtcVersion() >= 12.8 (else the correct but slower release arrive).
- Also measured: the owner gate load must come AFTER the release (a release waits for the thread's pending loads);
  staged-signal pull (per-source mbarriers, only the 5-tile parts own rows, pull as sources become ready) correct but
  10.2-10.7: each owner then pulls 1.5x the bytes and the per-source rounds serialize DSMEM latency; (m, l) loaded once
  per row + shuffle (PULLML) and plain LDS for the own stage (PULLOWN): neutral each, worse together.
- Phase budget with pull (nsys): merge exit 7.9 | stage + barrier, no loads 8.45 | full 9.54 (DSMEM loads + fold + epilogue
  ~1.1 us: bandwidth-bound, ~18 KB per owner).
- FINAL deliver_r5 (nsys, 2 passes, same job): v16s k34 9.98 / 9.95 -> k34xd_vs 9.50 / 9.54; k48 10.06 / 10.02 -> k48xd_vs
  9.58 / 9.50 (-0.45 .. -0.50); v32s unchanged (13.73-13.79; w4 / w8 SASS byte-identical). check.py 0 failures (both),
  t_v8w 2 seeds 0 failures + WH_ALLB 0 failures (both). Target 9.0 NOT reached: the remaining combine is DSMEM-bandwidth
  bound (~90 KB per cluster at ~20-30 GB/s per SM) and the tile phase is +1.7 us of chains over the 6.0 gather skeleton.

## 13. Round 6: GPU-scope membar audit of k51's cluster kernels (dev/xd/deliver_r6/: membar_audit.py + its Modal output)
- Coordinator: apply the restricted-fence pattern wherever a cluster barrier / fence lowers to MEMBAR.ALL.GPU.
- dev/xd/membar_audit.py compiles every NVRTC module of dev/coord/k51.py with Modal's NVRTC 13.0 (SRC with -DXDV_RESTRICT=1, as
  the host does) and lists MEMBAR.ALL.GPU / MEMBAR.SC.GPU per kernel with their SASS context (1 Modal job).
- Result: NO applicable site. Every cluster kernel (UCGABAR present) has zero GPU-scope membars: attn_1_1_1_1e3 / _1e / _1e6
  (runE1, d32s), all run8w cluster variants (_1x6 now MEMBAR.CTA via deliver_r5), run8p _1p2 / _1p3, WS _1s / _1h, WC _1c,
  RS8 _1q / _1r, XE's xe_* kernels: their end chains already use barrier.cluster.arrive.relaxed + st.async / mbarrier
  complete_tx (no release fence needed). The only MEMBAR.ALL.GPU are in non-cluster global-combine kernels (attn_1_*
  FLAT / _1a / _1w, attn_8_* FLAT, xa_b1f_*, attn_xd_v8f, xb_d1f): a release before a global atomic / strong global
  store that publishes partials in GLOBAL memory to CTAs outside any cluster -> the shared::cta-restricted fence does not
  apply there. Nothing to patch; no nsys job spent.

## 14. Round 7 (standing rule: stack every verified win): re-measure of the unfrozen faster variants on k51
Builder dev/xd/deliver_r7/build_r7.py <k51> <out> [KNOB=v ...] (knobs default 0 = k51; x6 only, w4 SASS byte-identical):
XDV_TLB (TLBS 16), XDV_ST3 (length load first + late tile-mbarrier inits + no early CTA barrier, cluster path only),
XDV_PULLOWN, XDV_PULLML; per-build XD split weights XDR7_OVH_E / _NEWW_E / _OVH_X / _NEWW_X.
Candidates from sections 3-12 that were CORRECT and measured faster or tied, never frozen: v16s TLB warm (r4 -0.1..-0.2),
start trims (r4 tie), pull own-stage LDS / (m, l) broadcast (r5 tie); v32f OVH 512 / NEWW 1024 (r4 sweep, ~tie).
Not rebuilt: the mb6 tile-energy "cuts" (ablations with wrong math, microbench only), every variant measured slower
(T128, TSPL, BK, NM, CLW, GP / GPA / GPB, RSC, PTPF, PTP, DST 4 / 6, r2 new-block-first, other OVH / NEWW points;
OVH 0 is +4.5 us in XC's r8 lineage), Q8 (rejected precision), XD_SP (never timed).
- Screening r7a (nsys 2 passes; k51 9.55 / 9.57, k51 copy 9.55 / 9.62): TLB 9.86 / 9.89 (+0.31) | ST3 9.60 / 9.57 (+0.02)
  | PULLOWN 9.57 / 9.57 (+0.01) | PULLML 9.60 / 9.54 (+0.01) | all four 10.11 / 10.14 (+0.56);
  v32f k51 174.93 / 174.85, copy 175.23 / 175.28, OVH 512 / NEWW 1024 175.06 / 175.09 (tie).
- Together (ST3 + PULLOWN + PULLML, 2 copies x 2 passes, r7n): v16s 9.95 / 10.02 and 10.08 / 10.05 vs k51 9.50 / 9.54 and
  9.68 / 9.62 (+0.45: the pieces interact badly); v32s / v16f / v32f identical within noise (13.71-13.82, 94.27-94.62,
  175.36-175.87; unchanged code). Correctness of the bundle: t_v8w 2 seeds, xcheck, check.py all 0 failures (r7c).
=> Nothing is faster: nothing to stack from round 7 (deliver_r7 = builder with every knob off + the logs).
