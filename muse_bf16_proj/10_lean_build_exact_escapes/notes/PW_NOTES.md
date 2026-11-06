# Agent PW — power-aware m = 256 (clock under the 700 W cap vs operand form / tile shape) (2026-11-02)

Base: claude/t3/bo17L.py (m = 256: qkv = Q3's g256q, o / pooler / unpooler = K's g256k via LN/g256k_lean.py).
Question: compute-bound m = 256 kernels run at ~1.60-1.70 GHz instead of 1.98. Does some operand form / tile shape
sustain a materially (>= 3 %) higher clock (FLOPs per joule), and can pooler / unpooler / qkv / o use it?

## Probe (claude/t3/PW/pwprobe.py)
Synthetic GEMM main loop, 128 CTAs (1 producer WG + NC consumer WGs, setmaxnreg 40 / 232, like K's g256k), every
form performs the pooler's 17.18 GFLOP (256 tokens; per CTA TM x BN tile x nkb k-blocks of 64).
- stream 1: every k-block's x (TM x 64) + W (BN x 64) stage images are bulk-copied into an NS-stage ring (W 67 MB
  from DRAM, pooler-like N(0, 1/sqrt(K)) bf16; x = reference generate_input incl. one-hot probe rows, from L2), with
  full / empty mbarriers = the real traffic of that tile shape.
- stream 0: the ring is filled once with the same data; consumers cycle through it without barriers (tensor + smem).
- Judge regime: graph replay right after a 512 MB read flush (claude/bench.py's method), empty-graph cost subtracted.
  Clock = per CTA (clock64 delta) / (globaltimer delta) over the consumer loop, median over CTAs and reps; profile =
  same ratio between samples every 4 k-blocks (2 us bins; short sample intervals are quantisation-noisy).
- Sustained regime: 0.5 s of back-to-back replays without flush, NVML power / SM clock / throttle reasons.
Forms (TM x BN tile, consumer layout):
| form | tile | wgmma per WG | L2->SM bytes (whole kernel) |
|---|---|---|---|
| f1_ss128x2 | 128 x 128 | 2 WG x SS m64n128 (K's o / unpooler) | 268 MB |
| f2_ss64x2 | 64 x 128 | 2 WG x SS m64n64, A shared (K's pooler LAY1) | 403 MB |
| f2k_ss128ks | 64 x 128 | 2 WG x SS m64n128, WGs alternate k-blocks (in-CTA split-K) | 403 MB |
| f1h_ss128x1 | 64 x 128 | 1 WG x SS m64n128 | 403 MB |
| f3_ss256x2 | 128 x 256 | 2 WG x SS m64n256 | 201 MB |
| f4_ss256x1 | 64 x 256 | 1 WG x SS m64n256 | 335 MB |
| f5_ss128n256 | 64 x 256 | 2 WG x SS m64n128, A shared | 335 MB |
| r1_rs128x2 | 128 x 128 | 2 WG x RS m64n128 (A by ldmatrix.x4 per k16) | 268 MB |
| r3_rs256x2 | 128 x 256 | 2 WG x RS m64n256 | 201 MB |
| r4_rs256x1 | 64 x 256 | 1 WG x RS m64n256 | 335 MB |
All builds: 168 regs (NC 2) / 232 (NC 1), 0 spills, LOCAL 0, 8 HGMMA in the (2x unrolled) loop, no C751x serialisation.

## Probe 1 (box 0, probe1.log): judge regime (25 reps) + sustained (0.5 s)
| form | stream 1: us / loop GHz / busy % | stream 0: us / loop GHz / busy % | sustained s1 GHz (NVML MHz) | sustained s0 GHz (NVML MHz) |
|---|---|---|---|---|
| f1_ss128x2 | 22.11 / 1.624 / 97.3 | 23.10 / 1.617 / 98.3 | 1.331 (1515) | 1.669 (1860) |
| f2_ss64x2 | 23.17 / **1.561** / 96.6 | 22.88 / 1.620 / 97.4 | 1.285 (1425) | 1.619 (1830) |
| f2k_ss128ks | 22.72 / 1.580 / 96.1 | 22.91 / 1.624 / 96.9 | 1.298 (1455) | 1.664 (1860) |
| f1h_ss128x1 | 22.91 / 1.595 / 94.2 | 22.78 / 1.613 / 98.2 | 1.296 (1440) | 1.676 (1860) |
| f3_ss256x2 | 22.27 / 1.628 / 97.5 | 23.36 / 1.630 / 99.3 | 1.363 (1545) | 1.696 (1905) |
| f4_ss256x1 | 22.24 / 1.613 / 95.7 | 22.62 / 1.624 / 98.9 | 1.345 (1485) | 1.710 (1890) |
| f5_ss128n256 | 22.30 / 1.599 / 97.8 | 22.72 / 1.624 / 98.3 | 1.344 (1500) | 1.688 (1875) |
| r1_rs128x2 | 24.42 / 1.617 / 94.7 | 23.07 / 1.623 / 98.0 | 1.337 (1515) | 1.630 (1830) |
| r3_rs256x2 | 22.56 / 1.608 / 97.3 | 23.46 / 1.635 / 98.5 | 1.365 (1530) | 1.700 (1905) |
| r4_rs256x1 | 23.10 / 1.615 / 91.5 | 22.78 / 1.629 / 98.9 | 1.314 (1485) | 1.671 (1875) |
(stream-0 wall times include the one-time ring fill at the start; compare loop GHz.) NVML throttle reason 0x4 (SW power
cap) in every sustained run; NVML power readings (333-556 W) lag (1 s averaging) and are not usable at this time scale.
- Spin kernel (no tensor work) right after the flush: 1.91-1.97 GHz -> the clock at kernel start is the 1.98 boost.
- In-kernel profile (n256 forms, 2.5 us sample intervals): the clock is already ~1.61 GHz at the first sample (t = 3 us)
  and flat until CTAs start finishing, then rises (1.66-1.79 in the last 2-4 us). => the limiter acts within ~1-2 us of
  the start of tensor work (not a slow ms-scale average) and releases within ~1 us when load drops.
- Judge regime, static ring: every operand form / N runs at 1.613-1.635 GHz (spread 1.4 %): SS vs RS, n64 vs n128 vs
  n256, 1 vs 2 WGs make no material difference in the judge regime.
- Judge regime, streaming: the clock falls with the L2->SM byte volume of the tile shape: 201 MB (128x256) 1.608-1.628,
  268 MB (128x128) 1.617-1.624, 335 MB (64x256) 1.599-1.615, 403 MB (64x128 pooler tile) 1.561-1.595.
- Sustained regime (power-capped average): static n256 forms 1.70-1.71 vs n64 1.62 (+5 %: fewer smem operand bytes
  per FLOP); with DRAM streaming everything falls to 1.29-1.37 GHz (n256 / 128x256 highest).

## Probe 2 (probe2.log; no sustained runs in between): mechanism
globaltimer resolution 32 ns (spin kernel, occasional 64). Spin kernel right after the flush: 1.92 GHz.
Interleaved repeats (3 rounds, stream 1, 128 CTAs): f1 1.620-1.627 GHz (busy 97.4-97.7 %, effective = clock x busy
1.577-1.586), f3 1.629-1.633 (1.587-1.590), f5 1.607-1.611 (1.566-1.574), f2k 1.592-1.595 (1.519-1.527),
f2 1.616-1.620 but busy 89 % (effective 1.438-1.443; in probe 1 the same kernel ran 1.561 GHz x 96.6 %: the f2 probe is
placement-sensitive, see probe 4).
| sweep (judge regime) | f1_ss128x2 loop GHz | f3_ss256x2 loop GHz |
|---|---|---|
| 16 CTAs, static | 1.710 | 1.696 |
| 32 CTAs | 1.700 | 1.696 |
| 64 CTAs | 1.666 | 1.694 |
| 96 CTAs | 1.667 | 1.673 |
| 128 CTAs | 1.636 | 1.632 |
| 132 CTAs | 1.616 | 1.651 |
| x = 0 (128 CTAs, static / stream) | 1.700 / 1.682 | 1.696 / 1.682 |
| W = 0 (static / stream) | 1.710 / 1.698 | 1.739 / 1.708 |
| real data (static / stream) | 1.624 / 1.627 | 1.647 / 1.635 |
(f2 static: real 1.625, x = 0 1.689, W = 0 1.711.)
Tensor duty (static, idle cycles after each drained k-block, f1): busy 98 / 85 / 71 / 53 / 35 % -> clock 1.624 / 1.667 /
1.693 / 1.730 / 1.760 GHz -> effective 1.594 / 1.417 / 1.206 / 0.924 / 0.609. f3: 99 / 97.5 / 96.6 / 82 / 58 % ->
1.632 / 1.652 / 1.653 / 1.670 / 1.728 GHz. Lowering tensor duty never pays (clock rises far less than duty falls).
=> Two limits: (1) a tensor-active ceiling of ~1.70-1.74 GHz that applies even with 16 busy SMs (tiny total power) and
with zero operands; (2) below it, a fast power/current limiter driven by the tensor datapath's DATA-DEPENDENT switching:
with real (random) operands on >= ~96 SMs it pulls the clock to 1.62-1.65 GHz; zero operands (x or W) restore 1.68-1.74.
The operand DELIVERY path (smem reads per FLOP: SS n64 128 B/clk vs n256 80 B/clk, RS via ldmatrix, 1 vs 2 WGs) is
second order (<= 1-1.5 %); DRAM / L2 streaming at the real rates costs 0-1 %.

## Probe 3: invalid (instrumentation bug, kept as a lesson)
A runtime `k % sstride` in the sample check put an integer-division chain (I2F, MUFU.RCP, F2I, IMAD.HI; ~100+ cycles
latency) on warp 0 of each consumer warpgroup every k-block. wgmma is warpgroup-synchronous, so the whole group waits:
f2 (m64n64 groups, 128 tensor clk per group) fell to 47 % busy, f1 (n128) 77 %, f3 (n256) 93 %. Same lesson as the
producer-leanness finding of K2: any per-k-block latency on a consumer warp is paid in full when the wgmma group is short.
Fixed with a counter (probe 4).

## Probe 4 (probe4.log, fixed build): 128 vs 132 CTAs, 2 interleaved rounds; staircase
| form | stream 1, 128 CTAs: loop GHz / busy / effective | 132 CTAs: TF/s vs 128 | stream 0, 128 CTAs: GHz / effective |
|---|---|---|---|
| f1_ss128x2 | 1.619 / 97.2-97.4 % / **1.574-1.577** | 786 vs 766 (+2.6 %) | 1.625 / 1.597-1.600 |
| f3_ss256x2 | 1.616-1.628 / 97.7-98.1 % / **1.586-1.591** | 783 vs 763 (+2.6 %) | 1.651 / 1.624-1.626 |
| f2k_ss128ks | 1.593-1.598 / 96.0-96.2 % / 1.533-1.534 | 756 vs 747 | 1.644 / 1.585-1.590 |
| f2_ss64x2 | 1.615-1.628 / 85.8-86.1 % / 1.389-1.397 | 698 vs 667 | 1.633 / 1.569-1.572 |
(132 CTAs keep the per-CTA work, i.e. 3.1 % more FLOP: per-SM rate is unchanged within noise, so a perfectly balanced
132-SM decomposition would be ~2.6-3 % faster; see conclusions for why the tile granularity prevents it.)
Staircase (f3 static, 6 us no-tensor spin before and after the loop, 1 us bins, samples every 2 k-blocks = 1.24 us):
before 2.04-2.05 (profile reads ~3 % high vs the 1.98 boost), first us of tensor work still 2.04, next us 1.60,
then flat 1.62 (one run: 1.72 from 13.5 us), loop end -> 1.79-1.88 within 1 us -> 2.05 after ~2 us. Stream f3: flat
1.62 from 2.5 to 19.5 us. The limiter engages ~1 us after tensor work starts and releases ~1 us after it stops.
(n128 forms sampled every 0.62 us show a bimodal 1.37 / 2.06 profile: sample-timing artefact, ignore; loop averages agree.)

## Iteration analysis block (probe kernels; no shipped kernel, so no bo17L-vs-mine nsys line)
1. Timing: graph-replay times after the flush are in the tables above (event timing, fixed cost subtracted: the same
   quantity nsys in-graph reports for a one-kernel graph). The base kernels' real clocks were measured by K2 (pooler
   1.600, unpooler 1.633, o 1.641 GHz) and Q3 (qkv 1.63-1.65): they sit exactly where the probe puts their forms.
2. ncu (ncu_probe.sh: --set full, --clock-control none, flushed caches; ncu replays each kernel ~40x so its clock is
   1.67-1.74 = the "sustained-ish" regime, compare utilisations, not clocks):
   | form (stream 1) | dur us | tensor % el / act | SM % | issue % | elig/clk | smem pipe % | L2 % | DRAM MB (TB/s) | occ % | top stalls |
   |---|---|---|---|---|---|---|---|---|---|---|
   | f1_ss128x2 | 23.94 | 78.2 / 88.4 | 78.2 | 21.8 | 0.24 | 58.9 | 69.6 | 69.2 (2.89) | 14.2 | barrier 26, wait 23, long_sb 17, wg_arrive 12 |
   | f3_ss256x2 | 24.03 | 77.3 / 88.9 | 77.3 | 12.7 | 0.14 | 48.5 | 67.5 | 68.2 (2.84) | 14.0 | barrier 36, long_sb 22, wait 15, wg_arrive 8 |
   | r3_rs256x2 | 24.54 | 77.0 / 88.7 | 77.0 | 13.2 | 0.14 | 48.4 | 65.0 | 68.2 (2.78) | 14.0 | long_sb 23, short_sb 22 (LDSM), barrier 18 |
   | f2_ss64x2 | 27.33 | 69.5 / 79.7 | 69.5 | 34.3 | 0.41 | 69.8 | 66.1 | 71.3 (2.61) | 14.2 | wait 37, long_sb 15, wg_arrive 15 |
   (barrier = the end-of-kernel / long_sb = mbarrier waits; smem pipe % shows the smem-read saving of n256: 48 vs 59 vs
   70 % for n256 / n128 / n64 at the same FLOPs, which buys <= 1 % clock in the judge regime.)
3. SASS (nvcc 12.1 sm_90a on the NVRTC source; NVRTC on the box identical attrs): 168 regs (NC 2) / 232 (NC 1),
   0 spills, LOCAL 0; per 2-k-block unrolled loop: 8 HGMMA (64x128x16 / 64x64x16 / 64x256x16; RS forms read A from
   R176-R183), 2 WARPGROUP.ARRIVE + DEPBAR.LE gsb0 1, 2 TRYWAIT + 2 SYNCS.ARRIVE (stream); RS adds 4 LDSM.x4 per
   k-block per warp; producer 2 UBLKCP per stage; no LDG; no C751x wgmma serialisation.
4. Binding unit: the tensor pipe at 97-99 % of its rate at the clock the power limiter grants (1.61-1.65 GHz with
   real data on 128 SMs); the clock, not the issue / smem / L2 / DRAM path, is the binding limit, and no operand form
   or tile shape moves it by >= 3 %. Physical ceiling under tensor load ~1.70-1.74 GHz (reached only with zero
   operands or <= 32 busy SMs).

## Conclusions (STEP 1) and STEP 2 decision
- In the judge regime (each replay right after a 512 MB flush) the clock is NOT set by the 700 W average-power cap:
  a non-tensor kernel runs at 1.92-1.98 GHz right after the flush. A fast (~1 us) on-die limiter takes the clock to
  1.61-1.65 GHz within ~1 us of tensor work starting on >= ~96 SMs and releases it ~1 us after the work stops.
  It is driven by the tensor datapath's data-dependent switching power (x = 0 or W = 0 -> 1.68-1.74 GHz), on top of a
  tensor-active ceiling of ~1.70-1.74 GHz that holds even for 16 busy SMs.
- Operand form at equal FLOPs (effective GHz = clock x tensor busy, real streaming, 128 SMs; probe 2 + 4 medians):
  SS m64n128 x 2 WG (128x128) 1.574-1.586 | SS m64n256 x 2 WG (128x256) 1.586-1.591 (+0.5-0.8 %) | RS m64n256 x 2 WG
  1.565 vs SS n256 1.587 in the same run (probe 1: RS is not better) | SS m64n256 x 1 WG (64x256) 1.544, 2 WG n128
  sharing A 1.564-1.574 | pooler tile 64x128:
  m64n128 in-CTA split-K 1.519-1.534, m64n64 x 2 WG (K's pooler form) 1.39-1.51 in the probe (placement-sensitive;
  K's real pooler kernel reaches 98-99 % of 1.600 GHz = 1.57-1.58, i.e. the probe's x layout is worse than K's TMA).
  The best form (n256, 128x256 tiles) gains <= 1 % over K's form; concurrent DRAM streaming costs 0-1 %.
- FLOPs per joule DOES depend on the form in the average-power-capped regime (0.5 s back-to-back, no flush): static
  n256 1.70-1.71 GHz vs n64 1.62 (+5 %), and with DRAM streaming everything drops to 1.29-1.37 GHz. That regime is
  what serving would see, but the judge never reaches it (its flush + host sync gaps keep the average low).
- 132 vs 128 SMs: per-SM rate unchanged -> +2.6 % throughput, but no exchange-free decomposition gets every CTA
  below the 128-CTA tile: N = 8192 / 4096 / 2304 with 8-column wgmma granularity always leaves some CTA with the
  current tile size (e.g. unpooler 64 x 248 tiles: 33 x 248 = 8184 columns, 8 left over), and the n248 / 1-WG / in-CTA
  split-K forms that 132 SMs would need lose 2-3 % effective rate vs the 128x128 form; with an exchange it is K2's
  measured +2.5..+7 us. Not a lever.
- => No operand form or tile shape sustains a >= 3 % higher clock (best: +0.5-1 %, worth <= 0.2 us on pooler /
  unpooler, less than any restructuring it needs: the 128x256 tile needs 2-way split-K for 128 CTAs). STEP 2 (building
  pooler / unpooler / qkv / o kernels in a new form) is NOT warranted: no pw.py / sub_pw.py is delivered; bo17L's
  m = 256 kernels stay.
- Not pursued (rejected): grouping the judge's one-hot probe rows to cut switching power (input-distribution gaming,
  and worth << 1 %); intentionally lowering tensor duty (clock rises far less than duty falls: 85 % busy -> 1.667 GHz).

## Files
pwprobe.py (probe; --forms/--stream/--ncta/--idle/--zero/--pre/--post/--rounds/--sustain), ncu_probe.sh,
probe1.log (forms, judge + sustained regimes), probe2.log (mechanism: SM count, duty, zero data, repeats),
probe3.log (INVALID: modulo bug), probe4.log (fixed: 128 vs 132 CTAs, staircase), ncu_probe.log.
