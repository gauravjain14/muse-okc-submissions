# HV: bs128 / bs256 (moe_8_8_1_3_4 / moe_8_8_1_3_8), base dev/moe_v4111.py

## MISSION 4 (2026-11-03 PM): fewer item edges through granularity -- CLOSED by simulation (no kernel built)
Tool: t3/HV/qsim2.py (qsim's calibrated model; an item = one claim = a list of 256-row segments; a claim costs EDGE =
0.6 us, a segment boundary inside a merged item EDGE - EI: the 256-row output drain stays, EI = what a merged claim saves:
claim, item top, first-stage data wait). bs32 added to qsim.py: 32-token tiles, count-desc queue, MOE_SPLW 1.0 -> 9 split
tiles x KSX1 4, first data 6.2, end 2.8 us, clock = bs128's shape scaled to PW's 1.727 GHz; calibrated vs the coordinator's
calib job (t3/HV/calib_coord.json, v4111 copies b1/b2, 60 jm seeds, H200 routings t3/HV/hv_routes.json): bs32 needs R =
4.7 TB/s (stream-bound; OV insensitive): sim 109.9 vs 109.7 us, per-seed -1.4 / +0.4 / +2.3%, corr 0.995; bs256 OV 40
re-checked on the coordinator's copies: -1.4 / -0.1 / +1.2%.
Paired median over the 60 H200 routings vs the calibrated v4111 baseline (seeds better / worse):
| policy | bs256 | bs128 | bs32 |
|---|---|---|---|
| ceiling: no item edges at all | +2.94% | +2.50% | +4.29% |
| split tiles 3 (bs32: 5) instead of 5 (9) | +0.06 | -0.01 | +0.04 |
| split tiles 7 (bs32: 12) | -0.22 | -0.10 | -1.05 |
| split k ways 2 instead of 4 | -0.47 | -0.56 | +0.22 |
| no split | -2.07 | -2.23 | -2.21 |
| G1 row blocks claimed in pairs, EI 0 (scheduling only) | +0.49 (53/60) | +0.05 | +0.07 |
| same, EI 0.05 / 0.10 us | +0.58 / +0.66 (59/60) | +0.13 / +0.19 | - / +0.25 |
| same, EI 0.15 / 0.30 / 0.45 us | +0.73 / +0.99 / +1.28 | +0.27 / +0.52 / +0.81 | - / +0.61 / - |
| G1 pairs for the first 70 / 85% of G1, EI 0.1 | +0.29 / +0.38 | +0.06 / +0.06 | - |
| G1 + G2 pairs everywhere, EI 0.15-0.45 | -2.36..-1.54 | -0.89..-0.13 | - |
| G1 + G2 pairs, first 70%, EI 0.15 / 0.3 / 0.45 | +0.44 / +0.72 / +1.05 | +0.22 / +0.49 / +0.75 | -0.29 (EI 0.1) |
| G1 x4 + G2 x2, first 50%, EI 0.15-0.45 | +0.33..+0.86 | -1.08..-0.42 | - |
| G2 pairs first 50%, EI 0.1 | +0.19 | +0.15 | - |
Reading: today's split (5 tiles x 4 at bs65-256, 9 x 4 at bs32) is at the simulated optimum (graded GSS-style sizes add
nothing: the split wave already is the "short items only in the last wave"). A policy reaches +1.0% only if a merged
claim saves EI >= 0.3-0.45 us per merged boundary. Measured counter-evidence: WP-L2 run 9's MOE_L2_PAIR (the producer
claims G1 row blocks in pairs and runs them in one lean loop, consumers unchanged) left the G1 boundaries unchanged
(last warp done -> next stamp 1440 vs 1399 cycles) and lost -0.3..-0.65%: the edge is the consumer's per-output-block
transition (staging, warp skew, item top), i.e. EI ~ 0 unless the consumer transition itself gets cheaper (worker EN's
lever). The EI-free scheduling share (+0.49% bs256, ~0 bs128/32) is below the +1.0% gate and was measured negative as PAIR.
VERDICT: no granularity policy predicts >= +1.0% at a realistic EI; not built.
Addendum (coordinator input from EN: consumer item transition ~1260 / 1340 cycles per item per warp at bs128 / bs256,
delay slope 0.5-0.7 -> ~0.39-0.54 us per item at ~1.68 GHz, consistent with the fitted 0.6 us edge): TWO-ROW-BLOCK ITEMS
FOR 1-GROUP TILES (a consumer-side merge: two accumulator sets, one transition per pair), EI = 0.30 / 0.45 / 0.55 us
saved per merged boundary (qsim2 build(..., ntl_max=1); 60 H200 routings, paired median, seeds better / worse):
| policy | bs256 | bs128 | bs32 (moe_8_4_1) |
|---|---|---|---|
| 1-group G1 x2 | +0.56 / +0.72 / +0.77 (57-60/60) | +0.38 / +0.59 / +0.69 (51-58/60) | +0.66 / +1.02 / +1.23 (57-59/60) |
| 1-group G1 + G2 x2 | +0.40 / +0.61 / +0.67 | -1.16 / -0.82 / -0.67 | -0.20 / +0.20 / +0.33 |
| 1-group G1 x4 + G2 x2 | +0.42 / +0.66 / +0.75 | -0.90 / -0.54 / -0.34 | -3.48 / -2.81 / -2.44 |
| <= 2-group G1 + G2 x2 | -1.85 / -1.50 / -1.39 | -0.27 / +0.02 / +0.18 | -0.61 / -0.13 / +0.04 |
| 1-group G1 x2, first 85% only (EI 0.45) | +0.35 | +0.37 | +0.85 |
| 1-group G1 + G2 x2, first 85% (EI 0.45) | +0.44 | +0.13 | +0.57 |
=> best policy = 1-group GATE/UP items in pairs (down items single: merging them coarsens the down phase next to the
split wave and its readiness waits): bs256 +0.72..+0.77%, bs128 +0.59..+0.69% even with EN's full per-item cost saved
-> below the +1.0% gate at both. bs32 reaches +1.02..+1.23% only at EI >= 0.45 (i.e. the merge must remove ~80-90% of a
whole transition, with no codegen cost); its 1-group loop is the shape's hot loop (ST's area at bs8-32). Not built.

## MISSION 3 (2026-11-03 PM): coupling / concurrent heavy-light ROLES -- CLOSED by simulation (no kernel built)
Simulator t3/HV/qsim.py (offline, no GPU): the kernel's queue (rank count-desc, mix5 with K 5, 64-token tiles, G1 32 x 16
stages, G2 16 x 16, last 5 tiles' G2 split 4 ways into 4-stage pieces, static first wave, in-order claims), 132 SMs,
per-stage cycles from t3/HV/hv_wgb.py (mma.sync ntl1/2 1460/1774 resident; wgmma ntl3..8 1932..2750 streaming) + OV,
the measured in-call clock profile (nsys jm v3720, GHz per 10 us), 35840 B per stage from DRAM (0 for an expert's
continuation tiles: L2), shared DRAM cap R = 4.5 TB/s (max-min fair) and 45 GB/s per SM, 0.6 us item edge, G2 stage
k waits for G1 row blocks 2k, 2k+1 + 2 us, start at first data (11.0 / 8.7 us; bs128 cold wgmma-first +1 us), end +3.8 /
3.3 us. Routings = the H200's own (t3/HV/hv_routes.json from t3/HV/route_dump.py on box 0: torch's CUDA RNG differs by
device -- my earlier local route_stats.py routings matched 0/60 seeds; the statistics in mission 1 are still fine).
Calibration vs the per-rep v4111 control copies of my runs b1 + b2 (6 copies, jm seeds 0..59):
| | OV (cycles) | sim median | measured | per-seed error p10 / median / p90 | per-seed correlation |
|---|---|---|---|---|---|
| bs256 | 40 | 197.1 | 196.5 | -1.2 / -0.2 / +1.2% | 0.995 |
| bs128 | 0-20 | 163.3-164.1 | 160.5 | -0.6 / +1.0 / +2.4% | 0.992 |
DRAM timeline (8 seeds, per 10 us from 30 us on) sim vs nsys: bs256 3.47 4.23 4.17 3.98 3.79 3.82 3.92 4.06 4.28 4.09 3.86 vs
3.51 4.19 4.09 3.96 3.83 3.86 4.00 4.14 4.33 4.23 3.98; bs128 4.21 4.42 4.38 4.34 4.17 4.21 4.22 4.19 vs 4.32 4.55 4.54 4.45 4.21
4.27 4.25 4.24 (within ~0.1-0.2 TB/s; the first bins differ by the 13-us nsys window smear of the startup).
NOTE: on the jm routings with v4111's mix5 the DRAM shape is FLAT (3.5-4.3 TB/s from 30 us); the 0.53 ... 4.56 ...
2.72 profile in the brief is the v1511 sprint-seed capture before MIX.

Bounds from the sim (60 seeds, paired median vs the calibrated mix5 baseline):
| what-if | bs256 | bs128 |
|---|---|---|
| no DRAM cap at all (R = 100 TB/s) = ceiling of ANY schedule / overlap change | +0.76% | +3.31% |
| DRAM cap 4.0 TB/s | -3.79% | -8.09% |
| clock 1.78 GHz throughout (no power dip) | +5.88% | +2.09% |
| no item edges | +2.94% | +2.50% |
| wgmma stages at their data-resident cost | +1.43% | +0.20% |
| no G2 readiness latency | +0.00% | +0.00% |
Roles (heavy tiles ntl >= 3 and light tiles as two queues, G1 then G2 each, heavy group = frac x the work-proportional
share, static first wave per group, stealing from the other queue's front):
| frac (heavy SMs median) | bs256 | bs128 |
|---|---|---|
| 0.6 (28 / 11) | -4.39% | -8.82% |
| 0.8 (38) | -3.40% | - |
| 1.0 (48 / 18) | -2.22% | -1.66% |
| 1.2 (56) | +0.06% | - |
| 1.4 (66 / 26) | -0.72% | -0.20% |
Reading: in full mode the consumers are compute-paced almost everywhere and DRAM binds only in short all-light windows
(at most +0.8% at bs256 even with infinite bandwidth); the 20-us "coupling" gap to compute-only is the power-dipped clock
(~6% at bs256) and per-stage costs under streaming, not idle overlap. Roles fragment the LPT queue: the smaller group's
long items end up in the tail. VERDICT: below the >= 3% bs256 gate (ceiling +0.76%); not built.

## MISSION 2 (2026-11-03 PM): L2-partition-local weight streaming -- CLOSED at the bound stage (no kernel built)
Gate (coordinator): a near-only copy form must reach >= today's contiguous 1 x 36 KB rate under consumer load.
It does not: the single 5-D tensor-map copy of a near class block runs 3.52 TB/s under full load vs 4.58 (-23%).

### B1: full-scale bare ring (t3/HV/l2b.py; 132 CTAs, 4 x 36 KB slots, no consumers, graph replay after a 512 MB flush,
median of 15; box 1 then box 0). Probe: SM sides = PW's pattern; page k bits 513/1024 pages = 1, 30 / 8192 chunk votes
against parity((chunk) & 0xAB) ^ k; 32768 near boxes per side.
| form (36 KB per stage) | 238 MB (bs8-size) | 457 MB (bs32-size) | 701 MB (bs256-size) |
|---|---|---|---|
| c1: 1 x 36 KB contiguous (today) | 4.058 | 4.335 / 4.349 / 4.350 | 4.463 / 4.443 |
| c9: 9 x 4 KB contiguous, lane 0 | 3.570 | 3.806 | - |
| c9p: 9 x 4 KB contiguous, 9 lanes | - | 4.219 | 4.320 |
| n9: 9 x 4 KB near, lane 0 (PW's form) | 3.733 | 3.925 | - |
| n9p: 9 x 4 KB near, 9 lanes | 3.873 | 4.103 / 4.102 | 4.180 |
| nt: ONE 5-D tensor-map copy (8 near 4 KB chunks, index bits 2/4/6 varying) + 1 x 4 KB near | 3.829 | 4.067 / 4.132 | 4.218 |
| f9 / f9p: far only | 2.166 | 2.272 / 2.406 | 2.445 |
=> at equal copy structure near beats mixed by ~3-5% (n9 vs c9, n9p vs c9p small), but every near form is 5-6% BELOW the
contiguous single copy; 4 KB granularity itself costs (c9p -3%). (PW's earlier "+5%" was near vs a 9 x 4 KB mixed ring.)

### B2: PW's coupled harness (t3/HV/hv_near.py = copy of t3/PW/pw_near.py + cmodes 10/11; 300-us calls, flush before,
consumers = v4111's stage_compute<1,1,S6O> on 8 (full) or 4 (half) warps, 2 rounds interleaved; box 1). Ring TB/s:
| cmode | no consumers | full load | half load |
|---|---|---|---|
| 0: 1 x 36 KB contiguous (today) | 4.584-4.593 | 4.581-4.582 | 4.581-4.583 |
| 2: 9 x 4 KB near, one per lane | 4.581 | 3.41-3.50 | 4.12 |
| 7: near, adjacent pairs merged (~6 copies) | 4.583 | 3.74 | 4.56 |
| 10: ONE 5-D tensor-map copy, 8 x 4 KB near class block (32 KB) + 1 x 4 KB | 4.561-4.564 | 3.52 | 4.36-4.38 |
| 11: ONE 5-D tensor-map copy, 4 x 8 KB near runs (chunks 4g+1, 4g+2; g bits 0, 2) + 1 x 4 KB | 4.567-4.571 | 3.53-3.54 | 4.33-4.34 |
=> one copy instruction does not remove the loss: the scattered 2-8 KB pieces themselves (not the producer's issue
count) cost ~23% under full load and ~5% at half load; without consumers no near form exceeds the contiguous ceiling.
(Geometry check: PW is right, the longest contiguous near run is 8 KB: parities of chunks 0..7 are 0 1 1 0 0 1 1 0;
the coordinator's "adjacent chunks always differ" hint is wrong for odd/even pairs 4g+1, 4g+2 and 4g+3, 4g+4.)
Energy columns of the harness are too noisy here to read (cmode 0 alone 124-199 mJ per call across rounds); consumer
work under full load +2% for the near forms comes with their LOWER stream rate (less power -> clock 1.785 vs 1.74).
VERDICT: closed. Any near layout needs >= 4-8 pieces per stage; the stream-bound shapes would lose 5-23% of rate to gain
PW's estimated -0.5..-1% from the clock.

## STATUS (2026-11-03)
- Direction 2 (heavy wgmma stage cost): CLOSED by microbenchmark. The kernel's stage form is already the fastest of every
  arrangement tried; cross-stage pipelining (group in flight across the boundary) is 3-5% SLOWER; HMUL2 -> XOR saves
  0-2%; moving decode work to either pipe (no IMAD / DQH IMAD.HI rotations) costs 6-38%. wgmma for 2-group tiles = mma.sync.
- Directions 1 / 3 (first-phase DRAM use, startup): BOUNDED SMALL. Removing the all-CTA gather barrier from every CTA's
  first item (timing bound, wrong outputs) is worth only +0.72% bs128 / +0.30% bs256; a light-first queue (so every
  first item could take its x straight from x, XDIR-style) LOSES 1.07% / 1.54% (correct build), and even with the
  barrier removed on top it is only +0.29% / +0.18%. No feasible startup form can beat the +0.7% / +0.3% bound.
- 16 consumer warps for the heavy stage (one row group each): equal or slower in isolation (closed).
- NOTHING BANKED. Defaults of every HV switch = v4111 SASS (moe_hv_1.py default sha 595ff80197a9 / 571003e3fd87).

## Routing facts (jm seeds, t3/HV/route_stats.py, local CUDA generator)
| bs | pairs | experts | tiles | 1-group | 2-group | ntl3..8 (heavy, wgmma) | busiest med |
|---|---|---|---|---|---|---|---|
| 128 | 190 | 23.1 | 23.3 | 17.0 | 3.84 | 1.10 0.41 0.25 0.20 0.14 0.38 (2.48) | 43 |
| 256 | 399 | 25.0 | 26.6 | 12.7 | 6.51 | 2.75 1.34 0.81 0.39 0.25 1.87 (7.41) | 100 |
Stage-time share (tiles x 768 stages x measured per-stage cost from the WP-MIX trace): bs256 heavy 35%, 2-group 24%,
1-group 41%; bs128 heavy 14%, 2-group 18%, 1-group 68% (1-group loop = worker ST's).

## Run 1-4: heavy-stage microbenchmark (t3/HV/hv_wgb.py, H200 box 0/1, 132 CTAs x 384 thr, setmaxnreg 216/72 as the
kernel, 3-slot ring of 69632 B with full mbarriers + named-barrier release (288), the kernel's own helpers: S6O decode
dequant_o, wgmma_nN, stage_compute_wg, stage_ds_c). data 0 = smem-resident (producer only arrives), data 1 = producer
streams 35840 B weights+scales from a 2 GB HBM buffer + 32 KB x from an L2-resident buffer per stage. 400 stages,
median over CTAs of 3 launches; MEASURED.

cycles per stage, data resident (data 1 in brackets):
| mode | NTL3 | NTL4 | NTL6 | NTL8 |
|---|---|---|---|---|
| 0 kernel stage_compute_wg | 1814 (1932) | 1980 (2067) | 2324 (2411) | 2560 (2750) |
| 1 pipelined: last group in flight across the stage boundary, next stage's wait+LDS+chunk-0 decode under it | 1910 (2077) | 2033 (2123) | 2374 (2412) | 2666 (2770) |
| 4 next-stage wait + LDS hoisted before wait 0 (WGSN-like) | 1990 (1983) | 2053 (2105) | 2414 (2409) | 2657 (2787) |
| 3 wgmma only (raw words as A, no decode) | 1097 | 1140 | 1571 | 2080 |
| 11 no IMAD (shifts as SHF: +64 ALU, -64 FMA) | 1937 | 2088 | 2474 | 2742 |
| 12 HMUL2 -> XOR (bound: no scale multiply) | 1824 | 1946 | 2274 | 2543 |
| 13 no IMAD + XOR (all ALU) | 2034 | 2150 | 2349 | 2746 |
| 14 HMUL2 -> IMAD (x*16+R) | 1843 | 1961 | 2298 | 2613 |
| 15 DQH=1 (rotl 10 as IMAD + IMAD.HI) | 2071 (2143) | 2231 (2334) | 2695 (2775) | 3108 (3256) |
| 16 DQH=2 (both rotations on FMA pipe) | 2408 (2477) | 2646 (2713) | 3180 (3168) | 3555 (3611) |
NTL1 / NTL2: wgmma kernel form 1620 / 1783 vs the kernel's mma.sync stage_ds_c 1460 / 1774 (data 1: 1879/1838, 1875/1884).
Clocks 1.76-1.80 GHz resident, 1.58-1.63 streaming (power).
Readings:
- The pipeline-drain model (TC idle during chunk-0 decode, decode idle during the last chunk) is WRONG: removing the
  drain makes every NTL slower. Stage = decode + tensor - ~750..1000 cycles at every NTL: a shared resource, not a
  schedule bubble (ptxas: no C75xx in any variant; the pipelined loop is a single 561-instruction loop, no spills).
- Neither pipe has slack under tensor load: shifting 64 ops to the ALU pipe +5..7%, shifting rotations to the FMA pipe
  +14..39%, removing all 128 HMUL2 (XOR bound) only -0..-2%. (The old "XOR saves 200 cycles at NTL8" does not reproduce:
  -17 cycles.)
- Bound for direction 2 inside the BF16/FP32-accumulation rules: <= 2% of heavy stage time (= ~0.7% of bs256 compute).
  Not pursued. wgmma for 2-group tiles: equal to mma.sync (closed).

CW16 (run 5, same harness, hvc_*: 640 threads, 16 consumer warps x 1 row group, setmaxnreg 104 / 40): NTL3 1934 (kernel
form 1814), NTL4 2056 (1980), NTL6 2326 (2322), NTL8 2631 (2560); streaming 2008/2143/2409/2707 vs 1934/2068/2411/2751.
=> no gain from more consumer warps (gj's v5100 header claims -6..-7% isolated in its own harness; not reproduced with the
ring boundary). (A first CW16 build hung for 300 s on box 0: setmaxnreg.inc 112 exceeded what the dec released; fixed to
104 and smoke-tested with 20 iterations before the full run.)

## Run b1 / b2: startup bounds (t3/HV/mk_hv1.py -> t3/HV/moe_hv_1.py; bench2 jm 60 reps, fillers both ends, 3 middle
v4111 controls, 2 copies per variant interleaved; t3/HV/ab.py = median over reps of mean(controls)/candidate)
Switches (all off by default = v4111 SASS; wpcgs_search: LF / LF+NWL EXACT (all stage + producer loops identical); NB1
stage loops 16/16 identical, producer loops renamed; GATHER_MIN0=9 re-schedules all 8 bs256 stage loops):
- MOE_HV_NB1 (bound, wrong outputs): the producer's first item (item == blockIdx.x) does not wait for the gather barrier.
- MOE_HV_LF: light first: the nf = ceil(grid / 32) = 5 smallest experts before mix5's split tail take queue positions
  0..4 (every CTA's static first item from a light tile), then mix5 over the rest. Correct (hang1 65-256 ok).
- MOE_HV_NWL: the bs256 kernel's producer posts no warm-up stage when the first item is a light gate/up item.
| variant (% vs control mean; every copy vs every control min..max) | bs128 | bs256 |
|---|---|---|
| controls (each vs the other two) | -0.09 / -0.24 / +0.22 | -0.02 / -0.17 / +0.22 |
| NB1 (b1) | +0.75 / +0.70 (+0.56..+0.86) | +0.32 / +0.28 (+0.02..+0.37) |
| NB1 + GATHER_MIN0=9 (b1; bs256 loops re-scheduled) | +0.83 / +0.84 | +0.58 / +0.59 (+0.42..+0.75) |
| LF + NWL (b2, correct) | -1.08 / -1.06 (-1.32..-0.90) | -1.49 / -1.58 (-1.76..-1.35) |
| LF + NWL + NB1 (b2) | +0.23 / +0.35 | +0.20 / +0.16 |
Readings: the barrier gate converts at ~55% (bs128 first data ~8.0 -> ~6 us = -2 us, measured -1.15 us): the producer's
cold first-copy path (~4.9 us) is the next gate (WP-START's "second gate"). Heavy-first (LPT) order is worth ~1.1-1.5%:
any scheme that makes the first wave light (the only way to get x without the gather: XDIR row copies need row-major
mma.sync tiles) gives back more than the barrier is worth. A per-CTA direct gather for heavy first items would need the
wgmma permutation + swizzle per stage (GCP conversion rate ~10 GB/s per SM vs ~19 GB/s needed): infeasible; its bound is
NB1 anyway (+0.7 / +0.3%). XD8-style (direct x only for the CTAs whose mix5 first item is light) is bounded by about half
of NB1 (~+0.35 / +0.15%) and measured -0.4..-0.9% by WP-L2. => startup closed for bs128/256.

## Where bs128 / bs256 time remains (for the coordinator)
- bs256 (196 us): compute-bound at a power-limited clock (time-weighted 1.668 GHz, P-CLK bound 14 us = 7%: PW's lever);
  heavy stages at decode + tensor - ~800 cycles (fundamental under BF16 / FP32 accumulation, this file); item boundaries
  ~0.6 us x ~11 items per CTA (~3%, worked by WP-IT/IT2/W0/HELP, remaining bound ~1.6% needs stage-loop-exact
  consumer epilogue forms); startup <= +0.3%; end chain closed (WP-END/TAIL).
- bs128 (160.5 us): 1-group stages are 68% of the stage time (ST's loop); startup <= +0.7%; boundaries ~1.2% remaining.
