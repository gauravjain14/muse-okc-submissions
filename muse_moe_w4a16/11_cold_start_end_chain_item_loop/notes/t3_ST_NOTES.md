# ST: per-stage round trip of the 1-group stage loops (bs8/16/32, then (8,8,1) 1-group) -- worker notes

## Status (top) -- 2026-11-03 ~14:30 PT: CLOSED, nothing banked
- The brief's premise does not hold in the real build: the 1-group stages of bs8/16/32 are STREAM-PACED. Zero-
  perturbation ncu: the stage-top TRYWAIT blocks 150-229 cycles per warp-stage (full build) vs 33-37 (compute-only
  build = the satisfied wait itself) -> ~115-190 cycles of true data wait per stage; the consumers' own cycle is
  ~1540-1560 vs the stream's ~1850-1940 (the stage carries 32 KB weights + 3-4 KB scales; the brief's 1690-cycle
  stream period counted 32 KB only). The WP-RING "consumer-paced" reading came from its stamps (~115 cycles each).
- Bounds (bench2 jm, 2+ copies, controls): consumer-speed ceiling MOE_NO_DEQ (decode 13 -> 3 ops per word, all
  stages) only -3.0 / -3.5 / -3.75% at bs8/16/32; MOE_HMUL_NONE (-25% decode instructions) 0 / -1.2 / -0.5%. A round-
  trip cut is a few % of the consumer cycle -> its ceiling is ~0-0.3%. (8,8,1) light phase: ST_FDS (mma.sync 1-/2-group
  decode -> 3 ops) -2.95% bs128 / -5.1% bs256 (incl. 2-group stages and power) -> 1-group round-trip ceiling ~0.3%.
- Built + measured the one stream-side consumer lever (ST_ER early slot release: NS slots in flight instead of NS-1,
  correct, check_mt 0 failures): bs8 +0.04% (noise), bs16 -0.6%, bs32 -3.3% / -5.0% (two forms): rejected. The stream
  is rate-bound, not in-flight-bound.
- Producer: +~410 ns of issue latency per stage costs 10-12.6% (the producer, ~1200 cycles/stage of a ~1900-cycle
  period, is near saturation), but WP-N's lean producers (X1 / LEAN / ELX) already measured neutral on bs8-32.

## Files
- st_loop.py: dump the innermost HMMA stage loops of a cuobjdump -sass file (text + stall / barrier fields).
- st_ncu1.sh: ncu SourceCounters job (no code change), env V / TAG / RUNS / MOE_DEFINES.
- st_ncu_an.py: per-instruction decomposition of the 1-group loop from the ncu CSV (head / refill / body / tail /
  retry stub), cycles per warp-iteration = samples x 256 x 12 / iterations (one sample per SM per interval over 12
  warps; absolute values read ~13% low vs the DRAM-rate period, ratios are what matter).
- ncu1/ (v4111 full build), ncu2/ (v4111 + MOE_NO_LOAD): CSVs + kernel metrics.
- st_mk1.py BASE OUT -> moe_st_1.py: bound probes ST_CDLY=N (consumer nanosleep at each 1-group stage boundary,
  before the wait), ST_PDLY=N (producer nanosleep after each slot's empty wait), ST_KM kernel mask. Default build =
  v4111 SASS in all 16 kernels (wpcgs --full EXACT). ST_CDLY=0/1/200/400: identical schedules (sim 1.000, only the
  immediate differs); vs v4111 the 1-group loops change by the one NANOSLEEP (sim 0.987-0.991).
- st_tab.py: bench2 json -> per-shape % vs control mean, every copy vs every control copy.
- st_mk2.py moe_st_1.py OUT -> moe_st_2.py: + ST_ER (early slot release, section 3), ST_ERK mask, ST_ERF form bits
  (2 x loads first, 4 no __syncwarp, 8 pin on chunk 1). st_mk3.py moe_st_2.py OUT -> moe_st_3.py: + ST_FDS (timing-only
  ds_dec stand-in, section 4). All defaults = v4111 SASS in all 16 kernels.
- st_run1.sh .. st_run4.sh (GPU jobs), bench/st_run1..4.json (bench2 outputs), st_ncu3/ (bs128/256 full), st_ncu4/
  (bs128/256 compute-only), st_mb_ns.py (nanosleep / try_wait calibration), st_mb_tw.py (superseded, not run).

## 1. Decomposition of the 1-group stage (ncu SourceCounters, v4111 MOE_S5R_FIT build, jm seeds, cold L2)

Cycles per warp-iteration (x12 scale), regions of the SASS loop (bs8 BOT loop @0x6280 513 instr; bs16 TOP @0x63b0 511;
bs32 TOP @0x9e80 514):

| build / shape (seed) | head (top .. first LDS) | of which TRYWAIT->BRA long_sb | refill (LDS .. 1st HMMA) | body | tail | total |
|---|---|---|---|---|---|---|
| full bs8 (70261 / 71516) | 291 / 312 | 223 (BRA @0x63a0) | 93 / 97 | 1372 / 1349 | 17 / 16 | 1773 / 1774 |
| full bs16 (73249 / 70877) | 311 / 245 | 218 / 148 | 83 / 86 | 1348 / 1341 | 11 / 12 | 1753 / 1684 |
| full bs32 (73736 / 73787) | 249 / 288 | 150 / 188 | 92 / 95 | 1295 / 1292 | 12 / 12 | 1648 / 1687 |
| compute-only bs8 (71516) | 114 | 33 | 87 | 1341 | 19 | 1561 |
| compute-only bs16 (70877) | 142 | 35 | 95 | 1289 | 13 | 1538 |
| compute-only bs32 (73787) | 132 | 37 | 101 | 1296 | 12 | 1541 |

- The try_wait retry stubs run in 0.2-6% of the iterations and hold ~0 samples: the hardware TRYWAIT blocks until the
  phase completes, so its blocking time shows as long_sb on the branch that reads its predicate.
- Compute-only (MOE_NO_LOAD: re-scheduled loops, sim 0.40-0.55, same structure) shows the intrinsic cost of the
  satisfied wait in situ: 33-37 cycles. The full build's 150-229 is therefore ~115-190 cycles of real data wait per
  warp-stage (7-11% of the stage); the non-wait boundary (head bookkeeping ~70-100, refill ~85-100, tail ~15) is what
  any round-trip mechanism could cut, and in a data-paced loop that would turn into more data wait.
- Clock under ncu: full 1.69-1.81 GHz, compute-only 1.81 GHz; kernel time compute-only vs full: bs8 53.2 vs 60.3,
  bs16 76.3 vs 88.3, bs32 97.5 vs 112.9 us (same seeds).
- Stream arithmetic: bs32 1-group stage = 32768 B weights + 4096 B scales (8-bit, KW 1) from DRAM (+ x rows from L2);
  bs8 = 32768 + ~2.6-3 KB (S5R); 132 CTAs x 36.9 KB at 4.25 TB/s (bs32 steady window) = 1.145 us = ~1940 cycles at
  1.69 GHz; consumers alone ~1540-1560 (+ ~13% scale on the ncu numbers ~1750). Stream-paced either way.

bs128 / bs256 ((8,8,1) MW4 / MW8 1-group loops, ncu st_ncu3 full / st_ncu4 compute-only, seeds 71637 / 73011):
head 242 / 232 (long_sb 122 / 108) vs compute-only 179 / 154 (long_sb 54 / 40); total 1575 / 1573 vs 1503 / 1514:
~68 cycles of data wait per stage (4%): nearly balanced (consumers ~1505 vs period ~1575). Clock under ncu full
1.62-1.64 GHz vs 1.81 compute-only (power).

## 2. Run 1 (st_run1.sh, box 0): slack tests, bench2 jm 60 reps, 2 copies each, v4111 controls a1/a2, fillers
% time vs control mean (+ = slower); z = ST_CDLY=0 (same code shape, nanosleep 0), c/d = consumer sleep 150 / 300 ns
before every 1-group stage wait (after the release), p = producer sleep 300 ns after each slot's empty wait.

| | bs8 | bs16 | bs32 |
|---|---|---|---|
| controls a1 / a2 | -0.25 / +0.25 | +0.12 / -0.12 | +0.08 / -0.08 |
| z CDLY=0 | +0.46 / +0.67 | +0.81 / +1.32 | +1.27 / +1.25 |
| c CDLY=150 | +3.29 / +3.25 | +2.76 / +2.60 | +2.85 / +2.95 |
| d CDLY=300 | +4.24 / +4.26 | +5.17 / +5.09 | +4.81 / +4.62 |
| p PDLY=300 | +12.60 / +12.08 | +9.63 / +9.75 | +10.64 / +10.50 |

Reading:
- A consumer delay AFTER the release (outside the slot's hold) converts only partly (c - z: +2.7 / +1.6 / +1.6% for
  150 ns, d - z: +3.7 / +4.1 / +3.5% for 300 ns per stage of ~1.05-1.15 us, i.e. 20-35% of the added time; the
  nanosleep duration itself is approximate). Consistent with ~10% consumer slack per stage that is partly jitter.
- A producer delay of 300 ns per stage (in the slot cycle: release -> producer -> copies -> landing) costs 10-12.6%:
  the 1-group stream is bound by the SLOT CYCLE LATENCY: period ~ (hold + turn + issue + landing) / NS. The consumer
  round trip (head / refill) is not on that cycle except through the hold (compute + boundary of the stage that holds
  the slot), which is the part a consumer-side change can cut: release the slot as soon as its bytes are in registers.
- => mechanisms 1-3 of the brief (round trip / try_wait / boundary pinning) have ~0 bound at bs8/16/32 (they shorten
  the consumers' own cycle, which is not binding); the measured lever on this cycle is the hold time (and the producer
  issue time, ~1150-1350 cycles per stage, WP-RING / WP-N).

## 3. ST_ER: early slot release (st_mk2.py -> moe_st_2.py)
1-group stages of the bs8/16/32 kernels issue every LDS of the slot at the stage top (weights, scales and all four
chunks' activation fragments; the base double-buffers x per chunk), then __syncwarp + lane-0 empty-barrier arrive,
then decode + MMA from registers. Pin: ptxas sank the hoisted LDS (and with them the release) to their uses (first
build: arrive at the end of the loop); the arrive's state token AND an opaque zero (%dynamic_smem_size >> 24) is XORed
into the first chunk's scale register (ST_ERF 8: the second chunk's), so the release must issue before the first
decode and every LDS before the release (SASS: bs8 loop 522 instr, 18 LDS at body lines 13-47, ARRIVE line 49, first
HMMA 61; bs16 1-group 516: LDS 23-57, ARRIVE 59; bs32 1-group 518: LDS 19-38, ARRIVE 46). SEPI kernels release every
1-group stage this way and skip the item end's release for 1-group items; bs32 (AEPI) keeps the item's last slot held.
Ordering: LDS (generic proxy reads) program-ordered before mbarrier.arrive (.release.cta), as in the base (the base
arrives after the compute); ptxas emits the arrive with no scoreboard wait in both (MIO in-order), the producer's
try_wait acquire then the bulk copy, i.e. the base protocol at an earlier point. Default build: all 16 kernels = v4111.
No spills (168 regs, 0 LDL/STL). The loops are re-scheduled by construction (sim 0.23-0.47).

Run 2 (st_run2.sh, box 1): hang1 1,2,4,8,9,16,17,32,33,64 ok (both forms); check_mt M 1 2 4 7 8 9 12 15 16 17 20 24 31 32
(normal + skewed, 3 seeds, 2 calls): 0 failures (both forms). wpcgs --full with ST_ER: only moe_4_1_1_4_1, moe_4_2_1_4_1,
moe_8_4_1_4_1 change (60/67 stage loops identical = the other kernels'). bench2 jm 60 reps, controls a1-a3, fillers
fA (v4111) / fB (ER):

| % vs control mean | bs8 | bs16 | bs32 |
|---|---|---|---|
| controls a1 / a2 / a3 | -0.03 / +0.00 / -0.08 | +0.13 / -0.02 / -0.05 | +0.03 / +0.05 / -0.10 |
| ER (pin chunk 0) e1 / e2 / fB | +0.65 / -0.57 / -0.30 | +0.64 / +0.71 / +0.61 | +3.31 / +3.36 / +3.44 |
| ER (pin chunk 1, ST_ERF=8) f1 / f2 | +0.14 / -0.09 | +0.60 / +0.42 | +5.01 / +5.02 |

=> ER does not raise the stream (bs8 noise, bs16/32 slower in every copy vs every control). Combined with the ncu
data wait this says the 1-group stream is RATE-bound (DRAM / request path under compute), not bound by bytes in
flight or the slot round trip; releasing earlier wakes the producer during the consumers' decode on SMSP0 and
(bs32) re-schedules a loop that was not the bottleneck. (WP-RING's BRC, the timing-only version, also lost 0.7-1%.)

## 4. Consumer-speed ceilings (run 3, run 4; timing only, wrong outputs)
Run 3: MOE_HMUL_NONE / MOE_NO_DEQ on v4111, 2 copies each, controls a1/a2:

| % vs control mean | bs8 | bs16 | bs32 |
|---|---|---|---|
| controls | +0.16 / -0.16 | +0.10 / -0.10 | -0.02 / +0.02 |
| HMUL_NONE (no BF16 scale multiply: 4 of ~16 instr per word) | +0.49 / -0.47 | -1.28 / -1.13 | -0.56 / -0.39 |
| NO_DEQ (decode = 3 ops per word, every decode incl. wgmma) | -3.16 / -2.85 | -3.68 / -3.39 | -3.47 / -4.02 |

Run 4: ST_FDS (st_mk3.py -> moe_st_3.py: the WP-DS ds_dec of the (8,8,1) mma.sync 1-/2-group stages as 3 ops):
bs128 -2.75 / -3.14 (controls +-0.06), bs256 -5.03 / -5.19 (controls +-0.03).

Reading: even a ~2x faster decode buys only 3-3.75% at bs8/16/32, and that includes the multi-group stages, item
edges and power; the mid-item 1-group stages, where the round trip lives, are stream-paced. The round-trip pieces a
mechanism could remove (non-wait head ~70-100 + refill ~85-100 + tail ~15 cycles, mostly latency that the second warp
of the SMSP partly covers) are <= ~5% of the consumer cycle -> ceiling ~0-0.3% per shape. Mechanisms 1-4 of the brief
were therefore not built (each would also re-schedule the loops, +-0.5-1% lottery, larger than the ceiling).

## 5. Calibration facts (st_mb_ns.py, H200 box, idle 1.98 GHz)
- nanosleep.u32 N actual extra cycles over an empty clock64 pair: N=0 ~51, 1 ~33, 50/100 ~53, 150/200 ~307,
  300 ~813, 600 ~1827 (granularity: ~155 ns for 150-200, ~410 ns for 300, ~920 ns for 600). The run-1 slack values
  CDLY 150/300 and PDLY 300 are therefore ~155 / ~410 / ~410 ns real.
- Satisfied try_wait clock-to-clock +106-128 cycles (serialised by the clock reads); in situ (ncu compute-only) ~35-50.

## 6. SASS notes
- moe_st_1.py / moe_st_2.py / moe_st_3.py default builds: v4111 SASS in all 16 kernels (wpcgs --full EXACT, 67/67 stage,
  106/106 producer loops).
- ST_CDLY=N: the 1-group loops gain one NANOSLEEP (sim 0.987-0.991 vs v4111; identical across N). ST_PDLY: stage loops
  identical, one producer loop +NANOSLEEP.
- ST_ER: moe_4_1_1_4_1 6112 -> 5592 instr (bs8 1-group loop 513 -> 522: 18 LDS, release, first HMMA at body line 61),
  moe_4_2_1_4_1 (1-group loop 511 -> 516), moe_8_4_1_4_1 (514 -> 518); every 1-group loop re-scheduled (sim 0.23-0.47),
  168 registers, 0 LDL/STL, no C7511/C7512. Without the state-token pin ptxas sank the hoisted LDS and the arrive to the
  end of the loop (arrive at 0x61a0 of 0x42f0-0x6350): a release only moves as early as the last LDS ptxas schedules.

# MISSION 2 (coordinator, 2026-11-03 afternoon): the EDGES of bs8 / bs16 / bs32 -- CLOSED, nothing built past probes

## E1. Files
- st_mk4.py moe_st_3.py OUT -> moe_st_4.py: edge delay probes (nanosleep N, ST_KM kernels): ST_Q1 producer before its first
  copies (bs8/16 early path, bs32 before wp32_fst), ST_Q2 producer after the early path / FST (before its first pass through
  the steady loop), ST_Q3 consumers at the combine entry, ST_Q5 consumers after eis (before their first wait). All
  defaults = v4111 SASS in 16 kernels; Q2 / Q5 keep every stage loop identical, Q1 changes the (4,*,1) producer retry-stub
  pseudo-loops (0.96-0.98), Q3 renames the bs8 stage loop (register names only).
- st_mk5.py moe_st_4.py OUT -> moe_st_5.py: ST_ES lean globaltimer stamps (tid 0, ctr[50000 + 64 b + 56..59]); ST_ES=13
  (entry, combine entry = last item done, combine done) keeps every stage loop identical or renamed only; the first-data
  stamp (bit 2, at the consumer item head) re-schedules all stage loops (0.38-0.55) and was not used.
- st_edge.py (per-CTA end timeline, judge-like: other-compile replay -> flush -> replay), st_run5.sh, bench/st_run5.json.

## E2. Run 5 (box 0): lean end stamps, 8 jm seeds x 2 cold replays, us after the first CTA's entry (medians)

| | bs8 | bs16 | bs32 |
|---|---|---|---|
| K (last combine done) / event time incl. launch | 59.90 / 69.47 | 84.70 / 94.70 | 110.35 / 120.13 |
| CTAs' last item done p10 / p50 / p90 / max | 54.38 / 55.20 / 56.59 / 57.18 | 76.98 / 78.80 / 81.15 / 81.65 | 103.60 / 105.58 / 107.12 / 107.89 |
| K - median last item | 4.26 | 5.03 | 4.82 |
| = tail imbalance (latest - median last item) | 1.50 | 2.34 | 2.19 |
| + end chain (K - latest last item) | 2.69 | 2.80 | 2.56 |
| idle SM time per CTA in the tail (mean K - own last item) | 4.21 | 5.20 | 4.91 |
(The build's stage loops are the base's; WP-TAIL's lean stamps on v1512 read 4.86 / 5.81 / 4.80 with +1.1-3.8 us of stamp
cost.) Start (WP-FAST / WP-TAIL on the same kernels): first data p50 4.8 / 4.9 / 6.2 us vs a ~1.7-2.4 us floor; the
producer's first steady-loop pass is cold (rtd -> stage-2 issue 2.3-2.9 us).

## E3. Delay slopes (bench2 jm 60 reps, B2_NOVERIFY, 2 v4111 controls + fillers; probe at N=600 (~920 ns real) minus
## its N=0 twin (~25 ns), i.e. +~0.9 us per CTA at the probed point; conversion = wall-time change / added sleep)

| point delayed | bs8 | bs16 | bs32 |
|---|---|---|---|
| Q1 producer's first copies (whole producer timeline) | +0.75 us (0.84) | +0.63 (0.70) | +0.51 (0.57) |
| Q2 producer's first steady-loop pass (ring fill) | +0.34 (0.38) | +0.16 (0.18) | +0.28 (0.32) |
| Q5 consumers' first wait | +0.46 (0.51) | +0.11 (0.13) | -0.04 (0) |
| Q3 combine entry (every CTA) | +0.12 (0.13) | +0.11 (0.12) | -0.03 (0) |
| NO_COMBINE (timing only, 2 copies) | -1.88 us (-3.01%) | -2.04 (-2.43%) | -1.34 (-1.21%) |
(controls a1 / a2: +-0.18 bs8, +-0.02 bs16, +-0.08 bs32 %.)

## E4. What converts, and why no new mechanism reaches the targets
- END: entering the combine later costs 0-13% of the delay, so the units' start / claim / polls are not on the critical
  path; the end is the chain behind the LATEST tile (2.6-2.8 us after the latest last item: SEPI epilogue + signaler wake
  + red.release ~1.2, poll ~0.45, the critical unit's loads + reduction + store ~1.0-1.7: at bs8 the unit of a k-split
  pair reloads the whole 4096-column fp32 row of every split pair for its sum of squares, WP-END 1.66 us) plus the
  imbalance of the last claims (1.5-2.3 us). NO_COMBINE (-1.3..-2.0 us) bounds every combine-side mechanism; single-hop
  cuts of this chain (TF / PB / LL / RO / LC / PS / KEL / IEL / YEL: WP-END, WP-TAIL, WP-E2) all measured ~0, and the
  split part is inherent to the bs8 last-wave k-split that W075 found worth +0.55% net. A "last arriver combines" design
  removes the signal + poll hops (~0.6-0.8 us, ~1% bs8) but needs an acq_rel arrival per token and the same reloads;
  below the 2% bs8 target and a multi-hour rewrite of the SEPI epilogue + fast units: not built.
- START: the producer's timeline converts 0.57-0.84 (the strongest edge lever), the consumers' start 0.51 at bs8 only;
  the producer's ring fill 0.18-0.38. Both producer pieces are cold-instruction-fetch bound (WP-FAST: first copy fixed at
  ~2.5-2.9 us whatever moves before it, the first steady-loop pass 2.3-2.9 us wherever it sits; WP-COLD: one per-SM miss
  path). Mechanism considered: issue the first item's stages 2..NS+1 from the warm early path with slot waits (the
  coordinator's "first copies from a minimal cold path" family): a stage-by-stage timeline model from WP-FAST's stamps
  (bs16: stage 2/3 at 8.06 / 9.38, refill 10.44, consumers' first stage 2.07 us cold) shows the cold steady-loop pass
  then lands on the stage after the early ones (the ring holds <= 4 stages ~ 4 us < cold pass 2.4-2.9 + landing ~1.8),
  modelled gain ~0.1-0.2 us; an earlier first copy hits the cold fetch (WP-FAST B: init barrier -0.5 us, first copy
  +0.2 us later) and at bs8 the consumer gate (4.99 us, NW1S) behind it. A dry pass of the producer loop by an idle warp
  competes for the same per-SM fetch path. Not built.
- Ceiling summary (measured): end <= NO_COMBINE -3.0 / -2.4 / -1.2% (any combine change), tail imbalance 1.5-2.3 us
  (only finer last pieces, ~0.8-1.1 us per extra item transition), startup: the producer's cold first copy + ring fill,
  multi-gated. The 2% (bs8) / 1% (bs16, bs32) targets would need several of these together (structural: a smaller cold
  startup path for M <= 32, or a combine without the split-pair reloads).
