# WP-START: startup of the (8,8,1) bs128 / bs256 kernels (and bs32), 2026-11-01

Worker WP-START. Base line B dev/moe_v1511.py (= v3101), line A dev/moe_v1504.py. Namespace v3300-v3399, defines MOE_WS_*,
scripts dev/wpstart_*, logs kda_ws/runs/wpstart_runN.log.

Files:
- dev/wpstart_mk3300.py: the patch script (applies to both lines): `python3 dev/wpstart_mk3300.py dev/moe_v1511.py
  dev/moe_v3300.py` (line B) / `... dev/moe_v1504.py dev/moe_v3350.py` (line A). Every switch off by default; default
  builds = the bases' SASS in all 16 kernels (checked after every patcher change, local NVRTC 13).
- dev/wpstart_probe.py: per-CTA startup chain from MOE_WS_TS globaltimer stamps (ctr[50000 + 64 b + k]), judge protocol
  (graph replay after a 512 MB flush) and no-flush; slot map in its header.
- dev/wpstart_dpchk.py: MOE_WS_DP's tokens vs the consumers' pair list (MOE_WS_DPCHK).

## Run 1: where the startup goes (v1511 + MOE_WS_TS stamps, 6 jm seeds x 3 reps, us after the first CTA's entry, per-CTA medians)

| event | bs32 (8,4,1) MW1 | bs128 (8,8,1) MW4 (no warm-up) | bs256 (8,8,1) MW8 |
|---|---|---|---|
| init barrier / bitmaps / tables | 0.69 / 1.60 / 2.05 | 0.78 / 1.67 / 2.30 | 0.77 / 1.78 / 2.45 |
| consumers' pair list | 2.33 | 2.66 | 3.11 |
| warps 8-11 past setmaxnreg.dec | w8 4.14 (wp32_fst first), w9-11 2.38 | 2.52 | 2.67 |
| GCP: plr seen / cp.async issued / landed | - | 2.92 / 3.71 / 4.05 | 3.34 / 4.55 / 4.87 |
| GCP: converted / zeroed / released (max) | - | 5.21 / 5.90 / 6.75 (6.92) | 7.69 / 8.10 / 8.60 (9.89) |
| producer: setup / slot 0 / stage-0 copies / gather wait | - | 3.22 / 4.55 / 4.90 / 5.41 | 3.35 (warm-up posted 3.64) / 5.00 / 5.41 / 5.75 |
| all-CTA gather barrier seen | - | 7.31 | 10.31 |
| warm-up record / end (wg64-first CTAs / small-first) | - | - | 3.59 / 10.01 (10.39 / 6.96) |
| first data (median / p90) | 4.97 / 8.55 (wgmma-first CTAs) | 8.02 / 8.16 | 11.31 / 11.43 |
Durations (median): GCP issue 1.21 (bs256) / 0.78 (bs128) us, conversion 2.79 / 1.15, zeroing 0.41 / 0.62, release 0.50 / 0.84;
warm-up stage 6.42 us at bs256 (record -> compute done); barrier seen -> first data 1.0 / 0.7 us.
- No flush (code + data in L2): bs256 warm-up 4.59 us (ends 8.09), GCP conversion 3.08 (not faster), first data 10.76;
  bs128 first data 7.69 (-0.33). The GCP chain is not a cold-code cost.
- MOE_WS_D2C (the first batch converted twice): bs256 cold pass 2.97 us, warm pass 2.79; bs128 1.29 / 1.04: the conversion
  is slow when warm too (latency-bound per-thread chain: ~60 instructions per 16-byte chunk incl. 4 scattered LDS + 4 STG.E.32,
  5-6 chunks per row per thread, plus the incremental tile scan from tile 0 for every CTA).
- MOE_NO_WARM (bs256): gather released 7.84, barrier 8.77, first data 9.44 (-1.9 us) but the CTAs' second item starts
  2.3 us LATER (38.57 vs 36.24): the cold first stage costs ~4.2 us under the stream (as WP-RT measured).
- Code map (bs256 moe_8_8_1_3_8 174.6 KB; WP-DEAD's ncu counts of v1506 = the same SASS): routing 0x0-0x3e50 15.6 KB static,
  14.8 KB executed, ~5950 warp instructions per CTA; consumer item loop + paths 0x3e60-0x1b760 94 KB (one first-item path =
  item top ~1.5 KB + one ntl path 8.8 KB (mma.sync) / 9-11 KB (wgmma)); setmaxnreg.dec + GCP 0x1b770-0x1d0a0 6.3 KB, ~3680
  warp instructions per CTA (~1200 per GCP warp); producer / epilogue / signaler / out-of-line waits 54 KB. bs128: routing
  15.2 KB, GCP 6.3 KB (~2780 warp instr / CTA).

## Run 2: the GCP gather rewritten (loop-exact), first data earlier, no bench gain

Switches (dev/moe_v3300.py; every stage loop and producer loop byte-identical incl. control bits, dev/wpcgs_search.py; all
loops after the GCP region (producer, epilogue, signaler) text-identical too, checked separately):
- MOE_WS_GI: the cp.async issue per row (the row's token read once, a thread's 5-6 chunks unrolled).
- MOE_WS_GC + MOE_WS_GCK=9: the conversion of wgmma rows with the per-thread constant swizzle (ch = gt + 96 h keeps
  ch & 7: 4 LDS + 1 store per chunk at constant offsets, rolled h loop; row-major rows keep the original loop). The fully
  unrolled form (GCK 0) and the __noinline__ form (MOE_WS_GCF) re-schedule the bs256 kernel's stage loops (sim 0.37).
- MOE_WS_DP: the GCP warps start after the tables barrier and compute their pairs' tokens from the tables (lane-parallel
  binary search over the tiles, word prefix, k-th set bit) instead of waiting for the consumers' pair list. DP's tokens ==
  the pair list's on 105942 rows (dev/wpstart_dpchk.py, M 65-512 normal + skewed). MOE_WS_DPS (the consumers skip the pair
  list) re-schedules every stage loop (0/16): not used.
check_mt M 64 65 100 127 128 129 200 256 257 300 512 (normal + skewed, 3 seeds, 2 calls): 0 failures for V1 and V2.

| (flush, us) | GCP issue / land / conv / zero / release | release (max) | barrier seen | first data | 2nd item |
|---|---|---|---|---|---|
| bs128 base | 0.78 / 0.35 / 1.15 / 0.62 / 0.84 | 6.75 (6.92) | 7.31 | 8.02 | 28.69 |
| bs128 V1 = GI + GC9 | 0.58 / 0.46 / 0.95 / 0.45 / 1.16 | 6.56 (6.72) | 7.11 | 7.83 | 28.64 |
| bs128 V2 = V1 + DP | 1.42 (incl. tokens) / 0.43 / 0.82 / 0.82 / 0.78 | 6.83 (6.99) | 7.37 | 7.98 | 28.93 |
| bs256 base | 1.21 / 0.27 / 2.79 / 0.41 / 0.50 | 8.60 (9.89) | 10.31 | 11.31 | 36.24 |
| bs256 V1 | 0.76 / 0.57 / 2.34 / 0.41 / 0.54 | 7.97 (8.72) | 9.13 | 10.38 | 35.34 |
| bs256 V2 | 1.63 / 0.61 / ~2.0 / 0.54 / 0.53 | 8.00 (8.61) | 9.01 | ~10.4 | - |
- ~8x fewer conversion instructions cut the conversion by only 15-20%: it is not issue-bound (and not cold-code-bound, run
  1's D2C); the zeroing + release fence (0.9-1.6 us at bs128) wait on the memory system (the release fence also waits for
  the producer's stage-0 weights in flight, issued 4.9 us).
- bs256: the warm-up stage (wg64-first CTAs end 10.39-10.6) gates first data as soon as the barrier moves: V1's first data
  10.38 is warm-up-bound.
bench2 jm 100 reps x 2 passes, 5 middle v1511 controls (dev/wpstart_tab.py; % vs the control mean):
| variant | bs128 | bs256 |
|---|---|---|
| V1 (2 copies) | -0.07 / -0.10 | +0.15 / +0.07 |
| V2 (2 copies) | -0.46 / -0.44 | +0.16 / +0.11 |
(controls bs128 0.9940-0.9964, bs256 0.9929-0.9946.) => bs256 first data -0.93 us converts to ~+0.1% (~0.2 us); bs128
nothing; V2 loses at bs128 (its GCP starts during the consumers' pair list; same loops as V1 otherwise). Not banked.

## Run 2b: what the warm-up stage spends (bs256, MOE_WS_WTS per-warp stamps; MOE_WS_W2 = a second warm-up stage)
| (flush, us) | record seen -> stage loop entry | stage body (record -> done) | wg64-first CTAs | small-first CTAs |
|---|---|---|---|---|
| first (cold) warm-up stage | 1.59 (item top, dispatch, path setup) | 4.62 | body 4.85 (5.45 -> 10.30) | body 1.4 (4.49 -> 5.88) |
| second (warm) warm-up stage (W2) | 0.46 | 1.61 | 1.7 | 0.9-1.1 |
The cold first wgmma stage costs ~3.2 us more than a warm one, the cold item top ~1.1 us; a cold mma.sync stage only ~0.4 us
(similar code size: 8.8 KB vs 9-11 KB per path). All 8 consumer warps start / end the stage within 0.02 us of each other.

## Run 3: does the startup convert? (GDLY = the GCP gather started later, i.e. first data later; WF = warm-up first)
- MOE_WS_WQ (diagnostic: warps 8-11 wait for the warm-up's end: no GCP / weight traffic during the warm-up): the warm-up still
  takes 6.0 us (stage body 4.56 vs 4.62): its cold cost is not memory-traffic contention.
- MOE_WS_WF=1 / 3 (the GCP gather, + the producer's first copies, after the warm-up; on V1, loop-exact): the gather then
  ends at 13.4 us, first data 15.6.
bench2 jm 100 x 2 passes, 8 middle v1511 controls (bs128 0.9958-0.9984, bs256 0.9959-0.9979), % vs the control mean:
| variant | bs128 | bs256 |
|---|---|---|
| GDLY 1000 ns | -0.65 | -0.50 |
| GDLY 2000 ns (2 copies) | -1.39 / -1.34 | -0.98 / -0.95 |
| V1 | +0.01 | +0.03 |
| V1 + WF=1 (2 copies) | -0.08 / +0.03 | -2.46 / -2.52 |
| V1 + WF=3 | -0.04 | -2.41 |
=> a LATER first data converts ~1:1 (2 us -> 2.2 us at bs128, 1.9 us at bs256), an EARLIER one (V1) does not: a second gate
sits right behind the current one. bs128: barrier seen -> first data is the x copy (0.6-0.8 us) and V1's earlier barrier
(-0.17) does not move first data (7.79 both in this run); bs256: the warm-up (wg64-first CTAs end 10.1-10.4) gates as soon as
the barrier moves. In-kernel clock (MOE_CLK, probe mode) 1.65 GHz for base / V1 / GDLY at both shapes.

## Run 4: inside the cold first wgmma stage (MOE_WS_KTS: tid 0 stamps in the CTA's first stage_compute_wg call)
(KTS reads a global flag in every wgmma stage: the kernels run ~25% slower, only the first-call stamps are meaningful.)
| us | entry -> kk0 waited (first chunk decoded, 8 HGMMA issued) | kk0 / kk1 / kk2 groups done | kk3 done | whole stage |
|---|---|---|---|---|
| bs256 warm-up stage, flush | 1.45 | 0.76 / 0.87 / 0.85 | 0.51 | 4.56 |
| bs256 warm-up stage, no flush (code in L2) | 1.09 | 0.66 / 0.64 / 0.61 | 0.35 | 3.42 |
| bs256 warm-up stage, flush, + MOE_WS_TW=1 (dummy wgmma at 1.0 us) | 1.24 | 0.86 / 0.84 / 0.90 | 0.50 | 4.48 |
| bs128 first real wgmma stage, flush | 0.78 | 0.65 / 0.61 / 0.62 | 0.49 | 3.25 |
| warm stage (steady state) | | ~0.3 per kk group (tensor-bound 64-token stage ~1.2 us) | | 1.2-1.6 |
- The early dummy wgmma (warpgroup 2, done at 1.02 us) does not speed up the first stage: no tensor-path wake-up cost.
  The whole first stage is ~2.5-3x slower than warm, every kk group alike, less so with the code in L2: the first execution of
  the ~10 KB wgmma stage code (decode + HGMMA) runs at the instruction-fetch rate (~2-3 KB/us), as does the item top / path
  preamble (record seen -> stage_compute entry 2.2 us cold vs 0.5 warm). Quiet HBM (run 3's WQ) does not change it.
- MOE_WS_TW is not loop-exact in any form tried (inline at the init barrier: 3/8 stage loops; __noinline__: 2/8) and costs
  0.2 us at the bitmap barrier: dropped.

## Run 5: bs128, the release fence vs the first weights in flight (MOE_WS_WR)
MOE_WS_WR: in the (8,8,1) kernels without a warm-up (bs65-128) the producer's first copies wait for this CTA's GCP release
(v969's MOE_RT_WREL extended to the GCP kernels; one producer loop register-renamed, not byte-exact). check_mt M 64-129: 0
failures (WR, WR + V1).
| bs128 (flush, us) | zero / release step | release med (max) | producer first copies | barrier seen | first data |
|---|---|---|---|---|---|
| base | 0.68 / 0.74 | 6.61 (6.74) | 4.68 | 7.14 | 7.84 |
| WR | 0.32 / 0.49 | 5.90 (6.37) | 6.22 | 7.11 | 7.95 |
| WR + V1 | 0.32 / 0.48 | 5.64 (5.99) | 5.96 | 6.79 | 7.68 |
- With no weights in flight the zeroing stores and the release fence take half as long (the gpu fence waits for the SM's
  in-flight TMA loads): the gather's max release moves 0.75 us earlier. But first data does not follow: the weights now
  leave at ~6.0 and land (~1.5 us HBM latency) after the barrier, and the producer's acquire poll, issued with its own
  weights in flight, sees the full count 0.8 us after the last release (0.4 us in the base).
bench2 (6 middle controls bs128 0.9943-0.9970): WR -0.09 / -0.02 / -0.17%, WR + V1 +0.03 / +0.05%; bs256 (WR identical,
V1 changes it) wr +0.11 / -0.03 / -0.08, WR + V1 +0.05 / +0.07%. Rejected.
=> bs128's startup is an equilibrium of memory latencies: the first weights must land by first data (~1.5 us), their
flight delays the gather's release fence and the barrier poll; the gather's own steps (cp.async land 0.35, conversion,
zeroing stores, release) are latency-bound hops. Shortening any one hop moves the gate to the next.

## Run 6: bs32 warps 10-11 late (MOE_WS_L10) and V1 at bs256 again
MOE_WS_L10=ns: warps 10-11 (the epilogue warps, idle until the first item ends) spin ns after the setmaxnreg.dec before
their role (LATE9 without warp 9, whose XREL release is on bs32's startup path); loop-exact on both lines (bs17-32 and
bs33-64 kernels). check_mt M 17-64: 0 failures. Probe: warps 10-11 reach their role 3.84 us instead of 2.31; first data
4.91 (p90 8.65) vs 5.01 (8.54).
bench2 (7 middle controls bs32 0.9970-0.9986, bs256 0.9944-0.9964): L10 1500 ns -0.09 / -0.20%, 3000 ns -0.12% (bs256
placebo +0.07 / +0.10 / +0.10); V1 at bs256 +0.02 / +0.03 / -0.06% (bs32 placebo -0.07 / +0.09 / +0.07). Both rejected:
V1 over 3 jobs at bs256 +0.15 / +0.07 | +0.03 | +0.02 / +0.03 / -0.06% (mean +0.04%, noise).

## Run 7: L2 prefetch of the first item's stage blocks (MOE_WS_WP=n + MOE_WS_WPG; loop-exact on both lines) - REJECTED
The GCP thread (gt 0) issues cp.async.bulk.prefetch.L2 for the first item's first n stage blocks (36 KB each) at the start of
its gather (~2.9 us), so the producer's own (unchanged) copies read L2 and land before this CTA's GCP release fence.
MOE_WS_WPW: also at bs256. check_mt M 64-512: 0 failures.
- Probe (bs128): the prefetch traffic delays everything else: n = 1 producer's first copies 4.86 -> 5.03 us, zeroing done
  6.10 -> 6.49, first data 8.11 -> 8.27; n = 3: first copies 6.35, first data 9.64. The release fence step does shrink
  (0.75 -> 0.50 us) but the extra early HBM traffic slows every cold code fetch / memory hop before it.
- bench2 (8 middle controls bs128 0.9964-0.9981, bs256 0.9947-0.9970): bs128 n=1 -0.53%, n=3 -1.12 / -0.99%, n=3 + V1 -1.15%;
  bs256 (n=3 + WPW) -0.71 / -0.71%, + V1 -0.34%; bs256 placebos (bs128-only builds) -0.11..+0.10%.
=> confirms the closed finding (early DRAM traffic before the gather barrier costs more than it saves), now for L2-only
prefetches of 4.75-14 MB.

## Run 8 (planner request): startup sensitivity bounds, slopes dT/dD
Timing-only delays (dev/moe_v3300.py, every stage loop byte-identical; producer loops scoreboard-permuted for PDLY):
MOE_WS_PDLY=D: the producer's FIRST copy issue held back D (lane-0 globaltimer spin right before it: the (4,*,1) early-start
path at bs8, warp 8's wp32_fst call at bs32, the start of the producer role at bs65+ (which at bs256 also delays the warm-up
record)); MOE_WS_CDLY=D: every consumer thread spins D right after setmaxnreg.inc (first record / stage wait later). With
run 3's MOE_WS_GDLY=D (the GCP gather started D later). bench2 jm 100 reps x 2 passes, 7 middle v1511 controls (bs8 0.9937-
1.0010 (one low control), bs32 0.9964-0.9989, bs128 0.9958-0.9985, bs256 0.9945-0.9966); dT = loss in us (kernel 62.0 /
110.1 / 161.4 / 197.1 us), slope = dT / D:

| shape | PDLY 0.5 / 1 / 2 us | CDLY 0.5 / 1 / 2 us | GDLY 1 / 2 us (run 3) |
|---|---|---|---|
| bs8 | +0.58 / +1.31 / +2.31 us (slope 1.15 / 1.31 / 1.16) | +0.35 / +0.42 / +0.95 (0.69 / 0.42 / 0.47) | - |
| bs32 | +0.36 / +0.83 / +1.88 (0.73 / 0.83 / 0.94) | -0.08 / +0.24 / +0.33 (~0 / 0.24 / 0.17) | - |
| bs128 | +0.27 / +0.68 / +0.95 (0.55 / 0.68 / 0.48) | +0.34 / +0.26 / +0.44 (0.68 / 0.26 / 0.22) | +1.05 / +2.20 (1.05 / 1.10) |
| bs256 | -0.18 / +0.22 / +1.12 (~0 / 0.22 / 0.56) | -0.04 / -0.20 / +0.81 (~0 / ~0 / 0.40) | +0.99 / +1.90 (0.99 / 0.95) |
(per-variant noise ~+-0.15% = +-0.1..0.3 us; bs8 +-0.3%.)
Reading:
- Delays of the chain that currently gates first data cost ~1:1: the producer's first copy at bs8 / bs32, the GCP gather at
  bs128 / bs256. The consumers' own start has slack at every shape (slope <= 0.25 up to 1-2 us except bs8's ~0.5); the
  producer's first copy has ~1 us slack at bs256 and ~0.5 at bs128.
- But advances do NOT convert at the same slope: V1 moved bs256's barrier 1.2 us earlier (first data -0.9 us) for +0.04%
  (3 jobs), bs128's max release -0.2 us for 0; WR + V1 moved bs128's max release 0.75 us earlier for +0.04%; P12 (bs8 / 16)
  0.7 us earlier first copy for ~0. A delay slope is only an UPPER bound on an advance's gain: each startup chain has a second
  gate within ~0.2-1 us behind the first (bs128: the first weights' landing, the x copy after the barrier and the barrier
  poll; bs256: the warm-up end of 40-64-token first tiles, and the HBM-burst after the barrier), so shortening one hop moves
  the gate to the next.

## Summary (WP-START)
Cold-path map (v1511; bs128 / bs256 kernels 173 / 175 KB, 10832 / 10912 instructions; per CTA, us after the first entry,
judge flush):
| role | code before first data | phase timings (bs128 / bs256) | what bounds it |
|---|---|---|---|
| all 12 warps: routing | 15.2 / 15.6 KB (entry+init 2.5, bitmaps 5.1, rank/tables 5.4-5.7, pair list 2.3), ~3.8k / 5.9k warp-instr | init bar 0.78, bitmaps 1.67 / 1.78 (ids HBM), tables 2.30 / 2.45, pair list 2.66 / 3.11 | ids latency, warp 0's ranking chain |
| GCP warps 9-11 | 6.3 KB (dec + gather), ~1200 warp-instr each | plr 2.9 / 3.3 -> issued +0.8 / +1.2 -> landed +0.35 -> converted +1.15 / +2.8 -> zeroed +0.6 / +0.4 -> released +0.8 / +0.5 (6.75 / 8.60, max 6.9 / 9.9) | memory latency of each hop (conversion not issue- or code-bound: 8x fewer instructions -15-20%); the release fence waits for the SM's in-flight TMA weights |
| producer warp 8 | setup + first item path 4.6 / 5.1 KB | dec 2.5 / 2.7, setup 3.2 / 3.35 (warm-up posted 3.64), first copies 4.7 / 5.4, gather wait 5.4 / 5.75, barrier seen 7.31 / 10.31 | cold code; then the all-CTA gather barrier |
| consumers (8 warps) | item top ~1.4 KB + one path 8.8 (mma.sync) / 9-11 KB (wgmma) + 2.4 KB preamble | bs256 warm-up: record 3.6 -> stage entry +1.6 -> stage body 4.6 (wgmma; warm 1.6) -> 10.0 (40-64-token first tiles 10.4); bs128: first real wgmma stage 3.3 us vs 1.2 warm | the first execution of the stage code runs at the instruction-fetch rate (~2-3 KB/us), not HBM traffic (quiet-HBM warm-up 6.0 us), not tensor wake-up (early dummy wgmma: no change) |
| first data | | 8.0 / 11.3 (barrier seen + 0.7 / 1.0: x copy from L2) | |
Epilogue warps 10-11 / signaler 9 reach their roles after the gather (6.0-6.8 / 8.2-8.7 us), off the path.

Variants (all check_mt 0 failures where run; % vs the middle-control mean, bench2 jm 100 x 2):
- V1 = MOE_WS_GI + MOE_WS_GC (GCK 9), loop-exact both lines: bs128 -0.07 / -0.10 / +0.01; bs256 +0.15 / +0.07 / +0.03 /
  +0.02 / +0.03 / -0.06 (3 jobs). Not a gain.
- V2 = V1 + MOE_WS_DP (tokens from the tables, 0 mismatches on 106k rows): bs128 -0.46 / -0.44, bs256 +0.16 / +0.11.
- MOE_WS_WF=1 / 3 (gather after the warm-up): bs256 -2.46 / -2.52 / -2.41. MOE_WS_WR (first copies after the release):
  bs128 -0.09 / -0.02 / -0.17, with V1 +0.03 / +0.05. MOE_WS_WP + WPG (L2 prefetch of the first stage blocks, loop-exact):
  bs128 -0.53 (n 1), -1.12 / -0.99 (n 3); bs256 -0.71 / -0.71. MOE_WS_L10 (bs32 warps 10-11 late): -0.09 / -0.20 (1.5 us),
  -0.12 (3 us).
- Not loop-exact in any form tried (not benched): MOE_WS_DPS, MOE_WS_ZE, MOE_WS_ZC, MOE_WS_TW, MOE_WS_SW (selective bs128
  warm-up: enabling the warm-up machinery at MW 4 re-schedules all 8 stage loops), MOE_WS_EW (early first-stage weights: the
  lean producer loop re-schedules).
BANKED: nothing. Every switch stays off; dev/wpstart_mk3300.py applies to both lines (dev/moe_v3300.py from v1511,
dev/moe_v3350.py from v1504; default builds = the bases' SASS in all 16 kernels, re-checked after the last change).
Why the startup does not pay at bs128 / bs256: delays of the gating chain cost 1:1 (GDLY), but every advance found here
lands on a second gate 0.2-1 us behind (run 8). A real gain needs several hops shortened together (bs128: gather max release
+ first weights landed + x copy; bs256: gather + the cold first wgmma stage), and the two cheap levers for the second gate
(early weights / L2 prefetch, an earlier or quieter warm-up) both cost more than they save: early DRAM traffic slows every
cold code fetch and memory hop, and the warm-up's cold cost is the instruction-fetch rate itself.
Levers left (not built): a startup without the all-CTA gather for the first wave (e.g. first-wave x straight from x rows
with a tensor-map swizzle, which needs a weight packing whose k order matches natural x order = stage-loop changes); fewer
cold bytes on the first wgmma path (the stage code is ~10 KB per ntl path).
