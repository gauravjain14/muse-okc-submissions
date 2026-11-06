# WP-RT: startup chain of the main kernel at bs32 / 128 / 256 (2026-10-27)

Worker WP-RT. Base dev/moe_v944.py, then dev/moe_v719.py (v944 + WP-A TMP), final candidate rebased on dev/moe_v984.py.
Files: dev/moe_v960.py (v944 + MOE_RT_TR stamps; default build = v944 SASS), dev/moe_v961.py (v960 + MOE_RT_SG),
dev/moe_v962.py-v968.py (v719 + the stamps and every probe / switch below, all off by default = v719 SASS),
dev/moe_v969.py (FINAL: v984 + MOE_RT_FR / PL / INIT / WREL baked, escape MOE_NO_WPRT = v984 SASS), dev/wprt_probe.py (per-CTA startup chain: judge protocol =
graph replay, 512 MB L2 flush, jm seeds; MOE_WP32_TR event trace + RT stamps at ctr[32768 + 512 b + 480 + k]),
dev/wprt_run*.sh. Logs: scratchpad wprt/run*.log.

## 1. Measured chain on v944 (wprt_run1: trace build, flush, 8 jm seeds; per-CTA medians [p90 / max], us after the first entry)

| event | bs32 (8,4,1) MW1 | bs128 (8,4,1) MW4 | bs256 (8,8,1) MW8 |
|---|---|---|---|
| init barrier passed (mbarrier inits) | 0.94 | 0.83 | 0.78 |
| warp 0's bitmap words (ids landed + 32 ballots) | 1.50 | 1.39 | 1.56 |
| bitmap barrier + PAR zeroing (c10) | 1.81 | 1.70 | 1.90 |
| warp 0 ranked (32 shuffles) | 2.46 | 2.24 | 2.53 |
| tables written / barrier (c11) | 2.72 / 2.79 | 2.67 / 2.74 | 3.00 / 3.09 |
| pair list (c12) | 3.08 | 3.25 | 3.96 |
| KRED zeroing done / consumer gather start | 3.38 | 3.52 | (GCP: plr seen 4.21) |
| gather done | c13 3.98 [7.08 / 7.60] | c13 5.09 [5.66 / 6.35] | GCP rows landed 5.77, release 9.00 [9.58 / 10.16] |
| signaler: rdy seen / release returned (XREL) | 4.29 / 7.05 [7.60 / 8.10] | 5.16 / 5.70 [6.72 / 7.00] | - |
| producer dec / loop top / first claim | 4.51 / 5.93 / 7.18 | 2.92 / 3.31 / 3.98 | 3.46 / 4.30 / 5.32 |
| first weights issued | FST 2.38 (104 mma CTAs); wgmma CTAs 8.06 | 5.65 | 6.67 |
| all-CTA gather barrier seen | ~8 (wgmma CTAs) | ~7.5 | 10.96 |
| first x copy issued | 9.71 (wgmma CTAs) | 7.7-8.0 (wgmma) | 11.58 |
| consumers' setmaxnreg.inc (c15) | 5.33 | 5.29 | 4.17 |
| warm-up stage end | - | - | 11.63 [12.41 / 12.42] |
| first data | mma 5.62, wgmma (32 CTAs) 10.05 | wgmma (98) 8.28, mma XDIR (39) 7.46 | 12.28 [13.12 / 13.47] |

No L2 flush (4 seeds): routing ~0.3 us earlier, producer path dec -> first weights 2.7 -> 1.4 us at bs128, bs256 warm-up
3.98 -> 7.26 (3.3 us vs 7.4 with the flush), first data bs32 mma 4.94 / wgmma 8.18, bs128 7.47 / 5.73, bs256 11.27.

Readings:
- bs32: the 32 wgmma-first CTAs (tiles of 17-32 tokens) start 4.4 us after the others. Their all-CTA gather waits for the
  XREL releases, whose gpu release fence waits for the FST weight copies in flight (rdy seen 4.29 -> release returns 7.05).
- bs128: the wgmma CTAs' x is gated by the barrier (~7.5 = the last release ~7.0 + poll); the XDIR CTAs (x direct) are gated
  by the producer's weights (issued 5.65: 2.7 us of cold producer code after the dec).
- bs256: two gates of the same size: the GCP gather (rows landed 5.77 -> converted/released 9.00, also 3.1 us without the
  flush) -> barrier 10.96, and the cold warm-up stage (4.2 -> 11.6).

## 2. Log
- run2/run3 (dev/moe_v961.py MOE_RT_SG): rejected, see candidates.jsonl v961-SG. The self-gather of 64 KB / CTA (cold x,
  issued after the pair list) took 2.4 us median at bs128 (up to 5.5 us at bs32, behind the FST weight fill), pushed every
  CTA's share of the global gather and the consumers' setmaxnreg.inc 3 us later, so first data got later for all CTAs
  (bs128 wg 10.3 / ms 9.9). bench2 vs v944: bs32 x0.993 / 0.992, bs128 x0.992 / 0.992 (controls 0.999 / 0.998).
- run4 (v962 on v719): same chain on the new base; XPF (x prefetched into L2 at entry) leaves the gather time unchanged.
- run5: XPF bs32 0.997 / 1.000, bs128 0.998 / 0.996, bs256 0.994 / 0.994 (controls 1.000 / 0.998 / 0.996): rejected.
  SG + XPF 0.993 / 0.990: rejected.
- run6 (v963 probes): rt_sgx's fence.proxy.async costs 0.01-0.03 us; its loads are what is slow. bs256 GCP conversion
  (rows landed -> converted) 2.9 us, 2.2 without the warm-up, unchanged with the weights 2 or 4 us later (MOE_RT_WDLY).
  NO_WARM first data 12.3 -> 10.3 us.
- run7: bs256 WDLY 1000 / 2000 0.994 / 0.995, NO_WARM 0.992 (control 0.995): the first weights' timing is not a bs256 lever.
- run8 (v964 MOE_RT_WREL): trace bs128 last release 7.23 -> 6.84, barrier 7.73 -> 7.50, first data wg 8.64 -> 8.18; bench
  1.0004 / 1.0021 (control 0.9993).
- run9 (v965 / v966): conflict-free self-gather still 4 us (bs128) / 6.7 us (bs32). 16 batched LDG.128 per thread is ~2048
  sector misses per SM: 64 KB of generic loads per CTA is bound by the SM's outstanding misses, and TMA cannot do the 4-byte
  permutation of the wgmma layout. RR probe: the 32-shuffle ranking costs 0.17 us cold vs 0.10 warm, so the routing is not
  instruction-fetch bound (the ncu v944 source counters agree: the routing's stall samples are barrier / wait, ~5% no_inst).
  INIT: init barrier 0.83 -> 0.62 us.
- run10: SG 0.992 / 0.991, SG + WREL 0.991 / 0.989 (dead); WREL bs128 0.9997, INIT bs32 1.0006 / bs128 0.9998 / bs256
  0.9978 (controls 0.9991 / 0.9985 / 0.9953).
- run11 (v967): WG1=5 (every bs17-32 tile on mma.sync, so every first item fast-starts) bs32 0.9715 / 0.9709: rejected.
  WREL + INIT bs128 1.0020 / 1.0005 / 1.0044 (control 0.9979).
- run12 (v968): FR + PL + INIT + WREL bs128 **1.0068 / 1.0058** (control 0.9995), bs32 0.9994 / 1.0009 (0.9981), bs256
  0.9998 / 0.9995 (0.9971). FR + PL alone bs128 1.0024, INIT + WREL alone 1.0031: they add up.
- run13: check_mt 14 sizes 0 failures, max rel-L2 3.30e-3. Chain, candidate vs base build of the same file (per-CTA
  medians, us):

  | event | bs32 base / cand | bs128 base / cand | bs256 base / cand |
  |---|---|---|---|
  | init barrier | 0.87 / 0.67 | 0.81 / 0.63 | 0.90 / 0.67 |
  | ranked (c10 -> r3) | 0.65 / 0.22 | 0.53 / 0.18 | 0.64 / 0.23 |
  | tables | 2.75 / 2.15 | 2.74 / 2.30 | 3.20 / 2.56 |
  | pair list | 3.02 / 2.47 | 3.25 / 2.72 | 4.07 / 3.30 |
  | gather done / GCP release | 3.79 / 3.26 | 5.02 / 4.47 | 9.19 / 8.67 |
  | last release (max) | 7.71 / 7.65 | 7.22 / 6.13 | 10.28 / 9.63 |
  | all-CTA barrier seen | 9.54 / 9.11 (wg) | 7.82 / 6.78 (wg) | 11.13 / 10.49 |
  | first data | ms 5.29 / 4.88, wg 10.26 / 9.98 | wg 8.85 / 7.50, ms 7.74 / 7.17 | 12.16 / 11.68 |

## 3. Summary
- Gates found: (1) the routing chain (mbarrier-init fence waiting for the hoisted ids loads, warp 0's ranking, the pair
  list), which every later link inherits; (2) gpu release fences waiting for in-flight TMA and for a loaded memory system,
  so the all-CTA gather barrier sits behind the weight fill (bs128 / bs32); (3) at bs256 the GCP conversion (2.2-2.9 us for
  ~2 rows) and the cold warm-up stage together; (4) at bs32 the 32 wgmma-first CTAs (first data 10 vs 5 us). No cheap fix
  was found for (4): self-gather, wgmma-free bs32 and FST for wgmma tiles all lose.
- Kept: MOE_RT_FR, MOE_RT_PL (KW 1), MOE_RT_INIT, MOE_RT_WREL, baked in dev/moe_v969.py on v984 (MOE_NO_WPRT = v984 SASS;
  only the (8,*,1) kernels differ, their stage loops byte-identical to v984's, no new spills). Final measurement vs v984:
  wprt_run14 (all shapes) bs32 1.0018 / 1.0011 (control 1.0001), bs128 1.0053 / 1.0070 (0.9990), bs256 0.9998 / 1.0051
  (0.9974), bs1/8/16 byte-identical kernels within noise; wprt_run15 (x3) bs32 1.0022 / 1.0007 / 1.0015 (0.9989), bs128
  1.0042 / 1.0058 / 1.0049 (0.9979), bs256 1.0008 / 0.9996 / 1.0045 (0.9962). hang1 13 shapes, check_mt 14 sizes 0 failures
  (max rel-L2 3.30e-3). Means: bs32 +0.17%, bs128 +0.6%, bs256 +0.3% (bs256 at the edge of its control spread);
  projected geomean ~+0.18%. Recommend integration.
- Rejected: SG / SGO / SGE / XPF (self-gather and x prefetch), WDLY, NO_WARM, WG1=5.
- Not built: per-tile readiness of the first tile with the release on the signaler; a 16-byte-store GCP conversion at bs256
  (the trace build's conversion could not be confirmed in ncu: 0 samples there).

## 4. bs32 first-wave order (planner task, base v969; dev/moe_v970.py, v971.py; wprt_run16 / 17)
v969 ranks heavy first, so the 1-2 wgmma tiles (17-32 tokens) are tiles 0-1. CTAs 0..63 then start on items that wait for
the all-CTA gather: first data ~9.9 us, against ~4.9 us for the FST / XDIR CTAs. The variants are rank permutations in warp
0; wp32_fst searches mma.sync experts only, so it agrees with the table. Every stage loop is byte-identical to v969.

| variant | order | bench2 bs32 vs v969 (control) | trace (8 seeds) |
|---|---|---|---|
| WQ=1 | first wave = the 5 largest mma tiles, then wgmma, then the other mma tiles | 0.9927 / 0.9935 (0.9966); 0.9948 (0.9978) | all first items mma, data0 med 5.0 / max 8.4; but the 32 extra FST CTAs' early fill slows every first item (1131 -> 1222 ns/stage), kernel +0.6 |
| WQ=2 | every mma tile before the wgmma tiles | 0.9795 / 0.9785 | the wgmma tiles become the KSX-split tail: end 3.0 -> 4.2 us, tail +0.5 |
| WQ=3 | WQ=1, but the moved CTAs skip FST (v969's fill size) | 0.9943 / 0.9968 (0.9978) | stages normal again (1090 ns), moved CTAs' first data p90 9.2 (weights from the producer loop) |

All three are rejected. Recovering the 32 CTAs' ~5 us needs early weights for them, and every early-fill increase costs the
other CTAs more than that (FSTW, FST4, WQ1, SG all show the same mechanism). check_mt 0 failures for each variant.

## 5. bs256 startup on v1002 (planner task; dev/moe_v1010-v1013.py, dev/wprt_probe2.py, wprt_run18-22)
Stamps (v1010; the full event-trace build and a light stamps-only build (WPRT_LIGHT=1) agree within 0.1-0.2 us), 6 jm seeds,
flush, per-CTA medians (max), us:
- routing: tables 2.54, pair list 3.29 (consumers), GCP warps see plr at 3.54.
- GCP: cp.asyncs issued 4.68, rows landed 4.97 (5.58), conversion 1.0-1.15 us PER ROW (2 / 3 / 4 rows per CTA:
  2.2 / 3.2 / 4.1 us), batch converted 8.19 (9.33), KRED zeroing 0.43, proxy fence 0.06, release fence 0.43 ->
  release returned 9.11 (10.12). The producer sees CB+3 = grid at 10.49 (0.37 us after the last release), x issued 11.19.
- warm-up: record posted by the producer at 3.96 (consumers ready 3.48), warm-up end 10.75 (11.28). With two warm-up
  stages (MOE_RT_W2) the second ends 2.5 us after the first: the first costs ~4.7 us of cold code.
- first data 11.75 (12.43).
What the GCP interval waits on: not cp.async completion (rows landed 0.3 us after issue), not the zeroing (0.4), not the
fences (0.06 + 0.43). It is the conversion itself, and not its instruction count or L2 write transactions:
- GV (4x fewer instructions): 3.1 -> 2.4 us.
- GV2 (4x fewer L2 transactions, full-line st.v4): still 3.4 us.
- EP (the pair list on the same GCP warps): 2.5 us, against 0.75 us on the 256 consumer threads.
Everything the 3 GCP warps run during the cold warm-up is slow. The warm-up's 8 consumer warps (2 per SMSP) share the SM
with them, so the two chains are coupled.
Bench (bs256 vs v1002 x2, control 0.997): GV 0.9935, GV+GZ 0.9930, GV+GZ+EW 0.9953 / 0.9957, EW 0.9981 (wprt_run20);
GV2 0.9898, GV2+EW 0.9937, +EP 0.984, EW+EP 0.987 (wprt_run22). bs128 placebo kernels are within noise. All rejected:
- The trace gains (barrier 10.7 -> 9.7, first data 11.75 -> 11.1 with GV+GZ+EW) do not convert.
- The GCP-side code changes cost 0.3-0.7% at bs256, probably through the producer warpgroup's code (epilogue warps 10-11
  share it).
