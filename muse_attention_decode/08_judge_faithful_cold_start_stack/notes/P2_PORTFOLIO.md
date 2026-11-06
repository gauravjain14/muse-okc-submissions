# P2 portfolio: batch-1 FULL (d1f = xb_d1f, v1f = xg_v8f), round 3, phase 1 (no GPU)

Sensitivity (ROUND3.md): d1f 0.79 %/judge-us (judge 10.88, local nsys 10.7-10.85), v1f 0.64 %/us (judge 12.60,
local nsys 12.22-12.30). Sources of every measured number are named; "est." means my estimate.

## 0. First-principles budget vs measured phases (nsys-equivalent us from kernel start)

Physics of one batch-1 full call: 80 000 tokens x 2 rows (K, V) x 128 B = 20.48 MB of random 128-B rows, all from
DRAM after the flush. Dependency chain: launch -> page-table row (320 KB, one cold DRAM trip) -> K/V addresses ->
stream 20.48 MB at the random-gather ceiling 4.3-4.6 TB/s (4.5-4.8 us) -> last tile's math -> cross-CTA combine
(release + acquire + dependent load = ~1.1-1.2 us on an idle GPU: MOE section 17 in docs/sibling_lessons.md; WB r26
4-KB publish incl. release 1.15-1.34 us) -> RMSNorm x gate -> store. Per-warp limit ~16 LDGSTS in flight (XA r3j5):
16 warps x 16 x 512 B = 128 KB in flight per SM vs 155 KB per CTA, so ~83 % of the call is in flight at once.

| d1f phase | floor | measured now (source) | gap |
|---|---|---|---|
| start + gather to last landing | ~5.9 (probe 'a' CTA done max 5.89, XA j6, trace) | last tile lands med 6.2 / max 7.0 (XB j38 judge-mode trace, +0.5 trace inflation) | ~0.3-0.6 |
| last-tile math | ~0.2-0.3 (one wgmma QK + softmax + PV chain) | stage 0.55-0.65, "latency-bound regardless of tokens" (XB j38); no-math AB=1 9.50 nsys (XB job35) | ~0.35 (math exposure ~1.3 total) |
| in-CTA merge of 16 warp partials | ~0.1 (2 KB) | store 0.61-0.82 + bar.red 0.06 + merge 0.47 (XB j25 / j39 clock64) | **~1.1** |
| reds + release + arrival atomic | ~0.6 (2 RTT) | 0.56-0.66 (XB j25/j39) | 0 |
| shard load RTT | ~0.45 | 0.37-0.48 | 0 |
| epilogue + store + exit | ~0.2 | call 0.13 + rsqrt 0.21 + store 0.11 + sync/reset 0.22 = 0.67 | ~0.45 |
| total | ~8.0-8.3 nsys | 10.7-10.85 nsys | ~2.6 |

| v1f phase (deliver_r5) | floor | measured now (source) | gap |
|---|---|---|---|
| skeleton: start + gather, no math, no publish (TB3) | ~6.4-6.6 (gather floor 5.44 bench, PLAN, + ~1 bench->nsys) | 7.74 (XG g3) | ~1.2 |
| exposed tile compute (TB1 - TB2) | ~0.5 (one tile chain after last landing) | 10.6 - 8.1 = **2.5** (XG r4a/r4c); 1.47 of it CUDA-core work, ~1.0 wgmma/barrier chain | **~2.0** |
| publish (TB2 - TB3) | ~0.3 | ~0.4; per-CTA publish - loop end 0.93 (XG r2g trace) | ~0.1-0.6 |
| combine after the last publish (full - TB1) | ~0.8 | ~1.6 (r4c / r5fin); trace r2g: slowest publish 9.71 -> gate 10.29 -> bulk 10.93 -> out 11.73 | ~0.8 |
| total | ~8.6-9.0 | 12.22-12.30 | ~3.4 |

v1f tile phase is compute-chain bound (XG r2e/r2g: step waits 0.13-0.17 us; step 0 med 2.98, max 4.85; loop end
med 8.06, max 8.78; last data ~6.3). Issue estimate: ~1007 SASS per warp for the call (2.5 pair steps x 310 + 232
deferred issue, XG notes) = ~2.2 us per SMSP vs a ~4.9 us chain: ~45 % issue utilization. XB job 7 microbench: 4 WGs
per SM run a QK/ALU/PV iteration in 994 cycles vs 566 with 2 WGs and "~550 if staggered" (convoy).

Local SASS facts (NVRTC 13 = image, dev/p2/sass_of.py): xb_d1f 106 regs, stage loop 324 SASS (0x3e90-0x5250), TORD
burst = 12 unrolled tile blocks (~7.9 KB, 0x0c90-0x2b50), post-loop path ~450 SASS + noinline epilogue; the arrival
is MEMBAR.ALL.GPU + ATOMG (red acks + atomic = 2 RTT). xg_v8f 127 regs, C7511, 4000 SASS (64 KB), PV =
QGMMA.64x72 (ones column for row sums). FP8 RS-A wgmma with A = V^T built by 2 LDSM.16.MT88.4 + 8 PRMT per thread
per 64-token head-tile compiles cleanly (dev/p2/toy_rsa.py: QGMMA.64x8x32.F32.E4M3.E4M3 R24, R32, gdesc, 40 regs).

## B1 (d1f) Head-disjoint lane partials: WG-level FP8 PV, no 16-way warp merge  [rank 1]
- Bottleneck: after the last landing, d1f pays the 0.55-0.65 us per-warp stage chain plus ~1.1 us of in-CTA merge
  (16 warps x 4-KB warp partials: store 0.61-0.82 + merge 0.47, XB j25/j39). Math exposure ~1.3 (AB=1 9.50 vs 10.8).
- Mechanism: lanes become (KV head h, tile parity j). Per 64-token tile a lane runs QK = 2 x m64n8k32 (one head,
  as today's descriptors kd + 4h), a lane-level lazy max (TAU 8; P written with the current lane max, the vote rides
  on the lane barrier that publishes P^T; rare raise = slow path), P in E4M3 to a 512-B P^T buffer (double-buffered),
  and PV = O^T[64 d x 8 rows] += V^T x P^T as 2 x m64n8k32 RS-mode: A = V^T from 2 LDSM.T + 8 PRMT (d and token
  permutations chosen so the S^T fragment's bytes land contiguous in P^T), B = P^T descriptor. The accumulator IS the
  lane partial (4 floats/thread). End: lane (h,1) hands 2 KB to lane (h,0) (one STS.128 / LDS.128 + fold at REF),
  then the same 1040-float red, arrival and epilogue as today. Per-token SASS ~20.5 -> ~9 (est.). Registers drop
  (~50), so an 8-lane 1024-thread variant (32 warps: whole 155-KB burst in flight; probe 'c' 32 warps/SM was
  -0.13..-0.17 vs 'a', XA j6) is an option.
- New twist vs closed levers: fresh_eyes_r2 #3 was "closed" by XB AB=4 (+5.5..+15 us), which red'ed 16 WARP partials
  from registers = 16x red volume (an L2-atomic-throughput test). B1 keeps red volume at 1x because the lanes are
  head-disjoint. AB=3 (quarter-size warp partials, -0.2) still stored 16 warp partials; B1 has no per-warp store step.
  Different from NL 2 (keeps 1 CTA/SM) and from MMA-free (more MMA, not less).
- Expected: d1f -0.6..-1.2 judge us = +0.5..+0.95 %. P(works) 0.30.
- Probe P2-J1 (~12 GPU-min, nsys d1f only, 2 copies x 2 passes): timing-only knobs on a copy of k55's XB1 module whose
  default SASS is verified identical: AB=5 = skip X1SYNC/store/bar.red/merge, 256 threads red their own registers at
  today's addresses (same 1040-float volume), arrival + epilogue unchanged; AB=6 = AB 1 + 5 (no math, no merge).
  Rule: build B1 iff AB=5 <= base - 0.6 us, or AB=5 <= base - 0.35 and AB=6 <= base - 1.3. Otherwise close B1 and
  record that the post-loop latency is not the merge.

## B3 (d1f) Dedicated finisher SMs (which SMs do what)  [rank 2, bound rides on J1]
- Bottleneck: after the last CTA's reds the chain is MEMBAR + ATOMG 0.6 -> bar + LDS 0.04 -> noinline call 0.13 ->
  shard LDG 0.48 -> rsqrt/gate 0.21 -> store 0.11 -> bar + counter reset + exit 0.22 = ~1.8 us (XB j25).
- Mechanism: 128 gather CTAs (gather is CTA-count insensitive: 132/120 CTAs 12.14/12.05, WS job46) + 4 token-free
  finisher CTAs that, during the gather, precompute sigmoid(gate) for their 4 rows, run their inline epilogue dry
  (free: they are off every critical path) and spin on a never-reset u64 arrival counter (XA8_MONO pattern, -0.6 on
  v1f). Gatherers end with reds + red.release and exit (no return trip, no epilogue, no reset). Finishers: detect
  (~0.15-0.3), one shard load round for 4 rows, store. Removes the atomic's return, call, bar/LDS, cold epilogue,
  gate math and end reset from the critical path.
- New twist: LAST2 / RK / PDL changed who finishes among gatherers or added a node (+0.75); nothing used idle SMs.
- Expected: d1f -0.3..-0.5 = +0.25..+0.4 %. P 0.30.
- Bound in P2-J1: AB=8 = the last arrival exits right after its atomic (no epilogue, no reset). Rule: build iff
  base - AB8 >= 1.2 us (B3 adds back ~1.25: release one-way + detection + load + store). Free rider in J1: XB1_NC 8
  (WB r26: 4 KB red into 8 copies 1.25 vs 4 copies 1.34) - keep if >= 0.05 faster in both copies.

## B2 (v1f) De-convoy the 4 warpgroups (soft stagger, then ping-pong)  [rank 3]
- Bottleneck: exposed compute 2.5 us; steps run ~1.4-1.6 us each (r2e/r2g) at ~45 % issue use, the 4 WGs enter
  their tensor and ALU phases together (data is resident: waits 0.13-0.17/step). Convoy evidence: XB job 7 994 vs 566
  cycles (4 vs 2 WGs), ~550 if staggered.
- Mechanism: offset the parity-1 WGs by ~half a step so their QK/PV overlap the parity-0 softmax, first softly (a
  start delay; WGs keep their own pace), then if needed with a 2-barrier tensor-phase hand-off (FA3 ping-pong).
- New twist vs XB PP (verify sliding, lost +0.15..+0.4) / W8 1-lane pipelining: those changed the tile code first; this
  first measures, in the real v1f kernel, whether delayed WGs absorb the delay. A pure-latency chain gives +X; a
  contended convoy absorbs it; a soft stagger that is already faster ships as a knob.
- Expected: v1f -0.4..-0.8 = +0.25..+0.5 %. P 0.15.
- Probe P2-J2 (~11 GPU-min, nsys v1f only): copy of XG5 with XG_STAG = ns of __nanosleep for parity-1 WGs before step
  0, on XG_DEFER 0 (so deferred issue is not delayed): DEFER0 x2, STAG 400 x2, STAG 800 x2, k55 x2. Rule: if best
  STAG <= DEFER0 - 0.15 -> freeze soft stagger (then port to DEFER 4 with parity-0 issuing the deferred wave); if
  STAG 800 - DEFER0 <= +0.25 (absorbed, no gain) -> build the ping-pong hand-off (one more job); if >= +0.5 -> close.

## B4 (v1f) Late combiners fetch their inputs during their own publish  [rank 4]
- Bottleneck: the critical combiner is the last-publishing combiner: publish (0.93/CTA) -> gate RTT (+0.58) -> bulk
  RTT (+0.64) -> reduction (+0.75) (r2g: slowest publish 9.71, gate max 10.29, bulk max 10.93, out 11.73);
  full - TB1 = ~1.6 nsys.
- Mechanism: at loop end tid 0 cp.async's the publish counter into smem (no live register, unlike EG); if >= G - 16
  have published, the CTA is late: it cp.async's row c of all 132 slots (18 KB, 128-B AL lines + ML words) into the
  dead K/V ring BEFORE/while it folds and publishes, skips the gate, then checks tags in smem and LDG-polls only stale
  words (the last few CTAs). Early combiners keep today's gate (their slot lines would miss to DRAM: BF +3 us).
- New twist vs EG (+0.27, ld.relaxed into a register, bulk still after the publish), BF (all combiners early) and
  PFS (own-slot prefetch): late-only, async into smem, overlapping the bulk RTT with the publish.
- Expected: v1f -0.3..-0.5 = +0.2..+0.3 %. P 0.25. No cheap bound (full - TB1 = 1.6 caps it at ~0.8);
  falsify with the build: one job, nsys 2 copies x 2 + t_xg quick; keep iff <= base - 0.15 in both copies.

## Ranking by expected score gain per GPU minute
1. P2-J1 (12 min) gates B1 (EV ~0.22 %) and B3 (EV ~0.10 %) + NC 8 rider. Build B1 first if it passes (largest gap,
   1.1 us of merge + ~0.8 of math are the two biggest SM-local items on the d1f critical path).
2. P2-J2 (11 min) gates B2 (EV ~0.06 %; cheap probe, low prior).
3. B4 build + 1 job (EV ~0.06 %). 4. B3 build (if J1's AB8 passes), stacks on B1.
Honest total for this pool: EV ~+0.4 %, best case ~+1.8 % (d1f -1.5, v1f -1.0). Batch-1 full alone cannot carry +5 %.
