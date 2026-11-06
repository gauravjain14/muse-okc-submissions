# WP-FUSE: fused, statically balanced kernel for small batches (bs8 first), base dev/moe_v1002.py

Worker WP-FUSE, started 2026-11-01 ~05:00 UTC. Namespace v1800-v1899 (dev/moe_v18NN.py), scripts dev/wpfuse_*.
Kernel name moef (opt-in switch MOE_FUSE; default build = v1002 SASS in every kernel).

## G0. Design (no GPU)

### 0.1 What there is to win at bs8 (v1002, wpedge.md per-CTA budget, trace build ~64.5 us)
startup 5.37 (first data) + steady loops 50.2 + item ends 2.02 + gaps 2.43 + tail 1.67 + end 2.83.
The stream between first data and the last hand-off already runs at 4.35 TB/s (bare ring 4.4-4.5), so item edges are
mostly absorbed (WP-EDGE2: ~50% of an item-start delay is exposed). A fused static kernel can remove: the claim /
item-decode startup (~2-2.5 us: moe1t / moe8t-FENT2 reach first data at ~2.5-3.5 us), most of the tail (~1 us, if the
static split stays balanced under DRAM-rate variance), and some of the item-edge exposure (stream 4.35 -> ~4.45 TB/s:
~1 us). The end chain stays ~2.5-3 us (one cross-CTA level, as today). Realistic target: ~-3..-6% at bs8
(62.2 -> ~58.5-60 us bench2); the bare-ring bound (56.6 us + an end) caps it at ~-7%.

### 0.2 Work decomposition (static, identical plan in every CTA, no queue, no scheduling atomics)
- Unit = 32 intermediates of one active local expert (G = 32). Global unit list = active experts in index order x 128
  units; U = 128 E. CTA c takes units [floor(c U / 132), floor((c+1) U / 132)): 1 or 2 pieces (expert, j0, j1).
  bs8 (E med 9): 8.73 units per CTA (8 or 9). Imbalance is at most one unit (6 stages of ~52); the 36 lighter CTAs
  stop streaming one unit early, which frees DRAM rate for the 96 heavier ones (needs per-SM headroom >= 1.4x: G1 measures).
- Why G = 32, not 16 / 8 / 64: G = 64 (the (4,1,1) item size) gives 4.36 units per CTA (4 or 5: up to +3..7% unless
  48 SMs can pull the whole DRAM rate). G = 16 / 8 breaks the down GEMM's scale amortization (below). With G = 32 every
  piece is whole down chunks (no compute-bound partial stages anywhere) and every stage is a full 35840 B block.
- Per unit: gate/up 64 W13 rows (the repacked row groups are already gate 8 + up 8 rows of 8 intermediates) x K 4096 =
  4 stages of 35840 B; down W2[e][:, 32 j .. 32 j + 31] = 4096 rows x 32 k = 2 stages of 35840 B. 6 uniform stages
  per unit, one bulk copy each.

### 0.3 Gate/up phase (reuses the (4,1,1) per-warp stage body unchanged: stage_compute<1, 1, S6O>)
- Stage s (0..3) of unit j: 8 cells, cell of warp w = (row pair rp = 2 j + (w & 1), 256-k block kc = 4 (w >> 1) + s):
  2 row groups x 256 k = the (4,1,1) cell (4 KB words + 384 B S6O scales). Each warp accumulates its row pair over
  its 1024-k segment (4 stages), so a unit end is a 4-way k reduction (the (4,1,1) item end is 2-way over 8 stages).
- New lossless layout wf13[e][j 128][s 4][8 x 4 KB words | 8 x 384 B scales] (a permutation of w13p / s13c6 blocks).
- Activations: the piece's T token rows resident in shared memory, x_res[8][4096 + 8] BF16 (64 KB, row stride 8208 B
  = 16 mod 128 like the (4,1,1) x block's 1040 B), loaded once per piece by T bulk copies of 8 KB; the stage body's
  B-fragment loads point at x_res + g * 8208 + 512 kc + 32 tig. Slots carry no activations.
- Unit end: consumers store their 8 accumulators (2 STS.128 per lane) into an 8 KB staging buffer and arrive on an
  mbarrier (no CTA barrier); a reducer warp (9) sums the 4 k segments, applies gs13 * 2^14 and SwiGLU, rounds to BF16
  (as SGLang / v1002: f2bf(g / (1 + e^-g) * u)) and writes h_j[8 tok][32] (512 B) into a per-CTA h buffer, then
  publishes "units reduced" with st.release.cta (consumers ld.acquire.cta before a down slot that needs it).

### 0.4 Down phase (new per-warp body stage_down; same decode, same MMA, B from h)
- k permutation inside a 32-k chunk (2 NVFP4 blocks): lane tig <-> block tig >> 1, words jj = 0, 1:
  k = 16 (tig >> 1) + 8 (tig & 1) + 4 jj + 2 pairsel + h. A lane needs ONE scale pair (rows g, g + 8 of its block)
  per row group per chunk, amortized over 2 words (the (4,1,1) layout amortizes over 4: +~1 cycle per word, ~+5%).
  The B fragment of both k steps is h[g][8 tig .. 8 tig + 7]: one LDS.128 per sub-stage, natural h order.
- Sub-stage = 1024 rows (quarter qq) x one unit's 32 k: warp w = 8 row groups (rows 1024 qq + 128 w + 16 r + {g, g+8})
  x 2 k steps = 16 words per lane, 8 scale pairs (3 S6O words per (g, block), shared by lanes tig 0/1 and 2/3:
  192 B per warp, no duplicated scale bytes). Sub-stage bytes 16384 + 1536 = 17920; a slot = units j, j + 1 of the
  same quarter (35840 B); an odd piece ends each quarter with one half slot (half bytes, half compute: no idle lanes).
- New lossless layout wf2[e][qq 4][j 128][8 x 2 KB words | 8 x 192 B scales].
- Accumulators acc[8][4] (32 floats) per quarter, carried over the piece's units; after the quarter's last unit the
  consumers red.add them straight from registers: red.global.add.f32 into yacc[parity][pair][row] (32 B per 8 lanes,
  predicated on token column < T; T <= 2: 16 instructions per warp per flush). 4 flushes per piece, quarter order
  rotated by CTA index so the ~15 CTAs of one expert hit different rows at a time.
- Order per piece: x copy, all gate/up units, then down quarter by quarter. The down's first slots need only the
  first units' h (reduced long before); only pieces of <= 3 units can wait for the reducer (~0.3-0.5 us).

### 0.5 Red volume, zeroing, end protocol
- Red volume per call: sum over pieces of 4096 x T x 4 B: ~140 pieces x ~2.1 tokens x 16 KB ~= 4.7 MB (v179 moet:
  ~39 MB, 128 CTAs on the same rows at once). 32-B sector reds ~150 K per call over ~50 us, spread over all L2 slices.
- yacc[2][64 pairs][4096] FP32 (2 MB per parity) in state; call index c = ctr / 132 read at entry (a CTA reads before
  its own increment, so at most 131 of this call's increments are visible: formally exact, as WP-B1c FXSS), parity
  c & 1. The combine of call c zeroes the rows it read after reading them; the kernel boundary orders that before call
  c + 2's reds. No in-call ordering needed.
- End: consumers' last flush -> mbarrier arrive (CTA-scope release) -> signaler (warp 10) waits (acquire), then
  red.release.gpu.add(ctr, 1) (MEMBAR.GPU: no TMA loads in flight any more at that point, ~0.4 us). Combine: CTA t < M
  (token t), after its own work: one thread ld.acquire polls ctr >= 132 (c + 1), CTA barrier, 256 threads x 16 rows:
  np <= 8 pairs' FP32 rows (16 KB each, __ldcg) loaded together, z = y * 2^14 gs2[e] * nw1, sums of squares
  (shuffles + one barrier for all pairs), out = sum_q w_q rsqrt(ss_q / 4096 + 1e-8) z_q -> BF16, then zero the rows.
  Expected chain ~2.3-2.8 us (v1002: 2.83).
- FP32 accumulation everywhere (MMA f32, red.add.f32, f32 combine), BF16 MMA operands, BF16 intermediate (as SGLang).

### 0.6 Shared memory (227 KB max)
ring 4 x 35840 = 143360; x_res 8 x 8208 = 65664; staging 8192; h 14 units x 512 = 7168; bars + plan ~1 KB:
~225.4 KB. One producer copy per stage (the cheapest producer path; WP-EDGE2's item-start instruction overhead is gone).

### 0.7 Expected per-CTA budget (bs8, E = 9)
first data ~3.0 us (ids + plan ~1.2, first copy + DRAM ~1.8) + 52.4 stages at the ring rate (~245 MB / 4.35-4.45 TB/s
= 55-56 us over the whole GPU) + unit/piece transitions ~0.3 + end ~2.5 = ~61-62 us of kernel span vs ~64.5 for v1002
in the same accounting (-4..-6%). Riskiest parts, in order: (1) red.add of ~4.7 MB into L2 under the stream (cost and
end visibility latency), (2) DRAM per-SM headroom for the static split, (3) stage_down's real cost (C/M; a new body),
(4) a new kernel's codegen / cold code.

### 0.8 Cheapest experiment that bounds the risky part (G1, dev/wpfuse_g1.py, one Modal job)
132 CTAs stream 35840 B stages through a 4-slot TMA ring (one bulk copy per stage, as in the design) for 52 stages each;
consumers (a) only release, (b) run the real gate/up stage body, (c) run stage_down; optionally they issue the design's
register red.adds (T = 1, 2, 8; 4 or 8 flushes per CTA in the last third, rotated or not) into a 2 x 64 x 4096 FP32
buffer, 15 CTAs per expert group. Measured per mode: wall time (graph replay after a 512 MB L2 flush), per-CTA end spread,
fence cost and last-signal -> seen latency with a red.release counter + poller, per-SM rate at 132 / 99 / 66 / 44 CTAs,
compute-only stage cycles of both bodies. Kill if the reds cost > ~2% of the stream.

## Implementation (2026-11-01 ~06:00 UTC)
- dev/moe_v1800.py = v1002 + MOE_FUSE (kernel moef, M 2..8, opt-in; MOE_FUSE_TR = per-CTA stamps into state["trc"]).
  Default build: cubin byte-identical to v1002 (local NVRTC 13, all 16 kernels). moef: 4784 SASS instructions, 144
  registers, no spills (gate/up stage loop 2x unrolled as the (4,1,1) loop; combine stages z in the idle ring instead of
  128 live registers, which had spilled 76 B).
- dev/wpfuse_layout.py (CPU, random NVFP4 bytes): every word / S6O scale field moef reads decodes to the codes / scale
  bytes its addressing formulas assign: wf13 0 mismatches, wf2 0 mismatches.
- dev/wpfuse_cc.py: local compile check of a microbenchmark source; dev/wpfuse_g1.py (G1), dev/wpfuse_trace.py
  (moef timeline), dev/wpfuse_g1.sh = run 1 (G1 + the G2 smoke / check_mt / trace / bench2 bs8 + bs4 in one job,
  because the shared GPU queue is serialized).

## G1 result (run 1, Modal H200, dev/wpfuse_g1.py, 132 CTAs x 52 stages of 35840 B = 246 MB, graph replay after a 512 MB
## L2 read, 30 reps x 2 rounds in opposite order; full log: scratchpad wpfuse/g1_run1.log)

| mode | us (round 0 / 1) | vs ring |
|---|---|---|
| ring (consumers wait + release) | 58.22 / 58.14 (4.23 TB/s) | 1 |
| ring + reds T=1, 4 flushes (16 KB / CTA) | 58.91 / 59.06 | 1.014 |
| ring + reds T=2, 4 flushes (32 KB / CTA = the design's volume) | 59.89 / 59.63 | **1.027** |
| same, quarter order not rotated | 60.10 / 59.73 | 1.030 |
| T=2, 8 flushes | 63.02 / 62.88 | 1.082 |
| T=8, 4 / 8 flushes | 64.1 / 71.1 | 1.10 / 1.22 |
| gu (real (4,1,1) gate/up body) / dn (stage_down) / mix (2/3 gu, 1/3 dn) | 65.1 / 65.9 / 65.3 | 1.118 / 1.133 / 1.122 |
| mix + reds T=2 x 4 / T=2 x 8 / T=8 x 8 | 69.4 / 74.1 / 78.9 | 1.192 / 1.273 / 1.356 |
| gu / dn compute only (resident slots) | 58.6 / 59.6 (1890 / 1930 cyc per stage) | |
| ring at 99 / 66 / 44 / 33 CTAs (same bytes per CTA) | 42 / 63 / 89 / 93 GB/s per SM | |

- **Red cost: KILL criterion met (marginally) for the planned register-direct flush.** The design's volume (T ~ 2, 4 flushes)
  costs +2.7% on the bare stream (+1.6 us), +6.3% with compute; it scales with the red sector count (T=8: +10%), not
  with address contention (rotation changes nothing). Mechanism (consistent with WP-L2's prefetch result): the 32-B red
  requests share the SM's memory-request path with the TMA ring (each flush is ~256 sector requests, ~1 stage of the
  ring's request count). Fence after the stream 0.35-0.45 us, last arrival -> seen 0.2-0.45 us, combine read of 3 x 16 KB
  1.6-2.4 us (end chain ~2.5-3 us, as v1002's 2.8).
- Per-SM headroom is large: one SM pulls up to ~90 GB/s (2.8x its share at 132 CTAs), so the static split's one-unit
  imbalance costs ~nothing (even G = 64 units would balance). Static end spread with compute: max - median ~1 us.
- The microbenchmark's compute modes run at C/M ~1 (1890 cyc compute-only per stage vs WP-16W's 1429 for the same body):
  its runtime-switched consumer loop is slower than the real kernel's, so the coupled numbers are pessimistic; the
  ring-only red numbers are the clean ones.
- Next (G1b): flush through shared memory as 512-B vector reds, or as TMA bulk reductions (cp.reduce.async.bulk .add.f32,
  one 4 KB op per token column per quarter), which use the copy engine's large transactions instead of 32-B requests.
- G2 smoke passed in run 1 (bs 2 / 4 / 8 rel-L2 2.54e-3 / 2.51e-3 / 2.54e-3); the rest of run 1 was cut by a bad STOP
  grep (it matched "error" in "relative L2 error"); run 2 repeats check_mt / trace / bench2 with timing bounds.

## Run 2 (Modal H200; log scratchpad wpfuse/run2.log): G2 prototype + G1b
Correctness: hang1 bs 2/4/8 ok (2.54e-3 / 2.51e-3 / 2.54e-3; V4 build identical errors); check_mt M 2..8 normal +
skewed, 3 seeds, 2 calls: 84 checks, 0 failures, worst rel-L2 2.63e-3.

bench2 jm (100 reps bs8, 100 bs4; B2_SHARE_EQ):
| | bs8 us | vs v1002 | bs4 us | vs v1002 |
|---|---|---|---|---|
| v1002 / v1002b | 62.6 / 62.8 | 1 / 0.9995 | 44.6 / 45.3 | 1 / 0.997 |
| fz (MOE_FUSE, scalar register reds) x2 | 70.1 / 69.6 | 0.903 / 0.900 | 49.0 / 48.7 | 0.917 / 0.923 |
| fzv4 (MOE_FUSE_V4: 512-B v4 reds) | 66.7 | **0.926** | 45.6 | **0.984** |
Bounds (B2_NOVERIFY, 40 reps, same job): fz 0.898, NORED 0.914, NOWAIT (combine without the arrival wait) 0.934.

Trace (MOE_FUSE_TR, 10 jm bs8 seeds, medians; us from the first CTA entry): producer's first copy issued 2.97,
x ready 2.66, first weight data 4.88 (v1002: 5.37), piece-0 gate/up done 39.1, consumers done med 59.8 (max - med
2-5 us: E 8 seed 56.8 / 59.7, E 13 seed 88.8 / 94.0), last release -> seen 0.4, seen -> combine done 1.8-2.7.
- The static split does NOT give a small tail: consumers-done spread max - median is 2-3 us at E 8-10 (v1002's tail
  1.67), and the end chain (~2.5 us) sits behind the slowest CTA (NOWAIT +3.6%).
- Startup did not improve: the first copy waits ~3 us for ids + 13 mbarrier inits on one lane + warp 0's serial plan +
  a CTA barrier + cold producer code. moe1t's first copy is at ~0.6 us.
- G1b flush mechanisms (ring-only, vs ring): scalar register reds T=1 / 2 / 8 x 4 flushes +2.6 / +4.4 / +10.7%;
  per-warp 512-B red.v4 (MOE_FUSE_V4) +0.0 / +0.8 / +6.2%; TMA bulk reduce (cp.reduce.async.bulk .add.f32, 4 KB per
  column) +0.8 / +1.7 / +6.5%. With compute (mix): scalar +8.6%, v4 +1.5%, bulk +1.2% at T=2. v4 is the flush to use
  (needs only 4.2 KB of smem); the reductions are cheap now, the kill criterion is lifted for the v4 flush.
- G2 status at bs8: x0.926 (FAIL vs the >= v1002 gate so far); deficits are startup, tail and end, not the stream.

## Run 3 (log scratchpad wpfuse/run3.log): fast start, v4 flush default, MOE_FUSE1 (M = 1 on moef)
Changes: the producer warp loads the routing ids before the CTA barrier and computes the plan itself (published on
mbarrier 13), the barrier inits do not wait for any global load, every role reads the call index itself; v4 flush is
the default (MOE_FUSE_SCL = run 1's scalar reds); MOE_FUSE1 routes M = 1 to moef.
Correctness: hang1 bs 1/2/4/8 ok; check_mt M 1..8 normal + skewed, 3 seeds, 2 calls: 96/96, worst 2.73e-3 (M 1 2.73e-3).
(The trace's "FAIL" marks were a trace-script bug: the cloned graph inputs went out of scope before the stamped replay;
fixed in run 4. bench2's own verification of the first reps passed.)

bench2 jm (100 reps, two passes in opposite order for bs1; same job):
| | bs1 pass A / B | bs8 (60 reps) |
|---|---|---|
| v1002 | 24.7 / 24.7 us | 62.8 us |
| v1002b | 0.998 / (1.0024) | 0.998 |
| v1301 FXSS (current best bs1) | 1.008 / 1.004 (vs v1002) | |
| moef (f1 / fz) x2 | **0.893 / 0.885** (27.6 / 27.9 us); pass B 27.6 / 27.7 | **0.922 / 0.926** (68.7 / 67.1 us) |

Trace (fast start), us from the first CTA entry, medians over seeds:
| | bs1 (12 seeds, ne 2-5) | bs8 (8 seeds) |
|---|---|---|
| producer's first copy issued | 1.74 (run 2: 2.97) | 1.98 |
| first weight data, median / max over CTAs | 3.52 / 4.2-10.0 | 3.70 / 4.8-10.2 |
| consumers done, max - median (the tail) | 2.76 (0.8-4.4) | 3.5-5.2 (ok seeds) |
| last release -> combine's seen | 0.3-0.5 | 0.4-0.5 |
| seen -> combine done (CTA t reads np x 16 KB FP32) | 1.9-2.3 | 1.8-2.9 |
| per-seed replay vs v1002 | 0.89-0.93 | 0.95-0.975 (pairs ~10), 0.87-0.90 (pairs 20-24) |
- The fast start works for the median CTA (first copy -1.2 us), but the first-data spread over CTAs is 2-6 us: each
  CTA's 4 initial copies (18.9 MB in all) are queued at once and some CTAs' stage 0 is served after others' stages 1-3.
  The consumers-done tail follows that spread seed by seed (bs1: +0.5 / +0.8, +2.1 / +1.5, +2.6 / +3.1, +4.7 / +4.4 us):
  a static split cannot recover a late start. v1002's queue absorbs it.
- High-T routings lose more (bs8 0.87-0.90 at 20-24 pairs vs 0.95-0.975 at ~10): the partial flush is 16 KB x T per
  piece and red bytes cost about as much as streamed bytes on the SM's request path (G1b: T = 8 x 4 flushes +6.2%);
  T-heavy experts' CTAs become the slowest.
- bs1: moef also pays a longer end than moe1t-FXSS (re-reading ne x 16 KB of FP32 y for the norm: ~2.3 us after the
  last arrival vs ~1 us), which the fused K-split cannot avoid.
Run 4 tests MOE_FUSE_S0W (the rest of each CTA's initial burst only after its x + stage 0 landed) and MOE_FUSE_TW=g
(token-weighted split: unit weight 384 + g T_e in 1/64-stage units), with traces.

## Run 4 (log scratchpad wpfuse/run4.log): tail levers
Correctness: hang1 bs 1/2/8 ok for S0W and S0W + TW4; check_mt M 1..8 (S0W + TW4) 96/96, worst 2.73e-3.

| bench2 jm (same job) | bs8 (60 reps) | bs1 (60 reps) |
|---|---|---|
| v1002 / v1002b | 61.9 / 61.7 us (1 / 1.0006) | 24.0 us |
| v1301 FXSS | | 1.0048 |
| moef (fz / f1) | 0.9197 | 0.8840 |
| + MOE_FUSE_S0W | 0.9154 | 0.8707 |
| + S0W + MOE_FUSE_TW=4 / =8 | 0.8814 / 0.8889 | |

Trace (inputs now kept alive: every seed checks ok):
- S0W does what it should for the start: bs1 first-data max over CTAs 3.4-3.6 us (run 3: 4.2-10), consumers-done tail
  1.0 us (run 3: 2.8). It still does not move the wall time (each CTA's stage 1 now waits ~0.5 us after stage 0).
- bs8 keeps a ~5 us tail even in step: the slowest CTAs cluster on specific experts (consecutive CTA ids, e.g. 48-59,
  0-16), the high-T experts whose pieces flush 16 KB x T. TW4 / TW8 do not fix it: the weighted plan's serial loops on
  the producer lane push the first copy from 1.97 to 3.1 us, and the tail stays at 5.7 us.
- bs1 budget (seed 73235, ne 3, S0W): first data 3.2 us, 18 stages at 1.12 us/stage (moe1t: 1.088), last consumers done
  24.7, last release 25.1, seen 25.5, combine done 27.8. moe1t (v1002) is ~23.7 us on the same seed: moef loses ~0.6 us
  in the stream, ~0.3 at the start, ~1.3 in the tail and ~2 us at the end (global arrival + re-reading ne x 16 KB of
  FP32 y for the norm vs moe1t's per-CTA output rows).

## Verdict: G2 KILLED (2026-11-01 ~07:30 UTC)
Best moef after ~1 day of tuning: bs8 x0.915-0.926 vs v1002 (gate: >= 1), bs4 x0.984, bs1 x0.871-0.893 (gate:
>= v1301 FXSS, which measured x1.004-1.008). Correct throughout (check_mt M 1..8 normal + skewed, 96/96, worst rel-L2
2.73e-3, the same as v1002's). Nothing to integrate. dev/moe_v1800.py stays as an opt-in reference: the default build is
cubin-identical to v1002 (local NVRTC 13, all 16 kernels, re-checked after the last edit).

Why it lost (for future static / split-K designs):
1. The dynamic queue's absorption of per-CTA variation is worth more than the static plan's savings at these sizes.
   Every per-CTA delay converts 1:1 into the tail under a static split: the initial-burst ordering (2-6 us first-data
   spread, fixed by S0W), per-expert work differences (high-T flush bytes: ~5 us at bs8) and per-SM / per-region DRAM
   variation (wpbcps.md 6c). v1002's queue gives the late CTAs fewer items. Its item edges are mostly hidden by the
   stream anyway, so removing them buys little.
2. The fused K-split's end is structurally longer. The norm needs the whole y row, so the combine must wait for a global
   arrival and re-read the FP32 rows: 0.3-0.5 us arrival -> seen + 1.8-2.9 us combine. moe1t-FXSS writes per-CTA output
   rows ~1 us after its last stage; v1002's main kernel end is 2.8 us.
3. The startup is not better than v1002's (first data 3.2-3.5 us vs 5.4) by enough to pay for 1 + 2. Cold code and the
   DRAM ramp set ~3 us regardless of the plan.

### Cost of red bytes vs streamed bytes (G1 / G1b, ring-only, 132 CTAs x 1.86 MB per CTA streamed in 57.9 us = ~31 ns
### per streamed KB per CTA; the bound for any split-K partial flush on this machine)
| flush mechanism (per CTA: 4 flushes x T x 4 KB) | T=1 (16 KB) | T=2 (32 KB) | T=8 (128 KB) | ns per red KB per CTA |
|---|---|---|---|---|
| scalar red.global.add.f32 from registers (32-B requests) | +1.4..2.6% | +2.7..4.4% | +10..10.7% | ~45-95 (1.5-3x a streamed byte) |
| per-warp 512-B red.global.add.v4.f32 (smem-staged, 4.2 KB) | +0.0% | +0.8% | +6.2% | ~14-28 (0.5-0.9x) |
| TMA bulk reduce cp.reduce.async.bulk .add.f32 (4 KB per op) | +0.8% | +1.7% | +6.5% | ~30 (~1x) |
With compute on the same SMs (mix mode, T=2): scalar +8.6%, v4 +1.5%, bulk +1.2%. Rule of thumb: red bytes cost like
streamed bytes on the SM's request path. A split-K design pays ~(partial bytes / weight bytes) of the stream, and for
the slowest CTAs, not the average. The address pattern (rotating which rows ~15 CTAs hit) does not matter.
Other G1 facts: one SM pulls up to ~90 GB/s (2.8x its share at 132 CTAs); fence (red.release) at the end of the stream
0.35-0.45 us; last arrival -> poller sees 0.2-0.5 us.

### Files and defines
- dev/moe_v1800.py: MOE_FUSE (moef for M 2..8), MOE_FUSE1 (M = 1 on moef), MOE_FUSE_SCL (scalar register reds instead
  of the default v4 flush), MOE_FUSE_S0W (start in step), MOE_FUSE_TW=g (token-weighted split), MOE_FUSE_PCOST=B
  (boundary-weighted split, not measured), MOE_FUSE_NORED / NOWAIT / NOROT (timing bounds), MOE_FUSE_TR (stamps).
  The extra prepare() layouts wf13 / wf2 (+880 MB) exist only with MOE_FUSE / MOE_FUSE1.
- dev/wpfuse_g1.py + dev/wpfuse_g1.sh (G1 / G1b microbenchmark), dev/wpfuse_run2.sh / run3.sh / run4.sh, dev/wpfuse_trace.py
  (per-CTA timeline + same-seed reference time), dev/wpfuse_layout.py (CPU layout check), dev/wpfuse_cc.py (local compile),
  dev/wpfuse_clu.py (cluster-occupancy probe for a 2-CTA DSMEM pre-reduction; written, not run, moot after the kill).
