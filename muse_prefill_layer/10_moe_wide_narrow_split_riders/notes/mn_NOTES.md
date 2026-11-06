# mn worker: more consumer warps per SMSP on the narrow NVFP4 expert items (2026-10-31)

Idea under test: the narrow (N <= 32) consumer loop is issue-bound at 2 consumer warps per SMSP (ms: IPC 0.69-0.82,
~1.3 cycles per instruction), so 3-4 consumer warpgroups per CTA should raise IPC and cut the per-item time.
Scope (orchestrator, 2026-10-31): mx owns the W/N kernel split (plan, launch, concurrency, combine) and kernel W;
mn owns kernel N's internals (consumer WG count, register split, narrow decode loop).

## 0. Prior data
- moe4's narrow kernel (moe_narrow_persistent): 640 threads = two half-engines, each a producer warp + publisher warp +
  2 consumer WGs, i.e. **16 consumer warps = 4 per SMSP**, consumers at 112 registers (setmaxnreg; launch 96).
  Per item vs the incumbent on 32 SMs: N=8 (mma.sync) -8.2%, N=16 +4.6%, N=32 +3.3%, N=64 +7%. Issue efficiency
  rose 67 -> 74% (N=8) and 55 -> 69% (N=32), but the N=32 code had +42% instructions (KPS 2, 112-register remat).
- Register arithmetic (found here): Hopper's register file is 16384 per SMSP and warps go to SMSPs round-robin, so an
  extra producer *warp* beside 16 consumer warps puts 5 warps on SMSP 0 and caps every thread at 96 registers (ptxas
  spills N=16 at 96). Options for 4 consumer WGs: producer WG + setmaxnreg (consumers <= 112, moe4) or no producer
  warp at all (512 threads x 128 registers: the last consumer warp to release a slot issues its refill, "P2").

## 1. Microbenchmark (ub/nw_ub.cu)
Production narrow loop (GROUP4: 2 k16 per commit group, wait<1>, NACC 4 for N <= 32, NACC 2 / KPS 2 for N = 64; or
the MAT_SMALL N=8 mma.sync path), NWG = 1..4 consumer WGs, each WG 64 weight rows of one shared X tile (super-item of
NWG x 64 rows), TMA ring (4 slots, full/empty mbarriers, all-lane release after the next stage's first wait) with W
from L2 (WSRC 0) or HBM (WSRC 1); FEED 0 = smem-resident. Cycles per k16 per 128 rows (= super-item x 2 / NWG).

### 1.1 Round 1 (logs/ub1.log; 132 CTAs, warp median, items 1..5 of 6; cycles per k16 per 128 rows, us at 1.65 GHz)
p0 = producer WG + setmaxnreg (2 WG: 40/232, 3 WG: 24/160, 4 WG: 24/112); s = smem-resident; c = constant A (no
decode) with the TMA ring. INVALID: every "p2" build (no producer warp, last releaser refills) is C7513 (the atomic +
TMA issue between the wgmma groups serializes every wgmma), and 4 WGs at 112 registers is C7512 for N = 32 and 64.
| N | TMA 2 WG | TMA 3 WG | TMA 4 WG | smem 2 WG | smem 4 WG | const-A 2 WG | const-A 4 WG |
|---|---|---|---|---|---|---|---|
| 8 wgmma | 61.1 (9.49 us) | 55.7 (-8.8%) | 53.9 (-11.8%) | 52.5 | 49.2 (-6.3%) | 52.6 | 34.7 |
| 8 mma.sync (MAT_SMALL) | 59.6 (9.25) | | 54.9 (-7.9%) | | | | |
| 16 | 65.7 (10.19) | 59.6 (-9.3%) | 58.0 (-11.7%) | 60.2 | 54.9 (-8.8%) | 53.2 | 35.8 |
| 32 | 73.7 (11.43) | 68.5 (-7.1%) | C7512 | 67.7 | 62.8 (-7.2%) | 56.7 | 38.5 |
| 64 | 107.4 (16.67) | 91.8 (-14.5%) | C7512 | 92.4 | 83.1 (-10.1%) | 100.4 | 70.9 |
Main-loop SASS per k16 per warp: 20.5-21.0 instructions at every WG count (LOP3 5.2, SHF 1.2, PRMT 1, IMAD 4.1,
HMUL2 4, LDS 0.5, + HGMMA / fence / descriptor ops), i.e. the same work per row. SMSP IPC in the loop (2 x instr / cycles):
TMA N=16 2 WG 0.64, 3 WG 0.70, 4 WG 0.72; smem N=16 2 WG 0.67, 4 WG 0.74.
=> Doubling the consumer warps per SMSP raises IPC by only ~10%: the loop is NOT bound by too few warps to issue from.
Best gain at N <= 32: -11.8% (N=8, TMA) -- under the 20% kill criterion. (CORRECTED in 1.3: the constant-A rows
above are "p2" builds; with a real producer warp the decode-free TMA loop runs at the tensor floor, the feed is not
what binds.)

### 1.2 ncu of the microbenchmark (logs/ncu_ub1.log; one launch, real clocks; s = smem-resident, w = TMA p0)
| build | SMSP IPC | issue active | warps eligible / active per clk | ALU | FMA-heavy | FMA-lite | tensor | loop stall mix (top) |
|---|---|---|---|---|---|---|---|---|
| s8 2 WG | 0.70 | 70.5% | 1.05 / 2 | 52.6% | 56.2% | 25.9% | 12.7% | selected 39, not_sel 20, wait 11, dispatch 11, math 10 |
| s8 4 WG | 0.78 | 77.6% | 1.98 / 4 | 57.6% | 62.1% | 28.7% | 14.1% | not_sel 32, selected 22, math 19, dispatch 11, wait 6 |
| s16 2 WG | 0.63 | 63.2% | 0.94 / 2 | 47.3% | 49.7% | 22.9% | 22.2% | selected 34, not_sel 17, wait 14, dispatch 12, math 12 |
| s16 4 WG | 0.71 | 71.3% | 1.81 / 4 | 52.8% | 56.5% | 26.2% | 25.4% | not_sel 29, math 19, selected 18, dispatch 15, wait 8 |
| s32 2 WG | 0.58 | 57.7% | 0.81 / 2 | 45.7% | 43.1% | 20.5% | 38.5% | selected 30, dispatch 16, wait 16, not_sel 13, math 10 |
| s32 4 WG | 0.63 | 63.5% | 1.58 / 4 | 49.1% | 48.2% | 23.1% | 43.7% | not_sel 24, dispatch 20, math 19, selected 16, wait 9 |
| w16 2 WG (TMA) | 0.62 | 61.9% | 0.90 / 2.3 | 47.2% | 47.6% | 21.4% | 20.7% | selected 31, wait 14, not_sel 14, dispatch 11, math 9, long_sb 4-6 |
| w16 4 WG (TMA) | 0.71 | 70.7% | 1.73 / 4.3 | 53.6% | 55.0% | 25.0% | 24.3% | not_sel 26, selected 18, math 17, dispatch 14, wait 7 |
=> With 4 warps per SMSP there are ~2 eligible warps every cycle on average, yet the SMSP issues on only 71-78% of
cycles: eligibility is bursty and the added stall is pipe throttle ("math" 10 -> 19%) and dispatch, not latency.
FMA-heavy (IMAD shifts + HMUL2 scale multiplies) and ALU (LOP3/SHF/PRMT) run at 55-62% of peak; with the wgmma
interference moe2 measured (int/fp16 pipes slowed while a wgmma runs) that is close to what they deliver. The
narrow loop is pipe/dispatch-bound with ~40 slow-class instructions per SMSP per 128-row k16, not warp-starved:
the issue floor at IPC 1 would be ~40 cycles (N=8/16), the measured 4-WG loop is 49-55.

### 1.3 Round 2: ring depth, HBM-streamed W, CTA count (logs/ub2.log; p0 builds; L = W from L2, H = W from HBM,
### distinct per CTA and item, L2 flushed; C = constant A + TMA ring from L2; cycles per k16 per 128 rows)
| N, WGs | L 4 slots | L 6 / 8-9 slots | H 4 slots | H 6 / 9 slots | C 4 slots | H at 33 / 66 CTAs |
|---|---|---|---|---|---|---|
| 8, 2 | 61.1 | 61.8 / 61.8 | 66.2 | 65.0 / 64.0 | 29.1 | |
| 8 mma.sync, 2 | 59.6 (round 1) | 59.6 (9) | 66.1 | 63.6 (9) | | |
| 8, 4 | 53.9 | 53.7 (5) | 64.3 | 63.7 (5) | 25.5 | |
| 16, 2 | 65.7 | 66.9 / 64.9 | 66.9 | 66.2 / 65.2 | 32.2 | 66.6 / 66.1 |
| 16, 4 | 58.0 | | 61.8 | | 25.9 | - / 58.1 |
| 32, 2 | 73.7 | 73.8 | 73.3 | 73.7 | 36.6 | |
| 64, 2 | 107.6 | 102.4 / 102.3 | 107.9 | 102.2 / 102.3 | 64.2 | |
1. The decode-free loop on the real TMA ring (C) runs at the tensor floor (29.1 / 32.2 / 36.6 / 64.2 vs smem-resident
   26.1 / 28.1 / 33.5 / 64.2 in ms): the per-SM feed (ring depth x latency) does NOT limit narrow items. Round 1's
   52-57-cycle constant-A rows were an artifact of the producer-free p2 design (refill only after the slowest warp's
   release + atomics), not of the feed.
2. Ring depth: 4 -> 6 -> 9 slots changes nothing for N <= 32 (W from L2 or HBM); N = 64 gains 5% from 6 slots.
3. HBM: streaming W from HBM costs +8% at N = 8 (2 WG) and +2% at N = 16, nothing at N >= 32, and the N = 16 cost is
   the same at 33, 66 and 132 CTAs (per-SM latency, not bandwidth). But 4 WGs at N = 8 / 16 from HBM: 64.3 / 61.8 vs
   53.9 / 58.0 from L2 -- the aggregate HBM ceiling: 132 x 294,912 B per 9.97 / 9.58 us = 3.9 / 4.06 TB/s; at 66 CTAs
   the 4-WG N = 16 row is back at 58.1.

## 2. Are narrow items feed-bound? (orchestrator question, 2026-10-31)
Weight bytes per 128-row item: 128 x 4096 x 0.5625 = 294,912 B (64 k-blocks x 4608). Measured HBM ceiling for this
stream (ub, 132 CTAs): ~4.0 TB/s = 30.3 GB/s per SM -> **9.7 us per item when every SM streams narrow items at once**
(1.65 GHz: 62.6 cycles per k16). Per item time vs that floor:
| | g8 | g16 | g32 | g64 | g96 |
|---|---|---|---|---|---|
| s_b in graph (mx r1a, mixed queue) | 9.72 | 10.45 | 11.5 | 15.97 | 19.1 |
| mx sw0t kernel N (all-narrow phase after W) | 9.45-9.51 | 10.35 | 11.25 | 15.4 | 18.1-18.4 |
| aggregate W rate if all 132 SMs ran this width | 4.1 TB/s | 3.75 | 3.46 | 2.5 | 2.1 |
| ub 2-WG TMA loop (L2 / HBM) | 9.49 / 10.27 | 10.19 / 10.38 | 11.43 / 11.38 | 16.7 / 16.7 | |
| ub 3-WG / 4-WG TMA loop from L2 | 8.65 / 8.36 | 9.25 / 8.99 | 10.62 / C7512 | 14.25 / - | |
- Per SM, narrow items are NOT feed-bound today: the per-SM feed sustains the tensor floor (1.3.1), the item time
  does not depend on how many SMs run narrow items (mx 1, ub 33/66/132 CTAs), and the in-graph item times equal the
  ub's SM-local 2-WG loop (decode-bound, section 1.2).
- In aggregate they are AT the edge: N = 8 items already need 4.1 TB/s when all SMs run them, N = 16 3.75 TB/s.
  Today's all-narrow phases (mx: 2.8-3.0 TB/s measured) stay below the ceiling only because ~100 of 132 SMs are
  narrow at a time. Any SM-local speedup of N <= 16 items (more WGs: -9..-12%) turns those phases HBM-bound:
  4-WG N = 8 from HBM gives back all but 3% of its gain at 132 CTAs.
- Totals at T=753 draw 11 case 0: 14 tiles x 96 items x 294,912 B = 396 MB per MoE call -> 99 us at 4.0 TB/s,
  vs a 243 us span; the narrow tiles alone 255 MB -> 64 us at 4.0 TB/s vs ~82 us/CTA of narrow SM time (sw0t).
  The MoE as a whole is SM-time-bound (149 wide + 82-88 narrow SM-us per CTA); HBM binds only where narrow items
  run without wide items beside them.
=> The faster narrow loop is worth ~-9..-12% (N <= 16) / -7% (N = 32) / -14% (N = 64) per item only where narrow
items share HBM with wide items on other SMs. In mx's split as launched (W first, N on each SM after W's CTA exits)
N's phase is mostly all-narrow: expect N = 8 to gain ~0, N = 16 ~-6% (capped at ~9.7 us), N >= 32 most of the ub
gain. The lever for the rest is W/N concurrency (narrow weight streaming overlapped with wide compute), as the
orchestrator suspected -- but the narrow items are decode-bound, not feed-bound, until they get faster.

## 3. 3-WG kernel N for mx's split ("R3"; cu/n3.cuh, build.py n3*, plugged into mx/split.py's PART == 2 launch)
Orchestrator direction (2026-10-31): keep the -8.8..-11.8% (3-4 WG, TMA) per-item win; build kernel N as a 3-WG
variant against mx's interface (mx NOTES 4.1) and measure it inside mx's split vs s_b and vs mx's 2-WG split (sw0).
Design (4 WGs are out: 112 registers -> C7512 at N = 32 / 64, and the 4-WG N <= 16 gain is HBM-capped, section 2):
- moe_n3_persistent<FINE>: 512 threads = producer WG (setmaxnreg.dec 40: warp 0 producer, warp 1 publisher) + 3
  consumer WGs (setmaxnreg.inc 152). 4 ring slots of 44032 B (largest stage: N = 32, 4 x (4096 + 3 x 2304)), 3
  per-WG staging buffers of 13824 B.
- Work unit = 3 consecutive 64-row halves of one GEMM's weight rows for one token tile: one stage = the tile's X once
  + 3 W halves (codes 2048 + scales 256 B per half per k-block, cut out of the unchanged 4608-B production blocks).
  GEMM1: 43 units per tile (128 halves; the last unit has 2), GEMM2: 22 (64 halves; the last has 1). Idle WGs of a
  partial unit still wait / release every stage. Order: GEMM1 units 0..21 across tiles, 22..42 tile-major, GEMM2
  tile-major (s_b's MAT_SMALL shape).
- Consumer WG w of unit s runs half hq = 3s + w (n-tile hq / 2, rows 64 (hq & 1) ..) with the production narrow
  loop: N = 8 mma.sync (NACC 4), 16 / 32 GROUP4 NACC 4, 64 GROUP4 NACC 2 (KPS 2), 96 one k16 per group NACC 1; the
  same accumulation split and the same epilogue arithmetic per element as s_b (bit-identical by construction). A
  128-token tile runs as two 64-token sub-tiles (N = 64 path: NACC 2 by k-block parity = N = 128's split).
- Producer = the whole warp, one lane per bulk copy (1 X copy, or 2 for a sub-tile, + 2 per half per k-block: <= 25
  per stage in one warp instruction), L2 prefetch of the next stage's pieces by the same lanes. GEMM2 units wait for
  done[tile] >= 128 per sub-tile (GEMM1 halves count one each).
- Epilogue: per WG staging (gate: tile-layout chunks q = 4h + w; down: 64 columns per token, pitch 144) and a
  sequence-numbered 8-entry job queue (3 producers); the publisher copies a gate half as one 64-B piece per token
  (the half's 4 chunks are contiguous after the 128B swizzle: offset ((h ^ (tok >> 2)) & 1) x 64) and a down half
  as 128 B per token, then fence.acq_rel.gpu + done++ for gate halves.

### 3.1 Build checks (nvcc 13, sass.sh; uni/ = reduced compile unit: the MoE namespace + one kernel-N instantiation,
### ~1 s per probe instead of 5 min for the whole file)
- First build: no C75xx, but 44 / 96 B spills (producer WG at setmaxnreg 40: its per-lane copy state) and the wgmma
  main loops at 25.3-25.6 instructions per k16 (s_b: 20.6) with 2 R2UR + 64-bit descriptor adds per HGMMA: `slot`
  lived in a vector register. Probes (uni1.log): removing the idle-WG branch, the virtual-tile table, wg's thread-id
  derivation or the staging-wait spins changed nothing. Cause (uni2.log): no converged point after the consumer's
  item-ring mbarrier spin (the SASS had BRA.DIV guards before every SHFL); production has named_bar_sync(1, 256)
  right there. With a per-WG bar.sync (.aligned) after the item read: loops 20.38 (N=16/32) / 20.44 (N=8 mma.sync) /
  23.38 (N=64) / 20.75, i.e. s_b's 20.62 / 20.62 / 23.5. Lesson for any kernel-N edit: every consumer path needs an
  .aligned barrier between a divergent spin and the next slot/descriptor use, or ptxas leaves the uniform datapath.
- The barrier cost 16 B of spills (2 ints of outer-loop state); folding the item-ring k / parity into one counter
  removes them (uni4.log). Folding slot / parity into one stage counter the same way puts `slot` back into a vector
  register (uni3.log): kept as two variables. Producer WG at 56 registers (128 x 56 + 384 x 152 = 65536).

### 3.2 First production measurement (v1 = cu/n3_v1.cuh; logs/r1.log, r2.log; MoE-only, draw 11, cold L2)
| case | s_b | sw0 (mx 2-WG split) | n3 v1 |
|---|---|---|---|
| 0 | 260.4 | 270.2 (-3.6%) | 369.8 (-30%) |
| 1 | 261.4 | 271.3 (-3.6%) | 406.9 (-36%) |
Outputs bit-identical to s_b (errs 0.0 / 0.0, both cases): the per-element accumulation order is preserved.
Trace (r2, ana3.py): kernel N units ran ~22-23 us per 128-row equivalent for N = 8, 16 and 32 alike (sw0t: 9.0 /
9.9 / 10.9) -- width-independent = feed-bound. Cause: a bulk copy is UBLKCP with uniform-register operands, so the
per-lane copies (and per-piece L2 prefetches) compile to an ELECT / 5x R2UR / UBLKCP / BRA.U.ANY waterfall, one
copy per iteration: ~49 serialized iterations per stage (25 copies + 24 prefetches) ~ 2.1 us per stage. Production
pays the same waterfall for 2-3 copies per stage.
v2 (current cu/n3.cuh): one producer thread; a stage copies the unit's 1-2 whole n-tile weight blocks (each n-tile's
kps k-blocks are contiguous: one copy each) + X: 2-3 copies + 1-2 L2 prefetches per stage, production's cost. Each
WG reads its half with production's in-block offsets (warp q = 4h + w). Cost: the n-tile shared by two adjacent units
is copied into both (+33% L2 -> SM weight bytes per unit; HBM ~unchanged, the neighbor unit runs concurrently and hits
L2). Largest stage 53248 B (N = 32, KPS 4) -> 3 ring slots (n3); n3k2 = N = 32 at KPS 2 with 4 slots. Registers:
producer WG 56 / consumers 152 (32 / 160: the producer WG spills); V and the unit count are re-read from shared
memory per unit (at 152 registers ptxas otherwise spilled 3 loop invariants, 16 B).

### 3.3 v2 measurements (logs/r3a-r8; MoE-only, draw 11 case 0 unless noted; per-128-row-item times from item traces)
- Two hangs / faults found and fixed on the way: NSLOT 3 released slots on the wrong mbarrier (8 * N3_NSLOT offset;
  now fixed bars + 32), and threads 1-31 of the producer warp fell into the publisher branch (full-mask shuffles
  without lane 0: deadlock; I killed my own hung remote bench).
- Single-thread producer (2-3 copies + 2 prefetches per stage): 294.5 us (n3k2) -- still the per-copy waterfall:
  N = 32 at KPS 2 (twice the stages, same bytes) 17.9 vs 11.6 us at KPS 4 = ~1000 cycles per stage of producer cost.
- Warp producer (convergent warp: vote-exit mbarrier waits, lane-0 shuffles, copies under elect.sync with operands
  computed outside: straight-line UBLKCP, no BRA.U.ANY loops): n3 275.7, n3 without L2 prefetch 269.6, sw0 ~270.2,
  s_b 259.3 (r5). Outputs bit-identical to s_b in every run. Per 128-row item vs sw0t:
  | | g8 | g16 | g32 | g64 | g96 |
  | sw0t (2-WG kernel N) | 9.02 | 9.88 | 10.88 | 14.90 | 17.73 |
  | n3t (3-WG, whole n-tile blocks, 3 slots) | 9.13 | 10.35 | 11.05 | 17.41 | 19.07 |
  => no 3-WG gain in production; N = 64 / 96 worse.
- Microbenchmark round 3 (logs/ub3.log; n3's stage structure: WFULL = whole n-tile blocks, NSLOT 3, H = W from HBM
  for all 132 CTAs, cycles per k16 per 128 rows):
  | N | L 2 WG | L 3 WG pieces | L 3 WG WFULL 3 slots | H 2 WG | H 3 WG pieces | H 3 WG WFULL 4 / 3 slots |
  | 8 | 61.2 | 55.7 | 58.5 | 65.6 | 65.1 | 84.9 / 86.4 |
  | 16 | 65.7 | 59.6 | 62.6 | 66.9 | 63.4 | 84.7 / 86.0 |
  | 32 | 73.7 | 68.4 | 68.0 | 73.4 | 68.7 | 81.8 / 83.1 |
  | 64 | 107.6 | 91.8 | 90.2 | 107.9 | 91.8 | 90.1 / 93.1 |
  From HBM with all SMs streaming, 3 WGs gain nothing at N = 8 (the ~4 TB/s ceiling), -5% at 16, -6% at 32, -15% at
  64; whole-block copies (+33% bytes) are HBM-capped at ~85 cycles for N <= 32 (the ub's duplicate is pure HBM
  traffic; in production the neighbor unit's copy partly hits L2).
- Production stream rate (r5 traces, bytes copied into shared memory per 10 us, kernel-N phase 170-250 us): sw0t 2.8
  - 3.5 TB/s, n3t 3.5 - 4.0 TB/s (incl. the duplicated half-blocks): n3's narrow phase sits on the ~4 TB/s ceiling.
- W || N concurrency probe (n3c: kernel N launched first, waits for the gather, triggers W, keeps G CTAs from a
  per-width cost model; W fills the other SMs at once, waits for N only at its end): works and stays bit-identical --
  W ran on 89 SMs and N on 43 from t = 0 -- but 318.8 / 341.7 us (cases 0 / 1): N's units were no faster even with
  HBM headroom (g8 9.9, g16 11.0, g64 18.7 us) and N, under-provisioned, ended at 288 us vs W at 238.
- Per-unit wait instrumentation (r7, r8; trace builds): consumer full-barrier waits 10-13% of a unit at N = 8 / 16,
  18-25% at N = 64 / 96 (both the producer and the consumers wait: latency-bound feed), and 4 slots (n3k2) did not
  remove them -> the 3-WG kernel needs more feed per SM than 2 WGs (faster per row, + the duplicated half-block):
  ~31 B/clk/SM at N = 8 / 64 vs ~23-30 for the 2-WG kernel.
v3 (current cu/n3.cuh, N3_TM = 1): exact 64-row halves by TMA tensor copies -- 8 CUtensorMaps (GEMM x codes / scales x
kps 4 / 2) over the unchanged 4608-B blocks as [k-block][half][2048 or 256 B] in 8-byte elements, built per call in
the glue (driver entry point) and passed as a __grid_constant__ parameter: per stage 1 X copy + 2 tensor copies per
half (straight-line UTMALDG.3D), 4 slots of 44032 B, no L2 prefetch (N3_PF 0), the same bytes per row as 2 WGs.

### 3.4 v3 (tensor-map halves) measurements (logs/r9; draw 11 case 0; SASS: kernel N 0 spills, no C75xx, 14 UTMALDG,
### no copy waterfalls, main loops 20.4-20.9 / 23.3 / 24.4 instructions per k16 = s_b's)
| | s_b | sw0 (2-WG N) | n3 v3 (3-WG N, sequential split) | n3c v3 (concurrent, N first, G = 43) |
|---|---|---|---|---|
| MoE-only us | 259.9 | 269.4 (-3.5%) | 268.7 (-3.3%) | 313.3 |
Per 128-row item (n3t vs sw0t r5): g8 8.78 vs 9.02 (-2.7%), g16 9.68 vs 9.88 (-2.0%), g32 11.13 vs 10.88 (+2.3%),
g64 15.88 vs 14.90 (+6.6%), g96 17.98 vs 17.73 (+1.4%). Consumer full-barrier waits per unit fell to ~1 us (N <= 32)
and ~2.5-3 us (N = 64 / 96, still ~10%: a 30-KB stage in a fixed 44-KB slot leaves only ~90 KB in flight).
In the concurrent probe the narrow units run slower (g8 9.36, g16 10.12, g64 16.64) next to W's X-heavy stream, and
the item work is the same as in every other arrangement: W 19.5 + N 11.2 ms SM ~ s_b's 31.3 ms SM -> a perfect
partition is ~233 us + tails; N-first with surplus N CTAs exiting cannot rebalance W's coarse (40 us) tail.
Next probe (n3w): mx's order (W first, then N), W keeps nctas - G CTAs (G = share x N's estimated work fraction):
N starts at once on G SMs and takes every SM a finishing W CTA frees.

### 3.5 W-shrink concurrency (n3w) and case 1 (logs/r10, r11; MoE-only, draw 11, 5 reps; all outputs bit-identical)
n3w: mx's launch order, W keeps nctas - G CTAs (G = share x N's estimated SM-work fraction, per-width cost table);
N's CTAs start at once on the G freed SMs and on every SM a finishing W CTA leaves.
| case | s_b | sw0 | n3 (v3) | n3w (share 0.9) | n3w7 (0.75) | n3w10 (1.0) |
|---|---|---|---|---|---|---|
| 0 | 260.4 | 268.5 (-3.0%) | 268.2 (-2.9%) | 264.2 (-1.4%) | 267.4 | 268.7 |
| 1 | 260.6 | 272.6 (-4.4%) | 280.5 (-7.1%) | 273.5 (-4.7%) | 274.0 | 272.5 |
Case 1's extra loss is the 128-token tile (117 tokens): kernel N runs it as two 64-token sub-tiles (to keep the
N = 128 path's NACC 2 split bit-identical), which doubles the decode: 2 x 15.9 = 31.8 us per 128-row item vs 20.8 us as
one N = 128 item (sw0t), ~+8 us/CTA. A 3-WG N = 128 path needs 128 accumulator registers at NACC 2 (152 available):
bit-identity and 3 WGs are incompatible for N = 128 (NACC 1 would fit but changes the accumulation order).
Idle SM time per CTA (span - item SM-time): s_b 5.4 us, sw0t 11.8, n3t 12.2 (the W -> N hand-off + N's tail); the
3-WG units (1.5 items) also coarsen kernel N's tail (last-end spread 15-16 us vs 10.9 for sw0t at case 1).

## 4. In-graph check (ncu source-level, one graph replay of case 0; logs/st_sb0.*, st_n3_0.*; loop_ana.py)
"issue/warp" = selected samples / loop samples (fraction of time a warp issues); SMSP IPC ~ warps per SMSP x that.
| loop | s_b (2 consumer warps / SMSP) | kernel N v3 (3 / SMSP; ncu replays it alone) |
|---|---|---|
| N = 8 (mma.sync) | 0.36 -> IPC 0.72; selected 36, not_sel 19, wait 15, math 14, dispatch 8 | 0.25 -> 0.75 (+4%); not_sel 27, selected 25, math 20, dispatch 11 |
| N = 16 | 0.30 -> 0.60; selected 30, not_sel 13, wait 13, dispatch 10, math 9 | 0.24 -> 0.72 (+20%); not_sel 20, dispatch 12, math 12 |
| N = 32 | 0.27 -> 0.54; selected 27, dispatch 15, wait 13, not_sel 11, math 9 | 0.21 -> 0.63 (+17%); wait 18, dispatch 17, not_sel 14 |
| N = 64 | 0.23 -> 0.46; selected 23, wait 22, dispatch 12 | 0.14 -> 0.42 (-9%); dispatch 20, long_sb 17 (data waits) |
- The production narrow loops show the ub's 2-warp profile (ub s8 2 WG: selected 39, not_sel 20, wait 11, dispatch 11,
  math 10; IPC 0.70): the microbenchmark finding holds in the real kernel. More warps raise loop IPC only where the
  pipes are not already throttled (N = 16 / 32: +17-20%; N = 8: +4%, math throttle 14 -> 20%) and N = 64 turns
  feed-bound (long_sb 17%).
- Whole s_b MoE kernel: issue slots busy 40%, 2.5 active / 0.60 eligible warps per scheduler; kernel N alone 53%,
  3.5 / 1.19.

## 5. Panel (logs/r10, r12; MoE-only paired, cold L2, us; outputs bit-identical to s_b in every row)
| draw / case | s_b | sw0 (mx 2-WG N) | n3 (3-WG N, v3) | n3w (3-WG N, W shrunk) | 128-token tiles |
|---|---|---|---|---|---|
| 11 / 0 | 260.4 | 268.5 (-3.0%) | 268.2 (-2.9%) | 264.2 (-1.4%) | 0 |
| 11 / 1 | 260.6 | 272.6 (-4.4%) | 280.5 (-7.1%) | 273.5 (-4.7%) | 1 |
| 22 / 0 | 285.9 | 296.6 (-3.6%) | 310.3 (-7.9%) | 304.1 (-6.0%) | 2 |
| 22 / 1 | 316.0 | 323.9 (-2.5%) | 342.5 (-7.8%) | 339.8 (-7.0%) | 3 |
| 33 / 0 | 290.5 | 300.6 (-3.4%) | 310.5 (-6.4%) | 307.9 (-5.6%) | 2 |
| 33 / 1 | 262.0 | 279.2 (-6.2%) | 297.8 (-12.0%) | 287.3 (-8.8%) | 3 |
(% = vs s_b; 128-token tile counts approximate, from the expert counts.) T = 3012 (cases 2 / 3) does not use the
split (512 <= T < 2048 only): kernel N is not launched there. No gate / replica runs: no variant beats s_b.

## 6. Verdict (2026-10-31): more consumer warps per SMSP on the narrow items -- does not pay; line closed
1. Microbenchmark gate (plan step 1): NOT met. 3-4 consumer WGs cut the narrow loop by 9-12% at N <= 16 and 7% at
   N = 32 with W from L2 (-14.5% at N = 64, 3 WG); 4 WGs at 112 registers are C7512 for N >= 32. The loop is not
   warp-starved: with 4 warps per SMSP there are ~2 eligible warps per cycle, IPC still stops at 0.71-0.78, and the
   added stall is pipe throttle / dispatch on the FMA-heavy (IMAD + HMUL2) and ALU pipes (section 1.2). With W from
   HBM for all SMs the gain shrinks to 0% (N = 8), -5% (16), -6% (32), -15% (64) (section 3.3 table).
2. In-graph (section 4): s_b's narrow loops have the ub's 2-warp stall profile (the ub finding holds in the real
   kernel); kernel N's 3 warps raise loop IPC +4% / +20% / +17% / -9% at N = 8 / 16 / 32 / 64.
3. Feed (section 2): narrow items are decode-bound per SM today, but an all-narrow phase sits at the ~4 TB/s weight
   stream ceiling (9.7 us per 128-row item when every SM streams narrow items; production kernel-N phase measured at
   2.8-3.5 TB/s with 2 WGs and 3.5-4.0 TB/s with 3 WGs). In mx's split, kernel N's phase is all-narrow, so faster
   narrow loops have almost nothing to harvest there.
4. Production 3-WG kernel N (cu/n3.cuh v3; bit-identical; 0 spills, no C75xx): per item -2.7% (N=8), -2.0% (16),
   +2.3% (32), +6.6% (64) vs mx's 2-WG kernel N; MoE-only parity with sw0 at draw 11 case 0 and -2.3 .. -6.7% elsewhere
   (128-token tiles: a bit-identical N = 128 path does not fit 3 WGs' registers, so they run as two 64-token passes at
   +52% per item). W || N with W shrunk (n3w) recovers 1-3 points but stays below sw0 except at draw 11 case 0, and
   every split variant stays below s_b (the split's own W -> N hand-off / tail: ~12 vs 5.4 idle us per SM).
=> No candidate. Kept for reference / reuse: cu/n3.cuh + build.py (n3*, n3w*, n3c*), ub/nw_ub.cu (+ WFULL), loop_ana.py,
ana3.py, uni/ (reduced compile-unit probes).
Reusable lessons (for mx's kernel N / W and any warp-specialized edit of this kernel):
- A divergent mbarrier spin must be followed by an .aligned barrier (bar.sync) before the next slot / descriptor
  use, or ptxas takes the slot math off the uniform datapath (2 R2UR + 64-bit adds per HGMMA, +23% loop instructions).
- cp.async.bulk takes uniform-register operands: per-lane copies (or copies inside a divergent single-thread region
  with vector operands) compile to an ELECT / R2UR / BRA.U.ANY waterfall, ~1 iteration per copy. A convergent warp
  with lane-0-shuffled values and elect.sync issues straight-line UBLKCP / UTMALDG.
- Cutting 64-row halves out of the 4608-B blocks costs no re-layout: two 3-D CUtensorMaps (codes [k-block][half][2048
  B], scales [k-block][half][256 B] at +4096, 8-byte elements) give 2 TMA copies per half per stage.
- With 3 consumer WGs + a producer WG the register pool is 128 x 56 + 384 x 152; the producer WG spills below ~48.
- The W || N launch shape that works (n3w): W first, its CTAs above nctas - G exit after the gather wait, N's grid of
  nctas fills those SMs at once and every SM a finishing W CTA frees (dynamic balance at the end).
GPU: done with the sandbox.
