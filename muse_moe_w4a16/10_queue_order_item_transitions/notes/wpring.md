# WP-RING: what paces the 1-group stages of the one-k-warp kernels, and is a deeper ring for them worth building? (2026-11-01)

Worker WP-RING. Base dev/moe_v1511.py (line B); every patch also applies to dev/moe_v1504.py (line A). Namespace
v3600-v3699, scripts dev/wpring_*, logs kda_ws/runs/wpring_runN.log. Brief: WP-K1 (wpk1.md section 3B) read the 1-group
(<= 8-token, mma.sync) stages of moe_8_8_1_3_4 (bs128) / moe_8_8_1_3_8 (bs256) as DATA-PACED by the NS 3 ring of 69.6 KB
slots (a 1-group stage needs ~40 KB, so ~5 would fit); wp32.md section 3 says a 1-group CTA "cannot go above ~1 stage/us
with its ring". Task: verify the mechanism, bound the lever, design a variable-depth ring if the bound is >= ~1%.

## Files
- dev/wpring_mk3600.py BASE OUT: text patch (both lines) -> dev/moe_v3600.py (line B) / dev/moe_v3650.py (line A). All
  switches off by default; default build = the base's SASS in all 16 kernels (scratchpad wpend/sass13.sh, MOE_S5R_FIT,
  re-checked after every patcher change). No ptxas C7511/C7512 in any build.
  - MOE_WPR_TR (diagnostic, with MOE_WPN_TRACE): extra per-stage stamps: each consumer warp's arrival at the stage's
    full-barrier wait, the producer's arrival at the slot acquire (bar.sync / empty mbarrier).
  - MOE_WPR_KM: kernel mask of the bounds (1 (8,8,1) MW <= 4, 2 (8,8,1) MW >= 8, 4 (8,4,1)); default 7.
  - MOE_WPR_BRC=mask (timing-only, wrong outputs): 1-group (mask 1) / 2-group (mask 2) mma.sync items release each
    slot (but the held last one) right after its full-barrier wait, before the compute: NS slots in flight instead of
    NS - 1 (phase-safe: a slot is released only after its bytes landed).
  - MOE_WPR_BDW=mask (timing-only, wrong outputs): those items' stages 1.. do not wait for their bytes; each stage's
    wait moves to after its compute, before its release (the last stage's after the loop): phase-safe, and the consumers
    never wait for bytes that land within one stage's compute.
  - MOE_WPR_BDWC (with BDW): control: BDW's code shape with the stage-top wait kept (correct outputs).
  - MOE_WPR_NPL (timing-only): the producer takes every chunk of a down item's intermediate as ready (no readiness poll).
- dev/wpring_trace.py: per-stage analysis of the MOE_WPN_TRACE + MOE_WPR_TR stamps by stage class (phase, 1-group /
  2-group / multi-group, position in the item: first = st 0, early = 1 <= st < NS (the slot's previous stage is the
  previous item's), mid, last); module globals per build (e.g. NS_MAX).
- dev/wpring_smoke.py: hang1 with module globals (NS_MAX). dev/wpring_lcmp.py: stage / producer loop identity of two
  SASS dumps (dev/wpcgs_search.py's loop finder).
- dev/wpring_run1.sh .. run3.sh; logs kda_ws/runs/wpring_run1.log .. run3.log (run 1: trace + BRC / BDW bounds; run 2:
  BDW / BDWC + NS_MAX sensitivity + their traces; run 3: NPL / BH / BDW + traces). GPU check printed before every timing
  step: no compute apps, 0-3%, 4-130 MiB.

## 1. Mechanism (runs 1-2, v1511 trace build, same routings; cycles, medians unless noted)

The trace build's stamps cost ~115-130 cycles between a warp's wait-arrival stamp and its data-ready stamp even with no
wait between them (BDW trace: "wait" 113-135 with the stage-top wait removed): that is the floor of every "wait" below.
Slot chain of a mid-item 1-group stage i (slot reused from stage i - NS of the same item):
release(i - NS) -> producer acquire (turn) -> copies issued (piss) -> bytes land (L) -> consumer needs them, at
release(i - NS) + (NS - 1) periods (top-tested loop: stage i - NS is released at the top of stage i - NS + 1).

| shape / seed (NS) | period | comp | pre | wait mean / p50 / p90 | producer slot wait pw | late p50 (% < 0) | turn | piss | L p50 / p90 | slack p50 / p90 |
|---|---|---|---|---|---|---|---|---|---|---|
| bs32 73736 (4) | 1877 | 1462 | 157 | 169 / 133 / 184 | 396 | -219 (83%) | 180 | 1160 | 2517 / 3361 | ~1770 / ~930 |
| bs32 73787 (4) | 1884 | 1465 | 156 | 171 / 134 / 213 | 396 | -217 (81%) | 183 | 1158 | 2314 / 3710 | ~2000 / ~600 |
| bs128 71637 (3) | 1746 | 1358 | 168 | 203 / 159 / 261 | 1177 | -1138 (97%) | 42 | 258 | 2487 / 3264 | ~700 / ~-70 |
| bs128 72808 (3) | 1725 | 1359 | 161 | 182 / 155 / 242 | 1180 | -1142 (99%) | 40 | 249 | 1942 / 2711 | ~1220 / ~450 |
| bs256 73011 (3) | 1718 | 1318 | 168 | 180 / 150 / 246 | 1184 | -1141 (100%) | 40 | 246 | 1653 / 2435 | ~1500 / ~720 |
| bs256 71497 (3) | 1720 | 1345 | 168 | 191 / 151 / 261 | 1197 | -1156 (100%) | 39 | 232 | 1848 / 2451 | ~1320 / ~720 |

(period = slowest warp's compute-done to compute-done, mean; comp / pre / wait per warp: pre = compute done -> wait
arrival (release + loop), wait = wait arrival -> data ready; late = producer arrival at the acquire minus the slot's last
release (< 0: the producer waited for the slot); L = copies issued -> first data-ready; slack = (NS - 1) x period -
turn - piss - L, a lower bound since data-ready is the consumers' arrival when they come late.) Gate/up (G1_1) mid
stages shown; down (G2_1) mid stages are the same within ~20 cycles.

What this says:
- The 1-group period is NOT set by bytes in flight, by the producer's issue rate or by the slot turnaround. It is set by
  the consumers: compute (1318-1360 cycles at bs128/256, 1462 at bs32) + release / loop / try_wait round trip. The
  consumers find their stage landed in nearly every mid stage: the wait above the stamp floor is 40-70 cycles per warp
  on average, which is the try_wait round trip itself (BDW moves that instruction out: its "pre" grows by 35-64 cycles
  while "wait" drops to the floor), i.e. ~0-20 cycles of actual data wait per warp per stage.
- The producer IS ring-paced (it waits ~1150-1200 cycles per stage at bar.sync, 97-100% of mid stages, at bs128/256;
  ~400 at bs32), but that is harmless: the ring is full because the consumers are slower than the stream. The slot
  turnaround (last consumer release -> producer awake) is ~40 cycles with WP-N's named barriers ((8,8,1)), ~180 with the
  empty mbarrier (bs32); issue ~250 cycles in (8,8,1) (one combined weight+scale copy + one x copy), ~1150 at bs32
  (separate weight / scale copies and one bulk copy per token row).
- Median slack between landing and need: ~700-1500 cycles at bs128/256 with NS 3, ~600-2000 at bs32 with NS 4; only the
  landing-latency tail (p90 L 2435-3264) reaches the need time, at bs128 (seed 71637) only.
- WP-K1's "comp + gap 1600-1650 vs period 1688-1793" is the mean-vs-median difference of item boundaries (first / early
  stages: periods 2650-3020 / 1670-1910), and its gap is release + loop + try_wait + its own stamps, not a data wait.
- Same for bs32 ((8,4,1), NS 4): wp32.md's "~1 stage/us = the ring's DRAM rate" coincides with the consumers' own rate
  (1880 cycles at ~1.7 GHz); the consumers do not wait for data in mid stages either.
- Item boundaries are producer-side late, not ring-blocked: first stages of gate/up items: the producer reaches the
  acquire ~580-650 cycles after the slot was free (late > 0 in 95-100%); down items' first stages take 1935-1983 cycles
  from acquire to issue (the readiness poll of the tile's chunk counters + proxy fence after the acquire and the
  expect-tx) and their early stages are reached ~1900-2060 cycles after the slot was free. Run 3 bounds that poll.

## 2. Bounds (bench2 B2_NOVERIFY, jm seeds, 100 reps x 2 passes, fillers at both ends, % vs the mean of the middle
## v1511 controls; spread = min..max of the controls)

| variant | bs32 | bs128 | bs256 |
|---|---|---|---|
| run 1 controls (5 x 2) | -0.20..+0.19 | -0.09..+0.14 | -0.30..+0.24 |
| BRC=1 (1-group: NS in flight) | -0.96 / -0.72 | -0.23 / -0.35 | -0.24 / -0.06 |
| BRC=3 (+ 2-group) | -0.80 / -0.57 | -0.61 / -0.46 | -0.20 / -0.33 |
| BDW=1 (1-group: no data waits within a compute) | +0.06 / +0.25 | +0.43 / +0.25 | +0.28 / +0.31 |
| BDW=3 (+ 2-group) | -0.35 / -0.26 | +0.41 / +0.46 | +0.29 / +0.45 |
| run 2 controls (7 x 2) | -0.12..+0.14 | -0.17..+0.13 | -0.16..+0.15 |
| BDW=1 (2 copies) | +0.10 / +0.27, +0.16 / +0.23 | +0.37 / +0.47, +0.24 / +0.24 | +0.35 / +0.38, +0.26 / +0.16 |
| BDWC (BDW's code shape, top wait kept; correct) | -0.49 / -0.44, -0.46 / -0.43 | -0.67 / -0.46, -0.52 / -0.41 | -0.20 / -0.22, -0.57 / -0.51 |
| NS_MAX=3 (same source; bs32 4 -> 3 slots; (8,8,1) unchanged = placebo) | -1.52 / -1.62 | +0.17 / +0.15 | +0.04 / -0.01 |
| NS_MAX=2 (bs32 4 -> 2; (8,8,1) 3 -> 2 slots) | -12.62 / -12.73 | -4.33 / -4.26 | -3.90 / -3.76 |

- BRC (one more slot's bytes in flight for 1-group stages, refilled under the compute) LOSES at every shape: the trace
  shows no wait reduction (mid-stage waits unchanged) and the compute slows (1318 -> 1379 cycles at bs256: the refill's
  smem writes under the consumers' reads).
- BDW, the bound for removing every 1-group data wait up to one stage compute (what any deeper ring could at most do
  for those stages, at item edges included), is +0.16..+0.47% at bs128 / bs256 (4 copies, 3 jobs) and +0.06..+0.27% at
  bs32. Its correct control BDWC (an extra, always-satisfied try_wait per 1-group stage) costs -0.4..-0.7%: one extra
  SYNCS round trip per stage is worth ~0.5% here, so BDW's gain is a mix of the waits it removes and of the try_wait
  moving off the compute-start path; either way it is the ceiling of the lever, and it is below the 1% threshold.
- Depth sensitivity with no code change: removing a slot costs -1.6% at bs32 (4 -> 3) and -3.8..-4.3% at bs128/256
  (3 -> 2): the current depths sit just past the knee (NS 3 in (8,8,1) is enough for 64-token-slot kernels whose
  producer issues in ~250 cycles; bs32's slower producer needs 4). The traces of NS 2 / NS 3 show where the loss comes
  from: mid-stage waits reappear (bs32 NS 3: wait p90 175 -> 426; (8,8,1) NS 2: 254 -> 386-408) plus the held slot one
  stage earlier (early stages' waits x2-3).

Run 3 (dev/wpring_run3.sh; controls 7 x 2: bs32 -0.12..+0.20, bs128 -0.27..+0.18, bs256 -0.23..+0.10):

| variant | bs32 | bs128 | bs256 |
|---|---|---|---|
| NPL (no readiness poll for down items; every stage + producer loop byte-identical) x2 | +0.13 / +0.02, +0.06 / -0.09 | +0.35 / +0.06, +0.10 / -0.02 | +0.09 / +0.14, +0.26 / +0.04 |
| MOE_IT_BH (held slot released at the epilogue's wake: the most extra slots could give the held slot) x2 | +0.01 / -0.03, +0.06 / +0.02 | +0.40 / +0.30, +0.27 / +0.20 | +0.31 / +0.36, +0.16 / +0.10 |
| NPL + BH | +0.26 / +0.19 | +0.45 / +0.35 | +0.07 / +0.26 |
| BDW=1 (third job) | +0.20 / +0.26 | +0.35 / +0.19 | +0.16 / +0.24 |

- NPL: the down items' first-stage issue moves ~1000 cycles earlier in the trace (acquire -> issue 1967 -> 262 cycles;
  the producer's per-item path then takes ~650 cycles longer before the acquire) but the consumers' head / early-stage
  periods barely move (G2_1_first 2556 -> 2579, G2_1_early 1894 -> 1823 cycles; span -0.2% bs128 / -0.5% bs256 in the
  trace build); in bench2 it is within the controls' spread except one copy. The item start is consumer-bound (WP-IT /
  WP-W0: warp 0's own transition), not producer- or ring-bound. Not a lever.
- BH (+0.1..+0.4%): the ceiling of the held-slot part of a deeper ring, consistent with WP-IT's ~0.3-0.4% after TMG.

Verdict: the variable-depth ring is not worth building. Its ceiling (BDW) is ~0.2-0.45% at bs128/256 and ~0.1-0.25% at
bs32, and the dual-layout implementation would add code to every role on the hot paths: a second barrier set and slot
stride in the 1-group loop (a re-scheduled loop is -0.5..-1% by itself), an item-head that must learn the next item's
layout before its first wait (an extra mailbox wait or a second polled barrier on warp 0's critical path: an extra SYNCS
round trip measured -0.5% per stage here, WP-W0 / WP-SEPI32 per item), producer transitions between the layouts, and the
epilogue's held-slot base / release barrier per layout. Expected net <= 0.

Design notes kept for the record (in case a later base makes the 1-group stages data-paced):
- Header room: with NBAR the (8,8,1) kernels never use their empty mbarriers (bytes 24-47) and bytes 48-63 are free:
  exactly 5 mbarriers for a small-slot full set, keeping the ring's barriers at offset 0.
- An overlay without a drain bubble exists: 5 small slots of 40960 B over the 3 x 69632 B ring (S0 in B0, S1 over
  B0/B1, S2 in B1, S3 over B1/B2, S4 in B2). Starting the small ring at the small slot that begins in the first big slot
  to be released (B0 -> S0, B1 -> S2, B2 -> S4) gives the same issue times as today for the first two stages and needs
  the held big slot only at the 4th small stage; the reverse switch (big slot b waits for the small slots overlapping it)
  likewise. Protocol: every slot use leaves exactly one outstanding release (named barrier, 288 or 96 threads); before
  writing any byte range the producer bar.syncs every outstanding release overlapping it, once. Small slots have
  896 B spare past the 40064 B stage for the record; XS staging of 1-group items does not fit (8320 B) and would fall
  back to staging at offset 0 behind the item-end barrier.
- A 4 x 52224 B layout fits every tile up to 32 tokens (1-2 group mma.sync, 3-4 group wgmma): the alternative if a
  later finding shows 3 -> 4 slots matters for <= 32-token tiles.

## 3. Leads for the planner (not built)
- The per-stage try_wait round trip: one extra, always-satisfied try_wait per 1-group stage costs ~0.5% (BDWC), and BDW
  (the wait moved behind the compute) gains +0.2..+0.45%. A correct form would test the next stage's full barrier
  (mbarrier.test_wait, non-blocking) right after a stage's compute and skip the stage-top try_wait when it succeeded
  (landing precedes the need by ~700-1500 cycles at the median). Expected small (the two consumer warps per SMSP are
  issue-bound, so one warp's SYNCS latency is largely covered by the other) and it edits the hottest loop (re-schedule
  risk -0.5..-1%; WP-W0's similar item-head form HP lost): only with dev/wpcgs_search.py for a loop-exact form.
- bs32 ((8,4,1) MW 1) producer: ~1150 cycles per 1-group stage from acquire to issue (separate weight / scale copies, one
  bulk copy per token row) with only ~220 cycles of slot-wait margin (late p50 -220): if anything makes its consumers
  faster, this producer becomes the pacer (the LEAN / WSC forms that (8,8,1) uses, 250 cycles per stage, are off for
  MW 1 by MOE_WPN_MW1OFF).
- Nothing to bank from WP-RING: every switch above is a diagnostic or a timing-only bound; dev/moe_v3600.py /
  dev/moe_v3650.py default builds are v1511 / v1504 SASS (no check_mt needed: no kernel changed).
