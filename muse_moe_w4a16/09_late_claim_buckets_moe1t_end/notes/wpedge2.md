# WP-EDGE2: item-boundary bubbles in the bs8 / bs16 / bs32 kernels (2026-10-31)

Worker WP-EDGE2. Base dev/moe_v1002.py. Namespace v1100-v1199, defines MOE_ED2_*, scripts dev/wped2_*.
Kernels: moe_4_1_1_4_1 (bs<=8), moe_4_2_1_4_1 (bs9-16), moe_8_4_1_4_1 (bs17-32, MW1).

Files:
- dev/wped2_mk1100.py -> dev/moe_v1100.py: v1002 + MOE_ED2_ITR (item-boundary trace), MOE_ED2_DLY=D (timing
  sensitivity: D more cycles at every producer item start after the CTA's first), MOE_ED2_INC (non-aggregated
  claim). Default build SASS-identical to v1002 in every kernel (local NVRTC 13 hashes); DLY / INC builds keep every
  consumer stage loop byte-identical (dev/wpc_cmp.py all regions).
- dev/wped2_itr.py: per item boundary, producer and consumer attribution (see its docstring).
- dev/wped2_mk1101..1110.py -> dev/moe_v1101..v1110.py (each = the previous + switches, default build = v1002 SASS in
  every kernel, checked by local NVRTC 13 hashes): EP (1101), PN + TOP trace (1102), PN2 (1103), PN1 (1104), GP (1105),
  DP / DP1 (1106), GP1 / GP1B (1107), GPK (1108), CTOP trace + INC2 (1109), B8 / BS bounds (1110).
- dev/wped2_mk1111.py -> dev/moe_v1111.py: the baked candidate (v1002 + the (4,*,1) claim as a bounded inc).
- dev/wped2_top.py: spill-free item-start split (TOP / CTOP traces). dev/wped2_run1..3.sh: the three Modal jobs.
- Logs: kda_ws/runs/wped2_run1..4.log (copies of the job outputs).
- (dev/wpe2_mk1100.py, dev/wpe2_itr.py: first copies of the above, created by mistake in WP-E2's file prefix; superseded,
  left in place. WP-E2's dev/wpe2_run1.sh was overwritten by mistake for a few minutes and restored byte-exact from
  WP-E2's session transcript.)

## 1. Is item-start lateness on the critical path? Sensitivity (wped2_run1, bench2 jm100, same job, v1002 x2)

MOE_ED2_DLY=D: the producer sleeps D cycles (nanosleep-paced, no issue slots taken) at every item start after the
CTA's first. Consumer stage loops byte-identical to v1002 in every kernel.

| | bs8 | bs16 | bs32 | bs128 |
|---|---|---|---|---|
| v1002 (us) | 62.2 | 84.2 | 110.8 | 165.1 |
| v1002b (control) | 0.9978 | 0.9973 | 0.9984 | 1.0005 |
| DLY=1500 | 0.9536 (+3.3 us) | 0.9717 (+2.2) | 0.9702 (+3.0) | 0.9825 (+2.3) |
| DLY=4000 | 0.8652 (+11.3) | 0.8983 (+9.2) | 0.9176 (+10.1) | 0.9484 (+8.5) |
| INC (non-aggregated claim) | 1.0035 | 1.0060 | 0.9967 | 1.0019 |

Delayed items per CTA: bs8 ~6.6, bs16 ~7.9, bs32 ~8.1, bs128 ~9.2. Exposure of the added delay (wall time / (D x
items per CTA at ~1.95 GHz)): bs8 65% at 1500 and 84% at 4000; bs16 36% / 53%; bs32 48% / 62%; bs128 33% / 46%.
Item-start lateness is NOT absorbed by the other SMs' streams: roughly half of it lands on the wall time, more at bs8.
A quadratic through (0, 1500, 4000) at bs8 gives a marginal exposure of ~54% at D = 0. (Section 5: the curve is NOT
smooth below 0: the producer has about a stage of lead, so removing item-start time gains nothing; only delays past the
slack cost.)

## 2. Attribution trace, first attempt (MOE_ED2_ITR, wped2_run1): distorted, do not use

12 clock stamps per item on the producer and consumers. The extra registers made the trace kernels spill (24-40 B
stack in moe_4_1_1 / 4_2_1 / 8_4_1_4_1, 0 in the base), the spill reloads sit on the item-start path right after the
down poll's CCTL.IVALL (L1 invalidated, so each LDL is an L2 round trip), and each stamp costs ~70+ cycles of starved
SMSP0 issue. What survives: (a) every producer instruction at the item start costs ~10-20 cycles (the producer shares
SMSP0 with two decode-bound consumer warps); (b) mid-item, the slot the producer is about to refill was released
~1700-2300 cycles before it gets there (free < 0): the producer is never waiting for a slot, it is blocked in the
weights copy issue (TMA backpressure, piss ~1100) plus ~600 cycles of its own instructions per stage; (c) the first
stage's data latency after issue is ~950-1100 cycles vs ~2000 mid-item: at an item start the TMA queue has drained.
A spill-free split (MOE_ED2_TOP: only the loop-top clock, in word 19 of the first-stage WPN record) replaces it.

## 3. The producer's item-start path (SASS + lean trace history)

(Read with section 5: run 2 shows this path is slack, not the cause of the late first stage.)

- Every item start of the (4,*,1) / MW1 producers runs ~120 (4,1,1) to ~250 (MW1) SASS instructions between the
  previous item's last copy and the next item's first weights copy, against ~70 between two mid-item stages: the item end
  (read the claimed index), per-call constants re-loaded from shared memory (the 72-register producer cannot keep them),
  the claim atomic (in v1002 still warp-aggregated: its SHFL waits for the L2 round trip), the queue decode (~12), the
  tile record LDS and ~35-45 instructions of per-item address / record / expected-bytes setup, then the stage-0 path.
- The producer shares SMSP0 with two decode-bound consumer warps and issues ~1 instruction per 10-20 cycles while they
  compute (WP-L measured the same: ~70 instructions in 1.4-1.5 us; warp ids do not set priority, WREMAP). So the item
  start costs ~1200-1700 cycles of producer time (v1002 lean WPN trace, bs8 pidle 1700-2300 vs 510-590 mid-item).
- Mid-item the producer is blocked ~1000 cycles per stage in the weights copy issue (TMA backpressure) and the TMA queue
  holds about one stage, so ~1000 cycles of the item-start excess are absorbed; the remainder becomes a stream bubble:
  bs8 first-stage period +400..600 cycles (median/mean), first-stage data latency after issue ~1300-2000 vs ~2200-3200
  mid-item (the queue had drained). bs32: period +1400..1800, but the first-stage latency stays at the mid-item value
  (~2600-3100), and WP-32's v860 INC showed that issuing ~575 cycles earlier does not bring that data earlier.
- Fix candidates, all keeping every consumer stage loop byte-identical (innermost HMMA loops, dev/wpc_cmp.py + a
  direct innermost-loop hash):
  INC (v1100: non-aggregated claim), EP (v1101: a down item's first-stage readiness loads before the slot wait),
  PN / PN2 (v1102 / v1103: the next item's decode / tile record read mid-item; (4,*,1) only: in MW1 PN re-schedules the
  consumer loops even through a __noinline__ decode, v1104 MOE_ED2_PN1, similarity 0.35), GP (v1105: gate/up items past
  the first wave claimed and streamed in row-block pairs, (4,*,1): half the gate/up item starts become mid-item stages),
  DP (v1106: whole down items in pairs too; (4,2,1) only by default: in (4,1,1) it re-allocates the consumer loops and
  spills 4 B). GP1 (v1107, pairs in MW1) spills 48 B and re-schedules the MW1 consumer loops: not run.

## 4. Planner's structural proposal (descriptor ring on a helper warp) -- assessment before building

(Outcome, section 5: GP's paired second items, the ring's best case, run at the same first-stage period as
ordinary item starts; the ring would remove producer time that is already hidden. Not built.)

Proposal: a helper warp off SMSP0 claims, decodes and writes ready-to-issue per-stage descriptors; the producer (or the
last releasing consumer warp) only waits, LDS's a descriptor and issues. Points from the code / earlier measurements:
- GP (v1105) is the cheap bound for it: the second item of a gate/up pair has no claim, no decode, no tile lookup and
  no per-item setup at all (only the stage-0 record and the x row pointers), i.e. exactly what a perfect descriptor ring
  would leave on SMSP0, for half of the gate/up item starts. If GP's paired starts run at the mid-item period and the
  bench gain matches the sensitivity (bs8 ~2 starts per CTA saved), the ring (all starts) is worth building; if not, not.
- Helper candidates: warp 9 lane 0 runs the signaler loop (sleeps on epi[]); warps 10-11 (SEPI) sleep on aq_full and
  run ~1.1-1.4 us per item ~3.5 stages behind the producer, so they can only serve claims >= 2 items ahead, which fails
  for the 2-stage k pieces of bs8's last round (needed 2 stages after the previous item) and moves the claim earlier
  (tail balance). A helper waiting on two event sources has to poll: WP-L's MOE_XH (x copies + polls on warp 9) lost 5%
  from exactly that (the poll loop's issue slots on SMSP1 and the hand-off latency).
- Refill by the last releasing consumer: every consumer-side addition on the item path measured as a loss so far (CSO
  -4%, BSTG, KDIR -2%, PWS -0.75%); a per-stage atomic + issue on the consumer warps would be per stage, not per item.

## 5. wped2_run2 (v1106; smoke 8 builds ok, check_mt M 2 4 8 9 12 16 17 32 33 normal + skewed x 3 seeds x 2 calls:
## 0 failures for INC+EP+PN+PN2 and for GP+DP+INC+EP)

bench2 jm100, one job, ratio vs v1002 (in brackets: vs the mean of the two control copies):

| | bs8 | bs16 | bs32 | bs128 |
|---|---|---|---|---|
| v1002b (control) | 0.9986 | 0.9969 | 0.9997 | 0.9986 |
| INC | 0.9998 (+0.05%) | 1.0016 (+0.32%) | 0.9961 (-0.38%) | 0.9990 (-0.03%) |
| INC+EP | 1.0000 (+0.07%) | 1.0019 (+0.35%) | 0.9948 (-0.51%) | 0.9980 (-0.13%) |
| INC+PN | 0.9980 (-0.13%) | 1.0004 (+0.20%) | 0.9964 (-0.35%) | 0.9985 (-0.08%) |
| INC+EP+PN | 0.9904 (-0.89%) | 0.9995 (+0.11%) | 0.9935 (-0.64%) | 0.9949 (-0.44%) |
| INC+EP+PN+PN2 | 0.9907 (-0.86%) | 1.0007 (+0.23%) | 0.9945 (-0.54%) | 0.9971 (-0.22%) |
| GP+INC+EP | 0.9854 (-1.39%) | 0.9990 (+0.06%) | 0.9958 (-0.41%) | 0.9966 (-0.27%) |
| GP+DP+INC+EP | 0.9879 (-1.14%) | 0.9410 (-5.75%) | 0.9954 (-0.45%) | 0.9967 (-0.26%) |
| DLY=500 | 0.9855 (-1.38%) | 0.9929 (-0.56%) | 0.9914 (-0.85%) | 0.9928 (-0.65%) |

(run 1 had INC at +0.46 / +0.74 / -0.25 / +0.16%.) Nothing beats the controls by a margin; GP and DP lose.

Lean trace (MOE_WPN_TRACE + MOE_ED2_TOP, spill-free), first stage of gate/up 1-group items after gate/up items
(G1>G11w), medians (cycles):

| | top2acq | pedge | piss | lat | gap | per (first) | per (mid) |
|---|---|---|---|---|---|---|---|
| bs8 base | 1584 | 1891 | 1064 | 1905 | 627 | 2216 | 1845 |
| bs8 INC | 695 | 1010 | 1116 | 1899 | 613 | 2214 | 1856 |
| bs8 GP, pair's 2nd item (PG1>G11w) | 451 | 693 | 1262 | 2352 | 587 | 2213 | 1846 |
| bs16 base | 1194 | 1482 | 1048 | 1754 | 700 | 2376 | 1879 |
| bs16 INC | 617 | 889 | 1125 | 2210 | 698 | 2358 | 1876 |
| bs32 base | 1636 | 1960 | 914 | 3188 | 1789 | 3632 | 1785 |
| bs32 INC | 991 | 1303 | 898 | 3935 | 1778 | 3696 | 1784 |

**The producer's item start is not what makes the first stage late.** INC removes the claim's L2 round trip (top2acq
-600..-900 cycles: the aggregated SHFL really did wait for it) and GP's paired second item has almost no item-start
work at all (pedge 693 vs 1891), yet the first stage's period, the consumers' gap and the moment the data is seen are
unchanged: the copy issued earlier just takes longer (lat +450 at bs16, +750 at bs32), and at bs8 the whole producer
timeline shifts relative to the consumers. Mid-item the producer is ring-gated (free ~0: the slot it refills is released
right as it gets there), so it has a stage of lead and its item-start time is slack. DLY shows only what happens once a
delay exceeds that slack (+500 cycles already costs 1.4% at bs8).

Why GP / DP lose: down items first-stage piss/gap explode (bs8 G1>G21s piss 1569 -> 5486, gap 1957 -> 4676; mean gaps x2-3
on every down category): a tile's gate/up rows are done later when its row blocks go to one CTA in pairs, and its down
items' readiness polls wait for them; DP at bs16 also coarsens the down phase's end (no k split there): -5.75%.

## 6. wped2_run3 (queued; v1110): replication + the two-path test

Hypothesis from runs 1-2: DLY=+500 already costs (no producer slack) but INC=-600..-900 gains nothing, so the first
stage is gated by two near-equal paths: the producer's issue and something on the consumer / stream side. If the other
path is the consumers' item end, then the item-end bound alone (B8 / BS, as WP-EDGE measured: ~0 at bs8, +0.7% bs32)
and INC alone show nothing, but INC + bound should beat both. Run 3: bench2 (verified) v1002 x2, INC2 ((4,*,1) only)
x3, INC2+PN x2, INC; bench2 (B2_NOVERIFY) v1002 x2, INC, B8+BS, B8+BS+INC, B8+BS+INC2; lean trace with consumer
item-top stamps (MOE_ED2_CTOP: item-end work vs data wait at each item start) for base, INC, B8+BS, B8+BS+INC.

### Run 3 results (v1110; smoke ok; check_mt INC2+PN M 2 4 8 9 12 16: 0 failures)

bench2 jm100, verified section (vs the control mean):

| | bs8 | bs16 | bs32 | bs128 |
|---|---|---|---|---|
| v1002b | 1.0037 | 0.9980 | 1.0000 | 0.9971 |
| INC2 x3 ((4,*,1) only; bs32/128 kernels = v1002) | -0.20 / +0.01 / -0.07% | **+0.33 / +0.45 / +0.28%** | -0.17 / -0.13 / -0.22% (placebo) | +0.02 / -0.22 / -0.22% (placebo) |
| INC2+PN x2 | -0.13 / +0.16% | -0.04 / +0.31% | | |
| INC | +0.16% | +0.44% | -0.09% | +0.19% |

Bounds section (B2_NOVERIFY, outputs wrong): B8+BS -0.30 / +0.07 / +0.90 / +0.12%; B8+BS+INC -0.24 / +0.26 / +1.17 /
+0.52%; B8+BS+INC2 +0.11 / +0.56 / +1.13 / +0.72% (bs8 / 16 / 32 / 128). INC alone in that section -0.04 / +0.16 /
-0.26 / -0.23%. No interaction at bs8/16 (the two-path idea fails there); at bs32 INC adds ~+0.25% only once the item
end is free (within noise).

Consumer item-top stamps (CTOP, lean), first stage of a gate/up item after a gate/up item, medians (cycles):

| | warp 0 item end | warp 0 data wait | warp 2 item end | warp 2 data wait | mid-item gap |
|---|---|---|---|---|---|
| bs8 base / INC / B8 | 564 / 518 / 493 | 418 / 415 / 423 | 409 / 412 / 396 | 375 / 372 / 376 | 348 |
| bs16 base / INC / B8 | 621 / 640 / 568 | 473 / 488 / 473 | 478 / 481 / 395 | 364 / 355 / 354 | 273 |
| bs32 base / INC / BS | 537 / 535 / 523 | 400 / 400 / 522 | 2244 / 2326 / 379 | 417 / 418 / 506 | 262 |

## 7. Conclusions

1. Root cause of the late first stage of an item (bs8 / bs16 / bs32), attributed with spill-free traces:
   - NOT the producer. Its item start is long (claim with an exposed L2 round trip, constants re-loaded, decode, tile
     lookup, ~40 setup instructions at ~10-20 cycles each on SMSP0), but it is slack: INC cuts it by 600-900 cycles and
     GP's paired items remove it almost entirely, and neither moves the first stage's data, period or gap. The producer
     leads by about a stage; only delays beyond that cost (DLY sensitivity). Neither the claim round trip, the copy-plan
     computation, the G2 dependency polls (EP: 0) nor a held slot gate it.
   - It is the consumers' own item end on the slowest warp: warp 0 (SMSP0, beside the producer) spends ~520-640 cycles
     between its last stage's compute and the next item's top (SEPI staging + buffer wait + record + hand-off at bs8/16;
     staging + bar.sync + fence + hand-off at bs32), and then still waits ~400-490 cycles for data (vs ~260-350
     mid-item). At bs32 the AEPI bar.sync also makes the warps that ran ahead (warp 2: ~2250 cycles) wait for warp 0,
     which is why the max-over-warps "first-stage period" looks so large there; without the barrier and staging (BS
     bound) that collapses (warp 2 379) and buys +0.9%, the most any bs32 item-end fix can give (WP-EDGE's PWS, the one
     real attempt, lost 0.75%).
   - The staging stores themselves are cheap (B8: -50..-70 cycles of warp 0's item end, bench ~0).
2. What converts: only the non-aggregated claim at bs16 (+0.35% over 3 copies in one job; +0.4% mean over 7 copies of
   the same (4,2,1) kernel in 3 jobs). bs8 ~0, bs32 slightly negative when applied there (-0.25% mean), so it is baked
   for the (4,*,1) kernels only: **dev/moe_v1111.py** (default on, escape MOE_NO_ED2; only moe_4_1_1_4_1 /
   moe_4_2_1_4_1 differ from v1002, byte-identical to the measured INC2 build; consumer stage loops byte-identical; no
   spills, no C75xx). check_mt on v1111 itself: 0 failures (section 8).
3. Rejected: EP, PN, PN2, PN1 / GP1 / GP1B (MW1: re-schedule the consumer loops / spill 36-48 B, not run), GP (bs8
   -1.4%), DP (bs16 -5.75%), GPK=4 (not run; same mechanism as GP). The planner's helper-warp descriptor ring is
   bounded by GP's paired items (no item-start work left on SMSP0) and would not pay either.
4. For whoever continues: the edge cost that remains is consumer-side (~500 cycles of item-end instructions on warp 0
   per item, and bs32's bar.sync), i.e. WP-EDGE's closed levers; no producer-side change can recover it.

## 8. Final summary (for integration)

- Banked switch: **dev/moe_v1111.py** = v1002 + the (4,*,1) producer's next-item claim as `atom.global.inc.u32 [head],
  0x7fffffff` (on by default; escape `MOE_NO_ED2` = v1002 SASS in every kernel). Only moe_4_1_1_4_1 and moe_4_2_1_4_1
  change (byte-identical to the measured MOE_ED2_INC2 build; consumer stage loops byte-identical; 0 spills; no C75xx);
  the bs17-512 kernels (bs32 / bs128 / bs256) are untouched.
- Effect: bs16 +0.35..0.4% (3 copies in one job +0.33 / +0.45 / +0.28% with controls +-0.10%; 7 copies of that (4,2,1)
  kernel over 3 jobs, mean +0.39%); bs8 ~0 (7 copies, mean +0.05%). Geomean ~+0.06%.
- Correctness (wped2_run4, kda_ws/runs/wped2_run4.log): hang1 bs 1 2 4 8 9 15 16 17 32 33 128 256 257 all ok (rel-L2
  2.5e-3..3.2e-3); check_mt on v1111 itself, M 2 4 8 9 12 15 16 17 32 33, normal + skewed, 3 seeds x 2 calls:
  **0 failures**.
- Everything else measured on the item edges is rejected (sections 5-7): the producer's item start is slack; the
  remaining edge cost is the consumers' item end on warp 0 (and bs32's bar.sync, bound +0.9%).
