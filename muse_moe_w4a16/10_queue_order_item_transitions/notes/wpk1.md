# WP-K1: the one-k-warp 1-group stage-loop "slowdown" ((8,*,1) ~1603-1621 vs (4,1,1) ~1429 cycles/stage), 2026-11-01

Worker WP-K1. Base dev/moe_v1511.py (line B), patches must also apply to dev/moe_v1504.py (line A). Namespace
v3500-v3599, scripts dev/wpk1_*, logs kda_ws/runs/wpk1_runN.log.

Prior work on the same question (read first): kda_ws/docs/findings/wpw.md (WP-W, base v567: "the 1-group loop's compute
is the same in every binary (ncu, same routing); MOE_TIMING's 1470 vs 1580 is an artifact of timers inside the loop;
the per-stage difference is the ring structure around the loop: SMSP0's consumer pair (with the producer) paces the
ring, the asynchronous epilogue holds the item's last slot") and wpn.md (WP-N, base v730: the producer's cost on SMSP0,
LEAN3 / LEAN4 / NBAR; residual SMSP0 skew 50-100 cycles per stage at bs128/256 = a per-wake floor, not instruction
count).

## Files
- dev/wpk1_loops.py: stage-loop inventory of a SASS dump (per kernel: innermost HMMA / HGMMA loops, hashes, opcode mix),
  --cmp: pairwise opcode-order similarity of the ntl-1 loops.
- dev/wpk1_trace.py: dev/wpn_trace.py (MOE_WPN_TRACE per-stage per-warp clock64) with a per-build bucket override, so
  one routing runs on (4,1,1) / (8,4,1) / (8,8,1).
- dev/wpk1_smoke.py: hang1 with a bucket override and capped routings.
- dev/wpk1_run1.sh: reproduction job (smoke, traces, ncu, MOE_TIMING).

## 1. Loop identity on v1511 (local NVRTC 13, MOE_S5R_FIT build = the judged build)
ntl-1 stage loops: moe_4_1_1_4_1 513 instructions (S5R; the S6O fallback build 510), moe_8_4_1_4_1 514, moe_8_8_1_3_4 511,
moe_8_8_1_3_8 511. Same opcode mix (32 HMMA, 18-19 LDS, 170-171 LOP3, 128 HMUL2, 71-74 IMAD, 66-68 SHF); differences:
(4,1,1) is the bottom-tested MOE_RUN_ITEM_BOT loop with a lane-0 mbarrier release (2 SYNCS); (8,8,1) MW >= 2 is the
top-tested MOE_RUN_ITEM_TOP loop with WP-DS's DEF2 carry (MOE_DS81 = 100000002) and a named-barrier release (BAR.ARV
288) + 1 SYNCS try_wait + 1 LDC. Opcode-order similarity: (4,1,1) vs (8,8,1) 0.31-0.44, (8,4,1) vs (8,8,1) 0.46,
(8,8,1) MW4 vs MW8 0.99. "Same SASS" is the same instruction mix, not the same schedule.

## 2. Run 1 (dev/wpk1_run1.sh, kda_ws/runs/wpk1_run1.log): the "slowdown" does NOT reproduce on v1511

A. Per-warp per-stage trace (MOE_WPN_TRACE timing build: clock64 per stage per consumer warp; dev/wpk1_trace.py), SAME
routings (jm bs32 seeds capped at 8 tokens per expert: every tile 1 group) forced onto each configuration, cycles per
stage, medians; comp = data-ready -> compute-done, gap = previous compute-done -> data-ready, period = mean per-warp
stage period; G1_1 = gate/up 1-group stages:

| routing | config (binary) | comp w0..w7 | comp SMSP1-3 warps | sk0 (SMSP0 comp+gap excess) | period |
|---|---|---|---|---|---|
| cap32/73736 | (4,1,1) moe_4_1_1_4_1 | 1543 1443 1444 1445 1557 1440 1442 1442 | ~1443 | 177 | 1894 |
| cap32/73736 | (8,4,1) moe_8_4_1_4_1 | 1514 1434 1436 1435 1472 1432 1435 1435 | ~1434 | 136 | 1850 |
| cap32/73736 | (8,8,1) moe_8_8_1_3_1 | 1460 1430 1436 1435 1474 1428 1434 1433 | ~1432 | 50 | 1892 |
| cap32/73787 | 411 / 841 / 881 | | 1446 / 1431 / 1428 | 159 / 118 / 54 | 1847 / 1804 / 1858 |
| cap32/73658 | 411 / 841 / 881 | | 1446 / 1429 / 1426 | 163 / 108 / 56 | 1854 / 1788 / 1837 |

Native shapes (same tool, v1511 default build): G1_1 comp of the SMSP1-3 warps / sk0 / period: bs8 (4,1,1) 1441 / 163-165
/ 1808-1816; bs32 (8,4,1) 1427 / 123 / 1837; bs128 (8,8,1) MW4 1373-1381 / 44-46 / 1716-1793; bs256 (8,8,1) MW8 1370-1375
/ 38-40 / 1688-1723.

B. ncu SourceCounters (no code change; fixed --warp-sampling-interval 3; dev/wpw_ncu_loop.py), the ntl-1 stage loop,
sampled cycles per warp-iteration (units: samples x 256 / iterations; x ~12 = cycles), total (data wait / rest):
cap32/73736 on (4,1,1) 145 (wait not split: bottom-tested loop), (8,4,1) 145 (20 / 125), (8,8,1) MW1 152 (29 / 123);
native bs8 151, bs32 140 (16 / 124), bs128 135 (15 / 121), bs256 127 (6 / 120). Stall mix of the rest identical in all
(selected 42-44, wait 20-24, not_selected 21-23, math 21-23, dispatch 13-16, short_sb 2-3, no_inst ~1 per iteration).
Shared-memory bank conflicts 1.2-1.9% of load wavefronts in every 1-group run (bs128/256 2.9-3.4%: wgmma / epilogue code).

C. MOE_TIMING (dev/timing_ntl.py, the tool behind the historical numbers; timers inside the loop): G1 ntl1 compute per
stage bs8 ~1486 (743 x 2: the tool normalizes (4,1,1) items by 16 stages, they have 8), bs32 1571, bs128 1421, bs256 1420.

=> On v1511 the one-k-warp 1-group stage is NOT slower: on identical routings the 1-group loop computes in the same
cycles in all three configurations (trace 1426-1446 for the warps without a helper warp, ncu rest 123-125), and the
bs128 / bs256 binaries' loops are the fastest of all (trace 1370-1381, ncu rest 120-121, MOE_TIMING 1420). The
historical 1603-1621 vs 1429 compared an in-kernel MOE_TIMING number (timers inside the loop re-schedule it: WP-W's
artifact) of older binaries with a harness number; what was real then (SMSP0 producer skew, AEPI held slot) has since
been cut by WP-N (NBAR, LEAN3/4: sk0 in (8,8,1) now 38-56 vs 160-177 in (4,1,1)), WP-DS (DEF2 carry) and WP-IT.
The SMSP0 skew is now LARGEST in the (4,1,1) kernel (warps 0/4 +100 cycles of compute per stage), whose 1-group ring is
memory-paced at bs8 (WP-16W), so it costs nothing there.

D. Other candidate causes from the brief, checked in run 1:
- Clock: ncu sm__cycles_elapsed.avg.per_second bs8 1.81, bs32 1.73, bs128 1.66, bs256 1.62 GHz; all numbers above are in
  cycles, so the clock does not enter.
- Instruction cache: stall_no_inst in the 1-group (and 2-group) mma.sync loops is 0.1-0.9% of the loop's samples in
  every kernel (bs8 0.1%, bs32 0.4%, bs128 0.6%, bs256 0.8%): the 175 KB (8,8,1) kernel does not lose its 1-group loop to
  instruction fetch.
- Bank conflicts: see B (no difference). Register banks / schedule: different schedules (section 1) but the same
  cycles per iteration (B), so the schedule lottery did not land against the (8,*,1) loops this time.
- mbarrier wait placement: the (8,8,1) loop waits at its top (TOP form, DEF2 carry), (4,1,1) at its bottom; both cost
  the same per iteration.

## 3. Run 2 (dev/wpk1_run2.sh, kda_ws/runs/wpk1_run2.log): what does pace the (8,8,1) stages at bs128 / bs256

A. wgmma stage loops: run 1 showed 5-12% stall_no_inst in the n40-n64 HGMMA loops (bs256 n56 10.7%, n64 10.3%, n40 5.8%;
bs128 n64 10.4%, n40 5.2%; n24 / n32 1.8-2.7%). Hypothesis "the epilogue warps' code (20 KB executed per item, on SMSP2/3)
evicts the 9.4 KB loop": REJECTED. MOE_IT_BE (timing only: the epilogue warps do no work at all) leaves the no_inst
shares unchanged (bs256 n56 12.3%, n64 10.0%; bs128 n64 8.8%); MOE_IT_BH likewise. The no_inst samples sit on the
instructions that follow an HGMMA group (+0x230 ULOP3 after the stage-top wait 294 samples, +0x930 HMUL2 before
WARPGROUP.ARRIVE 108, +0xc30 / +0xf30 / +0x1830 after HGMMA pairs 105-176, ...), and the stage-end
WARPGROUP.DEPBAR.LE gsb0, 0x0 holds 1149 of the n64 loop's 12299 samples (barrier): the register-sourced HGMMAs' use of
the register file / dispatch while the decode wants it, as WP-WG concluded (wpwg.md section 7). Not an i-cache problem.
Microbenchmark (dev/wpk1_ifetch.py: 2 consumer warps per SMSP on a 9.4 KB straight-line loop, a helper warp on the same
SMSP running the SAME loop vs OTHER code of the same size, same issue load): other code costs +2..6% more than the same
code (sleep 0 / 500 / 2000 ns: +46.2 vs +52.1%, +40.2 vs +42.3%, +34.3 vs +39.5%); no capacity cliff between 2 and
16 KB of other code. Concurrent different code on one SMSP costs a few percent, not the 10-20% a thrashing loop would.

B. Per-warp trace bs256 / bs128 (wpk1_trace, 2 seeds each, 3 reps), v1511 vs MOE_IT_BE, comp per warp w0..w7:

| | G1_m (gate/up wgmma) comp | period | G2_m comp | period |
|---|---|---|---|---|
| bs256/73011 v1511 | 1747 1713 1802 1778 1783 1713 1814 1793 | 2219 | 1748 1717 1752 1737 1794 1718 1786 1757 | 2275 |
| bs256/73011 BE | 1742 1711 1710 1710 1789 1712 1711 1711 | 2213 | 1732 1714 1713 1714 1785 1717 1716 1717 | 2256 |
| bs256/71497 v1511 | 1749 1713 1778 1750 1785 1712 1791 1757 | 2138 | 1752 1713 1761 1745 1791 1713 1781 1760 | 2189 |
| bs256/71497 BE | 1741 1710 1710 1710 1786 1711 1711 1711 | 2125 | 1742 1710 1710 1710 1789 1712 1711 1711 | 2157 |
| bs128/71637 v1511 | 2078 2128 2124 2125 2164 2167 2166 2164 | 2471 | | |
| bs128/71637 BE | 2062 2093 2100 2097 2176 2175 2174 2173 | 2463 | | |

- bs256 wgmma stages: the SMSP2/3 warps (beside epilogue warps 10/11) compute 40-100 cycles per stage more than the SMSP1
  warps (beside the light signaler); with the epilogue work removed (BE) they drop to the SMSP1 level (1710-1712). The
  SMSP0 warps (beside the producer) stay +30 (w0) / +75-78 (w4) in both builds. The stage period moves only -6..-32
  cycles (0.3-1.5%) under BE because SMSP0's w4 then paces (max 1814 -> 1789). Taking the helper warps' interference
  off BOTH SMSP0 and SMSP2/3 would bring every warp to ~1711 (-100 cycles, ~5% of a wgmma stage, ~55% of bs256's stage
  time): an upper bound of ~2-3% at bs256 for a helper-warp layout change that nobody has a design for within the
  register budget (12 warps: no free warp on SMSP1-3; a 13th warp has 32 registers left; 16 warps leave 40 per helper).
  WP-IT's measured wall bound for the epilogue's work share alone (be - bh) is +0.75% at bs256 / +0.3% at bs128.
- bs128 wgmma stages: no per-SMSP pattern; warpgroup 1 (warps 4-7) computes 45-75 cycles longer than warpgroup 0 and
  waits correspondingly less (comp + gap equal): both warpgroups' HGMMAs share each SMSP's tensor unit, the second one
  drains later. BE changes nothing there.
- 1-group stages (G1_1) at bs128/256 are not compute-paced: comp 1366-1398 + gap 222-281 ~ 1600-1650 vs period 1688-1793;
  the excess is data (NS 3 ring with 69.6 KB slots: refill path turn ~270 + issue ~280 + DRAM latency 1600-2450
  cycles); BE does not change them (period 1702 -> 1709, 1714 -> 1732).

## 4. Conclusion (WP-K1)
- The "(8,*,1) 1-group stage at 1603-1621 vs (4,1,1) 1429 cycles" does not exist on v1511. Same routing, three
  configurations: the 1-group loop computes 1426-1446 cycles per stage on the warps without a helper warp in all of them
  (trace), ncu's non-wait cycles per iteration are equal (123-125), and the bs128 / bs256 binaries' 1-group loops are the
  fastest measured (1370-1381 trace, ncu 120-121, MOE_TIMING 1420). The historical numbers were MOE_TIMING builds (timers
  inside the loop re-schedule it, WP-W) of older binaries, compared with a harness figure; the real effects of that time
  (SMSP0 producer skew, slow slot turnaround) were removed by WP-N's NBAR / LEAN3 / LEAN4 and WP-DS's DEF2. Today the
  SMSP0 skew of 1-group stages is 38-56 cycles in (8,8,1) and 159-177 in (4,1,1).
- Nothing to recover in the 1-group loop, so no switch was built (no patch script, no v35xx file).
- What paces bs256 instead: 1-group stages are data-paced (NS 3 ring); wgmma stages are compute-paced, and there the
  helper warps' issue load sets the pace: epilogue warps 10/11 slow the SMSP2/3 consumers by 40-100 cycles per stage,
  the producer the SMSP0 pair by 30-78. Both must go to gain (removing only the epilogue's share moves the period
  0.3-1.5%); upper bound ~-100 cycles per wgmma stage (~2-3% of bs256). No register-feasible layout for it is known
  (section 3B). The wgmma loops' 5-12% stall_no_inst is not instruction-cache eviction (BE leaves it unchanged); it is
  the HGMMA register-file / dispatch interplay WP-WG measured.
- Process note: runs 1-2 predate the planner's stray-process rule (their GPU_BEFORE lines show an idle H200, 80 W,
  345 MHz); dev/wpk1_gpucheck.sh (gpucheck "label") prints the compute-apps / utilization lines for any later run. No
  bench2 / check_mt step was run by WP-K1 (smoke = hang1 at bs 8/32/128/256 + the forced cap32 configurations, all ok).
