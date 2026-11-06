# WP-W: why the 1-group stage costs more in the NT>1 binaries (2026-10-27)

Worker WP-W. Base dev/moe_v567.py. Files: dev/moe_v660.py (v567 + probe / timing switches, default byte-identical to
v567: per-function SASS hashes checked with dev/wpw_sass.py), dev/moe_v661.py (+ MOE_WPW_PWS), dev/moe_v662.py
(+ MOE_WPW_AREL / AREL0 / AESUS / PSUS), dev/wpw_probe.py (per-warp item probe), dev/wpw_ab.py (graph-replay A/B on the
same routings, cold L2, optional cap-8 routings), dev/wpw_ncu_loop.py, dev/wpw_ncu_one.py, dev/wpw_ncu_met.py,
dev/wpw_loops.py (SASS loop census), dev/wpw_sass.py (WP-C's compile script with a chosen instantiation list),
dev/wpw_run1.sh, dev/wpw_run2.sh. Logs: scratchpad wpw/run1.log, run2.log; ncu CSVs in dev/remote_out/wpw_ncu/.

## 1. The 1-group loop's compute is the same in every binary (run 1)

SASS (local NVRTC 13.0, sm_90a): the 1-group stage loop is 510 instructions in (4,1,1) (bottom-tested inner loop of
stages 2-8 plus a peeled first stage), 514 in (8,4,1), 513 in (8,8,1), 511 in (4,2,1): same opcode mix (32 HMMA,
19 LDS, 171 LOP3, 71-73 IMAD, 128 HMUL2, 68 SHF), schedules differ in detail (opcode-order similarity 0.32-0.44).

ncu, same routing forced onto each binary (bs8 seed 70261: 5 experts, all tiles 1 group; bs32 jm seed 73736 capped at 8
tokens per expert), SourceCounters at one fixed --warp-sampling-interval, so samples per loop iteration compare across
kernels (units arbitrary, x100):

| routing | binary | samples / iteration | of which the TRYWAIT branch (data wait, long_sb) | rest (compute) |
|---|---|---|---|---|
| cap32/73736 | (4,1,1) inner loop | 57.4 | 8.2 | 49.2 |
| cap32/73736 | (8,4,1) | 58.9 | 9.6 | 49.3 |
| bs8/70261 | (4,1,1) inner loop | 57.3 | 7.1 | 50.2 |
| bs8/70261 | (8,4,1) | 60.1 | 10.8 | 49.4 |

The non-wait part has the same stall mix in all of them (selected 43, not_selected 21-23, wait 21-24, math 20-22,
dispatch 14-15, short_sb 3 per iteration; (8,8,1) the same as (8,4,1)). Shared-memory bank conflicts are 1-2% of
wavefronts everywhere and lower in (8,*,1) (ld conflicts 46k vs 82k of 4.3M wavefronts at cap32). ncu kernel time
cap32: 78.4 us (4,1,1), 83.2 us (8,4,1), 87.3 us (8,8,1).
=> There is no compute penalty in the loop. WP-Q's MOE_TIMING split (1470 vs 1580 "comp") is an artifact of the timers
inside the loop (they change the schedule; WP-S S1 saw the same tool not add up to wall time).

## 2. Where the per-stage difference is (run 1 per-warp probe, MOE_WPW_PROBE: clock64 at item boundaries only)

cap32 routings (jm seeds 73736 / 73787 / 73658 capped at 8), cycles per stage, warp mean (warps 0..7):

| | G1 loop / stage | G1 epilogue / item | G1 per stage incl. epilogue | G2 loop / stage | G2 stages / item | G2 epilogue |
|---|---|---|---|---|---|---|
| (4,1,1) | 1760 / 1710 / 1716 | 261 | 1793 / 1743 / 1749 | 1642 / 1676 / 1678 | 8 | 260 |
| (8,4,1) | 1842 / 1761 / 1745 | 1459 / 1713 / 1804 | 1933 / 1868 / 1858 (+7.8 / +7.2 / +6.2%) | 1801 / 1727 / 1752 | 4.3 / 6.0 / 6.4 | 799 / 1048 / 1040 |
| (8,8,1) | 1869 / 1823 / 1796 | 1044 / 1206 / 1288 | | 1767 / 1723 / 1710 | 16 | 1333-1515 |

- In (8,4,1) warps 0 and 4 (SMSP0, with the producer) are 4-5% slower than warps 1-3 / 5-7 (1902 vs 1820); in (4,1,1)
  there is no such skew (1752 vs 1763). Through the item-end consumer barrier (AEPI staging) and the slot release that
  needs all 8 warps, the slowest pair sets every warp's pace.
- The (8,4,1) item epilogue costs the consumers 1460-1800 cycles per 16-stage item (consumer_sync wait for the slowest
  warp + staging) against SEPI's 261 per 8-stage item: +60-75 cycles per stage (+3.5-4%).
- The stage loop itself is +30-80 cycles per stage: the data waits ncu shows (section 1). The ring slot that the
  asynchronous epilogue holds (until warps 10-11 wake, read and release it) leaves a 3-slot ring at every item start.
- bs32's down phase in (8,4,1) is mostly MOE_KSX1=4 split items (nspl = one wave of down items): 4.3-6.4 stages per
  item, each with a first-stage wait and a 800-1050-cycle epilogue -> G2 +12..18% per stage over (4,1,1)'s whole items.
- MOE_NO_LOAD (no copies) does not isolate compute here: (8,4,1) 1675 vs (4,1,1) 1567 per stage with no data, because
  the producer's suspended slot wait and the epilogue warps' slot hold still gate the ring.
- Timing ablations (per-stage numbers only; the run-1 wall times were host-launched and are not valid):
  MOE_WPW_NOCSYNC (no staging barrier, wrong results): epilogue 1459 -> 430 cycles/item, loop +34 cycles/stage, net
  -1.2..-2.4% per G1 item. MOE_WPW_STAG=800 (warps 4-7 start each item 800 cycles late): loop -1.5..-2.7%.
  MOE_WPW_ISYNC on (4,1,1) (lockstep item start): loop -3..-5% with the spin outside the measured window (no
  half-stage-offset effect is needed to explain the gap).

## 3. Barrier-free staging and early slot release do not pay (run 2, dev/moe_v662.py, dev/wpw_ab.py)

Graph replays, cold L2, jm routings (cap32 = the same seeds capped at 8 tokens per expert), alternating order, ratio =
median of t(v567) / t(variant) per rep:

| variant | cap32 (60) | bs32 (80) | bs128 (50) | bs256 (50) |
|---|---|---|---|---|
| b (v567 copy) | 1.0004 | 0.9993 | 0.9984 | 0.9934 |
| MOE_WPW_PWS (barrier-free per-warp staging, 1-group items) | 1.0008 | 0.9981 | 0.9960 | 0.9921 |
| PWS + AREL (warps 10-11 copy their reads, free the slot, then SwiGLU / stores) | 1.0025 | 0.9989 | 0.9930 | 0.9843 |
| PWS + AESUS=200 | 0.9990 | 0.9990 | 0.9954 | 0.9904 |
| PWS + AREL + AESUS=200 | 1.0049 | 0.9969 | 0.9932 | 0.9833 |
| PSUS=200 (producer slot-wait hint) | 0.9999 | 0.9983 | 0.9979 | 0.9915 |
| NOCSYNC (timing: no staging barrier) | 1.0022 | 1.0026 | 1.0004 | 0.9934 |
| STAG=800 (timing: warps 4-7 late per item) | 0.9978 | 0.9948 | 0.9931 | 0.9917 |
| AREL0 (timing: every staged slot freed before it is read; NaN outputs) | 1.0167 | 1.0135 | 1.0173 | 1.0240 |
| c411 (bs32 buckets on (4,1,1)) | **1.0687** | 0.9591 | = | = |

check_mt (13 sizes 17-257, normal + skewed) 0 failures for PWS and PWS+AREL. Probe: PWS cuts the consumers' epilogue
from 1255-1782 to ~310 cycles per item, and the stage loop absorbs the same time: the slowest warp sets the pace.

## 4. Per-stage trace (run 3, dev/moe_v663.py MOE_WPW_TRACE, dev/wpw_trace.py; clock64 per stage, all 8 warps + producer)

cap32/73736, cycles, medians (gap = data-ready(i) - compute-done(i-1) of the same warp; lat = first ready - copies
issued; piss = producer acquire -> copies issued; turn = slot released by its last warp -> producer acquired):

| | (4,1,1) G1 | (8,4,1) G1 | (4,1,1) G2 | (8,4,1) G2 |
|---|---|---|---|---|
| compute w0 / w4 / others | 1471 / 1478 / 1454-1458 | 1508 / 1495 / 1445-1448 | 1421 / 1428 / 1453-1457 | 1531 / 1529 / 1442-1445 |
| gap w0 / others (median; mean ~420-500 all) | 375 / 249 | 298 / 215 | 366 / 247 | 268 / 215 |
| last warp to finish a stage | w0 in 96% | w0 in 80%, w4 9% | | |
| lat / piss | 2010 / 1305 | 3441 / 991 | 1539 / 1169 | 1960 / 1068 |
| turn free / held (item-last slot) | 1373 / 1398 | 435 / **4908** | 467 / 500 | **3272 / 2922** |

- Warp 0 releases (is the last to finish) almost every stage in both binaries; its compute is +1% over the other warps
  in (4,1,1) and +4% in (8,4,1) (SMSP0, with the producer). The producer issues the same instruction count per stage in
  both (ncu: 153 vs 137 executed warp-instructions per stage, 4.7 vs 3.0 bulk copies), so it is not a heavier producer.
- Release -> data back in the slot (turn + piss + lat) is ~4700-4900 cycles in both against ~5700 of 3-stage lead, so
  the 4-slot ring has <1000 cycles of slack; a held slot (AEPI) takes 4908 in (8,4,1) vs 1398 (SEPI frees at once).
- The (8,4,1) down phase at bs32: the producer reaches a released slot ~3300 cycles late (it spins on the tile's G1
  completion counters for the KSX=4 split items that start before their tile's gate/up is done).
- bs128 / bs256 multi-group gate/up items (both binaries are the one-k-warp ones there): held-slot turn 9238-9488
  (bs128) and 8452-8612 (bs256) cycles against a lead of ~3 x 2000 (NS 4) / 2 x 2300 (NS 3) cycles: the epilogue warps'
  SwiGLU + stores of a 32-64-token tile gate the next item's stage NS-1 (WP-Q's #2 diagnosis, confirmed).

## 5. Shorter epilogue SwiGLU (v664 / v665 MOE_WPW_FSW): small keep; the signaler joining (HELP) loses

MOE_WPW_FSW: the asynchronous-epilogue warps' SwiGLU as ex2.approx / rcp.approx with the product (g u)(gsc^2 r) (the
same FP32 approximations as __expf / __fdividef, without the latter's range code), the staged values loaded without
adds into zeros, and the wgmma-layout loop's (chunk, token) stepped instead of divided: epilogue loops 143 -> 95 and
180 -> 118 instructions; only the (8,4,1) / (8,8,1) kernels change ((4,*,1), moe1t byte-identical to v655).
bench2 jm 100 reps (wpw_run5), vs v655 (control v655b in brackets):

| | bs1 | bs8 | bs16 | bs32 | bs128 | bs256 |
|---|---|---|---|---|---|---|
| control | (1.0019) | (1.0005) | (0.9986) | (1.0009) | (0.9991) | (0.9975) |
| FSW x2 | 0.9910 / 1.0036 | 0.9997 / 0.9993 | 0.9982 / 0.9995 | 1.0035 / 1.0026 | 0.9989 / 0.9983 | 1.0028 / 1.0035 |
| FSW + HELP=4 x2 | 1.0055 / 1.0036 | 0.9978 / 1.0018 | 0.9963 / 1.0010 | 0.9977 / 0.9964 | 0.9928 / 0.9945 | 0.9964 / 0.9971 |
| FSW + HELP=1 | 1.0008 | 1.0009 | 1.0014 | 0.9976 | 0.9913 | 1.0001 |

check_mt 15 sizes (1-257, normal + skewed, 3 seeds, 2 calls) 0 failures for all three (max rel-L2 3.3e-3). FSW: about
+0.26% bs32, +0.44% bs256, flat elsewhere (graph runs wpw_run4/7: 0.999-1.006). HELP (warp 9 does a third of the
epilogue of tiles with >= n groups, help word + mbarrier past the ring, slot releases 3/3/2) costs SMSP1's consumers more
than the shorter hold saves.

## 6. What paces the ring: the consumer pair on the producer's SMSP (wpw_run6)

- MOE_WPW_PSWAP (logical producer on physical warp 9 by a tid remap): the compute skew moves with it. cap32 G1 compute per
  warp [1407 1543 1419 1418 1415 1533 1414 1413] (warps 1/5 +9%, the last-finishing warps become 1/5); wall time vs
  v655 1.0030 / 0.9983 / 1.0007 at bs32/128/256 (control 0.9984 / 1.0001 / 0.9967).
- Warp 0 has no per-stage duties (warp 4, as slow, has none at all): it is the SMSP, not warp-0 bookkeeping.
- Delay sensitivity (graphs vs v655): MOE_PDELAY=200 (producer sleeps 200 ns before each stage's copies) 0.939 / 0.950 /
  0.962; MOE_WPW_PBUSY +64 / +128 / +256 producer ALU instructions per stage 0.962 / 0.967 / 0.980, 0.913 / 0.918 /
  0.965, 0.826 / 0.832 / 0.925 (bs32 / bs128 / bs256). Per added cycle both slopes agree (~0.018% per cycle), i.e. PBUSY
  is a delay of the producer's path too.

## 7. Making the producer's path faster does not help (wpw_run7, run8)

| variant (vs v655) | bs8 | bs32 | bs128 | bs256 |
|---|---|---|---|---|
| control b (run7 / run8) | 1.0008 / 0.9947 | 0.9985 / 1.0027 | 0.9985 / 0.9997 | 0.9952 / 0.9972 |
| PSPIN 0 (tight test_wait poll) | 0.984 | 0.968 | 0.970 | 0.980 |
| PSPIN 32 / 64 / 100 / 150 / 250 / 500 ns | 0.996 / 0.997 / 1.004 / 1.001 / 0.993 / 0.999 | 0.998 / 0.998 / 1.001 / 1.000 / 1.000 / 1.001 | 0.996 / 1.000 / 1.000 / 0.998 / 1.000 / 1.001 | 0.994 / 0.998 / 0.996 / 0.994 / 0.997 / 0.993 |
| MOE_NO_SUSPEND_PS | 1.000 | 1.000 | 0.998 | 0.995 |
| PCF (weights before record / expect-tx; +1 spill pair) | 0.997 | 0.996 | 1.000 | 1.002 |
| BSTG=2 / =1 (consumer SwiGLU, BF16 buffer, slot freed at once) | 1.005 / 1.001 | 0.966 / 0.967 | 0.955 / 0.957 | 1.001 / 1.000 |

The producer's release -> copy path sits just inside its slack: later hurts (PDELAY, PBUSY), earlier gains nothing. A
consumer-issued copy (the last releaser issues stage s + NS from a producer-made descriptor) would only issue earlier,
so its prize is bounded near 0, while its ~25-40 instructions per stage would land on the releasing warp, the pacer.
Freeing the item's last slot early was measured three times (SEPI8, AREL, BSTG) without gain.

## 8. Answer

The 1-group stage is not slower in the NT>1 binaries: on identical routings the loop's compute is identical (ncu), and
MOE_TIMING's 1470 vs 1580 is an artifact. The x1.07 of the lean binary on 1-group work (cap8) comes from the item / ring
structure around the loop: (1) the slot-releasing consumer pair shares SMSP0 with the producer and computes 4% slower in
(8,4,1) (1% in (4,1,1)), and the 4-slot ring runs at that pair's pace; (2) the asynchronous epilogue holds the item's
last slot (4.9k cycles at bs32, 8.4-9.5k for multi-group items at bs128/256 vs a 4.6-6k lead); (3) at bs32 the down phase
runs as KSX=4 pieces whose producer waits on the tile's G1 counters (~3.3k cycles per slot) with 800-1050-cycle
epilogues per 4-6 stages; (4) the gather-barrier start (WP-K's v655 took part of it). None of PWS / AREL / AESUS / PSUS
/ HELP / BSTG / PSPIN / PCF pays; FSW (+0.26% bs32, +0.44% bs256) is the only keep. The remaining lever is the pacing
pair's 4%: the producer's instruction footprint on its SMSP, which only a different warp layout removes.
