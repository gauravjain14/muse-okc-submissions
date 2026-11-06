# WP-N: what in the producer slows its SMSP's consumers, and fixes (2026-10-27)

Worker WP-N. Base dev/moe_v730.py. Files (all switches off by default; default builds byte-identical to v730 per-kernel
SASS hashes): dev/moe_v740.py (trace + probes), v741 (+NBAR), v742 (+LEAN, LWSC, PBL), v743 (+LEAN2), v744 (+LEAN3),
v745 (+HALFW, SPLITW, NBC8); dev/wpn_trace.py (per-stage trace analysis), dev/wpn_run1..8.sh. Logs: scratchpad wpn/run*.log.
Generators: scratchpad wpn/mk74*.py.

## 1. Decomposition (wpn_run1/2, per-stage trace, cap32/73736, bs32/73736, bs128/71637, bs256/73011, 3 cold calls each)

sk0 = SMSP0 warps (w0, w4) comp+gap median minus the SMSP2/3 warps' (w2, w3, w6, w7); 1-group stages (G1_1 / G2_1).

| build | cap32 G1_1 sk0 (comp part) | bs32 | bs128 | bs256 | period cap32 G1_1 | producer piss | turn_free |
|---|---|---|---|---|---|---|---|
| v730 (trace) | 132 (76) | 106-111 (37-42) | 100-103 (28-30) | 120-128 (79-92) | 1848-1868 | 1070-1170 | 470-710 |
| +64 LOP3 / producer stage | 191 (131) | 160 | 152 | 156 | 1921 | 970 | |
| +64 FFMA | 206 (124) | 172 | 164 | 150 | 1909 | | |
| MOE_NO_LOAD (wait+record+arrive only) | 88 (17) | 76 (4) | 90 (10) | 62 (8) | 1723 | 530 | 352 |
| NOREC (record only at item ends) | 130 (78) | 101 | 98 | 125 | 1847 | | |
| NBAR (named-barrier release) | 139 (111) | 128 | 136 | 133 | 1901 | 1055 | 290-980 |
| NBAR + NO_LOAD | 78 (70) | 77 | 79 | 38 | 1724 | 243 | 224 |

- The producer's empty-slot try_wait does not poll: 0.03-0.27 false returns per stage (the TRYWAIT blocks in hardware; the
  NANOSLEEP.SYNCS retry path is almost never taken).
- Extra producer instructions after the copies cost the SMSP0 warps ~0.8-1.0 cycle each and the ring period the same.
- Without copies the compute skew disappears; the w0/w4 gap skew (+60-90, the stage boundary right after w0's release,
  when the producer wakes) stays. With NBAR the wake is ~300 cycles faster and the same cost lands in compute instead.

## 2. Lean producer (wpn_run5/6): fewer producer instructions do NOT shrink the skew

LEAN (v742): every stage whose activations need no wait takes a lean path (record only at item ends, per-lane copy
descriptors per item); LEAN2 (v743): the item's remaining stages in a tight loop (87 static instructions); LWSC: the
32-token config's weights + 6-bit scales as one combined block (lossless wc13_6 / wc2_6 layout, as (8,8,1) WSC).
check_mt 17 sizes 1..512 (normal + skewed, 3 seeds, 2 calls) 0 failures for LEAN, LEAN+NBAR+LEAN2, +LWSC.

| build | cap32 sk0 (comp) | bs32 | bs128 | bs256 | piss | period cap32 / bs256 |
|---|---|---|---|---|---|---|
| v730 trace | 132 (80) | 105 | 101 | 128 | 1165 | 1848 / 1892 |
| LEAN+NBAR+LWSC | 160 (127) | 137 | 124 | 114 | 725 | 1898 / 1842 |
| LEAN2+NBAR | 158 (132) | 152 | 130 | 107 | 867 | 1872 / 1826 |
| LEAN2+NBAR+LWSC | 149 (130) | 138 | 114 | 104 | 819 | 1916 / 1827 |

bench2 jm 100 reps (wpn_run6), vs v730 (control v730b): bs1 0.9975, bs8 0.9964, bs16 0.9995, bs32 0.9965, bs128 0.9963,
bs256 0.9950:

| | bs32 | bs128 | bs256 |
|---|---|---|---|
| NBAR | 1.0008 | 1.0023 | 1.0134 |
| LEAN+NBAR+LWSC | 0.9880 | 1.0011 | 1.0131 |
| LEAN2+NBAR | 0.9954 | 1.0009 | 1.0019 |
| LEAN2+NBAR+LWSC | 0.9928 | 1.0002 | 1.0055 |

The producer's stage path got 300-450 cycles shorter, yet the SMSP0 skew stayed or grew: removing 25-35 producer
instructions per stage does not give back the ~1 cycle each that adding them costs. bs256 (NS 3: two stages of lead)
gains from the faster slot turnaround (NBAR wake ~230 vs 470 cycles; LEAN issue).

## 3. The mechanism: divergent copy-issue waterfalls, not bytes, not polls (wpn_run7)

ptxas compiles every lane-predicated `cp.async.bulk` (`if (lane < k) bulk_g2s(...)`, operands in vector registers) into
an ELECT / 5x R2UR / 2x PLOP3 / UBLKCP / BRA.U.ANY loop with one iteration per active lane, inside BSSY / BSYNC / BRA.DIV
reconvergence. v730's producer runs 3 of these per 1-group stage (weights + scales on lanes 0-1, x on lane 0; wgmma tiles:
4 x lanes), ~108 warp instructions per stage, issued over ~1100 cycles while its SMSP's two consumer warps compute.

| probe (timing, trace) | what changes | SMSP0 skew cap32 G1_1 / G2_1, bs32 G2_1, bs128 G1_1 / G2_1 |
|---|---|---|
| v730 | | 137 / 120, 113, 99 / 123 |
| HALFW: half the weight bytes | bytes -50%, same instructions | 116 / 122, 118, 102 / 124 |
| SPLITW: weights as 2 x 16 KB | +1 waterfall iteration, same bytes | 164 / 152, 148, 129 / 148 |
| NO_LOAD (no copies at all) | | 88 / 88, 82, 90 / 72 |

One more waterfall iteration (~10 instructions) costs the SMSP0 pair ~30 cycles per stage (~3 per instruction, against
~1 per plain LOP3 / FFMA in the PBUSY calibration); the bytes do not matter. Fix: issue the copies from ONE lane chosen
with `elect.sync` from warp-uniform operands. Then ptxas emits ELECT + R2UR + UBLKCP (no loop, no divergence;
standalone test kernel dumped with cuobjdump: the waterfall disappears).

## 4. LEAN3: elected-lane copies in a tight per-item loop (v744/v745, baked in v748)

Once every remaining stage of an item needs no activation wait (gate/up after the gather; down once its chunks are
seen ready), the producer runs them in a 102-instruction loop: named-barrier slot wait (NBAR), then one elected lane
writes the record (first / last stage only: the consumers read an item's first record, the epilogue warps its last),
posts the expected bytes and issues the combined weights + 6-bit scales block (WSC layout; LWSC for (8,4,1)) and the
activation block (or the 4 wgmma k chunks) from per-item descriptors advanced by constant strides. ~70 instead of ~108
warp instructions per 1-group stage, no waterfall. Producer stage path 1166 -> ~300 cycles.

Trace (wpn_run7 / run10; sk0 / ring period per stage, cycles):

| | v730 | v748 |
|---|---|---|
| bs128/71637 G1_1 | 99 / 1817 | 103 / 1786 |
| bs128/72808 G1_1 | 106 / 1797 | 106 / 1770 |
| bs128 G1_m / G2_m (wgmma) | 78-88 / 2234-2256 | 58-80 / 2211-2229 |
| bs256/73011 G1_1 / G2_1 | 125 / 1897, 123 / 2056 | 81 / 1774, 77 / 1984 |
| bs256/71497 G1_1 | 124 / 1886 | 83 / 1770 |
| bs256 G1_m / G2_m | 96-128 / 2330-2421 | 130-146 / 2264-2374 |
| cap32 / bs32 (LEAN3 on the MW 1 kernel; not baked) | 137 / 105 | 92 / 76, but the other warps' compute +2.4% (codegen) |

At bs17-32 the (8,4,1) MW 1 kernel's consumer loop drew a slower schedule with LEAN3 (1425 -> 1460 cycles per stage on
the warps without the producer): the skew cut (-30..-45) was eaten, bench neutral (0.9964-1.0020). v748 keeps that
kernel byte-identical to v730 (MOE_WPN_MW1OFF).

## 5. Wall time (bench2, jm seeds, 100 reps, graphs, cold L2; ratios vs v730, control = second v730 copy)

| run | build | bs1 | bs8 | bs16 | bs32 | bs128 | bs256 |
|---|---|---|---|---|---|---|---|
| run8 | control | 0.9975 | 1.0004 | 0.9995 | 0.9979 | 0.9968 | 0.9969 |
| run8 | LEAN3 (all, incl. MW 1) x2 | 0.986 / 1.004 | 1.000 / 1.000 | 0.999 / 0.999 | 1.0004 / 1.0020 | 1.0063 / 1.0099 | 1.0148 / 1.0196 |
| run8 | NBAR x2 | | | | 1.0027 / 1.0011 | 1.0020 / 1.0026 | 1.0116 / 1.0109 |
| run9 | control | 0.9925 | 0.9988 | 1.0018 | 0.9961 | 0.9967 | 0.9967 |
| run9 | LEAN3 x2 | | | | 0.9975 / 0.9964 | 1.0047 / 1.0047 | 1.0144 / 1.0137 |
| run9 | LEAN3 + MW1OFF x2 (= v748) | 1.000 / 1.002 | 0.995 / 1.005 | 1.000 / 1.000 | 0.9961 / 0.9962 | 1.0070 / 1.0063 | 1.0147 / 1.0188 |
| run10 | control | 0.9830 | 1.0002 | 1.0011 | 0.9966 | 0.9975 | 0.9973 |
| run10 | v748 x2 | 0.993 / 0.990 | 1.0011 / 1.0006 | 1.0003 / 1.0017 | 0.9984 / 0.9951 | 1.0075 / 1.0083 | 1.0156 / 1.0146 |
| run10 | v748 + L3P (lean-path poll) x2 | | | | | 1.0055 / 1.0081 | 1.0145 / 1.0163 |

bs1-32 run byte-identical kernels in v748 (their spread is the noise). Over runs 9-10 against the control means:
bs128 +0.8..1.0%, bs256 +1.6..1.9%; projected geomean +0.4%.
check_mt (M 1 2 4 8 9 16 17 32 33 64 65 128 129 256 257 384 512, normal + skewed, 3 seeds, 2 calls): 0 failures,
max rel-L2 3.30e-3, for NBAR, LEAN, LEAN2, LEAN3 (all / NBC8 / MW1OFF), L3T, L3P, SPLITW and v748.

## 6. LEAN4 and the final build (wpn_run11 / run12)

LEAN4 (v749): the lean loop's steady stages (the slot's previous stage is the same item's, not the item's last) in a
sub-loop with no record, no held-slot bookkeeping, an immediate 288-thread count. Trace bs256: G1_1 sk0 78 -> 60
(compute part 66 -> 8), wgmma G1_m sk0 147 -> 68, period 1786 -> 1754. bs128: sk0 106 -> 53 but the (8,4,1) consumer
loop drew a slower schedule (non-producer warps 1381 -> 1455 cycles/stage): bench equal to v748 -> baked for (8,8,1) only.

| run | build | bs1 | bs8 | bs16 | bs32 | bs128 | bs256 |
|---|---|---|---|---|---|---|---|
| run11 | control | 1.0074 | 0.9973 | 0.9995 | 0.9967 | 0.9968 | 0.9953 |
| run11 | v748 x2 | | | | 0.9988 / 0.9970 | 1.0047 / 1.0059 | 1.0160 / 1.0143 |
| run11 | LEAN4 all x2 | | | | 0.9988 / 0.9998 | 1.0046 / 1.0057 | 1.0224 / 1.0287 |
| run12 | control | 1.0048 | 0.9956 | 1.0004 | 0.9968 | 0.9993 | 0.9967 |
| run12 | **v749 (final) x2** | 1.0049 / 1.0089 | 0.9959 / 0.9985 | 0.9981 / 1.0019 | 0.9961 / 0.9996 | 1.0071 / 1.0094 | 1.0217 / 1.0247 |

dev/moe_v749.py = the integration candidate: NBAR + LEAN3 (elected-lane copies, tight loop) + LWSC for the (8,4,1) MW
2 / 4 kernels (bs33-128), + LEAN4 for (8,8,1) (bs129-512); every other kernel byte-identical to v730 (per-kernel SASS
hashes; MOE_NO_WPN = v730 everywhere). hang1 13 shapes ok, check_mt 17 sizes 0 failures, no C7511/C7513, producer stage
loops spill-free (a few per-item spill reloads outside them at 72 registers). vs the control means: bs128 +0.6..1.0%,
bs256 +2.5..2.8%, geomean ~+0.5%.

## 7. What is left

- The remaining SMSP0 skew at bs128/256 is 50-80 cycles per stage (the elected block's ~45-70 instructions: R2URs
  of loop-invariant operands are re-issued every stage; the record at item ends; the BAR wake). A 4-producer layout
  (one stage-issuing warp per SMSP, a 4th warpgroup) would spread it, but changes the launch to 512 threads.
- bs17-32: the same LEAN3 cut the skew (105 -> 76) but ptxas scheduled the MW 1 kernel's consumer loop 2.4% slower;
  a codegen-neutral formulation for that kernel would be worth ~+1.5% at bs32.
- (4,1,1) / (4,2,1) (bs8/16) issue their per-token x rows as an n-lane waterfall (n iterations per stage): the same
  elected-lane rewrite applies (row pointers per item in shared memory), not built.
- NBAR alone is worth +1.1% at bs256 (faster slot turnaround at NS 3), included in v749.

## 8. Follow-up round (v760-v764; base v749, then v752)

| build | change | result |
|---|---|---|
| v760 MOE_WPN_X1 | (4,1,1)/(4,2,1): the stages issued by one elected lane (combined block, token rows one by one from the pair list) | consumer loops unchanged (4_2_1 same modulo registers, 4_1_1 sim 0.996); trace producer path 1343 -> 1078 (bs8), 1242 -> 952 (bs16); SMSP0 skew bs8 149/96 -> 128/61, bs16 150/86 -> 153/62; 1-group period bs8 -0.7%, bs16 -3%; bench2 bs8 0.998 / 0.997 / 1.000, bs16 0.997 / 0.995 / 0.998 (control 0.9985 / 0.9936): neutral, rejected (bs8/16 are not paced by the 1-group stage) |
| v761 MOE_WPN_ELX | one-k-warp full path (every bs17-32 stage) with elected-lane copies | MW1 consumer loops byte-identical to v749, but skew unchanged (111/116 -> 110/124), bs32 0.9906 / 0.9904 / 0.9926 (control 0.9975): rejected |
| v763 MOE_WPN_L3M | LEAN3 loop with mbarrier waits in the MW1 kernel | MW1 consumer loop still re-scheduled (sim 0.44 vs v752) and a spill pair inside the producer loop: not run |
| v763 MOE_WPN_1TNB | moe1t (bs1): consumers -> warp 9 -> producer slot hand-offs on named barriers (no 200 ns / 1 ms sleeping try_waits), warp-wide producer loop with an elected-lane issue (no waterfall) | only moe1t changes; check_mt 0 failures; bench2 200 reps bs1 1.0141 / 1.0077 (control 1.0000), bs8 (identical kernel) 0.9937 / 1.0015 (control 0.9980) |

bs32 conclusion: the skew there only falls with LEAN3's tight loop, and every LEAN3 formulation tried (NBAR or mbarrier
waits) changes ptxas's schedule of the MW1 kernel's 1-group consumer loop (opcode-order similarity 0.44-0.65 vs v730),
which measured 2.4% slower; formulations that keep the consumer loop byte-identical (ELX) keep the producer's general
path and do not cut the skew.

v764 = v752 + MOE_WPN_1TNB baked (escape MOE_NO_WPN1T): wpn_run15 bs1 1.0131 / 1.0039 (six-shape run, control 1.0053)
and 1.0076 / 1.0083 / 1.0051 at 300 reps against controls 1.0000 / 0.9974 / 0.9914: +0.8..1.1% at bs1 over two jobs;
the other kernels are byte-identical to v752; check_mt 17 sizes 0 failures. Integration candidate: dev/moe_v764.py.

## 9. bs128 residual SMSP0 skew (base v837; v850-v852, wpn_run20)

ncu SASS counts (profile/v764_ncu/cs_bs128.csv, moe_8_4_1_4_4; the producer code is v749's): 10.2 items and 110 stages per
CTA, 94% of the stages in the LEAN3 elected-lane loop at ~77 warp instructions per stage (1-group stages ~60, wgmma tiles
~92: their x as 4 k-chunk copies); 6% in the full path. bs256 (8,8,1) runs LEAN4's steady sub-loop (~45 per stage).
Other SMSP0 work: none per stage (warp 8 only; warp 0's per-item aq_slot write). The signaler (warp 9, SMSP1) re-polls
its per-item epi wait ~4500 times per CTA (NANOSLEEP.SYNCS 200 ns wakes on every mbarrier event) without a measurable
SMSP1 skew.

| build | consumer loops vs v837 | producer | bs128 bench2 (control 0.9967) | trace bs128 sk0 G1_1 / G2_1 |
|---|---|---|---|---|
| v837 | | piss 299-308 | 1 | 71-74 / 87-92 |
| L4ALL (inline LEAN4 at MW4) | flipped (1100/465/499 = v749-L4ALL's, measured +5% compute on every warp in run 11) | | not run | |
| v850 L4NI (LEAN4 sub-loop __noinline__) | the same flipped schedule | | not run | |
| v852 X4 (one 16 KB x copy for wgmma tiles) | same modulo registers | loop 102 -> 69 static | 0.9963 / 0.9976 | |
| v852 X4 + T4 (immediate counts, record hint) | same modulo registers | loop 67 static, piss 197-204 | 0.9957 / 0.9996 | 92-98 / 90-94 |

Cutting the MW4 producer path by a third (piss -35%) leaves the SMSP0 skew and the wall time unchanged: at bs128 the
residual (~70-100 cycles/stage, as the ~60-75 floor of the minimal producers: NBAR + NO_LOAD 75, LEAN4 at bs256 60) is
a per-stage cost of the producer warp being woken and running a short burst on SMSP0, not its instruction count. The
only cut measured so far is LEAN4's (bs128 sk0 106 -> 53 in run 11), and every LEAN4 formulation re-schedules the MW4
consumer loops into a 5% slower form. Remaining levers: (a) two producer warps alternating stages on two SMSPs (warp 9
freed by moving the signal posts to the epilogue warps), halving the per-SMSP floor; (b) freezing the consumer stage
loops' codegen (per-(NT, NTL) __noinline__ item functions, WP-Q lever #3) so that LEAN4-class producer changes stop
moving the consumer schedule.

## 10. (4,1,1) / (4,2,1) item starts (base v837; wpn_run21, dev/wp32_stage.py; cycles mean / median)

| bs8 (3 seeds) / bs16 (3 seeds) | first stage | middle stages | last stage |
|---|---|---|---|
| gate/up 1-group items: period | 2311-2394 / 2482-2631 | 1937-2014 / 1944-1964 | 1824-1988 / 1859-1939 |
| gate/up: mean consumer gap | 693-761 / 820-947 | 382-438 / 346-364 | 296-415 / 278-344 |
| down whole items: period | 2185-2235 / 2371-2637 | 1840-1941 / 1880-1934 | 1778-1829 / 1829-1860 |
| down 2-stage k pieces (bs8 last round): period | 2556-2839 | | 1887-1948 |
| producer: slot acquired - previous copies issued (pidle) | 1592-2423 | 517-622 | 547-633 |

Every item's first stage costs +350..+750 cycles (bs8 +400, bs16 +550..+700): the producer reaches its slot ~1100-1800
cycles late. Its item-start path, in SASS (moe_4_1_1_4_1): the claim of the next item is warp-aggregated by ptxas
(VOTEU / FLO / POPC / ATOMG / SHFL.IDX of the atomic's result) despite the inline-PTX lane-0 form, and that SHFL holds
the producer for the global atomic's round trip right at the item start, before the decode (~25 instructions), the tile
lookup and the address setup (~60). The same aggregated claim is in the (8,*,1) kernels (WP-32's item-start finding).

Fix tried (dev/moe_v870.py / v871.py, wpn_run22; consumer loops byte-identical to v837, producers spill-free):
MOE_WPN_CLM = the claim through an address offset ptxas cannot prove zero (head + (lane & (M >> 31))): a plain ATOMG,
the result first read at the item's end. MOE_WPN_PN4 = MOE_PNEXT's mid-item decode for the (4,*,1) configs only.

| | v837 | CLM | CLM + PN4 |
|---|---|---|---|
| producer item-start lateness (pidle, first stage) bs8 / bs16 | 1667-1992 / 1578-1697 | 1009-1025 / 885-905 | 1037-1050 / 920-942 |
| first gate/up stage period bs8 / bs16 | 2456-2566 / 2432-2629 | 2223-2237 / 2408 | 2211-2529 / 2471-2481 |
| bench2 bs8 (control 0.9997) | 1 | 0.9985 / 0.9966 | 0.9995 / 1.0015 |
| bench2 bs16 (control 0.9977) | 1 | 0.9982 / 0.9974 | 0.9949 / 0.9987 |

The aggregated claim's round trip was ~650-970 cycles of each item-start delay. Removing it shortens the first stages,
but bs8/16 do not move: the ring there is memory-paced, and the consumers' idle at an item start is off the critical
path. Rejected for (4,*,1). The decode (PN4) is not part of the remaining ~450 cycles. The same aggregated claim sits at
every item start of the (8,*,1) kernels, which are compute/latency-paced: handed to WP-32 (one-line change; the CLM
expression with the KW == 2 condition removed).

## 11. LEAN4 at MW 4 on v944 (dev/moe_v950.py, wpn_run30)

v944 re-allocated the MW 4 kernel, so the LEAN4 forms land elsewhere than in v749. Local SASS vs v944: MOE_WPN_L4NI
(__noinline__ wpn_steady) keeps all four MW 4 consumer loops byte-identical; MOE_WPN_L4M4 (= L4ALL's MW 4 code) keeps
the 1-group and wgmma loops the same modulo registers and re-schedules the 2-group loop (similarity 0.54); neither is
the known-slow v749-L4ALL schedule. Both producers are spill-free in their copy loops; check_mt 17 sizes 0 failures.
bench2 jm 100 x2 vs v944 (control 0.9986): bs128 L4NI 0.9952 / 0.9950, L4M4 0.9945 / 0.9953; the other shapes run
identical kernels. Even with the consumer schedule untouched (L4NI), the steady sub-loop loses ~0.4% at bs128:
not baked. With X4/T4 (section 9) this is the second clean measurement that a cheaper per-stage producer path does not
pay at bs128.
