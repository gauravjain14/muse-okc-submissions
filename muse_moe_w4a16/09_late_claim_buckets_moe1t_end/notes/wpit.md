# WP-IT: consumer-side item transition of the (8,8,1) 64-token kernels (bs65-512), 2026-11-01

Worker WP-IT. Base dev/moe_v1502.py (integration build). Namespace v2100-v2199, defines MOE_IT_*, scripts dev/wpit_*.
Target: moe_8_8_1_3_4 (bs65-128, MW 4) and moe_8_8_1_3_8 (bs129-256, MW 8); WP-L2 measured ~1 us per item boundary at
bs256 / ~0.55 us at bs128, consumer-side (wpl2.md runs 4-9).

Files:
- dev/wpit_mk2100.py -> dev/moe_v2100.py: v1502 + timing-only bound switches (all off by default; default build = v1502
  SASS in all 16 kernels, local NVRTC 13):
  MOE_IT_BS (no item-end consumer_sync), MOE_IT_BST (no staging stores, runtime-false P.M < 0 guard), MOE_IT_BF (no
  consumer fence.proxy.async), MOE_IT_BH (held slot released at the epilogue warps' wake), MOE_IT_BE (epilogue warps do
  no work: release + signal only). Only the three (8,8,1) kernels change; every consumer stage loop (8 per kernel:
  ntl 1, ntl 2, wgmma n24..n64) byte-identical to v1502 in every bound build (dev/wpl2_loops.py).
- dev/wpit_mk2101.py -> dev/moe_v2101.py: + MOE_IT_XS (real candidate, below).
- dev/wpit_mk2102.py -> dev/moe_v2102.py: + MOE_IT_S6E (real candidate, below).
- dev/wpit_edge.py: per-warp item-boundary decomposition (MOE_WPN_TRACE): slowest warp, its own transition, slowest-warp
  periods across the boundary vs mid-item, first-stage compute excess per warp, data waits of the next item's stages 1-2,
  total boundary excess per CTA.
- dev/wpit_cmp.py: output comparison of two builds (and run-to-run noise of the first).
- dev/wpit_runN.sh; logs kda_ws/runs/wpit_runN.log.

## Candidates (loop-identical formulations)

MOE_IT_XS (v2101): in the (8,8,1) TMP kernels (MW <= 8), an item of <= 16 tokens (ntl 1-2, the mma.sync tiles: the
majority of items) stages its FP32 accumulators into its held last slot's activation area past the first 16 activation
rows (slot bytes WB + SB + 16 XP = 44288 .. 60928 of 69632). In that stage every warp reads only activation rows 0..15
(rows >= n are the padded columns of the last token group, never stored), weights / scales of its own row groups: no
staging store can hit a byte another warp may still read, so the item-end bar.sync 1, 256 is skipped for these items.
Layout unchanged (token-major [tok][RB + 4]), so the epilogue warps only add the offset (from f.n <= 16) to the slot
base; TMP's bulk copy source is 128-byte aligned. Items > 16 tokens (wgmma tiles, whose activation rows fill the area)
keep the barrier. Codegen: the 8 consumer stage loops of moe_8_8_1_3_4 / _3_8 byte-identical to v1502, same instruction
count and spills; the barrier becomes a predicated BAR.SYNC; moe_8_8_1_3_16 (MW 16, no TMP room) and all other kernels
identical. MOE_IT_XSB = the offset with the barrier kept (control for the address change).

MOE_IT_S6E (v2102, with XS): the consumers stage the raw accumulators and the epilogue warps fold MOE_S6O's x64 into the
expert scale (gsc * 64; power-of-two scaling, bitwise the same products): the item end loses one FMUL per staged
accumulator (8 per thread at ntl 1 .. 64 at ntl 8). Alone it re-schedules all 8 stage loops (lottery); with XS the 8
loops are byte-identical to v1502 (80 instructions fewer per kernel). An unused constexpr for it re-allocated registers in
moe_4_2_1_4_1's routing code, so the switch's kernel condition is a macro (default build and S6E builds: only the two
(8,8,1) TMP kernels differ).

MOE_IT_GC (v2105, wpit_mk2105.py): items of <= MOE_IT_GCN tokens (default 12, the items TMP copies whole) reach the
epilogue's room by a generic copy of the 64 epilogue threads (LDS.128 / STS.128) instead of the smem -> smem bulk copy,
so nothing reads their staging through the async proxy and the consumers skip their fence.proxy.async (MEMBAR.ALL.CTA +
FENCE.VIEW.ASYNC.S on every consumer warp, warp 0's critical path) for them; larger items keep the bulk copy and the
fence. Epilogue: copy, fence.proxy.async per epilogue thread (the room's generic writes before a later item's bulk copy
into the room: async-proxy WAW), bar.sync of the two epilogue warps, slot release, work from the room. GCN=64: every
item, consumer fence gone. All four GC builds (GC, GC64, with / without XS + S6E): 8 / 8 stage loops byte-identical in
moe_8_8_1_3_4 and _3_8; other kernels identical.

MOE_IT_DB (v2104, wpit_mk2104.py; built, NOT benched): full-k down items staged by the consumers as the final BF16 y rows
(bf16(acc * 64 * gsc), bitwise v1502's values) with one stmatrix.x4.trans per token group, moved by the epilogue warps
with one 512 B cp.async.bulk.global.shared::cta per token (release after wait_group.read, signal after wait_group 0).
The epilogue half alone keeps every stage loop; any consumer-side BF16 staging re-schedules all 8 loops of both kernels
(9 formulations: stmatrix x4 / x2, one asm block for mul + cvt + stmatrix, opaque scale, pre-pass multiply, STS.U16 in
the FP32 loop or a separate loop, with and without XS / GC; the mma.sync loops keep their opcode order with other
registers, the wgmma loops gain 3-5 UMOV). Not run (loop rule).

MOE_WPA_TMG_ALL (existing WP-A switch, v984 baked it for the then bs65-128 (8,4,1) kernel only): the global-scratch spill
of tokens 12.. of the held slot on the (8,8,1) MW 4 / 8 kernels too (bs65-128 runs on (8,8,1) since v1033 and lost it).
Stage loops byte-identical in both kernels (epilogue-only).

## Log
### run 1 (dev/wpit_run1.sh, kda_ws/runs/wpit_run1.log)
Smoke ok (bounds finish; XS 6/6 ok at bs 32-257). check_mt XS (M 64 65 100 128 129 200 256 257, normal + skewed, 3 seeds,
2 calls): 0 failures. dev/wpit_cmp.py XS vs v1502: max |a - b| equal to v1502's own run-to-run KRED noise (a - a') in
every case, 99.99-100% bitwise-equal elements, worst rel-L2 1.6e-5.

Per-warp boundary decomposition on v1502 (dev/wpit_edge.py, MOE_WPN_TRACE, 3 jm seeds x 2 reps, cycles; class = ending
item -> next item; medians; the trace build spills 68 B more than the real kernel, so its transition times are inflated):

| class (bs256) | n | skew | slowest warp | slowest's own transition | first warp starts | P1 / Pm | excess | w3 (next stage 2 wait) |
|---|---|---|---|---|---|---|---|---|
| G1n1 -> G1n1 | 1512 | 1138 | w0 99% | 1291 | +1263 | 2909 / 1647 | 1210 | 209 |
| G2n1 -> G2n1 | 1800 | 564 | w0 95% | 1343 | +1311 | 2931 / 1654 | 1316 | 241 |
| G1n2 -> G1n1 | 528 | 1159 | w0 54%, w4 31% | 1326 | +1276 | 2995 / 1647 | 1427 | 220 |
| G1n3 -> G1n1 (wgmma) | 264 | 79 | w0 59% | 1314 | +1284 | 3031 / 1648 | 1653 | 250 |
| G1n8 -> G1n2 | 128 | 810 | w0 72% | 1562 | +1519 | 3688 / 1975 | 6446 | 4677 |
| G2n8 -> G2n3 | 64 | 884 | w0 77% | 2208 | +2172 | 4365 / 2123 | 5647 | 3312 |
Total boundary excess (sum over a CTA's boundaries of the slowest-warp periods of the next item's first 3 stages minus
3 mid-item periods): bs256 20.3k cycles = 11.9 us per CTA = 6.8% of the traced span; bs128 16.7k = 9.8 us = 6.8%.
- Warp 0 (SMSP0, beside the producer) finishes the last stage last in 95-99% of the 1-group boundaries, ~1100 cycles after
  the fastest warps; the barrier makes every warp start the next item ~1300 cycles after warp 0's last compute; the next
  item's stages 1-2 then run at the normal period (w2 / w3 ~220 cycles) unless the ending item had > 32 tokens: then its
  held slot (stage NS - 1 = 2 of the next item) waits 2-5k cycles for the epilogue warps (the large classes above).
- First-stage compute excess per warp: warps 2, 3, 6, 7 (SMSP2/3, beside the epilogue warps) +150-300 cycles, warps 0, 1,
  4, 5 +0-100.
- Without barrier / stores / fence (BSA) warp 0's own transition is still ~900-1000 cycles and the earliest warp starts at
  +150 (G1n1): the next item's stage-0 data is there ~150 cycles after warp 0's last stage; the excess drops 1210 -> 666
  (G1n1), total 6.8% -> 5.2% (bs256) / 5.0% (bs128). With BH as well: 4.0% / 3.9%. XS: G1n1 excess 969, total 6.4% / 6.1%.

bench2 B2_NOVERIFY jm 100 reps, 2 passes, ratio vs the first v1502 copy (non-first copies v1502b / c are the controls):

| variant | bs256 p1 / p2 | bs128 p1 / p2 | bs32 (placebo) p1 / p2 |
|---|---|---|---|
| v1502b / v1502c (controls) | 0.9952 / 0.9969, 0.9952 / 0.9978 | 1.0001 / 0.9971, 0.9985 / 0.9984 | 0.9989 / 0.9994, 0.9974 / 0.9977 |
| bs (no item-end bar.sync) | 0.9992 / 0.9990 | 1.0033 / 1.0014 | 0.9985 / 0.9978 |
| bsa (no bar.sync, no staging stores, no fence) | 1.0072 / 1.0076 | 1.0109 / 1.0083 | 1.0000 / 1.0000 |
| bh (held slot freed at the epilogue's wake) | 1.0079 / 1.0083 | 1.0080 / 1.0048 | 0.9995 / 0.9988 |
| bsah (bsa + bh) | 1.0204 / 1.0207 | 1.0181 / 1.0163 | 0.9983 / 0.9978 |
| be (no epilogue work; slot freed at wake) | 1.0150 / 1.0166 | 1.0096 / 1.0092 | 0.9985 / 0.9986 |
| XS | 0.9981 / 0.9967 | 1.0004 / 0.9980 | 0.9997 / 0.9986 |
| XS copy | 0.9987 / 0.9990 | 1.0021 / 0.9988 | 0.9996 / 0.9986 |
| XSB (XS offset, barrier kept) | 0.9983 / 0.9973 | 1.0008 / 0.9991 | 0.9994 / 0.9988 |
Prize vs the control mean (bs256 0.9963, bs128 0.9985): bar.sync alone +0.3% / +0.35%; barrier + staging stores + fence
+1.1% / +1.1%; held slot +1.2% / +0.8%; both +2.4% / +1.9%; epilogue work (be - bh) +0.75% / +0.3%. XS +0.2% / +0.15%,
the same as XSB: removing the barrier for <= 16-token items buys nothing measurable (warp 0 is the last warp anyway, so
the barrier never delays it); the cost is in the work on warp 0's path (stores + fence) and the held slot.

### run 2 (dev/wpit_run2.sh, kda_ws/runs/wpit_run2.log): decomposition of the bound, held slot by phase, XS + S6E
Smoke ok (bounds finish; XS + S6E 6/6 ok). check_mt XS + S6E (M 64 65 100 128 129 200 256 257, normal + skewed): 0 failures;
wpit_cmp vs v1502: worst rel-L2 2.2e-5 = v1502's own KRED run-to-run noise (bitwise the same math).
bench2 B2_NOVERIFY jm 100 reps x 2 passes (dev/wpit_tab.py), % vs the mean of the three non-first v1502 copies
(bs256 0.9956, bs128 0.9973, bs32 0.9991; the controls agree within +-0.05%):

| variant (v2103 / v2101 / v2102) | bs256 | bs128 | bs32 (placebo) |
|---|---|---|---|
| bf: no consumer fence | +0.21 | +0.27 | -0.09 |
| bst: no staging stores (barrier + fence kept) | +0.84 | +0.74 | -0.16 |
| bsf: no barrier, no fence (stores kept) | +0.35 | +0.26 | +0.08 |
| bsst: no barrier, no stores (fence kept) | +1.13 | +1.09 | -0.01 |
| bsa: none of the three | +1.22 | +1.42 | +0.07 |
| bsh: half the staging stores (row group 0 only) | -0.11 | +0.04 | -0.01 |
| bh1: held slot freed at wake, gate/up items only | +0.81 | +0.63 | -0.13 |
| bh2: same, down items only | +0.40 | +0.20 | -0.02 |
| XS | +0.22 | +0.20 | +0.04 |
| XS + S6E (list position 12) | +0.24 | +0.41 | -0.20 |
| XS + S6E copy (position 15) | +1.03 | +0.80 | -0.00 |
- The item-end bound is mostly the staging stores (+0.75..0.85%), then the barrier (+0.1..0.35%) and the fence (+0.1..0.2%).
  Halving the stores gains nothing, so it is not store bandwidth: removing all of them also removes the S6O FMULs and
  (with every item's staged data being stale slot bytes) changes the data the epilogue / down phase compute on (power:
  WP-L2 measured a fast power limiter at these shapes), so the "no stores" bounds may overstate the store cost.
- The held slot costs more for gate/up items (+0.6..0.8%) than for down items (+0.2..0.4%).
- XS + S6E: both copies above every control at both shapes, but the two identical copies differ by 0.8% (bs256) / 0.4%
  (bs128) between list positions: per-variant buffer placement / position noise of up to ~0.8% for a modified kernel
  (the v1502 copies happened to agree). Replication with three copies at spread positions in run 3.

### run 3 (dev/wpit_run3.sh, kda_ws/runs/wpit_run3.log): XS + S6E replication, GC, TMG_ALL (verified bench2)
Smoke 5 builds 6/6 ok at bs 32-257. check_mt (M 64 65 100 128 129 200 256 257, normal + skewed, 3 seeds, 2 calls) 0 failures
for XS + S6E + GC, XS + S6E + GC64, TMG_ALL, XS + S6E + TMG_ALL; wpit_cmp vs v1502 worst rel-L2 2.0-2.2e-5 (= KRED noise).

LIST-POSITION BIAS (new, matters for every bench2 A/B here): the LAST variant of the list measures fast like the first one.
bench2 alternates the order every rep (0..N-1, N-1..0, ...), so the first and the last variants are replayed twice in a
row at each turnaround; the 512 MB flush empties L2 but not the SMs' instruction caches, so those two run with warm code
(a different variant's cubin ran in between for all others). Run 3: v1502e (last) +0.51% (bs256) / +0.31% (bs128) over
the middle copies v1502b-d; run 2: XS + S6E last +1.03% vs the same build in the middle +0.24%. => controls and
candidates are compared in the middle of the list only (first = reference, last = an excluded v1502 copy from run 4 on).
(Side fact: cold instruction fetch is worth ~0.3-0.6% of these kernels.)

% vs the mean of the middle controls v1502b / c / d (bs256 0.9954, bs128 0.9981; bs32 0.9982):

| variant (dev/moe_v2105.py) | position | bs256 p1 / p2 | % | bs128 p1 / p2 | % | bs32 % |
|---|---|---|---|---|---|---|
| XS + S6E | 2 | 0.9982 / 1.0000 | +0.37 | 1.0038 / 1.0016 | +0.46 | +0.02 |
| XS + S6E | 8 | 0.9964 / 0.9977 | +0.16 | 1.0014 / 1.0018 | +0.35 | -0.11 |
| XS + S6E | 12 | 0.9966 / 0.9991 | +0.25 | 1.0010 / 1.0011 | +0.29 | +0.04 |
| XS + S6E + GC | 4 | 0.9951 / 0.9956 | -0.01 | 0.9974 / 0.9971 | -0.09 | +0.11 |
| XS + S6E + GC | 13 | 0.9952 / 0.9988 | +0.16 | 1.0000 / 0.9974 | +0.06 | +0.08 |
| XS + S6E + GC64 | 7 | 0.9879 / 0.9917 | -0.56 | 0.9969 / 0.9969 | -0.12 | -0.02 |
| GC64 | 11 | 0.9854 / 0.9891 | -0.82 | 0.9927 / 0.9947 | -0.44 | -0.09 |
| TMG_ALL | 5 | 0.9997 / 1.0006 | +0.48 | 1.0019 / 1.0020 | +0.38 | +0.00 |
| XS + S6E + TMG_ALL | 9 | 1.0040 / 1.0041 | +0.87 | 1.0059 / 1.0048 | +0.72 | +0.00 |
| v1502e (last) | 14 | 1.0015 / 1.0028 | +0.67 | 1.0028 / 1.0017 | +0.42 | +0.17 |
- XS + S6E over runs 2-3 (4 middle copies): bs256 +0.24 / +0.37 / +0.16 / +0.25% (mean +0.26%), bs128 +0.41 / +0.46 /
  +0.35 / +0.29% (mean +0.38%); bs32 placebo (identical kernel) -0.2..+0.04%.
- GC rejected: the epilogue's generic copy costs more than the consumer fence it removes (GC64, which also copies the
  first 12 tokens of every big item before the slot work, lengthens their hold: -0.4..-0.8%).
- TMG_ALL (+0.4..0.5%) and XS + S6E + TMG_ALL (+0.7..0.9%, additive): replication in run 4.
Edge trace (v1502 / XS + S6E / XS + S6E + GC64, bs256): G1n1 -> G1n1 excess 1214 / 926 / 1039 cycles, warp 0's own
transition 1290 / 1180 / 1130, first warp starts +1259 / +622 / +634; total boundary excess 6.84 / 6.47 / 7.20% of the
span (bs128 6.58 / 6.15 / 6.83%). The ending item's last 3 stages run +30..110 cycles over the mid-item period (the
producer's item transition on SMSP0: small). The next item's first copies go out ~2150 cycles before warp 0's last
compute (G1 items; ~250-330 for G2, whose first stage polls its tile's counters) and the first warp sees data ~550-620
cycles after warp 0's last compute with XS: warp 0's own transition (~1180) stays the critical path.

### run 4 (dev/wpit_run4.sh, kda_ws/runs/wpit_run4.log): XS + S6E + TMG_ALL replication, TMG2, AQW=7 (verified bench2)
Smoke 3 builds 8/8 ok (bs 32-512). check_mt (M 64 65 100 128 129 200 256 257 300 512, normal + skewed, 3 seeds, 2 calls)
0 failures for XS + S6E + TMG_ALL, + AQW=7, + TMG2; wpit_cmp vs v1502 worst rel-L2 2.2e-5 (KRED noise).
% vs the middle controls v1502b / c / d (bs256 0.9953, bs128 0.9969); the last copy v1502z (excluded) +0.54 / +0.34%:

| variant (dev/moe_v2106.py) | position | bs256 p1 / p2 | % | bs128 p1 / p2 | % | bs32 % |
|---|---|---|---|---|---|---|
| XS + S6E + TMG_ALL | 2 | 1.0045 / 1.0057 | +0.98 | 1.0044 / 1.0043 | +0.75 | +0.01 |
| XS + S6E + TMG_ALL | 9 | 1.0020 / 1.0039 | +0.77 | 1.0032 / 1.0033 | +0.64 | -0.05 |
| TMG_ALL | 4 | 0.9997 / 1.0010 | +0.50 | 1.0006 / 1.0021 | +0.45 | +0.14 |
| TMG_ALL | 10 | 0.9998 / 0.9986 | +0.39 | 1.0005 / 0.9984 | +0.26 | +0.12 |
| XS + S6E | 5 | 0.9982 / 0.9982 | +0.29 | 0.9995 / 1.0012 | +0.35 | +0.16 |
| XS + S6E | 12 | 0.9973 / 0.9989 | +0.28 | 0.9991 / 0.9990 | +0.22 | +0.04 |
| XS + S6E + TMG_ALL + TMG2 | 6 | 0.9984 / 0.9969 | +0.23 | 1.0026 / 1.0030 | +0.60 | +0.13 |
| XS + S6E + TMG_ALL + AQW=7 | 8 | 1.0017 / 1.0034 | +0.73 | 1.0043 / 1.0036 | +0.71 | +0.08 |
| XS + S6E + TMG_ALL + AQW=7 | 13 | 1.0033 / 1.0066 | +0.97 | 1.0058 / 1.0071 | +0.96 | +0.19 |
Over runs 3-4 (3 copies each, middle positions, every copy above every middle control at both shapes):
- XS + S6E + TMG_ALL: bs256 +0.87 / +0.98 / +0.77% (mean +0.87%), bs128 +0.72 / +0.75 / +0.64% (mean +0.70%).
- TMG_ALL alone: bs256 +0.48 / +0.50 / +0.39% (+0.46%), bs128 +0.38 / +0.45 / +0.26% (+0.36%).
- XS + S6E alone (runs 2-4, 6 copies): bs256 mean +0.27%, bs128 +0.34%.
- AQW=7 (aq_slot by warp 7): indistinguishable from XS + S6E + TMG_ALL (+0.85 vs +0.87% bs256, +0.84 vs +0.70% bs128;
  its second copy sat one place before the excluded last copy): not banked. TMG2 (double-buffered reloads): -0.6% at
  bs256 vs TMG: rejected.
Baked: dev/moe_v2107.py (wpit_mk2107.py) = the three defines on by default, escape MOE_NO_WPIT (= v1502 SASS in all 16
kernels, local NVRTC 13); default build = the measured kernels (moe_8_8_1_3_4 7046dac34b36, moe_8_8_1_3_8 3aa241705c85;
every other kernel = v1502's). Validation in run 5.

### run 5 (dev/wpit_run5.sh, kda_ws/runs/wpit_run5.log): validation of the baked v2107, ESP follow-up
Smoke v2107 14/14 ok (bs 1 4 8 16 32 33 64 65 128 129 256 257 300 512); v2108 ESP=0 / 200 6/6 ok. check_mt on v2107 itself
(M 64 65 100 128 129 200 256 257 300 512, normal + skewed, 3 seeds, 2 calls): 0 failures; v2108 ESP=0 (M 65-257): 0
failures. wpit_cmp v2107 vs v1502 (M 65 128 200 256, normal + skewed): worst rel-L2 2.0e-5, = v1502's own run-to-run
KRED noise. % vs the middle v1502 controls (bs256 0.9956, bs128 0.9977, bs32 0.9978); last copy v1502z +0.44 / +0.43%:

| variant | position | bs256 p1 / p2 | % | bs128 p1 / p2 | % | bs32 % |
|---|---|---|---|---|---|---|
| v2107 | 2 | 1.0032 / 1.0034 | +0.78 | 1.0058 / 1.0078 | +0.91 | -0.09 |
| v2107 | 5 | 1.0000 / 1.0012 | +0.50 | 1.0043 / 1.0031 | +0.60 | -0.02 |
| v2107 | 8 | 1.0026 / 1.0040 | +0.78 | 1.0057 / 1.0059 | +0.81 | -0.01 |
| v2108 ESP=0 (polling wake) | 4 / 10 | 1.0027 / 1.0016, 1.0012 / 1.0032 | +0.66 / +0.66 | 1.0048 / 1.0048, 1.0061 / 1.0060 | +0.71 / +0.83 | -0.01 / -0.08 |
| v2108 ESP=200 | 7 / 11 | 1.0032 / 1.0041, 1.0052 / 1.0041 | +0.81 / +0.91 | 1.0041 / 1.0048, 1.0065 / 1.0065 | +0.67 / +0.88 | -0.17 / -0.08 |
XS + S6E + TMG_ALL over three jobs (runs 3-5, 6 copies, middle positions): bs256 +0.87 / +0.98 / +0.77 / +0.78 / +0.50 /
+0.78% (mean +0.78%), bs128 +0.72 / +0.75 / +0.64 / +0.91 / +0.60 / +0.81% (mean +0.74%); every copy above every middle
control; bs32 placebo (byte-identical kernel) -0.09..+0.01%.
ESP: polling 0 vs v2107; 200 ns hint +0.17% bs256 / +0.01% bs128 over the v2107 copies (one job): replication in run 6.

### run 6 (dev/wpit_run6.sh, kda_ws/runs/wpit_run6.log): ESP / AQW on top of v2107 (controls = v2107 copies)
Smoke 4 builds 6/6 ok; check_mt ESP=200 and ESP=200 + AQW=7 (M 64-257): 0 failures. % vs the middle v2107 copies b / c / d
(bs256 0.9949, bs128 0.9970); last copy v2107z +0.64 / +0.25%:

| variant (dev/moe_v2108.py on v2107) | bs256 | bs128 | bs32 |
|---|---|---|---|
| ESP=200 x3 (positions 2 / 7 / 12) | +0.17 / -0.06 / -0.02 | +0.23 / +0.05 / +0.00 | +0.03 / -0.09 / -0.00 |
| ESP=100 / ESP=500 | -0.07 / -0.16 | +0.03 / +0.07 | +0.06 / +0.11 |
| AQW=7 x2 | -0.05 / +0.05 | -0.04 / +0.17 | -0.05 / +0.02 |
| ESP=200 + AQW=7 | -0.01 | +0.06 | -0.02 |
| v1502 x2 (middle positions 9 / 14) | -0.73 / -0.52 | -0.65 / -0.60 | -0.13 / +0.06 |
=> ESP and AQW: no gain (rejected). Fourth job for v2107 vs v1502 (middle vs middle): bs256 +0.63%, bs128 +0.63%.

## Summary (WP-IT)
Where the (8,8,1) item boundary goes (v1502, per-warp trace, bs256 / bs128): 6.8% / 6.6-6.8% of the CTA span is boundary
excess (slowest-warp periods of the next item's first 3 stages over 3 mid-item periods). Warp 0 (SMSP0, beside the
producer) is the last to finish the last stage in 95-99% of the 1-group boundaries; its own item end + item top (~1300
cycles in the trace build) is the critical path, the item-end bar.sync only makes the other warps start with it. Items
of > 32 tokens add 2-5k cycles each through their held slot (the next item's stage 2).
Bounds (timing only, every stage loop byte-identical; prize vs the middle controls, bs256 / bs128): barrier +0.3 /
+0.35%; barrier + staging stores + fence +1.2 / +1.4% (mostly the stores; half the stores 0); held slot +1.2 / +0.8%
(gate/up +0.8 / +0.6, down +0.4 / +0.2); everything +2.4 / +1.9%; epilogue warps' work (be - bh) +0.75 / +0.3%.
Bench caveat found here: bench2's first AND last list positions measure 0.3-0.8% fast (order reversal replays them twice
in a row with warm instruction caches); compare in the middle of the list.

BANKED: dev/moe_v2107.py (default build; = dev/moe_v2106.py with MOE_IT_XS,MOE_IT_S6E,MOE_WPA_TMG_ALL; escape
MOE_NO_WPIT = v1502 SASS in all 16 kernels). Only moe_8_8_1_3_4 (bs65-128) and moe_8_8_1_3_8 (bs129-256) change; all 8
consumer stage loops of each byte-identical to v1502's (dev/wpl2_loops.py + an exact / opcode / renamed loop comparison);
same instruction-count class (10872 / 10904 vs 10840 / 10872), same spills (16 B / 44 B).
- MOE_IT_XS: items of <= 16 tokens stage into their held last slot's activation area past row 16 (no warp reads it in
  that stage) and skip the item-end bar.sync; the epilogue warps add the offset (from f.n).
- MOE_IT_S6E: MOE_S6O's x64 moved from the consumers' item end into the epilogue's expert scale (power of two: bitwise
  the same products); with XS it keeps the stage loops identical (alone it does not).
- MOE_WPA_TMG_ALL: WP-A's existing global-scratch spill of the held slot's tokens 12.. (v984 had it on the then (8,4,1)
  bs65-128 kernel only; since v1033 bs65-128 runs on (8,8,1) without it) on the (8,8,1) MW 4 / 8 kernels.
Measured (bench2 jm 100 reps x 2 passes, verified, middle positions, % vs middle v1502 controls), 4 jobs:
bs256 +0.87 / +0.98 / +0.77 / +0.78 / +0.50 / +0.78 / +0.63% (mean +0.76%), bs128 +0.72 / +0.75 / +0.64 / +0.91 / +0.60 /
+0.81 / +0.63% (mean +0.72%); parts: TMG_ALL alone +0.46 / +0.36% (3 copies), XS + S6E alone +0.27 / +0.34% (6 copies),
XS alone +0.2 / +0.15%. Projected geomean ~+0.25% (bs128 + bs256 of six loads).
Correctness: hang1 bs 1 4 8 16 32 33 64 65 128 129 256 257 300 512 ok; check_mt on v2107 (M 64 65 100 128 129 200 256 257
300 512, normal + skewed, 3 seeds, 2 calls) 0 failures; outputs equal v1502's up to v1502's own KRED run-to-run noise
(worst rel-L2 2.0e-5 between the builds).
Rejected / not run: GC (generic room copy to drop the consumer fence) -0.1..-0.8%; TMG2 -0.6% vs TMG at bs256; ESP
(faster epilogue wake) 0; AQW (aq_slot off warp 0) 0; DB / consumer-side BF16 staging (stmatrix + bulk-copy epilogue)
built but every consumer formulation (9 tried) re-schedules all 8 stage loops: not run.
For the planner: (1) the remaining item-end prize (~+0.9%) is in the consumers' staging stores of > 16-token items and
small items' stores; the only formulation that would cut them (BF16 / stmatrix staging) needs a stage-loop schedule
search. (2) The held slot of > 32-token gate/up items still costs ~0.3-0.4% after TMG. (3) Cold instruction fetch
(warm-i-cache list positions +0.3-0.8%) is a separate lever: the judge alternates SGLang's kernel with ours, so our
~170 KB (8,8,1) kernel starts every replay with cold SM instruction caches.
