# WP-A: early release of the asynchronous epilogue's held ring slot (2026-10-27)

Worker WP-A. Lever (WP-W wpw.md section 4 / WP-Q wpq.md #2): in the one-k-warp configs (8, 4, 1) (bs 17-128) and
(8, 8, 1) (bs 129-512) an item's last ring slot holds the consumers' staged FP32 accumulators until the two
asynchronous-epilogue warps 10-11 have finished SwiGLU + stores; the next item's stage NS - 1 needs that slot.
AREL0 (timing-only, slot freed before it is read) bounds the prize at +1.35% bs32, +1.7% bs128, +2.4% bs256.
Base: dev/moe_v730.py (planner: v695 + FSW baked). Controls v730 x2 in every A/B. Builds are made by patch scripts
(exact-string replacements on the parent file): dev/wpa_mk710.py .. dev/wpa_mk714.py. Every default build (no
define) is SASS-identical to v730 (dev/wpc_sass.py hashes); with a define only the (8, 4, 1) / (8, 8, 1) MW <= 8
kernels change.

Files: dev/moe_v710.py (CPO), v711 (CPX, RT), v712 (F32C, CPO batched, ET stamps, WPW_TRACE port), v713 (TMC),
v714 (TMC + 1-CTA cluster attribute); dev/wpa_et.py (epilogue-item timeline), dev/wpa_spill.py (where LDL/STL sit),
dev/wpa_rec.py (bench2 log -> benchmark.csv), dev/wpa_run1..N.sh; logs kda_ws/runs/wpa_runN.log.

## Room past the ring

| config | MW | smem used by v730 | free (of 232448) | BF16 gate/up tokens (512 B) | FP32 staged tokens (1040 B) |
|---|---|---|---|---|---|
| (8, 4, 1) | 1 (bs32) | 216064 | 16384 | 32 (all) | 15 |
| (8, 4, 1) | 2 | 217088 | 15360 | 30 | 14 |
| (8, 4, 1) | 4 (bs128) | 219136 | 13312 | 26 | 12 |
| (8, 8, 1) | 8 (bs256) | 219136 | 13312 | 26 (of 64) | 12 |
| (8, 8, 1) | 16 | 228352 | 4096 | - | - |

The route tables stay live to the combine (lid / bm / wpre), so that area cannot be reused. Tile sizes on the jm
routings (30 seeds): bs128 735 tiles, 514 with 1 group, 156 with 9-24 tokens, 65 with > 24 (53 of them exactly 32);
bs256 775 tiles, 366 / 283 / 126 (54 of 64 tokens).

## Run 1: v710 MOE_WPA_CPO (BF16 copy-out of multi-group gate/up items) -- rejected

Warps 10-11 convert the first 32 (MW 1) / 24 (MW 2-8) tokens' staged FP32 gate/up to BF16 (XOR-swizzled rows so the
wgmma-layout SwiGLU loop's 4 tokens per warp hit disjoint banks), SwiGLU the remaining tokens straight from the slot,
release, then SwiGLU + store from the copy. check_mt 15 sizes 0 failures (max rel-L2 3.95e-3 vs 3.30e-3: the BF16
gate/up rounding), spill-free.

| bench2 jm 100 reps vs v730 | bs1 | bs8 | bs16 | bs32 | bs128 | bs256 |
|---|---|---|---|---|---|---|
| v730 copy (control) | 0.9922 | 1.0049 | 0.9992 | 0.9985 | 0.9993 | 0.9962 |
| CPO=2 | 0.9960 | 1.0025 | 0.9992 | 1.0018 | 1.0002 | 0.9971 |
| CPO=3 | 0.9933 | 0.9982 | 0.9996 | 1.0014 | 1.0021 | 0.9956 |
| CPO=1 (1-group items too) | 1.0006 | 1.0029 | 1.0015 | 1.0017 | 1.0039 | 1.0042 |

Held-slot turnaround of multi-group gate/up items (wpw_trace, median cycles): bs128 8598 -> 6574 (seed 71637),
6075 -> 4159 (71433); bs256 7142 -> 6584 (73011), 4678 -> 2830 (71497). Two reasons it does not pay:
1. The copy is not cheap on these warps: ~150 cycles per token against ~270 for the SwiGLU itself (inferred from
   the turnarounds), although it is ~1/4 of the instructions: the epilogue warps are latency / issue starved
   (~12 cycles per instruction in the SwiGLU loop, ~25 in the copy loop).
2. Multi-group gate/up items are a minority of the held slots. Held-slot events per call at bs128 (seed 71637):
   G1 1-group 352 (2.4k cycles each), G2 1-group 608 (2.4k), G1 multi 124 (8.6k), G2 multi 128 (2.1k); bs256:
   288 / 384 / 220 (7.1k) / 176. Multi-group G1 is ~30% (bs128) / ~45% (bs256) of the held cycles, and CPO removed
   ~25% of those.
Batched copy loops (4 tokens or 6 vectors in flight) and an 8-token register tail (v711 RT) push the 72-register
producer warpgroup into spills (dev/wpa_spill.py: in the epilogue and, for (8, 8, 1), the producer setup).

## Run 2: v713 MOE_WPA_TMC (one bulk smem -> smem copy) -- illegal instruction

UBLKCP.S.S (cp.async.bulk.shared::cluster.shared::cta) traps with "Illegal Instruction Encoding" (Xid 13) in the
non-cluster launch at bs32. v714 compiles the RW = 8 kernels as 1-CTA cluster kernels (__cluster_dims__(1, 1, 1),
EIATTR_CTA_PER_CLUSTER 1 1 1; SASS unchanged).

## Run 3: v714 MOE_WPA_TMC (bulk smem -> smem copy, 1-CTA cluster kernels) -- no gain

Outputs bitwise unchanged (check_mt 15 sizes 0 failures, max rel-L2 3.30e-3). bench2 (v730 x2): TMC x2 bs32
0.9987 / 1.0000, bs128 0.9998 / 1.0015, bs256 1.0001 / 1.0030; the cluster attribute alone (MOE_WPA_CLU) 1.0019 /
1.0001 / 1.0012 (neutral). Epilogue timeline (dev/wpa_et.py, bs128 seed 71637, median cycles after the consumers post
the item): wake 270-400 everywhere; G1 1-group hold 1531 -> 1084 (the 8.3 KB bulk copy itself ~550 cycles = ~16 B per
cycle), G2 1-group 1006 -> 982 (its epilogue is as short as the copy). Multi-group items untouched (n > 12).

## Run 4: the early-release prize per item category on v730 (v716 MOE_WPA_AREL0 m, timing only)

| AREL0 mask (slot freed at wake, outputs wrong) | bs32 | bs128 | bs256 |
|---|---|---|---|
| control v730 copy | 0.9978 | 0.9980 | 0.9955 |
| 15 all items | 1.0094 | 1.0101 | 1.0235 |
| 5 1-group items (G1 + G2) | 1.0032 | 0.9963 | 0.9932 |
| 2 multi-group gate/up | 0.9997 | 1.0058 | 1.0116 |
| 10 multi-group gate/up + down | 1.0006 | 1.0094 | 1.0207 |

The prize sits in the multi-group items (gate/up AND down); releasing 1-group items early is worth nothing (or
negative). Held cycles of multi-group items (ET, bs128): gate/up 32-token items 7.0-7.5k, down 32-token 4.3-4.8k;
bs256: gate/up 2.5k (9-15 tokens) .. 8k (48+), down 1.9k .. 6.5k.

v716 MOE_WPA_TMC + MOE_WPA_TMP (every item: first min(n, 12) tokens by the bulk copy while warps 10-11 do the rest
from the slot; then wait, release, the copied tokens): bs32 1.0039 / 1.0057, bs128 1.0044 / 1.0046, bs256 1.0010 /
1.0016. Hold (ET bs256): gate/up multi 3237 -> 1985, down multi 2664 -> 1966; bs128 gate/up multi 6748 -> 5944, down
multi 4684 -> 6000 (the range down loop is slower than the hoisted full-k loop).

## Run 5: consumer-side staging past the ring (v717 MOE_WPA_CSO 2) -- large loss; TMP multi-only -- loss

CSO (multi-group items with <= 12-15 tokens: consumers stage FP32 into the room, release the slot like any stage,
queue bound by per-entry mbarriers): bs32 -3.7%, bs128 -4.7%, bs256 -4.7% (both copies), the other kernels
SASS-identical. As BSTG (v668): per-item work or waits on the consumers (they pace the ring) cost far more than the
hold they save. (First build overflowed the room: whole token groups were staged; fixed, check_mt 0 failures.)
TMPM=2 (TMP for multi-group items only) loses at bs256 (-1.3%): its runtime branch keeps the TMC and default paths
live (+400 instructions in the (8, *, 1) kernels). v716 TMP repeated: bs32 1.0051, bs128 1.0033, bs256 1.0032.
v718 (BF16 consumer-side staging of gate/up items, MOE_WPA_CSB) was built (spill-free, other kernels identical) but
not run: it shares CSO's consumer-side protocol.

Spill rule: every variant was checked with ptxas -v and dev/wpa_spill.py; batched warp copies (v711 RT, v712
F32B 4-6 / CPB 4) spill in the 72-register producer warpgroup and were not run; TMC / TMP / CSO / CSB are spill-free.

## Run 6: v716 TMC + TMP confirmation (x3 vs v730 x2)

| vs v730 | bs1 | bs8 | bs16 | bs32 | bs128 | bs256 |
|---|---|---|---|---|---|---|
| v730 copy | 0.9995 | 1.0021 | 1.0013 | 0.9975 | 0.9988 | 0.9967 |
| TMP x3 | 1.0026 / 1.0032 / 0.9856 | 1.0027 / 0.9988 / 0.9967 | 1.0018 / 1.0006 / 0.9998 | 1.0014 / 1.0020 / 1.0036 | 1.0010 / 1.0018 / 1.0044 | 1.0005 / 1.0000 / 1.0048 |

(bs1 / 8 / 16 kernels are SASS-identical: their spread is noise.) Over runs 4-6 (7 copies): bs32 +0.36%, bs128 +0.33%,
bs256 +0.19% vs c, i.e. ~+0.45 / +0.40 / +0.30% vs the mean of the two controls. Held-slot turnaround (median
cycles, base -> TMP): bs32 G1 1-group 3488 -> 2480; bs128 G1 multi 8670 -> 7150 and 6934 -> 4953, G1 1-group
2426 -> 1556, G2 multi 2119 -> 1904 and 1776 -> 1402; bs256 G1 multi 7309 -> 6090 and 4702 -> 3196, G2 multi
1694 -> 1616 and 1790 -> 1646.

## Conclusions

- On v730 the early-release prize is +0.9% bs32, +1.0% bs128, +2.35% bs256 (AREL0), all of it in the multi-group
  items. The 13 KB past the ring holds 12 FP32 or 24-26 BF16 tokens; most of the multi-group hold is in 32- and
  64-token items, which no copy past the ring can absorb.
- Work on warps 10-11 is slow (latency and issue starved: ~12 cycles per instruction in the SwiGLU loop, ~25 in a
  copy loop); the bulk copy engine (smem -> smem, 1-CTA cluster) moves ~16 B per cycle with no warp cost and is the
  only copy that paid. Consumer-side staging (CSO, and BSTG before it) costs 4-5%: the consumers pace the ring.
- Best: v716 MOE_WPA_TMC,MOE_WPA_TMP, ~+0.3..0.45% at bs32 / 128 / 256, bitwise-identical outputs. Small, close to
  the control spread; worth carrying only together with other (8, *, 1) changes.

## Run 7: port onto v944, baked as dev/moe_v719.py (MOE_WPA_TMP on, escape MOE_NO_WPA)

dev/wpa_mk719.py applies only the TMC + TMP pieces to v944 (slot release through v944's NBAR named barriers; the two
epilogue warps meet on named barrier 15; the down range loop with the full-k loop hoisted as EHOIST, which removed the
extra spill in (8, 8, 1)). MOE_NO_WPA compiles to v944's SASS (all hashes); with it on, only the (8, 4, 1) MW 1-4 and
(8, 8, 1) MW 8 kernels change, their consumer and producer stage loops identical (or same modulo registers), no C75xx,
spills at v944's level.

| vs v944 (jm 100 reps) | bs8 (placebo) | bs32 | bs128 | bs256 |
|---|---|---|---|---|
| v944 copy | 1.0019 | 0.9996 | 0.9981 | 0.9948 |
| v719 x2 | 1.0007 / 1.0022 | 1.0049 / 1.0046 | 1.0040 / 1.0048 | 1.0072 / 1.0104 |

hang1 13 shapes ok, check_mt 15 sizes 0 failures (max rel-L2 3.30e-3), dev/wpa_mix.py 100 interleaved calls 0 failures.

## Runs 8-10: the remaining hold on v719, and spilling the tokens past the room to global scratch (v980-v982)

Remaining hold on v719 (dev/wpa_et.py, median cycles from the consumers' post to the slot release; run 9/10):

| | bs128 32-token | bs256 24-31 | bs256 32-39 | bs256 48-63 |
|---|---|---|---|---|
| gate/up | 6624-6664 | 5799 | 5578-5607 | 6756-6774 |
| down | 4336-4340 | 4024 | 5424-5465 | 6246-6270 |

v980 MOE_WPA_TMG (bs65-512 kernels only): tokens 12 .. n-1 leave the held slot by bulk STORES
(cp.async.bulk.global.shared::cta) into a per-CTA global scratch (67 KB per CTA, passed in the unused MOE_F16_NT4
pointer P.w2h; allocated once in prepare), warp 10 lane 0 waits for the stores' source reads, the slot is released;
after the 12 room tokens the rest come back 12 rows at a time by bulk loads into the room. Hold (v719 -> TMG):
bs128 gate/up 32-token 6664 -> 2692, down 4340 -> 2256; bs256 gate/up 48+ 6774 -> 2848, down 48+ 6270 -> 2842. But
the part after the release grows from 1-9k to 18-23k cycles: each scratch reload waits behind the SM's weight
stream. v981 TMG2 (double-buffered halves of the room) does not shorten it (worse at bs256); v982 TMGS (the reload
waits suspend instead of polling) is the best of the three.

| vs v719 (jm 100 reps) | bs8 placebo | bs128 | bs256 |
|---|---|---|---|
| run 8: v719 copy / TMG x2 | 1.0015 / 0.9989, 0.9990 | 1.0015 / 1.0046, 1.0065 | 0.9983 / 1.0014, 1.0037 |
| run 9: v719 copy / TMG / TMG2 x2 | 0.9994 / 0.9964 / 0.9945, 0.9988 | 0.9993 / 1.0027 / 1.0026, 1.0041 | 0.9977 / 0.9995 / 0.9948, 0.9965 |
| run 10: v719 copy / TMG / TMGS x2 / TMG2S | 0.9994 / 0.9999 / 0.9978, 1.0015 / 1.0010 | 0.9978 / 1.0028 / 1.0043, 1.0042 / 1.0043 | 0.9962 / 0.9997 / 1.0009, 1.0023 / 0.9971 |

All bitwise identical (check_mt 15 sizes, 100 interleaved calls: 0 failures), stage loops identical to v719, spills
at v719's level, bs1-32 and MW 2 / 16 kernels identical.

## Runs 11-12: spill threshold (rejected) and the baked bs128 part (dev/moe_v984.py)

v983 MOE_WPA_TMGN 24 / 32 (spill only the larger items): bs128 1.0008 / 1.0007, bs256 0.9982 / 0.9980 vs TMGS 1.0036 /
1.0043, 1.0001 / 0.9986 in the same job (controls 0.9982 / 0.9978): both paths live costs more than it saves.
bs256 never showed a clear gain from the global spill (+0.09 / +0.23 / +0.01 / -0.14% over four copies, controls
-0.38 / -0.22%): its 48-64-token items wait 18-23k cycles for their scratch reloads after the release.

dev/moe_v984.py = v719 + TMG + TMGS on by default for moe_8_4_1_4_4 only (escape MOE_NO_WPA2 = v719's SASS; only that
kernel differs, byte-identical to the measured v982 TMGS kernel). wpa_run12 (v719 x2): bs128 1.0051 / 1.0070 (control
1.0000), placebos bs8 1.0010 / 0.9990, bs256 0.9971 / 0.9982 (controls 0.9982 / 0.9973). Over six copies of that
kernel: bs128 +0.48% vs c. check_mt 15 sizes and 100 interleaved calls: 0 failures, bitwise the same outputs.

## Runs 13-14 (base v969): generic-path scratch reload, bs32 KRED pieces

v985 MOE_WPA_TMGC (scratch rows back by cp.async.cg through the LSU path, double-buffered halves; with TMG_ALL also
bs256): slower than the bulk loads. Time after the release: bs128 32-token gate/up 20.4k -> 34.6k cycles, bs256 48+
23.7k -> 32.2k. bench vs v969 x2: bs128 0.9942 / 0.9940, bs256 0.9851 / 0.9861 (controls 0.9983 / 0.9980);
TMG_ALL with bulk loads at bs256 0.9990 (still ~0).

bs32 KRED down pieces on v969 (wpa_et, median cycles): TMP already copies them whole (F32T 15 at MW 1); hold 925
(1-group: wake 281 + bulk copy ~500) and 1852 (9-15 tokens). The rest of their per-item cost is the consumers' own
staging + barrier, which no epilogue-side release removes. v986 MOE_WPA_KDIR (consumers add the piece's rows from
registers, 32 scalar red.add per thread, no staging or barrier; the epilogue only releases and signals): piece hold
925 -> 642, but bs32 0.9780 / 0.9799, bs128 (KDIR 4) 0.9890 vs controls 0.9982 / 0.9985: consumer-side work loses
again (CSO, BSTG, KDIR).
