# WP-DS: source-level arrangements of the (8,*,1) consumer stage loop, 2026-10-27

Worker WP-DS. Base dev/moe_v969.py. Files: dev/wpds_occ.py (harness), dev/wpds_tab.py (result table), dev/wpds_loops.py /
dev/wpds_kloops.py (SASS loop fingerprints: exact / register-renamed / opcode-order hashes), dev/moe_v1000.py (first carrier,
superseded), dev/moe_v1001.py (real-kernel carrier, MOE_DS* codes), dev/wpds_run1-4.sh. Logs: scratchpad wpds/run*.log.

## 1. Harness (dev/wpds_occ.py)

v969's (8,*,1) stage body (KW 1: 8 consumer warps, warp w = row groups 2w, 2w+1 x 256 k, S6O scales, XP 528) in a TMA ring
on the H200: 132 CTAs x 256 stages, NS 4 (NT 4, 53 KB stages) / NS 3 (NT 8, 68 KB). The consumer loop has the kernel's
MOE_RUN_ITEM_TOP shape (release + wait at the top, bar.arrive release on named barrier 4 + slot, 288 threads, as the
MW >= 2 kernels' NBC; the producer bar.syncs before reusing a slot). Clock stamps are outside the loop: an in-loop stamp
changes the schedule. Modes: sync = ring handshake without copies (compute only, same consumer SASS as load); load = one
35840 B weight + scale bulk copy (HBM) + one n x 528 B activation copy (L2) per stage; loadg = load on 88 CTAs (streaming,
compute-bound grid); free = no waits (other SASS: ptxas schedules the free loop differently, so it is informational only).
Variant = stage_ds<ORD, DORD, SCL, WLD, XLD, MO, SPL, DEF> (knobs: header of dev/wpds_occ.py); "k0" (all knobs 0)
compiles to v969's loop byte for byte in the harness AND in all 16 kernels (v1001 code 100000000). Every variant is checked
on real data against v969's stage_compute (WPDS_CHECK=1): bit-exact except SPL (rel 7e-7, a different FP32 sum order).
Precision: identical code (v969, k0, v969b) gives the same cycles per stage to 0.01% in sync / loadg / free (wall +-0.3%).

v969 per 32 KB stage (warp 0, median CTA): NTL1 sync 1572 cyc, loadg 1598-1608, load 1705-1716 (1.67-1.70 GHz, at the
ring floor ~1100-1125 ns / stage), free 1513; NTL2 sync 1851-1890, loadg 1899-1905, load 1891-1911, free 1768.

## 2. Sweep 1: 72 order-only variants (wpds_run1)

Word order / interleave (ORD 1-9), the order inside a word's decode (DORD 1-5), scale-pair placement (SCL), weight / x LDS
placement (WLD / XLD), MMA order (MO), SPLITACC (SPL). 57 distinct NTL1 loops, 43 NTL2 (all 502-504 SASS instructions in
free, 519 in sync: instruction counts never change, only order and registers).

| | best | worst | knob means |
|---|---|---|---|
| NTL1 sync | +2.64% (6500000 = ORD 6 lag-2 words, DORD 5 HMUL2s reversed; 9500000, 7500000 same) | -3.6% | DORD5 +1.4%, DORD1 -1.2%, rest +-0.5% |
| NTL1 loadg | +2.86% (2500000) | -3.2% | |
| NTL2 sync | +2.15% (0000001 = SPL), +1.6% (2500000) | -3.9% | v969's own NTL2 order is near the top |
| NTL2 loadg | +3.6% (SPL) | -1.8% | |

Reading: order-only changes give a +-3% spread that is mostly schedule luck. The same source lands at only 0.2-0.5
opcode-order similarity between harness and kernel loops, and moe_8_4_1_4_4 vs moe_8_8_1_3_8 (same source) are 0.37 apart,
so an order winner in the harness says little about the kernel.

## 3. Sweep 2: structural levers (wpds_run2, 68 variants x NT 4 / 8 x NTL 1 / 2)

DEF (new): cross-stage pipelining. The last 64-k chunk's MMAs (DEF 1; DEF 2: only its jj 2-3) are not issued in the stage:
their A fragments (16 regs per row group) and x fragments (8 x NTL regs) stay in registers across the slot release and the
next full-barrier wait and are issued at the start of the next stage's block (right after its first LDS). An item's first
stage issues them on zero fragments (acc += 0 exactly), the item's loop is followed by one flush. Bit-identical math (each
accumulator still receives its MMAs in k order). Why it pays: the wait loop is a basic-block boundary ptxas never schedules
across, so every v969 stage ends with the last chunk's 8 x NTL dependent HMMAs and nothing to overlap, and starts with the
weight-LDS latency; with DEF the block starts with HMMAs (SASS positions 15-43 after the wait) and ends in decode.

Cycle ratio vs v969 (> 1 = faster), same job:

| variant | NT4 NTL1 sync / loadg / load | NT8 NTL1 | NT4 NTL2 sync / loadg / load | NT8 NTL2 |
|---|---|---|---|---|
| 00000001 (DEF 1) | 1.046 / 1.059 / 1.002 | 1.043 / 1.056 / 1.006 | 1.044 / 1.039 / 1.039 | 1.044 / 1.038 / 1.037 |
| 00000002 (DEF 2) | 1.029 / 1.046 / 1.005 | 1.056 / 1.062 / 1.005 | 1.034 / 1.023 / 1.023 | 1.034 / 1.025 / 1.025 |
| 00000011 (SPL + DEF 1) | **1.068 / 1.078** / 1.005 | **1.068 / 1.076** / 1.002 | 1.032 / 1.027 / 1.028 | **1.054 / 1.054 / 1.052** |
| 00000012 (SPL + DEF 2) | 1.026 / 1.026 / 1.003 | 1.042 / 1.050 / 1.002 | **1.071 / 1.061 / 1.057** | 1.031 / 1.010 / 1.017 |
| 25000012 | 1.030 / 1.050 / 1.004 | 1.028 / 1.034 / 1.005 | 1.068 / 1.054 / 1.053 | 1.048 / 1.049 / 1.048 |
| 2000001 (ORD 2 + SPL) | 1.037 / 1.030 / 1.003 | 1.037 / 1.030 / 1.001 | 1.058 / 1.046 / 1.046 | 1.055 / 1.048 / 1.047 |
| v969b (control) | 1.000 / 1.000 / 1.001 | 1.000 / 1.000 / 0.997 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |

NTL1 full-grid load stays at ~1.00 for every variant: at 132 CTAs the 1-group stage is at the TMA ring floor (as WP-16W found
for (4,*,1)); loadg (88 CTAs) is the streaming mode where compute matters. NTL2 is compute-coupled even at 132 CTAs.

## 4. Real kernel (dev/moe_v1001.py)

v1001 = v969 + the harness's stage_ds verbatim behind per-kernel codes (MOE_DS41 / 42 / 81 / 82 for the MW >= 2 (8,4,1) /
(8,8,1) kernels' 1- / 2-group tiles, MOE_DS41M1 / 42M1 for the MW 1 bs17-32 kernel; 9-digit 1<ORD..DEF>). DEF's carried
fragments live in MOE_RUN_ITEM_TOP (zeroed per item, flushed after the item's loop); SPL's acc2 is folded into acc in the
same flush (the global MOE_SPLITACC stays off, so no other kernel changes). Default and code 100000000: SASS of all 16
kernels = v969. Candidates: only the targeted (8,*,1) kernels change, no C75xx, spills = v969's (20-24 B per-item spills of
the MW >= 2 kernels, 0 in MW 1). In-kernel loops have the harness shape (carried HMMAs right after the wait, blocks end in
decode). hang1 ok; check_mt (M 1..257, 16 sizes, normal + skewed, 3 seeds, 2 calls) 0 failures for c1, c2, c3.

bench2 jm100 vs v969 (run 3 / run 7, same-job controls):

| | bs32 (placebo) | bs128 | bs256 |
|---|---|---|---|
| v969b control | 0.9981 / 0.9968 | 0.9995 / 0.9988 | 0.9965 / 0.9987 |
| c1 DEF1 all | 0.9976 | 1.0008 | 0.9960 |
| c2 SPL+DEF1 all | 0.9965 / 0.9996 | 0.9961 / 0.9968 | 0.9971 / 0.9987 |
| c3 ntl1 SPL+DEF1, ntl2 25000012 | 0.9991 | 0.9986 | 0.9986 |
| c4 ntl1 05000011, ntl2 2000001 | 0.9986 | 0.9992 | 0.9981 |
| c5 ntl1 DEF2, ntl2 SPL+DEF2 | 0.9977 / 0.9985 | 1.0016 / 1.0014 | 1.0032 / 1.0038 |
| MOE_NO_COMPUTE (data-only) | 1.0830 | 1.0923 | 1.1695 |

The +5..8% per-stage harness gains do not reach the wall clock: c5 is the only one above its controls, +0.2% bs128 and
+0.4..0.5% bs256 (replicated), below the 1% the brief asked for.

## 5. Why: the 1-group stages of bs128 / bs256 are paced by the ring, not by the loop

In-kernel per-stage cycles (warp 0, MOE_TIMING + MOE_TIMN build, dev/timing_ntl.py, 4 seeds; compute + wait per stage):

| | v969 | c2 | c5 | c1 |
|---|---|---|---|---|
| bs128 G1 ntl1 (512 items / call) | 1459 + 91 | 1463 + 87 | 1465 + 85 | 1500 + 82 |
| bs128 G1 ntl2 (96) | 1911 + 173 | 1852 + 193 | 1798 + 180 | 1817 + 177 |
| bs128 G2 ntl2 (48) | 1861 + 79 | 1790 + 83 | 1737 + 83 | 1760 + 77 |
| bs256 G1 ntl1 (392) | 1497 + 94 | 1457 + 97 | 1477 + 91 | 1443 + 93 |
| bs256 G1 ntl2 (192) | 1799 + 149 | 1724 + 153 | 1730 + 153 | 1783 + 144 |
| bs256 G2 ntl2 (96) | 1834 + 211 | 1736 + 210 | 1744 + 222 | 1790 + 213 |

The kernel's 2-group stages get 3..7% faster (as the harness predicts), the bulk 1-group stages of bs128 not at all and of
bs256 by 1.5..3.5%. Harness at rising streaming pressure (wpds_run8, load mode, 88..132 CTAs, wall ratio vs v969):

| CTAs streaming | 88 | 100 | 110 | 120 | 126 | 132 |
|---|---|---|---|---|---|---|
| NT4 NTL1 00000011 | 1.062 | 1.058 | 1.048 | 1.000 | 0.996 | 1.005 |
| NT8 NTL1 00000011 | 1.059 | 1.060 | 1.053 | 1.025 | 1.008 | 0.995 |
| NT4 NTL2 25000012 | 1.045 | 1.044 | 1.039 | 1.039 | 1.041 | 1.035 |
| NT8 NTL2 00000011 | 1.045 | 1.048 | 1.041 | 1.039 | 1.037 | 1.032 |

From ~120 streaming CTAs on, a 1-group stage runs at the TMA ring's pace (~1085-1117 ns per 35 KB stage, 4.2-4.3 TB/s)
whatever its compute: the ring floor WP-16W found for (4,*,1). bs128 is that regime (651 MB in 166 us, ~4.3 TB/s between
startup and tail; 1-group items are ~60% of its stage time), so its 9% gap to the data-only floor is not in the
1-group loop. Only multi-group stages are compute-coupled under full streaming. 2-group mma.sync stages are 15% of bs128's
stage cycles and 24% of bs256's, so the 3..7% stage gain is worth ~0.3-0.5% of wall, as measured.

## 6. The bs32 kernel (MW 1, mbarrier ring) and replication (wpds_run9 / run11, jm100, v969 x2)

Harness with the mbarrier release (NB 0, the MW 1 kernel's handshake): NTL2 00000012 +7.3..7.8% in sync / loadg / load, NTL1
SPL+DEF +4..5% sync. Kernel (A = c5's MW >= 2 codes + MOE_DS42M1=100000012; E = 2-group codes only; F = E with DS82
100000011; B / C / D = other MW 1 codes):

| | bs32 | bs128 | bs256 |
|---|---|---|---|
| v969b control (run 9 / 11) | 1.0000 / 0.9989 | 1.0007 / 0.9984 | 0.9967 / 0.9962 |
| A | 1.0028 / 1.0024 | 1.0012 / 1.0014 | 0.9992 / 1.0027 |
| B (DS42M1 125000012) | 1.0027 | 1.0002 | 0.9975 |
| C (+ DS41M1 SPL+DEF1) | 0.9916 | 1.0029 | 0.9974 |
| D (DS42M1 120000011) | 0.9901 | 1.0014 | 0.9988 |
| E (2-group only) | 1.0027 | 0.9912 | 0.9978 |
| F | 1.0022 | 0.9941 | 0.9959 |

Placebos (run 11, kernels byte-identical): bs1 A 0.9974 (control 0.9852), bs8 0.9988 (1.0012), bs16 1.0002 (1.0009).
- The MW 1 kernel's 2-group code gives bs32 +0.25% (A, B, E, F alike); its 1-group DEF costs ~1% (C).
- E / F (without the 1-group code in the MW >= 2 kernels) lose 0.6..0.9% at bs128 although those stages are ring-paced:
  the whole-kernel schedule moves with any change (+-0.5..1%), the same size as the effect being measured.
- Correctness: check_mt (17 sizes 1..257, normal + skewed, 3 seeds, 2 calls) 0 failures for c1, c2, c3, A, C, E, F; hang1
  ok at 1..257; rel-L2 = v969's (3.0-3.25e-3).

## 7. Conclusion

- Best harness variant: cross-stage pipelining + split accumulators (SPL+DEF). Order-only re-arrangements are schedule luck
  (+-3%), as the brief expected. SPL+DEF is structural: the stage-boundary wait is a basic-block edge that ptxas never
  schedules across. The harness shows +6.8% (NTL1, compute-only), +7.6% (NTL1 streaming on 88 CTAs) and +3..7.8% (NTL2,
  every mode including full-grid streaming).
- In the kernel the 2-group stages speed up by 3..7% (timing build), but bs128 / bs256 do not follow. Their 1-group stages
  (bs128: ~60% of stage time) run at the TMA ring's pace (~4.3 TB/s): from ~120 streaming CTAs on, a 1-group stage costs
  the same whatever its compute, as WP-16W found for (4,*,1). The 9% (bs128) / 17% (bs256) gap to the data-only floor is
  therefore not in the 1-group mma.sync loop. It is in the multi-group stages (wgmma tiles: 36% of bs256's stage cycles),
  the item edges (epilogue 0.28-0.6 us per item) and startup / tail.
- Kept as an option: dev/moe_v1002.py (baked A, escape MOE_NO_WPDS). Measured: bs32 +0.25%, bs128 +0.15%, bs256 ~+0.3%
  (+0.5% vs the control copies), about +0.1% geomean / ~5 us. Positive in all 4 runs but within ~2 control spreads.
  Integrate only together with other changes to the (8,*,1) kernels, and re-measure there (the whole-kernel schedule shifts
  with any edit).
- Not pursued: the wgmma stage (WP-WG: decode + tensor work additive; its stage-boundary hoist WGSN is already in v969 for
  MW 4).
