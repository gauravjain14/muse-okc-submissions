# WP-X findings (activation shared-memory traffic at bs128 / bs256, BF16 only)

Worker X, started 2026-10-27. Base: dev/moe_v475.py (MOE_S6O,MOE_S6X) == dev/moe_v365.py for bs17-512. Files:
dev/moe_v540.py (timing ablations), dev/moe_v541.py (MOE_XR), dev/wpx_*.

## Quantification (offline, dev/wpx_bytes.py, 100 judge-matched routings per shape)
Per stage of a 256-row x 256-k item (32 KB words + 3 KB 6-bit scales): copies L2 -> SMEM = wgmma tiles (ntl >= 3)
4 x 8 ntl x 128 B, mma.sync tiles n x 528 B; B reads from SMEM = mma.sync 8 warps x ntl x 4 KB (LDS), wgmma
4 m64 x ntl x 4 KB (tensor core).

| shape | weights+scales | x copies | per weight byte | B reads | copy share by ntl |
|---|---|---|---|---|---|
| bs32 | 453 MB | 24.9 MB | 0.055 | 441 MB | 1: 0.56, 2: 0.13, 3: 0.28, 4: 0.03 |
| bs128 | 668 MB | 78.6 MB | 0.118 | 773 MB | 1: 0.33, 2: 0.24, 3: 0.15, 4: 0.28 |
| bs256 | 731 MB | 167.3 MB | 0.229 | 1108 MB | 1: 0.12, 2: 0.19, 3: 0.16, 4: 0.10, 5-7: 0.15, 8: 0.28 |

512-row x 128-k stages ("XR": one x chunk serves two 256-row halves) halve both the copies and the mma.sync B loads
per weight byte. Register budget: 512 rows x T tokens of FP32 accumulators over 256 threads = 2T per thread: fine for
<= 32 tokens (64 regs), impossible for 64 (128 regs + 64 A + 32 words > 216). So at bs256 only tiles of <= 32 tokens
qualify (per expert, so an expert's tiles keep one weight layout: 45.8 of the 47.3 MB a per-tile rule would save;
mixed experts would cost 15 MB extra HBM): copies 167 -> 121 MB (1.39x); bs128 (T = 32) 78.6 -> 39.3 MB (2.0x).
Gate/up only (down items unchanged): bs128 78.6 -> 52.4 MB, bs256 -> 136 MB.

## Bounds (run 1, Modal H200, wpe_energy judge-style = flush + 1 call, 8 jm routings, timing-only builds)

| bs | base us | NO_X | XHALF (half the copy bytes) | BHALF (half the mma.sync B LDS) |
|---|---|---|---|---|
| 128 | 175.74 (153.8 mJ) | 169.35 (-3.6%, 138.6 mJ) | 174.02 (-1.0%, 151.4 mJ) | 174.79 (-0.5%) |
| 256 | 216.20 (186.5 mJ) | 205.45 (-5.0%, 160.2 mJ) | 211.80 (-2.0%, 180.8 mJ) | 215.20 (-0.5%) |

Back to back (power-capped, 1.62-1.65 GHz) bs256: NO_X -7.2%, XHALF -4.2%. NO_X also skips the producer's wait for
every CTA's gather (its whole x section is compiled out), so XHALF, not NO_X, is the bound for the bytes: halving the
copies is worth ~1% at bs128 and ~2% at bs256; XR additionally halves the mma.sync B loads (~0.5%). Expected XR:
bs128 ~1.5%, bs256 ~1-1.5% (only 28% of its copy bytes go away).

## MOE_XR design (dev/moe_v541.py, built from v365)
- Tiles: every tile of the 32-token configs (bs33-128; bs17-32 with MOE_XR_MW=1), and in the 64-token config the
  tiles of experts with <= 32 tokens; the rank order (largest first) puts those last, so tiles [nbx, ntiles) are XR and
  the queue decode stays arithmetic (scal[12] = nbx).
- Gate/up items of XR tiles: 512 rows (two 256-row items) x all k, 32 stages of 128 k: 16 items per tile.
- Stage: one 35840-byte copy from P.wx13 ([expert][128-k chunk][512-row block][32 KB words | 3 KB S6O scales],
  lossless permutation, checked offline by dev/wpx_chk_layout.py: 0 mismatches) + the x chunk (mma.sync tiles: rows of
  256 B, 16-byte group q of row r at q ^ (r & 1) -> conflict-free LDS without a pad, fits the old per-tile stride;
  wgmma tiles: 2 of the existing 64-k chunks).
- Consumer warp w: row groups (h, a) = 32 rb + 16 h + 2 w + a, 2 x 64-k chunks; S6O pair registers reused as
  P(r = a, kk = 2 h + kk'); wgmma tiles: one commit group per (kk, h), A double-buffered by h.
- Accumulators: 64-token config h = 1 in acc[a][4..7]; 32-token configs an extra acx[2][4][4] written before read.
- Epilogue: SwiGLU in registers, BF16 intermediate staged [token][264] (17 KB), epilogue warps 10-11 store it in the
  layout of the tile's (unchanged) down items; the item counts 2 on cnt1 (phase + 8 in sinfo).

## Log
- run1: bounds above (bench2 step aborted on the timing-only variants' check: needs B2_NOVERIFY).
- run2 (jm 100 reps, 2 copies each, one job; v365 controls 0.9946 / 0.9982 / 0.9961 at bs32 / 128 / 256):
  v541 XR bs32 0.9971 / 0.9968 (config unchanged), bs128 0.9821 / 0.9806, bs256 0.9948 / 0.9940;
  v542 XR + XR2 bs32 1.0006 / 0.9989, bs128 0.9837 / 0.9836, bs256 0.9722 / 0.9742. check_mt M 33..257 (normal +
  skewed, 3 seeds, 2 calls) 0 failures for both, rel-L2 2.5-3.3e-3 (unchanged). NVRTC 13: XR no new spills (bs256
  kernel 11224 -> 14952 instr, bs128 7792 -> 11464); XR2 bs128 kernel 43 LDL / 35 STL (base 10 / 32).
- run3 (MOE_TIMING + MOE_XRT build dev/moe_v545.py, judge protocol, 16 jm seeds; consumer thread 0, cycles per 32 KB
  stage; timing builds of XR spill 88-188 B, so treat the 256-row loops' numbers with care):

  | bs128 | n1 | n2 | n3 | n4 |
  |---|---|---|---|---|
  | base G1 compute | 1568 | 1836 | 2106 | 2238 |
  | XR G1 compute | 1547 | 1762 | 1862 | 2014 |
  | base G1 epilogue / 256-row item | 601 | 662 | 677 | 822 |
  | XR G1 epilogue / 512-row item (SwiGLU in the consumers) | 1754 | 3526 | 2682 | 2815 |
  | base G2 compute | 1556 | 1876 | 2068 | 2266 |
  | G2 compute in the XR build (same code) | 1611 | 2062 | 2378 | 2397 |
  | XR2 G2 compute | 1518 | 1585 | 1979 | 2075 |

  bs256 likewise: XR G1 -2.6 / -8 / -11 / -11% per stage (n1..n4), the 256-row G2 loops +4 / +5 / +8.6 / +7.4% in the
  XR build. => the XR stage itself is cheaper per weight byte (fewer B loads, fewer copy bytes, wgmma tiles -10%);
  the losses are (a) the in-consumer SwiGLU epilogue (+550..+2200 cycles per item, on the consumers' critical path)
  and (b) the unchanged 256-row loops running 4-15% slower in XR builds (code size / allocation; or a timing-build
  artifact). Estimated at bs128: +1.9% from XR stages, -0.9% epilogue, -1.5% slower G2 loops.
- v546 (= v544 + fixes): FP32 staging [token][516] + SwiGLU on the epilogue warps for XR gate/up items that fit the
  slot below the combine scratch (bs33-128: ntl <= 2, bs129-512: all), XR code emitted after the 256-row loops in
  source order (ptxas still places the XR loops first), MOE_XR_NONE (code-presence ablation), MOE_XR_MAXNTL (XR tiles
  of at most that many 8-token groups; 2 drops the wgmma XR loops). run4 measures them.
- run4 (jm 100 reps; v496 controls 0.9983 / 0.9977 / 0.9960 at bs32 / 128 / 256):

  | variant | bs32 | bs128 | bs256 |
  |---|---|---|---|
  | v546 XR (FP32 staging where it fits) | 0.9985 / 1.0009 | 0.9869 / 0.9860 | 0.9972 / 0.9970 |
  | XR code compiled, no XR tiles (MOE_XR_NONE) | 1.0012 | **0.9686** | **0.9816** |
  | XR on <= 2-group tiles only (MAXNTL=2) | 1.0011 | 0.9860 | 0.9973 |
  | XR + XR2 | 0.9992 | 0.9882 | 0.9691 |

  => the XR items gain +1.9% (bs128) / +1.6% (bs256) over the same build without XR tiles, but the mere presence of
  the XR code costs 3.1% / 1.8%. NVRTC 13 SASS (fetched): the 256-row loops keep their instruction counts but are
  re-scheduled (opcode-order similarity 0.4-0.7 vs v496), and the producer's stage loop grows 260 -> 342 instructions
  (bs256 273 / 336 -> 364). XR2 loses at bs256 in both runs (2x longer down items before the tail).
- run5 (code-presence bisection, v548 with MOE_XR_NONE; v496 controls 0.9996 / 0.9972 at bs128 / 256):
  all XR code 0.9686 / 0.9810; without the producer's 0.9717 / 0.9839; without the consumers' 0.9883 / 0.9861;
  without the epilogue warps' + gather's 0.9719 / 0.9870; without all 1.0002 / 0.9966. => a global codegen effect
  of the bigger kernel (any role's extra code costs; the consumers' 4 extra stage loops the most), not one path.
  MOE_XR_OPQ (opaque XR constants) changes nothing (0.9852 / 0.9957 with XR items).
- v550 (= v610 + unified XR, dev/wpx gen script mk_xu.py in the scratchpad): ONE stage loop per group count (<= 4)
  serves both item kinds: a stage is 4 row-group units (a, h) x 2 64-k chunks; a 256-row item maps h to the stage's
  k half (row group 2 w + a, chunk 2 h + kk: exactly the base word layout), an XR item to the row half; both
  accumulate h = 1 apart (acx), a 256-row item adds it in at its end. B addressing per item (row stride 528 / 256,
  h offset 256 / 0 bytes; wgmma chunk offset 2 T 128 / 0). The producer has one stage loop for these configs: every
  stage one 35840-B block (XR: P.wx13 / wx2 with words re-laid [warp][a][h][kk]; 256-row: the S6X wc13_6 / wc2_6
  blocks) + x. Code: bs128 kernel 9128 instructions (v610 7792; v546 XR 12064), bs256 11960 (10816; v546 15472);
  NVRTC 13 no C75xx, no new spills (local nvcc 12.1 reports C7512 at bs256: not the target compiler). Layout check
  dev/wpx_chk_layout2.py ... ah: 0 mismatches. run6 measures it.
- run6 (first try) hung at bs64; runs 7-8 bisected it on hang1: every bs >= 48 hung, even with MOE_XR_NONE; with the
  256-row producer loop (MOE_XU_NOPU) all sizes pass and are correct (so the unified consumer loop is right in
  256-row mode). Cause: the unified producer loop had no __syncwarp after lane 0's slot wait + expect_tx, so lanes
  1-3 issued the wgmma tiles' x chunk copies into a slot still being read (and onto the previous phase of its full
  barrier). Fixed; run6 re-launched.
- run6 (v550 fixed; jm 100 reps, v610 controls 0.9951 / 0.9980 / 0.9978 at bs32 / 128 / 256):

  | variant | bs32 | bs128 | bs256 |
  |---|---|---|---|
  | v550 unified XR | 0.9975 / 0.9990 | 0.9878 / 0.9849 | 0.9963 / 0.9945 |
  | unified code, no XR tiles (XR_NONE) | 0.9980 | 0.9869 | 1.0027 |
  | unified XR + XR2 | 0.9993 | 0.9861 | 0.9753 |

  check_mt M 33..257 (normal + skewed, 3 seeds, 2 calls) 0 failures for XR and XR + XR2, rel-L2 2.8-3.3e-3.
  Unification removed the code-presence cost at bs256 (XR_NONE 1.0027) but at bs128 the unified loop running the
  256-row items is 1.3% slower than v610's dedicated loops (the h = 1 accumulators acx and the per-item B offsets),
  and the XR items themselves add nothing at bs128 and cost ~0.7% at bs256 (2x longer gate/up items; the h = 1
  B loads, which the dedicated v546 loop skipped, are back). XR2 loses 2.5-3% at bs256 in all three runs.

- run9 (wpe_energy, 8 jm routings, MOE_CLK builds; v610 vs v550):

  | | bs128 judge us / mJ / kclk | bs128 b2b us / mJ / kclk | bs256 judge | bs256 b2b |
  |---|---|---|---|---|
  | v610 | 174.30 / 154.7 / 1.716 | 200.30 / 139.6 / 1.385 | 213.32 / 186.3 / 1.723 | 242.65 / 169.0 / 1.403 |
  | XR | 175.40 / 159.2 / 1.726 | 199.58 / 133.1 / 1.408 | 212.05 / 190.4 / 1.732 | 241.59 / 168.1 / 1.450 |
  | XR + XR2 | 175.66 / 153.3 / 1.730 | 199.67 / 138.6 / 1.398 | 215.58 / 187.1 / 1.726 | 244.14 / 169.3 / 1.479 |
  | no XR tiles | 175.60 / 157.9 / 1.719 | 200.24 / 138.6 / 1.424 | 212.28 / 190.7 / 1.731 | 242.12 / 164.5 / 1.445 |

  The in-kernel clock under the power cap rises 0.5-3% with XR (b2b), i.e. the fewer copies do lower power, but
  time per call stays within +-0.6% and the energy per call does not drop consistently (judge-mode mJ is noisy
  after the flush subtraction).

## Conclusion (2026-10-27)
Activation bytes per call: bs128 78.6 -> 39.3 MB with XR + XR2 (2.0x; gate/up only: 52.4 MB), bs256 167 -> 121 MB
(1.39x; the 64-token tiles cannot hold 512-row accumulators). It does not pay here:
1. The ceiling is small. Halving ALL copy bytes with nothing else changed (XHALF, timing-only) is worth 1.0% (bs128)
   and 2.0% (bs256); halving the mma.sync B loads 0.5% each. XR reaches 50% (bs128) / 28% (bs256) of that bound.
2. Every implementation costs about as much: separate XR loops add 3.1% / 1.8% just by being in the kernel (run4/5:
   NVRTC re-schedules the untouched 256-row loops; any role's extra code costs), unified loops slow the 256-row
   items 1.3% at bs128, items get 2x longer (XR2 always loses at bs256), the 512-row epilogue needs either FP32
   staging that only fits <= 16 tokens at bs128 or a consumer-side SwiGLU (+550..2200 cycles per item).
3. Per stage the XR geometry is cheaper (run3 timing build: 1-11% fewer cycles per 32 KB on gate/up stages), so a
   variant that keeps the 256-row loops byte-identical to v610 and adds XR without perturbing their codegen could
   still net ~+1% at bs128 / bs256; no such variant was found in runs 2-6.
Side finding for every worker: bs128/bs256 kernels are sensitive to unrelated code (run5: 1.8-3.1% from unused XR
code, no change in the hot loops' instruction counts) -> A/B every added code path with its runtime use switched off.
