# WP-S5 findings: fewer scale bytes, losslessly (5-bit / 4-bit block-scale codes)

Worker S5, 2026-10-27. Base: dev/moe_v894.py. Files: dev/moe_v930.py (timing bound switches), dev/wps5_ana.py,
dev/wps5_run1.sh. Modal job wps5_run1 (ap-<id>).

## Step 1: timing-only bound (dev/moe_v930.py, default build = v894 SASS byte-identical)
- MOE_S5B n (main kernels): every stage copies SB - n/6 SB scale bytes (n = 1: 2560 of 3072 B, i.e. a 5-bit layout;
  n = 2: 2048 B, 4-bit). Copy sizes, expect_tx counts and the GLOBAL strides / offsets of s13c6 / s2c6, of the WBLK
  blocks (wb13 / wb2, bs 2-16) and of the WSC blocks (wc13_6 / wc2_6, bs 33-512) all use the shrunk sizes, so each stage
  reads contiguous, fewer bytes. Sites: the (4,*,1) early start, the per-item descriptors (csrc / cstep / cbytes / ltx),
  LEAN3 / LEAN4 (wq / wqs / ltx3), the full path (expect_tx, WSC + WBLK copies), wp32_fst (bs 17-32 fast start).
  The smem layout and the consumers are unchanged: consumer stage loops byte-identical (wpc_cmp loop / gloop), producer
  loops same length, constants only. Outputs are wrong (stale scale bytes): bench2 with B2_NOVERIFY=1.
- MOE1T_S5CUT n (moe1t, bs1): its 36864-byte stages (32 KB words + 4 KB 8-bit scales) copy n bytes fewer, global stride
  36864 - n (1536: 35328-byte 5-bit stages; 2048: 4-bit).

Results (bench2 jm100, B2_SHARE_EQ, same job; ratios vs v894, control v894b in the first column):

| shape | v894b | 5-bit b5 / b5b | net 5-bit | 4-bit b4 (1 copy) | net 4-bit |
|---|---|---|---|---|---|
| bs1   | 1.0019 | 1.0052 / 1.0031 | +0.32% | 1.0138 | +1.28% |
| bs8   | 0.9995 | 1.0014 / 1.0074 | +0.47% | 1.0158 | +1.60% |
| bs16  | 0.9983 | 1.0066 / 1.0087 | +0.85% | 1.0167 | +1.75% |
| bs32  | 0.9977 | 1.0056 / 1.0039 | +0.59% | 1.0110 | +1.21% |
| bs128 | 1.0005 | 1.0046 / 1.0033 | +0.37% | 1.0113 | +1.11% |
| bs256 | 0.9975 | 1.0024 / 1.0040 | +0.45% | 1.0104 | +1.17% |

(net = candidate mean / control mean - 1; geomean 5-bit ~+0.5%, 4-bit ~+1.35%.)
- 5-bit: -1.43% bytes per main-kernel stage buy +0.47 / +0.85 / +0.59% at bs8 / 16 / 32 (time-per-byte elasticity
  0.33-0.59: the memory-paced shapes sit at the C/M knee, not on a pure DRAM line); moe1t -4.2% bytes buy +0.3%
  (elasticity 0.08: bs1's stages are not byte-limited enough for scale bytes to matter).
- VERDICT: the 5-bit bound is below the +1% gate on every memory-paced shape BEFORE any decode cost: the lossless
  5-bit layout was NOT built (rule of the task). The 4-bit bound passes (+1.1..1.75%) but no lossless 4-bit layout
  exists for these weights (below).

## Data (dev/wps5_ana.py on the judge weights, seed 5252, local experts 0-31; run on Modal, CUDA generator)
- 100.7M block scales, all in 0x5C..0x7E (no zeros, none outside S6O's 0x40..0x7F). Row spans (max - min code over a
  row's 256 blocks) 11..29 codes, mean ~18. Row min octave (code >> 3): 11 for 13 rows, 12 for 268274, 13 for 124929.
- 5-bit, octave-aligned per-row base (OR decode `(w & M5) | C` + per-row 2^k factor at the item end): 11 rows misfit
  (9 w13, 2 w2; min 0x5E-0x5F, max 0x78-0x7C = five octaves). Any OR / XOR-per-row scheme has these misfits (the 32
  codes an OR decode can produce form an octave-aligned 4-octave window), so it needs a fallback for the judge weights.
- 5-bit, any per-row base: 0 rows misfit. Decode R = (f5 << 4) + C_row with C_row = 0x7800 + ((base_row - 0x40) << 4)
  per BF16 half gives exactly S6O's pair register (no item-end factor change, the global x64 stays), at the cost of an
  add instead of an OR (2 registers per lane per item: C for row groups 0 and 1 of the lane's pair).
- 4-bit per 64-block group: 57% (octave-aligned) / 38% (any base) of groups misfit; per 8-block groups still 10%
  (octave-aligned) and the bases then cost ~1 bit per scale: no lossless 4-bit layout.

## Why the real 5-bit version would land below the bound (design notes, not built)
- Per lane and stage S6O decodes 16 fields (8 BF16 pair registers) from 3 words in 17 ops (13 ALU + 4 IMAD): 3 pairs in
  place (1 LOP3 each), 3 shifted (2 ops), 2 split across words (4 ops). A pair needs its two fields 16 bits apart, so a
  32-bit word holds at most 3 pairs (30 bits, 2 bits unusable); 16 fields x 5 bits = 80 bits = LDS.64 + LDS.U16 forces
  the last pair's fields onto the u16 + the 4 spare bits: estimated ~20 ops (OR or additive alike: off-place fields cost
  mask + shift-add = 2 ops either way), i.e. ~+3 ops per stage (~+0.6% consumer issue) against a bound of +0.5%, at the
  C/M knee where compute costs more than linear. Packing 3 pairs per word without split fields needs 86+ bits per lane
  (lane triples sharing 8 words: -341 B per stage instead of -512, ~-0.95% bytes, lane-dependent extraction).
- Expected net of a built 5-bit layout: ~0 to +0.2%, possibly negative. Not worth a scored candidate.

## Recommendation
- Do not integrate (nothing built beyond the timing switches). The scale-byte lever is exhausted at 6 bits for these
  weights: 5-bit pays at most ~0.5% before decode cost, 4-bit is not lossless.
- For the planner: the byte elasticity measured here (0.33-0.59 on bs8-32 main kernels, 0.08 on moe1t) is a general
  conversion factor for any byte-only lever on the memory-paced shapes (e.g. x / intermediate copy traffic).
