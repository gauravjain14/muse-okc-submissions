# WP-E findings (energy and bytes at bs128 / bs256)

Worker E, started 2026-10-26. Base: dev/moe_v200.py (== submission.py). Files: dev/moe_v47x.py, dev/wpe_*.

## Tools
- dev/wpe_energy.py: per variant and shape, K CUDA graphs (first K judge-matched seeds), replayed back to back
  (b2b: 20 per sync, no flush) and judge-style (flush.sum + 1 replay + sync, flush-only energy subtracted);
  NVML total-energy counter sampled every 2 ms (power = slope), NVML SM clock and power-cap reason, and the
  in-kernel clock from MOE_CLK stamps (clock64 vs globaltimer per CTA, also binned by 20 us of the call).
- dev/wpe_cc.py: local nvcc (12.1, sm_90a) compile check + SASS opcode counts (no GPU / cuda-python needed).
- MOE_CLK (v470+): per-CTA stamps in ctr[40000 + 32 cta]: start, 14 item starts, end.

## Static facts (offline, judge-matched routings, 100 seeds per shape)
Tile token-group (ntl) histogram = fraction of all stages (every tile has 48 items x 16 stages):

| shape | T | experts | pairs | ntl1 | ntl2 | ntl3 | ntl4 | ntl5-8 | mma.sync stages | padding B rows |
|---|---|---|---|---|---|---|---|---|---|---|
| bs32  | 32 | 16.45 | 58.9  | 0.913 | 0.039 | 0.044 | 0.004 | - | 0.952 | 0.607 |
| bs128 | 32 | 23.05 | 190.2 | 0.710 | 0.166 | 0.051 | 0.072 | - | 0.876 | 0.341 |
| bs256 | 64 | 25.04 | 398.9 | 0.476 | 0.245 | 0.104 | 0.050 | 0.124 | 0.721 | 0.201 |

=> at bs128/256 most stages are 1-2 group mma.sync tiles (BF16, FP32 accumulators), and 20-34% of the B
columns the tensor cores multiply are padding: in v200 they hold stale shared memory (mma.sync tiles copy only
the tile's n rows) or uninitialized global rows (wgmma tiles copy whole 8-row groups of xs / inter).

## Log
- run1 (v470, Modal H200, 2026-10-27): smoke + check_mt 17..257 pass for MOE_CLK / MOE_ZPAD.
  bench2 jm100 x2 (vs v200; control v200b 0.9981 / 0.9978 / 0.9964 at bs32 / 128 / 256):
  MOE_ZPAD bs32 0.9986/0.9997, bs128 0.9941/0.9946, bs256 0.9967/0.9982 -> no gain (bs128 -0.5%): rejected.
  MOE_SUSPEND_ALL 0.9972 / 0.9978 / 0.9988 -> noise: rejected.
  Energy numbers of run1 are INVALID (tool bug: the K graphs' captured input tensors were not kept alive,
  so later allocations overwrote the routings; per-call times then varied 30-50% between identical copies).
  What survives: no-compute (MOE_NO_COMPUTE) runs at 1.97-1.98 GHz in-kernel at 490-625 W, i.e. the
  data path alone is NOT power-capped; with compute every shape hits the 700 W cap back to back.
- dbg1 (dev/wpe_dbg.py, v470 + MOE_CLK): per-routing kernel times differ a lot inside one shape (bs128
  150-200 us, bs256 175-237 us over the first 8 jm seeds), so any K-graph rotation must use the mean.
  Back to back (no flush) the in-kernel clock is 1.55 GHz at bs128 and 1.53 GHz at bs256 (judge-style
  with the 512 MB flush between calls: ~1.70-1.75 GHz, run1's judge-mode stamps); all CTAs start within
  0.3 us and end within 7-11 us of each other.

## Candidates built (v471, v472)
- v471 MOE_F16MS: the FP16 configs' (bs17-256, (8,4,1) / (8,8,1)) mma.sync tiles (1-2 token groups, 72-95% of
  the stages) on FP16 operands and FP16 accumulators like their wgmma tiles: _POSH words for every tile, the
  gather's row-major rows and the SwiGLU's row-major intermediate converted to FP16, stage_compute_h
  (mma.sync m16n8k16 f16.f16.f16.f16 into hacc), hacc -> fp32 x 0.5 at the item end for every tile.
  Static cost: dec16 is 7 ALU + 3 IMAD per word (the BF16 decode 7 + 2), +1 IMAD per word.
- v471 MOE_S6X: 6-bit scales also for the one-k-warp configs (bs17-256): (8,4,1) through the existing
  s13c6 path, (8,8,1) through new 35840-byte combined stage blocks (wc13_6 / wc2_6 / wc13h_6 / wc2h_6:
  32 KB words + 3 KB scales); FP16 6-bit scale decode scale6h (the existing MOE_S6_WG path applied the
  BF16-format scale6 to the FP16 wgmma tiles: wrong with MOE_F16_NT4). Stage loop +15 instr (NTL1).
- v472 MOE_S6O: 6-bit scales in a layout built for the decode. f = E4M3 byte - 0x40 (lossless for every
  scale in 0x40..0x7F; zero scales have zero codes). Exponent offsets chosen so the constant only ORs and the
  largest octave stays finite: BF16 0x7800 | f << 4 = s * 2^112, FP16 0x4000 | f << 7 = s (both give products
  E2M1 * s * 2^-14 = 64x the 8-bit path's 2^-8 convention: accumulators x 64 at the item end; FP16 sums are
  128x smaller than today's, i.e. further from overflow). Each register holds the scales of rows g (low half)
  and g + 8 (high half) of one (row group, 64-k chunk); the multiplies select the half (HMUL2 .H0_H0/.H1_H1,
  one instruction, checked in SASS for BF16 and FP16). Per lane and stage: 3 words, 8 pair registers from 13
  ALU + 4 IMAD (8-bit path: 16 PRMT + 16 IMAD; scale6: ~30 ALU + 16 IMAD). The OR constant must come from the
  constant bank (two immediates made ptxas split every (w & M) | C into 2 LOP3s).
  SASS stage loops (nvcc 12.1): (8,4,1) NTL1 516 instr vs 528 (8-bit) / 543 (S6X scale6); FP16 wgmma NTL3
  609 vs 618 / 633; bs8 config (4,1,1) NTL1 517 vs 544 (production S6). Packing checked offline against an
  emulated decode (31193 scales, 0 mismatches; the first try with BF16 s * 2^120 made the top octave Inf).

## Results (Modal H200, bench2 judge-matched jm100, 2 copies per variant, same job)
- run3 (v472): MOE_S6O + MOE_S6X vs v200 (control v200b 0.9998 / 0.9989 / 0.9972 / 0.9985 / 0.9972):
  bs8 1.0082 / 1.0079, bs16 1.0129 / 1.0115, bs32 1.0128 / 1.0133, bs128 1.0127 / 1.0142, bs256 1.0146 / 1.0160.
  MOE_S6O alone (only the bs8-16 configs change): bs8 1.0066 / 1.0124, bs16 1.0069 / 1.0130, others flat.
  check_mt M 2..257 (normal + skewed, 2 calls) 0 failures; rel-L2 unchanged (bs8 3.22e-3, bs128 3.52e-3,
  bs256 5.19e-3). NVRTC 13.0 ptxas: no C7511/C7513, no spills.
  Energy (wpe_energy, 8 routings): b2b bs256 233.9 vs 236.2 us/call, 162.2 vs 164.5 mJ/call, in-kernel clock
  1.45-1.48 vs 1.42 GHz; bs128 197.1 vs 199.1 us, 137.0 vs 138.6 mJ. Judge-style bs256 210.3 vs 212.1 us,
  bs128 174.4 vs 175.8 us. KEPT: v472 with MOE_S6O,MOE_S6X.
- run4 energy decomposition of v200 (v470 + MOE_CLK; timing-only switches), judge-style (flush + 1 call):

  | bs256 | us | vs full | in-kernel GHz | b2b mJ/call |
  |---|---|---|---|---|
  | full | 217.1 | 1 | 1.66 | 170 |
  | data only (NO_COMPUTE) | 183.1 | 0.843 | 1.97 | 122 |
  | compute only (NO_LOAD) | 186.9 | 0.861 | 1.82 | 78 |
  | no activation copies (NO_X) | 203.7 | 0.938 | 1.78 | 158 |
  | no scale multiplies (HMUL_NONE) | 203.6 | 0.938 | 1.72 | 155 |
  | no tensor work (NO_MMA + NO_WGMMA) | 200.6 | 0.924 | 1.85 | 144 |
  | MOE_ZPAD | 217.0 | 1.000 | 1.69 | 166 |

  bs128: full 178.3, data only 162.6 (0.912), compute only 153.8, NO_X 172.5 (0.967), HMUL_NONE 169.5 (0.950),
  no tensor 172.6 (0.968), ZPAD 178.6 (1.002). Back to back the GPU sits at 697 W, in-kernel 1.34-1.40 GHz
  (hot GPU, 74 C; 1.42 at 64 C in run3); data only draws 650 W at 1.98 GHz, compute only 430-490 W at 1.82.
  Readings: (1) the data side is ~2/3 of the energy (122 of 170 mJ b2b at bs256) and does not shrink with the
  SM clock, so every byte off it lifts the clock; (2) the activation copies (L2 -> SMEM, ~8 KB per 36 KB stage
  on average, 32 KB at ntl 8) cost 6% of the bs256 time and 7-8% of its energy: the largest data sink left after
  the weights; (3) zero padding (20-34% of B columns) does not change power or time: the data-dependent tensor
  power here is below noise.

## Composition with WP-D's post-scale (planner request, 2026-10-27)
- v471 MOE_F16MS (FP16 mma.sync tiles) overlaps WP-D's MOE_PS2 almost exactly: not developed further and not
  measured (its run 2 died after the ptxas step; the rerun was stopped under the 5-instance rule).
- WP-D's post-scale stage (stage_ps, 1-group tiles) needs one (row g, row g + 8) scale pair per MMA block: 32
  pairs per lane and stage, quad-redundant, from `cvt.rn.f16x2.e4m3x2` (1 ALU op per pair) on an ALU-bound
  stage (257 ALU). The S6O pair register has exactly the f16x2 (s_g, s_g8) form the HFMA2 .H0_H0/.H1_H1 need,
  but 6 bits do not fit 12-bit-per-pair "direct" slots densely: the best dense layout found costs 1.25 ALU +
  0.9 IMAD per pair (+8 ALU per stage, ~3%) for -2.8% bytes on those tiles. Not built; likely a wash.
- v473 = v450 + MOE_S6X (+ MOE_S6O): with MOE_PS + MOE_PS2, the 2-group (stage_h) and FP16 wgmma tiles of
  bs17-256 read MOE_S6O 6-bit scales (s13x / s2x, and 35840-byte combined blocks wc13hx / wc2hx for the
  64-token config), the post-scale tiles keep their 8-bit s13ps / wc13ps, the ring slots keep the 8-bit size
  (6-bit tiles copy 3 KB into the 4 KB scale area and post 1 KB less per stage). 6-bit stages:
  bs128 29%, bs256 52% of all stages. MOE_S6O (bs8-16 configs) ported too.
  Touched in v473: s6o_h / s6o_bf / dequant_ho4 / dequant_o4 (new), stage_compute_wgh (S6V = MOE_S6O),
  stage_h (S6V template parameter), stage_compute (MOE_S6O branch), moe_body (S6P / S6PF, MOE_SOFF6, the
  dispatch of K = 52 and wgmma tiles, the two FP16 conversion factors, the producer's per-tile scale source,
  sizes and expect_tx, the BF16 x64 factor for MOE_S6O), Params (s13x, s2x, wc13hx, wc2hx), Python
  (_pack_s6o, _s6x_blocks, the kernel_fn pointers, MOE_S6O switch of _pack_s6).
- run5 (one job, bench2 jm100, vs v200; control v200b 0.9974 / 0.9994 / 0.9980 / 0.9984 / 0.9970 at bs8/16/32/128/256):

  | variant | bs8 | bs16 | bs32 | bs128 | bs256 |
  |---|---|---|---|---|---|
  | v450 PS+PS2 (WP-D) | 0.9987 / 0.9977 | 0.9999 / 1.0002 | 1.0093 / 1.0090 | 1.0086 / 1.0084 | 1.0082 / 1.0086 |
  | v473 PS+PS2+S6X | 1.0014 / 1.0008 | 1.0026 / 0.9998 | 1.0086 / 1.0088 | 1.0121 / 1.0146 | 1.0100 / 1.0139 |
  | v472 S6O+S6X | 1.0059 | 1.0155 | 1.0128 | 1.0144 | 1.0139 |

  => post-scale and the 6-bit pre-scale path do not stack: on 1-group tiles they buy the same, and the 6-bit
  scales on the post-scale config's other tiles only bring it to v472's level. v472 S6O+S6X stays the best.
- v474 = v472 + MOE_F16MS2. Since post-scale does not stack on S6O, the remaining FP16 lever that keeps the
  6-bit decode is FP16 accumulation on the mma.sync tiles (WP-D isolation: HMMA.F16 costs ~2.6 issue cycles vs
  ~4.6 for HMMA.F32; NTL2 stage -9.4%, NTL1 -2.7%). v472's MOE_F16MS does that for 1- and 2-group tiles (FP16
  S6O pairs, dequant_ho); MOE_F16MS2 restricts it to the 2-group tiles (1-group tiles stay BF16, whose decode is
  one IMAD per word cheaper): per-tile choice in the gather, the SwiGLU stores, the producer's word source and
  the stage dispatch. Measured in run6.

## MOE_S6O scale layout, exactly (for rebuilding WP-S's WBLK / SEPI on it)
Source: dev/moe_v472.py `_pack_s6o` (Python) and `s6o_bf` / `s6o_h` (kernel). Same for every config that has
S6 (bs8-16 (4,1,1) / (4,2,1) with MOE_S6O; plus bs17-256 (8,4,1) / (8,8,1) with MOE_S6X).

Notation (unchanged from v200's 8-bit layout): rows are the repacked rows (w13: `_gate_up_rows` order, row group
j = gate rows 8j..8j+7 then up rows 8j..8j+7); row group rg = 16 rows; 16-k block index b = 16 kc + 4 kk + tig
(kc = 256-k chunk 0..15, kk = 64-k sub-chunk 0..3, tig = lane & 3); g = lane >> 2; row = 16 rg + 8 h + g (h = 0:
row g, h = 1: row g + 8). 8-bit v200 layout: per (kc, rg) 256 B = [g 8][tig 4][kk 4][h 2], i.e. a lane (g, tig)
owns 8 bytes, byte kk * 2 + h.

6-bit field: f = E4M3 byte - 0x40 (lossless iff every byte is 0 or 0x40..0x7F; zero scales -> f = 0, harmless
because their codes are zero; prepare() checks and falls back to 8 bits otherwise).

Global arrays (`state["s13c6"]` / `state["s2c6"]` when MOE_S6O, same tensors the kernel gets as P.s13 / P.s2):
  [expert 32][kc 16][row-group pair p][lane 32][12 B]     (p = rg / 2; w13 256 pairs, w2 128 pairs)
  = 192 B per row group, 384 B per pair, contiguous over (kc, p, lane). Byte offset of (e, kc, rg pair p):
  ((e * 16 + kc) * NPAIR + p) * 384, NPAIR = 256 (w13) / 128 (w2).
Per lane the 12 bytes are three little-endian u32 words w0, w1, w2 holding the lane's 16 fields as 8 pairs
P(r, kk) (r = rg % 2 within the pair, kk 0..3), P.lo = f(row 16 rg + g, block b), P.hi = f(row 16 rg + g + 8, b):
  w0: [0..3] P(0,3).lo bits 0-3 | [4..9] P(0,0).lo | [10..15] P(1,0).lo | [16..19] P(0,3).hi bits 0-3 | [20..25] P(0,0).hi | [26..31] P(1,0).hi
  w1: [0..3] P(1,3).lo bits 0-3 | [4..9] P(0,1).lo | [10..15] P(1,1).lo | [16..19] P(1,3).hi bits 0-3 | [20..25] P(0,1).hi | [26..31] P(1,1).hi
  w2: [0..1] P(0,3).lo bits 4-5 | [2..3] P(1,3).lo bits 4-5 | [4..9] P(0,2).lo | [10..15] P(1,2).lo |
      [16..17] P(0,3).hi bits 4-5 | [18..19] P(1,3).hi bits 4-5 | [20..25] P(0,2).hi | [26..31] P(1,2).hi
Decode (per lane, once per stage): R[r * 4 + kk] = pair register, BF16 (w & 0x03F003F0) | 0x78007800 etc.
(s6o_bf), FP16 with the field moved to bits 7..12 (s6o_h); the OR constant must come from the constant bank.
Multiplies: a0 = o0 * R.H0_H0, a1 = o1 * R.H1_H1, a2 = o2 * R.H0_H0, a3 = o3 * R.H1_H1. Products are
E2M1 * s * 2^-14, so the item's FP32 accumulators are multiplied by 64 at the item end (before any staging).

Stage slot for the (4,*,1) configs (RW 4, KW 2: 128 rows x 512 k, RGS 8): [WB = 32768 B words][SB = 3072 B
scales][x]. Scales: k-warp kw (= k chunk kc0 + kw) at WB + kw * RGS * 192, its RGS = 8 row groups = 4 pairs x
384 B in rg order = the global bytes ((e * 16 + kc0 + kw) * NPAIR + rb * 4) * 384 .. + 1536. The producer copies
them as one 1536-byte bulk copy per k chunk (lane 2 kw + 1), exactly like the 8-bit layout but 192 B per rg.
Consumer warp (rw, kw) reads its pair (row groups 2 rw, 2 rw + 1) at WB + (kw * RGS + 2 rw) * 192 + lane * 12
(MOE_SOFF). A WBLK-style combined block per stage would therefore be: [kc0 words 16 KB][kc0 + 1 words 16 KB]
[kc0 scales 1536 B][kc0 + 1 scales 1536 B] = 35840 B per 128-row x 512-k stage (the 8-bit one is 36864 B).
- run6 (bench2 jm100 x2, vs v200; control v200b 0.9986 / 0.9983 / 0.9953 at bs32 / 128 / 256):
  v472 S6O+S6X 1.0125/1.0139, 1.0113/1.0134, 1.0103/1.0096;
  v474 S6O+S6X+F16MS 1.0224/1.0201, 1.0267/1.0272, 1.0259/1.0255 (KEPT: FP16 accumulation stacks on S6O);
  v474 +F16MS2 (2-group tiles only) 1.0072/1.0068, 1.0099/1.0107, 1.0149/1.0177 (rejected).
  check_mt 17..257 0 failures for both (F16MS rel-L2 5.2-5.4e-3, tolerance 1.5e-2).
- Activation-copy traffic (offline, 40 jm routings; gate/up x and down intermediate alike, per stage: wgmma tiles
  4 x 8 ntl x 128 B, mma.sync tiles n x 528 B): bs128 81 MB per call (0.12 of its 686 MB of weights + scales),
  bs256 164 MB (0.22 of 739 MB), all L2 -> SMEM. With run4's NO_X ablation (bs256 -6.2% time, -7% energy;
  bs128 -3.3%) this is the largest data-side sink left: per byte it costs about half of an HBM byte. Only a
  tiling change reduces it (e.g. 512-row x 128-k stages for the wgmma tiles, which halve x bytes per weight byte,
  or pairing two row blocks of one tile in a 2-CTA cluster with x multicast); out of WP-E's scope (WP-L).

## Rule change 2026-10-27: BF16 only (no FP16 operands or accumulators)
- v474 MOE_F16MS (+2.6% bs128/256 on top of S6O) and every FP16 part of v472's S6X gain are void. New baseline
  dev/moe_v601.py (= v200 without MOE_F16 / MOE_F16_NT4). v475 = v472 with the same two defines commented out:
  MOE_S6O + MOE_S6X then give every main-kernel config 6-bit scales with the BF16 S6O decode (0x7800 | f << 4),
  BF16 wgmma tiles (stage_compute_wg) and the 64-token config's combined BF16 blocks (wc13_6 / wc2_6) included.
  v475 without defines compiles to the same SASS as v601 (nvcc 12.1 counts identical). Measured in run8.
- run8 (BF16 only, one job): v475 MOE_S6O + MOE_S6X vs v601 (control v601b 0.9895 / 0.9927 / 1.0020 / 0.9994 /
  0.9987 / 0.9986 at bs1/8/16/32/128/256): bs1 0.9907 / 0.9853 (one-token kernel unchanged: noise), bs8 1.0061 /
  1.0028, bs16 1.0128 / 1.0128, bs32 1.0120 / 1.0154, bs128 1.0127 / 1.0145, bs256 1.0172 / 1.0185; geomean over
  six x2.4032 / 2.4017 vs v601 x2.3801 / 2.3783 (+0.95%). check_mt M 1..257 (33 sizes, normal + skewed, 2 calls):
  0 failures, rel-L2 ~3.2e-3. NVRTC 13.0 ptxas clean. KEPT: dev/moe_v475.py with MOE_S6O,MOE_S6X is the BF16 base.

## Worth trying next (WP-E view)
1. The activation copies (L2 -> SMEM, 81 / 164 MB per call at bs128 / 256; NO_X -3.3% / -6.2% time, -7% energy):
   stage shapes that reuse x across two row blocks (512 x 128 k for the wgmma tiles) or 2-CTA x multicast.
2. A 6-bit post-scale stage for 1-group tiles (WP-D's MOE_PS with S6O-style pairs): needs a dense pair layout at
   <= 1.25 ALU + 0.9 IMAD per pair (32 quad-redundant pairs per lane and stage); post-scale and the 6-bit
   pre-scale path gave the same in run5, so only a 6-bit post-scale can add to S6O.
3. S6O for the one-token kernel (bs1, WP-1): its stages still carry 8-bit scales (4 KB per 36 KB).
4. Not worth more time: zero padding (no measurable power effect), consumer suspend hints (noise).
