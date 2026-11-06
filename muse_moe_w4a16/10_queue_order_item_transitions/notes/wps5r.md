# WP-S5R findings: 5-bit block scales with a per-row base (lossless) for the stream-bound bs8/16 kernels

Worker WP-S5R, 2026-11-01. Base dev/moe_v1504.py (= submission.py, scored 5157.2), then dev/moe_v1506.py. Files:
dev/moe_v2400.py (v1504 + MOE_S5R for the (4,*,1) kernels), dev/moe_v2401.py (v1506 + MOE_S5R incl. bs17-32 and
per-kernel opt-outs; all switches default OFF, default build = v1506 SASS), dev/wps5r_cpu.py (host lossless check),
dev/wps5r_bitid.py (GPU bit identity, nondeterminism-aware), dev/wps5r_run1-3.sh (Modal jobs).

TL;DR: BANK dev/moe_v2401.py MOE_S5R,MOE_S5R_NO16,MOE_S5R_NO8: bs8 +0.75% (3 jobs, 12 copy-passes +0.52..+1.01%), other
shapes neutral, ~+0.12% geomean; lossless with an automatic S6O fallback. 5-bit at bs16 (-0.35%) / bs32 (-1.5%) loses:
the 21-op decode (S6O: 17) costs more than the 1.4% fewer bytes buy in those near-consumer-bound kernels.

## Design (MOE_S5R, the (4,*,1) WBLK configs = moe_4_1_1_4_1 / moe_4_2_1_4_1, bs 2-16)
- Per weight row (repacked order) a base b = its smallest non-zero E4M3 byte; per scale f = byte - b in 5 bits. prepare()
  checks every row of w13 / w2 (non-zero bytes span <= 31, all bytes 0 or 0x40..0x7F) and otherwise falls back to the
  S6O layout for the whole module (the CUDA source then `#undef`s MOE_S5R: the fallback build is SASS-identical to the
  default = v1504). Zero scales (their codes are zero) decode to any value.
- The pair register is exactly S6O's 0x7800 + ((byte - 0x40) << 4) per BF16 half, built as C + (f << 4): no change to
  the products, the accumulators or the item end (x64 as before) -> outputs bit-identical to v1504.
- C = 0x7800 + ((b - 0x40) << 4) per half (rows g / g + 8), one u32 per (lane row group): the item's 64 C words (256 B,
  [rw][g][rg]) ride in the global stage block past the 5-bit scales and are copied with the item's FIRST stage only
  (copy size WB + 2560 + 256 at st == 0, WB + 2560 otherwise; global block stride 35584): the consumers read their two C
  registers (one LDS.64) right after the item's first stage wait, outside the stage loop. Stage bytes read: 35328 per
  stage + 256 per item (v1504: 35840 per stage) = -1.40% at 8 stages per item. Shared-memory slot layout unchanged.
- A warp's 320 B per stage: [lane][W0, W1] (LDS.64, conflict-free) + one shared word S per lane pair (LDS.32,
  broadcast, conflict-free). Fields lo at x / hi at x + 16:
  W0: P(0,0) x=1, P(1,0) x=6, P(1,1) x=11, bits 0/16 = bit 4 of P(0,2); W1: P(0,1) x=1, P(1,2) x=6, P(1,3) x=11,
  bits 0/16 = bit 4 of P(0,3); S rotated right by 8 (lane & 3): [0..3]/[16..19] P(0,3) bits 0-3, [4..7]/[20..23] P(0,2).
  Row group 0's pairs: LOP3 + IMAD (x 8 + C0, constant-bank multiplier); row group 1's: LOP3 + LEA.HI (>> k + C1);
  the split pairs: 2 LOP3 + 2 IMAD. 21 ops (15 ALU, 6 IMAD) vs S6O's 17 (13 ALU, 4 IMAD).

## CPU / SASS (before any GPU job)
- dev/wps5r_cpu.py, seeds 5252 (judge weights), 1, 2: every non-zero scale's pair-register half equals S6O's and the
  reference formula (100.7M scales per seed, 0 differences); WBLK block layout byte-checked; row spans max 29 / 30 / 29
  (limit 31: 0 misfit rows on all three seeds); synthetic span-31 row fits and decodes, span 32 and byte 0x3F fall back.
- SASS (NVRTC 13, dev/wpc_sass.py): default build and MOE_S5R-without-fit (fallback) build: all 16 kernels
  hash-identical to v1504. MOE_S5R: only moe_4_1_1_4_1 / moe_4_2_1_4_1 change, no spills, 168 registers;
  stage loops (cuobjdump, back-branch spans): 4_1_1 1-group 510 -> 513, 4_2_1 2-group 558 -> 561, 1-group 512 -> 516.
  The compiler's 2x unroll of the 1-group loop survives (4_1_1 still has the 64-HMMA 2-stage block, 1093 -> 1103).
- Dead ends on the way (SASS): (a) mad.hi for the right shifts: IMAD.HI.U32 takes a 64-bit addend, so each use needs a
  {C : 0} register pair, which the allocator rebuilds with 2 IMAD.MOV per use at 168 registers (+13 instructions per
  stage); mad.wide with a {C1 : C0} pair is split by ptxas into IMAD.WIDE + a 64-bit add. (b) Without pinned lane
  offsets the 2-group loop of 4_2_1 re-derives them from %tid in the loop (+10 per stage); pinning the two offsets and
  the rotation in opaque registers (as MOE_PIN does) brings it to +3. (c) Rotation by the own words' smem address: worse.
- Timing-only sub-switch MOE_S5R_BOUND: the 5-bit producer (same bytes) with the S6O consumer decode (garbage outputs):
  its stage loops are instruction-identical to v1504's -> isolates the byte gain from the decode cost.

## Run 1 (dev/wps5r_run1.sh, Modal H200, 2026-11-01 ~01:20 local; v2400 vs v1504)
- hang1 bs 1..128 ok (S5R prints "5-bit scales on" for the judge weights); bound runs (wrong outputs, as designed);
  check_mt M 2..16 normal + skewed x 3 seeds x 2 calls: 0 failures.
- Bit identity (first version of wps5r_bitid.py): 224 "mismatches", but every one printed was in the forced-fallback
  case, whose kernels are SASS-identical to v1504 -> the base itself is not run-to-run deterministic (float red.add of
  the k-split pieces: max |diff| 1e-8 .. 4e-3, one BF16 ulp). The test was rewritten (3 runs of each version per case:
  deterministic cases must match exactly, others must lie within the base's own spread).
- bench2 jm seeds, 150 reps, B2_SHARE_EQ, fillers at both ends; ratio = median paired v1504 / variant, mean over the two
  v1504 controls (>1 = faster):
  | shape | S5R | S5R copy 2 | BOUND | BOUND copy 2 | v1504b |
  |---|---|---|---|---|---|
  | bs8  | 1.0069 | 1.0070 | 1.0042 | 1.0043 | 0.9983 |
  | bs16 | 0.9971 | 0.9987 | 1.0061 | 1.0059 | 1.0000 |
  | bs32 (placebo) | 0.9981 | 0.9992 | 0.9985 | 1.0002 | 0.9996 |
  bs8: +0.7% (the decode costs nothing there: the 1-group kernel has consumer slack). bs16: the bytes alone are worth
  +0.6% but the decode costs ~0.8% (the 2-group kernel is near consumer-bound): net -0.2%.

## Why the decode needs an add (and why the 21 ops)
- An OR / XOR / LOP3 against a per-row constant can only produce the 32 codes of one 32-aligned window (4 octaves): the
  octave-aligned base (fold 2^k into the item end, OR decode) misfits 5-9 rows per matrix on every seed tried (5252, 1,
  2: span + misalignment up to 37), so it would always fall back. Any-base needs C + (f << 4) with carries: one add per
  pair register (8 per stage) on top of the masks; S6O's in-place pairs are a single LOP3 with the OR constant.
- A 32-bit word holds at most 3 pairs (fields 16 bits apart, 5-bit windows at x <= 11: 30 bits, the spare bit at
  0/5/10/15 and +16), so 80 bits per lane force two pairs split across a shared word and the own words' spare bits.
  Only one window per word sits at x <= 4 (left shift: IMAD with the add); the others need a right shift + add (LEA.HI,
  ALU, or IMAD.HI with the {C:0} pair problem). Exhaustive check of the split pairs: no layout with 3-op split pairs
  and a single lane rotation exists (the lane's nibble set and its partner's must tile Z_16; the in-place shape
  {0,2,3,4} cannot appear in any tiling).
- Static schedule (sum of SASS stall counts per stage loop, cuobjdump control words; v1506 -> S5R -> S5R_IMADHI):
  4_1_1 1-group 743 -> 754 -> 750; 4_2_1 2-group 879 -> 877 -> 844, 1-group 753 -> 763 -> 756; 8_4_1 (bs32) DS 757 ->
  800 -> 769, 1-group 739 -> 738 -> 770, wgmma n24 773 -> 768 -> 745, n32 795 -> 800 -> 779.

## v2401 = v1506 + MOE_S5R incl. the bs17-32 kernel (planner: measure bs32 against v1506)
- S5R patch applied cleanly to v1506; added the (8,4,1,*,1) config: scales from s13c5 / s2c5 (s13c6's strides, per
  (k chunk, 16 row groups) [2560 B 5-bit | 512 B C registers]), copy 2560 B (+512 at an item's first stage) in the main
  path and in wp32_fst; 5-bit decode in its 1-group (stage_compute5), 2-group WP-DS (stage_ds S5 flag) and wgmma
  (stage_compute_wg S5 flag) stages. Sub-switches MOE_S5R_NO16 / MOE_S5R_NO8 (per-kernel opt-out; with NO16 the bs 2-8
  kernel gets its own wb13s5 / wb2s5 copy), MOE_S5R_IMADHI, MOE_S5R_NOPIN, MOE_S5R_ROTA, MOE_S5R_BOUND.
- SASS: default and fallback builds = v1506 (16/16 kernels). bs32 stage loops +2 (DS) / +5 (1-group) / +5 / +5 (wgmma).
  Lesson: unused constexpr locals in moe_body (S5P, SB5, ...) alone flip a bistable ptxas schedule of the routing prefix
  scan in moe_4_2_1 (same 5336 instructions): they are macros now; several other dead edits flip the same choice.
- CPU check (S5R_MOD=moe_v2401.py dev/wps5r_cpu.py): the bs17-32 arrays byte-checked, same size as S6O's.

## Run 2 (dev/wps5r_run2.sh, 2026-11-01 ~01:35 local; v2401 vs v1506, two bench passes x 300 reps, pass 2 reversed)
- hang1 bs 1..128 ok for S5R / IMADHI / NO16; check_mt M 1..33 (S5R, IMADHI): 0 failures.
- Bit identity (nondeterminism-aware): judge weights 198 cases: all 80 deterministic cases exact; 134 exact matches in
  all; 5 nondeterministic cases (M 17, 17, 23, 26, 28 = the bs17-32 kernel) where the closest pair (<= 4.9e-4, one BF16
  ulp) exceeded the base's 3-run spread; fallback weights (kernels SASS-identical to v1506): 1 such case (2x spread):
  the 3-run spread criterion is too tight for identical kernels too -> no evidence of a difference; none at M <= 16.
- ratio vs v1506 controls (>1 = faster), pass 1 / pass 2:
  | shape | S5R | S5R b | IMADHI | IMADHI b | NO16 | BOUND | v1506b |
  |---|---|---|---|---|---|---|---|
  | bs8  | 1.0067 / 1.0092 | 1.0081 / 1.0074 | 0.9998 / 0.9994 | 0.9966 / 1.0010 | 1.0099 / 1.0087 | 1.0012 / 1.0090 | 0.9973 / 0.9992 |
  | bs16 | 0.9960 / 0.9970 | 0.9951 / 0.9981 | 0.9960 / 0.9945 | 0.9956 / 0.9953 | 1.0005 / 0.9999 | 1.0060 / 1.0074 | 1.0015 / 0.9992 |
  | bs32 | 0.9853 / 0.9861 | 0.9847 / 0.9848 | 0.9777 / 0.9776 | 0.9792 / 0.9778 | 0.9856 / 0.9862 | 1.0015 / 1.0009 | 1.0003 / 0.9994 |
- Reading: bs8 +0.8% (decode free, bytes ~1:1 or better); bs16 bytes +0.65% but decode -1.0% -> -0.35%; bs32 the bytes
  are worth only +0.1% (its stage is not byte-limited) and the decode costs 1.5% (IMADHI 2.2%) -> rejected.
  IMADHI (fewer ALU, +8 issue) is worse everywhere: the stage loops are issue-limited, not ALU-pipe-limited.
- Candidate: MOE_S5R,MOE_S5R_NO16,MOE_S5R_NO8 (5-bit only in moe_4_1_1_4_1; moe_4_2_1 = the run-2 NO16 kernel
  aef9ae76c3f5, neutral at bs16; moe_8_4_1 = v1506's).

## Run 3 (dev/wps5r_run3.sh, 2026-11-01 ~01:47 local): the candidate v2401 MOE_S5R,MOE_S5R_NO16,MOE_S5R_NO8 vs v1506
- hang1 18 sizes (1..257) ok; check_mt M 1..33, 64, 65, 128, 129, 256, 257 normal + skewed: 0 failures; bench2 verified
  the first reps of every shape (no B2_NOVERIFY): ok.
- Bit identity: all 78 deterministic cases exact; the 8 flagged nondeterministic cases are all at M 17-32, whose kernel is
  SASS-identical to v1506 in this build (closest pair <= 4.9e-4 vs a 3-run spread): base noise; none at M <= 16.
- ratio vs v1506 controls (3 candidate copies, own 5-bit block copies each), pass 1 / pass 2 (reversed order):
  | shape | c | cb | cc | v1506b |
  |---|---|---|---|---|
  | bs8  | 1.0061 / 1.0071 | 1.0067 / 1.0101 | 1.0052 / 1.0072 | 1.0011 / 1.0034 |
  | bs16 | 0.9990 / 1.0009 | 0.9994 / 1.0008 | 0.9993 / 0.9993 | 0.9991 / 1.0009 |
  | bs32 | 0.9997 / 0.9982 | 0.9974 / 0.9981 | 0.9997 / 1.0012 | 0.9984 / 0.9997 |
  placebo pass (150 reps): bs1 1.0021 / 0.9986 / 0.9974 (v1506b 0.9987), bs128 0.9994 / 1.0009 / 1.0008 (0.9983),
  bs256 0.9994 / 0.9999 / 0.9998 (0.9983).

## Result / recommendation
- BANK: dev/moe_v2401.py with MOE_DEFINES=MOE_S5R,MOE_S5R_NO16,MOE_S5R_NO8 (5-bit block scales in moe_4_1_1_4_1 only,
  bs 2-8): bs8 +0.75% (12 copy-passes over 3 jobs, all +0.52..+1.01%; v1504 / v1506 control-vs-control -0.17..+0.34%),
  every other shape neutral -> ~+0.12% geomean (~5157 -> ~5151 us). Lossless (pair registers = S6O's), outputs
  bit-identical on every deterministic case; prepare() falls back to S6O (= v1506 SASS) when any row spans > 31 codes.
- Kernels vs v1506 in that build: moe_4_1_1_4_1 c87f6cfdda76 (changed), moe_4_2_1_4_1 aef9ae76c3f5 (stage loops = v1506's;
  routing prefix scan reordered by ptxas, same count; measured neutral 4 times), all others identical.
- Rejected: S5R at bs16 (-0.35%) and bs32 (-1.5%): the decode (+3..+5 instructions per stage, +2..+3 ALU) costs more
  than the 1.4% fewer bytes buy there (bs32's byte bound is only +0.1%). IMADHI decode worse at every shape.
- Lever exhausted for 5 bits unless a stage decode at <= S6O's 17 ops exists (it does not with a per-row base: see
  "Why the decode needs an add"); a cheaper bs16 decode would be needed to bank the +0.65% byte bound there.
