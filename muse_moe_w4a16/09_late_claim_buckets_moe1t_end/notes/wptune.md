# WP-TUNE: compile-time knob re-tune on the integration base dev/moe_v1502.py (2026-11-01)

Worker WP-TUNE. Base dev/moe_v1502.py (projected ~5215 us). Scripts dev/wptune_runN.sh, logs kda_ws/runs/wptune_runN.log,
bench JSON dev/remote_out/wpt*.json. No new source file was needed (see "per-kernel knobs" below).

## Setup / method
- Knobs via MOE_DEFINES (bench2 --variants third field). Kernels per scored shape: bs8 moe_4_1_1_4_1, bs32 moe_8_4_1_4_1
  (MW1), bs128 moe_8_8_1_3_4 ((8,8,1) MW4 since v1033), bs256 moe_8_8_1_3_8 (MW8).
- Compile-time validity (local NVRTC 13.0, sass13.sh): MOE_KSX8=8 and MOE_KSX1=8 do NOT compile (static_assert "LEAN: items
  of >= NS stages": 2-stage pieces < NS 3 / 4). MOE_E2_KSXNT1=8 compiles (1-stage pieces in the KW 2 kernel).
- MOE_L2_CLD is baked unguarded in v1502 (a prepended define would be overridden): D is swept on dev/moe_v1502u.py, whose
  MOE_L2_CLD=3,MOE_L2_CLDI build is source-identical to v1502 (kernel hashes identical).
- MOE_E2_KSXNT1 is baked unguarded too: swept with MOE_NO_WPE2S + MOE_E2_KSXNT1=k + MOE_E2_KSPL + MOE_E2_SPLF (k = 4 gives
  v1502's SASS in every kernel).
- Per-kernel knobs WITHOUT a source change: every knob here is used only in C++ constant expressions inside moe_body, where
  the template parameters MW / NT / KW are in scope, so a define can be an expression, e.g. MOE_SPLW8=(MW<8?0.3f:0.6f),
  MOE_WGSN_NT=(4+4*(MW<8)), MOE_L2_CLD=(MW<8?2:3), MOE_KSX1=(MW<2?2:4). Checked locally: each such build has, kernel by
  kernel, exactly the SASS hash of the corresponding global-define build (e.g. SPLW8 expression: moe_8_8_1_3_4 = the
  SPLW8=0.3f kernel, moe_8_8_1_3_8 / _16 = v1502's). (No '=' or ',' inside the expression: bench2 / hang1 / check_mt
  replace '=' and split on ','.)
- Which kernels each knob changes (local SASS hashes vs v1502): KSX8 / SPLW8 / WGMIN / WGSN_NT / CLD: the three (8,8,1)
  kernels; KSX1 / SPLW / WGMIN4: moe_8_4_1_4_1 and _2; KSXNT1 and MOE_SPLW=(KW>1?f:1.0f) without KSPL: moe_4_1_1 only.
- A/B: bench2 B2_SHARE_EQ=1 B2_SEEDS=dev/jm_seeds.jseeds --no-base, 100 reps x 2 passes per job, v1502 copies spread over
  the variant list; ratios = median over reps of (mean of the non-first v1502 copies' times) / variant time ("vsC").

## Log
### run 1 (dev/wptune_run1.sh; bs128 + bs256, jm 100 reps x 2 passes, 17 variants, v1502 x5)
hang1 65-512 ok and check_mt 65 100 128 129 200 256 257 (normal + skewed, 3 seeds, 2 calls) 0 failures for all 12 new
define sets. GPU before: H200 345 MHz idle 33 C; after: 1980 MHz, 320 W, 39 C, no throttle reason.

POSITION BIAS (new, matters for every bench2 A/B): bench2 reverses the variant order every rep, so the FIRST and the LAST
position are symmetric, and both read high: v1502 (first) +0.42 / +0.39%, v1502e (last) +0.30 / +0.44% (bs128 / bs256)
vs the middle copies v1502b/c/d. Earlier jobs that put a candidate last (e.g. WP-L2 run 8's "cl") over-read it by ~0.3%.
Controls below = mean of the middle copies b, c, d (positions 3, 7, 11); per-pass ratios, then the mean.

| variant (defines) | bs128 p1 / p2 (mean) | bs256 p1 / p2 (mean) |
|---|---|---|
| v1502b / c / d (controls) | 1.0006 0.9998 1.0001 / 1.0002 0.9998 0.9988 | 0.9983 1.0021 1.0000 / 0.9986 1.0015 0.9995 |
| s03 SPLW8 0.3f (3 split tiles; K4) | 1.0020 / 1.0031 (+0.26%) | 1.0013 / 1.0011 (+0.12%) |
| s045 SPLW8 0.45f (4 tiles) | 1.0015 / 1.0030 (+0.22%) | 1.0004 / 1.0005 (+0.04%) |
| k2s03 KSX8 2, SPLW8 0.3f | 0.9943 / 0.9943 (-0.57%) | 0.9968 / 0.9978 (-0.27%) |
| k2s06 KSX8 2 (SPLW8 0.6f) | 0.9964 / 0.9948 (-0.44%) | 0.9989 / 0.9994 (-0.08%) |
| k2s10 KSX8 2, SPLW8 1.0f | 0.9969 / 0.9957 (-0.37%) | 0.9981 / 0.9971 (-0.24%) |
| k1 KSX8 1 (no tail split) | 0.9890 / 0.9878 (-1.16%) | 0.9930 / 0.9928 (-0.71%) |
| cld2 (v1502u CLD 2 + CLDI) | 1.0007 / 1.0005 (+0.06%) | 0.9991 / 0.9996 (-0.06%) |
| cld4 | 1.0004 / 1.0006 (+0.05%) | 1.0007 / 1.0019 (+0.13%) |
| cld6 | 0.9987 / 0.9983 (-0.15%) | 0.9988 / 0.9989 (-0.11%) |
| wg2 WGMIN 2 | 0.9976 / 0.9974 (-0.25%) | 0.9948 / 0.9944 (-0.54%) |
| wg4 WGMIN 4 | 0.9962 / 0.9975 (-0.31%) | 0.9928 / 0.9920 (-0.76%) |
| wsn8 WGSN_NT 8 (WGSN in (8,8,1)) | 0.9986 / 0.9981 (-0.17%) | 1.0010 / 1.0008 (+0.09%) |

Reading: the split itself is still worth 0.7-1.2% (k1); 2-way pieces lose at both shapes; WGMIN 3, CLD 3 and WGSN off stay
optimal (CLD 2/4/6 within +-0.15%). Only a SMALLER bs128 split (3-4 tiles instead of 5) is above every control in both
passes (+0.2..0.3%, borderline) -> run 2 sweeps SPLW8 for the MW4 kernel alone (1 / 2 / 3 / 4 tiles, s03 twice).

### run 2 (dev/wptune_run2.sh; bs128 / bs32 / bs8, jm 100 reps x 2 passes; v1502 copies: 2 fillers at each end + 3-4 middle controls)
hang1 ok, check_mt 0 failures for all 15 new define sets (bs128 sets M 65 100 128; bs32 sets M 17 20 24 25 32 33 48 64; bs8 sets
M 2-8; normal + skewed, 3 seeds, 2 calls). GPU before 345 MHz idle 32 C, after 1980 MHz 259 W 41 C, no throttle reason.
Position check: the fillers at positions 1 and N-2 read like the middle copies (bs128 +0.06 / +0.04%, bs32 +0.08 / +0.05%);
only the two END positions read high (bs128 +0.32 / +0.32%). bs8 is noisier: middle controls -0.27..+0.48%.

| bs128: SPLW8 for MW4 only (MOE_SPLW8=(MW<8?f:0.6f)) | p1 / p2 vs mean(b, c, d) |
|---|---|
| controls v1502b / c / d | 1.0012 0.9984 1.0011 / 1.0008 0.9991 1.0004 |
| s01 (1 split tile) | 0.9909 / 0.9880 (-1.05%) |
| s02 (2 tiles) | 0.9960 / 0.9954 (-0.43%) |
| s03 (3 tiles), 2 copies | 1.0007 / 1.0006, 1.0011 / 1.0012 (+0.07 / +0.11%) |
| s045 (4 tiles) | 1.0008 / 1.0000 (+0.04%) |
-> run 1's +0.26% for 3 tiles does not replicate (+0.07 / +0.11%, inside the control spread). bs128 split depth: CLOSED
(5 tiles = v1502 stays; 1-2 tiles lose, 7-9 tiles lost in WP-L2 run 8).

| bs32 (moe_8_4_1_4_1) | p1 / p2 vs mean(b, c, d, e) |
|---|---|
| controls v1502b / c / d / e | 1.0003 0.9996 1.0009 1.0019 / 1.0002 0.9994 1.0013 1.0005 |
| s05 SPLW 0.5f (K4) | 1.0020 / 0.9996 (+0.08%) |
| s15 SPLW 1.5f (K4) | 0.9975 / 0.9963 (-0.31%) |
| k2s05 KSX1 2, SPLW 0.5f | 0.9607 / 0.9592 (-4.0%) |
| k2s10 KSX1 2 | 0.9795 / 0.9796 (-2.0%) |
| k2s15 KSX1 2, SPLW 1.5f | 0.9805 / 0.9788 (-2.0%) |
| wm2 WGMIN4 2 | 0.9918 / 0.9910 (-0.86%) |
| wm4 WGMIN4 4 | 0.9859 / 0.9863 (-1.39%) |
-> v1502's bs32 knobs (KSX1 4, SPLW 1.0, WGMIN4 3) are the optimum; 2-way pieces cost 2-4%. bs32: CLOSED.

| bs8 (moe_4_1_1; B2_NOVERIFY, see wps2.md harness note) | p1 / p2 vs mean(b, c, d, e) |
|---|---|
| controls v1502b / c / d / e | 0.9972 0.9972 0.9968 1.0064 / 1.0009 0.9981 0.9978 1.0031 |
| n2 KSXNT1 2 (KSPL) | 0.9897 / 0.9875 (-1.14%) |
| n8 KSXNT1 8 (KSPL; 1-stage pieces) | 1.0015 / 1.0044 (+0.29%) |
| w025 K4, no KSPL, always 2 split tiles | 0.9897 / 0.9872 (-1.16%) |
| w05 K4, no KSPL, always 3 split tiles | 1.0023 / 1.0014 (+0.18%) |
-> n8 and w05 are the only candidates above the control mean (not above every copy: control e reads high in pass 1);
run 3 repeats them with 3 / 2 copies, plus n8 without KSPL (3 tiles), K4 always-4-tiles, MOE_EARLY_NS 1.

### run 3 (dev/wptune_run3.sh; bs8 / bs16 / bs128 / bs256, jm 100 reps x 2 passes, fillers at both ends)
hang1 ok and check_mt 0 failures for all 8 new sets (EARLY_NS 1: M 2 4 6 8 9 12 16; EARLY_NS 3 for (4,2,1): M 9 12 16; n8w05 /
w075: M 2-8; wu0: M 65 100 128; MW8-only SPLW8 / CLD / both: M 129 200 256 257). (The check_mt log labels show "MW>4MW<16":
sed ate the "&&" in the label only; the defines themselves were intact.) GPU before 345 MHz 44 C, after 1980 MHz 254 W 53 C.

| shape: variant | p1 / p2 vs control mean (mean) |
|---|---|
| bs8 controls v1502b / c / d / e | 1.0009 0.9974 0.9971 1.0049 / 1.0028 0.9956 0.9985 1.0056 |
| bs8 n8 KSXNT1 8 (KSPL), 3 copies | 0.9980 / 1.0001, 0.9998 / 0.9987, 0.9946 / 1.0024 (-0.10 / -0.08 / -0.15%) |
| bs8 w05 (K4, always 3 tiles), 2 copies | 1.0014 / 1.0008, 1.0034 / 1.0031 (+0.11 / +0.32%) |
| bs8 w075 (K4, always 4 tiles) | 1.0012 / 1.0029 (+0.20%) |
| bs8 n8w05 (KSXNT1 8, always 3 tiles) | 0.9995 / 1.0008 (+0.01%) |
| bs8 e1 MOE_EARLY_NS 1 | 1.0001 / 1.0026 (+0.13%) |
| bs16 controls b / c / d | 0.9985 0.9991 1.0031 / 1.0013 0.9999 1.0003 |
| bs16 e1 MOE_EARLY_NS 1, 2 copies | 0.9995 / 1.0020, 0.9987 / 1.0003 (+0.08 / -0.05%) |
| bs16 e3 MOE_EARLY_NS 3 ((4,2,1) only) | 0.9934 / 0.9941 (-0.62%) |
| bs128 controls b / c | 0.9995 1.0005 / 0.9995 1.0005 |
| bs128 wu0 no warm-up stage in the MW4 kernel, 2 copies | 1.0013 / 1.0007, 1.0023 / 1.0026 (+0.10 / +0.24%) |
| bs256 controls b / c / d | 1.0012 0.9997 0.9989 / 1.0008 0.9993 1.0010 |
| bs256 s3m8 SPLW8 0.3f for MW8 only, 2 copies | 0.9991 / 0.9993, 1.0006 / 0.9994 (-0.08 / 0.00%) |
| bs256 c4m8 CLD 4 for MW8 only, 2 copies | 1.0002 / 1.0005, 1.0000 / 1.0000 (+0.04 / 0.00%) |
| bs256 s3c4m8 both | 0.9988 / 0.9995 (-0.08%) |
-> n8 (run 2 +0.29%) does not replicate: rejected. bs256 split depth / CLD: CLOSED (run 1's +0.12 / +0.13% were noise).
EARLY_NS: 2 stays (1 neutral at bs8 / bs16, 3 loses). bs8 always-split (w05 / w075) and bs128 wu0 sit at +0.1..0.3% in
every copy: run 4 repeats them with 150 reps.

### run 4 (dev/wptune_run4.sh; 150 reps x 2 passes)
hang1 ok (wu8 129 / 256 / 257); check_mt wu8 M 129 200 256 257 and wu0 M 33 64 65 100 128 129: 0 failures. GPU before 345 MHz
35 C, after 1980 MHz 269 W 40 C, no throttle reason.

| shape: variant | p1 / p2 vs control mean (mean) |
|---|---|
| bs128 controls v1502b / c / d | 0.9995 0.9997 1.0011 / 0.9993 1.0006 1.0003 |
| bs128 wu0 (no warm-up, MW4 kernel), 3 copies | 1.0018 / 1.0016, 1.0023 / 1.0026, 1.0034 / 1.0022 (+0.17 / +0.24 / +0.28%) |
| bs256 controls b / c | 0.9993 1.0007 / 0.9995 1.0005 |
| bs256 wu8 (no warm-up, MW8 kernel only) | 0.9948 / 0.9961 (-0.45%) |
| bs8 controls b / c / d / e | 1.0042 0.9939 1.0003 1.0017 / 1.0003 0.9973 1.0013 1.0002 |
| bs8 w05 (always 3 tiles), 2 copies | 1.0033 / 1.0011, 1.0022 / 1.0037 (+0.22 / +0.30%) |
| bs8 w075 (always 4 tiles), 2 copies | 1.0051 / 1.0051, 1.0055 / 1.0091 (+0.51 / +0.73%) |
-> wu0: in runs 3 + 4, all 5 copies beat every middle control in every pass, but only by +0.10..0.28% (mean +0.21%). It is
robust but below the 0.3% bar. wu8 rejected (the bs256 warm-up stays, as WP-K found on v655). bs8 w075: +0.5 / +0.7% here, above
every control in both passes, but run 3's single copy read +0.20%. Run 5 repeats it x3 with 4 / 5 / 6 / 3-tile neighbours.

### run 5 (dev/wptune_run5.sh; 150 reps x 2 passes)
hang1 ok (w10 / w125, bs 2 4 8); check_mt w075 / w10 / w125 M 2-8 0 failures. GPU before 345 MHz 37 C, after 1980 MHz 329 W 44 C.

| shape: variant | p1 / p2 vs control mean (mean) |
|---|---|
| bs8 controls v1502b / c / d / e / f | 1.0013 1.0004 1.0002 0.9975 0.9998 / 1.0031 0.9953 1.0024 0.9980 0.9999 |
| bs8 w075 (always 4 split tiles), 3 copies | 1.0014 / 1.0065, 1.0049 / 1.0028, 1.0052 / 1.0058 (+0.40 / +0.38 / +0.55%) |
| bs8 w10 (5 tiles), 2 copies | 1.0065 / 1.0018, 1.0020 / 1.0022 (+0.41 / +0.21%) |
| bs8 w125 (6 tiles) | 0.9974 / 0.9987 (-0.20%) |
| bs8 w05 (3 tiles) | 1.0067 / 1.0026 (+0.47%) |
| bs128 controls b / c | 1.0002 0.9998 / 0.9994 1.0006 |
| bs128 wu0, 2 copies | 1.0024 / 1.0032, 1.0017 / 1.0006 (+0.28 / +0.11%) |

bs8 split rule over runs 2-5 (per copy, vs that job's middle-control mean):
- w075 (4 tiles): +0.20 | +0.51 +0.73 | +0.40 +0.38 +0.55 -> mean +0.46% (6 copies, 3 jobs). In runs 4 and 5 the copies beat
  every middle control in 10 of 12 copy-passes; the two misses were by 0.0002 / 0.0003.
- w05 (3 tiles): +0.18 | +0.11 +0.32 | +0.22 +0.30 | +0.47 -> mean +0.27%. w10 (5 tiles): +0.41 +0.21. w025 (2) -1.16%, w125 (6) -0.20%.
- v1502's KSPL rule (split only a short last round) is therefore ~0.45% slower at bs8 than always splitting 4 of the ~9 tiles.
  The w075 kernel is also leaner (moe_4_1_1 6080 vs 6120 instructions: the KSPL rule's code is gone).
bs128 wu0 over runs 3-5: +0.10 +0.24 | +0.17 +0.24 +0.28 | +0.28 +0.11 -> mean +0.20% (7 copies). Above every middle
control in 13 of 14 copy-passes (one tie). Robust, but below the 0.3% bar.

### run 6 (dev/wptune_run6.sh): validation of the accepted set on all six shapes + nearby shapes
comb = W075 + WU0 (defines below; local NVRTC 13: only moe_4_1_1_4_1 (= the w075 kernel, 7771cd686d91) and moe_8_8_1_3_4
(= the wu0 kernel, aa48e2af06f1) differ from v1502; every other kernel hash identical). hang1 15 sizes (1-257) ok; check_mt
M 1 2 3 4 5 6 7 8 9 12 16 17 32 33 64 65 100 128 129 200 256 257 (normal + skewed, 3 seeds, 2 calls) 0 failures.
GPU before 345 MHz 42 C, after 1980 MHz 134 W 49 C, no throttle reason.

Nearby shapes (same (4,1,1) kernel; default seeds, 100 reps x 2 passes, vs middle controls), w075 x3:
bs2 +2.87 / +2.19 / +1.64%, bs4 +2.37 / +1.47 / +1.67%, bs6 +1.33 / +0.94 / +0.89% (controls -0.6..+0.84 at bs2, -0.2..+0.1
at bs4 / bs6). Medians: bs2 34.0-34.4 us vs 39.2-39.8, bs4 44.5-45.3 vs 45.5-46.4, bs6 56.0-56.5 vs 56.7-57.1. Always splitting
4 tiles helps the small calls more (at bs2-4 the KSPL rule rarely splits: every tile's last round is short).

Six scored shapes (jm 80 reps x 2 passes; middle controls b / c / d), comb x2:
| | bs1 | bs8 | bs16 | bs32 | bs128 | bs256 |
|---|---|---|---|---|---|---|
| comb (vsC, p1 / p2) | 1.0103 / 1.0161 * | 1.0087 / 1.0051 | 1.0021 / 1.0013 * | 1.0019 / 1.0012 * | 1.0022 / 1.0015 | 0.9989 / 0.9985 * |
| combb | 1.0068 / 1.0025 * | 1.0077 / 1.0115 | 1.0008 / 1.0005 * | 1.0005 / 0.9984 * | 1.0026 / 1.0020 | 0.9998 / 0.9990 * |
| controls b / c / d range | 0.9996..1.0067 | 0.9973..1.0025 | 0.9979..1.0021 | 0.9980..1.0019 | 0.9987..1.0004 | 0.9987..1.0006 |
(* = kernel identical to v1502's: placebo; bs1 copies spread +-0.5%.) bs8 is now a third job where w075 beats every control
(+0.69 / +0.96%). bs128 wu0 +0.19 / +0.23%.
project2 (anchor v969 in position 2): comb 5204.0 / 5211.1, combb 5219.8 / 5217.5 (mean 5213.1) vs middle v1502 controls
5224.3 / 5229.5 / 5217.3 / 5219.4 / 5222.0 / 5215.7 (mean 5221.4). That is ~-8 us, with the identical-kernel shapes'
noise (+-10 us per copy) dominating. Expected from the per-shape gains: (bs8 +0.5% + bs128 +0.2%) / 6 = ~+0.12% geomean,
~-6 us.

## Summary
ACCEPTED (defines on dev/moe_v1502.py; no source change; all kernels other than the two named are SASS-identical to v1502):
1. bs8 (and bs2-7): `MOE_NO_WPE2S,MOE_E2_SPLF,MOE_E2_KSXNT1=4,MOE_SPLW=(KW>1?0.75f:1.0f)`. It keeps v801's 4-way k split and
   the SPLF combine but drops the MOE_E2_KSPL rule (split only a short last round). Instead, every bs 2-8 call splits the down
   items of its last ceil(0.75 x 132 / 32) = 4 tiles. Only moe_4_1_1_4_1 changes (6120 -> 6080 instructions). bs8: +0.20 |
   +0.51 +0.73 | +0.40 +0.38 +0.55 | +0.69 +0.96% over 8 copies in 4 jobs (mean +0.55%); above every middle control in 3 jobs.
   Nearby bs2 / 4 / 6: +1.6..2.9 / +1.5..2.4 / +0.9..1.3%. Neighbours: 3 tiles +0.27% (6 copies), 5 tiles +0.31% (2 copies),
   2 tiles -1.16%, 6 tiles -0.20%; KSXNT1 8 (1-stage pieces) with KSPL ~0 (-0.1 x3 after +0.29); KSXNT1 2 -1.1%.
   Bake: in v1502's `#ifndef MOE_NO_WPE2S` block, remove `#define MOE_E2_KSPL` and add `#define MOE_SPLW (KW > 1 ? 0.75f : 1.0f)`
   (MOE_SPLW is only read inside moe_body, where KW is in scope; the (8,4,1) kernels keep 1.0f).
2. bs128 (small, below the 0.3% bar, robust): `MOE_GATHER_MIN0=(MW<8?9:5)`. This turns off the instruction-cache warm-up stage
   in the MW4 (8,8,1) kernel only (bs65-128). Only moe_8_8_1_3_4 changes. +0.10 +0.24 | +0.17 +0.24 +0.28 | +0.28 +0.11 |
   +0.19 +0.23% over 9 copies in 4 jobs (mean +0.20%). Above every middle control in 17 of 18 copy-passes (one tie). The same
   switch at bs256 (MW8) loses 0.45%. Bake: `#define MOE_GATHER_MIN0 (MW < 8 ? 9 : 5)` in place of `5` (read only in moe_body's
   `WARM = NT >= MOE_GATHER_MIN0`).
Combined: `MOE_NO_WPE2S,MOE_E2_SPLF,MOE_E2_KSXNT1=4,MOE_SPLW=(KW>1?0.75f:1.0f),MOE_GATHER_MIN0=(MW<8?9:5)`. Projected
~+0.12% geomean (~-6 us; run 6 measured ~-8 us). Correctness: see run 6, plus run 7 (interleaved calls).

REJECTED / at optimum already (v1502's values stay):
- bs128 / bs256 (8,8,1): KSX8 2 (-0.1..-0.6%, every SPLW8), KSX8 1 = no tail split (-0.7..-1.2%), KSX8 8 does not compile (LEAN
  static_assert); SPLW8 0.3 / 0.45 (+0.0..0.26%, not replicated), MW4-only SPLW8 1 / 2 / 3 / 4 tiles (-1.05 / -0.43 / +0.07..0.11
  / +0.04%), MW8-only SPLW8 0.3 (~0); CLD 2 / 4 / 6 (-0.17..+0.13%), MW8-only CLD 4 (~0), CLD4 + SPLW8 0.3 at MW8 (-0.08%);
  WGMIN 2 / 4 (-0.25..-0.76%); WGSN_NT 8 = WGSN in (8,8,1) (bs128 -0.17%, bs256 +0.09%); no warm-up at MW8 (-0.45%).
- bs32 (8,4,1) MW1: SPLW 0.5 (+0.08%), 1.5 (-0.31%); KSX1 2 (-2..-4%); KSX1 8 does not compile; WGMIN4 2 / 4 (-0.86 / -1.39%).
- bs16: MOE_EARLY_NS 1 (~0), 3 for (4,2,1) (-0.62%). (bs16 split rules: closed by WP-S2, not redone.)
- bs8: MOE_EARLY_NS 1 (+0.13%, 1 copy: noise).

Method note for every worker (POSITION BIAS): bench2 reverses the variant order every rep, so the first and last variants
share the "first replay after the input copy" slot, and BOTH read +0.3..0.5% high (8 jobs here: first +0.12..0.54%, last
+0.12..0.55%; positions 1 and N-2 read like the middle). A candidate placed last is over-read by about as much as the
first-position control. Put filler copies at both ends and use only middle copies as controls. (Examples: WP-L2 run 8's last
"cl" variant; plan_v1502.sh's last copy v1502b.)

### run 7 (dev/wptune_run7.sh): extra correctness of the accepted set
hang1 comb bs 2 4 8 128 ok. dev/wps2_mix.py (200 interleaved bs 1-33 calls in random order, normal + skewed, each checked):
comb 0 failures of 200 calls (worst rel-L2 3.317e-3), v1502 0 of 200 (worst 3.317e-3, identical). check_mt comb M 2-8 65 80 100
120 128: 0 failures. GPU before 345 MHz 29 C, after 1980 MHz 265 W 34 C.

Files: dev/wptune_run1..7.sh, logs kda_ws/runs/wptune_run1..7.log, bench JSON dev/remote_out/wpt{1..6}_*.json. Local SASS
checks used the session's NVRTC 13 wrapper (scratchpad wpend/sass13.sh + wptune/hdiff.sh). No source file was created or
edited. No okc check / submit.
