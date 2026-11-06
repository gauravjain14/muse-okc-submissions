# WP-TUNE2: knob re-tune on the current build dev/moe_v1510.py (2026-11-01)

Worker WP-TUNE2. Base dev/moe_v1510.py (projected ~5119 us; scored v1504 5157.2, rank 10). Scripts dev/wptune2_runN.sh,
logs kda_ws/runs/wptune2_runN.log, bench JSON dev/remote_out/wpt2_r*.json. Tables: dev/wpit2_tab.py <log> '^v1510[b-e]$'.

## Setup
- dev/moe_v2800.py (made by dev/wptune2_mk2800.py) = v1510 with `#ifndef` guards around the four knobs v1510 bakes
  unguarded at the top of CUDA_SRC: MOE_LATE9 (1500), MOE_LATE9_CFG (1), MOE_SPLW ((KW>1?0.75f:1.0f)), MOE_L2_CLD (3).
  A prepended MOE_DEFINES value now overrides them. Default build: all 16 kernels SASS-identical to v1510
  (local NVRTC 13, MOE_S5R_FIT, scratchpad wpend/sass13.sh).
- MOE_S2_SPLW2 / MOE_E2_KSXNT2 / MOE_SPLW8 are already overridable in v1510 (guarded / not baked).
- Per-kernel expressions (no '=' or ',', the harness replaces '=' and splits on ','): bs8 split only:
  MOE_SPLW=(KW>1?(NT<2?f:0.75f):1.0f) (NT==1 would be mangled to "NT  1"); MW4 only: (MW<8?a:b); MW8 only:
  (MW>4?(MW<16?a:b):b).
- Kernels changed (local SASS hash vs v1510; every set: same instruction count unless noted, 0 new spills):
  | define set | kernel changed |
  |---|---|
  | MOE_S2_SPLW2 0.5 / 1.0 / 1.25 / 1.5 (0.75 = v1510: identical) | moe_4_2_1_4_1 only (6104 instr, immediates) |
  | MOE_E2_KSXNT2=4 (+ SPLW2 0.25 / 0.5 / 1.0) | moe_4_2_1_4_1 only (6176 / 6168 instr: 4-way pieces) |
  | MOE_LATE9_CFG=3 | moe_4_2_1_4_1 only (6112 instr: the spin) |
  | MOE_LATE9 1000 / 1250 / 1750 / 2000 / 2500 | moe_4_1_1_4_1 only |
  | MOE_SPLW=(KW>1?(NT<2?0.5f / 1.0f:0.75f):1.0f) | moe_4_1_1_4_1 only |
  | MOE_SPLW8 MW4-only 0.45 / 0.7, MOE_L2_CLD MW4-only 2 / 4 | moe_8_8_1_3_4 only |
  | MOE_SPLW8 MW8-only 0.45 / 0.7, MOE_L2_CLD MW8-only 2 / 4 | moe_8_8_1_3_8 only |
- Split amounts in tiles: nspl = ceil(f x 132 / (NRB2 x KS2)); (4,*,1): NRB2 x KS2 = 32 -> f 0.25 / 0.5 / 0.75 / 1.0 /
  1.25 / 1.5 = 2 / 3 / 4 / 5 / 6 / 7 tiles (bs16 has ~12.3 tiles of 16 tokens per call); (8,8,1): 16 -> SPLW8 0.45 /
  0.6 / 0.7 = 4 / 5 / 6 tiles.
- A/B: bench2 B2_SHARE_EQ=1 B2_NOVERIFY=1 B2_SEEDS=dev/jm_seeds.jseeds --no-base, 100 reps x 2 passes, two v1510 fillers
  at each end, 4 middle v1510 controls (b-e) + a v2800 default placebo (p2800); % = mean of the two passes' "vs first"
  ratios vs the middle-control mean; * = above every middle control. Correctness separately (check_mt, normal + skewed,
  3 seeds, 2 calls) because KRS variants share state["ykz"] under B2_SHARE_EQ.

## Log
### run 1 (dev/wptune2_run1.sh; bs16 split amount + LATE9_CFG, bs8 LATE9 / split amount; 100 reps x 2 passes)
hang1 v2800 default 11 sizes (1-256) ok; every define set: hang1 ok + check_mt (bs16 sets M 9-17, bs8 sets M 2-9; normal +
skewed, 3 seeds, 2 calls) 0 failures. GPU before 345 MHz idle 31 C; benches at 1980 MHz, 253-307 W, 32-35 C, no throttle.
Controls = v1510b-e + p2800 (v2800 default, SASS = v1510).

| bs16 (moe_4_2_1) variant | p1 / p2 | % vs controls |
|---|---|---|
| controls v1510b / c / d / e, p2800 | 0.9997 0.9998 1.0004 0.9982, 0.9982 (mean 0.9993) | spread -0.13..+0.15 |
| K2 SPLW2 0.5 (3 tiles) | 0.9957 / 0.9955 | -0.37 |
| **K2 0.75 (4 tiles) = v1510** | | 0 |
| K2 SPLW2 1.0 (5 tiles) | 1.0010 / 0.9992 | +0.08 |
| K2 SPLW2 1.25 (6 tiles) | 0.9938 / 0.9958 | -0.45 |
| K2 SPLW2 1.5 (7 tiles) | 0.9959 / 0.9953 | -0.37 |
| K4 SPLW2 0.25 (2 tiles) | 0.9920 / 0.9898 | -0.84 |
| K4 SPLW2 0.5 (3 tiles) | 0.9990 / 0.9981 | -0.07 |
| K4 0.75 (4 tiles) | 0.9954 / 0.9952 | -0.40 |
| K4 SPLW2 1.0 (5 tiles) | 0.9920 / 0.9948 | -0.59 |
| LATE9_CFG=3 (LATE9 1500 also in (4,2,1)) | 1.0004 / 1.0010 | +0.14 * |

| bs8 (moe_4_1_1) variant | p1 / p2 | % vs controls |
|---|---|---|
| controls v1510b / c / d / e, p2800 | 0.9983 0.9972 0.9974 0.9990, 0.9970 (mean 0.9978) | spread -0.10..+0.16 |
| LATE9 1000, 2 copies | 1.0025 / 0.9967, 0.9982 / 0.9954 | +0.18 *, -0.10 |
| LATE9 2000, 2 copies | 0.9970 / 0.9982, 0.9996 / 0.9995 | -0.02, +0.18 * |
| LATE9 2500 | 0.9972 / 0.9993 | +0.05 |
| SPLW (NT 1) 0.5 (3 tiles), 2 copies | 0.9983 / 0.9972, 0.9970 / 0.9959 | 0.00, -0.13 |
| SPLW (NT 1) 1.0 (5 tiles), 2 copies | 0.9943 / 0.9919, 0.9934 / 0.9950 | -0.47, -0.36 |
-> bs16: v1510's K2 + 0.75 waves is the optimum of the 9-point grid (K2 1.0 ties: +0.08, as WP-CGS2 found on v1508);
every other amount loses 0.07-0.84%. LATE9 in the (4,2,1) kernel: +0.14% (above every control by 0.03%), below the
0.3% bar; one test only (WP-COLD's v1504 bs16 result did not replicate either) -> rejected. bs8: LATE9 1000 / 2000 / 2500
within noise of 1500 (copies -0.10..+0.18); the 4-tile split (W075) stays (3 tiles 0 / -0.13, 5 tiles -0.4).

### run 2 (dev/wptune2_run2.sh + dev/wptune2_c4.sh; candidate 4 bs128 / bs256, bs32 split amount, bs8 LATE9 curve; 150 reps x 2)
Every new set: hang1 ok, check_mt 0 failures (bs32 sets M 17 20 24 25 32 33; bs8 M 2-9; MW4 sets M 65 100 128; MW8 sets
M 129 200 256 257). GPU 1980 MHz, 318-405 W, 44-48 C, no throttle reason.

| shape: variant | p1 / p2 | % vs controls |
|---|---|---|
| bs128 controls v1510b-e, p2800 | 0.9974 0.9962 0.9966 0.9973, 0.9971 | -0.09..+0.07 |
| bs128 SPLW8 MW4 0.45 (4 tiles), 2 copies | 0.9974 / 0.9976, 0.9983 / 0.9982 | +0.06 *, +0.13 * |
| bs128 SPLW8 MW4 0.7 (6 tiles), 2 copies | 0.9948 / 0.9954, 0.9961 / 0.9976 | -0.18, -0.01 |
| bs128 CLD MW4 2, 2 copies | 0.9963 / 0.9967, 0.9978 / 0.9977 | -0.04, +0.08 * |
| bs128 CLD MW4 4, 2 copies | 0.9978 / 0.9965, 0.9970 / 0.9963 | +0.02, -0.03 |
| bs256 controls v1510b-e, p2800 | 0.9937 0.9961 0.9957 0.9970, 0.9952 | -0.24..+0.19 |
| bs256 SPLW8 MW8 0.45, 2 copies | 0.9961 / 0.9966, 0.9954 / 0.9958 | +0.08, +0.01 |
| bs256 SPLW8 MW8 0.7, 2 copies | 0.9950 / 0.9953, 0.9954 / 0.9946 | -0.04, -0.06 |
| bs256 CLD MW8 2, 2 copies | 0.9973 / 0.9963, 0.9960 / 0.9958 | +0.13, +0.04 |
| bs256 CLD MW8 4, 2 copies | 0.9964 / 0.9942, 0.9960 / 0.9948 | -0.03, -0.02 |
| bs32 controls v1510b-e, p2800 | 0.9973 0.9969 0.9995 0.9963, 0.9975 | -0.15..+0.25 |
| bs32 SPLW (8,4,1) MW1 0.75 (7 tiles), 2 copies | 1.0004 / 1.0001, 1.0001 / 0.9987 | +0.27 *, +0.19 |
| bs32 SPLW 1.25 (11 tiles), 2 copies | 0.9956 / 0.9949, 0.9979 / 0.9970 | -0.23, -0.01 |
| bs8 controls v1510b-e, p2800 | 1.0009 1.0003 0.9988 1.0002, 0.9996 | -0.15..+0.12 |
| bs8 LATE9 off (MOE_LATE9_CFG=0 = v1508's S5R kernel), 2 copies | 0.9920 / 0.9932, 0.9934 / 0.9919 | -0.73, -0.73 |
| bs8 LATE9 1000 / 1250 / 1750 / 2000 | 0.9979 / 0.9949, 0.9967 / 0.9978, 0.9980 / 0.9974, 0.9984 / 0.9972 | -0.35 / -0.27 / -0.22 / -0.21 |
-> candidate 4 CLOSED: bs128 / bs256 SPLW8 (4 / 6 tiles) and CLD (2 / 4) per kernel all within +-0.18% (inside the
control spread); v1510's 5 tiles / CLD 3 stay. bs8: LATE9 itself is worth +0.73% on v1510 (both copies), 1500 ns is the
optimum of {1000, 1250, 1500, 1750, 2000, 2500} over runs 1-2 -> stays. bs32: 7 split tiles (0.75 waves) +0.19..+0.27%
-> run 3 replicates it with the 5 / 6 / 8-tile neighbours and nearby shapes.

### run 3 (dev/wptune2_run3.sh; bs32 split-amount curve + nearby shapes 20 / 24; 200 reps x 2 passes)
New sets (SPLW for the (8,4,1) MW1 kernel 0.5 / 0.625 / 0.875): hang1 ok, check_mt M 17 20 24 25 28 32 33 0 failures.
GPU 1980 MHz, 321-358 W, 33-36 C.

| variant (tiles of ~16.5) | bs32 (jm) | bs24 | bs20 |
|---|---|---|---|
| controls v1510b-e, p2800 (spread) | -0.16..+0.12 | -0.13..+0.14 | -0.13..+0.07 |
| 0.5 (5) | -0.54 | -0.94 | -1.15 |
| 0.625 (6) | -0.08 | -0.37 | -0.33 |
| 0.75 (7), 4 copies | +0.12 * / +0.12 * / +0.11 * / +0.15 * | -0.05 / -0.23 / -0.13 / -0.14 | -0.13 / -0.13 / -0.07 / -0.05 |
| 0.875 (8) | +0.07 | +0.07 | -0.15 |
| **1.0 (9) = v1510** | 0 | 0 | 0 |
-> bs32 7 tiles: 6 copies over runs 2-3 +0.27 +0.19 | +0.12 +0.12 +0.11 +0.15 (mean +0.16%), all but one above every
middle control, but below the 0.3% bar and slightly negative on the nearby shapes bs24 / bs20 (-0.05..-0.23%) -> NOT
accepted; v1510's 1.0 (9 tiles) stays. 5 / 6 tiles lose at every size.

### run 4 (dev/wptune2_run4.sh; early-start depth MOE_EARLY_NS after S5R + LATE9; 150 reps x 2 passes)
MOE_EARLY_NS is read in moe_body: bs8-only MOE_EARLY_NS=(RW>2?(NT<2?k:2):2) (plain NT<2 also changes the unused
moe_2_1_1_3_1), bs16-only (NT<2?2:1). Only moe_4_1_1_4_1 (5968 / 6184 / 6256 instr for 1 / 3 / 4) resp. moe_4_2_1_4_1
(5960) change. hang1 ok, check_mt (bs8 M 2-9, bs16 M 9-17) 0 failures.

| shape: variant | p1 / p2 | % vs controls |
|---|---|---|
| bs8 controls v1510b-e, p2800 | 0.9983 0.9963 0.9979 0.9983, 0.9964 | -0.14..+0.11 |
| bs8 EARLY_NS 1, 2 copies | 0.9936 / 0.9976, 0.9991 / 0.9979 | -0.18, +0.11 * |
| bs8 EARLY_NS 3, 2 copies | 0.9878 / 0.9876, 0.9856 / 0.9854 | -0.98, -1.20 |
| bs8 EARLY_NS 4 (whole ring), 2 copies | 0.9691 / 0.9721, 0.9715 / 0.9723 | -2.69, -2.56 |
| bs16 controls v1510b-e, p2800 | 0.9987 0.9980 0.9972 0.9965, 0.9969 | -0.12..+0.16 |
| bs16 EARLY_NS 1, 2 copies | 0.9992 / 0.9991, 0.9973 / 0.9990 | +0.17 *, +0.07 |
-> EARLY_NS 2 stays at both shapes (1 = noise, deeper early starts flood HBM before the routing and lose 1-2.7%).

## Summary
NOTHING ACCEPTED: every knob re-tuned on v1510 sits at v1510's value; no source or define change to bank.
dev/moe_v2800.py (v1510 + #ifndef guards; default SASS = v1510, placebo p2800 within the control spread in all 9
sections) stays available for later sweeps of LATE9 / LATE9_CFG / SPLW / L2_CLD.

| knob (kernel) | values measured (% vs middle v1510 controls; copies) | verdict |
|---|---|---|
| MOE_S2_SPLW2 with KSXNT2 2 (bs16) | 0.5 -0.37; **0.75 = v1510**; 1.0 +0.08; 1.25 -0.45; 1.5 -0.37 | v1510 |
| MOE_S2_SPLW2 with KSXNT2 4 (bs16) | 0.25 -0.84; 0.5 -0.07; 0.75 -0.40; 1.0 -0.59 | v1510 (K2) |
| MOE_LATE9_CFG=3 (LATE9 also in bs16's (4,2,1)) | +0.14 * (1 copy, once only by brief) | rejected (< 0.3%) |
| MOE_LATE9 (bs8) | off -0.73 / -0.73; 1000 +0.18 -0.10 -0.35; 1250 -0.27; **1500 = v1510**; 1750 -0.22; 2000 -0.02 +0.18 -0.21; 2500 +0.05 | v1510 (LATE9 itself worth +0.7% on v1510) |
| MOE_SPLW NT 1 (bs8 split tiles) | 0.5 (3) 0.00 -0.13; **0.75 (4) = v1510**; 1.0 (5) -0.47 -0.36 | v1510 |
| MOE_SPLW (8,4,1) MW1 (bs32) | 0.5 (5) -0.54; 0.625 (6) -0.08; 0.75 (7) +0.27 +0.19 +0.12 +0.12 +0.11 +0.15 (mean +0.16; bs24 -0.05..-0.23, bs20 -0.05..-0.13); 0.875 (8) +0.07; **1.0 (9) = v1510**; 1.25 (11) -0.23 -0.01 | v1510 (7 tiles below bar, negative nearby) |
| MOE_SPLW8 MW4 (bs128) | 0.45 (4 tiles) +0.06 +0.13; **0.6 (5) = v1510**; 0.7 (6) -0.18 -0.01 | v1510 |
| MOE_L2_CLD MW4 (bs128) | 2: -0.04 +0.08; **3 = v1510**; 4: +0.02 -0.03 | v1510 |
| MOE_SPLW8 MW8 (bs256) | 0.45 +0.08 +0.01; **0.6 = v1510**; 0.7 -0.04 -0.06 | v1510 |
| MOE_L2_CLD MW8 (bs256) | 2: +0.13 +0.04; **3 = v1510**; 4: -0.03 -0.02 | v1510 |
| MOE_EARLY_NS (bs8) | 1 -0.18 +0.11; **2 = v1510**; 3 -0.98 -1.20; 4 -2.69 -2.56 | v1510 |
| MOE_EARLY_NS (bs16) | 1 +0.17 +0.07; **2 = v1510** | v1510 |

Correctness: every define set run (34 sets + the v2800 default build) passed hang1 and check_mt (normal + skewed, 3 seeds, 2 calls; M ranges of its
kernel) with 0 failures. No okc check / submit. Files: dev/wptune2_mk2800.py, dev/moe_v2800.py, dev/wptune2_run1-4.sh,
dev/wptune2_c4.sh, logs kda_ws/runs/wptune2_run1-4.log, bench JSON dev/remote_out/wpt2_r{1..4}_*.json and
wpt2_r2_s{128,256}_p*.json (c4 sections), SASS hash lists in the session scratchpad wptune2/hashes_run*.txt.
