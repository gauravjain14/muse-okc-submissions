# WP-BCPS: block-column post-scale (BCPS) for T <= 2 tiles, base dev/moe_v1002.py

Worker WP-BCPS, 2026-10-31/11-01. Namespace v1700-v1799, dev/wpbcps_*.

**Bottom line.**
- The BCPS stage is real: -10..-14% stage cycles compute-only, -8.6% in a consumer-bound ring, -10% stage compute
  inside moe1t; rel-L2 6.7e-8 vs 8.2e-8 with x * 2^48.
- It converts nowhere in the bench build. bs1 MOE1T_BCPS=1 is x0.987..1.008 over four passes (mean -0.5%), BCPS=2 x0.90,
  BCPS + FXSS does not stack (0.979 / 1.002 vs FXSS). bs8 / bs16 is NO-GO: v1050's x0.95 is code presence (the
  never-taken control loses the same), and executing BPC adds only +0.1..0.4%. T = 4 loses (+11..14% stage).
- At bs1 the call ends at the slowest CTA's data stream (a stable per-TPC DRAM-share deficit plus noise) plus the end
  chain. Faster consumers in the median CTAs cannot move it (section 6c, for WP-B1c).
- Nothing banked. dev/moe_v1700.py / v1701.py default builds = v1002 (SASS-identical).

## 0. Prior work (found before any change): this idea was built and measured as WP-RES "BPC"

kda_ws/docs/findings/wpres_decode.md (#1, "block-per-column post-scaling"), dev/wpres_stage.py, dev/wpres_occ.py,
dev/moe_v1050.py (MOE_BPC, built from v1033 by dev/wpres_mk1050.py):
- gate 1, (4,*,1) stage, compute only, H200: T = 1 -11.4% (S6O words), -12.7% (8-bit); T = 2 -5.7%; + one rotation on
  IMAD.WIDE: worse.
- gate 2, streaming 132 CTAs, cold HBM: P1 x1.009 / x1.003, P2 x0.997 / x1.000 (the steady 1-group stage is within 2% of
  the ring floor).
- gate 3, full bs8 kernel (moe_4_1_1_4_1, items with one token on the BPC stage): correct (check_mt 0 failures) but
  **x0.957** vs v1033 (dev/remote_out/sprint_bpc_full.json, 48 jm reps; sprint_state "BPC_full_kernel_v1050").
- never tried: the moe1t (bs1) kernel. Per WP-B1c (wpb1c.md section 2) moe1t's phase B (down rows) is consumer-bound
  (stage intervals 1.22-1.31 / 0.99 us vs phase A 1.03-1.05 data-bound), so this package targets moe1t phase B first
  (planner, 2026-11-01).

## 1. Numerics (local sm_89 + H200): x must be pre-scaled

The unscaled codes are BF16 E2M1 * 2^-126, so a block partial D = sum_16 v x * 2^-126 sits at the FP32 subnormal edge.
With unscaled x (dev/wpbcps_stage.py, local RTX 4060 Ti, random NVFP4 expert, moe1t layout, 64 items x 16 rows):

| input | x unscaled | x * 2^32 | x * 2^48 |
|---|---|---|---|
| token row (rms 1.0) | 7.1e-8 | 7.1e-8 | 7.1e-8 |
| intermediate (rms 0.6) | 8.0e-8 | 8.0e-8 | 8.0e-8 |
| token row * 2^-20 | **2.2e-2** (over budget) | 7.1e-8 | 7.1e-8 |
| token row * 2^-40 | - | 2.5e-7 | 7.1e-8 |
| token row * 2^-55 | 1.0 (all zero) at 2^-60 | 1.7e-1 | 1.3e-7 |
| token row * 2^30 / 2^60 | | 7.1e-8 / 7.1e-8 | 7.1e-8 / 7.1e-8 |
| intermediate * 2^-40 | 1.0 | 1.2e-6 | 8.0e-8 |

So the B fragments hold x * 2^48 (exact: a power of two; valid for |x| in about [2^-60, 2^78]); the FP32 scale is
s * 2^104 (scale byte's 7 bits at [26:20] | 0x70000000), so the stage output is E2M1 * s * x * 2^(48 - 22) = 2^26 times
the product (the epilogue's factor DQF = 2^-26 instead of 2^-8's 256).

## 2. Files

- `dev/wpbcps_stage.py`: moe1t-layout stage harness (base v1002): V0 = v1002 stage_two + the per-stage partial store;
  V1-V7 BCPS T = 1 variants (reduction form, 1 / 2 MMA chains per group, byte-3 scale by IMAD.HI, one rotation on
  mul.wide); modes bench (SMEM-resident, no ring), ring (producer + 4-slot ring + reducer warp; L2-resident source =
  consumer-bound, HBM source = data-bound), error vs FP64 (moe1t's own repack of the judge's weights, token row and
  reference intermediate, plus 2^-55..2^60 stress inputs). `WPB_ARCH=sm_89 WPB_RANDW=5252` runs the BCPS correctness
  locally; `WPB_NOLOAD=1 ... sass` prints the loop SASS of an sm_90a build without a GPU.
- `dev/wpbcps_stage411.py`: copy of dev/wpres_stage.py ((4,*,1) layout) + V24 / V25 = T = 4 (two n8 tiles, 2 HMMA per
  word, 8 FFMA per row group and chunk), timing only.
- `dev/moe_v1700.py`: v1002 + MOE1T_BCPS (default off; NVRTC 13 SASS of every moe1t kernel byte-identical to v1002,
  moe1t_4 sha d0da71311c80). Local to moe1t: a new stage function `stage_bcps1t` (+ bc_dec / bc_mma0 / bc_scf / bc_xs),
  the B-fragment construction at it 0 (token row) and at a phase-B expert change, the per-stage partial store, the
  fetcher's ibuf store (x * 2^48 via BC_ST), warp 9's DQF, and 256 zero bytes after the ring (`_smem1t` + 256).
  - MOE1T_BCPS=1: both phases on the BCPS stage (one stage body; moe1t_4 2312 instr, 148 regs). Phase A's token row is
    loaded by every lane as today (wpb1c.md: the row load's lateness is load-bearing) and multiplied by 2^48 (lanes
    g == 2 tig) or 0 (64 HMUL2 per lane, once per call); MOE1T_BCPS_ZA: off lanes read the zero area instead.
  - MOE1T_BCPS=2: phase B only (two stage bodies; 2744 instr, 164 regs).
  - phase B: off lanes read the zero area; the fetcher warps store the intermediate as x * 2^48 (2 HMUL2 per 8-byte
    word, after any check of the raw words: MOE1T_FXSS's max|h| stays on the unscaled values).
- `dev/moe_v1750.py`: dev/moe_v1050.py + MOE_BPC_OFF (code-presence control for v1050's bs8 x0.957).
- `dev/wpbcps_run1.sh` (gate 1), `dev/wpbcps_run2.sh` (moe1t kernel: smoke, check, bench2 bs1 + bs8 placebo x2 passes,
  traces, T = 4 harness), `dev/wpbcps_run3.sh` (bs8 code-presence diagnostic), `dev/wpbcps_sass.sh` (NVRTC 13 SASS),
  `dev/wpbcps_loops.py` (HMMA-loop opcode histograms).

- (added later) `dev/moe_v1701.py` = v1700 + v1301's moe1t diff (MOE1T_FXSS, v1301 trace stamps; BL4 not carried),
  `dev/wpbcps_trace.py` (per-CTA moe1t timelines: last CTAs, SM ids, last-vs-median milestones, correlations),
  `dev/wpbcps_smstat.py` (per-SM statistics from its saved data), `dev/wpbcps_run4.sh` / `run5.sh` (merged into
  `dev/wpbcps_run6.sh`, the job that ran). Logs: scratchpad wpb/run1.log, run2.log, run6.log; bench JSON
  dev/remote_out/wpbcps_run4_{1..4}.json; traces dev/remote_out/wpbcps_trace5.pt.

Codegen note: a per-lane `on ? x : 0` select of the resident fragments is sunk by ptxas into the stage loop (31-64 SEL
per stage in the harness); per-lane pointer selection into a zero area (or a per-lane multiplier) avoids it.

## 3. Gate 1 (dev/wpbcps_run1.sh, Modal H200, 1.80 GHz; two passes, order swapped, identical to 0.3%): GO

moe1t stage (16 rows x 4096 k, 8 warps, resident B fragments) + the per-stage partial store. bench = SMEM-resident,
no ring; ring l2 = producer + 4 slots + reducer warp, source L2-resident (consumer-bound); ring hbm = 64 cold
36 KB stages per CTA (data-bound). Pass 1 / pass 2.

| V | stage | loop instr | bench cyc/stage | ring l2 ns/stage | ring hbm ns/stage |
|---|---|---|---|---|---|
| 0 | today (v1002 stage_two) | 504 | 1453.3 / 1453.5 | 912.2 / 911.7 | 1229.2 / 1224.1 |
| 1 | BCPS, transposed quad reduction (2 SHFL + 2 FSEL) | 400 | 1310.6 / 1310.6 (-9.8%) | 834.8 / 832.2 (**-8.6%**) | 1228.7 / 1218.5 |
| 2 | BCPS, plain reduction (4 SHFL) | 402 | 1264.6 / 1264.7 (-13.0%) | 874.8 / 874.9 (-4.1%) | 1203.3 / 1222.1 |
| 3 | BCPS, no shuffles (4 partials per row to the reducer) | 393 | 1253.5 / 1253.4 (-13.7%) | 834.1 / 833.9 (-8.6%) | 1222.1 / 1214.0 |
| 4 | BCPS, 2 MMA chains per group (4 FFMA) | 426 | 1455.6 / 1455.5 (0%) | 880.5 / 879.8 (-3.5%) | 1226.2 / 1226.7 |
| 5 | V1, byte-3 scale by IMAD.HI | 400 | 1357.0 / 1354.3 (-6.6%) | 841.5 / 842.2 (-7.7%) | 1224.6 / 1223.6 |
| 6 | V1 + one rotation on mul.wide (WP-RES #2) | 400 | 1421.6 / 1421.6 (-2.2%) | 927.6 / 926.7 (+1.7%) | 1228.2 / 1201.3 |
| 7 | V1, groups in pairs | 400 | 1296.8 / 1296.8 (-10.8%) | 837.4 / 836.0 (-8.3%) | 1222.6 / 1222.1 |

- The isolated stage is 10-14% cheaper; in the consumer-bound ring (wait + arrive + reducer) BCPS saves 8.6% per stage
  (78 ns). When the ring is data-bound nothing changes (as WP-RES's gate 2 found for the (4,1,1) stage).
- Schedule sensitivity: V2 is the best bench variant but loses half its gain in the ring; V1 / V3 / V7 hold -8.3..-8.6%.
  The kernel uses V1's form.
- Error vs FP64 (judge weights, expert 3, moe1t repack, 64 items x 16 rows): token row V0 8.2e-8, V1 6.7e-8;
  intermediate V0 1.4e-7, V1 7.6e-8; stress inputs x * 2^-55 .. 2^60 and intermediate * 2^-40: V1 6.7e-8 .. 1.3e-7,
  0 nonfinite (x * 2^48 pre-scale; the H200 matches the local sm_89 numbers).

## 4. Kernel run 2 (dev/wpbcps_run2.sh, Modal H200): moe1t MOE1T_BCPS does not convert at bs1

Correctness (dev/wpb1c_check.py, check_mt M 1 2 normal + skewed x 3 seeds x 2 calls, 40 sequential bs1 calls):
bc1 (BCPS=1), bc2 (BCPS=2), bc1za (BCPS=1 + ZA): 0 failures, max rel-L2 identical to v1002 (M1 2.73e-3 / 2.58e-3,
M2 3.23e-3 / 3.27e-3), sequential max 2.62e-3; outputs vs v1002 max |diff| 3.9e-3 (bc1; one BF16 ulp at 0.5-1).

bench2 (jm seeds, 100 reps, B2_SHARE_EQ; ratio vs v1002 copy a, per-ne columns = the variant's own speedups):

| variant | bs1 pass 1 | bs1 pass 2 (swapped) | ne 2 / 3 / 4 / 5 (pass 1) | bs8 placebo |
|---|---|---|---|---|
| v1002b (control) | 0.9926 | 0.9984 | 4.007 / 3.214 / 2.918 / 2.681 | 1.0003 |
| v1002 | 1 | 1 | 4.033 / 3.262 / 2.936 / 2.628 | 1 |
| bc1 = MOE1T_BCPS=1 | 0.9922 | 0.9918 | 3.937 / 3.248 / 2.915 / 2.696 | 1.0042 (placebo) |
| bc2 = MOE1T_BCPS=2 | **0.9066** | 0.8986 | 3.660 / 2.927 / 2.704 / 2.469 | 1.0042 (placebo) |
| bc1za = BCPS=1 + ZA | 0.9638 | 0.9527 | 3.756 / 3.161 / 2.860 / 2.663 | 0.9969 (placebo) |

(pass 2 ratios rebased from "vs bc1za": bc1 1.0410, v1002b 1.0479, v1002 1.0496.) bc1: ne 2 -2% in both passes, ne 3
(the median's block) +-0, ne 5 +1%: neutral to -0.8% overall.

Trace (dev/wpb1_trace2.py, MOE1T_TRACE, seeds 73235 ne 3 / 70055 ne 2, 7 reps; warp 0 medians over CTAs):
- stage compute ("ready + compute"): phase A 0.86 -> 0.77 us, phase B 0.80-0.83 -> 0.64-0.74 us (-10%: the harness gain is
  real inside the kernel).
- phase A stage interval 1.045 -> 0.997 us (ready 4.48..14.91 vs 4.35..14.32 over 10 intervals). 36 KB x 132 CTAs per
  0.997 us is ~4.9 TB/s: phase A is now at (beyond) the measured stream ceiling, i.e. data-bound.
- phase B (ne 3, stages 12-17): within an expert 0.87-0.91 us (base 0.96-1.00), but every expert change 1.39-1.41 us
  (base 1.21-1.28); the phase-B average stays ~1.09 us/stage in both (data-bound on average).
- milestones ne 3: A2st 16.26 -> 15.81, B0fr 16.06 -> 15.78, c-done 22.50 -> 22.14, seen 23.94 -> 23.68, out 24.32 -> 24.10
  (median CTA -0.22 us); ne 2: c-done 16.45 -> 16.19 but max arrive 16.90 = 16.90 (the slowest CTA does not gain), out
  18.14 -> 18.05.
- Reading: the -10% stage removes the consumer bound, and the kernel then sits on the HBM stream (phase A) and on the
  phase-B data + expert-change latency; the median CTA finishes ~0.1-0.3 us earlier but the slowest CTA and the end
  chain do not move, so the score (median paired ratio, ne-3 block) is flat.
- bc2 (two stage bodies in one loop, 164 regs): -9..-10% at every ne (the phase-A body in the same loop as the BCPS body
  is far slower; not a code-fetch effect of that size). bc1za (zero-area token row, i.e. BL4's fast row load): -4..-5%,
  the same mechanism WP-B1c found for MOE1T_BL4 (the token-row load's lateness is load-bearing).

T = 4 ((4,1,1) harness, dev/wpbcps_stage411.py, timing only, 2 passes identical): V0 today 1450.3, V1 S6O 1424.4,
V14 BCPS T=1 1268.6 (-12.5%), V15 T=2 1311.4 (-9.6%), V20 S6O T=1 1261.7, V21 S6O T=2 1352.4, **V24 T=4 1611.1 (+11.1%),
V25 S6O T=4 1622.7 (+13.9% vs V1)**: T >= 3 (2 HMMA per word) loses, as predicted.

## 5. Why the -10% stage does not convert at bs1 (interim; per-CTA trace pending in dev/wpbcps_run6.sh)

- WP-B1c's observer trace (wpb1c.md section 4) puts the consumer period at 1.088 us/stage in phase A vs the 1.037 us HBM
  floor (128 x 36 KB at 4.55 TB/s), and notes that the trace builds run ~0.05 us/stage slower than the bench build. So in
  the bench build phase A is within ~0-2% of the floor before BCPS; BCPS (0.86 -> 0.77 us compute) can only remove that
  remainder. In the trace build BCPS brings phase A to 0.997 us/stage, i.e. to the floor (section 4).
- Phase B: BCPS shortens the within-expert stages (0.96-1.00 -> 0.87-0.91 us) but every expert change gets longer by
  about the same amount (1.21-1.28 -> 1.39-1.41 us); the phase-B average stays ~1.09 us/stage: the stage data (copy
  latency ~1.5 us, WP-B1c) and the expert-change dependency set it, not the consumer.
- End chain: median-CTA c-done / arrive / out move 0.1-0.36 us earlier, but the max arrive (the slowest CTA) moves 0-0.16 us,
  and seen = max arrive + ~0.8-1.1 us. The tail spread (max - median arrive) is ~0.45 us at ne 2 and ne 3.
- dev/wpbcps_trace.py (queued, run 6) dumps every CTA's stamps (SM id, per-stage ready / computed of warps 0 and 5, producer
  issue times, milestones) for base / BCPS / FXSS / BCPS+FXSS on 4 ne-3, 2 ne-2 and 1 ne-4 routings x 15 reps, and reports
  which CTAs / SMs arrive last, the last CTA's timeline vs the median CTA's, and the correlation of a CTA's arrive with
  its phase-A rate, its copy latency (issue -> ready) and its last B fragments.

## 6. Run 6 (dev/wpbcps_run6.sh, Modal H200): BCPS + FXSS, v1050's bs8 loss, and what sets the slowest CTA at bs1

### 6a. BCPS + MOE1T_FXSS (dev/moe_v1701.py = v1700 + v1301's moe1t diff; NVRTC 13: default = v1002 sha d0da71311c80,
FXSS = v1301 FXSS sha 14ed64c8d10b, BCPS=1 = v1700's). check_mt M 1 2 + 40 sequential: 0 failures, errors = v1002's.

| bs1 (jm 100 reps) | pass 1 vs v1002 | pass 2 vs v1002 (rebased) |
|---|---|---|
| fx (v1301 FXSS) | 1.0053 | 1.0113 |
| fxb (copy) | 1.0013 | 1.0139 |
| bcfx = BCPS=1 + FXSS | 0.9837 | 1.0132 |
| bc1 = BCPS=1 | 0.9869 | 1.0080 |

bcfx / fx: 0.979 and 1.002 (ne 2 -3..-5% in both passes). They do not stack. Over the four bs1 passes of runs 2 and 6,
bc1 / v1002 = 0.9922, 0.9918, 0.9869, 1.0080 (mean -0.5%): **MOE1T_BCPS is not banked.**

### 6b. v1050's bs8 x0.957 is code presence, not BPC execution

| bs8 (jm 100 reps) | pass 1 vs v1033 | pass 2 vs v1033 (rebased) |
|---|---|---|
| v1033b | 0.9989 | 0.9974 |
| bpc (v1050 MOE_BPC) | 0.9433 | 0.9510 |
| bpcoff (dev/moe_v1750.py MOE_BPC + MOE_BPC_OFF: the BPC branch compiled, never taken) | 0.9425 | 0.9470 |

The build with the BPC code present but never executed loses the same 5-5.5%; executing BPC on the one-token items is
worth +0.1..+0.4% on top (bpc / bpcoff 1.0008, 1.0042). The extra loop (moe_4_1_1_4_1 6128 -> 7000 instructions, same
168 registers) perturbs the existing loops' code generation (WP-RES: today's 1-group loop same 510 instructions,
re-scheduled, opcode-order similarity 0.46). Consequence for the plan's step 3 (bs8 / bs16 runtime choice between the
BCPS stage and today's): NO-GO. Even with zero code-presence cost the ceiling is the +0.1-0.4% execution gain, because
the steady 1-group stage there is ring-bound (WP-RES gate 2).

### 6c. What determines the slowest CTA's end at bs1 (dev/wpbcps_trace.py, dev/wpbcps_smstat.py; trace builds of v1701)

420 traced calls (4 builds x 7 routings: 4 ne 3, 2 ne 2, 1 ne 4 x 15 reps), per-CTA stamps incl. %smid and the producer's
copy-issue time per stage (data: dev/remote_out/wpbcps_trace5.pt):
- The call ends at seen = (last CTA's arrive) + the end chain: seen - max arrive = 0.74-0.77 us (base), 0.42 (FXSS),
  0.51-0.54 (BCPS + FXSS).
- The last arrival comes from a broad tail. max - median arrive is 0.26 / 0.32 / 0.45 us at ne 2 / 3 / 4 (base), max - p90
  0.10-0.16. The last CTA's lag builds up linearly over the whole call: vs the median CTA it is +0.06..0.13 us at A0st,
  +0.16..0.22 at A1st, +0.22..0.29 at A2st, +0.32 at arrive (ne 3). Its compute per stage is identical (0.86-0.90 us).
  Its stage-ready times drift +0.02-0.03 us per stage, and its copy latency (issue -> ready) is higher.
  corr(arrive, phase-A stage interval) is +0.56..0.79, corr(arrive, copy latency) +0.2..0.8.
- Per-SM speed is a stable hardware property. The 128 CTAs always occupy the same 128 SMs (117, 119, 121, 123 unused)
  with a different CTA -> SM mapping per launch. A SM's mean arrive lag has rank correlation 0.956 between the first and
  second half of the reps. The slowest SMs come in TPC pairs: tpc64 (sm128/129) +0.22 us, tpc21 (42/43) +0.21,
  tpc43 (86/87) +0.20, tpc13 (26/27) +0.18, tpc57 (114/115) +0.18, tpc5 (10/11) +0.17, tpc36 (72/73) +0.17. Each has a
  +10..13 ns (1%) longer phase-A stage interval and +20..30 ns copy latency (a smaller DRAM / crossbar share). The
  fastest TPCs are at -0.08 us. sm128/129 are among the last 4 arrivals in 23% of calls (fair share 3%).
- The systematic per-TPC part (up to +0.22 us) plus per-call noise gives the tail. Dropping the 2-8 slowest SMs' CTAs
  would not move the median call's max (the last CTA is spread over many TPCs).
- BCPS widens the tail (ne 3: max - median 0.32 -> 0.45 us, max - p90 0.12 -> 0.22). The median CTAs get faster and
  issue their copies sooner, and the extra bandwidth comes out of the slow CTAs' share. So the max (and the bench) barely
  moves. In the trace builds (per-stage stamps add ~0.05 us/stage, WP-B1c) the consumers are slower, and there BCPS does
  move the max (-0.3..-0.5 us) and 'out' (-0.4 us at ne 3). The bench build is at the data floor; the trace build is not.
- Is there a "BCPS + target the slow CTA" fix? The slow CTA is data-limited, not compute-limited, so BCPS cannot help it.
  The lever is balancing bytes across SMs, worth at most max - mean arrive = 0.23-0.42 us (1.0-1.7% of bs1, ~0.2-0.3%
  geomean). A static per-SM table will not transfer: which TPCs are slow depends on the chip's floorsweeping and the
  judge's H200 is another chip. It would have to be dynamic, e.g. the last phase-B row blocks handed out by a global
  ticket so fast CTAs take more. The rmsnorm combine then needs each output row's expert terms from several CTAs, i.e. a
  cross-CTA reduction of y, a structural change on WP-B1c's end chain for <= ~1.5% at one shape. BCPS would add on
  top only if that balanced kernel became consumer-bound again.

**For WP-B1c (moe1t end chain):** at bs1 the end is (slowest CTA's data stream) + (end chain). The slowest CTA is set by
a stable per-TPC DRAM-share deficit (~1% per stage) plus noise, which accumulates over all 6 ne stages; consumer-side
savings in the median CTA (BCPS, PIPE, ...) do not move it and can widen the tail. FXSS works because it shortens the
part after the last arrival (0.77 -> 0.42 us). Remaining levers: (1) the end chain after the last arrival (0.42 us
with FXSS); (2) byte balancing toward fast SMs (<= 0.2-0.4 us, dynamic only).

## 7. Final summary (WP-BCPS closed)

Closed:
- BCPS in moe1t (bs1): MOE1T_BCPS=1 neutral (mean -0.5% over 4 passes), =2 -9%, +ZA -4%, + FXSS no stack. Not banked;
  dev/moe_v1700.py / v1701.py default builds stay SASS-identical to v1002 / v1301.
- BCPS in the bs8 / bs16 kernels (runtime per-item choice): NO-GO. The code-presence cost of a second stage body is
  -5% (v1750 control), and the execution gain is +0.1..0.4%.
- T >= 3 (2 HMMA per word): +11..14% stage cycles.
- Numerics note for any future post-scale: with unscaled codes (E2M1 * 2^-126) the partials sit at the FP32 subnormal
  edge. x needs an exact power-of-two pre-scale (2^48 here); unscaled x already fails at |x| ~ 2^-20.
- Codegen note: ptxas sinks a per-lane select of loop-invariant fragments into the loop. Use pointer selection into a
  zero area or a per-lane multiplier.

Per-SM / per-TPC tail facts (for WP-B1c and WP-FUSE):
- On this H200 (Modal), under a full 128-CTA HBM stream, a stable subset of TPCs runs ~1% slower per stage: +10..13 ns
  per 36 KB stage, +20..30 ns copy latency. Examples: tpc64 sm128/129, tpc21 sm42/43, tpc43 sm86/87, tpc13 sm26/27,
  tpc57 sm114/115, tpc5 sm10/11, tpc36 sm72/73. Their rank across reps is stable (rank corr 0.956). The fastest TPCs are
  ~-0.08 us vs the median. So over a ~23 us bs1 call the end spread is 0.26-0.45 us (max - median arrive), and it grows
  with the number of stages (ne).
- 128-CTA grids land on the same 128 SMs every launch (117/119/121/123 idle here), but the CTA -> SM map changes per
  launch, and which TPCs are slow depends on the chip (floorsweeping): a static plan cannot pre-assign less work to the
  slow SMs.
- Any statically planned kernel (WP-FUSE's included) will meet the same variance: its completion is the max over
  ~128 per-SM byte streams. The full spread is ~1-2% of a data-bound call. A small dynamic slack mechanism at the very
  end would recover most of it, at a granularity well under the ~0.3 us tail (e.g. the last few row blocks handed out
  by a global ticket, or half-stages for the final items). Faster consumers alone do not help: they widen the tail
  (BCPS ne 3: max - p90 0.12 -> 0.22 us) because the fast CTAs take bandwidth from the slow ones.
