# muse_moe_w4a16: dead ends

Failures that changed direction: what was tried, the number, why it failed, which milestone it informed. Numbers are local same-job A/Bs (bench2 or the judge lab) unless marked "judge"; negative = slower. Paths are relative to ``: `M/` = `muse-opt/muse_moe_w4a16/`, `F/` = `M/kda_ws/docs/findings/`, `mem:` = `<agent-state>/memory/`.


### 10-22 → 10-24 (milestones 01-04)

| idea | number | why it failed | informed | source | date |
|---|---|---|---|---|---|
| three kernels per call (route / GEMM / combine, v1) | H100 0.862x geomean; bs256 GEMM 1607.7 µs; the NT=8 GEMM at 168 registers + 520 B spill | split launches and a spilling 64-token tile | 01 (one persistent kernel from v2) | `01_.../evidence/transcript_v1_profile_ablation.txt` | 10-23 |
| decode shifts on the FMA pipe (mul.lo/hi v8, mul.wide v13, IMAD left shifts v14) | v8 1.484x vs 1.612x (H100); v13 / v14 worse | IMAD.HI / IMAD.WIDE cost more than the SHF they replace | 01 (v30 keeps exactly 2 IMAD per word) | `01_.../evidence/transcript_h100_bench_filtered_v1_v40.txt` | 10-23 |
| stream-K static schedule (v11) | H100 1.559x vs v9 1.796x | the dynamic queue absorbs per-CTA spread | 01, 09 | session transcript 10:17 UTC 10-23 (summarized in `01_.../README.md`) | 10-23 |
| wgmma for all tiles (v16), for 2-4 groups (v53), for 3-4 groups in 32-token configs (v107) | H100 1.77x vs 1.832x; -0.5..-2 %; -1.7..-2 % | decode and wgmma do not overlap; ring slots lost to 1 KB alignment | 01, 02, 04, 05 (wgmma kept for ≥ 5 groups, then 3-4 once dead paths went) | `02_.../evidence/sweeps/`; `04_.../evidence/transcript_*` | 10-23/24 |
| L2 prefetch of later stages (`MOE_PD`, bulk and per-line, v51 / v60) | -14..-21 % geomean (Modal) | prefetches compete with the ring's copies | 02, 05, 09 | `02_.../evidence/sweeps/` (s7, s13) | 10-23 |
| two 4-warp CTAs per SM (v45) | 2.079 vs 2.210 (Modal) | worse latency hiding | 02 | `02_.../evidence/sweeps/` (s2) | 10-23 |
| epilogue warps finishing items (v46) | judge 2.1270 vs 2.1335 | faster only in compute-only mode on the H100 | 01 | `01_.../evidence/judge_v46_7440dd65_dead_end.txt` | 10-23 |
| SSQ fast combine at every batch size (v54, v86) | bs128/256 +4..+8 % time | multi-group / k-split configs pay for the slot sums | 02 (SSQ only for NT == 1) | `02_.../evidence/sweeps/` (s8); `04_.../evidence/sweeps/` (s51) | 10-23 |
| cross-CTA hand-off variants: claim-ahead (v61, v104), signaler partner reduction (v64, v74), global-barrier or static combine (v105, v106), background combiner (v95) | -0.6..-3.6 %; bs1/8/16 -7..-9 %; -4.0 / -1.8 %; -10..-12 % | every cross-CTA hop costs 2-6 µs under streaming load (`red.release` blocks its writer ~4.4 µs) | 03 (the one-token kernel avoids mid-call waits), 06 | `02_` / `03_` / `04_.../evidence/sweeps/`; `03_.../evidence/membar_result.txt` | 10-23/24 |
| bottom-tested per-item loop | bs32 +2 % time; bs256 kernel 17,776 vs 12,216 SASS instructions | the compiler's loop inversion emitted the unrolled stage body twice | 02 | `02_.../evidence/transcript_v65_v66_loop_sass.txt` | 10-23 |
| tail k-splits at bs8/16 (v68, v81, v101; row halves v88) | -4..-9 % | per-piece overhead plus a slow combine of partials | 02-05 (see also 10-24..10-26 below) | `02_.../evidence/sweeps/` (s23); `03_.../evidence/sweeps/` (s38) | 10-23/24 |
| `yacc` replicas in the one-token kernel (YR 2/4/8) | bs1 +9..+21 % time (H100) | more lines and adds per combine | 03 | `03_.../evidence/sweeps/` (s48) | 10-23 |
| tensor-map weight copies in the main kernel (v80) | judge GPU bs8 -6.3 %, bs16 -4.8 %, bs32 -1.8 % (neutral on H100) | mechanism not established | 03 (kept only in the one-token kernel) | `03_.../evidence/labs/lab1.7870d1bfb78c491a.log` | 10-24 |
| header re-layout with the ring mbarriers off dynamic-smem offset 0 (v85) | bs8 -5.4 %, bs16 -4.2 % vs v83; 6-layout sweep -3 % at bs8-32 | mechanism unknown | 04 | `04_.../evidence/transcript_bench2_v83_v103_bisect_v112.txt`, `transcript_header_layout_sweep_v132.txt` | 10-24 |
| per-variant weight copies in A/Bs | identical builds read 3.6 % apart at bs1 | physical placement bias | 04 (shared-weight bench2 from then on) | `04_.../evidence/transcript_bench2_placement_control.txt` | 10-24 |
| bs8/16 early start | +0.7..+1.2 % on v85's layout; -1.3..-1.5 % on v83's | it only offset the header loss | 04 | `04_.../evidence/sweeps/` (s57), `labs/` | 10-24 |
| cross-stage operand prefetch (v114, v116); hand 2x unroll (v117) | bs8/16 -10 %, bs32-256 -2.4..-7 %; bs32 -1.5..-3.8 % | lost the compiler's own 2x unroll of the 1-group loop; v114 spilled | 04 | `04_.../evidence/transcript_bench2_v114_v119.txt` | 10-24 |
| main-kernel ring of 3 stages | bs8 -4.8 %, bs16 -4.3 % (shared weights) | not enough prefetch depth | 04 | same | 10-24 |

### 10-24 → 10-27 (milestones 05-06)

| idea | number | why it failed | informed | source | date |
|---|---|---|---|---|---|
| L2 prefetch of upcoming stages (`MOE_PD`) | -8 % (later -11..-88 %) | the prefetches share the copy path the producer waits on | 05, 09 | `M/NOTES_status.md` ("Rejected today"); `mem:okc-moe-versions.md` | 10-24 |
| wgmma for 3-4 groups with all paths compiled in | bs128 -2..-3 % | the faster wgmma stages made the kernel's mma.sync stages 4-11 % slower (code presence) | 05 (fixed by v141's dead-path removal, then +2.2 %) | `M/NOTES_status.md` | 10-24 |
| paired ring slots at bs256 | -1.6 % (judge lab x0.982 / x0.985) | halves the prefetch depth | 05 | `05_.../evidence/judge_lab/lab7.*.log` | 10-24 |
| 32-token tiles at bs256 | -7 % | twice the decode per byte for the hot experts | 05 | `M/NOTES_status.md` | 10-24 |
| tail k-splits at bs8/16 (KSXK2, FIXUP, HSPL / HSPL2 / HROW halves) | judge lab x0.912-0.931; Modal -3.6..-9.1 % (bs8), -1..-5.5 % (bs16) | per-piece cost (~0.75-0.9 µs first-stage wait) ≈ the tail gain; split-aware code re-schedules the consumer loop (~5 %); FIXUP's `__threadfence` costs ~4 µs per CTA | 05, 07, 10 (the split that finally won at bs16 was a loop-drawn port) | `05_.../evidence/judge_lab/lab13.*.log`; `mem:okc-moe-versions.md` 10-25, 10-26 | 10-24..10-26 |
| tensor-map (4-D box) weight copies in the main kernel | judge lab bs8 x0.937, bs16 x0.952 | extra copy-issue latency under the stream on H200 | 03, 05 | `05_.../evidence/judge_lab/lab1.*.log` | 10-23 |
| WGX / WGP pipelined wgmma across stages | WGX -1..-1.6 %; WGP U=2 -4.4 %, U=4 -11..-13.9 % | ptxas C7513 serialization when groups cross the back edge; loss grows with code size (i-cache) | 05, 08 | `mem:okc-moe-versions.md` 10-25 | 10-25 |
| FP16 wgmma tiles at bs256 (`MOE_F16`, v174/v177) | judge-case bs256 -4.0 % (195.9 → 188.0 µs); judge v177 2.3975 / 2.3872 / 2.3677 | **worked**, but voided by the 10-27 BF16 rule (rules, not performance) | 06 (BF16 rebase from v601) | `mem:okc-moe-versions.md`; `M/kda_ws/docs/draft.md` §0 | 10-25..10-27 |
| FP16 post-MMA block scale (`MOE_PS/PS2/PS1`, v450 → v600) | judge check 2.4470, scored 2.4203 | voided by the BF16 rule | 06 | `06_.../evidence/judge_v600_fp16_2dbd0bc9.txt`; `F/wpd.md` | 10-26 |
| BF16 post-MMA scale (FP32 FFMA, PSB / V12 / V13) | stage +12.5..+30 % cycles | 4 FFMA per HMMA = the 4 HMUL2 it replaces; the 2^-126 decode bias then needs x pre-scaling | 06, 09 (BCPS) | `M/kda_ws/docs/draft.md` §5g-5h | 10-27 |
| CUDA-core (tensor-free) decode + HFMA2 pools for bs8/16 (WP-P), bs1 (WP-1C) | bs8 86 vs 65 µs; two pools 70 vs 65.6; bs1 x1.11 only in FP16 (invalid) | ≥ 2.25 instructions per weight vs 1.75 on the tensor path; avoids the 1.79 GHz cap but loses on issue | 06 | `F/wpp.md`, `F/wp1c.md` | 10-27 |
| cheaper decode than 9 integer ops (IMAD/IMAD.HI rotations, MOE_DQH on the FMA pipe, pure-shift layouts) | rotations +17-33 % decode cycles; DQH -10..-19 %; F16S -1.7 % | exhaustive layout search: the BF16 field cannot be tiled by 4 rotations (0 solutions at any period 32-256 bits); the FMA pipe competes with the tensor core | 05, 06, 09 | `F/wpres_decode.md`; `M/kda_ws/docs/draft.md` §7 P-DECODE | 10-25..11-01 |
| fused multi-token kernel `moet` (bs1 kernel's units for 2-8 tokens, v179) | bs8 84.1 vs 48.8 µs | per-unit partials cost ~24 µs of `red.v4` at bs8; split-K over the intermediate needs a free cross-CTA reduction | 06 | `mem:okc-moe-versions.md` 10-26 | 10-26 |
| 5-bit block scales with a per-(k chunk, row group) base (`MOE_S5`, v180) | bs8 -1.4..-1.9 %, bs16 -1.1..-1.4 % | saves only 0.9 % of stage bytes; funnel shifts, base load and bank conflicts cost more | 06, 10 (a per-row-base form later won at bs8 only) | `mem:okc-moe-versions.md` 10-26 | 10-26 |
| x sharing in a 2-CTA cluster (multicast, v181 / v182) | bs128 -7 %; v182 -20 / -16 % | cluster handshakes on the critical path; even free x copies (`MOE_NO_X`) are worth only +2.7 / +4.2 % | 06 | `mem:okc-moe-versions.md` 10-26 | 10-26 |
| 16 consumer warps (v184 bs1, v186 main) | bs1 -8..-10 %; bs8 -11 %, bs16 -9 % | `setmaxnreg.inc` only takes what the CTA's `.dec` released; 16 warps lose B-fragment reuse | 06, 08 | `mem:okc-moe-versions.md` 10-26 | 10-26 |
| two-phase kernel for M ≤ 8 (`moe8t`, WP-B8) | best x0.905-0.94 vs the main kernel | same stage cost, worse edges; bs16 infeasible | 06 | `F/wpb8.md`; `M/kda_ws/docs/draft.md` §5n | 10-27 |
| 512-row stages for 3-4-group tiles (WP-X XR) | copy bytes / 2 at bs128, no speedup | code presence 1.8-3.1 % at bs128/256 | 06 | `F/wpx.md` | 10-27 |
| bs256 row-split tail halves / x-helper warps (WP-L RSPL, XH) | -2.5..-3.3 %; -5 % | tail stages ~0.85 µs whatever their size; helper warps cost their SMSP's consumers | 06 | `M/kda_ws/docs/draft.md` §5j-5k | 10-27 |

### 10-27 (milestones 07-08)

| idea | number | why it failed | informed | source | date |
|---|---|---|---|---|---|
| last-arriving split piece sums the partials (`MOE_KFIX`) | bs128 x0.985 | synchronous release atomic (`MEMBAR.GPU` + `ATOM`) once per piece on the epilogue warps, plus round trips at the end of the call | 07 (fence-free KRED instead) | `F/wpk.md` | 10-27 |
| KRED k-splits at bs8/16 | x0.93 | the dead slow-combine body becomes live (`moe_4_1_1` 5712 → 7592 instructions) | 07 | `M/kda_ws/docs/draft.md` §5s | 10-27 |
| PDL two-kernel hybrid (WP-H) and PDL tail kernel (WP-T) | hybrid -1..-3.5 % vs v655; tail kernel bs8/16 x0.91 / 0.95 | CTAs launched mid-call pay ~5 µs to first data (cold start) | 07 | `F/wph.md`, `F/wpt.md` | 10-27 |
| `__noinline__` item functions to freeze codegen (WP-C) | loses | called loops pay ~3x instruction-fetch stalls; the real code-presence cost is the producer spilling at 72 registers | 07 | `F/wpc.md` | 10-27 |
| bs1 k-split of the last expert (WP-F) | x0.90-0.98 | the split pieces' end chain costs more than the dependency it removes | 07 | `F/wpf.md` | 10-27 |
| moe1t poll by TMA (FTMA) / conflict-free x fragments (XN) | -6 %; -1 % | the poll copy queues behind the ring's weight copies in the TMA unit; XN's earlier start spreads the CTAs' phase-A stores and the first polls see stale words | 07 | `F/wpb1.md` runs 13-14 | 10-27 |
| freeing the held slot early by moving staging (SEPI8, AREL, BSTG) | -3..-4 % | moved epilogue work into compute | 08 (TMP's bulk-copy form won instead) | `M/kda_ws/docs/draft.md` §5u | 10-27 |
| 16 consumer warps for bs8/16 (WP-16W) | probe x0.973, bs8 x0.962, bs16 x0.959 | the bs8/16 stage loop under streaming is within 1.7 % of the bare ring (1138 vs 1119 ns/stage): the gap is outside the loop | 08 | `F/wp16w.md` | 10-27 |
| fast start ported to bs128 (FST4), FSTW, FDRY | lose | early weights slow the cold x gather (4.98 → 6.86 µs); a dry pass holds the bitmap barrier | 08 | `F/wp32.md` | 10-27 |
| bs256 startup chain edits (WP-RT GV / GV2 / GZ / EW / EP) | x0.984-0.998 | trace gains (barrier -1 µs) never converted; GCP-side code costs 0.3-0.7 % | 08 | `M/kda_ws/docs/draft.md` §5zzc | 10-27 |
| scratch reloads by generic loads (TMGC) / consumers `red.add` pieces from registers (KDIR) | x0.985; x0.979 | generic-load reloads are slower than TMA; register reds lengthen the consumer path | 08 | `M/kda_ws/docs/draft.md` §5zza | 10-27 |
| decoupled stream by L2 run-ahead prefetch (v1030 / v1031) | x0.85-0.94 | same-SM L2 prefetch contends with the ring copies | 09 | `M/kda_ws/docs/draft.md` §5zzf | 10-27 |
| two item streams per SM (WP-DUAL) | x0.83-0.98 | heavy and light items share the decode ALU / dispatch; 4-warp streams hide latency worse | 09 | `F/wpdual.md`; draft §5zzg | 10-27 |
| deeper rings / variable slot sizes (plan_bwm) | ring depth 2-6 identical at 4.58 TB/s | only the 34 KB activation bytes per stage cost, and only on compute-bound 64-token tiles | 09 | `mem:okc-moe-versions.md` 10-27 20:35 | 10-27 |

### 10-31 → 11-01 (milestones 09-10)

| idea | number | why it failed | informed | source | date |
|---|---|---|---|---|---|
| static byte-balanced fused kernel for small batches (WP-FUSE `moef`) | bs8 x0.915-0.926, bs1 x0.871-0.893 | a static split turns every per-CTA delay into tail; `red` bytes cost about as much as streamed bytes; the K-split end needs a global arrival | 09 (keep the dynamic queue) | `F/wpfuse.md` | 11-01 |
| block-column post-scale (BCPS / BPC) | bs8 x0.957 even when never taken; moe1t -0.5 % | code presence: the extra loop re-schedules the existing 1-group loop | 09 | `F/wpbcps.md` | 11-01 |
| end-of-call combine rewrites (WP-END TF / PB / LL / ...) | noise or worse (9 jobs) | a shorter combine moves the last items later by the time it saves: the end is bound by total traffic | 09 | `F/wpend.md` | 11-01 |
| bs1 one-round exchange forms (WP-B1X XRA, XHC, XLEAN) and BCPS + early start (WP-B1d) | XRA -8..-9 %; BCPS -1.0 / -1.8 % | gpu-scope release under the stream costs 1.7 µs writer / 5-7 µs fetcher; faster consumers need one more fetch round | 10 | `F/wpb1x.md`, `F/wpb1d.md` | 11-01 |
| moe1t phase-B fragment-reload fix (`MOE1T_BLD`) | -7.8 % (bound +1.9 %) | any moe1t edit flips ptxas's allocation (136 → 116 registers costs ~9 %) | 10, 11 | `F/wpb1p.md` | 11-01 |
| hot expert decoded once at bs256 (WP-HOT pair path) | -4..-8 % (bound +0.78 %) | 128-token stages add decode + tensor; code presence 0.6-3.7 % | 10 | `F/wphot.md` | 11-01 |
| startup advances (WP-START V1 / V2 / WF / ...) and a producer dry pass (`MOE_FT_DRY`) | +0.04 %; dry pass +3.3 µs up front | a second gate sits 0.2-1 µs behind every advanced one; startup is bound by per-SM instruction fetch of cold code | 10, 11 | `F/wpstart.md`, `F/wpfast.md` | 11-01 |
| deeper ring for 1-group stages (WP-RING) | ceiling +0.2-0.45 %; NS 2 -4..-13 % | 1-group stages are consumer-paced; depths sit just past the knee | 10 | `F/wpring.md` | 11-01 |
| tail remedies (WP-TAIL LC / DU / PS / PSK=32 / END_RO_MW=8) | noise or negative | warp 0's wait at an item's first stage (0.8-1.25 µs) is the per-piece cost behind every tail trade-off | 10, 11 | `F/wptail.md` | 11-01 |
| BF16 staging / dedicated FP32 staging for bs32 items (WP-IT3, WP-SEPI32) | -1.1..-1.3 %; -0.4..-0.7 % | re-schedules all four loops; a never-blocking `try_wait` at the item end costs ~0.5 % on the critical warp | 10 | `F/wpit3.md`, `F/wpsepi32.md` | 11-01 |
| cold i-cache levers (WP-COLD) | prize ≤ +0.5 % geomean; nothing robust beyond LATE9 at bs8 | the SM i-cache holds ~28 KB; own code is not readable as data, so it cannot be L2-prefetched | 10 | `F/wpcold.md` | 11-01 |

### 11-03 until the cut (milestone 11)

| idea | number | why it failed | informed | source | date |
|---|---|---|---|---|---|
| bs8/16/32 consumer-side stage work (ST: NO_DEQ bound, early slot release) | NO_DEQ bound only -2.9..-4.0 %; early release bs32 +3.3 % slower | 1-group stages are stream-bound (consumer slack ~20 %) | 11 (go after the edges) | `M/t3/ST/NOTES.md` | 11-03 09:40 |
| heavy/light SM roles at bs128/256 (HV mission 3) | -0.2..-8.8 % | calibrated queue simulator: granularity already at its optimum | 11 | `M/t3/HV/NOTES.md` | 11-03 11:00 |
| L2-partition-local streaming (HV mission 2) | every near form 5-23 % slower; coupled 3.52 vs 4.58 TB/s | one 36 KB contiguous copy beats any 4 KB-granular near form | 11 | `M/t3/HV/NOTES.md`; `M/t3/PW/NOTES.md` | 11-03 10:16 |
| power / clock levers (PW) | ceiling ~+1.7 % geomean | the limiter is a sub-ms power-step limiter invisible to NVML; time/clock elasticity 0.1-0.6 | 11 | `M/t3/PW/NOTES.md` | 11-03 10:05 |
| bs128/256 transitions and combine (EN: TOK2 2x combiners, INA / TOKA) | +0.8..+1.7 % slower; ~0 | not throughput-bound; transitions reachable only by a rewrite (which IL then built) | 11 | `M/t3/EN/NOTES.md` | 11-03 11:20 |
| in-flight cap / shallow queue (RV, QC) | C=1 +25..+31 % slower; C=2 +0.4..+1.4 %; bs1 cap 1 -1.3..-6.7 % | producer issue work sits in the slot cycle, so caps cost stream rate | 11 | `M/t3/RV/NOTES.md`, `M/t3/QC/NOTES.md` | 11-03 12:10-12:25 |
| bigger early bursts at start (SU E=4, SU_F32) and bs1 producer start shifts (SU mission 2) | bs8 +0.9..+1.9 % slower; bs32 +0.6..+0.9 %; bs1 E=1 -0.42 %, held back 400 ns -2.4 % | every round trip and cold fetch issued under the burst waits for it to drain; bs1's start is consumer-bound | 11 | `M/t3/SU/NOTES.md` | 11-03 11:00-11:35 |
| IL with a blocking early record (K4096 / K12288) | deadlock (`check_mt` timeout) | consumers wait for the next item before handing the held slot to the epilogue warps | 11 | `M/t3/IL/NOTES.md` §5 | 11-03 12:30 |
