# 04: a shared-weight A/B method finds a header-layout regression; bs1 kernel trims and two re-tuned knobs (v83 to v119)

Paths: `M/` = `muse-opt/muse_moe_w4a16/`, `R/` = `okc-retro/`, `mem:` = `<agent-state>/memory/` (outside the tree), all relative to ``. "transcript, hh:mm UTC" = the main session log `<agent-state>/<session>.jsonl` (outside the tree; 10-24 UTC unless marked; PT = UTC-7). Copies in this folder are cited as `evidence/...` and `notes/...`.

This milestone is as much about measurement method as about the kernel. About twenty builds after v83 were each "faster" on the local A/B, and the judge read four of them below v83 at bs8 and bs16. The fix was to change how two builds are compared; that comparison then found the culprit.

| | before (03) | after (04) |
|---|---|---|
| build | `../03_fused_one_token_kernel/build.py` = v83 (md5 c690651a) | `build.py` = `M/dev/moe_v119.py` (md5 af9f04bd, 127,274 B; copied to `submission.py` at 10:27 UTC: transcript, cp command at 10:27 UTC) |
| judge | 2.2639 / 2.2823 (fc7f84de, e83db22b), real 2.2731 | **2.3119** (scored, job f3c48e94). On the way: v92 2.2476, v98 2.2419, v100 2.2773, v103 2.2755, v112 2.2904, v118 2.2879 (all scored) |
| real effect | | shared-weight same-job A/Bs chained per shape: bs1 about +7 %, bs32 +1.6 %, bs256 +1.0 %, bs8/16/128 +0.5 % → about **+1.8 %** geomean over v83 (derived). The judge read +1.7 % |
| shapes changed | | bs1 (`moe1_5` → `moe1_4`); bs32 moves to config (8,2,1) (`moe_8_2_1_4_1`); the main kernel keeps v83's shared-memory header layout and gains BF16 down outputs and a 2-pair combine |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: one-line comments credit the designs or reference code that parts of it follow. The code is unchanged. The judged file has md5 `af9f04bd`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

Terms (as in 01-03): **item**, **stage**, **producer**/**consumers**/**signaler**; kernels `moe_<RW>_<NT>_<KS2>_<NS>_<MW>`; **moe1_N** is the one-token kernel with an N-stage ring. New here:
- **Header**: the first ~1 KB of the main kernel's dynamic shared memory: the ring's full/empty mbarriers (`bars`), per-stage info, signal slots, routing scalars, and the per-expert global scales. The ring stages come after it.
- **Early start** (`EARLY`, `MOE_EARLY_ALL`): the producer derives its first item from `topk_ids` and issues its first stages while the consumers are still routing.
- **bench2** (`evidence/bench2_snapshot_20261024_1142PDT.py`, from `M/backups/dev_20261024_133453.tar.gz`; the live `M/dev/bench2.py` was edited later): times several builds on the same routings in one process on a Modal H200. It uses many fresh routings per shape (normal and skewed popularity), CUDA-graph replays after an L2 flush, and paired ratios against SGLang. A **shared state** run gives every variant the first variant's prepared weights (`st = subs[0][2]`, from 05:43 UTC), and from 07:22 UTC equal tensors are shared across different versions too (`B2_SHARE_EQ`). Each build is loaded twice ("v112", "v112b") so the two copies give the noise floor.

## 1. Starting point

v83 on the judge, mean of its two runs (`evidence/judge_before_v83_e83db22b.txt`, 03's check): bs1 2.792x, bs8 2.176x, bs16 2.056x, bs32 1.694x, bs128 2.448x, bs256 2.664x.

The next ~20 builds followed the same loop: write, test, A/B against the parent in a Modal sweep or an early bench2, submit when ahead. The pieces:

| build | change | where |
|---|---|---|
| v84 | one-token combine: experts in groups of 4, all loads issued before any store (a store behind a load of the same address waits for it) | bs1 |
| v85 | main-kernel header re-laid out to free room for a 5th stage: barriers moved from offset 0 to 704 | bs2+ |
| v87 | one-token gate/up stages by bulk copies instead of tensor maps | bs1 |
| v90 | BF16 down outputs for full-k pairs (FP32 only for split ones); bs8/16 early start on by default | bs2+ |
| v92 | combine in groups of 2 pairs (CG = 2) for 1-k-warp configs | bs32+ |
| v98 | constant-bank warm-up at the start of the one-token kernel | bs1 |
| v99 / v100 | one-token down stage restructured; combine inputs prefetched in the last unit | bs1 |
| v103 | early-issue barrier (`eis`): consumers wait until the producer's early stages are issued | bs8/16 |

## 2. What we saw

**(a) The judge disagreed with the A/Bs at bs8 and bs16.** Judge speedups (`evidence/judge_*`):

| build | bs1 | bs8 | bs16 | bs32 | bs128 | bs256 | score |
|---|---|---|---|---|---|---|---|
| v83 (mean of 2) | 2.792 | **2.176** | **2.056** | 1.694 | 2.448 | 2.664 | 2.2731 |
| v92 (9dffea34) | 2.881 | 2.101 | 2.010 | 1.662 | 2.403 | 2.654 | 2.2476 |
| v98 (d4f55614) | 2.743 | 2.101 | 1.997 | 1.677 | 2.454 | 2.681 | 2.2419 |
| v100 (c00c4cc7) | 2.903 | 2.124 | 2.020 | 1.694 | 2.466 | 2.680 | 2.2773 |
| v103 (16782d95) | 2.857 | 2.146 | 2.038 | 1.681 | 2.459 | 2.687 | 2.2755 |
| mean v92-v103 | 2.846 | **2.118 (-2.7 %)** | **2.016 (-1.9 %)** | 1.679 (-0.9 %) | 2.446 | 2.676 | |

Four independent runs agreed: bs1 up 2 %, bs8/16 down 2-3 %. Locally, each step had looked neutral or positive at bs8/16. For example, the bs8/16 early start measured +0.7 % on Modal (`evidence/sweeps/s57_result.txt`, 4 rounds) and +1.2 % / +0.6 % in a judge lab on v87 (`evidence/labs/lab2.e4af893717b641d0.log`).

**(b) Two copies of the same build read 3.6 % apart.** bench2 at bs1, each variant with its own `prepare()` (05:37 UTC, `evidence/transcript_bench2_placement_control.txt`):

| variant | separate weight copies | one shared weight placement (05:43 UTC) |
|---|---|---|
| a100 (v100) | x1.0000 | x1.0000 |
| b100 (v100 again) | **x1.0361** | **x1.0005** |
| v98 | x1.0303 | x0.9859 |
| v99 | x1.0397 | x0.9912 |
| v92 | | x0.9845 |

With separate copies, the identical second copy of v100 was 3.6 % "faster" than the first, and v98/v99 looked 3-4 % faster than v100. With one placement, the same builds were 0.9-1.4 % slower than v100, and v92 → v100 is a real +1.6 % at bs1.
- The bias depended on which weight copy a variant got (physical placement), not on its code. It was larger than most of the gains being measured.
- Every A/B before this point that loaded a separate copy per variant carried it: the Modal `sweep_list.sh` runs (one process per variant) and bench2 before 05:43 UTC.

**(c) With shared weights, v83 really was faster than v103 at bs8-32.** Same job, two copies each, against a warm-up filler (07:24 UTC, `evidence/transcript_bench2_v83_v103_bisect_v112.txt`):

| shape | v83 / v83b | v103 / v103b |
|---|---|---|
| bs1 | 0.9876 / 0.9718 | 1.0217 / 1.0332 |
| bs8 | **1.0253 / 1.0215** | 0.9977 / 1.0000 |
| bs16 | **1.0164 / 1.0169** | 1.0010 / 1.0014 |
| bs32 | 1.0104 / 1.0083 | 1.0000 / 1.0027 |

v103's one-token kernel was ~4-5 % faster at bs1; v83's main kernel was 2.3 % / 1.5 % / 1 % faster at bs8/16/32.

**(d) Bisect on one shared placement.** Every build relative to v103, bs8 / bs16 / bs32 (07:29 UTC, same file):

| build | bs8 | bs16 | bs32 |
|---|---|---|---|
| v83 | +1.6 % | +1.4 % | +0.7 % |
| **v85** (header moved) | **-3.9 %** | **-2.9 %** | +0.3 % |
| v87 | -2.3 % | -1.8 % | +0.4 % |
| v90 (early start on) | -1.3 % | -1.5 % | 0.0 % |
| v92 | -1.2 % | -1.5 % | +0.2 % |
| v103 without early start | -2.3 % | -1.6 % | 0.0 % |
| v103b (control copy) | -0.6 % | +0.2 % | -0.1 % |

v85 (header moved, plus a 5th ring stage) cost bs8 5.4 % and bs16 4.2 % against v83. v87, back at 4 stages with v85's header, was still 3.8 % / 3.1 % behind v83. The early start (v90) and the early-issue barrier (v103) won some of it back on that layout, which is why each looked like a gain.

**(e) What in v85 mattered.** On top of v103 (07:30-07:50 UTC, same file):
- v109 restored only the 8 KB gap v83 left before the ring: no change (v109 and a size-only control both within ±0.2 % of v103).
- v110 restored v83's header offsets: bs8 +0.5 %, bs16 +0.1 %.
- v111 restored v83's exact layout, header and gap: bs8 +0.6 %, bs16 +0.5 %.
- **v111 without the early start: bs8 +2.1 %, bs16 +1.7 %**, equal to v83 (v83 / v83b +1.9 / +1.7 %, +1.5 / +1.5 %).

With v83's layout the early start costs 1.3-1.5 %; with v85's it had gained 1-2 %. That became v112.

**(f) Later on 10-24 the header effect was pinned to one field.** A sweep of 6 header layouts on v132 (`HL_*` defines; 16:05 UTC, `evidence/transcript_header_layout_sweep_v132.txt`), relative to the current layout L0 (bars at offset 0):

| layout | bars at | bs8 | bs16 | bs32 | bs128 |
|---|---|---|---|---|---|
| L1, L3, L5 (other fields moved) | 0 | -0.1 to +0.1 % | -0.1 % | 0 to +0.2 % | -0.2 to -0.4 % |
| L2 | 192 | **-3.3 %** | **-3.1 %** | **-2.1 %** | -0.8 % |
| L4 | 256 | **-3.5 %** | **-3.2 %** | **-2.1 %** | -0.7 % |
| L0b (control) | 0 | +0.2 % | -0.1 % | +0.1 % | 0 |

Every layout with the ring mbarriers at dynamic-shared-memory offset 0 performs the same; any layout with them elsewhere is ~3 % slower at bs8-32, wherever the other fields go. The mechanism is unknown (`M/NOTES_status.md`, "Header layout rule").

**(g) Re-tuning on shared weights moved two knobs** (`evidence/transcript_bench2_v114_v119.txt`):
- bs ≤ 32 config: (8,2,1), 16-token tiles with 2 groups, beats (8,4,2) by 1.5 % at bs32 (x1.7167 vs x1.6912) and by 0.4 % over nearby sizes 20-32, where (8,2,2) read +0.7 %. At bs32 the others lose: (8,4,1) x1.6840, (8,2,2) x1.7099, (8,1,2) x1.6593.
- One-token ring depth `NS1`: 4 beats 5 by 1.5-2.3 % in three interleaved copies (x1.0150 / 1.0229 / 1.0184 vs 0.9970 / 0.9981 / 0.9962); `NS1 = 3` read +0.9 % in one run.
- Confirmed already best: bs8/16 (4,1,1) (others -3 to -7 %); bs128 (8,4,1) ≈ (8,8,1); bs256 (8,8,1); `MOE_WGMIN 5`; `MOE_SPLW 1.0`; main-kernel NS 4 (NS 3: bs8 -4.8 %, bs16 -4.3 %).

## 3. Diagnosis and projection

- **Method.** An A/B in which each build gets its own weight copy carries a placement bias of a few percent. Pre-fix A/Bs (before v101) are unreliable at the 1-3 % level (`mem:okc-moe-findings.md`, "A/B method"). The fix: one placement shared by every variant, two copies of each build as a noise control, and fillers around them.
- **Kernel.** The bs8/16 regression was v85's header layout (about -4 to -5 %), partly masked by the early start (+1-2 % on that layout only). Restoring v83's layout and turning the early start off at bs8/16 recovers v83's main kernel. v103's faster one-token kernel (+4-5 % at bs1) and v90/v92's large-batch changes stay.
- **Projection at the time** (transcript, 07:28 UTC): v83's bs8/16/32 plus about +2 % at bs1 gives ~2.29x (~5.87 ms).

## 4. Change

`change.diff` is v83 → v119. Per piece, measured with shared weights unless noted:

| piece | what | shapes | gain (same job) | judge | source |
|---|---|---|---|---|---|
| one-token kernel, v84-v100 | grouped combine loads before stores, bulk-copy gate/up stages, constant-bank warm-up (`warm_consts`), restructured down stage, combine inputs prefetched in the last unit | bs1 | v103 vs v83 **+4-5 %** at bs1 (2 copies each); v92 → v100 +1.6 % on its own | v100 bs1 2.903, v103 2.857 vs v83 2.79 | `transcript_bench2_v83_v103_bisect_v112.txt`, `transcript_bench2_placement_control.txt` |
| BF16 down outputs + 2-pair combine (v90, v92) | full-k down items store BF16 y (`yh`); the combine claims pairs in groups of 2 for 1-k-warp configs | bs32-256 | v112 vs v83: bs128 +0.5 %, bs256 +1.0 % | | `transcript_bench2_v83_v103_bisect_v112.txt` (07:50 UTC) |
| early-issue barrier (v103) | `eis` mbarrier at the start of the consumers | bs8/16 | +1.3 % over v100 at bs8/16, on v85's layout | | transcript, 06:19 UTC |
| **v112: v83's header layout, no bs8/16 early start** | bars at offset 0, StageInfo at 128, `rdy`/`rtd` at 672/688, the 8 KB gap before the ring; `EARLY` only with `MOE_EARLY_ALL` | bs8/16/32 | vs v103: **bs8 +2.2 %, bs16 +1.7 %**, bs32 +0.5 %; geomean x2.3686 / 2.3647 vs v103 2.3428 / 2.3522 (+0.8 %) and v83 2.3323 | **2.2904** (ae45204d) | same file, 07:50 UTC |
| **v118: bs ≤ 32 on (8,2,1)** | `BUCKETS` entry 32 → (8,2,1) | bs17-32 | **bs32 +1.6 %** (x1.0164 / 1.0163, controls x1.0000 / 0.9985) | 2.2879 (85892d49) | `transcript_bench2_v114_v119.txt` (09:48 UTC) |
| **v119: `NS1 = 4`** | one-token ring 4 × 36 KB (`moe1_4`) | bs1 | **+1.5-2.3 %** (three interleaved copies) | **2.3119** (f3c48e94) | same file (09:59 UTC) |

`build.py` keeps the code paths the bisect cleared: v87's bulk-copy one-token gate/up, v90's BF16 y, v92's 2-pair combine. The early start, and with it v103's `eis` barrier, now compiles only into the (2,1,1) config, which M = 1 no longer reaches (`ONE_TOKEN = True`), so neither runs at the scored shapes.

Tried in this window and dropped (shared weights, two copies each, unless noted):

| idea | build | result | source |
|---|---|---|---|
| cross-stage operand prefetch (next stage's words in registers) | v114, v116 | v114: **bs8 -10 %, bs16 -10 %, bs32 -7 %, bs128 -5.4 %, bs256 -4.2 %** (it also spilled: 96 B stack, 40 B spill stores). v116 (boundary inside the last chunk): bs8/16 -10 %, bs32/128/256 -2.4 to -3.2 %. Both lost the compiler's own 2x unroll of the 1-group loop | `transcript_bench2_v114_v119.txt` (08:39, 08:44, 09:10 UTC) |
| hand 2x-unrolled stage loops | v117 | bs32 -1.5 to -3.8 %, bs256 -0.6 to -0.8 % | same file (09:17 UTC) |
| balanced item rounds at bs8/16 | v115 | -0.2 to +0.1 % | same file (09:01 UTC) |
| claim-ahead combine | v104 | bs32 -2.7 %, bs128 -1.9 % | transcript, 06:28 UTC |
| global-barrier elementwise combine / static combine after a barrier | v105 / v106 | -4 % (x2.1168 vs 2.2044) / -1.8 % | transcript, 06:46 and 06:56 UTC |
| wgmma for 3-4-group tiles in the 32-token configs | v107 | -1.7 to -2 % at bs32/128 | transcript, 07:03 UTC |
| interleaved tile order | v108 | -0.3 % | transcript, 07:09 UTC |
| k-half splits of the last down items at bs8/16 | v101 | -4.3 % | transcript, 05:51 UTC |
| main-kernel ring of 3 stages | v119 knob | bs8 -4.8 %, bs16 -4.3 % | same file (10:03 UTC) |
| row-half splits at bs8/16; weights issued before the gather release (`MOE_GEARLY2`); SSQ fast combine everywhere | v88, v89, v86 | -2.3 % (H100); -1.0 %; bs128/256 +4 to +8 % time (H100) | `evidence/sweeps/s54_result.txt`, `s55_result.txt`, `s51_result.txt` |
| background combiner | v95 | -10 to -12 % | `evidence/sweeps/s62_result.txt` |

## 5. Result

| check | result |
|---|---|
| v112 validation, all six shapes, shared weights | vs v103: bs1 +2.5 / +1.5 % (v103b +2.1 %, noise), bs8 +2.2 / +1.9 %, bs16 +1.7 / +1.6 %, bs32 +0.5 %, bs128 +0.3 / +0.5 %, bs256 +0.3 / +0.2 %. v83 in the same job: bs8/16 within 0.6 % of v112, bs128/256 0.5 / 0.9 % behind it |
| v118 validation | vs v112: bs32 +1.6 %, other shapes within the control spread (bs1 controls themselves 2.7 % apart) |
| chained per shape vs v83 (derived) | bs1 ≈ +7 % (+4.9 % v103 vs v83 × +1.9 % NS1 = 4), bs8 +0.5 %, bs16 +0.5 %, bs32 +1.6 %, bs128 +0.5 %, bs256 +1.0 % → geomean ≈ **+1.8 %** |
| correctness | 5/5 judge tests in all scored runs. Local tests rel-L2 3.27e-3 (bs4), 3.03e-3 (bs33), 3.20e-3 (bs257), 2.65e-3 (bs128 skewed), 2.47e-3 (bs1); up from 2.5e-3 because full-k down outputs are stored in BF16 since v90; limit 1.5e-2 (`evidence/transcript_bench2_v83_v103_bisect_v112.txt`) |
| SASS | the v112 and v118 validation jobs grepped the ptxas log for C75xx (wgmma serialization) and printed nothing (transcript, 07:50 and 09:48 UTC); v119 changes only the host constant `NS1` |
| judge | v83 2.2731 → v112 **2.2904** → v118 2.2879 → v119 **2.3119** |

Judge per case, speedup (`evidence/judge_*`):

| shape | v83 (mean of 2) | v112 (ae45204d) | v118 (85892d49) | v119 (f3c48e94) | v119 vs v83 |
|---|---|---|---|---|---|
| bs1 | 2.792 | 2.865 | 2.876 | **2.967** (26.32 µs) | +6.3 % |
| bs8 | 2.176 | 2.181 | 2.187 | 2.189 | +0.6 % |
| bs16 | 2.056 | 2.059 | 2.041 | 2.074 | +0.9 % |
| bs32 | 1.694 | 1.686 | 1.690 | 1.695 | +0.1 % |
| bs128 | 2.448 | 2.474 | 2.455 | 2.480 | +1.3 % |
| bs256 | 2.664 | 2.690 | 2.692 | 2.697 | +1.2 % |
| **score** | 2.2731 | 2.2904 | 2.2879 | **2.3119** | +1.7 % |

**Read honestly:**
- **v112 restored bs8/16 on the judge** (2.181 / 2.059 vs v83's 2.176 / 2.056) and kept bs1 at 2.865. It landed exactly on the projection: 2.2904 vs ~2.29 (5871.6 µs vs ~5.87 ms).
- **bs32:** v118's +1.6 % at bs32 never showed in one judge reading (1.686 → 1.690 → 1.695). It is within the ±1 % single-run spread per shape.
- **bs1:** v119's +3.2 % over v118 (2.876 → 2.967) is larger than the +1.5-2.3 % the A/B gave `NS1 = 4`. The next scored build (v122, same bs1 kernel) read bs1 2.842 (`R/index/judge_runs.tsv`, job 705329f7), so v119's bs1 reading was a good draw by ~2-3 %.
- **Overall:** the chained A/B estimate (+1.8 %) and the judge (+1.7 %) agree, but per shape they disagree in opposite directions: the judge read bs1 high and bs32 low (inferred from the A/Bs above).

**vs projection:**
- The v112 step matched its projection (2.29x).
- v118 and v119 were each projected from shared-weight A/Bs at +0.27 % and +0.3 % geomean (derived from +1.6 % bs32 / 6 and +1.9 % bs1 / 6). Together that is +0.6 %, against a judged +0.9 % over v112.

## 6. Takeaways, and what came next

- **The A/B harness had a bias larger than the effects it was measuring.** Separate weight copies made identical builds read 3.6 % apart. From here on every A/B shared one placement (`B2_SHARE_EQ`), carried two copies of each build as controls, and was read only against those controls.
- **Bisect against the last judged-good build, not the parent.** Each of v84-v103 was a gain over its parent on a biased measurement, and an early start that helped only on the regressed layout hid the regression. Restoring v83's layout and removing the early start at bs8/16 recovered 2 % at bs8/16 with no new code.
- **Shared-memory header rule:** keep the ring mbarriers at dynamic-shared-memory offset 0. Moving them costs ~3 % at bs8-32 for unknown reasons (6-layout sweep). Later layout changes respected it.
- **Re-tune knobs that were chosen with a biased A/B.** Two of them moved (bs32 config, `NS1`); the rest held.
- **What came next** (`../05_wgmma_heavy_tiles_async_epilogue/`): the large batches. At bs128/256 the consumers are compute-bound: the producer waits 47-66 µs per call for free slots, while consumers wait only 10-11 µs for data (`M/NOTES_status.md`, "Measured facts"). Milestone 05 added an asynchronous epilogue on warps 10-11 and wgmma for 3-4-group tiles.

Raw notes:
- `notes/NOTES_status.md`: status notes written on 10-24 at v119, with the measured facts, the header rule and the tried-and-rejected list; its last section covers v138-v148, milestone 05's work.
- `M/ROOFLINE.md`: the 10-24 roofline with per-shape floors (cited by path).
- `notes/Improvements.txt`: the improvements log; its first line is v119.
- `mem:okc-moe-versions.md` and `mem:okc-moe-findings.md` (cited by their original paths; their copies stay private).

## Reproduce

```bash
cd muse_moe_w4a16/tools
python3 check_mt.py ../04_shared_weight_ab_header_layout/build.py 1 4 8 16 17 32 33 128 256 257
# shared-weight same-job A/B vs milestone 03, two copies each
P=../03_fused_one_token_kernel/build.py; N=../04_shared_weight_ab_header_layout/build.py
B2_SHARE_EQ=1 B2_SEEDS=jm_seeds.jseeds python3 bench2.py - --reps 100 --shapes 1,8,16,32,128,256 \
  --variants "a1|$P|;c1|$N|;a2|$P|;c2|$N|"
```

The same experiments with the 10-24 bench2 (from `M/`, on a Modal H200 through `dev/modal_run.py`; `B2_OWN_STATE=1` restores the old per-variant weight copies):

```bash
# placement control: the same build twice, own copies vs one shared copy
B2_OWN_STATE=1 python3 dev/bench2.py - --variants 'a100|dev/moe_v100.py|;b100|dev/moe_v100.py|' --reps 300 --shapes 1
python3 dev/bench2.py - --variants 'a100|dev/moe_v100.py|;b100|dev/moe_v100.py|' --reps 300 --shapes 1
# bisect with one placement across versions
B2_SHARE_EQ=1 python3 dev/bench2.py - --reps 150 --shapes 8,16,32 \
  --variants 'v103|dev/moe_v103.py|;v83|dev/moe_v83.py|;v85|dev/moe_v85.py|;v87|dev/moe_v87.py|;v90|dev/moe_v90.py|;v92|dev/moe_v92.py|;v103b|dev/moe_v103.py|'
# header-layout sweep: dev/moe_v132.py with HL_* defines (see evidence/transcript_header_layout_sweep_v132.txt)
```

Originals: `M/dev/moe_v84.py` … `moe_v122.py`, `M/dev/moe_v132.py`; judge results `M/profiles/submission.{9dffea34,d4f55614,c00c4cc7,16782d95,ae45204d,85892d49,f3c48e94}*`; early bench2 outputs `M/dev/sweeps/b2_*.txt`; judge labs `M/dev/checks/profiles/lab2.*`, `lab3.22a45ce6*`.
