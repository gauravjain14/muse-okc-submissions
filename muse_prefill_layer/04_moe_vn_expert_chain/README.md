# 04: the native expert path "vN chain" (routing, gather and expert-kernel issue trims)

| | before (03) | after (04) |
|---|---|---|
| build | `03_expq_fp8_exponential/build.py` = `cand_full11.py` (md5 9d946f1a, 358,480 B) | `build.py` = `cand_full17.py` = `submits/submit2_full17.py` (md5 09d36a74, 409,762 B) |
| judge | 1.8310 (check, job 6da1e379) | **1.8329** (scored, job 2a57ce70). In between: full12 1.8010 (scored, c6e0ecdb), full14 1.8030 (check, fa01f258) |
| real effect | | same-run local bench (`evidence/bench_era1.py`) **+3.7 %** compounded over four builds; turn 1 +2.3 %, turn 4 +1.9 %; the 16K part (+6.9 %) is not trustworthy (see Result). The judge shows +0.1 % because the "before" reading was a high draw |
| shapes changed | | every case (expert path at all T; attention ping-pong off; bulk-copy boundaries for >= 2048 rows = cases 2-7) |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: code copied or adapted from other projects is marked in place with `SPDX-SnippetBegin` / `SPDX-SnippetEnd` comments (source, copyright, license, our changes); see `NOTICE` and `LICENSES/` at the repository root. The code is unchanged. The judged file has md5 `09d36a74`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

Terms used below. **Expert path**: everything between the pooler and the unpooler: the pre-expert RMSNorm plus top-8 routing kernel (`pnr*`), the activation gather, the fused NVFP4 expert kernel (`moe_<cfg>`, both expert GEMMs and SwiGLU) and the combine (`_native_finish`, Triton). **vN**: the expert worker's numbered expert-path versions, `muse-opt/muse_prefill_layer/cand_dense2_moe_vN.py`, each one a full layer on the FA3 dense base so it can be benched alone. **full N**: the composed layer builds (`cand_fullN.py`) that carry our own attention (`attn8`) on top. **Item**: one (expert tile, k range) unit of work in the persistent expert kernel; **tile**: up to 128 tokens routed to one expert. **Roofline** (`evidence/roofline.py`): per-kernel max(bytes / 4.8 TB/s, FLOP / 989 TF/s) from an nsys trace, and the "over" column = measured minus roofline.

## 1. Starting point

full11 (milestone 03) ran expert path v7. Its nsys roofline table (`evidence/roofline_full12.txt`; full12 = full11 + one attention flag, same expert path) per case:

| case | span µs | expert kernel µs (roofline eff.) | routing `pnr` µs | separate gather µs | expert path over its roofline |
|---|---|---|---|---|---|
| 0 (T 753 sliding) | 599.3 | 315.7 (44 %) | 12.9 | (in kernel) | 177.4 + 8.7 of 265.7 µs total over |
| 1 (T 753 full) | 814.0 | 294.0 (45 %) | 12.7 | | 160.3 + 8.6 |
| 2 (T 3012 sliding) | 1899.0 | 848.2 (67 %) | 29.5 | 35.5 | 284.0 + 12.8 + 16.6 of 533.2 |
| 4 (16K ctx0 full) | 9879.5 | 4118.5 (75 %) | 109.6 (`pnr2`) | 177.8 | 1022 + 19 + 74 of 2040 |

The judge's own per-kernel profile of the full11 check says the same: the expert kernel is 49.6 % of case 0 (294.9 µs) and 42.5 % of case 2 (809.1 µs), plus `pnr` 12.5 / 29.5 µs and `gather` 33.2 µs (`evidence/judge_before_full11_6da1e379.txt`, profiles 0 and 2).

## 2. What we saw

**(a) The expert path is the largest gap to the roofline in every case.** In full12's table the expert kernel alone is 67 % of case 0's over-roofline time and 53 % of case 2's; with routing and gather, 70 % and 59 % (`evidence/roofline_full12.txt`, "cost over roofline by kernel" lines). The earlier load-level estimate was that the experts' excess alone is 23.9 / 12.2 / 8.8 % of the turn-1 / turn-4 / first-turn loads ((private notes, not included) line 28, from `muse-opt/muse_prefill_layer/logs/roofline_full5.txt`). The routing kernel and the gather are pure byte movers but ran at 32 % (`pnr`, 753) and 53 % (`gather`, 3012) of their byte rooflines.

**(b) The expert kernel is issue/latency limited, not DRAM limited** (ncu of v9 at T = 753, `evidence/moe_ncu_v9.log`):

| metric | value |
|---|---|
| duration, SM clock | 301.3 µs at 1.71 GHz |
| DRAM / L2 throughput | 31.9 % / 48.0 % |
| issue slots busy | 33.5 %; no eligible warp 64 % of cycles |
| warps per scheduler | 2.48 active, 0.51 eligible; 168 registers, 15.6 % occupancy |
| top stalls (of 30,437 samples) | wait 5195, selected 4592, long_sb 4331, dispatch 3480, math 2735, barrier 2519 |
| top sampled opcodes | LOP3 4581, HGMMA 3110, IMAD 2491, BRA 2486, WARPGROUP 2445, HMUL2 2285, NANOSLEEP 1986, HMMA 1876, SHF 1789, **R2UR 909** |

LOP3/IMAD/HMUL2/SHF are the NVFP4 decode feeding the register-sourced wgmma (RS-wgmma) and the mma.sync paths. `R2UR` (moving a value from a regular to a uniform register) shows up 909 times: the wgmma descriptors were being rebuilt from per-thread registers every MMA. That observation is what v24b fixes.

**(c) Every expert-path change was measured twice:** expert kernel alone (`moe_bench`, MoE-only µs at T = 753 / 3012 / 16384, e.g. `evidence/moe_v9v10b.log`) and in the whole layer against SGLang in the same run (`evidence/bench_v*.log`, judge-style paired ratio per load). Only steps that won in the layer were kept.

**(d) The real routing has a long tail of tiny experts** (`evidence/moe_trace753_v9.log`, layer routing at T = 753): 24 tiles, of which the hot experts carry 563 / 205 / 202 / 146 / 139 routes and the rest 44, 36, 28, 19, 2, 2 and single tokens. Hot items take 43-44 µs of compute each, small items 22 µs regardless of size, and the small ones run at the end of each phase. This trace did not change the steps below; it drove the next milestone's mma.sync work (v28 / v29) and, much later, the wide/narrow split (milestone 10).

## 3. Diagnosis and projection

- At T = 753 the expert kernel sits at 44-45 % of its roofline with one third of issue slots busy. No single unit is saturated, so the levers are fewer instructions and fewer dependent round trips, not bandwidth.
- The routing and the gather sit on the critical path between the pooler GEMM and the expert kernel, and both run far below their byte rooflines. Folding the gather into routing and shortening the routing kernel is worth up to `pnr` + `gather` minus their rooflines: about 8.7 µs at case 0 and 29 µs at case 2 (roofline rows above). Score sensitivity is 0.042 %/µs at case 0 and 0.0126 %/µs at case 2 (`muse-opt-codex/muse_prefill_layer/research/harness/NOTES.md:138-141`), so roughly +0.4 % + 0.4 % (inferred).
- No single-number projection was written for the in-kernel items. Each was accepted only on a same-run layer gain.

## 4. Change

Four composed builds, each benched against its predecessor in the same run (`evidence/bench_full1{2,4,6,7}.log`):

| build | what changed | turn 1 | turn 4 | first turn | geomean |
|---|---|---|---|---|---|
| full12 | attention `ATT_PP 0`: consumer warpgroups no longer issue their wgmmas in turn (ping-pong off) | +0.22 % | +0.28 % | +1.76 % | +0.77 % |
| full14 | expert path v9 + `patch_bnd`: persistent bulk-copy boundary kernels (`bnd4_b1/b2`) from 2048 rows | +0.27 % | +0.56 % | +1.16 % | +0.68 % |
| full16 | attention epilogue issues all 16 gate loads before any store (the compiler could not reorder loads past the output stores) | +0.11 % | +0.16 % | +1.84 % | +0.69 % |
| full17 | expert path v14 → v24b (table below) | +1.71 % | +0.93 % | +1.93 % | +1.52 % |

The expert-path steps, each benched alone on the FA3 base (geomean ratio of the layer bench; `evidence/bench_v*.log`, MoE-only numbers from `muse-opt/muse_prefill_layer/logs/moe_v*.log`):

| step | mechanism | layer bench | what it taught |
|---|---|---|---|
| v9 | `MOE_XPU 1`: one stage per pipelined block for builds narrower than 512 tokens. Half the main-loop code (instruction-cache footprint); the in-flight wgmma group no longer crosses stage boundaries | +0.63 % (`bench_v9.log`); MoE-only vs v8 ×1.029 / 1.036 at 753, ×1.019 at 3012 (`moe_v9v10b.log`) | code size matters in a kernel with several instantiated loop variants |
| v14 | `pnr9`: the routing moves onto a ninth warp of the pre-expert-norm kernel (`pnr`); each top-k round's winner by one warp max-reduction (REDUX) of an order-preserving key instead of five shuffle-compare steps; PDL-triggers the expert kernel | +0.65 % (`bench_v14.log`) | |
| v15 | 16K: the router writes each local route's row straight into a pre-gathered, row-major [128 rows][4096] tile slot (one contiguous 8 KB store per route); the expert kernel's TMA tensor maps load stage chunks from the slot in the wgmma layout. The gather kernel disappears | +0.78 % (`bench_v15.log`) | |
| v16 | the same slots at T = 3012 | +0.29 % (`bench_v16.log`) | |
| v21 | 753: in-kernel gather in two passes (the first wave's tiles first), so the first stages wait only for the first pass | +0.22 % (`bench_v21.log`) | |
| v22 | combine inside the expert kernel (CTAs out of items take tokens from a counter) | +0.49 % in the layer bench, but MoE-only ×1.011-1.015 at 753, ×0.952 at 3012, ×0.930 at 16K (`bench_v22.log`, `moe_v22.log`) | **rejected**: the layer gain sat entirely on case 5 (+2.3 %), where the MoE itself got 7 % slower; the 16K layer bench was not trustworthy |
| v23 | `pnr9` in 160 threads (four warps normalize, the fifth routes): 7 CTAs per SM, one wave at 753 tokens. Built on v21, not v22 | +0.34 % (`bench_v23.log`) | |
| v24b | lane-0 `__shfl_sync` of the stage address: the compiler then knows it is warp-uniform and keeps the wgmma descriptors in uniform registers (no `R2UR` per wgmma) | +0.63 % (`bench_v24.log`) | the ncu `R2UR` count in 2(b) was the lead |

The v7 → v24b steps compound to +3.6 % on the FA3 base (computed from the ratios above). In the shipped build the routing kernels are `pnr5` (T = 753), `pnr5g` (2048-4096, writes the slots) and `pnr2y` (16K) (`build.py` lines 6046-6130, launch at 8286-8308). Dead branches tried in the same stretch are listed in 5 and in `notes/okc-prefill-moe-worker.md`.

## 5. Result

**Local, same-run layer bench** (`evidence/bench_era1.py`: one CUDA graph per case, flush, paired with SGLang in the same process, fixed router seed 728034927264498030; ratios computed from `evidence/bench_full1{2,4,6,7}.log`):

| load | full11 → full17, compounded |
|---|---|
| turn 1 (cases 0, 1) | **+2.3 %** |
| turn 4 (cases 2, 3) | **+1.9 %** |
| first turn (cases 4-7) | +6.9 %, not trustworthy |
| geomean | +3.7 % |

Why the first-turn number is not trustworthy: the same build read 1.925, 1.914 and 1.900 on the first-turn load in three runs (full14 in `bench_full14.log`, `bench_full15.log`, `bench_full16.log`), and two attention-only changes (full12, full16) each "gained" 1.8 % there, although attention is a few % of a 16K case. A position bias of the bench at the power-capped 16K shapes was found later the same day ((private notes, not included) line 52). The v22 row above is the same effect seen from the other side.

**Where the time went, nsys** (`evidence/roofline_full12.txt` → `evidence/roofline_full17.txt`, µs):

| case | routing + gather + experts, full12 | full17 | change |
|---|---|---|---|
| 0 | 12.9 + 0 + 315.7 = 328.6 | 6.2 + 305.9 = 312.1 | −16.5 |
| 1 | 12.7 + 294.0 = 306.7 | 6.3 + 273.4 = 279.7 | −27.0 |
| 2 | 29.5 + 35.5 + 848.2 = 913.2 | 33.1 + 820.9 = 854.0 | −59.2 |
| 3 | 29.5 + 34.0 + 833.3 = 896.8 | 32.1 + 813.1 = 845.2 | −51.6 |
| 4 | 109.6 + 177.8 + 4118.5 = 4405.9 | 133.0 + 4064.3 = 4197.3 | −208.6 |

(The 16K traces are back-to-back replays under nsys; case 5 moved the other way, 4315.6 → 4494.9, which is a clock-state artefact of that trace, inferred.)

**Correctness:** the 8 scored cases pass the reference check in every bench run, and the vN builds also pass the 5 judge tests (`evidence/bench_v24.log`); full17's errors are identical to full16's on every case (`evidence/bench_full17.log`); the judge run passed 5/5 tests.

**Judge, read honestly.** 1.8310 → 1.8329. Per case (`evidence/judge_before_full11_6da1e379.txt`, `evidence/judge_after_full17_2a57ce70.txt`):

| case | ours µs before → after | SGLang µs before → after | ratio before → after |
|---|---|---|---|
| 0 | 584.6 → 537.2 (−8.1 %) | 1094.7 → 998.4 (−8.8 %) | 1.873 → 1.859 |
| 1 | 818.4 → 810.4 | 1474.0 → 1480.1 | 1.801 → 1.826 |
| 2 | 1871.7 → 1839.8 | 3462.4 → 3361.4 (−2.9 %) | 1.850 → 1.827 |
| 3 | 2843.5 → 2808.5 | 4941.5 → 4978.9 | 1.738 → 1.773 |
| 4 | 9886.9 → 9640.1 | 18150.1 → 17570.0 | 1.836 → 1.823 |
| 5 | 9358.3 → 9251.7 | 17117.8 → 16979.1 | 1.829 → 1.835 |
| 6 | 11591.5 → 11734.0 | 21170.9 → 21873.4 | 1.826 → 1.864 |
| 7 | 13277.0 → 13388.3 | 24684.0 → 25250.1 | 1.859 → 1.886 |

- Our times fell on six of eight cases. On case 0, the most valuable case, SGLang also ran 8.8 % faster, which took the whole gain away. The judge draws a fresh router for every run (found later: `muse-opt-codex/muse_prefill_layer/research/claude/NOTES.md:8-10`), and both MoE implementations move with it. The judge's per-kernel rows show it: our expert kernel went 809.1 → 837.5 µs on case 2 and 854.1 → 729.6 µs on case 3 between the two runs, although the code change is the same on both cases. The routing kernel moved cleanly: `pnr` 12.5 µs → `pnr5` 6.5 µs (case 0), and `pnr` + `gather` 62.7 µs → `pnr5g` 32.0 µs (case 2).
- The "before" reading was high. Judge / local-bench ratios of the builds judged in this stretch: full10 0.981 (1.7359 / 1.7695, (private notes, not included) line 29), **full11 0.995** (1.8310 / 1.8396, `evidence/bench_full12.log`), full12 0.972 (1.8010 / 1.8537), full14 0.967 (1.8030 / 1.8647), full17 0.964 (1.8329 / 1.9005). A dedicated calibration check later the same day put the judge at about 0.963 × this bench, because SGLang runs 3-11 % faster on the judge node while ours gains only 1.5-6 % ((private notes, not included) line 41; `05_sk_only_attention_fused_unpool/evidence/judge_node_lab_plab1_4092c8e9.log`). At that calibration full11's real level is about 1.77 (1.8396 × 0.963, inferred), and full12, a strict superset, scored 1.8010.
- **Real effect:** the local turn-1 / turn-4 gains (+2.3 / +1.9 %) are the trustworthy part; with the first turn counted at zero the score effect is about +1.4 % (inferred), at the compounded geomean +3.7 %.
- **vs projection:** routing + gather moved by −6.0 µs (case 0) and −30.7 µs (case 2) in the judge's own rows, against the projected 8.7 / 29 µs. The in-kernel steps had no projection.

## 6. Takeaways, and what came next

- **Measure the expert kernel alone and in the layer, and keep only what wins in the layer.** v22 is the cautionary case: a layer gain that sat entirely on the noisiest load while the kernel itself got slower.
- **Read SASS-level counters (R2UR, decode LOP3/IMAD) before tuning schedules.** The cheapest win of the chain (v24b, one shuffle) came from an opcode count.
- **Judge readings at this stage were single draws with a ±2-3 % spread**; full12 and full14 (both strict improvements locally) read 1.80. Calibrate the bench against the judge before trusting a step (done the next day, milestone 05).
- **What came next (05):** the real-routing trace in 2(d) showed the small-tile items, which led to v28 (two 256-k chunks per mma.sync stage) and v29 (4-group tiles moved to wgmma) in milestone 05. The same milestone took the attention tile loop's code-size lesson further (a stream-K-only attention module) and fused the 16K unpooler with its boundary.

## Reproduce

All you need is torch and an H200. `tools/` holds the problem spec, `reference.py`, `checks.py`, `baseline.py` and the judge-faithful harness. The era-1 bench that produced the numbers above is `evidence/bench_era1.py` (it expects the era-1 directory layout; the replica below is its successor).

```bash
cd muse_prefill_layer/tools
python research/harness/check.py ../04_moe_vn_expert_chain/build.py --edges          # judge tests + 8 scored cases + nearby shapes, 2 replays each
python research/judge/replica.py --candidate before=../03_expq_fp8_exponential/build.py \
       --candidate after=../04_moe_vn_expert_chain/build.py --panel 11,22,33 --reps 8   # judge-faithful paired A/B over 3 router draws
```

Originals: `muse-opt/muse_prefill_layer/cand_full17.py` (= `submits/submit2_full17.py`, sha256 1b5c8dc2…), the expert-path chain `muse-opt/muse_prefill_layer/cand_dense2_moe_v{9,14,15,16,21,22,23,24b}.py`, logs in `muse-opt/muse_prefill_layer/logs/`, tools `muse-opt/muse_prefill_layer/dev/{bench,roofline,moe_bench,moe_ncu,moe_trace2}.py`. Judge results: `muse-opt/muse_prefill_layer/{checks,submits}/profiles/`.
