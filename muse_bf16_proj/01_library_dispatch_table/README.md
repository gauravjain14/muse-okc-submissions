# 01: a per-shape dispatch table over libraries, Triton and TinyGEMM (the Codex-era baseline)

| | before (SGLang) | after (01) |
|---|---|---|
| build | `tools/baseline.py` = SGLang's projections (md5 1a35b94b): cuBLAS `bf16_gemm_dispatch` for the four BF16 layers; router FP32 `F.linear` up to 8 tokens, two BF16 hi/lo GEMMs above | `build.py` = Codex `candidate_v2.py` (md5 ef833cda, 80,153 B) |
| judge | 1.0 by definition | **1.2471** [1.2284, 1.2658] (job 70a24525, check). Its predecessor `candidate_v1.py` (md5 c5159994) read 1.2417 (job e94b1b5d, check) |
| real effect | | about +25 % over SGLang; v1 to v2 (+0.43 %) is inside the ±1.5 % interval |
| shapes changed | | all 30 (one table entry per (m, layer)); the gain sits in the six routers (1.37-3.85x) and the four m = 1 GEMVs (1.22-1.82x); the eight m = 128/256 BF16 shapes stay at 0.98-1.03x |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: code copied or adapted from other projects is marked in place with `SPDX-SnippetBegin` / `SPDX-SnippetEnd` comments (source, copyright, license, our changes); see `NOTICE` and `LICENSES/` at the repository root. The code is unchanged. The judged file has md5 `ef833cda`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

This milestone is the starting point of the ladder. It was built on 2026-10-27 (07:13-08:27 PT) by a Codex agent team ("the Codex era"), before the Claude workstreams that produced milestones 02-05. Sources: the Codex coordinator log `notes/codex_plan.md` (= `muse-opt-codex/muse_bf16_proj/docs/plan.md`) and the ncu reports under `muse-opt-codex/muse_bf16_proj/profile/*/REPORT.md` (copied into `evidence/codex_profile_*_REPORT.md`).

## 1. Starting point

SGLang's per-shape times and the judge "floor" (speed of light: bytes at 4.8 TB/s or FLOPs at 989 TFLOP/s, whichever is longer) from the judge run of this build (`evidence/judge_after_candidate_v2_check_70a24525.txt`), µs:

| load | m | qkv_gate | o | pooler | unpooler | router | load sum (SGLang / floor) |
|---|---|---|---|---|---|---|---|
| decode 1 | 1 | 14.03 / 7.87 | 6.77 / 3.50 | 21.34 / 13.99 | 7.01 / 1.75 (1024-row slice) | 10.40 / 1.75 | 59.55 / 28.86 |
| decode 16 | 16 | 13.71 / 7.93 | 6.69 / 3.56 | 21.38 / 14.06 | 18.74 / 14.06 | 14.56 / 1.81 | 75.08 / 41.42 |
| decode 32 | 32 | 14.08 / 8.00 | 6.66 / 3.62 | 22.03 / 14.14 | 18.94 / 14.14 | 15.36 / 1.86 | 77.07 / 41.76 |
| verify 1 | 8 | 14.08 / 7.90 | 6.53 / 3.53 | 21.60 / 14.02 | 6.78 / 1.76 (slice) | 26.02 / 1.78 | 75.01 / 28.99 |
| verify 16 | 128 | 16.86 / 8.42 | 7.42 / 3.99 | 24.96 / 14.64 | 19.87 / 14.64 | 16.26 / 2.21 | 85.37 / 43.90 |
| verify 32 | 256 | 19.58 / 9.77 | 9.66 / 4.48 | 30.22 / 17.37 | 23.94 / 17.37 | 17.30 / 2.68 | 100.70 / 51.67 |

(Load sums computed from the per-case rows.) Up to m = 128 every shape is memory-bound: the arithmetic intensity is about 1 FLOP/byte at m = 1 and 112-122 FLOP/byte at m = 128, below the 206 FLOP/byte ridge of 989 TFLOP/s over 4.8 TB/s. At m = 256 the BF16 layers reach 200-234 FLOP/byte and their floor becomes the tensor bound: qkv 9.77 µs against 8.98 µs of bytes, pooler/unpooler 17.37 against 15.29 (`evidence/codex_roofline_initial.json`; ridge computed). SGLang sits 1.3-2.2x over the floor on the four BF16 layers (3.9-4.0x on the two 1024-row unpooler slices) and 5.9-14.6x over it on the routers (ratios computed from the table).

## 2. What we saw

The Codex team built v2 at 08:27 PT. Items (a), (b, first row) and the first row of (c) were measured before that; the rest are the follow-ups of 08:45-10:12 PT that tried to beat the table and did not (report mtimes, `notes/codex_plan.md` section times). They are kept here because they set the direction for 02.

**(a) A Triton GEMM at m = 16 was memory-concurrency bound, not math bound** (`evidence/codex_profile_triton-m16-pooler-unlocked_REPORT.md`, ncu, pooler m = 16):

| metric | cuBLAS | Triton `matmul_transpose` |
|---|---|---|
| main kernel | 21.76 µs | 22.50 µs |
| split-K reduction | 4.26 µs | 2.43 µs |
| DRAM read | 67.39 MB at 3.10 TB/s | 67.38 MB at 3.00 TB/s |
| achieved occupancy | 14.1 % (96 CTAs) | 12.0 % (256 CTAs) |
| tensor-core active | 5.3 % | 5.1 % |

Long-scoreboard stalls sat on the weight load (1,274 of the source samples). The report ranked three levers: more concurrent work, TMA loads, and fewer weight bytes via a lossless format. The first two were swept: BK128/SK4 reached 1.059x and TMA loads 1.063x on this shape, and larger token counts regressed.

**(b) Lossless compression read fewer bytes but lost on decode cost.** Every codec was exact, and every one was slower than plain BF16:

| codec / kernel (pooler) | time | DRAM | why it lost | source |
|---|---|---|---|---|
| Triton block codec (8-bit sign+mantissa plane + 4-bit exponent deltas), m = 8 | 102.56 µs vs cuBLAS 21.25 | 52.8 MB (−21.4 %) at 0.515 TB/s | 254 registers/thread; decoder instructions | `codex_profile_compressed-m8-pooler-initial_REPORT.md` |
| native CUDA fixed-plane GEMV, m = 1 | 40.32 µs (ncu), 41.12 µs paired | 57.1 MB at 1.416 TB/s | first use of the wide loads and dependent escape loads dominate stalls | `codex_profile_fixed-native-m1-pooler-initial_REPORT.md` |
| + aligned fast path and prefetch | 35.31 µs (m = 1), 69.81 µs (m = 8) | | still slower than the checkpoint | `notes/codex_plan.md` (09:38) |
| fixed-slot block codec, m = 1 | 25.09 µs (ncu), 27.42 µs paired = 0.9757x | 59.1 MB at 2.36 TB/s | integer/logic ALU at 67 %; exponent-plane first-use stalls | `codex_profile_block-native-m1-pooler-initial_REPORT.md` |

A read-only probe showed the opposite potential: sign rotation plus separated byte planes read about 8-9 % faster than plain BF16 (`notes/codex_plan.md`, 09:05), and the bytes do shrink by about 22 %. The loss was in the decoders as written, not in the idea.

**(c) Custom large-m GEMMs lost on their split-K reduction** (pooler m = 256, ncu):

| design | main kernel | reduction | paired graph result | source |
|---|---|---|---|---|
| Triton TMA, host descriptors | 40.99 µs vs cuBLAS 31.01 | | slower | `codex_profile_tma-m256-pooler-initial_REPORT.md` |
| 256-token tile (`wgmma.m64n256k16`) | 43.23 µs | 5.02 µs | slower; occupancy 6.25 % | `codex_profile_tma-m256-pooler-token256-v2_REPORT.md` |
| adaptive 8-warp TMA, 128x128x64, split 2, 6 stages | **26.98 µs** vs 31.94 | 5.06 µs | 36.96 µs = 0.9444x | `codex_profile_tma-m256-pooler-adaptive8-v3_REPORT.md` |
| native producer + 2 consumer warpgroups, wgmma m64n128 | **26.27 µs** vs 31.39 | 6.82 µs | 1.00024x | `codex_profile_native-wgmma-m256-pooler-initial_REPORT.md` |

Both custom main loops beat cuBLAS's main kernel by about 5 µs, and both gave the 5 µs back in a separate FP32 split-K reduction kernel.

**(d) A cluster GEMM at m = 8 only tied.** Its main kernel took 20.64 µs against cuBLAS's 21.34, at the same 67.3 MB (`codex_profile_cluster-m8-pooler-initial_REPORT.md`). Paired, it reached 24.61 µs = 1.103x, which `notes/codex_plan.md` (09:05) called "essentially unchanged", with no material integrated win.

**(e) The router had the most room.** SGLang's router uses FP32 `F.linear` at m ≤ 8 (a 25-26 µs path at m = 8) and two BF16 GEMMs plus two reductions plus an add above 8 tokens. Codex's interleaved hi/lo router packs the BF16 hi and lo halves of each FP32 weight into alternating 128-element blocks, so one dot kernel does both. At m = 256 its dot kernel took 11.71 µs and its FP32 reduction 3.78 µs, at 1.075 TB/s and 19.3 % tensor activity, with weight-load stalls on top (`codex_profile_router-m256-interleaved-initial_REPORT.md`). Even so, it was 1.28-3.24x over SGLang locally (`evidence/codex_v2_selected_20261027.json`).

## 3. Diagnosis and projection

- With BF16 weights, every non-router shape up to m = 128 is bounded by streaming 8-67 MB from HBM (m = 256 sits at the tensor bound), and SGLang's cuBLAS runs 1.3-2.2x over the floor (section 1).
- The custom kernels the team could write that day either decoded too slowly (codecs) or reduced too slowly (split-K at m = 256). That left a per-shape choice among measured candidates.
- **Projection:** Codex's paired local bench gave 1.1708x for the v1 table (`evidence/codex_v2_selected_20261027.json`, `projected_speedup`) and 1.1861x for v2 (`notes/codex_plan.md`, 08:45). The routers were projected at 1.28-3.24x per shape, the m = 1 GEMVs at 1.16-1.52x, and m = 128/256 at 0.98-1.11x.

## 4. Change

`candidate_v2.py` is one file with a 30-entry table `_CHOICES[(m bucket, layer)] -> (family, config)`. `kernel_fn` buckets m into 1, ≤8, ≤16, ≤32, ≤192 and above, and calls one family:

| family | used for (scored shapes) | what it is |
|---|---|---|
| Triton `pr_gemv` | qkv, o, pooler, unpooler at m = 1 | one program per BN rows × full K, FP32 accumulate |
| TinyGEMM (TensorRT-LLM `tinygemm2`, see NOTICE) | qkv, pooler, unpooler slice at m = 8 | Hopper BF16 mma + TMA kernel, extended in-house |
| Triton TMA `tm_tma_matmul` + split-K reduce | qkv, pooler at m = 16/32 | host TensorDescriptors, BN 64, BK 128, split 4 |
| Triton `pa_packed_mm` / `pr_matmul_transpose` | o at m = 8/32 and m = 16 | packed-weight and swap-AB Triton GEMMs |
| `torch.mm` on a pre-transposed W | unpooler m = 16/32, o m = 128, pooler m = 256 | cuBLAS with a different layout heuristic |
| static cuBLASLt configs (`lt_run`) | qkv m = 128/256, unpooler m = 128 | fixed algo/tile/split-K picked from a sampled sweep |
| plain `torch.mm` | pooler m = 128, o and unpooler m = 256 | cuBLAS default |
| router: exact FP32 SIMT / interleaved hi+lo dot + reduce / `dot` + `reduce` | m = 1 / m = 8, 128, 256 / m = 16, 32 | all FP32 accumulation; hi+lo only above 1 token |

v1 to v2 (`notes/codex_plan.md`, 08:05-08:45): v2 added one-token GEMV configurations, interleaved router paths and selected static cuBLASLt choices. It passed 89 fresh normal/outlier/nearby checks plus all 30 timing shapes.

What the Codex team tried that day, in order (`notes/codex_plan.md`; reports in `evidence/`):

| experiment | best result | kept? |
|---|---|---|
| Triton small-M split-K sweep (423 cases) | pooler m16 1.059x | yes (table entries) |
| Triton TMA loads (80-case sweep) | pooler m16 1.063x; larger m regressed | yes, at m = 16/32 |
| lossless block codecs (Triton, native CUDA fixed-plane, fixed-slot block) | 102.56 / 40.32 / 25.09 µs vs ~21 µs | no |
| hardware byte planes (sign rotation) | read-only +8-9 %, lost once decoded | no |
| cluster GEMM, split-K in one launch | m8 pooler 1.103x local, no integrated win | no |
| TMA / native wgmma at m = 256 | main −5 µs, reduction +5-7 µs | no |
| interleaved hi/lo router | 1.28-3.24x | yes |
| expanded-activation router + one library GEMM | m128 12.51 µs, m256 14.72 µs, slower | no |

## 5. Result

| check | result |
|---|---|
| local, Codex paired bench | v1 1.1708x, v2 1.1861x (`notes/codex_plan.md`) |
| correctness | 9/9 judge tests; 89 fresh normal/outlier/nearby cases + 30 timing shapes (`notes/codex_plan.md`, 08:05) |
| judge | v1 **1.2417** (job e94b1b5d), v2 **1.2471** [1.2284, 1.2658] (job 70a24525); both checks, kept off the board (`runs/official_v2_20261027.json`: `"mode": "check"`) |
| per load (judge v2, computed) | decode 1 1.458, decode 16 1.217, decode 32 1.201, verify 1 1.516, verify 16 1.100, verify 32 1.059 |
| **judge, read honestly** | v1 to v2 is +0.43 %, inside the ±1.5 % interval. Shapes v2 did not change also moved: verify 32 unpooler (cuBLAS in both) went 24.50 → 23.66 µs and SGLang's own time 24.54 → 23.94, while verify 1 pooler (TinyGEMM in both) went 18.31 → 18.78 µs (`evidence/judge_candidate_v1_check_e94b1b5d.txt` vs `evidence/judge_after_candidate_v2_check_70a24525.txt`). The real effect of v1 → v2 is "no worse". The level, about 1.24-1.25, is robust. |
| vs projection | Codex's local bench under-predicted the judge by 5-6 % (1.1708 → 1.2417, 1.1861 → 1.2471, computed). A later re-bench of v2 with the Claude harness (`claude/bench.py`) gave 1.2765 local, with judge = local + 1.0 µs per shape (milestone 02, `evidence/calib.log` there). |

Kernel-level picture in the judge profile of v2: the m = 1 Triton GEMVs run at 7.9 / 4.0 / 16.5 / 2.9 µs of GPU time (qkv / o / pooler / unpooler slice) against floors of 7.87 / 3.50 / 13.99 / 1.75. The TMA qkv/pooler shapes at m = 16/32 pay a separate 1.1-1.3 µs split-K reduction kernel. The m = 128/256 shapes are cuBLAS `nvjet` kernels plus, for qkv, a cuBLASLt `splitKreduce_kernel`.

## 6. Takeaways, and what came next

- **The m = 1 GEMVs were already near the BF16 byte floor.** Below it, the only lever is fewer bytes. The codecs here lost because of how they decoded: Triton block decoders at 254 registers, dependent escape loads on the critical path, 67 % integer ALU. They did not lose because the format could not pay.
- **At m = 128/256, a faster main loop is worth nothing if the split-K reduction runs as a second kernel.** Two independent designs gained 5 µs in the loop and lost 5-7 µs in the reduction.
- **The router was the outlier:** 5.9-14.6x over its floor in SGLang, and v2 still 2.3-4.7x over it (computed from the judge rows).
- **Next, 02:** a Claude workstream recalibrated the bench against this judge run. It measured the 12-bit streaming floor, and rebuilt the m ≤ 16 shapes on a lossless 12-bit format that decodes in registers in a few integer ops per weight. A dedicated router worker went alongside.

## Reproduce

Needs an H200 and the SGLang image of the challenge (torch 2.14 + CUDA 13.0, Triton 3.8, `sglang` for the baseline). `build.py` compiles its CUDA extensions (TinyGEMM, cuBLASLt wrapper) with `torch.utils.cpp_extension.load_inline` (nvcc + ninja) during `prepare()`, which took 74.2 s locally (milestone 02 `evidence/calib.log`). It reads no files outside itself.

```bash
cd muse_bf16_proj/tools
python3 check.py ../01_library_dispatch_table/build.py --graph          # judge tests + scored + nearby m, eager and graph
python3 bench.py --cand v2=../01_library_dispatch_table/build.py --reps 30   # paired with SGLang; prints per-shape, per-load, judge-projected
```

Originals (relative to ``):
- build: `muse-opt-codex/muse_bf16_proj/candidate_v2.py`; v1: `muse-opt-codex/muse_bf16_proj/candidate_v1.py`
- judge results: `muse-opt-codex/muse_bf16_proj/profiles/candidate_v{1,2}.*.{txt,json,log}`
- Codex log and run data: `muse-opt-codex/muse_bf16_proj/docs/plan.md`, `muse-opt-codex/muse_bf16_proj/runs/`
- ncu reports: `muse-opt-codex/muse_bf16_proj/profile/<run>/` (`REPORT.md` + `reports/` + `analysis/`; the binary reports stay there)
- TinyGEMM upstream copy: `muse-opt-codex/muse_bf16_proj/upstream/tinygemm2/`
