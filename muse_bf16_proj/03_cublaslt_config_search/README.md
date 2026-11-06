# 03: exhaustive cuBLASLt configuration search at m = 128/256, and 32-token register streaming (K1t) at m = 32

| | before (02) | after (03) |
|---|---|---|
| build | `02_.../build.py` = `int_v1.py` (md5 19dd35d6) | `build.py` = `int_v4.py` (md5 cf0feeb7, 207,880 B) |
| judge | 1.3112 (job 97b5d8aa, check) | **1.3482** [1.3279, 1.3684] (job 8bb51ba3, check) |
| real effect | | **+2.8 %** judge; same-run local judge-projected 1.3364 vs 1.2979 (+3.0 %) |
| shapes changed | | qkv / o / pooler / unpooler at m = 128 and 256 (all 8: searched cuBLASLt configs); pooler and unpooler at m 17-32 (scored m32: K1t) |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: code copied or adapted from other projects is marked in place with `SPDX-SnippetBegin` / `SPDX-SnippetEnd` comments (source, copyright, license, our changes); see `NOTICE` and `LICENSES/` at the repository root. The code is unchanged. The judged file has md5 `cf0feeb7`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

Terms used below:
- **cuBLASLt config**: the tuple a `cublasLtMatmul` call runs with: algorithm id, tile, stages, cluster shape, split-K, swizzle, custom option.
- **K1t**: K1s from milestone 02, built for 32-token x tiles.

## 1. Starting point

Judge times in `evidence/judge_before_int_v1_check_97b5d8aa.txt`, µs. Every m = 128/256 BF16 shape was still SGLang's library at parity, and m = 32 pooler/unpooler were Codex v2 paths:

| shape | SGLang | int_v1 | floor | int_v1 kernel |
|---|---|---|---|---|
| qkv m128 / m256 | 17.06 / 19.78 | 16.60 / 20.58 | 8.42 / 9.77 | cuBLASLt static pick (Codex) + `splitKreduce` |
| o m128 / m256 | 7.15 / 9.66 | 7.29 / 9.81 | 3.99 / 4.48 | cuBLAS |
| pooler m128 / m256 | 25.22 / 30.56 | 24.67 / 29.49 | 14.64 / 17.37 | cuBLAS |
| unpooler m128 / m256 | 19.82 / 24.08 | 18.96 / 24.26 | 14.64 / 17.37 | cuBLASLt pick / cuBLAS |
| pooler m32 / unpooler m32 | 22.02 / 18.88 | 19.68 / 18.47 | 14.14 / 14.14 | Triton TMA + reduce / cuBLAS |

Load speedups at m = 128/256 were 1.134 and 1.074 (computed from the per-case rows). These two loads were the lowest of the six.

## 2. What we saw

**(a) Two custom-GEMM workers lost to cuBLAS on every m = 128/256 shape.**

- **K2** (`notes/k2_NOTES.md`, final at 11:51): BF16 TMA + wgmma, pre-swizzled W stages, staged epilogue, parallel split-K. Against Codex v2 in the same run (local µs): qkv m128 17.10 vs 15.89, o m128 7.70 vs 6.32, unpooler m128 20.98 vs 17.77, pooler m256 32.40 vs 28.46. Its only tie was pooler m128 (24.45 vs 24.72). The per-CTA traces show where it went:
  - The first TMA stage lands 1.5-2.6 µs after entry.
  - The end-of-kernel split-K tail costs 4-7 µs.
  - The S = 1 main loops are L2→SM bound by x re-reads, about 75 GB/s per SM.
- **K2c** (`notes/k2c_NOTES.md`, final at 12:47): swap-AB wgmma with 12-bit W decoded into the A registers. It was slower still, e.g. qkv m128 23.30 µs and o m128 15.60 µs against Codex v2's 15.86 and 6.62. Its main loops were L2→SM ingress bound at 54-70 GB/s per SM. A `red.add` split-K drained at only about 2 TB/s.
- Both notes come to the same conclusion: cuBLAS is within about 20-40 % of the HBM floor at m128, and close to the compute floor at m256.

**(b) Other libraries in the image were no better.** `evidence/lib3.log` (13:12, local, paired with SGLang, all 30 shapes; variants from `claude/sub_lib*.py`) gave these geomeans over the six loads:
- `deep_gemm.bf16_gemm_nt`: 1.0005x.
- flashinfer `mm_bf16` with the cuDNN backend: 1.0184x.
- flashinfer's `cutlass` and `cute-dsl` backends raise `BackendSupportedError` on capability 90, and `flashinfer.tinygemm_bf16` does not exist in this version. Those three variants fell back to `torch.mm` and measured 1.0126-1.0219x, which is the spread of plain cuBLAS.

None of the libraries moved m = 128/256 by more than a few percent.

**(c) The library's own heuristic was the problem.** `lt_tune.py` (`evidence/lt_tune.py`) enumerates every `cublasLtMatmul` configuration: algorithm id × tile × stages × cluster shape × split-K × swizzle × custom option. Each one is validated by `cublasLtMatmulAlgoCheck`, 44,199 valid configs per layout and shape (`evidence/lt2.log`). Every candidate is timed judge-style: graph replay after a cold-L2 read. Best vs `torch.mm` (local µs, `evidence/lt2.log`, winners in `evidence/lt_128_256.json`):

| shape | torch.mm | best searched | layout | change |
|---|---|---|---|---|
| pooler m128 | 24.67 | **18.50** | nn | **−25 %** |
| pooler m256 | 28.51 | **22.34** | nn | **−22 %** |
| qkv m128 | 15.74 | 13.76 | tn | −13 % |
| qkv m256 | 17.92 | 16.26 | tn | −9 % |
| unpooler m128 | 18.72 | 17.54 | nn | −6 % |
| unpooler m256 | 23.17 | 22.34 | nn | −4 % |
| o m128 | 6.05 | 5.76 | tn | −5 % |
| o m256 | 7.87 | 7.55 | nn | −4 % |

The two pooler shapes moved most. For pooler m128, the heuristic choice is `nvjet_sm90_tst_64x64_64x13_2x1`: SGLang runs it too, at 26.18 µs in milestone 05's `evidence/nsys_int_v7.log`, and int_v1's plain cuBLAS call ran it at 25.5 µs in the judge profile. The searched config runs `nvjet_sm90_tst_128x32_64x10_1x1` (21.6 µs in the int_v4 judge profile): a 128 × 32 tile with 10 stages on a 1 × 1 cluster instead of 64 × 64, 13 stages, 2 × 1.

**(d) At m = 32, K1s's 16-token tiles were the wrong size.** K1t takes K1s's register-streaming kernel, adds a 32-token x tile (`_k1t_TOKBS = (8, 16, 32)`) and gives each layer its own `_k1t_CFG`. In `evidence/k1t7.log` (12:52, local µs, same run), pooler m32 measured: K1t 17.35, K1s 20.05, Codex v2 18.58. The int_v4 integration run (`evidence/int4.log`, `evidence/int4_bench.json`) confirms it: pooler m32 17.66 vs 18.72 and unpooler m32 16.64 vs 17.34.

## 3. Diagnosis and projection

- **At m = 128/256 the binding problem was the configuration, not the kernel.** For pooler m128 the heuristic's 64 × 64 / 2 × 1-cluster kernel is 25 % slower than the searched 128 × 32 / 1 × 1 one, on the same library.
- A custom GEMM would have to beat both that config and the split-K tail that K2 and K2c could not hide. That would need on-chip split-K (cluster-pair DSMEM) plus x multicast (`notes/k2c_NOTES.md`, "Why it cannot win as designed").
- **Projection, per shape:** `lt2.log` against `torch.mm`: pooler m128 and m256 −6.2 µs each, qkv m128 −2.0, qkv m256 −1.7, unpooler −1.2 / −0.8, o −0.3 / −0.3. int_v1 ran `torch.mm` on both pooler shapes. On qkv m128/256 and unpooler m128 it ran Codex's pinned cuBLASLt configs, and the qkv m256 pin turned out slower than `torch.mm` (int_v1 19.78 µs in `int4.log`). Plus K1t: pooler m32 −1.2, unpooler m32 about −0.7.
- **Projection for the integrated build:** judge-projected local 1.3364 vs 1.2979 for int_v1 in the same run (`evidence/int4.log`), i.e. **+3.0 %**.

## 4. Change

- **`_LT` table:** maps (m, layer) to (layout, searched settings) for all eight m ∈ {128, 256} BF16 shapes. `kernel_fn` checks it before every other path and calls Codex's cuBLASLt wrapper (`_cx_lt_run`) with the pinned config. The settings are the first entries of `lt_128_256.json`, e.g. pooler m128 `nn [66, 16, 1, 0, 0, 1, 35, 0, 2, 0, 130101]`.
- **K1t** (`_k1t_*`, NVRTC): pooler and unpooler at 17 ≤ m ≤ 32 (`_K1T_USE`), with 32-token tiles. Its `prepare` packs `pooler32` / `unpooler32` variants of the 12-bit layout.
- **`_LIB` hook** for a flashinfer backend: added but left empty, so nothing routes to it.
- No file is read at run time: the configs are literals in `build.py`.

Iterations on the way (local µs):

| step (log) | what | result |
|---|---|---|
| `int2.log` (13:04) | int_v1 + K1t at m 17-32 (`claude/sub_int2.py`) | decode-32 load x1.280 vs x1.239; overall 1.3189 vs 1.3187 (other loads noise) |
| `int3.log` (13:15) | + flashinfer cuDNN picks for qkv m256, o m128/256, pooler m128 (`claude/sub_int3.py`, `_LIB`) | judge-projected 1.3221 vs 1.3216: no gain, dropped (`_LIB` left empty) |
| `lt1.log` (13:19) | first `lt_tune.py` build | compile error: `CUBLASLT_ALGO_CAP_CLUSTER_SHAPE_IDS` not in this cuBLAS; fixed |
| `lt2.log` (13:35) | m = 128/256, both layouts | the table in 2(c) |
| `int4.log` (13:38) | int_v1 + `_LT` + K1t | 1.3364 vs 1.2979: shipped as int_v4 |
| `lt3.log` (13:59, after the build) | the same search at m ≤ 32 | best searched cuBLASLt configs (pooler m16 16.51, unpooler m16 15.97, pooler m32 16.83) are no better than the 12-bit kernels already in the build (K1s pooler m16 15.40, K1t pooler m32 17.66 in `int4.log`, a different run); m ≤ 32 stays on K1s/K1t |

## 5. Result

| check | result |
|---|---|
| local A/B, same run (`evidence/int4.log`, `evidence/int4_bench.json`) | pooler m128 19.82 vs 25.37, pooler m256 23.74 vs 28.28, qkv m256 17.08 vs 19.78, qkv m128 14.74 vs 15.18, pooler m32 17.66 vs 18.72, unpooler m32 16.64 vs 17.34; o m128 6.48 vs 6.61, o m256 8.61 vs 8.35, unpooler m128/m256 18.35/23.43 vs 18.47/23.99. Judge-projected **1.3364 vs 1.2979** |
| correctness | sweep "all ok" before the bench (`int4.log`); judge tests 9/9 |
| judge | 1.3112 → **1.3482** (+2.82 %) |
| judge, changed shapes (µs) | pooler m128 24.67 → 21.01 (−3.66); pooler m256 29.49 → 24.86 (−4.63); qkv m256 20.58 → 18.32 (−2.26); qkv m128 16.60 → 15.54 (−1.06); pooler m32 19.68 → 18.38 (−1.30); unpooler m32 18.47 → 17.45 (−1.02); o m128 +0.25, o m256 −0.44, unpooler m128 −0.04, m256 +0.01 |
| **judge, read honestly** | The m = 1 shapes run **identical** code in both builds, yet decode 1 went the other way: qkv 10.58 → 11.21 (+0.63), pooler 16.42 → 17.37 (+0.95), router 3.59 → 3.85 (+0.26). The decode-1 load speedup fell from 1.501 to 1.428 (−4.9 %) with no code change. The local bench ran those same kernels at 9.94 vs 9.81 (qkv m1) and 16.28 vs 16.30 (pooler m1) µs. So this node was slow on small kernels, and that hid part of the gain. The two loads that changed most improved: verify 16 1.134 → 1.223 and verify 32 1.074 → 1.171 (computed). The real effect is at least the +2.8 % read here, and probably closer to the local +3.0 %. |
| vs projection | +2.8 % judge vs +3.0 % projected. Per shape, pooler m256 matched (judge −4.63 µs, local −4.54 in `int4.log`) and pooler m128 came in smaller (judge −3.66, local −5.55). The m128 gap (1.9 µs) is larger than the ±0.6 µs per-shape noise seen in 02; the notes do not isolate its cause. |

## 6. Takeaways, and what came next

- **Search the library before replacing it.** One afternoon of exhaustive cuBLASLt enumeration was worth +2.8 %, more than the two custom-GEMM workers produced all day (both lost).
- **The searched configs are still far from the floor.** In the judge, pooler m128 at 21.0 µs is 1.43x its 14.64 µs floor, and qkv m128 at 15.5 is 1.85x its 8.42. In milestone 05's ncu (`05_.../evidence/ncu_int_v7.log`, same configs), the pooler m128 cuBLASLt kernel shows 14 % occupancy and 60 % long-scoreboard stalls. These shapes needed a custom kernel with the split-K exchange hidden on-chip, which was not built until later.
- **The m = 1 shapes, the most valuable per µs, were still on Codex's BF16 Triton GEMVs** (11.2 / 5.4 / 17.4 / 3.7 µs judge for qkv / o / pooler / unpooler slice). That is above even the BF16 floor, while the 12-bit floor is ~20 % lower.
- **Next, 04:** a many-CTA GEMV at m = 1 on the 12-bit format.

## Reproduce

Needs an H200 and the SGLang image (torch 2.14 + CUDA 13.0, Triton 3.8, cuda-python, `sglang`). `prepare()` builds the Codex cuBLASLt and TinyGEMM extensions with `load_inline` (nvcc + ninja) and compiles K1s/K1t/router with NVRTC. The cuBLASLt configs are literals in `build.py`; nothing is read from disk. cuBLASLt algorithm ids are tied to the cuBLAS version in the image, and another version may reject them.

```bash
cd muse_bf16_proj/tools
python3 check.py ../03_cublaslt_config_search/build.py --graph
python3 bench.py --cand int_v1=../02_f12_register_streaming/build.py --cand int_v4=../03_cublaslt_config_search/build.py \
       --shapes 32,128,256 --reps 30                                            # changed loads; the m = 1 shapes as control
# the config search itself (original location claude/; needs Codex's dev/cublaslt_candidate.py beside it):
#   python3 claude/lt_tune.py --shapes 128,256 --json /tmp/lt.json
```

Originals (relative to ``):
- build: `muse-opt-codex/muse_bf16_proj/claude/int_v4.py`
- search: `muse-opt-codex/muse_bf16_proj/claude/lt_tune.py`, `claude/lt_128_256.json`
- logs: `claude/out/{lib3,lt1,lt2,lt3,int3,int4,k1t7}.log`, `claude/out/20261027T133827_int4/int4.json`
- K2 / K2c workers: `claude/k2/`, `claude/k2c/` (NOTES, kernels, run logs)
- judge: `muse-opt-codex/muse_bf16_proj/claude/profiles/int_v4.8bb51ba316a14813.{txt,json,log}`
