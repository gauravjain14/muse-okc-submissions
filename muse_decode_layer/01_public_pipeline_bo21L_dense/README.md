# 01: a valid whole layer: the public pipeline with our lossless bf16 dense kernels (all shapes)

| | before | after (01) |
|---|---|---|
| build | `variants/public_native.py`: vjeux's public whole-layer pipeline (md5 bfdcb290, 568,655 B) | `build.py` = `variants/d01_dense21.py` (md5 7c2f3de5, 858,358 B) |
| judge | not judgeable: cold `prepare()` took 364.5 s, the limit is 300 s | **2.0568** (check, job 499a15f4, CI 2.0259-2.0876, 13/13 tests) |
| real effect | | single reading 2.0568; with SGLang fixed at its 70-run median per case, 2.0606 |
| shapes changed | | all 12: the qkv / o / pooler / router / unpooler GEMMs, and `prepare()` |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: code copied or adapted from other projects is marked in place with `SPDX-SnippetBegin` / `SPDX-SnippetEnd` comments (source, copyright, license, our changes); see `NOTICE` and `LICENSES/` at the repository root. The code is unchanged. The judged file has md5 `7c2f3de5`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

The problem, the scoring and the shape names (d1s = decode 1 sliding, v32f = verify 32 full, ...) are explained in `../README.md`.

## 1. Starting point

The starting file is vjeux's Apache-2.0 public whole-layer pipeline (see the root `NOTICE`). It strings together component kernels and a prepare-time tuner. For every (T, matrix) the tuner times cuBLAS, a bf16_proj kernel and a Triton split-K GEMM, and pins the fastest (docstring of `variants/public_native.py`).

- It passed 26/26 fresh hard-case checks locally, but cold `prepare()` took **364.5 s** (`muse-opt-codex/muse_decode_layer/runs/public_native_hard.log`, "PREPARE 364.53s"). TASK.md allows 300 s (`muse-opt-codex/muse_decode_layer/TASK.md`, "runs once, untimed, for up to 300 s"). So there is no judge reading for it.
- Its router fed BF16 activations at T ≤ 8. The reference uses an FP32 activation and FP32 router weights there (`runs/research.json` E001: "small-token router input needs FP32"; `baseline.py`).

## 2. What we saw

**(a) Where a build spends prepare time.** The tuner benchmarks every backend for every (T, matrix) on the judge GPU. The dense kernels themselves were the muse_bf16_proj entry's. Our own lossless bf16 build at that time, `bo21L`, already carried the per-(layer, m) kernel choice baked in. It needs no tuning, only NVRTC compiles. That is the structural fix for the 364.5 s.

**(b) Where the time goes in d01.** Per-kernel nsys + ncu over the 12 scored cases, one fresh draw each, default router (`evidence/d01_analysis.txt`, from `muse-opt-codex/muse_decode_layer/profiles/d01_analysis.txt`). Ten kernels per shape:

| case | span (µs) | experts `moe_fused` | pooler (+router) | qkv | attention | o | boundary 1 + 2 | `_route_split` |
|---|---|---|---|---|---|---|---|---|
| d1s | 75.6 | 25.4 (2.56 TB/s, issue 36.8%) | 12.7 + 3.9 | 8.3 (3.30 TB/s) | 4.6 | 4.5 | 5.2 + 4.8 | 4.4 |
| d16s | 120.7 | 53.8 (3.19 TB/s) | 14.6 + 4.3 | 8.9 | 7.9 | 5.3 | 4.6 + 5.0 | 4.4 |
| d32f | 313.7 | 84.1 | 15.1 + 4.6 | 9.6 | **166.8 (4.12 TB/s)** | 5.7 | 4.6 + 4.9 | 4.2 |
| v16s | 222.7 | **129.2 (3.43 TB/s, issue 56.2%)** | 20.5 + 5.4 | 13.6 | 11.6 | 6.5 | 5.4 + 5.9 | 4.3 |
| v32s | 309.2 | **188.8** | 26.6 + 6.7 | 16.1 | 14.7 | 8.4 | 9.6 + 8.0 | 5.3 |

Readings from that table:
- The dense projections stream at 3.0-3.9 TB/s: `n2_pooler` 3.84 TB/s, `pu5_unpooler_16` 3.92 TB/s. They are near the cold-L2 DRAM line.
- The glue kernels move almost no bytes: `_boundary1` 0.02 TB/s, `_route_split` 0.01 TB/s. Yet each costs 4-6 µs. They are latency and launch, not bandwidth.
- The routed experts are the single largest kernel on every shape.

`runs/research.json` (findings, `profiles/d01_nsys.json`) puts experts at 58-61% of the graph span in the large sliding cases. The judge's own profile of the next scored build later showed this understates it (64-67%, see 02).

## 3. Diagnosis and projection

- **Validity first.** Only a build whose `prepare()` fits in 300 s can be judged. Replacing the tuned dense path with a fixed, already-tuned dense build removes the tuning cost.
- **bo21L** stores the BF16 weights losslessly in 11/12-bit formats (n2 `compress11`, F12 tiles, exact escapes). That cuts the weight stream of every projection.
- At T = 1 the dense projections are pure weight streams, so fewer bytes are the lever. At T = 128/256 the m128/m256 kernels are tensor-bound and unchanged.
- No numeric projection was made for this step. The goal was a valid baseline.

## 4. Change

`tools/build_dense21.py` (copied to `evidence/build_dense21.py`) emits `d01_dense21.py`. The change is a 581 KB diff that is almost all insertion:

1. **bo21L inserted verbatim.** The `muse_bf16_proj/claude/t3/bo21L.py` build (sha256 e0b57002…, `evidence/d01_build.json`) goes in with every name prefixed `dlbf_`. It replaces the pipeline's own `bfp__*` dense section. Its attribution block (vjeux, alygg77, Nathen Fernandes, Matsvei) stays (`NOTICE`).
2. **Tuner replaced by fixed choices.** `BFP_MAX_T = 256`, so the bf16_proj kernels serve every T, and "force all tuned matrices": the kernel per (T, matrix) is fixed and no longer timed in `prepare()`.
3. **FP32 router input kept.** `_boundary1` gets a second output (`ROUTER_OUT`, `KEEP_ROUTER`) that keeps the unrounded FP32 normalized activation for the router at T ≤ 8, as the reference does.
4. **Single-CTA boundaries.** The residual/norm boundary kernels run one CTA per token.

The bo21L kernels are our own work from the muse_bf16_proj task (sibling task `../muse_bf16_proj/`). They are not re-derived here.

## 5. Result

| check | result |
|---|---|
| correctness | 26/26 fresh hard-case graph checks and 24/24 scored-shape checks locally (`runs/d01_hard.json`, `runs/research.json` findings); judge 13/13 |
| prepare | 364.5 s → **19.0 s** local (`evidence/d01_hard.log`, "PREPARE 18.98s"); **70.1 s** on the judge (`evidence/d01_check.json`, `prepare_s`) |
| judge | **2.0568**, CI 2.0259-2.0876 (`evidence/judge_after_d01_dense21_499a15f4.txt`) |
| judge per case, ours / SGLang (µs) | d1s 76.9 / 170.1, d1f 82.1 / 180.0, d16s 141.9 / 266.1, d16f 217.2 / 436.7, d32s 160.6 / 260.9, d32f 318.0 / 617.2, v1s 119.1 / 259.8, v1f 128.3 / 278.5, v16s 246.3 / 515.8, v16f 334.5 / 693.9, v32s 325.0 / 734.9, v32f 484.5 / 1064.1 (judge JSON `muse-opt-codex/profiles/499a15f417bd4de4.json`) |
| judge, read honestly | One reading. With SGLang fixed at its median over all 70 decode-layer judge runs, the same per-case times give 2.0606 (`../03_pool_router_unions_k80_full_attention/evidence/judge_readings_early.tsv`, column `fixed_sglang`). This draw's SGLang was close to typical. |
| vs projection | none made |

This reading later became one of the two calibration anchors for the local judge-alike bench (`tools/common.py`, `JUDGE_D01_*`). The local SGLang-anchored projection of this build was 2.0644 vs the judge's 2.0568, i.e. +0.4% out of sample (`muse-opt-codex/muse_decode_layer/cdl/H/NOTES.md`, line 9).

## 6. Takeaways, and what came next

- **A build that cannot pass `prepare()` scores nothing.** Prefer fixed, pre-tuned kernel choices over tuning at prepare time.
- **The profile set the agenda for 02:**
  - The ten kernels per shape include four byte-free glue kernels (two boundaries, `_route_split`, a separate FP32 router at T = 1/8) at 4-6 µs each.
  - The expert kernel dominates every large shape.
  - Sliding attention at T ≥ 16 still ran the generic pipeline kernel.
- **Next:** [02](../02_kperm_vector_boundaries_k80_sliding/) attacks the glue (vectorized boundaries, parallel routing, PDL on the T = 1 dense chain), the expert weight layout (lossless K permutation), and sliding attention (our k80 kernels).

## Reproduce

```bash
cd muse_decode_layer/tools
python check.py --cand ../01_public_pipeline_bo21L_dense/build.py --suites tests,scored,nearby --seeds 2
python bench.py --cand d01=../01_public_pipeline_bo21L_dense/build.py            # local judge-alike, SGLang-anchored
```

Originals (relative to ``): build `muse-opt-codex/muse_decode_layer/variants/d01_dense21.py`, recipe `muse-opt-codex/muse_decode_layer/tools/build_dense21.py`, profiles `muse-opt-codex/muse_decode_layer/profiles/d01_*` (`d01_ncu.ncu-rep`, 9.9 MB, not copied), lab ledger `muse-opt-codex/muse_decode_layer/runs/research.json` (E001, E002; copied as (private notes, not included) because it was appended with later work).
