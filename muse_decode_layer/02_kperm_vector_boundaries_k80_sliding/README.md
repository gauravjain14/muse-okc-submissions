# 02: cheaper glue, a scale-sharing expert layout and k80 sliding attention (all shapes)

| | before (01) | after (02) |
|---|---|---|
| build | `01_.../build.py` = `d01_dense21.py` (md5 7c2f3de5) | `build.py` = `variants/d09_stack.py` (md5 4db1b641, 986,177 B) |
| judge | 2.0568 (check, job 499a15f4) | **2.1817** (scored, job f9a581c0, CI 2.1489-2.2144, 13/13). Part of the jump is a slower SGLang draw; see Result |
| real effect | | local same-run weighted ratio **+6.1%** vs d01 (36 fresh draws). Judge with SGLang held fixed: 2.0606 → 2.1517 (**+4.4%**) |
| shapes changed | | all 12: boundaries, routing and the T ≥ 17 expert layout everywhere. The T ≤ 8 router, T = 1 dense PDL and sliding attention only on their shapes |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: code copied or adapted from other projects is marked in place with `SPDX-SnippetBegin` / `SPDX-SnippetEnd` comments (source, copyright, license, our changes); see `NOTICE` and `LICENSES/` at the repository root. The code is unchanged. The judged file has md5 `4db1b641`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

## 1. Starting point

d01 judge, ours / SGLang in µs (`evidence/judge_before_d01_dense21_499a15f4.txt`):

| d1s | d1f | d16s | d16f | d32s | d32f | v1s | v1f | v16s | v16f | v32s | v32f |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 76.9 / 170.1 | 82.1 / 180.0 | 141.9 / 266.1 | 217.2 / 436.7 | 160.6 / 260.9 | 318.0 / 617.2 | 119.1 / 259.8 | 128.3 / 278.5 | 246.3 / 515.8 | 334.5 / 693.9 | 325.0 / 734.9 | 484.5 / 1064.1 |

Ten kernels on every shape.

## 2. What we saw

Sources: d01's per-kernel nsys + ncu (`../01_public_pipeline_bo21L_dense/evidence/d01_analysis.txt`) and the judge's own per-kernel profile of d01 (`evidence/judge_before_d01_dense21_499a15f4.txt`, "profile N" tables).

**(a) Four glue kernels per shape that move no bytes.** In the judge profile, `_boundary1` takes 4.4-5.6 µs, `_boundary2` 4.4-5.3 µs and `_route_split` 4.0-4.4 µs on every shape. ncu reads them at 0.01-0.35 TB/s, 24% occupancy, and stalls led by long scoreboard / mio throttle / short scoreboard (d01_analysis cases 0-5). They do one program per token and a serial top-8. At T = 1 and T = 8 a separate `_router_fp32_exact` costs another 4.1 / 9.0-9.2 µs. All of these are latency, and on the B = 1 shapes they are worth the most score per µs.

**(b) The expert kernel's inner loop.** `moe_fused` is the largest kernel on every shape:
- 129.2 µs at v16s (442.6 MB at 3.43 TB/s, issue 56.2%), and 188.8 µs at v32s (`d01_analysis.txt` cases 8, 10). That is below the ~4.5 TB/s cold stream.
- alygg77's public KPERM3 re-layout permutes the NVFP4 codes along K, so one scale pair serves four 16-wide K steps. "The inner loop stops spilling registers"; the codes, scales and every dequantized value stay the same (`muse-opt-codex/muse_decode_layer/sources/alygg77/README.md`, lines 59-63).

**(c) Sliding attention ran the generic kernel.** The generic `attn_kernel` took 5.0 µs at d1s, 8.2 at d16s and 6.9 at v1s (judge d01 profile). These read only the 2,048-token window, so they are latency chains. Our attention task's k80 build has dedicated cluster kernels for them (the sibling task `../muse_attention_decode/`: the XE class kernels from its milestone 06, as shipped in k80, its milestone 09).

**(d) PDL at T = 1 only.** The m = 1 dense kernels are pure weight streams that can be issued before the predecessor ends. At T = 8 the same edit lost: E008 measured +2.60 / +2.62 µs at d1s / d1f but −0.41 / −1.45 µs at v1s / v1f (`evidence/ab/d07_vs_d06.log`).

Every piece was measured as a separate whole-layer A/B with `tools/compare.py`: paired AB/BA replays on the same fresh inputs, 3 fresh draws × 12 pairs per case (`evidence/compare.py`; results `evidence/ab/d0N_vs_d0M.{json,log}`; ledger `runs/research.json` E003-E009):

| step | piece | measured saving (µs, median by case) | kept |
|---|---|---|---|
| d02 | weight-only residual coefficients precomputed (no sigmoid / sqrt per row) | 0.5-2.4 in 11 cases, −0.86 in case 5 | yes |
| d03 | KPERM3 expert layout (alygg77, Apache-2.0) | d32s 3.86, d32f 3.21, v16s 3.42, v16f 3.52, v32s 4.47, v32f 6.23; 215/216 changed-case pairs positive | yes (T ≥ 17 builds) |
| d04 | FP32 router: weight loads shared over the 8 token rows in one CTA | T = 8: −1.30..−1.36; **T = 1: +2.18..+2.32 worse** | T = 8 only |
| d05 | both residual/norm boundaries vectorized (4 adjacent elements per lane) | +0.78 (case 5) to +2.48 (case 11), all 12 cases | yes |
| d06 | pool-normalize and top-8 selection in separate CTAs of one launch (`_dl_tile_route`) | +1.18 to +2.96, all 12 cases | yes |
| d07 | PDL + W prefetch on the small dense chain | d1s/d1f +2.60/+2.62; v1s/v1f −0.41/−1.45 | T = 1 only |
| d08 | k80 sliding-attention cluster kernels (`xe_*`) | d1 0.48, d16 2.02, v1 4.92 | yes |

## 3. Diagnosis and projection

- **What limits each piece.** The glue is latency-bound and the experts are issue-bound in the unpack loop. Sliding attention is a latency chain.
- **No joint projection was written down.** The stack was assembled from the per-piece wins, keeping each piece only on the shapes where it won (the "measured dispatch").
- **Size of the stack, from the per-piece table** (inferred: per-piece savings do not add exactly):
  - Summed, it is about 6-14 µs per shape on 70-480 µs shapes.
  - The largest relative wins land on d1s, v1s and d16s.

## 4. Change

`tools/build_d09.py` (in `evidence/recipes/`) builds d09 from d08 plus the measured dispatch. Its manifest reads: "original one-token router; PDL dense only at T=1; native shared FP32 router at 2..8; k80 sliding; vector boundaries; parallel route; KPERM3 MoE" (`evidence/recipes/d09_build.json`). Concretely:

- `_boundary1_vector` / `_boundary2_vector` replace `_boundary1/2` (d05), with precomputed α/β (d02).
- `_dl_tile_route` replaces `_route_split` (d06).
- `router_fp32_kernel` replaces `_router_fp32_exact` at 2 ≤ T ≤ 8 (d04). T = 1 keeps the original one-row router.
- The expert kernel's T ≥ 17 builds read KPERM3-permuted codes (d03). The permutation is applied in `prepare()` and is lossless.
- The m = 1 dense kernels (`n2_*`, `nk_o1`, `sm5_cgemv`) are PDL-launched with their weights prefetched before the wait (d07).
- k80's `xe_r1_1` (d1s), `xe_dm_8_4_2` (d16s) and `xe_b1s_8c6` (v1s) run sliding attention (d08). d32s keeps the generic kernel (k80's B = 25..44 sliding kernel was later measured unstable, E031).

## 5. Result

| check | result |
|---|---|
| local, same run, d09 vs d01 (router seed 11171, pad 23, 3 fresh × 12 pairs; `evidence/ab/d09_vs_d01_router11171.log`) | every case faster in every draw. d1s +8.3 µs; v1s +9.7..+10.3; v16s +8.4..+10.2; v32s +13.7..+14.2; v32f +12.0..+14.1; d32f +5.4..+9.0. **Weighted ratio 1.0611** (`runs/research.json` E010) |
| correctness | 36/36 fresh scored-shape draws, 26/26 hard cases (`runs/d09_hard.json`); judge 13/13 |
| judge | 2.0568 → **2.1817** (`evidence/judge_after_d09_stack_f9a581c0.txt`) |
| judge per kernel, d01 → d09 run | `_boundary1` 5.6 → 3.7 (`_vector`), `_boundary2` 4.7 → 3.2, `_route_split` 4.4 → `_dl_tile_route` 2.1 (d1s). Attention: d1s 5.0 → 3.5 (`xe_r1_1`), d16s 8.2 → 6.7 (`xe_dm_8_4_2`), v1s 6.9 → 4.0 (`xe_b1s_8c6`). v1s router 9.2 → 7.1 |
| **judge, read honestly** | Our load-geomean time fell 10,953.7 → 10,489.9 µs (−4.2%). SGLang in this run was slower than in d01's, e.g. v16s 515.8 → 541.3 µs and v32f 1064.1 → 1096.6 µs. With SGLang fixed at its 70-run median per case, the two readings are 2.0606 → 2.1517 = **+4.4%**, against a local +6.1% (`../03_pool_router_unions_k80_full_attention/evidence/judge_readings_early.tsv`) |
| the routing caveat | The judge picks a router seed per run, so `moe_fused` moved both ways between the two runs on byte-identical expert code: d1s 30.1 → 22.8 µs, d1f 22.7 → 29.8 µs, v16s 151.2 → 179.5 µs. Single-run kernel deltas for the experts are mostly routing, not code |
| vs projection | E010 records a local weighted ratio of 1.0611 and a judge ratio of 1.0607. The match is partly luck: our own times moved 4.2% and the SGLang draw supplied the rest |

**Measurement caveat found afterwards.** A review of the early harness (`muse-opt-codex/muse_decode_layer/dumb_mistakes.md`, H1/H2) found three problems:
- `compare.py` always placed the control first, with a fixed pad layout.
- There was no A/A run.
- Most A/Bs ran on the default router, whose MoE load differs from the judge's.

Effects below about 0.6-1 µs per case are therefore not resolved by the table in section 2. The 6-14 µs per-case savings of the full stack are far above that.

## 6. Takeaways, and what came next

- **Glue and launch latency were a large, cheap target.** Each byte-free kernel costs 3-6 µs in the judge's graph, and the judge's own trivial-graph timing shows about 1.06 µs of marginal cost per extra kernel node (`muse-opt-codex/muse_decode_layer/dumb_mistakes.md`, H3).
- **Keep a piece only where it measures faster.**
  - The T = 8 router and PDL both lost on some shapes and were dispatched per T.
  - Every T ≤ 8 router stays FP32, as the reference requires.
- **Calibrate the local bench to the judge's routing before trusting sub-µs deltas.** This became the anchor method in 03 (`tools/bench.py --cand d09=...`). This build's judge reading is the d09 anchor.
- **Next:** [03](../03_pool_router_unions_k80_full_attention/) folds the T = 1/8 pooler and FP32 router into one launch, and moves full-context verify and single-request decode/verify attention to k80's dedicated kernels.

## Reproduce

```bash
cd muse_decode_layer/tools
python check.py --cand ../02_kperm_vector_boundaries_k80_sliding/build.py --suites tests,scored,nearby --seeds 2
python bench.py --cand d01=../01_public_pipeline_bo21L_dense/build.py \
                --cand d09=../02_kperm_vector_boundaries_k80_sliding/build.py   # paired ratio d09 vs d01, judge-alike
```

The original per-piece A/Bs used `evidence/compare.py` (with `evidence/measure.py`), run from `muse-opt-codex/muse_decode_layer/`: `python tools/compare.py variants/d01_dense21.py variants/d09_stack.py --fresh 3 --reps 24 --router-seed 11171 --pad-seed 23`.

Originals (relative to ``): builds `muse-opt-codex/muse_decode_layer/variants/d02_*.py` … `d09_stack.py`; recipes `muse-opt-codex/muse_decode_layer/tools/build_d0{2..9}.py`; A/B outputs `muse-opt-codex/muse_decode_layer/runs/d0N_vs_d0M.*`; judge JSON `muse-opt-codex/profiles/f9a581c0ca2346a5.json`; ledger `muse-opt-codex/muse_decode_layer/runs/research.json` E003-E010 (copied as (private notes, not included)).
