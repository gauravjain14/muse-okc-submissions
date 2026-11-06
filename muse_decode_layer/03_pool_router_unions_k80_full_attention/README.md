# 03: one-launch pooler + FP32 router at T = 1/8, and k80's full-context attention kernels (d1, v1, v16f, v32f)

| | before (02) | after (03) |
|---|---|---|
| build | `02_.../build.py` = `d09_stack.py` (md5 4db1b641) | `build.py` = `variants/d23_single_verify.py` (md5 18d93d14, 994,930 B; 5,070 B under the 1,000,000 B cap) |
| judge | 2.1817 (scored, job f9a581c0) | **no judge run of this file.** Local d09-anchored projection **2.2053** (mean of 3 router seeds). A sibling build (this file + later expert/verify edits) checked 2.2074 (job fe345625) |
| real effect | | local same-run paired ratio vs d09 **+1.09%** (1.0110 / 1.0109 / 1.0107 over 3 router seeds) |
| shapes changed | | d1s, d1f, v1s (pooler + router union); v1f, v16f, v32f (full-context verify attention); d1f (single-request full decode attention) |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: code copied or adapted from other projects is marked in place with `SPDX-SnippetBegin` / `SPDX-SnippetEnd` comments (source, copyright, license, our changes); see `NOTICE` and `LICENSES/` at the repository root. The code is unchanged. The judged file has md5 `18d93d14`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

## 1. Starting point

d09 judge, ours (µs) and the judge's per-kernel profile (`evidence/judge_before_d09_stack_f9a581c0.txt`):

| shape | ours | the kernels this milestone targets |
|---|---|---|
| d1s | 69.9 | `n2_pooler` 13.1 + `_router_fp32_exact` 4.2: two launches |
| d1f | 76.4 | same pair, 13.0 + 4.1; full attention `attn_kernel<1,false,11>` 11.4 |
| v1s | 115.8 | `m8_pooler` 13.8 + `router_fp32_kernel` 7.1 |
| v1f | 119.8 | same pair 13.3 + 7.3; full attention `attn_kernel<8,false,11>` 14.6 |
| v16f | 332.4 | full verify attention `attn_kernel<8,true,0>` 104.5 |
| v32f | 484.0 | full verify attention `attn_kernel<8,true,0>` 188.1 |

## 2. What we saw

**(a) Full-context verify attention ran the pipeline's generic kernel well below the gather rate.**
- In d01's ncu, `attn_kernel` at v16f read 348.2 MB at 3.48 TB/s (100.1 µs), and at v32f 694.6 MB at 3.77 TB/s (184.5 µs). Issue was 48-51% and occupancy 36%.
- The decode-full kernels at d16f/d32f read their K/V at 4.05-4.12 TB/s (`../01_public_pipeline_bo21L_dense/evidence/d01_analysis.txt`, cases 3, 5, 9, 11).
- Our attention task's k80 build has a dedicated verify-full module, XD/XC, and dedicated single-request kernels (XB1 for B = 1 decode, XG5 for B = 1 verify). The sibling task describes them in its milestones 06 (the per-class XE / XA / XD modules) and 09 (k80) (`../muse_attention_decode/`).

**(b) At T = 1 and T = 8 the pooler and the FP32 router are two launches on the same input.**
- Each launch costs a node in the graph, about 1.06 µs marginal at the judge (`muse-opt-codex/muse_decode_layer/dumb_mistakes.md`, H3, from the judge's own `replay_overhead`).
- The router CTAs read only 8.4 MB of FP32 weights. ncu: `_router_fp32_exact` 3.9 µs at 2.1 TB/s with long-scoreboard stalls at the top. They can share the grid with the pooler's 48.7 MB stream (`d01_analysis.txt` case 0).

**(c) Single-kernel experiments on the boundaries and the experts did not pay.** These closed other directions in the same window (`runs/research.json`):

| experiment | result |
|---|---|
| E015/E016/E021/E022: two CTAs/SM for the large norms, 8 warps, delayed loads, 108-config boundary sweep | neutral or regressions (T = 128 −3.7 µs with 8 warps) |
| E018: sum-of-squares partials in phase-B epilogues | T = 128 −1.6, T = 256 −4.7 µs |
| E019: 96-token expert jobs | T = 32 −2.8, T = 128 −9.9, T = 256 −11.4 µs |
| E020: L2 count-based TMA cache hint for expert weights | mixed (+0.42 / −0.35 µs) |

## 3. Diagnosis and projection

- **Full verify (v16f/v32f).** It is gather-bound, and the generic kernel's consumer loop and merge sit 10-15% above the gather rate the decode kernels already reach. Projection: a large share of the 10-15% gap, i.e. ~10 µs at v16f and ~15 µs at v32f (inferred from the TB/s gap).
- **T = 1/8 pooler + router.** It is launch- and latency-bound. Fusing them saves one node plus the router's ramp: about 1-2 µs per shape.
- **Score impact.** Full verify carries only 16 of 62 weight in its load, and B = 1 shapes earn the most per µs. Expected total: about +1%. These ratios come from the `score%/us` column of `evidence/floors/floors_d23.txt`: d1s 0.178 %/µs, v16f 0.016, v32f 0.012.

## 4. Change

Built step by step, each step a paired whole-layer A/B (recipes in `evidence/recipes/`, A/B outputs in `evidence/ab/`, ledger `runs/research.json`):

| step | change | measured | what it taught |
|---|---|---|---|
| d10 | `n2_pool_router`: T = 1 pooler and exact FP32 router CTAs interleaved in one grid | −1.57 µs sliding, −1.74 full; 71/72 pairs positive (E011) | one node less is worth more than its bytes |
| d12 | k80 XD/XC full-verify attention for B 2..128 | v16f **−10.85**, v32f **−12.25 µs**; 72/72 pairs positive (E013) | the verify-full path was the largest single attention gap |
| d13 | `m8_pool_router`: same union at T = 8 | −1.70 sliding, −1.24 full (E014) | |
| d16 | stack of d10 + d12 + d13 | local weighted ratio 1.0087 vs d09 (E017) | |
| d21 → d22 | k80 single-request full decode (XB1) at d1f, inactive XD branches pruned to fit the 1 MB cap | d21 failed a routing check: offline preprocessing had replaced the NVRTC-supplied `XB1_GC` with 0. d22 fixed it, verified against the original preprocessor output at 4 grid sizes; d1f −0.27 µs (E023, E024) | when pruning source for size, compare against the real preprocessor output |
| d23 | k80 single-request full verify (XG5, `xg_v8f`) with its shape guard and fallback | v1f −3.25 µs, 36/36 pairs (E025) | |

The file is 994,930 B, leaving 5,070 B under the 1,000,000 B cap for anything else.

## 5. Result

| check | result |
|---|---|
| local A/A control (d23 vs itself, same harness; `evidence/ab/d23_aa_vs_d23.log`) | per-case medians within −1.04..+0.86 µs; positive-pair counts 1/12 to 10/12. The harness noise floor is ~±1 µs per case |
| local, d23 vs d09, router 11171 (`evidence/ab/d23_vs_d09_router11171.log`) | d1s −1.1..−1.8, d1f −1.5..−1.7, v1s −0.8..−1.7, v1f −3.1..−4.2, **v16f −8.8..−15.1**, **v32f −10.8..−14.5 µs** (all draws 11-12/12 positive). d16/d32/v16s/v32s within the A/A band |
| local, judge-alike bench, d09-anchored (`muse-opt-codex/muse_decode_layer/cdl/H/BASELINE.md`) | d23/d09 paired 1.0110 / 1.0109 / 1.0107 on router seeds 3043 / 11064 / 15316 → **projected 2.2053** (2.2055 / 2.2054 / 2.2050) |
| correctness | 72 scored, 26 hard and 20 nearby checks (E025; `evidence/ab/d23_hard.log`, `d23_nearby.log`); 298 local checks in the later harness (H NOTES) |
| judge, sibling | `d34_verify_expert_stack.py` = d23 + WH sliding-verify attention + expert retained-value edits: **2.2074** (check, job fe345625; fixed-SGLang 2.1829). In that run the d23 kernels read: `n2_pool_router` 16.0 µs (vs 13.1 + 4.2 for the pair in d09's run), `m8_pool_router` 19.0 (vs 13.8 + 7.1), `xg_v8f` 12.4 (vs 14.6), `attn_xd_v8f` **97.0 at v16f (vs 104.5) and 174.2 at v32f (vs 188.1)** (`evidence/judge_d34_verify_expert_stack_sibling_fe345625.txt` vs `judge_before_d09_stack_f9a581c0.txt`) |
| vs projection | The verify-full gain landed as projected (−9..−15 µs). The pool/router unions gave 1.3-1.9 µs per shape. Total +1.09% locally, slightly above the ~+1% estimate |

This file was never judged on its own. Its real level is the projection: about 2.205. The sibling's 2.2074 (fixed-SGLang 2.1829) supports it.

## 6. Takeaways

- **For attention, reuse the dedicated kernels the attention task already tuned** (XD/XC verify-full, XB1, XG5). Full-context verify gained 9-15 µs per shape with no new kernel code.
- **Merging two launches on the same input pays on the B = 1 shapes** even when the bytes do not change.
- **Calibrate first.** The anchor method (judge time = judge's d09 time / local paired ratio vs d09) predicted d01's judged 2.0568 as 2.0644 (+0.4%). It became the standard projection. A local SGLang-ratio projection ran about 2% high, because the local box's SGLang is 6.5% slower on decode 1 (`notes/H_NOTES.md`).
- **Where d23 still lost, by class** (`evidence/floors/floors_d23.txt`, score % if the class ran at its floor, linearized):

  | class | gap (score %) |
  |---|---|
  | MoE expert kernel | **15.31** |
  | attention | 5.06 |
  | boundary 2 / boundary 1 | 2.46 / 2.11 |
  | unpooler | 1.86 |
  | exposed gaps | 1.65 |
  | route | 1.60 |
  | o / qkv / pooler | 1.15 / 1.10 / 0.99 |
  | pool+router / router | 0.95 / 0.86 |

  The top single items were all the expert kernel: v32s 2.34%, v16s 2.27%, d32s 1.80%, d16s 1.80%, d1s 1.56%.
- **What this build embeds from the sibling tasks.**
  - bf16: the bo21L dense kernels.
  - Attention: the k80 kernels (XE, XB1, XG5, XD/XC).
  - Experts: the public pipeline's expert kernel with alygg77's KPERM3 layout.

## Reproduce

```bash
cd muse_decode_layer/tools
python check.py --cand ../03_pool_router_unions_k80_full_attention/build.py --suites tests,scored,nearby --seeds 2
python bench.py --cand d09=../02_kperm_vector_boundaries_k80_sliding/build.py \
                --cand d23=../03_pool_router_unions_k80_full_attention/build.py --router-seed 3043   # d09-anchored projection
python kprof.py prof --cand ../03_pool_router_unions_k80_full_attention/build.py --out /tmp/kp_d23 && \
python floors.py /tmp/kp_d23.json                                                                   # per-class floors table
```

Originals (relative to ``): build `muse-opt-codex/muse_decode_layer/variants/d23_single_verify.py` (= `cdl/d23_base.py`, byte-identical); recipes `muse-opt-codex/muse_decode_layer/tools/build_d1{0..6}.py`, `build_d2{1..3}.py`; A/Bs `muse-opt-codex/muse_decode_layer/runs/d1*_vs_*.log`, `d2*_vs_*.log`; projection `muse-opt-codex/muse_decode_layer/cdl/H/BASELINE.md`, `cdl/H/data/`; sibling judge `muse-opt-codex/profiles/fe345625a1ca47fa.json`; ledger `runs/research.json` E011-E030 ((private notes, not included)); harness review `muse-opt-codex/muse_decode_layer/dumb_mistakes.md` ((private notes, not included)).
