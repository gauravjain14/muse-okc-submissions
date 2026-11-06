# 02: native expert kernel v1 → v7, f16 softmax, attn8 on sliding layers, PDL (a stack)

| | before (01) | after (02) |
|---|---|---|
| build | `../01_fp8_attention_full_layers/build.py` = `cand_att8c.py` (md5 04b318ac) | `build.py` = `cand_full10.py` (md5 0c429b21, 356,831 B) |
| judge | 1.6369 (check e4e4ea4469c94941) | **1.7359** (check 436bc4864b4547f0); on the way: full1 1.6678 (8809bba425dd4c33), full3 1.7004 (ba074101f7524773), full5 1.7055 (8fd0e221f34d4a9b) |
| real effect | | ≈ **+6 to +7 %**: nine same-run bench steps of +0.3 to +1.4 % each, chained +6.6 % from `moe1` to full10 alone; judge +6.0 % |
| cases changed | | all 8 (expert kernel, boundaries, routing, attention on every layer, PDL) |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: code copied or adapted from other projects is marked in place with `SPDX-SnippetBegin` / `SPDX-SnippetEnd` comments (source, copyright, license, our changes); see `NOTICE` and `LICENSES/` at the repository root. The code is unchanged. The judged file has md5 `0c429b21`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

This milestone is a stack: ten builds on 2026-10-24, each kept only when a same-run bench against its predecessor was positive. Two lanes ran in parallel: the **expert path** ("MoE worker": the native NVFP4 expert kernel `moe_<cfg>`, its routing and finish kernels, `cand_dense2_moe_vN.py`) and **attention** (attn8 from [01](../01_fp8_attention_full_layers/)). Glue kernels and launch overlap came with them.

## 1. Starting point

att8c's judge profile (`evidence/judge_before_att8c_e4e4ea44.txt`), largest kernels per case:

| case | ours µs | experts `moe_8_16_1_2_*` | attention | two SGLang norm_residual boundaries | FA3 on sliding |
|---|---|---|---|---|---|
| 0 sliding 753 | 652.7 | 332.5 (50.4 %) | – | 41.9 | 25.2 + 7.2 combine |
| 1 full 753 | 1014.3 | 326.4 (32.2 %) | attn_fwd 376.6 (37.2 %) | 42.6 | – |
| 2 sliding 3012 | 1993.5 | 900.5 (44.8 %) | – | 154.7 | 61.2 |
| 3 full 3012 | 3447.8 | 849.8 (24.7 %) | 1482.6 (43.2 %) | 155.0 | – |
| 5 sliding 16K | 10158.5 | 4520.6 (44.4 %) | – | 798.6 | 307.1 |
| 7 full 16K 64K | 16234.4 | 3775.5 (24.0 %) | 6714.1 (42.7 %) | 797.7 | – |

The experts were the largest kernel in every sliding case; attention in the full cases.

## 2. What we saw

**(a) Where each case sits against its floors** (`evidence/roofline_full5.txt`, from `evidence/dev_roofline.py` over an nsys trace of full5, mid-stack; "over" = measured minus the kernel's roofline time):

| case | span µs | experts: µs / % of roofline / over | attention: µs / % / over | gaps |
|---|---|---|---|---|
| 0 | 628.5 | 313.1 / 44 % / 174.8 | 34.4 / 17 % / 28.5 | 6.8 |
| 1 | 931.0 | 285.3 / 47 % / 151.6 | 348.7 / 66 % / 117.1 | 6.5 |
| 3 | 3222.3 | 824.8 / 66 % / 282.3 | 1300.3 / 71 % / 374.0 | 22.4 |
| 5 | 9536.1 | 3917.7 / 79 % / 813.4 | 307.1 / 42 % / 178.7 | 27.7 |
| 7 | 17146.6 | 4486.9 / 66 % / 1547.6 | 7185.1 / 64 % / 2563.2 | 28.2 |

Two kernels hold most of the distance to the floor; inter-kernel gaps are under 1 % of every span. Attention's roofline here is the MUFU exponential bound from 01.

**(b) Attention: half of attn_fwd is the per-tile softmax chain** (`evidence/a8_noexp.log`, attention microbenchmark, ablations that break the output):

| case | attn_fwd | without exp2 | without exp2 and without the row max |
|---|---|---|---|
| 1 full 753 | 379.0 µs | 318.5 (−16 %) | **201.3 (−47 %)** |
| 6 full 16K ctx 32K | 3820.3 µs | 3187.5 (−17 %) | **1910.0 (−50 %)** |

The matmuls plus data movement alone run in about half the time. The chain between the two wgmmas (row max across the tile, subtract, exp2, convert to E4M3, rescale) is the other half, and it runs serially inside each consumer warpgroup.

**(c) The chain is register- and ALU-heavy.** S in FP32 takes 64 registers per thread per 128-key tile, and every max/subtract/convert is a 32-bit op. Computing S in f16 halves both (two scores per register, `HMNMX2`/`HSUB2` on pairs). ncu of attn_fwd, case 1 (`evidence/ncu_a8f.log`): f16 scores + 3/16 of the exps as an f16 polynomial ran 352.5 µs with XU (MUFU) 64.3 %, ALU 46.0 %, FMA 27.8 %, issue 53.5 %; the FP32 version 382.3 µs with XU 72.7 %, FMA 14.2 %, issue 49.0 %.

**(d) attn8 on sliding layers needs a different work split.** Stream-K over 132 CTAs is wasteful when every query block sees at most 2048 keys. "Unit mode" (whole (request, KV head, query block) units dealt round-robin to CTAs) vs stream-K (`evidence/a8_unit1.log` vs `evidence/a8_unit0.log`): case 2 attention 67.9 vs 154.4 µs, case 5 312.0 vs 337.6 µs. At 753 tokens there are too few units for 132 SMs. Splitting a unit in two, with the low half handing its partial O and row sums to the high half via plain stores and a flag (`evidence/a8_split_flag.log`), took case 0 to 29.8 µs (unsplit units: 37.1 µs).

**(e) Expert path** (the MoE worker's real-routing benches; `notes/okc-prefill-moe-worker.md`):
- the expert kernel ran far below its tensor floor at 753 tokens (44-47 % of roofline above);
- routing ran as separate route / count / finalize / pairs kernels;
- mma.sync tiles (the narrow, few-token tiles) re-read activations in their own row-major layout;
- every expert-path launch drained before the next started.

## 3. Diagnosis and projection

- **Experts:** reduce the per-call fixed cost and the per-tile overheads at 753/3012 tokens. The worker's isolated benches projected about +1 % per step.
- **Attention:** the softmax chain is ~50 % of attn_fwd. Halving its register and ALU cost (f16 scores) and moving part of MUFU's load onto the FMA pipe should take 5-10 % off attn_fwd, i.e. +0.5-1 % on full-layer loads (projection from 2(b)-(c)).
- **Sliding attention on attn8:** unit mode with split units beats FA3 from 753 tokens up (2(d)).
- **Glue:** fused residual boundaries and PDL save a few µs per kernel edge.
- No single item was projected above ~1.5 %. The plan was to stack them.

## 4. Change

`change.diff` against 01 (+2,196 / −157 lines). In build order, with the same-run bench gain of each step against its predecessor (`evidence/dev_bench.py`: fixed router seed 728034927264498030, 6 reps, SGLang-paired, cold L2):

| build | what it added | bench step | log |
|---|---|---|---|
| dense2 | **bnd3_b1 / bnd3_b2** boundary kernels replace SGLang's two norm_residual kernels: boundary 1 writes only the FFN input; boundary 2 recomputes the residual stream from s and a instead of storing and reloading it. Router: pre-expert norm + sigmoid / biased top-8 / renormalize fused (**pnr**) | vs dense: case 0 ×1.0138, 1 ×1.0086, 2 ×1.0080, 3 ×1.0040, 16K ±1 % | `dense2_bench_04.log` |
| att8d | attn8 also on sliding layers of ≥ 4096 tokens | +1.07 % vs dense2 | `bench_att8d.log` |
| moe8d / moe1 | MoE worker v1 (`cand_moe_on_att8d[_v1].py`): the native expert kernel retuned for prefill shapes | +1.35 % (16K +2-3 %) | `moe_full_att8d.log` |
| full1 | MoE v2; attn8 on every sliding layer (unit mode) | +0.60 % | `bench_full1.log` |
| full2 | MoE v3 (adds SiLU as 0.5·g·(1 + tanh(g/2)), one MUFU op per intermediate); `ATT_PP 2` (the ping-pong turn covers only the QK issue); PDL into `qkv_post2` and `attn_fwd` | +0.60 % | `bench_full2.log` |
| full3 | MoE v4; `qkv_post2` writes the new keys straight into packed tiles (no tail kv_pack) | +0.37 % | `bench_full3.log` |
| full4 | **`ATT_F16S`**: QK accumulates in f16 (32 instead of 64 registers per S tile); max, subtract, exp and the FP8 cast on f16x2 pairs | +0.76 % | `bench_full4.log` |
| full5 | **`ATT_POLYH 3`**: 3 of the 16 n8 blocks compute exp2 as an f16x2 degree-2 polynomial on the FMA pipe (round through the 1551 magic, exponent from its low 6 bits), the rest on MUFU | **+1.03 %** (full cases +2.3 to +3.9 %) | `bench_full5.log` |
| full6 | `patch_pdl`: PDL for bnd3 and pnr (`griddepcontrol.wait` / `launch_dependents`) | +0.29 % | `bench_full6.log` |
| full7 | direct new-key packing only up to 1024 tokens (its byte scatter loses on larger blocks) | +0.79 % (16K +1 to +2.3 %) | `bench_full7.log` |
| full9 | **MoE v6p**: tokens bucketed per expert inside the router kernel (route/count/finalize/pairs kernels removed), PDL between the expert-path kernels, at 16K one N = 512 router GEMM over [hi; lo] + pnr2. attn8: sliding units split in two with the plain-store handoff (`hacc`/`hl` scratch), conflict-free kv_pack (each thread turns 4×4-byte blocks with PRMT) | **+1.40 %** (case 0 +3.8 %) | `bench_full9.log` |
| full10 | **MoE v7**: mma.sync tiles read the wgmma activation layout through ldmatrix (`MOE_LMX`: one activation layout for all tiles). Attention source identical to full9 | +0.59 % | `bench_full10.log` |

The steps from moe1 to full10 multiply to **+6.6 %** (computed from the logs above). Other expert-kernel features of v1-v6p, with the gains the worker measured for them in the source comments (`change.diff`): wgmma groups kept in flight across stage pairs, weight words loaded one group ahead (+1-1.7 %), the gate/up epilogue straight from the accumulators (+1-1.5 % at 3012/16K). The expert-kernel versions alone, isolated in one bench each: v4 → v6p +1.56 % (`evidence/bench_v6.log`, 1.6842 → 1.7105), v6p → v7 +0.96 % (`evidence/bench_v8.log`, 1.7023 → 1.7186). Attention steps in the attention microbenchmark, case 1: f32 → f16 scores 377.3 → 371.4 µs (`evidence/a8_f16s.log`); `POLYH` 0 → 3: 369.6 → 351.6 µs (case 3 1441.4 → 1366.6, `evidence/a8_polyh2.log`).

## 5. Result

| check | result |
|---|---|
| bench, chained same-run steps | moe1 → full10 **+6.6 %**. Not measured in one run: att8c → moe1 (dense2 boundaries, attn8 on 16K sliding, MoE v1), whose own steps were +0.4 to +1.4 %. Bench geomean of att8c 1.6470 (`01/evidence/bench_att8c.log`) vs full10 1.7695 (`evidence/bench_full10.log`), different runs: +7.4 % |
| correctness | judge 5/5 tests on every check; bench `--check` on all 8 cases each step (full10, case 1: update relative L2 6.21e-2 against a bound of 0.150, `evidence/bench_full10.log`) |
| judge | 1.6369 → 1.6678 (full1) → 1.7004 (full3) → 1.7055 (full5) → **1.7359** (full10) |

Judge, att8c → full10, per load (computed with the judge's formula from the per-case medians in `evidence/judge_*`): turn 1 1.6434 → 1.7526, turn 4 1.5960 → 1.6931, first turn 1.6722 → 1.7628. Per kernel:

| case | ours µs | experts | attention | boundaries (2 kernels) |
|---|---|---|---|---|
| 0 sliding 753 | 652.7 → 594.7 | 332.5 → 300.8 | FA3 25.2 + 7.2 → attn_fwd 24.7 | 41.9 → 37.7 |
| 1 full 753 | 1014.3 → 928.0 | 326.4 → 313.2 | 376.6 → 354.4 | 42.6 → 35.3 |
| 2 sliding 3012 | 1993.5 → 1870.2 | 900.5 → 838.4 | FA3 61.2 → 61.6 | 154.7 → 143.2 |
| 3 full 3012 | 3447.8 → 3236.7 | 849.8 → 830.2 | 1482.6 → 1376.3 | 155.0 → 144.9 |
| 4 full 16K | 10290.9 → 10307.4 | 4141.8 → **4444.4** | 895.6 → 824.3 | 798.4 → 768.1 |
| 5 sliding 16K | 10158.5 → 9327.7 | 4520.6 → **3911.3** | FA3 307.1 → attn_fwd 317.5 | 798.6 → 764.8 |
| 6 full 16K 32K | 13268.2 → 12992.9 | 3897.5 → 3946.8 | 3847.7 → 3517.9 | 797.6 → 770.5 |
| 7 full 16K 64K | 16234.4 → 15601.9 | 3775.5 → 3938.3 | 6714.1 → 6272.9 | 797.7 → 766.8 |

**Read honestly:**
- The 16K expert rows move ±10 % in both directions (case 4 +302 µs, case 5 −609 µs). The judge draws a new router for every run, so the expert work itself changes between runs; these rows are not the kernel's effect.
- The 753/3012 rows (case 0 −32 µs, case 2 −62 µs) agree with the bench steps.
- SGLang also moved between the two runs (case 4 17249 → 18662 µs). Re-scoring full10's times against att8c's SGLang medians gives 1.7618 (+7.6 %); against full5's, 1.7424.
- So the judge places the stack at +6.0 to +7.6 % depending on which SGLang draw it is paired with. The bench says +6.6 % from moe1 alone and ~+7.4 % across runs. Real effect: **≈ +6 to +7 %**.
- The judge ran about 1.9 % below our bench on full10 (bench 1.7695 vs judge 1.7359), consistent with the calibration of the time ((private notes, not included):29`).

**Against the projection.** The expert steps delivered about what the worker's benches promised (+1.0 to +1.6 % per step). The attention changes delivered less than projected: f16 scores + POLYH took attn_fwd only −6 % (case 1 376.6 → 354.4 µs in the judge), against the ~50 % that 2(b) showed the softmax chain to cost. The f16 path still sends every exponential through MUFU (`ex2.approx.f16x2` costs two MUFU issues per pair, (private notes, not included):24`), and the polynomial blocks only trade MUFU slots for FMA/ALU slots that are just as busy (XU 64 %, ALU 46 %, issue 53 % in 2(c)).

## 6. Takeaways, and what came next

- **Ablate the chain before tuning it.** "No exp, no max" halved attn_fwd. That bounded every softmax lever at ~50 % of the kernel, and it showed the exponential alone at ~16 %.
- **Moving work between MUFU and FMA does not pay when both pipes and the issue stream are busy.** The polynomial helped only at 3/16 of the blocks.
- **Sliding attention wants whole units, not stream-K.** A two-way split with a plain-store handoff fixes the 753-token case.
- **Small steps compound:** nine positive steps of 0.3-1.4 % gave +6.6 %. The judge could only resolve the sum.
- Next, [03](../03_expq_fp8_exponential/): P is consumed as an FP8 E4M3 code anyway, so the exponential can be produced *as that code* with one f16 FMA and no MUFU at all.

## Reproduce

```bash
cd ../tools
python research/harness/check.py ../02_native_moe_f16_softmax_pdl/build.py --edges
python research/judge/replica.py --candidate att8c=../01_fp8_attention_full_layers/build.py \
       --candidate full10=../02_native_moe_f16_softmax_pdl/build.py --panel 11,22,33 --reps 8
```

`evidence/dev_roofline.py`, `dev_prof_table.py`, `dev_layer_prof.py` and `dev_patch_pdl.py` are this era's profiling and patch tools; `full5_cuda_gpu_trace.csv` + `full5_nvtx_pushpop_trace.csv` are the nsys exports `roofline_full5.txt` was computed from (`python3 dev/roofline.py <gpu_trace.csv> <nvtx.csv>`). The attention microbenchmark is `../01_fp8_attention_full_layers/evidence/dev_attn_test.py`. These scripts expect the layout of `muse-opt/muse_prefill_layer/`.

Originals: `muse-opt/muse_prefill_layer/cand_full10.py` (= `checks/full10.py`), the intermediates `cand_full{1..9}.py`, `cand_dense2.py`, `cand_att8d.py`, `cand_moe_on_att8d*.py`, `cand_dense2_moe_v{1..7}.py`, `dev/attn8.py`, `dev/build_cand.py`, `dev/patch_pdl.py`, `logs/`, `prof/`. Judge results: `muse-opt/muse_prefill_layer/checks/profiles/{full1,full3,full5,full10}.*`.
