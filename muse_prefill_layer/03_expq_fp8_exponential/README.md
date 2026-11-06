# 03: EXPQ, the softmax exponential written directly as its FP8 code (no MUFU)

| | before (02) | after (03) |
|---|---|---|
| build | `../02_native_moe_f16_softmax_pdl/build.py` = `cand_full10.py` (md5 0c429b21) | `build.py` = `cand_full11.py` (md5 9d946f1a, 358,480 B) |
| judge | 1.7359 (check 436bc4864b4547f0) | **1.8310** (check 6da1e37996f849e5) |
| real effect | | ≈ **+4.3 to +4.5 %** of score (the judge's +5.5 % re-scored against one SGLang draw); attn_fwd −25 to −33 % on full layers, −11 to −22 % on sliding layers (judge profiles) |
| cases changed | | attention on all 8 cases (attn8 runs every layer since 02); the full-layer cases 1, 3, 4, 6, 7 carry the gain |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: code copied or adapted from other projects is marked in place with `SPDX-SnippetBegin` / `SPDX-SnippetEnd` comments (source, copyright, license, our changes); see `NOTICE` and `LICENSES/` at the repository root. The code is unchanged. The judged file has md5 `9d946f1a`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

The whole change is one compile-time switch in attn8, `ATT_EXPQ 1` in place of `ATT_POLYH 3` (`change.diff`, +38 lines).

## 1. Starting point

full10's judge run (`evidence/judge_before_full10_436bc486.txt`):

| case | ours µs | ratio | attn_fwd µs | attn_fwd share of the case |
|---|---|---|---|---|
| 0 sliding 753 | 594.7 | 1.852 | 24.7 | 4 % |
| 1 full 753 | 928.0 | **1.570** | 354.4 | **38 %** |
| 2 sliding 3012 | 1870.2 | 1.793 | 61.6 | 3 % |
| 3 full 3012 | 3236.7 | **1.527** | 1376.3 | **43 %** |
| 4 full 16K ctx 0 | 10307.4 | 1.811 | 824.3 | 8 % |
| 5 sliding 16K | 9327.7 | 1.795 | 317.5 | 3 % |
| 6 full 16K ctx 32K | 12992.9 | 1.691 | 3517.9 | 27 % |
| 7 full 16K ctx 64K | 15601.9 | **1.626** | 6272.9 | **40 %** |

The four lowest ratios were full-attention cases, and attention was their largest kernel.

## 2. What we saw

**(a) attn_fwd sat at 59-71 % of a floor set by the exponential unit** (`evidence/roofline_full10.txt`, from `evidence/dev_roofline.py` over the full10 nsys trace in `evidence/full10_*.csv`). The tool takes a kernel's floor as the largest of FLOP at the FP8 peak, bytes at 4.8 TB/s, and, for attention, one MUFU `ex2` per score at 16 per clock per SM (`dev_roofline.py:17-20, 64-69`). For attention the exponential term is the largest: case 1 has 968 M scores, so 231.6 µs for MUFU against 125 µs for the tensor cores (247.9 GFLOP / 1979 TF/s, computed).

| case | attn_fwd µs | MUFU floor µs | % of floor | over floor µs | rank among the case's kernels |
|---|---|---|---|---|---|
| 1 | 350.9 | 231.6 | 66 % | 119.3 | 2nd (experts 159.2) |
| 3 | 1301.0 | 926.3 | 71 % | 374.7 | **1st** |
| 4 | 820.0 | 513.6 | 63 % | 306.4 | 2nd |
| 6 | 4366.2 | 2567.7 | 59 % | 1798.5 | **1st** |
| 7 | 7129.3 | 4621.9 | 65 % | 2507.4 | **1st** |

**(b) MUFU was still the busiest pipe after 02's f16 changes** (`evidence/ncu_a8f.log`, case 1): with f16 scores and 3/16 of the exps on the FMA pipe, XU (the MUFU pipe) 64.3 %, ALU 46.0 %, FMA 27.8 %, issue 53.5 %; FP32 scores with all exps on MUFU: XU 72.7 % (`evidence/ncu_poly.log`: 73.0 %). `ex2.approx.f16x2` costs two MUFU issues per pair ((private notes, not included):24`), so f16 scores did not halve MUFU's work. Removing the exponential alone took attn_fwd from 379.0 to 318.5 µs at case 1 and from 3820 to 3188 µs at case 6, −16 to −17 % (`evidence/a8_noexp.log`).

**(c) The exponential's output is consumed at FP8 resolution.** P goes into the P·V wgmma as E4M3: 3 mantissa bits, a relative step of 6.25-12.5 %. An E4M3 byte read as an integer is `(exponent + 7) · 8 + mantissa`. That is a piecewise-linear function of log2(P): for x = S − m ≤ 0, the code 8x + 56 is exactly the encoding of 2^⌊x⌋ · (1 + frac(x)). This is Schraudolph's linear-mantissa approximation of 2^x; its worst relative error is 6.1 % at frac(x) = 0.443 (computed), the same size as E4M3's own rounding. So the code can come straight from an affine function of the score, with no `exp2` at all.

**(d) The f16 FMA can produce that integer.** For y in [1024, 2048) the f16 unit in the last place is exactly 1, so `y = 8·S + (1080 − 8m)` computed by one `HFMA2` is rounded to the nearest integer. The low byte of y's bit pattern is then `y − 1024 = 8(S − m) + 56`, the E4M3 code of 2^(S−m). Scores more than 7 below the running max give y < 1024; one `HMNMX2` against 1024.0 (`0x6400`) clamps them to code 0. A `PRMT` with selector `0x6420` gathers the low bytes of two f16x2 registers into the four FP8 codes of the P fragment. For the offset to be exact, 8m must be an integer, so m is rounded to a multiple of 1/8 whenever it is raised.

## 3. Diagnosis and projection

- In the per-tile softmax chain (row max → subtract → exp2 → cast to E4M3 → pack), everything after the max collapses to `HFMA2` + `HMNMX2` per score pair plus one `PRMT` per 4 scores. MUFU, the subtract, the clamp and the `cvt.e4m3x2` all go away.
- **Projection:** the no-exp ablation bounds the exponential alone at −16 to −17 % of attn_fwd. Applied to the judge's attn_fwd times, −16 % is **+2.5 %** of score and −20 % is +3.1 % (computed with the judge's load formula). The subtract and the cast also disappear, so the real number should be larger.
- **Accuracy cost:** up to 6.1 % per P element, before E4M3 rounding. Averaged over thousands of keys, it should move the layer error by a few % relative.

## 4. Change

`change.diff` (attn8 source in `ATT_CUDA_SRC`, plus the define list):

```cuda
// on a max raise: m := t - 8, rounded so that 8 m is an integer (the code offset is then exact)
ma2 = rintf(ma2 * 8.f) * 0.125f;
// y = 8 S + (1080 - 8 m) lands in [1024, 2048) (ulp 1) for kept scores: its low byte is the
// E4M3 code of 2^(S - m) with a linear mantissa; below 1024 (S - m < -7, masked) -> code 0
const float ca = 1080.f - 8.f * fminf(fmaxf(ma, -1000.f), 1000.f);
hs[n8][0] = h2max(h2fma(hs[n8][0], 0x48004800u /* 8.0 */, cha), 0x64006400u /* 1024.0 */);
...
pf[kc][0] = prmt_b(hs[4 * kc][0], hs[4 * kc + 1][0], 0x6420);   // low bytes -> 4 FP8 codes
```

The rest of attn8 is unchanged: f16 scores (`ATT_F16S 1`), the rescale trigger (a new max more than 8 above m raises m to t − 8, so P ≤ 2^8), and the row sum from the ones rows of Vᵀ. The rare rescale factor (`ex2_mufu(m_old − m_new)`) stays on MUFU. `ATT_DEFINES` becomes `['ATT_CW 3', 'ATT_POLY 0', 'ATT_PP 2', 'ATT_F16S 1', 'ATT_EXPQ 1']`.

Microbenchmark of the change (`evidence/a8_expq.log`, attention path only; "cur" = full10's attention):

| case | full10 attn_fwd µs | EXPQ µs | speedup | error vs BF16-q reference: EXPQ / SGLang FA3 |
|---|---|---|---|---|
| 0 sliding 753 | 29.0 | 25.7 | 1.13× | 5.84e-2 / 5.76e-2 |
| 1 full 753 | 353.3 | **255.2** | **1.38×** | 7.27e-2 / 7.13e-2 |
| 2 sliding 3012 | 66.6 | 54.5 | 1.22× | 6.78e-2 / 6.76e-2 |
| 3 full 3012 | 1369.3 | **975.5** | **1.40×** | 7.54e-2 / 7.34e-2 |
| 4 full 16K | 826.4 | 611.9 | 1.35× | 2.41e-2 / 2.48e-2 |
| 5 sliding 16K | 311.8 | 249.3 | 1.25× | 3.79e-2 / 4.11e-2 |
| 6 full 16K 32K | 3490.5 | **2421.7** | **1.44×** | 6.77e-2 / 6.48e-2 |

Follow-up sweep with EXPQ on (`evidence/a8_expq_sweep.log`): `ATT_PP 0` (no ping-pong) was faster again on full layers, case 6 2433.9 → 2386.2 µs. That became the next build, full12, outside this milestone.

## 5. Result

| check | result |
|---|---|
| attention microbenchmark | above: full layers −26 to −31 %, sliding −11 to −20 % |
| layer correctness (`evidence/bench_full12.log`, full11 rows; full10 in `evidence/bench_full10.log`) | update relative L2 error, full10 → full11: case 1 6.21e-2 → 6.46e-2 (bound 0.150), case 3 6.29e-2 → 6.53e-2, case 6 5.21e-2 → 5.36e-2, case 7 5.95e-2 → 6.11e-2: **+2.6 to +4.1 % relative**, all far inside the bounds; judge 5/5 tests |
| ncu after (`evidence/ncu_expq.log`; case 1 by its 253.8 µs duration, inferred) | 253.8 µs; ALU 50.6 %, FMA 23.6 %, issue 48.7 %; top stalls wait 1.29, not_selected 0.84, math_pipe_throttle 0.79 per issue. MUFU dropped out of the list: XU 1.3-1.9 % in the next day's ncu, with tensor ~67-70 % of the FP8 peak at 1.77 GHz ((private notes, not included):61`) |
| roofline after (`evidence/roofline_full12.txt`, full12 = full11 + `ATT_PP 0`) | attention at 94 % (case 1), 101 % (case 3), 86 % (cases 4, 6), 89 % (case 7) of the old MUFU floor, which no longer binds. Against the FP8 tensor floor, case 1 now runs at ~2.0× (246.5 vs 125 µs, computed) |
| judge | 1.7359 → **1.8310** (`evidence/judge_after_full11_6da1e379.txt`) |

Judge per case, full10 → full11 (per kernel from the judge's profiles):

| case | ours µs | SGLang µs | ratio | attn_fwd µs |
|---|---|---|---|---|
| 0 | 594.7 → 584.6 | 1101.2 → 1094.7 | 1.852 → 1.873 | 24.7 → 21.9 |
| 1 | 928.0 → **818.4** | 1456.9 → 1474.0 | 1.570 → **1.801** | 354.4 → **257.6** |
| 2 | 1870.2 → 1871.7 | 3352.9 → **3462.4** | 1.793 → 1.850 | 61.6 → 48.8 |
| 3 | 3236.7 → **2843.5** | 4943.7 → 4941.5 | 1.527 → **1.738** | 1376.3 → **982.4** |
| 4 | 10307.4 → 9886.9 | 18661.8 → 18150.1 | 1.811 → 1.836 | 824.3 → 620.7 |
| 5 | 9327.7 → 9358.3 | 16742.5 → 17117.8 | 1.795 → 1.829 | 317.5 → 249.0 |
| 6 | 12992.9 → **11591.5** | 21968.2 → 21170.9 | 1.691 → **1.826** | 3517.9 → **2418.3** |
| 7 | 15601.9 → **13277.0** | 25365.7 → 24684.0 | 1.626 → **1.859** | 6272.9 → **4210.1** |

Per load: turn 1 1.7526 → 1.8492, turn 4 1.6931 → 1.8112, first turn 1.7628 → 1.8329 (computed).

**Read honestly:**
- The kernels whose code did not change stayed put: the nvjet GEMMs within ±1 % (case 4: 1646.0 → 1648.0, 1349.6 → 1348.7, 749.5 → 749.8 µs) and the boundaries within ±1 % (bnd3_b2 485.6 → 484.0).
- The expert kernel moved by ±5 % (case 5 3911 → 4057, case 7 3938 → 3676 µs) because the judge draws a new router every run.
- SGLang moved too: case 2 ran 3.3 % slower, which alone lifted case 2's ratio +3.2 % while our time was flat (1870.2 → 1871.7 µs).
- Re-scoring with full10's SGLang medians: full10 plus only the judge's attn_fwd deltas gives 1.8101 (**+4.3 %**); full11's own medians give 1.8148 (**+4.5 %**). The remaining ~1 % of the judge's +5.5 % was the SGLang draw.
- The CIs do not overlap (1.710-1.762 vs 1.804-1.859), so this is the one step of this era the judge resolves on its own.

**Against the projection.** The exp-only ablation projected −16 % of attn_fwd and +2.5 % of score. EXPQ took −25 to −33 % on full layers and +4.3 % of score. The ablation had kept the subtract, the E4M3 conversion and the packing; EXPQ folds all three into the same FMA and a PRMT. Of the per-tile chain, only the row max is left (the "no exp, no max" ablation was 201 µs at case 1, `evidence/a8_noexp.log`; EXPQ's case 1 is 255 µs).

## 6. Takeaways, and what came next

- **When an output is quantized, compute it at the output's resolution.** E4M3 needs 3 mantissa bits; Schraudolph's linear mantissa is accurate to that level, and the f16 FMA's integer rounding in [1024, 2048) emits the FP8 code directly. That removes MUFU, the subtract and the conversion from the softmax in one instruction per pair.
- **Check what an ablation leaves in.** "No exp" kept the work around the exp, so it underestimated the gain by 2×.
- **The floor moved.** With MUFU gone, attention is bounded by the tensor cores (~2× away at case 1), and the remaining chain (row max, rescale) runs at ~50 % issue with no saturated pipe ((private notes, not included):61`). The case-1 roofline now puts the experts first: 160 µs over their floor, against 15 µs for attention (`evidence/roofline_full12.txt`).
- Next: `ATT_PP 0` (full12, the first scored run) and then [04](../04_moe_vn_expert_chain/), the expert-kernel chain v9 → v24b, since the experts had become the largest gap in every case.

## Reproduce

```bash
cd ../tools
python research/harness/check.py ../03_expq_fp8_exponential/build.py --edges
python research/judge/replica.py --candidate full10=../02_native_moe_f16_softmax_pdl/build.py \
       --candidate full11=../03_expq_fp8_exponential/build.py --panel 11,22,33 --reps 8 --cases 1,3,6,7
```

The attention microbenchmark (`../01_fp8_attention_full_layers/evidence/dev_attn_test.py`) compares attn8 variants by define: `ATT_MOD=attn8 python3 dev/attn_test.py --defines "ATT_CW 3,ATT_POLY 0,ATT_PP 2,ATT_F16S 1,ATT_EXPQ 1"` from `muse-opt/muse_prefill_layer/`. The roofline tables come from `python3 dev/roofline.py evidence/full1{0,2}_cuda_gpu_trace.csv evidence/full1{0,2}_nvtx_pushpop_trace.csv`.

Originals: `muse-opt/muse_prefill_layer/cand_full11.py` (= `checks/full11.py`), `dev/attn8.py` (`ATT_EXPQ`), `logs/a8_expq*.log`, `logs/ncu_expq.log`, `logs/roofline_full1{0,2}.txt`, `prof/full1{0,2}_*.csv`. Judge result: `muse-opt/muse_prefill_layer/checks/profiles/full11.6da1e37996f849e5.*`.
