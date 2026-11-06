# 06: the public okc-2026 expert kernel for T ≥ 2048, plus an exact early vote in full attention

| | before (05) | after (06) |
|---|---|---|
| build | `05_sk_only_attention_fused_unpool/build.py` = `cand_full28.py` (md5 d624097a, 506,194 B) | `build.py` = Codex's `research/attention/bundle2_public_vote_full.py` (md5 21732359, 788,295 B, sha256 9f8e2ce3…) |
| judge | 1.8866 (check b1033df7778e4586) | **1.9232** (check ee0692843fb94ded) and **1.9201** (scored 455b4f71a2a940b9), same file; real level **1.9217** |
| real effect | | local same-run paired A/B vs 05: **+2.16 %** (one router draw); judge mean +1.86 % |
| shapes changed | | all 8 cases; the large part is the expert path at T = 3012 and 16384 (cases 2-7) and full attention (cases 1, 3, 4, 6, 7) |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: code copied or adapted from other projects is marked in place with `SPDX-SnippetBegin` / `SPDX-SnippetEnd` comments (source, copyright, license, our changes); see `NOTICE` and `LICENSES/` at the repository root. The code is unchanged. The judged file has md5 `21732359`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

Third-party code: this build is the first that contains `okc_moe::moe_fused_persistent` from vjeux/okc-2026
(Apache-2.0). See the root `NOTICE`. Among the judged files it is the only one that carried the upstream notice inline
(judged `build.py` lines 11106-11313); in this repository every build marks the borrowed code in place.

Source shorthand: paths are relative to ``; `R/` = `muse-opt-codex/muse_prefill_layer/research/`.
`evidence/` and `notes/` are this folder's copies.

Terms used below:
- **Expert path / MoE**: the 32 local NVFP4 experts (W13 8192×4096, W2 4096×4096 each, top-8 of 256 routing). The
  earlier native kernel `moe_8_16_1_2_<M>` (one persistent 384-thread CTA per SM, 256 weight rows × 128 tokens per
  item, FP4 codes decoded in registers and fed to RS-wgmma) is the vN chain of milestones 04-05 (v29 in 05).
- **T** (or M): tokens in the layer call: 753 (cases 0, 1), 3012 (cases 2, 3), 16384 (cases 4-7).
- **PG**: the native "pre-gathered" path that writes routed rows straight into per-expert slots at large T.
- **Codex**: the agent that owned the task on 10-29 (this milestone and 07). Its harness `R/harness/bench.py`
  times control and candidates as AB/BA pairs on one fixed router seed and reports a paired relative speedup.
- **Paired ratio / load speedup**: the judge's per-case number is a paired SGLang/ours ratio; a load (turn 1 = cases
  0-1, turn 4 = cases 2-3, first turn = cases 4-7) is Σw·t_SGLang / Σw·t_ours; the score is the geomean of the
  three loads (`tools/TASK.md`).

## 1. Starting point

05's judged run (`evidence/judge_before_full28_b1033df7.txt`) and the expert kernel's share of each case in that
run's per-kernel profile:

| case | shape | judge ours / SGLang (µs) | ratio | native MoE kernel (µs, share) | local control (µs) |
|---|---|---|---|---|---|
| 0 | T753 sliding | 556.4 / 1061.2 | 1.907 | 279.2 (48.8 %) | 573.7 |
| 1 | T753 full | 761.9 / 1450.7 | 1.904 | 276.6 (34.4 %) | 788.1 |
| 2 | T3012 sliding | 1840.6 / 3442.7 | 1.870 | 816.7 (44.4 %) | 1840.1 |
| 3 | T3012 full | 2620.9 / 4947.6 | 1.888 | 839.7 (29.4 %) | 2670.3 |
| 4 | 16K ctx0 full | 9378.9 / 17332.4 | 1.848 | 3954.2 (41.9 %) | 9580.4 |
| 5 | 16K sliding | 9178.8 / 17132.8 | 1.867 | 4061.7 (44.2 %) | 9193.6 |
| 6 | 16K ctx32K full | 11307.3 / 21667.5 | 1.916 | 4185.6 (36.8 %) | 11333.5 |
| 7 | 16K ctx64K full | 12956.0 / 25039.8 | 1.933 | 3948.0 (30.7 %) | 13146.8 |

Local control = `cand_full28.py` in `evidence/public_vote_bench.log`. Judge loads: turn 1 1.9060, turn 4 1.8760,
first turn 1.8779 (computed from the judge table with the TASK.md weights; the geomean reproduces 1.8866).
Full attention (`attn_fwd`) was the second-largest kernel on full layers: 214.1 µs (case 1), 817.2 (case 3),
2183.8 (case 6), 3890.6 µs (case 7) in the same profile.

## 2. What we saw

**(a) The native expert kernel was already near its tensor floor in the main loop; the losses were around it.**
From the 10-25/26 measurements carried in `notes/codex_moe_NOTES.md` ("Existing evidence", "Round 2 and structural
bounds"):
- At T = 16384 a 256×128×256 stage took 4241 clocks against a 4096-clock tensor ceiling: 3.543 ms of compute against
  a 3.421 ms cycle floor at 1.679 GHz, plus ~0.25 ms of epilogue and ~0.16 ms of stage waits that were not hidden.
- Small experts (≤ 32 routes) took ~35 % of CTA time at T = 753 for ~8 % of the routes (`moe_tl_v42_v29.log`, cited there).
- ptxas warnings were part of the picture: variants that kept more accumulator chains live (tiny-WGMMA "group8") hit
  C7512, "wgmma serialized due to insufficient register resources", which slows every wgmma in the kernel, not only
  the new path (`notes/codex_moe_NOTES.md`, "Round 2").

**(b) A public kernel had a different shape on exactly those losses.** vjeux/okc-2026 (Apache-2.0, commit f35a41ee,
cloned to `R/public_okc_2026/`) was read side by side with ours (`notes/codex_moe_NOTES.md`, "Public implementation
architecture"):

| | native (milestone 04) | public `okc_moe::moe_fused_persistent` |
|---|---|---|
| item shape | 256 weight rows × 128 tokens | **128 weight rows × 256 tokens**: each decoded FP4 weight feeds twice the tokens, so decode work per output FLOP halves |
| accumulator chains | one per width | 4 independent chains at width 32, 2 at 64/128 (hides wgmma recurrence latency) |
| epilogue | consumers copy their tile out before taking the next item | staging buffer + a **publisher warp** issues the gate bulk S2G and down copies while consumers start the next item |
| queue / plan | expert-ordered | one persistent queue, all gate items before all down items, tiles sorted by **width** (32/64/128/160/192/256) |

**(c) MoE-only A/B on the layer's real routing** (`evidence/moe_public_bound.log`, same inputs, reference error
2.87e-3 for both kernels at every point):

| T | seed 0 | seed 1 |
|---|---|---|
| 753 | ours 302.8, public 301.3 µs (×1.005) | ours 306.0, public 318.2 µs (**×0.963**) |
| 3012 | 904.4 → 877.4 µs (**×1.029**) | 896.8 → 856.8 µs (**×1.046**) |
| 16384 | 4696.2 → 4461.3 µs (**×1.054**) | 4540.3 → 4296.3 µs (**×1.057**) |

The public kernel wins where items are wide and loses at T = 753, where the tiny-expert tail dominates. Decision:
route **2048 ≤ T ≤ 16384** to the public kernel and keep the native kernel below.

**(d) The public attention was not a candidate, but one idea in it was.** On our cases the public attention variant
(FP32 QK scores, FP32 exp2 on the SFU) was far slower than ours (`evidence/public_attention.log`): case 1 368.5 vs
243.4 µs, case 3 1425.9 vs 896.4 µs, case 4 848.1 vs 544.0 µs (16K full: 6848 vs 3862 µs, `notes/codex_attention_NOTES.md`).
Its row-max update order, however, ports exactly: every lane already holds its partial row maxima; a warp-wide
`__any_sync` vote on "does any lane exceed the current threshold m + 8" decides whether the four row-max shuffles
are needed at all. If no lane exceeds it, no row can trigger a rescale. Same file: `epi_votemax` error 0.0 on all
probed cases and −4.1 µs on case 1, −12.2 µs on case 3, −5.0 µs on case 4 (`evidence/public_attention.log`).

**(e) Small exact gains from the takeover audit of 05** (`notes/codex_RUN_STATE.md`, "Current session results" and
"Later session update"; `notes/codex_dense_NOTES.md`):
- attention epilogue with a fast division and a factored norm: −1.8 / −2.8 / −15.6 µs on sliding cases 0 / 2 / 5,
  output error ≤ 2.13e-5;
- native MoE "pair-diet" (two block scales converted and applied per bf16x2 multiply, `mopt_dequant_pair`) and
  `stmatrix` staging of the down epilogue: +0.3-0.5 % MoE-only, error unchanged at 2.87e-3;
- `stmatrix` staging of the gate/up epilogue. An earlier attempt (v46, 10-25) had been written off with relative
  error 1.0. The audit found the cause in its builder, which had dropped the W2 repack for every expert but #31.
  Repaired: +0.2-0.4 % at T = 753, +0.3-0.4 % at 3012, +0.1 % at 16K;
- **bk4** at 16K: the o_proj GEMM with the post-attention boundary (bnd1) time-sliced into it, from the `gemm_ub2`
  family of milestone 05. Sweep `evidence/oproj_sweep.log`: best G4/R16/cap32, 597.2 → 572.1 µs (−4.3 %).

## 3. Diagnosis and projection

- The expert path was 29-49 % of every case (table above). At T ≥ 2048 its losses were structural (decode issue per
  FLOP, an unhidden epilogue), not a knob, so a kernel built around a different tile shape was the lever.
- Projection from (c): MoE −3.7 % (3012) and −5.5 % (16K), times its ~44 % share, gives about −1.6 % on cases 2-3 and
  −2.4 % on cases 4-7. The case 0-1 MoE stays native. With the vote (≈ −2 % of `attn_fwd`, 20-30 % of a full case)
  and the exact items in (e) adding a few tenths: **≈ +2 %** score (inferred).
- What it would cost: the public kernel needs its own gather of normalized rows and its own combine. The native PG
  path and `_native_finish` are switched off on the public branch, so those costs move into new kernels.

## 4. Change

The build is the end of a five-step chain. Each step is a frozen file under
`muse-opt-codex/muse_prefill_layer/`; `evidence/change_by_step.diff` holds the five diffs in order (their end points
match `../05_*/build.py` and `build.py` byte for byte). The local numbers are Codex's paired benches on its fixed
router seed 728034927264498030.

| step | file | what it adds | local paired gain (source) |
|---|---|---|---|
| 1 | `research/integrated/full28_epi_pair_stsm.py` | attention epilogue fast division + factored norm; MoE pair-scale decode diet; down-epilogue `stmatrix` (`MOE_STSM`) | +0.51 % vs 05 (`evidence/bundle1_bench.log`) |
| 2 | `research/integrated/bundle1_bk4.py` | bk4: 16K o_proj + bnd1 fused, G4 / R16 / cap 32 | first-turn load +0.25 % (`evidence/bundle1_bk4.log`) |
| 3 | `research/integrated/bundle2.py` | gate/up `stmatrix` epilogue (builder bug repaired) | +0.21 % vs step 2 (`evidence/bundle2_bench.log`) |
| 4 | `research/moe/bundle2_public.py` | public expert kernel for 2048 ≤ T ≤ 16384 | +1.35 % vs step 2, step 3 included (`evidence/bundle2_bench.log`) |
| 5 | `research/attention/bundle2_public_vote_full.py` | `ATT_VOTEMAX` early vote, in a separate stream-K module used only for full layers (`window == 0`) | step 3 + vote (`research/attention/bundle2_vote_full.py`): +0.43 % vs step 2, i.e. ≈ +0.22 % over step 3 (`evidence/bundle2_bench.log`) |
| all | `build.py` | | **+2.16 %** vs 05 (`evidence/public_vote_bench.log`) |

How the public kernel was integrated (`notes/codex_moe_NOTES.md`, "Public MoE gain and full-layer integration";
the inline header at `build.py:11106`):
- The upstream source is appended as a self-contained component: 34 symbols with a `_pm_` prefix and a C++/CUDA
  extension built once in `prepare` with `torch.utils.cpp_extension.load_inline` (`-O3 --use_fast_math`, `sm_90a` on H200, `build.py:11444-11447`;
  44-49 s cold in `evidence/moe_public_bound.log` / `evidence/bundle2_bench.log`). The upstream attention, QK and
  boundary functions are compiled along with it but never called (a `--moe-only` trim came later, `notes/codex_moe_NOTES.md`).
- Routing stays ours: the native fused router's bucket counts and positions are reused, local expert ids are guarded
  against ≥ 32, and the native PG gather is disabled on the public branch. The router writes normalized rows, and the
  public gather (`okc_moe::moe_gather_kernel`) only rearranges them (identity pre-norm).
- The 32 routing counters are reset inside the ordered combine's CTA 0 (`_pm__combine_kernel`), so every call and
  every graph replay starts fresh without an extra launch.
- Dispatch: `_experts` sends `2048 <= M <= MAX_M` to the public path and everything else to the previous native
  `_experts` (`build.py:11478-11483`); the router skips its PG gather in the same range (`build.py:10998`).

The vote (step 5) is 9 lines of CUDA under `#ifdef ATT_VOTEMAX` plus a third `_AttCuda` module (`mod_sk_vote`)
selected when `not prm.window and use_sk` (see the last section of `evidence/change_by_step.diff`). Sliding layers
keep the original max path, and so does every shape outside the stream-K slot support.

## 5. Result

| check | result |
|---|---|
| local A/B, same run vs 05 (`R/harness/bench.py`, AB/BA, 5 pair ratios per case, `evidence/public_vote_bench.log`) | per case ×1.0058 / 1.0121 / 1.0191 / 1.0154 / 1.0392 / 1.0396 / 1.0375 / 1.0383; loads turn 1 +0.78 %, turn 4 +1.79 %, first turn +3.92 %; **×1.0216**. Every one of the 40 pair ratios is > 1. |
| correctness | `evidence/public_vote_check.log`: 5 tests + 4 edge shapes × 2 replays, 18/18 PASS; the bench's 16 scored-shape checks PASS; MoE error identical to native (2.87e-3); vote bit-identical (error 0.0) |
| judge | 1.8866 → **1.9232** (check) / **1.9201** (scored), 5/5 tests both |

Judge per case, paired ratio (`evidence/judge_*.txt`):

| case | 05 (b1033df7) | 06 check (ee069284) | 06 scored (455b4f71) | judge Δ (mean of two) | local Δ |
|---|---|---|---|---|---|
| 0 | 1.907 | 1.928 | 1.897 | +0.3 % | +0.6 % |
| 1 | 1.904 | 1.928 | 1.882 | +0.1 % | +1.2 % |
| 2 | 1.870 | 1.884 | 1.915 | +1.6 % | +1.9 % |
| 3 | 1.888 | 1.907 | 1.928 | +1.6 % | +1.5 % |
| 4 | 1.848 | 1.957 | 1.929 | +5.1 % | +3.9 % |
| 5 | 1.867 | 1.937 | 1.942 | +3.9 % | +4.0 % |
| 6 | 1.916 | 1.978 | 1.958 | +2.7 % | +3.8 % |
| 7 | 1.933 | 2.005 | 2.000 | +3.6 % | +3.8 % |
| loads | 1.9060 / 1.8760 / 1.8779 | 1.9280 / 1.8917 / 1.9504 | 1.8921 / 1.9194 / 1.9491 | +0.2 / +1.6 / +3.8 % | +0.8 / +1.8 / +3.9 % |

Where the time went, judge per-kernel profiles (each judge run draws its own router, so expert work differs a few
percent between runs; read these as approximate):
- case 5 expert path: native 4061.7 (kernel) + 147.9 (`_native_finish`) + 134.7 (`pnr2y`, router with PG)
  = 4344.3 µs → public 3631.0 (kernel) + 136.8 (gather) + 128.4 (combine) + 78.6 (`pnr2y`, no PG) = 3974.8 µs
  (**−8.5 %**; the kernel alone −10.6 %);
- case 2 expert path: 816.7 + 32.6 + 31.1 = 880.4 → 723.7 + 32.2 + 25.2 + 22.8 = 803.9 µs (−8.7 %);
- 16K o_proj + bnd1: nvjet 337.2 + `bnd4_b1` 255.8 = 593.0 µs → `ub2_2_192_4_4` 555.0 µs (case 5);
- full attention `attn_fwd`: case 6 2183.8 → 2102.5 / 2132.8 µs, case 7 3890.6 → 3681.8 / 3693.1 µs.

**Judge, read honestly.** No case runs identical code between the two builds (step 1 touches every case), so there is
no clean control shape. The check is that the judge reproduces the local pattern: the first-turn load moved +3.8 %
(local +3.9 %), turn 4 +1.6 % (local +1.8 %) and turn 1 +0.2 % (local +0.8 %). The turn-1 gap is inside the run-to-run
spread: the two readings of this very file disagree by 1.7 % on turn 1 (1.9280 vs 1.8921), and the native MoE at
case 0 read 300.9 µs in one run and 268.3 µs in the other. The real effect is **≈ +2 %** (judge +1.86 %, local +2.16 %).
For release purposes the real level is the mean of the two readings of this file, **1.9217**.

**Against the projection.** The projection was ≈ +2 % and the result is +2.16 % local, +1.86 % judge. The MoE-only
gain at 16K (5.4-5.7 %) became −8.5 % on the whole expert path in situ, because the public kernel's slot layout also
removed the PG pre-gather from the router (`pnr2y` 134.7 → 78.6 µs). The turn-1 gain came only from the exact items,
since T = 753 stayed native.

## 6. Takeaways, and what came next

- **Changing the tile shape changed the cost model.** 128 rows × 256 tokens halves FP4 decode work per FLOP, and the
  publisher warp hides the epilogue. At T ≥ 2048 that beat further tuning of our 256 × 128 kernel. At T = 753 the
  tiny-expert tail decides, and there it did not help (×0.963-1.005).
- **Read a faster public design for its mechanisms, not as a drop-in.** The public attention lost by 1.5× on our
  cases, yet its vote ordering ported exactly and kept bit-identical output.
- **Audit dead ends that were never explained.** The gate/up `stmatrix` epilogue had been written off at error 1.0.
  The cause was a builder bug, and the repaired version banked +0.2-0.4 %.
- **Next: 07.** After this build, Codex's per-kernel audits (ncu stall PCs, SASS wgmma counts, item traces) found about
  twenty exact 0.03-0.6 % savings across the public kernel, attention, router and boundaries. One of them extended
  the public kernel down to T ≥ 512. Stacked, they are milestone 07.

## Reproduce

Needs torch, nvcc (the public kernel is built with `load_inline`) and an H200. `../tools/` mirrors the problem root
that the harness expects.

```bash
cd ../tools
python research/harness/check.py ../06_public_okc2026_expert_kernel/build.py --edges        # 5 tests + 8 scored + nearby edges, 2 replays each
python research/judge/replica.py --candidate full28=../05_sk_only_attention_fused_unpool/build.py \
       --candidate public_vote=../06_public_okc2026_expert_kernel/build.py --panel 11,22,33 --reps 8   # judge-style paired A/B on 3 router draws
```

The original A/B used Codex's `R/harness/bench.py --control cand_full28.py --candidate public_vote=... --reps 10`
on the fixed router seed. `replica.py` came later; it is the judge-faithful version, with several router draws and
the AB/BA position-bias fix. The MoE-only A/B is `R/moe/public_moe_adapter.py` (log `evidence/moe_public_bound.log`).

Originals: `muse-opt-codex/muse_prefill_layer/research/attention/bundle2_public_vote_full.py` (build),
`research/moe/build_public_integration.py` (integration builder), `research/attention/build_votemax.py` (vote),
`research/public_okc_2026/` (upstream clone, Apache-2.0). Notes: `research/RUN_STATE.md`, `research/moe/NOTES.md`,
`research/attention/NOTES.md`, `research/dense/NOTES.md`. Judge results: `muse-opt-codex/muse_prefill_layer/profiles/{b1033df7778e4586,ee0692843fb94ded,455b4f71a2a940b9}.txt`.
