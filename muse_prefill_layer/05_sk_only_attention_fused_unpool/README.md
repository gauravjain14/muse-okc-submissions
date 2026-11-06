# 05: a stream-K-only attention module, SM-shared QKV, and the 16K unpooler fused with its boundary

| | before (04) | after (05) |
|---|---|---|
| build | `04_moe_vn_expert_chain/build.py` = `cand_full17.py` (md5 09d36a74, 409,762 B) | `build.py` = `cand_full28.py` (md5 d624097a, 506,194 B) |
| judge | 1.8329 (scored, job 2a57ce70, 2026-10-25) | **1.8866** (check, job b1033df7, run 2026-10-29). In between, all 10-25: full19 1.8386 (check 4092c8e9, full19 + a judge-node lab appendix), full20 1.8339 (scored a74062ce), full25a 1.8455 (scored 0432ea6e) |
| real effect | | same-run local bench **+3.9 %** compounded over eight builds (turn 1 +2.4 %, turn 4 +2.0 %). Scored-to-scored on the same day (full17 → full25a): +0.7 %, with the full-layer cases +1.0 to +4.2 %. Most of the last +2.2 % (full25a → full28) is a judge-day shift; see Result |
| shapes changed | | full layers (cases 1, 3, 4, 6, 7): attention module, QKV SM sharing, partial slots; 753 / 3012 (cases 0-3): o_proj GEMM, zero-workspace cuBLASLt plans at 3012; 16K (cases 4-7): fused unpooler + boundary; all: expert kernel v29 |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: code copied or adapted from other projects is marked in place with `SPDX-SnippetBegin` / `SPDX-SnippetEnd` comments (source, copyright, license, our changes); see `NOTICE` and `LICENSES/` at the repository root. The code is unchanged. The judged file has md5 `d624097a`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

Terms used below. **attn_fwd**: our FP8 attention kernel (`attn8`, milestones 01-03); full layers split long key ranges across CTAs (**stream-K**) and combine the partial results; sliding layers use **units** (fixed token blocks), which may be split in two with a **handoff** of the first half's partial state. **attn_comb**: a small PDL-launched grid that sums a split block's partials. **kv_pack**: repacks a full layer's cached FP8 K/V into attention's layout; it runs on a side stream beside the QKV projection. **sk2**: our persistent BF16 GEMM (`dev/gemm_sk2.py`). **bnd2**: the post-FFN boundary (FP32 residual + two RMSNorms) after the unpooler GEMM. **ub2**: the fused unpooler + bnd2 kernel. **bench**: the era-1 local judge-alike (`04_moe_vn_expert_chain/evidence/bench_era1.py`), paired with SGLang in the same run.

## 1. Starting point

full17's full-layer cases were dominated by attention, and its 16K cases by the unpooler + bnd2 pair. Judge per-kernel profile of full17 (`evidence/judge_before_full17_2a57ce70.txt`) and the nsys roofline (`evidence/roofline_full17.txt`):

| case | judge "yours" µs | attn_fwd µs (share) | other targets (µs) |
|---|---|---|---|
| 1 (753 full) | 810.4 | 252.3 (30.7 %) | kv_pack 41.4 on the side stream, 21 % of its byte roofline; o_proj (cuBLAS) 20.4 |
| 3 (3012 full) | 2808.5 | 944.3 (34.2 %) | kv_pack 131.6 + 7.7, 26 % of roofline; o_proj 70.4 |
| 2 (3012 sliding) | 1839.8 | 45.1 | o_proj 68.7 (74 % eff.), pool 281.4, unpool 250.3 after a workspace memset |
| 5 (16K sliding) | 9251.7 | 218.4 (2.3 %) | o_proj + unpooler (one cuBLAS kernel, 2 calls) 1676.4, then bnd4_b2 481.3 |

## 2. What we saw

**(a) The attention tile loop paid for code it did not run.** `ATT_NOCOMB` is an ablation that compiles the "this CTA finishes its block" flag as a constant true (outputs wrong by design). It made `attn_fwd` 11-12 % faster on the 753 / 3012 full layers. A runtime-only version of the same switch recovered only a quarter of that (`evidence/attn_nocomb.log`, `evidence/attn_nocomb_rt.log`):

| case | attn_fwd µs | NOCOMB (compile-time) | NOCOMB_RT (runtime only) |
|---|---|---|---|
| 1 (753 full) | 246.5 | 218.8 (−11.2 %) | 239.6 (−2.8 %) |
| 3 (3012 full) | 943.4 | 839.6 (−11.0 %) | 911.8 (−3.3 %) |
| 6 (16K ctx32K full) | 2361.2 | 2178.8 (−7.7 %) | |
| 7 (16K ctx64K full) | 4164.6 | 3888.6 (−6.6 %) | |

So the cost was the presence of the split-unit handoff and the atomic-combine paths in the same kernel (register allocation and scheduling of the loop), not the work they do.

**(b) Partial slots instead of atomics help short full layers and hurt 16K.** `ATT_COMB 2` writes stream-K partials to per-CTA slots (plain stores, the last arriver sums): case 1 251.5 → 247.0 µs, ctx-3000 test shape 42.8 → 35.4 µs, but case 6 2380 → 2455 and case 7 4147 → 4308 µs (`evidence/attn_comb2.log`). It was therefore built as a second module used only up to 4096 tokens (`PSLOT_MAX_T`, `build.py` line 8450).

**(c) The full-layer QKV projection starved the side-stream kv_pack.** cuBLAS's persistent QKV GEMM takes every SM, so kv_pack could not co-reside and ran after it (kv_pack standalone 52.7 µs at turn-4 full, 3.1 TB/s: (private notes, not included) line 42). A cuBLASLt plan with `CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET` leaves SMs free for it (`evidence/bench_smt.log`, `evidence/bench_smt2.log`, full-layer cases):

| SMs for QKV | case 1 vs all SMs | case 3 vs all SMs |
|---|---|---|
| 92 | −0.18 % | +0.30 % |
| 100 | +0.30 % | +0.41 % |
| 108 | +0.58 % | **+0.98 %** |
| 112 (vs 108) | **+0.07 %** | −0.08 % |
| 116 / 120 (vs 108) | −0.11 / −0.16 % | −0.11 / −0.36 % |

**(d) Dense GEMMs at 753 / 3012 were already at the power-limited wgmma rate**, except o_proj. The GEMM worker's measurements (`notes/gemm2_notes.md`): in-kernel clock 1.60-1.72 GHz with 126-132 busy SMs; qkv / pool / unpool mainloops at 96-99 % of the wgmma rate; our best standalone configs reached 0.943-1.036× cuBLAS (`gemm2_notes.md` entry 12). o_proj (K = 1024, so fixed costs are a large fraction) gained in the layer: 72.4 → 69.4 µs (case 2) and 74.5 → 69.8 µs (case 3), partly because cuBLAS at 3012 also pays a workspace memset node; zero-workspace cuBLASLt plans removed that memset for pool and unpool at 3012 too (281.8 → 275.5 µs, 248.8 → 244.7 µs; "prof06" table).

**(e) The expert kernel's small tiles were HMMA-throughput bound** (the real-routing trace of 04, `04_moe_vn_expert_chain/evidence/moe_trace753_v9.log`: small items ~22 µs each regardless of size). On the real routing, MoE-only (`evidence/moe_real_v28.log`, `evidence/moe_real_wgmin.log`): v28 (two 256-k chunks per mma.sync stage) ×1.012 at 753; v28 + `MOE_WGMIN 4` (tiles of 4 row groups move from mma.sync to wgmma) ×1.015 at 753, ×1.005 at 3012.

**(f) At 16K the memory-bound glue around the GEMMs is about 11 % of every case**: bnd1 255, bnd2 473, combine 143, routing 136, qkv_post 20 µs ((private notes, not included) line 74; the same rows in `evidence/judge_mid_full25a_0432ea6e.txt`, profile 5). Hiding bnd2 under the unpooler GEMM was tried two ways:

- **Concurrent lane on the same SM** (`evidence/mlane_probe.log`: three producer warps of the GEMM stream bnd2's bytes): the GEMM slows by +10.2 % (753), +13.2 % (3012), +23.2 % (16K), and the lane moves only 0.86-1.18 TB/s. The GEMM's wgmma + TMA traffic already uses about 95 % of the 128 B/clk L1/shared-memory port ((private notes, not included) line 75).
- **Time slicing**: boundary rows become work items in the persistent GEMM's own queue, claimed as soon as their m-tile is complete (see 4).

## 3. Diagnosis and projection

- **Attention:** build one module per path. A stream-K-only module for full layers (and unsplit sliding units) should take most of the ~11 % from 2(a). With attention at 30.7 % of case 1 and 34.2 % of case 3, −10 % of attn_fwd is −25 µs and −94 µs. At 0.0146 / 0.0044 %/µs (`muse-opt-codex/muse_prefill_layer/research/harness/NOTES.md:138-141`) plus cases 6/7, that is about **+1.0 %** of score (inferred).
- **QKV SM sharing:** 108 SMs at 3012, 112 at 753, from the sweep; worth +1.0 / +0.6 % on those cases.
- **o_proj / memset:** −3 to −5 µs per case at 753 / 3012: about +0.1-0.2 % (inferred).
- **16K fused unpooler:** standalone, cuBLAS + bnd2 1885.6 µs vs ub2 1770.8 µs (`evidence/ub2_v7.log`), −115 µs per 16K case = −1.2 % of the first-turn load, **+0.4 %** of score (one load in three, inferred). At 753 / 3012 it loses (×0.865 in v6b, ×0.858 in v7): too few waves, and our GEMM alone is 3-6 % behind cuBLAS there (`evidence/ub2_v4.log`). Ship it at 16K only.
- **MoE v29:** +1.0-1.5 % of the MoE at 753 ≈ +0.4 % of case 0 (inferred).

## 4. Change

The composed builds, each benched against its predecessor in the same run (paired per-case ratios, `evidence/bench_full*.log`):

| build | change | cases that moved (same-run ratio) | geomean |
|---|---|---|---|
| full19 | expert path v29 (`MOE_WGMIN 4`, double-k mma.sync stages) | case 0 +1.03 %, case 1 +0.42 % | +0.88 % |
| full20 | `dev/patch_gemm2.py`: o_proj on sk2 at 753 / 3012 with an early PDL trigger from attention; zero-workspace cuBLASLt plans for pool / unpool at 3012 | case 2 +0.41 %, case 3 +0.54 % | +0.45 % |
| full21 | `ATT_COMB 2` partial slots up to 4096 tokens (second module) | case 1 +0.61 % | +0.37 % |
| full22 | `dev/patch_smt.py` (`QKV_SMT = {3012: 108, 753: 112}`, `build.py` line 10318) + `ATT_USHIFT 1` (the handing-over first half of a split sliding unit gets one tile fewer, so it finishes before the merging half) | case 1 +1.10 %, case 3 +1.74 % (2-variant rerun `bench_full22b.log`) | +0.48 % (rerun); +1.67 % in the 3-variant run |
| full23 | **`ATT_SK_MODULE`**: full layers and unsplit units from a build with `ATT_COMB 3` + `ATT_SK_ONLY 1` (partials always to slots, no fence or atomic; `attn_comb` sums each split block and runs the epilogue); the main module keeps only the split-unit handoff | case 1 +2.22 %, case 3 +3.12 %, case 6 +1.81 %, case 7 +2.65 % | +0.84 % |
| full24 | main (sliding) module built `ATT_UNIT_ONLY 1`; the all-paths module only compiled lazily for > 8 requests | small, +0.1-0.3 % | +0.35 % |
| full25a | two compile-time-off experiments added (`ATT_INCOMB 0`, `KVP_ILP 0`; `diff` of the two files) | none | +0.02 % (`bench_full25all.log`) |
| full28 | `dev/patch_ub2.py`: fused unpooler + bnd2 at T = 16384 (`UB_ROUTE`, `build.py` lines 10321-10330) | case 4 +1.52 %, case 5 +1.95 % | first turn ×1.953 → ×1.971 |

The stream-K-only module is bit-identical to the all-paths kernel ("sk vs mine 0.00e+00") and on attn_fwd alone (`evidence/attn_skonly.log`): case 1 248.6 → 224.5 µs (−9.7 %), case 3 951.0 → 847.6 (−10.9 %), case 6 2448.4 → 2201.4 (−10.1 %), case 7 4321.5 → 3875.0 (−10.3 %). The unit-only build is neutral on sliding units (23.7 → 23.4 µs case 0, `evidence/attn_unitonly.log`).

**How ub2 converged** (standalone, `dev/ub2_test.py` = `evidence/ub2_test.py`; µs, fused vs cuBLAS + bnd2 in the same run):

| version | design point | result | what we learned |
|---|---|---|---|
| v0 | persistent sk2-style GEMM (256×192 tiles, 3 × 64 KB stages) plus boundary items | M 1000: 182.7 vs 122.5 (×0.670); 3012: 396.9 vs 344.6 (`muse-opt/muse_prefill_layer/logs/ub2_v0.log`) | |
| v1 | traced | boundary items 10.84 µs per row at M 1000 (`logs/ub2_v1.log`) | rows far too slow |
| v2 | head-pointer queue | 1.55 µs/row, but CTA ends spread 69.5-225.2 µs (`logs/ub2_v2.log`); a compare-and-swap head pointer serializes 132 CTAs ((private notes, not included) line 75) | claim boundary rows as "units" published per finished m-tile, no CAS loops |
| v4 | GEMM-only variants | plain-epilogue G4: 1456.0 vs cuBLAS 1482.0 at 16K (×1.018); 266-273 vs 257 at 3012 (`evidence/ub2_v4.log`) | parity at 16K only; a fused f'/sum epilogue cost 121 µs at 16K, so the epilogue stays plain BF16 |
| v3-v6b | tile groups, a cap on CTAs streaming boundary rows, a boundary row staged in one 64 KB stage, split-phase mbarrier reductions (design notes: (private notes, not included) line 75) | 753: ×0.805-0.865 (`logs/ub2_v5.log`, `ub2_v6.log`, `ub2_v6b.log`) | the row loop must be `#pragma unroll 1` (unrolled ×16 thrashed the i-cache, 7.8 µs/row; same note) |
| v7 | final | **16K: 1770.8 vs 1885.6 (×1.065); GEMM-only 1399.4 vs cuBLAS 1397.6**; 3012 ×0.858 (`evidence/ub2_v7.log`) | shipped at 16K |

## 5. Result

**Local, same-run bench, compounded full17 → full28** (computed from `evidence/bench_full*.log`; the ub2 step from Codex's 10-29 re-measure `evidence/codex_initial_bench_full28_vs_full25a.log`):

| load | gain |
|---|---|
| turn 1 (cases 0, 1) | **+2.4 %** |
| turn 4 (cases 2, 3) | **+2.0 %** |
| first turn (cases 4-7) | +7.0 %, partly noise |
| geomean | +3.9 % |

The first-turn figure is partly noise. Two steps that do not touch 16K at all "gained" there: full20 +0.2 to +1.7 % and full21 +0.1 to +1.6 % on cases 4-7 (`bench_full20.log`, `bench_full21.log`). The 3-variant full22 run read +2.5 to +3.5 % at 16K and its 2-variant rerun −0.4 to +1.5 %. The bench's position bias at the power-capped 16K shapes was found the same day ((private notes, not included) line 52).

**Codex's independent re-measure of the last step** (10-29, `evidence/codex_initial_bench_full28_vs_full25a.log`, 6 paired reps, full25a as control with a self-copy A/A): cases 0-3 within ±0.26 % (A/A ±0.25 %), cases 4-7 +1.10 / +1.25 / +0.93 / +1.89 %; first-turn load −1.26 %; projected score **+0.47 %**, i.e. 1.854 if full25a's judge 1.8455 is taken as the anchor (`muse-opt-codex/muse_prefill_layer/research/prefill_initial.json`, `judge_projection`).

**Correctness:** every bench run passes the reference check on all 8 cases; the attention modules are bit-identical to the all-paths kernel; ub2's output vs the layer path differs by 2.4e-6 / 1.1e-5 at 16K (`evidence/ub2_v7.log`); the full28 judge check passed 5/5 tests.

**Judge, read honestly.** Ratios per case (`evidence/judge_before_full17_2a57ce70.txt`, `evidence/judge_mid_full25a_0432ea6e.txt`, `evidence/judge_after_full28_b1033df7.txt`):

| case | full17 (10-25) | full25a (10-25) | full28 (10-29) | code changed full25a → full28? |
|---|---|---|---|---|
| 0 sliding 753 | 1.859 | 1.864 | 1.907 | no |
| 1 full 753 | 1.826 | **1.869** | 1.904 | no |
| 2 sliding 3012 | 1.827 | 1.809 | 1.870 | no |
| 3 full 3012 | 1.773 | **1.848** | 1.888 | no |
| 4 16K ctx0 full | 1.823 | 1.856 | 1.848 | yes (ub2) |
| 5 16K sliding | 1.835 | 1.831 | 1.867 | yes |
| 6 16K ctx32K full | 1.864 | 1.883 | 1.916 | yes |
| 7 16K ctx64K full | 1.886 | **1.928** | 1.933 | yes |

- **full17 → full25a (same day, both scored): +0.69 %.** The full-layer cases, where the attention and QKV changes act, moved +2.4 % (case 1), +4.2 % (case 3), +1.8 / +1.0 / +2.2 % (cases 4, 6, 7). The sliding cases, which got only the small MoE and o_proj items, moved −1.0 to +0.3 %, inside noise. The judge's own attn_fwd row fell 252.3 → 218.9 µs (case 1) and 944.3 → 816.5 µs (case 3). The new `attn_comb` row (26.8 / 70.7 µs) is mostly PDL wait overlap, not work.
- **full25a → full28 (+2.2 %) is mostly the judge day.** Cases 0-3 run identical code and moved +1.9 to +3.4 %. The 16K cases, where ub2 acts, moved −0.4 to +2.0 %, i.e. no clearer than the unchanged ones. In the judge's per-kernel profile the fused kernel replaces the unpooler + bnd2 pair on case 5: about 1382.8 + 473.0 = 1855.8 µs → 1692.3 µs (−163.5 µs; the unpooler's share of full25a's two-call cuBLAS row is inferred by subtracting full28's single o_proj call, 337.2 µs). That matches the local −171 µs (`bench_full28.log`, case 5).
- **Real level** by the release rule is the single reading, 1.8866. Our best local estimate of the build is about 1.854, the 10-25 judge level × the measured +0.47 % (inferred).
- **vs projection:** attention came out larger than projected: the judge's attn_fwd rows −13 %, against the −10 % projected from the standalone numbers. ub2 matched (−115 µs projected; −105 / −115 µs per 16K case in Codex's re-measure, −163.5 µs in the judge's kernel rows). The o_proj and memset items are below what one judge run resolves.

## 6. Takeaways, and what came next

- **Code that a kernel never executes can still cost 10 %.** When a loop serves several paths, build one module per path and dispatch on the host. The same lesson as v9's i-cache step in 04, at a larger scale.
- **Concurrency inside an SM with a wgmma GEMM does not work here; time slicing does.** The GEMM uses about 95 % of the shared-memory port, so a co-resident lane slows it by 10-23 %. The same work as queue items inside the GEMM wins at 16K and loses at 753 / 3012, where there are too few waves to hide it.
- **SM-count targets let a side stream share a GEMM's SMs**: the cheapest way to overlap kv_pack with the projection.
- **This bench could not resolve 1 % at 16K**, and judge days drift by about 2 %. The next stage replaced both: a judge-faithful replica paired with SGLang over several router draws (milestone 08 onwards), and a judge-calibration check (`evidence/judge_node_lab_plab1_4092c8e9.log`: judge GPU = Modal H200, idle 1.98 GHz, 16K GEMM 744 TF/s at 1.649 GHz, HBM copy 4.26 TB/s).
- **What came next (06):** on 10-29 Codex took over from exactly this build. It checked it (this milestone's judge reading) and integrated the public okc-2026 expert kernel for T ≥ 2048, plus an exact early-rescale vote in full attention. The last era-1 build, full29 (stmatrix staging of the expert epilogue, about +0.15 % bench), was never judged; its idea returned in Codex's bundle as gate stmatrix staging.

## Reproduce

All you need is torch and an H200. `tools/` holds the problem spec, `reference.py`, `checks.py`, `baseline.py` and the judge-faithful harness.

```bash
cd muse_prefill_layer/tools
python research/harness/check.py ../05_sk_only_attention_fused_unpool/build.py --edges      # judge tests + 8 scored cases + nearby shapes
python research/judge/replica.py --candidate before=../04_moe_vn_expert_chain/build.py \
       --candidate after=../05_sk_only_attention_fused_unpool/build.py --panel 11,22,33 --reps 8
python ../05_sk_only_attention_fused_unpool/evidence/ub2_test.py   # standalone ub2 vs cuBLAS + bnd2 (needs the era-1 dev/ layout: dev/gemm_ub2.py)
```

Originals: `muse-opt/muse_prefill_layer/cand_full28.py` (also `muse-opt-codex/muse_prefill_layer/cand_full28.py`, same md5), intermediates `cand_full{19,20,21,22,23,24,25a}.py`, patches `muse-opt/muse_prefill_layer/dev/{patch_gemm2,patch_smt,patch_ub2,gemm_sk2,gemm_ub2}.py`, attention source `dev/attn8.py`, logs `muse-opt/muse_prefill_layer/logs/`. Judge results: `muse-opt/muse_prefill_layer/{checks,submits}/profiles/`, and for full28 `muse-opt-codex/muse_prefill_layer/profiles/b1033df7778e4586.txt` (Codex's run state: `muse-opt-codex/muse_prefill_layer/research/RUN_STATE.md`).
