# 07: twenty exact micro-gains, stacked (Codex bundles 3-23)

| | before (06) | after (07) |
|---|---|---|
| build | `06_public_okc2026_expert_kernel/build.py` (md5 21732359, 788,295 B) | `build.py` = Codex's `research/integrated/bundle23_gate_queue.py` = `BEST_SUBMITTED.py` (md5 bea50449, 964,737 B, sha256 bd39eec3…) |
| judge | 1.9232 (ee0692843fb94ded) / 1.9201 (455b4f71a2a940b9); real 1.9217 | **1.9955** (scored fa9e150bebe74d1f), single reading. Intermediate bundles: 1.9203 … 1.9748, see Result |
| real effect | | local chain of paired A/Bs **+1.97 %** (one router draw); judge-based estimate ≈ +2.5 %. The 1.9955 reading is a favourable draw, about +1.3-1.8 % high |
| shapes changed | | all 8 cases (attention, router, public and native expert kernels, gather/combine, boundaries) |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: code copied or adapted from other projects is marked in place with `SPDX-SnippetBegin` / `SPDX-SnippetEnd` comments (source, copyright, license, our changes); see `NOTICE` and `LICENSES/` at the repository root. The code is unchanged. The judged file has md5 `bea50449`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

Third-party code: contains `okc_moe::moe_fused_persistent` (vjeux/okc-2026, Apache-2.0); the judged file dropped the inline
notice from this build onward; here every borrowed block is marked in place (see the root `NOTICE`).

Source shorthand: paths are relative to ``; `R/` = `muse-opt-codex/muse_prefill_layer/research/`.

Terms: **bundle N** = Codex's integrated stack N (`R/integrated/bundleN_*.py`), each one validated before the next
was stacked on it. **Exact** = output bit-identical to its control, or a documented sub-ulp FP32 change. **PG** =
the pre-gathered slot path used at large T. **mod_sk / unit 1-2** = attention modules for the stream-K full path and
the split/unsplit sliding units. **Replica** = `tools/research/judge/replica.py`, the judge-faithful local scorer
built right after this milestone (see 2e). The other terms are defined in 06's README.

## 1. Starting point

06 in its two judge runs (ours / SGLang µs, paired ratio; `evidence/judge_before_public_vote_*.txt`):

| case | check ee069284 | scored 455b4f71 | biggest kernels in the scored run's profile |
|---|---|---|---|
| 0 | 563.8 / 1086.9 (1.928) | 536.7 / 1018.0 (1.897) | native MoE 268.3 µs (47.9 %), pooler+router 80.0, unpooler 67.0, QKV 39.8 |
| 1 | 772.7 / 1489.5 (1.928) | 737.7 / 1388.0 (1.882) | native MoE 271.2, `attn_fwd` 212.9 |
| 2 | 1783.8 / 3360.9 (1.884) | 1756.4 / 3363.0 (1.915) | public MoE 723.7 (40.4 %), gather 32.2, combine 25.2, `pnr5` 22.8 |
| 3 | 2587.0 / 4932.5 (1.907) | 2550.9 / 4919.1 (1.928) | `attn_fwd` 804.8, public MoE 735.3 |
| 4 | 9033.3 / 17676.5 (1.957) | 8909.0 / 17189.4 (1.929) | public MoE 3455.7, bk4 563.7, `attn_fwd` 502.6 |
| 5 | 8808.3 / 17063.7 (1.937) | 8692.8 / 16884.2 (1.942) | public MoE 3631.0 (41.6 %), unpool+bnd2 1687.1, gather 136.8, combine 128.4 |
| 6 | 10629.7 / 21022.0 (1.978) | 10522.9 / 20608.0 (1.958) | public MoE 3632.0, `attn_fwd` 2102.5 |
| 7 | 12182.8 / 24429.7 (2.005) | 12158.0 / 24315.2 (2.000) | public MoE 3965.1, `attn_fwd` 3681.8 |

## 2. What we saw

**(a) Where a microsecond is worth the most.** From the scored profile, Codex computed the log-score sensitivity per µs
saved (`notes/codex_harness_NOTES.md:136-143`): case 0 0.0420 %, case 1 0.0146 %, case 2 0.0126 %, case 3 0.0044 %,
case 5 0.0027 %, cases 4/6/7 0.0003 %. A 1 % score gain needs 17.5 µs off both T = 753 cases, or 58.3 µs off both
T = 3012 cases, or 272 µs off all four 16K cases.

**(b) The bounds left no single large lever inside the existing kernels** (`notes/codex_harness_NOTES.md:144-160`,
`R/moe/NOTES.md` "Integrated public success"):
- **Wide expert items** run at the tensor floor. Per-item timers in the public kernel at 16K
  (`06_*/evidence/public_moe_trace.log`): a 256-wide gate item has a median of 39.36 µs against a 39.03 µs tensor floor
  at 1.679 GHz. Narrower items sit further above it: width 128 +15 %, 64 +65 %, 32 +129 %. Across all CTAs the
  consumer span is 3662.8 µs: 3521.6 µs in items, 118.3 µs of tails, 18.7 µs of gaps.
- **Tiny experts are bound by data movement.** In-graph ncu (`evidence/narrow_memory_summary.json`): 12 active tiny
  experts need 339.7 MB of weights. The kernel moved 347.6 MB in 103.2 µs (3.37 TB/s) at 18.5 % tensor-active.
  The L2 slices are evenly loaded, so partition camping is ruled out.
- **The MoE in situ** (ncu, `evidence/bundle14_moe_roofline_summary.json`): case 0 275.4 µs at 63.0 % tensor,
  39.5 % DRAM, 37.5 % issue, 1.68 GHz; case 2 81.6 % tensor; case 5 90.5 % tensor (`notes/codex_RUN_STATE.md`, "Current roofline").
- **Glue and dense GEMMs.** Deleting every public gather would be worth 1.05 % and every combine 0.90 %. Bandwidth
  leaves about 0.3-0.4 % of that reachable. The dense mainloops already ran at 96-99 % of the clock-limited peak. The
  T = 3012 boundaries need 113 µs at 4.8 TB/s against ~142 µs measured. Whole-layer roofline
  (`evidence/bundle14_roofline.txt`): ideal 128.2-140.9 ms against 144.6-165.0 ms for the current stage structure.

**(c) What the instruments did find was many small, exact costs.**
- **SASS wgmma counts exposed a compiler cliff.** Adding tile widths to the public kernel (tiny N8/N16 widths, then
  seven fine widths) made ptxas serialize every wgmma in the kernel (C7512). In SASS the HGMMA/ARRIVE/DEPBAR counts
  became equal: 68/68/68 (tiny) and 52/52/52 (seven widths) against a healthy 36/12/16. Grouping all ten widths'
  accumulators globally ("G2") restored a healthy cadence (88/56/66) (`notes/codex_RUN_STATE.md`, "Small-MoE
  compiler/occupancy round" and "Latest fine-width / occupancy follow-up").
- **Instruction counts in the attention softmax.** The EXPQ bias (milestone 03) was recomputed on every KV tile.
  Caching it, and refreshing only inside the rescale branch, removes 8 instructions per tile whose max did not move:
  case 1 239.95 → 235.86, case 3 885.44 → 865.79, case 6 2210.4 → 2155.0, case 7 3941.6 → 3809.4 µs, bit-identical
  (`evidence/expq_cache_attention.log`).
- **Memory transactions in the gather.** In the non-PG gather each lane made two vector loads with alternating
  half-sector addresses. Excess sectors at T = 3012: 2,269,617 → 1,713 after the fix (`evidence/round15_status.json`).
- **Dependent loads (long-scoreboard stall PCs)** in PG metadata (8 copy CTAs per expert) and in combine
  (`evidence/round15_status.json`; raw PCs in `R/bundle14_stall_pcs.json`).
- **Queue order, from per-item traces** (`evidence/queue_item_trace_summary.json`). Fully interleaving tiny jobs into
  the queue raised CTA idle from 7.5 to 27.2 µs and the span from 262.6 to 288.3 µs (seed 0). It also delayed the
  down items' dependencies (down max 77 vs 40 µs, `notes/codex_RUN_STATE.md`, "Stack23"). Mixing only half of the
  small gate jobs, in chunks of 8, kept the largest-first drain (`evidence/gate_queue_component.log`).

**(d) The measurement problem.** It did not change the code, but it decided what got judged.
- Codex's local number (`local_score_us`, e.g. 197,708 µs for this build in `evidence/gate_queue_local_score.json`)
  was our absolute weighted latency. The judge's `score_us` is 387,403.9 / speedup, a constant divided by the
  paired-ratio score. In the 1.9955 run our own absolute geomean was ~199.5 ms and SGLang's ~398 ms
  (`notes/prefill-judge-metric.md`), so the two quantities are not comparable.
- After b17's check the user's judge gate tightened to "no judge run until local < 194 ms" (`notes/codex_RUN_STATE.md`,
  "User tightened the judge gate"; `R/progress.json` `user_strategy`). Measured in that quantity it could not open:
  b18-b23 never reached the judge, though their relative gains were real, until the user overrode the gate for one
  scored run of b23.
- Codex's bench always used one router seed (728034927264498030). The judge draws a new router every run
  (`reference.weights_fields`). On judge-style draws the first-turn load of b23 read 2.0524 (fixed seed), 1.9456
  (draw 11) and 2.0179 (draw 22), and case 5 read 2.054 / 1.935 / 2.013 (`evidence/replica_calib1.log`,
  `evidence/replica_router_seed_test.log`). On draw 11, b23 ranked below b12 and b14 on that load. Gains selected on
  one routing partly do not transfer.

## 3. Diagnosis and projection

- The layer was a chain of kernels each within a few percent of its own bound (b). What was left was fixed costs
  inside kernels: recomputed constants, extra instructions on common paths, uncoalesced sectors, an extra launch,
  dead instrumentation, tail rows of padding. Each worth 0.1-30 µs, each exact.
- Policy (Codex, `R/gains.json` "strategy"): keep every repeatable, exact gain; validate each stack with all 16
  scored-shape checks plus 18 hard/edge checks before stacking the next one.
- Projection: the sum of the measured full-layer steps (table below), about +2 %. No single-step projection was
  made in the judge's quantity: Codex's projections were in its absolute metric, e.g. "198,820.5 µs, NOT official"
  for b12 (`notes/codex_RUN_STATE.md`).

## 4. Change

The ladder, as recorded in `evidence/gains.json` and `notes/codex_RUN_STATE.md`. "Local" is the full-layer paired A/B
on Codex's harness, all 8 cases, fixed router seed. "Component" is a kernel-level A/B on fresh inputs.

| bundle | what it added | local (vs previous) | judge |
|---|---|---|---|
| b3 `native_epi` | native expert epilogue cleanup; direct-small `stmatrix` epilogue | ×1.00108 | |
| b4 `public_ten` | public kernel with ten tile widths and globally grouped accumulators (no C7512) | ×1.00181 | |
| b5 `ten_pg` | compact PG tails, large T only (the same tails cost 0.72 % on case 2, `notes/codex_RUN_STATE.md`) | ×1.00132; ×1.00316 vs 06 (`evidence/combined_vs_submitted.log`) | 1.9203 check (7493a449) |
| b6-b8 `expq_stack` → `compact` | empty-prefix packed attention mask; public combine without unit scales; identity BF16 gather copy; boundary reuses its RMS factor; **cached EXPQ bias** in full attention. The raw stack (1,026,168 B) was over the 1,000,000 B upload cap; compaction removed one unused source literal | ×1.00302; **×1.00613 vs 06** (`evidence/bundle89_vs_submitted.log`) | 1.9375 check (23df26ab) |
| b9-b10 `clear` | EXPQ bias cache also in unsplit sliding unit 1; clear only the live native accumulator columns | ×1.00092 | |
| b11 `threshold_keep` | full-attention rescale thresholds cached as scalars (−1.6 / −6.5 / −4.8 / −14.0 / −30.6 µs on cases 1/3/4/6/7); boundary coefficients kept across persistent rows | ×1.00094 | 1.9423 check (02dd9d6c) |
| b12 `small_router` | scalar thresholds in unsplit sliding; router FP32 sigmoid via `__expf`/`__fdividef` (same top-k, max weight Δ 5.96e-8); zero-accumulator first-tile cleanup; **public "fine" kernel from T ≥ 512** (turn 1 ×1.0016-1.0026); PG-only sentinel routing (−5.6 µs at 16K) | ×1.00319 (`evidence/bundle12_full_bench.log`) | 1.9561 check / 1.9500 scored |
| b13 `cleanup_stack` | down-epilogue address rematerialization (non-PG); dead per-item timers removed; PG router candidates sorted once; work-counter reset fused into combine (one launch fewer) | ×1.00212 | 1.9466 check |
| b14 `fast_normalize` | FP32 fast reciprocal in the route-weight normalization | ×1.00026 | 1.9748 check / 1.9475 scored |
| b15 `n208_nonpg` | 208-row tail tiles (non-PG) instead of rounding to 224 | ×1.00076 | |
| b16 `coalesced_gather` | contiguous per-warp vector addresses in the non-PG gather | ×1.00318 | |
| b17 `parallel_metadata` | 32 copy CTAs per expert for PG metadata, redundant prefix scans removed (−17.7 / −18.4 µs component) | ×1.00099 | 1.9336 check |
| b18 `medium_combine_cache` | combine keeps routing metadata in registers while streaming values (T = 3012) | ×1.00053; b18 vs b14 ×1.00404 | |
| b19 `small_matrix_stores` | matrix stores in the small-expert epilogue (component −0.24 / −1.01 / −0.30 µs) | ×0.99991 (noise) | |
| b20 `small_warp_mma` | warp MMA for 8-row tiles at 512 ≤ T < 2048 | ×1.00076 | |
| b21 `decode_reuse` | reuse one rotated FP4 word in the small-shape decode | ×1.00101 | |
| b22 `medium_decode_reuse` | the same in medium non-PG kernels | ×0.99990 | |
| b23 `gate_queue` | first half of the small gate jobs mixed into the queue in chunks of 8; largest-first gate drain and down order kept | ×1.00032 (small load ×1.00119; component −1.90 / −2.98 / −0.75 µs on 3 seeds, 116/120 pairs positive) | **1.9955 scored** |

Chained over the measured full-layer steps (b8 vs 06, then b10, b11, b12, b13, b14, b18 vs b14, b19-b23):
**×1.0197** (computed from the values above).

## 5. Result

| check | result |
|---|---|
| local, chained paired A/Bs vs 06 (Codex harness, fixed router seed) | ×1.0197 |
| correctness | every bundle: 16 scored-shape checks + 18 hard/edge checks PASS (`evidence/gains.json`, `notes/codex_RUN_STATE.md`); judge 5/5 tests on every run |
| judge | 1.9217 (06, two runs) → **1.9955** (b23, one scored run) |
| replica, after the fact | b23 on 3 judge-style router draws: 1.9592 (draws 1.9629 / 1.9571 / 1.9575) with the position bias fixed (`muse-opt-codex/muse_prefill_layer/research/claude/NOTES.md:68-69`); 1.9778 with the earlier biased ordering (`evidence/replica_panel_base.log`) |

Judge per case (paired ratio):

| case | 06 (mean of 2) | b23 | ours µs, b23 vs the 06 runs | SGLang µs, b23 vs the other 12 runs of 10-29 |
|---|---|---|---|---|
| 0 | 1.913 | **2.076** | 563.6 vs 536.7 / 563.8 | **1170.3** vs 1018.0-1099.2 |
| 1 | 1.905 | 1.984 | 752.6 vs 737.7 / 772.7 | 1493.0 vs 1388.0-1491.1 |
| 2 | 1.900 | 1.965 | 1796.4 vs 1756.4 / 1783.8 | **3530.7** vs 3296.3-3442.7 |
| 3 | 1.918 | 1.954 | 2579.6 vs 2550.9 / 2587.0 | 5041.4 vs 4797.1-5001.9 |
| 4 | 1.943 | 2.023 | 9459.0 vs 8909.0 / 9033.3 | **19135.4** vs 16374.3-18943.7 |
| 5 | 1.940 | 1.958 | 8491.3 vs 8692.8 / 8808.3 | 16624.1 vs 16614.0-17319.4 |
| 6 | 1.968 | 2.007 | 10689.3 vs 10522.9 / 10629.7 | 21458.4 vs 20440.4-21762.5 |
| 7 | 2.003 | 2.046 | 12087.5 vs 12158.0 / 12182.8 | 24729.5 vs 24023.9-25309.8 |
| loads | 1.910 / 1.906 / 1.950 | 2.047 / 1.961 / 1.979 | | |

(The "other 12 runs" are 05's check, 06's two runs and the nine bundle runs b5-b17, `evidence/judge_*.txt`.)

**Judge, read honestly.**
- In b23's run our own medians were not faster than in 06's runs on cases 0, 2 and 4. SGLang, in the same run, was
  slower on those cases than in any of the other 12 judge runs that day. Case 0, the most valuable case, went from
  1.913 to 2.076 almost entirely on SGLang's side (1170 µs against 1018-1099).
- The same-code pairs from the same day show the spread: b12 read 1.9561 and 1.9500, b14 1.9748 and 1.9475.
- Two independent estimates of b23's expected judge level both land near 1.97:
  - judge means × the local chain: b12 (1.9531) × 1.0083 (b13-b23) ≈ 1.969, and b14 (1.9612) × 1.0060 (b18-b23) ≈ 1.973;
  - the replica: 1.959, which ran ~0.8 % below the judge on the next build (`muse-opt-codex/muse_prefill_layer/research/claude/NOTES.md:74-76`) → ≈ 1.975.
- So the 1.9955 reading is about +1.3-1.8 % high (inferred). The real effect of this milestone is **≈ +2 to +2.5 %**:
  local +1.97 %, judge-based 1.97 / 1.9217 = +2.5 %.
- For the release rule the real level is the single reading, 1.9955.

**Against the projection.** The projection (+2 %, the sum of the steps) matched the local chain by construction. The
judge-based estimate (+2.5 %) agrees within its ±1 % noise. The 1.9955 reading over-states it.

## 6. Takeaways, and what came next

- **Stacking exact 0.03-0.6 % items is a real lever when every kernel is near its own bound:** twenty of them made
  about +2 %. What found them was instruction-level evidence: SASS wgmma counts, stall PCs, excess sectors, per-item
  traces. Kernel-wide percentages alone would not have.
- **Measure the judge's quantity.**
  - An absolute-latency gate (< 194 ms local) cannot be compared with `score_us`. It kept every build from b15 to b23
    off the judge.
  - A single router seed rewarded changes tuned to one routing.
  - One judge reading is ±1-1.5 %. The 1.9955 here is the clearest case of a lucky draw in this ladder.
- **Next: 08.** The next step began by building `research/judge/replica.py`: SGLang-paired, AB/BA per candidate, a
  panel of judge-style router draws (calibrated on this build: fixed seed 1.9963 locally vs 1.9955 judged,
  `evidence/replica_calib1.log`). All later milestones are measured with it. 08 itself moved to the dense GEMMs and
  the 16K boundary tail: exhaustive cuBLASLt configurations without workspace, and finer boundary items.

## Reproduce

```bash
cd ../tools
python research/harness/check.py ../07_exact_micro_gain_bundles/build.py --edges           # tests + scored + nearby, 2 replays each
python research/judge/replica.py --candidate public_vote=../06_public_okc2026_expert_kernel/build.py \
       --candidate b23=../07_exact_micro_gain_bundles/build.py --panel 11,22,33 --reps 8          # judge-style paired A/B, 3 router draws
python research/judge/replica.py --candidate b23=../07_exact_micro_gain_bundles/build.py --panel fixed,11,22 --reps 8   # 'fixed' = Codex's router seed vs two judge-style draws
```

Originals: `muse-opt-codex/muse_prefill_layer/research/integrated/bundle{3..23}_*.py` (the stacks; this build is
`bundle23_gate_queue.py` = `BEST_SUBMITTED.py`), `research/{moe,attention,router,packing}/build_*.py` (builders),
`research/gains.json` (ledger), `research/RUN_STATE.md`, `research/CURRENT.md`, `research/harness/NOTES.md`.
Judge results: `muse-opt-codex/muse_prefill_layer/profiles/{7493a4491e4d4707,23df26ab67904f5d,02dd9d6ca1c748a9,8e6838c863164d3b,cc229f30eb514d6a,f69dd0b96a124ca1,f4dcd2496f2c439f,46b2d9100c8049d0,b12db764d7fa4a40,fa9e150bebe74d1f}.txt`.
Replica logs: `research/judge/{calib1,panel_base,router_seed_test}.log`.
