# muse_moe_w4a16

The routed mixture-of-experts layer of one GPU under 8-way expert parallelism, with 4-bit NVFP4 weights and BF16 activations, on H200, judged against SGLang's Marlin W4A16 MoE path.

Source paths in the milestone READMEs are relative to the authors' workspace. `M/` = `muse-opt/muse_moe_w4a16/` (main workspace), `R/` = `okc-retro/`, `mem:` = dated notes kept outside the tree. Files copied into this repository are cited relative to their milestone (`evidence/...`, `notes/...`).

## The problem and the score

Per call (`tools/TASK.md`, `tools/reference.py`):
- M tokens, each routed to its top-8 of 256 experts; this GPU holds 32 of them (EP8). H = I = 4096; SwiGLU.
- Weights are NVFP4: E2M1 codes, one E4M3 scale per 16 values, an FP32 global scale per expert. One expert is 28.31 MB (25.17 MB codes + 3.15 MB scales; `M/ROOFLINE.md`).
- `out[t] += topk_w · RMSNorm(y · (1 + post_norm_w))` for each (token, local expert). Routing is skewed toward this GPU and changes on every repetition.
- Correctness: rel-L2 ≤ 1.5e-2 against an FP32 reference, per row ≤ 5x that; tests at bs 4, 33, 257, 128 (skewed) and 1.

Scoring: six loads, one shape each, equal weight: decode bs1 / bs16 / bs32 and verify bs8 / bs128 / bs256.
- Each shape's speedup is the median, over repetitions, of the paired ratio SGLang time / our time.
- Each repetition is a CUDA-graph replay after the judge reads a buffer larger than L2 (cold L2), with the ~4 µs graph overhead subtracted on both sides.
- The score is the geomean of the six speedups; `score_us` = 13448.5 / speedup.
- One µs is worth 1/(6t) of the score: ~0.77 % at bs1 (t ≈ 21.6 µs) vs ~0.086 % at bs256 (t ≈ 195 µs).

**Precision rule.** BF16 wherever SGLang keeps BF16 (activations, the SwiGLU intermediate, dequantized MMA operands), FP32 accumulation, weights only losslessly repacked. From 10-27 00:15 PT the team also banned FP16 operands and accumulators; the FP16 builds of 10-25..10-27 are dead ends (`dead_ends.md`).

**Reading the judge.** A single judge run carries ±1-3 % noise per shape and ~±1 % on the score: identical kernels moved 4-5 % between runs (milestones 10, 11), and SGLang's own per-shape medians move as much. So every milestone gives the **real level**: the mean over judge runs of the same code, or the single reading when there is only one. The local same-job A/B against controls is the measure of each change.

## How the kernel works (at the last milestone here; the milestones build it up)

- **`prepare()`** (untimed) repacks the E2M1 codes losslessly into MMA-fragment order and re-encodes the E4M3 block scales losslessly (6-bit, or 5-bit with a per-row base at bs ≤ 8) when the weights allow. It also builds slice-major copies for the one-token kernel.
- **One kernel launch per call**, chosen by M alone.
- **M = 1: `moe1t`**, a two-phase one-token kernel (milestone 06). In phase A each CTA computes a fixed share of every touched expert's gate/up and publishes epoch-tagged BF16 intermediates. In phase B each CTA owns 32 output rows of every expert. One fence-free arrival word per expert ends the call (milestone 09).
- **M ≥ 2: one persistent kernel**, 132 CTAs x 384 threads, named `moe_<RW>_<NT>_<KS2>_<NS>_<MW>`: row-warps, 8-token groups per tile, down k-split, ring stages, capacity class by M.
  - Warps 0-7 are **consumers**: they decode E2M1 → BF16 in registers (9 integer ops + 4 HMUL2 per 32-bit word, a proven floor) and run `mma.sync` or `wgmma`. Warp 8 is the **producer**: it streams each **stage** (32 KB of codes + 3-4 KB of scales + the activation tile) into a shared-memory **ring** with `cp.async.bulk`. Warp 9 signals completions; warps 10-11 run the asynchronous epilogue.
  - Every CTA computes the routing redundantly. A dynamic queue then hands out **items** (a token tile x a block of weight rows over all of K, gate/up or down). Tail down items are k-split into fence-free `red.add` partials.
  - Per token, a combine unit applies RMSNorm x (1 + w) x topk weight from per-64-row sums of squares (SSQ) and sums.

Jargon used across the milestones: **jm seeds** = 100 judge-matched routings per shape (`tools/jm_seeds.jseeds`), on which local per-shape speedups match the judge within ~1 %. **bench2** = the local judge-alike (`tools/bench2.py`): graph replays after a 512 MiB flush, several builds in one job on shared weights, fillers at both ends. **project2** = a judge-scale projection that scales an anchor build's judged per-shape ratios by bench2's same-routing ratios (the script is not shipped: it embeds a leaderboard snapshot). **Loop-exact** = a change after which every hot stage loop's SASS is byte-identical (`tools/wpcgs_search.py`). Worker scopes (WP-*, B1, SU, EC, IL, ...) are defined in the milestone READMEs that use them.

## Milestones

| # | milestone | judge readings (job) | real level | what drove it |
|---|---|---|---|---|
| 01 | [persistent fused kernel with a TMA ring](01_persistent_fused_kernel_tma_ring/) (v40) | 2.1339 (48fc3fa7), 2.1331 | 2.1335 | one launch per call, lossless MMA-order repack, register decode, bulk-copy ring |
| 02 | [per-item loops, SSQ fast combine](02_per_item_loops_ssq_fast_combine/) (v66) | 2.2041 (348e75ba) | 2.2041 | the compiler's loop inversion duplicated the unrolled stage body |
| 03 | [fused one-token kernel](03_fused_one_token_kernel/) (v83) | 2.2639, 2.2823 (e83db22b) | 2.2731 | bs1 carries ~9x the score per µs of bs256 |
| 04 | [shared-weight A/B and the header layout](04_shared_weight_ab_header_layout/) (v119) | 2.3119 (f3c48e94) | 2.3119 | "gains" v84-v103 were placement artifacts; ring mbarriers must sit at smem offset 0 |
| 05 | [wgmma for heavy tiles, async epilogue](05_wgmma_heavy_tiles_async_epilogue/) (v169) | 2.3611, 2.3640 (72f0763c) | 2.3625 | stage-cost table by group count; code presence cancelled wgmma until dead paths went |
| 06 | [BF16 rebase, 6-bit scales, two-phase bs1 kernel](06_bf16_rebase_two_phase_bs1_kernel/) (v567) | 2.4346 (69f9e304) | 2.4346 | BF16 rule; bs1 at 1.4-1.54x its streaming floor because of cross-CTA waits |
| 07 | [fence-free split partials, lean producer](07_kred_partials_lean_producer/) (v814) | 2.5359, 2.5197 (b9a6bf4a) | 2.5278 | first ncu report; SASS showed per-lane copy waterfalls on the producer |
| 08 | [fast start, early held-slot release](08_fast_start_early_slot_release/) (v969) | 2.5603, 2.5551 (a142c92b) | 2.5577 | DRAM idle for the first 4.7 µs at bs32; the held slot is the ring turnaround |
| 09 | [fence-free bs1 end, late claims, buckets](09_late_claim_buckets_moe1t_end/) (v1504) | 2.6077 (1084b630) | 2.6077 | time ≈ first copy + bytes/ring + end + tail; warp-aggregated claims |
| 10 | [queue order, item transitions, 5-bit bs8 scales](10_queue_order_item_transitions/) (v4111) | 2.5793 (195eb93e) | 2.5793 | four-phase nsys timeline at bs128/256; +0.85 % local that the judge never showed |
| 11 | [cold start, end chain, item loop](11_cold_start_end_chain_item_loop/) (i4) | 2.6094, 2.6169, 2.6315 (9f97658f) | 2.6315 | stamped edge budgets; divergent spin poll; per-path transitions |

FP16 builds v174-v600 (judge up to 2.4470) are not milestones: they were voided by the 10-27 rule (see `dead_ends.md`).

**Known latent bug in milestones 09-11** (described in their READMEs). The item-transition path applies a 6-bit-scale compensation without checking that `prepare()` chose 6-bit scales. On valid weights that need the 8-bit fallback for some rows, the bs32-256 paths scale outputs wrongly. The task's weight generator never produces such weights, so every judge test passed. The fix is to add `S6 && MOE_S6OV` to `#define ITE` and `#define ILK`.

## Where the time goes at milestone 11 (i4)

Judge numbers are i4's run (`11_cold_start_end_chain_item_loop/evidence/judge_after_i4_9f97658f.txt`). The bare-ring column is a bulk-copy ring with no consumers streaming the same jm-routed bytes, net of the 5.3 µs empty-graph cost, measured on 11-03 against v4111's bench2 medians (`M/t3/STATE.md`, "RV review"). Budgets come from the measurement lanes of 11-01 and 11-03, sources per column below.

| shape | SGLang µs | i4 µs | judge SOL µs | bare ring µs (v4111 bench2 µs) | start | tail + end chain µs | GPC clock (time-weighted) | binding resource |
|---|---|---|---|---|---|---|---|---|
| bs1 | 76.90 | 21.58 | 11.42 | streamer ~19.2; no-compute 21.94 (24.0-24.2) | consumers' cold path to the first release ~4.8 µs | slowest `red` → last store 0.512 | (MMA cap ~1.79) | consumer path, 1:1 in both phases |
| bs8 | 146.99 | 60.62 | 37.32 | ~53.0 (62.6) | first copy ~1.4 µs after SU (was 2.49); first data p50 4.80 | 1.5 + 2.7 | 1.775 | edges |
| bs16 | 188.05 | 82.87 | 52.52 | ~74.5 (84.2) | first copy ~1.4 µs after SU (was 2.86); first data 4.93 | 2.3 + 2.8 | 1.720 | edges + throttled second half |
| bs32 | 212.27 | 114.06 | 70.76 | ~99.5 (110.5) | first copy 2.82; first data 6.21 | 2.2 + 2.6 | 1.710 | stream in steady state, edges |
| bs128 | 460.06 | 160.95 | 101.34 | ~140 (160.9) | first data 8.67 | ~5.8 (end − median last stage) | 1.685 | compute-bound heavy phase + power-throttled clock |
| bs256 | 620.13 | 194.76 | 108.94 | 152.4 (196.8) | first data 11.04 | ~6.5 | 1.668 | compute: heavy stage = decode + tensor − 750..1000 cycles |

Sources:
- bs1: `11_.../notes/t3_B1_NOTES.md` (no-compute and streamer bounds), `t3_SU_NOTES.md` mission 2 (consumers' cold path), `t3_EC_NOTES.md` §1 (end chain after the EC change).
- bs8-32 tail + end chain: `t3_ST_NOTES.md` mission 2.
- First copies: `t3_SU_NOTES.md` budget.
- First data p50, and bs128/256 kernel end − median last stage: `10_.../notes/wptail.md` (v1512).
- Clocks: P-CLK in `M/kda_ws/docs/draft.md` §7 (v1512, judge-matched nsys).
- Heavy stage: `t3_HV_NOTES.md` round 1.

How to read it:
- **bs8/16/32 stream in steady state.** 1-group stage-top waits of 150-223 cycles show the stream pacing, with ~20 % consumer slack (ST, ncu SourceCounters). The ~10-18 % over the bare ring is start, tail and end chain. These are latency chains of dependent hops: cold instruction fetch after the judge's flush, routing, the first copy, signal → release → poll → combine.
- **bs128/256 are compute-bound in their heavy phase.** Decode and tensor work do not overlap within a warp. The all-heavy first waves leave DRAM at 16-63 % for 30-50 µs; then a light phase saturates DRAM and the power limiter pulls the clock to ~1.5 GHz (P-NSYS1511, `M/kda_ws/docs/draft.md` §7). The clock ceiling, if 1.80 GHz held, is ~+1.7 % geomean (`11_.../notes/t3_PW_NOTES.md`).
- **The decode loops run at ~62 % of issue peak** with 2 compute warps per scheduler: 13 instructions per 8 weights, 45 % of them on the half-rate ALU (P-NCU1514, `10_.../evidence/ncu1514/`). No lossless layout gives fewer than 9 integer ops per 32-bit word.
- **What was still open at this point:** bs8/16 tails and end chains; bs256's heavy phase (decode/tensor overlap, energy per weight); bs1's one cross-CTA exchange. See `dead_ends.md` for what was tried.

## Tools

`tools/` is the harness as it stood at milestone 11 (11-03):
- the problem files `TASK.md`, `reference.py`, `checks.py`, `baseline.py` (SGLang's path; needs sglang + sgl_kernel) and `nvfp4.py`;
- the local judge-alikes `bench.py` (first version) and `bench2.py` (jm seeds, same-job A/B on shared weights, `B2_SHARE_EQ=1`), with the routing-calibration tools `route_pool.py` and `jdist.py` that produced `jm_seeds.jseeds`;
- correctness gates `hang1.py` (20 s watchdog per size) and `check_mt.py` (many M, normal + skewed, 2 calls);
- codegen tools `kern_sass.py` (per-kernel SASS summary, wgmma serialization check) and `wpcgs_search.py` (loop-exact search);
- `calib.sh` and `val_i3.sh`, the calibration and validation recipes of milestone 11 (paths refer to the original `dev/` layout).
