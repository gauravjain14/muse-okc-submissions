# muse_attention_decode

Decode and verify attention over an FP8 paged KV cache on an H200, timed against SGLang's FA3 kernel. This was a problem of the Open Kernel Challenge (closed 2026-11-04). This tree keeps every build that moved the score, with the profiling evidence behind each decision and the measured result.

## The problem

One call computes one decode step (q_len L = 1) or one 8-token speculative verify block (L = 8, causal inside the block) of grouped-query attention for B requests (`tools/TASK.md`):

```
q, k, v, gate = split(qkv[t])                 # fused projection output, BF16
q = rmsnorm(q) * 5.473;  k = rmsnorm(k)       # QK-norm
q, k = rope(q, p), rope(k, p)                 # sliding layers only
cache[cache_loc[t]] = fp8(k), fp8(v)          # write the new token(s) into the FP8 E4M3 cache
a = softmax(q @ K.T / 8) @ V                  # last 2,048 tokens (sliding) or the whole ~80K context (full)
out = rmsnorm(a) * sigmoid(gate)              # BF16 [B, L, 16 * 64]
```

- TP8 geometry: 16 query heads, 2 KV heads, head_dim 64. One K row and one V row are 128 B each per token.
- The cache is paged through a page table. The page size is 1 in the scored shapes and 64 in most tests.
- The 12 scored shapes are 6 loads (decode / verify × B = 1 / 16 / 32), each with a sliding and a full layer at ~80K context. Short names used throughout:
  - `d` = decode (L = 1), `v` = verify (L = 8);
  - `1` / `16` / `32` = batch size;
  - `s` = sliding (2,048-token window with RoPE), `f` = full context.

  For example, `d1s` is decode, batch 1, sliding; `v32f` is verify, batch 32, full.
- **Score.** A load's time is 46 × t(sliding) + 16 × t(full). The score is the geometric mean over the 6 loads of SGLang's load time divided by ours.
  - A µs saved therefore counts in proportion to 1 / (our load time): it is worth most on the B=1 shapes. Per µs: d1s ≈ 2.3-2.5 %, v1s ≈ 1.9-2.0 %, d1f ≈ 0.8 %, v1f ≈ 0.7 %, the batched sliding shapes 0.23-0.46 %, the batched full shapes 0.08-0.16 %.
  - These weights are computed from the load times of the 10 judge run.
- **Judge.** Each case is a CUDA-graph replay after an L2 flush, paired rep by rep with SGLang. The case time is the median of the paired ratios, after ~3.6-4 µs of graph-replay cost is subtracted from both sides. Correctness is checked on every timed call.
- **Rules that shaped the designs** (`tools/TASK.md`, plus the team's rulings recorded in the notes):
  - CUDA-graph safe: no host syncs, no allocation or branching on input values.
  - Fresh inputs every call; nothing cached.
  - BF16 activations; FP8 queries at most.
  - Every key attended (no sparsity).
  - The source is compiled by NVRTC at import, and the file must stay < 1,000,000 bytes.
  - B=1 length speculation is allowed only when it is verified in the kernel.
  - No page-table window guesses for B > 1.

## How to read the judge numbers

- **The judge is noisy.**
  - One run moves ±1.5-2 % on identical code; per-run CIs are ±1.5 %.
  - Era-0 judge nodes were bimodal: byte-identical d32f code read ~158.5 µs on fast nodes and ~165 on slow ones (`../autoresearch-kernels/attn/docs/PLAN.md`, 10-28 17:20 and 10-29 17:05).
  - SGLang's own d32f reps are bimodal (381-391 or 508-525 µs). This can move the paired median of one shape by 10-15 %: in 04 and 09, d32f read 146.2 and 144.8 µs while our raw times were unchanged (171.7 and 168.9 µs; typical readings ~165).
  - So every decision here was made on local same-run A/Bs. The judge only confirms the level.
- **Real level** = the mean over judge runs of byte-identical code (or byte-identical CUDA); with one run, that run.
- **Local method.**
  - `tools/bench.py` is a local judge-alike: graph replays after a flush, paired with SGLang.
  - `tools/ab.py` interleaves several files in one process and gives each its own graph instances. Identical code differs by 0.07-0.15 µs between module copies, so small effects were only trusted with ≥ 2 copies per arm and repeated runs.
  - "raw" local times include the ~3.6-4 µs graph-replay cost that the judge subtracts.

## Who did what (names used in the milestone records)

- **Era 0** (2026-10-27 → 10-30; `../autoresearch-kernels/attn/`, outside this tree's root). A coordinator and worker agents built k1 … k72.
  - Workers are named by pool: WB, WC, WD, WE, WF, WH, WS, W1-W21, P1-P3, S1-S5. The ROUND-2 module owners are XA-XG.
  - Builds are `dev/kNN.py`; the judged copies are `subs/kNN_{check,submit}.py`.
- **Era 1** (10-30 → 10-31; `opt-muse-decode/`). k72 → k80, with workers `base`, `xbg`, `xd` and `xe`.
- **Final week** (11-03; `muse-opt-codex/muse_decode_layer/cdl/AT*`). One lane (worker) per shape class: AT2 (the small shapes), AT3 (v1f), AT4 (d1f).
- **Kernel modules.** Each module is its own CUDA source string with its own launcher. `kernel_fn` is a chain of overrides, and the last one defined wins:
  - base `SRC`: era-0 kernels such as `run1f`, `run8`, `runE1`, `run8w`;
  - `XE`: B=1 sliding and d16s;
  - `XB1`: d1f;
  - `XG5`: v1f;
  - `XD`: verify full B > 1.

## Milestones that moved the score

Each milestone folder holds its decision record (`README.md`), the exact judged build (`build.py`), the diff against the previous milestone (`change.diff`), text evidence, and the raw lab notes it came from. Judge readings are speedups over SGLang; "chk" = an off-board `okc check`, "scored" = a board run. The real level is the mean over readings of byte-identical code or byte-identical CUDA.

| # | milestone | judged build (md5) | judge readings | real level | what drove it |
|---|---|---|---|---|---|
| 01 | [one fused FP8 split-KV kernel, FLAT token split](01_fused_flat_splitkv_fp8/) | era-0 k3 (0937facd) | 1.2010 chk | 1.2010 | SGLang runs 6-7 kernels per call and FA3 gathers 64-B half rows. One fused launch with whole 128-B rows, a FLAT split over 132 CTAs and a self-resetting in-kernel combine gave decode 1.6-1.9×. A no-compute skeleton showed that verify full was math-bound (0.65× SGLang) |
| 02 | [native QGMMA tiles, FLAT boundary balancing, B=1 cluster combine](02_native_qgmma_and_cluster_sliding/) | era-0 k6 (d544651d) | k4 1.5724 chk; k6 1.8412 chk | 1.8412 | SASS: FP8 `mma.sync` is emulated on sm_90a (12 unpacks + 2 HMMA; 1083 vs 417 cycles per tile). Native wgmma tiles for q_len 8, then q_len 1; a 16-CTA cluster DSMEM combine with verified length speculation took d1s 13.2 → 8.1 µs |
| 03 | [cutting the start and end chains](03_start_and_end_chains/) | era-0 k13 (cb51ad0f) | k8 2.0001, k10 2.0561, k11 2.0928, k13 2.1518 (all chk) | 2.1518 | ablations put 6-18 µs per call in the fixed end chains, and traces showed long start chains. Ten worker pieces shortened both: v1s −5.7, v16f −28.7, v32f −43.0 µs |
| 04 | [B=1 sliding on per-KV-head clusters](04_b1_sliding_per_head_clusters/) | era-0 k19 (5474cffe) | k15 2.1372, k16 2.2747, k19 2.4136 (chk; ≈ 2.37 without an SGLang-side d32f artifact) | 2.4136 | traces showed ~2 µs of start chain and 2.4 µs of tail in the B=1 sliding kernels. One 16-CTA cluster per KV head, producer warps with a straight-line predicated prefix, and a register-push tail: d1s 7.42 → 4.37, v1s 10.99 → 5.51 µs |
| 05 | [per-request clusters for batched sliding, FP16 softmax in the verify tile](05_batched_sliding_clusters_and_fp16_softmax/) | era-0 k32 (752d7dec) | 6 chk 2.39-2.48 (incl. byte-identical k29 chk 2.4347 / scored 2.5175); k32 2.5543 scored | 2.5543 | ncu: the verify-full tile is issue / ALU-bound at the 128-register cap. An instruction diet plus FP16 softmax arithmetic (once the team allowed FP16) cut v32f by 26 µs locally; DSMEM clusters for batched sliding; fixed-reference sums for B=1 full |
| 06 | [one compact kernel per shape class: XE, XA, XD](06_per_class_modules_xe_xa_xd/) | era-0 k37 (187c7ec5) | k36 2.6572 chk; k37 2.6836 chk, 2.6724 scored | 2.6780 | a 12-shape profile and a SASS census: the monolith's start paths ran ~13 KB of cold SASS after the L2 flush. Dedicated modules for B=1 sliding, d1f and verify full: d1s 4.17 → 3.9, d1f 14.1 → 11.7, v16f 105 → 99.6 µs |
| 07 | [shorter end chains, cluster kernels for d16s and v1s](07_end_chain_and_cluster_kernel_stack/) | era-0 k55 (f0bff1b7) | 10 runs 2.70-2.85; k55 2.9266 scored (fast node), 2.8712 chk (slow node) | 2.8989 | end-chain anatomy of each module (publish → arrive → wait → combine). Fewer dependent round trips (FP16 partials, never-reset counters, tagged slots): v1f 17.3 → 12.4 µs; new cluster kernels for d16s and v1s |
| 08 | [a judge-faithful bench and the cold-start pieces it selected](08_judge_faithful_cold_start_stack/) | era-0 k72s (614bc266) | 11 runs 2.81-2.93; k72s 2.8832 chk, 2.9033 scored, 2.8620 chk (era 1, same CUDA) | 2.8828 (all slow nodes; +1-2 % like for like) | the judge frees and reallocates memory per rep pair, so the first rep runs on cold GPU TLBs, which the local bench never saw. A bench that reproduces it picked early page touches, `L2::evict_first` on verify-full K/V copies and K/V-first start reorders |
| 09 | [nine latency fixes: start loads, cross-CTA combines, f16x2 masks](09_start_combine_mask_fixes_stack/) | era-1 k80 (32ab55f1) | k72s 2.8620 chk, k77 2.9367 chk; k80 2.9036 chk, 3.0182 scored (≈ 2.96 without an SGLang-side d32f artifact) | 2.9609 | per-worker traces and SASS of the start and combine chains. Compile-time grid size, staggered combine polls, f16x2 masks, cp.async staging: −1.0 % board time in a same-run A/B |
| 10 | [B=1 latency trims](10_b1_latency_trims_stack/) | AT4 `cand_b2` (f14ec998) | fr1 2.9573, c4 2.9585, stack1 2.9094; `cand_b2` 2.9746 (all scored) | 2.9746 | d1s sat 0.73 µs above its chain floor; ncu found a hoisted `seq_lens` wait before the v1f burst; d1f lanes were compute-latency-bound. Fixed-reference merges, a v1f combine and the `SRL` fix, 2-tile lane passes: ≈ +0.8 % locally |

Per-shape judge times of each milestone build (µs; mean of the readings above; * = an SGLang-side artifact, see 04 and 09):

| build | d1s | d1f | d16s | d16f | d32s | d32f | v1s | v1f | v16s | v16f | v32s | v32f |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SGLang (k3 run) | 20.38 | 31.58 | 26.98 | 200.9 | 31.89 | 384.8 | 23.09 | 35.90 | 28.99 | 205.4 | 34.72 | 390.8 |
| 01 k3 | 13.35 | 18.46 | 15.50 | 111.8 | 19.03 | 195.6 | 22.99 | 31.31 | 22.55 | 318.1 | 30.54 | 608.4 |
| 02 k6 | 8.12 | 17.23 | 10.76 | 91.8 | 13.53 | 170.3 | 16.73 | 21.39 | 16.13 | 147.8 | 19.34 | 262.0 |
| 03 k13 | 7.42 | 16.04 | 10.36 | 86.2 | 12.35 | 161.3 | 10.99 | 20.02 | 13.83 | 119.1 | 17.00 | 219.0 |
| 04 k19 | 4.37 | 15.80 | 9.61 | 89.1 | 11.89 | 146.2* | 5.51 | 19.06 | 13.45 | 119.4 | 15.73 | 219.3 |
| 05 k32 | 4.17 | 14.11 | 8.66 | 87.8 | 10.34 | 165.5 | 5.55 | 17.57 | 10.36 | 105.0 | 14.48 | 188.6 |
| 06 k37 | 3.90 | 11.73 | 8.87 | 87.9 | 10.47 | 166.0 | 5.25 | 17.34 | 10.77 | 99.6 | 14.48 | 180.7 |
| 07 k55 | 3.54 | 10.88 | 7.15 | 85.2 | 10.22 | 161.6 | 4.70 | 12.59 | 10.11 | 94.6 | 14.12 | 173.3 |
| 08 k72s | 3.84 | 10.83 | 6.49 | 86.9 | 10.51 | 165.3 | 4.57 | 12.46 | 9.70 | 93.4 | 13.23 | 173.7 |
| 09 k80 | 3.58 | 10.31 | 6.70 | 86.7 | 10.43 | 155.0* | 4.55 | 12.00 | 9.34 | 93.4 | 12.48 | 172.6 |
| 10 cand_b2 | 3.25 | 10.00 | 6.96 | 87.2 | 10.33 | 165.6 | 4.07 | 11.69 | 9.46 | 94.4 | 12.58 | 174.1 |

Sources: `../autoresearch-kernels/attn/subs/judge_ledger.jsonl` (01-08); the judge JSONs in `opt-muse-decode/runs/profiles/` (08's era-1 reading, 09) and in `muse-opt-codex/muse_decode_layer/cdl/AT4/profiles/` (10).

Two rows are judge artifacts:
- **SGLang-side artifacts.** The d32f readings marked * come from SGLang's bimodal reps. The kernels' raw times were unchanged: 171.7 µs in 04, 168.9 µs in 09.
- **Node speed.** The judge nodes differ by up to ~4 % in HBM speed, so the full-layer columns move ±2-4 % on identical code (compare 07's two readings of the same k55).



## Where the time goes in the last released build (10, `cand_b2`)

Judge times come from the build's scored run (1c64d7ee; `10_*/evidence/judge_after_sub_b2_1c64d7ee.txt`). Floors are measured gather or chain floors. They are given as local raw µs (graph replay including the ~3.6 µs fixed cost), because that is how they were measured; compare them with our local raw times in the same row.

| shape | kernel | SGLang (µs) | ours (µs) | share of its load | floor vs ours, local raw (µs) | what the rest is |
|---|---|---|---|---|---|---|
| d1s | `xe_r1_1` | 20.2 | 3.25 | 48 % of d1 | memory chain + DSMEM push 6.85-6.90 vs ~7.46 (k80's 7.58 in the same session, less the 0.11-0.13 trim; `muse-opt-codex/muse_decode_layer/cdl/AT2/NOTES.md`, `mb_d1s.py`) | compute → CTA merge → push → owner merge → store, run in series |
| v1s | `xe_b1s_8c6` | 22.3 | 4.07 | 50 % of v1 | gather chain for this layout 7.55-7.79 vs 8.21 (`AT2/NOTES.md`, `mb_hd.py`) | the same chain with 8 query rows per KV head |
| d1f | `xb_d1f` | 31.1 | 10.00 | 52 % of d1 | B=1 full gather 11.14 vs ~13.5 (`AT2` `mb_g1.py`; `AT4/NOTES.md`) | each lane is compute-latency-bound after tile 0 (~1050 cycles per tile); the end chain is 3 dependent L2 round trips (~1.5 µs) |
| v1f | `xg_v8f` | 36.2 | 11.69 | 50 % of v1 | 11.14 vs ~15.1 (`AT3/NOTES.md`) | per-warpgroup serial pair steps (~1.34-1.47 µs each, power-capped clock) + an end chain of ~3 µs (publish, gate, bulk loads, final) |
| d16s | `xe_dm_8_4_2` | 27.1 | 6.96 | 19 % of d16 | B16 sliding gather 9.36-9.57 vs 10.35 (k80; `AT2/NOTES.md` "v16s end phase") | push → owner end chain |
| d32s | `attn_1_1_1_1e3` | 31.5 | 10.33 | 15 % of d32 | 11.46 vs 13.70 | prologue and combine |
| v16s | `attn_8_1_1_1x6` | 29.3 | 9.46 | 22 % of v16 | 9.36-9.57 vs 13.04 | Q prep before the first tile; ~2.6 µs end phase; the DSMEM pull of ~110 KB per cluster takes ~0.7 µs |
| v32s | `attn_8_1_1_1w4` | 34.5 | 12.58 | 17 % of v32 | 11.46 vs 16.17 | global-slot combine |
| d16f | `attn_1_0_1_0g` | 200.4 | 87.23 | 81 % of d16 | pure page-table-order gather 86.2 (first sandbox GPU) vs 91.0 (k80, A/B session; GPUs differ by up to 4 %; `opt-muse-decode/NOTES.md`, `opt-muse-decode/runs/ab_k73_k80.txt`) | the era-1 notes put this kernel at the random-gather ceiling for its access order (~4.0 TB/s) |
| d32f | `attn_1_0_1_0` | 385.1 | 165.58 | 85 % of d32 | 163.5 vs 170.0 (same sources) | same |
| v16f | `attn_xd_v8f` | 204.7 | 94.40 | 78 % of v16 | ~6-8 µs above the same ceiling (`opt-muse-decode/NOTES.md`) | start 5.8, steady 3.4, tail 4.0, end 5.4 µs lost (era-0 W10 split of a k63 trace, `../autoresearch-kernels/attn/docs/PLAN.md` 10-30 00:20) |
| v32f | `attn_xd_v8f` | 390.5 | 174.08 | 83 % of v32 | same | start 5.2, steady 11.0, tail 4.5, end 5.1 µs lost (same source) |

- **B=1 loads.** Sliding and full each take about half of the load time. These four shapes carry the most score per µs (d1s 2.5 %, v1s 2.0 %, d1f 0.86 %, v1f 0.71 %; computed from this run's load times). All four are chains of fixed latencies, 0.4-4 µs above their floors.
- **Batched loads.** The full layer is 78-85 % of each load. Those kernels stream at the random-gather ceiling for page-table order (~4.0 TB/s on 128-B rows); a sequential read of the same bytes is ~10 % faster (147 vs 163.5 µs on d32f, `opt-muse-decode/NOTES.md`). The verify-full kernels also carry 6-8 µs of start, coupling and end chain on top of that.
- **Combines.** The split-KV combine costs ~1 µs per dependent L2 round trip under load; polling a line that other SMs are writing costs 0.8-1.2 µs per iteration (`AT4/NOTES.md` §4). Every faster protocol tried was slower (`dead_ends.release.md`).

## Layout and how to reproduce

```
muse_attention_decode/
  README.md          this file: the problem, the ladder, where the time goes
  dead_ends.release.md   every failure that changed direction (idea | number | why | milestone it informed)
  tools/     harness as at the last milestone (10): TASK.md (problem spec), reference.py, checks.py,
                     baseline.py (SGLang's kernel), bench.py (local judge-alike), ab.py (same-run interleaved A/B),
                     nearby.py (86 nearby-shape correctness cases: 56 + 30 B=1 full decode), nvk.py (NVRTC module helper)
  NN_<mechanism>/    README.md (decision record), build.py (exact judged build), change.diff (vs the previous milestone),
                     evidence/ (judge results, microbenchmarks, trace / probe scripts, ncu / nsys / SASS text),
                     notes/ (raw lab notes, unedited)
```

All you need is torch, SGLang's kernels for the baseline, and an H200. Each build is self-contained: its CUDA source is compiled by NVRTC (CUDA 13) at import.

```bash
cd muse_attention_decode/tools
python bench.py ../NN_*/build.py --tests                                   # the 16 judge tests + local judge-alike timing of the 12 shapes
python ab.py ../A_*/build.py ../B_*/build.py --shapes d1s,v1f --seeds 7,8,9 --caps 2   # same-run A/B; pass a file twice for 2 module copies
python nearby.py ../NN_*/build.py                                          # nearby-shape correctness
```

Binary profiler reports (`.ncu-rep`, `.nsys-rep`) are not copied; each milestone names the originals. Original trees, relative to ``: era 0 `../autoresearch-kernels/attn/`, era 1 `opt-muse-decode/`, final week `muse-opt-codex/muse_decode_layer/cdl/`.
