# 05: per-request clusters for batched sliding layers, and an FP16 softmax in the verify tile (verify full, batched sliding, batch-1 full)

Shape names: d / v = decode (q_len 1) / verify (q_len 8); 1 / 16 / 32 = batch; s / f = sliding (2048-token window) / full (~80K context). Era-0 files live outside this tree, under `../autoresearch-kernels/attn/` (written "E0" below).

| | before (04) | after (05) |
|---|---|---|
| build | `04_*/build.py` = E0 `subs/k19_check.py` (md5 5474cffe, 295,378 B) | `build.py` = E0 `subs/k32_submit.py` = `dev/k32.py` (md5 752d7dec, 488,392 B) |
| judge | 2.4136 (okc check 13ed784f). Its d32f read 146.2 µs on a raw median of 171.7: SGLang's bimodal reps, worth +1.9 % of score. Without that artifact the reading is ≈ **2.369** | **2.5543** (scored run 820fe912, one reading) |
| real effect | | +5.8 % as read; **≈ +8.5 % on our side** (SGLang times held fixed, k19's d32f artifact removed); same-job local chain +12.0 % (judge-projected +11.2 %) |
| shapes changed | | v16f, v32f (tile diet + FP16 softmax); d16s, d32s, v16s, v32s (new per-request cluster kernels); d1f, v1f (combine + start); d16f, d32f (page-table prefetch); v1s −0.2 µs locally (FP16 exchange). d1s runs an identical kernel |

This milestone is a stack of 13 deliveries from five worker lanes of the era-0 team. **W9, WS, WE, WH and WB** are agent lanes; each owned a shape class and handed back a builder patch on the current build (`dev/k20.py` … `dev/k32.py`, each with a lineage docstring).

## 1. Starting point

k19's judge times and kernels (`evidence/judge_before_k19_check_13ed784f.txt`; local from E0 `out/verify_k19.txt`, k19 column).

Kernel names read `attn_<q_len>_<RoPE>_<page 1>_<ALIGNED><variant>`:
- **ALIGNED** splits each request over its own CTAs (batch 1, and the sliding layers).
- **FLAT** concatenates every request's keys and splits them evenly over 132 CTAs (full layers, B > 1).
- `run1*` are the q_len-1 templates; `run8*` is the q_len-8 template, built on FP8 wgmma (QGMMA e4m3).

| shape | k19 kernel | judge "yours" (µs) | local (µs) | SGLang (µs) |
|---|---|---|---|---|
| d1s | `attn_1_1_1_1s` (runA1h, clusters of 16) | 4.37 | 3.33 | 21.07 |
| d1f | `attn_1_0_1_1` (run1 ALIGNED, 132 CTAs) | 15.80 | 14.04 | 31.07 |
| d16s / d32s | `attn_1_1_1_1` (run1 ALIGNED, combine through global memory) | 9.61 / 11.89 | 8.60 / 10.75 | 26.56 / 31.44 |
| d16f / d32f | `attn_1_0_1_0` (run1f FLAT) | 89.09 / 146.17 (raw 171.7) | 84.82 / 159.42 | 200.14 / 386.27 |
| v1s | `attn_8_1_1_1h` (runA8h) | 5.51 | 5.07 | 22.64 |
| v1f | `attn_8_0_1_1` (run8 ALIGNED) | 19.06 | 18.13 | 35.82 |
| v16s / v32s | `attn_8_1_1_1` (run8 ALIGNED, combine through global memory) | 13.45 / 15.73 | 12.00 / 14.48 | 29.23 / 34.27 |
| v16f / v32f | `attn_8_0_1_0` (run8 FLAT) | 119.38 / 219.29 | 112.30 / 208.53 | 205.34 / 389.76 |

Per-load speedups on the k19 judge: d1 3.23, d16 2.37, d32 2.64, v1 2.89, v16 1.83, v32 1.85.
- The two weakest loads, v16 and v32, were ~80 % verify-full time.
- d1f and v1f were 56 % and 55 % of their loads (`notes/PLAN.md`, "Judge k19").

## 2. What we saw

**(a) Verify full B > 1: the lockstep run8 FLAT tile was instruction-bound, and restructuring it lost.**
- The loop of `attn_8_0_1_0` executed 475 SASS instructions per warp-tile, 34 of them MUFU (`evidence/w9_out_f1_diet_sass.txt`).
- Restructuring lost:
  - W8's FA3-style producer / consumer ring capped at ~181 µs for the b32 gather, against a 158.7 µs lockstep floor.
  - W9's rounds 1-4 were all neutral or worse: chained PV, O in smem, V^T one tile ahead, lane phase offset, lazy max, hybrid exp (`notes/PLAN.md`, "DEAD END (W8 round 5)" and the entry labelled 09:05).
- WS's copy probe closed the warp-specialized route at its root (`notes/WS_NOTES.md`, Round 5):
  - TMA / `cp.async.bulk` one-row copies are issue-bound: 3.36 / 3.28 TB/s at 328 MB, against 4.09 for the cp.async 16-B ring.
  - With one issuing warp per lane they reach only 1.0-1.6 TB/s.

**(b) Cheaper softmax arithmetic: the in-rules version lost; FP16 won.**

Resident tile microbenchmark, in cycles per 64-token tile per lane (`evidence/w9_out_h1_fp16_tile.txt`):

| softmax | cycles |
|---|---|
| FP32 ex2 | 1676 |
| EXQ (P's E4M3 code computed by one FFMA, Schraudolph style) | 1665 |
| FP16 variant A | 1402 (−16 %) |
| FP16 variant B (QK also accumulated in f16) | 1241 (−26 %) |

- **The in-rules alternative lost.** It folded the softmax scale and max into the QK wgmma (fresh-eyes IDEA 1).
  - It passed precision, but it needed 810-845 resident cycles per SM-tile against a ≤ 700 target.
  - End to end it was 7 % slower: v16f 114.8 vs 107.2 µs, v32f 210.1 vs 196.4 (`notes/PLAN.md`, "DEAD END … (W9 round 8)").
- **FP16 needed a policy change.**
  - The team's own rule was "FP8 Q/K/P/V + FP32 accumulation, BF16 elsewhere, no FP16", which is stricter than TASK.md.
  - The user dropped it (`notes/PLAN.md`, "POLICY CHANGE"). Only the competition rules remain: the tolerance, BF16 where SGLang keeps BF16, an FP8 cache, and queries no coarser than E4M3.

**(c) Batched sliding: global-memory end chains dominated, and per-request clusters fit on the chip.**
- In the k18 trace of the verify-sliding global path (`notes/WS_NOTES.md`, Round 3), about 4.5 µs of merge, publish, release and fold follow the tiles:
  - b16: tiles done 6.51 / 7.92 µs (median / max), combined 11.25 / 11.89.
  - b32: tiles 9.46 / 10.78, combined 13.62 / 14.37.
  - Balancing tiles across parts gained nothing (v16s 11.91 vs 11.90).
- WE's co-residency probe for 512-thread CTAs, 1 per SM (`notes/WE_NOTES.md`, job1):
  - Clusters that fit at once: 17 of C6, 39 of C3, 30 of C4, 15 of C8.
  - So per-request clusters fit as 16 × C6 (b16) and 32 × C3 (b32), 96 CTAs each.
  - The alternative, per-(request, KV head) clusters, gathers 64-B half rows, which is 6-13 % slower.
  - Gather floor with the real seq → page table → K/V chain: b16 C6 5.70 µs, b32 C3 8.26 µs.
- run8's ~1.1-1.25 µs lane step is compute, not fences: removing both proxy fences moved v16s only 11.68 → 11.53 µs (`notes/WH_NOTES.md`, job1).

**(d) Batch-1 full: the gather sat at its floor; the combine did not.**
- d1f, k19 trace (`notes/WS_NOTES.md`, Round 4):

  | event | K/V issued | tiles (med / max) | merged | published | released | combine loads | combined |
  |---|---|---|---|---|---|---|---|
  | µs | 3.67 | 8.20 / 8.91 | 9.02 | 10.02 / 10.70 | 11.34 | 12.62 | 14.02 |

  About 5 µs follow the last tile. The tile phase (~6.5 µs) already equals the 20.5 MB gather floor of this launch shape (WS Rounds 5-6).
- v1f has a ~6 µs end chain: merge 0.9 + publish 0.83 + release ~0.6 + combine 2.9 µs (`notes/WB_NOTES.md`, Round 14).
  - It is latency: the 132-partial L2 burst takes 1.26 µs, then the reductions run serially.
  - It is not cold code: the flush and warm runs are identical.
- **RMS-normalized q and keys bound every score:** |q·k|·sm_scale·log2 e ≤ 63.2 (the `WS_REF` comment in `build.py`).
  - So a fixed reference REF = 64 keeps every partial weight 2^(m − REF) finite, and partials can be summed with no max merge.
  - A range flag (any max outside [REF − 90, REF + 40]) falls back to the exact merge, so the result stays correct for any input.
- Atomics pay only with few rows per CTA:
  - One 256-thread burst of `red.global.add.v4.f32` from d1f's merged 16-row partial gained 0.9 µs.
  - Reds straight from 132 × 512 threads' registers took 23.4 vs 14.1 µs, because they collide with the gather (WS Round 4, job22).
  - At 128 rows per CTA (v1f), the reds slow the publish by ~1.9 µs, as much as the combine they remove (`notes/WB_NOTES.md`, Round 15).

**(e) Decode full B > 1: page-table traffic sat inside the bandwidth-bound phase.**
- The coordinator's ruling (c) allowed prefetching a full layer's whole page-table rows into L2 at t = 0: a shape-determined superset of what the layer reads anyway.
- WS's d32f trace, prefetch off vs at entry (`notes/WS_NOTES.md`, Round 7):
  - The length barrier moved later, 0.83 → 2.13 µs.
  - Tiles ended 155.2 → 154.0 and the last stamp 159.8 → 159.0.
  - The page table's ~1.7 % of the bytes moved out of the HBM-bound phase into the idle start.

**(f) The judge's noise was as large as most single pieces.**
- k29 was read twice with byte-identical code (same sha256): check 2.4347, scored 2.5175 (+3.4 %).
  - d16f read 87.75 vs 84.43 and d32f 166.47 vs 159.58 (`evidence/judge_readings.txt`).
  - The raw medians moved too, 91.9 vs 88.4 and 170.2 vs 163.3 µs (E0 `subs/profiles/k29_*.json`): a faster node.
- k19's d32f "146.2 µs" was SGLang's: 16 of 60 SGLang reps ran at ~514 µs instead of ~390 and flipped the paired median. Our raw median was 171.7 (`evidence/judge_readings.txt`; `notes/PLAN.md`, "FACT CORRECTION").

## 3. Diagnosis and projection

- **Per-shape value, from the k19 judge** (`notes/PLAN.md`, "Judge k19"):
  - d1f → 10 µs is worth ≈ +3.9 %; v1f → 11 µs ≈ +4.5 %.
  - Verify full → 90 / 170 µs local is worth ≈ +5.7 %.
- **Projection after k20** (`notes/PLAN.md`, entry labelled "09:20 PROJECTION"):
  - Realistic per-shape judge targets: d1f 11, v1f 12, d16s 7.5, d32s 9, v16s 10, v32s 12, d16f 82, v16f 100, v32f 185 µs.
  - Even all of them give a board time of ~1328 µs, ≈ x2.84 (inferred: the judge's SGLang step time 1560.5 × 2.4136 = 3766 µs, divided by 1328).
  - These targets set this milestone's work list.
- **Limiters per class:**
  - Verify-full tile: issue slots per 64-token tile (FP16 saves 16 % of resident cycles).
  - Batched sliding: the global end chain. WE's floors for a cluster structure are b16 ~5.5 µs and b32 ~7.6 µs local.
  - b1 full: the end chain (~5-6 µs) and the start. The gather is already at its floor.
  - Decode full: 7-8 µs above the 76.4 / 152 µs gather floors (local) (`notes/PLAN.md`, "Measured").

## 4. Change

`change.diff` adds 3,907 lines and removes 48 (`diff -u subs/k19_check.py subs/k32_submit.py`). The local effects below are same-job deltas against the mean of two control copies of the previous build (`evidence/verify_chain.txt`, from `evidence/verify_kNN.txt`).

| build | piece (key code names) | lane | shapes | local effect (µs) | source |
|---|---|---|---|---|---|
| k20 | Instruction diet of run8's FLAT tile loop (`W9F_*`): window = 0 at compile time; one predicated cp.async per K/V row (no branch); next tile's page-table rows read before the lane barrier; both V^T `ldmatrix.trans` issued together; the proxy fence moved to the pre-PV barrier. Loop 475 → 415 SASS per warp-tile | W9 | v16f, v32f | −5.73, −11.77 | `verify_k20.txt`, `w9_out_f1_diet_sass.txt`, `notes/W9_mk_k18f.py` |
| k21 | `run1w` for d1f with `WS_B1A`: the merged CTA partial is scaled to 2^(m − 64) and added by one burst of `red.global.add.v4.f32` into a zeroed sum; the last arrival reads 4 KB | WS | d1f | −0.89 | `verify_k21.txt`; WS Round 4 (−0.7) |
| k22 | Compact start of run8 FLAT (`W9S_RL`): the first DST − 1 tiles' page-table rows are loaded before the Q / gate / new-token loads | W9 | v16f, v32f | −2.30, −1.88 | `verify_k22.txt`, `notes/W9_mk_k21f.py` |
| k23 | `runE1`: decode sliding B > 1 on one cluster of C CTAs per request (C6 at b16, C3 at b32), DSMEM combine (details below) | WE | d16s, d32s | −1.78, −2.19 | `verify_k23.txt`; WE job15 |
| k24 | Fixed-reference atomic combine. `W9D` (run1f, decode full B > 1) is on. `W9A` (run8 FLAT) ships **off**: its variants cost +0.3 … +7.8 µs | W9 | d16f, d32f | −0.42, −0.32 (noise level) | `w9_out_a2_*.txt`, `w9_out_a1_*.txt`, `notes/W9_mk_k23a.py` |
| k25 | `WB_FDA`: k20's diet ported to the ALIGNED verify-1 full kernel `attn_8_0_1_1` (loop 470 → 405 SASS) | WB | v1f | −0.75 (k24 + k25) | `verify_k25.txt`; WB Round 17 (−0.5 … −1.0) |
| k26 | `W9S`: verified length speculation at the d1f start (details below) | W9 | d1f | −0.43 | `verify_k26.txt`, `notes/W9_mk_k25d.py` |
| k27 | `run8w`: verify sliding B > 1 on per-request clusters (B ≤ 16) or a global owner path (B ≤ 33) (details below) | WH | v16s, v32s | −1.51, −0.77 | `verify_k27.txt`; WH job19 |
| k28 | `WB_SUMC` + `WB_SE1`: v1f sum-only combine. Partials are stored pre-scaled to 2^(M − 64) with plain stores; out-of-range rows are stored classic with l negated; combiners just add; compact epilogue | WB | v1f | −0.71 | `verify_k28.txt`; WB Round 19 (−0.82) |
| k29 | `WS_PTP 1`: run1f prefetches a 1/G slice of the whole [B][max_pages] page table into L2 at entry (`prefetch.global.L2`) | WS | d16f, d32f | −1.33, −0.50 | `verify_k29.txt`; WS Round 7 (−1.1 … −1.2 / −1.0 … −1.7) |
| k30 | `W9H` variant A: FP16 softmax arithmetic in the q_len-8 wgmma tile (`wg_tile_h`, h16.cu) | W9 | v32f, v16f, v1f, v16s, v32s | −12.79, −5.71, −1.12, −0.61, −0.48 | `verify_k30.txt`, `w9_out_h1/h3_*.txt`, `notes/W9_mk_k30h.py` |
| k31 | `WS_PTP 10`: the prefetch moves off the start chain. Lanes 0-3 of warp 1 issue `cp.async.bulk.prefetch.L2` for 5/8 of the slice right after the length barrier, and the last 3/8 after the first segment's rows | WS | d16f, d32f | −0.3 … −0.4, −0.55 … −0.6 | WS Round 8; `verify_k31.txt` |
| k32 | `WB_H16A8/W/G/E`: FP16 partial exchange in runA8h, run8w (cluster and global) and runE1. Rows carry O / l as FP16 plus FP32 (m, l), or O × 2^-13 as FP16. Off for runA1h and v1f, where it was neutral | WB | v16s, v1s, v32s, d16s | −0.58, −0.22, −0.18, −0.11 | WB Round 21 (job62); `verify_k32.txt` (v16s −0.71) |

**The verify-full tile (k20, k22, k30): −26.4 µs on v32f and −13.7 µs on v16f, locally.** The tile is issue-bound, so each instruction removed showed up as time.

| tile of `attn_8_0_1_0` | loop SASS / warp-tile | MUFU | resident cycles / tile / lane | v16f / v32f, same job (µs) | what it taught |
|---|---|---|---|---|---|
| k17 tile (in k19) | 475 | 34 | | 112.1 / 208.6 | |
| + diet (k20) | 415 (407 in a later build) | 34 | | 107.9 / 197.7 | instruction count is time here (`w9_out_f5_diet.txt`) |
| + compact start (k22), as in k29 | 398 | 34 | 1676 | 106.7 / 197.6 | |
| + W9H variant A (k30) | 362 | 2 | 1402 | 101.0 / 184.8 | MUFU gone; one f16 FMA per score pair produces the code (`w9_out_h1/h3`) |
| W9H variant B (QK accumulated in f16) | | | 1241 | 111.1 / 206.7 | fewer cycles but slower end to end; not shipped. XD later traced it to a ~3× wider per-CTA spread (`06_*`) |

How variant A computes P (build.py, h16.cu header):
1. The FP32 QK accumulators are converted to f16 (`cvt.rn.f16x2.f32`).
2. An f16x2 max tree plus 2 quad shuffles gives the row max; m is that max rounded up to 1/8.
3. Then y = `fma.rn.f16x2`(S, 8c, 1144 − 8m), clamped by `max.f16x2`(y, 1024).
4. An f16 in [1024, 2048) has an ulp of 1, so its low byte is 8(s·c − m) + 120: the E4M3 code of P = 2^(s·c − m + 8). This is the code EXQ computed in FP32.
5. PRMT packs those bytes into the PV wgmma's A fragment.

Everything else keeps its precision: PV, O, l and alpha stay FP32, and P stays E4M3.

Range:
- |s| ≤ 8·8·448 < 65504, so the f16 scores are always finite.
- The bias is exact for |m| ≤ 399 log2 units.
- Beyond that, no FP8-query kernel can meet the tolerance anyway. For such rows, `wh_sane` turns the row's partial into an empty one after the loop, so no NaN reaches the output.

**Batched sliding on per-request clusters (k23, k27, k32): −1.8 / −2.2 µs on d16s / d32s, −1.5 / −0.8 µs on v16s / v32s.**

**`runE1`** (decode, WE):
- 8 producer warps issue K/V with cp.async.
- 2 consumer warpgroups attend pairs of 64-token tiles with an exact per-step max. The 2 producer warpgroups take the last 2 steps.
- Each CTA pushes its merged partial with `st.async` into the row owners' shared memory; the owners fold it and run the epilogue.
- C is a compile-time constant (kernels `attn_1_1_1_1e6` / `e3`).

How it converged (`notes/WE_NOTES.md`; same-job d16s / d32s in µs, k16 control ≈ 8.5 / 10.6):

| version | change | d16s / d32s | what it taught |
|---|---|---|---|
| v1 (job3) | producers issue, 2 consumer WGs, `st.async` push | 8.13 / 11.34 | ~0.6-0.8 µs per tile per WG: compute-bound at b32 |
| v2 (job5) | tile pairs: one QK batch, one max vote, one PV chain | 7.79 / 10.39 | 3 consumer WGs with only 4 producer warps lose: cp.async issue is capped per warp (~13 KB in flight) |
| v3 (job6) | 32-token issue units | 7.54 / 9.42 | when consumers also issue K/V, their Q prep slips |
| v5 (job9) | exact step max instead of the lazy vote | 7.26 / 9.33 | logits have a std of ~6 log2 units, so the lazy rescale fired on most steps |
| v7 (job12) | compile-time C (`_1e3`, `_1e6`) | 6.59 / 8.54 | with a runtime C the owner combine was a 16-way predicated unroll (~470 SASS, 1008-1140 cycles) |
| k17e (job15) | final defaults | 6.98 / 8.64 (k16: 8.72 / 10.63) | |

**`run8w`** (verify, WH):
- B ≤ 16: `attn_8_1_1_1x6`, one cluster of 6 CTAs per request. Merged rows go by `st.async` to the owner CTAs.
- 16 < B ≤ 33: `attn_8_1_1_1w4`, 4 parts per request. They publish to self-resetting sentinel slots in global memory, each a whole number of 32-B sectors. The owners prefetch their slot region into L2 at start; the new block is a 32-token tile.

What WH learned along the way (`notes/WH_NOTES.md`):
- Owners polling full rows saturated L2 (v16s 15.0 µs).
- Writing partial sectors cost ~1.3 µs.
- `setmaxnreg` producer warps stalled the consumers.
- The C6 cluster with `st.async` read 10.25-10.40 vs k19's 11.85-11.89.

**FP16 exchange (k32)** paid where the exchange is DSMEM-bound: run8w C6 halved both bytes and message count, −0.58 µs on v16s. On L2 combines, which are latency-bound, it did nothing (v1f neutral, WB Round 21).

**Batch-1 full: fixed-reference sums and speculation (k21, k25, k26, k28): d1f −1.3, v1f −1.5 µs locally** (v1f gained another −1.1 from the FP16 tile in k30).
- `WS_B1A` (d1f) and `WB_SUMC` (v1f) replace the max merge with a plain sum at REF = 64.
  - In the v1f trace, released → combined fell 2.85 → 1.62 µs (WB job56).
- `W9S` (verified length speculation):
  - Before, the d1f start waited for the length: length visible 1.12 µs, page-table loads 1.57, burst issued 3.73, tile 0 at 4.54 (`notes/W9_mk_k25d.py`).
  - W9S guesses seq = width − 1. The coordinator allowed this guess for B = 1 only, because SGLang's FA3 page table at B = 1 is sliced to exactly the past tokens plus the block.
  - Page-table loads go out without the length barrier, the guess is checked after the burst, and a miss drains the thread's own copies and redoes the start.

**Decode full (k24, k29, k31): d16f −1.5, d32f −1.4 µs summed over the 12 same-job verifies.** Most of it is the page-table prefetch above, made cheaper by moving it off the start chain (mode 10).

## 5. Result

| check | result |
|---|---|
| local, same job | 12 verify jobs. Each ran the previous build, the new build and an identical copy of the previous build in one container. Chained geomean **+12.0 %** (judge-projected with +1 µs per call: +11.2 %) (`evidence/verify_chain.txt`). Identical-code shapes moved up to ±0.5 µs (d32f), and once −1.67 µs (v32f in `verify_k27.txt`), so a single-piece delta below that is noise |
| correctness | every verify job: `check.py` 0 failures (the 16 tests plus the 12 scored shapes, twice each); judge 16/16 tests. FP16 exchange errors within ±5e-5 of k31 (WB job62) |
| judge | 2.4136 → **2.5543** (`evidence/judge_*`) |

Per case: k19 → k32 judge, the sum of the same-job local deltas, and the score each judge change is worth alone with SGLang's times held fixed (`evidence/judge_readings.txt`, `evidence/verify_chain.txt`):

| shape | k19 → k32 judge (µs) | Δ judge | Σ local Δ | alone |
|---|---|---|---|---|
| d1s | 4.37 → 4.17 | −0.20 | −0.17 | +0.34 % (identical kernel: node) |
| d1f | 15.80 → 14.11 | −1.69 | −2.26 | +1.03 % |
| d16s | 9.61 → 8.66 | −0.95 | −2.78 | +0.40 % |
| d16f | 89.09 → 87.76 | −1.33 | −1.52 | +0.19 % |
| d32s | 11.89 → 10.34 | −1.55 | −2.15 | +0.42 % |
| d32f | 146.17 → 165.47 | +19.30 | −1.44 | −1.68 % (k19 artifact) |
| v1s | 5.51 → 5.55 | +0.04 | −0.07 | −0.05 % |
| v1f | 19.06 → 17.57 | −1.49 | −3.10 | +0.73 % |
| v16s | 13.45 → 10.36 | −3.09 | −3.12 | +0.97 % |
| v16f | 119.38 → 104.99 | −14.39 | −15.47 | +1.60 % |
| v32s | 15.73 → 14.48 | −1.25 | −1.69 | +0.23 % |
| v32f | 219.29 → 188.63 | −30.66 | −28.88 | +2.07 % |

**Reading the judge honestly:**
- **k19's d32f is an artifact.** Set it to raw − the usual 4.19 µs offset and k19 reads 2.369, so our side gained ≈ +8.5 % (SGLang times held fixed).
- **d1s ran an identical kernel** (runA1h) and still read −0.20 µs: that is the node.
- **The verify-full work transferred in full:** v16f −14.4 judge vs −15.5 local; v32f −30.7 vs −28.9.
- **The cluster kernels transferred only partly:** d16s −0.95 vs −2.78, d32s −1.55 vs −2.15. The judge-minus-local offset on cluster shapes was later measured at d32s +1.44 µs against SGLang's +0.74 (E0 `dev/xe/NOTES.md`, Round 6); the cause was never isolated.
- **v1f also transferred only partly:** −1.49 vs −3.10.
- **SGLang read ~0.6 % faster in k32's run than in k19's:** the fixed-base score is 2.5704 vs 2.5543 as read.

**Against the projection** (judge targets from the 09:20 entry):

| shape | target | k32 judge |
|---|---|---|
| v32f | 185 | 188.6 |
| v16f | 100 | 105.0 |
| v16s | 10 | 10.36 |
| d16s | 7.5 | 8.66 |
| d32s | 9 | 10.34 |
| d16f | 82 | 87.8 |
| v32s | 12 | 14.48 |
| d1f | 11 | 14.11 |
| v1f | 12 | 17.57 |

- Verify full and v16s came close.
- The batch-1 full shapes missed by 3-5.5 µs. The sums shortened the combine, but two dependent L2 round trips and the start chain remained. A later probe put d1f's tiles-end → output at ~2.3 µs whatever the partial count (`notes/PLAN.md`, "FINAL (WS round 10)").
- The local chain (+12 %) runs ahead of the judge (+8.5 % on our side) for three reasons:
  - A/A noise between module copies is ~0.5 % geomean per job (`notes/PLAN.md`), summed over 12 jobs in which pieces were kept partly because they read well (inferred).
  - The cluster offsets above.
  - The b1 shapes' larger judge offsets.

## 6. Takeaways, and what came next

- **Check the team's own rules against the competition's before calling a structure finished.** The largest single piece, W9H (+3.05 % local), existed only after the user dropped a precision rule stricter than TASK.md.
- **In the verify-full tile, instruction count is time:** the diet saved 11.8 µs on v32f and FP16 another 12.8. 06 shows why: that tile is also power-capped.
- **For batched sliding, per-request clusters with a DSMEM push to row owners beat global combines.** On DSMEM, bytes and message count matter; on L2, latency does.
- **Wherever normalization bounds the scores, use fixed-reference partial sums** (REF = 64 log2 units, with an exact fallback) instead of max merges. Use atomics only with few rows per CTA.
- **Byte-identical code reads ±1.7 % apart on the judge** (k29), and one shape can carry a 25 µs SGLang-side artifact (k19 d32f). Read the per-case raw medians in the judge JSON before crediting a step.
- **After k32 the 10K-line monolith resisted further structural edits.**
  - W9 rounds 13-17 on the verify-full loop all lost.
  - Batch-1 full on clusters was killed.
  - So `06_*` rebuilt the shapes with the most score per µs as compact, dedicated kernels in their own modules: XE (batch-1 sliding), XA (d1f), XD (verify full).

## Reproduce

```bash
cd muse_attention_decode/tools
python bench.py ../05_batched_sliding_clusters_and_fp16_softmax/build.py --tests       # correctness + local judge-alike timing
python ab.py ../04_*/build.py ../05_batched_sliding_clusters_and_fp16_softmax/build.py \
       --shapes v32f,v16f,v16s,d16s,d1f,v1f,d1s --seeds 7,8,9 --caps 2                # same-run A/B; d1s is the control
# no GPU: the judge and same-job readings used above
cd ../05_batched_sliding_clusters_and_fp16_softmax/evidence
python judge_readings.py judge_before_k19_check_13ed784f.txt judge_k29_check_afe9b7c2.txt \
       judge_k29_submit_446324a0.txt judge_after_k32_submit_820fe912.txt judge_before_k19_check_13ed784f.json
python verify_chain.py verify_k2[0-35-9].txt verify_k3[0-2].txt
```

Originals:
- Builds: `../autoresearch-kernels/attn/subs/k32_submit.py` (= `dev/k32.py`); the base is `subs/k19_check.py`. Intermediate builds are `dev/k20.py` … `dev/k31.py`.
- Lanes: `dev/{w9,ws,we,wh,wb}/`. W9's 10-28 NOTES.md was later overwritten by another W9 lane. Its builders `dev/w9/mk_k*.py` and its outputs `dev/w9/out_*.txt` remain, and the builders are copied here.
- Judge results: `subs/profiles/`, `subs/judge_ledger.jsonl`. Verifies: `out/verify_k20.txt` … `out/verify_k32.txt`.
