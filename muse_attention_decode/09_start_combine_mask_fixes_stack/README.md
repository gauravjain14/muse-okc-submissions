# 09: nine latency fixes from four worker lanes: start loads, cross-CTA combines, f16x2 masks (d1s, d1f, v1s, v1f, v16s, v16f, v32s, v32f)

| | before (08) | after (09) |
|---|---|---|
| build | `08_.../build.py` = era-0 `k72s_submit.py` (md5 614bc266, 875,212 B) | `build.py` = era-1 `runs/k80.py` (md5 32ab55f1, 880,642 B) |
| judge | 2.8832 (okc check c4f2a37a), 2.9033 (scored cea25010). Era 1's own reading of the same k72 code: 2.8620 (okc check 8cfd4f01) | 2.9036 (okc check 7fa0b21c), **3.0182** (scored 667297c3). Real level **2.9609** (mean). The 3.018 contains a lucky d32f reading worth about 1.8 %; see Result |
| real effect | | local same-run A/B k73 → k80: estimated board time 1303.7 → 1290.9 µs, **−1.0 %** (≈ **+1.0 %** score) |
| shapes changed | | d1s, d1f, v1s, v1f, v16s, v16f, v32s, v32f. The d16s, d16f, d32s and d32f kernels are untouched |

Shape names: d = decode (q_len 1), v = verify (q_len 8); 1 / 16 / 32 = batch; s / f = sliding / full layer.

**Era 1** (`opt-muse-decode/`, 2026-10-30 evening to 10-31 morning PDT) was a fresh start from the k72 code. One coordinator and four worker lanes shared one H200 sandbox. Each lane owned one module of the file:

- `base`: the base `SRC` kernels for d32s, v16s and v32s;
- `xbg`: the `XB1` (d1f) and `XG5` (v1f) modules;
- `xd`: the `XD` module (v16f, v32f);
- `xe`: the `XE` module (d1s, v1s, d16s).

`kernel_fn` is a chain of overrides (XG5 → XB1 → XD → XE → base); each module has its own CUDA string, compiled by NVRTC at import (`notes/coordinator_NOTES.md`). Every lane delivered its wins as `#define` defaults, each with a same-run A/B. The coordinator merged them one by one into k74 … k80. This milestone is that merge.

**Why `change.diff` is large.** It holds 13,257 lines, almost all of them comment-stripping differences.
- The 08 build is era 0's comment-stripped k72.
- Era 1 restarted from the unstripped k72 (`opt-muse-decode/k72.py` = `runs/k72_base.py`, md5 b2c5a13f, 1,067,483 B). The judge rejected that file for exceeding the 1,000,000-byte cap (`runs/check_k72_base.json`).
- Era 1 then stripped it again (`runs/k72s.py`, md5 b74b9b66, okc check 8cfd4f01).
- From that, era 1 made `runs/k73.py` (md5 e0f34916, 866,318 B): comments stripped and the dead `XA` module removed. XA was an older d1f module, already shadowed by XB1. k73 passes 16/16 at the same speed.

The meaningful change is `evidence/k73_to_k80.diff` (502 lines).

## 1. Starting point

k72 per-case judge times, "yours" (µs), with three readings of the same code. The era-0 file and era 1's `runs/k72s.py` are two comment strips of k72; with comments removed they are identical (checked by diffing comment-normalized copies). The value of 1 µs saved is the geomean gain per µs, computed from the k72s run (`notes/coordinator_NOTES.md`). The local raw time is k73 in the k73 → k80 A/B (`evidence/ab_k73_k80.txt`) and includes about 3.6-4.9 µs of graph launch.

| shape | kernel (module) | c4f2a37a / cea25010 / 8cfd4f01 | SGLang | value of 1 µs | local raw k73 |
|---|---|---|---|---|---|
| d1s | `xe_r1_1` (XE) | 3.69 / 3.73 / 4.11 | 20.4-21.0 | 2.13 % | 7.51 |
| v1s | `xe_b1s_8c6` (XE) | 4.78 / 4.36 / 4.58 | 22.5-23.0 | 1.87 % | 8.14 |
| d1f | `xb_d1f` (XB1) | 10.99 / 10.84 / 10.67 | 30.8-32.0 | 0.74 % | 13.98 |
| v1f | `xg_v8f` (XG5) | 12.49 / 12.46 / 12.42 | 35.3-36.2 | 0.65 % | 15.88 |
| d16s | `xe_dm_8_4_2` (XE) | 6.55 / 6.48 / 6.45 | 26.4-26.6 | 0.45 % | 10.10 |
| v16s | `attn_8_1_1_1x6` (base run8w, cluster of 6) | 9.67 / 9.69 / 9.75 | 29.1-29.2 | 0.39 % | 12.80 |
| d32s | `attn_1_1_1_1e3` (base runE1, cluster of 3) | 10.39 / 10.51 / 10.63 | 31.4-31.6 | 0.245 % | 13.58 |
| v32s | `attn_8_1_1_1w4` (base run8w, global combine) | 13.12 / 13.43 / 13.15 | 34.2-34.6 | 0.227 % | 16.54 |
| d16f | `attn_1_0_1_0g` (base run1f) | 87.04 / 86.68 / 86.92 | 200.3-201.4 | 0.158 % | 91.04 |
| v16f | `attn_xd_v8f` (XD build) | 93.04 / 93.68 / 93.40 | 204.4-205.5 | 0.137 % | 96.47 |
| d32f | `attn_1_0_1_0` (base run1f) | 165.03 / 165.61 / 165.39 | 383.0-384.4 | 0.085 % | 169.82 |
| v32f | `attn_xd_v8f` (XC build) | 174.42 / 173.59 / 173.00 | 388.9-390.4 | 0.079 % | 177.46 |

Sources: `evidence/judge_before_k72s_check_c4f2a37a.txt`, `judge_before_k72s_scored_cea25010.txt`, `judge_k72s_era1_check_8cfd4f01.txt`. The B=1 shapes carry most of the score per µs.

## 2. What we saw

**(a) Floors and fixed costs** (coordinator microbenchmarks `evidence/mb_launch.py` and `evidence/mb_gather.py`; numbers from `notes/coordinator_NOTES.md`, "Measured facts"):
- Graph replay of an empty 1×32 kernel costs ~4.45 µs. 132×512 CTAs cost 4.9 µs. A cluster or more than 32 KB of dynamic smem adds ~0.4 µs (bimodal). The judge subtracts ~3.6-4.2 µs.
- One dependent cold DRAM hop plus a write costs **+1.2 µs**. Each further dependent hop adds 0.4-0.5 µs.
- Pure page-table gather of all K+V rows, no math: d32f 163.5 µs (4.04 TB/s), d16f 86.2, d1f 11.3. A sequential read of the same bytes: 147 / 77.6 / ~10.
- **Conclusions:**
  - d16f and d32f already sit at the random-gather ceiling.
  - d1f and v1f are ~2.5 µs above it; v16f and v32f are 6-8 µs above it.
  - So in every other shape, the excess is dependent latency: the start chain, the per-tile chain and the end (combine) chain.

**(b) d1f / v1f** (`xbg`; per-CTA `%globaltimer` traces with `evidence/xbg_trace.py`; `notes/xbg_LOG.md`)
- **d1f timeline:**
  - The page table arrives at ~0.7 µs.
  - The K/V burst is fully issued at 2.8 µs median, 3.9 µs max. cp.async back-pressure holds ~73 KB in flight per SM.
  - Tile data lands at 3.1-6.2 µs; the loop ends at 7.9 µs max.
  - Each lane's loop ends ~1.4 µs after its last tile landed: the per-tile chain (~920 cycles; ~1600 for the first tile) is latency-bound.
  - Before the page-table loads, every CTA computed its slice start as a runtime u32 division by `G = p.G`.
- **v1f, the trigger for XG_CLE:**
  - The new block's `cache_loc` load was issued after the burst and waited **~1.9 µs behind ~20 MB of queued K/V requests**.
  - So warpgroup 0 of CTA 131 started step 0 1.9 µs late, and CTA 131 was always the last publisher.
- **Kernel start in general:** the start is fetch-bound after the judge's L2 flush. In `xd`'s ncu warp sampling, the first constant-bank `ULDC.64 c[0x0][0x208]` drew 463 `imc` samples, ~40 % of the samples in the first 0x1860 bytes of SASS (`notes/xd_LOG3.md`). Any dependent load before the first page-table load is on the critical path.
- **Controls that ruled out other fixes:**
  - Code first executed after the burst is fetched from DRAM behind the burst queue: a dry tile chain started at 0.8 µs took ~5 µs. Warm-up passes therefore lose: `XB1_HWARM` +3.3 µs, producer warps `XB1_PROD` +1.7.
  - `xg_v8f`'s wgmma serialization (ptxas C7511) comes from the 36-register m64n72 PV accumulator. Un-serializing it measured +0.13 µs with unchanged step times. Large ALU cuts buy little: no V^T −0.22, no clamp/packing −0.36. The steps are issue/latency bound.

**(c) d1s / v1s** (`xe`; `notes/xe_LOG.md`)
- **d1s timeline** (median CTA, ns): K/V issued 640 → K lands 1408 → softmax 1648 → PV 1760 → CTA barrier 1888 → combine and push 2080 (max 2272) → owner mbarrier complete 2432 → merge 2688 → store issued 2880.
- **Ablations:** the owner merge + norm costs 0.36-0.37 µs of critical path; math 0.48; CTA combine 0.14. On v1s: math 0.46, owner 0.25.
- **The merge's weights:** each owner lane computed all 16 source weights itself (16 `ex2` per lane).
- **v1s issue cost:** in the register-load path, ptxas spent ~200 instructions per lane issuing the K/V loads (comment on `XE_CPA` in the diff).
- **Controls:**
  - FP8 `mma.sync` is emulated on sm_90 (F2FP + 2 HMMA), so it gives no gain.
  - Online owner merges: +0.5 to +1.2 µs.
  - v1s clusters of 5, 8 or 16: +0.21 to +0.64.
  - The lane-parallel merge on v1s's 6-source owner: +0.03 (dropped).
  - cp.async staging on d16s: +0.23. It pays only where the issue phase is the bottleneck.

**(d) v16s / v32s** (`base` and `xbg` round 5; `notes/base_LOG.md`, `base_LOG2.md`, `xbg_LOG5.md`)
- **Ablations on k73** (timing only):
  - v32s 17.20: no cross-CTA combine **−2.86**, no tile math −1.26.
  - v16s 13.70: no cross-CTA combine −1.31, no tile math −1.55.
- **The q_len-8 tile loop (`w8_tile`):**
  - ~1265 cycles per 64-token tile, with only ~100 cycles waiting for data once started: compute-bound from the first landed tile.
  - ncu warp sampling: mbarrier try-wait 17 %, LDSM→PRMT (the V^T transpose) 11 %, `fence.proxy.async` (MEMBAR.ALL.CTA) 7 %, PV DEPBAR 7.5 %.
- **v32s global combine** (trace, `base_LOG2.md`): 4 CTAs per (request, head pair); owners poll the other 3 CTAs' FP16 partial rows.

  | event (median / p90, µs from first CTA start) | |
  |---|---|
  | owner stream merge done | 8.35 / 8.96 |
  | last of the 3 sources published | 9.02 / 9.47 |
  | owner folded − max(last publish, own merge) = L2 propagation + detection + loads | **1.76 / 2.30** |
  | poll rounds | 2 (max 4) |
  | kernel end (last CTA) | 12.50 |

  A cross-CTA visibility microbenchmark (`evidence/base_mb_vis.py`, 128 CTAs, `ld.relaxed.gpu` polls) measured ~1.5 µs from store to observation, with rounds of ~0.65 µs. **The protocol's cost is L2 round trips, so the rounds must overlap.**
- **Hot-tile SASS** (`xbg_LOG5.md`): ~170-200 instructions per warp per tile. A boundary tile's mask block alone is 128 SASS instructions. The mask + max + bias phase costs 349-357 cycles at the median, but 312 / 581 at p10 / p90. Masked tiles cost ~250 cycles more, and each stream's last (masked) tile is on the critical path.
- **Controls:**
  - Lazy rescale `W8_LZ`: v16s +0.3.
  - Last-arrival finisher `BZ_LA`: v32s +1.78.
  - Token re-split `BZ_SPL`: +0.17 to +0.32.
  - v16s's DSMEM pull combine is bandwidth-bound at ~12-14 B/clk per SM (~14.5 KB per CTA, 0.6-0.9 µs), so it was left alone.

**(e) v16f / v32f** (`xd`; `notes/xd_LOG.md`, `xd_LOG3.md`)
- **Trace:** the first K/V issue is at ~4.0 µs and the first pair lands at ~6 µs. Ring ends span 78-89 µs (median 85.8). Steady consumption is ~4.15 TB/s, the random-gather rate.
- **ncu:** DRAM 73.8 %, L2 82 %, issue 0.47 per scheduler. The kernel sits at 126/128 registers.
- **End phase, ~4.6 µs:** merge + partial write 0.8, publish (fence / atomic / fence) 1.7, combine (wait + loads + epilogue) 2.1.
  - In the combine, each thread issued 27 loads for P = 9 partials: m, l and O for each partial.
  - The loads of a round land 1130-1160 cycles after issue.
  - Cold code costs only ~0.35 µs of the combine.
- **The split:**
  - The FLAT split charges `OVH` virtual units at each request start for the restart cost.
  - CTAs whose range holds a request boundary finished ~4 µs early: OVH = 1024 overcharges a restart.
  - Lowering OVH hurt v16f (+1 µs) but helped v32f (−0.5 to −1.2).
- **Controls:**
  - One `atom.add.acq_rel` instead of fences (`XD_AR`): no gain.
  - Combiner-count sweep: the default stays.
  - A dry warm-up of the combine needs a non-inlined call: +19 µs (STACK 328 at the 128-register cliff).

**(f) d16f / d32f were left alone** (`notes/xe_LOG2.md`, `xe_LOG3.md`).
- An equal-cost static split removed the systematic spread (B32 max − mean 3.73 → 2.68 µs) but cost +0.87 / +0.93 µs.
- The fitted marginal cost of a token (5.6 ns) is two-thirds of the average per-CTA time per token (8.4 ns). With HBM saturated, an early finisher's bandwidth goes to the others, and the max is set by run-to-run noise.
- The verify full kernels behave the other way: there, overcharging the restart cost did cost time (the v32f OVH result above).

**(g) How small effects were measured** (`opt-muse-decode/tools/bench.py`, `tools/ab.py`)
- **bench.py** is the local judge-alike: one CUDA graph per shape, replayed after a cold-L2 read of 512 MB; median of reps; an estimated board time from the k72s SGLang times (`evidence/sgl_times.json`).
- **ab.py** was taken from `xe`'s `work/xe/ab.py`:
  - every candidate file is loaded as its own module;
  - for every instance (seed × capture), fresh graphs are captured for each candidate;
  - the reps alternate flush + replay in rotated order;
  - it reports the mean ± stderr of the per-instance delta.
- **Why so many instances:** each graph or module instance carries a sticky offset of ~0.1-0.3 µs (`xe_LOG.md`: ±0.3 µs, split halves agree, not caused by placement). An A/A over 20 instances read d1s −0.001 ± 0.066 µs. A graph replayed twice in a row is 0.1-0.2 µs faster (`xbg_LOG.md`). Single bench.py runs swing the estimated board by ±10 µs (coordinator notes).
- **Only same-run deltas count:** k73 read 1311.8 µs in one run and 1303.7 µs in the next (`evidence/ab_k73_k79.txt` vs `ab_k73_k80.txt`).
- This is the origin of the same-run multi-instance A/B used in every later milestone: `tools/ab.py` in this task is the same script with paths changed.

## 3. Diagnosis and projection

- **No bandwidth lever was left.** Big full shapes run at the random-gather rate, so their remaining excess is start, imbalance and end phase. Small shapes are short dependent chains.
- **Every win had to remove a step from a dependent chain:**
  - a load or division before the first page-table load (d1f), or a cold load queued behind the burst (v1f);
  - serial work in a combine: 16 `ex2` per lane (d1s), 27 loads per thread (v16f), polling rounds that do not overlap (v32s);
  - instructions in the masked tile at the end of each stream (v16s, v32s, v1f);
  - issue overhead (v1s);
  - an overcharged restart cost in the split (v32f).
- **Ceiling (inferred):** d1f and v1f sat ~2.5 µs above their gather floor each. Closing both fully would be worth ~3.5 % (2.5 × 0.74 + 2.5 × 0.65); no single piece came near that.
- **Projection (inferred):** applying the value-per-µs table to each lane's own A/B gives **+0.98 %**: d1s −0.05, v1s −0.07, d1f −0.16, v1f −0.36, v16s −0.23, v32s −0.70, v16f −0.33, v32f −0.87 µs.
- **Plan:** judge noise (±1.5-2 %) exceeds this, as the coordinator notes state. So each piece was banked behind a define, and the stack was verified once against k73 in one A/B run before judging.

## 4. Change

Each piece is a `#define` default or a Python constant, all present in `evidence/k73_to_k80.diff`. Local effects are each lane's own same-run A/B.

| piece | lane | shape (kernel) | mechanism | local effect (µs) | source |
|---|---|---|---|---|---|
| `XB1_GCONST` 0 → 1 | xbg r1 | d1f (`xb_d1f`) | grid size G compiled in (host passes `-DXB1_GC`): no runtime u32 division and no `p.G` constant-bank load before the page-table loads | −0.16 (full file 13.90 → 13.73; 7/7 slim rounds) | `notes/xbg_LOG.md` |
| `XG_CLE` 0 → 1 | xbg r1 | v1f (`xg_v8f`) | the new block's `cache_loc` row is loaded in the prologue (`ld.global.nc.b64`), not after the burst | −0.2 (15.84 → 15.62) | `notes/xbg_LOG.md` |
| `XE_OLP` | xe r1 | d1s (`xe_r1_1`) | owner merge: lane s computes source s's weight once (`ex2`), shuffled to all lanes; two FFMA chains | −0.056 ± 0.038 (40 inst), −0.041 ± 0.029 (64) | `notes/xe_LOG.md` |
| `XE_CPA` | xe r1 | v1s (`xe_b1s_8c6`) | K/V staged by `cp.async` (4 lanes × 16 B = the head's 64-B slice of each key row, one shuffle per 8 keys, K group then V group); fragments read by LDS | −0.089 ± 0.014 | `notes/xe_LOG.md` |
| `_XC_OVH = 256` | xd r1 | v32f (`attn_xd_v8f`, XC build, B ≥ 22) | restart charge per request start 1024 → 256 units; the XD build (v16f) keeps 1024 | −0.87 mean (seeds 7-10: −0.98 / −0.72 / −1.31 / −0.48) | `notes/xd_LOG.md` |
| `BZ_FL` | base r1 | v16s, v32s (`w8_tile`) | `fence.proxy.async` for the V^T stores moved from before the QK wait to right before the PV barrier (same ordering) | −0.07 / −0.08 (6 seeds × 30 reps) | `notes/base_LOG.md` |
| `XD_CMB2` | xd r3 | v16f (XD build) | combine: lane u of a row's 16 lanes loads partial u's (m, l) and computes its weight; O chunks weighted by shuffle: 2 + P loads and 1 `ex2` per thread instead of 3P and P | −0.33 mean (−0.54 / −0.14 / −0.18 / −0.46) | `notes/xd_LOG3.md` |
| `W8_HM` | xbg r5 | v16s, v32s (`w8_tile`) | boundary-tile mask with `set.lt/ge.u32.f16x2`; masked halves OR'ed to NaN (0xffff), which `max.f16x2` and the clamp drop; mask block 128 → 69 SASS instructions | v16s −0.16, v32s −0.12 (7 rounds) | `notes/xbg_LOG5.md` |
| `XG_HM` | xbg r5 | v1f (`xg_v8f`) | the same NaN-OR mask in the masked step | −0.17 (15.53 → 15.37) | `notes/xbg_LOG5.md` |
| `BZ_PP`, `BZ_PPD 160` | base r2 | v32s (`attn_8_1_1_1w4`) | owner warps keep two full-row poll batches in flight, staggered by 160 ns; gate sigmoids computed while the first batch flies; slot resets issued before the epilogue | −0.46 / −0.48 / −0.44 (three runs); −0.51 vs k77 (full files) | `notes/base_LOG2.md` |

Integration order (md5 of each file in `opt-muse-decode/runs/`):

| file | md5 | adds | judged |
|---|---|---|---|
| k73 | e0f34916 | k72 stripped, XA removed | |
| k74 | 67fa58cd | XB1_GCONST, XG_CLE (= `work/xbg/cand.py`) | |
| k75 | e617b733 | XE_OLP, XE_CPA | |
| k76 | 51346e93 | `_XC_OVH` 256 (+ an `XD_PTRACE` hook, compiled out) | |
| k77 | 0c73ca8f | BZ_FL | okc check a37ea66b: 2.9367 |
| k78 | 6123eabe | XD_CMB2 (+ `XD_AR` option, off) (= `work/xd/cand3.py`) | |
| k79 | 2a719c94 | W8_HM, XG_HM | A/B vs k73: estimated board 1311.8 → 1299.5 µs |
| k80 | 32ab55f1 | BZ_PP, BZ_PPD | okc check 7fa0b21c 2.9036; scored 667297c3 3.0182 |

**v1f: XG_CLE + XG_HM (−0.32 µs, the largest score share, 0.21 %).**
- In `xg_v8f`, warpgroup 0 of the last CTA (`lastc`) loads the new block's K and V from `qkv` and writes them into the cache.
- Before XG_CLE, it read the cache row index after the burst. The load then queued behind ~20 MB of K/V requests and arrived 1.9 µs late, which made CTA 131 the last publisher every time.
- Issuing it next to the K/V loads of the new token, in the prologue, removes that wait.
- XG_HM replaces 32 compare/select pairs in the single masked step with one `set.ge.u32.f16x2` per word, OR'ed into the score halves as NaN.

**v32s: BZ_PP (+ W8_HM, BZ_FL), −0.79 µs, 0.18 %.**
- In the k73 owner poll, round 1 probed one word per source and later rounds re-read full rows (with a 64 ns nanosleep). One visibility round costs ~0.65 µs (microbenchmark), and the owner needed 2 rounds at the median, 4 at most.
- BZ_PP issues full-row loads for all 3 sources twice, 160 ns apart. It consumes batch A while batch B is in flight, then re-issues A, and so on; a source's row counts once its sentinel words are gone.
- The 16 gate sigmoids move under the first batch, and the slot resets go out before the output stores so that they drain under the epilogue.
- Stagger 0 / 160 / 320 ns measured −0.45 / −0.46 / −0.50 µs.

**d1f: XB1_GCONST, −0.20 µs, 0.16 %.**
- `cstart(c) = min(n, c·(n + LD) / G)` sits between kernel entry and the first page-table load.
- With G as a runtime parameter it costs a constant-bank load (cold after the flush) plus a u32 division. Compiled in, the division by a constant becomes a multiply-shift (inferred), and the whole d1f pipeline starts earlier.

**v1s / d1s: XE_CPA, XE_OLP (−0.07 / −0.04 µs, 0.13 % / 0.10 %).**
- On B=1 sliding shapes a 0.05 µs gain is worth 0.1 %.
- XE_CPA moves each warp's K/V loads (the head's 64-B slice of each 128-B row, 16 B per lane) to `cp.async`, zero-filled when masked. This cuts the issue work of each compute warp; the fragments are then read from shared memory.
- XE_OLP turns the owner's 16 `ex2` per lane into one per lane plus shuffles.

## 5. Result

| check | result |
|---|---|
| local A/B, same run, k73 → k80 (`evidence/ab_k73_k80.txt`, 12 instances per shape) | v32s **−0.79 ± 0.04**, v32f −0.71 ± 0.12, v1f **−0.32 ± 0.03**, v16s −0.25 ± 0.04, v16f −0.24 ± 0.07, d1f **−0.20 ± 0.05**, v1s −0.07 ± 0.05, d1s −0.04 ± 0.05. Estimated board 1303.7 → 1290.9 µs (speedup 2.8890 → 2.9177, **+0.99 %**) |
| controls (code untouched) | d16s −0.01 ± 0.03, d16f −0.04 ± 0.05, d32s −0.08 ± 0.04, **d32f +0.15 ± 0.05**: the module-instance offset on large shapes |
| BZ_PP in the merged file | k73 → k79 (no BZ_PP) v32s −0.36 ± 0.04; k73 → k80 −0.79 (`evidence/ab_k73_k79.txt`) |
| local kernel times (nsys, `evidence/nsys_k80_kernel_summary.txt`, median of 30) | d1s 3.18, v1s 3.82, d1f 9.98, v1f 11.55, d16s 6.00, v16s 8.69, d32s 9.34, v32s 11.74, d16f 87.49, v16f 92.91, d32f 164.96, v32f 173.06 µs |
| correctness | Judge: 16/16 in both k80 runs. Per piece (lane deliverables): xbg `nearby.py` 56/56 (k74 and W8_HM / XG_HM); xe `robust.py` 145 XE cases + 96/96 speculation-miss cases; base verify-sliding sweeps over B 2-33, run twice to check that the slot sentinels are restored; xd q_len-8 full at B 1-128. W8_HM / XG_HM output is bitwise equal on v16s and v1f. No nearby sweep of the merged k80 itself is recorded |

Judge per case, "yours" (µs). k72 mean = the three readings in section 1; k80 mean = 7fa0b21c and 667297c3.

| shape | k72 mean | k77 check a37ea66b | k80 check 7fa0b21c | k80 scored 667297c3 | k80 mean − k72 mean | local Δ |
|---|---|---|---|---|---|---|
| d1s | 3.84 | 3.76 | 3.72 | 3.43 | −0.27 | −0.04 |
| d1f | 10.83 | 10.22 | 10.37 | 10.25 | −0.52 | −0.20 |
| d16s | 6.49 | 6.63 | 6.59 | 6.81 | +0.21 | −0.01 |
| d16f | 86.88 | 86.80 | 86.70 | 86.80 | −0.13 | −0.04 |
| d32s | 10.51 | 10.07 | 10.65 | 10.21 | −0.08 | −0.08 |
| d32f | 165.34 | 164.47 | 165.21 | **144.81** | −10.33 | +0.15 |
| v1s | 4.57 | 4.37 | 4.73 | 4.37 | −0.02 | −0.07 |
| v1f | 12.46 | 11.92 | 11.77 | 12.23 | −0.46 | −0.32 |
| v16s | 9.70 | 9.43 | 9.45 | 9.23 | −0.36 | −0.25 |
| v16f | 93.37 | 94.22 | 93.49 | 93.38 | +0.06 | −0.24 |
| v32s | 13.23 | 12.82 | 12.84 | 12.13 | −0.75 | −0.79 |
| v32f | 173.67 | 173.02 | 172.79 | 172.37 | −1.09 | −0.71 |
| **speedup** | 2.8828 | 2.9367 | 2.9036 | 3.0182 | | |

**Reading the judge honestly:**
- **Same code, different readings.** d16s, d16f and d32s are code no piece touched; they moved −0.13 to +0.21 µs between the means. The three k72 readings of d1s span 3.69-4.11. Small-shape readings swing by 0.3-0.4 µs, more than most single pieces.
- **The scored run's d32f (144.81 µs) is not ours.**
  - Our own 60 raw reps were unchanged: 164.4-173.0 µs, median 168.85 (check run: 164.6-173.9, median 168.96). The judge's own kernel profile shows 169.6 µs.
  - SGLang's d32f reps are bimodal: 381-392 µs or 509-525 µs. 18 of 60 were slow in the scored run, against 11 of 60 in the check (and 11 in the k77 check).
  - The judge's `paired_ratio` is the median over 30 pairs of the geometric mean of the two reps' ratios. 16 of the 30 pairs held a slow SGLang rep, so the median pair was a slow one: 2.659, against a ratio of medians of 2.334. In the check, 11 of 30 pairs held a slow rep and the median was 2.319.
  - Source: `evidence/judge_after_k80_*.json`, fields `ratios`, `baseline_runs_ns`, `paired_ratio`, `speedup_of_medians`.
  - Re-scoring the scored run with d32f = SGLang median / 2.334 = 164.9 µs gives **2.964** (≈ 1271 µs, as the coordinator notes say).
- **Levels.** The real level by the release rule is the mean of the two readings, **2.9609**. With d32f corrected the two readings are 2.904 and 2.964 (mean 2.934). Against the k72 mean of 2.8828 that is +1.8 %; against the era-0 mean of 2.8933, +1.4 %.
- **vs projection.** The local A/B (+0.99 %) matches the projection from the lanes' own numbers (+0.98 %). The pieces add up and do not interfere. The judge's extra +0.4 to +0.8 % (after the d32f correction) is node noise: the judge deltas on d1s (−0.27) and d1f (−0.52) are 3-7× the local ones (−0.04, −0.20), and the code-untouched d16s moved +0.21. **The real effect of this milestone is the local +1.0 %.**

## 6. Takeaways, and what came next

- **On short shapes, the critical path is a chain of dependent loads.**
  - A single load issued in the wrong place cost 1.9 µs on one CTA (XG_CLE).
  - A parameter read plus a division before the first page-table load cost 0.16-0.2 µs (XB1_GCONST).
  - Find these with per-CTA `%globaltimer` timelines, not with ncu averages.
- **Cross-CTA combines are bound by L2 round trips** (~0.65 µs per visibility round, ~1 µs per dependent global hop under load). Keep the rounds overlapped (BZ_PP), and cut loads per thread (XD_CMB2) rather than bytes.
- **Many small, independent pieces add up linearly** (projection +0.98 %, measured +0.99 %). But the judge's ±1.5-2 % node noise hides the whole stack, so each piece has to be proven locally with a multi-instance same-run A/B.
- **Negative rounds after k80 closed several lines** (`notes/*_LOG*.md`). The lanes found nothing more to merge, and k80 stayed the best era-1 file.
  - Warp specialization: v16f / v32f +3 / +5 µs; v16s / v32s +2-3 µs.
  - Dynamic tail queues: +6 to +35 µs at 126/128 registers.
  - Equal-cost splits for d16f / d32f: +0.9 µs.
  - Two-tile softmax steps `w8_pair`: +0.8 to +2 µs, despite −25 % per-tile cost.
  - Warpgroup ping-pong: +0.1 to +0.35.
  - A fixed-reference red accumulator for v16f / v32f: +0.3 to +1 µs.
  - A query-row split for v16s / v32s: fails a feasibility gate, since one CTA gathers random rows at only ~60-65 GB/s.
  - TMA multicast of 128-B rows: 43-94 µs.
- **Next:** `10_*` keeps trimming the B=1 chains on top of k80: `XE_FR`, a v1f end-chain candidate and `XB1_BAT=2`.

## Reproduce

```bash
cd muse_attention_decode/tools
python bench.py ../09_start_combine_mask_fixes_stack/build.py --tests       # 16 tests + local judge-alike timing
python ab.py ../08_*/build.py ../09_start_combine_mask_fixes_stack/build.py \
       --shapes d1s,d1f,v1s,v1f,v16s,v16f,v32s,v32f,d32s --seeds 7,8,9 --caps 4   # same-run A/B; d32s = untouched control
# with all 12 shapes, ab.py also prints the estimated board time if sgl_times.json (copy in evidence/) sits beside it
PYTHONPATH=. python ../09_start_combine_mask_fixes_stack/evidence/mb_gather.py d32f   # pure-gather floor (uses evidence/nvk.py)
PYTHONPATH=. python ../09_start_combine_mask_fixes_stack/evidence/mb_launch.py         # graph-replay and dependent-hop costs
PYTHONPATH=. python ../09_start_combine_mask_fixes_stack/evidence/xbg_trace.py \
       ../09_start_combine_mask_fixes_stack/build.py v1f xg                             # per-CTA timeline of xg_v8f (XG_TRACE build)
python3 -c "import json,statistics as s; b=json.load(open('../09_start_combine_mask_fixes_stack/evidence/judge_after_k80_scored_667297c3.json'))['benchmarks'][5]; \
print(sum(x>450e3 for x in b['baseline_runs_ns']), s.median(b['runs_ns'])/1e3, b['paired_ratio'], b['speedup_of_medians'])"   # d32f: 18 slow SGLang reps
```

`base_mb_vis.py` and `xbg_trace.py` were written for the era-1 sandbox (`/work/tools` on `sys.path`); with `PYTHONPATH=.` they fall back to `tools/`.

Originals:
- `notes/coordinator_NOTES.md` = `opt-muse-decode/NOTES.md`; `notes/<lane>_LOG*.md` = `opt-muse-decode/work/<lane>/LOG*.md`.
- Builds: `opt-muse-decode/runs/k73.py` … `k80.py`.
- Judge results: `opt-muse-decode/runs/profiles/{k72s,k77,k80}.*`, plus the era-0 `../autoresearch-kernels/attn/subs/profiles/k72s_{check,submit}.*`.
- A/B: `opt-muse-decode/runs/ab_k73_k{79,80}.txt`. nsys: `opt-muse-decode/runs/profiles/nsys/` (only the text summary is copied; the `.nsys-rep` / `.sqlite` reports stay there).
- Microbenchmarks: `opt-muse-decode/tools/`, `work/xbg/trace.py`, `work/base/t/mb_vis.py`.
- Previous build: `../autoresearch-kernels/attn/subs/k72s_submit.py`.
