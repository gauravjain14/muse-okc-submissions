# 01: one fused FP8 split-KV kernel per call, FLAT token split over all CTAs (all 12 shapes)

Shape names: d / v = decode (q_len 1) / verify (q_len 8); 1 / 16 / 32 = batch; s / f = sliding (2,048-token window) / full (~80K context). E0 = the era-0 lab tree, `../autoresearch-kernels/attn/` relative to `Compilers/`.

| | before (SGLang baseline) | after (01) |
|---|---|---|
| build | `../tools/baseline.py`: SGLang's chain of 6-7 kernels around FA3 (md5 07906528, 6,178 B) | `build.py` = era-0 `k3` (`E0/subs/k3_check.py`, md5 0937facd, 33,204 B) |
| judge | 0.9969 (SGLang submitted against itself, okc check e9c98045, CI 0.982-1.012) | **1.2010** (okc check 4a7aa2f3, CI 1.183-1.219, 3,136.1 µs). One reading, no scored run |
| real effect | | 1.2010 (single reading). Local judge-like bench projected x1.2053 (−0.4 % vs the judge). Decode loads 1.59-1.90x, verify loads 0.71-1.05x |
| shapes changed | | all 12 (new kernel). Decode 1.5-2.0x faster; v16f and v32f **1.55x slower** than SGLang |

## 1. Starting point

SGLang runs QK-norm (Triton), the cache store, an FP8 cast of q, FA3's prepare kernel (sliding and verify only), the FA3 kernel, FA3's combine kernel and `rms_norm_gate`: 6-7 dependent kernels per call (`notes/research.md` §1.1; `evidence/judge_before_calib_sglang_e9c98045.txt`, kernel table per shape).

| shape | SGLang, judge (µs) | SGLang GPU time: total / FA3 main / FA3 combine (µs) | floor (µs) |
|---|---|---|---|
| d1s | 20.38 | 18.2 / 8.2 / 2.5 | 0.11 |
| d1f | 31.58 | 27.4 / 14.8 / 6.9 | 4.27 |
| d16s | 26.98 | 22.2 / 9.9 / 3.1 | 1.65 |
| d16f | 200.91 | 198.4 / 186.8 / 3.1 | 63.18 |
| d32s | 31.89 | 25.8 / 13.0 / 3.1 | 3.29 |
| d32f | 384.75 | 385.5 / 373.2 / 3.5 | 126.35 |
| v1s | 23.09 | 21.0 / 8.4 / 4.0 | 0.12 |
| v1f | 35.90 | 32.1 / 15.0 / 7.4 | 4.28 |
| v16s | 28.99 | 24.7 / 10.8 / 3.8 | 1.86 |
| v16f | 205.42 | 205.2 / 187.7 / 4.3 | 63.39 |
| v32s | 34.72 | 30.4 / 14.8 / 4.4 | 3.72 |
| v32f | 390.82 | 393.8 / 375.5 / 3.9 | 126.78 |

"SGLang, judge" is the SGLang column of the k3 run (`evidence/judge_after_k3_check_4a7aa2f3.txt`), which is what that run's score divides by. GPU times come from the calibration run's profile. The floor is the task's speed-of-light column.

## 2. What we saw

**(a) SGLang's FA3 is far from the memory floor.** Read from the FA3 source (`notes/research.md` §0 items 1, 5, 6 and §1):
- With FP8 and head_dim 64, FA3 fixes a 192 × 160 tile. Packed GQA leaves 8 valid rows (decode) or 64 (verify) of the 192, but softmax runs on all of them.
- Each KV head goes to its own CTA, so every gather is a 64-B half row. Only 2 stages are in flight, and the page table is reloaded synchronously for every 160-key block.
- Result: d32f moves ~666 MB in 373 µs of FA3 time, ≈ 1.8 TB/s (inferred from the profile above; research.md estimated 1.7 TB/s). Verify costs the same as decode (390.8 vs 384.8 µs judge).

**(b) The gather floor.** `evidence/probe_gather.py` is a no-math kernel: page table → random 128-B K and V rows, cold L2, CUDA graph. Results (`notes/PLAN.md` "Measured"):

| K+V bytes | shape it stands for | µs | TB/s |
|---|---|---|---|
| 655 MB | b32 full | 152 | 4.30 |
| 328 MB | b16 full | 76.4 | 4.29 |
| 20.5 MB | b1 full | 5.44 | 3.8 |
| 16.8 MB | b32 sliding | 4.38 | 3.8 |

- Reaching this needs ≥ 16 warps per SM with ≥ 8 KB in flight per warp (a cp.async ring or LDG).
- A later, more faithful launch-shaped probe put 20.5 MB at 5.98 µs. PLAN's 5.44 came from a fast container (`notes/WORKERS.md`, "GATHER FLOORS, faithful").

**(c) Where the score is** (`notes/PLAN.md` "Where the score is"):
- **b1 loads:** 65 % of SGLang's time is the 46 sliding calls at ~20 µs. Every 10 % on b1 sliding is worth ≈ 1.5 % of the score.
- **b16/b32 loads:** dominated by the 16 full calls. These are memory-bound (floors 76 / 152 µs vs SGLang 198 / 384), which caps those loads at ~2.6-2.8x.
- **v32f:** 84 GFLOP and 328 M exps per call. With mma.sync, compute is ~55-85 % of the memory time, so it needs good overlap or wgmma.
- **Ceiling:** every shape at its floor + 1-2 µs would give ≈ 3.6x.

**(d) PTX/SASS facts checked before writing code** (`notes/research.md` §0, verified with offline SASS; nothing run on a GPU):
- FP8 `mma.sync.m16n8k32.e4m3` is **emulated** on sm_90a: each one becomes 12× `F2FP.F16.E4M3.UNPACK_B` + 2× `HMMA.16816.F32`. `wgmma ... m64n8k32 .e4m3` becomes a native `QGMMA`.
- Small shapes are a chain of three dependent cold-DRAM trips, seq → page_table → K/V, at ~0.7-1 µs each.

**(e) The prototypes k1 and k2 measured the structure before k3** (local bench, same job as SGLang; `evidence/bench_k1.txt`, `bench_k2.txt`).
- **The k2 no-compute skeleton** (`ABL_NOCOMPUTE`: same loads, no math) ran x1.978 against x1.236 for the real k2:

| shape | k2 µs | k2 no-compute µs | compute share |
|---|---|---|---|
| d1s | 10.57 | 8.88 | 1.7 |
| d32f | 189.95 | 173.81 | 16.1 |
| v16f | 330.09 | 124.44 | 205.7 |
| v32f | 638.60 | 231.65 | 406.9 |

  So q_len 1 was already close to its skeleton. On verify full, the MMA/softmax tile is 64 % of the time.
- **k2 phase traces** (`evidence/trace.py` with globaltimer stamps; `evidence/trace_k2_b1_flush_warm.log`; build inferred as k2, the script's default file at 22:56). d1s timeline in µs, flush mode:
  - seq landed 0.58
  - Q and new-token tile done 3.44
  - tile0 3.62, tiles done 4.55
  - merged 5.60, published 6.02
  - combiner released 8.14, combined 10.45

  The Q/new-token preparation finished 2.9 µs after the length landed; k3 moves those loads to kernel entry (its docstring). The end chain after the tiles (merge → publish → release → combine) is 5.9 µs of the 10.45. With a warm L2 the end is 9.26.
- **ncu on k2** (`evidence/ncu_k2_d1s_*`, `ncu_k2_d1f_*`):
  - d1s (16 CTAs): DRAM 1.0 %, issue slots 2.9 %. Stalls: barrier 30 %, long scoreboard 14 %. The top lines are a `__syncthreads`, two `__threadfence` and the RoPE `sincosf`.
  - d1f (132 CTAs): DRAM 22.6 % (1.08 TB/s), issue 19.6 %. Stalls: barrier 28 %, long scoreboard 27 %.
  - Both kernels: 128 registers, 24 % occupancy.

## 3. Diagnosis and projection

- **Kernel boundaries.** SGLang loses on small shapes to the 6-7 kernel boundaries, and on big shapes to FA3's padded softmax and half-row gathers. One fused launch that gathers whole 128-B rows (both KV heads per CTA) attacks both.
- **Targets** (`notes/PLAN.md` "Where the score is"): ~3 µs sliding and ~7 µs full b1 calls would give ~5x on the b1 loads. The b16/b32 loads are capped at ~2.6-2.8x by the gather floor.
- **No explicit per-build projection was recorded for v1.**
- **The emulated MMA was a known risk.** research.md flagged FP8 `mma.sync` as emulated, and PLAN noted that verify full needs good overlap or wgmma. v1 still used mma.sync m16n8k32 (PLAN "Kernel v1 design"). The native-wgmma path was split off to a worker once k3 was the base (`notes/orchestrator-state.json`, hop "workers").

## 4. Change

One kernel per call (`run<L, ROPE, PAGE1, DST, ALIGNED>`, launched as `attn_{L}_{rope}_{page1}_{aligned}` with 512 threads and 1 CTA per SM). It does QK-norm (+ RoPE) of q and the new k, writes the FP8 cache rows of the new k/v, runs split-KV attention with E4M3 Q/K/P/V and FP32 accumulation, and applies RMSNorm × sigmoid(gate) → bf16.

**Work split** (`plan()` in the Python wrapper):
- **FLAT** (full layers, B > 1): request b contributes its past tokens + 1 "new block" unit. The flattened unit list is split evenly over G = min(132, units/256) CTAs, so lengths that differ by ±7.5 % still balance. Warp 0 builds the prefix of units with a shuffle scan of `seq_lens`. A CTA then walks its request segments in order.
- **ALIGNED** (sliding layers, and B = 1): each request gets cpr = ceil(span/256) CTAs (capped at 132/B), one segment each.

**Lanes and the ring:**
- A lane is a group of warps sharing one 32-token tile.
  - L = 1: 8 lanes of 2 warps (one per KV head).
  - L = 8: 2 lanes of 8 warps.
- Each lane has a DST-stage cp.async ring of 8 KB stages (32 tokens × [K 128 B | V 128 B]); DST = 3 (L = 1) or 6 (L = 8). Lanes take tiles round-robin. There is one named barrier per tile.

**Tile math** (`tile_compute<L>`):
- QK: `mma.sync.m16n8k32.e4m3` (`mma8`) with a k-permutation, so each thread's B fragment is one 16-B shared-memory load.
- PV: the same MMA. P comes from the S accumulators; V is transposed in registers by 4×4 byte `prmt` transposes (`tr4`).
- A 9th "ones" n-tile gives the row sums.
- Softmax uses FA3's FP8 offset: P = 2^(s·c − m + 8).

**Merge and combine:**
- Lanes merge in shared memory.
- A CTA's partial (m, l, O) goes to a global workspace with one `atom.add.acq_rel` arrival per partial request.
- The last K arrivals each combine a slice of rows. When the partials total ≤ 32 KB (`K3_COMB_BYTES`), the last arrival combines alone.
- The combiners apply the epilogue and reset the counters themselves, so no memset node is needed in the graph.

**Instrumentation:** `-DK2_TRACE` adds `TR(k)` globaltimer stamps; `-DABL_NOCOMPUTE` builds the no-compute skeleton.

How v1 converged (local bench, same job as SGLang):

| build | what changed | local geomean | what we learned |
|---|---|---|---|
| k1 | fused kernel, FLAT split for every shape, one (last-arriving) combiner per request | x0.569 | d1f 92.7, v1f 376.5, v32f 887.7 µs: a serial combine over 132 partials is the tail (`evidence/bench_k1.txt`; orchestrator-state "serial combine") |
| k2 | ALIGNED mode for sliding and B = 1; the last K arrivals combine row slices in parallel | x1.236 | decode 1.7-2.0x, verify full 0.60x; the no-compute skeleton shows the verify tile is the limiter (§2e) |
| k3 | latency restructure: every length-independent load (queries, new k/v, positions, cache rows, gates → smem) issued at entry; page-table rows and all of a lane's tiles issued as soon as the length lands; one release atomic per CTA | x1.214 vs k2 x1.229, same job | v32f 646.7 → 608.6 and v16f 332.1 → 315.3 improved, but d1s got worse (10.79 → 12.90): ptxas spilled the `inv_freq` loads with an STL right after the LDG, so every warp waited one DRAM trip before issuing its other early loads (`notes/orchestrator-state.json` hop k3; `../02_*/notes/WC_NOTES.md`, "SASS k3") |

k3 was the team base when the first judge run was spent. It was checked as is, which also calibrated the local judge-projection.

**The invariant that deadlocked twice** (`notes/PLAN.md` "Invariants"):
- The combiners are the last K of P arrivals. K must be ≤ P, and every combiner must own at least one row (rows per combiner ≥ ceil(R/P)).
- Otherwise some rows are never written, and the self-resetting counters never reset: the next call reads stale counts or hangs.
- It was hit in k2 (FLAT, L = 1) and k3 (FLAT, L = 8). k3's fix is the `rpc = max(..., (R + P - 1) / P)` line marked `// K <= P combiners`.

## 5. Result

| check | result |
|---|---|
| local, same job as SGLang | k3 x1.2137 (judge-projected, +1 µs per call: x1.2053) vs k2 x1.2286 (`evidence/bench_k3.txt`); a second job read k3 x1.2271 (`../02_*/evidence/verify_k3w.txt`) |
| correctness | judge 16/16 tests; local `check.py` 0 failures on k3 (`../02_*/notes/WC_NOTES.md`, run_base) |
| judge | **1.2010**, 3,136.1 µs per step (`evidence/judge_after_k3_check_4a7aa2f3.txt`) |
| **judge, read honestly** | One reading. SGLang against itself read 0.9969 the same evening, so the judge has no bias at this level. The local projection was within 0.4 %. |

Per case (judge; per-load speedups recomputed by `evidence/judge_score.py`):

| shape | SGLang | k3 | speedup | k3 kernel time | local k2 no-compute |
|---|---|---|---|---|---|
| d1s | 20.38 | 13.35 | 1.53x | 11.7 | 8.88 |
| d1f | 31.58 | 18.46 | 1.71x | 14.8 | 14.15 |
| d16s | 26.98 | 15.50 | 1.74x | 13.0 | 10.58 |
| d16f | 200.91 | 111.85 | 1.80x | 110.3 | 94.10 |
| d32s | 31.89 | 19.03 | 1.68x | 15.8 | 12.79 |
| d32f | 384.75 | 195.64 | 1.97x | 193.0 | 173.81 |
| v1s | 23.09 | 22.99 | 1.00x | 19.3 | 11.78 |
| v1f | 35.90 | 31.31 | 1.15x | 28.2 | 16.66 |
| v16s | 28.99 | 22.55 | 1.29x | 20.8 | 13.74 |
| v16f | 205.42 | **318.14** | **0.65x** | 314.3 | 124.44 |
| v32s | 34.72 | 30.54 | 1.14x | 28.1 | 16.92 |
| v32f | 390.82 | **608.36** | **0.64x** | 602.7 | 231.65 |

Per load: d1 1.586, d16 1.780, d32 1.903, v1 1.050, v16 0.754, v32 0.705 → geomean 1.2009.

**Against the targets:**
- **Decode full** is at 1.8-2.0x, but its judge times are 36-44 µs above the local gather floors. Even the local k2 skeleton sits 18-22 µs above them. WB's trace of k3 at d32f (`../02_*/notes/WB_NOTES.md`, job1) shows where that goes:
  - ~10 µs of startup before the first gather;
  - ~13 µs of CTA spread (median → max);
  - ~11 µs of tail after the slowest CTA's last tile.
- **b1 sliding** is at 13.35 µs against a ~3 µs target. It is a latency chain (§2e).
- **Verify full** is 1.55x slower than SGLang. Taken alone, v16f and v32f cost −5.3 % and −5.9 % of the score against SGLang's times (`judge_score.py` on the two reports). The cause is the emulated FP8 MMA in the tile: the k2 skeleton runs v32f in 231.7 µs, the real kernel in 638.6.

## 6. Takeaways, and what came next

- **Fusing the chain pays immediately on decode.** One launch, whole 128-B rows, a FLAT split and an in-kernel self-resetting combine gave 1.6-1.9x on every decode load.
- **Check what an MMA instruction lowers to before building a tile on it.** FP8 `mma.sync` on sm_90a is FP16 HMMA plus unpacks. That is tolerable at 8 rows per KV head (decode), and fatal at 64 (verify).
- **A no-compute skeleton next to a no-math gather probe splits the gap correctly.** Tile math vs scheduling: verify full was math; decode full and the small shapes were scheduling and latency chains.
- **Self-resetting counters need every combiner to own work.** Otherwise the failure shows up one call later as a hang.
- **What came next (02_\*):** four worker lanes started in parallel on k3 (`notes/WORKERS.md`; orchestrator-state hop "workers"):
  - W8: native wgmma for q_len 8.
  - WD: native wgmma for q_len 1.
  - WB: FLAT full-layer scheduling.
  - WC: small-call latency.

## Reproduce

All you need is torch and an H200. `tools/` holds the problem spec, `reference.py`, `checks.py`, `baseline.py` and our harness.

```bash
cd muse_attention_decode/tools
python bench.py ../01_fused_flat_splitkv_fp8/build.py --tests                  # correctness + local judge-like timing
python ab.py ../01_fused_flat_splitkv_fp8/build.py ../01_fused_flat_splitkv_fp8/build.py \
       --env ';ABL_NOCOMPUTE' --nocheck --shapes d1s,d32f,v32f                # real kernel vs no-compute skeleton (wrong output)
python ../01_fused_flat_splitkv_fp8/evidence/judge_score.py \
       ../01_fused_flat_splitkv_fp8/evidence/judge_before_calib_sglang_e9c98045.txt \
       ../01_fused_flat_splitkv_fp8/evidence/judge_after_k3_check_4a7aa2f3.txt   # per-load speedups, per-shape shares
# gather floors and phase traces ran inside the era-0 tree E0 (they import its tools/cu.py and tools/bench.py):
cd $E0 && python3 dev/probe_gather.py && python3 dev/trace.py --file dev/k3.py --shape 1,1,sliding
```

Originals: `../autoresearch-kernels/attn/subs/k3_check.py` (= `dev/k3.py`); prototypes `dev/k1.py`, `dev/k2.py`; judge reports `subs/profiles/k3_check.4a7aa2f34cfd44c2.*` and `subs/profiles/calib_sglang.e9c98045b4184779.*`; local runs `out/bench_k1..k3.txt`, `out/trace{1,2}.log`, `out/20261027T230030_prof1/`.
