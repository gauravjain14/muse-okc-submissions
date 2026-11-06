# 01: one persistent fused kernel over a bulk-copy ring of losslessly repacked NVFP4 (v2 to v40)

Paths: `M/` = `muse-opt/muse_moe_w4a16/`, `R/` = `okc-retro/`, `mem:` = `<agent-state>/memory/` (outside the tree), all relative to ``. "transcript, hh:mm UTC" = the main session log `<agent-state>/<session>.jsonl` (outside the tree; 10-23 UTC unless noted; PT = UTC-7). Copies in this folder are cited as `evidence/...` and `notes/...`; `evidence/transcript_*.txt` are verbatim excerpts of that log.

| | before | after (01) |
|---|---|---|
| build | `M/baseline.py`: SGLang's Marlin W4A16 MoE path, the 1.0x reference. First judged build of ours: `M/dev/moe_v2.py` as copied to `submission.py` at 07:50 UTC (transcript, cp command at 07:50 UTC) | `build.py` = `M/dev/moe_v40.py` (md5 8fb0d862, 78,014 B) |
| judge | v2: **1.4753** (check, job f0aae482) | **2.1339** (check of the dev path, job 48fc3fa7) and **2.1331** (scored, job 95e61683; `submission.py` = cp v40 at 14:34 UTC). Real level **2.1335** |
| real effect | | +44.6 % over the first judged build (2.1335 / 1.4753, derived). Every shape moved: bs1 1.96x to 2.32x, bs32 1.09x to 1.60x, bs256 1.63x to 2.55x |
| shapes changed | | all six (a new kernel) |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: one-line comments credit the designs or reference code that parts of it follow. The code is unchanged. The judged file has md5 `8fb0d862`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

Terms used below.
- **Item**: one unit of work from the queue: a token tile times a block of weight rows over all of K, for gate/up (phase 1, "G1") or for down (phase 2, "G2").
- **Stage**: one shared-memory slot of the ring: 32 KB of FP4 weight codes, 4 KB of E4M3 block scales and the activation rows of the tile.
- **Producer / consumers / signaler**: warp 8 issues `cp.async.bulk` copies into the ring; warps 0-7 decode E2M1 to BF16 in registers and run the MMAs; warp 9 publishes finished items to other CTAs.
- **Kernel names** are `moe_<RW>_<NT>_<KS2>_<NS>_<MW>`: RW consumer row-warps (8/RW warps split K, "KW"), NT 8-token groups per tile, KS2 k-split of down items, NS ring stages, MW capacity class by M. **NTL** is the number of token groups actually present in a tile (at most NT).
- **Judge lab**: a check run whose `prepare()` runs several variants on the judge H200 and prints `DIAG` lines, then raises (`evidence/make_diag.py`). It costs one of the 20 daily checks.
- **H100 bench**: `evidence/bench.py` on an sfcompute H100 node (the only GPU until a Modal H200 was approved at 15:35 UTC): CUDA-graph replays after an L2 flush, paired against SGLang in the same process, geomean of six per-shape median ratios. Comparisons are only valid inside one run.

## 1. Starting point

The judge times SGLang's routed-expert path (`M/baseline.py`): `moe_align_block_size`, two `moe_wna16_marlin_gemm` calls (W4A16, FP32 reduce), `silu_and_mul`, and a Triton post-expert-norm weighted sum. Its per-case times in the first judge run (`evidence/judge_first_run_v2_f0aae482.txt`):

| shape | SGLang µs | unique weight MB | byte floor µs |
|---|---|---|---|
| bs1 | 80.08 | 73.3 | 21.1 |
| bs8 | 143.84 | 238.4 | 55.0 |
| bs16 | 189.38 | 332.5 | 78.1 |
| bs32 | 214.11 | 444.6 | 100.4 |
| bs128 | 468.59 | 641.1 | 144.9 |
| bs256 | 615.82 | 698.3 | 156.5 |

MB and floors are from `M/ROOFLINE.md` (written 10-24: the touched experts' bytes for judge-like routings, read cold at the measured plain-load rate). SGLang ran at 2.1-3.9x its floor (derived).

Two Triton candidates handed to us at the start measured 0.633x and 0.313x geomean against SGLang on the H100: they decode each 4-bit weight in FP32 through `tl.where` chains, and the small-batch path re-reads an expert's weights once per token (130-290 GB/s) (transcript, 07:20 UTC). They were not used.

Our own v1 (written 07:28 UTC, never judged) used three kernels: a one-CTA route kernel, a persistent tensor-core GEMM kernel for both GEMMs, and a norm/combine kernel. It passed all tests at rel-L2 2.5e-3 (the SGLang path shows 4.9e-3 on the same tests: transcript, 07:30 UTC) but measured **0.862x** on the H100, with bs128/256 at 435-458 GB/s (`evidence/transcript_v1_profile_ablation.txt`).

## 2. What we saw

**(a) v1: one GEMM config spilled, and the bare pipeline was most of the time.** Per-kernel profile at bs256: the GEMM kernel took 1607.7 µs, route 4.0 and combine 28.1 µs; the 64-token GEMM (`moe_gemm_8_8_2_3`) sat at 168 registers with 520 B of local memory (`evidence/transcript_v1_profile_ablation.txt`). Ablations at bs32 (H100, µs): full 267.0, no decode 224.4, no decode and no MMA 216.9, no activation copies 272.8. Taking the decode out bought 16 %; the skeleton without decode or MMA still took 81 % of the call. v2 fused everything into one launch and capped tiles at 32 tokens.

**(b) The ring itself can stream at 4.3 TB/s on the judge.** A TMA microbenchmark inside a judge-lab run (`evidence/diag_v9.30bce642b3eb4331.log`, code `evidence/tma_bench.py`):

| stage bytes | stages | GB/s |
|---|---|---|
| 16384 | 4 | 3988 |
| 32768 | 3 | 4303 |
| 32768 | 4 | 4321 |
| 36864 | 4 | 4305 |
| 49152 / 65536 | 4 / 3 | 4313 / 4312 |

So 32-36 KB stages are enough; the H100 tops out near 3.0 TB/s (transcript summary at 10:17 UTC).

**(c) The consumers are nearly as long as the stream at every shape.** v9's ablations on the judge H200, µs per call (`evidence/diag_v9.30bce642b3eb4331.log`):

| variant | bs1 | bs8 | bs16 | bs32 | bs128 | bs256 |
|---|---|---|---|---|---|---|
| full | 37.2 | 72.5 | 106.4 | 141.1 | 219.9 | 293.5 |
| compute only (`MOE_NO_LOAD`) | 31.4 | 64.9 | 96.6 | 128.2 | 196.0 | 255.5 |
| no decode (`MOE_NO_DEQ`) | 34.7 | 68.8 | 95.2 | 126.2 | 188.5 | 258.0 |
| data only (`MOE_NO_MMA+MOE_NO_DEQ`) | 34.2 | 70.9 | 94.2 | 124.0 | 177.2 | 231.6 |
| fixed costs only (`MOE_NO_WORK`) | 7.9 | 7.6 | 7.3 | 9.8 | 11.7 | 16.0 |

Compute-only is within 9-16 % of full, and data-only within 2-21 %. Neither side alone sets the pace. What is left is overlap, plus 7-16 µs of fixed cost per call: routing, startup, combine and the end of the call. That fixed cost is 21 % of bs1.

**(d) The decode is at its instruction floor.** One 32-bit word holds 8 E2M1 codes. A BF16x2 A-fragment register needs each code's sign and three magnitude bits at fixed positions, so the decode is shifts and masks, then one `HMUL2` by the block scale, which also fixes the exponent bias.
- A first search (07:15-07:19 UTC) showed that no single shift-and-mask extracts all four outputs and that 8 translates of the target bit set cannot tile the word; the cheapest extractor set costs **9 integer ops** (`evidence/transcript_decode_layout_search.txt`). v6 used a pure-rotation layout: rotations 4, 10, 6 and 14, i.e. 5 LOP3 + 4 SHF. It cut the hot loop from 915 to 775 instructions, but the time did not move (H100 1.603x vs v5's 1.620x) (`evidence/transcript_v6_v9_stage_analysis.txt`).
- Moving all shifts to the FMA pipe as `mul.lo`/`mul.hi` lost (v8, 1.484x); so did `mul.wide` rotations (v13) and other IMAD left-shift layouts (v14).
- v30 found the form that pays: the rotations by 4 and 6 never wrap inside the masked bits, so they are plain left shifts and run as `IMAD q*16`, `q*64` (constants from `__constant__` memory so ptxas keeps them as multiplies): **7 ALU + 2 IMAD + 4 HMUL2 per word**. On 10-24 an exhaustive search over all 96 layouts with three single-rotation outputs and one two-piece output confirmed 7 ALU + 2 IMAD as the minimum (`evidence/transcript_decode_layout_search.txt`, last block, 10-24 08:29 UTC).
- Pipe rates measured on the H100: LOP3/SHF/IMAD/PRMT 2 cycles per warp instruction per SMSP, HMUL2 and FFMA ~1.1, `mma.sync m16n8k16` ~8.7 cycles per HMMA per SMSP (~40 % of tensor peak) (transcript summary at 10:17 UTC; `evidence/mma_bench.py`, `evidence/deq_bench.py`).

**(e) Most tiles did MMA work for token groups they did not have.** v6's per-CTA stall counters on the H100 showed the producer waiting 33-88 µs per call for free slots at bs16-256, i.e. the consumers were the bottleneck (`evidence/transcript_v6_v9_stage_analysis.txt`, 08:17 UTC). Most experts at bs8-32 see 1-4 tokens (bs32: ~50 pairs over ~16 experts, `M/ROOFLINE.md`; derived), yet a stage ran all NT groups of its config. v9 templated the stage compute on NTL, the live group count, so a 1-group tile issues 1/NT of the MMAs and B-fragment loads.

**(f) Where the time went at v40** (judge lab, per-call medians in µs; `evidence/diag_timing_v40.76e2a356a5844017.log`):

| shape | routed | first data | items end (min/med/max) | combine end | per-CTA compute | consumer wait | epilogue | producer waits for slots | data-only call (no decode/MMA) |
|---|---|---|---|---|---|---|---|---|---|
| bs1 | 2.7 | 5.0 | 26.6 / 28.8 / 29.8 | 33.8 | 15.8 | 3.8 | 2.9 | 1.8 | 28.9 |
| bs8 | 2.6 | 4.9 | 62.6 / 63.9 / 67.5 | 70.3 | 42.6 | 5.7 | 4.3 | 8.3 | 63.2 |
| bs16 | 2.6 | 4.8 | 82.5 / 85.2 / 88.1 | 91.3 | 57.9 | 7.3 | 5.8 | 9.4 | 82.9 |
| bs32 | 4.4 | 6.2 | 122.3 / 124.9 / 127.4 | 130.9 | 87.4 | 9.1 | 5.3 | 20.6 | 116.4 |
| bs128 | 6.0 | 6.8 | 170.8 / 172.6 / 175.5 | 180.9 | 130.4 | 8.8 | 7.3 | 39.2 | 152.3 |
| bs256 | 9.0 | 9.9 | 211.1 / 214.7 / 216.7 | 227.0 | 156.0 | 11.1 | 12.2 | 50.8 | 178.7 |

Reading it:
- At bs128/256 the producer waits 39-51 µs for a free slot while consumers wait only 9-11 µs for data: the consumers are compute-bound. The full call is 19-27 % longer than the data-only call.
- At bs1-16 the call ends 5-6 µs after the median CTA's last item: tail spread plus the combine.
- First data lands at 4.8-9.9 µs: routing (2.6-9.0 µs, more at bs32+ because of the activation gather) comes first.
- The earlier v19 timeline (`evidence/diag_timing_v19.faa34bc1ce5746e3.log`) measured the in-kernel SM clock at 1.77-1.82 GHz under load and the per-stage compute at 891-1171 ns.

**(g) The H100 misleads on compute-bound shapes.** The H100 has ~30 % less bandwidth, so bs128/256 look bandwidth-bound there.
- Per-item stage loops measured flat on the H100 (v39 vs v38: 2.042-2.066 vs 2.043-2.058).
- On the judge they were **-2.5 % time at bs128** (183.8/181.8 vs 187.0/187.9 µs), -0.9 % at bs256, and **+2.0 % at bs32** (127.6/127.1 vs 125.0/124.6) (`evidence/diag_abl_v39a.b88005358ff4453d.log`). v40 took them only where they won.
- The same happened to v46 (epilogue warps finishing items): faster in H100 compute-only mode, judged 2.1270 against v40's 2.1335 (`evidence/judge_v46_7440dd65_dead_end.txt`).

## 3. Diagnosis and projection

- **Every shape is memory-bound by FLOP/B, but the decode is a second ceiling.** At 13 instructions per 8 weights (9 integer + 4 HMUL2) and ~900-1000 ns per 32 KB stage, the consumers need about as long as the ring takes to stream the stage. The data-only skeleton runs within 2-21 % of the full call.
- **The design therefore has to:**
  - stream only the touched experts' bytes, once;
  - keep the ring full while 8 warps decode;
  - not compute empty token groups;
  - hide routing, the gather and the combine behind the stream wherever possible.
- **Projection at the time:** "Theoretical max for this design on the H200 is about 2.7x" (transcript summary at 10:17 UTC; the floors in section 1 give the same order: geomean of SGLang/floor ≈ 2.9x, derived).

## 4. Change

`change.diff` is against `M/baseline.py` (it replaces the file). The v40 kernel, one launch per call:

1. **Weights, prepared once.** `prepare()` permutes each local expert's E2M1 codes into the order the `mma.sync m16n8k16` A fragments read them: `_POS` places each code's bits where the decode's rotations expect them, and `_repack` adds a k permutation so each thread needs one scale pair per 64-k chunk. It is the same NVFP4 values in the same bytes, a lossless repack. Gate and up rows are interleaved 8 by 8 (`_gate_up_rows`) so SwiGLU is local to a thread. A zero block scale gets zero codes. Nothing is dequantized ahead of time.
2. **One CTA per SM, 384 threads.** `setmaxnreg.dec` to 64 for warps 8-11 and `.inc` to 216 for the consumers. Routing runs redundantly in every CTA with no atomics: token bitmaps per local expert from `__ballot_sync` (v40), a rank by size, the pair list, and tiles of up to 8·NT tokens. Configs with NT ≥ 4 then gather activations tile-major into global memory behind one grid-wide count.
3. **Work queue.** The first wave is static (CTA b takes item b, v24); the rest come from an atomic queue. Order: all gate/up items, then down items, tile-major. A down item waits on a per-tile counter of its gate/up items. The last tiles' down items are split 4 ways in k (`KSX`, v25) to shorten the tail.
4. **Producer.** One weight copy and one scale copy per k-chunk (v17) into an NS ≤ 4 stage ring with mbarriers, plus the activation rows. The producer warpgroup leaves routing early, so the first item's weights stream while the consumers build the pair list (v24).
5. **Consumers.** Per stage: `dequant()` (7 ALU + 2 IMAD), `HMUL2` by the scale pair (fast path `prmt + x*16 + 0x77007700` when all scales are normal E4M3, v5), then `mma.sync` over the tile's NTL live groups (v9). The 64-token config at bs > 128 uses `wgmma m64nNk16` (A from registers, B from 128B-swizzled shared memory) for tiles with at least 5 groups (`MOE_WGMIN 5`, v19). Configs with KW > 1 or KS2 == 1 run a per-item loop that reads the stage info once per item (v40).
6. **Epilogue.** Accumulators are staged in the item's held last ring slot and stored coalesced (v21): SwiGLU to a BF16 intermediate, or FP32 down partials. The signaler warp publishes with `red.release` (cumulative, so no extra fence: v24).
7. **Combine.** Per token, after its tiles' down items are done (per-tile readiness for KW ≤ 2, v29/v32): RMSNorm of y·(1+w) per (token, expert), weighted sum, BF16. Batched loads for multi-k-warp configs (v33) and a per-pair loop with next-pair prefetch for KW = 1 (v38).
8. **End.** The last CTA to arrive resets the counters for the next call (CUDA-graph safe).

Buckets (`BUCKETS`, confirmed by a judge-lab sweep of 11 configs, `evidence/diag_sweep_v17.fbe7638c93224280.log`):

| M | config (RW, NT, KS2) | judge kernel |
|---|---|---|
| 1 | (2, 1, 2) | `moe_2_1_2_4_1` |
| 2-16 | (4, 1, 2) | `moe_4_1_2_4_1` |
| 17-32 | (8, 4, 2) | `moe_8_4_2_4_1` |
| 33-128 | (8, 4, 1) | `moe_8_4_1_4_4` |
| 129-512 | (8, 8, 1), wgmma for ≥ 5 groups | `moe_8_8_1_3_8` |
| > 512 | PyTorch fallback | |

How the design converged. The H100 column is the paired geomean vs SGLang on the H100 bench, compared only within one run. The bench's L2 flush was switched to a read-only `flush.sum()` at 11:25 UTC, which lifted all levels by ~5 %, so rows before and after v21 are not comparable (`evidence/transcript_h100_bench_filtered_v1_v40.txt`; earlier rows: transcript summary at 10:17 UTC):

| build | change | H100 (same-run control) | judge | what it taught |
|---|---|---|---|---|
| v1 | 3 kernels | 0.862 | not judged | the split and an NT=8 spill cost more than the decode |
| v2 | one persistent kernel: routing per CTA, queue, bulk-copy ring, fused SwiGLU/down/combine, tiles ≤ 32 tokens | 1.447 | **1.4753** check (f0aae482) | the H100 tracked the judge within 2 % here |
| v3 / v4 | tile-major order, pipelined combine / signaler warp | 1.474 / 1.557 | | pipeline control fences were on the consumers' path |
| v5 / v6 / v7 | fast scales / 9-op rotation decode / register prefetch | 1.620 / 1.603 / 1.612 | | fewer instructions alone did not move time |
| v8 | shifts on the FMA pipe | 1.484 | | dead |
| v9 | stage compute specialized per NTL | 1.796 | **1.8678** scored (6badc871) | the largest single step: +26.6 % on the judge |
| v10 / v11 | combine items / stream-K static schedule | 1.793 / 1.559 | | dead: the dynamic queue absorbs per-CTA spread |
| v12 | global scales in shared memory, L2 prefetch off, split accumulators | 1.832 | | L2 prefetch competes with the ring (worse the further ahead) |
| v13-v16 | `mul.wide` rotations, IMAD shifts, `setmaxnreg`, wgmma | all ≤ v12 | | wgmma alone (v16) 1.77: it did not overlap with the decode |
| v17 | one weight + one scale copy per k-chunk, stage-major intermediate, gather for NT ≥ 4 | 1.864-1.871 | **1.9686** scored (10b4d2c3) | the producer's copy count starved it; it shares SMSP0 with two consumer warps |
| v18 | per-(tile, 256-k) dependency counters | 1.818-1.864 | | dead (tie at best) |
| v19 | `setmaxnreg` role branches + wgmma (N ≤ 64) for 64-token tiles at bs > 128 | 1.868 / 1.877 vs v18 1.853 | | bs256 -5 % time; bs128 stays on 32-token mma.sync tiles |
| v21 | staged coalesced epilogue | 1.884 / 1.878 vs v20 1.873 / 1.865 | | |
| v22 / v23 | per-item loops / elementwise combine | bs32 +2.6 % time / 1.95 vs 1.99 | | per-item loops returned later where they win |
| v24 | static first wave, producer leaves routing early, no signaler fence | 2.002 (fence-free) vs v21 1.986 / 1.987 in one run | **2.0267** check (7b64da29) | the first weights can stream during the pair list |
| v25 | 4-way k-split of the last tiles' down items | 2.000 / 2.003 vs 1.997 / 1.998 | | bs256 -1.5 % time |
| v29 | per-tile combine readiness | ±0.2 % | | |
| v30 | decode 7 ALU + 2 IMAD; IMAD scale conversion only for 1-group stages | 2.021 / 2.029 vs 2.005 / 2.002 | **2.0829** check (32cdc83b) | IMAD scales help at 1 group (-1.8 %) and hurt at 2-8 (+1.4 to +3 %) |
| v33 | batched combine for KW > 1 | 2.034 / 2.036 vs 2.018 / 2.027 | **2.0874** check / **2.0913** scored | |
| v34-v38 | slot release, per-tile gather readiness, batched gather, faster ranking, per-pair combine | within ±0.3 % | | |
| v39 | per-item loops everywhere | flat | judge lab: bs128 -2.5 %, bs32 +2.0 % | the H100 hides compute-bound effects |
| **v40** | per-item loops only for KW > 1 or KS2 == 1; ballot bitmaps | 2.055-2.067 vs v39 2.061-2.064 | **2.1339** check / **2.1331** scored | |

Judge ablations at three points of this path are in `evidence/diag_abl_v20...`, `diag_abl_v30...` and `diag_abl_v33...` (tuning knobs `MOE_GATHER_MIN`, `MOE_SPLW`, `MOE_WGMIN`: all within ±1 % of the defaults).

**Byte-stream provenance of the judged builds** (transcript, cp commands; file mtimes in `M/dev/`):
- `dev/moe_v2.py` was edited at 07:52-07:53 UTC, after its 07:50 copy (ablation `#ifdef`s were being added, inferred). `dev/moe_v9.py` was edited at 08:53 UTC, after its 08:28 copy (a plain-argument launcher for the diagnostics, inferred). So the judged v2 and v9 byte streams are **not preserved**.
- `dev/moe_v30.py` was edited at 12:47-12:48 UTC, after its 12:43 check upload. The edits were experiment-only paths under `#ifdef` (`MOE_HMUL_NONE`, `MOE_BFMA`, the `MOE_NO_WORK` variables), so the judged code path is unchanged (inferred from the transcript).
- v17 (copied in the same minute as its mtime), v24, v33 and **v40** (checked at 14:23 UTC and copied at 14:34 UTC, both after its 14:18 UTC mtime) are preserved. `build.py` here is v40 byte-identical (md5 8fb0d862).

## 5. Result

| check | result |
|---|---|
| local, H100, same runs vs v39 | 2.055-2.067 vs 2.061-2.064: flat, as expected on the H100. v40's gain over v33 shows only on the H200 (judge lab, row (g) above) |
| local, Modal H200 (later, same job as v50) | v40 2.181 vs v50 2.186 / 2.190 (`../02_per_item_loops_ssq_fast_combine/evidence/sweeps/s1_result.txt`). That harness reads ~3 % above the judge (transcript, 19:01 UTC) |
| correctness | 5/5 judge tests in both runs; local tests rel-L2 2.47e-3 to 2.55e-3 (bs 1, 4, 33, 257, 128-skewed) against the 1.5e-2 limit (`evidence/transcript_h100_bench_filtered_v1_v40.txt`) |
| judge | v2 1.4753 → v40 **2.1339 / 2.1331** |

Judge per case, "yours" µs and speedup (`evidence/judge_*`):

| shape | v2 (f0aae482) | v9 (6badc871) | v17 (10b4d2c3) | v33 (a3b124a9) | v40 (48fc3fa7) | v40 (95e61683) |
|---|---|---|---|---|---|---|
| bs1 | 40.88 (1.959x) | 33.88 (2.288x) | 33.16 (2.340x) | 34.75 (2.292x) | 34.70 (2.307x) | 34.13 (2.331x) |
| bs8 | 103.45 (1.390x) | 79.55 (1.778x) | 77.62 (1.948x) | 68.58 (2.088x) | 71.14 (2.106x) | 67.74 (2.107x) |
| bs16 | 138.56 (1.367x) | 104.88 (1.771x) | 102.90 (1.826x) | 97.91 (1.958x) | 94.02 (1.997x) | 95.62 (1.968x) |
| bs32 | 196.61 (1.089x) | 147.51 (1.419x) | 142.72 (1.443x) | 135.60 (1.571x) | 133.52 (1.603x) | 129.11 (1.591x) |
| bs128 | 300.08 (1.562x) | 224.21 (2.044x) | 213.66 (2.131x) | 202.18 (2.285x) | 193.89 (2.373x) | 195.58 (2.405x) |
| bs256 | 378.22 (1.628x) | 300.71 (2.031x) | 276.97 (2.274x) | 250.51 (2.488x) | 240.79 (2.557x) | 243.45 (2.547x) |
| **score** | 1.4753 | 1.8678 | 1.9686 | 2.0913 | 2.1339 | 2.1331 |

**Read honestly:**
- The two v40 runs agree to 0.04 %, but single shapes moved more: bs8 71.14 vs 67.74 µs, bs32 133.52 vs 129.11 µs. SGLang's own bs8 moved with it (149.81 vs 142.75 µs), so the paired speedups barely changed (2.106 vs 2.107). Treat per-shape judge µs as ±3 %.
- v30 → v33 read +0.3 % on the judge (H100 +0.6 %). v33 → v40 read +2.1 % on the judge, while the H100 saw v39 and v40 flat against their parents. Most of that +2.1 % is the per-item loop at bs128/256, which only the judge lab could see.

**vs projection:** 2.13x against a ~2.7x design ceiling. The timeline in 2(f) shows the remaining gaps:
- 5-10 µs before first data;
- 4-10 µs from the median CTA's last item to the end of the combine;
- at bs128/256, the full call 19-27 % longer than the data-only call.

## 6. Takeaways, and what came next

- **Fuse first, then measure where the stream waits.** The single kernel, the per-NTL specialization (v9) and the lighter producer (v17) carried most of the 1.48x → 2.13x. Instruction-count cuts to the decode (v6, v8) did not move time until the decode matched the pipe mix (v30).
- **Measure on the judge's GPU.** The H100 was right about memory-bound shapes and wrong about compute-bound ones (v39/v40, v46). From here on, judge-lab checks and (from 15:35 UTC) a Modal H200 were the A/B tools.
- **Two leads were left open:**
  - the fixed costs at both ends of the call (end-of-kernel reset, combine latency, startup);
  - bs32's per-stage loop, the one config v40 left off the per-item path.

  Both are milestone 02 (`../02_per_item_loops_ssq_fast_combine/`): fence-free end reset, suspend-hinted waits, an inline-PTX queue claim, the sum-of-squares fast combine for 1-group configs, and per-item loops everywhere with a top-tested loop.

Raw notes: the dated memory notes `mem:okc-moe-versions.md` (version log, written during the session) and `mem:okc-moe-findings.md` (measured facts), cited by their original paths (they also hold board standings, so their copies stay private).

## Reproduce

```bash
cd muse_moe_w4a16/tools
python3 check_mt.py ../01_persistent_fused_kernel_tma_ring/build.py 1 4 8 16 32 33 128 256 257   # correctness vs reference
# same-job A/B against SGLang and a control copy (judge-matched routings)
B2_SEEDS=jm_seeds.jseeds python3 bench2.py - --reps 60 --shapes 1,8,16,32,128,256 \
  --variants "v40|../01_persistent_fused_kernel_tma_ring/build.py|;v40b|../01_persistent_fused_kernel_tma_ring/build.py|"
```

As run at the time (H100 node, then Modal H200), from `M/`:

```bash
sh dev/setv.sh moe_v40 && python3 dev/bench.py dev/abl.py --reps 20            # paired vs SGLang, one process
MOE_DEFINES=MOE_NO_LOAD python3 dev/bench.py dev/abl.py --no-check --no-base   # ablation (compute only)
python3 dev/make_diag.py dev/moe_v40.py dev/diag_timing_v40.py timing "full;MOE_NO_MMA+MOE_NO_DEQ"
okc check dev/diag_timing_v40.py                                               # judge-lab timeline
```

Originals: `M/dev/moe_v1.py` … `moe_v46.py`, `M/dev/diag_*.py` and `M/dev/profiles/diag_*.log`, judge results `M/profiles/submission.*` and `M/dev/profiles/moe_v*.*`, `M/dev/check_v*.json` / `submit_v*.json` (job id per version).
