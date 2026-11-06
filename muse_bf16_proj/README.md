# muse_bf16_proj

The five dense projections of one Muse Spark Open layer, as one H200 runs them under TP8: `out = x @ W^T` for m = 1 to 256 tokens, timed against SGLang's cuBLAS dispatch (`tools/TASK.md`, `tools/baseline.py`). It was an Open Kernel Challenge problem; the boards closed on 2026-11-04.

## The problem and how it is scored

| layer | W [n, k] | weight dtype | bytes |
|---|---|---|---|
| qkv_gate | [2304, 8192] | BF16 | 37.7 MB |
| o | [8192, 1024] | BF16 | 16.8 MB |
| pooler | [4096, 8192] | BF16 | 67.1 MB |
| unpooler | [8192, 4096] | BF16 | 67.1 MB; at m ≤ 8 only this GPU's slice, the first 1024 rows (8.4 MB) |
| router | [256, 8192] | **FP32**, FP32 output | 8.4 MB |

Shapes from `tools/reference.py`; bytes computed from them.

- **Loads.** Six: decode 1 / 16 / 32 (m = 1, 16, 32) and verify 1 / 16 / 32 (m = 8, 128, 256, i.e. 8-token verify blocks). Each load runs all five layers, each with weight 62. A load's time is the sum of its five shapes.
- **Score.** The geometric mean over the six loads of SGLang's load time over ours. Each load counts equally, so a µs saved is worth most on the small loads. From the release-cut build's judge run (computed, `11_small_shape_tails_stack/evidence/judge_after_bo23L_scored_43c064cb.txt`): decode 1 0.51 %/µs, verify 1 0.48, decode 16 0.37, decode 32 0.34, verify 16 0.27, verify 32 0.21.
- **Timing.** Each shape is a CUDA graph, replayed from a cold L2 (the judge reads a buffer larger than L2 first), alternating with SGLang on the same fresh inputs. A shape's speedup is the median of order-swapped paired ratios. The judge subtracts a graph replay's fixed cost (~4 µs) from both sides. Every run reports a 95 % interval of at least ±1.5 %.
- **"SOL"** in the judge output is max(BF16 bytes / 4.8 TB/s, FLOPs / 989 TFLOP/s). Our weight formats ship fewer bytes than BF16, so a kernel can sit below 1.0× SOL.
- **SGLang.** `bf16_gemm_dispatch` (cuBLAS) for the four BF16 layers. The router runs FP32 `F.linear` at m ≤ 8 and two BF16 GEMMs on a hi/lo split of the weight above that.

**Rules that shaped every design** (`tools/TASK.md`):
- Weights may be repacked and compressed **losslessly**; they may not change precision. The router's FP32 weight may run as a BF16 hi + lo pair with FP32 accumulation.
- Activations stay BF16 where SGLang keeps BF16; accumulate in FP32.
- Tolerance is 1 % relative L2 (0.01 % for the router). About one row in eight of x is one-hot (a single power of two), so its output is a column of W and must be within 0.1 %: rare weights that a compressed format cannot represent ("escapes") must be corrected exactly, not dropped.
- Added on 11-02: files must be ≤ 1,000,000 bytes, with no compressed, encoded or obfuscated source. Our own rule throughout: kernels ship as source and are compiled inside `prepare()` (untimed, up to 300 s); no precompiled cubins or hand-written SASS.

## Terms used in the records

- **F12 / F11.** Our lossless weight formats. F12 stores each BF16 weight as its low byte plus a 4-bit code for sign and exponent within a 16-exponent window, 12.03-12.13 bits per weight including escape storage. F11 uses an 8-exponent window, 11.45-11.58 bits per weight. Measured weight entropy is 10.46 bits per weight (`muse-opt-codex/muse_bf16_proj/claude/t3/ROOFLINE.md`, lines 10-12 and 103-104; the builds' format tables).
- **Escapes.** Weights whose exponent falls outside the window: ~0.007-0.01 % of weights in F12, ~1.3 % in F11. They are stored separately and must be corrected exactly, because the one-hot test rows read single weights.
- **GEMV / tile kernel / ring.** At m = 1 a many-CTA GEMV; at m = 8 one-wave tile kernels; at m = 8-32 persistent "rings": one CTA per SM, a producer issuing TMA bulk copies of W (and x) into a stage ring in shared memory, consumer warps decoding into tensor-core MMAs.
- **Split-K, cluster pair, solo phase.** At m = 128/256 two CTAs of a cluster split K and exchange FP32 partials through distributed shared memory (DSMEM) instead of global memory. In the "solo phase" each CTA computes only its own tokens for the last stages, so the partner's partial can be pushed early.
- **Build names.** `candidate_vN`: the Codex era (10-27 morning). `int_vN`: the first Claude push (10-27). `int_t3x`, `boN`: the second push (11-02 on); `boNr`: the readable rebuild; `boNL`: builds on the lean base. 
- **Lanes.** One agent's work directory, named by letters (G, Q2, S5, ...). Each milestone defines the ones it uses.
- **Judge runs.** A *check* is a judge run kept off the board; a *scored* run counts. Both use the same harness. The same file can read differently from run to run (next section).
- **Local bench.** `tools/bench.py`, a judge-alike: CUDA graphs, 512 MB cold-L2 read, paired with SGLang. Judge ≈ local + 1.0-1.1 µs per shape, because the judge does not subtract the marginal cost of a kernel inside a graph (`ROOFLINE.md` lines 407-412).
- **x clones, alternating processes, control shapes.** The placement of W and x in memory moves small kernels by 0.1-0.5 µs, and a process can bias all its kernels by several percent. Sub-1 % A/Bs therefore use several graphs with x at random addresses, alternate candidates across fresh processes, and correct every ratio against a shape whose code did not change.

## How to read the judge numbers

A single judge reading carries about ±1-1.5 % of node noise (`tools/TASK.md` puts the 95 % interval at ≥ ±1.5 %). The same file, judged twice (computed from the judge results in each milestone's `evidence/`):

| build | first run | second run | Δ |
|---|---|---|---|
| bo1 (milestone 06) | 1.5690 check | 1.5688 scored | −0.01 % |
| bo11 (08) | 1.6338 check | 1.6085 scored | −1.55 % |
| bo13r (09) | 1.6062 check | 1.6326 scored | +1.64 % |
| bo21L (inside 11) | 1.6337 scored | 1.6124 check | −1.30 % |

Per shape, identical code moves 0.3-0.6 µs between runs (sd ~3 %). "Real level" below is the mean over judge runs of the same file; with one reading it is that reading. Each milestone splits its judge delta into changed and unchanged shapes, and leans on same-run local A/Bs with controls for anything under ~1 %. From milestone 09 on the judge readings stay flat at ~1.62-1.63 while the local A/Bs bank +1-2 %: the stacked gains are real but smaller than one run's noise.

## Milestones that moved the score

| # | milestone | judge readings | real level | what drove it |
|---|---|---|---|---|
| 01 | [per-shape dispatch table over libraries, Triton, TinyGEMM](01_library_dispatch_table/) (Codex era) | 1.2417, 1.2471 | 1.2471 | ncu of every custom path: lossless codecs at 254 registers and 0.5 TB/s (102.6 / 40.3 / 25.1 µs vs ~21), m256 GEMMs that gained 5 µs in the loop and paid 5-7 µs in a split-K reduction → pick the best measured library path per (m, layer) |
| 02 | [12-bit lossless weights streamed into registers (K1s), router worker](02_f12_register_streaming/) | 1.3112 | 1.3112 | m ≤ 16 is a pure weight stream and the 12-bit floor is 17-24 % below BF16; five TMA-ring 12-bit kernels lost to register streaming |
| 03 | [exhaustive cuBLASLt config search at m = 128/256](03_cublaslt_config_search/) | 1.3482 | 1.3482 | custom GEMMs K2/K2c lost to cuBLAS (split-K tail 4-7 µs); 44,199 searched configs gave pooler m128 −25 %, m256 −22 % |
| 04 | [many-CTA 12-bit GEMV at m = 1 with in-row escapes (GV3)](04_many_cta_gemv_m1/) | 1.3997 (int_v5), 1.3999 | 1.3999 | GV's two dependent cold loads per CTA (offset table → escape list) doubled CTA lifetime; inline escape slots removed them |
| 05 | [11-bit F11 format and load-sinking-free GEMVs at m = 1](05_f11_gemv_m1_no_load_sinking/) | 1.4168 | 1.4168 | SASS: without min-blocks ptxas sank 6 of 19 loads below the first FMA; F11 cuts 4.5 % of the bytes |
| 06 | [per-shape best-of on the public alygg77 file](06_public_per_shape_best_of/) | 1.5690, 1.5688 | 1.5689 | our m = 8-32 kernels were latency-bound (25 % occupancy, 0.5 eligible warps); a same-run bench put the public file at 1.5835 vs ours 1.4594 |
| 07 | [sk11: ring escape lists in registers (+ router, m128 pooler, m256 qkv)](07_sk11_register_escape_ring/) | 1.5763, 1.5757, 1.5910 | 1.5910 | ring DRAM at 96 % of peak, escape warps ~0.85 µs and slowing the consumer on their SMSP |
| 08 | [BF16 wgmma GEMMs at m = 128/256 with a cluster-pair split-K exchange](08_cluster_pair_splitk_wgmma/) | 1.6338, 1.6085 | 1.6212 | the m128 loops already ran at the BF16 HBM rate; the cost was the start and a 2.5-3 µs global-memory exchange |
| 09 | [compliant readable rebuild + four lane kernels](09_compliant_readable_rebuild/) | 1.6062, 1.6326 | 1.6194 | 11-02 rules (≤ 1 MB, no encoded source) made the public base unusable; m256 bound by per-SM ingress (~150 GB/s), egress 25-30 GB/s per SM |
| 10 | [lean, provably identical build + exact-escape small-shape modules](10_lean_build_exact_escapes/) | 1.6277 | 1.6277 | no room left under 1 MB, and the m ≤ 8 / m = 1 kernels flushed tiny weights to zero; first build to pass every gate |
| 11 | [small-shape tails stack (router m128 egress, qkv m8/16 cold epilogue, router m16 zero barrier, o m8 one wave)](11_small_shape_tails_stack/) | 1.6337, 1.6124, 1.6315, 1.6232 | 1.6232 | globaltimer traces and SASS of each kernel's tail; N2's "cold fetch" re-diagnosed as load sinking |

Readings are in run order; whether each was a check or a scored run, and its job ID, is in the milestone's header table and `evidence/`. Milestones 01-05 and 07 are check runs only. Milestone 11 is the last milestone in this tree.

## Where the time goes at milestone 11

`11_small_shape_tails_stack/build.py` (bo23L), its judge run (`11_small_shape_tails_stack/evidence/judge_after_bo23L_scored_43c064cb.txt`; groups computed).

| load | layer | m | SGLang µs | ours µs | SOL µs | ours / SOL | kernel |
|---|---|---|---|---|---|---|---|
| decode 1 | qkv_gate | 1 | 14.06 | 8.12 | 7.87 | 1.03 | n2_qkv_gate |
| decode 1 | o | 1 | 6.94 | 4.78 | 3.50 | 1.37 | nk_o1 |
| decode 1 | pooler | 1 | 21.58 | 12.73 | 13.99 | 0.91 | n2_pooler |
| decode 1 | unpooler | 1 | 7.20 | 3.70 | 1.75 | 2.11 | sm5_cgemv |
| decode 1 | router | 1 | 10.53 | 3.70 | 1.75 | 2.11 | sm5_rz1 |
| decode 16 | qkv_gate | 16 | 13.74 | 9.28 | 7.93 | 1.17 | qkvb |
| decode 16 | o | 16 | 6.78 | 5.39 | 3.56 | 1.51 | sk11_o_16 |
| decode 16 | pooler | 16 | 21.20 | 13.51 | 14.06 | 0.96 | pu5_pooler_16 |
| decode 16 | unpooler | 16 | 17.98 | 13.10 | 14.06 | 0.93 | pu5_unpooler_16 |
| decode 16 | router | 16 | 14.88 | 4.16 | 1.81 | 2.30 | s6_rz2 |
| decode 32 | qkv_gate | 32 | 14.05 | 9.83 | 8.00 | 1.23 | sk11_qkv_gate_32 |
| decode 32 | o | 32 | 6.88 | 5.92 | 3.62 | 1.64 | o3_o_32 |
| decode 32 | pooler | 32 | 21.65 | 14.46 | 14.14 | 1.02 | pu5_pooler_32 |
| decode 32 | unpooler | 32 | 18.27 | 14.22 | 14.14 | 1.01 | skh_unpooler_32 |
| decode 32 | router | 32 | 15.26 | 4.62 | 1.86 | 2.48 | ya |
| verify 1 | qkv_gate | 8 | 14.21 | 9.08 | 7.90 | 1.15 | qkvb |
| verify 1 | o | 8 | 6.86 | 5.10 | 3.53 | 1.44 | o8_m8 |
| verify 1 | pooler | 8 | 21.14 | 13.52 | 14.02 | 0.96 | m8_pooler |
| verify 1 | unpooler | 8 | 6.66 | 3.54 | 1.76 | 2.01 | unps_8 |
| verify 1 | router | 8 | 26.30 | 3.63 | 1.78 | 2.04 | sm5_rz1 |
| verify 16 | qkv_gate | 128 | 16.70 | 12.98 | 8.42 | 1.54 | kg_qkv_gate (g128q) |
| verify 16 | o | 128 | 7.28 | 6.47 | 3.99 | 1.62 | q5_o |
| verify 16 | pooler | 128 | 25.34 | 19.40 | 14.64 | 1.33 | kg_pooler (g128q) |
| verify 16 | unpooler | 128 | 19.97 | 17.58 | 14.64 | 1.20 | u128_unpooler |
| verify 16 | router | 128 | 16.64 | 5.57 | 2.21 | 2.52 | r5 |
| verify 32 | qkv_gate | 256 | 20.06 | 15.77 | 9.77 | 1.61 | ph_qkv (g256q) |
| verify 32 | o | 256 | 9.28 | 8.47 | 4.48 | 1.89 | kg_o (g256k) |
| verify 32 | pooler | 256 | 31.18 | 23.82 | 17.37 | 1.37 | kg_pooler (g256k) |
| verify 32 | unpooler | 256 | 24.29 | 23.00 | 17.37 | 1.32 | kg_unpooler (g256k) |
| verify 32 | router | 256 | 17.94 | 7.02 | 2.68 | 2.62 | r4 |

| load | SGLang sum | ours sum | SOL sum | load speedup | ours / SOL | value of 1 µs |
|---|---|---|---|---|---|---|
| decode 1 | 60.31 | 33.03 | 28.86 | 1.826 | 1.14 | 0.51 % |
| decode 16 | 74.58 | 45.44 | 41.42 | 1.641 | 1.10 | 0.37 % |
| decode 32 | 76.11 | 49.05 | 41.76 | 1.552 | 1.17 | 0.34 % |
| verify 1 | 75.17 | 34.87 | 28.99 | 2.156 | 1.20 | 0.48 % |
| verify 16 | 85.93 | 62.00 | 43.90 | 1.386 | 1.41 | 0.27 % |
| verify 32 | 102.75 | 78.08 | 51.67 | 1.316 | 1.51 | 0.21 % |

The geomean of the load speedups reproduces the reading, 1.6231. Summed over the 30 shapes we take 302.5 µs against 236.6 µs of SOL, **+65.9 µs (+27.8 %)**:

| group | µs over SOL | what limits it (measured; see the milestone) |
|---|---|---|
| qkv / pooler / unpooler at m = 128 / 256 (6) | +30.3 | BF16 bytes at m128 (the loops already stream at the BF16 HBM rate, 08); at m256 the tensor clock under the H200's on-die limiter (1.61-1.65 GHz within ~1 µs of tensor load, 10) and per-SM ring ingress (~150 GB/s, 09); split-K exchange tails (08, 09) |
| routers (6) | +16.6 | tiny kernels: first byte, the split-K reduction after the stream, partial-sum egress at 25-30 GB/s per SM, the zero barrier (09, 11); hi + lo doubles the MMAs |
| o, all m (6) | +13.5 | 13-21 MB kernels dominated by start and tail: a lone stage lands ~0.9 µs after entry, ~0.45 µs entry barrier (10, 11) |
| qkv m ≤ 32 (4) | +4.6 | ring overheads; at m32 W delivery while x streams is aggregate L2/DRAM-bound (11) |
| sliced unpooler m1 / m8 (2) | +3.7 | 6.4 MB kernels: fixed latency (10) |
| pooler m ≤ 32 and unpooler m16 / 32 (6) | −2.9 | already 0.91-1.02× SOL, because F11/F12 ship fewer bytes than BF16 (05, 09, 10) |

## Provenance and licensing

- **01-05** carry NVIDIA TensorRT-LLM's tinygemm2 kernel (Apache-2.0) and follow the Triton persistent-matmul tutorial pattern (MIT; no code copied).
- **06 on** start from the public file of team alygg77, `okc-muse-spark-kernels` (Apache-2.0), which itself builds on vjeux/okc-2026 (Apache-2.0, commit f35a41e), Nathen Fernandes' muse-bf16-proj routes (MIT) and code posted by Matsvei (Apache-2.0 as a challenge submission). 06-08 also carry a CUTLASS-template route (BSD-3-Clause notice).
- 06-08 (bo1, bo7b, bo11) embed 10 zlib + base64 CUDA source literals from that public base. They broke the readable-source rule of 11-02 and were withdrawn; milestone 09 is the readable rebuild. The code here is unchanged from what was judged; only attribution comments were added (see below).
- From 09 on, the judged build headers said Nathen Fernandes' MIT notice was "reproduced below" and credited Matsvei, but the lean builds had dropped the MIT text and contain no Matsvei-derived code. Here those headers are replaced by a pointer to the in-place markers.
- Every `build.py` marks each borrowed block in place (`SPDX-SnippetBegin` / `SPDX-SnippetEnd`: source, copyright, license, our changes). The root `NOTICE` lists the sources and the milestones that use them; `LICENSES/` holds the full texts.

## tools/

The harness at milestone 11:
- `TASK.md`, `reference.py`, `checks.py`, `baseline.py`: the official problem files (`okc init`, 2026-10-27).
- `bench.py`: local judge-alike (CUDA graphs, 512 MB cold-L2 read, paired with SGLang; `--xclones N` for x placements).
- `check.py --graph`: 259-case correctness gate (the judge's 9 tests, every scored shape, nearby m, normal and outlier inputs, CUDA-graph replay) with a NaN-poisoned allocator.
- `exact_all.py`: losslessness gate (injected out-of-window weights, one-hot rows must reproduce W bit-exactly). `exact_router.py`: the same for the router.

They are `muse-opt-codex/muse_bf16_proj/claude/bench.py`, `claude/t3/check.py`, `claude/t3/exact_all.py` and `claude/t3/Y/exact_router.py` as of 11-02, in flat copies that differ only in `PROB` pointing at their own directory.

```bash
cd muse_bf16_proj/tools
python3 check.py ../11_small_shape_tails_stack/build.py --graph          # correctness, 259 cases
python3 exact_all.py ../11_small_shape_tails_stack/build.py --nesc 200   # exact escapes
python3 exact_router.py ../11_small_shape_tails_stack/build.py           # exact router
python3 bench.py --cand m11=../11_small_shape_tails_stack/build.py --xclones 4   # judge-alike, all 30 shapes
```

See `dead_ends.release.md` for the failures that changed direction.
