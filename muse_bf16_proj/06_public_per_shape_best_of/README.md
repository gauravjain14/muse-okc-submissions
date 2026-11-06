# 06: a per-shape best-of on the public alygg77 file (all loads except decode-1)

| | before (05) | after (06) |
|---|---|---|
| build | `05_.../build.py` = `int_t3b` (md5 ab353ff0, 273,097 B) | `build.py` = `bo1` (md5 a5cf713f, 450,150 B) |
| judge | 1.4168 (check, job a5086e19) | **1.5690** (check, job fa06cf83) and **1.5688** (scored, job 1b299d12): same file |
| real effect | | **+10.7 %** (real level 1.5689, the mean of the two readings); identical-code shapes moved only -0.5 % / +0.3 % |
| shapes changed | | 25 of 30: everything except qkv_gate / pooler at m = 1, unpooler m = 8, and the two m = 256 shapes that stayed on the same cuBLAS kernel |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: code copied or adapted from other projects is marked in place with `SPDX-SnippetBegin` / `SPDX-SnippetEnd` comments (source, copyright, license, our changes); see `NOTICE` and `LICENSES/` at the repository root. The code is unchanged. The judged file has md5 `a5cf713f`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

`bo1` is alygg77's public `muse_bf16_proj` file (Apache-2.0, itself built on vjeux/okc-2026 and Nathen Fernandes' routes; see the root `NOTICE`) used as the base, with three of our modules dispatched in front of it for the shapes where ours measured faster. It is the largest single step on this board.

## 1. Starting point

`int_t3b` was our own stack: the 10-27 kernels (K1s/K1t register-streaming GEMVs at m ≤ 32, tuned cuBLASLt plans at m = 128/256, router kernels) plus the m = 1 F11 GEMV of milestone 05. Its judge loads (`evidence/judge_compare.txt`, from the judge files):

| load | SGLang sum (µs) | int_t3b sum (µs) | load speedup | % of score per µs saved |
|---|---|---|---|---|
| decode 1 (m = 1) | 60.42 | 33.63 | 1.797 | 0.50 |
| decode 16 | 75.75 | 56.64 | 1.337 | 0.29 |
| decode 32 | 77.03 | 60.77 | 1.268 | 0.27 |
| verify 1 (m = 8) | 75.59 | 41.18 | 1.836 | 0.40 |
| verify 16 (m = 128) | 85.88 | 71.10 | 1.208 | 0.23 |
| verify 32 (m = 256) | 102.53 | 85.56 | 1.198 | 0.19 |

The last column is 100 / (6 × load sum) (computed). decode 1 was already near its byte floor; the other five loads were 1.2-1.8x.

## 2. What we saw

**(a) The roofline said the gap was kernel efficiency, not the weight format** (`ROOFLINE.md` lines 14-58; "ROOFLINE line N" below always means `muse-opt-codex/muse_bf16_proj/claude/t3/ROOFLINE.md`, the coordinator's running log, copied unedited as (private notes, not included); model in `claude/t3/roofline.py`):

| weights streamed at their floor | judge geomean |
|---|---|
| BF16 | 1.551 |
| 12-bit lossless (our F12) | 1.865 |
| 11-bit | 1.968 |
| our base at the time (int_v7, local) | 1.404 |

The per-shape gap table put +3.2 to +5.3 µs (judge) over the 12-bit floor on every m = 16/32 qkv/pooler/unpooler shape and +5 to +7 µs on qkv/pooler/unpooler at m = 128.

**(b) The m = 8-32 kernels were latency-bound, not bandwidth-bound.** `evidence/ncu_v7.log` (ncu full set, flushed L2):

| kernel (int_v7) | shape | ncu µs | DRAM MB | TB/s (computed) | occupancy | issue | eligible warps/clk | top stalls |
|---|---|---|---|---|---|---|---|---|
| k1sp16 (K1s) | pooler m16 | 18.14 | 53.1 | 2.93 | 24.8 % | 32.4 % | 0.50 | long_scoreboard 43, wait 17, barrier 10 |
| k1tq16 (K1t) | qkv m16 | 12.51 | 29.2 | 2.33 | 24.6 % | 34.2 % | 0.53 | long_scoreboard 39, wait 20 |
| k1tp32 (K1t) | pooler m32 | 20.16 | 53.4 | 2.65 | 24.8 % | 34.8 % | 0.53 | long_scoreboard 24, wait 23, barrier 11 |
| lt128p (cuBLASLt) | pooler m128 | 22.91 | 69.2 | 3.02 | 14.1 % | 24.5 % | 0.26 | long_scoreboard 60, wait 19 |
| rw64 (router) | router m128 | 8.35 | 10.6 | 1.27 | 7.8 % | 15.0 % | 0.15 | long_scoreboard 39 |

One CTA of 16 warps per SM, 122-128 registers, half an eligible warp per scheduler: too few bytes in flight per SM, plus a serial prologue (x by TMA) and a split-K epilogue that nothing overlaps (ROOFLINE lines 61-66). The router at m = 128 streamed on only 64 of 132 SMs (line 68).

**(c) Our own rewrites of those shapes closed only part of the gap.** Three agents rebuilt m = 2-32 in our style (S pooler m ≤ 8, Q qkv m 2-16, P pooler/unpooler m 9-32) and the result `int_t3f` projected to about 1.46 on the judge (ROOFLINE lines 139-151). Agent S found its main kernel at 13.84 µs against an 11.5 µs pure stream, the rest being x delivery and a CTA barrier (line 133). W (m > 32, 12-bit RS-wgmma) stayed slower than int_v7's tuned cuBLASLt on all 8 scored shapes through four rounds (ROOFLINE lines 110-131; `claude/t3/W/NOTES.md`).

**(d) A same-run bench of three whole files showed where public code was ahead.** `evidence/bench_ext.log`: one process, our `int_t3f` (t3f), vjeux's file (vj) and alygg77's file (al), each paired with SGLang, local µs:

| load | SGLang | t3f | vj | al |
|---|---|---|---|---|
| decode 1 | 52.38 | **26.13** (x2.005) | 27.53 | 26.92 |
| decode 16 | 69.09 | 47.39 (x1.458) | 40.90 | **40.45** (x1.708) |
| decode 32 | 71.15 | 51.38 (x1.385) | 45.00 | **44.79** (x1.588) |
| verify 1 | 69.33 | 33.64 (x2.061) | 32.29 | **29.88** (x2.321) |
| verify 16 | 79.39 | 65.16 (x1.218) | 61.87 | **59.21** (x1.341) |
| verify 32 | 94.62 | 79.02 (x1.198) | 81.33 | **74.33** (x1.273) |
| geomean, judge-projected (+1.0 µs/shape) | | 1.4594 | 1.5251 | **1.5835** |

Per shape (`evidence/bestof_projection.txt`), ours won only qkv_gate m1 (6.75 vs 7.28), pooler m1 (11.21 vs 11.73) and the unpooler slice at m8 (2.50 vs 2.87). Of the other 27 shapes alygg77's file won 20 and vjeux's 7. On the qkv/pooler/unpooler shapes at m = 8-32 the public kernels were 1.4-2.1 µs faster than ours, and they won every m = 128/256 shape.

**(e) Why the public kernels were faster at m = 8-32.** alygg77's m = 8-32 kernel is vjeux's `sk2`: a persistent ring per SM, fed by TMA bulk copies of 12-bit weights, consumed by `mma.sync` warps, with separate escape warps. Agent X's port and ablations (`notes/` of milestone 07, X iteration 0) measured pooler m32 at 13.76 µs against 12.83 µs for the same ring with no x and no compute. That ring streams at about 96 % of 4.8 TB/s in its bulk, against our K1s/K1t at 2.3-2.9 TB/s under ncu. At m = 128 the public file had hand-written TMA + wgmma kernels (`qb_kernel` for qkv, `ol_projection` for o) and an `rk` router GEMM spread over 96-128 CTAs; at m = 256 it pinned cuBLASLt plans and added a CUTLASS route for pooler. In both bo1 judge runs pooler m256 ran a cuBLAS `nvjet` kernel, so the CUTLASS route evidently did not engage on the judge (inferred).

## 3. Diagnosis and projection

- The shapes outside decode 1 were limited by kernel structure (bytes in flight, prologue/epilogue serialization), and a public file already contained structurally better kernels for them.
- So the step was a **per-shape best-of**, not a new kernel: take the public file whole and dispatch our faster modules in front of it.
- **Projection** (`evidence/bestof_projection.py` on `bench_ext.log`): the oracle best-of is 1.6000 judge-projected; the `bo1` composition (alygg77 everywhere except our m = 1 qkv/pooler GEMV and our m ≤ 8 unpooler slice) is **1.5952**, i.e. +12.6 % over int_t3b's 1.4168.
- Agent M's unpooler m128 kernel was measured separately at 17.39 vs 17.83 µs against int_v7's route (`notes/M_NOTES.md`); against alygg77's 17.51 µs in `bench_ext.log` it is worth about 0.1 µs (inferred), negligible in the projection.

## 4. Change

`bo1.py` is assembled by `claude/t3/build_t3.py`. Each module's top-level names get a prefix so modules cannot collide (`_b7_` = the base file, `_g1_` = G, `_mq_` = M). `kernel_fn` is an if-chain (end of `build.py`):

| shapes | module | kernel (judge profile) | origin |
|---|---|---|---|
| qkv_gate, pooler at m = 1 | G `gemv1` | `g1_qkv_gate_g311`, `g1_pooler_g311` | ours: F11 11-bit lossless many-CTA GEMV (milestone 05) |
| unpooler, 2 ≤ m ≤ 8 (the 1024-row slice) | G `gemv1` (tcm) | `g1m_unpslice` | ours: F12 16-row tensor-core tiles, 64 CTAs × 1024 threads |
| unpooler, 97 ≤ m ≤ 128 | M `m128` | `mk_unpooler` | ours: 12-bit RS-wgmma, split-K 2 as a cluster pair with a DSMEM reduce-scatter (`notes/M_NOTES.md`) |
| everything else | alygg77 file | `cgemv`, `rgemv`, `sk2`, `dd_cap2`, `rr_small`, `v18s_gemv`, `smallm2_gemv`, `qb_kernel`, `ol_projection`, `rk_kernel`, `rl_early_projection`, cuBLASLt `nvjet` plans | public, Apache-2.0 / MIT (`NOTICE`) |

In `build.py` every block from these sources is marked in place, and the upstream notices are kept verbatim inside the marked blocks; `evidence/ext_HEADER_bestof.txt` is the header the judged file carried.

Agent M's module is the one new kernel here. Its lessons, measured in its lane:
- The decode had to sit between `WARPGROUP.DEPBAR` and the next HGMMA in one basic block. Branches or try-wait loops that split the block cost about 3 µs.
- Writing any future wgmma input register between `wgmma.fence` and the wait makes ptxas 13 serialize every HGMMA (C7513).
- A split-K exchange of fp32 partials through global memory plus a converter kernel cost 3.3-3.8 µs. A cluster-pair DSMEM reduce-scatter (32 KB per CTA, one bulk copy) cost 3.5 µs less.
- Its data-movement probe (`evidence/M_mprobe.py`): x alone drains 512 KB/SM in about 4.3 µs, x+W 906 KB/SM in about 11 µs. Moving W to LDGSTS changes nothing (13.07 vs 13.10 µs), so the W HBM stream is the limit.

**Encoded source in the public base (stated plainly).** alygg77's file ships several CUDA kernels as zlib-compressed, base64-encoded string literals that are decompressed at import. `evidence/blob_inventory.py` finds **10** such literals in `build.py`. They cover the kernels behind qkv m128 (`qb_kernel`), o m128 (`ol_projection`), o m256 (`rl_early_projection`), the routers (`rr_kernel`, `rr_small_kernel`), o m8 (`v18s_gemv`), pooler m8 (`smallm2_gemv`), pooler m16 (`dd_cap2_b_cpssm_gemv`), `smtrace_gemv` and `m32w_gemm`. Eight of the 30 judged shapes run code from them: qkv m128, o m8/m128/m256, pooler m8/m16, router m8/m16 (kernel names in `evidence/judge_after_*`).

On 11-02 the challenge rules changed to require readable source with no compressed or encoded code (ROOFLINE lines 271-276). That made this build, and 07 and 08 which inherit the same literals, non-compliant. They were withdrawn and replaced by the readable rebuild of milestone 09. `build.py` here stays byte-identical to the judged file, literals included.

## 5. Result

| check | result |
|---|---|
| local projection (`evidence/bestof_projection.txt`) | 1.5952 judge-projected at the +1.0 µs/shape calibration |
| correctness | judge 9/9 tests in both runs. The sibling builds with this base passed `check.py --graph` (259 cases × 2 seeds, NaN-poisoned outputs; ROOFLINE line 173). `exact_all` would flag G's m = 1 / slice kernels, which round out-of-window weights below ~2^-39 to 0; that never happens for real weights (ROOFLINE lines 142-145). Every exactness gate was first passed at milestone 10 (inferred for this build) |
| judge | 1.4168 → **1.5690** (check fa06cf83) / **1.5688** (scored 1b299d12) |
| judge per load (check / scored) | decode 1 1.797 → 1.799 / 1.792, decode 16 1.337 → 1.585 / 1.595, decode 32 1.268 → 1.503 / 1.505, verify 1 1.836 → 2.107 / 2.077, verify 16 1.208 → 1.316 / 1.325, verify 32 1.198 → 1.255 / 1.259 |
| largest per-shape moves (judge µs, before → check) | pooler m16 16.72 → 13.57, unpooler m32 17.54 → 14.31, pooler m32 18.07 → 15.02, router m256 9.82 → 7.09, qkv m128 15.89 → 13.56, router m128 8.07 → 5.82, unpooler m16 16.30 → 14.10, qkv m8 11.22 → 9.06 |
| **judge, read honestly** | The 5 shapes whose kernel did not change moved -0.48 % (check) and +0.29 % (scored); SGLang moved +0.06 % / +0.27 %. The 25 changed shapes moved -11.7 % / -11.6 %. Removing both drifts re-reads the two runs at 1.5604 / 1.5693 (`evidence/judge_compare.txt`). Two same-code runs 8 minutes apart agreed to 0.01 %. So the +10.7 % is real, not node luck. Among the "unchanged" shapes, qkv m256 ran the same cuBLAS `nvjet` kernel but read 18.51 → 17.17 / 17.07. qkv m256 is the shape later found to move by up to ~1 µs with the physical placement of identical code (ROOFLINE lines 187-188, 261-263), not code |
| vs projection | Read 1.5689 against 1.5952 projected. Against `bench_ext.log`'s own times, the two judge runs measured +1.37 µs per shape on our side and +1.39 on SGLang's (mean over 30 shapes, computed), not the +1.0 the projection assumed. Re-projecting the `bo1` composition at +1.4 µs/shape gives **1.5687**, which is the judge reading. So the whole gap is the fixed per-shape offset, which compresses every ratio, not a kernel that failed to transfer |

Where the time still went (`evidence/dash_bo1.md`; nsys in-graph time, ncu bytes and stalls, floor = 0.81 µs + DRAM bytes / 4.56 TB/s):
- The 30 kernels sat 76.7 µs over their stream/MMA floors in total.
- The sk2 rings at m = 8-32 were +1.7 to +3.5 µs each, with tensor pipe 9-28 % and issue 26-37 %.
- `sass_bo1.txt`: the sk2 hot loops are dominated by decode integer ops (94-545 per kernel body) and LDS, with only 8-64 HMMA: issue/latency-bound around an idle tensor pipe.
- m = 128: qkv +3.9 µs (`qb_kernel`, 2.96 TB/s in-graph), pooler +3.5 (cuBLAS), unpooler +6.0 (M).
- m = 256: +4.4 to +6.3 µs on the cuBLAS kernels.

## 6. Takeaways, and what came next

- **Measure whole public files against yours, same run, per shape, before writing kernels.** One `bench_ext` run (three candidates, 30 shapes) located a +8.5 % judge-projected difference that our own lanes (S, Q, P at m ≤ 32, W at m > 32) had not closed.
- **The ring design was the transferable idea.** Persistent per-SM ring, TMA bulk copies, compressed weights, `mma.sync` consumers. Our register-streaming kernels had half an eligible warp per scheduler; the ring kept DRAM at about 96 % during its bulk.
- **Calibrate the offset per build.** At this level the judge sat about 1.4 µs/shape above this bench's local times, not 1.0. The ratio projection is sensitive to it on short kernels.
- **Provenance and compliance travel with the base.** The base's encoded literals had to be removed once the rules required readable source (milestone 09).
- **Next ([07](../07_sk11_register_escape_ring/)):** with the public kernels as the base, the lanes attacked what `dash_bo1.md` showed: escape-warp work in the sk2 ring (agent F's sk11), the routers (Y), m = 128 pooler (Z), m = 256 qkv (V), and o at m = 1 (G's tensor-core tile kernel).

## Reproduce

All you need is torch and an H200; `tools/` holds the problem spec, reference, checks, SGLang baseline and our harness. Run from `tools/`:

```bash
cd muse_bf16_proj/tools
python3 bench.py --cand base=../05_f11_gemv_m1_no_load_sinking/build.py \
                 --cand new=../06_public_per_shape_best_of/build.py --reps 30 --xclones 4   # same-run, 30 shapes
python3 check.py ../06_public_per_shape_best_of/build.py --graph                            # 259 cases x 2 seeds + graph replay
python3 exact_all.py ../06_public_per_shape_best_of/build.py --nesc 200                     # losslessness (flags G's m=1 kernels)
python3 exact_router.py ../06_public_per_shape_best_of/build.py
python3 ../06_public_per_shape_best_of/evidence/bestof_projection.py ../06_public_per_shape_best_of/evidence/bench_ext.log
python3 ../06_public_per_shape_best_of/evidence/judge_compare.py ../06_public_per_shape_best_of/evidence/judge_*.txt
python3 ../06_public_per_shape_best_of/evidence/blob_inventory.py ../06_public_per_shape_best_of/build.py
```

`evidence/M_mprobe.py` needs agent M's lane helpers (`cu.py`); run it from its original directory.

Originals (relative to ``):
- build: `muse-opt-codex/muse_bf16_proj/claude/t3/bo1.py`, assembled by `claude/t3/build_t3.py`;
- public files as staged: `claude/t3/ext/alygg_bf16.py`, `ext/vjeux_bf16.py`; upstream copy with LICENSE/NOTICE: `muse-opt-codex/muse_decode_layer/sources/alygg77/`;
- lanes: `claude/t3/G/`, `claude/t3/M/`;
- dashboard: `claude/t3/dash_bo1*.{md,log}`, `claude/t3/sass_bo1.txt`;
- judge results: `claude/t3/profiles/bo1.{fa06cf83e343466f,1b299d120f5d4dfe}.txt`, `claude/t3/profiles/int_t3b.a5086e19abb541ba.txt`.
