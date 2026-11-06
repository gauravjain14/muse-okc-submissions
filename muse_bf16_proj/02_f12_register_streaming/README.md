# 02: 12-bit lossless weights streamed into registers (K1s) at m ≤ 16, plus a dedicated router

| | before (01) | after (02) |
|---|---|---|
| build | `01_.../build.py` = Codex `candidate_v2` (md5 ef833cda) | `build.py` = `int_v1.py` (md5 19dd35d6, 178,957 B): Codex v2 paths + K1s + the router worker's `router.py` |
| judge | 1.2471 (job 70a24525, check) | **1.3112** [1.2915, 1.3308] (job 97b5d8aa, check) |
| real effect | | **+5.1 %** judge; same-run local judge-projected 1.3195 vs 1.2604 (+4.7 %) |
| shapes changed | | qkv m 2-16 and pooler m 1-16 (scored: qkv m8/m16, pooler m1/m8/m16), unpooler m 9-16 (scored: m16): K1s. Router: all m. Everything else is still Codex v2 |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: code copied or adapted from other projects is marked in place with `SPDX-SnippetBegin` / `SPDX-SnippetEnd` comments (source, copyright, license, our changes); see `NOTICE` and `LICENSES/` at the repository root. The code is unchanged. The judged file has md5 `19dd35d6`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

Terms used below:
- **K1** is the Claude workstream for the m ≤ 32 BF16 layers. k1c ... k1s are its successive kernel files.
- **12-bit format**: each BF16 weight stored losslessly in 12 bits (the low byte + a 4-bit exponent code).
- **Escape / raw tile**: a weight whose exponent falls outside the code's window, and the tile that holds it.
- **cx**: Codex v2 measured in the same run.

## 1. Starting point

Judge times of the shapes this milestone targets, from `evidence/` of milestone 01 (`judge_after_candidate_v2_check_70a24525.txt`), µs:

| shape | SGLang | Codex v2 | floor | Codex v2 kernel |
|---|---|---|---|---|
| pooler m = 16 | 21.38 | 18.96 | 14.06 | Triton TMA 17.5 + split-K reduce 1.1 |
| unpooler m = 16 | 18.74 | 18.02 | 14.06 | cuBLAS (`torch.mm`, transposed W) |
| pooler m = 8 | 21.60 | 18.78 | 14.02 | TinyGEMM |
| qkv m = 8 / 16 | 14.08 / 13.71 | 13.10 / 12.06 | 7.90 / 7.93 | TinyGEMM / Triton TMA + reduce |
| routers m = 1, 8, 16, 32, 128, 256 | 10.40, 26.02, 14.56, 15.36, 16.26, 17.30 | 4.02, 6.76, 6.50, 7.71, 9.51, 12.66 | 1.75-2.68 | SIMT / interleaved hi+lo dot + FP32 reduce |

## 2. What we saw

**(a) The local bench had to be calibrated first.** `claude/bench.py` times every shape as a CUDA graph after a 512 MB cold-L2 read and subtracts the empty-graph cost (5.25 µs). It pairs each candidate with SGLang in the same run. On Codex v2 it gave **1.2765** local against the judge's 1.2471 (`evidence/calib.log`). The rule adopted from this: judge ≈ local + 1.0 µs per shape on both sides (`notes/PLAN.md`, "Measured facts").

**(b) The BF16 floor was already reached at m = 1; the 12-bit floor was 17-24 % lower.** `evidence/bw1.log` streams the actual byte counts with no math, cold L2, for several patterns: LDG floods of 132-264 CTAs and a single-thread bulk-copy ring of 4-16 stages:

| weights | BF16 bytes, best µs | 12-bit bytes, best µs | Codex v2 at m = 1, local (`calib.log`) |
|---|---|---|---|
| pooler / unpooler | 67.11 MB, 15.23 (ring 6 × 32 KB) | 50.33 MB, 12.54 (LDG 264 × 512) | pooler 15.85 |
| qkv_gate | 37.75 MB, 9.57 | 28.31 MB, 7.30 | 9.75 |
| o | 16.78 MB, 4.45 | 12.58 MB, 3.68 | 4.49 |
| unpooler slice | 8.39 MB, 2.66 | 6.29 MB, 2.18 | 2.82 |

Codex v2's m = 1 GEMVs were within 0.04-0.62 µs of the BF16 streaming floor (computed). Hence the PLAN.md line: "BF16-only cannot win m ≤ 32; lossless compression is required". The 12-bit payload removes 25 % of the bytes and 17-24 % of the floor time (computed).

**(c) The weights compress well enough for a fixed 12-bit code.** Each tile is 8 rows × 32 k:
- **Low byte:** exponent bit 0 plus the 7 mantissa bits, stored raw.
- **4-bit code:** the sign plus (exponent >> 1) − H, for a per-matrix 16-exponent window [2H, 2H+15].
- **Decode:** the BF16 bits are rebuilt in registers with shifts, masks and adds plus 4 PRMTs per 8 weights (`decode12` in `build.py`; "decode 12 ALU/8 weights" in `notes/memory_bf16-proj-findings.md`).
- **Escapes:** weights outside the window number 1351 / 830 / 2306 / 3299 in qkv / o / pooler / unpooler (`evidence/k1c2.log`, `[k1c]` lines).
- **Raw tiles:** at tile granularity, 2.3-3.3 % of tiles hold an escape and are stored raw, with exact BF16. Pooler is then 51.90 MB with headers and overflow (`evidence/k1t5.log`, `[k1t]` lines: the same format, logged by the next kernel generation; `notes/memory_bf16-proj-findings.md` gives "3.5 % of tiles").

**(d) TMA-ring kernels lost; register streaming won.** The K1 workstream wrote eight kernel generations in about 70 minutes (`evidence/k1*.log`; local µs, same run within each row):

| kernel (log, time PT) | design | pooler m16 / pooler m8 / qkv m16 | BF16 same kernel | Codex v2 same run | what it taught |
|---|---|---|---|---|---|
| k1c (`k1c2.log`, 11:14) | 1 producer thread: bulk-copy W ring + TMA x; mma.sync m16n8k16 (x = A, W rows = B); escapes fixed in the epilogue from a per-CTA list | 21.34 / 18.42 / 13.75 | 17.57 / 18.04 / 11.98 | | 12-bit slower than BF16 in the same ring |
| k1c, no-escape control (`k1c3.log`) | escapes skipped (wrong, timing only) | 17.10 (pooler m16) vs 22.10 | 18.45 | | the escape epilogue alone cost ~5 µs |
| k1d (`k1d1.log`, 11:24) | escape block per stage, corrected from smem | 24.89 / 23.61 / 20.72 | 18.91 / 18.37 / 12.17 | 18.60 / 18.30 / 11.90 | worse |
| k1f (`k1f1.log`, 11:44) | two rings (deep W, shallow x), producer warps | 21.75 / 22.03 / 15.55 | 18.56 / 18.49 / 12.98 | 17.82 / 17.42 / 11.62 | still behind cx |
| k1g (`k1g2.log`, 11:51) | k1f family, tuned | 16.93 / 17.37 / 13.02 | 16.99 / 17.36 / 12.53 | | 12-bit = BF16: the ring, not the bytes, binds |
| k1r (`k1r3.log`, 12:12) | **no smem ring**: each warp streams its own contiguous W slice straight into registers (D k32-steps in flight); x resident in smem via one TMA | 16.90 / 15.68 / 13.66 | 18.12 / 17.07 / 12.86 | 18.08 / 16.86 / 11.69 | first 12-bit win on pooler |
| **k1s** (`k1s2.log`, 12:20) | k1r + x delivered by chunked TMA in k-step order; raw-tile overflow issued with x | **15.63 / 15.29 / 11.34** | 19.15 / 18.87 / 14.10 | 17.83 / 17.62 / 11.77 | wins all three; ships |

The memory note gives the reason the rings lost: "per-stage overhead + bunched arrivals" (`notes/memory_bf16-proj-findings.md`). With register streaming, every warp keeps its own loads in flight and the hardware interleaves them. A ring delivers whole stages to one consumer set at a time.

**(e) K1s itself was latency-bound.** ncu of K1s, qkv m = 8 (`evidence/ncu1.log`, 12:42):

| metric | value |
|---|---|
| duration | 12.90 µs |
| DRAM throughput | 47 % of peak (2.27 TB/s) |
| grid | 132 CTAs × 512 threads, 128 registers/thread, 1 CTA/SM |
| achieved occupancy | 24.8 % |
| eligible warps per scheduler | 0.50 |
| issue slots busy | 25.8 % |
| top stall samples | long_scoreboard 440, wait 216, barrier 112 |

The kernel won on bytes but left bandwidth on the table. Milestone 04 returns to this.

**(f) The router was a separate worker.** Its notes are `notes/router_NOTES.md` (R = the router worker):
- m ≤ 4: `rs`, exact FP32 SIMT, 2 rows × full K per CTA, sitting at the 8.4 MB floor.
- 5 ≤ m ≤ 64: `rt`, mma.sync on the BF16 hi/lo pair with split-K. It runs 128 adder CTAs plus 4 "zero CTAs" in one kernel: the zero CTAs clear `out` and release a counter, and the adders `red.add` after an acquire-poll.
- m > 64: `rw`, wgmma with W hi/lo as A, plus a tiny `rzero` kernel and a PDL launch.

Measured facts that shaped it:
- A global split-K fixup costs +1.4 µs at small m and +4 µs at m = 256.
- In-kernel zero CTAs are about 0.8 µs better than PDL at m = 8-64.
- First weight bytes land 1.5-1.8 µs after kernel start.
- A lossless 24-bit hi/lo compression was exact but gained nothing: the stream is latency-bound at m ≤ 64.

## 3. Diagnosis and projection

- At m ≤ 16 the four BF16 layers are pure weight streams: x is at most 256 KB.
- Codex v2 was 4.0-4.9 µs over the judge's BF16 floor on pooler/unpooler m16 and pooler m8, and the 12-bit floor is about 2.7 µs lower still (computed from section 1 and (b)).
- So a kernel streaming 12-bit weights at the probe's rate could save up to 6.7-7.6 µs on each of those three shapes (inferred upper bound). The k1s run measured −2.2 to −2.3 µs against Codex v2 on pooler m8/m16.
- The router projection came from its own same-run A/B. Against Codex v2 in the `int1.log` run (local): m8 4.21 vs 5.88, m16 4.11 vs 5.46, m32 4.38 vs 6.22, m128 6.73 vs 8.56, m256 8.93 vs 11.86.
- **Projection for the integrated build** (`evidence/int1.log`, same run, judge-projected): int_v1 1.3195 vs Codex v2 1.2604, i.e. **+4.7 %**.

## 4. Change

`build.py` = `claude/build_sub.py` output: Codex v2 kept whole as a fallback, plus two components.

1. **K1s** (12-bit, `_k1_*`).
   - **Grid:** 132 CTAs × 16 warps. Config `(C, Q, GW, WR, WK, D)`:
     - qkv `(2, 66, 5, 1, 16, 4)`: 2-CTA clusters split K; partner halves are pushed through DSMEM.
     - pooler `(2, 66, 4, 2, 8, 4)`; pooler at m ≤ 8 `(1, 132, 4, 1, 16, 4)`; unpooler `(1, 132, 4, 2, 8, 4)`.
   - **Per warp:** each warp owns a contiguous slice of 12-bit tiles. It keeps D = 4 k32-steps of loads in flight (`ld.global.nc.L1::no_allocate`), decodes in registers, and issues mma.sync m16n8k16 with x from swizzled smem.
   - **Raw tiles:** their 4 extra bytes per lane come from a per-CTA overflow area in smem.
   - **Output:** partials go to padded smem slices, then a CTA sum, then the cluster push, then BF16 out.
   - **Dispatch:** `_K1_USE`: qkv 2 ≤ m ≤ 16, pooler 1 ≤ m ≤ 16, unpooler 9 ≤ m ≤ 16. o and unpooler m ≤ 8 stay on Codex v2.
2. **Router** (`_rt_*`, from `claude/router/router.py`): `rs` / `rt` / `rw` as in 2(f), all exact or hi+lo with FP32 accumulation, one or two launches.

Both compile at `prepare()` through NVRTC (cuda-python `cuda.bindings`, the judge's toolchain) and launch on torch's current stream, so they are graph-capturable.

## 5. Result

| check | result |
|---|---|
| local A/B, same run (`evidence/int1.log`, 30 reps) | judge-projected **1.3195** vs Codex v2 **1.2604**. Per load: decode 16 x1.389 vs x1.251, verify 1 x1.814 vs x1.599, decode 32 x1.270 vs x1.218 |
| correctness | "all ok" from the correctness sweep that precedes the bench in `int1.log` (inferred to be `claude/t_k1.py`, the only script in `claude/` that prints it: m 1, 2, 5, 7, 8, 9, 15, 16, 17, 31, 32 × five layers × normal and outlier inputs through `reference.check_implementation`); judge tests 9/9 |
| judge | 1.2471 → **1.3112** (+5.14 %), `evidence/judge_before_*` vs `evidence/judge_after_*` |
| judge, changed shapes (µs) | pooler m16 18.96 → 16.61 (−2.35); unpooler m16 18.02 → 16.02 (−2.00); pooler m8 18.78 → 15.94 (−2.84); qkv m8 13.10 → 12.27 (−0.83); qkv m16 12.06 → 11.97 (−0.09); pooler m1 16.68 → 16.42 (−0.26); routers m1 −0.43, m8 −1.30, m16 −1.19, m32 −1.80, m128 −1.39, m256 −2.30 |
| **judge, read honestly** | Eight shapes run identical cuBLAS code in both builds (qkv/o/pooler/unpooler at m = 128 and 256). They moved by −0.50 to +0.60 µs (mean +0.14). The Triton o kernels at m = 8/16 (identical code) moved +0.34 / +0.41. So this node was neutral to slightly slow, and per-shape noise is about ±0.6 µs. The changed shapes moved 2-6x further than that. **The +5.1 % is real.** |
| vs projection | +5.14 % judge vs +4.7 % projected. The projection held. |

Kernel view in the judge profile of int_v1: `k1s_pooler_16_1` 16.8 µs, `k1s_unpooler_16_1` 15.6 µs, `k1s_pooler1_8_1` 15.7 µs (m8) and 16.8 µs (m1), `k1s_qkv_gate_16_1` 11.7 µs. The routers: `rs1` 2.7, `rt16_f16` 4.3, `rw64` + `rzero` 5.8 + 0.9 µs.

## 6. Takeaways, and what came next

- **Fewer bytes only pay if the kernel stays a stream.** Five ring designs (k1c-k1g) carried the 12-bit format and still lost or tied, because the ring's per-stage overhead dominated. Register streaming let 16 warps per SM each keep their own loads in flight, and only then did the 25 % byte saving show up.
- **Measure the escape path on its own.** Skipping escapes made k1c 5 µs faster on pooler m16. Escape handling has to be off the critical path: here, raw tiles plus an smem overflow read alongside x.
- **K1s is not at the floor.** ncu shows 25 % occupancy, 0.5 eligible warps and 47 % DRAM: it is latency-bound. K1s at m = 1 (pooler 16.42 µs judge) loses to the BF16 floor (13.99) because one wave of 132 CTAs cannot hide its start and epilogue.
- **Next, 03:** at m = 128/256 the custom-GEMM workers (K2, K2c) were losing to cuBLAS. The remaining lever there was the library configuration itself. At m = 32, K1s was extended to 32-token tiles (K1t).

## Reproduce

Needs an H200 and the SGLang image (torch 2.14 + CUDA 13.0, Triton 3.8, cuda-python, `sglang` for the baseline). `prepare()` builds the Codex extensions with `load_inline` (nvcc + ninja) and compiles K1s and the router with NVRTC. The build reads no files outside itself.

```bash
cd muse_bf16_proj/tools
python3 check.py ../02_f12_register_streaming/build.py --graph
python3 bench.py --cand v2=../01_library_dispatch_table/build.py --cand int_v1=../02_f12_register_streaming/build.py --reps 30
python3 bench.py --cand v2=../01_library_dispatch_table/build.py --cand int_v1=../02_f12_register_streaming/build.py \
       --shapes 8,16 --layers pooler,unpooler,qkv_gate --reps 60      # the K1s shapes only
```

Originals (relative to ``):
- build: `muse-opt-codex/muse_bf16_proj/claude/int_v1.py` (assembled by `claude/build_sub.py`)
- K1 kernels: `muse-opt-codex/muse_bf16_proj/claude/k1*.py` (k1s = `claude/k1s.py`)
- router: `claude/router/router.py` + `NOTES.md`
- logs: `claude/out/*.log` (copied: `calib`, `bw1`, `k1c2`, `k1c3`, `k1d1`, `k1f1`, `k1g2`, `k1r3`, `k1s2`, `int1`, `ncu1`)
- binary ncu report: `claude/out/20261027T124249_ncu1/k1s_qkv8.ncu-rep`
- judge: `muse-opt-codex/muse_bf16_proj/claude/profiles/int_v1.97b5d8aaaece47c7.{txt,json,log}`
