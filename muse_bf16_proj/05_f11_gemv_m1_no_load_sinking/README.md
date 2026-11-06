# 05: an 11-bit lossless format and load-sinking-free GEMV kernels at m = 1

| | before (04) | after (05) |
|---|---|---|
| build | `04_.../build.py` = `int_v7.py` (md5 558722c4) | `build.py` = `int_t3b.py` (md5 ab353ff0, 273,097 B) = int_v7 + module G (`claude/t3/G/gemv1.py`) |
| judge | 1.3999 (job 71a5d9d6, check) | **1.4168** [1.3955, 1.4380] (job a5086e19, check) |
| real effect | | **+1.2 %** judge, matching the +1.1-1.4 % projected; decode-1 load 1.684 → 1.797 |
| shapes changed | | qkv_gate, o, pooler and the unpooler slice at m = 1; the unpooler slice at 2 ≤ m ≤ 8 (scored: verify 1, m = 8). Nothing else |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: code copied or adapted from other projects is marked in place with `SPDX-SnippetBegin` / `SPDX-SnippetEnd` comments (source, copyright, license, our changes); see `NOTICE` and `LICENSES/` at the repository root. The code is unchanged. The judged file has md5 `ab353ff0`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

Terms used below:
- **t3**: the second Claude push on this problem, which started on 2026-11-02 at about 01:00 PT after a five-day pause (work dir `claude/t3/`).
- **G**: its m = 1 lane.
- **F12**: the 12-bit format of 02-04.
- **F11**: an 11-bit format: the low byte plus a 3-bit code (sign + 2 exponent bits) over an 8-exponent window.
- **g311**: G's FMA GEMV kernel on F11.
- **tc**: G's tensor-core tile GEMV on F12.
- **Load sinking**: the compiler (ptxas) placing global loads after the first FMA/MMA that consumes earlier data, so each load becomes a serialized DRAM round trip.

## 1. Starting point

int_v7's m = 1 kernels and the 12-bit floor. Local times are from the t3 re-bench (`evidence/bench_int_v7_base_v7.log`); judge times are from `evidence/judge_before_int_v7_check_71a5d9d6.txt`; µs:

| shape | SGLang (judge) | int_v7 judge | int_v7 local | floor12, judge units (`evidence/roofline_excerpt_lines1-146.md`) | kernel |
|---|---|---|---|---|---|
| qkv m1 | 13.82 | 8.78 | 7.83 | 8.02 | GV3 (2, 512) |
| o m1 | 6.88 | 5.28 | 4.51 | 4.57 | GV3 (8, 64) |
| pooler m1 | 21.70 | 13.76 | 12.67 | 12.85 | GV3 (2, 512) |
| unpooler slice m1 | 6.85 | 3.58 | 2.58 | 3.19 | GV3 (1, 256, 8) |
| unpooler slice m8 | 7.01 | 4.79 | 3.94 | 3.21 | TinyGEMM (BF16) |

## 2. What we saw

**(a) A roofline over all 30 shapes said the remaining gap was kernel efficiency, not format** (`evidence/roofline_excerpt_lines1-146.md`, lines 6-23; model in `evidence/roofline.py`). The model:
- **Cold streaming:** local t = 0.81 µs + bytes / 4.56 TB/s, a least-squares fit of the 6.3-67 MB probes.
- **Judge:** local + 1.0 µs, calibrated on Codex v2, int_v5 and int_v7.
- **Tensor:** 132 SMs × 4096 FLOP/clk at 1.755 GHz under tensor load, 85 % achievable.
- **Weight entropy:** measured over all four BF16 layers at **10.46 bits/weight** (exponent 2.54 b, sign 1, mantissa 7).

The judge geomean if every shape sat at its floor:

| weights at the floor | geomean |
|---|---|
| BF16 | 1.551 |
| 12-bit (F12) | **1.865** |
| 11-bit | 1.968 |
| entropy (10.46 b) | 2.021 |
| int_v7 now (local + 1.0) | 1.404 |

The value of 1 µs is highest at decode 1 (0.47 % per µs), then verify 1 (0.39). At m = 1 int_v7 was already only 0.39-0.94 µs over the 12-bit floor (the gap table, lines 25-59). From there, only fewer bytes or less fixed cost could pay.

**(b) Profiles of int_v7 at m = 1.**
- **ncu, GV3 pooler** (`evidence/ncu_int_v7.log`): 50.49 MB in 15.23 µs under ncu, 2048 CTAs × 512 threads, 30 registers, 88 % occupancy, 57.6 % issue, long_scoreboard 47 % then barrier 16 %. It is a stream.
- **nsys, in-graph kernel time** (`evidence/nsys_int_v7.log`): qkv 8.59, o 5.20, pooler 13.50, unpooler slice 3.39 µs. SGLang takes 14.16, 6.38, 20.64, 6.77 µs, including its split-K reduce kernels.

**(c) Launch and streaming probes** (`evidence/launchprobe.py`, excerpt lines 81-96; local µs after the flush, empty graph subtracted):
- An empty kernel of 2048 × 512 CTAs costs +1.09 µs of CTA launch throughput; 1056 × 128 costs +0.64.
- A pure LDG stream of 50 MB takes 12.00 µs at 2048 × 512, 12.91 at 264 × 512 and 13.42 at 132 × 512.

Many short CTAs win on large streams; on the 6 MB slice, 2048 × 512 is the worst (4.00 vs 2.11 µs).

**(d) G's structure sweep at m = 1** (`notes/G_NOTES.md`, "(A) kernel structure"; local µs, GV3 control in the same runs):

| layer | best new structure | GV3 control | verdict |
|---|---|---|---|
| o | `tc(1, 8)`: CTA = one 16-row F12 tile, 8 warps split K = 1024, 96 B/lane issued up front, 48 regs, 512 CTAs = one wave | 3.76-4.07 vs 4.30-4.54 | −0.35 to −0.49 in 5 runs; one-wave configs only |
| unpooler slice | `tc(1, 32)`, 64 CTAs × 1024 threads | 2.42-2.66 vs 2.46-2.68 | equal |
| qkv / pooler | `tc`, `tb`, one-warp-per-row `rw`, F12 GV3 variants | best 8.05 / 13.2 vs 7.73-7.95 / 12.41-12.75 | GV3's many small CTAs stay best |

**(e) The 11-bit format** (`notes/G_NOTES.md`, "(B)" and "F11 format spec"):
- **Window:** per matrix, the unaligned 8-exponent window holding the most weights.
- **Escapes:** 1.25-1.36 % of weights fall outside it.
- **Size with escape slots:** 11.45-11.58 bits/weight, against 12.03-12.13 for F12, i.e. 4.5 % fewer bytes.
- **Decode:** a "fake" BF16 = true × 2^(120 − E_lo), exactly. BF16 × BF16 products are exact in FP32, so Σ x · fake × 2^(E_lo − 120) rounds like the true sum. The cost is 33 ALU ops per 32 weights, against 28 for F12.
- **Round trip:** bit-exact for all four layers and for stress matrices (`evidence/roundtrip_test.py`).

**(f) The first F11 kernels were slower than F12 GV3, and the cause was load sinking, not the format.**
- The first g311 / g311b measured 8.4-10 µs (qkv) and 13.2-15.4 µs (pooler), against 7.8 / 12.6 for GV3.
- `__launch_bounds__(NT)` without a min-blocks value lets ptxas pick a low register target and sink loads below the first FFMA/HMMA.
- `evidence/sasschk.py` and `evidence/nvrtc_sass.py` compile with NVRTC 13.0, the judge's toolchain, and count global loads issued after the first compute instruction:
  - GV3 o (8, 64): **6 of 19** late;
  - GV3 qkv: 1 of 7;
  - tc11w without min-blocks: 8 of 13;
  - g311b without min-blocks: 5 of 10.
- With min-blocks set so that every load is hoisted:

| layer | kernel | local µs | GV3 control, same run |
|---|---|---|---|
| qkv | g311 R=1, 256 threads, min-blocks 6 (11-bit, 40 regs) | **7.16** | 7.81 |
| pooler | g311 R=1, min-blocks 6 | **11.71** | 12.54 |
| o | tc F12 (1, 8), min-blocks 4, W pointer baked into the code | **3.72-3.85** | 4.40-4.45 |
| unpooler slice | g311 R=1, 128 threads, min-blocks 12 | **2.48** | 2.60 |

Tiny source changes flip this. Folding the escape FMA into the main chain of g311 sank 2 loads, and qkv went to 8.03 µs from 7.22.

**(g) x delivery matters at m = 1** (`notes/G_NOTES.md`, "x delivery" table; probes with x loads replaced by constants):
- Baking the W pointer as an NVRTC constant issues W loads 0.2-0.45 µs earlier. The kernel-parameter bank is a cold miss after the flush (agent R's finding, excerpt lines 99-101).
- But then x, whose address comes from that cold parameter bank, queues behind the W flood: **+1.2-1.4 µs** on the multi-wave GV3/g311 kernels.
- So the W pointer is baked only for the one-wave `tc` kernel (o), and x is never issued after W.

## 3. Diagnosis and projection

- At m = 1 the kernels were streams with three removable costs:
  - 4.5 % of the bytes (F11);
  - loads the compiler had serialized (min-blocks);
  - for o, a kernel shape that left most SMs short of warps (78 registers, 0.65 waves in milestone 04's ncu).
- The unpooler slice at m = 8 was still the BF16 TinyGEMM (3.94 µs local, floor12 3.21).
- **Projection** (`notes/G_NOTES.md`, final bench; `evidence/G_final_bench2.log`): decode-1 −2.05 µs (× 0.47 %/µs) and verify-1 −1.13 µs (× 0.39 %/µs), about **+1.4 %**. The t3 log gives the decode-1 load alone: 1.806 vs 1.694 judge-projected, **+1.1 % geomean** (excerpt lines 124-126).

## 4. Change

Module G (`gemv1.py`), dispatched in `kernel_fn` for any non-router layer at m = 1, and for the unpooler slice at 2 ≤ m ≤ 8:

| shape | kernel | format | geometry |
|---|---|---|---|
| qkv m1 | `g1_qkv_gate_g311` | F11 (11.58 b/w) | 2304 CTAs × 256 threads, R = 1, min-blocks 6 |
| pooler m1 | `g1_pooler_g311` | F11 | 4096 CTAs × 256 |
| unpooler slice m1 | `g1_unpslice_g311` | F11 (11.63 b/w) | 1024 CTAs × 128 (parity with GV3) |
| o m1 | `g1_o_tc` | F12 (12.13 b/w) | 512 CTAs × 256 = one wave, W pointer baked |
| unpooler slice m2-8 | `g1m_unpslice` (`tcm_body`) | F12 tiles | 64 CTAs × 1024, all x B-fragments loaded up front (`XS = 5`) |

Details:
- **g311:** contiguous warp loads, x loaded first, escape x-gathers issued right after the slot load, and a warp-segmented scan by row with one atomic per segment.
- **tcm** (`notes/G_NOTES.md`, "Small M extension"): its reduction layout is `[w][tok][row]`, which removes the 32-way bank conflicts of `[w][row][tok]`. An mbarrier spin loop is a basic-block boundary, so ptxas cannot sink the weight loads below a `__syncthreads`.
- Everything is compiled by NVRTC at `prepare()`, in about 2-4 s.

int_t3a and int_t3b differ only in that tcm option: t3b loads all x fragments up front (`XS = 5`) and extends the dispatch to the slice at m ≤ 8 (diff of `claude/t3/int_t3a.py` vs `int_t3b.py`).

## 5. Result

| check | result |
|---|---|
| local, same run (`evidence/G_final_bench2.log`) | m = 1: qkv 7.19 vs int_v7 7.81, o 3.89 vs 4.36, pooler 11.73 vs 12.69, slice 2.60 vs 2.56; slice m8 2.91 vs 4.04. Repeat runs: qkv 7.14-7.22, pooler 11.51-11.73, o 3.72-3.91 (`notes/G_NOTES.md`) |
| ncu / nsys of G (`notes/G_NOTES.md`, RESULT) | qkv 27.30 MB read, issue 52 %, occupancy 63 %; pooler 48.65 MB vs GV3 50.49, 14.43 vs 15.30 µs; o 5.70 vs 6.02 µs. nsys in-graph: qkv 8.24 (GV3 8.61), o 4.62 (5.15), pooler 12.64 (13.49) |
| SASS | every G kernel: 0 weight loads after its first FFMA/HMMA (`nvrtc_sass.py`, NVRTC 13.0; `notes/G_NOTES.md`, RESULT) |
| correctness | `check.py --graph` (judge tests + scored + nearby m, normal + outliers, eager + graph): 207 cases × 2 seeds ALL OK; m 1-8: 64 cases ALL OK; F11/F12 round trip bit-exact |
| judge | 1.3999 → **1.4168** (+1.20 %). decode 1 load 1.684 → 1.797 (+6.7 %), verify 1 1.776 → 1.836 (computed) |
| judge, changed shapes (µs) | qkv m1 8.78 → 8.01 (−0.77), o m1 5.28 → 4.67 (−0.61), pooler m1 13.76 → 13.10 (−0.66), slice m1 3.58 → 3.81 (+0.23), slice m8 4.79 → 3.57 (−1.22) |
| **judge, read honestly** | The other 25 shapes run identical code. They moved by a mean +0.10 µs (−0.35 to +1.11; qkv m256 +1.11, router m128 +0.53), and SGLang's own times by +0.06 (computed). So the node was neutral to slightly slow: the +1.2 % is not node luck. The slice at m1 (+0.23) was expected at parity, and the 0.23 µs is inside the per-shape noise. |
| vs projection | +1.2 % judge vs +1.1 % (t3 log) and +1.4 % (G's estimate). Recorded as "projection held" (excerpt line 131). |

## 6. Takeaways, and what came next

- **Check SASS for load sinking after every kernel edit.** It was the root cause of most "slow" variants at m = 1. Fix it with a min-blocks value in `__launch_bounds__` sized to the occupancy target, and count late loads with NVRTC (the judge's compiler), not nvcc.
- **Baking constants is not free.** A baked weight pointer helps a one-wave kernel. In a multi-wave kernel it pushes the parameter-bank load of x behind the weight flood.
- **11 bits pay only where the escape gather overlaps other CTAs' streaming.** That holds in many-CTA g311. It does not hold in the one-wave tile kernel for o (tc11: +0.3-0.5 µs from the escape path, `notes/G_NOTES.md` "Lessons").
- **Known gap:** the t3 losslessness gate `exact_all.py` (built the same morning) later found that g311 and tcm turn escapes smaller than about 2^−39 into 0 (excerpt lines 142-144). This is equivalent to FP32 accumulation rounding and never occurs for the real weights.
- **What was left after 05** (this judge run against the floor12 column of the gap table, excerpt lines 25-59; computed):
  - m = 1: 0.0-0.6 µs over the 12-bit floor.
  - The m = 8-32 qkv / pooler / unpooler shapes (K1s, K1t, Triton TMA): 2.8-5.1 µs over.
  - The m = 128/256 cuBLASLt shapes: 0.8-7.3 µs over, worst on qkv and pooler at m = 128.

  The last two groups held most of the remaining gap.
- **Next, 06:** instead of writing new m = 8-256 kernels from scratch, a per-shape best-of built on a public Apache-2.0 kernel file, combined with this milestone's m = 1 kernels.

## Reproduce

Needs an H200 and the SGLang image (torch 2.14 + CUDA 13.0, Triton 3.8, cuda-python, `sglang`). `prepare()` builds the Codex extensions with `load_inline` (nvcc + ninja) and compiles everything else with NVRTC. The build reads nothing from disk.

```bash
cd muse_bf16_proj/tools
python3 check.py ../05_f11_gemv_m1_no_load_sinking/build.py --graph
python3 bench.py --cand int_v7=../04_many_cta_gemv_m1/build.py --cand int_t3b=../05_f11_gemv_m1_no_load_sinking/build.py \
       --shapes 1,8 --reps 80                 # changed shapes; the other m = 8 layers are controls
python3 ../05_f11_gemv_m1_no_load_sinking/evidence/roofline.py   # roofline / floor table (edit its probe table to re-fit)
# load-sinking check and F11 round trip need the G lane modules (claude/t3/G/g1x.py, g11.py, gemv1.py) on the path:
#   cd claude/t3/G && python3 nvrtc_sass.py gemv1 "<expr returning the CUDA source>" && python3 roundtrip_test.py
```

Originals (relative to ``):
- build: `muse-opt-codex/muse_bf16_proj/claude/t3/int_t3b.py` (assembled by `claude/t3/build_t3.py`); int_t3a: `claude/t3/int_t3a.py`
- G lane: `muse-opt-codex/muse_bf16_proj/claude/t3/G/` (`gemv1.py`, `g1x.py`, `g11.py`, `tcm.py`, `NOTES.md`, `logs/`)
- t3 roofline: `claude/t3/ROOFLINE.md` (lines 1-146 copied in `evidence/`; the whole file is (private notes, not included)), `claude/t3/roofline.py`, `claude/t3/launchprobe.py`
- int_v7 profiles: `claude/t3/{base_v7.log, ncu_v7.log, nsys_v7.log, b_v7.json}`
- judge: `muse-opt-codex/muse_bf16_proj/claude/t3/profiles/int_t3b.a5086e19abb541ba.{txt,json,log}`
