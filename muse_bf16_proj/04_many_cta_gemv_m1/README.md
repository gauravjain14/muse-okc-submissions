# 04: a many-CTA 12-bit GEMV at m = 1 (GV3), with escapes stored inside each row

| | before (03) | after (04) |
|---|---|---|
| build | `03_.../build.py` = `int_v4.py` (md5 cf0feeb7) | `build.py` = `int_v7.py` (md5 558722c4, 237,526 B). The step itself was `int_v5.py` (md5 f15261da, 220,950 B); int_v7 = int_v5 + four small per-shape pieces |
| judge | 1.3482 (job 8bb51ba3, check) | int_v5 **1.3997** [1.3787, 1.4207] (job a0c53537, check); int_v7 **1.3999** [1.3789, 1.4209] (job 71a5d9d6, check) |
| real effect | | +3.8 % judge (int_v4 → int_v5), of which about +2.5 % is the code (same-run local) and the rest the previous reading's slow node reverting; int_v5 → int_v7 is +0.02 %, i.e. no change |
| shapes changed | | int_v5: qkv / o / pooler / unpooler slice at m = 1 (GV3); router at 5 ≤ m ≤ 8 (epilogue flag); qkv at m ≤ 16 (K1s register depth 4 → 3). int_v7 adds: o at 2 ≤ m ≤ 8 (GM), qkv at 2 ≤ m ≤ 16 (K1t, 4-CTA clusters), GV3 qkv config (2, 512), a K1t epilogue using `st.async` |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: code copied or adapted from other projects is marked in place with `SPDX-SnippetBegin` / `SPDX-SnippetEnd` comments (source, copyright, license, our changes); see `NOTICE` and `LICENSES/` at the repository root. The code is unchanged. The judged file has md5 `558722c4`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

Terms used below:
- **GV / GV3**: generations of a many-CTA GEMV for m = 1, in which each CTA owns R rows and the full K.
- **GM**: the same structure using tensor cores, for o at small m.
- **Slot**: a per-row escape entry `(k << 16) | true BF16 bits`.
- **wdec**: the value an escaped weight decodes to. Its code is 0 / low byte 0, which decodes to 2^(2H−127); the kernel adds x[k] · (true − wdec).

## 1. Starting point

The decode-1 load (m = 1) was the most valuable per µs: 100 / (6 × 41.44 µs) = 0.40 % of score per µs saved (computed from `evidence/judge_before_int_v4_check_8bb51ba3.txt`). Its BF16 layers were still on Codex's Triton GEMV or on K1s:

| shape (m = 1) | SGLang | int_v4 judge | int_v4 kernel | 12-bit stream floor, local + 1.0 (milestone 02 `evidence/bw1.log`) |
|---|---|---|---|---|
| qkv_gate | 14.13 | 11.21 | Triton `pr_gemv` (BF16) | 8.30 |
| o | 6.59 | 5.36 | Triton `pr_gemv` (BF16) | 4.68 |
| pooler | 21.38 | 17.37 | K1s `k1s_pooler1_8_1` (12-bit, one wave of 132 CTAs) | 13.54 |
| unpooler slice | 6.75 | 3.65 | Triton `pr_gemv` (BF16) | 3.18 |

## 2. What we saw

**(a) The right structure for m = 1 was Codex's, not K1s's.**
- Codex's BF16 Triton GEMV runs one program per few rows × full K, thousands of CTAs. It already streamed within 0.04-0.62 µs of the BF16 floor (milestone 02, section 2b).
- K1s at m = 1 is one wave of 132 CTAs, each with a serial start (x TMA) and a split-K epilogue. Its m = 1 pooler (16.4-17.4 µs judge) sat 3-4 µs above the 12-bit floor.
- The many-CTA structure lets the hardware scheduler overlap each CTA's start and reduction with the other CTAs' streaming (`claude/gv.py` docstring).
- The plan: put the 12-bit format into that structure.

**(b) The first version (GV) lost**, local µs in the same run:

| log | GV | Codex v2 / K1s same run |
|---|---|---|
| `evidence/gv1.log` (13:08) | o 6.23, pooler 15.64, unpooler slice 4.26 | cx 4.72, 15.98, 2.94 |
| `evidence/gv2.log` (13:57) | pooler 17.73, unpooler slice 3.58 | cx 16.26 / 2.97; s12 16.52 / 4.59 |

The cause, recorded in `claude/gv3.py`'s docstring and in `notes/` (memory file in 02): every GV CTA first read a per-row offset table, then the escape list it pointed to. Those are two dependent cold DRAM round trips before the CTA could finish. That **doubled CTA lifetime**, and with thousands of short CTAs it doubled the time.

**(c) GV3 stores each row as one contiguous blob: [low bytes: K][codes: K/2][escape slots: E × u32]**, so a row's escapes arrive with its weights.
- Escapes: 2661 / 986 / 4559 / 391 for qkv / o / pooler / unpooler slice, at most 6 per row. E = 8 / 4 / 8 / 4 slots per row.
- Bytes: 28.39 / 12.71 / 50.46 / 6.31 MB (`evidence/gv3a.log`).
- x is read straight from global through L1, since all CTAs on an SM share it.

| log | qkv | o | pooler | unpooler slice | note |
|---|---|---|---|---|---|
| `gv3a.log` (14:11), GV3 best config | **8.39** | 4.88 | **13.86** | **2.60** | |
| `gv3a.log`, int_v4 same run | 9.83 | 4.55 | 16.49 | 2.85 | |
| `gv3s.log` (14:12), config sweep | 8.24-9.15 | 4.26-4.70 | 12.58-13.89 | 2.56-3.17 | (rows × threads) per layer; picks in `_GV_PREF` |

GV3 removed 1.4 µs from qkv and 2.6 µs from pooler. o stayed behind Codex's GEMV in this run; the integrated run below shows it ahead.

**(d) The rest of int_v5 came from two same-day lanes:**
- **Router round 2** (milestone 02 `notes/router_NOTES.md`, "Round 2"). With FLAGS = 272 at m ≤ 8, each consumer warp acquire-polls the zero counter itself and `red.add`s its mma fragments directly: no smem cross-warp reduction and no `__syncthreads`. Router m8 −0.25 to −0.3 µs. At m ≥ 32 it costs +1 to +4 µs, so it is gated to m ≤ 8.
- **K1s qkv register depth 4 → 3** (`_k1_CFG["qkv_gate"]` D field). Same-run `evidence/v5a.log`: qkv m16 10.26 vs 10.60, m8 9.91 vs 10.35.

**(e) ncu of GV3** (`evidence/ncu_gv3_compare_qkv_o_unpslice.txt` and `evidence/ncu_gv3_stall_hotspots_*.txt`, 15:25, int_v7 configs):

| kernel | duration (ncu) | DRAM | regs | achieved occupancy | eligible warps | waves | top stall (source line) |
|---|---|---|---|---|---|---|---|
| qkv (2 rows × 512 thr) | 9.60 µs | 61.8 % | 30 | 84 % | 2.00 | 2.18 | long_scoreboard on the weight load (gv3.cu:28) |
| o (8 rows × 64 thr) | 6.08 µs | 43.7 % | 78 | 22.5 % | 0.58 | 0.65 | long_scoreboard (gv3.cu:81, :28) |
| unpooler slice (1 × 256) | 4.45 µs | 29.8 % | 26 | 84 % | 1.52 | 0.97 | barrier (gv3.cu:109) |

- qkv and pooler are streaming. The t3 re-profile (milestone 05 `evidence/ncu_int_v7.log`) gives GV3 pooler 50.49 MB in 15.23 µs, 88 % occupancy, 57.6 % issue, 47 % long-scoreboard.
- **o is not:** 78 registers leave 0.65 waves and 0.58 eligible warps.

**(f) K1s still pays about 1 µs at each end** (`evidence/tr16.log`, `evidence/trx.log`; `evidence/trace_k1.py` writes globaltimer stamps per CTA; µs from the earliest CTA start, median / max over CTAs):

| shape | first data | loop end | kernel end | after the loop |
|---|---|---|---|---|
| qkv m8 | 0.93 / 1.06 | 8.10 / 8.54 | 9.54 | 1.0-1.4 µs (sync → partial exchange → sums → store) |
| pooler m16 | 1.25 / 1.38 | 12.99 / 13.63 | 14.56 | 0.9-1.6 µs |
| unpooler m16 | 1.25 / 1.31 | 12.90 / 13.50 | 13.86 | 0.4-1.0 µs |

The ncu of the K1s/K1t kernels (`evidence/ncu_k1s_k1t_compare.txt`, 14:33) shows the same picture:
- 121-128 registers, 1 CTA × 16 warps per SM.
- 24.7-24.9 % occupancy, 0.46-0.55 eligible warps per scheduler, 31-35 % issue.
- DRAM at 48-63 % of peak (2.3-3.0 TB/s under ncu).
- Tensor pipe 10-20 % active.

## 3. Diagnosis and projection

- **At m = 1 the limiter is bytes plus CTA lifetime.** GV3 keeps the many-CTA overlap and the 12-bit bytes, and takes the dependent cold loads off each CTA's path.
- **Projection** (same run, `evidence/v5a.log`): decode-1 load 28.39 µs vs 33.71 local (x1.838 vs x1.547). Whole build judge-projected **1.4215 vs 1.3871 = +2.5 %**.
- Floor check: GV3 qkv and pooler land at 7.79 and 12.16 µs local in that run. That is 0.5 µs above the 12-bit LDG floor for qkv, and 0.4 µs *below* the pooler probe of `bw1.log`, which was taken in a different container (local µs; containers differ by up to 25 % on small kernels per `notes/router_NOTES.md`).

## 4. Change

int_v5 (`claude/int_v5.py`, built 14:15):
1. **GV3** (`_gv_*`, NVRTC):
   - **Dispatch:** at m = 1, `kernel_fn` sends qkv / o / pooler / unpooler slice to GV3 first.
   - **`_GV_PREF`:** qkv (2 rows, 256 thr), o (8, 64), pooler (2, 512), slice (1, 256). The first config whose slot count fits (rows × E ≤ 32 escape lanes) is used. A matrix with more than 64 escapes in a row keeps its old path.
   - **Kernel:** each thread loads 16-B chunks of low bytes and codes, decodes with the same `decode12`, and FMAs in FP32 against x.
   - **Escapes:** lanes of warp 0 each load one slot word at CTA start, alongside the weights. Each adds x[k] · (true − wdec) into its accumulator before the warp shuffle reduction (`claude/gv3.py`, `hasesc`).
2. **Router** `_rt_SMALL_FLAGS = None`: FLAGS 272 for m ≤ 8, 16 otherwise.
3. **K1s** `_k1_CFG["qkv_gate"] = (2, 66, 5, 1, 16, 3)`.

int_v7 (`claude/int_v7.py`, built 15:11), each a same-run local win on its shape:

| piece | evidence (local µs, same run) |
|---|---|
| GV3 qkv config (2, 512) first in `_GV_PREF` | `gvq.log`: q1 (2, 512) 7.94 vs current 8.40 |
| **GM** for o at 2 ≤ m ≤ 8: many-CTA tensor-core GEMV. CTA = one 16-row 12-bit tile × full K, mma.sync with W as A, x fragments read from global via L1; `_GM_O = (4 warps, min blocks 6)` | `gm3.log`: variant d o m8 4.30 vs int_v5 5.09; GM loses at m16/32 (`gm1.log`: 5.78 / 8.42 vs 4.92 / 4.97), so it is gated to m ≤ 8 |
| **K1t** for qkv at 2 ≤ m ≤ 16 with 4-CTA clusters (120 SMs) and 8-row groups: `_k1t_CFG["qkv_gate"] = (4, 30, 5, 2, 8, 3)` | `q4.log`: qkv m16 10.56 vs 10.85, m8 10.43 vs 10.58 |
| K1t epilogue: partner partials pushed with `st.async ... mbarrier::complete_tx` (`claude/k1t.py` `st_async4`) | mixed. The `old` / `new` K1t A/B of 14:49-14:53 measured −0.29 to +0.45 µs per shape across two runs (`sae1.log`, `sae2.log`). That this A/B is the `st.async` change is inferred: it is the only K1t A/B before int_v7 |

Kernel generations for m = 1 on the way:

| version | design | result | lesson |
|---|---|---|---|
| Codex Triton `pr_gemv` | BF16, one program per row block | near the BF16 floor | the structure is right, the bytes are not |
| K1s at m = 1 | 12-bit, one wave of 132 CTAs | pooler 16.4-17.4 µs judge | one wave cannot hide its start and epilogue |
| GV (13:08) | 12-bit many-CTA; per-row offset table → escape list | o 6.23, slice 4.26 (lost) | two dependent cold loads per CTA double its lifetime |
| **GV3** (14:11) | escapes as slots inside the row blob | qkv 8.39, pooler 13.86 | ships |
| GV3 config (2, 512) (15:10) | 512 threads for qkv | qkv 7.94 | ships in int_v7 |

## 5. Result

| check | result |
|---|---|
| local, same run (`evidence/v5a.log`) | int_v5 vs int_v4: decode 1 x1.838 vs x1.547; m = 1 qkv 7.79 vs 9.25, pooler 12.16 vs 15.27, o 3.89 vs 4.01, slice 2.05 vs 2.60; judge-projected **1.4215 vs 1.3871** |
| local int_v7 (`evidence/v6a.log`, `evidence/v7a.log`) | v6 vs v5 1.3871 vs 1.3895; v7 vs v6 1.4007 vs 1.3992: flat |
| correctness | the correctness sweep before each bench printed "all ok" (`v5a.log`, `v7a.log`); judge 9/9 |
| judge int_v4 → int_v5 | 1.3482 → **1.3997** (+3.82 %). decode 1 load 1.428 → 1.705. m = 1: qkv 11.21 → 9.22 (−1.99), pooler 17.37 → 13.43 (−3.94), o 5.36 → 5.03 (−0.33), slice 3.65 → 3.38 (−0.27). qkv m16 12.14 → 11.32 (−0.82), qkv m8 11.69 → 11.52 (−0.17); router m8 5.22 → 5.31 (+0.09: the −0.25 µs local gain is invisible) |
| **judge, read honestly** | 22 shapes run identical code in int_v4 and int_v5 (the K1s/K1t pooler/unpooler, the Triton o kernels, the m = 128/256 cuBLASLt pins, the m ≥ 16 routers). They moved by a mean −0.21 µs (range −1.31 to +0.39; qkv m256 −1.31), so this node was slightly fast. The previous reading (03) had caught a node about 5 % slow on decode-1 identical code. Part of the +3.8 % is therefore the two nodes, not the code. **The same-run local +2.5 % is the better estimate of this step.** The decode-1 change itself (−6.5 µs on a 41.4 µs load) is far outside the noise. |
| judge int_v5 → int_v7 | 1.3997 → **1.3999** (+0.02 %). GM o m8 6.10 → 5.55 (−0.55) and GV3 qkv m1 9.22 → 8.78 (−0.44) show up. The K1t qkv m16 piece reads +0.25 µs, inside the noise. Identical-code shapes moved −0.49 to +0.61 (qkv m128 −0.49, router m1 +0.55, unpooler m16 +0.61). Treat int_v7 as int_v5 plus a few small, locally verified pieces. |
| vs projection | +2.5 % projected (same run), +3.8 % read on the judge, with the node effects above explaining the difference. |

## 6. Takeaways, and what came next

- **For many-CTA kernels, every dependent cold load is paid once per CTA.** Put everything a CTA needs (weights, codes, escape slots) into one contiguous blob, issued at its start. That single change turned GV's loss into GV3's −1.4 / −2.6 µs.
- **At m = 1, structure beats bandwidth tricks.** Short CTAs with smooth turnover let the hardware overlap start and reduction. A one-wave persistent kernel (K1s) cannot, even when it reads the same bytes.
- **Not every local same-run win shows on the judge.** int_v7's four pieces were each a local win and summed to nothing visible on one judge run (±1.5 % interval). The memory note of the day says so directly: "micro-gains <2%/shape are invisible at the judge CI" (`claude` memory, milestone 02 `notes/memory_bf16-proj-findings.md`).
- **What was left:**
  - o at m = 1 with 78 registers and 0.65 waves.
  - All the K1s/K1t shapes: about 1 µs start and about 1 µs epilogue per CTA, 25 % occupancy, 0.5 eligible warps.
  - m = 128/256, still cuBLASLt.
- Three workers continued that afternoon (`notes/k3_NOTES.md`, `notes/k4_NOTES.md`):
  - k3: m = 128/256 swap-AB RS-wgmma plus router, close to cuBLASLt but not ahead.
  - k4: m = 32 `K4S`, −0.3 µs on qkv / unpooler.
  - k5: m = 8/16 producer/consumer, behind.
- The work was paused at about 17:05 PT on 10-27 with int_v8 (int_v7 + K4S) built but never validated (milestone 02 `notes/memory_bf16-proj-findings.md`). **Next, 05**, on 11-02: a roofline over all 30 shapes, then an 11-bit format and a load-sinking fix at m = 1.

## Reproduce

Needs an H200 and the SGLang image (torch 2.14 + CUDA 13.0, Triton 3.8, cuda-python, `sglang`). `prepare()` builds the Codex extensions with `load_inline` (nvcc + ninja) and compiles GV3 / GM / K1s / K1t / router with NVRTC (84.1 s locally for int_v7, milestone 05 `evidence/bench_int_v7_base_v7.log`). The build reads nothing from disk.

```bash
cd muse_bf16_proj/tools
python3 check.py ../04_many_cta_gemv_m1/build.py --graph
python3 bench.py --cand int_v4=../03_cublaslt_config_search/build.py --cand int_v7=../04_many_cta_gemv_m1/build.py \
       --shapes 1,8,16 --reps 30            # m = 1 (GV3), m = 8 (GM o, router), m = 16 (K1t qkv); other loads as control
# per-CTA phase timeline of a K1 kernel (original location claude/; imports reference.py and the candidate):
#   python3 claude/trace_k1.py <candidate.py> qkv_gate:8 pooler:16
```

Originals (relative to ``):
- builds: `muse-opt-codex/muse_bf16_proj/claude/int_v5.py` (the judged step, not copied here) and `claude/int_v7.py`
- kernels: `claude/gv.py`, `claude/gv3.py`, `claude/gm.py`, `claude/k1t.py`, `claude/router/router.py`
- logs: `claude/out/{gv1,gv2,gv3a,gv3s,gvq,gm1,gm3,q4,sae1,sae2,tr16,trx,v5a,v6a,v7a}.log`
- ncu: `claude/out/20261027T152542_profgv1/analysis/` (GV3), `claude/out/20261027T143341_prof2/` (K1s/K1t; 38 MB with `reports/*.ncu-rep`, not copied)
- judge: `muse-opt-codex/muse_bf16_proj/claude/profiles/int_v5.a0c53537e6ec4bd6.*` and `int_v7.71a5d9d6a4924409.*`
