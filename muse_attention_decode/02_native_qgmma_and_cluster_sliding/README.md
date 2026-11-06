# 02: native FP8 wgmma (QGMMA) tiles, FLAT boundary balancing and a batch-1 cluster combine (verify via k4, then decode and verify sliding via k6)

Shape names: d / v = decode (q_len 1) / verify (q_len 8); 1 / 16 / 32 = batch; s / f = sliding (2,048-token window) / full (~80K context). E0 = the era-0 lab tree, `../autoresearch-kernels/attn/` relative to `Compilers/`.

Worker lanes: four agents, each with its own directory `E0/dev/<lane>/`, building on the 01 kernel in parallel (`notes/PLAN.md`; `notes/*_NOTES.md`):
- **W8**: q_len 8 on wgmma.
- **WD**: q_len 1 on wgmma.
- **WB**: the FLAT full-layer schedule.
- **WC**: small-call latency.

This milestone is the stack of their deliveries, judged twice:
- **k4** = k3 + W8.
- **k6** = k4 + WD + WB + W8 round 2 + WC.

| | before (01) | after (02) |
|---|---|---|
| build | `../01_*/build.py` = k3 (md5 0937facd) | `build.py` = k6 (`E0/subs/k6_check.py`, md5 d544651d, 163,415 B). Intermediate judged build k4 = `E0/subs/k4_check.py` (md5 8046e004, 71,726 B) |
| judge | 1.2010 (okc check 4a7aa2f3) | k4 **1.5724** (okc check c284726b, CI 1.549-1.600); k6 **1.8412** (okc check c763bae8, CI 1.814-1.869). One reading each, no scored runs |
| real effect | | **+53.3 %** on the judge (single readings). Local same-job steps (raw geomean): k3 → k4 +31.8 %, k4 → k5 +9.2 %, k5 → k6 +9.7 %. Chaining the local judge-projected steps gives +55.8 % |
| shapes changed | | k4: all six verify shapes. k6: all six decode shapes, plus v1s, v1f, v16s, v32s. v16f and v32f moved only at k4 |

## 1. Starting point

k3 judge times (`evidence/judge_before_k3_check_4a7aa2f3.txt`) and the local same-job reading (`evidence/verify_k3w.txt`):

| shape | k3 judge (µs) | SGLang (µs) | speedup | k3 local (µs) | k3 kernel |
|---|---|---|---|---|---|
| d1s | 13.35 | 20.38 | 1.53x | 12.62 | ALIGNED `attn_1_1_1_1` |
| d1f | 18.46 | 31.58 | 1.71x | 16.63 | ALIGNED, 132 CTAs |
| d16s | 15.50 | 26.98 | 1.74x | 13.88 | ALIGNED |
| d16f | 111.85 | 200.91 | 1.80x | 107.71 | FLAT `attn_1_0_1_0` |
| d32s | 19.03 | 31.89 | 1.68x | 17.76 | ALIGNED |
| d32f | 195.64 | 384.75 | 1.97x | 189.80 | FLAT |
| v1s | 22.99 | 23.09 | 1.00x | 21.04 | ALIGNED `attn_8_1_1_1` |
| v1f | 31.31 | 35.90 | 1.15x | 30.08 | ALIGNED |
| v16s | 22.55 | 28.99 | 1.29x | 21.14 | ALIGNED |
| v16f | 318.14 | 205.42 | **0.65x** | 315.83 | FLAT `attn_8_0_1_0` |
| v32s | 30.54 | 34.72 | 1.14x | 28.57 | ALIGNED |
| v32f | 608.36 | 390.82 | **0.64x** | 606.61 | FLAT |

Per load (`evidence/judge_score.py`): d1 1.586, d16 1.780, d32 1.903, v1 1.050, v16 0.754, v32 0.705.

## 2. What we saw

**(a) The verify tile was bound by the emulated FP8 MMA.** W8's `evidence/t_core.py` times the tile core with resident shared memory, 512 threads and `clock64`; output in `evidence/w8_tcore_cycles_sass.txt`:

| tile core | cycles per 32-token tile per SM | SASS per tile iteration |
|---|---|---|
| k2's `mma.sync.m16n8k32.e4m3` | 1083.1 | 89 `F2FP.F16.E4M3.UNPACK_B` + 34 `HMMA.16816.F32` |
| wgmma, 32-token stages (NB=1) | 540.8 | 3 `QGMMA.64` |
| wgmma, 64-token stages (NB=2) | 416.7 | 4 `QGMMA.64`, 34 `MUFU.EX2` |

The budget per 32 tokens per SM is about 497 cycles of memory time (8 KB at 4.3 TB/s over 132 SMs, at 1.98 GHz) and about 264 cycles of MUFU. mma.sync was at 2.2× the memory budget; QGMMA fits under it (`notes/W8_NOTES.md`, "Log"). This matches 01's skeleton: k2 ran v32f in 638.6 µs with math, 231.7 µs without.

**(b) A precision trap in QGMMA accumulation.** QGMMA accumulates FP8 internally with ~14 bits. Chaining PV across tiles in the wgmma accumulator drifted the ones-column row sum by −22 % after 512 tiles.
- Fix: every tile's PV starts from scale-d 0 into a fresh accumulator, then `o = o·alpha + t` in FP32 registers.
- After the fix, 512 tiles give O 1.2e-4 and l 9e-5 (`notes/W8_NOTES.md`; unit test in `w8_tcore_cycles_sass.txt`: "CORE PASS", S rel ~2e-4).
- FA3's FP8 path does chain O.

**(c) A hidden DRAM trip per tile.** k2/k3's ring loads the page-table entry for tile it + DST − 1 with an LDG at the top of iteration it, then consumes it right after the barrier. That exposes one cold-DRAM trip per tile.
- Fix: prefetch the entries with 4-B `cp.async` into a per-lane shared-memory ring, DST − 1 tiles ahead.
- Effect: v32f 292 → 262 µs, v16f 159 → 146 µs (`notes/W8_NOTES.md`).

**(d) What limits the QGMMA verify tile next.**
- ncu, k2w + page-table ring, v32f (`evidence/ncu_k2w_v32f_*`): DRAM 51.7 % (2.49 TB/s), issue slots 54.6 %, 128 registers. The top stall line (21 %) is the `ex2`: wait plus mio_throttle, because MUFU shares the MIO queue with LDGSTS / LDSM / STSM / SHFL.
- Ablations on k3w, same job, v32f / v16f in µs (`notes/W8_NOTES.md`):

| variant | v32f | v16f |
|---|---|---|
| base | 256 | 142 |
| no exp | 242 | 133 |
| no PV wgmma | 209 | 116 |
| no compute | 174 | 97 |

  The PV (issued, then waited on at once) costs ~47 µs at v32f; MUFU only ~14. Another job measured the same skeleton at 181 / 99 µs (`evidence/bench_w8_k3w_nocompute.json`).

**(e) The decode tile carried the same emulation.**
- k3's q_len-1 loop is ~640 SASS instructions per 32-token warp tile, including 77 `UNPACK_B` + 34 `HMMA` (`notes/WD_NOTES.md`, "SASS").
- WD measured compute overhead (real minus `ABL_NOCOMPUTE`, same job wd_t2):

| shape | k4 | WD's QGMMA run1 |
|---|---|---|
| d32f | 15.4 µs (188.6 vs 173.1) | 5.8 µs |
| d1s | 3.9 µs (12.40 vs 8.50) | 1.5 µs |

**(f) FLAT decode full: request boundaries cost ~12 µs.** WB traced k3, and then its first variant k3b, at d32f with per-CTA globaltimer stamps (`notes/WB_NOTES.md` job1-job3):
- ~10 µs of startup before the first gather.
- Median → max CTA spread ~13 µs.
- ~11 µs of tail after the slowest CTA's last tile.
- CTAs whose slice spans two requests finished at a mean of 174.9 µs, against 162.9 µs for one-request CTAs.

Charging virtual units at each request start (OVH) balances the two groups: 166.1 vs 169.3 µs at OVH 1536-2048. That is worth ~6 µs at b32 and ~10 µs at b16.

**(g) Batch-1 sliding is a latency chain that ends in a global combine.**
- k3 trace, d1s, in µs (`notes/WC_NOTES.md` run_base):
  - page-table rows issued 2.62
  - q/new done 4.31
  - tile0 4.67
  - tiles 5.76 (max 7.46, the CTA with the new block)
  - merged 6.90, published 7.86
  - released 9.86, combined 12.69

  About 5 µs of global publish/combine after the tiles.
- k5m (WD's QGMMA run1) still spends ~4.3 µs from tiles (4.74) to combined (9.01) (`notes/WD_NOTES.md`).
- Two SASS facts set the design (`notes/WC_NOTES.md`):
  - `barrier.cluster.arrive` (release) and `mbarrier.arrive.release.cluster` lower to `MEMBAR.ALL.CTA + MEMBAR.ALL.GPU`.
  - `st.async` with `mbarrier.try_wait.acquire.cluster` lowers with no MEMBAR.
- Cluster co-residency at 1 CTA per SM (`cuOccupancyMaxActiveClusters`): size 16 → 7 clusters, 8 → 15, 6 → 17, 4 → 30, 3 → 39, 2 → 66.

## 3. Diagnosis and projection

- **Verify.** The tile cost (1083 cycles) exceeded the memory budget (497). A native QGMMA tile (417) fits under it, so verify full should fall to roughly the structure's no-compute time. That time, measured once k3w existed, is v32f 181 µs and v16f 99 µs locally (`evidence/bench_w8_k3w_nocompute.json`), against 607 / 316 for k3.
- **Decode tile.** Swap-AB (`S^T = K·Q^T`, M = 64 keys, N = 8 query heads) gives a padding-free `m64n8k32` QGMMA (`../01_*/notes/research.md` §0 item 4). It removes the unpack/HMMA work: compute overhead at d32f goes from 15 to ~6 µs.
- **FLAT boundaries.** Charge ~1536 virtual units per request start, projected −6 / −10 µs at b32 / b16 (WB job3).
- **Batch-1 sliding.** Replace the global publish → acquire → combine (~4-5 µs) with a DSMEM push inside a cluster formed by the request's CTAs. Start the page-table loads before the length is known.
  - The second part is allowed as *verified* speculation only (`notes/PLAN.md`, ruling 2026-10-28): at B = 1, SGLang's FA3 page table is sliced to the batch's maximum length, so its width is seq + L. Guess from it, then confirm or reload. Every input must stay correct.
  - Projection: a review of sibling kernels (`E0/docs/sibling_lessons.md` §0) estimated b1 sliding 12.8 → ~6-7 µs from a one-round-trip cluster combine plus a shorter start chain, and +5-9 % score for the combine alone.

## 4. Change

| piece | lane | shapes | local effect (same job) | source |
|---|---|---|---|---|
| `run8<>`: q_len 8 on QGMMA, page-table cp.async ring (**k4**) | W8 | all verify | v32f 606.6 → 253.3, v16f 315.8 → 142.4, v1f 30.1 → 21.6, v1s 21.0 → 17.4, v16s 21.1 → 18.2, v32s 28.6 → 21.8; x1.2271 → x1.6177 | `evidence/verify_k3w.txt` |
| `run1<>`: q_len 1 on QGMMA (k4d = k5) | WD | all decode | d1s 12.72 → 9.30, d16s 13.76 → 10.26, d32s 17.38 → 12.61, d1f 16.37 → 14.96, d16f 106.1 → 94.8, d32f 187.8 → 170.4; x1.6261 → x1.7764 | `evidence/verify_k4d.txt` |
| j4 FLAT schedule: OVH = 1536, fast full-tile issue, combine loads in flight, barrier-free single combiner | WB (merged by WD as `k5m`) | d16f, d32f; small calls | on k4: d16f 106.1 → 95.5, d32f 187.8 → 173.7 (`verify_k4d.txt`). On k5: d16f 97.0 → 90.0, d32f 177.0 → 169.5, d1s 9.47 → 8.70 | `notes/WD_NOTES.md` wd_t4 |
| `run8<>` round 2: ring depth 3 on sliding, gate handling per path, lane-merge row stride 68 | W8 | v1s, v16s, v32s, v1f | k5 → k5w: v1s 17.29 → 16.91, v16s 18.51 → 16.71, v32s 22.00 → 19.90, v1f 21.97 → 20.24 | `evidence/bench_w8_k5w.json` |
| `runA1<>`: one cluster per B = 1 sliding request, DSMEM combine, verified length speculation, RoPE table, spill-free prologue/epilogue; plus run8's explicit exit from its single segment | WC | d1s; v1s, v1f via the run8 exit | d1s 8.88 → 6.81 vs k5m (WC job8, 40 reps). The run8 exit alone: v1s 17.25 → 15.66, v1f 21.62 → 20.04 (job3b) | `notes/WC_NOTES.md` |
| **k6 = all of the above** | WC (merge, `dev/wc/build_k6.py`) | | k5 → k6: x1.7496 → x1.9191 (judge-projected x1.7229 → x1.8796) | `evidence/verify_k6.txt` |

**`run8<>`, q_len 8 on native wgmma (W8, kernels `attn_8_*`)**
- **Lane:** 2 warpgroups (one per KV head) sharing a cp.async ring of 64-token stages. K is stored with the hardware 128-B swizzle.
- **QK:** 2× `wgmma m64n64k32` in RS mode. A = Q in registers; the 64 rows are 8 tokens × 8 query heads, which is exactly one wgmma M, so there is no padding.
- **Softmax:** FA3's FP8 scaling.
- **V^T:** each warp builds it with one `ldmatrix.x4.trans`, 4 `prmt` and one `stmatrix.x4` into a K-major tile.
- **PV:** 2× `wgmma m64n72k32`. The extra 8 columns are a ones block that yields the row sums. Each PV runs into a fresh accumulator that is folded into FP32 registers (§2b).
- **Page table:** entries ride in the cp.async ring DST − 1 tiles ahead (§2c).
- **SASS** of `attn_8_0_1_0`: 2× `QGMMA.64x64x32` + 2× `QGMMA.64x72x32`, 0 HMMA, 0 `UNPACK_B`.
- **k6 ring depth:** 3 for sliding instantiations, 4 for full (`K8_DST_S`, `K8_DST`).

**`run1<>`, q_len 1 on native wgmma (WD, kernels `attn_1_*`)**
- **Lane:** 1 warpgroup; 4 lanes per CTA; a ring of 3 × 16 KB stages of 64 tokens.
- **QK:** swap-AB, 2× `wgmma m64n8k32` SS per KV head, with Q^T written to shared memory once per segment.
- **Softmax:** speculative, against a warpgroup-uniform reference max. One `bar.red.or` vote ("any s·c > m_ref + 0.75") is fused with the P^T visibility barrier. Only if the vote fires do the warps exchange the tile max, raise m, rescale and recompute P; that is rare after the first tiles.
- **PV:** `O^T = V^T·P^T`, 2× `m64n8k32` RS. V^T is built in registers from 1 `ldmatrix.x4.trans` + 4 `prmt`, with no shared-memory transpose. Each PV goes into a fresh accumulator added in FP32.
- **Defaults:** `WD_NEWFIRST=1` attends the locally built new-token tile before the ring loop. `WD_EARLY=1` refills a stage at the vote barrier, keeping all DST stages in flight.
- **SASS** (image NVRTC 13): tile loop 428 instructions, 128 registers, 0 spills. The 8 `attn_1_*` run1 kernels of k5m hold 128 QGMMA in total, with 0 HMMA and 0 `F2FP.F16` (`notes/WD_NOTES.md` wd_t3, wd_t4).
- **Unit test** `t_dec.py`: O 5.7e-5, end-to-end 6.2e-3.

**j4 FLAT schedule (WB, inside `run1<>`)**
- **OVH virtual units:** every request in the FLAT unit list is charged `ovh` virtual units (`u = s − lo + 1 + p.ovh`, default 1536). CTAs that hold a request boundary therefore get fewer real tokens, which pays for the restart: merge + page-table hop + gather hop at loaded latency.
- **Fast issue path:** full tiles issue cp.async without bounds checks or src-size (`WD_FULLFAST`): 5 SASS instructions per copy instead of 10.
- **Combine:** the loads of up to 4 partials go out before any merging. When a request has a single combiner, it skips the barrier (`FASTCOMB`).
- **Rejected:**
  - PTQ (a page-table register queue): +8 µs.
  - RELFIRST / PTS (refill first, page-table ring for q_len 1): +10 / +3 µs from 7 LDL spills in the loop.
  - PEEK: +4-6 µs, 3 LDL (`notes/WB_NOTES.md` job2-job5).

**`runA1<>`, cluster combine for B = 1 sliding (WC, kernel `attn_1_1_1_1c`)**
- **Cluster:** the CTAs of a sliding q_len-1 request form one cluster, the largest size in (16, 8, 6, 5, 4, 3, 2) whose clusters all fit co-resident with ≥ 128 tokens per CTA. At B = 1 that is 16 CTAs × 128 tokens. Launched with `cuLaunchKernelEx` and a non-portable cluster size.
- **Combine:** after the lane merge, each CTA pushes every partial row (m, l, O) by `st.async` into the shared memory of the CTA that owns the row. The owner merges on an mbarrier with an expected transaction count. There is no global workspace, atomic or GPU-scope fence.
- **Verified speculation:** the host passes a guess `seq_g = max_pages·page_size − L` and its window bounds (`glo`, `gn1`). Each CTA loads its page-table rows for that guess at kernel entry, while tid 511 fetches the true length by cp.async (that thread issues nothing else, so the length barrier waits only for the length). After the barrier, the guess is confirmed (same first position, no more past tokens than guessed) and K/V are issued at once; otherwise the rows are reloaded for the true range.
- **RoPE table:** 32 `sincosf` per request instead of 4 per thread.
- **Spill-free tail:** nothing length-independent is held in a register across the barrier. The single segment leaves the loop explicitly, and post-loop values are re-derived from special registers and shared memory. Before this, post-loop spill reloads missed L1, since the ~219 KB of shared memory leaves little L1 (72 KB of stack per CTA).
- **The same single-segment exit was added to `run8<>`:** stack 72 → 32 B.
- **Batch > 1:** sliding stays on `run1`. The cluster path measured slower there: b16 9.71-10.07 vs 9.52 µs, b32 12.61-12.97 vs 11.81 (WC job7c).

How `runA1` converged (d1s, local µs, each row a same-job A/B; `notes/WC_NOTES.md`):

| job | build | d1s before → after | what we learned |
|---|---|---|---|
| 1 | cluster combine on k3 | 12.76 → 9.49 (8-CTA clusters, prefetch off; 16 CTAs 10.03) | prefetches placed before the first barrier cost +1.5 µs; post-loop spill reloads are cold |
| 2 | spill-free segment exit | 12.13 → 7.78 | |
| 3b | on k4: padded merge rows, unrolled cluster combine, RoPE table | 12.90 → 7.56 | |
| 4 | runA1 on k5: speculative K/V issued before the length barrier | 9.64 → 7.32 | the barrier moved 1.18 → 2.16 µs, because every warp now waited for page-table rows first |
| 5 | speculative rows as plain loads; K/V right after the barrier | 9.63 → 6.94 | runA1 without a cluster is 1.2 µs slower than run1, so b16/b32 and full b1 keep run1 |
| 7c | k6 v1 = k5m + W8 round 2 + runA1 | 8.50 → 6.63 | clusters for B = 1 only |
| 8 | k6 v2: length and speculative rows loaded at kernel top (**judged**) | 8.88 → 6.81 | top-loading itself −0.06 µs (noise) |

How `run8` converged (W8, local µs, same-job pairs; `notes/W8_NOTES.md`):

| step | v32f / v16f | what we learned |
|---|---|---|
| k2 (mma.sync tile) | 633 / 327 | |
| k2w: QGMMA core, 64-token stages, ones block N = 72 | 292 / 159 | tile 1083 → 417 cycles. Fence on/off and ring depth 3/4/5: no change |
| + page-table entries by cp.async | 262 / 146 | the per-tile page-table LDG was an exposed DRAM trip |
| k3w = k3 + run8 = **k4** | 253 / 142 | skeleton 181 / 99 |
| K8_PVDEFER (retire PV(i) in tile i+1) | 523 / 273 | 23 LDL / 16 STL in the loop at 128 registers |
| K8_PVSPLIT (P and PV per 32-token half) | | ptxas hoists every MUFU above the first PV QGMMA: no overlap |
| K8_PV64 (N = 64 + FP32 row sums) | | wrong output in the RoPE instantiation; not used |

## 5. Result

| check | result |
|---|---|
| local, same job | k3 x1.2271 → k3w (= k4) x1.6177 (`evidence/verify_k3w.txt`); k4 x1.6261 → j4 x1.6930 → k4d (= k5) x1.7764 (`evidence/verify_k4d.txt`); k5 x1.7496 → k6 x1.9191 (`evidence/verify_k6.txt`) |
| controls | WC job8, 40 reps: k6 x1.9108 vs k5m x1.7989 and k5w x1.7766. b1 shapes carry ±2-3 % run-to-run noise: identical code read d1s 9.08 vs 8.85 and d1f 15.24 vs 15.04 in one job (`notes/WD_NOTES.md` wd_t6) |
| correctness | `check.py` 0 failures (16 tests + 12 scored shapes, each called twice) on k3w, k4d and k6 (verify logs above; WC job8 56/56). Judge 16/16 on k4 and k6. Unit tests: W8 `t_core` CORE PASS; WD `t_dec` PASS |
| judge | k3 1.2010 → k4 **1.5724** → k6 **1.8412** |

Per case (judge µs; the share column = the score change from that shape's new time alone, `evidence/judge_score.py`):

| shape | SGLang (k6 run) | k3 | k4 | k6 | k3 → k4 share | k4 → k6 share |
|---|---|---|---|---|---|---|
| d1s | 20.86 | 13.35 | 13.20 | **8.12** | +0.13 % | **+5.17 %** |
| d1f | 31.39 | 18.46 | 18.04 | 17.23 | +0.12 % | +0.24 % |
| d16s | 26.59 | 15.50 | 15.47 | **10.76** | +0.01 % | +1.52 % |
| d16f | 200.86 | 111.85 | 111.84 | **91.78** | 0.00 % | +2.32 % |
| d32s | 31.47 | 19.03 | 19.07 | **13.53** | −0.01 % | +1.10 % |
| d32f | 384.38 | 195.64 | 195.34 | **170.28** | +0.02 % | +1.77 % |
| v1s | 22.93 | 22.99 | **18.69** | **16.73** | +2.29 % | +1.28 % |
| v1f | 36.46 | 31.31 | **23.13** | 21.39 | +1.47 % | +0.38 % |
| v16s | 28.98 | 22.55 | 19.60 | **16.13** | +0.37 % | +0.84 % |
| v16f | 205.62 | 318.14 | **148.62** | 147.80 | **+10.23 %** | +0.07 % |
| v32s | 34.26 | 30.54 | 23.05 | **19.34** | +0.53 % | +0.55 % |
| v32f | 390.03 | 608.36 | **263.71** | 262.03 | **+12.06 %** | +0.09 % |

Per load:

| run | d1 | d16 | d32 | v1 | v16 | v32 |
|---|---|---|---|---|---|---|
| k4 | 1.613 | 1.778 | 1.903 | 1.322 | 1.414 | 1.482 |
| k6 | 2.252 | 2.260 | 2.270 | 1.473 | 1.488 | 1.538 |

**Judge, read honestly**
- **k3 → k4:** the six decode shapes ran the same q_len-1 code. They moved by −0.42 to +0.04 µs (d32f −0.30), so the two nodes were comparable, and the +30.9 % is the verify path.
- **k4 → k6:** v16f and v32f did not change locally (k5 → k6: 142.05 → 142.66 and 255.37 → 253.78). On the judge they read −0.8 / −1.7 µs: again a comparable node.
- **Judge vs local:** on k6, the judge reads +0.65 to +1.9 µs per call above local on the small and decode-full shapes, and +5.1 / +8.3 µs on v16f / v32f. The local judge projection (+1 µs per call) was 1.7 % (k4) and 2.1 % (k6) too optimistic.
- **Real level:** 1.5724 and 1.8412. Each build has a single reading.

**Against the projection**
- **Verify full** reached 253 / 142 µs locally, not the 181 / 99 skeleton. About 72 µs of compute remains at v32f; the ablations put ~47 µs of it on the serialized PV wait and ~14 on MUFU (§2d). The judge's v32f is still 1.49x faster than SGLang, against the 2.6x the gather floor allows.
- **b1 sliding** landed where the estimate said: 6.51 µs locally (`verify_k6.txt`), 8.12 µs on the judge.
- **Decode full** (k5m): d32f 169.5 against its own skeleton at 167.4. Compute is down to ~2-3 µs, and the remaining ~15 µs gap to the 152 µs gather floor is scheduling (`notes/WB_NOTES.md` job6).
- **d1f** improved only −0.8 µs. Its ALIGNED run1 spends ~7.4 µs after the tiles (`notes/WD_NOTES.md`, k5m trace: tiles 8.30 → combined 15.73).

## 6. Takeaways, and what came next

- **On sm_90a, FP8 must go through wgmma.** The native QGMMA tile is 2.6× the emulated mma.sync tile per SM. It turned verify full from 0.65x to 1.4-1.5x SGLang in one step, worth about +22 % score (v16f + v32f shares).
- **Never chain FP8 QGMMA accumulation across tiles.** Start each tile's PV from zero and fold in FP32.
- **For q_len 1, swap-AB `m64n8k32` with a vote-gated speculative max is padding-free and cheap.** Decode compute overhead fell to ~6 µs at d32f and ~1.5 µs at d1s.
- **In a FLAT split, a request boundary is a real cost (~12 µs).** Model it as virtual units in the split rather than balancing tokens alone.
- **At B = 1, the end chain is the lever, and a DSMEM push beats the global combine:**
  - `st.async` + mbarrier carries no GPU-scope membar;
  - but at B > 1 the plain run1 stays faster;
  - nothing may be spilled across the segment loop, because the reloads after it miss L1.
- **Verified speculation pays only if it doesn't delay the barrier.** Issuing K/V before the length barrier made every warp wait for the page-table rows.
- **What came next** (`notes/PLAN.md`, "The real bar", written from the k6 judge run):
  - The judge adds ~1.5 µs per call to every shape.
  - Every shape now had to approach its floor. Local targets: sliding b1 ≤ 3.7 µs, full b1 ≤ 6.5, b16/b32 full ≤ 80 / 158 for both decode and verify.
  - The following builds (k7-k13) attacked the start and end chains of the run8 ALIGNED and FLAT paths and W8's verify-full loop: `03_*`.

## Reproduce

All you need is torch and an H200. `tools/` holds the problem spec, `reference.py`, `checks.py`, `baseline.py` and our harness.

```bash
cd muse_attention_decode/tools
python bench.py ../02_native_qgmma_and_cluster_sliding/build.py --tests                  # correctness + local judge-like timing
python ab.py ../01_*/build.py ../02_native_qgmma_and_cluster_sliding/build.py \
       --shapes d1s,d16f,v1s,v32f --seeds 7,8,9 --caps 2                                 # same-run A/B, 01 vs 02
python ab.py ../02_*/build.py ../02_*/build.py --env ';NOSPEC' --shapes d1s              # verified B=1 speculation on/off
python ab.py ../02_*/build.py ../02_*/build.py --env ';NOCLU' --shapes d1s               # cluster combine on/off
python ../02_*/evidence/judge_score.py ../02_*/evidence/judge_mid_k4_check_c284726b.txt \
       ../02_*/evidence/judge_after_k6_check_c763bae8.txt                                # per-shape shares
# tile microbenchmark: runs inside the era-0 tree E0 (needs dev/w8/k2w.py beside it)
cd $E0 && python3 dev/w8/t_core.py
```

Originals: `../autoresearch-kernels/attn/subs/k6_check.py` (= `dev/k6.py` = `dev/wc/k6.py`, built by `dev/wc/build_k6.py` from `dev/wd/k5m.py`, `dev/w8/run8.cu` + `dev/w8/k2w.py` and `dev/wc/runa1.cu`); intermediate `subs/k4_check.py` (= `dev/k4.py` = `dev/w8/k3w.py`) and `dev/k5.py` (= `dev/wd/k4d.py`); lanes `dev/w8/`, `dev/wd/`, `dev/wb/`, `dev/wc/`; judge reports `subs/profiles/k{3,4,6}_check.*`; local runs `out/verify_k3w.txt`, `out/verify_k4d.txt`, `out/verify_k6.txt`, `out/20261027T233403_k3w/`, `out/20261027T235727_w8final/`, `out/20261027T232635_w8prof/`, `out/20261028T004446_k5w/`.
