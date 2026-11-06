# 06: one compact kernel per shape class, each in its own module: XE, XA, XD (d1s, v1s, d1f, v16f, v32f)

Shape names: d / v = decode (q_len 1) / verify (q_len 8); 1 / 16 / 32 = batch; s / f = sliding (2048-token window) / full (~80K context). Era-0 files live outside this tree, under `../autoresearch-kernels/attn/` (written "E0" below).

| | before (05) | after (06) |
|---|---|---|
| build | `05_*/build.py` = E0 `subs/k32_submit.py` (md5 752d7dec, 488,392 B) | `build.py` = E0 `subs/k37_submit.py` = `subs/k37_check.py` = `dev/coord/k37.py` (md5 187c7ec5, 677,675 B) |
| judge | 2.5543 (scored 820fe912) | okc check **2.6836** (ab77bf33) and scored **2.6724** (a90f3cd5), byte-identical (same sha256 49a97ccc…): real level **2.678**. Intermediate k36 (= without XD): okc check 2.6572 (ad018c2e) |
| real effect | | +4.8 % as read (real level vs k32's single reading). **≈ +3.7 % on our side**: +3.96 / +3.46 % with SGLang's times held fixed; the rest is SGLang reading slower in the k37 runs. Local same job vs k34: +4.2 % (judge-projected +3.6 %) |
| shapes changed | | d1s, v1s (XE), d1f (XA), v16f, v32f (XD). The other 7 shapes run byte-identical kernels |

**ROUND 2** is the team's second phase. It stopped patching the monolithic kernel and gave each shape class to a lane that writes a new, small kernel in its own NVRTC module, from scratch:
- **XE**: batch-1 sliding.
- **XA**: batch-1 full.
- **XD**: verify full B > 1.

A module is appended to the base file with a `kernel_fn` wrapper whose predicate looks only at shapes and config. Anything outside the predicate falls back to the base kernel (`notes/ROUND2.md`, "Integration convention").

## 1. Starting point

The base for ROUND 2 was k34 = k32 + two unscored round-1 pieces, both later overridden on the scored shapes (see Change). Times for the targeted shapes:

| shape | k34 kernel | judge k32 (µs) | local k34 (µs) | nsys k34 (µs) | SGLang (µs) |
|---|---|---|---|---|---|
| d1s | `attn_1_1_1_1s` (runA1h, 2 clusters of 16) | 4.17 | 3.3 | 3.81 | 20.3 |
| v1s | `attn_8_1_1_1h` (runA8h) | 5.55 | 4.94 | 5.52 | 22.3 |
| d1f | `attn_1_0_1_1w` (run1w, 132 CTAs) | 14.11 | 12.0-12.6 | 13.25 | 30.7 |
| v16f | `attn_8_0_1_0` (run8 FLAT) | 104.99 | 98-100 | 100.0-100.8 | 205.3 |
| v32f | `attn_8_0_1_0` | 188.63 | 178-182 | 184.3-184.9 | 389.8 |

Sources:
- Judge: `evidence/judge_before_k32_submit_820fe912.txt`.
- Local: `notes/ROUND2.md`, gap table.
- nsys d1s / v1s: `evidence/xe_job9_nsys/k34_*_parsed.json`; d1f: `evidence/xa_j14_final.txt`; v16f / v32f: `evidence/xd_n1_nsys.log`.

Score per µs saved (ROUND2.md): d1s 1.36 %, v1s 0.95 %, d1f 0.46 %, v16f 0.10 %, v32f 0.06 %. The batch-1 shapes carry the most score per µs.

## 2. What we saw

**(a) The 12-shape profile of k33 (WP lane, `evidence/WP_PROFILE_k33.md`, "Cross-shape patterns"):**
- **The judge sees the kernel duration.** The flush → kernel gap is a constant 4.1-4.3 µs on every shape and on the empty graph alike. Judge-visible time ≈ the nsys kernel duration.
- **Sliding shapes are latency chains:** issue active 25-48 %, barrier the top stall (21-43 % of samples on the cluster kernels), and 5-14 % front-end stalls (cold instruction fetch after the judge's L2 flush).
- **Batch-1 full layers use 26-33 % of HBM.** The 20.5 MB gather takes 5.4-6.0 µs, so 7-11 µs is start burst and merge / publish / combine chain.
- **Verify full B ≥ 16 is issue / ALU-limited:** issue active 57-58 %, ALU 49-51 %, tensor 26-27 %. It executes 48 / 94 M warp instructions, against 35.5 / 65.8 M for decode full on the same bytes.

**(b) The SASS census (WI lane, `evidence/WI_CENSUS_k33.md`, summary table):**
- Only the FLAT full kernels have a steady-state loop. Every other kernel runs fewer than 6 tiles per warp, so its cost is its one-shot code.
- Executed code before the loop: d1s 13.0 KB, v1s 20.7 KB, d1f 21.4 KB.
- Executed code after the loop: d1f 8.3 KB, v1f 15.7 KB, v16f / v32f 18.8 KB.
- One-shot start-path code cost ~0.25 µs per KB: 3.4 KB more of it, on a prologue already 13 KB long up to the last LDGSTS, cost v32s ~0.8 µs (`notes/PLAN.md` "ROUND 2"; E0 `dev/wh/NOTES.md`, Round 3).

**(c) Round 1 had run out of in-place levers** (`notes/PLAN.md`, entries after k32):
- Verify full: rounds 13-17 (W9, with WB) all lost: half-row decoupling, mid-tile refill (WB: +13 / +22 µs), DST 3/5, 3 lanes, an issue-path diet, freeing registers. The run8 FLAT loop was at its practical floor in that structure.
- Batch-1 full on clusters was killed: 7 × C16 cost +0.30 µs, 15 × C8 +0.1-0.2.

**(d) XE, d1s and v1s.**

k34's critical path, flush traces in µs from the first CTA start (`notes/XE_NOTES.md`, job1):

| event | prefix loads | page table issued | length barrier | K/V issued | tile 0 | tiles done | output |
|---|---|---|---|---|---|---|---|
| d1s (runA1h) | 0.47 | 0.82 | 1.09 | 1.25 | 1.86 | 2.37 | 3.18 |
| v1s (runA8h) | 0.84 | | 1.41 | 1.57 | 2.17 | 2.96-3.09 | 4.92 |

- The first load leaves 0.5 / 0.85 µs after CTA start: a cold prologue plus constant misses.
- XE's floor probe on the real d1s inputs runs the skeleton: start, page table, K/V, merge, push, owner store. It takes **~1.8 µs**, against k34's 3.04 in the same job. The room was the cold prologue (~0.5 µs), the tile phase (~0.5) and the tail.

**(e) XA, d1f.** Critical-path budget of k34 (`notes/XA_NOTES.md`, "k34 baseline profile"), with a no-math probe for comparison:

| phase | k34 (µs) | probe (µs) |
|---|---|---|
| start → issue done | 3.8 | 2.27 |
| gather to the last tile | 9.0 | 5.9 (CTA completion) |
| end chain (merge 1.0 + publish 1.0 + combine 1.5) | 3.5 | |

ncu: DRAM at 33 % of peak; stalls are barrier 22 % and long-scoreboard 16 %.

**(f) XD, verify full: power, not the loop skeleton** (`notes/XD_NOTES.md` §0-§3; `evidence/xd_mb2_microbench.log`, `evidence/xd_pw2_power.log`, `evidence/xd_prof0_ncu_k34_v32f_*.txt`).

ncu of k34's v32f: DRAM 73.1 %, issue slots busy 55.3 %, 1.22 eligible warps per scheduler out of 4.03.

Microbenchmark of the tile phase alone, b32, 643 MB:

| variant (job xd_mb2) | µs | SM clock (MHz) |
|---|---|---|
| k34 lockstep loop, no compute (skeleton) | 156.7 | 1931 |
| compute alone, resident (no gather) | 144.0 | 1806 |
| both together | 170.0 | 1656 |
| k34 full kernel, same data | 182.4 | |

In the first microbenchmark job (xd_mb1) the same skeleton ran at 150.3 µs, i.e. 4.28 TB/s: the gather ceiling.

Power soak, 3 s back-to-back:

| variant | power | SM clock | throttle |
|---|---|---|---|
| gather alone | 656.5 W | 1980 MHz | none |
| tile compute alone | 396-400 W | 1980 MHz | none |
| both together | 685.6 W | 1390-1583 MHz | SW POWER CAP (700 W limit) |

- **Verify full is power-bound:** its time tracks total energy, so every instruction, smem byte and tensor op removed also buys clock.
- **Decoupling the two warpgroups lost:** XD-M was +6.5 µs (b32) and +2.6 (b16), because the refill issue piles onto the lagging warpgroup.
- **A lean K/V issue cut resident compute 145 → 134 µs,** but end to end only −1.5 … −4.5.
- **A 128-token step (T128: one max / rescale / fold per 128 tokens) cut resident compute 12-18 %,** but end to end only −1.3 / −2.5 µs.

## 3. Diagnosis and projection

- **For batch-1 shapes, k34's cost was one-shot code and serial chains, not bandwidth.**
  - Dedicated kernels in their own modules carry small code, have no generic dispatch, and keep prologue work off the critical path.
  - ROUND2.md targets (local), with the score each would add: d1s ≤ 2.8 (+1.0 %), v1s ≤ 3.9 (+1.9 %), d1f ≤ 9.0 (+3 %).
- **For verify full, the bare tile at the gather ceiling is ~150 µs (b32).**
  - The k34 kernel spends ~7.5 µs on start / segment / end structure and ~20 µs on power-capped compute coupling.
  - Targets: v16f ≤ 88 (+1.4 %), v32f ≤ 168 (+0.9 %).
  - A tile with less energy per token (lean issue, T128 steps, f16-accumulated QK) was the lever XD could pull without fighting the power cap.
- **Even with every ROUND2 target hit, the projection was only ~x2.97 on the judge** (`notes/ROUND2.md`, coordinator note).

## 4. Change

`change.diff` (`diff -u subs/k32_submit.py subs/k37_submit.py`) adds 3,747 lines and removes 14. Most of it is three appended modules: `XE_SRC`, `_XA_SRC`, `_XD_SRC`, each compiled by NVRTC with its own wrapper.

The rest is k34's two round-1 pieces. They were never judged on their own, and the new modules override both on the scored shapes:

| build | piece | lane | shapes | local effect | source |
|---|---|---|---|---|---|
| k33 | `W9E` d1f end chain on run1w: the fixed-reference accumulator split into 4 shards (`W9E_NC`), double-buffered by call parity (`W9E_PB`); the last arrival issues all its loads at once (`W9E_LD`) | W9 | d1f, superseded on scored d1f by XA | −0.40 µs | `evidence/verify_k33.txt`, `notes/W9_mk_k34d.py` |
| k34 | branch-free producer K/V issue in runA1h (`WS_PD 1`) | WS | d1s, superseded by XE | −0.09 … −0.11 µs | `05_*/notes/WS_NOTES.md`, Round 9 |
| k35 | **XE** round 1 (`xe_b1s_1`, `xe_b1s_8`) | XE | d1s, v1s | d1s 2.53-2.66 vs k34 2.92-3.00; v1s 3.82-3.90 vs 4.63-4.81. nsys d1s 3.36 vs 3.81, v1s 4.61 vs 5.52 | `notes/XE_NOTES.md` job9; `evidence/xe_job9_nsys/` |
| k36 | **XA** round 2 (`xa_b1f_1L`) | XA | d1f | 10.26 vs k34 12.16 / 12.33; nsys 11.36 vs 13.25 | `notes/XA_NOTES.md` j14; `evidence/xa_j14_final.txt` |
| k37 | **XD** round 1 (`attn_xd_v8f`) | XD | v16f, v32f | v16f 96.79 vs 99.74 / 99.78; v32f 179.78 vs 182.37 / 182.19. nsys −2.9 / −4.7 µs | `notes/XD_NOTES.md` §4; `evidence/xd_b3_bench.log`, `xd_n1_nsys.log` |

**XE (`xe_b1s_1` for d1s, `xe_b1s_8` for v1s).**

Layout:
- One cluster of 16 CTAs per KV head for d1s (grid 16 × 2, 160 threads). For v1s, one per (KV head, group of 4 query tokens) (grid 16 × 4, 256 threads).
- CTA rank r owns keys [r·C, r·C + C) of the window, C = ⌈past / 16⌉ ≤ 128. Each compute warp owns 32 of them.

Math is register-direct, with no shared memory for K/V:
- Each thread loads exactly its operands of FP16 `mma.sync.m16n8k16`: 16 B of K per key for S^T = K·Q^T, and 8 B of V per key for O^T = V^T·P^T.
- E4M3 converts to FP16 exactly; accumulation is FP32. P^T comes out of the QK accumulators by `movmatrix.trans`; row sums come from an all-ones tile.

Work split:
- Prep warps (one for d1s, four for v1s) build the RoPE rows, the Q fragments and the new tokens' cache rows, off the critical path. The compute warps only issue K/V and wait for "Q ready".
- Each CTA merges with one named barrier and pushes its partial by `st.async` into the row owners' shared memory (mbarrier completion).
- The owner warps fold the 16 pushes, attend the new token(s), and apply RMSNorm × sigmoid(gate). There is no final cluster barrier.
- Batch-1 length speculation as in 05: the guessed seq = width − L, confirmed by the length load.

How it converged (`notes/XE_NOTES.md`; same-job µs, k34 controls):

| version | change | d1s / v1s | what it taught |
|---|---|---|---|
| v1 (job2) | compute warps also prepare Q, RoPE and the new tokens | 3.35 / 5.49 vs 3.08-3.12 / 4.93 | the ~355-instruction straight-line prep chain took 1.2 µs, longer than the ~0.6 µs K/V trip |
| v2 (job3) | separate prep warps | 2.79-2.83 / 3.88-3.96 vs 2.87-2.93 / 4.63-4.81 | v1s −0.8 µs |
| job4 | push variants | `st.async` 2.63 / 3.80 vs bulk copy 2.79 / 4.45 | staging for a bulk copy (STS + fence + barrier) costs 0.3 / 0.7 µs; FP16 O payloads neutral |
| v4 (job7) | conflict-free reader; owner pre-warm for q_len 1 | 2.63 vs 2.75 without pre-warm | the owner's first LDS after the mbarrier took 574 cycles cold vs 164 warm (job6): instruction fetch |
| v5 = final (job9) | | 2.53-2.66 / 3.82-3.90 | errors 3-7e-3 (k34 6.9e-2) |

**XA (`xa_b1f_1L`, d1f).**

Layout:
- 132 CTAs × 512 threads.
- WS's best gather shape: 4 lanes × 3 stages of 64 tokens. A lane is the 4 warps sharing one ring.
- The QK of each tile runs on the lane's FP8 wgmma (m64n8k32, Q^T in E4M3 in smem). The softmax and the PV `mma.sync` run per warp.

Start:
- The batch-1 split is speculative (n = width − 1), verified after the burst.
- The burst is issued before waiting for Q.
- The Q copies get an opaque dependency on the page-table entries (`XA_QDEP`), so ptxas cannot schedule them ahead of the critical page-table loads.

End:
- CTA partials are added at the fixed reference REF = 64 into 4 shards (double-buffered by call parity).
- Out-of-range CTAs store an exact partial and a list entry.
- The last arrival sums the shards and runs the epilogue.

How it converged (`notes/XA_NOTES.md`; d1f µs, k34 ≈ 12.2-12.5 in the same job):

| version | change | d1f | what it taught |
|---|---|---|---|
| v1 (j2) | warp-private key ranges, FP16 `mma.sync` | 13.12 | the end chain took 4.5 µs: cold, unrolled merge and epilogue in a ~40 KB kernel |
| v2 (j3-j4) | Q staged with `cp.async.cg` by every warp | 32-35 | 270K L1-bypassing requests on one 2 KB L2 hot spot: an ~18 µs stall. `cp.async.ca` fixed it |
| v2 (j5) | warm-up dry pass at kernel start | 12.53 vs 11.43 without | the pass is itself a 1.5 µs cold fetch, and it delays the burst |
| v3 (j7-j8) | burst issued before the Q wait | 11.11 / 10.95 | |
| v6 (j13) | lane-structured gather, QK on FP8 wgmma (`XA_L1`) | 10.19 | the lane shape issues faster, and the per-warp K LDS + cvt + QK HMMA disappear |
| final (j14) | | 10.26 vs 12.16 / 12.33 | check.py and t_xa: 0 failures |

**XD (`attn_xd_v8f`, verify full B > 1).**

A new kernel replaces run8 FLAT for q_len 8, full layer, page 1, tp8, 1 < B ≤ 128:
- **Split:** k34's FLAT split, segments, lane merge and end chain, as a compact copy.
- **Tile loop:** T128 pairs (`xd_tile2`), one max / rescale / fold per 128 tokens.
  - QK is accumulated in f16 (QV 2): 179.78 vs 181.30 µs with an FP32 QK on v32f.
  - 5 ring slots of 16 KB.
  - Lean K/V issue: tokens strided by 8, so the swizzled destinations are immediates, and one warp copies each tile's page-table chunk.
- **Smem:** the gates move out of shared memory (an L2 evict_last prefetch per segment) to fit 5 stages.
- **Reductions** go through smem + `__syncwarp`, because shuffles there serialize the wgmmas (ptxas warning C7515).

Correctness: `xcheck.py` (12 edge cases) and `check.py` give 0 failures. Errors match k34 (v16f 2.77e-2 at bound 0.082; v32f 2.68e-2 at 0.080).

Remaining gap on v32f, against a decode-like ~158 µs (XD §6):
- ~3 µs of structural coupling under the power cap.
- ~1-2 µs of possible CUDA-core cuts.
- ~5-6 µs of start to tile 0.
- ~7.5 µs of end chain.
- ~4 µs of per-CTA spread with equal work.

**Stacking from frozen snapshots** (`notes/SNAPSHOTS.md`). Each lane froze its deliverable, and the coordinator built only from the frozen copies:
- k35 = k34 + `snap/xe_mod.py`.
- k36 = k35 + `snap/xa_mod.py`.
- k37 = k36 + `snap/xd_mod.py`.

The first k36 was stacked from XE's live working file, in which round-2 work was in progress. It read d1s 4.00 / 3.79 µs against 2.72 / 2.59 for the frozen XE build in the same job (`evidence/k36ab_live_vs_frozen.txt`). It was rebuilt from snapshots before any judge run.

## 5. Result

| check | result |
|---|---|
| local, same job vs two k34 copies | k36 (XE + XA) +4.26 % (judge-projected +3.64 %); k37 (+ XD) +4.23 % (+3.60 %) (`evidence/verify_chain.txt`). k37 per shape: d1f −2.09, v1s −0.73, v16f −2.74, v32f −2.73, d1s −0.17 µs (`evidence/verify_k37.txt`). The same job moved d32s +0.47 and d32f +0.48 on identical code |
| correctness | `check.py` 0 failures for k36 and k37; judge 16/16 tests (all three runs) |
| judge | 2.5543 → k36 check 2.6572 → k37 check **2.6836**, k37 scored **2.6724** |

Per case, judge µs (`evidence/judge_readings.txt`):

| shape | kernel in k37 | k32 | k36 | k37 check | k37 scored | mean k37 − k32 | local k37 − k34 |
|---|---|---|---|---|---|---|---|
| d1s | `xe_b1s_1` | 4.17 | 3.58 | 3.79 | 4.01 | −0.27 | −0.17 (k36c: −0.34) |
| d1f | `xa_b1f_1L` | 14.11 | 11.59 | 11.70 | 11.76 | **−2.38** | −2.09 |
| v1s | `xe_b1s_8` | 5.55 | 5.04 | 5.32 | 5.18 | −0.30 | −0.73 |
| v16f | `attn_xd_v8f` | 104.99 | 106.61 | 99.16 | 99.95 | **−5.43** | −2.74 |
| v32f | `attn_xd_v8f` | 188.63 | 190.14 | 181.18 | 180.28 | **−7.90** | −2.73 |
| d16s, d32s, v16s, v32s | k32's | 8.66, 10.34, 10.36, 14.48 | 8.86, 10.23, 10.29, 14.49 | 8.82, 10.43, 10.56, 14.43 | 8.92, 10.51, 10.97, 14.53 | +0.21, +0.13, +0.41, 0.00 | identical code |
| d16f, d32f, v1f | k32's | 87.76, 165.47, 17.57 | 88.10, 166.04, 17.11 | 87.57, 166.29, 17.45 | 88.19, 165.68, 17.22 | +0.12, +0.52, −0.24 | identical code |

**Reading the judge honestly:**
- **The 7 identical-code shapes moved −0.35 … +0.82 µs between runs.** Summed, they are worth −0.1 … −0.3 % of score.
- **d1s read 3.58 / 3.79 / 4.01 on identical XE code across the three runs.** The judge's node spread on d1s is ~0.45 µs in absolute terms (XE Round 6, measured later). The mean d1s gain (−0.27) is barely above it.
- **On k36 the new batch-1 kernels read above their local times** by d1s +0.97, v1s +1.26 and d1f +1.39 µs (`notes/PLAN.md`, okc check k36 entry). The judge shrank XE's v1s gain (local −0.73 … −0.94) to −0.30 on the k37 mean.
- **XD's v16f / v32f read −5.4 / −7.9 µs on the judge** (−6.2 / −8.7 against the mean of k32 and k36, which ran the same run8 FLAT kernel). Locally they were −2.7 / −2.7, and in nsys −2.9 / −4.7 (`evidence/xd_n1_nsys.log`). A plausible reason is that judge nodes sit deeper in the power cap, where XD's lower energy per tile buys more clock (inferred; not measured on the judge).
- **SGLang read slower in both k37 runs.** With SGLang's times held fixed, k32 → k37 is +3.96 % (check) and +3.46 % (scored). As read it is +5.1 / +4.6 %.
  - So the real effect is ≈ +3.7 %, matching the local judge-projected +3.6 %.
  - The real level 2.678 (mean of the two byte-identical runs) is the number to carry forward.

**Against the targets** (local, ROUND2.md):

| shape | target | best local | met |
|---|---|---|---|
| d1s | ≤ 2.8 | 2.16-2.66 | yes |
| v1s | ≤ 3.9 | 3.50-3.90 | yes |
| d1f | ≤ 9.0 | 9.74-10.26 | no |
| v16f | ≤ 88 | 96.4 | no |
| v32f | ≤ 168 | 178.4 | no |

- **d1f:** a ~2.5 µs end chain was left after the slowest CTA's loop.
- **Verify full:** stuck ~20 µs above the gather ceiling. The cause is structural coupling under the 700 W cap, plus the start, the end chain and the CTA spread. XD's later ablations (round r2) bounded any further tile-energy work at ~1-2 µs.

## 6. Takeaways, and what came next

- **When the cost is one-shot code and serial chains, write a compact kernel per shape class.** That is the case for every batch-1 shape. XE and XA took 13-21 KB of executed prologue code out of the path and moved non-critical work (Q prep, RoPE, the new tokens) onto separate warps.
- **Verify full is power-capped: time ≈ energy.** HBM alone draws ~650 W of the 700 W cap. The levers are fewer instructions, smem bytes and tensor ops per token; overlap and decoupling do not help.
- **Integrate only frozen deliverables.** One stack built from a live worker file lost ~1.2 µs on d1s before anyone noticed.
- **Read the judge per shape:** identical code moved ±0.6 µs per shape between runs, and SGLang's side moved the score by ~1 %. Byte-identical repeats give the real level.
- **Next, `07_*`** kept iterating inside these modules:
  - XE round 2 (v1s on clusters of 8) and round 4 (decode sliding B ≤ 16).
  - XA round 3 onward on v1f.
  - XB's d1f start and end chain, XC's verify-full end chain, XG's tagged v1f combine, and XF's run1f end chain.
  - All of it stacked to k55.

## Reproduce

```bash
cd muse_attention_decode/tools
python bench.py ../06_per_class_modules_xe_xa_xd/build.py --tests                       # correctness + local judge-alike timing
python ab.py ../05_*/build.py ../06_per_class_modules_xe_xa_xd/build.py \
       --shapes d1s,v1s,d1f,v16f,v32f,d16s --seeds 7,8,9 --caps 2                      # same-run A/B; d16s is identical code (control)
# no GPU: the judge and same-job readings used above
cd ../06_per_class_modules_xe_xa_xd/evidence
python judge_readings.py judge_before_k32_submit_820fe912.txt judge_k36_check_ad018c2e.txt \
       judge_after_k37_check_ab77bf33.txt judge_after_k37_submit_a90f3cd5.txt
python verify_chain.py verify_k37.txt
```

The lane probes ran on Modal from the era-0 tree and need its `tools/`:
- XE floor probe: `dev/xe/probe.py`.
- XD tile microbenchmark and power soak: `dev/xd/mb.py` with `dev/xd/xdcore.cu`.
- Profiles: WP's `dev/wp/run_nsys.py` / `run_ncu.py`.

Originals:
- Build: `../autoresearch-kernels/attn/subs/k37_submit.py` (= `subs/k37_check.py` = `dev/coord/k37.py`); the intermediate is `dev/coord/k36.py` (= `subs/k36_check.py`), and the base is `dev/k34.py`.
- Frozen modules: `dev/coord/snap/{xe,xa,xd}_mod.py`.
- Lanes: `dev/{xe,xa,xd}/` (XD's frozen hand-off: `dev/xd/deliver_r1/`).
- Profiles: `dev/wp/PROFILE.md`, `dev/wi/CENSUS.md`.
- Judge results: `subs/profiles/k3{6,7}_*`. Verifies: `out/verify_k36c.txt`, `out/verify_k37.txt`, `out/k36ab.txt`.
