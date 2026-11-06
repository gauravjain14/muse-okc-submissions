# 07: shorter end chains, and cluster kernels for d16s and v1s (10 of 12 shapes; mostly v1f)

| | before (06) | after (07) |
|---|---|---|
| build | `06_*/build.py` = era-0 `k37` (md5 187c7ec5, 677,675 B) | `build.py` = era-0 `k55` (md5 f0bff1b7, 974,512 B; sha256 e75d0cbd...) |
| judge | 2.6836 (okc check ab77bf33) and 2.6724 (scored a90f3cd5): real **2.6780** | **2.9266** (scored 945f42a6, fast node) and **2.8712** (okc check 489b863d of the byte-identical file, `k55_recheck.py`, slow node): real **2.8989** |
| real effect | | **≈ +6 %**, from two estimates that agree. Weighting each changed shape's slow-to-slow Δ by its score sensitivity gives +6.1 %. Re-scoring the judge JSONs with one common replay overhead (W4's `rejudge.py`, made in `08_*`; `evidence/rejudge_k37_k55.txt`) gives 2.6999 → 2.8571 = +5.8 % slow-to-slow, with the controls d1s / d32s flat (−0.05 / −0.07 µs). The raw slow-to-slow move is +7.2 % and the real-level means move +8.2 %; the rest is node and estimator luck. Locally ≈ +6.4 % (inferred) |
| shapes changed | | v1f (most), d1f, d16s, v1s, v16f, v32f, v16s; small: d16f, d32f, v32s. **d1s and d32s run unchanged code** (controls) |

Shape names: d = decode (q_len 1), v = verify (q_len 8); 1 / 16 / 32 = batch; s / f = sliding window / full context.
The **end chain** is everything after a CTA's last K/V tile. The CTA publishes its partial (O, m, l), signals arrival, and
waits for the other CTAs that share its request. A combiner then merges the partials, applies RMSNorm × sigmoid(gate), and stores bf16.
On the latency-bound shapes, each dependent L2 or DRAM round trip in that chain costs ~0.4-1 µs.

This milestone stacks the round-2 work of seven workers. Each worker owned one shape class and one NVRTC module inside the build:

| worker | module / kernel | shapes |
|---|---|---|
| XA | `_XA_SRC`: `xa_b1f_1L` (d1f), `xa_b1f_8` (v1f) | batch-1 full |
| XB | `_XB1_SRC`: `xb_d1f` | d1f (from k45 on); the start of the verify-full kernel |
| XC | the end chain of `attn_xd_v8f` (`XC_END`, `XC_XDF`) | v16f, v32f |
| XD | `_XD_SRC`: `attn_xd_v8f`; also the v16s combine | v16f, v32f, v16s |
| XE | `XE_SRC`: `xe_r1_1` (d1s), `xe_b1s_8c6` (v1s), `xe_dm_8_4_2` (d16s) | batch-1 sliding, d16s |
| XF | patches to the base `SRC`: run1f `attn_1_0_1_0` end chain, run8w `attn_8_1_1_1w4` | d16f, d32f, v32s |
| XG | `_XG5_SRC`: `xg_v8f` | v1f (from k47 on) |

The base `SRC` also holds runE1 (`attn_1_1_1_1e3`, d32s) and run8w x6 (`attn_8_1_1_1x6`, v16s).

## 1. Starting point

Judge times of k37 in µs (`evidence/judge_before_*`). The kernel per shape is from the same files' profiles.
The %/µs column is score gain per µs saved at these times: 100 × weight / (6 × load time) (inferred from TASK.md's formula).

| shape | kernel (k37) | check ab77bf33 | scored a90f3cd5 | %/µs |
|---|---|---|---|---|
| d1s | `xe_b1s_1` | 3.79 | 4.01 | 2.09 |
| d1f | `xa_b1f_1L` | 11.70 | 11.76 | 0.73 |
| d16s | `attn_1_1_1_1e6` (runE1) | 8.82 | 8.92 | 0.42 |
| d16f | `attn_1_0_1_0` (run1f) | 87.57 | 88.19 | 0.15 |
| d32s | `attn_1_1_1_1e3` (runE1) | 10.43 | 10.51 | 0.24 |
| d32f | `attn_1_0_1_0` | 166.29 | 165.68 | 0.085 |
| v1s | `xe_b1s_8` | 5.32 | 5.18 | 1.48 |
| v1f | `attn_8_0_1_1` (base `SRC` run8) | 17.45 | 17.22 | 0.51 |
| v16s | `attn_8_1_1_1x6` (run8w) | 10.56 | 10.97 | 0.37 |
| v16f | `attn_xd_v8f` | 99.16 | 99.95 | 0.13 |
| v32s | `attn_8_1_1_1w4` (run8w) | 14.43 | 14.53 | 0.21 |
| v32f | `attn_xd_v8f` | 181.18 | 180.28 | 0.075 |

Both k37 runs were slow nodes (d32f 165.7-166.3; a fast node reads ~158.5, see Result). v1f was still on the old kernel and
the furthest from its floor: 17.3 µs against 4.3 µs of data.

## 2. What we saw

**(a) v1f: the end chain, not the gather, held most of the excess** (`notes/XA_NOTES.md` rounds 3-4, `notes/XG_NOTES.md`).
- nsys split of XA's kernel at 14.88 µs: skeleton (start + gather + publish, no compute, no combine) 9.8, exposed tile compute 3.5, combine after the slowest arrival 1.8.
- XA's trace: from the loop end to done took ~5 µs. The cold call of the noinline combiner was ~0.8, the slot loads ~0.8, the warp-0 epilogue ~0.7.
- A guarded load loop `for r < 5: if (q < G) { ld; use }` compiled to 5 *dependent* branches, i.e. 5 serial L2 round trips (~2 µs).
- A warp keeps only ~16 LDGSTS (16-B cp.async) in flight. A 512-thread burst of 20 per warp blocks every warp's issue until ~4 µs.
- XG traced the chain after the slowest CTA's publish (10.48 µs): gate passed 10.43 → bulk slot load returned 11.26 (one round ≈ 0.8 µs) → all slots fresh 11.31 → out median 12.11 / max 12.45.
- A slot load issued before that slot was written in this call missed L2 (the judge's flush had evicted it) and went to DRAM behind the K/V stream: ~4 µs while the stream ran.

**(b) v1s and d1s: a latency chain with a measured floor** (`notes/XE_NOTES.md` rounds 2b, 4).
- The C8 trace (v1s on 8-CTA clusters), µs from the first CTA start: page-table issued 0.22 → landed 0.54 → K/V issued 0.80 → K landed 1.66 → PV 2.26 → CTA merge 2.43 → reader 2.75 → pushed 2.84 → owner received 3.20 → out 3.52.
- **MMA clock cap:** any MMA instruction capped the SM at 1.75-1.77 GHz on d1s, against 1.91 GHz with none (clock64 / globaltimer). Removing every tail phase *and* the MMAs (diag 63) gave a floor of d1s 2.94 and v1s 3.47-3.52 µs nsys, against 3.39 / 4.29 for the shipped kernels.
- An MMA-free d1s needs ≥ 0.27 µs of HFMA2 issue per SM, so it is at best break-even. XB measured the bare CUDA-core work of an MMA-free d1f stage at +1.3 µs nsys over the MMA kernel (`notes/XB_NOTES.md` round 6).

**(c) d16s: the per-request cluster runE1 was beatable, d32s was not** (`notes/XE_NOTES.md` round 3).
- `xe_dm_8_4_2`: one 8-CTA cluster per (request, KV head), with fragment-exact K/V loads into registers. nsys d16s 6.50-6.53 µs, against 7.23-7.30 for runE1 in the same jobs.
- d32s needs 64 co-resident clusters, and only 62 fit. The second wave cost 11.8-12.3 µs against 9.1-9.3.
- Keeping everything in flight (smem-staged passes) made the issue stall on per-SM memory backpressure: d16s 7.07, d32s 10.7.

**(d) verify full B ≥ 16: throughput- and power-bound** (`notes/ROUND2.md` shared findings, `notes/XC_NOTES.md`, `notes/XD_NOTES.md`).
- XD's v32f microbench: k34's lockstep loop skeleton alone ran 150.3 µs (4.28 TB/s), against 177.7 for the full kernel. That is ~7.5 µs of start / segment / end structure plus ~20 µs of compute coupling.
- nvidia-smi soak: the gather alone draws ~650 W and the tile compute adds ~320 W. The 700 W cap drops the SM clock to ~1400-1750 MHz.
- XC's pipelined segment switch cut a straddling CTA's bubble 6.7 → 4.3 µs, and nsys did not move. While one CTA idles, the others take its bandwidth. Only the start, the tail and the end chain count.
- Tail bound (XC_CUT, timing-only, wrong output): even dropping all the work the late half still has buys only −0.1 µs (v16f) / −1.1 µs (v32f). The poll that does it costs +2.7 µs.
- XC's single FINAL combiner beat XD's 5 combiners at B = 32, where each request has ~5 contributing CTAs: v32f −3.5 µs nsys. It lost at B = 16, with ~10 contributors (v16f bench +1.1).

**(e) d1f: lanes waited on tiles that all landed together** (`notes/XB_NOTES.md` round 7).
- Stage s of all 4 lanes landed at the same time, so every lane idled ~1 µs between finishing stage 0 (4.7 µs) and stage 1 landing.
- CTA 131, which owns the new token, was the critical CTA in 10 of 10 reps: its new-token loads queued behind the burst.
- The no-math skeleton ran 9.50 µs nsys, against 11.4 for the kernel.

**(f) Controls that ruled alternatives out.**
- *Page-ordered gather.* Bucketing each lane's rows by 2-MB page made a gather-only probe faster: d32f 157.31 → 148.29 µs (4.13 → 4.38 TB/s), d16f 78.02 → 73.41 (`evidence/probe_sort.py`, `notes/XF_NOTES.md` round 7). Every in-kernel way to produce that order cost more than it saved. Shared-memory atomics run 0.46 cycles per key per SM, against ≤ 0.1-0.15 needed. A pre-pass kernel costs 10.8 / 15.8 µs. XB's best in-kernel build on verify full sorted the v32f tile phase 7-10 µs faster and still lost +7.5 to +12 µs net (`notes/PLAN.md` 22:30, `notes/XB_NOTES.md` round 10).
- *A judge-only cluster launch delay* (suspected from judge-minus-local offsets) did not reproduce. Under the judge-like sequence, CTA start spread was ≤ 0.13 µs. Clusters wait only when SMs are busy at launch (`notes/XE_NOTES.md` round 6).
- *A PDL combine kernel* instead of in-kernel end chains: +0.75-0.8 µs per extra graph node (`notes/PLAN.md` 22:00, `notes/fresh_eyes_r2.md` §1).

**(g) Judge noise.** k42 and k43 have byte-identical d32f code, and the judge read 158.49 vs 165.27 µs (`evidence/judge_k42_*`, `judge_k43_*`). k42's check was a fast node. From then on, decisions were made on same-job local A/Bs and nsys with 2 copies; judge runs only confirm a level (`notes/PLAN.md` 17:20).

## 3. Diagnosis and projection

- **Batch-1 shapes are latency chains.** On v1f and d1f, the removable part is the end chain (dependent L2 round trips, a cold combiner, serial slot loads) plus the start burst's issue limit. On d1s and v1s, it is the work per SM after the K/V land.
- **d16s is a gather-latency chain** whose runE1 layout left 0.7-0.9 µs on the table.
- **Verify full B ≥ 16 is throughput / power-bound mid-stream.** Only its start, tail and end chain are worth cutting.
- **Projection.** "Jump mode" (`notes/ROUND2.md`, 17:00) set floor-based targets per shape, with the score each was worth: v1f 15.53 → 10 (+3.6 %), d1f 11.41 → 9 (+2.0 %), v16s 10.89 → 7 (+1.5 %), v16f 99.3 → 88 (+1.5 %), v1s 4.63 → 3.8 (+1.4 %), d1s 3.59 → 3.0 (+1.35 %), v32f 180.5 → 165 (+1.2 %), v32s 14.46 → 9.5 (+1.1 %), d16s 8.76 → 6.5 (+1.0 %), d16f 87.9 → 81 (+1.0 %), d32f 165.4 → 156 (+0.8 %), d32s 10.69 → 8.5 (+0.55 %).
- The coordinator's note on the earlier gap table: even if every target were hit, the projection was only ~2.97 on the judge (`notes/ROUND2.md`).
- **Rule adopted:** remove whole phases (barriers, round trips, waves), not instructions. Decide on nsys with 2 copies. From 22:45: stack every verified win, with no minimum size.

## 4. Change

`change.diff` is 6,971 lines. Most of it is new module source: XG's `_XG5_SRC`, XB's `_XB1_SRC`, the XE and XD sections rebuilt from later deliveries. The base `SRC` gets the run1f end-chain and run8w patches.
Each worker froze a deliverable `dev/<w>/deliver_rN/` with a `build.py <base> <out>`. The coordinator stacked frozen snapshots only (`notes/SNAPSHOTS.md`).

| piece | worker | shapes | local effect (nsys unless noted, same-job controls) | source |
|---|---|---|---|---|
| r2: v1s on 8-CTA clusters × 4 row groups (`xe_b1s_8c8`), 16 rows per cluster | XE | v1s | 4.64 → 4.32 (−0.33); bench −0.58 | XE_NOTES job18 |
| r4: `xe_dm_8_4_2`, one 8-CTA cluster per (request, KV head), for 2 ≤ B ≤ 16 | XE | d16s | runE1 7.30 → 6.43 (−0.86) | XE_NOTES job25 |
| `XE_BMAX` 24 (from XC's nearby-shape audit) | XE / XC | d-sliding B 17-24 | B 20: 7.46 → 6.24 bench (removes a −16 % cliff) | XC_NOTES audit |
| r6: v1s on 16 clusters of 6 (`xe_b1s_8c6`); `XEL` lane-level publish in run1f | XE | v1s; d16f | v1s 4.36 → 4.29; d16f −0.07 (noise level) | XE_NOTES round 8 |
| r3: `xa_b1f_8` end chain: FP16 O/l partials (`XA8_H16P`), loads issued before use (`CLD`), never-reset u64 arrival counter (`MONO`) | XA | v1f | 16.66 → 14.82 | XA_NOTES r3 |
| r4 + r5: 128-token steps (`T128`), Q off the start barrier (`QLATE`), one mbarrier per tile pair (`MBP`), deferred wave (`DEFER 4`) | XA | v1f | 14.82 → 13.94-14.00 | XA_NOTES r4, r5 |
| r1-r5: `xg_v8f`, tagged end chain (below), 128-B O rows (`MF`, `AL`), branch-free burst (`BFI`), alpha skip (`ASK`), tile-0-first issue (`PFT 150`) | XG | v1f | XA r5 14.05-14.14 (same job as XG r1) → XG r5 12.22-12.30 | XG_NOTES |
| r7: tile-ordered burst (`XB1_TORD`) + new-token rows at kernel start (`XB1_NTE`) | XB | d1f | XA's kernel 11.26-11.30 → 10.82-10.85 | XB_NOTES round 7 |
| r2: split + rows issued before the burst, no CTA barrier (`XD_START`) | XD | v16f, v32f | −0.4..−1.2 / −0.5..−1.4 | XD_NOTES §7 |
| r3: `OVH` 1024 / `NEWW` 512 work weights, spill-free start | XD | v16f | 93.55 → 92.54 (−1.1) | XD_NOTES §8 |
| r6: hybrid end chain, XC's single FINAL combiner when 6B ≥ 132 (B ≥ 22) | XC | v32f | 178.91 → 174.94 (−3.4) | XC_NOTES deliver_r6 |
| r8: `XC_XDF`, B = 16 combine with FP16 normalized partials, all loads in flight, tanh sigmoid | XC | v16f | 95.46 → 93.89 (−1.6) | XC_NOTES r8 |
| r10: `XC_RND` = 1 | XC | v32f | −0.12 (mean) | SNAPSHOTS |
| r15: `XB_VSTART=2`, warp 0 computes every segment's parameters once | XB | v32f | 175.37 → 175.08 (mean) / −0.57 (median) | XB_NOTES round 15 |
| r5: v16s combine as owner *pull* (`ld.shared::cluster`) + restricted cluster fence | XD | v16s | 10.06 → 9.58 (−0.45..−0.50) | XD_NOTES §12 |
| r3: run1f end chain, parallel arrivals (`XFE_PUB`), one-shot range check (`XFE_RCK`), P / K off the end path (`XFE_PK`) | XF | d16f, d32f | −0.75 / −0.74 | XF_NOTES r3 |
| r10: run8w owner poll back-off `WH_NAP` 128 → 64 ns | XF | v32s | 13.745 → 13.67 (−0.08) | XF_NOTES round 10 |

Not stacked: XD's int8 cluster partials (`XDV_Q8`, v16s −0.20). The coordinator rejected them because they are coarser than SGLang's FP32 partials (`notes/XD_NOTES.md` §11).

**The v1f end chain (XA r3 → XG r5): −4.7 µs on the judge, the largest single move of the milestone.**

- **XA r3: cheaper bytes and fewer trips.**
  - Each partial row became FP16 O/l (64 halves) plus a float2 (M, W): 136 B per row instead of FP32. That alone was −1.0..−1.3 µs.
  - The combiner's loads were clamped and weighted 0 past G instead of branching, so they issue together: −0.5.
  - A u64 counter that is never reset (call k uses arrivals kG .. kG+G−1) removed the DONE atomic and the reset at the kernel end: −0.6.
- **XG r1: tagged slots.**
  - Every 64-bit word of a partial carries a 1-bit call tag (the call's parity) and is stored with `st.relaxed.gpu.b64`. There is no fence and no atomic round trip on the publish path.
  - A combiner waits until the publish counter reaches G − 8. It then loads all slots in one round and re-polls only the words whose tag is stale.
  - After the slowest publish, one poll round trip plus the reduction remain.
- **XG r2: whole-line rows.** O rows of 128 B make every store a full sector and every combiner load one line: −0.54 µs, all of it in the combine after the last publish.

| step | v1f nsys µs | what it taught |
|---|---|---|
| k34 `attn_8_0_1_1` | 16.66-16.8 | |
| XA r3 (H16P + CLD + MONO) | 14.82 | guarded loads serialize into N round trips; FP16 partials halve the slot bytes |
| XA r4 / r5 (T128, QLATE, MBP / DEFER 4) | 14.34-14.46 / 13.94-14.00 | traced builds run +0.5-0.7 µs slow and mislead: decide on nsys only |
| XG r1 (tagged end chain, gate 8) | 13.25-13.28 | with no gate (early bulk round + polling) 16.74: polls flood L2 |
| XG warm start (combine run dry at entry) | 15.74 | **dead (+2.6)**: the dry pass delays the burst by ~2.5 µs, and the slow combine tail is not cold code |
| XG r2 (MF + AL + D32 + PIN + GSM) | 12.50-12.59 | full-sector rows are what made the combine cheap |
| XG r3 / r5 (BFI / ASK + PFT 150) | 12.35-12.38 / 12.22-12.30 | small issue-path wins; −0.11 for ASK + PFT together |

**d16s cluster kernel (XE r4).**
- One cluster of 8 CTAs per (request, KV head), grid (8, 2B).
- Each CTA has 4 compute warps and 1 prep warp; warp w, pass p owns 32 keys and loads its K/V fragments straight into registers.
- Online softmax runs across 2 passes. Owners (rank < 8) fold 8 sources plus the new token, received by `st.async` push with one mbarrier per owner CTA.
- No length guess: seq → page table → K/V.
- It serves 2 ≤ B ≤ 24 after `XE_BMAX` 24. B = 32 stays on runE1, because 64 clusters do not fit.

**v1s on more, smaller CTAs (XE r2 → r6).**
- C8 put 8 compute warps × 32 keys and 16 rows on each 8-CTA cluster: half the compute, merge, push and owner work per CTA, for ~+0.1 µs later K/V landing.
- C6 (16 clusters of 6: one (token, KV head) per cluster, 96 CTAs) is the shape that fits co-residency. One token per 8-CTA cluster would need 16 clusters, and only 15 are co-resident.

**Verify-full end chains (XC r6, r8 + XD r2, r3).**
- B ≥ 22 uses XC's single FINAL combiner. B < 22 keeps XD's 5-combiner protocol with XC's FP16 normalized partials (`XDF`).
- The source is compiled twice (`XC_END` = 1 / 0) and picked per call. The counters self-reset under both protocols, so alternating calls is safe.

## 5. Result

| check | result |
|---|---|
| local, same-job vs two k34 controls (era-0 `tools/bench.py`) | k37 x2.9358 vs x2.8068 / x2.8218 (+4.3 %, `evidence/verify_k37.txt`); k55 x3.0307 vs x2.7304 / x2.7288 (+11.0 %, `evidence/verify_k55.txt`) ⇒ k37 → k55 ≈ **+6.4 %** (inferred across the two jobs through the k34 controls) |
| local per shape (k55 − k34) − (k37 − k34), µs | v1f −4.57, v32f −4.74, v16f −3.80, v16s −0.72, d16s −0.68, v1s −0.63, d1f −0.39; d1s −0.13 and d32s −0.41 with unchanged code show the cross-job noise |
| correctness | `tools/check.py` on k55: 56 PASS / 0 failures, i.e. 16 tests + 12 scored shapes, twice each (`evidence/verify_k55.txt`); judge 16/16 tests. XC's nearby-shape audit of k49: 96 cases (B 1-64, both q_len, both layers, kv_scale 1 and 1/64, page 64) 0 failures, never slower than k34 (`notes/XC_NOTES.md`) |
| judge level | 2.6780 → **2.8989** (real-level means). Slow node vs slow node: 2.6780 → 2.8712 (+7.2 %). Common-overhead rescoring: 2.6999 → 2.8571 (+5.8 %, `evidence/rejudge_k37_k55.txt`) |

Judge per case, µs (`evidence/judge_*`). "Profile" is the judge's own per-kernel GPU time; it is a warm profile, not the timed number.

| shape | k37 (mean of 2) | k55 scored (fast) | k55 re-check (slow) | Δ slow-to-slow | profile k37 → k55 | code changed? |
|---|---|---|---|---|---|---|
| d1s | 3.90 | 3.68 | 3.39 | −0.51 | 3.0 → 3.1-3.3 | **no** (same `xe_r1_1` SASS) |
| d1f | 11.73 | 11.04 | 10.71 | −1.02 | 10.0-10.4 → 9.5-9.7 | yes |
| d16s | 8.87 | 7.23 | 7.07 | −1.80 | 7.2-7.3 → 5.0-5.2 | yes |
| d16f | 87.88 | 83.55 | 86.82 | −1.06 | 89.3-90.0 → 86.7-88.5 | small |
| d32s | 10.47 | 10.22 | 10.22 | −0.25 | 9.2-9.3 → 9.3-9.5 | **no** (runE1, untouched by the diff) |
| d32f | 165.99 | 158.53 | 164.67 | −1.32 | 171.7-175.0 → 166.4-171.6 | small |
| v1s | 5.25 | 4.60 | 4.80 | −0.45 | 4.0 → 3.7-3.8 | yes |
| v1f | 17.34 | 12.76 | 12.43 | **−4.91** | 14.7-14.9 → 11.8-11.9 | yes |
| v16s | 10.77 | 10.30 | 9.92 | −0.85 | 9.4-9.5 → 8.9-9.0 | yes |
| v16f | 99.56 | 93.14 | 96.12 | −3.44 | 100.7-102.5 → 93.4-97.7 | yes |
| v32s | 14.48 | 14.08 | 14.17 | −0.31 | 13.0-13.2 → 12.9-13.2 | small |
| v32f | 180.73 | 170.43 | 176.09 | −4.64 | 181.3-182.4 → 171.2-177.7 | yes |

**Reading the judge honestly.**
- **The node.** The scored k55 run was a fast node: d32f 158.53 and d16f 83.55 against 164.67 / 86.82 on the byte-identical re-check. That alone is 2.9266 vs 2.8712. The fair comparison is slow node to slow node: +7.2 %.
- **The controls moved too.** d1s and d32s run unchanged code and still moved −0.51 and −0.25 µs. d1s alone is worth +1.07 % of the +7.2 % at 2.09 %/µs; the k55 re-check's 3.39 is lower than every earlier d1s reading of this code (3.58-4.03 since k36, `evidence/judge_ledger_era0.jsonl`; `notes/PLAN.md` 19:45 calls it a ±6 % same-code spread).
- **The overhead estimator.** The judge subtracts a replay overhead it measures per shape in every run, with sd ~0.25 µs (`notes/PLAN.md` 10-29 ~17:00 W4 findings). On a ~3.5 µs shape that alone moves the ratio several percent. Re-scored with one common 4.15 µs overhead (`evidence/rejudge_k37_k55.txt`), the controls stop moving: d1s 3.68 → 3.63, d32s 10.33 → 10.26 µs. The changed shapes keep their gains: v1f 17.23 → 12.60, v32f 180.89 → 176.07, v16f 99.39 → 96.07, d16s 8.59 → 7.02, d1f 11.52 → 10.91, v1s 5.08 → 4.68 µs (k37 mean → k55 re-check). Score 2.6999 → 2.8571, +5.8 %.
- **Real effect.** Weighting each changed shape's slow-to-slow Δ by its %/µs gives ≈ +6.1 %: v1f +2.5, d16s +0.76, d1f +0.74, v1s +0.66, v16f +0.44, v32f +0.35, v16s +0.31, and +0.3 for d16f / d32f / v32s together. That agrees with the common-overhead +5.8 % and the local +6.4 %.
- **Stacking vs noise.** On slow nodes, the judged builds of this period rose fairly steadily: k38 2.6985, k39 2.7486 / 2.7418, k43 2.7826, k44 2.7618, k45 2.7950, k47 2.7719, k49 2.8426, k51 2.8536, k55 2.8712 (`evidence/judge_ledger_era0.jsonl`). Single steps below ~2 % were invisible to one judge run.

Judged builds along the way. Pieces are as listed in `notes/SNAPSHOTS.md`. "Local" is raw GEOMEAN vs the mean of two same-job k34 controls (`../autoresearch-kernels/attn/out/verify_kNN.txt`).

| build | added | local | judge |
|---|---|---|---|
| k38 | XE r2 (v1s C8) | +5.8 % | 2.6985 (check e200c479) |
| k39 | XA r3 (v1f end chain) | +6.7 % | 2.7486 (check e1704ff2) / 2.7418 (scored 7ac0d4e2, same sha) |
| k42 | XF r3 run1f end chain, XE r4 (d16s), XA r4, XD r3 | +8.4 % | 2.8476 (check 6c182dda, fast node) |
| k43 | k42 with XC r4 (XD r2 + `XC_END`) in place of XD r3 | local A/B: v32f 178.15 → 174.61 (`evidence/k42v43.txt`) | 2.7826 (check 756e90fe) |
| k44 | XA r5, XC r5 | +8.7 % | 2.7618 (scored a6df2aa0) |
| k45 | XC r6 hybrid, XB r7 (d1f) | +9.5 % | 2.7950 (check 438b47dc) |
| k47 | XC r8 (`XDF`), XG r1 | +11.1 % | 2.7719 (check 3ff8bc64) |
| k49 | XG r3 | +11.8 % | 2.8426 (scored f9dab3f3) |
| k51 | `XE_BMAX` 24, XD r5 (v16s pull) | +11.4 % | 2.8536 (scored 05daba72) |
| k55 | XE r6, XG r5, XC r10, XF r10, XB r15 | +11.0 % | 2.9266 (scored 945f42a6) / 2.8712 (check 489b863d) |

**Against the projection.** The jump-mode targets added up to ~+17 %. The milestone delivered ~+6 %. Each gap was measured, not guessed:
- v1f stopped at 12.4-12.8 vs 10. The 4-warpgroup tile convoy and the publish remain.
- d1s/v1s sit 0.4-0.8 µs above the diag-63 floor, which is the architecture's floor while MMAs cap the clock.
- d32s cannot use the cluster kernel (co-residency).
- Verify full is power-capped mid-stream.
- The page-ordered gather, the one lever measured to lower the full-layer floor, had no affordable in-kernel sort.

## 6. Takeaways, and what came next

- **End chains are where batch-1 time hides.** Count the dependent round trips after the last tile and remove them one by one. Use FP16 normalized partials, issue all combiner loads before use, use never-reset counters, tag the data instead of fencing it, and write whole lines. Each step was 0.5-1.3 µs on v1f.
- **SASS and ptxas decide latency chains.** Guarded loads become serial branches, `.nc` loads sink below barriers, and a cluster arrive placed after the loads made ptxas duplicate the issue loop, so the V loads waited for K (XE job13: d1s 3.46-3.55 vs 2.69-2.86 µs bench). Read the SASS before timing a latency-chain change.
- **For HBM- and power-bound kernels, only the phases where DRAM is not saturated count:** start, tail, end chain. Mid-stream bubbles are free.
- **One judge run reads ±2-3 %, and nodes are bimodal.** Decide on same-job A/Bs with 2 copies; use the judge to confirm a level.
- **What came next (`08_*`).**
  - A re-check of this exact build read 2.8712: the true level was ~2.90, not 2.93.
  - The next round looked for what the judge sees and the local bench does not. It found that the judge's harness frees and reallocates memory per rep pair, so the first rep of each pair runs with cold GPU TLBs.
  - Every piece from then on was decided on a judge-faithful bench.
  - The page-ordered gather was reopened in `08_*`, with all SMs sweeping the same pages together, and closed again there (see its dead ends).

## Reproduce

All you need is torch and an H200. `tools/` holds the problem spec, `reference.py`, `checks.py`, `baseline.py` and our harness.

```bash
cd muse_attention_decode/tools
python bench.py ../07_end_chain_and_cluster_kernel_stack/build.py --tests            # correctness + local judge-alike timing
python ab.py ../06_*/build.py ../07_end_chain_and_cluster_kernel_stack/build.py \
       --shapes v1f,d16s,v1s,d1f,v16f,v32f,d1s,d32s --seeds 7,8,9 --caps 2          # same-run A/B; d1s and d32s are controls
python nearby.py ../07_end_chain_and_cluster_kernel_stack/build.py                   # nearby-shape correctness
# in the era-0 tree: gather-order probe (needs tools/cu.py, tools/bench.py, dev/xf/probe.py) and the common-overhead rescoring
cd ../../../../autoresearch-kernels/attn && python3 dev/xf/probe_sort.py --shapes 16,32
python3 dev/w4/rejudge.py --ovh 4.15 subs/profiles/k37_check.ab77bf338d3b4052.json subs/profiles/k37_submit.a90f3cd55dcb45ce.json \
        subs/profiles/k55_submit.945f42a6df25433c.json subs/profiles/k55_recheck.489b863dea7a4bb5.json   # no GPU
```

Originals: `../autoresearch-kernels/attn/` (era 0). Build `subs/k55_submit.py` (= `dev/coord/k55.py` = `subs/k55_recheck.py`), base `subs/k37_submit.py`; judge results `subs/profiles/k37_*`, `k55_*`, per-case ledger `subs/judge_ledger.jsonl`; local verifies `out/verify_k37.txt` .. `out/verify_k55.txt`, `out/k42v43.txt`. Notes: `docs/PLAN.md` (coordinator log, 2026-10-28 13:00-23:55), `docs/ROUND2.md`, `docs/research_r2.md`, `docs/fresh_eyes_r2.md`, `dev/coord/SNAPSHOTS.md`, `dev/{xa,xb,xc,xd,xe,xf,xg}/NOTES.md` (copied here as `notes/<W>_NOTES.md`), frozen deliverables `dev/<w>/deliver_rN/`.
