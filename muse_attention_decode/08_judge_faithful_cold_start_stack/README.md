# 08: a judge-faithful bench (cold-TLB first rep), and the start-path, TLB-touch and L2-policy pieces it selected (11 of 12 shapes)

| | before (07) | after (08) |
|---|---|---|
| build | `07_*/build.py` = era-0 `k55` (md5 f0bff1b7, 974,512 B) | `build.py` = era-0 `k72s` (md5 614bc266, 875,212 B; sha256 5ed8458f...): k72 with the CUDA comments stripped |
| judge | 2.9266 (scored 945f42a6, fast node), 2.8712 (okc check 489b863d, slow node): real 2.8989 | **2.8832** (okc check c4f2a37a), **2.9033** (scored cea25010), **2.8620** (era 1's okc check 8cfd4f01 of the same CUDA, see below): real **2.8828**. All three on slow nodes |
| real effect | | **≈ +2.3 %**, the estimate with the judge's overhead noise removed. Re-scoring the slow-node JSONs with one common replay overhead (W4's `evidence/rejudge.py`) gives 2.8571 → 2.9218 for this build's three runs, with the d32f control flat (`evidence/rejudge_k55_k72s.txt`). On the official readings, slow node to slow node: 2.8712 → 2.8828 (+0.4 %) for this build, 2.8712 → 2.905 (+1.2 %) for the five slow-node runs of k70-k72. The real level is *lower* than 07's 2.8989 only because 07's mean contains one fast-node reading. Locally ≈ +3.8 % on the judge-faithful bench (inferred: +3.40 % k70 vs k55 directly, plus the k71 / k72 pieces) |
| shapes changed | | d1f, v1f, d1s, v1s, d16s, v16s, d32s, v32s, v16f, v32f; d16f only by a gated start fix. **d32f runs unchanged code** (control) |

Shape names: d = decode (q_len 1), v = verify (q_len 8); 1 / 16 / 32 = batch; s / f = sliding / full.
Kernels, by module:
- `XE_SRC`: `xe_r1_1` (d1s), `xe_b1s_8c6` (v1s), `xe_dm_8_4_2` (d16s).
- `_XB1_SRC`: `xb_d1f` (d1f).
- `_XG5_SRC`: `xg_v8f` (v1f).
- `_XD_SRC`: `attn_xd_v8f` (v16f / v32f).
- Base `SRC`:
  - run1f `attn_1_0_1_0` (d16f / d32f) and the gated copy `attn_1_0_1_0g` (d16f);
  - runE1 `attn_1_1_1_1e3` (d32s);
  - run8w `attn_8_1_1_1x6` (v16s) and `attn_8_1_1_1w4` (v32s).

Workers were named by role:
- **P1-P3** wrote "portfolios" (round 3).
- **S1-S5** ran inside an orchestrator loop (an LLM coordinator funding bets).
- **W1-W21** each owned one lever (rounds 4-5).

**Identity of the era-1 reading.** Era 1 judged `opt-muse-decode/runs/k72s.py` (md5 b74b9b66, 976,824 B), a different comment strip of the same k72.
`evidence/cmp_src.py` extracts the six CUDA source strings (`SRC`, `XE_SRC`, `_XA_SRC`, `_XD_SRC`, `_XB1_SRC`, `_XG5_SRC`) from both files and from the unstripped `dev/coord/k72.py`. It strips comments with era 0's `dev/coord/strip_cuda_comments.py` and compares them with whitespace normalized, both fully and with line structure kept (preprocessor lines).
All six are equal in all three files, and the Python outside the strings is identical (1,099 lines). Output: `evidence/cmp_src_k72s.txt`. That reading therefore counts toward the real level.

## 1. Starting point

k55 per case, µs (`evidence/judge_before_*`). %/µs is the score gain per µs saved at the slow-node times: 100 × weight / (6 × load time).

| shape | kernel | k55 scored (fast) | k55 re-check (slow) | %/µs |
|---|---|---|---|---|
| d1s | `xe_r1_1` | 3.68 | 3.39 | 2.34 |
| d1f | `xb_d1f` | 11.04 | 10.71 | 0.82 |
| d16s | `xe_dm_8_4_2` | 7.23 | 7.07 | 0.45 |
| d16f | `attn_1_0_1_0` | 83.55 | 86.82 | 0.156 |
| d32s | `attn_1_1_1_1e3` | 10.22 | 10.22 | 0.25 |
| d32f | `attn_1_0_1_0` | 158.53 | 164.67 | 0.086 |
| v1s | `xe_b1s_8c6` | 4.60 | 4.80 | 1.83 |
| v1f | `xg_v8f` | 12.76 | 12.43 | 0.64 |
| v16s | `attn_8_1_1_1x6` | 10.30 | 9.92 | 0.38 |
| v16f | `attn_xd_v8f` | 93.14 | 96.12 | 0.13 |
| v32s | `attn_8_1_1_1w4` | 14.08 | 14.17 | 0.22 |
| v32f | `attn_xd_v8f` | 170.43 | 176.09 | 0.077 |

The scored 2.9266 was a fast node. The byte-identical re-check read 2.8712, so the true level was ~2.90 (`notes/ROUND3.md`).

## 2. What we saw

**(a) Judge nodes are bimodal, and small shapes carry estimator noise.**
- `k60xs_a` and `k60xs_b` differ by one comment line (`# copy b`). They read **2.9297 vs 2.8115**, with d32f 158.79 vs 165.81 (`evidence/judge_k60xs_*`).
- The judge subtracts a per-shape replay overhead, ovh = 2·one − two, measured in each run: 3.68-4.64 µs, sd ~0.25 (`notes/W4_NOTES.md` §1, `evidence/rejudge.py`).
- On a ~3.5 µs shape against SGLang's ~20 µs, 0.3 µs of ovh error moves that shape's ratio ~7 %. Re-scored with a common ovh of 4.15, the two k55 runs read 2.9524 / 2.8571.

**(b) The judge pays a cost the local bench never saw: cold GPU TLBs on the first rep of every pair** (`notes/W4_NOTES.md` §2-3).
- Every judge run alternates slow even reps and fast odd reps, for both SGLang and us.
- Our even-minus-odd penalty (k55a / k55b / k60a / k60b): d1s 0.24-0.83, d32s 1.71-2.14, v16s 0.50-0.90, v16f 1.07-1.73 µs. If it vanished, the score would be +1.8-2.7 %.
- Probe `evidence/probe_jseq.py` tried the input modes gen / gens / copy / none / sleep / check: none reproduced the penalty.
- The `free` mode reproduced it: fresh inputs in new allocations, freed with `torch.cuda.empty_cache()` before each pair, so `cudaFree` invalidates the GPU TLBs. rep0 − rep1 was d1s +0.77, v1s +0.90, d32s +2.66, v16s +1.04 µs.
- nsys put most of it **inside our kernels**: span +0.51 (d1s), +0.43 (v1s), +2.13 (d32s), +1.44 (v16s). These are page walks on the first touch of K/V-cache, page-table and output pages.
- The window itself (gap, span, tail) matched a trivial graph's within 0.05-0.15 µs. There was no removable launch or end cost.

**(c) Where the rest of the time was** (`notes/ROOFLINE.md`, ncu `--set full` of all 12 k65 kernels, cold L2).
- The small shapes are latency-bound: tensor pipe 2-14 %, DRAM 2-41 % of peak. d1s / v1s / d16s lose 14-22 % of PC samples to instruction or constant fetch at kernel entry.
- The full B16/32 shapes average 75-87 % of DRAM peak.
- Decode full lost 11.6 µs at d16f: start 7.4, steady 0.1, tail 0.4, end 3.2. At d32f it lost 15.6: 8.2 / 3.2 / 1.4 / 3.1.
- In the d16f start, the burst issued 24 cp.async per warp, over the ~16 in flight a warp can hold. Iteration 0 waited 2.3 µs for a page-table row stuck behind the burst (W10).
- Verify full lost 18.9 µs at v16f (5.8 / 3.4 / 4.0 / 5.4) and 26.4 at v32f (5.2 / 11.0 / 4.5 / 5.1). The steady-state shortfall at B32 is TLB locality.
- The aggregate random-row gather ceiling was re-measured at 4.06-4.12 TB/s (`notes/ROOFLINE.md` §5).

**(d) Hardware facts that closed whole families of ideas** (`notes/ORCH_LEDGER.md`).
- `red.shared.add.f32` compiles to a CAS loop (`ATOMS.CAST.SPIN`) on sm_90a, so shared-memory atomic merges are dead.
- MMA caps the SM clock at ~1.70-1.78 GHz, against 1.90-1.96 GHz without it. MMA-free math is HFMA2 issue-bound on every shape measured.
- PDL or a second graph node costs +0.75 µs. Each `mbarrier.init` costs 25 ns.
- Convoy: 4 warpgroups per SM run one QK + softmax + PV iteration in 994 cycles, against 566 with 2.

**(e) Controls.**
- W6 found that 40-50 % of the first-rep penalty is *not* TLB: it is present without `empty_cache`. It is node-dependent: 0.09-1.01 µs on the same code. Same-job controls are therefore mandatory (`notes/PLAN.md` 10-30 00:20).
- ncu single-launch deltas overstate small-kernel gains. For k55 → k70, ncu said d1s −0.35 / v1s −0.83 µs (raw counters `evidence/ncu_ab_k55.csv`, `ncu_ab_k70.csv`). nsys per replay in a judge-loop replica said −0.08 / −0.25 (`notes/ROOFLINE.md` §6-7).
- An empty kernel with d1s's launch configuration already costs 1.3-1.5 judge-µs (W18).

## 3. Diagnosis and projection

- **The local bench was blind to a judge-only cost.** From k61 on, every piece was ranked on `evidence/bench_jt.py`, W4's judge-faithful bench:
  - reps come in pairs on one input set, with the order swapped;
  - with `--tlbcold 1`, the inputs are freed and `empty_cache` runs before each pair;
  - the judge's ovh = 2·one − two is subtracted;
  - an unjudged dummy candidate sits in the last slot, because late slots run 5-10 µs slow on the big shapes (`notes/PLAN.md` 10-29 19:05, `notes/ROUND5.md`).
  - The decision rule changed with it: TLB levers are decided on `--tlbcold 1` alone. Earlier warm guards had rejected real judge wins (verify-full touch, cold −0.8 / −1.3 µs; `notes/PLAN.md` 18:35).
- **Pools** (`notes/ROUND3.md`):
  - Batch-1: −1 µs on each of d1s / v1s / d1f / v1f would be +5.5 %.
  - B16/32 full sat 17-22 % above the ~4.3 TB/s gather ceiling; closing half of that would be +4.8 %.
  - The three portfolios priced their bets at EV ~+1.35 %, best case ~+4 % (`notes/PLAN.md` 10-29 08:45).
  - The orchestrator's gap table: ~17 % theoretical, ~1.5-2.5 % realistic from measured bounds (`notes/ORCH_LEDGER.md`).
- **Projections before the judge runs:**
  - k65: +2.93 % judge-faithful → slow ~2.955 / fast ~3.01 (`notes/PLAN.md` 10-29 19:30).
  - k70: +3.40 % directly vs k55 → slow 2.969 / fast 3.026 (10-30 03:45).

## 4. Change

`change.diff` is 14,267 lines, mostly because k72s has its CUDA comments stripped and k55 does not. The judge rejects sources over 1,000,000 B, and k60 had reached 1,010,147 B (`notes/ROUND4.md`).
The stripped file's CUDA is token-identical to `dev/coord/k72.py` (`evidence/cmp_src_k72s.txt`). After comment stripping, five of the six modules differ from k55 (`SRC`, `XE_SRC`, `_XD_SRC`, `_XB1_SRC`, `_XG5_SRC`); `_XA_SRC`, which no scored shape uses any more, is token-identical (`evidence/cmp_src.py subs/k55_submit.py subs/k72s_submit.py`). Builds were stacked from frozen `dev/<w>/deliver_rN/` builders (`notes/SNAPSHOTS.md`).

"jt" = `bench_jt.py --tlbcold 1`; nsys = per-replay kernel duration. Both use 2 copies and same-job controls.

| piece | worker | shapes | local effect | source |
|---|---|---|---|---|
| compile-time `__cluster_dims__` on the XE kernels (`CD`) + K-before-V in `xe_dm` (`KFM`/`KFP`) | P1 | d16s | nsys 6.528 → 6.072 (−0.46) | P1_NOTES p1_j2 |
| per-head lanes `XB1_HD` (4) with each lane's PV wgmma in flight (`HDPV`) | P2 | d1f | nsys 10.745 → 10.37 (−0.37) | P2_NOTES |
| per-warp arrival counters + per-warp float4 epilogue (`XB1_PW=2`), head-pair 256-thread barriers (`PWB`) | S1 | d1f | nsys 10.444 → 10.01 (−0.43); with PWB −0.50 vs controls | S1_NOTES |
| mask-free hot step (`MFREE`), `ALB`, step-0-ness by type (`KF`): hot SASS 1581 → 899 | S2 | v1f | nsys 12.19-12.27 → 11.90-11.98 (−0.29) | S2_NOTES, ORCH_LEDGER |
| table-free `sincosf` on the prep warp (no libdevice table load = no cold DRAM trip); dead vote removed | S5 | v1s | nsys 4.192 → 4.096 (−0.10) | S5_NOTES |
| `TP`: one predicated L2 prefetch per 2-MB page of K / V cache and output, right after the page-table loads | W4 | d1s | jt 4.48 / 4.25 → 4.02 / 3.99; warm −0.08 | W4_NOTES j4 |
| verify full: `pb[]` in smem (`PB`), mask-free hot pair loop + early election (`EE1`), last pair as two T64 steps (`LT`) | W2 | v16f, v32f | nsys −1.59 / −0.40 | W2_NOTES |
| v32s line-aligned slots (`LAY`) + TLB touch (`TPG`); v16s early cluster arrive (`EA`) | W3 | v32s, v16s | nsys −0.264 / −0.084; jt v32s cold −0.9 | W3_NOTES |
| v16s TLB touch (`TPC`) | W3 | v16s | jt 11.87 / 11.39 → 11.15 / 11.12 (−0.5); warm +0.26 | `evidence/ab_k64c.txt` |
| verify-full early TLB touch (`TP=1` in XD) | W2 | v16f, v32f | jt −0.40 / −0.98 | W2_NOTES |
| d32s `e3` early real touches (4-B cp.async per 2-MB page of K / V + own page-table row + output row) | W7 | d32s | jt −0.34..−0.70 on 2 nodes; warm +0.03 | W7_NOTES |
| run8w: FP16-accumulated QK (`A16`), 32-bit descriptor adds (`CH`): 218 → 192 instr per unit | W8 | v16s, v32s | nsys −0.15..−0.20 / −0.48..−0.50; jt −0.20 / −0.33 | W8_NOTES, PLAN |
| `L2::evict_first` cache policy on every verify-full K/V copy (`EVF`) | W11 | v16f, v32f | nsys −1.09 / −2.16; jt −1.61 / −3.18 | W11_NOTES |
| v16s K/V tiles 0-2 issued as soon as the page-table entries land, before the Q prep (`x3`); `E3CD` on d32s | W12 | v16s, d32s | nsys −0.46..−0.49; −0.064 | W12_NOTES |
| RMSNorm sum via one smem round + `rsqrt.approx` (`NS`) | W9 | d1s, v1s | nsys −0.026 / −0.031 | W9_NOTES |
| `CT2`: warp 1 pre-touches the constant-bank lines and loads the k / v scales while warp 0 waits for the lengths; compiled only into `attn_1_0_1_0g` for B ≤ 16 | W10 | d16f | nsys −0.17 | W10_NOTES |
| v32s role-split start: warps 0-7 issue K/V right after the length, warps 8-15 build RoPE / Q | W19 | v32s | nsys 12.91 → 12.61 (−0.30); ncu: barrier stalls before the first LDGSTS −82 % | W19_NOTES |
| `QG=3`: v1s Q·Kᵀ on native FP8 wgmma m64n16k32 (Q rounded to E4M3, as the reference's FP8 path does) | W21 | v1s | span −0.11; jt −0.19 vs k70 | W21_NOTES, `evidence/verify_k72.txt` |

**What carried the gain.**

- **The cold-TLB first rep.** Five pieces touch the pages early: d1s `TP`, v16s `TPC`, v32s `TPG`, the verify-full `TP` and d32s `e3`. While the kernel still waits on its page-table loads, these predicated L2 prefetches (or 4-B real loads, for `e3`) start the page walks for every 2-MB page the kernel will read. Each moved jt by 0.3-1.0 µs, while warm timing moved −0.3..+0.26. A warm guard had already rejected the verify-full touch once, and would have rejected `TPC` (`notes/PLAN.md` 10-29 18:35, `notes/W2_NOTES.md`).
- **`EVF`, the largest single piece.**
  - A verify-full call streams ~650 MB of K/V through L2 and evicted the end chain's own lines (partials, counters, gate rows: inferred).
  - Marking every K/V copy `L2::evict_first` keeps them resident: v32f −2.16 µs nsys, −3.18 jt.
  - run1f already had this (`WF_CPH=2`), so the lever did not transfer to decode full (`notes/W10_NOTES.md`).
- **Start-path order on the sliding B > 1 kernels.**
  - These are cluster or latency chains: length → page table → K/V → compute → combine. Every reorder that issues K/V earlier paid: `x3` (v16s), role split (v32s), K-before-V + compile-time cluster dims (d16s).
  - The W19 trace shows the mechanism: the page-table entry landed at 1.84 instead of 2.40 µs, and the whole chain shifted −0.65 µs. The stream itself did not change.
- **d1f end chain and v1f hot step.**
  - S1 deleted three CTA barriers, an smem broadcast and a counter reset. Each warp now releases and counts its own arrival and runs the epilogue for its own two rows.
  - S2 moved the masked tile into a separate instance reached once, which removed ~680 instructions from the hot instances. Cold code behind forward skip-branches inside a hot instance costs instruction fetch (~0.1 µs per branch family, `notes/ORCH_LEDGER.md`).

Internal iterations: the judged and verified builds (`notes/SNAPSHOTS.md`, `../autoresearch-kernels/attn/out/verify_kNN.txt`).
"std" = era-0 `tools/bench.py` raw GEOMEAN vs two same-job k55 copies. "jt" = the same on `bench_jt.py --tlbcold 1`. Judge readings: `evidence/judge_ledger_era0.jsonl`.

| build | adds | local | judge (slow nodes unless marked) | what it taught |
|---|---|---|---|---|
| k56 | P1 + P2 | std +0.5 % | — | below the judge's ~2 % resolution: no run spent |
| k59 | S1 + S2 | std +1.5 % | — | |
| k60 | S5 | std +1.9 % | k60s 2.8315 (scored 38ed22c1), 2.8650 (check df09e7f7) | d1s / v1s read *worse* than k55; suspected the XE changes |
| (k60xs) | k60 with k55's XE section | — | 2.9297 (c01fe571, **fast**) / 2.8115 (9cf6e7bb) | identical code 4.2 % apart: XE not guilty, nodes bimodal |
| k61 | W4 `TP` | std +2.0 %, **jt +2.1 %** (`evidence/verify_k61.txt`) | — | the first build ranked on the judge-faithful bench |
| k62 | W2 r2 | std +2.1 % | — | |
| k63 | W3 r1 | std +1.5 %, jt +2.7 % | 2.9248 (scored 3c860120) | best slow-node reading so far (+1.9 % over k55's) |
| k64 | W3 `TPC` | jt v16s −0.5 (`evidence/ab_k64c.txt`) | — | warm +0.26: a warm guard would have rejected it |
| k65 | W2 `TP` | **jt +2.93 %** (`evidence/verify_jt_k65.txt`) | 2.9092 (check 7638639c), 2.8953 (scored 2a251f2c) | |
| k66 | W5 `TKMF` (last-ticket lane fold, run1f) | nsys −0.18 / −0.21 | **hung**: timed replay > 60 s at rep 13 (check f86c925c) | rejected. A 60-replay hang test (`t_rep`) became mandatory |
| k67 | W7 + W8 | jt v16s −0.46, v32s −0.70 | — | |
| k69 | W11 `EVF` + W12 | jt +0.99 % vs k67 | 2.8926 (scored fdb183e7) | changed shapes improved on the judge; untouched d1f / d16f swung +0.5 |
| k70 | W9 `NS` + W10 `CT2` | **jt +3.40 % vs k55 directly** (`evidence/verify_k70.txt`) | 2.9061 (scored a1d0fb56) | chained estimates (+4.2-4.5 %) overstated the direct one: compare directly |
| k71 | W19 | nsys v32s −0.30 | 2.9266 (check a2f46a01), 2.9046 (scored 7c7b9bc9) | |
| k72 | W21 `QG=3` | jt v1s −0.19, d1s control ±0.0 (`evidence/verify_k72.txt`) | 2.8832 (check c4f2a37a), 2.9033 (scored cea25010); era 1: 2.8620 (check 8cfd4f01) | per-run spread exceeds the 0.07-0.3 % pieces |

## 5. Result

| check | result |
|---|---|
| local, judge-faithful, k70 vs k55 directly (2 + 2 copies, dummy last; `evidence/verify_k70.txt`) | x2.8538 / x2.8445 vs x2.7482 / x2.7627: **+3.40 %** (matched copies +3.84 / +2.96 %) |
| local, k72 vs k70 (`evidence/verify_k72.txt`) | v1s 4.98 / 5.11 → 4.82 / 4.89 (−0.19); d1s control 4.22 / 4.26 → 4.28 / 4.19 (0.00); v32s 15.21 → 15.17 |
| correctness | `tools/check.py` on the stripped k72s: 56 PASS / 0 failures; `t_rep` 60 replays × {v1s, v32s, d1s}: 0 failures, no hang (`evidence/verify_k72.txt`); judge 16/16 tests in all three runs. v1s error rises 4.5e-3 → 7.0e-2 with `QG=3`, against a bound of 0.150 (worst err/bound 0.585, `notes/W21_NOTES.md`) |
| judge level | 2.8989 → **2.8828** (real-level means; the "before" mean includes one fast node). Slow node to slow node: 2.8712 → 2.8828 (+0.4 %); the k70-k72 slow family 2.9061 / 2.9266 / 2.9046 / 2.8832 / 2.9033, mean 2.905 (+1.2 %). Common-overhead rescoring: 2.8571 → 2.8952 / 2.9492 / 2.9210, mean 2.9218 (**+2.3 %**, `evidence/rejudge_k55_k72s.txt`) |

Judge per case, µs (`evidence/judge_*`). Local Δ = bench_jt k70 − k55 (`evidence/verify_k70.txt`), plus k72's v1s step. Profile = the judge's own warm per-kernel time, k55 re-check → mean of the three k72s runs.

| shape | k55 fast | k55 slow | k72s check | k72s scored | era-1 check | k72s mean | Δ vs k55 slow | local Δ | profile |
|---|---|---|---|---|---|---|---|---|---|
| d1s | 3.68 | 3.39 | 3.69 | 3.73 | 4.11 | 3.84 | +0.45 | −0.21 | 3.1 → 3.1 |
| d1f | 11.04 | 10.71 | 10.99 | 10.84 | 10.67 | 10.83 | +0.12 | −0.84 | 9.5 → 8.8 |
| d16s | 7.23 | 7.07 | 6.55 | 6.48 | 6.45 | 6.49 | −0.58 | −0.72 | 5.2 → 4.8 |
| d16f | 83.55 | 86.82 | 87.04 | 86.68 | 86.92 | 86.88 | +0.06 | slot artifact (−0.51 / +3.12) | 88.5 → 89.3 |
| d32s | 10.22 | 10.22 | 10.39 | 10.51 | 10.63 | 10.51 | +0.29 | −0.61 (−1.15 / −0.08) | 9.3 → 9.3 |
| d32f | 158.53 | 164.67 | 165.03 | 165.61 | 165.39 | 165.34 | +0.67 | −0.13 | 171.6 → 172.2 |
| v1s | 4.60 | 4.80 | 4.78 | 4.36 | 4.58 | 4.57 | −0.23 | −0.31, −0.19 | 3.7 → 3.4 |
| v1f | 12.76 | 12.43 | 12.49 | 12.46 | 12.42 | 12.46 | +0.03 | −0.32 | 11.9 → 11.4 |
| v16s | 10.30 | 9.92 | 9.67 | 9.69 | 9.75 | 9.70 | −0.22 | −1.32 | 8.9 → 8.4 |
| v16f | 93.14 | 96.12 | 93.04 | 93.68 | 93.40 | 93.37 | **−2.75** | −1.84 | 97.7 → 96.2 |
| v32s | 14.08 | 14.17 | 13.12 | 13.43 | 13.15 | 13.23 | −0.94 | −1.72 | 13.2 → 12.2 |
| v32f | 170.43 | 176.09 | 174.42 | 173.59 | 173.00 | 173.67 | **−2.42** | −3.35 | 177.7 → 178.4 |

**Reading the judge honestly.**
- **The node.** Every k60-k72 run except k60xs_a was a slow node: d32f 164.99-165.81, against 158.5-158.8 on the two fast readings (k55 scored, k60xs_a). So "before" must be the slow re-check 2.8712, not the scored 2.9266.
- **The control.** d32f runs untouched code and moved +0.67 µs. That is the node-to-node drift on the big shapes (~0.4 %).
- **What the judge saw.** v16f −2.75, v32f −2.42, v32s −0.94, d16s −0.58, v1s −0.23 and v16s −0.22 µs. Weighted by %/µs that is ≈ +1.5 % of score, about two-thirds of what the local bench measured on the same shapes (+2.3 %).
- **What the official numbers hid.**
  - On the official readings, d1s read +0.45 µs (−1.06 % of score on its own), d1f +0.12 and v1f +0.03. Yet d1f's kernel got faster in the judge's own profile (9.5 → 8.8 µs).
  - The judge subtracts a replay overhead measured per shape in each run (sd ~0.25 µs, `notes/W4_NOTES.md` §1). On 3.5-12 µs shapes that noise is as large as the gains.
  - Re-scored with one common 4.15 µs overhead (`evidence/rejudge_k55_k72s.txt`), k55 re-check → mean of the three k72s runs: d1s 3.63 → 3.58, d1f 10.91 → 10.44, d16s 7.02 → 6.54, v1s 4.68 → 4.31, v1f 12.60 → 12.22, v16s 10.00 → 9.39, v32s 14.18 → 13.09, v16f 96.07 → 93.25, v32f 176.07 → 173.51. The controls stay put: d32f 165.12 → 165.15, d16f 86.82 → 86.82.
  - Only d1s shows no gain, which matches its −0.08 µs in-kernel gain per replay (`notes/ROOFLINE.md` §7). The k55 re-check's official 3.39 was a good draw.
- **Net.** Official readings: ≈ +0.4 % for this build, ≈ +1.2 % for the k70-k72 family; ROUND 5 concluded that the judge realized about a third of k70's bench_jt gain (`notes/ROUND5.md`). With the overhead noise removed: +2.3 % for this build (2.8571 → 2.9218), +2.6 % for the six k70-k72 runs, era 1's included (mean 2.9303). That is about 60 % of the local +3.8 %. The rest is spread over d1s, d1f, v16s and v32s, whose judge gains are about half their local ones (inferred from the two tables).
- **Against the projection.** k70 was projected at +3.4 % (slow ~2.969, `notes/PLAN.md` 10-30 03:45). Officially the k70-k72 family landed at +1.2 %; with the overhead noise removed, at +2.6 %. The remaining miss is batch-1 sliding:
  - ncu put k55 → k70 at ~+3.8 % at kernel level, and the official judge gap sat entirely in d1s / v1s (`notes/ROOFLINE.md` §6).
  - In the judge-like replay those two gains were only d1s −0.08 / v1s −0.25 µs, smaller than ncu's single-launch numbers (§7).

## 6. Takeaways, and what came next

- **Model the judge's harness, not just its timing loop.** It frees and reallocates per rep pair, so the first rep runs on cold TLBs. That cost was 0.1-2.7 µs per shape, mostly inside our kernels, and the standard bench never saw it. Early page touches and cache-policy fixes were only selectable on a bench that reproduces it.
- **Keep the end chain's lines resident.** A 650-MB stream evicts everything, so mark the stream `evict_first`.
- **Read the judge in families, and rescore before reading small shapes.**
  - Identical code read 2.81-2.93, and nodes are bimodal on the big shapes.
  - The per-run overhead estimate alone moves 3-12 µs shapes by tenths of a µs. A common-overhead rescoring (`evidence/rejudge.py`) turns the official +0.4 % for this build into +2.3 %, with a flat control.
  - Pieces of 0.1-0.3 % cannot be seen in one or two runs.
  - Compare slow node to slow node, use an untouched shape as the control, and judge-check stacks rather than pieces.
- **Gate every build with a replay-hang test.** k66's last-ticket fold passed `check.py` locally and hung in the judge's replays.
- **Dead ends that set the next direction** (`notes/PLAN.md` 10-30):
  - The aligned 2-MB-page gather order, with all SMs sweeping the same pages together (W13, W15, W16, W20). The bound was −4.5 / −8.6 µs (d16f / d32f) and −4.7 / −9.1 (v16f / v32f), worth ~+2.8 % if free. All nine correct builds lost 1.7-14 µs: the sort costs 5-11.5 µs per lane, and 512 threads × 128 registers leave no spare warps.
  - W17 / W18 (batch-1 full / sliding redesigns): nothing beat the shipped kernels. W18 measured the empty-kernel floor at 1.3-1.5 judge-µs.
- **What came next (`09_*`).** Era 1 started from this build (`opt-muse-decode/runs/k72_base.py` = `dev/coord/k72.py`, md5 b2c5a13f), re-checked it at 2.8620 (the third reading above), and stacked its own workers' pieces on it up to k80.

## Reproduce

All you need is torch and an H200. `tools/` holds the problem spec, `reference.py`, `checks.py`, `baseline.py` and our harness.
The judge-faithful bench `evidence/bench_jt.py` imports era 0's `reference.py` and `baseline.py` from its grandparent directory. Run it in the era-0 tree as below.

```bash
cd muse_attention_decode/tools
python bench.py ../08_judge_faithful_cold_start_stack/build.py --tests             # correctness + local (warm-TLB) timing
python ab.py ../07_*/build.py ../08_judge_faithful_cold_start_stack/build.py \
       --shapes v16f,v32f,v32s,d16s,v16s,v1s,d32f --seeds 7,8,9 --caps 2          # same-run A/B; d32f is the control
python nearby.py ../08_judge_faithful_cold_start_stack/build.py
# judge-faithful (cold-TLB pairs, judge ovh, dummy last) -- in the era-0 tree:
cd ../../../../autoresearch-kernels/attn
python3 dev/w4/bench_jt.py --cand k55=dev/coord/k55.py --cand k72s=subs/k72s_submit.py --cand k55b=subs/k55_recheck.py \
        --cand k72sb=subs/k72s_check.py --cand dummy=dev/coord/k72s.py --tlbcold 1 --reps 80   # 2 + 2 copies, dummy last
python3 dev/w4/rejudge.py --ovh 4.15 subs/profiles/k55_recheck.489b863dea7a4bb5.json \
        subs/profiles/k72s_check.c4f2a37a72e74c0e.json subs/profiles/k72s_submit.cea2501043034a2b.json \
        ../../Compilers/opt-muse-decode/runs/profiles/k72s.8cfd4f01ff7d4b03.json   # common-overhead rescoring (no GPU)
python3 ../../Compilers/okc-kernels-private/muse_attention_decode/08_judge_faithful_cold_start_stack/evidence/cmp_src.py \
        subs/k72s_submit.py ../../Compilers/opt-muse-decode/runs/k72s.py dev/coord/k72.py      # CUDA identity check
```

Originals: `../autoresearch-kernels/attn/` (era 0). Build `subs/k72s_submit.py` (= `subs/k72s_check.py` = `dev/coord/k72s.py`; unstripped `dev/coord/k72.py`, md5 b2c5a13f), base `subs/k55_submit.py`.
- Judge results: `subs/profiles/{k55,k60*,k63s,k65s,k66s,k69s,k70s,k71s,k72s}_*`; ledger `subs/judge_ledger.jsonl`; era 1's `opt-muse-decode/runs/profiles/k72s.8cfd4f01ff7d4b03.txt`.
- Local: `out/verify_k56.txt` .. `out/verify_k72.txt`, `out/verify_jt_k65.txt`, `out/ab_k64*.txt`; ncu A/B `out/20261030T084734_coord_ncuab/`; the ncu full sets `out/20261029T233923_roof_1_1_sl/ncu/` and `out/20261029T234300_roof_32_1_s/ncu/` (`.ncu-rep`, not copied).
- Notes: `docs/PLAN.md` (10-29 08:00 to the end), `docs/ROUND3.md`, `ROUND4.md`, `ROUND5.md`, `dev/orch/LEDGER.md`, `dev/roof/ROOFLINE.md`, `dev/coord/SNAPSHOTS.md`, and the `NOTES.md` / `PORTFOLIO.md` of `dev/{p1,p2,s1,s2,s5,w2,w3,w4,w7,w8,w9,w10,w11,w12,w19,w21}` (copied to `notes/`). The dead-end lanes are in `dev/{p3,s3,s4,w1,w5,w6,w13,w14,w15,w16,w17,w18,w20}/NOTES.md`.
