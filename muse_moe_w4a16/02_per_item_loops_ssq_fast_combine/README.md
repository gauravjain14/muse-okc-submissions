# 02: per-item stage loops with a top-tested loop, the sum-of-squares fast combine, and lighter end-of-call sync (v40 to v66)

Paths: `M/` = `muse-opt/muse_moe_w4a16/`, `R/` = `okc-retro/`, `mem:` = `<agent-state>/memory/` (outside the tree), all relative to ``. "transcript, hh:mm UTC" = the main session log `<agent-state>/<session>.jsonl` (outside the tree; 10-23 UTC; PT = UTC-7). Copies in this folder are cited as `evidence/...` and `notes/...`.

| | before (01) | after (02) |
|---|---|---|
| build | `../01_persistent_fused_kernel_tma_ring/build.py` = v40 (md5 8fb0d862) | `build.py` = `M/dev/moe_v66.py` (md5 9e7512eb, 89,654 B; mtime 21:09 UTC, copied to `submission.py` at 21:18 UTC: transcript, cp command at 21:18 UTC) |
| judge | 2.1339 / 2.1331 (48fc3fa7, 95e61683), real 2.1335 | **2.2041** (scored, job 348e75ba). Intermediates: v50 2.1361 (055608b9), v59 2.1503 (80f39693) |
| real effect | | same-job Modal H200 A/Bs chained: about **+1.6 %** geomean (derived). Most of it is v66 at bs32/128/256 (-3.9 / -2.5 / -2.8 % time) |
| shapes changed | | all. The bs1 and bs8/16 configs change to KS2 = 1 (`moe_2_1_1_4_1`, `moe_4_1_1_4_1`) |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: one-line comments credit the designs or reference code that parts of it follow. The code is unchanged. The judged file has md5 `9e7512eb`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

Terms (as in 01): an **item** is a token tile times a block of weight rows over all of K, gate/up or down; a **stage** is one ring slot (32 KB codes + 4 KB scales + activations); **producer** = warp 8, **consumers** = warps 0-7, **signaler** = warp 9. Kernels are `moe_<RW>_<NT>_<KS2>_<NS>_<MW>`, and KW = 8/RW is the number of k-warps. New here:
- **SSQ**: sum of squares. A down item that covers all of K writes, per (token, 64-row slot), the sum of squares of y·(1+w), so the combine needs no row reduction.
- **Per-item loop**: a consumer loop over all stages of one item with the token-group count fixed at compile time, so the stage info is read and dispatched once per item instead of once per stage.
- **Modal A/B**: `evidence/sweep_list.sh` runs `bench.py` once per variant line on a Modal H200 (approved at 15:35 UTC), each line paired against SGLang in its own process. Variants are interleaved over 2-4 rounds. That harness reads ~3 % above the judge (transcript, 19:01 UTC: v40 2.198 there vs 2.133 judged). It gives each variant its own weight copy; milestone 04 shows this can skew a variant by a few percent.

## 1. Starting point

v40 per case on the judge, mean of its two runs (`evidence/judge_before_v40_48fc3fa7.txt` and 01's second run):

| shape | yours µs | speedup | config |
|---|---|---|---|
| bs1 | 34.4 | 2.32x | (2,1,2) `moe_2_1_2_4_1` |
| bs8 | 69.4 | 2.11x | (4,1,2) `moe_4_1_2_4_1` |
| bs16 | 94.8 | 1.98x | (4,1,2) |
| bs32 | 131.3 | **1.60x** | (8,4,2) `moe_8_4_2_4_1`, per-stage loop |
| bs128 | 194.7 | 2.39x | (8,4,1) |
| bs256 | 242.1 | 2.55x | (8,8,1) |

bs32 had the lowest speedup and was the one config left on the per-stage loop. At bs1-16 the call ended 5-6 µs after the median CTA's last item (01, section 2(f)).

## 2. What we saw

**(a) The end of the call sat behind three grid-wide fences.** v40's last-arriver reset did `__threadfence(); atomicAdd(ctr+2); __threadfence(); <reset counters>; __threadfence(); ctr[2] = 0`, and every call waited on it. It was estimated at 1-3 µs per call (transcript summary at 15:28 UTC).

**(b) The waiting warps spin on the consumers' schedulers.** ncu region profile of v50 on Modal (`evidence/ncu_regions_v50.log`, built by `evidence/ncu_prof.py` / `ncu_ctx.py`): the signaler warp was 9.8-11.5 % of all executed instructions at bs1-32 while doing almost no work (stall reason `long_sb`). Warps 0 and 4 share SMSP0 with the producer and ran 3-7 % slower at bs128/256 (`mem:okc-moe-findings.md`).

**(c) bs32 at v50, one kernel on the judge-like flush** (ncu, `evidence/ncu_v50_bs32.log`, `moe_8_4_2_4_1`):
- 117 µs at an SM clock of 1.68 GHz;
- DRAM 71.0 % of peak (3.42 TB/s), L2 79.2 %;
- issue slots busy 48.9 %, ALU the top pipe at 45.8 %.

The Modal timeline at bs32 (`evidence/timing_v50_h200.log`) shows a per-CTA compute of 72.4 µs and 888 ns per stage, against 814-816 ns at bs8/16. The producer waited 16.8 µs per CTA for free slots, so the consumers were the limiter there too.

**(d) Why bs32 stayed on the per-stage loop, and why the per-item loop had lost there.**
- In a stage microbenchmark, the per-stage info read and `switch` dispatch cost ~100 cycles per stage on top of ~140 for the mbarrier ring, about 7 % of a 1-group stage (transcript summary at 15:28 UTC; transcript, 20:52 UTC).
- Yet forcing the per-item loop at bs32 measured 125.1 / 124.5 µs vs 122.2 / 122.8 µs, about 2 % slower (`evidence/transcript_v65_v66_loop_sass.txt`).
- The SASS showed why (`evidence/kern_sass.py` opcode counts, same excerpt). With the bottom-tested per-item loop, `for (st = 0;;) { compute; if (++st == nst) break; arrive; wait; }`, the compiler inverted the loop and emitted the fully unrolled stage body twice:

| kernel | loop form | SASS instructions | HMMA | HGMMA | HMUL2 |
|---|---|---|---|---|---|
| `moe_8_4_2_4_1` (bs32) | per-stage loop | 6,448 | 320 | 0 | 512 |
| `moe_8_4_2_4_1` (bs32) | per-item, bottom-tested | 8,872 | 640 | 0 | 1,024 |
| `moe_8_4_2_4_1` (bs32) | per-item, top-tested | 6,600 | 320 | 0 | 512 |
| `moe_8_8_1_3_8` (bs256) | per-item, bottom-tested (v59) | 17,776 | 640 | 384 | 2,560 |
| `moe_8_8_1_3_8` (bs256) | per-item, top-tested | 12,216 | 320 | 192 | 1,280 |

So v40's bs128/256 kernels were already carrying a duplicated body. Every multi-group kernel paid for it in i-cache and scheduling.

**(e) The small-batch combine was a serial chain.** At bs1 one CTA combined the token after a grid-wide "all down items done" wait. A single token's combine took 5-7 µs on the H100: pair list ~0.7, first pair ~1.7, the rest ~3 (transcript summary at 15:28 UTC). Down items at bs ≤ 16 were k-split 2 ways (KS2 = 2), so the combine first had to add partials and then reduce each row's sum of squares for the RMSNorm.

**(f) Ablations at v51 on Modal** (`evidence/sweeps/s5_result.txt`, µs):

| variant | bs1 | bs8 | bs16 | bs32 | bs128 | bs256 |
|---|---|---|---|---|---|---|
| full | 33.4 | 70.7 | 89.4 | 127.3 | 186.3 | 229.0 |
| no compute | 27.0 | 63.1 | 81.6 | 111.5 | 155.6 | 171.7 |
| no compute, no combine | 23.9 | 61.0 | 78.6 | 106.8 | 149.7 | 164.4 |
| compute only (no loads) | 27.3 | 64.1 | 80.0 | 119.0 | 167.4 | 218.8 |

The combine alone was worth 3.1 µs at bs1 and 2-6 µs elsewhere. Without compute the data path ran 10-25 % faster, so compute coupling was still the larger loss at bs32-256.

## 3. Diagnosis and projection

- **End of call.** One acquire-release atomic gives the last arriver everything the three fences gave: each CTA's counter updates happen before its arrival, and the last arriver acquires them all. Kernel completion publishes the reset stores to the next launch. Expected 0.5-1 µs per call.
- **Small-batch combine.** With KS2 = 1 at bs ≤ 16 every down item holds final rows. If its epilogue also leaves the per-64-row sums of squares, the combine becomes one elementwise pass spread over all CTAs, with no per-token chain. Expected: bs1 combine ~2.7 → ~1.8 µs (measured after v59: transcript, 20:09 UTC), and 1-3 % at bs8/16.
- **Loops.** A top-tested per-item loop keeps one copy of the stage body. Expected: up to the ~7 % dispatch overhead at bs32, plus whatever the duplicated body cost bs128/256.

## 4. Change

| piece | what | code | shapes | same-job Modal A/B | source |
|---|---|---|---|---|---|
| v50 | fence-free end-of-call reset: one `atom.acq_rel.gpu.global.add.s32`, plain reset stores | end of `moe_body` | all | H100: 2.067 / 2.068 vs v40 2.049 / 2.058 (bs1/8/16 0.5-1 µs faster). Modal: 2.186 / 2.190 vs v40 2.181 (+0.3 %) | `evidence/transcript_h100_v50_vs_v40.txt`, `evidence/sweeps/s1_result.txt` |
| v51 | producer and signaler waits use `mbarrier.try_wait` with a suspend-time hint (`mbar_sleep`, `MOE_SUSPEND` 1 ms) | `mbar_try_s` | all | 2.192 / 2.179 vs v50 2.193 / 2.183 (0) | `evidence/sweeps/s3_result.txt` |
| v58 | non-blocking queue claim in inline PTX: lane 0's predicated `atom.global.add.u32`, broadcast by shuffle | consumer claim | all | 2.190 / 2.200 / 2.202 vs v51 2.196 / 2.202 / 2.190 (0) | `evidence/sweeps/s11_result.txt` |
| v59 | SSQ fast combine for 1-group configs, and KS2 = 1 at bs ≤ 16 (buckets `(1,(2,1,1))`, `(16,(4,1,1))`) | `SSQ = NT == 1`, `token_fast()` | bs1, bs8/16 | vs v58 (bs1/8/16 only): bs16 -4.0 %, bs8 -2.9 %, bs1 -1.0 % time; geomean 2.246 / 2.255 vs 2.198 / 2.208. All shapes vs v51: 2.200 / 2.206 / 2.213 vs 2.213 / 2.203 / 2.200 (bs8 -1.2 %, bs16 -0.9 %, bs1 +0.9 %) | `evidence/sweeps/s18_result.txt`, `s12_result.txt` |
| v66 | per-item loops for every config (`MOE_RUN_ITEM`); top-tested `MOE_RUN_ITEM_TOP` for NT > 1, bottom-tested `MOE_RUN_ITEM_BOT` kept for NT == 1 | consumer item loop | bs32/128/256 (bs1-16 code unchanged) | 2.246 / 2.235 / 2.240 vs v59 2.209 / 2.213 / 2.212 (**+1.3 %**); bs32 120.5 vs 125.4 µs (-3.9 %), bs128 183.2 vs 187.9 (-2.5 %), bs256 223.7 vs 230.3 (-2.8 %) | `evidence/sweeps/s21_result.txt` |

The deeper story is v66. It came in two steps (`evidence/transcript_v65_v66_loop_sass.txt`, 20:52-21:18 UTC):
1. **v65 kept one stage body.** The loop waits at its top (`for (st = 0; st < nst; ++st) { if (st > 0) { arrive empty[s]; advance; wait full[s]; } compute; }`), so ptxas emits the unrolled body once. Applied to all configs, it gave bs128 -2.6 % and bs256 -2.5 % time but bs8 +1.2 % and bs16 +2.5 % (the 1-group loop lost the compiler's own 2x overlap). Forcing the per-item path at bs32 on top gave bs32 -3.6 % (`evidence/sweeps/s20_result.txt`).
2. **v66 kept the bottom-tested form for 1-group tiles** and used the top-tested one elsewhere, with per-item loops everywhere.

The SSQ combine (v59), in more detail:
- A full-k down item's epilogue reduces, per token and 64-row slot, Σ (y·(1+w))² with half-warp shuffles, and stores it beside the BF16 rows.
- The combine is then an elementwise pass in (token, 1024-column) units over every CTA. Each warp sums a pair's 64 slot sums itself, so only the per-tile readiness polls remain.
- On larger configs (KS2 = 2 tails, multi-group tiles) SSQ lost: bs128/256 +4 to +7 % time (`evidence/sweeps/s8_result.txt`). Hence `SSQ = NT == 1`.

Tried in this lane and dropped (same-job Modal A/B unless noted):

| idea | build | result | source |
|---|---|---|---|
| L2 prefetch of later stages (bulk, 2-12 stages ahead) | v51 `MOE_PD` | -14 to -21 % geomean; worse the further ahead | `evidence/sweeps/s7_result.txt` |
| per-line L2 prefetch | v60 | -19 to -20 % | `s13_result.txt` |
| claim 2-3 items ahead | v61 | -0.6 % / -3.6 % | `s14_result.txt` |
| first stage issued during routing | v56 | bs1/8/16 -2.6 to -3.4 % | `s10_result.txt` |
| partner reduction of split down items in the signaler | v64 | bs1/8/16 -7 to -9 % | `s18_result.txt` |
| row/k tail splits at bs8/16 | v68 | -6 to -8 % | `s23_result.txt` |
| two 4-warp CTAs per SM | v45 | 2.079 vs 2.210 | `s2_result.txt` |
| BF16 gate/up staging; epilogue warps finishing items | v41, v46 | 2.155, 2.187 vs 2.210 | `s2_result.txt`; v46 judged 2.1270 (01 evidence) |
| routing loads overlapped | v52 | bs256 +3 % time: ptxas C7511 (wgmma serialized) after an unrelated edit | `s4_result.txt`; `mem:okc-moe-findings.md` |
| wgmma for 2-4-group tiles | v53 | -0.5 to -2 % | `s6_result.txt` |
| TLB warm-up of weight pages; producer copy reorder | v63, v62 | flat | `s17_result.txt`, `s15_result.txt` |
| other bs8/16 and bs32 configs | | the current ones best | `s1_result.txt`, `s26_result.txt` |

## 5. Result

| check | result |
|---|---|
| local, Modal H200, chained | v50 +0.3 %, v51 0, v58 0, v59 +0.05 % (all shapes vs v51), v66 +1.3 % → **about +1.6 %** over v40 (derived product of the same-job ratios above) |
| correctness | 5/5 judge tests in all three scored runs; local tests rel-L2 2.47e-3 to 2.55e-3 in every sweep job (`evidence/sweeps/*_result.txt`) |
| SASS | the v66 build check (`evidence/kern_sass.py`, grep for ptxas C7511 and spills) printed only the per-kernel opcode lines: no C7511, no spills (`evidence/transcript_v65_v66_loop_sass.txt`, 21:09 UTC) |
| judge | v40 2.1335 → v50 2.1361 → v59 2.1503 → **v66 2.2041** |

Judge per case, speedup (yours µs) (`evidence/judge_*`):

| shape | v40 (mean of 2) | v50 (055608b9) | v59 (80f39693) | v66 (348e75ba) |
|---|---|---|---|---|
| bs1 | 2.319 (34.4) | 2.364 (32.44) | 2.368 (32.65) | 2.396 (32.79) |
| bs8 | 2.107 (69.4) | 2.102 (72.41) | 2.127 (71.15) | 2.185 (65.23) |
| bs16 | 1.983 (94.8) | 1.988 (94.32) | 2.021 (95.34) | 2.022 (95.88) |
| bs32 | 1.597 (131.3) | 1.599 (131.08) | 1.590 (135.76) | **1.676** (123.19) |
| bs128 | 2.389 (194.7) | 2.387 (192.52) | 2.396 (194.48) | **2.444** (188.56) |
| bs256 | 2.552 (242.1) | 2.520 (244.33) | 2.549 (245.00) | **2.644** (233.86) |

**Read honestly:**
- **v66's own shapes moved as the A/B said.** bs32 +4.9 %, bs128 +2.3 %, bs256 +3.6 % in speedup, against Modal's -3.9 / -2.5 / -2.8 % time.
- **The rest of the +3.3 % on the judge (v40 → v66) is draw.** bs1 +3.3 % and bs8 +3.7 % came with code Modal measured flat or within ±1 % (v50/v51/v58 at bs1; v59 +1-3 % at bs8). bs1 is the noisiest judge shape: the same bs1 code read 32.44-32.79 µs across these runs, with SGLang's own bs1 moving 76.67-78.58 µs.
- **The real gain is the A/B's ~+1.6 %.** Later same-code readings were 2.1871 for v71 and 2.2057 for v72; milestone 03's local A/Bs put v71 at +0.6 %. Both suggest the judge read v66 about 1 % high.
- **v59 was submitted against its own A/B.** The same job that gave v59 2.206 gave its control v51 2.205 (transcript, 19:01 UTC). It read +0.7 % over v50 on the judge.

**vs projection:**
- End-of-call reset: 0.5-1 µs at bs1/8/16 on the H100, as projected; ≤ 0.3 % geomean.
- Fast combine: bs8/16 -1 to -4 %, inside the projected 1-3 %. At bs1 it did not show: v59 traded the k-split of bs1's down items (fine items, tight tail) for the fast combine, and the tail got worse. The median CTA finished its last item at 27.9 µs but the slowest at 32.1, against 32.0 / 32.8 with the split (transcript, 20:09 UTC).
- Loops: bs32 -3.9 % against a ~7 % dispatch bound. The duplicated body was a bonus at bs128/256.

## 6. Takeaways, and what came next

- **Check the SASS of every loop restructure.** A textually equivalent loop compiled to twice the stage body. Opcode counts per kernel (`kern_sass.py`) became a standing check, together with C7511.
- **Waits and fences at the end of the call are measurable but small (≤ 0.3 % each).** Spinning warps cost issue slots in ncu, but taking them away did not change time.
- **A fast path that needs full-k items trades against the tail.** At bs1 the fine k-split items kept CTAs finishing together; removing them for the fast combine moved the loss to the tail. Milestone 03 removed the cross-CTA steps at bs1 entirely.
- **What came next** (`../03_fused_one_token_kernel/`):
  - main-kernel pieces: dynamic combine claims, a lean producer copy plan, a parallel end reset, a bs1 producer early start;
  - then a separate fused one-token kernel for M = 1, which took bs1 from ~30 to ~27.6 µs on the judge.

Raw notes: the dated memory notes `mem:okc-moe-versions.md` ("Lineage of what helped" lists v50-v66) and `mem:okc-moe-findings.md`, cited by their original paths (their copies stay private).

## Reproduce

```bash
cd muse_moe_w4a16/tools
python3 check_mt.py ../02_per_item_loops_ssq_fast_combine/build.py 1 4 8 16 32 33 128 256 257
# same-job A/B vs milestone 01, two copies each (judge-matched routings)
P=../01_persistent_fused_kernel_tma_ring/build.py; N=../02_per_item_loops_ssq_fast_combine/build.py
B2_SHARE_EQ=1 B2_SEEDS=jm_seeds.jseeds python3 bench2.py - --reps 60 --shapes 1,8,16,32,128,256 \
  --variants "a1|$P|;c1|$N|;a2|$P|;c2|$N|"
```

As run at the time (from `M/`, on a Modal H200 through `dev/modal_run.py`):

```bash
sh dev/sweep_list.sh dev/sweeps/s21.lst 20                  # v59 vs v66, 3 interleaved rounds
python3 dev/kern_sass.py moe_v66                              # per-kernel opcode counts, ptxas C75xx warnings
```

Originals: `M/dev/moe_v41.py` … `moe_v70.py`; Modal sweeps `M/dev/sweeps/s1-s26.lst` and `_result.txt`; ncu/timing logs `M/dev/ncu_*_v50.log`, `M/dev/timing_v50_h200.log`; judge results `M/profiles/submission.{055608b9,80f39693,348e75ba}*`.
