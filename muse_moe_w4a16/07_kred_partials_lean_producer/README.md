# 07: fence-free split partials, a lean producer and a combine without local memory (v567 → v814)

Paths: `M/` = `muse-opt/muse_moe_w4a16/`, all relative to ``. Copies in this folder are cited as `evidence/...` and `notes/...`. The integrator's log is `M/kda_ws/docs/draft.md` §5q-§5zg.

| | before (06) | after (07) |
|---|---|---|
| build | `06_bf16_rebase_two_phase_bs1_kernel/build.py` = v567 (md5 3156a829) | `build.py` = v814, `M/dev/moe_v814.py` (md5 fdabb515, 359,867 B) |
| judge | 2.4346 check (69f9e304) | 2.5359 check (fa7c6551), **2.5197** scored (b9a6bf4a); real 2.5278 |
| real effect | | local, chained same-job bench2 steps ≈ **+2.9 %** geomean (derived from the per-step gains below). The judge's real-level gain is +3.8 %; the v567 check probably read low (see Result) |
| shapes changed | | all six: bs1 (moe1t named barriers + lean fetcher), bs8/16 (combine, bs8 last-round split), bs32 (KRED, combine), bs128 (KRED, lean producer), bs256 (lean producer, KRED tail split, combine) |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: one-line comments credit the designs or reference code that parts of it follow. The code is unchanged. The judged file has md5 `fdabb515`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

Terms used below. **Item**: a work unit from the dynamic queue (a token tile times a block of weight rows over all of K, gate/up or down). **Split pieces**: the last round's down items are k-split into pieces so the tail is shorter. **Held slot**: the item's last ring slot, kept by the consumers while warps 10-11 (the epilogue warps) store the staged accumulators. **SMSP0 pair**: consumer warps 0 and 4, which share a sub-partition (SMSP) with the producer warp. **jm seeds**: judge-matched routings (`M/dev/jm_seeds.jseeds`). **bench2**: same-job paired A/B with ≥ 2 copies of each build. **KDA ncu method**: the six-dimension ncu report procedure (`--set full` + `--set source`, `--clock-control none`) the team adopted at 07:30 PT on 10-27 ("profile-driven from here").

## 1. Starting point

v567's check (`evidence/judge_before_v567_69f9e304.txt`), µs: bs1 23.97, bs8 65.33, bs16 85.77, bs32 116.85, bs128 176.18, bs256 214.42. The KDA multi-agent pipeline (a planner plus workers WP-*, each writing findings to `M/kda_ws/docs/findings/`) was in its first morning, under the BF16-only rule.

## 2. What we saw

**(a) Split pieces at bs32/128 ended in a fenced fixup** (`notes/wpk.md`). The (8,4,1) kernel's tail pieces stored four FP32 partial slices per split pair, and the combine read all four. A first fix (`MOE_KFIX`: the last-arriving piece sums the partials) lost 1.5 % at bs128. Its release atomic is synchronous (`MEMBAR.GPU` before the `ATOM`) and sits on the epilogue warps once per piece, and the fixer adds atomic + L2 round trips at the end of the call. The same chain had sunk two earlier fixup designs (v189, v190).

**(b) The producer's copy-issue path is the ring's critical path** (`notes/wpw.md`, `M/kda_ws/docs/draft.md` §5t-5u):
- 1-group stage compute is identical in the (4,1,1) and (8,4,1) kernels: 49.2 vs 49.3 ncu fixed-interval samples per iteration.
- But slot release → producer acquire takes ~435 cycles (suspend-hinted `try_wait` wake), acquire → copies issued ~1000 cycles, and the ring's turnaround 4700-4900 cycles against a ~5700-cycle lead. The slack is under 1000 cycles.
- Injected `MOE_PDELAY=200 ns` per stage cost 4-6 % (bs32/128/256 x0.939 / 0.950 / 0.962); +64 producer instructions per stage cost 3-4 %.
- The SMSP0 pair runs +4-4.5 % slower than the other consumers in the (8,4,1) kernel and paces the ring. Moving the producer to SMSP1 (`PSWAP`) moved the skew with it.

**(c) The SASS showed why the producer was expensive: divergent copy-issue waterfalls** (`notes/wpn.md` §3). ptxas compiles every lane-predicated `cp.async.bulk` (`if (lane < k) bulk_g2s(...)`, operands in vector registers) into an `ELECT` / 5x `R2UR` / 2x `PLOP3` / `UBLKCP` / `BRA.U.ANY` loop with one iteration per active lane, inside `BSSY` / `BSYNC` / `BRA.DIV` reconvergence. v730's producer ran three of these per 1-group stage (~108 warp instructions per stage, issued over ~1100 cycles while its SMSP's two consumer warps compute). Timing probes on SMSP0 skew, in cycles per stage (cap32 G1 / G2, bs32 G2, bs128 G1 / G2):

| probe | what changes | SMSP0 skew |
|---|---|---|
| v730 | | 137 / 120, 113, 99 / 123 |
| HALFW: half the weight bytes | bytes -50 %, same instructions | 116 / 122, 118, 102 / 124 |
| SPLITW: weights as 2 x 16 KB | +1 waterfall iteration, same bytes | 164 / 152, 148, 129 / 148 |
| NO_LOAD (no copies at all) | | 88 / 88, 82, 90 / 72 |

One more waterfall iteration (~10 instructions) costs the SMSP0 pair ~30 cycles per stage. The bytes do not matter.

**(d) The first KDA ncu report (v730) put the waste outside the loop** (`evidence/ncu_v730/details_bs*.txt`; `M/kda_ws/docs/draft.md` §5z):
- DRAM 58-72 % of peak; issue slots 39-52 %.
- The stage loop is 510 instructions per 32 words: 13 decode instructions per 8 weights, the floor, at ~69 % issue efficiency, with no spills or bank conflicts.
- Off the loop:
  - the combine used **local memory** (a 64-B stack; 29K-212K local sectors, 72-79 % L1 miss);
  - mbarrier waits were 12-13 % of samples;
  - SM idle (IPC elapsed / active) bs1 11 %, bs8 19 %, bs16 14 %, bs32 8 %, bs128/256 3 %;
  - ptxas split 16/8-B stores (4x the L2 sectors).

**(e) The end phase, stamped** (`notes/wpe2.md` §1; globaltimer, judge protocol, 6-10 jm seeds). Last down stage → last warp exit: bs8 3.1-3.5, bs16 3.4-4.3, bs32 3.7-4.7, bs128 3.0-3.7, bs256 5.5-6.4 µs. The last tile's chain costs 1.0-1.5 µs: staging 0.13-0.35, epilogue wake 0.10, stores 0.22-0.96, signaler wake 0.19-0.26, `red.release` (`MEMBAR.GPU`) 0.38-0.43. The critical combine unit then needs 1.30-3.1 µs from its last poll to the stored output.

**(f) bs1's intermediate fetch took 1.1-2.5 µs per poll round** (`notes/wpb1.md` runs 13-16). In moe1t's phase B a fetcher warp polls tagged intermediate words written by other CTAs. In the judge SASS, v801 issued 13 of its 32 poll `LDG.128` *after* the first tag check, so every poll round was two round trips. ncu (`evidence/ncu_v764/details_bs1.txt`, per-line view in `M/profile/v764_ncu`) showed the fetch line at 77 % MIO throttle.

## 3. Diagnosis and projection

Each of these was a latency or issue cost outside the decode loop, which itself was at its floor. The workers' bounds: KRED was bound by the fixup chain it removes (~1.5 % bs128 from KFIX's loss, plus the combine's 4-slice reads). The producer's SMSP0 skew was worth ~4 % at bs32/128/256 (~2 % geomean) if it could be zeroed (WP-N bound, `M/kda_ws/docs/draft.md` §5w). The combine's no-combine bound was +1.1-4.0 % per shape (`M/kda_ws/docs/draft.md` §5zl, on v837). bs1's fetch rounds 3 → 1 were worth ~1 µs (~4 %). Summed over the day, the measured per-step gains came to ~2.9 % geomean (derived).

## 4. Change

| build | piece | shapes | bench gain (same job) | source |
|---|---|---|---|---|
| v655 | WP-K `MOE_KRED`: the (8,4,1) tail pieces `red.relaxed.gpu.global.add.v4.f32` their scaled FP32 rows into one accumulator row per split pair (zeroed by the consumers during the gather, ordered by the gather's existing barrier + `red.release`); the combine reads 1 row instead of 4. Plus `MOE_XDIR`, `MOE_XREL`, `GATHER_MIN0 5` (gather-free start at bs32) | bs32, bs128 | bs32 +2.0 % (x1.0068 / x1.0128), bs128 +0.7 % (x1.0046 / x1.0072) | `notes/wpk.md` |
| v730 (= v695) | WP-Z: `SDEC` division-free k-split decode, `GDIV32`, `EHOIST` epilogue store loop, bs129-512 KRED tail split (KSX8 4, SPLW8 0.6); + WP-W `FSW` shorter SwiGLU epilogue | bs32-256 | bs32 +1.0 / +1.2, bs128 +0.9 / +1.0, bs256 +0.9 / +1.35 % | `notes/wpz.md`, `notes/wpw.md` |
| v749 | WP-N lean producer: NBAR (named-barrier slot release), LEAN3 (elected-lane copies in a tight per-item loop), LWSC (one combined weights + 6-bit scales block), LEAN4 for (8,8,1) | bs33-512 | bs128 +0.6..1.0 %, bs256 +2.5..2.8 % (run 12; draft: bs128 +0.8, bs256 +2.3) | `notes/wpn.md` §4-6 |
| v752 | WP-E2: combine without local memory (per-slot register arrays, one polling lane per slot at bs2-16; lanes + `__shfl` pair lists at bs17-512; early next claim) | all M ≥ 2 | bs8 +1.3, bs16 +0.9, bs32 +0.8, bs128 +0.2, bs256 +1.3 % (2 runs) | `notes/wpe2.md` |
| v764 | WP-N `MOE_WPN_1TNB`: moe1t slot hand-offs on named barriers, elected-lane producer | bs1 | +0.8..1.1 % | `notes/wpn.md` |
| v801 | WP-E2 bs8 last-round 4-way k-split on the fast (SSQ) combine | bs8 | +1.0..1.9 % | `notes/wpe2.md` |
| v814 | WP-B1 lean fetcher: `MOE1T_FLEAN` (base pointers once per expert, immediate offsets, all 32 poll loads issued before the first tag check), `MOE1T_F2` (two fetcher warps x 16 loads), `MOE1T_FDLY=500` (last expert's first poll +500 ns) | bs1 | +2.2..2.7 % (run 16), +2.9 % (integration bench) | `notes/wpb1.md` run 16 |

**KRED: fence-free accumulation of split partials** (`notes/wpk.md` attempt 2). Correctness rests on ordering that already existed. The split pairs' accumulator rows are zeroed during the gather, before the gather's consumer barrier and `red.release(CB[3])`. Every piece's CTA has acquired CB[3] (or a gate/up item's counter whose CTA did) before it adds. The signal counts are unchanged (16 x 4 pieces), so the combine's existing acquire covers the adds. FP32 accumulation throughout; the add order of the 4 pieces varies per call (relative ~1e-7). SASS: (8,4,1) +40 instructions, no spills. One codegen lesson from this worker carried through the project: an unused `constexpr` local in the kernel body re-schedules the bs16 kernel. From then on every switch was a macro inside `#ifdef`.

**The lean producer: one elected lane, warp-uniform operands** (`notes/wpn.md` §3-6). Issuing the copies from one lane chosen by `elect.sync`, with warp-uniform operands, makes ptxas emit `ELECT` + `R2UR` + `UBLKCP`: no loop, no divergence. A standalone test kernel's `cuobjdump` showed the waterfall gone. LEAN3 moves every stage of an item that needs no activation wait into a 102-instruction loop. Per stage it does a named-barrier slot wait, a record write on the first and last stage only, the expected-byte post, and the combined weights + 6-bit scales block, plus activation copies from per-item descriptors advanced by constant strides. That is ~70 instead of ~108 warp instructions per 1-group stage, and the producer's stage path drops from 1166 to ~300 cycles. LEAN4 drops record and held-slot bookkeeping from steady stages: bs256 SMSP0 skew 78 → 60 cycles (compute part 66 → 8), wgmma-stage skew 147 → 68, period 1786 → 1754 cycles. At bs128 the same change drew a slower consumer schedule (1381 → 1455 cycles per stage), so LEAN4 is baked for (8,8,1) only.

**How the day converged** (judge checks; every build also passed `check_mt` M 1..257, normal + skewed):

| build | judge (check) µs | speedup | what the judge said | source |
|---|---|---|---|---|
| v567 | 5523.8 | 2.4346 | | `evidence/judge_before_v567_69f9e304.txt` |
| v655 | 5427.8 | 2.4777 | ~1 % lucky vs its bench projection (~5499) | `evidence/judge_v655_d17dc36e.txt` |
| v730 | 5497.8 | 2.4462 | unlucky: same-kernel shapes read -1..-5 % (bs1 25.8 vs 23.4 µs) | `evidence/judge_v730_1495f448.txt` |
| v749 | 5421.8 | 2.4805 | | `evidence/judge_v749_09f2a289.txt` |
| v752 | 5436.4 | 2.4738 | = v749 within noise | `evidence/judge_v752_9df0a63e.txt` |
| v764 | 5359.8 | 2.5091 | | `evidence/judge_v764_42961421.txt` |
| v801 | 5372.4 | 2.5032 | bs8 2.340 (+1.7 %, as bench); identical-kernel shapes -1.5 % | `evidence/judge_v801_dd9a4a5d.txt` |
| v814 | 5303.3 | 2.5359 | | `evidence/judge_after_v814_fa7c6551.txt` |
| v814 scored | 5337.4 | 2.5197 | same code, -0.6 % | `evidence/judge_after_v814_b9a6bf4a.txt` |

Rejected along the way (details in `../dead_ends.md`): KFIX (-1.5 %); WP-H PDL hybrid kernel (CTAs launched mid-call pay ~5 µs to first data); WP-T PDL tail kernel (x0.91 / 0.95 at bs8/16); WP-C `__noinline__` freeze (called loops pay ~3x instruction-fetch stalls; the real code-presence cost is the producer spilling at 72 registers); WP-F bs1 k-split of the last expert (x0.90-0.98); KRED splits at bs8/16 (x0.93: the dead slow-combine body becomes live, `moe_4_1_1` 5712 → 7592 instructions); the moe1t poll by TMA (FTMA, -6 %: the poll copy queues behind the ring's weight copies in the TMA unit); conflict-free x fragments in moe1t (XN, -1 %).

## 5. Result

| check | result |
|---|---|
| local A/B | per step, same job, ≥ 2 copies with controls (table above). Chained ≈ +2.9 % geomean (derived: ~+0.45 v655, +0.5 v730, ~+0.5 v749, +0.75 v752, ~+0.15 v764, ~+0.25 v801, ~+0.25 v814) |
| correctness | `check_mt` M 1..257 (normal + skewed, 3 seeds, 2 calls) 0 failures at each base; `hang1` per build; no ptxas C7511/C7513 (wgmma serialization) in any banked build; judge tests 5/5 |
| judge | v567 5523.8 → **v814 5303.3 (check) / 5337.4 (scored)**, real level 2.5278 |

Judge per case ("yours" µs):

| shape | v567 check | v814 check | v814 scored | change (mean of v814 runs vs v567) |
|---|---|---|---|---|
| bs1 | 23.97 | 22.36 | 23.30 | -4.8 % |
| bs8 | 65.33 | 63.87 | 64.90 | -1.5 % |
| bs16 | 85.77 | 85.27 | 83.72 | -1.5 % |
| bs32 | 116.85 | 112.23 | 118.49 | -1.3 % |
| bs128 | 176.18 | 171.42 | 169.19 | -3.3 % |
| bs256 | 214.42 | 203.28 | 202.46 | -5.4 % |

**The judge, read honestly.**
- Nine judge readings over the day moved in steps that tracked the local gains only loosely. v655 read 1 % lucky and v730 1 % unlucky. The two v814 runs differ by 0.6 % overall, with bs32 at 112.23 vs 118.49 µs (identical code).
- The team's own reading at the time: "judge level for v814-class builds ~5330 mean (5303 / 5337 scored / 5347)" (`M/kda_ws/docs/draft.md` §5zi).
- Against v567's single check, the real level moved +3.8 %, against a local +2.9 %. v567's check most likely sat ~1 % low: the same-kernel bs1 read 23.97 there vs 22.4-23.4 µs in later runs.

**vs projection.** The bs256 gain on the judge (-5.4 %) matches the lean producer plus KRED tail split plus combine (+2.5 + 1.1 + 1.3 % ≈ 5 %). bs32 is the noisiest shape: one v814 run read slower than v567.

## 6. Takeaways, and what came next

- **Read the SASS of anything lane-predicated that issues work.** A `cp.async.bulk` under `if (lane < k)` becomes a divergent per-lane loop. On a warp that shares an SMSP with the critical consumers, ~10 extra instructions per stage cost ~30 cycles of their time. `elect.sync` with warp-uniform operands removes it.
- **Reuse ordering that already exists instead of adding fences.** KRED needed no new synchronization: the zeroing rides on the gather's release, and the adds on the combine's acquire.
- **The ncu method paid immediately.** Local memory in the combine, waits, and SM idle were all invisible in the bench. From here on every lane started with an ncu/trace budget, and the waste outside the loop became the work list.
- **Single checks carry ±1 % geomean.** The day's judge checks only made sense as a running mean.
- **Next:** [`08_fast_start_early_slot_release`](../08_fast_start_early_slot_release/) continues on the edges: a fast start for bs32, the bs8 end chain, wgmma commit grouping, and releasing the held slot before the epilogue finishes.

## Reproduce

```bash
cd muse_moe_w4a16/tools
python3 check_mt.py ../07_kred_partials_lean_producer/build.py 1 2 4 8 9 16 17 32 33 64 65 128 129 256 257
P=../06_bf16_rebase_two_phase_bs1_kernel/build.py; N=../07_kred_partials_lean_producer/build.py
B2_SHARE_EQ=1 B2_SEEDS=jm_seeds.jseeds python3 bench2.py - --reps 100 --shapes 1,8,16,32,128,256 --out ab07.json \
  --variants "fA|$P|;a1|$P|;c1|$N|;a2|$P|;c2|$N|;fB|$P|"
python3 kern_sass.py ../07_kred_partials_lean_producer/build.py      # per-kernel SASS summary, C7511/C7513 check (local NVRTC)
```

ncu text exports: `evidence/ncu_v730/details_bs*.txt` (the first KDA report) and `evidence/ncu_v764/details_bs*.txt`. The `.ncu-rep` files (7-10 MB each) stay in `M/profile/v730_ncu/reports/` and `M/profile/v764_ncu/` (per-line `cs_bs*.csv` and SASS `sass_bs*.csv` there too). The candidate ledger of the KDA workers is `evidence/kda_candidates.jsonl` / `evidence/kda_benchmark.csv`.

Originals: `M/kda_ws/docs/draft.md` §5q-§5zg; findings `M/kda_ws/docs/findings/{wpk,wpz,wpw,wpn,wpe2,wpb1,wpq,wph,wpt,wpc,wpf}.md`; builds `M/dev/moe_v650`-`v655`, `v690`-`v730`, `v740`-`v749`, `v750`-`v752`, `v764`, `v801`, `v810`-`v815`; judge results `M/dev/checks/profiles/plan_v{655,730,749,752,764,801,814}.*`, `M/profiles/submission.b9a6bf4ad4f14ee2.*`.
