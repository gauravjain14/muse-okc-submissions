# 08: fast start, bs8 end chain, wgmma stage boundaries and early release of the held slot (v814 → v969)

Paths: `M/` = `muse-opt/muse_moe_w4a16/`, all relative to ``. Copies in this folder are cited as `evidence/...` and `notes/...`. The integrator's log is `M/kda_ws/docs/draft.md` §5zh-§5zz.

| | before (07) | after (08) |
|---|---|---|
| build | `07_kred_partials_lean_producer/build.py` = v814 (md5 fdabb515) | `build.py` = v969, `M/dev/moe_v969.py` (md5 24727739, 446,766 B) |
| judge | 2.5359 check / 2.5197 scored; real 2.5278 | 2.5603 check (99c8ef0c), **2.5551** scored (a142c92b); real 2.5577 |
| real effect | | local, chained same-job bench2 steps ≈ **+1.3 %** geomean (derived from the per-step gains below); judge real level +1.2 % |
| shapes changed | | bs8 (end chain), bs16 (EWG2), bs32 (fast start, TMP, RT), bs128 (WGSN, TMP, TMG, RT), bs256 (TMP, RT). bs1 unchanged |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: one-line comments credit the designs or reference code that parts of it follow. The code is unchanged. The judged file has md5 `24727739`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

Terms used below. **Held slot**: in the one-k-warp kernels (8,4,1) (bs17-128) and (8,8,1) (bs129-512), an item's last ring slot holds the consumers' staged FP32 accumulators until the asynchronous-epilogue warps 10-11 have finished SwiGLU and the stores, and the next item's stage NS-1 needs that slot. **Gather**: for tiles with many token groups the activations are first gathered tile-major into a global buffer, followed by an all-CTA count barrier. **wgmma tiles**: tiles with ≥ 3 token groups run `wgmma.m64nNk16` with A = decoded weights from registers and B = activations from 128B-swizzled shared memory. **KRS**: the fast combine's split rows (a down item's last-round k-split partials).

## 1. Starting point

v814's two runs (`evidence/judge_before_v814_b9a6bf4a.txt`; check in milestone 07), µs: bs1 22.4-23.3, bs8 63.9-64.9, bs16 83.7-85.3, bs32 112.2-118.5, bs128 169.2-171.4, bs256 202.5-203.3.

## 2. What we saw

**(a) bs32 sat idle on DRAM for its first 4.7 µs** (`notes/wp32.md` §4; per-CTA probe `wp32_probe.py` on v801). Routing bitmaps took 1.66 µs, tables 2.46, producer loop top 3.44, first weight copy ~4.7, first data 7.09 µs. DRAM is idle until ~4.7 µs, against 1.7-2.4 µs for a minimal kernel. Bounds: removing the gate/up → down dependency (`NODEP`) x0.999, so it is not a cost; no combine x1.017. Injected start delays convert ~1:1 into wall time.

**(b) The first stage of every item is late, and the cost sits on the consumer side** (`M/kda_ws/docs/draft.md` §5zk-5zl). Every item's first stage costs 3300-4500 cycles vs ~1800 mid-item (bs32). Per stage, down stages are no slower: the timing-only bounds NOX2 / XRES / W13 / NOEPI2 / NOPOLL / NOFENCE all read ~1.0. The SASS showed that ptxas warp-aggregates the producer's lane-0 item claim (a `SHFL` of the atomic's result right after `ATOMG`). Fixing that was wall-time neutral here: at bs8/16 the stream paces, and at bs32/128 the item-start cost is the consumers' item-end staging.

**(c) The held slot is the ring's turnaround at bs32-256** (`notes/wpa.md`; held-slot turnaround in median cycles, from the epilogue-item timeline `wpa_et.py`). A timing-only bound that frees the slot before it is read (`AREL0`) is worth +1.35 % bs32, +1.7 % bs128, +2.4 % bs256. Freeing it early by other means had already lost three times (SEPI8, AREL, BSTG -3..-4 %, `notes/wpw.md`).

**(d) wgmma stages cost ~30 % more in the kernel than decode + isolated wgmma** (`notes/wpwg.md`; ncu on v894 / v944, `evidence/ncu_v944/details_bs*.txt`). wgmma sync stalls were ~5.5 % of bs128 samples. ptxas pipelines each 64-k chunk's 8 HGMMAs into the next chunk's decode, leaving 16-24 HGMMAs per SMSP outstanding at the stage end, so `DEPBAR` waits are 9-14 % of the wgmma loops' samples. Dispatch stalls are 16 % vs 9 % in the mma.sync loops: register-file / dispatch contention between HGMMA with FP32 accumulators and the decode.

**(e) The bs8 end chain dominated its tail** (`notes/wpe2.md`; no-combine bound on v837: bs8 +4.0 %, bs16 +1.1, bs32 +1.3, bs128 +1.3, bs256 +1.1 %).

**(f) Startup at bs32-256 is a chain of cold code and barriers** (`notes/wprt.md`; per-CTA startup probe, judge protocol):
- bs32: the wgmma-first CTAs see first data at 10.05 µs vs 5.62 for the mma.sync CTAs.
- bs128: producer cold code 2.3 µs, gather done 5.09, all-CTA barrier ~7.5, first data 8.3 µs.
- bs256: GCP conversion + zeroing 3.2 µs, barrier 10.96, first data 12.3 µs.
- The ncu report on v944 put SM idle at bs1 9.9 %, bs8 12.4 % (19 % on v764), bs16 13.7 %, bs32 4.8 %, bs128 2.4 %, bs256 2.4 % (`M/kda_ws/docs/draft.md` §5zu).

## 3. Diagnosis and projection

Again the stage loop was not the limit. bs8/16 hot loops sit within 1.7 % of the bare ring per stage (1138 vs 1119 ns, WP-16W, `notes/wp16w.md`). The levers were the first copy at bs32, the held slot's turnaround, the wgmma stage boundary, and the startup chain. Bounds: fast start ≈ 2.5 % of bs32 (first copy 4.7 → ~2 µs at 1:1); held slot +1.35 / +1.7 / +2.4 % (AREL0); bs8 combine up to +4 %.

## 4. Change

| build | piece | shapes | bench gain (same job) | source |
|---|---|---|---|---|
| v837 | WP-32 `MOE_WP32_FST`: warp 8 ranks experts right after the routing bitmaps and issues the first item's 4 stages off the table barrier, through a `__noinline__` issue function so the consumer loops stay byte-identical | bs32 | x1.0068 / x1.0078 (control 0.9974): +0.75..1.05 %; first data 7.1 → 6.0 µs | `notes/wp32.md` §4 |
| v894 | WP-E2 bs8 end chain: KRS self-cleaning split rows + SPE preloads | bs8 | +0.5 % | `notes/wpe2.md` |
| v916 | WP-E2 EWG2 | bs16 | +0.24 % | `M/kda_ws/docs/draft.md` §5zr |
| v944 | WP-WG `MOE_WGSN`: wgmma items through their own stage loop; after the stage's last commit, the next slot's full-barrier wait and its weight/scale LDS are hoisted before `wgmma.wait_group 0` and the slot release | bs128 | +0.5..0.7 % | `notes/wpwg.md` §3 |
| v719 | WP-A `MOE_WPA_TMC` + `MOE_WPA_TMP`: every item's first min(n, 12) tokens of staged accumulators leave the held slot by one bulk copy while warps 10-11 process the rest from the slot; then the slot is released early | bs32, bs128, bs256 | +0.5 / +0.5 / +1.1 % | `notes/wpa.md` runs 4-6 |
| v984 | WP-A TMG / TMGS at bs65-128: held-slot rows spilled to a per-CTA global scratch | bs128 | +0.5 % | `notes/wpa.md` |
| **v969** | WP-RT startup: FR (ballot-based expert ranking), PL (pair list), INIT (mbarrier inits on warp 11), WREL (first weight copy right after the CTA's own gather release at bs33-128) | bs32, bs128, bs256 | bs128 +0.6, bs32 +0.17, bs256 +0.3 %; FR+PL alone bs128 +0.24, INIT+WREL alone +0.31: they add | `notes/wprt.md` run 12 |

**Fast start: how it converged** (`notes/wp32.md` §4; bench2 bs32 vs v801, control in brackets):

| build | change | bs32 | first data µs | finding |
|---|---|---|---|---|
| v831 | warp 8 issues the first item's 4 stages right after the bitmaps, inside the routing's table barrier | x0.9940 / 0.9959 (0.9968) | 7.78 | the ~2.9 µs issue call held the table barrier: tables 2.53 → 4.56 µs, consumers' entry 3.68 → 7.20 |
| v832 | warp 8 arrives on a named barrier that replaces that `__syncthreads` | x0.9993 / 1.0000 (0.9970) | 7.47 | the call holds the producer warpgroup's `setmaxnreg.dec`, so the consumers' `setmaxnreg.inc` waits too |
| v834 FSTD | the call after the dec | x0.9822 / 0.9819 (0.9970) | 9.26 | 2.2 µs later first stage costs ~2 µs of wall time (1:1) |
| **v835** | `redux.sync` selection (was a 1.3 µs shuffle ranking); copies issued stage by stage (stage 0's x rows had queued behind 140 KB of weights in the TMA FIFO) | **x1.0068 / 1.0078** (0.9974) | 5.95 | kept; the 32 wgmma-first CTAs still see first data at 9.7 µs (their gather waits for the FST CTAs' signaler) |
| v836 FDRY | dry pass of the call during the ids load (warm i-cache) | x0.9954 | 8.30 | the dry pass held the bitmap barrier: lost |

Two details mattered: the TMA unit serves bulk copies in FIFO order, so the small activation copies must not queue behind 140 KB of weights; and anything that holds a warpgroup-collective `setmaxnreg` delays every other warpgroup's start.

**TMP: early release of the held slot.** Held-slot turnaround, median cycles, base → TMP (`notes/wpa.md` run 6): bs32 G1 1-group 3488 → 2480; bs128 G1 multi-group 8670 → 7150 and 6934 → 4953, G1 1-group 2426 → 1556, G2 multi 2119 → 1904 and 1776 → 1402; bs256 G1 multi 7309 → 6090 and 4702 → 3196. The room for the copy is the shared memory past the ring: 16,384 B free in the (8,4,1) MW 1 kernel, enough for 15 staged FP32 tokens (`notes/wpa.md`, "Room past the ring"). A variant applying TMP only to multi-group items (TMPM=2) lost 1.3 % at bs256: its runtime branch kept both paths live (+400 instructions).

## 5. Result

| check | result |
|---|---|
| local A/B | per step, same job, ≥ 2 copies vs controls (table above). Chained ≈ +1.3 % geomean (derived: ~+0.15 v837, ~+0.1 v894, ~+0.04 v916, ~+0.1 v944, ~+0.35 v719, ~+0.1 v984, ~+0.15 v969, weighted by shape) |
| correctness | `check_mt` M 1..257 0 failures per base; no C7511/C7512 serialization in the banked forms (`MOE_WGSM=0x8080` serialized the (8,8,1) wgmma and was kept off there); judge tests 5/5 |
| judge | v814 5303.3 / 5337.4 → **v969 5252.8 (check) / 5263.3 (scored)** |

Judge readings across the day (µs; `evidence/judge_*.txt`):

| build | score µs | bs1 | bs8 | bs16 | bs32 | bs128 | bs256 |
|---|---|---|---|---|---|---|---|
| v814 scored | 5337.4 | 23.30 | 64.90 | 83.72 | 118.49 | 169.19 | 202.46 |
| v837 check | 5347.2 | 23.22 | 61.60 | 90.09 | 112.56 | 172.58 | 205.07 |
| v894 check | 5293.1 | 23.54 | 64.11 | 86.34 | 115.55 | 173.38 | 199.83 |
| v944 check | 5314.5 | 24.33 | 60.07 | 84.48 | 116.53 | 168.80 | 204.05 |
| v719 check | 5254.2 | 22.86 | 63.23 | 85.00 | 111.19 | 173.89 | 200.48 |
| v969 check | 5252.8 | 22.73 | 59.10 | 84.14 | 111.76 | 165.78 | 205.18 |
| **v969 scored** | **5263.3** | 23.11 | 63.99 | 84.29 | 111.58 | 164.68 | 202.70 |

**The judge, read honestly.**
- bs1 ran identical code in every build of this milestone and read 22.73-24.33 µs (7 %).
- bs8 changed only at v894, yet read 59.10-64.90 µs across these runs. Its two v969 runs differ by 8 % (59.10 vs 63.99) with the same code.
- What tracks the local gains is the per-milestone level: v814-class builds averaged ~5330 µs, v894/v944 ~5300, v719/v969 ~5255 (`M/kda_ws/docs/draft.md` §5zi, §5zs, §5zx).
- Real level 2.5577 vs 2.5278 (+1.2 %), consistent with the local ~+1.3 %.

**vs projection.** The pieces landed close to their bounds where one mechanism dominated: fast start reached ~0.9 % of its ~2.5 % bound (the wgmma-first CTAs still start late), and TMP reached roughly half of AREL0's bound at bs256. The bs8 end chain recovered ~0.5 % of a 4 % no-combine bound: its critical unit is the split rows' L2 round trip.

## 6. Takeaways, and what came next

- **Order the first copies by size.** The TMA FIFO serves bulk copies in issue order, so the small activation copies have to go before the 140 KB of weights.
- **Watch warpgroup-collective instructions on the startup path.** `setmaxnreg.dec`/`.inc` couple the producer's and the consumers' starts. A long call before the dec delayed every consumer.
- **The held slot is the turnaround in one-k-warp kernels.** Copy the first tokens out with one bulk copy and release the slot before the epilogue finishes.
- **Hoist the next stage's barrier wait above `wgmma.wait_group 0`.** The cost is RF/dispatch contention between HGMMA and the decode, not drain. Commit-group, pre-decode and accumulator-order variants all lost.
- **Next:** [`09_late_claim_buckets_moe1t_end`](../09_late_claim_buckets_moe1t_end/) resumes after a pause, with a bucket move for bs65-128, a fence-free bs1 end, late non-aggregated queue claims, and barrier-free item-transition staging.

## Reproduce

```bash
cd muse_moe_w4a16/tools
python3 check_mt.py ../08_fast_start_early_slot_release/build.py 1 2 4 8 9 16 17 32 33 64 65 128 129 256 257
P=../07_kred_partials_lean_producer/build.py; N=../08_fast_start_early_slot_release/build.py
B2_SHARE_EQ=1 B2_SEEDS=jm_seeds.jseeds python3 bench2.py - --reps 100 --shapes 1,8,16,32,128,256 --out ab08.json \
  --variants "fA|$P|;a1|$P|;c1|$N|;a2|$P|;c2|$N|;fB|$P|"
# judge-scale projection: project2.py (not shipped: it embeds a leaderboard snapshot) takes, per shape, an anchor build's judged speedup x the median same-routing time ratio anchor/candidate, then the geomean
```

ncu text exports of v944 (`--set full`, one jm seed per shape) are in `evidence/ncu_v944/`; the per-line and SASS CSVs (`cs_bs*.csv`, `sass_bs*.csv`, 1-6 MB each) stay in `M/profile/v944_ncu/`.

Originals: `M/kda_ws/docs/draft.md` §5zh-§5zz; findings `M/kda_ws/docs/findings/{wp32,wpe2,wpwg,wpa,wprt,wpds,wpedge,wps5,wp16w,wpdual,wpt96}.md`; builds `M/dev/moe_v830`-`v837`, `v890`-`v894`, `v916`, `v940`-`v944`, `v710`-`v719`, `v984`, `v960`-`v969`; judge results `M/dev/checks/profiles/plan_v{837,894,944,719,969}.*`, `M/profiles/submission.a142c92bf1e54c3d.*`.
