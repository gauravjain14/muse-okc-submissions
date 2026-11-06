# 08: s1, exhaustive cuBLASLt configs without workspace, finer 16K boundary tails, and a judge-faithful local scorer

| | before (07) | after (08) |
|---|---|---|
| build | `07_.../build.py` = Codex `bundle23_gate_queue.py` "b23" (md5 bea50449, 964,737 B) | `build.py` = `s1.py` (md5 01e3ed8b, 808,815 B) |
| judge | 1.9955 (job fa9e150b, single reading) | **1.9831** (check c6fe3962), **1.9901** (scored a7ea31e5); real level **1.9866** |
| real effect | b23's replica level is ~1.959, so its judge reading was a favourable draw (see 5) | judge-faithful replica, 3 router draws x 8 reps, all 8 cases: **+0.45 %** (1.0029 / 1.0033 / 1.0073) |
| shapes changed | | every dense GEMM slot at T = 753 / 3012 / 16384 except the 753 o_proj; 16K UB and D6 tail items (cases 4-7) |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: code copied or adapted from other projects is marked in place with `SPDX-SnippetBegin` / `SPDX-SnippetEnd` comments (source, copyright, license, our changes); see `NOTICE` and `LICENSES/` at the repository root. The code is unchanged. The judged file has md5 `01e3ed8b`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

Paths: `R/` = `muse-opt-codex/muse_prefill_layer/research/claude/`, `J/` = `muse-opt-codex/muse_prefill_layer/research/judge/`
(both relative to ``). Copies of the lane notes are in `notes/`.

Terms used below. **Case 0..7** are the judge's 8 scored shapes: 0 = T=753 sliding (turn-1 load, weight 46), 1 = T=753 full
(16), 2 = T=3012 sliding (turn 4, 46), 3 = T=3012 full (16), 4-7 = T=16384 "first turn" (4 full ctx0, 5 sliding ctx32K with
weight 138, 6/7 full ctx32K/64K). The score is a geomean over the 3 loads of the paired SGLang/ours ratio
(`tools/TASK.md`). **nvjet** is cuBLASLt's sm90 GEMM kernel family. **sk2** is our persistent o_proj GEMM.
**UB** (`ub2_2_192_3`) and **D6** (`ub2_2_192_4_4`) are the 16K-only persistent GEMMs that also run the post-FFN and
post-attention boundary rows (residual + RMSNorms) as time-sliced work items (milestone 05). **bnd3/bnd4** are the
standalone boundary kernels at 753 / 3012.

## 1. Starting point

b23 was Codex's last bundle: 23 rounds of exact micro-gains on top of the public expert kernel (milestone 07). Its one
judge reading, per case (`evidence/judge_before_b23_fa9e150b.txt`):

| case | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| SGLang us | **1170.3** | 1493.0 | 3530.7 | 5041.4 | 19135.4 | 16624.1 | 21458.4 | 24729.5 |
| ours us | 563.6 | 752.6 | 1796.4 | 2579.6 | 9459.0 | 8491.3 | 10689.3 | 12087.5 |
| ratio | 2.076 | 1.984 | 1.965 | 1.954 | 2.023 | 1.958 | 2.007 | 2.046 |

Its dense GEMMs in that run (profile rows of the same file): case 0 fused pool+router `tss_256x112` 80.5 us, unpooler
`tst_128x192` 65.2, QKV `tst_112x128` 39.1, sk2 o_proj 20.7; case 2 fused `tss_256x144_ssched` 282.4, unpooler
`tst_128x304_ssched` 249.8, QKV `tst_256x104` 147.6, sk2 o_proj 69.6; case 5 pooler `tst_128x256` 1356.9, QKV
`tst_256x128` 753.4, router `tss_256x128` 173.5, UB 1683.3, D6 582.8 us.

Where a case-0 layer went (local node trace of b23/b24, `R/NOTES.md:15-24`): QKV 39, attention 19+7, o_proj 21, bnd1 13,
pool+router 76, MoE 268, combine 9, unpool 65, bnd2 22 us. The dense GEMMs ran at ~89-92 % of the power-limited floor;
the MoE was the largest gap but already had three closed lanes behind it (milestones 09-10 pick that up).

## 2. What we saw

**(a) The local number Codex optimised was not the judge's quantity.** (`R/NOTES.md:3-13`, notes/memory_prefill-judge-metric.md)
- The judge's score is a constant 387,403.9 us divided by the speedup; the speedup is a geomean over loads of
  sum(w x SGLang) / sum(w x SGLang / paired ratio), with the paired ratio the median over AB/BA repetition pairs.
  Codex's "197.4 ms local" was our absolute latency, which moves with the GPU's state and never tracked the judge.
- The judge draws a new router (expert routing) every run. On a single fixed router seed b23 measured 1.9963 locally
  against its judge 1.9955 (`evidence/judge_replica_calib1.log`), which looked calibrated, but per-case ratios move 3-5 %
  between draws and candidate order flips: case 5 of b23 reads x1.9352 on fields seed 11 and x2.0127 on seed 22, and
  on seed 11 b14 beats b23 on the 16K load (1.9910 vs 1.9456) while on seed 22 it does not (2.0166 vs 2.0179)
  (`evidence/judge_replica_router_seed_test.log`). Codex's fixed `ROUTER_SEED` was a favourable draw, so its small
  gains were partly tuned to one routing.
- We wrote `J/replica.py` (copied to `tools/research/judge/replica.py`): SGLang-paired, judge overhead formula, cold L2
  before every replay, and `--panel` over judge-style router draws (fields seeds 11, 22, 33).

**(b) The first replica had a position bias of 0.4-1.1 %.** It timed `[sglang, c1, c2]` and then the reverse, so c1 always
ran in the middle and c2 alternated first / last. The last of three back-to-back 10-12 ms graphs runs slow
(power/thermal). Found by the boundary lane on the 16K tail items (`R/bnd/NOTES.md`, "REPLICA POSITION BIAS"):

| run | order | result |
|---|---|---|
| rep1 (`evidence/bnd_replica_rep1_tail_listed_second.log`) | base middle, tail first/last | tail/base 0.9987 / 0.9936; case 7 tail-base +213/+367/+426 us when tail ran last, -75/-58/-78 when first |
| rep2 (`evidence/bnd_replica_rep2_tail_listed_first.log`) | tail middle | base/tail 0.9909 / 0.9865 (biased the other way) |
| order-averaged | | tail **+0.39 %** on the first-turn load, consistent with nsys in situ |

Fix (`R/NOTES.md:59-63`): every candidate gets its own AB/BA pairs with SGLang and the candidate order rotates per pair;
the biased version is kept as `evidence/replica_v1_biased.py`. With it, b23 measures **x1.9592** over three draws
(`evidence/replica_s1_vs_b23compact_run_s1.log`), against its 1.9955 judge reading. Earlier panel numbers
(`evidence/judge_replica_panel_base_v1_biased.log`: b23 1.9778, b14 1.9582) carry the bias.

**(c) b23's dense plans came from cuBLASLt's heuristic list only, and many of them paid for a memset node.**
(`R/lt/NOTES.md`, notes/lt_NOTES.md)
- `lt_prefill.py` enumerated the full nvjet space (algo 66 x 634 tile ids x cluster 1x1..4x4 x custom option x swizzle x
  split-K x tn/nn layout x workspace mode): 84 k configs per shape at 753, 42-63 k at 3012, 36 k at 16K, each
  AlgoCheck-validated, then correctness, judge-flush timing, graphs, and a paired final round
  (`evidence/lt_standalone_results.json`).
- **Workspace = a memset node.** Many nvjet configs ask for a 4-byte workspace; cuBLAS then adds a Memset node before the
  GEMM. In the layer at 3012 every ws=4 config, including the incumbent's own algo rebuilt with a workspace, lost
  0.2-0.5 % of the layer. Passing `workspaceSize=0` for the same algo selects nvjet's memset-free `ssched` variant,
  often faster too (f3012 128x168: 283.0 vs 289.5 us).
- Standalone wins did not always survive in the layer. The in-layer sweep (`inlayer_sweep.py`: one GEMM slot swapped,
  whole layer re-captured and timed judge-style) was the arbiter:

| slot (case) | chosen config | in-layer GEMM delta | layer |
|---|---|---|---|
| 753 fused pool+router (0) | tn 144x192 1x1 (128 CTAs) | -2.3..-2.4 us | x1.0043..1.0052 |
| 753 unpooler (0) | nn 192x128 1x2 | -1.2..-1.5 us | x1.0029..1.0034 |
| 753 o_proj (0/1) | **not taken**: nvjet ws0 is 1.7 us faster per kernel, but layer x1.0033 / x0.998, because sk2 launches early behind the attention (PDL) and hides its fill | | |
| 3012 QKV (2) / capped QKV (3) | tn 256x104 / tn ws0 128x168 2x2 on 108 CTAs | -2..-3 / -3.1..-4.2 us | x1.0015 / x1.001 |
| 3012 o_proj (2/3) | tn ws0 128x168, replaces sk2 | | x1.0035 / x1.0014 |
| 3012 fused (2) / unpooler (2) | nn ws0 128x168 / tn ws0 128x168 | -6.1..-6.3 / -4.4..-6.6 us | x1.0043 / x1.0011..1.0032 |
| 16K QKV / pooler / router (5) | tn ws0 128x176 / tn ws0 128x160 / tn ws0 128x168 | -18 / -63..-65 / -6..-7 us | x1.0027..1.0039 / x1.0070 / noise |

- 16K standalone numbers drifted with thermal state (the incumbent's 16K QKV: 782 us eager at the start of a sweep,
  910 us after ~2 min of back-to-back GEMMs), and the ranking changed with it: hot favours 128x144..176 tiles, cool
  favours 256-wide tiles (`R/lt/NOTES.md`, finding 4). The 16K picks above were made in the hot state.

**(d) The 16K boundary-fused kernels had a 29 us tail, and the boundary rows themselves sat near a hardware floor.**
(`R/bnd/NOTES.md`, notes/bnd_NOTES.md)
- Traces (`UB_TRACE`, random data): UB = GEMM SM-time 1445 + boundary 240 us per CTA, idle 2.7 %, **tail 29 us after the
  last GEMM tile**; D6 = 392 + 165, idle 4 %, tail 29 us (its last group is 4 m-tiles = 1024 rows).
- Per-SM write port: 63 GB/s (32 B/clk) for STG.128 and for bulk S2G alike; a UB row writes 48 KB, so the floor is
  ~0.94 us/row at the fused clock against 1.93 measured.
- Cheaper row math does not pay: epilogue row sums (BnPlan c1-c6) cut the boundary to 1.73 us/row but cost each UB
  GEMM tile 1.2-1.5 us; async bulk stores (c4) hold the stage longer; L2 prefetch of later rows costs +100 us (UB) /
  +53 us (D6). The tail was the only clean lever.

## 3. Diagnosis and projection

- Dense GEMMs: the heuristic list misses better nvjet configs at most shapes, and the workspace configs add a memset node
  that costs more in the graph than standalone. Projection from the lt replica of `lt_v2.py` alone (panel 11,22, all 8
  cases, `evidence/lt_replica_exp4.out`): **x1.0049 / x1.0046, mean +0.48 %**.
- 16K boundaries: the tail after the last GEMM tile is idle SM time. Splitting the last m-tiles into 4-row items moves
  it into the schedule. Projection (order-averaged replica): **+0.39 % first-turn load, ~+0.13 % of score**.
- Together (inferred, additive): about **+0.6 %**.

## 4. Change

`s1.py` = `b23_compact.py` + `lt_v2` + `bnd_tail` (`R/NOTES.md:64-65`). `change.diff` is against b23; because s1 was built
on `R/b23_compact.py` (b23 with its two cached-EXPQ attention sources re-expressed as line edits of `ATT_CUDA_SRC` by
`R/compact.py`, verified byte-identical at import), the real edits are in `evidence/change_readable.diff` (11.7 KB).

1. **`LT_X_ALGOS` + `DenseLtX`** (`R/lt/lt_tail.py`): an explicit config table per slot (`evidence/lt_table_v2.txt`),
   rebuilt in the loaded cuBLASLt, AlgoCheck-validated, with `workspaceSize=0` wherever the config allows it. Plans are
   keyed by exact token count (753, 3012, 16384); nearby shapes keep b23's paths. If AlgoCheck rejects a config (a
   different cuBLASLt), that slot keeps b23's plan.
2. **3012 o_proj moves from sk2 to cuBLASLt** (ws0 128x168); the 753 o_proj keeps sk2.
3. `nn` configs keep a transposed weight copy (pool+router W^T 75 MB for the 3012 fused GEMM, unpooler W^T 67 MB at
   753). Guard added at stacking: an `nn` plan refuses a weight other than the one it transposed (`R/NOTES.md:64`).
4. **`bnd_tail`** (`R/bnd/bnd_tail.py`, built by `make_tail.py`): UB keeps its knobs (G4 R16 tail 2.1.1 bcap16) and runs
   its last m-tile as 4-row items (R4K1); D6 goes to G=2, tail (2,1,1) and its last 2 m-tiles as 4-row items (R4K2). Row
   math untouched: kernel outputs bit-identical to b23 (probe rel 0.0).

How the dense picks converged (`R/lt/NOTES.md`):

| step | result | lesson |
|---|---|---|
| exp1: 753/3012 configs from the heuristic-free search, no ws0 dimension | +0.38..0.48 % turn1/turn4 | the space beats the heuristic list |
| ws0 dimension added | 3012 ws=4 configs lose 0.2-0.5 % in-layer; ws0 variants win | memset nodes cost more in a graph than standalone |
| lt_v2o (+ 753 o_proj on cuBLASLt) | 0.9987 / 1.0027 | kernel time is not the metric when PDL hides a launch |
| lt_v2 | 1.0049 / 1.0046 | kept |

## 5. Result

| check | result |
|---|---|
| fixed replica, s1 vs b23_compact, panel 11/22/33 x 8 reps, all 8 cases (`evidence/replica_s1_vs_b23compact_run_s1.log`) | base x1.9592 (1.9629 / 1.9571 / 1.9575), s1 x1.9680 (1.9685 / 1.9636 / 1.9718): **RELATIVE 1.0029 / 1.0033 / 1.0073, mean +0.45 %** |
| per load (same run) | turn 1 +0.29 / +0.58 / +0.44 %; turn 4 +0.60 / +0.77 / +0.93 %; first turn -0.04 / -0.34 / +0.83 % (16K replica noise) |
| correctness | gate (`R/gate.sh`: 5 judge tests + 8 scored + nearby shapes, 2 replays each) CHECK_RC=0 PASS=42 CASES=21; replica reference checks pass |
| judge | check c6fe3962 **1.9831** (CI 1.953-2.013), scored a7ea31e5 **1.9901** (CI 1.960-2.020); no flags |

Per-kernel, judge profiles (b23 fa9e150b -> s1 a7ea31e5): case 0 fused 80.5 -> 77.1 us (now `tss_144x192`), unpooler
65.2 -> 64.2 (`tst_192x128 NNN`); case 2 o_proj sk2 69.6 -> nvjet 64.1, fused 282.4 -> 278.8, unpooler 249.8 -> 247.2;
case 5 pooler 1356.9 -> 1344.1, D6 582.8 -> 563.3, UB 1683.3 -> 1679.4, **QKV 753.4 -> 773.7 (+20 us)**.

**Reading the judge honestly.**
- The judge says s1 (mean 1.9866) is *below* b23 (1.9955). It is not. b23's single reading drew a slow SGLang on case 0:
  1170.3 us against 1082-1098 in s1's two runs, which lifts b23's case-0 ratio to 2.076 and its turn-1 load to 2.0468
  (s1: 2.0079 / 2.0333). Load speedups recomputed from the per-case tables.
- Calibration: judge / replica is ~+0.95 % for s1 (1.9866 / 1.9680) and ~+1.85 % for b23 (1.9955 / 1.9592). So b23's
  reading was about +0.9 % favourable relative to s1's (inferred), and b23's real level is ~1.96 (`R/NOTES.md:68-76`).
- The MoE is identical code in both builds, yet its case-0 kernel reads 284.5 us in b23's run and 244.7 us in s1's: the
  judge's fresh router changes the expert work per run, and that alone moves case 0 by tens of us.
- The real effect is the replica's **+0.45 %**, below the +0.6 % projection. The gap is the 16K first-turn load, which
  reads -0.34..+0.83 % per draw. One measurable loss is the 16K QKV pick: tuned in a hot, back-to-back state, it ran
  20 us slower than b23's `256x128` in the judge's cooler state (case 5 profile above).

## 6. Takeaways, and what came next

- **Measure what the judge measures.** Paired ratios against SGLang over a panel of judge-style router draws, each
  candidate in its own AB/BA pairs. Every later milestone was accepted on this replica, never on absolute latency or
  on one router seed.
- **In a CUDA graph, a GEMM's workspace is a memset node.** `workspaceSize=0` both removes the node and selects nvjet's
  memset-free scheduler.
- **Kernel time is not the metric** when PDL hides a launch (the 753 o_proj stays on sk2).
- **Thermal state picks tiles.** The 16K picks made in a hot sweep lost in the judge's cool state.
- Next, milestone 09: with the dense and boundary edges tuned, the remaining case-0 gaps were the attention critical
  path and the 16K expert kernel.

## Reproduce

```bash
cd ../tools
python research/harness/check.py ../08_exhaustive_cublaslt_configs/build.py --edges        # judge tests + 8 scored + nearby, 2 replays each
python research/judge/replica.py --candidate b23=../07_exact_micro_gain_bundles/build.py \
       --candidate s1=../08_exhaustive_cublaslt_configs/build.py --panel 11,22,33 --reps 8   # judge-faithful paired A/B, 3 router draws
python research/judge/replica.py --candidate s1=../08_exhaustive_cublaslt_configs/build.py --panel 11 --reps 8 --cases 0,1,2,3
```

The cuBLASLt search (`R/lt/lt_prefill.py`, `R/lt/inlayer_sweep.py`) and the boundary probes (`R/bnd/probe.py`) need the
lane's environment; their outputs are in `evidence/lt_standalone_results.json` and `notes/bnd_NOTES.md`.

Originals: `muse-opt-codex/muse_prefill_layer/research/claude/stack/s1.py` (= `submits/claude/s1.py`), lanes
`research/claude/{lt,bnd}/`, scorer `research/judge/`. Judge results: `submits/claude/profiles/s1.*.txt`.
