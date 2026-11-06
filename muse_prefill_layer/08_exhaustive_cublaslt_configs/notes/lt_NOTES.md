# Dense GEMM cuBLASLt exhaustive config search (muse_prefill_layer), 2026-10-30

Worker: dense/cuBLASLt. Base: research/claude/b23_compact.py. Files here:
- `lt_prefill.py`: the tuner (C++ via load_inline). Enumerates algo id 66 (nvjet, the only sm90 family that matters; ids
  21/23/24/... are sm80 mma.sync kernels and ignore cluster) x 634 tile ids x stages (only 64xAUTO for nvjet) x
  cluster 2..10 (1x1..4x4; 0/1 and >10 invalid for nvjet) x custom option 0..7 x swizzle x split-K (<=4 at 753, <=2
  at 3012, 1 at 16K; FP32-accurate reductions only) x layout (tn: W [n,k]; nn: W^T [k,n] copy) x **workspace mode**
  (split-1 configs that ask for a workspace are also run with workspaceSize=0, see below), each AlgoCheck-validated.
  Phase A: one async cold (256 MB read) rep of every config after an untimed warm launch (cuBLAS lazy-loads each
  kernel on first launch: without the warm launch every split-1 config looked +60 us slow); shuffled order.
  Phase B: top 300, correctness (rel L2 vs FP32 ref < 2e-2) + 3 reps with the judge's 1 GiB flush. Phase C: top 40
  as CUDA graphs, 5 reps. Phase D: top 8 + the incumbent's exact call (b23's own DenseLt/LtMatmul/sk2/sglang path),
  30 alternating reps, median + paired ratio (judge-style), kernel names + grids from a profiler trace.
- `lt_enums.json`: tile / cluster / stage enum names from the sandbox's cublasLt.h (cuBLASLt 13.1.1, version 130101).
- `inlayer_sweep.py`: one prepared candidate state, one GEMM slot swapped per config, the whole layer re-captured and
  timed judge-style alternating with the unmodified layer graph; `--kprof` adds per-GEMM in-layer kernel durations
  (CUPTI trace of the same replays). This is the arbiter: standalone wins do not always survive in the layer.
- `lt_tail.py`: `DenseLtX` (explicit config rebuilt in the loaded library + AlgoCheck; tn/nn; optional ws0) and a
  prepare() that installs `LT_X_ALGOS` into b23's tables (falls back to b23's plan if AlgoCheck fails).
- `build.py`: single-file candidate = b23_compact.py + lt_tail.py with a table. `check_plans.py`: installed plans +
  in-layer GEMM kernel names per case. `summarize.py` -> `results.json`.

## Findings
1. The incumbent's DenseLt.find() picks only from the heuristic list; the exhaustive space has better configs for
   most shapes, mostly non-default custom options / cluster shapes of the same or neighbouring tiles.
2. Workspace = a memset node. Many nvjet configs report a 4-byte workspace in AlgoCheck; with a workspace cuBLAS
   adds a Memset node before the GEMM. In the layer that node costs far more than standalone (in-layer sweep at
   3012: every ws=4 config, even the incumbent's own algo rebuilt with a workspace, lost 0.2-0.5% of the layer).
   Passing workspaceSize=0 for the same algo works and selects nvjet's memset-free `ssched` kernel variant (the same
   one b23's zero-workspace heuristic plans run), often faster as well (f3012 128x168: 283.0 vs 289.5 us).
3. o_proj at 753: the best nvjet ws0 config (128x96 / 128x128 ssched) is ~1.7 us faster per kernel than sk2, but the
   layer gain is mixed (case 0 x1.003, case 1 x0.998; 8-case replica no better): sk2 is launched with PDL behind the
   attention kernel's early trigger, which hides its launch/fill. Keep sk2 at 753. At 3012 cuBLASLt ws0 128x168
   (64.3 vs 69.6 us standalone) wins in the layer too (x1.0035 / x1.0014) and replaces sk2.
4. 16K standalone numbers drift with thermal state (incumbent q16k: 782 us eager at the start of a sweep, 910 us in
   phase D after ~2 min of back-to-back GEMMs; in-layer trace 741 us). Rankings change with it: hot -> 128x144..176
   tiles win, cool -> 256-wide tiles. Only in-layer comparisons count there, and those are noisy (2-3% per case).


## Standalone results (cold-L2 CUDA-graph replay, 30 alternating reps vs the incumbent's exact call; results.json)
Times in us incl. graph replay cost minus the fixed cost. ws0 = run with workspaceSize=0 (memset-free ssched kernel).
Settings = [algo id, tile id, splitK, reduction, swizzle, custom, stages, inner, cluster]; tile/cluster names in
lt_enums.json (cluster 2=1x1, 3=2x1, 5=1x2, 6=2x2, 8=1x4).

| shape | M,N,K | incumbent (b23) kernel | inc | best config | best | paired x |
|---|---|---|---|---|---|---|
| q753 | 753,2304,8192 | tst_112x128_64x7_1x2_h (sglang dispatch) | 38.48 | tn [66,165,1,0,0,6,35,0,5] 112x128 | 38.46 | 1.003 |
| q753c | same, SM cap 112 | tst_128x128_64x6_2x2_h | 41.50 | nn [66,20,1,0,0,6,35,0,5] 128x128 1x2 | 41.14 | 1.010 |
| o753 | 753,8192,1024 | sk2 (custom) | 20.43 | nn [66,20,1,0,0,0,35,0,2] ws0 128x128 ssched | 18.61 | 1.104 |
| f753 | 753,4608,8192 FP32 | tss_256x112_64x4_2x1_v_coopA | 76.78 | tn [66,185,1,0,0,4,35,0,3] 144x192 | 74.16 | 1.034 |
| u753 | 753,8192,4096 | tst_128x192_64x5_2x1_v_coopB (sglang) | 66.62 | nn [66,20,1,0,0,1,35,0,2] ws0 128x128 ssched | 64.77 | 1.031 |
| q3012 | 3012,2304,8192 | tst_256x104_64x4_1x1_h_coopA | 144.91 | tn [66,496,1,0,0,5,35,0,2] 256x104 v | 143.38 | 1.011 |
| q3012c | same, SM cap 108 | tst_128x256_64x4_2x2_h_coopA | 163.82 | nn [66,408,1,0,0,3,35,0,6] ws0 128x168 2x2 | 161.94 | 1.013 |
| o3012 | 3012,8192,1024 | sk2 (custom) | 69.58 | tn [66,408,1,0,0,1,35,0,3] ws0 128x168 ssched | 64.34 | 1.081 |
| f3012 | 3012,4608,8192 FP32 | tss_256x144_64x4_1x2_h_ssched_coopA | 295.66 | nn [66,420,1,0,0,5,35,0,3] ws0 128x280 | 290.85 | 1.018 |
| u3012 | 3012,8192,4096 | tst_128x304_64x3_2x1_v_ssched_coopA | 257.01 | tn [66,408,1,0,0,1,35,0,3] ws0 128x168 | 253.81 | 1.012 |
| q16k | 16384,2304,8192 | memset + tst_256x128_64x4_1x2_h_coopA | 850.16 | nn [66,409,1,0,0,3,35,0,2] ws0 128x176 | 833.33 | 1.021 |
| p16k | 16384,4096,8192 | memset + tst_128x256_64x4_2x1_v_coopA | 1614.0 | tn [66,29,1,0,0,1,35,0,3] ws0 128x160 | 1548.5 | 1.041 |
| r16k | 16384,512,8192 FP32 | tss_256x128_64x4_1x2_h_coopA | 179.90 | nn [66,408,1,0,0,3,35,0,2] ws0 128x168 | 176.40 | 1.019 |

Configs screened per shape: 84k (753, split<=4), 42-63k (3012, split<=2, +ws0), 36k (16K, +ws0). Top-3 per shape with
kernel names/grids: results.json; raw tuner output: v1/r753.json, v1/r3012.json (first pass, no ws0 dimension),
r2_mid.json, r2_16k.json (with ws0), logs *.out. The 16K standalone numbers are thermally inflated (see 4.).

## In-layer (inlayer_sweep.py; layer paired ratio over 24 reps x 2 rounds + per-GEMM kernel delta from CUPTI)
| slot (case) | chosen config | in-layer GEMM delta | layer |
|---|---|---|---|
| 753 fused (0) | tn [66,185,1,0,0,4,35,0,2] 144x192 1x1 (128 CTAs) | -2.3..-2.4 us | x1.0043..1.0052 |
| 753 unpooler (0) | nn [66,31,1,0,0,4,35,0,5] 192x128 1x2 | -1.2..-1.5 us | x1.0029..1.0034 |
| 753 o (0 / 1) | not taken: nvjet ws0 128x96 -1.7 us/kernel, but layer x1.0033 (case 0) / x0.998 (case 1); sk2 keeps PDL overlap | | |
| 753 qkv capped (1) | not taken: +0.0..1.0 us, noise | | |
| 3012 qkv (2) | tn [66,496,1,0,0,5,35,0,2] | -2..-3 us | x1.0015 |
| 3012 qkv capped (3) | tn [66,408,1,0,0,1,35,0,6] ws0 128x168 2x2 (108 CTAs) | -3.1..-4.2 us | x1.001 |
| 3012 o (2 / 3) | tn [66,408,1,0,0,3,35,0,5] ws0 (replaces sk2) | (sk2 is PDL-launched; kernel time n/a) | x1.0035 / x1.0014 |
| 3012 fused (2) | nn [66,408,1,0,0,3,35,0,5] ws0 128x168 | -6.1..-6.3 us | x1.0043..1.0045 |
| 3012 unpooler (2) | tn [66,408,1,0,0,1,35,0,3] ws0 128x168 | -4.4..-6.6 us | x1.0011..1.0032 |
| 16K qkv (5) | tn [66,409,1,0,0,1,35,0,3] ws0 128x176 | -17.8..-18.5 us | x1.0027..1.0039 |
| 16K pooler (5) | tn [66,29,1,0,0,1,35,0,3] ws0 128x160 | -63..-65 us | x1.0070 (both rounds) |
| 16K router512 (5) | tn [66,408,1,0,0,3,35,0,5] ws0 128x168 | -6.0..-6.9 us | noise |
Rejected in-layer: every ws=4 (memset) config at 3012 (x0.995-0.998, incl. the incumbent's own algo with a workspace);
at 16K 256x168 / 256x144 ws0 pooler (-24 us, less than 128x160).

## Candidate: lt_v2.py (= b23_compact.py + lt_tail.py, 805,9xx bytes)
Judge replica (run_exp.sh, panel 11,22, all 8 cases, 6 reps; exp4.out/json):
- draw 11: b23 x1.9741, lt_v2 x1.9839 (turn1 2.0175->2.0306, turn4 1.9021->1.9151, first 2.0047->2.0078)
- draw 22: b23 x1.9481, lt_v2 x1.9571 (turn1 1.9700->1.9771, turn4 1.9003->1.9170, first 1.9749->1.9777)
- RELATIVE lt_v2/b23: 1.0049, 1.0046, mean **1.0048**. lt_v2o (+ o753 on cuBLASLt): 0.9987 / 1.0027 -> rejected.
- 16K load noise: lt_v2o is identical to lt_v2 at 16K yet its first-turn speedup differed by 1.7% / 0.8%; the 16K
  gain (~88 us/case by kernel deltas, ~1% of the load) is below what one replica run resolves.
- Earlier 753/3012-only variant (exp1, before the ws0 dimension): +0.38..0.48% on turn1/turn4.
Correctness gate (`bash research/claude/gate.sh research/claude/lt/lt_v2.py`): CHECK_RC=0 PASS=42 CASES=21 (5 judge
tests + scored + nearby shapes, 2 replays each; gate_v2.log). Replica reference checks also passed (REPLICA_RC=0).
Rebuild: `python3 research/claude/lt/build.py out.py "{$(cat research/claude/lt/table_v2.txt)}"`.

## Notes for integration
- Plans are keyed by exact token count (753, 3012, 16384); nearby shapes keep b23's paths.
- nn configs keep a transposed weight copy (pool+router W^T 75 MB for 3012 fused, unpooler W^T 67 MB for 753).
- If AlgoCheck rejects a config (different cuBLASLt), that slot keeps b23's plan; the 3012 sk2 route is dropped only
  after its cuBLASLt plan was built.
- UbPlan / D6Plan (16K o_proj / unpooler) untouched.
