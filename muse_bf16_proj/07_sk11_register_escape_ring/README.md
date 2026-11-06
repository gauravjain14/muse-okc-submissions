# 07: sk11, escape lists in registers in the m = 8-32 ring (plus router, m128 pooler, m256 qkv, o m1)

| | before (06) | after (07) |
|---|---|---|
| build | `06_.../build.py` = `bo1` (md5 a5cf713f, 450,150 B) | `build.py` = `bo7b` (md5 d4f227ba, 729,324 B) |
| judge | 1.5688 (scored, job 1b299d12) | **1.5910** (check, job d34e34cf). Intermediate checks: `bo5` 1.5763 (job 4aeb8901), `bo6` 1.5757 (job 5ecc4fe7) |
| real effect | | between **+0.7 %** (both node drifts removed) and **+1.4 %** (raw); local projection +1.9 %. Real level 1.5910 (single reading) |
| shapes changed | | 14: qkv_gate m8/16/32, pooler m32 (sk11); o m16, unpooler m16/32 (sk3); routers m8/16/32/128 (rt3); pooler m128 (g128); qkv_gate m256 (g256); o m1 (tensor-core tile GEMV) |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: code copied or adapted from other projects is marked in place with `SPDX-SnippetBegin` / `SPDX-SnippetEnd` comments (source, copyright, license, our changes); see `NOTICE` and `LICENSES/` at the repository root. The code is unchanged. The judged file has md5 `d4f227ba`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

Six lanes ran on top of `bo1` and stacked in three judged builds. The largest piece, and the one this milestone is named after, is agent F's `sk11`: the vjeux/alygg77 `sk2` ring kernel with a new escape path, where each escape warp keeps its tile's rare-escape list in registers and only votes per stage.

"ROOFLINE line N" means line N of `muse-opt-codex/muse_bf16_proj/claude/t3/ROOFLINE.md`, the coordinator's running log (copied unedited as (private notes, not included)). Lane notes are in `notes/<lane>_NOTES.md`.

## 1. Starting point

`bo1`'s m = 8-32 shapes ran the public `sk2` ring (or Nathen's small-M routes), the routers ran alygg77's `rr_small`/`sk2`/`rk`. Judge times (`evidence/judge_before_bo1_scored_1b299d12.txt`):

| shape | kernel | judge µs | SGLang µs |
|---|---|---|---|
| qkv m8 / m16 / m32 | sk2 | 9.03 / 9.86 / 10.50 | 13.82 / 14.11 / 14.03 |
| pooler m32 | sk2 | 14.91 | 21.92 |
| unpooler m16 / m32 | sk2 | 13.98 / 14.82 | 18.91 / 19.14 |
| o m1 / o m16 | cgemv / sk2 | 5.04 / 5.46 | 6.85 / 6.82 |
| router m8 / m16 / m32 / m128 | rr_small / rr_small / sk2 / rk | 4.45 / 4.61 / 5.48 / 5.79 | 26.48 / 14.93 / 15.55 / 16.16 |
| pooler m128 | cuBLAS nvjet | 20.36 | 25.31 |
| qkv m256 | cuBLAS nvjet | 17.07 | 19.57 |

Value of 1 µs at this level, 100 / (6 × load sum) (computed from the same file): decode 1 0.49 %, verify 1 0.46 %, decode 16 0.35 %, decode 32 0.32 %, verify 16 0.26 %, verify 32 0.21 %.

## 2. What we saw

**(a) The rings were issue/latency-bound around an idle tensor pipe.** `../06_public_per_shape_best_of/evidence/dash_bo1.md`: every sk2 kernel at m = 8-32 sat +1.7 to +3.5 µs over its stream floor, tensor pipe 9-28 %, issue 26-37 %. The SASS (`../06_public_per_shape_best_of/evidence/sass_bo1.txt`) shows hot loops dominated by decode integer ops and LDS with only 8-64 HMMA.

**(b) What the time above the floor is made of** (agent X, NVRTC port `sk3` of sk2; `notes/X_NOTES.md` iteration 0). Ablations on pooler m32, same run, µs from the first CTA start:

| variant | total | first stage lands | loop end med / max |
|---|---|---|---|
| full | 13.76 | 1.50 | 12.74 / 13.41 |
| no x | 13.52 | 1.06 | 12.35 / 12.96 |
| no compute | 13.25 | 1.31 | 11.90 / 12.26 |
| no x, no compute (pure W ring) | 12.83 | 1.02 | 11.33 / 12.16 |

- The pure W ring moves 50.3 MB from the first landing (~1.0 µs) to the last loop end (~12.0 µs): 4.6 TB/s, 96 % of 4.8. DRAM in the bulk is not the problem.
- x costs only 0.24 µs and compute 0.5 µs.
- Escape warps: switching their work off saves **~0.85 µs** (12.99 vs 13.84, X iteration 2).
- The rest is startup (1.0-1.5 µs), end skew (0.5-0.9 µs) and the last stage plus epilogue (~0.6 µs).

**(c) The escape warps tax the consumer that shares their SM sub-partition.** Agent T's per-warp trace of pooler m32 (`notes/T_NOTES.md` iteration 2; scripts `evidence/T_tr.py`, `evidence/T_probe.py`):
- Consumer warp 0 finishes each stage 1.05 µs after it lands, against 0.63-0.79 for warps 1-3. Warp 0 shares SMSP 0 with the producer and an escape warp.
- Moving one escape warp so that a consumer shares its SMSP with two escape warps: pooler32 16.53 vs 13.66, qkv32 11.10 vs 9.92.
- Escape warps "wake, scan and arrive on EVERY stage". Each slot cycle (land → last release → producer wake → refill issued) takes ~2.1 µs for 5 slots, the measured 0.42 µs per stage.
- T's microbenchmarks: decode costs ~160 cycles per 64-k unit (half-rate INT ops). HMMA loops run at 1.67-1.72 GHz under DRAM load vs 1.90 without. pooler32 / qkv32 / qkv16 are consumer-bound in steady state.

**(d) Moving escapes into the consumers is worse.** X: inline escapes in the consumer warps 16.42 vs 13.63 µs (pooler32); ~150-300 cycles per escape event on the consumer's critical path. Agent F, trying the 11-bit format in the same ring: correcting 1.36 % escapes cost +2 to +19 µs with every mechanism (`notes/F_NOTES.md` iterations 1-4). So the escape work had to get *cheaper*, not move.

**(e) The other shapes each had a measured, local lever:**
- **Routers** (agent Y, `notes/Y_NOTES.md`, `evidence/Y_bench_rt3_v6.log`, `evidence/Y_iter8_prof.log`). The routers are an HBM stream (8.4 MB issued at t = 0, landing by ~2.6-2.8 µs) plus a 0.5-1 µs MMA/sum/red tail.
  - A helper warp's `red.release.gpu` on an SM with an LDG flood in flight completes only when the flood drains (~2.3 µs).
  - Four dedicated zeroing CTAs on idle SMs remove that from the critical path.
- **pooler m128** (agent Z, `notes/Z_NOTES.md` iteration 8): the W stream binds. W alone 13.3 µs for 50.3 MB = 3.8 TB/s; compute alone runs at 89 % of tensor peak; x adds only 0.26 µs.
- **qkv m256** (agent V, `notes/V_NOTES.md`):
  - cuBLAS's qkv tile uses only 96 of 132 SMs.
  - A non-cluster launch puts 16 consecutive blockIdx on one GPC, so per-GPC L2 delivery binds (`evidence/V_smid.py`).
  - The tensor pipe runs at 1.68-1.72 GHz under load, ~900 TF (`evidence/V_tcprobe.py`).
- **o m1** (agent G): a one-wave tensor-core tile GEMV beat the public `cgemv` (ROOFLINE line 173).

## 3. Diagnosis and projection

- **Ring shapes:** the consumer warps' per-stage critical path and their co-tenants (producer, escape warps) bind, not DRAM or the tensor pipe. The cheapest escape path is one that touches nothing on the slot-release path unless the stage actually holds an escape. Escapes are rare: 7e-5 of weights in F12, ~5-25 per 8-row tile per CTA (F_NOTES formats section).
- **Projection.** Each lane's same-run local delta, times the value per µs above (local deltas carry over because judge ≈ local + a constant per shape), all computed:

| piece | local delta (same run) | judge geomean, projected |
|---|---|---|
| G o m1 tc tile GEMV | 3.90 vs 4.14 µs | +0.12 % |
| V qkv m256 g256 | 15.45-15.83 vs 16.36-16.72 (V final table) | +0.19 % |
| Z pooler m128 g128 | -0.34 (in bo4) to -0.88 (Z's own runs) | +0.09..0.23 % |
| X sk3 configs + float4 corrections | decode32 -0.72, decode16 -0.29, verify1 -0.19 µs (X iteration 6) | +0.42 % |
| Y rt3 routers | m8 -0.12, m16 -0.40, m32 -0.74, m128 ~0 (ROOFLINE line 187) | +0.43 % |
| F sk11 | qkv32 -1.03, pooler32 -0.56, qkv16 -0.16, qkv8 -0.13 (bo7b vs bo6, ROOFLINE lines 206-208) | +0.63 % |
| **total** | | **≈ +1.9 %** (1.5688 → ~1.60) |

## 4. Change

`sk11` (`F/sk11.py`, module prefix `_f1_`) is X's `sk3` (vjeux's `sk2` ported to NVRTC) on agent G's F12 weight format (12.00 bits/weight, escapes 7e-5), with a new escape path:
1. **Escape lists.** Per (CTA, 8-row tile), a list of up to 64 entries `k | row << 13 | fake << 16` (~5-25 per tile), loaded **once** at kernel start into 1-2 registers per lane of the tile's escape warp.
2. **Per stage**, the escape warp only does `ballot(stage(entry) == st)` and arrives. No shared-memory access on the slot-release path unless that tile has an escape in that stage. Then lane = token: one `LDS.U16` of x and 8 predicated FFMAs into two correction registers.
   - `E1 = Σ(-d·x)` cancels the decoded stand-in value `d`; `E2 = Σ(true·x)` adds the true weight.
3. **Output** = `((Σ k-warp partials) + E1) + E2`, times 2^(E_lo - 112). One-hot rows come out exact.
4. **What it replaces.** sk2/sk3 scan a shared-memory escape table every stage in 4 escape warps and keep 4 correction copies.

Configs (R, NT, NWR, WSK, UPW, NACC, ring KB, flags) per shape are in F_NOTES iteration 5. In `bo7b` sk11 owns qkv m2-32 and pooler m17-32. F's other measured wins (pooler m16, unpooler m16, o m16/32) did not reproduce in the integrated runs (ROOFLINE line 208), so sk3 stays on o m16 and unpooler m16/32. The public sk2 stays on o m32, and Nathen's kernels on pooler m8/m16 and o m8.

The other pieces, in build order (dispatch at the end of `build.py`):

| build | adds | same-run local evidence (ROOFLINE lines 173-208) | judge |
|---|---|---|---|
| bo2 | G's `g1_o_tc` for o m1 (one-wave 16-row F12 tile GEMV) | 3.90 vs 4.14 µs | (not judged) |
| bo3 | V `g256` qkv m256: BF16 SS-wgmma 128 tok × 72 cols, 2-way asymmetric split-K through global memory (helpers publish, finisher's idle producer thread pulls the partial into free ring slots) | 16.28 vs 16.90 | |
| bo4 | Z `g128` pooler m128: 12-bit RS-wgmma (agent M's loop), 4 K-quarters as 2 cluster pairs (DSMEM inside the pair, global across) | 18.40 vs 18.74 | |
| **bo5** | X `sk3` (sk2 configs + float4 escape corrections, baked W pointer on unpooler16/32 and o16) | o16 4.37 vs 4.57, unp32 13.48 vs 13.65, qkv8 8.00 vs 8.14 | **1.5763** |
| **bo6** | Y `rt3` routers (`ys` m ≤ 8, `ya` + 4 zeroing CTAs m 9-32, `rk` K-split 16 at m 97-128) + V's final g256 (ring depth 9) | router m8 2.95 vs 3.07, m16 3.12 vs 3.52, m32 3.46 vs 4.20 | **1.5757** |
| **bo7b** | F `sk11` on qkv m2-32 + pooler m17-32 | qkv32 8.86 vs 9.89, pool32 13.39 vs 13.95, qkv16 8.39 vs 8.55, qkv8 8.25 vs 8.38 | **1.5910** |

How sk11 converged (`notes/F_NOTES.md`; harness `evidence/F_tf.py`, bit-exact decode emulation `evidence/F_loccheck.py`):

| step | pooler m32 (local µs) | what it taught |
|---|---|---|
| F11 (11.45-11.53 b/w) + tensor-core escape warps | 20.53 (escapes idle: 12.82; sk3 13.86) | 1.36 % escapes; the escape warp lags ~2 stages and the producer waits on its releases |
| gathers for all groups first, then MMA | 23.74 | ncu: +4.26 M instructions (~10 per escape) |
| escape tables + prefetch | 31.5 | an escape warp sharing an SMSP gets IPC ~0.3; any LDS round trip ~100 cycles under load |
| consumer-side exponent patch (probe) | 15.42 vs 13.26 | consumers pay ~1:1 for any per-stage work: F11 dead in this ring |
| **F12 + register-preloaded escape lists (sk11)** | 13.16-13.36 vs sk3 13.87-14.04 | ncu issue 29 % vs 36 % (pooler32), 26 % vs 36 % (qkv32): the per-stage scans are gone |

## 5. Result

| check | result |
|---|---|
| local A/B, sk11 vs bo5's kernels (two sandboxes, 30 reps; F_NOTES iteration 5) | qkv m32 8.89 / 8.85 vs 9.81 / 10.09; pooler m32 13.36 / 13.16 vs 13.87 / 14.04; qkv m16 8.31 / 8.23 vs 8.71 / 8.60; qkv m8 7.97 / 8.03 vs 8.24 / 8.17 |
| integrated A/B (bo5 + sk11 vs bo5, same run) | decode 16 x1.661 → x1.691, decode 32 x1.544 → x1.624, verify 1 x2.240 → x2.281 |
| nsys in-graph, sk11 vs bo5's kernel (F's candidate) | qkv m32 9.39 vs 10.59, qkv m16 9.10 vs 9.41, pooler m32 14.51 vs 14.58 µs |
| correctness | `check.py --graph` 259 cases × 2 seeds ALL OK for bo5, bo6 and bo7b (ROOFLINE lines 177, 188, 208). sk11 alone: `exact_all --nesc 200` and `--nesc 3000` EXACT ALL OK. Y: `exact_router --nesc 200` EXACT ROUTER ALL OK |
| judge | 1.5688 (bo1, scored) → 1.5763 (bo5) → 1.5757 (bo6) → **1.5910** (bo7b, check). Per load bo1 → bo7b: decode 16 1.595 → 1.625, decode 32 1.505 → 1.573, verify 1 2.077 → 2.113, verify 32 1.259 → 1.264, decode 1 1.792 → 1.795, verify 16 1.325 → 1.325 |
| judge per changed shape, bo1 → bo7b (µs) | qkv m8 9.03 → 8.59, m16 9.86 → 9.28, m32 10.50 → 9.89; pooler m32 14.91 → 14.50; router m16 4.61 → 4.01, m32 5.48 → 4.46; o m1 5.04 → 4.74; pooler m128 20.36 → 19.67; qkv m256 17.07 → 17.19 |
| **judge, read honestly** | See below the table. |
| vs projection | +1.9 % projected; +0.7 % (node-corrected) to +1.4 % (raw) read. The judge cannot separate the six pieces at this size. The clearest signal is the decode-32 load, where X, Y and sk11 all landed: the judge load rose 1.505 → 1.573 (+4.5 %, raw) against +6.3 % from summing the three lanes' local deltas (X -0.72, Y -0.74, sk11 -1.59 µs on a 51.65 µs load; computed). qkv m256's local -0.9 µs is invisible on the judge, because the same `g256` kernel read 16.05 and 17.19 µs in two runs |

How the judge readings should be read (`evidence/judge_compare.txt`):
- **bo5 → bo6 (identical-code shapes moved).** The 25 shapes whose kernel did not change got **1.55 % slower**, e.g. o m32 (same sk2 kernel) 5.84 → 6.45 µs and o m16 (same sk3) 5.28 → 5.80. So bo6's flat 1.5757 hides the router gain: the router shapes read 4.84 → 4.38 (m16) and 5.48 → 4.94 (m32).
- **bo6 → bo7b (same in reverse).** The 26 unchanged shapes got **1.67 % faster** (router m32, same `ya` kernel, 4.94 → 4.46). qkv m256, same `g256` kernel, read 16.05 in bo6 and 17.19 in bo7b.
- **Correction.** Removing both drifts (our same-kernel shapes, and SGLang's) re-reads bo7b on bo1's node at **1.5804**, i.e. +0.7 %.
- **Uncertainty.** The drift estimate is itself uncertain by roughly ±0.6-0.8 %: the per-shape sd of 3.1 % on identical code (milestone 08), averaged over 15-26 shapes (computed).
- **Reading.** The real step was about +1 %. The raw +1.4 % includes a fast node.

What remained, from `evidence/dash_bo7b.md` (nsys + ncu on bo7b):
- The sum over floors fell 76.7 → 74.5 µs.
- m = 128/256 shapes stood out: qkv m128 `qb_kernel` +4.0 µs, pooler m128 +6.7, unpooler m128 +5.9, qkv m256 +5.8, o m256 +3.2, pooler/unpooler m256 +4.4-4.5 µs over floor.

## 6. Takeaways, and what came next

- **In a TMA ring with `mma.sync` consumers, the binding resource is the consumer warp's per-stage critical path and whoever shares its SMSP.** Any work done every stage costs ~1:1, whatever it is. Rare events have to be found by a register vote, not a shared-memory scan.
- **The 11-bit format does not pay inside this ring** with any escape mechanism measured: free escapes would have saved 0.8-1.0 µs on pooler m32, but the cheapest correction cost +2.2 µs. The byte cut needs a ~free escape path.
- **Dead ends logged here** (detail in the notes):
  - T: last-arriver refill, stage rotation, per-warp private pipelines, owner-only escape warps (T_NOTES iterations 3-6).
  - U: no m128 win. It found the `setmaxnreg` hazard: dealloc'd warps doing long-latency global loads plus shared stores next to RS-wgmma consumers corrupt results (rel. err 2-6e-2). It also found that the rare-escape branch in the RS loop costs 0.35-1.0 µs (`notes/U_NOTES.md`).
  - X: TMA multicast of x and split-K clusters both lose (coupling costs more than the ingress saved).
- **Judge reading at this size.** Single checks move about ±1 % on identical code (ROOFLINE line 191), up to 1.55 % (milestone 08). Decisions below ~1 % had to come from local same-run A/Bs; checks were kept for large steps.
- **Next ([08](../08_cluster_pair_splitk_wgmma/)):** the m = 128 and m = 256 shapes were now the largest gaps over floor (+3 to +7 µs each). The next lanes replaced them with custom BF16 wgmma GEMMs: K at m = 256, Q2 at m = 128 with a cluster-pair split-K exchange. N took o at m = 1/8 and H unpooler at m = 32.

## Reproduce

Run from `tools/` (torch and an H200):

```bash
cd muse_bf16_proj/tools
python3 bench.py --cand base=../06_public_per_shape_best_of/build.py \
                 --cand new=../07_sk11_register_escape_ring/build.py --shapes 8,16,32 --reps 40 --xclones 4
python3 bench.py --cand base=../06_public_per_shape_best_of/build.py \
                 --cand new=../07_sk11_register_escape_ring/build.py --layers router --reps 60
python3 check.py ../07_sk11_register_escape_ring/build.py --graph
python3 exact_all.py ../07_sk11_register_escape_ring/build.py --nesc 200 --ms 8,16,32
python3 exact_router.py ../07_sk11_register_escape_ring/build.py --nesc 200
python3 ../07_sk11_register_escape_ring/evidence/judge_compare.py ../07_sk11_register_escape_ring/evidence/judge_before_bo1_scored_1b299d12.txt \
        ../07_sk11_register_escape_ring/evidence/judge_bo5_check_4aeb8901.txt ../07_sk11_register_escape_ring/evidence/judge_bo6_check_5ecc4fe7.txt \
        ../07_sk11_register_escape_ring/evidence/judge_after_bo7b_check_d34e34cf.txt
```

The lane harnesses in `evidence/` (`X_tx.py`, `F_tf.py`, `F_loccheck.py`, `T_probe.py`, `T_tr.py`, `V_tcprobe.py`, `V_smid.py`) import their lane modules (`sk3.py`, `sk11.py`, `skt.py`, `cu.py`/`k3.py`); run them from their original directories.

Originals (relative to ``):
- builds: `muse-opt-codex/muse_bf16_proj/claude/t3/{bo2,bo3,bo4,bo5,bo6,bo7b}.py`;
- lanes: `claude/t3/{X,Y,Z,V,T,U,F,G}/`;
- dashboard: `claude/t3/dash_bo7b*.{md,log}`;
- judge results: `claude/t3/profiles/{bo1.1b299d120f5d4dfe,bo5.4aeb890144e84940,bo6.5ecc4fe7149a413a,bo7b.d34e34cf321c417a}.txt`.

Third-party code (marked in place in `build.py`; see the root `NOTICE`): `build.py` still carries the public base (alygg77 / vjeux / Nathen Fernandes / Matsvei / CUTLASS) and its 10 encoded literals (see milestone 06).
