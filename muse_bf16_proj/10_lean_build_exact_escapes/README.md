# 10: a lean, provably identical build, then six small-shape modules, every shape exact

| | before (09) | after (10) |
|---|---|---|
| build | `09_.../build.py` = `bo13r` (md5 964777bd, 810,525 B) | `build.py` = `bo19L` (md5 dbdbe35a, 408,026 B) |
| judge | check 1.6062 (job 710ae9d7), scored 1.6326 (job 64811543): mean 1.6194 | scored **1.6277** (job f90aa169), single reading |
| real effect | | local full bench, alternating processes: judge-projected **+1.4 %** (1.6101 / 1.6152 → 1.6335 / 1.6368). Judge: the 12 changed shapes **+0.87 %**, the 18 unchanged −0.21 % (vs both runs of 09); the coordinator's drift-corrected reading ≈ 1.655 |
| shapes changed | | qkv m1, pooler m1 (exact), unpooler m1/m8, router m1/m8/m16, pooler m8, o m16, o m32, o m128, unpooler m128 |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: code copied or adapted from other projects is marked in place with `SPDX-SnippetBegin` / `SPDX-SnippetEnd` comments (source, copyright, license, our changes); see `NOTICE` and `LICENSES/` at the repository root. The code is unchanged. The judged file has md5 `dbdbe35a`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

Sources. Paths are relative to ``. `ROOFLINE:n` is line n of `muse-opt-codex/muse_bf16_proj/claude/t3/ROOFLINE.md`. `<LANE>/NOTES.md` is `muse-opt-codex/muse_bf16_proj/claude/t3/<LANE>/NOTES.md`; copies are in `notes/`. Lane names (LN, E8, O2, S5, U3, Q5, O3, N2, K2, PW) are just labels of agent workstreams.

Terms:
- **F12 / F11**: lossless 12-bit / 11-bit weight formats (low BF16 byte + a 4-bit / 3-bit exponent code). Weights whose exponent falls outside the code's window are **escapes**, corrected from a side list. F11 has ~1.3 % escapes, F12 ~0.01 %.
- **Exact**: the rules require that a one-hot row of x reproduces a column of W bit for bit (`tools/TASK.md`). Our gate `exact_all.py` injects 200 or 3000 out-of-window weights (including tiny ones like −1e-30) and checks one-hot rows; `exact_router.py` does the same for the router's hi + lo split.
- **24-bit router format**: the FP32 router weight split into BF16 hi + lo (as SGLang does above 8 tokens), packed losslessly into 3 bytes per weight instead of 4.
- **RS-wgmma**: a Hopper wgmma whose A operand comes from registers, so weights can be decoded in registers straight into the MMA.
- **local** = `tools/bench.py` (CUDA-graph replay after a 512 MB cold-L2 read, paired with SGLang); judge ≈ local + 1.0 µs per shape.

## 1. Starting point

bo13r (milestone 09) was compliant but 811 KB, two of its m = 1 replacements had lost on the judge, and some kernels were not exact. Judge µs, mean of its two runs (`evidence/judge_before_*`):

| shape | kernel in bo13r | judge µs | problem |
|---|---|---|---|
| qkv m1 / pooler m1 | G's F11 `g311` GEMV | 8.16 / 12.91 | flushes injected weights < 2^-39 to 0 (75 / 82 of 200 strict one-hot columns, N2/NOTES.md) |
| unpooler m1 / m8 | G's slice kernels | 3.54 / 3.77 | inexact (50-75 of 200 columns, S5/NOTES.md line 101); m1 −6 % on the judge vs the public kernel it replaced |
| router m1 / m8 / m16 | `rs1_i` / `ys` / `ya` (FP32 or raw hi+lo, 8.4 MB) | 3.93 / 3.88 / 4.51 | tiny latency-bound kernels |
| pooler m8 | sk11 ring | 13.62 | |
| o m16 / m32 / m128 | sk3 / sk11 / Q2 `kg_o` | 5.45 / 5.89 / 6.63 | |
| unpooler m128 | Q2 `kg_unpooler` (BF16 SS-wgmma) | 18.19 | 68 MB of BF16 W |
| (size) | 810,525 B | | only ~190 KB left under the 1,000,000 B cap |

## 2. What we saw

**(a) The build had to shrink before anything else could go in.** Lane LN rebuilt every module as a lean, readable copy and proved it identical (LN/NOTES.md, `evidence/LN_prove_final.md`, `evidence/LN_sass_final.log`):
- 130 traced shapes (every scored m, the 9 judge tests, nearby m 2..257): same owner module and same launch geometry (kernel, grid, block, dynamic smem, cluster) everywhere.
- Outputs bit-identical, except the split-K FP32 `red.add` routers, which are not run-to-run deterministic even in the original (max |A−B|/max|A| 1.1e-7..1.9e-7).
- All 30 launched kernels: **identical SASS** (NVRTC 13 + cuobjdump).
- Same-run sum over the 30 scored shapes: bo14 263.56 | ref14 263.44 | lean 263.30 µs (`evidence/LN_bench_final_run1.log`).
- Size: 314,960 B vs 826,332 for bo14 (−62 %).

**(b) pooler m8: tensor cores cost no issue headroom; the problem is the stream shape** (E8/NOTES.md iterations 0 and 5):
- Budget per SM at the stream rate: 254 K weights in ~20-22 K cycles = 80-88 K warp-issue slots. A CUDA-core FFMA GEMV at m = 8 needs ~80 K warp instructions per SM (~95-100 % of all issue slots). mma.sync with W as the A operand needs ~10 K (≈ 12 %; ncu measured 11.9-14.6 %).
- Shipped `m8_pooler`, ncu: 15.26 µs (ncu-inflated) vs the previous kernel's 16.16 under the same tool; DRAM 50.54 MB at 72.6 % of peak, issue 11.9 %, HMMA 2.2 %, occupancy 50 %, stalls long_scoreboard 81 %. SASS: 56 registers, 0 spills, 49 LDG, decode 7 ALU per 8 weights.
- It drains DRAM at ~4.55 TB/s from ~0.6 to ~11.7 µs; what is left is a 0.26 µs epilogue and CTA end skew.

**(c) Tiny routers: 2 bytes of the 4 are compressible, and the tail is a reduction** (S5/NOTES.md; `evidence/S5_prof_v5.log`):
- The FP32 router weight as a BF16 hi + lo pair codes losslessly in 3 bytes (escapes 0.017 %): 6.3 MB instead of 8.4 MB.
- ncu of the old kernels: `ys` (m8) and `ya` (m16) occupancy 12-13 %, issue 6.7-6.9 %, long_scoreboard 61-63 %, 8.5-8.7 MB.
- After the 6.3 MB stream lands (~2.5 µs in-kernel), ~0.8 µs of tail: decode ALU of the last-landing warps ~0.38 µs, CTA reduction / zero barrier 0.24, `red.add` issue 0.17.
- Dead ends on the way: escape fix-up via parameter list + x gathers (+0.5..0.6 µs), a 512-B-stride smem reduction (16-way bank conflicts, +0.45), a runtime-indexed accumulator (whole array in local memory, +1..2.5 µs).

**(d) unpooler m128: the old BF16 kernel moved 68 MB, a 12-bit loop moves 51.5 MB; the problem was the split-K tail** (U3/NOTES.md iterations 0, 5, 8):
- A 12-bit RS-wgmma loop (from lane L) ran at ~13 µs but its split-K exchange tail was 1.9 µs (unpooler).
- The tail, from traces: loop end → bulk push issued +0.83 µs (escape correction 0.21-0.29, 32 KB staging, fence, barriers), 32 KB DSMEM transfer ~1.0 µs (~33 GB/s per SM), add + 16 KB store ~0.5.
- Milestone 09's "solo phase" (compute the k-blocks near the split for your own token half only, so the partner's partial is final early) projected **~0.8-1.0 µs below** Q2's kernel (paper projection, U3/NOTES.md iteration 0).

**(e) o m128: the start is the largest fixed cost, and the order of the two copies per stage matters** (Q5/NOTES.md iteration 2; `evidence/Q5_probe_start.py`):

| probe (128 CTAs, 210 KB smem, after the flush) | first stage landed (µs after CTA entry, mean / max) |
|---|---|
| stage 0 alone, issued before any sync | 0.89 / 1.22 |
| `__syncthreads` first, then stage 0 | 1.34 / 1.64 |
| 12 stages by thread 0 after a sync | 1.64 / 2.13 |

- A `__syncthreads` at entry costs ~0.45 µs; the TMA unit issues ~75-90 ns per stage; a second issuing warp does not help.
- Inside the loop the o kernel streams W at 4.66 TB/s (97 % of 4.8); ~30 % of its time is outside the loop (first landing 1.55-1.66 µs, epilogue 0.30, end skew ~0.35).
- qkv m128, by contrast, is DRAM/L2-bound in the loop (4.43 TB/s of W + 64 MB of x from L2): every solo-phase stage costs a full ring cadence, as much as the exchange it hides. No qkv variant won.

**(f) m = 1 exactness costs almost nothing if the loads stay ahead of the math** (N2/NOTES.md; S5 iteration "coordinator update"):
- Exact "far" escapes need two terms: cancel the decoded fake value in the main sum, add x·true in a separate true-domain sum.
- On natural weights pooler has 1 far escape and qkv 0, so qkv compiles to G's kernel instruction for instruction.
- **Load sinking** is the trap: without `__launch_bounds__` min-blocks or an anti-sink dependency, ptxas moves a load below the first FMA and adds a DRAM round trip. S5's exact unpooler GEMV without anti-sink: 2.80 µs; with: 2.40-2.46 (G's inexact algorithm with anti-sink: 2.35).
- N2 also measured "+1.15 µs for a never-taken tail block" (`evidence/N2_ip_probe.py`). It attributed it to cold instruction fetch; milestone 11 showed it was load sinking.

**(g) m = 256: what K2 and PW ruled out.**
- The m256 loops already run at 98-99 % of the power-capped tensor rate. Stream-K over 132 SMs pays +2.5..+7 µs in partial exchange; epilogue overlap needs 2 accumulators (+0.3..0.4) (K2/NOTES.md, `ROOFLINE:328-331`).
- The clock under tensor load is set by a fast on-die limiter: 1.61-1.65 GHz within ~1 µs on ≥ 96 SMs, hard ceiling ~1.70-1.74; the best power-aware tile form buys +0.5-0.8 % (PW/NOTES.md, `ROOFLINE:352-355`). So m256 was left alone.

## 3. Diagnosis and projection

- Size: the lean build frees ~500 KB, enough for every queued module.
- The small shapes (m ≤ 8 and the routers) pay 0.4-0.5 % of score per µs, the most of any load. Their kernels sit 0.8-1.6 µs above their byte floors, mostly in fixed start and tail costs.
- Projection per lane, from same-run local deltas × value per µs (bo13r judge: decode-1 0.51 %/µs, verify-1 0.48, decode-16 0.36, decode-32 0.35, verify-16 0.27; computed from `evidence/judge_before_bo13r_scored_64811543.txt`):

| lane | shapes | local Δ (µs) | projected |
|---|---|---|---|
| E8 | pooler m8 | −0.3 net | ~+0.14 % (E8/NOTES.md) |
| S5 | router m1/m8/m16, unpooler m1 | −0.10 / −0.30 / −0.25 / −0.10 | ~+0.33 % (S5/NOTES.md) |
| U3 | unpooler m128 | −0.80 | ~+0.2 % (U3/NOTES.md) |
| Q5 | o m128 | −0.25..−0.31 (alternating) | ~+0.08 % (inferred) |
| O3 / O2 | o m32 / o m16 | −0.05..−0.16 / −0.07 | ~+0.02-0.05 % / ~+0.03 % |
| N2 | qkv/pooler m1 | tie / +0.05 | 0 (exactness) |

Sum: about **+0.8 %** (inferred), plus exactness on every shape.

## 4. Change

Built with `LN/build_t3.py` over the lean modules; intermediate builds, none judged: bo14 (+E8, 826 KB) → bo14_lean (LN, 315 KB) → bo15L (+S5, 339 KB) → bo16L (+U3, 375 KB) → bo17L (+Q5, 389 KB) → bo18L (+O3, 416 KB) → bo19L (+N2, G dropped, 408 KB) (`ROOFLINE:294-346`).

1. **LN lean modules**: every module rewritten without lab switches, proven identical (§2a). Lane O2's o m16 config (sk11 with 4 consumer warps instead of sk3's 2) folded into the lean sk11; sk3 dropped.
2. **E8 `m8_pooler`** (pooler m = 8): 256 CTAs (2 per SM) × 16-row F12 tile × full K; 16 K-warps with a software-pipelined register ring (W ~2 units ahead, all LDG); mma.sync with W as A; 1 exact escape warp; smem reduction.
3. **S5 `sm5`**: router m1-8 `sm5_rz1` and m9-16 `sm5_rz2` (24-bit hi/lo decoded in registers into mma.sync, split-K × 16, per-warp escape trailers, 4 zeroing CTAs, `red.add`); unpooler slice m1 = the public 12-bit `cgemv` (Apache-2.0 port) with exact escapes; m2-8 `unps` = G's tile kernel with exact two-term escapes.
4. **U3 `u128`** (unpooler m97-128): escape-free 12-bit RS-wgmma loop; split-K over a cluster pair with the solo phase (QS = 4 chunks); per-warp bulk-copy push from smem.
5. **Q5 `q5`** (o m97-128): Q2's o kernel with the W bulk copy issued **before** the x TMA in every stage.
6. **O3 `o3`** (o m32): 128 CTAs × 64 rows × all K, full residency (8 W items + 4 x pieces requested at t = 0), two warpgroups alternating whole items, RS-wgmma with F12 decoded into A registers, exact two-term escapes.
7. **N2 `n2`** (qkv/pooler m1): G's g311 statement for statement plus exact far escapes; G removed from the build.

How the lanes converged (internal iterations, local µs):

| lane | variant | result | what it taught |
|---|---|---|---|
| E8 | CUDA-core FFMA at m8 (paper) | ~95-100 % of issue slots | use mma.sync even at m = 8 |
| E8 | byte-balanced CTAs, 1 per SM (e8r) / hw-scheduled CTAs + cluster split-K (e8c) | pooler 12.74 vs 12.48 / 17.9-23.7 vs 12.19 | a CTA's progress is not proportional to its bytes in flight; cluster split-K 1.5-2× slower |
| E8 | qkv m8 (144 tiles on 132 SMs) | every fix lost | the TMA ring (qkvb) stays |
| S5 | rz v1, helper duties merged into warp 0 | m1 4.13 vs 2.72 | ticket atomic + poll in the tail cost ~1 µs: dedicated helper warp |
| S5 | rz v2, column-sum reduction, trailers in the W stream | m8 3.01 vs 3.33 | escapes must ride with the warp's own W |
| S5 | exact g311 slice vs public cgemv | cgemv with exact escapes is faster | ship the port |
| U3 | straight solo-phase transfer, both WGs push | — | st.async from registers stalls the issuing warp and blocks later smem loads: push from smem by bulk copy |
| U3 | pooler with the same design (4-way) | 17.89 vs 17.34 | 48 KB egress per CTA ~1.5 µs: pooler stays on Q2 |
| Q5 | Q3's solo phase on qkv m128 | tie (11.1-11.4) | at m128 the loop is DRAM/L2-bound, not ingress-bound |
| Q5 | rotated k start per CTA | qkv +3.3 µs | never desynchronise the k order across CTAs |
| Q5 | W-first per stage (WF) | o −0.05..−0.13 same run | takes the x descriptor fetch / TMA issue out of the W request path |
| O3 | mma.sync consumers (RG/KS variants) | ties | the F12 mma.sync consumer runs ~11-13 clk per HMMA per SMSP |
| O3 | warpgroups alternate whole items + RS-wgmma | first win at m32 | RS-wgmma pays only with straight-line items and compile-time descriptor offsets |
| N2 | o m1/m8 in F11 | no win | ~205 escapes per tile need x gathers in the tail |

## 5. Result

| check | result |
|---|---|
| full local bench, alternating processes, 2 pairs | bo13r judge-projected 1.6101 / 1.6152 → bo19L 1.6335 / 1.6368: **+1.4 %** (local geomean +1.55 %) (`ROOFLINE:347`) |
| E8 (6 runs, both boxes) | pooler m8 −0.38 µs (x1.596 → x1.648); identical-kernel controls −0.09, so net ~−0.3; alternating processes 12.14 / 12.42 / 12.20 vs 12.59 / 12.67 / 12.49 (`ROOFLINE:294`) |
| S5 (4 runs) | router m1 −0.10, m8 **−0.30**, m16 **−0.25**, unpooler m1 −0.10, m8 −0.04 (tie, now exact) (`evidence/S5_bench_final_*.log`) |
| U3 | alternating 4 + 4 processes **−0.80 µs** (x1.128 → x1.165); nsys 17.74 → 17.02; ncu 20.06 → 19.20 µs, DRAM 68.2 → 51.5 MB |
| Q5 | o m128 same run −0.06..−0.10, alternating −0.25..−0.31; ncu 7.46 → 7.33 |
| O3 | o m32 paired −0.05..−0.06 (3 runs); nsys 5.57 → 5.44 (`evidence/O3_bench_final_*.log`) |
| N2 | qkv tie (SASS-identical), pooler +0.05 (mean, inside ±0.1 noise) (`evidence/N2_bench_final.log`) |
| correctness | **first build to pass every gate**: `check.py --graph` 259 cases × 2 seeds ALL OK, `exact_router` ALL OK, `exact_all` OK at `--nesc 200` and `--nesc 3000` (`ROOFLINE:346`). Strict one-hot test: n2 0 / 0 mismatches vs G's 75 / 82; sm5 0 vs 50-75 |
| judge | 1.6277 (single scored run) |

**The judge, read honestly.**
- 1.6277 is below the previous build's scored run (1.6326) and above its mean (1.6194). The single reading cannot resolve a 1 % change.
- Per shape against both runs of 09 (`evidence/judge_attribution.txt`): the 12 changed shapes **+0.87 %**, the 18 unchanged −0.21 %, together +0.65 %. Largest: router m16 +8.9 %, router m8 +8.6 %, unpooler m128 +6.7 %, pooler m8 +4.9 %, o m128 +3.8 %.
- The coordinator compared against the scored run of 09 only: 13 changed shapes +2.17 %, 17 unchanged −1.66 % (sd 2.4 %), i.e. this judge run was ~1.7 % slow and the changed shapes gained ~+3.8 % relative; drift-corrected ≈ **1.655**, equal to the local projection (`ROOFLINE:357-360`). (The 13th "changed" shape there is router m128, whose lean module has identical SASS; I count 12.)
- The one shape to watch: unpooler m1 −4.3 % with the public `cgemv` port, which had won locally by 0.10 µs. The coordinator flagged it as a possible non-transfer (`ROOFLINE:361`). Tiny kernels keep only 0.78-0.80 of their local speedup on the judge (`ROOFLINE:412`), and their per-graph placement swings ±3 % (`ROOFLINE:366-367`).
- The user's call at the time: the 1.6277 was judge noise and bo19L was the base to build on (`ROOFLINE:363`).
- vs projection: +0.8 % projected from the lanes, +1.4 % on the full local bench (the bench includes ~0.3-0.5 µs per-process placement noise), +0.65 % on the judge's per-shape split. The m = 1 losses and the ~1 µs fixed judge cost per kernel absorb part of every small-shape gain.

## 6. Takeaways, and what came next

- **Prove a refactor, don't trust it.** Same owner, same launch geometry, bit-identical outputs, identical SASS for every launched kernel: after that, the lean build cost no measurement time.
- **For tiny kernels the fixed costs are the kernel.** Routers and slices stream 6-8 MB in ~2 µs; the 0.8 µs after the last byte (reductions, zero barriers, escape fix-up) is where S5's wins came from.
- **Exactness is a scheduling problem, not a math problem.** Exact two-term escapes cost ~0.05 µs once every load is issued before the first FMA; a sunk load costs a whole DRAM round trip.
- **The first landing and the tail dominate m128 o and the split-K kernels.** Q5's per-stage W-first order and U3's solo phase both attack fixed time around a loop that already streams at 87-97 % of DRAM.
- Next: [11_small_shape_tails_stack](../11_small_shape_tails_stack/). With every shape exact and ~590 KB of room, the remaining work was smaller: the router m128 egress tail, a cold-line fix on the qkv m8/16 ring epilogue, the router m16 zero barrier, and o m8's second wave.

## Reproduce

All you need is torch and an H200. Run from `tools/`.

```bash
cd muse_bf16_proj/tools
python3 check.py ../10_lean_build_exact_escapes/build.py --graph
python3 exact_all.py ../10_lean_build_exact_escapes/build.py --nesc 200     # and --nesc 3000
python3 exact_router.py ../10_lean_build_exact_escapes/build.py
# full judge-alike bench; run each candidate in its own process, alternate them, and compare speedups
python3 bench.py --cand base=../09_compliant_readable_rebuild/build.py --reps 60 --xclones 6
python3 bench.py --cand new=../10_lean_build_exact_escapes/build.py --reps 60 --xclones 6
```

The probe scripts in `evidence/` (`Q5_probe_start.py`, `N2_ip_probe.py`, `LN_prove.py`) import their lane's lab modules, which are not copied here; run them from the original lane directory (`claude/t3/<LANE>/`). `evidence/judge_attr.py` (with `jt.py`) recomputes the per-shape judge split from the judge `.txt` files.

Originals: builds `muse-opt-codex/muse_bf16_proj/claude/t3/{bo14,bo15L,bo16L,bo17L,bo18L,bo19L}.py`, `LN/bo14_lean.py`, `LN/build_lean.sh`; lanes `claude/t3/{LN,E8,O2,S5,U3,Q5,O3,N2}/` (deliverables `m8.py`, `olay.py`, `sm5.py`, `u128.py`, `q5.py`, `o3.py`, `n2.py`); dead-end lanes `claude/t3/{K2,PW}/`. Judge result: `claude/t3/profiles/bo19L.f90aa1696c8b4879.txt`.
