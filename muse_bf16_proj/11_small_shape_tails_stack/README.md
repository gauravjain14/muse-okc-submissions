# 11: small-shape tails, a stack of four (router m128 egress, qkv m8/16 cold epilogue, router m16 zero barrier, o m8 in one wave)

| | before (10) | after (11) |
|---|---|---|
| build | `10_.../build.py` = `bo19L` (md5 dbdbe35a, 408,026 B) | `build.py` = `bo23L` (md5 f3c2c415, 458,224 B) |
| judge | scored 1.6277 (job f90aa169) | scored **1.6232** (job 43c064cb). Intermediate builds of the stack: `bo21L` (md5 e785088b) scored 1.6337 (job b983282a) and checked 1.6124 (job b3039841), the same file; `bo22L` (md5 6a0d6f27) scored 1.6315 (job 98a43aa4) |
| real effect | | local, per shape, same run with controls: router m128 −0.27 µs, qkv m8 +1.1-2.8 %, qkv m16 +0.5-1.9 %, router m16 ≈ −0.07 µs, o m8 −0.22..−0.30 µs; ≈ **+0.3 %** projected (inferred). Below what one judge run resolves: the same file read 1.6337 and 1.6124 |
| shapes changed | | router m128 (R5), qkv m8/m16 (IC patch on qkvb), router m16 (S6), o m8 (O8) |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: code copied or adapted from other projects is marked in place with `SPDX-SnippetBegin` / `SPDX-SnippetEnd` comments (source, copyright, license, our changes); see `NOTICE` and `LICENSES/` at the repository root. The code is unchanged. The judged file has md5 `f3c2c415`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

Sources. Paths are relative to ``. `ROOFLINE:n` is line n of `muse-opt-codex/muse_bf16_proj/claude/t3/ROOFLINE.md`, the coordinator's log of this phase (not included here). `<LANE>/NOTES.md` is `muse-opt-codex/muse_bf16_proj/claude/t3/<LANE>/NOTES.md`; copies are in `notes/`. Lane names (R5, IC, S6, O8, and the dead-end lanes in §2f) are labels of agent workstreams.

Terms:
- **Tail**: everything a kernel does after its last weight byte lands: the last MMAs, cross-warp or cross-CTA reductions, zero barriers, output stores.
- **Egress**: bytes an SM pushes out (partial sums, outputs). Every path measured (red.add, plain stores, bulk reduce, DSMEM) runs at ~25-30 GB/s per SM (milestone 09).
- **Zero barrier**: split-K kernels accumulate into the output with `red.add`, so the output must be zeroed inside the same launch (CUDA-graph safe); compute CTAs wait until the zeroing is visible.
- **qkvb**: the TMA-ring kernel for qkv m8/16 from milestone 09. **x clones**: several graphs per candidate with x at random addresses, to average out placement (x placement alone moves a kernel 0.1-0.5 µs).
- **local** = `tools/bench.py`. judge ≈ local + 1.1 µs per shape (§5).

## 1. Starting point

Five kernels of bo19L with a large tail relative to their stream (judge µs from `evidence/judge_before_bo19L_scored_f90aa169.txt`; in-graph nsys and floors from the dashboard of the first stack build, `evidence/dash_bo21L.md`. That dashboard already shows the patched qkv kernel and the router m128 replacement, so router m128's nsys time is rt4's from R5/NOTES.md; its floor is for the same 10.5 MB):

| shape | kernel | judge µs | nsys in-graph µs | stream floor µs | % of 4.8 TB/s in-graph |
|---|---|---|---|---|---|
| router m128 | `r4` (rt4) | 5.59 | 5.65 (R5/NOTES.md) | 3.11 | |
| qkv m8 / m16 | `qkvb` | 9.29 / 9.37 | 8.67 / 8.96 (patched) | 7.06 / 7.08 | 68 / 66 |
| router m16 | `sm5_rz2` | 4.00 | 3.87 | 2.26 | 36 |
| o m8 | `nk_o8` | 5.27 | 5.07 | 3.60 | 52 |

(stream floor = 0.81 µs + DRAM bytes / 4.56 TB/s, the measured cold-stream fit, `evidence/dash_bo21L.md`.)

## 2. What we saw

**(a) router m128: a 2.3 µs stream between a 1.3 µs start and a 1.1 µs tail** (R5/NOTES.md iteration 0, globaltimer traces of rt4, µs from the first CTA start):

| prologue issued | first stage | last stage | MMA done | zero barrier ok | reds issued |
|---|---|---|---|---|---|
| 1.12 | 1.34 | 3.65 | 3.90 | 4.10 | 4.51 |

- 10.4 MB of DRAM in ~2.3 µs is DRAM-bound. The tail is last-stage MMA 0.25 + staging and zero barrier 0.2 + **16 KB of red.add per CTA 0.4** + drain.
- Since egress runs at ~25-30 GB/s per SM on every path (milestone 09, `09_compliant_readable_rebuild/evidence/R4_probe_red.py`), the lever is fewer partial bytes per CTA, not a different reduction instruction.
- Halving the egress alone (32-row tiles, NS 8) tied (4.53 vs 4.52): 256 KB instead of 192 KB of L2→SM per SM and twice the copies cancelled the saving (iteration 1).

**(b) qkv m8/16: what a "cold tail" really costs** (IC/NOTES.md; `evidence/IC_warmcode.py`, `IC_abm.py`, `IC_ic_sass.log`):
- Audit of all 30 launched kernels of bo19L: fetching kernel **code** cold after the judge's flush costs ≤ 0.19 µs per kernel (same kernel timed cold vs with its code pre-warmed in L2): 0.00-0.06 µs for 20 of 25 measurable shapes, 0.10-0.19 for the TMA-ring kernels.
- Milestone 10's "+1.15 µs for a never-taken tail block" was **load sinking**, not instruction fetch. In the NVRTC 13 SASS the escape-slot load `@!P2 LDG.E.NA.CONSTANT` moved from before the first HMMA (index 57) to after it (index 92), one more DRAM round trip. Warming the code did not remove it, it persisted with no flush at all, and it vanished when the same block was moved into a `__noinline__` function.
- What ncu did show: constant-cache miss samples on the epilogue's `ULDC c[0x0][0x208]` (global-memory descriptor) and `ULDC c[0x0][0x2d0]` (the output pointer). ptxas rematerialises parameters at their use, so an earlier `asm volatile` read of the output pointer had been dropped; the descriptor, the constant-bank line and the output page translation were all fetched in the tail.

**(c) router m16: the zero barrier sits on the critical path** (S6/NOTES.md iterations 1-6):
- m = 8 timeline: warps of one CTA do not land together (per-CTA landing spread 0.51 µs median); last landing → CTA end 0.67 µs.
- The zero-done poll succeeds at p50 2.3 µs, **after** the median CTA reaches its tail barrier (2.08).
- The zeroing CTAs issue their stores by 0.44 µs, but `red.release` (MEMBAR.ALL.GPU) completes at p50 1.69 / max 2.86-3.28 µs: an SM's store acks return behind that SM's W data. A zeroing CTA alone on its SM releases at p50 0.90.
- Ablation (invalid output, no zero wait and no red): −0.12 / −0.13 / −0.25 µs at m 1 / 8 / 16: the bound for any zero-barrier fix.
- Trap: one variant used 60 registers × 544 threads → 1 CTA per SM → a second wave (+0.9..+1.3 µs). Rule: check occupancy of every variant.

**(d) o m8: 23 % of the CTAs ran in a second wave** (O8/NOTES.md iterations 0-4; `evidence/O8_prof_v1.log`):
- The old kernel: 512 CTAs × 512 threads at 38-39 registers = 3 CTAs per SM (396 slots), so 116 CTAs start at ~2.6 µs. Its escape-slot threads gather x (8 global loads each) after the MMA; in a 32-row variant these gathers cost 0.4 µs.
- ncu, shipped `o8_m8` vs `nk_o8`: 5.73 vs 6.50 µs (ncu-inflated), same 12.7 MB, occupancy 88.7 vs 63.2 %, warp instructions 935 K vs 1190 K, smem bank conflicts 35 K vs 115 K, long_scoreboard 62 vs 44 %, barrier 7 vs 19 %.
- ptxas at 32 registers reorders the 6 loads per lane. When it issued the escape-list load after the third HMMA, the list became a third serial round trip per warp (~0.2 µs):

| load schedule (m8, d vs the old kernel, local µs) | d |
|---|---|
| lab kernel (dead slot-thread code still in the PTX) | −0.23..−0.31 (6 runs) |
| same, all loads forced up front | +0.05 (barrier stalls 33 % vs 11 %: late warps queue behind early ones) |
| clean module (list issued at SASS index 59) | −0.05..−0.14 |
| **module, list bit 30 → decode-mask bit 6** (list issued at index 17) | **−0.25 / −0.23** |

**(e) Where the time went at the start of this milestone** (first stack build, nsys in-graph vs floors, `evidence/dash_bo21L.md`; `evidence/dash_bo21L_ncu.log`, `dash_bo21L_nsys.log`):

| m | µs over the stream/MMA floor per shape (qkv / o / pooler / unpooler / router) | dominant stall |
|---|---|---|
| 1 | +1.41 / +1.04 / +1.15 / +1.15 / +1.29 | long_scoreboard 51-57 % |
| 8 | +1.61 / +1.47 / +1.38 / +1.63 / +1.33 | long_scoreboard 31-81 % |
| 16 | +1.88 / +1.49 / +1.32 / +1.37 / +1.61 | long_scoreboard 42-54 % |
| 32 | +2.32 / +1.86 / +2.37 / +2.21 / +1.64 | long_scoreboard 34-55 % |
| 128 | +3.37 / +1.70 / +3.83 / +4.68 / +2.20 | long_scoreboard 28-48 % |
| 256 | +4.37 / +2.77 / +4.32 / +4.07 / +2.83 | wait / barrier (tensor 66-87 % on the four BF16 layers) |

The sum is 65.7 µs over 30 shapes in nsys, about 0.8 µs per shape of which is the nsys-vs-bench offset. Small shapes sit 1.0-1.9 µs over their floors: mostly fixed start and tail cost, which is why every lever here is a tail lever.

**(f) What the parallel dead-end lanes established** (each falsified a lever for the remaining m ≤ 32 and m ≥ 128 time):

| lane | idea | measured | conclusion | source |
|---|---|---|---|---|
| DF | cheaper F12 decode on consumer-bound rings | decode removed entirely (wrong results) is no faster: qkv32 −0.04..−0.11, pooler32 −0.10..−0.14, o16 0.00 | ring- or structure-bound, not ALU; G's decode (7 ALU / 8 weights) is ALU-optimal | `ROOFLINE:390-395` |
| RS2 → Q7 | qkv m32 is per-SM-ingress bound (807 KB/SM at ~96 GB/s) | idle-SM helpers: owners alone 7.68, helpers alone 6.14, together 8.96-9.22 (sk11 9.22); ncu: L2 read sectors unchanged, DRAM 3.4 TB/s (71 %) vs W-only ring 4.6-4.8 | per-SM-ingress diagnosis **falsified**: aggregate L2/DRAM delivery of W while x streams (~1 µs) | `ROOFLINE:420-436` |
| PT | helper warps for pooler m32's tail | 0.68 µs of compute in the last ~2 stages; halving work in place saves half, moving it to helper warps saves nothing (+0.1..0.3) | per-SMSP throughput bound | `ROOFLINE:426-430` |
| WS | tail work stealing on rings | steal claims ~0.45 µs each (`acq_rel` = MEMBAR.ALL.GPU 1.0-1.2 µs) | projected < 0.3 µs everywhere; not built | `ROOFLINE:385-389` |
| E16 | E8's LDG + mma.sync template at m16/32 | x via registers 0.55-0.9 µs at m16, 3.9 at m32; any generic global load in a TMA-ring CTA +1.4..4 µs | the remaining 0.8-1.4 µs at m16/32 is the ring tail | `ROOFLINE:368-372` |
| Q6, U4 | 12-bit W at qkv m128 / pooler m128 | q6c full 18.87; smem decode 8.7 µs; U4 split-2 doubles x per CTA (stream alone 15.5-16.0 vs 13.1) | BF16 SS-wgmma stays at m128 except unpooler | `ROOFLINE:401-404, 438-443` |

## 3. Diagnosis and projection

- All five shapes are tail-bound small kernels: each loses 0.2-1.1 µs after its stream, against 0.27-0.48 % of score per µs.
- Projection per piece (local Δ × value per µs from the bo19L judge: verify-1 0.48 %/µs, decode-16 0.36, verify-16 0.27; computed from `evidence/judge_before_bo19L_scored_f90aa169.txt`):

| piece | shape | expected Δ | projected score |
|---|---|---|---|
| R5 | router m128 | −0.27 µs | +0.07 % |
| IC | qkv m8 / m16 | −0.09 / −0.16 µs | +0.04 % / +0.06 % |
| S6 | router m16 | −0.07 µs | +0.03 % |
| O8 | o m8 | −0.26 µs | +0.12 % |

Sum ≈ **+0.3 %** (inferred). One judge run has a ±1.5 % interval at best, so the stack could only be verified locally.

## 4. Change

Three intermediate builds; every one gated, the first two judged (`ROOFLINE:364, 382, 419, 451`):
- `bo20L` = bo19L + R5 (426 KB, not judged);
- `bo21L` = bo20L + IC's qkvb patch (426,242 B);
- `bo22L` = bo21L + S6 (443,149 B);
- `bo23L` = bo22L + O8 (458,224 B).

1. **R5 `r5`** (router m97-128): R4's design (x as A, router hi and lo rows stacked in wgmma N) retiled to 32 W rows × 64 tokens × 1024 k, 8 K splits: **egress 8 KB per CTA instead of 16 KB**. Stages of 2 k-blocks = ONE W bulk copy + ONE 3D x TMA (x first), 5-slot ring, so the extra copies of iteration 1 disappear.
2. **IC patch on `qkvb`** (qkv m8/16): lane 0 of every consumer warp issues one `ld.volatile.global.b32` from its CTA's first output word at consumer start (result unused). SASS: `LDC.64 c[0x208]` + `LDC.64 c[0x2d0]` + `LDG.E.STRONG.SYS` before the main loop; 85 → 88 registers; hot loop and epilogue instruction for instruction the same. It pulls the cold constant line, descriptor and output translation into the stage-0 wait.
3. **S6 `s6`** (router m16): S5's 24-bit router with 16 compute warps, separate hi / lo accumulator chains, **one** zeroing CTA and a ticket-free modular counter (zeroing CTA adds NCOMP, each compute CTA adds 1 after it saw the zeroing; done ⇔ bit NCOMP set; no generation ticket, no extra round trip), escape-trailer load first. `__launch_bounds__(544, 2)`: 45 registers, 2 CTAs/SM.
4. **O8 `o8`** (o m8): 32 registers → 4 CTAs × 512 threads per SM = **one wave** for all 512 tiles. Escapes fixed inside the warp that owns their k, from the x values already in its B fragment: `p = fma(x, true·2^(112−E_lo), fma(x, −2^−15, p))` (the decoded 2^−15 term cancels exactly first; exact for one-hot rows). The escape-list load is kept early by an always-zero dependency in the decode mask (list bit 30 → mask bit 6, which ptxas cannot fold).

How each piece converged (local µs, same process, x clones):

| lane | variant | result | what it taught |
|---|---|---|---|
| R5 it. 1 | 32-row tiles, NS 8 / 10 / 12 | 4.53 / 4.85 / 4.98 vs 4.52 | deeper rings are slower; halving egress alone ties |
| R5 it. 3 | 2-k-block stages, 3D x map (shipped) | 4.40 vs 4.64 | fewer, larger copies keep the egress saving |
| IC it. 4 | ring prefill from the warm prologue | dead end | |
| IC it. 5 | early output touch on qkvb / pu5 / sk11 / skh / q5 | qkvb −0.05 / −0.12; others +0.03..−0.07; q5 **+0.65** | per-kernel only: on q5 the load's scoreboard lands on the wgmma path |
| IC it. 6 | same touch on skh (unpooler m32) | alternating processes inside noise | not shipped |
| S6 it. 2-3 | one-barrier tail (RM 2) | +0.9..+1.3 µs (second wave) | fixed with `__launch_bounds__(544, 2)` |
| S6 it. 4-6 | memset node / distributed zeroing | +0.8..+1.1 / +0.4..+0.8 | every release waits for its own SM's burst |
| S6 it. 7-12 | ONE zeroing CTA + ticket-free counter (d6) | m16 −0.06..−0.14 | m1 / m8 never separated from placement noise: not claimed |
| O8 it. 1 | 32-row CTAs sharing x (E8 style) | −0.13 / +0.12 | gathers in the tail cost 0.4 µs |
| O8 it. 2 | in-warp exact escapes, 1 tile, 4 CTAs/SM | −0.245 | one wave + no gathers |
| O8 it. 3 | load-schedule table (§2d) | −0.25 / −0.23 | an always-zero dependency pins a load early |
| O8 it. 5 | same structure at o m1 | no same-run win | o m1 stays |

## 5. Result

| check | result |
|---|---|
| R5, router m128 | same process x clones 4.64 → 4.40; alternating single-candidate processes, 3 pairs: x3.328 → x3.542 (**−0.27 µs**, +6.4 %); nsys in-graph 5.65 → 5.33; ncu 6.66 → 6.27 µs, same 10.50 MB (R5/NOTES.md). With bo20L, alternating `--xclones 6`: x3.231 → x3.406 (+5.4 %, 3 of 3 pairs) (`ROOFLINE:364-367`) |
| IC, qkv m8 / m16 | same run, A/B/A'/B' instances: −0.05 (8 of 8 pairs) / −0.12; alternating processes x1.692 → x1.739 (+2.8 %) / x1.647 → x1.656 (+0.5 %); in bo21L, alternating `--xclones 6`, 3 pairs: m8 x1.707 → x1.726 (+1.1 %), m16 x1.624 → x1.656 (+1.9 %) (`ROOFLINE:382`); ncu 9.82 → 9.76 / 9.98 → 9.86; outputs bit-identical |
| S6, router m16 | same run `--xclones 6`, 6 of 6 wins, +3.2..+5.9 %; local 3.21 → 3.09 / 2.99 → 2.88 µs; placement lab −0.08..−0.10 at every W placement; control-corrected two-order runs −0.14 / +0.01 (≈ −0.07 µs) (`ROOFLINE:419`, `evidence/S6_run18.log`) |
| O8, o m8 | two-order same run, `--xclones 6`, 5 identical-kernel control shapes: **+8.1 % / +6.0 %** control-corrected (3.88 → 3.58, 3.85 → 3.63 µs) (`evidence/O8_bench_A1.log`, `O8_bench_B1.log`); nsys 4.96 → 4.77 |
| full bench, bo21L vs bo19L (2 alternating pairs, x clones 4) | judge-projected 1.6359 / 1.6366 vs 1.6355 / 1.6425: a tie within full-load noise (`ROOFLINE:396`) |
| correctness | every build: `check.py --graph` 259 cases × 2 seeds ALL OK, `exact_all` EXACT ALL OK at `--nesc 200` and `3000`, `exact_router` ALL OK (`ROOFLINE:382, 396, 419, 451`; `evidence/O8_gates_full.log`, `S6_run19.log`) |

**The judge, read honestly.**

| run | build | judge | the stack's changed shapes, vs bo19L | the 25 unchanged shapes, vs bo19L |
|---|---|---|---|---|
| b983282a (scored) | bo21L | 1.6337 | router m128 +8.6 %, qkv m8 +0.8 %, qkv m16 +3.1 % | +0.23 % |
| b3039841 (check) | bo21L, same file | 1.6124 | router m128 +2.3 %, qkv m8 +2.2 %, qkv m16 +0.8 % | −1.39 % |
| 98a43aa4 (scored) | bo22L | 1.6315 | router m128 +2.3 %, qkv m8 +4.4 %, qkv m16 +2.4 %, router m16 +0.8 % | +0.12 % |
| 43c064cb (scored) | bo23L | 1.6232 | router m128 +2.0 %, qkv m8 +1.3 %, qkv m16 +0.1 %, router m16 −2.9 %, o m8 +2.6 % | −0.51 % |

(`evidence/judge_attribution.txt`, computed by `evidence/judge_attr.py`.)
- The same file read 1.6337 and 1.6124: −1.3 % with no code change, and its 30 shapes moved −1.58 % as a block. The run-to-run spread is 4-5× the whole stack's projected effect.
- The changed shapes moved the right way in 14 of 15 shape-runs: router m128 +2.0..+8.6 % in all four runs (mean +3.8 %), qkv m8 +0.8..+4.4 % (mean +2.2 %), qkv m16 +0.1..+3.1 % (mean +1.6 %), o m8 +2.6 % in its one run. Router m16 is +0.8 % / −2.9 % in its two runs: unresolved.
- Attributed to the changed shapes only, the score moved +0.11 % (bo23L), +0.21 % (bo21L, both runs) and +0.32 % (bo22L), against the +0.3 % projection (`evidence/judge_attribution.txt`).
- The level: the four runs average 1.6252, about 1 % below the local projection of ~1.64 (`ROOFLINE:452`). The single reading of this milestone's build, 1.6232, is its real level by our rule.
- **Why the judge sits below local** (`ROOFLINE:407-414`, from two judge runs of bo21L and a per-shape local bench, `evidence/local_bo21L.txt`): the judge subtracts `2·one − two` = 3.71 µs, where one/two are graphs of 1/2 trivial kernels (4.86 / 6.02 µs). The marginal cost of a kernel in a graph (~1.15 µs) stays in the measured time. So judge ≈ local + 1.1 µs per shape: +0.9..+1.2 µs on small shapes, +0.5..−0.2 µs at m = 256. Tiny kernels keep only 0.78-0.80 of their local speedup (router/unpooler m1, routers m16/32); big shapes 0.96-1.03. Identical-code per-shape cv is mostly < 1.5 %, up to 3.3 % (pooler m256, router m128, qkv m128, pooler/qkv m32).
- A method fact from S6: alternating single-candidate process pairs carry a **process-level bias of +4..9 % on identical kernels** (allocation layout), so their ratios are only usable against a same-kernel control shape (`ROOFLINE:416-418`).

## 6. Takeaways

- **Tails are where tiny kernels lose, and each tail has its own cause**: egress volume (router m128), cold constant/descriptor lines (qkv m8/16), a zero barrier behind store acks (router m16), a second wave and a serial load (o m8). Every fix was found with a trace or ablation that isolated the tail, not with a sweep.
- **ptxas decides where loads go.** Three of this milestone's fixes (O8's mask dependency, S6's trailer-first, IC's re-attribution of the "cold fetch") are load-placement fixes. Re-check SASS load placement (`afterUse`) after any code change, even dead code.
- **Below ~0.5 % the judge cannot see a change.** The stack was verified per shape with controls and accepted on that evidence; the judge only bounds it.
- **What the remaining small-shape time is not**: decode ALU (DF), per-SM ingress (Q7), helper-warp throughput (PT), tail stealing (WS). Three more lanes closed in the same hours: x TMA multicast loses because Hopper already merges identical x reads from many SMs before the L2 (XM, `ROOFLINE:453-459`); 12-bit W at qkv m256 is only −0.1..−0.3 µs even with no decode (Q8, `ROOFLINE:461-465`); a wave/occupancy audit found every shipped kernel at one wave except `nk_o1` (o m1), whose one-wave variants were slower (S7, `ROOFLINE:466-472`).

## Reproduce

All you need is torch and an H200. Run from `tools/` (the harness as it was at this milestone).

```bash
cd muse_bf16_proj/tools
python3 check.py ../11_small_shape_tails_stack/build.py --graph
python3 exact_all.py ../11_small_shape_tails_stack/build.py --nesc 200       # and --nesc 3000
python3 exact_router.py ../11_small_shape_tails_stack/build.py
# same-run A/B with x clones; read the unchanged shapes as controls
python3 bench.py --cand base=../10_lean_build_exact_escapes/build.py --cand new=../11_small_shape_tails_stack/build.py \
       --layers router,qkv_gate,o --shapes 8,16,128 --reps 60 --xclones 6
python3 ../11_small_shape_tails_stack/evidence/judge_attr.py <before .txt,...> <after .txt,...> router:128,qkv_gate:8,qkv_gate:16,router:16,o:8
```

The lane scripts in `evidence/` (`IC_warmcode.py`, `IC_abm.py`) import modules from the original tree; run them from `claude/t3/IC/`.

Originals: builds `muse-opt-codex/muse_bf16_proj/claude/t3/{bo20L,bo21L,bo21L_c1,bo22L,bo23L}.py` (`bo21L_c1.py` is byte-identical to `bo21L.py`); lanes `claude/t3/{R5,IC,S6,O8}/` (deliverables `r5.py`, `IC/qkvb_ic.py`, `s6.py`, `o8.py`); dead-end lanes `claude/t3/{DF,RS2,Q7,PT,WS,E16,Q6,U4,XM,Q8,S7}/`; dashboard `claude/t3/dash_bo21L.md`, `local_bo21L.txt`. Judge results: `claude/t3/profiles/{bo21L,bo21L_c1,bo22L,bo23L}.*.txt`.
