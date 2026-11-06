# IC — cold-instruction audit of every kernel in the build (2026-11-02, base claude/t3/bo19L.py)

All times local us, box 0, judge style (512 MB read before every replay, CUDA graph, empty graph subtracted) unless
noted. Tools (claude/t3/IC, research only): `sass_ic.py` (NVRTC 13 SASS of every launched kernel from an LN/trace.py
pickle: loops, forward jumps, out-of-line code), `ncu_ic.sh` + `ncu_export.py` + `ncu_ic.py` (ncu SourceCounters per
SASS instruction, flushed caches, all 30 shapes in one process), `exit_cold.py` (executed EXITs followed by
never-executed code), `ip_ncu.py` (ncu on N2's probe), `abm.py` (same-run module A/B with x clones, bit-compare),
`warmcode.py` (cold vs code-warm timing of every shipped kernel), `sass_local.py` (local NVRTC 12.1 SASS structure).

## RESULT (read first)
- Audit of all 30 launched kernels of bo19L (+ R5 of bo20L): the cost of fetching kernel CODE cold after the judge's
  flush is <= 0.19 us per kernel (measured: same kernel timed cold vs with its code pre-warmed in L2; 0.00-0.06 for
  20 of 25 measurable shapes, 0.10-0.19 for the TMA-ring kernels qkvb / skh / kg_qkv / pu5 / sk11_qkv / q5; router
  kernels cannot run shrunk). A 1200-instruction cold tail executed after the DRAM stream costs the same as a warm one.
- N2's "+1.15 us for a never-taken tail block" is ptxas load sinking (a slot LDG moved after the first HMMA), not
  instruction fetch: warming the code does not remove it, it persists without any flush, and it vanishes when the same
  block is moved into a __noinline__ function (details below). Check load placement (afterUse) after any code change.
- One fix wins both gates: V2 qkvb (qkv m=8 / 16) + an early load from the CTA's output in each consumer warp, which
  takes the epilogue's cold constant-bank line / memory descriptor / output translation off the tail:
  same run -0.05..-0.12 us, alternating processes x1.692 -> x1.739 (m=8) and x1.647 -> x1.656 (m=16), ncu -0.06 / -0.13,
  bit-identical outputs. IC/qkvb_ic.py; IC/sub_ic.py = bo19L + it (check --graph ALL OK, exact_all 200 / 3000 OK,
  exact_router OK); IC/sub_ic20.py = bo20L + it (not gated separately; only V2 differs).
- Everything else measured neutral or worse (table at the end).

## READ FIRST (correction of the cross-cutting "cold tail code" fact, measured 2026-11-02 evening)
N2's +1.15 us probe effect (never-taken block after the tail store) is NOT an instruction-fetch cost. It is ptxas
LOAD SINKING caused by the extra code in the function: in the NVRTC 13 SASS of o2_ip1 the escape-slot load
`@!P2 LDG.E.NA.CONSTANT R3` moved from before the first HMMA (index 57 in o2_ip0) to after it (index 92), i.e. it is
issued only after W landed = one more DRAM round trip (~1.1 us). Evidence (ip_warm.py / ip_var.py, box 0):
- code made warm (flush -> same kernel on 2 CTAs -> timed): ip1 still 4.83-4.86 vs ip0 3.62 (warming removes nothing);
- no flush at all (everything in L2): ip1 still +0.72 us;
- the same block in a `__noinline__` function (only a CALL after the store): 3.58 vs 3.66 (cost gone, body SASS
  unchanged); non-uniform predicate: +1.10 (unchanged); NI 16: free, NI >= 24: +1.1 (scheduler threshold).
- a genuinely EXECUTED cold tail (1200 straight-line instructions, one warp per CTA, after the DRAM stream) costs the
  same cold or code-warm (ip_warm.py --taken 1: +0.06 / -0.03 us).
=> Rule: after ANY code change (even dead code) re-check load placement (N2's sass_n2.py afterUse, LDG/TMA issued
after the first use of loaded data); tail layout / out-of-lining rare code is not a lever by itself.
Every shipped kernel's total cold-code cost is <= 0.19 us (iteration 3 table).

## iteration 0 — static + ncu audit of the 30 launched kernels (bo19L)
Trace: LN/trace.py bo19L -> ic_tr19.pkl (30 scored shapes, 1 launch each; sm5_rz1 serves router m=1 and m=8).
SASS (ic_sass.log, ic_sass/*.sass): forward jumps >= 24 instr exist in every warp-specialised kernel, but they are
(a) role dispatch at kernel start (producer / escape / consumer blocks) and (b) out-of-line mbarrier try-wait retry
blocks after the last EXIT (6-35 instr, the same 1-3 lines for every wait site, warmed by the main loop's waits).
No shipped kernel has a far UNCONDITIONAL jump in a consumer tail except m8_pooler's `BRA` over the 120-instr escape
block (365 -> 486); its target is warmed by the escape warp, which gets there first (ncu: 2 no_inst samples).
Post-loop code sizes (instructions after the largest in-line loop to the last in-line EXIT, all roles included):
u128 1979, qkvb 403-404, o3 395, sk11 372-380, skh 375, pu5 362-368, kg (Q2/K) 189-296, q5 187, rz1/rz2 217,
r4 33-169, ya 105, m8 77, cgemv 86, n2 / nk / unps / ph / kg_qkv <= 4.

ncu (SourceCounters + WarpStateStats, --warp-sampling-interval 0, --cache-control all; 740 samples/us, ~5.6 per SM
per us): no_instruction is 1.1-7.8 % of samples per kernel and mostly ONE instruction: the kernel's first
(`LDC R1, c[0x0][0x28]`, cold entry fetch): 0.09-0.29 us per SM (qkvb_8 0.73, unps_8 0.41). Post-loop no_inst
<= 0.3 % of samples except u128 (1.9 % = ~0.36 us, agent U3's correction region) and sk11_qkv (1.6 %).
imc (cold constant-bank lines) in the tails: the first STG's `ULDC c[0x0][0x208]` (global memory descriptor) and
the late `p.out` / `p.N` params: pu5 0.14-0.15 us, qkvb 0.08-0.09, q5 0.08, kg_pooler_64 0.09, kg_o 0.06, u128 0.06,
sk11 0.03-0.07, skh 0.05 (per SM, estimated from samples).

## iteration 1 — calibration on N2's probe (ip_probe.py / ip_ncu.py)
Timing reproduced today: o2_ip0 3.60 | ip1 (never-taken NI-line block after the final store) 4.75 | ip2 (block at
kernel start) 3.73 (NI 200); NI 24: 3.68 / 4.80 / 3.78; NI 32: 3.71 / 4.93 / 3.87; NI 40: 3.74 / 4.91 / 3.87.
=> the cost starts between NI 16 (~50 SASS instr, N2: free) and NI 24 (~75 instr).
ncu durations show it too (ip0 5.54-5.82, ip1 6.56-6.78, ip2 5.70-5.76) but PC sampling does NOT: no_inst samples
are equal (ip0 123-179, ip1 127-264); ip1's extra samples are barrier (+270) and long_scoreboard (+170).
NVRTC 13 SASS of ip1: all 16 warps reconverge at a BSYNC after the tail (the block is common code) and execute
`@P0 EXIT` followed by 422 never-executed instructions. => sampling cannot see this cost; only timing can.
Detector (exit_cold.py: executed EXIT followed by >= N never-executed instructions): rz1/rz2 72, qkvb 95, sk11_o_16
79, pu5 16-shapes 79, ya 88, cgemv 50, u128 48.

## iteration 2 — sk11 o m=16: remove the never-run remainder loop behind the consumers' EXIT
Patch (IC/sk11_ic.py): compile-time M, fully unrolled `for it < ceil(R4*M/NC)` (SASS: consumer EXIT now followed by
the escape warps' code, no cold block). Same run (abm.py, 4 x clones x 30 reps x 2 rounds, A/A' and B/B' instances):
A 4.056 / 4.052 | B 4.100 / 4.068 | A2 4.164 / 4.152 | B2 4.012 / 4.016 -> B - A = -0.05 (inside the A/A' spread of
0.1). Bit-identical outputs. => the probe's cost does NOT transfer to this pattern: the detector is not sufficient.

## iteration 3 — total cold-code cost of every shipped kernel (warmcode.py, bo19L, box 0, 3 x clones x 30 reps)
cold = judge state; code-warm = flush, then the same kernel on a 2-CTA grid and a dummy x/out (instruction lines +
code-page translations in L2/TLB), then the timed launch. Router kernels spin on other CTAs (split-K semaphores):
not measurable this way (rz1/rz2/ya/r4); their ncu tail no_inst is <= 0.3 % of samples.
| shape | kernel | cold | code-warm | cold-code cost |
|---|---|---|---|---|
| qkv m=1 | n2_qkv_gate | 7.20 | 7.20 | +0.00 |
| o m=1 | nk_o1 | 3.68 | 3.65 | +0.03 |
| pooler m=1 | n2_pooler | 11.55 | 11.55 | +0.00 |
| unpooler m=1 | sm5_cgemv | 2.37 | 2.37 | +0.00 |
| qkv m=8 | qkvb_8 | 7.87 | 7.68 | +0.19 |
| o m=8 | nk_o8 | 4.10 | 4.06 | +0.03 |
| pooler m=8 | m8_pooler | 12.45 | 12.38 | +0.06 |
| unpooler m=8 | unps_8 | 2.88 | 2.82 | +0.06 |
| qkv m=16 | qkvb_16 | 8.16 | 7.97 | +0.19 |
| o m=16 | sk11_o_16 | 4.22 | 4.19 | +0.03 |
| pooler m=16 | pu5_pooler_16 | 12.35 | 12.26 | +0.10 |
| unpooler m=16 | pu5_unpooler_16 | 12.42 | 12.38 | +0.03 |
| qkv m=32 | sk11_qkv_32 | 8.58 | 8.48 | +0.10 |
| o m=32 | o3_o_32 | 4.58 | 4.51 | +0.06 |
| pooler m=32 | pu5_pooler_32 | 13.07 | 12.96 | +0.11 |
| unpooler m=32 | skh_unpooler_32 | 13.22 | 13.02 | +0.19 |
| qkv m=128 | kg_qkv (Q2) | 11.46 | 11.30 | +0.16 |
| o m=128 | q5_o | 5.34 | 5.25 | +0.10 |
| pooler m=128 | kg_pooler_64_64 (Q2) | 17.57 | 17.55 | +0.02 |
| unpooler m=128 | u128 | 15.73 | 15.71 | +0.02 |
| qkv m=256 | ph_qkv (Q3) | 14.32 | 14.30 | +0.02 |
| o m=256 | kg_o (K) | 7.20 | 7.14 | +0.06 |
| pooler m=256 | kg_pooler_64_128 (K) | 22.50 | 22.46 | +0.03 |
| unpooler m=256 | kg_unpooler (K) | 21.98 | 21.98 | +0.00 |
Single-role kernels (n2, nk, cgemv, unps, m8): ~0; warp-specialised TMA rings: 0.02-0.19. The kernel-entry fetch
that ncu samples as no_inst (0.1-0.7 us per SM) costs nothing measurable in time.

## iteration 4 — qkvb: issue the ring prefill (steps 1 .. NSTAGE-1) from the warm tid-0 prologue (dead end)
Idea: the producer warp's block sits ~520 instructions after the dispatch (cold far jump at start) and it issues x of
stage 0 (XLAG 1) and every later stage. Variant IC/qkvb_prefill_dead.py: tid 0 issues steps 1 .. NSTAGE-1 in the
producer's order after the escape list, the producer continues at NSTAGE. Same run (abm, 4 x clones x 30 x 2 rounds,
A/A'/B/B'): m=8 A 7.98/7.92, A2 7.94/7.96, B 8.01/8.04, B2 8.07/8.02 -> +0.08; m=16 A 8.19/8.13, A2 8.32/8.20,
B 8.29/8.30, B2 8.32/8.26 -> +0.08. Bit-identical. Slower: the consumers' __syncthreads waits for the longer prologue.

## iteration 5 — early output touch (cold constant-bank line / descriptor / output page in the epilogue)
ncu showed imc (constant-cache miss) samples on the epilogue's `ULDC c[0x0][0x208]` (global-memory descriptor) and
`ULDC c[0x0][0x2d0]` (p.out) in the TMA-ring kernels: ptxas rematerialises params at their use, so V2's existing
`asm volatile("mov.b64 ...")` early read of p.out is dropped (SASS: p.out only loaded in the tail; O2's fact 2).
Fix: one `ld.volatile.global.b32` from this CTA's first output word by lane 0 of every consumer warp at the consumer
start (SASS: `LDC.64 c[0x208]` + `LDC.64 c[0x2d0]` + `LDG.E.STRONG.SYS` before the main loop; +2-3 registers, still
1 CTA/SM; the result is unused). A `prefetch.global.L2` form compiles to `CCTL.E.PF2` without the descriptor.
Same run (abm.py, 4 x clones x 30 reps x 2 rounds, A / B / A' / B' instances, bit-identical outputs):
| module (shape) | A, A' | B, B' | B - A |
|---|---|---|---|
| V2 qkvb (qkv m=8) | 7.944 7.856 / 7.828 7.812 | 7.840 7.780 / 7.808 7.812 | -0.05 (8 of 8 pairs < 0) |
| V2 qkvb (qkv m=16) | 8.112 7.960 / 8.312 8.048 | 8.064 7.928 / 8.060 7.892 | -0.12 (A' high in r0) |
| P5 pu5 (pooler m=16) | 12.464 12.420 / 12.472 12.524 | 12.504 12.432 / 12.452 12.484 | +0.00 |
| P5 pu5 (unpooler m=16) | 12.556 12.452 / 12.492 12.452 | 12.580 12.500 / 12.536 12.468 | +0.03 |
| P5 pu5 (pooler m=32) | 13.100 13.056 / 13.128 13.156 | 13.136 13.080 / 13.044 13.068 | -0.03 |
| F sk11 (o m=16) | 4.152 4.148 / 4.152 4.164 | 4.188 4.116 / 4.172 4.144 | +0.00 |
| F sk11 (qkv m=32) | 8.612 8.636 / 8.620 8.616 | 8.720 8.608 / 8.552 8.576 | -0.01 |
| H skh (unpooler m=32) | 13.148 13.132 / 13.068 13.080 | 13.116 13.032 / 12.988 13.000 | -0.07 |
| Q5 q5 (o m=128) | 5.328 5.316 / 5.356 5.380 | 6.020 5.940 / 5.992 5.940 | +0.65 (!) |
Q5: the consumer warpgroups start their wgmma loop right after setmaxnreg; the volatile load's scoreboard (DRAM
latency behind the TMA prefill) lands on that path: dead end. Candidates: V2 and H (alternating-process check next).
Alternating single-candidate processes (claude/bench.py --reps 60, module + torch fallback via IC/sub_one.py, speedup
vs SGLang measured in the same process; A = LN/qkvb_lean.py, B = IC/qkvb_ic.py; order A B A B A B):
| shape | A run 1 / 2 / 3 | B run 1 / 2 / 3 | mean speedup A -> B |
|---|---|---|---|
| qkv m=8 | 8.29 x1.669 / 7.94 x1.705 / 8.03 x1.703 | 7.89 x1.736 / 7.87 x1.746 / 7.85 x1.736 | 1.692 -> 1.739 (+2.8 %) |
| qkv m=16 | 8.42 x1.630 / 8.28 x1.658 / 8.33 x1.652 | 8.34 x1.641 / 8.26 x1.654 / 8.17 x1.672 | 1.647 -> 1.656 (+0.5 %) |
=> V2 qkvb_ic ships (wins same-run and in alternating processes, outputs bit-identical).

## iteration 6 — H skh (unpooler m=32): early output touch, alternating processes (not shipped)
Same run (iteration 5): -0.07 (4 of 4 pairs < 0). Alternating processes (bench.py --reps 60, A = LN/skh_lean.py,
B = IC/skh_ic.py): A 13.15 x1.342 / 13.13 x1.353 / 13.25 x1.336, B 13.22 x1.343 / 13.20 x1.353 / 13.06 x1.356 ->
speedup 1.344 -> 1.351 but 2 of 3 time pairs worse: inside the noise, not a two-gate win -> H stays as is
(IC/skh_ic.py kept as research).
warmcode on the module alone (sub_one.py, same box): qkvb A cold 8.03 / 8.14 (m=8 / 16), code-warm gap 0.14 / 0.18;
qkvb B cold 7.87 / 7.98 (-0.16 / -0.16 vs A), gap 0.13 / 0.10; skh A 13.02 gap 0.10, B 12.99 gap 0.13.

## iteration 7 — sub_ic.py = bo19L with V2 -> IC/qkvb_ic.py (IC/build_ic.py; 408,726 bytes; no encoded data)
`python3 claude/t3/IC/build_ic.py -o claude/t3/IC/sub_ic.py --ic V2=IC/qkvb_ic.py --base ext/min_base.py
 --header ext/HEADER_bestof.txt --no R5` (drop `--no R5` for the bo20L version). diff vs bo19L: the V2 header line + 4
lines (3 comment lines + the load). Gates (box 0):
- check.py IC/sub_ic.py --graph (all layers, NaN-poisoned): "checked 259 cases x 2 seeds + graph: ALL OK"
- Y/exact_router.py --nesc 200: "EXACT ROUTER ALL OK"
- exact_all.py --nesc 200: "EXACT ALL OK"; exact_all.py --nesc 3000: "EXACT ALL OK"

## Analysis block for the shipped patch (V2 qkvb, A = LN/qkvb_lean.py, B = IC/qkvb_ic.py; analysis.sh, box 0)
1. nsys in-graph (nsys_shapes.py, one process each, +-0.1 placement noise): A m=8 8.54 / m=16 8.91 us; B 8.59 / 8.90.
2. ncu (--set full, flushed, boost; ncu_tab.py): m=8 A 9.82 us -> B 9.76; m=16 A 9.98 -> B 9.86. DRAM read 28.49 /
   28.63 MB both (= 2.9 TB/s over the ncu duration; 3.6 TB/s = 75 % of 4.8 over the local in-graph ~7.9 us);
   L2 61.8-62.7 %, SM 16.1-16.5 %, issue 26.8-27.8 %, HMMA pipe 15.5-16.4 %, achieved occupancy 11.8-11.9 % (8 warps,
   1 CTA/SM), eligible 0.30-0.32 warps/clk; stalls long_scoreboard 41-47 %, wait 17-20 %, selected 12-15 %.
3. SASS (NVRTC 13): 85 -> 88 registers, 0 stack / local; +3 instructions before the main loop (LDC.64 c[0x208],
   LDC.64 c[0x2d0], LDG.E.STRONG.SYS, result unused); hot loop and epilogue instruction for instruction the same.
4. Binding unit: the 28.5 MB W stream at ~75 % of DRAM peak in-graph (tensor 16 %, issue 27 %); the patch only takes
   the epilogue's cold constant-bank / descriptor / output-translation misses (~0.05-0.15 us after the last landing)
   off the serial tail. The rest of the gap is the stream start (~1.2 us first landing) and the 196-instruction tail.

## R5 (router m=128, bo20L) audit
Trace bo20L router m=128 -> r5_32_8_2_5 (132 x 192 threads, 58 regs, 693 instr). Static: role-dispatch jumps at start,
out-of-line try-wait blocks after the last EXIT (11 instr), no executed EXIT followed by never-executed code
(exit_cold.py), grid-wide semaphore (ATOMG + LDG.STRONG + CCTL.IVALL spin): cannot run shrunk, so no warmcode number.
ncu: no_inst 4.9 % of samples, 128 of them on the entry instruction; post-loop no_inst 0.2 %; imc: entry
`c[0x208]` 0.14 us per SM, tail `c[0x2c0]` 0.03 us. Nothing to fix for cold code.

## Audit table (bo19L + R5; tail = instructions from the end of the consumer main loop to its EXIT, NVRTC 13)
| shape | kernel (module) | tail instr | cold-code cost (warmcode) | other tail-cold signal (ncu) | fix tried | delta (same run / processes) |
|---|---|---|---|---|---|---|
| qkv 1 | n2_qkv_gate (N2) | no loop, 225 total | +0.00 | - | none | - |
| o 1 | nk_o1 (N) | 1 | +0.03 | - | none | - |
| pooler 1 | n2_pooler (N2) | no loop, 232 total | +0.00 | - | none | - |
| unpooler 1 | sm5_cgemv (S5) | 86 | +0.00 | 50 cold instr after the escape warp's EXIT | none (cost 0) | - |
| router 1 / 8 | sm5_rz1 (S5) | 186 | n/a (grid sync) | 72 cold after EXIT; tail no_inst 0.1-0.3 % | none | - |
| qkv 8 | qkvb_8 (V2) | 196 | +0.19 (+0.14) | imc 0.09 us | early output touch | -0.05 / -0.10 same run; x1.692 -> x1.739; ncu -0.06 |
| o 8 | nk_o8 (N) | 1 | +0.03 | - | none | - |
| pooler 8 | m8_pooler (E8) | 77 | +0.06 | BRA over 120-instr escape block (target warmed) | none | - |
| unpooler 8 | unps_8 (S5) | 1 | +0.06 | - | none | - |
| qkv 16 | qkvb_16 (V2) | 196 | +0.19 (+0.18) | imc 0.08 us | early output touch | -0.12 same run; x1.647 -> x1.656; ncu -0.13 |
| o 16 | sk11_o_16 (F) | 162 | +0.03 | 79 cold after EXIT | remainder loop removed; touch | +0.00 / +0.00 |
| pooler 16 | pu5_pooler_16 (P5) | 161 | +0.10 | imc 0.14 us | touch | +0.00 |
| unpooler 16 | pu5_unpooler_16 (P5) | 155 | +0.03 | imc 0.09 us | touch | +0.03 |
| router 16 | sm5_rz2 (S5) | 186 | n/a (grid sync) | 72 cold after EXIT | none | - |
| qkv 32 | sk11_qkv_32 (F) | 193 | +0.10 | imc 0.07 us | touch | -0.01 |
| o 32 | o3_o_32 (O3) | 137 | +0.06 | (already has O2's early store) | none | - |
| pooler 32 | pu5_pooler_32 (P5) | 171 | +0.11 | imc 0.15 us | touch | -0.03 |
| unpooler 32 | skh_unpooler_32 (H) | 167 | +0.19 (+0.10) | imc 0.05 us | touch | -0.07 same run; processes neutral: not shipped |
| router 32 | ya (Y) | 1 | n/a (grid sync) | 88 cold after EXIT | none | - |
| qkv 128 | kg_qkv (Q2) | 4 | +0.16 | entry imc (placement words c[0x400+]) 0.15 us | none (design) | - |
| o 128 | q5_o (Q5) | 129 | +0.10 | imc 0.08 us | touch | +0.65 (dead) |
| pooler 128 | kg_pooler_64 (Q2) | 199 | +0.02 | imc 0.09 us | none (bound 0.02) | - |
| unpooler 128 | u128 (U3) | 1704 | +0.02 | no_inst 0.36 us (U3's correction; warm code does not help) | none | - |
| router 128 | r5 (R5) / r4 (R4) | 1 | n/a (grid sync) | tail no_inst 0.2 % | none | - |
| qkv 256 | ph_qkv (Q3) | 4 | +0.02 | - | none | - |
| o 256 | kg_o (K) | 238 | +0.06 | imc 0.06 us | none | - |
| pooler 256 | kg_pooler_64_128 (K) | 132 | +0.03 | - | none | - |
| unpooler 256 | kg_unpooler (K) | 238 | +0.00 | - | none | - |
| router 256 | r4 (R4) | 1 | n/a (grid sync) | - | none | - |

## Deliverable
- `IC/qkvb_ic.py` (V2, qkv m=8/16): LN/qkvb_lean.py + one early load from the CTA's output in every consumer warp
  (4 lines). Outputs bit-identical (abm.py on normal + outliers inputs, every instance).
- `IC/sub_ic.py` = bo19L with V2 -> IC/qkvb_ic.py (IC/build_ic.py); gates in iteration 7. Expected: qkv m=8 -0.05..-0.16
  us, qkv m=16 -0.03..-0.16 us (verify-1 and decode-16 loads), everything else identical to bo19L.
- Research copies (not shipped): skh_ic.py (neutral), sk11_touch_dead.py, pu5_touch_dead.py, q5_touch_dead.py,
  sk11_mloop_dead.py, qkvb_prefill_dead.py.

## Dead ends (measured)
- Removing never-run code behind an executed EXIT (sk11 o m=16, 79 instr): 0.00.
- Issuing the ring prefill from the warm tid-0 prologue instead of the far producer block (qkvb): +0.08.
- Early output touch where the consumers start computing at once (Q5 wgmma warpgroups): +0.65.
- Prologue / dummy-pass warming of the epilogue: the ideal version (all code in L2 before launch) is worth <= 0.19 us
  per kernel (iteration 3) and a 1200-instruction cold tail costs the same as a warm one (iteration 1/READ FIRST), so a
  dummy pass (which adds its own instructions and register pressure, N2 / U3) cannot pay; not built per kernel.
