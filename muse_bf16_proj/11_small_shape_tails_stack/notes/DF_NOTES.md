# Agent DF — cheaper lossless decode for the "consumer-bound" F12 ring shapes (qkv m32, pooler m32, o m16)

## RESULT (read first): no win to ship — the F12 decode is NOT the binding unit in any of the three kernels
- Direct proof (same-run, XCL 4-6 x clones, judge style, box 0): a build of the base kernel with the decode REMOVED
  (raw lo / nibble words fed straight into the MMA, wrong results, opt 32768) is no faster within noise:
  | shape (bo19L kernel) | lab replica b | no decode at all | no compute (ring only) | decode share of the gap |
  |---|---|---|---|---|
  | qkv m32 (sk11 24,3,1,4,2) | 8.67 / 8.72 | 8.63 / 8.61 (-0.04 / -0.11) | 8.40-8.44 | <= 0.1 of 0.25 |
  | pooler m32 (pu5 32,2,2,2,2 XLAG) | 13.08 / 13.23* | 12.98 / 13.09* (-0.10 / -0.14) | 12.29-12.52 | <= 0.14 of 0.6-0.7 |
  | o m16 (sk11 32,2,2,2,2) | 4.12 | 4.12 (0.00) | 3.82-3.89 | 0 of 0.28 |
  (* LOOPR build as reference.) ncu agrees (durations ncu-inflated): no-decode o16 6.30 vs 6.27 us, qkv32 10.50 vs
  10.66, pooler32 16.13 vs 15.87; the ALU pipe is only 22-26 % of peak with the decode and 10-13 % without it.
  => the upper bound of ANY decode-format change is ~0.0-0.14 us per shape; the best ALU-only format (6 vs 7 ops per
  8 weights) would buy 1/7 of that. Nothing measurable is left in the decode.
- Also no win from: leaner consumer loops (20-25 % fewer ALU ops: o16 81 -> 61-64 ALU per stage; +0.0..+0.1 us),
  FMA-pipe decode variants (+0.14..+0.22 at qkv32), 8 consumer warps (qkv32 +0.37; pooler32 0 / -0.07..-0.26 noise),
  NACC 2, and five new escape mechanisms (escape warps cost 0.10-0.19 us at o16 / pooler32, ~0 at qkv32; every
  alternative was worse, see iteration 3). The premise "consumer-bound by decode" came from Q4's wgmma kernel (q32,
  consumer 0.165 us/chunk); bo19L's sk11 qkv32 is RING-bound (ring alone 8.40 vs full 8.64; LDG floor ~6.8).
- Nothing to integrate: no module / sub_df.py is delivered (a module returning None everywhere would be dead code
  under the lean-build rule). The lab (dfk.py + tdf.py) is kept for whoever attacks the ring or the tail next.
- Transferable facts (measured here):
  1. GENERIC GLOBAL LOADS DURING A TMA RING ARE CATASTROPHIC: one escape warp doing ~20 x 2-byte ld.global(.nc) of x at
     kernel start (or after stage 0 landed) made o16 4.2 -> 6.1-6.5 us (+2 us), with .nc.L1::no_allocate or plain
     ld.global alike; without the loads the same code is -0.13. (V2's "generic loads + TMA in flight" hazard has a
     timing face too.) Use only TMA / bulk copies for any global data in these kernels.
  2. Small 2D TMA gathers of x columns (box 8 k x M tokens, one per escape, issued by an escape warp at start) cost
     +0.4-0.5 us at o16 (x lines are the L2 hot spot; extra small TMAs on them slow the ring's x).
  3. ptxas re-derives per-lane smem addresses from %tid in EVERY stage of the ring loop (o16: S2R + ~25 ALU/IMAD per
     stage) even at 56 regs; an empty `asm volatile("" : "+r"(addr))` per stage pins them (LOOPR+PIN: 142 -> 112 instr
     per stage). Unrolling the ring by NSTAGE gives compile-time slot offsets but the unrolled bodies run slower
     (qkv32 pure consumer +0.38 us: instruction footprint).
  4. ptxas fuses mul.lo + mad.hi on the same operands into IMAD.WIDE + IMAD.MOVs, and lowers mad.wide with a 64-bit
     addend to IMAD.WIDE + IADD3 + IMAD.X: FMA-pipe funnel shifts cost 3.5-4.5 issue slots per nibble word vs 1 SHF.
  5. Escape-warp variants that skip stages cannot use ring parities (they get 2+ rounds ahead: deadlocked / crashed
     here until fixed). A relay barrier per (tile, slot) arrived by consumer warp 0 after its full wait is exact, but
     escape warps that only join escape stages hold those slots longer: o16 +0.1, pooler32 +0.4..+0.55 us.
  6. The pure consumer (no traffic) of pooler32 is HMMA + LDSM bound: removing HMMA -1.86 us, LDSM -1.33, decode +0.05
     (of 9.96 us); ncu stall mix of the pure consumer: wait 38 %, selected 29 %, long_sb 8 %, short_sb 7 %, issue 35 %
     (one warp per SMSP: dependent-latency bound).

Files (claude/t3/DF/): dfk.py (lab: sk11/pu5 F12 ring kernel with OPT bits for decode variants, loop variants, escape
mechanisms and ablations; class Lab), tdf.py (same-run harness: --base = the bo19L kernel of the shape, --v lab
variants; checks normal / outliers / one-hot / one-hot on escape columns, bit-exact, NaN-poisoned; XCL x clones,
interleaved replays, paired delta), sass_df.py (local NVRTC 12.1 sm_90a SASS census of sk11_lean.py or dfk variants),
sub_lab.py (candidate wrapper, env DFV / DFV_<label>), ncu_df.sh (ncu full set + ALU / FMA / LSU / smem / HMMA pipes).
All times local us (judge ~= local + 1.0) from tdf.py unless noted; `d` = paired median delta vs the first candidate.

## Lab config (dfk.py)
cfg = (R, NT, NWR, WSK, UPW, NACC, ring cap KB, XLAG, BAKE, OPT); base configs: qkv32 24,3,1,4,2,1,208,0,1 (F),
pooler32 32,2,2,2,2,1,176,1,7 (P5), o16 32,2,2,2,2,1,72,0,1 (O2). OPT: 1 LOOPU, 256 LOOPR, 512 PIN, 2 DEC1, 4 DEC2,
8 ESCG, 16384 ESCT, 262144 ESCL, 4194304 ESCP, 524288 ESLP, 1048576 EBAR; ablations 16 no compute, 32 no escape warps,
64 no traffic (pure consumer), 32768 no decode, 65536 no HMMA, 131072 no LDSM, 2097152 idle warps.
The lab replica of each base (OPT 0) times within +0.0..+0.12 of bo19L's own kernel in the same run (it carries a
second TMap in the params); compare variants with `b`.

## iteration 1 — SASS census of the decode, the minimal sequence, leaner loops, FMA-pipe decode
### Decode instructions per weight (local NVRTC 12.1 sm_90a; per consumer warp per ring stage; lane = 16 weights per
tile-unit: LDS.128 lo + LDS.64 nibbles)
| kernel (bo19L) | instr / stage | weights / lane | HMMA | decode ALU (PRMT+LOP3+SHF) | all ALU-pipe ops | ALU / weight | IMAD |
|---|---|---|---|---|---|---|---|
| o16 sk11 | 142 | 64 | 16 (MT 1) | 32+16+8 = 56 (0.875 / w) | 81 | 1.27 | 20 |
| qkv32 sk11 | 192 | 96 | 48 (MT 2) | 48+24+12 = 84 | 91 | 0.95 | 19 |
| pooler32 pu5 | 152 | 64 | 32 (MT 2) | 32+16+8 = 56 | 66 | 1.03 | 23 |
The extra ALU above 0.875 / weight is per-stage address / slot math (o16: S2R TID, 6 LOP3 swizzles, slot = st % 3 via
IMAD 0xaaab + PRMT sign extension, ~20 IMAD, all recomputed every stage). Decode ALU per HMMA: 3.5 (MT 1), 1.75 (MT 2).

### Minimal 12-bit -> BF16 fragment sequence on SM90 (argument)
- Every mma B register (bf16x2) is the output of >= 1 instruction: >= 4 per 8 weights.
- One bitwise instruction (LOP3: 3 inputs, two of them the mask / constant) can emit a finished bf16x2 only if both
  weights' 24 information bits already sit at their final positions in ONE stored word; that word's remaining 8 bits
  (the 0111 positions of the two hi bytes) then carry foreign code bits that no single uniform shift maps onto hi-byte
  information positions {7, 2, 1, 0} (4 contiguous bits are not a translate of that pattern) -> such "native word"
  layouts cost >= 8 ops per 8 weights in total.
- PRMT outputs need the 8 hi bytes (with the fake-exponent constant 0111) in 2 registers: >= 2 more ops. hA = one LOP3
  (mask | 0x38); hB's code bits sit at {6..3} and need a 4-bit rotation AND mask AND or-constant: no SM90 ALU
  instruction both shifts and masks (LOP3 cannot shift; SHF / LEA / IMAD cannot mask; IADD3 carries into the sign).
  => 4 PRMT + 2 LOP3 + 1 SHF = 7 ALU per 8 weights: agent G's F12 is optimal on the ALU pipe. The only cheaper ALU
  count (6 + an FMA-pipe shift) needs a funnel (the rotation's wrapped bits): with a plain IMAD.HI / IMAD.SHL shift
  one sign bit (right shift) or 3 code bits (left shift) of every word fall off; filling them costs >= 1 more op.
- Fake-exponent rebias is already folded into the post-MMA scale (2^(E_lo - 112)); a constant-free hi byte (bias 0)
  would make code 0 subnormal and products underflow FP32 (not exact). TF32 / IMMA / IDP routes produce 1 output per
  weight or cannot place bytes.
### Format variants measured (same escapes, bit-exact: pure relabelling of bit 3 between the two nibble words)
- DEC1 (hB0 = funnel(N.x, N.y) via mul.lo + mad.wide, hB1 via mad.hi): ptxas emits IMAD.WIDE + IADD3 + IMAD.X + IMAD.HI
  (+ IMAD.MOV of a zero) per word pair: ALU 12 + 2-3 IADD3, FMA 5 per 16 weights (was ALU 14).
- DEC2 (2 x (mul.lo + mad.hi)): ptxas fuses each pair into IMAD.WIDE + 2 IMAD.MOV: ALU 12, FMA 7 per 16 weights.
### Leaner loops
- LOOPU (ring loop unrolled by NSTAGE, slots / parities compile-time, per-lane addresses hoisted; tail stages
  separate) + PIN: o16 61 ALU / stage (was 81), pooler32 120.5 instr / stage (152), qkv32 169 (192).
- LOOPR + PIN (rolled, hoisted + pinned addresses, uniform slot offset): o16 112 instr / stage (142, ALU 64),
  pooler32 140 (152), qkv32 188 (192).
### Timing (o16, tdf box 0, XCL 4, 30 reps)
| variant | local | d vs bo19L |
|---|---|---|
| bo19L / lab b | 4.17 / 4.17 | 0 / +0.04 |
| LOOPU+PIN / LOOPR+PIN | 4.15 / 4.16 | -0.00 / +0.04 |
| LOOPU+PIN+DEC1 / DEC2 | 4.20 / 4.15 | -0.02 / -0.00 |
| no compute / no escape warps / LOOPU no escapes | 3.89 / 3.96 / 3.98 | -0.26 / -0.19 / -0.18 |
qkv32 (XCL 4, 30 reps): bo19L 8.64, b 8.72, LOOPU+PIN 8.76 (+0.11), DEC1 8.87 (+0.22); qkv32 (XCL 6, 40 reps): b 8.72,
LOOPR+PIN 8.77, DEC2 8.82. pooler32: b 13.00, LOOPU+PIN 13.06, LOOPR+PIN 12.94.
### iteration-1 analysis block
1. nsys: not run (no candidate ahead; see iteration 4 for the in-graph comparison).
2. ncu: not run this iteration (iteration 4 has b / no-decode / LOOPR for all three shapes).
3. SASS: o16 b 54-60 regs, LOOPU 56, DEC1/2 60, 0 spills, LOCAL 0 everywhere (NVRTC 13 on the box agrees: 54-60);
   instruction mix per stage in the tables above; no LDG in any consumer path (W bulk copies, x 3D TMA).
4. Binding unit: not the ALU pipe — 20-25 % fewer ALU ops and 2 ALU ops less per 16 weights change nothing. Next:
   ablations to find what the compute overhead (full - no compute) is made of.

## iteration 2 — what the compute overhead is made of (ablations)
Same run per shape (tdf, XCL 4, 25-30 reps), local us:
| variant | qkv32 | pooler32 | o16 |
|---|---|---|---|
| bo19L | 8.64 | 12.96 | 4.17 |
| lab b | 8.72 | 13.00 | 4.17 |
| no escape warps | 8.71 (+0.01) | 12.77 (-0.15..-0.28) | 3.96 (-0.15..-0.19) |
| no compute (ring + escapes) | 8.40 (-0.19..-0.27) | 12.29 (-0.71..-0.76) | 3.89 (-0.28) |
| pure consumer (no traffic, no escapes) | 6.07 | 9.55 | 1.79 |
| pure consumer, LOOPU+PIN | 6.45 (+0.38) | 9.67 | |
| pure consumer, LOOPR+PIN | | 9.72 | |
| 8 consumer warps (24,3,1,8,1 / 32,2,2,4,1): full / pure | 9.05 (+0.37) / 5.74 (-0.33) | 13.01-13.16 (0 / -0.07) / 9.06 (-0.56) | |
| NACC 2: full / pure | 8.66 / | 13.22 (+0.16) / 9.66 | |
| no decode (raw words to MMA): full / pure | 8.63 (-0.04) / 5.93 (-0.16) | 12.98 (-0.10) / 9.63 (+0.03) | 4.12 (0.00) / 1.65 (-0.14) |
Reading: qkv32 is ring-bound (the ring alone is 8.40; the consumer alone 6.07); o16 is stream-bound (pure consumer
1.8 us of a 4.1 us kernel; escape warps are half of the compute overhead); pooler32 has the largest compute overhead
(~0.7 us = the consumer backlog after the last-stage burst), but decode is ~free inside it.
### iteration-2 analysis block (pooler32; ncu full set, flushed, box 0, ncu_run.sh)
1. nsys: not run (no candidate).
2. ncu: lab b (= pu5 cfg) dur 16.10 us (ncu-inflated), DRAM 50.94 MB, issue 28.3 %, hmma 8.5 %, occ 13.3 %, elig 0.32,
   clk 1.73 GHz, stalls long_scoreboard 43 %, wait 23 %, selected 13 %, short_sb 7 %; pure consumer (no traffic, no
   escapes) dur 11.36, issue 34.8 %, hmma 11.9 %, elig 0.36, clk 1.82, stalls wait 38 %, selected 29 %, long_sb 8 %,
   short_sb 7 %, dispatch 5 %.
3. SASS: pooler32 b 80 regs (NVRTC 13), LOOPU 88, 0 spills, LOCAL 0; loop mixes in iteration 1.
4. Binding unit: full kernels = the ring (long_scoreboard = waits on full barriers, DRAM 4.5-4.8 TB/s inside the
   stream) + a consumer tail; the pure consumer is latency-bound on one warp per SMSP (wait / selected), not ALU.

## iteration 3 — escape mechanisms (the only measured consumer-side lever: 0.10-0.19 us at o16 / pooler32)
All variants exact (tdf one-hot on escape columns bit-exact, normal / outliers within tolerance) unless noted.
| mechanism | o16 | pooler32 | notes |
|---|---|---|---|
| F's per-tile escape warps (base) | 4.17-4.28 | 13.0-13.4 | x read from the ring slot, warps in every stage's release count |
| none (ablation, wrong) | -0.10..-0.19 | -0.15..-0.28 | ceiling |
| NTL idle warps instead (ablation) | -0.12 (= none) | -0.22 (= none) | warp count itself costs nothing |
| ESCG: 1 warp, x by ld.global at start | +1.9..+2.3 | | generic-load hazard (fact 1); same code without loads -0.13, empty warp -0.08 |
| ESCG, loads after stage 0 landed / plain ld.global | +2.35 / +1.9 | | |
| ESCT: 1 warp, x columns by 2D TMA (box 8 x M) at start | +0.39..+0.56 | | fact 2 |
| ESCL: 1 lazy warp (relay barrier), smem RMW accumulation | +0.32 | +1.24 | one serial warp gates the releases |
| ESCP: per-tile lazy warps (F's register path) + relay barriers | +0.10..+0.22 | +0.42..+0.55 | escape stages held longer |
| ESLP + EBAR: F warps sleep-wait + bar.arrive instead of cbar | -0.01..-0.05 vs b | -0.10 vs bo19L (noise) | |
First ESCL builds hung / crashed (two bugs: escape warps counted twice in the thread count; parity aliasing when a
skipping warp gets 2 ring rounds ahead) — always smoke-tested under `timeout 60-80`, no box impact.
### iteration-3 analysis block
1. nsys: iteration 4 (ESLP+EBAR 'es').
2. ncu: not run for the losing mechanisms (losses of 0.3-2 us are outside any profiling subtlety).
3. SASS: ESCG warp code ~800 instr (ESCG v1 2400 with shuffles / collectives); ESCT/ESCL/ESCP 64-108 regs (escape
   warps raise the kernel's register count), 0 spills, LOCAL 0.
4. Binding unit: escapes cost their per-stage participation (~0.1-0.15 us); every decoupled design adds latency to
   the release -> refill path or touches x outside the ring. Kept: F's design.

## iteration 4 — final measurements: decomposition, ncu pipes, nsys
### Pure-consumer / full decomposition (pooler32, LOOPR+PIN reference, XCL 4, 25 reps)
| removed | full kernel | pure consumer |
|---|---|---|
| nothing | 13.23 | 9.96 |
| HMMA (folded into LOP3+FADD) | 13.00 (-0.15..-0.23) | 8.10 (-1.86) |
| LDSM (x operands constant) | 13.25 (-0.06) | 8.63 (-1.33) |
| decode | 13.09 (-0.14) | 10.01 (+0.05) |
| all compute | 12.52 (-0.66) | |
### ncu (full set, flushed, clock control none, box 0; ncu_df.sh) — base vs no decode
| kernel | dur | DRAM rd | issue % | ALU pipe % | FMA pipe % | LSU % | HMMA cycles % | L1/smem % | elig | clk | instr | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| o16 b | 6.27 | 12.78 MB | 25.1 | 23.4 | 5.0 | 8.7 | 9.6 | 28.5 | 0.35 | 1.85 | 1.005 M | long_sb 49, wait 17, selected 9 |
| o16 no decode | 6.30 | 12.78 | 18.6 | 12.1 | 4.4 | 8.4 | 9.6 | 28.1 | 0.23 | 1.81 | 0.744 M | long_sb 50, wait 14, no_instr 9 |
| o16 LOOPR+PIN | 6.30 | 12.78 | 23.0 | 21.2 | 4.2 | 8.5 | 9.7 | 28.5 | 0.30 | 1.82 | 0.919 M | long_sb 47, wait 14, selected 10 |
| qkv32 b | 10.66 | 28.89 | 26.5 | 22.0 | 3.2 | 11.3 | 26.9 | 41.3 | 0.29 | 1.83 | 1.556 M | long_sb 39, wait 19, selected 16 |
| qkv32 no decode | 10.50 | 28.89 | 17.8 | 10.1 | 3.3 | 12.9 | 30.6 | 46.9 | 0.19 | 1.78 | 1.101 M | long_sb 44, wait 23, short_sb 10 |
| pooler32 b | 15.87 | 50.95 | 29.6 | 25.9 | 5.4 | 13.6 | 25.9 | 50.1 | 0.34 | 1.84 | 3.557 M | long_sb 45, wait 22, selected 15 |
| pooler32 no decode | 16.13 | 50.94 | 22.0 | 13.0 | 5.1 | 13.6 | 26.0 | 50.2 | 0.24 | 1.76 | 2.742 M | long_sb 44, wait 24, short_sb 12 |
Occupancy 25 % (o16, 2 CTAs x 9 warps), 11.9 % (qkv32), 13.5 % (pooler32); regs 56 / 104 / 80; 0 local.
### nsys in-graph (box 0, one process, 15 reps, flushed; single x placement per candidate: +-0.3 us placement noise)
| shape | bo19L | lab b | no decode | ESLP+EBAR |
|---|---|---|---|---|
| o16 | sk11_o_16 5.15 | 5.22 | 5.25 | 5.15 |
| pooler32 | pu5_pooler_32 14.34 | 14.18 | 14.21 | 14.27 |
| qkv32 | sk11_qkv_gate_32 9.57 | 9.89 | 9.57 | 9.98 |
### Final same-run check of the closest candidates (qkv32, XCL 6, 40 reps): bo19L 8.63, b 8.72, no decode 8.61,
LOOPR+PIN 8.77, DEC2 8.82, ESLP+EBAR 8.70 -> nothing ahead of bo19L.
### iteration-4 analysis block
1. nsys: table above — no variant ahead of bo19L beyond placement noise; removing the decode moves nothing.
2. ncu: table above — the decode is ~half of the ALU pipe's work but the pipe runs at 22-26 % of peak; removing it
   halves ALU utilization and cuts issue by 7-8 points with no duration change; HMMA pipe 10-27 % active; L1/smem
   28-50 %; the dominant stall is long_scoreboard (waiting for ring stages) 39-50 %.
3. SASS: regs 56 / 104 / 80 (NVRTC 13), 0 spills, LOCAL 0, no LDG in any consumer; decode 7 ALU per 8 weights.
4. Binding unit and distance: o16 and qkv32 = the TMA ring's DRAM stream + startup/tail (qkv32 ring-only 8.40 vs
   LDG-floor ~6.8: the 1.2-1.6 us "over the floor" belongs to the ring, not the consumer); pooler32 = ring + a 0.6-0.7
   us consumer tail whose per-stage time is set by HMMA / LDSM latency on one warp per SMSP (pure consumer: HMMA 1.86,
   LDSM 1.33, decode ~0 of 9.96 us), plus escape warps 0.10-0.19. What could still pay (not decode): (a) the ring
   itself at qkv32 / o16 (start + last-1 % skew + epilogue, ~1 us; agents T / P5 measured its anatomy); (b) the
   pooler32 tail: more independent work per SMSP at the END only (e.g. split the last 2-3 stages across idle warps),
   since 8 consumer warps throughout cost ring throughput; (c) a release-path-free escape scheme worth <= 0.15 us
   still needs x from the ring (TMA / LDG alternatives measured above all lose).
