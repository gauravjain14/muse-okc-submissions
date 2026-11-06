# C2 — M = 1 qkv_gate / pooler: compact exact escape slots on F11 (base bo9 = agent G's g311), 2026-11-02

## RESULT: no win. bo9 keeps qkv_gate / pooler at m = 1.
Deliverable `gemvc.py` (`prepare_gemvc(weights, enable=False)`, `run_gemvc` -> None for every shape by default) and
`sub_gemvc.py` (bo9 + gemvc). `enable=True` (env `GEMVC_ENABLE=1` in sub_gemvc, or `sub_gcl.py`) builds the best exact
compact-slot kernel for reference/testing only.

Final same-run, placement-averaged table (claude/t3/N/t_ab.py, box 1, 6 graphs per candidate x 40 reps, local us;
judge ~= local + 1.0):
| shape | SGLang | bo9 (G g311) | gemvc (exact compact, NB 4) | speedup vs SGLang bo9 -> gemvc | int_v7 speedup |
|---|---|---|---|---|---|
| decode-1 qkv_gate m=1 | 13.02 | **7.04** [7.01-7.07] | 7.30 [7.23-7.41] (+0.27) | x1.849 -> x1.784 | x1.713 |
| decode-1 pooler m=1 | 19.97 | **11.62** [11.55-11.71] | 12.48 [12.38-12.61] (+0.79) | x1.719 -> x1.600 | x1.579 |
Repeat with the G control (sub_g1l.py) and gemvc built at 40 regs (MB 6): qkv 7.20 vs 7.62, pooler 11.71 vs 12.45.
Correctness of the (disabled) exact kernel on the H200: `GEMVC_ENABLE=1 check.py sub_gemvc.py --layers qkv_gate,pooler
--graph` -> "checked 103 cases x 2 seeds + graph: ALL OK"; `exact_all.py sub_gcl.py --layers qkv_gate,pooler --ms 1
--nesc 200` -> EXACT ALL OK; `t_lab.py --exact` (200 injected columns per layer, one-hot each): gemvc 0 mismatches,
G's g311 73-76 mismatching columns (it flushes the injected 2^-40 / -1e-30 / -2^-126 values; exact_all at m = 1 only
probes ONE injected column, so it passes g311 by luck — use t_lab --exact for m = 1 exactness).

Why it loses (measured, below): G's g311 sits exactly at 40 registers / 6 CTAs per SM with every load issued before
the first FFMA; that kernel is at the hardware-scheduled LDG streaming floor for its bytes. Every compact-slot variant
needs a few extra instructions/registers; ptxas then either (a) sinks a W/x/slot load below the FMA chain (+0.9..+3 us),
(b) goes to 43-48 registers = 5 CTAs/SM (+0.5..+0.9 us), or (c) adds work to the per-CTA tail after the barrier
(+0.2 us per 8 smem loads). The byte saving available (pooler -0.54 MB / qkv -0.23 MB with 4-byte slots in buckets,
-1.0 / -0.5 MB with 3-byte slots) is worth only ~0.05-0.2 us at 4.56 TB/s. The best equal-bytes reproduction of g311
inside my build (`g65`: G's body + anti-sink) matches it (±0.05-0.2) but nothing beats it.

## Tools (claude/t3/C2)
- `lab.py` lab kernels + packers: FMT 3 (3-byte slots: u16 k<<3|class + u8 s|m, groups of 8, ext list), FMT 4 (u32
  slots), FMT 5/7 (G-style u32 pools, bucketed, two-term flag), FMT 6 (`g6`: G's g311 body verbatim, compile-time E/RB,
  bisection bits GB), FMT 8/9 (G body on bucketed blobs / one specialised body per bucket), FMT 10 (`gr`: G's R-row
  CTA body), ablation bits (ABL) per kernel, anti-sink helpers `pm0` / `pmall`.
- `t_lab.py` same-run placement-averaged timing of lab variants + G control (`--g1`, `--g1b`), `--check`, `--exact`
  (inject 400 out-of-window values, one-hot 200 injected columns), `--sass` (SASS stats of the REAL build: regs, LDG,
  late = LDGs after the first FFMA, lateNA = W/slot loads after it).
- `sass_lab.py` NVRTC-13 SASS stats with fake buckets (CPU; NOTE: fake buckets can hide spills — iteration 2).
- `ncu_lab.py` ncu driver/parser; `ncu_src.py` per-SASS-instruction executed counts + stall samples.
- `sub_gcl.py` lean candidate (gemvc enabled, torch elsewhere), `sub_g1l.py` lean G control, `sub_gemvc.py` bo9 + gemvc.

## iteration 0 — baseline (bo9 m=1 = G's g311: 1 row x 256 thr per CTA, 32 weights/thread, 40 regs, mb 6)
- Time (t_lab, same runs): pooler 11.20-11.62, qkv 6.82-7.09 local us (14 runs).
- nsys in-graph (nsys_shapes, 20 reps): g1_qkv_gate_g311 8.16 us, g1_pooler_g311 12.61 us.
- ncu (full set, flushed, boost clock; c2n3/c2n4/c2fin): pooler dur 13.95-15.5, DRAM read 48.65 MB = 80 % of 4.8 TB/s
  over the in-graph 12.6 us (3.85 TB/s), L2 67-75 %, SM 42-45 %, issue 55-56 %, tensor 0 %, occupancy 64-65 % (48 warps),
  eligible 1.3-1.44 warps/clk, L1 throughput 45.7 %, LSU pipe 21 %, stalls long_scoreboard 53-59 %, barrier 11-12 %,
  not_selected 7 %. qkv: dur 9.09, 27.30 MB, L2 66 %, SM 39 %, issue 52 %, occ 63 %, elig 1.27, long_sb 56 %.
  6.51 M warp-instructions (199 per warp) for pooler.
- SASS (NVRTC 13.0): 40 regs, 0 spill, 240 instr, 11 LDG (5 W, 1 slot, 4 x, 1 escape gather), all before the first FFMA
  (the gather is hoisted too); per 32 weights 33 decode ALU + 16 PRMT + 32 FFMA + 64 bf16->f32 unpack; tail = BAR + 8 LDS
  + 8 FADD + STG.
- Binding unit: DRAM streaming latency at the hw-scheduled LDG floor (probe: 50.3 MB in 11.17-11.41 us -> ~10.9-11.0 for
  48.6 MB; g311 11.2-11.6). The remaining lever is bytes: 2.42 MB of slots (E = 148-152 padded to the row max vs mean 111
  escapes = 1.36 %) out of 48.6 MB.
- Format analysis (local): escapes by exponent: e = E_lo+8 0.474 %, E_lo-1 0.44 %, -2 0.22 %, ... (geometric), e <= E_lo-8
  7e-5. Fixed-length joint exponent codes (top-N tuples, tables) do not beat 3 bits/exponent: pairs in 5 bits cover 90.7 %,
  triples in 8 bits 92.9 %, quads in 11 bits 94.0 % (entropy 2.54 b) -> F11 stays; only the slot pool can shrink.
  Slot bytes (pooler): G 2.42 MB; u32 exact-count 1.84; u32 in 8 buckets 1.88-1.92; 3-byte slots in buckets 1.47-1.55.

## iteration 1 — FMT 3/4 compact slots in a SEPARATE array (rows sorted, bucket constants in code)
- t_lab: pooler +2.3..+3.1 us, qkv +1.1..+2.0 vs g1. Ablations (same run): no escapes -0.8..-0.96; slots loaded but not
  used +1.0; store to out[b] instead of permuted row: no change.
- Diagnosis: the slot loads are a second DRAM stream away from the W stream; under the saturated sequential W stream they
  are served late (FR-FCFS row-hit priority) -> the escape step waits ~a DRAM round trip per CTA. => slot bytes must ride
  inside each CTA's contiguous blob (G's layout does this).
- SASS: 32 regs (no min-blocks) / 40 (mb 6) with W + x loads sunk below the first FFMA (late 7-8) -> mb 5 (48 regs) needed.

## iteration 2 — contiguous per-CTA blobs (main | hdr | ext | slots), per-bucket runtime caps
- t_lab: still +1.2..+2.2 (pooler), x-first variants up to +5.5 us. ncu: warp-instructions 8.2-10.4 M vs 6.5 M (G),
  issue 60-65 %, barrier stalls 22-25 %; per-instruction counts (ncu_src) show STL/LDL local-memory traffic: the 8-way
  bucket select chain spilled at 40 regs in the REAL build (my SASS tool used 4 fake buckets and showed stack 0).
- Fix attempts: b = blockIdx.x (uniform), select chains -> still 25 ISETP/SEL + duplicated recomputation.

## iteration 3 — compile-time slot roles (XC ext threads, SMAX slot threads), piecewise-linear offsets (VIADDMNMX),
over-read + hdr-count gating, escape correction BEFORE the chain (no slot register live across it)
- SASS: 40 regs, 0 spill, all W/x/slot loads before the first FFMA (late 1-2 = gathers), 312-328 instr.
- t_lab: no-escape ablation n8 10.85-11.07 (bytes only: 46.5 MB) vs g1 11.36-11.42; with escapes 12.2-12.6 (+1.1 over n8).
  ncu (c2n2): inst 8.16 M vs 6.51 M (+50 per warp), issue 65 % vs 55 %, the escape step waits on hdr (19 % of stall
  samples on the first hdr use). hdr-first / gathers issued before hdr arrives: -0.2..-0.3 only.

## iteration 4 — G-style u32 pools in buckets (FMT 5: word 0 = row, fillers, two-term flag bit 31)
- t_lab: +1.3..+1.9 us. SASS of ablations (t64, t256): the SLOT load sunk into the FMA chain (lateNA 1) -> +2..+3 us.
- Anti-sink trick: start the FMA chain from +-0 whose sign depends on the loads that must be issued early
  (`pm0(x quarters, slot)` = 2 LOP3 + AND; `pmall` = every load). With pm0 the slot load stays early; still +1.3..+1.9.
- L1 check (ncu c2n4): L1 throughput 44-46 %, LSU pipe 19-21 %, identical wavefronts/requests to G -> not L1-bound.

## iteration 5 — bisection from G's verbatim body (`g6`, compile-time E/RB/scale)
| variant (pooler, same run) | SASS | local us | vs g1 |
|---|---|---|---|
| g1 (G's own build, runtime params) | 40 regs, late 0 | 11.26-11.62 | 0 |
| g6 = G body, compile-time constants | 38 regs, 2 x loads after the 1st FFMA | 12.21-12.37 | +0.9..+1.1 |
| g65 = g6 + pm0 anti-sink (mb 5) | 40 regs, late 0 | 11.30-11.62 | +0.00..+0.16 |
| + unconditional gathers (GB1) / branch-free select (GB4) | 40 regs | +0.02 / +0.02 | ~0 |
| + second smem partial for two-term sums (GB2) | 40 regs | +0.23 | tail +8 LDS/FADD |
| + output row from slot word 0 (GB8, valid permutation) | 40 regs | +0.05..+0.25 | |
| x via coherent ld.global / L1::evict_last | 40 regs | ±0.05 | |
| L2 prefetch of x ahead of W | | +9 us | prefetch bypasses L1: 64 MB of L2 requests |
| vectorised tree tail (2 LDS.128) | 40 regs | ±0.05 | |
| 32-reg builds (7-8 CTAs/SM) | W loads sink | +1.4..+1.7 | |
Fact: x loads issued after the first FFMA cost ~1 us even though they hit L1 (in-order texture return behind the SM's
W misses); G's checker only counted .NA (W) loads — count every LDG.

## iteration 6 — gz: G's body + bucketed u32 pools + trailing two-term bucket (FMT 7 = shipped gemvc kernel)
- Escapes with e in [E_lo-16, E_lo-1] U [E_lo+8, E_lo+22] U zero use G's delta form (exact for power-of-two one-hot x);
  others (2 escapes in the real pooler, 0 in qkv) put their row into a trailing bucket that uses the two-term form.
- SASS (real build): MB 6 -> 40 regs, lateNA 0 with pmall, 248 (qkv) / 288 (pooler, slow path compiled in) instr;
  MB 5 -> pooler 43 regs (5 CTAs/SM), qkv 40.
- ncu (c2fin, flushed, no graph): pooler 15.04 us vs g1 15.20, DRAM 48.11 vs 48.65 MB, occupancy 54 % vs 65 % (43 regs),
  issue 60 % vs 56 %, inst 7.49 M vs 6.51 M, elig 1.63 vs 1.43, long_sb 52 %; qkv 9.15 vs 9.09, 27.07 vs 27.30 MB, 40 regs,
  occ 64 %, issue 56 %. (Under ncu, without the graph/placement, pooler is even slightly faster; in-graph it is not.)
- nsys in-graph (20 reps): gc_qkv_gate 8.30 vs g1 8.16; gc_pooler 13.82 vs g1 12.61.
- Binding unit: DRAM stream at 3.5-3.85 TB/s in-graph (73-80 % of 4.8 TB/s); the bytes saved (-0.54 / -0.23 MB) are
  smaller than the lost occupancy/latency hiding (pmall makes the chain wait for every load; 43 regs drop a CTA/SM).

## iteration 7 — other structures
- R = 2 rows per CTA thread-shared x (G's body, 62 regs, 4 CTAs/SM, x traffic halved): pooler +0.07..+0.38, qkv
  +0.16..+0.22 (as G found).
- One compile-time-specialised G body per bucket in one kernel (uniform branch): ptxas sinks W loads in the bodies
  (+2.3 us). Two rows per 512-thread CTA (RPC 2): W load sunk, +3 us.

## Dead ends / facts for others
- Escape metadata outside the CTA's contiguous W blob: +2 us (off-stream DRAM requests starve).
- Bucket parameters via select chains spill at 40 regs; check SASS of the REAL build (fake inputs hid it).
- ptxas load sinking also hits x loads and the slot load; NVRTC compile-time constants vs runtime params flip it.
  Robust fix: make the FMA chain's start value depend on the loads (pm0/pmall, 2-5 LOP3) — but pmall (every load) delays
  the chain until all data land (+0.3..+0.6 us); pm0 (x + slot) is free when W is not sunk.
- Anything after the CTA barrier is paid by every CTA (multi-wave kernel): +8 LDS/FADD = +0.2 us on pooler.
- Fixed-length sub-11-bit exponent codes are not viable (coverage table in iteration 0).
- exact_all at m = 1 probes a single injected column per layer: use t_lab --exact for m = 1.
