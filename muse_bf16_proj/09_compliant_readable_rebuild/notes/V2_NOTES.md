# Agent V2 — qkv_gate W[2304, 8192] at m = 8 / 16 / 32: beat bo10 (F's sk11)

## RESULT (read first)
- SHIPPED: claude/t3/V2/qkvb.py (prepare_qkvb / run_qkvb; generated from skv.py by mkqkvb.py) + sub_qkvb.py (bo10 +
  qkvb) + sub_qkvb_only.py (qkvb + torch). Owns qkv_gate m = 8 and m = 16 only (run_qkvb returns None elsewhere, so
  bo10 keeps m = 32 and every other shape). Same geometry as bo10's sk11 (R = 24, 96 CTAs, 5-stage ring) plus:
  (1) XLAG = 1: the producer issues the x tile of stage s together with the W block of stage s + 1, i.e. x lags W by
  one stage in each SM's TMA queue (prologue issues only W(0)); (2) baked W / escape-list pointers + M / N / scale.
- bench.py same run vs bo10 and SGLang (local us; judge ~= local + 1.0), 8 processes on both boxes, 60 reps:
  qkv m=16: bo10 8.20-8.54 -> qkvb 7.85-8.26, mean -0.36 us, 8/8 wins, speedup vs SGLang x1.606 -> x1.678 (+4.5%).
  qkv m=8:  bo10 7.84-8.05 -> qkvb 7.66-7.89, mean -0.13 us, 7 wins + 1 tie, x1.706 -> x1.734 (+1.6%).
  (decode-16 load: qkv is 1 of 5 shapes, ~+0.8% load time; verify-1 load ~+0.4%.)
- Correctness: check.py claude/t3/V2/sub_qkvb.py --graph (all layers, bo10 + qkvb) -> "checked 259 cases x 2 seeds +
  graph: ALL OK"; exact_all sub_qkvb_only.py --layers qkv_gate --nesc 200 and --nesc 3000 -> EXACT ALL OK, and qkvb
  verified to serve m = 8 / 16 itself with 3000 injected escapes (4350 escapes, max 27 per 8-row tile, list cap 32,
  one-hot mismatches 0). prepare_qkvb < 1 s, never raises.
- m = 32: no win. Consumer-bound at R = 24 (3 n8 tiles x 16 stages per CTA, loop ends tight at 8.1-8.3 us) and at the
  per-SM ingress limit (807 KB/SM); every way to cut per-SM bytes either keeps 3 tiles of consumer work (R = 18) or
  needs a cross-CTA reduction that costs 1.2-2.8 us (global) / ~0.6 (DSMEM, agent H). XLAG = 1 costs +1.3 us there.
- Hazard found (applies to every TMA-ring kernel): generic global loads from a warp in a CTA with >= ~200-215 KB of
  shared memory and in-flight TMA traffic corrupt whole CTAs at random (iteration 6). Clean at <= 181 KB.
- Integration: `out = run_qkvb(x, cfg, state_V2)` before bo10's sk11 dispatch for layer qkv_gate, 2 <= m <= 32.

Base kernel for these shapes in bo10 = agent F's sk11 (claude/t3/F/sk11.py), cfgs (R, NT, NWR, WSK, UPW, NACC, cap, fl):
m8 (24,3,1,4,2,2,144,F12), m16 (24,3,1,4,2,2,200,F12), m32 (24,3,1,4,2,1,208,F12|bake): 96 CTAs x 24 rows, 4 consumer
warps (K split), 1 producer warp, 3 escape warps.

Files (claude/t3/V2/): skv.py (sk11 F12 path + V2 options; prepare_skv / run_skv), tv.py (same-run timing harness:
`v:` = skv variant, `f:` = F's sk11 = bo10's kernel; judge-style graph replay after a 512 MB flush, empty graph
subtracted, interleaved; CHECK=1 normal + outliers + one-hot exact; ABL=1 skip self-tests; flag 2 = per-CTA trace with
epilogue phases and SM clock), sass.py (local NVRTC sm_90a build: regs / spills / SASS), dbg.py (error pattern).

skv cfg = (R, NT, WSK, UPW, NACC, cap KB, flags, KS); PR = R - 8 (NT - 1) real rows in the CTA's last n8 tile (only those
rows' bytes are in the W stream). Flags: 4 bake (W, escape list, split-K scratch pointers + M/N/scale as constants),
2 trace, 16 no compute, 128 no escape warps, 1<<29 no W, 1<<30 no x, bits 8-9 initial fill depth, 2048 no output
store, 16384 no epilogue, 8192 software-pipelined consumer, 65536 escape warps read x from global at start,
32768 single-RMW split-K protocol, 4096 out pointer via smem, 1024 late params.

All times: local us (judge ~= local + 1.0), tv.py same-process A/Bs unless noted. Per-process / per-kernel-instance
offsets of 0.3-0.5 us exist (agent H); single-run deltas below 0.3 us are not conclusions.

## iteration 1 — byte-balanced rows: R = 18 (128 CTAs, partial third tile, 221 KB W per SM instead of 295)
Correct on the first build (one-hot exact). Same run (box 0, 40 reps):
| variant | m32 | m16 | m8 |
|---|---|---|---|
| F sk11 (bo10) | 8.42 | 8.13 | 7.87 |
| skv R=24 (same cfg as F) | 8.35 | | |
| skv R=18 | 8.83 | 8.29 | 8.03 |
| ring only (no compute) R=24 / R=18 | 8.51 / 8.11 | | 7.41 / 7.42 |
| ring, no x (W only) R=24 / R=18 | 7.44 / 7.44 | | |
| ring, no W (x only) R=24 / R=18 | 4.53 / 4.88 | | |
Per-CTA trace (box 1/0, flag 2, median of 7 replays; us from first CTA start; trace adds ~0.3 us):
| variant | first stage | loop end min/med/p90/max | CTA end max | SM clock |
|---|---|---|---|---|
| m32 R=24 full | 1.47 | 8.10/8.16/8.22/8.32 | 8.90 | 1.73 GHz |
| m32 R=18 full | 1.50 | 8.19/8.35/8.48/8.54 | 9.22 | 1.70 |
| m32 R=24 no compute | 1.50 | 7.30/7.46/8.02/8.48 | 9.06 | 1.92 |
| m32 R=24 no x (W only, no compute) | 1.41 | 6.82/6.98/7.12/7.26 | 7.78 | 1.94 |
| m32 R=24 x only | 0.90 | 4.00/4.03/4.06/4.19 | 4.70 | 1.94 |
| m8 R=24 full | 1.70 | 7.20/7.42/7.60/7.81 | 8.19 | 1.72 |
| m8 R=18 full | 1.54 | 7.20/7.49/7.68/7.81 | 8.32 | 1.71 |
| m8 R=24 no compute | 1.50 | 6.78/6.94/7.09/7.20 | 7.78 | 1.91 |
Reading:
- The W-only ring is identical at 96 and 128 CTAs (7.44): DRAM-bound (28.3 MB from the first landing at 1.41 us to
  ~6.98 us median = 4.8 TB/s). Byte balance does not help the W stream.
- x costs ~1.0 us at m32 and ~0.6 at m16 (ring 7.97 vs 7.38 no-x). Per-SM ingress model (W + x share ~64 B/clk/SM ~
  100-110 GB/s, agent H): m32 R=24 807 KB/SM, R=18 733 KB/SM -> matches the no-compute rings 8.51 vs 8.11.
- But the m32 consumer is binding at R=24 (loop ends tight at 8.1-8.3 while the ring's median CTA has data at 7.46)
  and R=18 keeps 3 n8 tiles of decode + HMMA per CTA (the partial tile costs a full tile's instructions), on 128 SMs
  at a lower clock (1.70 vs 1.73 GHz): the R=18 consumer ends later (8.35 med) -> R=18 loses at every m.
- Clock: 1.70-1.73 GHz with HMMA, 1.91-1.94 without (power cap).

## iteration 2 — epilogue: params, output stores (flags 1024 / 4096 / 2048 / 16384)
Trace showed bar2 -> CTA end = 0.22 (m8) / 0.32 (m32) us for the final output loop. SASS: ptxas rematerialises the
early `p.out` read as ULDC.64 c[0x0][0x2d0] at the store (asm mov is transparent to ptxas); store is a generic ST.E.
Staging the pointer through smem (4096) or baking M/N/scale: no change (8.75/8.85 vs ..., noise). Bench ablations
(same run, box 0): no epilogue at all (16384) saves only 0.13 (m8 7.89 vs 8.02) / 0.27 (m16 8.03 vs 8.30) / 0.23
(m32 8.62 vs 8.85) -> the epilogue is a minor lever; initial fill depth limits (bits 8-9: D=1 / D=2) cost +2.3 / +0.6
(m32) and +0.6 (m8/m16): DRAM needs the full initial burst.

## iteration 3 — 8 consumer warps (WSK 8, UPW 1: 2 warps per SMSP, half the units per warp per stage)
Slower everywhere: m32 9.81-10.0, m16 9.62, m8 8.58 (vs F 8.4-8.9 / 8.1-8.4 / 7.9-8.1). Same conclusion as agent X.

## iteration 4 — split-K KS = 2, R = 36 (128 CTAs, 40 instead of 48 tile-stages per CTA, x per SM halved)
Global scratch + last-arriver (deterministic fixed-order sum). Correct (one-hot exact). Trace (m32): loop end
7.78/7.90/8.00/8.32 (0.25 earlier than R=24) but first stage 1.95 (60 KB stages) and the reduction (bar2 -> CTA end)
1.2-2.8 us; single-thread acq_rel RMW instead of per-thread fences (32768): still 1.2-2.5 us under the saturated memory
system. Bench m32 10.5-11.1, m16 10.8, m8 9.9 -> dead (agent H measured ~0.6 us for a DSMEM exchange; also too much).
Explains H's iteration-2 puzzle: every split H/X tried kept 48 tile-stages per CTA (R=24 full K, R=48 half K, R=96
quarter K) and the m32 kernel is consumer-bound, hence "halving x per SM did not shorten the stream".

## iteration 5 — software-pipelined consumer (flag 8192)
Next unit's W/x loads issued before the current unit's decode + HMMA, next stage prefetched when its full barrier tests
complete, slot released after its last loads. Correct; 136-167 regs, 0 spills. Much slower: m32 9.76 vs 8.35, m16 8.99
vs 8.35, m8 8.85 vs 7.87. SASS: ~50 predicated address / branch instructions per stage (slot-dependent addresses,
BSSY/BSYNC around the conditional prefetch); the consumer pays ~1:1 for added instructions (as F / X found).

## iteration 6 — escape warps off the ring (flag 65536: corrections from x read from global at kernel start)
Bit-identical math (list order). Correct at m8 (cap 144, 141 KB smem) and m16 (cap 200, 181 KB) but WHOLE CTAs come out
wrong (random CTAs per run, all rows and tokens) at >= ~215 KB of shared memory (m32 cap 200/208, m16 cap 208/220);
clean at 165 KB. Bisect (dbg.py): skipping the global loads -> clean; skipping the correction stores -> still corrupt;
plain ld.global instead of .nc -> still corrupt. The loaded values are not even used in that ablation, so generic
global loads (L1 allocations) in a TMA-ring CTA with ~216 KB of smem corrupt the ring data. Agent X saw the same
symptom ("escape warps take x from global ... non-deterministic tile corruption (1-3 CTAs per run)"). Layout verified
in bounds (smem 216576 B, last byte used at sbase + 215552). RULE: no generic global loads in kernels with >= ~200 KB
of shared memory and in-flight TMA traffic.
Benefit at safe sizes: none (m8 8.02 vs 8.06, m16 cap 176 8.72 vs 8.66; m32 needs the 208 KB ring: cap 176 9.25-9.73).

## tv.py harness bug found and fixed
Each variant's cloned x was freed when the loop rebound `x`, while its captured graph (and the cached tensor map) still
pointed at it -> sporadic illegal-address faults during graph warm-up in multi-variant processes. Inputs are now kept
alive for the life of the graphs (`keep`).

## iteration 7 — ring depth / bytes in flight (multi-instance A/Bs: INST=3 byte-identical kernel instances per spec)
tv.py INST=n builds n byte-identical instances per spec and reports the mean of instance medians (agent H: per-instance
offsets 0.3-0.5 us). With it, my kernel at F's exact configs = F (8.67 vs 8.61, 7.95 vs 7.93) once the epilogue uses
F's float4 / uint2 path and a 4-word escape header (was 0.06-0.09 slower with float2 + 8-word header). Baking alone,
NACC 2 (m32 +0.2), UPW 1 (+0.15..+0.5) and 8 consumer warps: no gain.
Stage count is the strongest knob (same run, INST 2-3, both boxes):
| shape | 3 stages | 4 stages | 5 stages | 6 stages |
|---|---|---|---|---|
| m8 R=24 (F = 5) | | 8.05 (F 7.96) | F | 8.56 (cap 176) |
| m8 R=18 | 9.00 / 8.85 | 8.01-8.25 (F 7.96-8.20) | 8.15 / 8.13 | |
| m16 R=24 (F = 5) | | 8.36 (F 8.27) | F | 8.95 (cap 216) |
| m16 R=18 | 9.24 / 9.11 | 8.18-8.42 (F 8.22-8.50) | 8.55 / 8.47 | |
| m32 R=18 | 9.74 (F 8.91) | 9.08 (F 8.61) | | |
Reading: each SM needs ~(its ingress rate x ~1.5 us loaded latency) bytes in flight (m16 R=18: 80 GB/s -> 120 KB = 4
stages; 3 stages starve), but more total in flight across the GPU is slower (5-6 stages: +0.1..+1.1 us; agent H's
"more DRAM concurrency = worse"). R=18 with a 4-stage ring beat F at m16 in 6 of 6 tv processes (-0.05..-0.14) but
only -0.04 +- 0.15 in 8 bench.py processes -> not shipped.

## iteration 8 — x lags W in the producer queue (flag bits 22-23: XLAG)
Idea from iteration 7: the in-flight budget hurts, and x (an L2 hit, every CTA requests the same lines) needs much less
lead than W (DRAM). XLAG = L: the producer issues W(s) when slot s frees and x(s - L) right after it (prologue: W(0)
only). Correct (CHECK one-hot exact). Same run, INST 2-3, both boxes (local us):
| variant | m8 | m16 | m32 |
|---|---|---|---|
| F sk11 (bo10) | 7.90-8.05 | 8.28-8.46 | 8.57-8.64 |
| XLAG 1, F's config (5 stages; m32 4 stages) | 7.79-7.95 (7 of 8 processes below F, mean -0.07) | 8.14-8.28 (6 of 6 below F, mean -0.13) | 9.91-9.98 |
| XLAG 2 | 8.41-8.58 | 8.73-8.82 | 13.3-13.5 |
| XLAG 1 + one more stage (6) | 8.18-8.22 | 8.75-8.82 | |
| XLAG 1 on R=18 4 stages | | 8.91-9.18 | |
x may lag W by exactly one stage when the ring has 5 stages; with 4 stages (m32, R=18) x arrives late.

## iteration 9 — XLAG 1 + baked pointers (SHIPPED as qkvb)
tv (INST 3, two processes): m8 7.90 / 7.74 vs F 8.02 / 7.89 (XLAG 1 unbaked 7.95 / 7.88); m16 8.14 / 8.01 vs F 8.37 /
8.36 (unbaked 8.23 / 8.10). Baking removes tid 0's cold param-bank loads before the first W copy, which matters more
once x(0) no longer rides in the prologue.

### same-run table vs bo10 and SGLang (claude/bench.py, 60 reps, local us; judge ~= local + 1.0)
qkvb-only candidate (sub_qkvb_only.py) vs bo10, --layers qkv_gate --shapes 8,16:
| run (box) | SGLang m16 | bo10 m16 | qkvb m16 | SGLang m8 | bo10 m8 | qkvb m8 |
|---|---|---|---|---|---|---|
| 1 (0) | 13.47 | 8.54 x1.577 | 8.26 x1.632 | 13.60 | 8.05 x1.690 | 7.84 x1.735 |
| 2 (0) | 13.54 | 8.36 x1.620 | 7.94 x1.705 | 13.54 | 7.84 x1.728 | 7.83 x1.728 |
| 3 (0) | 13.65 | 8.45 x1.616 | 8.15 x1.674 | 13.58 | 7.97 x1.704 | 7.89 x1.721 |
| 4 (1) | 13.47 | 8.21 x1.641 | 7.95 x1.695 | 13.26 | 7.84 x1.692 | 7.78 x1.704 |
| 5 (1) | 13.42 | 8.29 x1.618 | 7.99 x1.681 | 13.39 | 7.86 x1.704 | 7.79 x1.720 |
| 6 (1) | 13.47 | 8.45 x1.595 | 8.00 x1.685 | 13.68 | 7.95 x1.720 | 7.79 x1.757 |
Integrated candidate (sub_qkvb.py = bo10 + qkvb) vs bo10, --shapes 8,16,32:
| run (box) | bo10 m8 | bo10+qkvb m8 | bo10 m16 | bo10+qkvb m16 | bo10 m32 | bo10+qkvb m32 (same kernel) |
|---|---|---|---|---|---|---|
| 7 (0) | 8.02 x1.725 | 7.81 x1.771 | 8.52 x1.570 | 7.99 x1.674 | 8.72 x1.537 | 9.14 x1.467 |
| 8 (1) | 7.88 x1.685 | 7.66 x1.734 | 8.20 x1.608 | 7.85 x1.679 | 8.83 x1.492 | 8.54 x1.544 |
(m32 is bo10's sk11 in both columns: its +0.42 / -0.29 is the per-instance offset, i.e. the noise floor of one run.)

### iteration-9 analysis block (shipped kernels qkvb_qkv_gate_8 / _16 vs bo10's sk11_qkv_gate_8 / _16)
1. nsys in-graph (one process, 20 reps, flushed; claude/t3/nsys_shapes.py): m8 sk11 8.72 vs qkvb 8.78 us, m16 sk11 9.18
   vs qkvb 9.14. One process cannot resolve < 0.3 us here (instance offsets; agent H saw nsys disagree with bench for
   skh too); the 8-process bench series and ncu below are the evidence.
2. ncu (--set full, flushed caches, clock-control none; same run; ncu-inflated durations):
| kernel | dur | DRAM rd | L2 % | L1/smem % | SM % | issue % | hmma % | occ % | elig | inst | L2 rd sectors (SM side) | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sk11 m8 (bo10) | 9.95 | 28.49 MB | 62.0 | 33.7 | 14.7 | 28.0 | 5.6 | 12.0 | 0.32 | 1.427 M | 1.005 M | long_sb 42, wait 19, selected 14 |
| qkvb m8 | 9.76 | 28.50 | 63.0 | 33.4 | 16.6 | 27.6 | 5.5 | 11.9 | 0.31 | 1.441 M | 0.998 M | long_sb 41, wait 21, selected 13 |
| sk11 m16 (bo10) | 10.30 | 28.63 | 60.0 | 34.3 | 15.6 | 27.0 | 5.6 | 12.0 | 0.30 | 1.434 M | 1.134 M | long_sb 40, wait 21, selected 16 |
| qkvb m16 | 9.98 | 28.63 | 62.4 | 33.0 | 16.3 | 26.7 | 5.4 | 11.9 | 0.30 | 1.448 M | 1.109 M | long_sb 40, wait 18, selected 15 |
| bake only (no XLAG) m16 | 10.37 | 28.63 | 59.8 | 31.3 | 15.7 | 27.3 | 5.1 | 11.9 | 0.31 | 1.434 M | 1.144 M | long_sb 42, wait 19 |
   Same DRAM bytes; with XLAG the L2 delivers at a higher rate (62.4 vs 60.0 %) with fewer SM-side read sectors; the
   bake-only control equals sk11 -> XLAG is the lever, bake adds a little on top (bench).
   Earlier ncu (same tool): sk11 m32 10.75 us, 1.507 M inst, hmma 10.3 %, L1/smem 47 %; skv R=18 m32 11.04 us, 2.009 M
   inst (+33 %: the partial tile costs a full tile on 128 CTAs), clock 1.80 vs 1.96; split-K KS=2 12.58 us (membar 6 %).
3. PTX/SASS (NVRTC sm_90a; local 12.1, sandbox 13): qkvb m8 / m16 80 / 80 regs locally (85 on the sandbox), 0 spills,
   LOCAL_SIZE_BYTES 0, smem 139776 / 180736 B. Consumer stage loop (m16) = sk11's: 162 instructions = 24 HMMA,
   8 LDSM.x4, 12 LDS (6 x .128 + 6 x .64), 93 decode ALU (48 PRMT, 32 LOP3, 13 SHF = ~0.97 per weight), 17 address
   IMAD/IADD3, 2 SYNCS (try_wait + arrive); no LDG / sunk loads (all data via TMA / bulk copy). Prologue: UBLKCP W(0) +
   UBLKCP escape list (no x); producer loop: UBLKCP W(s) then UTMALDG x(s - 1).
4. Binding unit: DRAM / per-SM delivery. 28.6 MB in 7.8-8.0 us local = ~3.6 TB/s averaged over the whole kernel; the
   W-only ring of the same geometry (iteration 1) streams at ~4.8 TB/s (100% of the practical 4.8) between the first
   landing (1.41 us) and the median loop end (6.98 us), so what is left is the first-stage latency (1.4-1.7 us, the
   initial burst of 5 stages x 96 CTAs) and the tail (consumer backlog of the last stages + epilogue 0.13-0.27 us +
   per-SM end skew 0.3-0.5 us). Tensor 5.5 %, issue 27 %, L1/smem 33 %: idle. Next: the first-stage latency (all
   attempts to shrink the initial burst - fill-depth limits, smaller rings - lost; a cheaper first stage needs a
   different source for stage 0) and, for m32, a consumer under 3 tiles per CTA without a cross-CTA reduction.

## Profiling coverage note
Iterations 1-8 are backed by same-run benches, per-CTA traces (start / first stage / loop end / epilogue phases / SM
clock) and SASS (regs, spills, loop mix); ncu full-set numbers were collected for the contenders only (sk11 m8/16/32,
R=18 m8/m32, split-K KS=2, bake-only, shipped qkvb m8/m16; table in iteration 9) and nsys in-graph for the shipped
kernels. Intermediate dead ends (8 warps, pipelined consumer, UPW 1, deeper rings, XLAG 2) were not ncu-profiled.

## Not done / ideas for others
- XLAG 1 is a producer-order change that applies to every sk11 / sk3 ring with >= 5 stages (pooler m8/m16, o, unpooler
  m16): worth a same-run test there (skv only supports NWR = 1 layouts, so I did not test those layers).
- m32: any design that gives 132 SMs ~17.5 rows each with < 3 n8 tiles of consumer work and no cross-CTA reduction.
