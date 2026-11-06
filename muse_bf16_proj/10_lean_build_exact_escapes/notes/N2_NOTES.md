# N2 — (A) exact m=1 qkv_gate / pooler GEMVs, (B) o W[8192,1024] at m=1 / m=8 (2026-11-02, base claude/t3/bo15L.py)

All times local us (judge-style: 512 MB flush before every replay, CUDA graph, empty graph subtracted; judge ~= local
+ 1.0), box 0 only (user limit). "x clones" = claude/t3/N/t_ab.py / t_n2.py / t_o.py: several graphs per candidate on
different x / out buffers, timed interleaved in one process (P5's placement fix).

## RESULT (read first)
Deliverable: `claude/t3/N2/n2.py` (15.5 KB, readable, no encoded data): `prepare_n2(weights)` (never raises) /
`run_n2(x, config, state)` -> out for qkv_gate and pooler at m == 1, None for everything else (the base keeps every
other shape, including o at m = 1 / 8: nothing I built beats agent N's ta16 there). Candidates: `sub_n2.py` (bo15L +
n2, n2 first), `sub_n2_only.py` (n2 + torch). Integration into the build: before the base's G route,
`if layer in ('qkv_gate', 'pooler') and m == 1: out = run_n2(x, cfg, st)`; G's m = 1 qkv/pooler kernels can then
be dropped from the build (n2 replaces them; G's unpooler-slice/o paths are already served by S5 / N).

(A) EXACT: G's g311 with exact two-term escapes (design: iteration 1). Natural H200 weights: pooler has 1 "far" escape
(compiled with the thread-0 variant), qkv_gate 0 (compiles to G's kernel instruction for instruction).
| gate (box 0) | result |
|---|---|
| check.py sub_n2.py --layers qkv_gate,pooler --graph (NaN-poisoned) | checked 103 cases x 2 seeds + graph: ALL OK |
| exact_all.py sub_n2.py --layers qkv_gate,pooler --nesc 200 (all M incl. m = 1) | EXACT ALL OK |
| exact_all.py sub_n2.py --layers qkv_gate,pooler --nesc 3000 | EXACT ALL OK |
| exact_all EXDIAG=1 on bo15L (same run) | qkv_gate m=1 "got 0.000000e+00 want -9.984021e-31": EXACT FAILED |
| t_exact.py (= C2's t_lab.py --exact: 400 injected, 200 one-hot columns + 0 + K-1) | n2: 0 / 0 mismatches; bo15L: 75 (qkv) / 82 (pooler) |
| local sm_89: exact_all --nesc 200 / 3000 (m=1), t_exact, check | all OK |

Speed (same-run, x clones, t_ab.py --ng 6 --reps 40, three runs; SGLang in the same process):
| shape | SGLang | bo15L (G g311) | n2 | paired delta n2 - bo15L |
|---|---|---|---|---|
| decode-1 qkv_gate m=1 | 12.90-12.99 | 6.90-6.96 | 6.90-6.94 | -0.085 / -0.027 / -0.027 (identical SASS: placement noise) |
| decode-1 pooler m=1 | 19.73-19.81 | 11.38-11.47 | 11.44-11.50 | +0.096 / +0.059 / -0.035 (mean +0.04) |
Alternating single-candidate processes (bench.py --reps 60, 3 pairs, speedup vs SGLang of the same process):
| shape | bo15L | n2 |
|---|---|---|
| qkv_gate m=1 | 7.26 x1.896 / 7.20 x1.872 / 7.24 x1.878 | 7.40 x1.826 / 7.29 x1.848 / 7.28 x1.854 |
| pooler m=1 | 11.75 x1.687 / 11.66 x1.698 / 11.69 x1.686 | 11.77 x1.683 / 11.76 x1.681 / 11.75 x1.689 |
qkv's n2 kernel is SASS-identical to G's (FAR = 0 at these weights): its process-level -0.07 (x1.87 -> x1.84) is the
placement of a second copy of the 27 MB blob in the process, not code; the same-run x-clone A/B shows n2 qkv -0.05.
pooler: +0.02..0.10 in 5 of 6 comparisons, mean ~+0.05 us (0.4 %) = the 8 extra instructions of the far path (within
the +-0.1 noise band of single comparisons). nsys in-graph (one graph each): qkv 8.14 (G) / 8.29 (n2, same code),
pooler 12.77 / 12.70.

(B) o at m = 1 / 8: no win, n2 returns None. Answers to the two questions (iterations 2-8):
- Can N's structure carry F11 at K = 1024? F11 tiles are 12.12 MB vs F12 12.71 (-4.6 %). Free-escape bound (F11,
  escape work ablated, wrong results): m=1 -0.10..-0.14 us, m=8 -0.04..-0.14 vs N. Every exact escape path I built
  costs more than that: compact per-tile list + row header + smem-staged x/slots, warp-parallel tail: m=1 +0.55,
  m=8 +2.2 (F11); the same machinery with F12's rare escapes still +0.24 / +1.1. => no, not with a one-wave-ish tile
  kernel: 205 escapes per 16-row tile need an x gather and a per-row sum in the CTA tail, and anything added to the
  tail of this kernel costs 0.2-1 us (iterations 3-8).
- What keeps m=8 above m=1 (N's kernel, ablations, iteration 5): escapes (8 global x gathers per slot thread after
  the MMA) and the serial reduction (128 outputs x 16 partials by 128 threads); each alone -0.14 us, both + x -0.55
  us -> 3.54 = the m=1 kernel with the same ablations. x itself is free (no-x ablation +0.02). The tail is a serial
  chain; parallel tails I wrote were slower (barrier->end 0.42-0.61 us vs N's 0.29, traces in iteration 7).
- New general fact (iteration 6): a forward jump in a kernel's tail over >~1-2 KB of code costs a cold instruction
  fetch of ~1.15 us under a full DRAM stream (the judge's flush evicts code from L2): N's o m=1 kernel + a never-taken
  48-line block after its store: 3.70 -> 4.83 us; <= 16 lines (~40 SASS instr): no cost; the same block at kernel
  start: +0.1. Rare-path code (escape fix-ups, cold branches, out-of-line blocks) in tails should stay within ~1 KB
  of straight-line code.

## Tools written (claude/t3/N2; research code is NOT part of the deliverable)
- `lab.py` + `t_n2.py`: exact g311 lab (knobs far / fl / t0 / as / force), same-run x-clone timing vs G's lean module,
  --exact (C2's injection). `t_exact.py`: C2's t_lab --exact test applied to full candidates.
- `lab_o.py` + `t_o.py`: o-kernel lab: ta12 (= N's ta16 verbatim + ablation bits 1 no escapes / 2 no x / 4 no
  reduction / 4096 tail warm-up / 2048 trace), ta11 (F11 v1), tb (v2: F11/F12 compact escapes, staged x, parallel
  tail), tc (v3: N's per-row F12 slots, staged x, parallel tail; bits 16/32/64/8192), x-clone timing vs N's lean
  kernels, --exact.
- `tr_o.py`: per-warp clock64 timelines (CTA start, last warp's loads landed, barrier, end) under judge-style replay.
- `ip_probe.py`: instruction-fetch probe. `sass_n2.py`: NVRTC SASS stats incl. "afterUse" = LDGs issued after the
  first instruction that consumes ANY loaded register (stricter than "late": catches loads sunk below the decode).
- `ncu_o.sh` / `ncu_o.py` / `ncu_parse.py` / `ncu_src.py`: ncu for lab kernels (+ per-instruction stall samples).
- `gsrc.py` / `ssrc.py` / `nsrc.py`: G's / N's / n2's kernel sources for SASS comparison. `sub_g1lean.py`: G control.

## iteration 0 — baselines (box 0, x clones)
G g311 (bo15L m=1): qkv 6.82-7.07, pooler 11.38-11.58. N ta16 (bo15L o): m=1 3.62-3.74, m=8 4.00-4.21.
SASS (NVRTC 13.0 sandbox = judge; 12.1 local gives the same stats): g311 40 regs, 240 instr, 11 LDG, all W/x/slot
loads before the first FFMA (the escape x gather at 58, intended); N o1 38 regs / 264 instr, o8 39 / 328 (8 escape
gathers after the MMA, intended).

## iteration 1 — exact g311 (lab.py, n2.py)
Design: G's single-term delta x * (true * 2^(120-E_lo) - 2^-7) is exact when that delta is an FP32 value
(vectorized pack-time test: d = fp32(tv - 2^-7); exact iff d + 2^-7 == tv in fp64 and finite). Otherwise ("far"):
fake sum += -x * 2^-7 (cancels the decoded 2^-7 exactly), true sum at += x * true; out = bf16(fma(s, scale, at)).
Far escapes are sorted first in the row's slot pool and flagged (bit 16 + log2 K); the at sum must not need a CTA
reduction: T0 variant (<= 1 far per row: slot 0 = thread 0 keeps at in a register, nothing else), vote variant (<= 32
per row: warp 0's slots, `if (__any_sync(at != 0))` shuffle in warp 0). Compile-time FAR = 0 when a layer has none.
Counts: H200 weights (reference seed): pooler 1 far escape (of 456K escapes), qkv 0; injected tests 37-1125.
SASS (NVRTC 13 and 12.1): first versions with a branchy far path pushed the x gather after the FMA chain (168 vs 58)
or (G's body with my small edits: plain `a` instead of `acc[R]`, FFMA tail) delayed one x load past the first FFMA
(+1 L2 round trip, measured +0.03..0.1). Keeping G's statement order verbatim and expressing the far path as selects:
40 regs, 0 stack, 11 LDG, afterUse 1 (the gather, as G), 248 instr (T0) / 256 (vote) / 240 (FAR = 0).
Anti-sink (C2's ±0 trick) here made things worse (a W lo load sank below the first FFMA: lateNA 1).
Timing (t_n2.py, box 0, x clones, paired deltas vs G's own lean build in the same process):
run A (4 x 40, far path forced on both layers): pooler G-copy +0.028, select form (f0) +0.108, FFMA form (f1) +0.032;
qkv G-copy -0.008, f0 +0.100, f1 +0.068. Run B (2 x (5 x 50), H200 weights: pooler has 1 far escape, qkv 0, so qkv
variants compile to G's code): pooler G-copy +0.028 / -0.016, f1 -0.010, T0 -0.045; qkv G-copy +0.064 / +0.016,
T0 +0.029 / +0.006 (= noise of identical code). => T0 shipped (vote form only when a row has > 1 far escape).
Local exactness: G 73-76 mismatching columns of 202, n2 0.
Analysis block (shipped kernels, box 0):
1. nsys in-graph (20 reps, one graph each): qkv G 8.14 / n2 8.29 (identical SASS: placement), pooler G 12.77 / n2 12.70.
2. ncu (full set, flushed, ncu_run.sh): pooler n2 14.27 us vs G 14.30; DRAM 48.65 MB both = 3.41 TB/s over the ncu
   duration (in-graph 12.7 us -> 3.83 TB/s = 80 % of 4.8); SM 46.5 / 45.3 %, issue 56.3 / 55.3 %, occupancy 64.5 /
   64.7 % (6 CTAs x 8 warps), eligible 1.44 / 1.38 warps/clk, tensor 0; stalls long_scoreboard 54 / 56 %, barrier
   14 / 13 %. qkv n2 8.93 / G 8.86 us, 27.30 MB, issue 52.4 / 52.6 %, occ 63 %, long_sb 47 %, barrier 17-18 %.
3. SASS: above (40 regs, LOCAL 0, 11 LDG, every W/x/slot load before the first FFMA, 32 FFMA + 33 decode ALU per 32
   weights, tail BAR + 8 LDS + FADD + FFMA + STG).
4. Binding unit: DRAM streaming at the hw-scheduled LDG floor (C2's probe: 50.3 MB in 11.17-11.41 us -> ~10.9-11.0
   for 48.6 MB; n2 11.4-11.5 local) — 80 % of peak averaged over the in-graph time, the rest being launch / first
   landing / last-CTA tail. The exact path adds 8 instructions (selects + 1 FFMA) to 2-3 % issue headroom; the only
   remaining lever for these two shapes is bytes (F11 slot padding 2.4 MB, C2: not recoverable without sinking).

## iteration 2 — o: F11 in N's ta16 structure (lab_o ta11 v1)
Packer: F11 per lane = 2 x 16 B low bytes + 3 x 4 B code words (planes) per 64-k chunk (1408 B), compact per-tile
escape list (row-sorted, (row<<10|k)<<16|true, 256 slots = 1 KB padded, mean 205 used) + 16-word row header ->
23680 B / tile = 12.12 MB (F12 12.71). Exact on first build (local, 202 columns). Box 0 (x clones, 4 graphs x 40):
m=1 N 3.70 | ta12 copy 3.70 | ta11 4.56 | ta11 no-escapes (wrong) 3.73; m=8 N 4.05 | ta11 8.74 | no-esc 6.77 (!).
ncu ta11 no-esc m=8: 590K shared-memory bank conflicts (16-way: red[kc][tok][row] read with tok-parallel lanes),
short_scoreboard 30 % -> fixed by padded layouts in v2.

## iteration 3 — v2 (tb): conflict-free tail, x + slots staged in smem, header; load-sinking found
First v2: 42-48 regs with the escape loop unrolled 4x (544 instr) -> 2 CTAs/SM. `#pragma unroll 1` + min-blocks 3:
40 regs, 184-208 instr. Box 0: F11 no-esc m=1 3.47 (N 3.62: -0.14 = the F11 byte bound), m=8 3.87 (-0.14 vs
N 4.00), but WITH escapes m=1 +1.2 (F12 and F11 alike). Ablations: dropping just the escape loop (whose result was
the only reader of the header word) -> N speed. SASS: the per-warp header LDG was issued AFTER the decode PRMTs
(its address math was scheduled late), i.e. after W landed: a second DRAM round trip per warp (+1 us). My "late"
metric (LDG after the first FFMA/HMMA) missed it -> new metric afterUse. Fix: anti-sink on the decode masks
(M ^= (hd | es) >> 31, both words have bit 31 = 0) -> afterUse 0 for all v2 kernels. Then: F12+escapes m=1 3.90
(+0.24 vs N), F11 4.19 (+0.55), m=8 F12 5.22 (+1.1), F11 6.21 (+2.2).

## iteration 4 — v3 (tc): N's per-row F12 slots + staged x + parallel tail; bisection
v3 = N's kernel with x fragments / slots staged in smem before the barrier (no global gathers in the tail) and a
parallel tail (m=1: warp per row; m=8: 4 threads per output + 2 shuffles), padded red, 168-184 instr, all loads
before first use. Box 0: m=1 +0.45..+0.52, m=8 +0.78..+0.89 vs N. Bisection (abl bits, same run):
| m | full v3 | no x staging | no slot staging | neither | N's serial tail | serial tail, no staging |
|---|---|---|---|---|---|---|
| 1 | +0.49 | +0.64 | +0.27 | +0.01 | +0.12 | +0.01 |
| 8 | +0.78 | | | +0.30 (parallel tail) | +0.42 | +0.00 |
=> staging x/slots in smem costs 0.2-0.5 us and my parallel tail is slower than N's serial one at m=8; N's tail is
the local optimum. Compile-time grid stride in the tail (no gridDim read): no change.

## iteration 5 — what keeps m=8 above m=1 (N's kernel, ablations, box 0, x clones 4 x 40)
| | full | no escapes | no x loads | chunk-0-only reduction | all three |
|---|---|---|---|---|---|
| m=8 | 4.08 | 3.92 (-0.14) | 4.11 (+0.02) | 3.98 (-0.14) | 3.54 (-0.55) |
| m=1 | 3.63 | 4.11 (+0.44 !) | 3.79 (+0.13) | 3.60 (-0.06) | 3.54 (-0.13) |
m=8's extra 0.45 us = escape gathers (8 per slot thread, after the MMA, global) + the 128-output serial reduction,
non-additive (a serial tail chain). x (16 KB per CTA, 8 MB of L2->SM traffic) is free. With all three removed m=8
runs at the m=1 time (3.54), i.e. the stream itself is identical. (m=1 single ablations go the wrong way: code layout.)

## iteration 6 — instruction fetch after the flush (ip_probe.py)
N's o m=1 kernel + `if (q.pad == 12345) { NI dummy lines }` after the final store (never taken) vs the same block at
kernel start vs none (box 0, 4 graphs x 25-30 reps):
| NI (source lines, ~2.5 SASS each) | 4 | 16 | 48 | 100 | 200 | 2000 |
|---|---|---|---|---|---|---|
| none | 3.70 | 3.55 | 3.68 | 3.70 | 3.68 | 3.70 |
| after the store (tail) | 3.70 | 3.58 | **4.83** | **4.86** | **4.83** | **4.85** |
| at kernel start | 3.79 | 3.71 | 3.81 | 3.81 | 3.78 | 3.92 |
A forward jump in the tail over >= ~120 SASS instructions (~2 KB) costs one cold instruction fetch ~1.15 us (DRAM
latency under the saturated W stream; the flush evicted the code from L2), independent of the skipped size beyond
that; small skips are covered by the sequential prefetch. At kernel start the same fetch costs ~0.1. Explains part
of the "ablation goes the wrong way" noise of this project (code layout of rare paths); rule: keep tail code
straight-line or short-skip, never put a far rare path between the hot tail and EXIT.
Tail warm-up attempt (run the tail once while loads fly, `#pragma unroll 1` 2-pass loop): ptxas keeps the loaded
registers live across the warm pass -> 62 regs (2 CTAs/SM) or, at min-blocks 3, 24-56 B of stack. Not pursued.

## iteration 7 — per-warp timelines (tr_o.py, clock64, box 0, 12 replays, medians over CTAs)
| kernel | CTA start -> last warp landed | landed -> barrier | barrier -> end | kernel span | CTA start p90 |
|---|---|---|---|---|---|
| N o m=1 | 2.08 | 0.14 | 0.25 | 4.39 | 2.56 |
| v3 o m=1 | 2.11 | 0.14 | 0.45 | 4.72 | 2.56 |
| N o m=8 | 2.22 | 0.26 | 0.29 | 4.76 | 2.56 |
| v3 o m=8 | 2.14 | 0.21 | 0.61 | 5.39 | 3.07 |
| v3 serial tail, no staging | 2.28 | 0.12 | 0.43 | 4.90 | 2.82 |
(clock taken as 1.75 GHz; spans include the trace stores.) 512 CTAs on 3 CTAs/SM (38-40 regs x 512 threads) = 396
slots: ~23 % of the CTAs start in a second wave at ~2.6 us, so every first-wave CTA's tail delays a second-wave CTA;
the kernel is ~1.3 CTA latencies. A 4-CTA/SM build needs <= 32 regs: at min-blocks 4 ptxas sinks a W load or spills
8-16 B (checked for ta12 and v2/v3).
Binding unit (N's kernels): DRAM stream latency at ~3.4-3.6 TB/s averaged over the in-graph time (12.7 MB in
~3.6-3.7 us local + 1 us) with tensor 2 %, issue 27-34 %; per CTA ~2.1-2.2 us until the last warp's data lands, then
0.4-0.55 us of tail. N's tail is the local optimum among the 6 tail structures measured here.

## iteration 8 — o m=8 escape work per (slot, token) thread pair (lab_o ta12 ABL 16384)
N's m=8 slot threads each issue 8 x gathers + 16 STS after the MMA (a 91-instruction block that the other warps
branch over). Variant: 384 threads = 48 slots x 8 tokens, one gather each (272 instr, 1 late LDG, 32-instr skip).
Box 0 (x clones, 2 runs x 5 graphs x 40): N 4.03 | N copy 4.03 | pairs 4.32 (+0.23) | N without escapes 3.84-3.89
(-0.21). Slower: 8x the slot loads (the same 4 B read by 8 threads, inside the W stream). Escape work at m=8 is worth
~0.2 us in N's kernel; no cheaper exact form found.

## Dead ends (measured, this agent)
- F11 for o in any form tried (v1, v2) with exact escapes: +0.24..+2.2 us; free-escape bound only -0.1..-0.14.
- Staging x / escape slots in smem for o (instead of global gathers): +0.2..+0.5 us.
- Warp-parallel / 4-threads-per-output tails for o: slower than N's serial tail (+0.12..+0.3 at m=8).
- Anti-sink on g311: sinks a W load (lateNA 1). Branchy far path in g311: pushes the gather behind the FMA chain.
- Tail warm-up loop: register blow-up (62 regs) or spills.
