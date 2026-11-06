# Agent L — m = 128 pooler / unpooler: W-stream layout hypothesis, then an escape-free RS-wgmma loop

## STATUS (final, 2026-11-02 12:30 PST)
- Hypothesis "stage-major W + bigger copies + multi-thread fill speeds up the m128 W stream": FALSE for these kernels
  (iteration 1: all variants within noise or worse; the pure W ring of the kernel already streams at 4.5 TB/s).
- Built an escape-free 12-bit RS-wgmma kernel (g128l.py): beats bo6 (Z's g128 / M's m128) in every same-run bench, but
  LOSES to bo11 (agent Q2's BF16 SS g128q) on both shapes -> g128l.OWN_LAYERS = () (claims nothing; run -> None).
| same-run bench.py, 100 reps (local us) | SGLang | base | g128l |
|---|---|---|---|
| vs bo6, box 0: pooler / unpooler | 24.98 / 19.09 | 18.99 / 17.39 | **18.02 / 17.21** |
| vs bo6, box 1: pooler / unpooler | 24.38 / 19.26 | 18.58 / 17.41 | **18.25 / 16.96** |
| vs bo11, box 0: pooler / unpooler | 24.61 / 18.75 | **17.71 / 16.70** | 18.11 / 17.03 |
| vs bo11, box 1: pooler / unpooler | 24.90 / 19.28 | **18.03 / 16.88** | 18.12 / 16.91 |
| vs bo11, box 1 (earlier run): pooler / unpooler | 24.37 / 19.10 | **18.15 / 17.25** | 18.23 / 17.41 |
- Correctness (g128l on pooler + unpooler, torch elsewhere: sub_g128l_t.py): `check.py --layers pooler,unpooler --graph`
  (NaN-poisoned outputs) "checked 104 cases x 2 seeds + graph: ALL OK"; `exact_all.py --layers pooler,unpooler --nesc 200
  --ms 97,127,128` EXACT ALL OK (incl. injected -1e-30, 2^-40, -2^-126, 3.5, 1.0, zeros).
- Why 12-bit RS loses to Q2's BF16 SS here (measured): the RS decode costs ~0.27-0.29 us per 64x64 W block per
  warpgroup-pair regardless of N (iteration 5), so 12-bit needs all 128 tokens per decode (n128) -> 128 W rows x 128
  tokens tiles -> split-K (pooler S=4: 2.9 us exchange tail; unpooler S=2: 1.9 us) where Q2's BF16 tiles need none
  (0.4-0.6 us epilogue). The 12-bit loop (~13 us) beats Q2's BF16 loop (14.3-15.6 us) by less than the tail costs.

Base: claude/t3/bo6.py (pooler m128 = agent Z's g128 `mz_pooler_4_3_2_0`, unpooler m128 = agent M's m128 `mk_unpooler`).
All times local us (graph replay after a 512 MB flush, empty graph subtracted; judge ~= local + 1.0). Same-run A/Bs only.

## Files (claude/t3/L/)
- g128l.py      DELIVERABLE: prepare_g128l / run_g128l (pooler + unpooler at 97 <= M <= 128, None otherwise)
- sub_g128l.py  candidate = bo6 + g128l;  sub_g128l_t.py = g128l + torch elsewhere (fast checks)
- le.py         dev copy of the escape-free kernel family (diagnostic defines LE_*), g128l.py is generated from it
- lz.py         copy of Z's g128 kernel family with W-layout / copy / fill variants + diagnostics (LZ_*) + finer trace
- tl.py         same-run sweep (copy of Z's tz.py): correctness at M=128/97/127 normal+outliers, judge-style timing,
                per-CTA globaltimer trace converted in int64 (Z's float64 conversion quantised traces to 0.256 us),
                clock64 wait accounting; specs `module|layer:S,EPI[:DEF1+DEF2=v]`
- gensrc.py / gensrc_le.py  kernel source with fake addresses for local nvcc -cubin; sassloop.py hot-loop SASS mix;
  ncuq.py raw ncu metric query

## iteration 1 — the stage-major W layout hypothesis (Z's g128 pooler, lz.py), same run, box 1, 3 rounds x 20 reps
| variant | pooler local us |
|---|---|
| base (CTA-major blob, thread-0 interleaved fill) | 18.22 |
| LZ_SM stage-major blob across ALL 128 CTAs ([s][r][rb]) | 18.42 |
| LZ_W2 2-chunk (24.6 KB) W bulk copies | 18.53 |
| LZ_SM + LZ_W2 | 18.08 |
| LZ_MTF=1 multi-thread initial fill (x by t0, W by t32, lists by t64) | 18.48 |
| LZ_MTF=2 (W fill split over 2 threads) | 19.02 |
| LZ_TMPF prefetch.tensormap | 18.30 |
| LZ_INIT=2/3/4 (W initial burst limited, rest after W0 lands) | 18.90 / 18.51 / 18.34 (unpooler INIT=3 +0.7) |
| diag LZ_NOESC (escape fix-up branch removed, wrong result) | **17.71 / 17.52** |
| diag LZ_NOX (no x traffic) | 17.58 |
| diag LZ_NOW (no W traffic, no escapes) | 15.12 / 14.85 |
| diag LZ_NOMMA (data movement + decode + epilogue, no HGMMA) | 16.61 |
| diag LZ_NOMMA + LZ_NOX (pure W ring + decode + epilogue) | 16.35 |
=> Hypothesis FALSE for m128: stage-major layout, bigger copies and multi-thread fills are all within +-0.3 us (noise
or worse). Z (iteration 10) had found the same for a per-K-split chunk-major blob. The W ring is not what limits the
loop in this kernel (see the traces below); the one big measured lever is the escape fix-up branch (-0.5..-0.7 us).

Traces (trace build, median over CTAs, us from the first CTA start; q1/q3 = consumer at chunk 8 / 24):
| variant | firstW | firstX | q1->q3 per chunk | in-loop W rate | SM clock | loop end med/max | end med/max |
|---|---|---|---|---|---|---|---|
| base | 1.63 | 2.18-2.21 | 0.404-0.420 us | 3.8-3.9 TB/s | 1.66 GHz | 15.26-15.84 / 16.03-16.38 | 18.14-18.50 / 18.62-19.01 |
| NOESC | 1.60 | 2.11 | 0.396-0.412 | 3.83-3.98 | 1.67 | 14.91-15.62 / 15.17-16.10 | 17.86-18.24 |
| NOX | 1.54 | 2.21 | 0.386-0.400 | 3.94-4.09 | 1.72 | 14.75-15.20 / 15.65-15.94 | 17.63-17.86 |
| NOW (x + MMA only) | 0.99-1.06 | 1.22 | 0.356-0.386 | - | 1.67 | 12.58-13.79 | 15.42-16.48 |
| NOW + NOX (MMA only) | 0.83 | 1.02 | 0.336-0.368 | - | 1.735 | 11.84-12.96 | 14.69-15.71 |
| NOMMA (x + W + decode) | 1.57 | 1.95 | 0.374 | 4.22 | 1.78 | 13.60 / 14.24 | 16.42 |
| NOMMA + NOX (pure W ring) | 1.54 | - | 0.348 | **4.53** | 1.79 | 12.64 / 13.76 | 15.46 |
clock64 wait accounting (consumer thread 0, % of loop cycles): base W-full 9.4 %, x-full 7.4 % (compute-only: 8.9 / 9.3 %:
the try_wait instructions themselves); NOMMA: W 29 %, x 48 % (x delivery is the slower stream without compute).
Reading: (1) the pure W ring of this kernel streams at 4.5 TB/s mid-stream (no layout problem); with the RS-wgmma
consumer the loop runs at 0.40-0.42 us/chunk = 3.8-3.9 TB/s because the consumer itself is slower under traffic
(689 vs 638 clk/chunk in the trace build) and the clock drops 1.735 -> 1.66 GHz; (2) the first x tile lands 0.5 us
after the first W tile (2D TMA box over 128 DRAM pages, cold) — every start variant tried moved it by <= 0.1 us;
(3) loop-end spread across CTAs 0.8-1.1 us, epilogue (E3R split-K) 2.9 us after the median loop end.

## iteration 2 — escape-free loop (le.py -> g128l.py)
Design: 12-bit blob without raw units / overflow lists / header (6144 B per 64-row group per chunk); an escape
weight is stored as code 0 / lo 0 and decodes to WDEC = 2^(2H-127) (= 2^-19) inside the loop, which therefore has no
escape branch (90 SASS instr per chunk per warp vs 134 in g128, nvcc 12.1; 168 regs, 0 spills, LOCAL 0).
Exact correction: per CTA, the distinct escape k's of its K range (pooler <= 55, unpooler <= 39) get gather slots;
producer warps 2 / 3 (even / odd chunks) copy the x columns x[0:128, k] of the landed x stage into a gather buffer
(smem -> smem only, no global loads in the setmaxnreg-dealloc'd warps: U's hazard), then release the stage (9th
arrival on its empty barrier). Per-warp work lists are register queues (8-bit counts per chunk, 6-bit k's) loaded
once from a per-CTA table (bulk-copied with the first stage). After the loop every consumer thread corrects its two
rows: acc += x * (wtrue - WDEC) with one FMA when the difference is exact in fp32, else acc = fma(x, wt, fma(-x, WDEC,
acc)) (prepare flags those entries) -> one-hot rows stay bit-exact (exact_all incl. -1e-30, 2^-40, -2^-126, 3.5).

Build-up (same runs, pooler unless noted):
| version | pooler | unpooler | note |
|---|---|---|---|
| bound: no gather, no correction (escapes -> 2^-19, wrong for exact_all only) | 17.38-17.66 | 16.13 | loop end 14.27-14.50 (base ~15.3) |
| v0: x-producer warp gathers (serial LDS.U8 -> LDS.U16 -> STS) | 23.3 | 21.1 | gather 9-16 K clk per CTA on the x-issue path: x issue 5 us late |
| v1: separate gatherer warp, 9th arrival | 19.4-20.1 | 18.6 | gather still ~400 clk per escape (smem latency ~200 clk under load) |
| v2: two gatherer warps, register work queues | 18.06 | 16.95 | gather 0.39, correction 0.29 |
| same run vs Z/M kernels (lz) | le 17.66 vs 18.16 | le 16.61 vs 16.93 | |
Diagnostics on v2 (box 1): NOXE9 (gatherer does not hold the slots) 17.90 vs 18.28 => 0.38 us of the gather cost is the
gatherer releasing x stages late; the copy itself ~0.2; the correction 0.21.

### iteration 2 analysis block (v2 build = first shipped g128l; box 0/1)
1. nsys in-graph (nsys_shapes.py, same process as bo6, box 1): pooler m128 `le_pooler_4_3_0` 18.72 us vs bo6
   `mz_pooler_4_3_2_0` 19.20; unpooler m128 `le_unpooler_2_1_1` 17.76 vs bo6 `mk_unpooler` 18.18.
2. ncu (ncu_run.sh, full set, flushed, box 1; sub_g128l_t.py):
| kernel | dur | DRAM rd | DRAM % peak (TB/s) | L2 % | SM % | issue % | tensor (FLOP-derived) | occ % | elig/clk | smem wavefronts % / bank conflicts | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| le pooler | 22.11 | 52.5 MB | 49.5 (2.38) | 62.7 | 41.6 | 24.6 | 9.2 us = 42 % of dur | 17.1 | 0.37 | 36.5 % / 189 K (ld) | long_sb 37, wait 18, barrier 10, selected 9, math_pipe_throttle 7 |
| le unpooler | 19.71 | 51.5 MB | (2.61) | | 46.4 | 25.4 | 9.2 us = 47 % | 17.3 | 0.38 | | long_sb 36, wait 22, selected 10, math_pipe_throttle 7 |
   (ncu runs at ~1.73 GHz with its own launch overheads: dur is ~3.5 us above the bench time.)
3. PTX/SASS: NVRTC on the box 168 regs, LOCAL_SIZE_BYTES 0 (both kernels); nvcc 12.1 -cubin: 0 spills. Consumer hot
   loop (sassloop.py, 6 chunks unrolled, 24 HGMMA.64x128x16): pooler 90 instr / chunk / warp (g128: 134):
   PRMT 16, LOP3 16.8, IADD3 16.7, MOV 7.2, BRA 6.5, IMAD 5.3, SHF 4, HGMMA 4, SYNCS 4, LDS 3 (3 x LDS.128 W),
   WARPGROUP 2; unpooler 84. No LDG/TMA in the consumer loop. Producer-WG branch uses R0-R37 (< 40 after
   setmaxnreg.dec), no global loads in the dealloc'd warps.
4. Binding unit: none at its peak. In-loop the kernel runs ~0.40 us/chunk (W 3.9 TB/s = 81 % of 4.8; tensor ~72 %
   of the 1.66-1.73 GHz clock): the consumer critical path under traffic (689 vs 638 clk/chunk compute-only) plus a
   4 % power-capped clock drop; outside the loop: 1.4 us to the first W / 2.1 us to the first x tile, 0.6-0.8 us
   loop-end spread, and the split-K exchange (pooler E3R 2.9 us, unpooler DSMEM 1.9 us). Next: hide the exchange
   behind the last chunks (tail split per token half) and cut the gather / correction cost (0.4 + 0.25 us).

## iteration 3 — gather / correction variants, x start (XLD), all no gain (same runs, local us)
| variant | pooler | unpooler | note |
|---|---|---|---|
| v2 (shipped g128l) | 17.92-18.16 | 17.02-17.53 | |
| LE_GB4: gather batches of 4 escapes, stage released before the gbuf stores | 21.07 | | predicated LDS.U16 + per-batch pops: slower (cause not isolated) |
| LE_SPLITCORR (correct the outgoing token parts first, own part during the exchange) | +1.5 (with GB4) | +0.5 | runtime group masks |
| LE_XLD=1 / 2: first 1 / 2 x stages loaded by the 256 consumer threads (LDG -> swizzled STS) | 19.55 / 19.62 (vs 18.16) | 18.54 (vs 17.53) | consumer LDG at start costs more than the late x TMA box |
| v0 -> v2 history: x-producer gather 23.3 / separate warp 19.4-20.1 / two warps + register queues 18.06 | | | gather on the x-issue path is fatal (smem round trip ~200 clk under the loop's load) |
Analysis: these are all changes outside the hot loop; their trace effect was read from per-CTA globaltimer traces
(gather clock64 accounting: v1 15-16 K clk per CTA, v2 5.5-6 K clk). No ncu was run on variants that lost by > 1 us.

## iteration 4 — split the last TD chunks per token half to hide the unpooler DSMEM exchange (LE_TAILD), dead end
Design: main loop on chunks < n - TD (n128); the TD tail chunks decoded once into registers, MMA'd as n64 on the
partner's token half (separate 32-register accumulator: n64 on the upper half of the n128 accumulator array makes
ptxas 12.1 and NVRTC 13.0 serialise EVERY wgmma of the kernel, C7511), exchange, then the own half while the DSMEM copy
flies; escape corrections of non-tail chunks applied while the tail HGMMAs run.
| unpooler (box 0) | base 17.02 | TD 3: 18.69 | TD 4: 18.98 | TD 6: 19.65 |
Trace (TD 3): main loop ends 13.22 (base 14.59) but the tail waits for the LAST chunks' x / W (x(31) issued 12.70,
lands ~14.3): phase (a) ends 14.85, DSMEM issued 15.84 vs base 15.42. => at the end of the loop the kernel is
data-arrival bound; deferring work past the last arrival cannot hide the exchange (at most the own-half MMA of the
tail chunks, ~0.2 us).

## iteration 5 — could 12-bit use Q2's split-free geometry (64 tokens per CTA, 2 token tiles share W in L2)?
Measured the RS consumer at n64 (lz.py LZ_N64 diagnostic, same decode, HGMMA m64n64k16, trace build, box 0):
| build | per 64-k chunk (both WGs) | clock |
|---|---|---|
| compute only, n128 (NOW + NOX) | 0.362 us = 637 clk (512 tensor) | 1.76 GHz |
| compute only, n64 | 0.290 us = 512 clk (256 tensor: 50 %) | 1.77 GHz |
| n64 with W + x traffic | 0.394 us | 1.72 GHz |
=> the decode path, not the MMA, sets ~0.27-0.29 us per 64x64 W block per WG pair. A split-free 12-bit unpooler
(128 W rows x 64 tokens, 64 chunks per CTA, every W block decoded by both token-tile CTAs) would need ~17-18.6 us
of loop: worse than Q2's BF16 SS (14.3 us loop). The 12-bit RS route needs n128 (all tokens per decode), hence the
split-K tails. Remaining 12-bit idea (not built): a decode that is not on the consumer critical path at n128 AND no
split-K, e.g. 128 tokens x 64 W rows per CTA (RG1, x 1 MB per SM: M measured that x-limited at 13.4 us) -> no.

### iteration 3-5 analysis block
1. nsys: v2 (shipped) pooler 18.72 vs bo6 19.20, unpooler 17.76 vs bo6 18.18 (iteration 2); variants above lost in the
   same-run timing by 0.5-2 us and were not re-profiled.
2. ncu: v2 table in iteration 2 (pooler DRAM 2.38 TB/s over the ncu duration, ~3.9 TB/s in-loop, issue 24.6 %,
   occupancy 17 %, eligible 0.37, long_scoreboard 37 %).
3. SASS: v2 loop 90 instr / chunk / warp (pooler), 84 (unpooler); TAILD builds: 168 regs, 0 spills, no C7511 with the
   separate accumulator; tail HGMMA.64x64x16 groups issued back to back (12 per WG) with one DEPBAR.
4. Binding: in-loop the consumer critical path under traffic (0.40 us/chunk vs 0.336 compute-only, clock 1.66 vs 1.735
   GHz) with W at 3.9 TB/s (81 % of 4.8); at the end of the loop the arrival of the last chunks' data; then the split-K
   exchange (1.9-2.9 us). The BF16 SS design of Q2 has no exchange and streams W at 4.3-4.7 TB/s: it wins.

## Files (final)
- g128l.py (deliverable, OWN_LAYERS = () by default), sub_g128l.py (bo6 + g128l, both layers enabled),
  sub_g128l_b11.py (bo11 + g128l, for the same-run comparison), sub_g128l_t.py (g128l + torch)
- le_v2.py (source of g128l.py), le.py / le_tail_dev.py (TAILD v2 dev), lz.py (Z's g128 + LZ_* variants), tl.py,
  mk_g128l.py, gensrc*.py, nvrtc_check.py (local NVRTC sm_90a compile + C75xx warnings), sassloop.py, ncuq.py
