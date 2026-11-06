# b3 worker: T=753 boundary kernels bnd3_b1 / bnd3_b2 toward HBM peak (2026-10-31)

Base: research/claude/stk/s_b.py. Scope: only the bnd3 kernels (rows < BND4_MIN_ROWS = 2048); bnd4 (>= 2048),
pnr5 / gather / combine / MoE untouched (mf worker).

## Byte budget per call at T=753 (what a floor means here)
- b1: reads s 24.7 MB (FP32, DRAM: layer input) + a 12.3 MB (BF16, written by o_proj just before: likely L2), writes
  h 12.3 MB. b2: reads s 24.7 + a 12.3 (DRAM) + f 12.3 (unpool output: likely L2), writes s2 24.7 + out 12.3.
- Per row, bnd3 also re-reads the per-channel coefficients: b1 80 KB (al1, be1 FP32 + wpre BF16), b2 128 KB
  (A, Bv, C + wpost, wnext). From the s_b ncu L2 hit rates (b1 29.8%, b2 46.0%) these mostly hit L1.
- In-layer (stk base_trace): b1 13.1 us, b2 21.4 us.

## Tools
- bench.py: kernel-level probe. Base = BND3_SRC from s_b.py, candidate = BND3_SRC + bnd6.cu (+ -D knobs).
  Byte-exact check of every candidate kernel vs bnd3 (h, ra, out, s2) at T 9..2047; CUDA-event timing with a
  judge-style L2 flush + a producer copy leaving a (b1) / f (b2) dirty in L2; --trace: per-row globaltimer phases.
- bnd6.cu: r0 = bnd3 copy with optional stamps; b6 = persistent, all coefficients + (1 + w) resident in registers,
  bulk-copy stages, ra prefetched (no global round trip on the per-row chain).

## Log
### 1. Probe vs layer, correctness (bench.py, T=753, L2: read flush + producer copy of a / f)
- Every kernel variant below is byte-exact vs bnd3 (h, ra, out, s2) at T 9, 64, 300, 700, 753, 760, 1000, 1500,
  2047, including bnd4 run at T < 2048 (the "bnd4 arithmetic is not bit-exact" note in lat/NOTES does not hold here).
- globaltimer stamps (256 ns resolution) of the bnd3 copy: b1 span 13.06 us, b2 21.76 us = the layer's 13.1 / 21.4.
  CUDA-event times below carry ~3.7 us of launch overhead (bnd3 b1 16.8 / b2 24.5 event vs 13.1 / 21.8 span).
- bnd3 b1 timeline: wave 1 loads 0-2.5 us, then all 264 CTAs compute 2.5-4.5 us with no loads in flight; waves stay
  synchronized (DRAM idle during each compute phase). b2: loads 0-3.5, compute 3-7.5 (red1 -> red2 1.8 us median).

### 2. Floors: what the access pattern itself costs (event us, same L2 state; d = vs bnd3)
| kernel | b1 | b2 |
|---|---|---|
| bnd3 | 16.75 | 24.6 |
| fl: same loads + stores, no math (one-row CTAs) | 14.8 (-2.0) | 23.6 (-1.0) |
| fn: bnd3 math, coefficients = constants (no coefficient loads) | 15.25 (-1.5) | 23.9 (-0.7) |
| lo: loads only | 12.3 | 14.8 |
| so: stores only | 8.3 | 13.5 |
| em: empty kernel (launch overhead) | 5.25 | 5.25 |
| e0 / fe: bnd3 / fl with L2::evict_first on s, a, f | +0.2 / +0.0 | -0.0 / -0.0 |
- Stores alone run at ~3.8 TB/s whether L2 starts clean or full of dirty lines (so b2: 13.5 clean / copy / dirty):
  the writes reach DRAM during the kernel. fl ~= lo + so: loads and stores share one ~4.4 TB/s pipe.
- => b2 sits ~1 us above its streaming floor, b1 ~2 us (1.5 us of it the per-row coefficient traffic: 80 KB/row
  of L1 hits in b1, 128 KB in b2). The brief's "80% DRAM -> -9..10 us" is not available: ncu's DRAM% is low because
  ncu's cache control leaves the kernel's writes in L2, not because bandwidth is idle.
- Evict-first hints on the single-use reads change nothing (the writes, not the reads, are what reach DRAM).

### 3. Restructurings vs bnd3 (all byte-exact vs bnd3 unless noted; event us, d = vs bnd3 in the same run, 30 reps)
| variant | idea | b1 | b2 |
|---|---|---|---|
| bnd4 at 753 | existing persistent kernel | +0.1 | +2.1 |
| b6 | persistent, all coefficients + (1+w) in registers, 4 / 3 bulk-copy stages, ra prefetched | +0.4..0.6 | +1.4..1.8 |
| pf | persistent floor (b6 structure, no math): 15.5 / 23.7 vs fl 14.8 / 23.6 | (-1.3) | (-0.8) |
| c1 | coefficients cp.async'd to smem at CTA start (C1_PRE: before the PDL wait) | +0.0 (+0.75) | +1.7 (+3.0) |
| h1 | coefficient loads hoisted into registers next to the row loads | -0.06 | +0.37 |
| m2 | 2 rows / CTA, s in smem (cp.async), merged reductions, 2 CTAs / SM | +1.46 | - |
| m1 | 1 row / CTA, s in smem, y over s, chunked coefficients, 4 / 3 CTAs per SM (spills) | +3.4 | +1.8 |
| m1 OCC 3 / 2 | same, 3 / 2 CTAs per SM | +2.1 | +2.1 |
| r1 | a packed, coefficient addresses depend on reduction 1 | +0.43 | +1.1 |
| p1 | one thread bulk-prefetches the CTA's rows to L2 (after / before the PDL wait) | +1.1 / +1.0 | +1.0 / +1.2 |
| r2 | row loads as ld.relaxed.gpu (pinned up front, in order) | +1.4 | +1.8 |
| m3 | 2 rows / CTA in registers, 1 CTA / SM, half the coefficient bytes | +1.3 | +1.3 |
| m3 OCC 2 | same at 2 CTAs / SM | 116-228 B spills: not run | |
| q1 | bnd3 body in a 2-CTA/SM persistent loop | +0.19 | +4.0 (spills) |
| m4 | b1: 1 row / CTA at 3 CTAs / SM (40 regs): a packed, coefficients in 4-column chunks | **-0.27** | - |
| m4 OCC 2 | same at 2 CTAs / SM | -0.22 | - |
Diagnosis (ncu source counters, bench --once, ncu.sh -> prof/b3n1.*):
- bnd3_b1 SASS (NVRTC 13): ptxas issues the 4 `a` loads, stalls on a's first unpack (16% of all samples), and only
  then issues the 4 `s` loads: the CTA's DRAM fetch is serialized behind the a latency (register pressure at 64).
  bnd3_b2: part of the row loads (a chunk 0, s chunks 0 / 3) also issue after the first unpack stall (15.5%).
  The floor kernel fl (32 regs) issues all 8 loads up front. In r1 and m4 ptxas sinks the `.nc` s loads all the
  way below the first bar.sync (it may: .nc is read-only), so s's DRAM latency sits after reduction 1.
- Every variant that adds requests (L2 prefetch, coefficient staging) loses: ncu L2 throughput is 70-83% for these
  kernels, so duplicated traffic costs time directly. Multi-row CTAs lose CTA-level overlap (m3) or spill (m3 OCC 2).
| r3 | bnd3 body, row loads weak (non-.nc): all 8 row loads issued up front, before bar.sync | +0.24 | +0.13 |
| m4 weak | m4 with weak s loads (s before the barrier) | +3.2 (36 B spills) | - |
| m5 | b2 at 3 CTAs / SM (a, f packed, chunked A/Bv/C, y in smem or regs): spills 16-56 B | - | +3.5 / +4.1 |
| m5 OCC 2 | same structure at 2 CTAs / SM, no spills | - | +0.74 |
- Load order is falsified as a lever: r3 issues every row load up front and is not faster.
- m4 across shapes (b1, 2 runs x 40 reps, event us vs bnd3): T300 -1.06 / -1.09, T700 -0.27 / -0.11, T753 -0.22 / -0.14,
  T760 -0.16 / -0.24, T1000 +0.74 / +0.59, T1500 -0.27 / -0.67, T2047 +1.07 / +1.18. It follows last-wave fill (3 vs 2
  CTAs / SM: T1000 m4 2.53 waves, T2047 5.17 -> near-empty last wave), i.e. wave quantization, not a better kernel.
  At T=753: ~-0.18 us (~0.01% of score) and it would need T-gating: not delivered on its own.

### 4. Where the coefficient cost comes from (bench, event us vs bnd3; timing-only ablations k_b*, wrong math)
| ablation | b1 | b2 |
|---|---|---|
| fn: no coefficient loads at all | -1.66 | -0.83..-1.10 |
| K1: v1 (and v2) loads read v0's lines: same per-CTA bytes, fewer distinct lines | -0.94 | -0.16 |
| K2: every coefficient load reads the first 2 KB of its vector: same instructions and per-CTA bytes, ~all L1 hits | -0.90 | -1.09 |
| K3: no w loads | -0.43 | +0.37 |
- => the cost is the cold-L1 fetch of the coefficients in wave 1 (both co-resident CTAs miss on the whole 80 / 128 KB:
  ~21 / ~34 MB of L2-hit traffic at kernel start, competing with the row stream; matches the ncu L1 hit rates
  45.8% / 44.1%), not L1-hit throughput and not latency (h1 / c1 moved the loads and gained nothing).
- A one-row-CTA kernel at 2 CTAs / SM pays 2 x coefficients per SM in wave 1 by construction. Halving it needs the
  CTAs to share (cluster multicast into smem + persistent rows), i.e. the structures that lost in section 3.

### 5. In-layer headroom (gap/trace.sh nsys node trace, cases 0 / 1, 6 reps, medians; timing-only builds t_*.py)
| node | s_b | fl floor (no math) | K2 (coefficients made L1 hits) | m4 |
|---|---|---|---|---|
| b1 | 13.0 / 13.0 | 10.3 / 10.4 | 11.8 / 11.6 | 13.1 / 13.2 |
| b2 | 21.6 / 21.3 | 20.0 / 19.8 | 20.5 / 20.3 | (bnd3) |
- In-layer headroom vs the floor: b1 2.7 us, b2 1.5 us; the wave-1 coefficient misses are ~1.3 (b1) / ~1.0 (b2) of it.
- m4's probe gain does not carry into the layer (+0.1).

### 6. One coefficient fetch per SM in wave 1 (CTA placement: wave 1 = 2 CTAs on every SM; blocks 0..131 on 124 SMs)
| variant | b1 | b2 |
|---|---|---|
| pfc: blocks < 132 prefetch.global.L1 every coefficient line (CCTL.E.PF1) / all wave-1 blocks | +0.86 / +2.14 | +2.0 / +3.7 |
| cx: blocks < 132 cp.async.ca-touch every 32 B sector (no registers) | +0.05 | +0.9 |
| **hx: blocks < 132 run h1 (coefficient loads with the row loads), others bnd3's order** | **-0.40..-0.58 (5 runs)** | +0.7..0.9 |
| hx, 66 leaders | -0.34 | +0.9 |
| lx_b2: leader path computes fma(A, s, Bv rbf(a ra)) before reduction 1 (C, wnext early or not) | - | +0.14..+0.40 |
- hx b1 across shapes (2 runs, 40 reps): T129 -0.24/-0.21, T300 -0.43/-0.34, T700 -0.40/-0.50, T753 -0.50/-0.54,
  T760 -0.59/-0.58, T1000 -0.29/-0.50, T1500 0.00/-0.30, T2047 +0.19/-0.10: a fixed ~0.5 us wave-1 saving, no regression.
- b2 leader paths are byte-exact (confirms bnd3's contraction: y = fma(C, rbf(f rf), fma(A, s, Bv rbf(a ra))))
  but do not gain: b2's first reduction waits only on f (L2-resident), so the follower's coefficient demand overlaps
  the leader's fetch instead of following it.
- Candidate c_hx.py (= s_b + b3k.cu: bnd3_b1 -> b3_b1 for rows < 2048, b2 unchanged) built; in-layer trace run
  (b3t2, cases 0/1, 8 reps): EVENTS replay medians case 0 539.8 -> 538.6, case 1 715.2 -> 715.6 (span noise
  +-15 us per rep from routing); node-level numbers pending (sandbox lock contention).
- Next: delayed followers (fd: coefficient addresses + z(s[15]), z == 0 for any s, so they issue once the row's
  s has landed, after the leader's fetch) for b1 (hy) and b2 (hy / hz).

### ncu reference (bench --once, cold-ish L2 as in the probe, clock-control none; prof/b3n1.*)
| kernel | us (ncu) | DRAM% | L2% | L1 hit | occupancy | stalls (share of samples, all + not-issued) |
|---|---|---|---|---|---|---|
| bnd3_b1 | 15.7 | 54.0 | 69.6 | 45.8% | 45% | long scoreboard 53%, barrier 19%, short scoreboard 8% |
| fl_b1 (floor) | 14.3 | 62.4 | 72.4 | 0% | 69% (4 CTAs/SM) | long scoreboard 74%, drain 14% |
| bnd3_b2 | 24.0 | 62.0 | 79.0 | 44.1% | 46% | long scoreboard 50%, barrier 19%, short scoreboard 8% |
| fl_b2 (floor) | 22.1 | 67.6 | 83.0 | 0% | 39% | long scoreboard 67%, drain 14%, lg throttle 12% |
