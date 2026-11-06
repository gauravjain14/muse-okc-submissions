# E8 — verify-1 (m = 8) qkv_gate / pooler: beat bo12 (2026-11-02)

## RESULT (current; updated as I go)
- Deliverable: `claude/t3/E8/m8.py` (`prepare_m8(weights)` never raises, `run_m8(x, config, state)` -> out or None),
  `sub_m8.py` (bo12 + m8), `sub_m8only.py` (m8 + torch, fast iteration). Owns **pooler m == 8 only**; qkv_gate m = 8 stays
  with bo12 (V2's qkvb) - nothing I built beats it (below).
- Kernel `m8_pooler`: F12 tiles (agent G's format), CTA = 16-row tile x full K, 256 CTAs (2 per SM, 544 threads, 56 regs),
  16 warps split K (8 units of 64 k each, software-pipelined register ring, W ~2 units ahead, all LDG), 1 escape warp
  (exact two-term escapes), smem reduction, BF16 store. W pointer baked.
- Same-run numbers so far (local us, judge ~= local + 1.0): see the table at the end (bench.py vs bo12 + SGLang).
- Gates: integrated `check.py claude/t3/E8/sub_m8.py --graph` 259 cases x 2 seeds ALL OK; exact_all (pooler, qkv_gate,
  --nesc 200 / 3000) EXACT ALL OK at every M except m = 1, where bo12 alone fails identically (G's g311, < 2^-39 flush).
- Same-run gain (7 bench.py runs, both boxes): pooler m8 bo12 12.33-12.57 -> m8 11.89-12.36 local, mean -0.38 us
  (x1.596 -> x1.648 vs SGLang); net of the per-instance offset seen on identical kernels ~-0.3 us (~+0.14 % geomean).

## iteration 0 — per-SM instruction budget at the stream rate (picks the design)
pooler m8: 33.5 M weights / 132 SMs = 254 K weights per SM; stream time ~11.2-11.6 us at 1.75-1.9 GHz = ~20-22 K cycles
= 80-88 K warp-issue slots per SM (4 SMSPs).
- CUDA-core FFMA (g311-style with 8 token accumulators): 8 FFMA per weight = 63.5 K warp-FFMA per SM (FP32 pipe 128
  lanes/clk = 4 warp-instr/clk/SM -> 15.9 K cycles) + decode 1.03/w + bf16->f32 unpack 1/w (~16 K warp-instr) =>
  ~80 K warp-instr per SM = ~95-100 % of every issue slot. Not viable (g311 at m=1 already runs at 50-55 % issue).
- mma.sync m16n8k16 with W = A (16 rows x 16 k = 256 weights per HMMA, tokens = N 8): 993 HMMA per SM + F12 decode
  7 ALU per 8 weights per lane (= 7 warp-instr per HMMA) + 2 LDG.128 x per 4 HMMA => ~10 warp-instr per HMMA = ~10 K
  warp-instr per SM = ~12 % issue (ncu measured 11.9-14.6 %); tensor ~2 % (2048 FMA per HMMA, 1024 FMA/clk/SM).
  => tensor cores; m = 8 costs no issue headroom. Registers per 32 weights per lane: W 12 (F12) + x 8 (B fragments
  for 4 steps x 8 tokens) - x per lane is LESS than g311's (16 regs of x at m = 1) because B is shared by 16 rows.
- The blocker is structural, not FLOPs: a 16-row tile x full K is 188 KB (F12) per CTA and needs all 128 KB of x;
  it cannot be one all-loads-up-front g311-style CTA (register file), and splitting K across CTAs needs a cross-CTA
  reduction (S: PDL converter +0.6 launch +1.2 work; tickets +10 us). => CTA = tile x full K with a register ring.

## iteration 1 — e8: CTA = 16-row tile x full K, 16 K-warps, static register ring, F12 (claude/t3/E8/e8.py)
- Correct first build (normal + outliers + one-hot on escape columns: 0 mismatches).
- t_e8.py same run (box 0, 3 graphs x 25 reps, local us): pooler bo12 12.54 | e8 (16,3,2,MB2) 12.30 (-0.16..-0.24);
  qkv bo12 7.73 | e8 8.46 (+0.69).
- Per-CTA globaltimer trace (ABL 16, 7 replays, medians): pooler: all 256 CTAs start within 0.26 us; first warp of a CTA
  done 10.75, slowest warp p10/p50/p90 11.01/11.78/12.03, CTA end p50 12.29 / max 12.54; **in-CTA warp spread 1.02 us**;
  last warp -> end (barrier + reduction + store) 0.26. qkv: CTAs/SM {1: 120, 2: 12}; single-CTA SMs end ~7.4-7.7,
  the 12 doubled SMs (2 tiles) p90/max 8.2/9.0 => qkv's 144 tiles on 132 SMs is the qkv loss.
- Per-unit trace (ABL 32, perturbs the kernel by +2.5 us, relative only): first unit lands 4.6 (earliest warp) .. 6.1
  (latest) us after CTA start: the initial burst (256 CTAs x 16 warps x 2-3 units = ~20 MB) is a deep DRAM queue served
  in a random order; per-warp progress then stays offset (no warp index is systematically late: mean last-unit time by
  warp index 13.24-13.62).

## iteration 2 — ablations and knobs (same runs, local us)
| variant (pooler) | us | vs e8 |
|---|---|---|
| e8 16 warps, W 2 units ahead (ptxas picks the depth at the 56-reg cap; BW 2/3/4/5 give identical SASS) | 12.30-12.42 | 0 |
| no escape-warp work (ABL 1, wrong) | 12.38 | -0.04 (escapes ~free) |
| no x loads (ABL 4, wrong) | 12.19 | **-0.23 = cost of x** |
| x first (XF 1) | 12.30-12.42 | 0 |
| x with L1::evict_last | 12.38 | -0.04 (noise) |
| W with L1 allocation | 12.45 | +0.03 |
| NW 8 (288 thr, 16 units / warp, 3 CTAs/SM by regs) | 12.70-12.74 | +0.3 |
| interleaved units (warp w takes u*16 + w: CTA-contiguous per step) | 12.58-12.67 | +0.25 |
| dynamic unit claims in the CTA (smem atomic per unit, BW 2; BW 3 spills) | 13.94-15.70 | +1.6..+3.4 |
- SASS (NVRTC 12.1 local / 13 sandbox): 56 regs, 0 spill, 49 LDG (40 W/x of the 8 units, 1 slot + 8 gathers in the
  escape warp); the unrolled loop is software pipelined by ptxas: prologue issues W(0), W(1), x(0), x(1), then each
  unit's 4 HMMA are interleaved with the next units' LDGs (no sunk loads beyond the intended ring).
- Dynamic claiming fails because the claim -> address -> LDG dependency and the 2-slot ring leave ~1 unit in flight per
  warp (and a 3-slot ring needs ~80 regs: stack 72-232 B).

## iteration 3 — e8r: byte-balanced CTAs, one per SM (claude/t3/E8/e8r.py)
Idea: qkv 2304 rows = 108 x 16 + 24 x 24 rows (132 CTAs), pooler 4096 = 116 x 32 + 16 x 24. The extra 8 rows of a
24-row CTA are a half tile computed with W as the mma B operand and x as A (tokens g, rows g + 8 zero), reusing the
SAME x registers as the 16-row tile (pack_t8: lane (g, t) holds W[g][k0 .. k0 + 3] in the B fragment order) - no extra
x traffic. Correct first build (one-hot exact).
| variant | qkv m8 | pooler m8 | bo12 same run |
|---|---|---|---|
| e8r 108x16 + 24x24 (qkv) / 116x32 + 16x24 (pooler), 96 regs, 1 CTA/SM | 8.70-8.86 | 12.74 | 7.71 / 12.48 |
| e8r 96 x 24 rows (qkv) / 128 x 32 (pooler) | 8.32 | 12.70 | |
| e8r qkv ring W 1 unit ahead | 17.73 | | |
Trace: qkv per-SM end p10/p50/p90/max 6.91/7.68/8.45/8.96 - the 24-row CTAs end last; pooler in-CTA warp spread 1.79 us.
Reading: a CTA's progress is NOT proportional to its bytes in flight (24-row units are 1.3x the bytes in flight per
warp but 1.5x the work); per-SM delivery saturates. One CTA (16 warps) per SM is also worse than two (32 warps).

## iteration 4 — e8c: hardware-scheduled g311-style CTAs + cluster split-K (claude/t3/E8/e8c.py)
CTA = 16-row tile x K/C (C = 4 or 8), all loads at CTA start (32-56 regs, 2-3 CTAs/SM, 1024-2048 CTAs), smem
reduction, partials pushed into the leader CTA's smem (st.shared::cluster) + barrier.cluster arrive.release /
wait.acquire; leader sums in rank order. Correct (one-hot exact). Same run: pooler bo12 12.19 | C8 x 16 warps 22.4-23.7
| C8 x 8 warps x 2 units 18.2 | C4 x 16 x 2 17.9; qkv bo12 7.49 | C8 14.2 / 11.2. Dead (as G / Q2 / V found for
clusters): non-leaders leaving after arrive (LW) -1.2 us of 11, still 2x slower.

## iteration 4b — e8p (qkv): 132 full tiles + 12 remainder tiles in 11 K pieces each (one piece CTA per SM)
Piece CTAs (blockIdx 0..131, 12 warps x 1 unit, all loads at start) publish 16x8 fp32 partials into a sentinel buffer
(relaxed stores); the last piece of each remainder tile polls its 10 siblings, sums in piece order, adds escapes and
stores (one-hot exact). Same run: qkv bo12 7.79 | e8p 9.04. Trace (smid): the hardware scheduler does NOT place one
piece + one full CTA per SM: pieces landed on 124 SMs (up to 2 per SM) and 8 SMs got 2 full tiles - static per-SM
balance by blockIdx order is impossible; equal-work CTAs (stream-K) would be needed, and their fix-up waits sit at the
end of the kernel. qkv stays with bo12.

## iteration 5 — shipped m8 (pooler m8 = e8 16,3,2,MB2), mandatory analysis
1. nsys in-graph (nsys_shapes.py, 20 reps, box 0): m8_pooler 13.22 us (bo11/bo12 smallm2_gemv: 13.55 in dash_bo11).
2. ncu (--set full, flushed, single launch; ncu-inflated durations; claude/t3/ncu_run.sh e8a):
   dur 15.26 us (smallm2 16.16 under the same tool), DRAM read 50.54 MB, DRAM throughput 72.6 % of peak = 3.49 TB/s over
   the ncu duration, L2 72.7 %, L1/TEX 35.7 %, SM 15.0 %, issue slots busy 11.9 % (issue-active 14.6 %), hmma 2.2 %,
   achieved occupancy 50.2 % (theoretical 53 %, 2 CTAs x 17 warps, limit = registers), eligible 0.23 warps/clk,
   clock 1.74 GHz; stalls long_scoreboard 81 %, barrier 8 %, wait 3 %.
3. SASS (NVRTC 13.0, sass_m8.py): 56 regs, 0 spill/stack, LOCAL 0, 576 instr, 49 LDG; per warp 32 HMMA, decode
   7 ALU / 8 weights (2 LOP3 + 1 SHF + 4 PRMT per decode12f), 5 LDG.128 per unit (3 W .NA + 2 x), loads of units
   u + 1 / u + 2 interleaved with unit u's HMMAs (software pipeline), STS/BAR/LDS only in the epilogue.
4. Binding unit: DRAM stream (latency/queueing-bound tail, not bandwidth): 50.5 MB in ~11.4 us of kernel time in the
   graph (local 12.36 - ~1.0 launch/flush offset) = ~4.4 TB/s average vs ~4.56 sustained; tensor 2 %, issue 12-15 %.
   What is left above the hw-scheduled stream floor (11.17-11.41 for 50.3 MB): x delivery 0.23 us + the static per-warp
   assignment's tail (in-CTA warp spread ~1 us around a ~0.4 us mean lag, barrier + reduction + store 0.26 us).
   Next: cut bytes (F11 needs ~free escapes) or remove the static-assignment tail.

## iteration 6 — layout / cache-policy knobs and qkv with 8-warp CTAs (both boxes, 5 graphs x 40 reps)
| pooler variant | box 1 vs bo12 | box 0 vs bo12 |
|---|---|---|
| shipped (G's tile layout: lo and nibble regions = 2 streams per warp) | 12.16 (-0.24) | 12.45 (-0.17) |
| unit-contiguous blob [unit][lo 2U, lo 2U+1, nib U] (1 stream per warp) | 12.10 (-0.34) | 12.40 (-0.11) |
| unit-contiguous + x L1::evict_last | 12.13 (-0.27) | 12.32 (-0.30) |
| x L1::evict_last | 12.19 (-0.18) | 12.48 (-0.24) |
All within the +-0.1 us noise: not adopted. qkv with 8-warp CTAs (16 units / warp, 3 CTAs/SM by regs): 8.35 / 8.50 vs
16-warp 8.54 vs bo12 7.78: the 12 doubled SMs still end last (per-SM end p10 7.94, max 8.96).
ncu e8a extra: L1/TEX hit rate 19.9 % (x is 40 % of L1 requests -> ~50 % of x requests hit L1: the two CTAs of an SM
share x), L2 hit rate 16.5 %, mem busy 54 %, warp cycles per issued instruction 55.

## iteration 7 — e8q (qkv): placement-proof balance: 132 CTAs x 1024 threads (exactly one per SM)
CTA c = full tile c + piece c % 11 of remainder tile c / 11 (23-24 half-units of 32 k processed FIRST, one per warp;
the last piece warp (smem counter) publishes the piece partial ~1-2 us into the kernel); the finalizer (piece 10)
sums the 10 sibling partials in its epilogue. Work unit = half-unit (LDG.128 lo + LDG.64 nibble + LDG.128 x), 32 warps
x 8 half-units, ring 2 ahead, 64 regs. One-hot exact. Same run (local us):
| variant | qkv m8 | bo12 |
|---|---|---|
| e8q, finalizer polls one sibling at a time (10 dependent L2 round trips) | 11.09-11.30 | 7.86-7.97 |
| e8q, all 10 polls issued before the CTA barrier | 9.41 | 7.74 |
| no poll (wrong) / no piece work (wrong) | 8.93-8.98 / 8.94 | |
| e8q with escape smem writes inside the ring (gather stall at unit 1) | +0.2 | |
Reading: the fix-up itself is now ~0.45 us (one L2 round trip + the finalizer's extra epilogue), but the 1-CTA-per-SM
32-warp half-unit structure is already 1.2 us slower than bo12 before any fix-up (as e8r: one big CTA per SM loses to
two 17-warp CTAs). Dead end for qkv; no further qkv design tried.

## iteration 8 — x L1 prefetch, compute-free ablations
- Escape warp prefetching all of x into L1 (prefetch.global.L1, 1024 lines per CTA): 13.9-14.1 vs 12.03 (+1.8 us;
  same sign as C2's L2-prefetch finding). Dead.
- "No mma" ablations (loads consumed by xor) are slower (13.3-16.8) because ptxas re-schedules the loads without a
  consumer - not usable as stream probes for this kernel. No-x ablation: 12.02 vs 12.26 (x = 0.24 us), repeatable.

## Conclusions (measured)
- m = 8 needs no issue headroom with tensor cores (12-15 % issue, 2 % tensor); the whole problem is the stream shape.
- pooler m8 runs at "m = 1 speed for F12 bytes": e8 12.0-12.4 local vs agent G's m = 1 g311 11.62 on 3.5 % fewer bytes
  (F11) - scaled to F12 bytes g311 would be ~12.0. The remaining m = 1 advantage is F11, whose 1.36 % escapes have no
  cheap exact path in an mma kernel (gathers + per-escape work in the consumers, as agent F found; patching A fragments
  needs runtime-indexed registers; escape-MMA groups need ~5 % extra bytes plus dependent gathers).
- The e8 kernel drains DRAM at ~4.55 TB/s from ~0.6 us to ~11.7 us; the rest is the 0.26 us epilogue and CTA end skew.
- qkv m8: 144 tiles on 132 SMs. Static LDG designs leave 12 SMs with 2 tiles (end +1 us); byte-balanced shapes lose
  because a CTA's progress is not proportional to its bytes in flight (per-SM delivery saturates near 45-50 GB/s for LDG
  rings); clusters are 2x slower; blockIdx-ordered piece CTAs are placed unevenly by the hardware scheduler. bo12's TMA
  ring (V2 qkvb, 96 CTAs x 24 rows) stays best.

## same-run bench.py tables (local us, speedup vs SGLang in the same run; judge ~= local + 1.0)
| run (box) | SGLang pooler m8 | bo12 | m8 | delta | identical-kernel shapes (qkv/o/unp/router) bo12 vs m8 |
|---|---|---|---|---|---|
| 1 (0) m8only | 20.11 | 12.56 x1.601 | 12.36 x1.627 | -0.20 | |
| 2 (1) integrated sub_m8 | 19.82 | 12.33 x1.608 | 11.97 x1.656 | -0.36 | -0.05 / -0.01 / -0.05 / -0.13 |
| 3 (0) integrated sub_m8 | 19.97 | 12.57 x1.589 | 12.14 x1.644 | -0.43 | -0.15 / -0.04 / -0.09 / -0.09 |
| 4 (1) integrated, 2 layers | 19.76 | 12.40 x1.593 | 11.89 x1.662 | -0.51 | qkv -0.17 |
| 5 (1) integrated, 2 layers | 19.87 | 12.53 x1.586 | 12.21 x1.628 | -0.32 | qkv -0.13 |
| 6 (0) integrated, 2 layers | 20.10 | 12.53 x1.604 | 12.08 x1.663 | -0.45 | qkv +0.03 |
| 7 (0) integrated, 2 layers | 20.05 | 12.55 x1.598 | 12.26 x1.636 | -0.29 | qkv -0.18 |
Mean over the 6 integrated / m8only runs: pooler m8 -0.38 us (speedup vs SGLang x1.596 -> x1.648, +3.3 %); the
byte-identical control shapes move -0.09 us on average (per-instance offset), so the net kernel gain is ~0.3 us.
verify-1 load (5 shapes, runs 2/3): bo12 29.55 / 30.06 -> 28.97 / 29.26 us; judge-projected load speedup x2.1510 -> x2.1878
(+1.7 %) / x2.1294 -> x2.1786 (+2.3 %), of which ~0.1 us per shape is the per-instance offset seen on identical kernels.

Gates (integrated sub_m8 = bo12 + m8): `check.py claude/t3/E8/sub_m8.py --graph` -> "checked 259 cases x 2 seeds +
graph: ALL OK". `exact_all.py sub_m8.py --layers pooler,qkv_gate --nesc 3000 --ms 2,5,8,9,16,17,32,33,128,256` -> EXACT ALL
OK; at m = 1 exact_all reports 1-2 one-hot mismatches for pooler / qkv_gate with BOTH bo12.py alone and sub_m8 (agent
G's g311 flushes injected values < 2^-39 - the documented exception in AGENT_RULES2; not an m8 path).
`exact_all.py sub_m8only.py --layers pooler --ms 8 --nesc 200 / 3000` -> EXACT ALL OK.

## Files
- `e8.py` lab (static e8, dynamic e8d, IL / L1-policy / trace / ablation bits), `e8r.py` (byte-balanced CTAs + half
  tiles), `e8c.py` (cluster split-K), `t_e8.py` (same-run lab timing vs bo12's kernels built standalone from bo12.py,
  --check, --trace per-CTA globaltimer), `sass_e8.py` / `sass_m8.py` (NVRTC SASS stats), `m8.py` + `sub_m8.py` +
  `sub_m8only.py` (deliverable).

## Not done / ideas for others
- pooler: the e8 kernel is DRAM-drain + epilogue bound (~4.55 TB/s from 0.6 to 11.7 us, then 0.26 us epilogue); a lower
  number needs fewer bytes (F11 with an escape path costing < ~0.2 us; none found for mma consumers) or a stream at the
  hw-scheduled probe's ~4.8 TB/s, which needs short independent CTAs and therefore a cheap cross-CTA K reduction
  (everything measured here and by S/G/Q2 costs >= 0.5 us: clusters 2x slower, polls/atomics one L2 round trip in the
  tail, PDL converter +0.6).
- qkv m8: a byte-balanced split of the 12 remainder tiles is the only lever; it needs 2 CTAs per SM (the 1024-thread
  one-CTA-per-SM structure is 1.2 us slower) and the hardware scheduler's CTA placement is not controllable without a
  runtime per-SM role atomic (cold, ~0.7-1 us at kernel start).
