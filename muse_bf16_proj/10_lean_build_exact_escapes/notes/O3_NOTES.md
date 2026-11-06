# Agent O3 — layer o (W [8192, 1024] BF16, F12 stream 12.6 MB) at m = 16 / 32: short-tail full-residency kernel

## RESULT (read first)
- SHIPPED: claude/t3/O3/o3.py (25.9 KB, readable, no encoded data; prepare_o3(weights) / run_o3(x, config, state) ->
  None except layer o at m == 32) + sub_o3.py (LN/bo14_lean + o3) + sub_o3_only.py (o3 + torch). prepare (one F12
  pack + one NVRTC module + self-test; measured 0.53 s on box 0), never raises (any failure / > 252 escapes per CTA -> None -> base).
  Integration: `out = run_o3(x, cfg, st)` before the base for layer o (only m == 32 is claimed).
  Kernel: 128 CTAs x 64 rows x all K, full residency (8 W items + 4 x pieces requested at t = 0, stage-major), two
  warpgroups alternating whole items with an RS-wgmma consumer (A = decoded F12 W in registers, B = x by SW128
  descriptor), smem exchange of the two partial sums, exact two-term escapes from producer-built smem planes.
- m = 16: NOT owned (best o3 config ties the base: +0.04 / -0.02 / +0.05 / +0.08 in four runs).
- Same-run tables, final code, box 0 (local us; judge ~= local + 1.0):
  | run | harness | SGLang | base bo14L (sk11 o32) | o3 | delta / speedup vs SGLang |
  |---|---|---|---|---|---|
  | 1-3 | t3o.py, XCL 4 x clones, 40 reps (bench_final_same.log) | 5.46 / 5.51 / 5.49 | 4.58 / 4.55 / 4.52 | 4.54 / 4.49 / 4.47 | paired d -0.052 / -0.064 / -0.052 |
  | 4-6 | bench.py same process, 60 reps | 5.82 / 5.74 / 5.92 | x1.187 / x1.148 / x1.163 | x1.188 / x1.207 / x1.204 | +0.1 / +5.1 / +3.5 % |
  | 7-9 | bench.py alternating single-candidate processes (bench_final_alt.log) | 5.92-6.02 | x1.192 / x1.204 / x1.180 | x1.200 / x1.199 / x1.215 | +0.7 / -0.4 / +3.0 % |
  Earlier runs of the same design: t3o -0.09 / -0.05 / -0.12 / -0.16 / -0.15; bench same-run x1.164 -> 1.211,
  1.166 -> 1.208, 1.180 -> 1.211; alternating +1.4 / -0.1 / +3.3 %. int_v7 at o m32: x1.105 (base_v7.log).
  => o m32 -0.05..-0.16 us (~+1-3 %), ~+0.02-0.05 % of the final score (0.34 % per us).
- nsys in-graph (one process): base sk11_o_32 5.57 us -> o3_o_32 5.44 us. ncu: 6.75 -> 6.53 us (analysis block of
  iteration 5).
- Correctness (gates_final.log): check.py claude/t3/O3/sub_o3.py --layers o --graph (NaN-poisoned) -> "checked 51
  cases x 2 seeds + graph: ALL OK"; exact_all.py sub_o3_only.py --layers o --nesc 200 -> EXACT ALL OK; --nesc 3000 ->
  EXACT ALL OK; chk_esc.py --nesc 3000: o3 itself serves m32 with 3708 escapes (max 45 per CTA, cap 252), one-hot on
  the injected columns 0 mismatches of 262144. build_t3.prefix_names(o3.py, '_o3_') -> pfx/sub_prefixed.py: check.py
  --layers o --ms 31,32,33 --graph ALL OK.
- Lab / tools (not shipped): o3lab.py (all variants: RG/KS mma.sync, NA, NXP, EARLY, WGM, WGA, NSH, NB, trace with an
  observer warp, ablation flags), t3o.py (same-run harness, --v lab / --o3v module / --pk P5 ring variants, --trace),
  lsrc.py / lsrc_o3.py (local nvcc sm_90a build + SASS stats), ncu3*.sh, chk_esc.py; snapshots o3lab_v1..v7.py.
- Transferable facts (measured here): (1) mma.sync consumers of F12 W run at ~11-13 clk / HMMA / SMSP whatever the
  warp count or accumulator ILP (issue + decode + LDSM); (2) RS-wgmma only pays once the per-item code is straight-
  line (compile-time descriptor offsets, one batch per item, warpgroups alternating items): 105 -> ~45 instr / 4 k16;
  (3) every extra bulk copy issued by one thread costs ~0.04-0.07 us of stream end; (4) mbarrier try_wait on a
  completed phase still costs ~100 clk under load: wait each barrier once per warp; (5) a [&] lambda passing &acc to
  the wgmma asm, or any runtime index into acc, sends the accumulators to local memory and ptxas serialises every
  wgmma (C7514 / C7515); (6) trace stamps in a u64 array indexed by the item go to local memory (+1-2 us): stamp with
  direct stores from lane 0; (7) ablations that drop an LDSM's results let ptxas delete the LDSM (misleading).

Base for these shapes (claude/t3/LN/bo14_lean.py): agent F's sk11 (LN/sk11_lean.py), 256 CTAs x 32 rows, 2 CTAs / SM,
x re-streamed per stage; o16 cfg (32,2,2,2,2,1,72 KB,F12|bake) [O2], o32 cfg (32,2,2,2,2,1,100 KB,F12|bake) [F].

## Design (o3lab.py, inherited from the stopped first O3 run and checked line by line)
- 128 CTAs x 64 W rows x all K (one CTA per SM, 4 SMs idle). Every byte is requested at kernel start (full residency):
  the CTA's 96 KB of F12 W is cut into NI k-items (ITEMS = k64 counts, e.g. 4-4-4-2-2), one bulk copy and one mbarrier
  each, stage-major in global memory ([item][cta][warp][granule][lane][16 B]) so all SMs stream one region at a time.
- x: one 3D TMA (box 64 k x MT tokens x 16 k-blocks, 128B swizzle) issued after XPOS W items; x read once per SM
  (8 MB L2->SM at m32 vs 16 MB for sk11's 256 CTAs).
- consumers: RG x KS warps; warp (rg, ks) owns 16 * 4/RG rows x all MT tokens and k-split ks of every item.
  W is the mma A operand (16 rows, decoded from 3 LDS words per lane per k16: agent G's F12 decode, 7 ALU / 8 weights),
  x^T the B operand (ldmatrix.x4 from the swizzled x). KS = 1: no reduction; KS = 2/4: one smem exchange at the end.
- escapes (exact, agent N's two-term form): out = (sum x * fake + A) * 2^(E_lo - 112) + B with
  A = sum_esc(-x * 2^-15), B = sum_esc(x * true). The producer warp (lane = token) builds the A / B planes in smem from
  the resident x while W streams; the epilogue reads its two float2 per output pair (always-executed code, no list scan).
- flags: 32 early dummy store (warms the out param + page; same thread rewrites that word at the end), 64 dry pass of
  the exchange / epilogue at kernel start (instruction-cache warm-up), 2 per-warp trace, 256 W-word prefetch order.

## Measurement
claude/t3/O3/t3o.py: judge style (512 MB flush, CUDA-graph replay, empty graph subtracted), XCL = 4 x clones at random
4 KB-aligned addresses per candidate (agent P5), replays interleaved, `d` = paired delta vs the first candidate
(base = lean sk11 rebuilt in the same process). Correctness per candidate in the same run: normal, outliers, one-hot
(bit-exact) with NaN-poisoned memory.

## iteration 1 — first measurements of the inherited design (runs run1.log / run2.log, box 0, XCL 4, 30 reps, local us)
cfg = "MT,RG,KS,NA,ITEMS,XPOS,FL[,NXP]" (o3lab.parse_cfg). Same run, base = lean sk11 (bo14L):
| variant | m16 | m32 |
|---|---|---|
| bo14L (sk11) | 4.12-4.16 | 4.59 |
| a: RG4 KS2 (8 consumer warps), items 4-4-4-2-2, x after W0, early store | 4.48-4.52 (+0.32..+0.38) | 5.14-5.19 (+0.58) |
| b: RG4 KS1 (4 warps, no exchange), 4-4-4-2-1-1 | 4.71 | 5.27 |
| c: RG4 KS4 (16 warps), 4-4-4-4 | | 5.39 |
| d: RG2 KS2 (4 warps, 2 m16 tiles each) | | 5.18 |
| e: a + dry epilogue pass (FL 64) | 4.51 | 5.12 |
| x: a + EARLY exchange (FL 1024) | 4.61-4.68 | 5.04-5.18 |
| a + W-word prefetch order (FL 256) / 8 items of 2 / x first (XPOS 0) | | 5.16 / 5.04 / 5.20 |
| **ablation: no compute (FL 4)** | **3.47** | **3.91** |
| ablation: decode only, no HMMA (FL 512) | | 4.25 |
| ablation: no escapes (FL 16) | | 5.14 |
- The stream of this kernel (no compute) is 0.65-0.7 us BELOW the base (3.47 / 3.91 vs 4.12 / 4.59): the full-residency
  stream + x read once per SM + epilogue is fine. The loss is entirely the consumer: decode adds 0.34, HMMA 0.94 (m32).
- First trace build (run1) was perturbed (+1.3..2.4 us): the u64 ti[16] trace array went to local memory (128 B stack per
  thread, ~36 KB of STL traffic per CTA during the stream). Fixed: trace stamps stored directly by lane 0.
- Trace (run2, a32, us from first CTA start, mean/max over warps; the item time is when the consumer's wait for the
  item finished): x lands 1.98/2.07, items i0 2.04 | i1 2.84/3.46 | i2 3.46/4.07 | i3 4.05/4.68 | i4 4.41/5.03 | loop
  end 4.72/5.37 | exchange 4.87 | end 5.08/5.74. No-compute build: x 1.78, items land 1.87 / 2.64 / 3.24 / 3.43 / 3.54
  (max 3.83). => consumer is behind from item 1 on; it needs ~0.15-0.17 us per k64 (64 rows) at BOTH m16 and m32
  (4 vs 2 HMMA per warp-k16-step): the cost does not scale with HMMA count -> latency chain, not HMMA throughput.
  Per SMSP that is ~67 clk per warp-k16-step (2 warps / SMSP); the serial accumulator chain (k16 step j + 1 uses the
  accumulators of step j) x HMMA latency is the suspected bound (sk11 pooler32 shows the same ~80 clk per k16 step
  with 1 warp / SMSP, P5 / X traces).
- Also: x (one 64 KB TMA) lands at ~1.8-2.0 us, so the consumer starts late; the stream window is only ~1.9 -> 3.5 us.

### iteration 1 analysis block
1. nsys: not run for these variants (all lose by >= 0.3 us in the same-run bench; nsys follows for the candidate).
2. ncu: pending (next iteration's candidate); the ablations above give the split: stream+epilogue 3.91, + decode 0.34,
   + HMMA chain 0.94 (m32).
3. SASS (local nvcc 12.1 sm_90a, NVRTC on the box agrees: a32 70 regs, b32 65, a16 56, b16 49, LOCAL 0): 0 spills
   (trace builds had 128 B stack before the fix); hot group loop per warp (a32): 3 LDS.128 W words + 8 LDSM.x4 x +
   4 x (2 LOP3 + 1 SHF + 4 PRMT) decode + 16 HMMA.16816.F32.BF16, no LDG; producer 6 UBLKCP + 1 UTMALDG.3D.
4. Binding unit: the consumer's dependent HMMA chain (tensor pipe ~45 % busy during the consumer phase at m32, far
   less at m16), not DRAM (the no-compute stream finishes at 3.5-3.8 us, 4.6 TB/s over the stream window). Next:
   break the chain (NA accumulator sets: o3lab's NA was a no-op template parameter), start earlier (x in pieces
   that land with their W items: NXP), observer-warp trace with true landing times.

## iteration 2 — accumulator sets (NA), x in pieces (NXP), observer trace, ncu (run3-run5, box 0, XCL 4, 30 reps)
New lab features: NA accumulator sets (k16 step jj -> set jj % NA), NXP x pieces (box 64 k x MT x 16/NXP), observer warp
in trace builds (landing time of each item), all stamps rebased on the CTA's clock64 (globaltimer only aligns CTAs).
| variant (cfg) | m16 | m32 |
|---|---|---|
| bo14L | 4.15-4.24 | 4.58-4.70 |
| RG4 KS2 NA1 NXP1 (a) | 4.82 (run3; 4.48-4.52 in run1-2) | 5.56 (run3; 5.14-5.19 in run1-2) |
| NA2 / NA4, NXP1 | | 5.65 / 5.59 |
| NA4 + NXP4 (n4x) | 4.70-4.78 | 5.12-5.14 |
| NA1 + NXP4 | | 5.14 |
| n4x own W blob copy (placement control, same process) | 4.77 | 5.17 |
| n4x + 8 items of 2 k64, NXP8 | 4.88 | 5.29 |
| n4x x first (XPOS 0) | | 5.41 |
| RG4 KS4 (16 warps) NA2 4-4-4-4 + EARLY | | 5.98 |
| smaller first items 2-2-4-4-2-2 | 4.71 | 5.10 |
| RG2 KS4 (8 warps) 2-2-4-4-2-2 | 4.68 | |
| no compute (n4x + FL 4) | 4.17 | 3.98 |
- run-to-run: the same a32 binary moved 5.19 -> 5.56 between processes while the base did not; inside one process an own
  W-blob copy agrees within 0.04 (placement is not the cause of the iteration-1 losses; between-process drift is why
  every comparison here is same-run).
- NA (independent accumulator chains) changes nothing; 16 consumer warps are no faster per item than 8 (0.51-0.55 vs
  0.47-0.51 us per 4-k64 item) -> the consumer is throughput-bound on a shared SM resource, not latency-bound.
- x in 4 pieces: -0.45 at m32 (consumers start on piece 0 instead of waiting for all 64 KB).
- Trace (run5, CTA clock domains, n4x m32): items land 1.89 / 2.51 / 3.20 / 3.43 / 3.61 (mean; max 3.91); consumer busy
  0.51 / 0.49 / 0.47 / 0.22 / 0.22 us per item; at EVERY item boundary 0.15-0.20 us of pure overhead after the data had
  landed: 0.10-0.12 of it is the x-piece try_wait loop on barriers that completed long before, the rest the item
  barrier try_wait itself (SYNCS ops queue behind the saturated smem pipe). m16 the same: busy 0.38 / 0.37 / 0.38 /
  0.21 / 0.21, overhead 0.14-0.17 per boundary; loop end 4.56 vs last landing 3.55.
- Consumer bound = SHARED MEMORY BANDWIDTH: per SM at m32 the consumer moves x via LDSM 4x (each of the 4 row-group
  warps re-reads x: 256 KB) + W LDS 96 KB + TMA writes 160 KB = 512 KB = 4000 clk at 128 B/clk = 2.3 us at 1.72 GHz
  = the measured busy sum (~1.9-2.3 us); m16: 128 + 96 + 128 = 352 KB = 1.6 us = measured 1.55. (The decode-only
  ablation looked cheap only because ptxas deletes LDSMs whose results are unused.)

### iteration 2 analysis block (n4x = RG4 KS2 NA4 NXP4, items 4-4-4-2-2)
1. nsys: not run (candidate loses by 0.4-0.55 us in the same-run bench).
2. ncu (claude/t3/O3/ncu3.sh, full set, flushed; ncu_it2.log):
   | kernel | dur | DRAM rd | grid x blk | regs | smem | occ % | issue % | hmma % | elig/clk | top stalls |
   |---|---|---|---|---|---|---|---|---|---|---|
   | o3 n4x m32 | 7.30 | 12.67 MB | 128 x 288 | 133 | 194 KB | 13.5 | 30.1 | 5.4 | 0.41 | long_sb 25, wait 15, barrier 13, short_sb 11 |
   | sk11 o32 (base) | 6.82 | 12.80 MB | 256 x 288 | 92 | 98 KB | 24.7 | 20.8 | 5.9 | 0.27 | long_sb 43, short_sb 14, wait 13 |
   | o3 n4x m16 | 6.75 | 12.64 MB | 128 x 288 | 97 | 149 KB | 13.8 | 28.9 | 2.8 | 0.39 | long_sb 29, wait 17, barrier 14 |
   | sk11 o16 (base) | 6.18 | 12.76 MB | 256 x 288 | 54 | 68 KB | 25.3 | 24.7 | 3.3 | 0.35 | long_sb 52, wait 15, imc_miss 6 |
   SASS-level samples (o3 m32, 607 samples): start barrier (consumers wait while the producer issues ~10 copies) 21 %,
   item waits 33 %, group loop 33 % (short_sb on HMMA operands from LDSM / decode), exchange 4 %, epilogue 9 %.
3. SASS (local sm_90a, NVRTC on the box agrees): n4x m32 133 regs (NA4), 0 spills, LOCAL 0; group loop = 3 LDS.128 +
   8 LDSM.x4 + 28 decode ALU + 16 HMMA + ~12 address/loop ops per warp per 4 k16 steps; producer 5 UBLKCP + 4 UTMALDG.
4. Binding unit: shared-memory bandwidth of the consumer phase (~4000 of the ~4300 clk of busy time at m32; tensor pipe
   ~45 % during that phase: 256 HMMA/SMSP x 6.2 clk = 1590 clk), plus ~0.8 us of item-boundary barrier overhead.
   Next: one barrier per item (x pieces complete on the barrier of the first item that needs them), and warps that
   own more rows (RG 2 / 1: x LDSM traffic 2x / 1x instead of 4x).

## iteration 3 — one barrier per item, row-heavy warps, ablations, issue order (run6-run10, box 0, XCL 4, 30 reps)
- One barrier per item (x piece q completes on the barrier of the first item that needs it; expect_tx = W item + its
  pieces): m32 n4x 5.13 -> 4.83 (-0.30); x-wait at boundaries 0.10-0.12 -> 0.02-0.03 us.
- Row-heavy warps (fewer x re-reads): RG2 KS4 (g2, 32 rows / warp) m32 4.72-4.89 (best: +0.13..+0.21 vs base 4.55-
  4.71), m16 4.68-4.70 (+0.43..+0.56); RG1 KS4 (4 warps, 64 rows each) 5.12 / 4.75 (1 warp per SMSP: latency).
- Ablations on g2 (run7): no x LDSM 5.00, no W LDS 4.83, neither 4.88 vs 4.75 full -> NOT smem-bandwidth bound (the
  iteration-2 bandwidth model is wrong; the decode-only ablation of iteration 1 was misleading because ptxas deletes
  LDSMs with unused results). The consumer runs at ~11-13 clk per HMMA per SMSP = agent T's loopbench figure for
  LDS + decode + LDSM + HMMA (10.1-13.0; 6.0-6.3 for HMMA alone; decode on the half-rate INT pipe).
- 8 equal items (2 k64 each) / early non-blocking probe of the next barrier / test_wait spin: no gain (4.78-4.94).
  Trace stamps themselves cost ~0.05-0.08 us per item boundary (trace builds +0.3-0.4 us): the "excess" at
  boundaries in the trace is mostly instrumentation; the untraced bench barely depends on the item count.
- Distributed issue (each consumer warp issues its own W items after __syncthreads): +0.4..+0.7 (first landing
  1.5 -> 1.9 us, items land in a burst). Dead.
- No-compute floors (same kernel, FL 4): 5 items + 1 x copy 3.60 (m16) / 3.88 (m32); 8 items + 8 x pieces 4.19 /
  4.20 -> every extra bulk copy costs ~0.04-0.07 us of stream end (single-thread issue ~60 clk each + later last
  landing). Escape pass moved to per-piece processing (it had waited for the last x piece: up to 0.24 us at m16).
- m32 g2 timeline (run6 trace): items land 1.56 / 1.82 / 2.50 / 3.11 / 3.35 / 3.52 (max 3.67); half of K lands after
  3.1 us -> ~0.9 us of consumer work after that point; then exchange (KS4) 0.27 + epilogue 0.20 us.

### iteration 3 analysis block (g2 = RG2 KS4 NA2, items 2-2-4-4-2-2, NXP4)
1. nsys: not run (still +0.13..+0.56 vs base in the same-run bench).
2. ncu: not rerun (same kernel family as iteration 2; the ablations above replace the stall breakdown).
3. SASS (local sm_90a): g2 m32 86-100 regs, m16 ~92, 0 spills, LOCAL 0; group = 3 LDS.128 + 4 LDSM.x4 (m32) +
   4 x 7 decode ALU + 32 HMMA per warp per 2 k16 steps x 2 tiles; no LDG.
4. Binding unit: the mma.sync consumer issue/latency mix (~11-13 clk/HMMA vs 6.2 pipe peak; INT pipe for decode),
   aggravated by the end-of-stream burst (half of K lands in the last 0.4 us), plus 0.4-0.5 us of exchange +
   epilogue. The stream itself (no compute) is 0.3-0.6 us under the base.

## iteration 4 — pk ring (P5) on o, low-copy layouts, RS-wgmma consumer (run11-run12)
- P5's pk ring kernel applied to o (K_OF / N_OF extended at run time, t3o.py --pk): XLAG 1 +0.3..+0.7 (o has only 4
  stages per CTA: lagging x starves the first stage), ESC1 (1 escape warp) without XLAG -0.08 at m32 (noise level),
  pk base-equivalent config = base. Not pursued.
- Copy count vs start: 5 items + 1 x copy (n5x1) is the best no-compute floor but the consumer then starts after all
  of x has landed: m32 5.32, m16 4.66 (worse than NXP 4 with 6 items).
- RS-wgmma consumer (WGM, flag 262144; RG4 KS2 with warp = ks * 4 + rg so each k-split is one warpgroup; A = the
  decoded mma A fragments (identical layout), B = x by SW128 K-major descriptor (SBO 1024, +32 B per k16, agent O's
  osw.py form); accumulators = the mma C layout, so exchange / epilogue unchanged): one-hot exact on the first build,
  but NO faster (0.27 us per 2-k64 item, as mma.sync): the group loop was ~105 SASS instructions per warp per 4 k16
  (28 decode, 4 HGMMA, 3 LDS, ~40 descriptor rebuilds incl. 8 R2UR, register copies) and every 2-k64 item is a single
  LDS -> wait -> decode -> HGMMA chain per warpgroup => issue / latency bound, not tensor bound.

## iteration 5 — WGA: warpgroups alternate whole items (run13-run18) => the first win (m32)
- WGA (flag 524288): items packed for ONE warpgroup (layout KS 1, all items equal = CI 64-k blocks, compile-time
  descriptor offsets), warpgroup g processes items g, g + 2, ...: per item 3 CI LDS.128, retire previous batch, decode
  4 CI fragments, fence, 4 CI HGMMA, commit (~12 SASS instr per k16 instead of 26). x pieces read by an item can sit
  on the OTHER warpgroup's barrier (first item needing them): wait for it too (missing that wait = garbage x, found
  by the check, fixed).
- run13-15 (lab, XCL 4, 40 reps): m32 8 items NXP4 -0.08..-0.11 (base 4.66), 4 items -0.06; shared last items (k-split
  between both warpgroups, NSH 1..3) and two batches per item (decode of the 2nd half overlapping the 1st half's
  HGMMA, NB 2: no C7513 serialisation, verified in SASS) within noise; m16 best +0.08..+0.18.
- Lean module o3.py: two NVRTC hazards met and fixed while porting: (1) a [&] lambda calling the wgmma helper with
  &acc made acc escape to local memory (64 B stack) -> ptxas C7514 / C7515 (wgmma serialised); the item body is a
  __forceinline__ template taking acc by reference; (2) a runtime index acc[(ks ^ 1) * NO + o] in the exchange also
  forced local memory -> per-element selects. Final: 84-96 regs, 0 spills, LOCAL 0, no C75xx.
- Each warp waits each x-piece barrier only once (xdone): m32 -0.15..-0.16 (run17, run18; base 4.75-4.81 there),
  m16 +0.04 / -0.02 (tie: not owned). x issued one item after its W (XPOS 1) beats XPOS 0 (+0.10) and XPOS 2.

### iteration 5 analysis block (shipped o3_o_32: NI 8, NXP 4, XPOS 1, NSH 0, NB 1)
1. nsys in-graph (one process, prof_final.log): sk11_o_32 (base) 5.57 us vs o3_o_32 5.44 us.
2. ncu (full set, flushed, ncu_run.sh):
   | kernel | dur | DRAM rd | grid x blk | regs | smem | occ % | issue % | tensor | elig/clk | top stalls |
   |---|---|---|---|---|---|---|---|---|---|---|
   | o3_o_32 | 6.53 | 12.67 MB | 128 x 288 | 96 | 194 KB | 13.6 | 21.0 | gmma inst 3.1 % of peak | 0.25 | long_sb 38, barrier 17, wait 12 |
   | sk11_o_32 (base) | 6.75 | 12.80 MB | 256 x 288 | 92 | 98 KB | 24.7 | 22.3 | hmma 5.9 % | 0.29 | long_sb 47, wait 14, short_sb 12 |
   o3 extra: ALU pipe 22.7 % of peak (decode), L1/smem throughput 25.8 %, 859 K warp instructions.
3. SASS (local nvcc 12.1 sm_90a = NVRTC structure; box NVRTC 96 regs): 0 spills, LOCAL 0, no C7513/C7514/C7515;
   per item per warp 6 LDS.128 + 1 WARPGROUP.DEPBAR + 8 x (2 LOP3 + 1 SHF + 4 PRMT) + WARPGROUP.ARRIVE + 8 HGMMA.64x32x16
   with immediate descriptor offsets; producer 8 UBLKCP + 4 UTMALDG + 1 escape-list UBLKCP; no LDG; 2 STG per output pair.
4. Binding unit: DRAM stream + its end-of-stream burst. Items land 1.6 -> 3.5 us (mean, max 3.8; 12.7 MB = ~4.4 TB/s
   over the window, ~92 % of 4.8); the consumer keeps up until the last ~3 items, which land within ~0.3 us; what is
   left after the last landing is one item per warpgroup (~0.25 us) + exchange (barrier between the two warpgroups,
   ~0.1-0.25) + epilogue (~0.15). Tensor pipe ~3 % (not binding), ALU 23 % (decode). Next levers: per-item barrier
   cost (~100 clk per try_wait), the warpgroup imbalance at the end, epilogue stores.

## Notes on the logs
- bench_same.log: its three bench.py lines are INVALID (candidate label "base" collides with bench.py's internal SGLang
  key, so both columns show SGLang); the t3o lines in it are valid. bench_b.log / bench_final_*.log use label "bo14".
- run13.log first attempt: an apparent repeat failure was the old log re-read by my wait loop (stale REMOTE_EXIT);
  the WGA x-piece wait fix passed every check afterwards.

## Next levers (not done)
- m32 tail after the last landing (~0.5 us): (a) the item whose W lands last belongs to one warpgroup while the other
  idles at the exchange barrier: give the last item to both (NSH 1 was neutral with the old per-item waits; retest with
  the xdone waits), or let the idle warpgroup start the epilogue of its own half and receive the late partials;
  (b) ~100 clk per mbarrier try_wait: poll the next item's barrier while the current batch runs; (c) epilogue stores
  (2 x 4-B STG per output pair; 16-B stores via a shuffle transpose).
- m16 (tie): decode-bound (HGMMA n16 is 8 clk); same levers plus fewer copies (x is one 32 KB copy already).
