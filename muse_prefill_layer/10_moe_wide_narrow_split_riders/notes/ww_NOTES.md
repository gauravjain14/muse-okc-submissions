# ww worker "W": wide expert items of the fused NVFP4 MoE at T=753 / 3012, in a wide-only kernel (2026-10-31)

Base: research/claude/stk/s_b.py (MoE strings = mw/cu/sb_moe.cu + sb_glue.cu). Scope: kernel W (wide tiles z >= 160):
headroom, one shared epilogue, tile plan, per-item overheads, the split's own cost. Worker R (rw/) owns the rider path.
Tools (all research/claude/ww/): build.py (variants as text patches: `python3 build.py <names>` -> cu/, v_<name>.py;
`--uni <names>` -> uni/<name>.cu reduced compile unit), uni.sh + uni_rep.py (remote nvcc 13: per instantiation regs /
spills / C75xx / HGMMA / loops + DEPBAR->ARRIVE gaps, ~2 min), loopx.py (opcode mix / listing of a SASS address range),
pc.py (parallel module precompile, ~110 s per batch), mb.py (= mw/mb.py + per-SM trace grouping, smid kept, first-of-class
vs repeat epilogues), r.sh (MoE-only paired bench vs s_b, STK_FUSE=0, traces kept).

## 1. Compile probes (nvcc 13, reduced unit; logs/u_*.sass)
- Narrow (or all) consumer paths as __noinline__ device functions called from the consumer dispatch (nf1/nf2/nf3):
  ptxas C7510 "wgmma pipeline crossing function boundary at a function call" in every form (even when the kernel itself
  has no inline wgmma), callees capped at the 168-register launch bound. => one launch cannot give the wide and narrow
  paths separate allocations; separate kernels are required.
- Split (PART template on moe_fused_persistent: 1 = W = tiles z >= 160, 2 = N = the rest; build.py split()):
  W with s_b's per-width consumers (s0) or ONE consumer with a runtime-width shared epilogue (s1): no C75xx, wide loops
  all hoisted (DEPBAR->ARRIVE gaps [1, 0, 0]; s_b's 256 loop is [13, 13, 11]); W kernel 3.3-3.4K instructions (s0 6.7K,
  s_b 11.3K).
- BUT with nW from __syncthreads_count both kernels' loops left the uniform datapath (W 256 loop: 8 R2UR, BSSY/
  WARPSYNC, 131-139 instructions vs s_b's 109; N's N=16 loop 406 vs 330, 32 R2UR + vector 64-bit descriptor adds, gaps
  [36, 6, 38, ..] vs s_b's [17, 5, 13, ..]): ptxas does not treat the BAR.RED.POPC result as warp-uniform, so the item
  loop's control flow is "divergent" (mn's lesson, other cause). Fix: each warp counts with __ballot_sync (u2 = 'vote',
  the default now) or a lane-0 shuffle (u1): N's loops back to s_b's 328-331 instructions, W's 119-120 (the extra ~10 vs
  s_b = rematerialized woff/soff at the loop top, overlapping the TRYWAIT), descriptors on uniform registers.

## 2. Log
### r1 (logs/r1.json; MoE-only paired, draw 11, cases 0-3, 6 reps, cold L2): sequential split (W on all SMs, N's CTAs
### take each SM as W's CTA there exits)
| case | s_b | w0 (split, W = s_b's per-width consumers) | w1 (split + W's shared epilogue) |
|---|---|---|---|
| 0 (753) | 262.2 | 270.0 (-2.9%) | 266.7 (-1.7%) |
| 1 (753) | 263.2 | 273.3 (-3.7%) | 270.0 (-2.5%) |
| 2 (3012) | 817.0 | 819.7 (-0.3%) | 819.6 (-0.3%) |
| 3 (3012) | 757.8 | 761.1 (-0.4%) | 762.1 (-0.6%) |
All outputs bit-identical to s_b (errs 0 / 0). The shared epilogue buys 3.3 us at 753; the split itself costs 3-4% at
753, 0.3-0.4% at 3012. Traces next (where the split's cost sits).

### r2 (logs/r2.json; trace builds, draw 11 cases 0 / 2, per SM, ana.py -v): where the split's cost sits
| | s_b (sbt) | w0t (split) | w1t (split + shared epilogue) |
|---|---|---|---|
| c0 wide SM-us (ideal) | 147.0 (132.7) | 150.5 (131.0) | 147.4 (132.4) |
| c0 wide epilogue / first-stage per SM | 7.0 / 2.6 | 7.4 / 5.3 | 4.1 / 4.6 |
| c0 idle: W<->N switch / tail | 0.7 / 3.6 | 4.3 / 6.2 | 4.1 / 5.5 |
| c2 wide SM-us (ideal) | 589.0 (545.7) | 592.8 (548.7) | 589.5 (550.8) |
| c2 wide epilogue / first / main-loop excess | 22.6 / 8.3 / 7.6 | 20.8 / 7.7 / 9.2 | 16.7 / 7.2 / 9.1 |
| c2 first-of-class epilogue excess per SM | 6.87 | 7.26 | 2.07 (very first g256 epilogue 4.4-4.9 -> 1.10 us) |
- Shared epilogue confirmed by the item trace: first-of-class epilogue excess 6.9-7.3 -> 2.0 us/SM at 3012; epilogue
  total 25.3 -> 21.0 (w0t -> w1t); at 753 epilogue 7.0 -> 4.1 us/SM.
- 753: W keeps the MAT_SMALL interleave, so all wide tiles' gate items finish together and the down items wait for
  their gate items (d256 first-stage 0.54 -> 2.31 us, producer done-wait 0.52 -> 3.87 us; in s_b the narrow tiles' gate
  items sit between). Fix: W tile-major at every T (w2).
- 3012: g160 / d160 main loops +3.6..4.4 us per item in W vs +1.9..2.2 in s_b (all 64 items, not outliers), g192 +1.3
  vs +1.1; g256 better in W (+0.22 vs +0.33). SASS of the 160 loop is s_b's modulo ~9 rematerialized offset
  instructions before the TRYWAIT. In-graph ncu (below): narrow-wide loops are latency-bound at the k-block boundary.
- Clock: per-phase SM clock 1.60-1.79 GHz (all-wide phases slowest); at 3012 W's down phase runs ~1% below s_b's
  (1.66 vs 1.67-1.71 GHz) -> per-item pieces are compared in cycles, wall time only for verdicts.
- W -> N switch per SM (cta stamps, w1s, p50): W consumers leave the item loop -> publisher drained 2.3 us (its last
  down copy-out: a single-warp loop, LDS.128 -> ST.E.128 serialized on R12-R15 at 40 registers, ~1.35 us per 128
  rows), CTA teardown + N CTA launch 0.35 us, N prologue (barrier init 0.19, tile count 0.26, nW vote 0.35) 0.8 us,
  then N's first item hand-off 0.5 + first stage 1.4-1.7 us. ~5.5 us per switch, once per SM.

### r3 (logs/r3.json; MoE-only, draw 11, 6 reps): W tile-major (w2), W || N concurrency (w2c: W keeps
### ceil(132 cw / (cw + cn)) + 2 CTAs from a per-tile cost model, its other CTAs exit at once)
| case | s_b | w1 | w2 (W tile-major) | w2c (+ concurrent) |
|---|---|---|---|---|
| 0 | 262.0 | 266.5 (-1.7%) | 265.2 (-1.2%) | 267.0 (-1.9%) |
| 1 | 263.3 | 270.7 (-2.7%) | 264.3 (-0.4%) | 267.3 (-1.5%) |
| 2 | 821.3 | 822.8 (-0.2%) | 823.9 (-0.3%) | 833.3 (-1.4%) |
| 3 | 757.6 | 761.0 (-0.45%) | 761.9 (-0.6%) | 768.8 (-1.5%) |
Outputs bit-identical. Tile-major W removes the 753 down-item waits (r4 trace: d256 first 2.31 -> 0.76, pwait 3.87 ->
1.09 us). Concurrency loses: at 3012 W on ~105 SMs ran to the end (last wide end p50 745 us of 782), its items +8.8
SM-us/SM slower, tail 7.0 vs 3.8 us.

### ncu, in-graph, case 2 (fixed ROUTER_SEED routing; logs/st_sb2.*, st_w2_2.*; real clocks, cold caches)
| | s_b fused kernel | kernel W (w2) |
|---|---|---|
| duration / SM clock | 753.3 us / 1.68 GHz | 631.9 us (W only) / 1.66 |
| tensor pipe (% of elapsed) | 82.5 | 89.2 |
| issue slots busy / issued warp per scheduler | 30.3% / 0.31 | 27.3% / 0.28 |
| 256 loop: share of samples, wgmma-wait (barrier) | 51%, barrier 49% wait 13% short_sb 8% | 63%, barrier 54% wait 11% short_sb 8% |
| 160 loop | barrier 13% wait 24% dispatch 12% short_sb 11% long_sb 7% | barrier 18% wait 18% dispatch 13% short_sb 9% |
| 208 / 192 loops: barrier share | 29% / 20% | 33% / 26% |
Top non-wgmma stall in every wide loop: the k-block's first decode instruction waiting for the codes' LDS.128
(short_sb; 8% of the 256 loop) -- with one k-block per stage the codes are loaded only after the stage's full barrier,
and a 160-wide wgmma (80 cycles per warpgroup) does not cover trywait -> LDS -> decode.

## 3. Continuation (2026-10-31 ~20:50 PT, new instance; predecessor stopped mid-edit)
State found: build.py had variants w3*..z3 (compile-probed, never benched; z* = scale0 was missing its #define -> fixed:
`#define WW_SCALE0 {0,1}` in split() defs). Compile probes of the stopped instance (remote /tmp/ww_u*.res), kernel W:
| variant | W loop instr (160..256) | DEPBAR->ARRIVE gaps | W spills |
|---|---|---|---|
| w2 (vote nW, shared epilogue, tile-major) | 119-120 | [1, 0, 0] | 4 B st / 16 B ld |
| w3b / w4b / w6n / w6p (+ bigstg + pubcopy [+ wreg 88/208]) | 107 | [1, 0, 0] | 0 (with wreg) |
| w4x (xpf <= 224 + wreg, no bigstg) | 127 (256: 124) | [1, 0, 6] | 0 |
| w4 (xpf + bigstg + pubcopy + entry + wreg) | 134-135 | [1, 0, 14..17] | 0 |
| w4e / w6e / e1..e10 (entry hand-off, every mode) | 127-141 | [1, 0, 0] | 0 |
=> the producer -> consumer tile-entry hand-off (entry) costs 20-34 loop instructions in every form: dropped.
bigstg = kernel W ring slots of 37888 B + a 256-row staging buffer: a down item's whole output (both halves) goes to the
publisher (consumers no longer copy half 0 themselves); pubcopy = the publisher's row copy with 8 independent 16-B
loads in flight per lane; wreg = W's setmaxnreg split producer 88 / consumers 208 (W's consumers need <= ~190).

### r5 (logs/r5a.json, r5b.json; MoE-only paired vs s_b, draw 11, 6 reps; non-trace builds)
| case | s_b us | w2 | w4b (bigstg+pubcopy+wreg) | w4x (xpf+wreg) | w6p (w4b + tile plan TP160) | w6n (w4b + N wreg 56/224) |
|---|---|---|---|---|---|---|
| 0 (753) | 261.9 | -1.07% | -0.51% | -0.84% | -1.17% | -0.44% |
| 1 (753) | 262.0 | -0.70% | -0.16% | -0.47% | +0.52% | -0.02% |
| 2 (3012) | 819.6 | -0.17% | +0.46% | -0.22% | +0.36% | +0.23% |
| 3 (3012) | 757.5 | -0.67% | +0.15% | -0.32% | -0.23% | -0.04% |
All bit-identical except w6p (tile plan regroups rows: max err 0.0039, within tolerance). w4b = +0.5..0.8% over w2,
about s_b's speed. Next: isolate wreg vs bigstg/pubcopy vs xpf (w7r, w3b, w6x) and trace w4b (switch / tail / loops).

### r6-r7 (logs/r6.json, r7.json; MoE-only paired vs s_b, 6 reps; r7 two panels)
| variant | c0 d11 | c1 d11 | c2 d11 | c3 d11 | c0 d22 | c1 d22 | c2 d22 | c3 d22 |
|---|---|---|---|---|---|---|---|---|
| w7r (w2 + wreg 88/208 only) r6 | -0.60% | -0.29% | +0.72% | -0.04% | | | | |
| w3b (bigstg + pubcopy, NO wreg: W spills 124 B) r6 | -2.29% | -1.64% | +0.30% | +0.30% | | | | |
| w6x (w4b + xpf <= 224) r6 / r7 | -0.25 / -0.38 | -0.27 / -0.25 | +1.28 / +0.34 | +1.43 / +0.10 | -0.42 | +0.66 | +0.13 | -0.34 |
| z0 (w4b + scale0) r6 / r7 | -0.05 / -0.18 | +0.02 / +0.04 | +1.37 / +0.48 | +1.42 / +0.20 | +0.04 | +0.98 | +0.19 | -0.13 |
| z4 (z0 + xpf) r7 | -0.47 | +0.25 | +0.37 | -0.04 | +0.02 | +0.98 | +0.31 | -0.10 |
scale0 = the first k16 wgmma of an item overwrites the accumulators (scale-d 0) instead of zero-initialising 128 fp32
registers per width: kernel W 3240 -> 2592 instructions, loops 110. Bit-identical. xpf no longer pays once scale0 is in.
r6 base ran ~1% slow at 3012 (827.9 / 769.9 vs 819 / 758 in r7): single-run noise at 3012 is ~0.5-0.7%.

### Where the split stands (trace r7t, z0s vs sbt, draw 11; per SM)
| | c0 (753) sbt | c0 z0s | c2 (3012) sbt | c2 z0s |
|---|---|---|---|---|
| wide items (ideal) | 148.1 (133.0) | 142.9 (131.4) | 590.4 (545.7) | 580.7 (548.9) |
| wide epi / mainxs / first | 6.8 / 2.6 / 2.8 | 3.3 / 0.7 / 2.5 | 22.3 / 8.2 / 8.4 | 13.5 / 7.5 / 6.4 |
| narrow items | 86.1 | 85.0 | 174.3 | 171.9 |
| idle: switch / gaps / tail | 0.7 / 0.9 / 4.6 | 3.9 / 1.1 / 6.3 | 0.9 / 5.8 / 5.7 | 4.1 / 3.8 / 4.0 |
CAUTION: the trace instrumentation costs s_b ~2% and kernel W ~0.7% (r7t: base 817.7, sbt 833.9, z0s 823.2) -> trace
spans favour the split; use traces for structure only, the non-trace bench for verdicts.

### Lever 1: the W -> N switch (sw.py / sw2.py on cta-stamp builds; us per SM, means)
Full switch = last W item's main-loop end -> N's first item's first stage: 6.6-7.5 us (c0), 6.4-7.3 (c2):
| piece | z0s c0 / c2 | fix tried | after |
|---|---|---|---|
| last W epilogue | 1.2-1.3 / 0.6-1.2 | | |
| publisher drain of that item (64 KB rows, 1 warp) | 2.1 / 1.8 | lastcp: consumers store the last item | 1.7-3.0 epi+copy (moved, not removed) |
| teardown + N launch | 0.37 | - | |
| N barrier init + tile count + nW vote | 0.19 + 0.27 + 0.40 / 0.19 + 0.26 + 0.67 | pro: loads issued before the barrier init | 0.37 + 0.07 + 0.11 |
| N first hand-off | 0.6 / 0.5 | pro: N's first claim in the prologue | 0.3-0.4 |
| N first stage (ring empty, HBM-loaded) | 1.9 / 1.45 | npf: W's exiting CTA L2-prefetches N item #rank | no change |
- lastcp (z9 = bulk row stores + wait; z10 = wait.read; z11 = 256-thread 16-B stores): the last item's 64 KB write-out
  costs the consumers 0.9-1.7 us -- the SM's store path (~35-45 GB/s here) is the floor, whoever issues the stores.
  Totals (sw2): z0s 7.48 / 6.56, z9s 7.20 / 6.63, z10s 7.43 / 6.54, z11s 6.62 / 6.10 us. <= 0.9 us; not kept.
- npf: N's first item is NOT predictable from W's exit order (|rank - first N item| p50 48 at 753: early N CTAs run
  several items before late ones start); also N's first stage is HBM-bandwidth-bound (all-SM narrow streaming).
- pro (z6 = z0 + pro): switch 3.98 -> 3.60 (c0), 4.07 -> 3.56 (c2); W's first hand-off 0.54 -> 0.16 us.
| variant (r8 / r11 / r13 / r19: 2 panels x 4 cases, 8 reps each; mean of the 8 deltas vs s_b) | r8 | r11 | r13 | r19 |
|---|---|---|---|---|
| z0 | +0.15% | | | |
| z6 (z0 + pro) | +0.02% | +0.23% | +0.17% | +0.27% |
| z5 (z0 + npf) / z7 (z0 + npf + pro) | +0.19 / +0.03 | | | |
z6's 8 deltas (r19): c0 -0.10/+0.27, c1 -0.23/+0.61, c2 +0.69/+0.38, c3 +0.29/+0.11 (d11/d22). Parity-plus, not a win.

### The item boundary (producer stamps: pstamps builds, ps.py; us from the previous item's main-loop end, c2 g256)
producer: item resolved -1.57, ready (tile + done wait) -1.23, stage-0 slot free -0.70, stage 0 issued -0.49;
consumers: start +1.37 (epilogue ~1.2), got the item +1.42, reach the first-stage wait +1.66, data there +1.69.
=> the producer is ready 0.5 us before the slot frees (claim chain NOT critical), and the data is there when the
consumers arrive (wait 0.03 us). The 'first' piece of the item trace (0.25-0.4 us/item) is consumer instruction time
between the hand-off and the first wait -- and most of it is the trace's own overhead (t_mid stamps through a local
array; the smem tile table s1 removed the dependent tiles[] LDG but not this gap: 0.23-0.25 us remains).
Tried on that premise (all non-trace benches vs z6, 2 panels):
- pla (producer look-ahead: claim + hand-off + tile load D stages early): pa/pb/pd -1..-3% (N's 40-register producer
  spills 84 B; the claim atomic stalls the producer mid-item: g160 mainxs 3.3 -> 5.4 us); pc (W only) ~0; pe/pf/ph (+ next
  item's weights L2-prefetched ~2 us early) ~0, d22 c3 -2.7%. Dropped.
- xpf0 (X L2 prefetch at the item start), thr (stage s waits for stage s-1 at an item start): no change in arrival / loses.
- cla (consumers peek the next hand-off before the epilogue and load its tile entry): -0.2..-0.8% at 753.
- s1 (smem tile table): vs z6 +0.17..+0.43% at 753, -0.30..-0.53% at 3012 (N's 96/128 loops +5-7 instructions).
- q256 / q1k / q4k (publisher poll 64 -> 256 / 1024 / 4000 ns; r20): 753 -0.0..-0.5 / -0.5..-1.0 / -1.0..-1.8% vs z6:
  the publisher's reaction latency matters (staging free, gate done counters), its issue slots do not. Dropped.
- kernel N's producer stage (SASS, u_s1): 3 ELECT / R2UR / BRA.U.ANY waterfalls (L2 prefetch, X copy, W copy),
  ~60-80 executed instructions per narrow stage (4 k-blocks) on SMSP 0 beside two consumer warps (~2.6% of that SMSP's
  issue): <= ~1 us/SM at 753 -- R's lean producer (rw NOTES 6.3) covers it; not done here.

### Nearby shapes (r21n: mb.py --shapes, MoE-only, z6 vs s_b, 6 reps)
| T (batch x q_len) | 700 (1x700) | 760 (1x760) | 1500 (2x750) | 2048 (4x512) | 3040 (4x760) | 4096 (4x1024) |
|---|---|---|---|---|---|---|
| d11 | -0.08% | +1.41% | +0.54% | -0.42% | +0.48% | +0.52% |
| d22 | -0.33% | -1.30% | +2.27% | +0.79% | +0.11% | +0.86% |
No systematic regression (single 6-rep runs: +-1% noise at 760).

### Candidate ww_z6 (deliverable; build.py 'ww_z6', deliver.py) = s_b with the MoE string replaced
Kernel W / N split for 512 <= T < 8192 (T >= 8192 and T < 512: s_b's kernels, unchanged): vote nW, W tile-major,
ONE wide consumer with a runtime-width shared epilogue, scale0, bigstg + pubcopy, W register split 88 / 208, pro
(prologue loads + first claims across the barrier init). Lean source (no compiled-out options). Bit-identical to s_b.
research/claude/ww/ww_z6.py 924,788 B sha256 9f964e65332a8236a7aa961fed84101e44d881de44c87644bce9a6f114b7f7ba
research/claude/ww/ww_z6h.py (split only for T >= 2048) 924,789 B sha256 aaf7ec2900b93736befa1ed09a61e36fc172e8c73c9869b888c58e21449a6978
- gate (logs/gate_ww_z6.json): CHECK_RC=0 PASS=42.
- MoE-only (r19 / r20 / earlier z6 runs, 2 panels): z6 mean +0.17..+0.41% vs s_b; 753 ~+0.1 (c0 d11 -0.1..-0.3, c1 d22
  +0.6..+0.8), 3012 +0.3..+0.9.
- Replica (layer, cases 0-3, 8 reps, judge-faithful pairs vs SGLang):
| | d11 | d22 | d33 | mean | d44 | d55 | d66 | mean |
|---|---|---|---|---|---|---|---|---|
| ww_z6 / s_b | 1.0031 | 0.9972 | 1.0010 | 1.0004 | 1.0011 | 1.0004 | 1.0019 | 1.0012 |
| ww_z6h / s_b | 0.9997 | 0.9977 | 0.9992 | 0.9989 | 1.0012 | 0.9991 | 1.0006 | 1.0003 |
=> VERDICT: parity (+0.04 / +0.12%, inside the ~0.1-0.3% per-draw noise). The split no longer costs anything (w2 was
-0.3..-1.2% MoE-only), but on its own it is not a submission. Its value is as the base R's riders stack on.

### Stacking with R (r22; R's s16L rebuilt read-only from rw/var.py: build.py 'rs16L' == rw/rw_s16L.py's MoE string)
MoE-only vs s_b, T = 753, 8 reps:
| draw / case | 11/0 | 11/1 | 22/0 | 22/1 | 33/0 | 33/1 | mean |
|---|---|---|---|---|---|---|---|
| rs16L (R's s16L) | +2.42% | +1.91% | +0.07% | +0.40% | +1.98% | -1.09% | +0.95% |
| rs16Lz (+ my scale0 via R's sc0 hook; kernel W 4464 -> 3744 instr) | +2.67% | +1.98% | -0.36% | +0.42% | +1.48% | -0.55% | +0.94% |
=> scale0 does not add on R's build (its W time is mostly combo items, whose loop keeps its own init). My other W
patches do not fit R's kernel W: wreg 88/208 needs consumers <= 208 registers (R's combo consumer spills at 232);
bigstg's 37888-B slots cannot hold a combo stage. R already carries my shared epilogue, tile-major W and (its port of)
pro / pubcopy. My T >= 2048 split (ww_z6h) is additive in principle (R splits only T < 2048) but measures +0.03% on the
layer at cases 2 / 3 (replica, 6 draws): not worth a merge into R's file (size budget, risk).

## 4. Summary of this instance (levers in the orchestrator's order)
1. W -> N switch: measured end to end (6.5-7.5 us per SM from the last W main loop to N's first data). pro removes
   0.4-0.5 us (prologue chain) and W's first hand-off 0.38 us; lastcp moves the 64-KB drain onto the consumers without
   shortening it (SM store path ~35-45 GB/s); npf cannot target N's first item (claim order unpredictable) and N's
   first stage is HBM-bandwidth-bound. The floor is ~drain 1.2 + teardown 0.37 + prologue 0.5 + first stage 1.5 us.
2. Codes/scales LDS prefetch one stage ahead (xpf): neutral once scale0 is in (z4 vs z0, w6x vs w4b, 2 panels).
   scale0 itself (first k16 with scale-d 0) is the biggest single W gain: +0.3..+1.4% MoE-only over w4b.
3. Tile plan TP160 (w6p vs w4b, r5): -0.66 / +0.68 / -0.10 / -0.38% -> no gain (and not bit-identical). Dropped.
4. ho hand-off: not pursued (mw: +0.16% MoE-only, layer-neutral; my producer-stamp data shows the producer is ready
   0.5 us before the stage-0 slot frees, so hand-off latency is not on the critical path in kernel W).
Item boundary: the producer is never late; "first-stage" waits in item traces are mostly trace overhead.
Deliverable: ww_z6.py (parity on the layer: +0.04 / +0.12% on the two panels; gate PASS=42). GPU work done.
