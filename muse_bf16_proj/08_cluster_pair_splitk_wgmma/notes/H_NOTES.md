# Agent H — sk11 ring follow-ups (T's (a)/(b)/(c)), m = 8/16/32, base claude/t3/bo7b.py

## RESULT (read first)
- SHIPPED: claude/t3/H/skh.py (prepare_skh / run_skh) + sub_skh.py (bo7b + skh) + sub_skh_only.py (skh + torch):
  **unpooler m = 32 with R = 64 rows per CTA** (128 CTAs = 1 per SM, so x is read once per SM instead of twice; one
  escape warp with a stage-sorted per-CTA list = ESC1, needed for 8 tiles per CTA; baked W pointer). Same-run bench vs
  bo7b (4 runs, boxes 1/1/1/0): 13.18 vs 13.53, 13.07 vs 13.37, 13.11 vs 13.50, 13.31 vs 13.49 local (-0.18..-0.39); decode-32 load
  ~+0.6% judge-projected, i.e. ~+0.1% on the 6-load geomean. run_skh returns None for every other shape (bo7b keeps them).
- Correctness: check.py sub_skh.py --graph (all layers) 259 cases ALL OK; sub_skh_only --layers unpooler --graph --seeds 3
  ALL OK; exact_all --layers unpooler --nesc 200 and --nesc 3000 EXACT ALL OK; prepare_skh 0.56 s, never raises.
- T's (a) last-arriver refill: +4 us (dead). (b) half final stages: +0.15..+0.6 us on every shape (dead). (c) split-K
  cluster pairs / quads for qkv32 (and pairs for pooler32): +0.5..+1.4 (dead with an st.async reduce-scatter epilogue).
  ESC1 alone, PM placement spread, x rotation, L2 hints / prefetch / promotion, W TMA interleave, x box split, parallel
  initial fill, x via cp.async loader warps, R = 64 for o / unpooler16, config sweeps for o16 / o32: no robust win.
- Measured facts worth keeping (iterations 3, 5, 6): x costs 0.5-1.2 us of the data ring through per-SM ingress
  sharing with W (not latency: not waiting for x costs the same); same-binary kernels differ by up to 0.3-0.5 us between
  instances/processes (A/Bs below 0.3 us need >= 3 processes); nsys / ncu kernel durations disagree with the graph
  replay bench for the 1-CTA-per-SM 225 KB kernel (nsys 15.42 vs 14.27 while bench is -0.30): an okc check should
  confirm the unpooler32 gain before it is relied on.

Files (claude/t3/H/): skh.py (copy of F's sk11 + H options below; prepare_skh / run_skh), th.py (same-run timing harness:
`h:` = skh variant, `f:` = F's sk11 (no cfg = F's shipped cfg = bo7b's kernel for qkv m8-32 / pooler m32), `x:` = X's sk3
(no cfg = bo7b's kernel for o m16 / unpooler m16-32); judge-style graph replay after a 512 MB flush, empty graph
subtracted, interleaved order; CHECK=1 = normal + outliers + one-hot exact probe; ABL=1 skips the self-test for
ablations), sass.py (local NVRTC sm_90a SASS + regs; SKH_NOTMAP=1), one.py + ncuh.sh (ncu full set of one variant).
Times: local us (judge ~= local + 1.0). Only same-run deltas are meaningful; run-to-run noise of one shape's median is
+-0.15-0.3 us even at 40-60 reps, so every claim below is from >= 2 runs unless marked "1 run".

skh cfg tuple: (R, NT, NWR, WSK, UPW, NACC, cap KB, flags[, SK, PM, RG, XS, XP, HF, clone, IF, XL]); flags = F's (4 bake, 1<<20 F12, 2 / 16 /
64 / 128 / 256 as in sk11) plus H kernel flags: 1<<17 ESC1 (one escape warp, per-CTA stage-sorted escape list, smem
corrections; lifts F's R <= 32 limit), 1<<18 LR (last arriver issues the refill; needs ESC1), 1<<19 HTR (per-warp
per-stage globaltimer + clock64 trace of CTA 0 and ncta/2), 1<<21 relaxed LR atomics, 1<<22 no HMMA, 1<<23 no x LDSM,
1<<24 W L2 evict_first, 1<<25 x L2 evict_last, bits 26-28 W L2-prefetch distance, 1<<29 W as TMA tiles of 1 KB pieces
interleaved across CTAs, 1<<30 no x traffic (ablation). SK = split-K cluster size (ESC1 only; st.async reduce-scatter),
PM = placement spread (PM working CTAs per 16 blockIdx), RG = x-stage rotation groups (W re-laid out time-major),
XS = x box split into XS TMA loads, XP = 1/2 x L2 prefetch at start, 8 = no W (ablation), 9 = x loads on a never-waited
barrier (ablation), HF = last HF stages as 2 HF half stages, clone = separate W blob (noise test), IF = parallel initial
fill, XL = x via XL cp.async loader warps.

## Results so far (all measured, same-run vs F's sk11 / X's sk3 = bo7b's kernels)

| idea | shapes | result |
|---|---|---|
| (a) LR: last releasing warp issues the refill (no producer) | pooler32 | **+4.0 us** (17.5 vs 13.5): the issuing consumer becomes the last arriver of every later stage (issue ~0.15-0.2 us per stage on its own critical path, positive feedback); trace: w0 release-land 2.2 us vs 0.32 for w1-w3 |
| ESC1 (one escape warp instead of R/8) | pooler32 / o32 | -0.32 / -0.26 in one run, +0.48 / +0.06 in others: tie |
| ESC1 | qkv8 / qkv16 / qkv32 | +1.0 / +0.1..0.3 / +0.1..0.3 (one warp serialises 3 tiles' escapes; slot held per escape) |
| ESC1 batched (release before the correction RMW) | all | +0.5..+4 us (WARPSYNC.COLLECTIVE loop; reverted) |
| (c) split-K cluster pairs, R=48, st.async reduce-scatter | qkv32 | +1.3 (10.0-10.2 vs 8.7-9.0); stream not faster (loop end 8.42 vs 8.45 med), epilogue 1.05 vs 0.41 us |
| (c) split-K pairs R=64 | pooler32 | +0.5 (13.78 / 13.87 vs 13.31) |
| W L2 evict_first + x evict_last hints | qkv32 / pooler32 | tie |
| W L2 prefetch 2-4 stages beyond the ring | qkv32 | +1.1..+1.4 (more DRAM concurrency = worse) |
| W as TMA tiles of 1 KB pieces interleaved across CTAs | qkv32 / pooler32 | tie / +0.2 |
| PM 12-of-16 placement spread (all 8 GPCs) | qkv32 / qkv16 | -0.1 / -0.16 (1 run), +0.1 (other run): tie |
| x-stage rotation (RG 2/4/8, W re-laid out) | qkv32 / pooler32 | +0..+0.4 / +1.2..+2.0 |
| x box as 2 / 4 TMA loads | qkv32 / pooler32 | tie / +0.25 |
| x tensor-map L2 promotion NONE / 128B (vs 256B) | qkv32 | -0.09 / -0.29 (1 run each) |
| x L2 prefetch at kernel start | qkv32/16, pooler32, unp32, o32 | tie (qkv16 +0.3) |
| **R = 64 (128 CTAs, 1 per SM) for unpooler m32, cfg (64,4,2,2,2,1,160,ESC1+bake)** | unpooler32 | **-0.16 / -0.21 / -0.3 vs X sk3** (3 runs) |
| R = 64 | unpooler16, o16, o32 | +0.5..+0.7 / +0.2 / +0.2 |
| (b) last stage split into 2 half stages (half W, half x box, separate code path) | o32 / o16 / unp32 / pooler32 | +0.27 / +0.28 / +0.15 / +0.6 (first version with a shared code path: +0.3..+2.0) |
| parallel initial ring fill (stage s issued by lane 0 of warp s - 1) | o32 / o16 / unp32 / qkv32 | +0.2 / +0.6 / +0.5 / +0.6 (in-order serial issue is better) |
| config sweeps o16 / o32 (UPW 1, WSK 4, NACC 2, caps, ESC1) | o16 / o32 | all within +-0.15 of bo7b; 2-CTA/SM occupancy cliffs at cap >= 112 KB or > 113 regs (+1.5..+2 us) |

## SHIPPED: skh.py unpooler m = 32 (R = 64, 128 CTAs, cfg (64,4,2,2,2,1,160, F12 | ESC1 | bake))
bench.py same run vs bo7b (judge-style, paired with SGLang):
| shape | SGLang | bo7b | skh (bo7b + H) | delta |
|---|---|---|---|---|
| unpooler m32 (run 1, 80 reps, box 1) | 17.28 | 13.53 x1.277 | **13.18 x1.311** | -0.35 |
| unpooler m32 (run 2, 80 reps, box 1) | 17.23 | 13.37 x1.289 | **13.07 x1.319** | -0.30 |
| unpooler m32 (run 3, 60 reps, box 1) | 17.17 | 13.50 x1.272 | **13.11 x1.309** | -0.39 |
| unpooler m16 (run 3; same kernel as bo7b) | 17.07 | 12.56 x1.359 | 12.60 x1.355 | noise |
| unpooler m32 (run 4, 80 reps, box 0) | 17.58 | 13.49 x1.304 | **13.31 x1.321** | -0.18 |
| unpooler m16 / m8 (run 4; same kernels as bo7b) | 17.34 / 6.18 | 12.67 / 2.82 | 12.73 / 2.85 | noise |
| unpooler m16 (same kernel as bo7b: X sk3) | 17.06 | 12.39 x1.376 | 12.27 x1.390 | noise |
| unpooler m8 (same kernel: G slice) | 6.00 | 2.69 x2.232 | 2.70 x2.223 | noise |
th.py same-process A/B vs X's sk3 (bo7b's unpooler32 kernel): -0.16, -0.21, -0.30, -0.27, -0.30 (5 processes).
Correctness: check.py claude/t3/H/sub_skh.py --graph (all layers, bo7b + skh) -> "checked 259 cases x 2 seeds + graph:
ALL OK"; check.py sub_skh_only.py --layers unpooler --graph -> 53 cases ALL OK; exact_all sub_skh_only --layers unpooler
--nesc 200 and --nesc 3000 (all M) -> EXACT ALL OK. prepare_skh ~1-2 s (one NVRTC kernel), never raises.
Integration: `out = run_skh(x, cfg, state_H)` before bo7b's kernel_fn (returns None except unpooler m == 32).

### iteration-4 analysis block (unpooler32 R=64 vs bo7b's X sk3 R=32)
1. nsys in-graph (12 reps, flushed): skh_unpooler_32 15.42 us vs sk3_unpooler_32 14.27 (!). The profilers disagree with
   the graph-replay bench (-0.30/-0.35) and th.py (-0.16..-0.30 in 5 processes); my reading: the 225 KB / 1-CTA-per-SM
   kernel's start is counted earlier (or it waits for the flush kernel's tail SMs) in the kernel-duration view, while
   the judge-style event timing of the whole replay is shorter. The bench (calibrated to the judge) is the reference;
   flagging this for the coordinator (an okc check would settle it).
2. ncu (full set, flushed, clock-control none; durations ncu-inflated and at different clocks):
| kernel | dur | clk | DRAM rd | L2 % | L1/smem % | bank rd % | issue % | hmma % | occ % | elig | L2 rd sectors (SM side) | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| skh R=64 | 16.96 | 1.69 | 50.66 MB | 64.4 | 39.1 | 12.7 | 23.0 | 9.1 | 9.1 | 0.24 | 1.94 M | long_sb 38, wait 19, selected 19 |
| sk11-style R=32 (F12) | 15.90 | 1.85 | 50.74 | 71.2 | 53.4 | 19.1 | 30.9 | 8.9 | 26.1 | 0.42 | 2.30 M | long_sb 54, wait 13 |
| X sk3 R=32 (bo7b) | 16.42 | n/a | 50.67 | 68.4 | 56.4 | 20.4 | 32.6 | 8.8 | 23.3 | 0.44 | | long_sb 45, wait 15 |
   R=64 moves 16% fewer L2->SM sectors (x read once per SM instead of twice) and 29% fewer instructions (2.60 M vs 3.60 M:
   one producer/escape set per SM, x LDSM shared by 2x more rows).
3. SASS (NVRTC 12.1 local / 13 sandbox): 115-119 regs (NT 4: b[8][8] + acc 2x4x4), LOCAL 0, no LDG in the loop; per stage
   per consumer warp 64 HMMA, 16 LDSM.x4, 16 LDS (8 x .128 + 8 x .64), 56 PRMT/LOP3/SHF decode; 4 consumer warps + producer
   + 1 escape warp (192 threads).
4. Binding unit: the W+x delivery (data ring 12.29 us no-compute vs 13.0-13.2 full; DRAM 50.7 MB over ~11 us of streaming
   = 4.6 TB/s, 96% of 4.8 in the bulk). Remaining gap to the 12-bit floor (~12.0): startup ~1.3 us + consumer tail ~0.8.

## iteration 5 — noise characterisation (important for everyone's A/Bs)
- Two byte-identical kernels (same SASS: only an unused template flag differs) in one process differ consistently by
  0.3-0.5 us (qkv32 9.07 vs 8.69, 8.98 vs 8.51 in two processes; another process: 9.18 vs 8.90 with order swapped), and
  F's sk11 vs my identical-config skh flips sign between processes (+0.45, then -0.32). Six separate W-blob clones of
  one kernel in one process: spread only 0.15 (qkv32) / 0.16 (pooler32). => per-process, per-kernel-instance offsets of
  +-0.3 us that do not average out with reps; any A/B below ~0.3 us needs >= 3 processes (or bench.py runs).
- Most "wins" < 0.3 us I saw (ESC1 pooler32 / o32, PM qkv, promotion 128B, bigger pooler32 stages) reversed in a
  repeat process.


## iteration 1 — LR, ESC1, per-warp trace (pooler32)
Trace (HTR, trace adds ~2.5 us so only relative numbers): ESC1 non-LR: consumer w0 (shares SMSP0 with the producer)
and w1 (shares SMSP1 with the escape warp) release each stage 0.4-0.6 us after w2/w3; consumers are busy every cycle
(stage st+1 has landed before they release st). LR: w1-w3 process a stage in 0.32 us after it lands and then wait,
w0 (issuer) needs 0.55 us/stage -> period 0.55 us. With the escape warp sleep-waiting (FL 256) all four consumers run
at the same 832 cyc/stage. Clock 1.69-1.72 GHz with HMMA, 1.88-1.90 GHz with the HMMA ablated.
Ablations (40 reps, same run): ESC1 13.55 | escape sleep 13.41 | no compute 13.09 | no HMMA 13.39 | no x LDSM 13.36 |
no escape warps (F12, FL128) 13.25 | NACC 2 13.87 | F sk11 13.70.
=> pooler32 is within 0.3-0.45 us of its own data ring (no-compute 13.09); the consumer is not the main limit.
ncu (full set, flushed, clock-control none; durations ncu-inflated):
| kernel | dur | DRAM rd | L2 % | L1/smem % | issue % | hmma % | occ % | elig | clk | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|
| sk11 pooler32 (F12) | 16.26 | 50.94 MB | 69.2 | 51.2 | 29.5 | 8.9 | 13.6 | 0.33 | 1.80 | long_sb 41, wait 22, selected 15 |
| skh ESC1 pooler32 | 17.38 | 50.90 | 64.9 | 53.6 | 29.0 | 8.9 | 9.2 | 0.31 | 1.59 | wait 32, long_sb 28, selected 21 |
SASS (NVRTC 12.1 local, sm_90a): pooler32 F12 80 regs, ESC1 76, LR 76; LOCAL 0; LR release = MEMBAR.ALL.CTA + ATOMS.ADD
(acq_rel) or ATOMS.ADD alone (relaxed, flag 1<<21; same timing); no LDG in the consumer loop.
Binding unit: the W+x delivery into the SM (no-compute ring 13.09 vs full 13.4-13.6); DRAM 50.9 MB in ~11.5 us of
streaming = 4.4 TB/s (92% of 4.8); tensor 9%, issue 29%.

## iteration 2 — split-K cluster pairs (T's (c)) for qkv32
Implementation: CTA = row block rb x rank kr (cluster dims SK), W stream [local stage][rb][kr], x TMA from kr's K range,
per-CTA escape list with local stages, epilogue: own partial (WSK sum + corrections) -> partner's half via
st.async.v4 (mbarrier complete_tx on the partner's receive barrier) -> wait own receive barrier -> sum -> bf16.
Correct first try (normal, outliers, one-hot exact).
Trace (FL 2, 96 CTAs): split first stage 1.60 med | loop end 8.42 med / 9.22 max | CTA end 9.47 / 10.27;
non-split: 1.41 | 8.45 / 9.15 | 8.86 / 9.57. => halving x per SM did not shorten the stream; the exchange + pair
coupling adds ~0.65 us. Same-run ablations (40 reps): non-split no-compute 8.67, + no x 7.49 (x costs 1.18);
split no-compute 9.25, + no x 7.98 (x costs 1.27: halving x bytes per SM does not reduce it; split adds 0.5 even
without x).

## iteration 3 — what x costs in the ring (qkv32 / pooler32 / unpooler32, no-compute rings, same run)
| variant | qkv32 | pooler32 | unpooler32 R=32 | unpooler32 R=64 |
|---|---|---|---|---|
| full kernel | 8.6-9.0 | 13.1-13.5 | 13.3-13.6 | 12.96-13.15 |
| no compute (data ring) | 8.56-8.67 | 12.66-12.80 | 12.56 | 12.29 |
| no compute, x issued but not waited for (XP 9) | 8.59 | 12.74 | | |
| no compute, no x | 7.49-7.60 | 12.19-12.27 | 11.94 | 12.16 |
| x from stage 0 only (L2-hot after the first stage) | +0.74 (full) | | | |
=> x costs 1.0-1.2 us (qkv32) / 0.5 (pooler32) / 0.6 (unp32, 2 CTAs per SM) / 0.13 (unp32 R=64, 1 CTA per SM) in the
data ring, and NOT through its latency (not waiting for it costs the same): the x traffic slows the W stream. It does
not scale with x bytes per SM (split-K: same cost), is worse when all CTAs read the same lines longer (stage-0-only x
+0.74) and worse when CTAs read different stages (rotation +0.4..+2.0), and is neither cold-x (L2 prefetch: tie) nor
L2-promotion (tie) nor box granularity (tie). ncu (qkv32): L2rd sectors from the SM side +0.50 M with x (16 MB for 50
MB of x requests: merged at L2), L2 % 62.6 (with x, no compute) vs 64.4 (no x).
ncu table (flushed, no compute ablations of qkv32):
| kernel | dur | DRAM rd | L2 % | L2 hit % | L1/smem % | issue % | clk | top stalls |
|---|---|---|---|---|---|---|---|---|
| qkv32 full (F12 bake) | 10.98 | 28.89 MB | 59.0 | 25.0 | 47.6 | 26.6 | 1.75 | long_sb 41, wait 18, selected 12 |
| qkv32 no compute | 10.34 | 28.89 | 62.6 | 27.1 | 41.5 | 6.5 | 2.03 | long_sb 70 |
| qkv32 no compute no x | 9.41 | 28.37 | 64.4 | 2.5 | 16.0 | 6.0 | 1.98 | long_sb 70 |
Binding unit (qkv32): W delivery at 28.3 MB / ~7 us = 4.0 TB/s with x vs ~4.6 TB/s without; tensor 10%, issue 27%.

## iteration 4 — R = 64 for unpooler m32 (ESC1 lifts the 4-tile limit)
unpooler32 has 256 CTAs (2 per SM, each reading all of x) in bo7b; R=64 = 128 CTAs (1 per SM, x per SM halved).
Data ring 12.29 vs 12.56 (no compute, same run), x cost 0.13 vs 0.62. Full kernel, same run vs X's sk3 (bo7b):
(64,4,2,2,2,1,160) 13.49 vs 13.65 | 13.15 vs 13.36 (F 13.42) | 12.96 (2nd variant run) ; 8-warp layouts
(64,2,4,2,2) 13.66-14.03, (64,4,2,4,1) 13.57, (64,8,1,4,1) 13.20, NACC 2 13.23, cap 200 13.42.
R=64 loses on unpooler16 (13.04-13.26 vs 12.38-12.51), o32 (4.86 vs 4.70), o16 (4.51 vs 4.34).

## iteration 6 — what limits the x stream (no-compute ablations, same run, box 0)
| ring (no compute) | qkv32 | qkv16 | pooler32 | unpooler32 R=32 | unpooler32 R=64 |
|---|---|---|---|---|---|
| W + x | 8.50 | | 12.86 | 12.56 (other run) | 12.29 (other run) |
| W only | 7.62 | | 12.19-12.27 | 11.94 | 12.16 |
| x only (W copies skipped, XP 8) | 4.70 | 3.65 | 5.86 | 5.09 | 5.89 |
x alone (all CTAs re-reading 0.5 MB of L2-resident x) needs ~3.5 us of transfer for qkv32 (50 MB L2->SM in ~3.5 us,
~150 GB/s per SM); W alone 6.4 us at 4.4-4.6 TB/s; together 7.3 + 1.2 startup instead of max(6.4, 3.5) + 1.2.
The x cost grows with the per-SM x rate: unp32 R=64 24 GB/s/SM -> 0.13 us, pooler32 / unp32 R=32 48 GB/s/SM ->
0.5-0.6, qkv32 75 GB/s/SM -> 1.0-1.2 (per-SM ingress W + x: 59 / 83 / 117 GB/s). Consistent with a per-SM ingress
(TMA + L2->SM) limit of ~100-110 GB/s/SM that W and x share; the fix is fewer x bytes per SM per unit time.
- x through the LSU path instead (XL = 2 / 4 cp.async loader warps, 16 B per lane, 128B swizzle reproduced, arrive via
  cp.async.mbarrier.arrive.noinc; correct): qkv32 12.96 / 10.03 vs 8.88, pooler32 (XL 4) 13.39 vs 13.09 -> the LSU path
  is slower (issue-bound loaders + longer latency); dead.
- split-K S = 4 (R = 96, cluster of 4, x per SM 131 KB, ingress 426 KB/SM): qkv32 9.28-9.38 vs 8.59 (cluster + exchange
  overhead > ingress relief); S = 2 R = 48: 10.0-10.2. Split-K does not pay with this exchange (st.async reduce-scatter).
- parallel initial fill (IF): worse on every shape (in-order arrival of the first stages matters).

## iteration 7 — (b) half final stages: dead (see table). Even with a separate code path for the half stage and a
half-size x box (second tensor map), every shape got slower (o32 +0.27, o16 +0.28, unp32 +0.15, pooler32 +0.6): each
extra stage carries the full per-stage cost (barrier round trip + TMA issue); the backlog after the last landing is not
compute-bound per stage. First version (half-stage branch inside the unrolled uu loop) was +0.3..+2.0: the branch broke
the compiler's interleaving of the two units of a stage (SASS 1397 -> 1589 lines for the same 32 HMMA).
