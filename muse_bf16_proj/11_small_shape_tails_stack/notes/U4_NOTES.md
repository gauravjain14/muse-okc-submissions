# Agent U4 — m = 128 pooler W[4096, 8192] with 64-row x 128-token tiles (12-bit RS loop) + U3's remaining costs

Base build: claude/t3/bo21L.py (pooler m128 = Q2's BF16 SS kg_pooler_64_64: ~17.3-17.6 local, 19.82 nsys in-graph;
unpooler m128 = U3's u128: ~15.7-16.1 local, 16.78 nsys). Template: U3/u128.py (+ U3/u3k.py, U3/NOTES.md).
All times local us (graph replay after a 512 MB flush, empty graph subtracted; judge ~= local + 1.0) unless noted.
Box 0 only (shared by 4 agents). Same-run A/Bs only.

## STATUS (final, 2026-11-03): no shape wins -> nothing to integrate (pooler stays Q2, unpooler stays U3's u128)
- GOAL 1 (pooler W[4096,8192] m=128, 64-row x 128-token tiles, split-K 2): INFEASIBLE, measured by data-movement
  probes before building: the geometry doubles x per CTA (1 MB) and its x+W stream alone takes 15.5-16.0 us vs 13.1
  for U3's geometry (projected kernel ~18 vs Q2's 17.3); the fix (x TMA multicast between the two row tiles) is
  impossible in one wave (max 30 active clusters of 4 on this H200 = 120 SMs) and multicast is slower than plain
  loads anyway (cluster-2 probes 16.6-17.7 vs 14.3-15.6). Iterations 0 / 0b.
- GOAL 2 (U3's start + tail): the start is not a lever (x0 really lands at 1.25 us, before W0; U3's "2.1" is the
  consumer's W1 wait; the loop is HBM-bound and ends within 0.3 us of a data-only probe; prefetch / W-first / SPL /
  QS 6 all within noise). The tail: built u4.py (push the partner half uncorrected; receiver-side exact correction
  of the sender's escapes): correct + exact, but 0.29 us SLOWER (x-clone same run 16.26 vs 15.97; nsys 17.10 vs 16.77),
  because an earlier DSMEM push shares the SM's ingress with the last TMA loads (32 KB takes 1.61 instead of 1.10 us)
  and u128's two post-loop chains are already balanced. Iterations 1-2.
- Files: u4.py (39 KB, U4_ON = False -> run_u4 returns None; set True for research), sub_u4t.py (u4 + torch),
  sub_u4.py (bo22L + u4), u4k.py (U3's dev kernel + U_PFX / U_WF / U_NOCORR1 + start trace points), u4t.py (U3's
  same-run harness + "u4:" spec), u4probe.py (multicast / cluster / geometry data-movement probe), u4trace.py +
  u4_tr.py / u128_tr.py (trace builds), u4_diag_*.py (diagnostic variants, wrong results by design), gensrc*.py,
  nvc.py (Q5's NVRTC-only compile), u128_ref.py (copy of U3's u128).

## iteration 0 — is the 64-row x 128-token / split-K 2 geometry feasible? (paper + data-movement probes)
Per CTA (64 rows x 4096 k x 128 tokens): decode work = U3's (262 K weights), RS layout: the two consumer warpgroups
split K inside the CTA (stage = 128 k; warpgroup g takes the 64-k half g: per warpgroup per stage EXACTLY U3's work
per chunk: 64 rows x 64 k decode + 4 x m64n128k16), W blob [rb][stage][g][6144] = U3's format with g = k half.
What changes: x bytes per CTA double: 128 tok x 4096 k x 2 B = 1 MB (U3: 512 KB), W 384 KB (same) -> SM ingress
1.39 MB vs 0.9 MB, L2 -> SM total 182 MB vs 116 MB; x ring slots are 32 KB per stage (smem: W ring + x ring + rbuf).

Data-movement probes (no MMA; M/mprobe.py, box 0, 2 rounds x 20 reps, 128 CTAs, local us):
| probe | per SM | time | first data / end median |
|---|---|---|---|
| u2: U3 geometry (x 512 KB + W 384 KB, 6 x 16 KB x slots, 6 x 12 KB W slots) | 0.9 MB | **13.2** | 1.5-1.8 / 13.06 |
| u2x: U3 geometry x only | 0.5 MB | 4.6 | |
| p1: pooler 64-row geometry (8 x 16 KB x, 8 x 6 KB W) | 1.39 MB | 16.8 | 1.0-1.3 / 15.9-16.1 |
| p1x: pooler 64-row x only (1 MB / SM, 2 x-sets) | 1.0 MB | 6.9 | |
| p1w12: 12 x 6 KB W slots | | 17.2 | |
| p1w2: 6 x 12 KB W copies (= U3's W ring) | | **16.0** | 1.8 / 16.6 |
| p1w0: W only, 12 x 6 KB copies | 0.4 MB | 14.0 | (12 KB copies: ~12.7, M) |
| u2c / p1c (consumer spin per chunk = 0.378 us per 64x128 W block) | | 17.1 / 21.1 | |
=> x alone streams fine (1 MB/SM in 6.9 us) but x + W together cost +2.8 us over U3's geometry (16.0 vs 13.2): the
L2 -> SM side (182 MB in ~14.5 us = 12.5 TB/s) binds, not the per-SM ingress (x-only reaches ~150 GB/s/SM).
W copies must stay 12 KB (6 KB copies: W-only 14.0 vs 12.7). A straight 64-row kernel would end ~18.5 > Q2's 17.3.
Lever to test: x TMA multicast between the two row tiles of the same K half (cluster of 4 = 2 row tiles x 2 K
halves): L2 reads of x halve (back to U3's 116 MB), SM ingress unchanged.

### iteration 0b — x TMA multicast (U4/u4probe.py, same probe style, box 0, 2 rounds x 20 reps)
| probe (64-row tiles, 8 x 16 KB x slots, 6 x 12 KB W copies) | time |
|---|---|
| pooler S2, no cluster (= p1w2) | 15.5-15.65 |
| pooler, cluster 4 = (2 row tiles) x (2 K halves), no multicast | 27.0 (**max active clusters of 4 = 30**: 2 waves) |
| pooler, cluster 4 + x multicast | 87.7 |
| pooler, cluster 2 (two row tiles, same K half) + x multicast, release.cluster remote arrives | 45.3 |
| same, plain remote arrive / no remote arrive at all (timing only) | 17.7 / - |
| unpooler-like 64-row, NO split (every CTA reads all of x[128, 4096]) | 14.34 |
| same + cluster-2 x multicast, release.cluster / plain / no remote arrive | 45.9 / 16.97 / 16.61 |
| reference u2 (U3's unpooler geometry) | 13.13 |
=> (1) clusters of 4 cannot hold 128 CTAs on this H200 (30 x 4 = 120 SMs: GPC sizes), so "row-tile pair x split-K pair"
clusters are impossible in one wave; (2) TMA multicast of x is SLOWER than plain loads here (16.6-17.7 vs 14.3-15.6,
even with no cross-CTA release at all), and `mbarrier.arrive.release.cluster` per stage is pathological (+30 us);
(3) without multicast the 64-row geometry streams 1.2-2.4 us slower than U3's.
**GOAL 1 verdict: the 64-row x 128-token / split-K 2 pooler cannot beat Q2's BF16 kernel** (data movement alone 15.5-16.0
vs U3-geometry 13.1; projected ~18 vs 17.3). Other 12-bit pooler geometries are already measured losers (U3 it. 7:
128-row S4 17.89 vs 17.34; L: n64 decode-bound; U/Q6: decode-to-smem SS smem-bound). The pooler stays with Q2.
Side result: a split-free 64-row unpooler (no exchange tail at all) streams in 14.34 vs 13.13: projected ~15.7-15.9
vs U3's ~16.0 (gain <= 0.2-0.3 us for a full rewrite): not pursued.

## iteration 1 — U3's unpooler kernel: where the start and the tail go (u4k.py = U3's u3k.py + trace points)
Trace build (MZ_TRACE, box 0; medians over CTAs, us from the first CTA start; shipped cfg 2,1,4,3,1,8,0,1,1):
x0 lands 1.25 (new trace point in gatherer warp 2: BEFORE W0, 1.41), W1 lands 2.02, consumer passes the x0 wait at 2.18
(it waits for W1 first: U3's "x lands at 2.1" is the consumer's W1 wait, not x). Loop end (phase 1) 13.38 (max 13.73);
correction done 13.82 (+0.44; per-warp clock64 704 clk median, 1902 max); push issued 14.24 (+0.42: xe wait 125 +
8 STS 195 + fence 37 + bulk issue 252 clk); solo chunks issued 14.91; solo MMAs done 15.07; partner data 15.36
(+1.12 after the push: 32 KB at ~29 GB/s); staged 15.55; end 15.84 (max 16.32).
Start variants (same run, trace builds): x L2 prefetch of one token row per CTA (U_PFX) -0.10, W-first per stage (U_WF)
-0.16, both -0.13 (x0 then lands later: 1.44-1.60). Non-trace re-run (same run, 3 x 25): base 16.24 | WF 16.27 |
WF + SPL (two-part receive) 16.24 | WF + QS 6 16.34 | WF + QS 6 + SPL 16.56 | lean u128 16.24 => all within noise:
the loop is data-bound (probe u2 13.1 vs phase-1 end 13.4) so an earlier consumer start only waits later.
- (2026-11-03) coordinator: base is now claude/t3/bo22L.py (unpooler m128 = U3's u128, pooler m128 = Q2's g128q: unchanged vs bo21L).
Correction-removal bound (same run, non-trace, dev build u4k): base 15.98 | U_NOCORR1 (pre-push phase-1 correction
compiled out; wrong for escapes) **15.60 (-0.38)** | U_NOCORR (all corrections out) 15.73 | QS 2 + NOCORR1 16.30 |
lean u128 15.70 (the lean build is 0.1-0.3 faster than the dev u3k build of the same config: compare lean to lean).
### iteration 1 analysis block
1. nsys: not re-run for the dev variants (trace-only differences); shipped u128 in-graph 16.77 (iteration 2 run).
2. ncu: see iteration 2 (u128_unpooler 18.82 us, 51.5 MB DRAM).
3. SASS (nvcc 12.1 sm_90a; NVRTC 13 on the box for the shipped configs: 168 regs, 0 spill, LOCAL 0): U_PFX adds one
   UBLKPF (bulk L2 prefetch) before the barrier inits; U_WF swaps the UBLKCP / UTMALDG order per prologue stage; hot loop
   unchanged (76 instr / chunk / warp on the dev build).
4. Binding unit: the W stream from HBM (probe u2: data alone 13.1 us; phase-1 end 13.4 = within 0.3 of it): the start
   (first W 1.4 us after launch, DRAM latency under the 21 MB prologue burst) cannot move the loop end. The tail
   (p1end -> end 2.3-2.5 us) is the only lever: correction 0.44 + staging 0.42 + DSMEM 1.12 + epilogue 0.48.

## iteration 2 — u4.py: push the partner half UNCORRECTED, the receiver applies the sender's escapes after the add
Design (u4.py = U3's lean u128 + changes): (a) no correction before the push (extract, stage, bulk-push right after
the last full chunk); (b) the gatherer warps also send each gathered full-chunk column's partner token half (128 B)
to the partner's `pgbuf` (complete_tx on its `pgbar`; expect_tx posted by the receiver from its table); (c) all
corrections run on my token half only: my full-chunk escapes, my solo escapes, then + partner's partial, then the
partner's full-chunk escapes (table list 3). Exactness: a correction applied after the add sees, for a one-hot row,
an exact zero from every other summand, so the result is the same FMA chain as on the sender (bit-exact);
gbuf slots switched to a half-major word layout (each token half = 128 contiguous bytes, 2 x LDS.128 per entry).
Build-up (same runs, box 0, 3 x 25 reps, local us):
| variant | u4 | lean u128 | diag: no partner path | diag: pushes, no list-3 corr |
|---|---|---|---|---|
| v1: column push = fence.proxy.async + 128 B bulk copies in the gatherer before the xe release | 16.80-16.90 | 15.84-15.87 | 15.78 | |
| v2: column push = one st.async word per lane, inside the gather (no fence) | 16.13 | 15.65 | 15.55 | |
| v3: + the three correction passes as ONE runtime loop (code warm for the last pass; -1000 SASS instr) | 16.29 | 15.97 | 15.87 | 16.06 |
fence.proxy.async in the gatherer costs ~1 us of x-ring stall (the gatherer is the 9th releaser of every x stage).
Trace (u4trace.py, TRACE builds u4_tr / u128_tr, median / max over CTAs, 9 reps, us from the first CTA start):
| stamp | u128 | u4 v3 |
|---|---|---|
| W0 landed / x0 waited | 1.46 / 2.06 | 1.38 / 2.05 |
| phase-1 end | 13.41 / 13.70 | 13.41 / 13.63 |
| correction done (u128) | 13.82 | - |
| push issued | 14.10 / 14.62 | **13.73** / 13.92 |
| solo issued / solo MMAs done | 14.78 / 14.91 | 14.40 / 14.51 |
| own corrections done (u4: list 1, list 2) | 15.14 (list 2) | 14.91, 15.17 |
| partner partial landed (rbar) | 15.20 / 15.62 | **15.34** / 15.78 |
| partner corrections done | - | 15.71 |
| end | **15.68** / 16.10 | 16.10 / 16.42 |
Reading: (1) the push leaves 0.37 us earlier but LANDS 0.14 us later: issued at 13.73 the 32 KB DSMEM transfer takes
1.61 us (overlapping the solo chunks' TMA ingress into the same SM) vs 1.10 us when issued at 14.10 after the loads
landed — the SM's ingress is shared by DSMEM and TMA (U3 saw ~15 GB/s DSMEM under TMA traffic); (2) in u128 the two
post-loop chains are already balanced (solo chain + own corrections end 15.14, partner data 15.20), so taking the
correction off one chain only moves it onto the other (u4: list 1 0.40 cold, list 2 0.26, list 3 0.25-0.37 warm);
(3) every correction pass costs 0.25-0.40 us even warm (smem round trips under the incoming DSMEM / TMA writes).
Final A/B (bench.py same run, --xclones 4, 30 reps, box 0): unpooler m128 SGLang 18.82 | **u128 15.97 (x1.178)** |
u4 16.26 (x1.157) => u4 LOSES 0.29 us; u4.py ships with U4_ON = False (run_u4 -> None everywhere).
Gates (u4 enabled, sub_u4t.py): check.py --layers unpooler --graph "checked 53 cases x 2 seeds + graph: ALL OK";
exact_all.py --layers unpooler --nesc 200 --ms 97,100,113,127,128 EXACT ALL OK (u4 active); --nesc 3000 --ms 97,113,128
EXACT ALL OK (u4's per-CTA tables overflow -> it disables itself, as u128 does).
### iteration 2 analysis block (u4_unpooler v3 vs u128_unpooler, same process)
1. nsys in-graph (nsys_shapes.py, box 0): u4_unpooler **17.10** vs u128_unpooler **16.77** us.
2. ncu (ncu_run.sh full set, flushed, boost): u4 18.98 us / u128 18.82 us; DRAM read 51.57 / 51.54 MB (2.72 / 2.74 TB/s
   over the ncu duration = 57 % of 4.8; in-loop ~4.2 TB/s = 87 %); SM 44.6 / 48.7 %; issue-active 25.5 / 26.1 %;
   occupancy 17.6 / 17.7 % (12 warps, 1 CTA/SM); eligible 0.39 / 0.40 warps/clk; HGMMA pipe (hmma% metric) 1.9 / 2.0 %
   of the ncu window (tensor busy ~47 % from FLOPs: 8.6 GFLOP / 18.9 us = 455 TF of 949); top stalls u4 long_scoreboard
   35 %, wait 20 %, selected 9 %, math_pipe_throttle 8 %, warpgroup_arrive 5 % | u128 long_scoreboard 34 %, wait 19 %,
   selected 9 %, math_pipe_throttle 7 %, short_scoreboard 6 %.
3. SASS: NVRTC 13 on the box (nvc.py): u4 168 regs, 0 stack, 0 spill (LOCAL_SIZE_BYTES 0), no C75xx warnings; same
   for u128. nvcc 12.1: u4 2040 SASS instr (u128 2784: one correction body instead of three), hot loop 66.7 instr per
   chunk per warp vs u128 67.8 (same counting: first..last HGMMA.64x128x16 of the 6-chunk unrolled loop; LOP3 93,
   PRMT 80, IADD3 48, HGMMA 24, LDS.128 15, SYNCS 20 per 6 chunks); no LDG anywhere (3 UTMALDG.2D + 3 UBLKCP.S.G
   producer copies, 1 UBLKCP.S.S push, 2 STAS column pushes) -> no sunk loads (afterUse n/a).
4. Binding unit: the HBM W stream in the loop (~4.2 TB/s = 87 % of 4.8; data-only probe 13.1 us vs loop end 13.4) and,
   after it, the SM ingress port shared by the 32 KB DSMEM partial and the solo-chunk TMA loads (DSMEM 29 GB/s alone,
   ~20 GB/s under TMA). Nothing in the u128 tail is idle: both post-loop chains end within 0.06 us of each other.
