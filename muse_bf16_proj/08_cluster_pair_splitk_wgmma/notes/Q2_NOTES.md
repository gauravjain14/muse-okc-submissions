# Agent Q2 — m = 128 (verify-16) for qkv_gate W[2304,8192] and o W[8192,1024] (2026-11-02, base claude/t3/bo8.py)

Base at m=128: qkv_gate -> al's qb_kernel (BF16 SS-wgmma, 64 W rows x 128 tokens, cluster-3 split-K + DSMEM st.async
reduce, 108 CTAs); o -> al's ol_projection (BF16, packed W, 64 W rows x 128 tokens, ONE consumer WG, 128 CTAs).
Files: qg.py (kernel family = agent K's kg.py + mmax grid + RED 5), qt.py (same-run harness vs SGLang and bo8's qb / ol
rebuilt standalone, --trace globaltimer timelines, --nocheck for diagnostic builds), gensrc.py (source for local nvcc),
ncum.py (ncu report summary), mkg128q.py + g128q_cfg.txt -> g128q.py (deliverable), sub_g128q.py (g128q + torch
elsewhere), sub_bo8q.py (bo8 + g128q first).
Times: local us (graph replay after a 512 MB flush, empty graph subtracted; judge ~= local + 1.0). Same-run A/Bs only.
Trace columns (qt.py --trace, means over CTAs, us from the first CTA start): tma0 (first TMA issued), data0 (first
stage landed), loop-end, send (helper partial published), bar (finisher: partials landed), end.

## SHIPPED (current): g128q.py — qkv_gate, o, pooler, unpooler at 97 <= m <= 128 (all BF16 SS-wgmma, 128 CTAs)
- qkv_gate = (TM 64, BN 72, LAY 2, S 2, NS 12, MC 0, RED 9, MAP 0, PERM 1, ..., EPI 2, PRO 1, LASTD 0, WPK 2): 64 tiles of
  64 tokens x 72 W rows; the two consumer WGs of a CTA take alternate k-blocks (in-CTA k-split, combined through smem);
  split-K 2 as a CLUSTER PAIR (rank 0 k-blocks 0..63, rank 1 64..127); DSMEM reduce-scatter: each rank finalises 32 of
  the 64 token rows, the WG0 warps holding the partner's rows push them (9 KB) with st.async into the partner's smem
  (complete_tx on its mbarrier: no global memory, no fence, no polling); GPC-aware cluster placement (PERM 1, measured
  at prepare; without it the cluster launch costs +3.5 us).
- o = (64, 128, LAY 1, S 1, NS 9, EPI 2, PRO 1, WPK 2): 2 token tiles x 64 column tiles, both WGs on the same 64 tokens,
  WG g owns W rows 64g.. (m64n64k16), W prepacked into 16 KB swizzled stage images (one bulk copy, baked address).
- unpooler = the o config (N 8192, K 4096: 64 k-blocks).
- pooler = (64, 64, LAY 2, S 1, NS 14, EPI 2, PRO 1, WPK 2): 2 x 64 tiles of 64 tokens x 64 W rows, full K, in-CTA k-split.
| shape (bench.py same run, 80 reps, box 1) | SGLang | bo8 (x vs SGLang) | g128q (x vs SGLang) | delta | int_v7 (base_v7.log) |
|---|---|---|---|---|---|
| qkv_gate m128 | 16.46 | 12.73 (x1.293) | **11.33 (x1.453)** | **-1.40** | x1.111 |
| o m128 | 6.26 | 5.70 (x1.097) | **5.21 (x1.201)** | **-0.49** | x1.003 |
| pooler m128 | 24.13 | 18.46 (x1.307) | **17.31 (x1.394)** | **-1.15** | x1.267 |
| unpooler m128 | 19.12 | 17.23 (x1.110) | **16.51 (x1.158)** | **-0.72** | x1.073 |
| verify-16 (4 shapes, router excluded) | 65.97 | 54.12 (x1.219) | 50.36 (x1.310) | **-3.76** | |
Box 0 (same run, 100 reps): qkv 11.67 vs 12.79 (x1.421 vs x1.296), o 5.43 vs 5.83 (x1.172 vs x1.091), pooler 17.61 vs 18.66
(x1.432 vs x1.351), unpooler 16.84 vs 17.56 (x1.154 vs x1.106); verify-16 4 shapes 51.55 vs 54.84 (-3.29).
Robustness: qkv / o / pooler margins held in every run; the unpooler margin is noisier (qt.py runs -0.11 / -0.34 / -0.41,
bench.py -0.72 on both boxes).
Earlier same-run benches (qkv+o only): qkv 11.48 vs 12.58, o 5.18 vs 5.60 (box 1, 80 reps); qt.py runs: pooler 17.34-17.65
vs bo8(Z g128) 18.10-18.46, unpooler 16.74-16.75 vs bo8(M m128) 17.09-17.15.
Judge-projected for the 4 shapes (+1 us each): x1.2038 -> x1.2871 (~0.24 % of the final score per us -> ~+0.9 %).
Correctness: check.py sub_g128q.py --layers qkv_gate,o,pooler,unpooler --graph -> "checked 207 cases x 2 seeds + graph:
ALL OK"; exact_all.py sub_g128q.py --layers qkv_gate,o,pooler,unpooler --nesc 200 --ms 97,100,127,128 -> EXACT ALL OK.
prepare_g128q 0.4 s (4 NVRTC kernels + W packing (+188 MB device memory) + GPC probe + self-test incl. one-hot rows at
m 128/100). pooler/unpooler are outside my assignment (Z/M/L territory): integrate them only if nobody has better.
Integration: `if layer in ('qkv_gate', 'o', 'pooler', 'unpooler') and 97 <= m <= 128: out = run_g128q(x, cfg, st)`
(returns None for anything else; drop a layer from G128Q_CFG to leave it to the base).
Bugs found and fixed on the way (both caught only by checking on a NaN-prefilled output, now standard in qt.py):
(1) K's store_direct stored only WN/32*32 columns (WN = 72 dropped 8 columns per WG; checks passed because torch reused
SGLang's output buffer); (2) LAY 2 with EPI 2 and S 1 skipped the ring barrier before warpgroup 1 wrote its partial into
ring slots 0-1 (race with warpgroup 0's last k-block). K's shipped g256k configs (LAY 0/1, WN 64/128) hit neither.

## iteration 1: K's family at m = 128 (o: no split; qkv: new RED 5 = 4-way asymmetric split-K)
Same run (qt.py, box 0, 20 reps):
| variant | o | qkv |
|---|---|---|
| SGLang | 6.34 | 16.00 |
| bo8 (ol / qb) | 5.68 | 12.77 |
| o 128x64 LAY0 NS9 (128 CTAs) | 5.70 | |
| o 128x128 LAY0 NS6 LASTD3 (64 CTAs, K's m256 cfg) | 7.26 | |
| **o 64x128 LAY1 NS9 (128 CTAs)** | **5.38** | |
| qkv 128x72 LAY0 S4 RED5 D4 NS9 (128 CTAs) | | 14.43 |
| qkv 64x144 LAY1 S4 RED5 D4 NS8 (128 CTAs) | | 13.94 |
o sweep (box 0, 30 reps): NS 9/8/7 5.50/5.52/5.55 (bo8 5.73); PERM 1 5.79; LASTD 2 5.84 (worse).
Timelines: o 64x128: tma0 0.39, data0 1.69, loop-end 4.91 (16 k-blocks, 0.20 us each), end 5.31 (max 5.62).
qkv 64x144 S4: data0 1.64, loop-end 9.59 (max 10.77; helpers 31 k-blocks, finisher 35), send 10.64, bar 12.71, end max 14.2
=> the 4-way exchange (3 x 36 KB partials pulled into one finisher SM) costs ~3 us after the loop.
Isolation (qt.py --nocheck, defines VG_NOW / VG_NOX / VG_NOMMA, box 1; loop = data0 -> loop-end):
| build | qkv 64x144 S4: total / loop | o 64x128: total / loop |
|---|---|---|
| full | 14.27 / 8.1 | 5.33 / 3.2 |
| no W traffic | 10.94 / 5.55 (0.17 us/k-block = MMA rate) | 4.43 / 2.6 |
| no x traffic | 13.06 / 7.44 | 5.46 / 3.3 |
| no MMA | 14.08 / 8.0 | 5.09 / 3.06 |
=> both loops are bound by the W stream from HBM (qkv: 37.7 MB in ~8.1 us = 4.65 TB/s in the loop); compute alone
is 5.55 / 2.6 us; x adds 0.6 / ~0. 12-bit W (-25 % bytes) is the remaining loop lever; for qkv the exchange tail first.

### iteration 1 analysis block
1. nsys in-graph (box 0, same process): o: g128q kg_o_64_128 **6.34** vs bo8 ol_projection 6.61; qkv: bo8 qb_kernel 13.50
   (my RED5 qkv not in this nsys run; bench 13.94-14.43 vs qb 12.77).
2. ncu (full set, flushed, box 0):
   | kernel | dur | DRAM % / bytes / TB/s | L2 % | SM % | issue % | tensor % el / act | occ % | elig/clk | top stalls |
   |---|---|---|---|---|---|---|---|---|---|
   | g128q o 64x128 | 7.36 | 48.3 / 17.0 MB / 2.32 | 59.3 | 29.4 | 21.3 | 29.4 / 43.7 | 14.7 | 0.26 | long_sb 28, wait 18, barrier 16 |
   | bo8 ol_projection | 7.81 | 45.6 / 17.1 MB / 2.18 | 56.9 | 28.7 | 14.1 | 28.7 / 39.2 | 12.5 | 0.15 | long_sb 44, wait 18, barrier 15 |
   | bo8 qb_kernel | 15.39 | 56.9 / 40.0 MB / 2.60 | 64.9 | 33.5 | 17.3 | 33.5 / 49.5 | 8.1 | 0.18 | long_sb 43, wait 19, selected 12 |
   (ncu at 1.73-1.83 GHz; long_scoreboard = mbarrier full-waits)
3. SASS (local nvcc 12.1 -arch sm_90a on the NVRTC source; NVRTC on the box: 168 regs, LOCAL 0): o 168 regs, 0 spills,
   448 instructions, 4 HGMMA.64x64x16 per k-block per WG, 1 WARPGROUP.ARRIVE + DEPBAR + 2 SYNCS per k-block, producer
   1 UTMALDG (x) + 1 UBLKCP (W) per stage, no LDG anywhere, epilogue 4 STG.128 + 16 SHFL. qkv RED5: 656 instructions,
   4 HGMMA.64x72x16 per k-block per WG, 9 STG.128.STRONG.GPU (helper publish) + 2 LDG.STRONG.GPU (flag poll, after
   the loop) + 6 UBLKCP (W + 2 partial pulls) + NANOSLEEP + MEMBAR.ALL.GPU (fence.acq_rel).
4. Binding unit: the HBM W stream (o: 16.8 MB; DRAM 48 % of the ncu duration = 2.3 TB/s averaged over the kernel, but
   ~4.5 TB/s inside the loop from the trace; tensor idle 56 % of active cycles). Outside the loop: start (first stage
   lands 1.6 us after kernel start) and epilogue (0.4 us, 16 KB/CTA at the ~30 B/clk store path).
   Next: qkv exchange (reduce-scatter instead of one finisher pulling 108 KB), then 12-bit W.

## iteration 2: qkv exchange variants (RED 6 / 7 / 8)
- RED 6 = TM 64 x BN 72 (LAY 2: in-CTA k-split, the two WG partials combined in smem) + 2-way asymmetric split-K through
  global memory (helper publishes ONE 18 KB partial + fence + flag; the finisher's producer pulls it into a buffer after
  the ring). Same run (box 1, 20-30 reps): D (finisher surplus k-blocks) 4/8/12/16/20/24: 12.74/12.51/12.18/12.11/
  12.32/12.29 vs bo8 12.69-12.88; NS 12 > 11 > 10 > 9.
  Trace D16: helpers loop-end 8.89, send 10.31 (1.4 us for combine + 18 KB STG + MEMBAR.GPU + flag), finisher loop-end
  10.72, pull issued 10.99 (flag seen 0.7 us after send), partial landed 11.34, end 11.71 (max 12.08). The finisher-only
  phase (16 k-blocks on 64 SMs, 0.12 us each = ~2.4 TB/s of W) wastes ~0.9 us; the exchange ~0.9 us after the loop.
- RED 7 = RED 6 without flag/fence: helper stores the partial with bit 0 of every fp32 word = per-tile launch epoch
  parity (<= 1 ulp), the finisher polls the data itself (ld.global.cg). D 4/8/12: 14.75/14.00/12.10 (= RED 6): the
  global-memory store -> visible -> load latency (~1.2-1.5 us under the HBM stream) dominates whatever the signalling.
- RED 8 = RED 6 as a cluster pair (rank 0 helper, rank 1 finisher of one tile), the helper pushes its partial with
  st.async into the finisher's smem. Without placement: 15.28-15.39 (loop 13.4 us instead of 10: V's GPC effect: the
  cluster launch puts consecutive clusters on different GPCs). With PERM 1: D 2/4/6/8/12: 11.92/11.81/11.63/11.62/11.84
  vs bo8 12.56. Trace D8: helper loop-end 9.66, push issued 10.31, finisher loop-end 10.41, landed 10.89, end 11.25.

## iteration 3: RED 9 = symmetric cluster-pair DSMEM reduce-scatter (shipped)
Both ranks run 64 k-blocks (no idle phase); after the in-CTA combine, rank r keeps rows [32r, 32r+32) and pushes the
other 32 rows (9 KB) to the partner. Same run (box 1, 30 reps): RED 9 NS12 11.55 | NS11 11.71 | RED 8 D8 11.89 | bo8 12.64.
Trace: loop-end 10.15 (max 10.55), push issued 10.66, partner rows landed 10.98, end 10.99 (max 11.56).
### analysis block (iterations 2-3)
1. nsys in-graph (box 1, same process): qkv g128q RED 9 **12.19** vs bo8 qb_kernel 13.42 (RED 8: 12.32, RED 6: 13.01);
   o g128q 6.24 vs bo8 ol_projection 7.04.
2. ncu (full set, flushed):
   | kernel | dur | DRAM % / bytes / TB/s | L2 % | SM % | issue % | tensor % el / act | occ % | elig/clk | top stalls |
   |---|---|---|---|---|---|---|---|---|---|
   | qkv RED 9 | 13.70 | 62.7 / 39.9 MB / 2.91 | 67.7 | 37.9 | 19.6 | 37.9 / 47.1 | 13.9 | 0.22 | long_sb 51, wait 18, barrier 9 |
   | qkv RED 8 D8 | 13.89 | 62.4 / 39.9 MB / 2.87 | 66.4 | 37.8 | 21.1 | 37.8 / 48.2 | 14.2 | 0.24 | long_sb 47, wait 20, barrier 9 |
   | qkv RED 6 D16 | 14.53 | 60.0 / 39.9 MB / 2.74 | 67.7 | 32.5 | 18.4 | 32.5 / 47.3 | 13.9 | 0.21 | long_sb 48, wait 20, barrier 11 |
   | bo8 qb_kernel | 15.39 | 56.9 / 40.0 MB / 2.60 | 64.9 | 33.5 | 17.3 | 33.5 / 49.5 | 8.1 | 0.18 | long_sb 43, wait 19 |
   (long_scoreboard = full-barrier waits for TMA data; DRAM % is over the whole ncu duration incl. start and tail)
3. SASS (local nvcc 12.1 on the NVRTC source; NVRTC on the box: 168 regs, LOCAL 0): RED 9 688 instructions, 0 spills;
   hot loop per k-block per WG ~30 instructions: 1 SYNCS.PHASECHK (full wait), WARPGROUP.ARRIVE, 4 HGMMA.64x72x16,
   WARPGROUP.DEPBAR.LE gsb0 1, 1 SYNCS.ARRIVE (slot release), uniform address math; loop bound folded to a constant
   (no HGMMA serialisation warning); no LDG anywhere; producer 1 UTMALDG (x) + 1 UBLKCP (W) per stage; epilogue
   9 STS + 18 LDS (combine + received rows), 9 STAS.128 (st.async DSMEM push), UCGABAR arrive/wait (cluster barrier),
   4 STG.128 + 2 STG.32 + 16 SHFL (WN = 72 direct store).
4. Binding unit: HBM W stream in the loop: 64 k-blocks in 8.6 us (cons start 1.52 -> loop-end 10.15) = 37.7 MB at
   4.37 TB/s (91 % of 4.8; the bench model's 4.56 TB/s effective -> ~96 %). Outside the loop: DRAM ramp (first stage
   ~1.5 us after launch: 128 x 17 KB first-stage burst + latency) and the ~1.0 us tail (combine 0.2 + push 0.3 +
   transit 0.3 + add/store 0.2). Floor at this byte count: 0.81 + 37.7/4.56 = 9.1 local (we are at 11.5).
   Next: bytes (12-bit W) is the only large lever left for the loop; the tail could lose ~0.3 more.

## iteration 4: same family on pooler / unpooler m128 (outside my assignment, shipped as optional configs), dead ends
Same run qt.py (box 1, 30 reps; bo8 routes rebuilt standalone: Z's g128 / M's m128):
| variant | pooler | unpooler |
|---|---|---|
| bo8 | 18.10-18.46 | 17.09-17.15 |
| pooler 64x64 LAY 2 S 1 NS 12 / 14 | 17.39-17.65 / 17.46-17.57 | |
| pooler 64x128 LAY 2 S 2 RED 9 PERM NS 8 | 17.57-17.70 | |
| pooler 64x64 LAY 2 PERM 1 | 18.96 | |
| unpooler 64x128 LAY 1 NS 9 / 8 (o config) | | 16.74-16.75 / 16.83 |
| unpooler 128x64 LAY 0 NS 9 | | 17.60 |
- Stage-major packed W (WPK 3: all CTAs' images of one k-block contiguous, T's suggestion): qkv 11.84 vs 11.57 (worse),
  o 5.31 vs 5.34 (equal). Dead end (Z saw the same for pooler / unpooler).
- unpooler loop-end spread 0.6-0.8 us inside every GPC (mean 15.7, max 16.6); token-tile pairs that share W run in
  lockstep (|even - odd| 0.03 us), the spread is between column-tile pairs (DRAM service variance). qkv / pooler
  spreads are 0.2-0.35 us.
- 12-bit W (option b), why not built: qkv N = 2304 = 36 x 64; RS-wgmma needs W as the A operand in 64-row units ->
  36 / 72 / 18 row tiles do not map onto ~128 SMs without a 3- or 7-way split (108 CTAs: tensor time 6.4 us at 100 %
  = the 12-bit HBM time, so no gain; 126 CTAs: 64 KB partials). o (128 CTAs of 128 W rows x 64 tokens fits) is
  compute ~= memory at 12 bit (2.4 us MMA at 100 % + RS decode on the critical path vs 2.9 us of 12-bit HBM time) ->
  <= 0.3 us at best. Decode-to-smem (U's dk) for SS: smem traffic 47.5 KB per k-block vs 34 KB now. Z / M / U's 12-bit
  RS kernels lose to this BF16 family on pooler / unpooler too.
### analysis block (iteration 4)
1. nsys in-graph (box 1): pooler g128q 18.26 vs bo8 mz_pooler 19.90; unpooler g128q 17.63 vs bo8 mk_unpooler 18.21.
2. ncu: pooler 20.96 us, DRAM 71.8 % / 69.2 MB / 3.30 TB/s, L2 76.8 %, SM 46.0 %, issue 24.4 %, tensor 46.0 / 53.8 %,
   occ 14.3 %, elig 0.28, stalls long_sb 53, wait 21, selected 9 | unpooler 19.58 us, DRAM 75.5 % / 68.2 MB / 3.48 TB/s,
   L2 80.1 %, SM 46.5 %, issue 23.0 %, tensor 46.5 / 53.7 %, occ 14.3 %, elig 0.28, stalls long_sb 43, wait 22, selected 10.
3. SASS: same loop as qkv / o (4 HGMMA.64x64x16 per k-block per WG, 168 regs, 0 spills, LOCAL 0, no LDG).
4. Binding unit: HBM W stream (pooler 128 k-blocks in 15.6 us = 67 MB at 4.3 TB/s; unpooler 64 k-blocks in 14.3 us
   = 4.7 TB/s incl. the start prefetch); outside: start 1.5 us, epilogue 0.4-0.6 us, unpooler spread 0.9 us.
- Integration check: build_t3.prefix_names applied to g128q.py (mkprefixed.py -> g128q_prefixed.py + sub_prefixed.py):
  check.py --ms 97,113,128 --graph on all 4 layers ALL OK, route active (st ok, PERM on qkv).

## iteration 5 (final tries)
- RED 10 (RED 9 without the in-CTA combine: both warpgroups push their partner-half rows, 18 KB DSMEM per CTA): 11.57 vs
  RED 9 11.55 (tie): rows land 0.87 us after loop-end vs 0.80; DSMEM moves ~29 GB/s per SM (9 KB in 0.32 us), so
  doubling the push volume eats the saved combine.
- unpooler PERM 1: 17.15 vs 17.17 (no change); PRO 0: 17.36 (worse).
- Where the remaining time goes (qkv RED 9, us from launch): first TMA 0.3, first stage lands ~1.5 (128 x 17 KB first-
  stage burst at the HBM rate + latency), loop 8.6 at 4.37 TB/s, tail ~1.0 (WG1's last k-block 0.13, combine 0.2,
  push 0.15, DSMEM transit 0.32, add + store 0.2), CTA spread ~0.3. o: first stage 1.6, loop 3.2, epilogue 0.4.
  Both loops are at the practical HBM limit for BF16 bytes; the remaining levers are bytes (see the 12-bit analysis in
  iteration 4) and the ~1 us split-K tail of qkv.
