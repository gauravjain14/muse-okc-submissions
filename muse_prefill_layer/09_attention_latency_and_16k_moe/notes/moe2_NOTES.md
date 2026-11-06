# moe2 worker notes: per-item overhead of okc_moe::moe_fused_persistent at small T (2026-10-30)

Base: research/claude/stack/s1.py (MoE source extracted to cu/s1_moe.cu; variants built by bsrc.py as text patches).
Tools: mb.py (MoE-only paired graph bench, item trace incl. first-stage arrival, R2-aware item map, MOE_FLAGS),
ncu_one.py / ncu2.sh / ncuexp.sh / ncu_ana.py (per-loop SASS stall analysis), sass.sh / sassq.sh / sass_loops.py
(nvcc 13 resource + C75xx + per-loop SASS checks), ub/ub.cu (consumer main-loop microbenchmark), ub/ub2.cu
(tensor-core interference by instruction type).

## 0. Bottom line (for the root)
No variant beats s1; nothing to submit from this lane. Every candidate was measured slower or within noise (sections
3-5). Where the per-item time above the tensor floor goes:
| item | above TC floor | made of (measured) |
| hot gate n256 (T=753, 41.6 main + 1.25 epi) | ~3.5 us | X + W feed 2.1 (X 1.45, W 0.6; flags 1/2), epilogue 1.25 (0.9 silu/stmatrix + barriers; 3-4 us in the first wave of each width class), item transition 0.3 |
| n64 gate (16.3 + 0.64) | ~7 us | decode not hidden under wgmma 4.2, kernel loop overhead 1.3, feed 0.9, epilogue 0.64, transition 0.3 |
| n<=32 (10.4-11.7) | ~6-9 us | decode issue ~6 (floor 6.6 us per 128 rows at IPC 0.84) slowed by the concurrent wgmma |
moe3's "+5.6 us per n256 gate item" compares against a standalone loop at 1.75 GHz; the fused kernel runs power-limited
at 1.63-1.66 GHz (ncu), where the n256 TC floor is 39.7 us, not 37.4; with nomem the fused n256 main is 39.4-39.7 us,
i.e. the fused main loop is at the tensor floor and the rest is feed + epilogue.
Root cause for w <= 128: a running wgmma halves the throughput of every non-FP32 pipe on its SMSP (section 2), and
the NVFP4 decode (15 int / fp16 instructions per 8 weights) lives entirely in those pipes; no restructuring of the
loop (R2 rows, 3 WGs, deeper groups / waits) gets more than ~10-17% in isolation, and in the fused kernel each one
either breaks ptxas (C7512) or loses to side effects.

## 1. Where the per-item time goes (measured)
Item trace (logs/r1.json, d11 T=753, us, main/epilogue): g256 41.6/1.25, g224 35.8, g208 33.3, g96 19.3/0.74,
g64 16.3/0.64, g32 11.65/0.51, g16 10.6/0.48, g8 10.4/0.48 (MAT_SMALL: mma.sync), d256 40.5/1.5 ... d8 9.4/0.32.
Schedule is dense: per CTA gaps 1.4 us, head 0.1, tail median 2.3 (max 10) of a 238 us span.
Decomposition of a w=64 gate item (16.3 + 0.64 us):
| part | us | how measured |
| tensor time (64 x 4096 x 128 rows at 1.65 GHz) | 9.9 | ub p64m1 (constant A): 64.2 cyc/k16 = TC peak |
| NVFP4 decode not hidden under the wgmma | 4.2 | ub p64m0 14.1 us vs p64m1 10.0 (no memory, no producer) |
| real-kernel loop vs ub (barriers, slot release, item setup) | 1.3 | nomem 15.4 vs ub 14.1 |
| memory: X (activation tile) re-reads | ~0.6-1.3 | flags=1 (X loaded only for the first 4 stages) |
| memory: weight stream | ~0.4-0.8 | flags=2 |
| first-stage arrival at item start | 0.3 | trace2 (first full-barrier pass) |
| epilogue | 0.64 | trace |
Memory totals at T=753 (logs/r6): X off -9.7 / -12.8 us (d11 / d22), W off -6.9 / -10.9, both -12.2 / -17.1 us of
266 / 297. Activation (X) re-reads cost MORE than the weight stream: every 128-row weight n-tile item re-reads the whole
tile's X (64 gate + 32 down n-tiles per tile): ~1.2 GB of L2->SM X traffic vs 0.4 GB of weights per T=753 call.
ncu (logs/ncu_s1a, 753 d11): issue active 35%, IPC 1.45/SM; tensor pipe active 62.7% (root); per loop: n256 is
DEPBAR(barrier)-bound (TC-bound), n<=128 loops are spread over selected/wait/dispatch (latency + pipe interference).

## 2. Root cause: wgmma on an SMSP halves the integer / fp16 pipes of every warp there (ub2.cu)
Per-warp throughput of independent chains, alone vs with a concurrent RS wgmma stream (n256 or n64) on the SMSP:
LOP3/SHF/PRMT/IMAD/IMUL 0.47 -> 0.23 inst/clk; HMUL2/HFMA2 bf16 or f16 0.81 -> 0.235; F2FP pack 0.23 -> 0.12;
e4m3 unpack 0.46 -> 0.24; IADD 0.89 -> 0.45; FP32 FFMA/FMUL 0.88 -> 0.65 (only FP32 FMA keeps its rate).
The NVFP4 decode (15 instr per 8 weights: 5 LOP3, 1 SHF, 1 PRMT, 4 IMAD, 4 HMUL2) is entirely in the slowed class.
Split test (ub, WG0 wgmma const-A, WG1 decode only): decode warp 27 -> 65.8 cyc/k16 when WG0 runs n256; n64: both
slow (wgmma 32 -> 46, decode 27 -> 48). Model that fits the ub data: cycles per k16 step (2 consumer WGs, 128 rows)
T ~= max(N, 42 + 0.7 N): p64 measured 92 (model 87), p128 137 (132), p256 256 (256); n32 78 (model 64, TC-latency).
=> for w <= 128 the decode issue (~38 slow-class instr per k16 per SMSP) cannot overlap the wgmma; each decode
instruction per k16 costs ~0.34 us per item. The decode itself is near-minimal (bit layout allows 7-9 int ops per 8
codes; no lossless shortcut for the e4m3 scale multiply; FP32-only decode would need F2FP packs, which are slower).
Decode-only floor (2 warps, IPC 0.84): 6.6 us per 128-row item; TC floor n32 5.2 us; measured 11.6.

## 3. Microbenchmark of structural options (ub/, cfg1-4; us per 128-row item, no memory)
| width | production replica | 2 m64 row blocks per WG (R2) | 3 consumer WGs | wait<2> | 4-k16 groups |
| 8  | 8.98 (wgmma) | 8.35 | | | |
| 16 | 9.71 | 8.68-8.73 | | 8.93 | |
| 32 | 11.58 | 9.49-9.63 | 10.28 | 11.53 | 9.56 (R2) |
| 64 | 14.11 | 12.84-13.3 | 12.52 | 13.77 | 13.92 |
| 96 | 17.37 | 16.06 | | | |
| 128 | 20.65 | 19.18 (= TC floor) | | | |
Deeper wgmma pipelining / bigger groups: <= 4%. 3 consumer WGs impossible with n256 in the kernel (registers).

## 4. R2 items in the real kernel (256 weight rows per CTA item, each WG two m64 passes sharing X): FAILS
r2part.py: consumer_item_r2<N>, producer two weight n-tile copies per stage, publisher kinds (gate: one bulk copy of
both n-tiles, done += 2; down: 512-byte rows), item map r2_map (round-robin chunks of 1024 weight rows as before).
Correct (rel err 9e-5 vs s1). Findings:
- ptxas: any 2+ R2 widths in the MAT_SMALL kernel -> C7512 (all wgmma serialized), with every combination tried
  (NACC, single accumulator array, volatile-tid rematerialization, dropping fine widths). One R2 width fits.
- The producer must hand the decoded item to the consumers through lane-0 shuffles (__shfl_sync of the smem values):
  plain smem loads made ptxas move slot/descriptor math off the uniform datapath (R2UR per HGMMA, +18 instr per
  k-block in EVERY width's loop). Lesson for any edit of this kernel: keep queue state warp-uniform provably.
- R2<32> only (tiles <= 32 padded to 32): R2 item 25.5 us = 12.75 per 128 rows vs 11.9 for the 128-row path at the
  same padding (m32t control), and n208/n224/n96/n64 items slow 7-15% (ncu: long_sb 4% -> 10% in their loops; their
  SASS is unchanged). Net 753: +27 us (d11), +31 us (d22). T=3012 unchanged. Rejected.
- Padding 8/16-token tiles to 32 alone (m32t): +7.5 us at d11.

## 5. The activation (X) stream and the epilogue (T=753 and T=3012)
- X vs W (flags 1 / 2: load only the first 4 stages of X / of W; logs/r6, r8): T=753 X -9.7 / -12.8 us, W -6.9 / -10.9;
  T=3012 X -33 us (-4%), W -18 us; per item hot X costs 1.3-2.3 us (g256 39.55 -> 38.24 at 3012, d256 41.0 -> 38.7).
  16K (DIRECT_PG) per-item the same (-2 us per hot item), its graph totals vary +-3% between replays.
- Not smem contention: ub/ub.cu with an extra TMA warp streaming 32 KB bulk copies into a separate smem region at
  ~39 B/clk (the n256 item's X rate is 32 B/clk) leaves the n256 wgmma at 256.6 cyc/k16 (grid 132 and 16); n64 +1%.
- Not L2 misses: evict_last for X / evict_first for weights (l2part.py): within +-0.3% at 753 and 3012.
- Queue order: the MAT_SMALL interleaved order (chunks of 8 n-tiles per tile) at T=3012 is +15 us (both draws): many
  SMs on the same X tile at once is not the problem.
- => X arrives late because of L2->SM feed bandwidth/latency (the n256 stage is 32 KB X + 4.6 KB W per 1024 cycles per
  SM, 132 SMs: ~7.8 TB/s from L2). MC2 (mcpart.py): 2-CTA clusters run n-tiles (u, u+1) of one tile in lockstep and
  rank 0 multicasts X into both CTAs (halves the X L2 reads).
- MC2 results (mcpart.py; correct, bit-identical outputs; T=753 d11 / T=3012 d11 MoE call, base 265.6 / 833.6 us):
  v1 (rank 1's producer relays each slot release; rank 0 waits for both before W and X): 340 us at 753;
  v1 without multicast (lockstep handshake only, each CTA loads its own X): 358 us -> the multicast itself saves 18 us,
  the coupling costs ~90;
  v2 (rank 1 consumers arrive remotely on rank 0's per-slot barrier): ptxas C7512 (any extra live state in the
  main loops serializes all wgmma);
  v3 (dedicated relay thread = idle producer-WG warp 2 on rank 1; rank 0 loads W on its own release, multicasts X
  after the peer barrier): 283 / 875 us (+17.5 / +42). n256 items at parity (40.2 vs 39.6, d256 40.4 vs 41.0), narrower
  items pinned at ~0.6-0.7 us per stage whatever their width (g160 37.2 vs 26.8, g64 19.8 vs 16.2, g8 11.2 vs 10.0):
  with 4 ring slots the cross-CTA release -> relay -> multicast round trip (~2.5 us) bounds the stage rate.
  => multicast needs a deeper ring than the 227 KB of shared memory allows at 37 KB per n256 stage. Rejected.
- Deeper feed instead (logs/r18, MoE call, d11/d22): 5 ring slots of 37 KB (staging 40 KB, kps 1 for N in (64,128])
  +4.1 / +7.2 us at 753, +4.8 / -1.4 at 3012; L2 prefetch distance 2: +1.6 / +3.8 (753), +13.8 / +10.9 (3012);
  distance 3: worse. The feed cost is not a ring-depth / prefetch-distance problem that these knobs reach.
- Epilogue (trace3/trace4): staging-free wait ~0.19 us (negligible); the silu + stmatrix phase is 0.9 us for g256 in
  steady state but 3-4 us in the first wave of every width class (all SMs start the class at once); excess over each
  class's p25 is 3 us per CTA at 753, 8 us at 3012. Not the gscale global load (staging it in smem: no change); most
  likely cold instruction fetch of the class's straight-line epilogue code by 132 SMs at once (each L2 flush between
  graph replays and the real layer's traffic evict the 370 KB kernel image) - but a warm-L2 replay (no flush, logs/r19)
  keeps it (excess 8.35 -> 7.51 us per CTA at 3012, first-wave g256 epilogue 4.2 -> 4.0 us): cause not identified.

## Log
- logs/r1: trace + nomem; r2/r3: R2 variants; r4: padding control + R2 nomem; r5: first-stage arrival; r6: X/W
  memory split; r7: L2 cache-policy hints; r8: X/W at 3012/16K; r9-r11: epilogue split; r12: queue order at 3012;
  r13-r17: MC2 v1/v3; r18: 5 slots / prefetch distance; r19: warm vs cold epilogue.
