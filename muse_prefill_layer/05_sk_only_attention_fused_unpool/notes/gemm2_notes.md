# gemm2: shape-scheduled BF16 GEMMs for the 753 / 3012 dense projections

Files: dev/gemm_sk2.py (kernel + host schedules), dev/gemm_sk2_bench.py (standalone A/B vs the layer's exact
cuBLAS call), dev/gemm2_gtsweep.py (steady-state sweep of the old gemm_tuned configs), dev/patch_gemm2.py.

## What cuBLAS runs in the layer (cand_full17 nsys trace, kernel names + grids)
| shape | kernel | grid | tiles |
|---|---|---|---|
| qkv 753 | tst_112x128_64x7_1x2 (112 W x 128 tok) | 126 | 21x6 = 126, 1 per CTA |
| o 753 | tst_128x192_64x5_2x1 | 132 | 64x4 = 256 (1.94 waves) |
| pool 753 | tss_256x112_64x4_2x1 (FP32) | 126 | 18x7 = 126 |
| unpool 753 | tst_128x192_64x5_2x1 | 132 | 256 (1.94 waves) |
| qkv 3012 | tst_256x104_64x4_1x1 | 132 | 9x29 = 261 (1.98 waves) |
| o 3012 | tst_256x152_64x4_1x2 | 132 | 32x20 = 640 (4.85) |
| pool 3012 | tss_256x144_64x4_1x2 | 132 | 18x21 = 378 (2.86) |
| unpool 3012 | tst_256x152_64x4_1x2 | 132 | 640 (4.85) |

## Measured facts (Modal H200)
- In-kernel clock under these GEMMs is 1.65-1.80 GHz (clock64 / globaltimer), and it falls with the number of
  busy SMs: 126 CTAs 1.678 GHz, 108 1.762, 72 1.78, 48-54 1.79-1.80 (qkv 753 sweep, logs/gemm2_sweep_qkv753.log).
  So 132 SMs at ~1.66 GHz = ~900 TF/s peak, not 989; the 10% targets at 3012 need ~95% of that.
- ptxas C7515 ("wgmma serialized ... non wgmma instructions defining accumulator registers") fires for the old
  gemm_tuned kernels and for any mainloop whose scale-d is a runtime predicate ((kb > 0 || k > 0)). Peeling the
  first k-block (scale-d a compile-time constant everywhere) removes it.
- Steady-state k-block rate of gemm_tuned configs on qkv 753 (clk per k-block vs wgmma ideal): 128x112 2WG 95.9%,
  128x128 2WG 87% (!), 128x192 96.3%, 128x256 97.4%, 192x128 3WG 94.6%, 192x192 96.2%; 2-CTA multicast of W
  (share=2) 128x112: 86%, of x (share=1): 80% with an 11 us end spread. Multicast does not help here.
- unpool 753 with 192x256 (128 CTAs, gemm_tuned): 1590 clk/kb vs 1536 (96.6%), clock 1.696 GHz, fill 2.3 us
  (consumer start -> first k-block), last epilogue (direct stores, 96 KB) 2.3 us, end spread 2 us.
- L2 micro (dev/gt_micro.py --test l2) numbers are not trustworthy (its cl=2 loop is latency bound).

## Log
### 01 (logs/gemm2_01.log) qkv 753, first sk2 kernel (standalone cold graph replay, cuBLAS 42.2 us)
- DP 128x112 (126 CTAs, = cuBLAS tile): 43.6 us; fill 2.3 us, end 37.1-38.1 us, clock 1.70 GHz.
- stream-K 128x96 (144 tiles, 139 kb/CTA): 68.9 us; 128x104: 80.0; 128x112 sk: 72.6; 128x192 sk: 58.0;
  split-K 2 128x216 (66 tiles x 2): 48.2 us.
- Why stream-K loses: each CTA starts at a different k offset, so the cold-L2 start pulls the whole working set
  from DRAM at once: fill 4.6-7.7 us (DP 2.3 us). The hand-off (FP32 partial, 48-56 KB) then takes 2.3-3 us
  (runs in that DRAM-saturated phase) and the owner add + epilogue 2-5 us. Split-K: owner waits for the
  simultaneous partner, 5.4 us add+epilogue.
- harness bug found: graphs wrote into freed out buffers (expandable segments unmapped them) -> Xid 31 faults; fixed (keep outs alive).
### 02 (logs/gemm2_02.log) DP one-wave configs, standalone (cuBLAS: o 24.7, unpool 71.2, pool 81.6 us)
- o 753: 192x256 3WG (128 CTAs) 26.2-26.3; 256x192 (2WG RM2, 129) 26.2; 128x256 2WG (192 tiles, 1.45 waves) 31.5.
  trace 192x256: fill 2.56, CTA end 19.2-20.5 us, epilogue (direct) 2.4 us, clock 1.77 GHz.
- unpool 753: 192x256 70.8, 256x192 70.5 (x1.01). Trace: fill 2.6, epilogue 2.3, clock 1.71/1.67.
- pool 753 (FP32): 128x216 (132 CTAs) 80.5, 192x144 3WG (128) 81.0, 256x112 (126) 83.3. Trace 128x216: clock
  1.644 GHz, fill 2.56, mainloop ~67.8 us (= ~99% of wgmma peak at that clock), FP32 epilogue 3.8 us (14 MB).
- Conclusion: at 753 the mainloop is at the power-limited clock already; the only slack is fill (~2.6 us, hideable
  with PDL if the previous kernel triggers early), the epilogue (2.3-3.8 us) and the end spread (~1 us).
### 03 (logs/gemm2_03.log) EPI 1 = last tile staged through the drained ring + TMA store: slower than direct
  stores (o 753 epilogue 3.3 vs 2.3 us, unpool 3.1 vs 2.3). The final epilogue is bound by ~5 TB/s aggregate
  writes when every CTA stores at once; TMA store adds smem write + sync.
### 04 (logs/gemm2_04.log) EPI 2 = every tile but a CTA's last through a dedicated staging buffer + TMA store
  that drains under the next tile's mainloop; last tile direct.
- o 753: 192x128 3WG epi2, 2 tiles/CTA on 128 CTAs: 24.9 us (cuBLAS 24.6; DP 192x256 26.1); last epilogue 1.0 us.
  128x192 2WG epi2 (129 CTAs): 25.1.
- unpool 753: all ~70.6-71.3 (cuBLAS 70.8).
- o 3012: 128x256 2WG ns3 epi2 (6 tiles/CTA, 128 CTAs) 73.8 us vs cuBLAS 75.9 (x1.029); 128x192 epi2 74.4;
  192x128 3WG epi2 75.0; 192x256 DP (4 tiles, direct epilogues) 78.4.
### 05 (logs/gemm2_05.log) fill probes (qkv 753 DP 128x112): stagger (issue 1-3 stages, wait for stage 0 to land,
  then the rest): no effect (43.4-44.0 vs 43.8 us). npre=1 (only stage 0's W before the PDL wait, x interleaved
  after): fill 1.79 vs 2.30 us, 43.2 vs 43.8 us standalone (~1%).
### 06 (logs/gemm2_06.log) pingpong (PP: 2 WGs own alternate 128-row tiles, ordered handshake)
- o 753 PP 128x192 (129 CTAs x 2 tiles): 26.1 us vs EPI2 192x128 24.8 vs cuBLAS 24.5.
- unpool 753 PP: 72.2 vs EPI2 71.0 vs cuBLAS 71.2.
- o 3012: PP 128x192 74.5, PP 128x128 78.1, EPI2 128x256 74.3, cuBLAS 76.3 (x1.028 best).
- unpool 3012: PP 128x192 278.9, PP 128x128 307.2, EPI2 128x256 265.5, cuBLAS 262.8 (clock 1.51-1.58 GHz:
  long 3012 GEMMs run lower clocks; cuBLAS 769 TF/s ~ 92% of 132*4096*1.55 GHz).
### layer01 (logs/gemm2_layer01.log) bench.py cases 0-3, reps 8, --check all pass. (Variant label "base" collides
  with bench.py's SGLang key, so its "sglang" column there is cand_full17; use the "vs base" ratios.)
- g1 = o@753 192x128 EPI2 + o@3012 128x256 EPI2 + attn_fwd early trigger: x1.0018 / 1.0016 / 1.0001 / 1.0011.
- g1n = same without the attn trigger: x1.0004 / 1.0025 / 1.0007 / 1.0023.
- g2 = g1 + unpool@753 (128x192 EPI2) + pool@753 (128x216 DP): x0.9728 / 0.9802 / 1.0006 / 1.0013 -> ~16 us
  LOSS at 753 although both are at standalone parity. Profiling (gemm2_prof01).
### prof01 (logs/gemm2_prof01.log) nsys in-layer, per-GEMM effective cost (end minus previous kernel's end)
| case | GEMM | cuBLAS (full17) | ours | |
|---|---|---|---|---|
| 0 | o 753 | 21.1 | 19.5 (g1, attn trigger) / 20.6 (g1n) | attn_fwd +0.9 us with the trigger |
| 1 | o 753 | 21.0 | 19.4 / 20.8 | |
| 2 | o 3012 | 71.5 | 70.7 / 69.9 | |
| 3 | o 3012 | 72.0 | 71.2 / 71.1 | |
| 0 | pool 753 | 76.0 | 92.7 (128x216, 132 CTAs) | stragglers, see ltrace |
| 0 | unpool 753 | 65.8 | 65.6 | |
### ltrace01/02 (dev/gemm2_layer_trace.py: SK_TRACE build inside the layer)
- pool 753 128x216 (132 CTAs) in the layer: CTA ends 72-98 us (median ~77), a few CTAs ~20 us late; standalone
  the same kernel's ends span 3 us. Every 132-CTA config (m or n order) shows it; 126/128-CTA configs do not:
  192x144 3WG ends 73.7-78.6, 256x112 76.3-79.6, 128x224 74.8-79.4. None beats cuBLAS's 76.0 us in the layer.
  -> pool 753 stays on cuBLAS.
### prof02 (logs/gemm2_prof02.log) o_proj W L2 prefetch before griddepcontrol.wait (pf k-blocks), +- attn trigger
| variant | case0 o753 eff | case1 | case2 o3012 | case3 | spans 0/1/2/3 |
|---|---|---|---|---|---|
| full17 cuBLAS | 21.2 | 21.3 | 71.5* | 72.0* | 584.1 / 793.2 / 1817.0 / 2729.1 |
| o1 trigger, pf 0/0 | 20.4 | 19.7 | 69.1 | 69.8 | 584.8 / 793.6 / 1812.4 / 2724.0 |
| o2 trigger, pf 12/13 | 20.2 | 20.3 | 70.0 | 70.4 | 585.4 / 790.1 / 1817.6 / 2725.3 |
| o3 trigger, pf 28/45 | 19.4 | 19.1 | 82.3 | 71.7 | 584.3 / 790.3 / 1829.9 / 2730.6 |
| o4 no trigger, pf 28/45 | 27.4 | 27.6 | 90.5 | 91.0 | 592.6 / 801.6 / 1835.5 / 2748.1 |
(* at 3012 the cuBLAS o_proj follows a CUDA memset; eff is measured from that memset's end)
- Prefetch without an early trigger floods DRAM right when the GEMM needs its own loads: much worse.
### 07 (logs/gemm2_07.log) unpool 3012 standalone: 128x256 EPI2 ns3 262.4 vs cuBLAS 259.3 (x0.988); 192x128 3WG
  275.5; 128x192 269.0. Clocks 1.56-1.61 GHz. In the layer cuBLAS also pays memset + gap (~3 us) -> ~parity.
### 08 (logs/gemm2_08.log) qkv 3012: PP 128x144 153.3, 128x144 EPI0 154.6 vs cuBLAS 144.9 (x0.945). pool 3012:
  PP 128x144 317.5, 192x144 3WG 302.1 vs cuBLAS 288.2 (x0.954). -> qkv/pool/unpool 3012 stay on cuBLAS.
### layer02 (logs/gemm2_layer02.log) bench.py reps 10 --check (all pass): g3 = o@753 192x128 EPI2 + o@3012 128x256
  EPI2 + attn trigger (full layers / full grid only) vs full17: x1.0005 / 0.9992 / 0.9997 / 1.0044; geomean
  1.8682 vs 1.8681. The 1-4 us o_proj savings are inside the layer-level noise.
### prof04 (logs/gemm2_prof04.log) 753 unpool/pool with predecessor early triggers / prefetch (case 0 / case 1 eff)
- unpool 128x192 EPI2 + finish trigger: 65.9 / 69.8 vs cuBLAS 65.3 / 65.4; + pf16: 84.3 / 80.1 (much worse).
- pool 192x144 (bnd1 already triggers at start): 75.9 / 76.3 vs cuBLAS 75.7 / 76.1; + pf16: 80.4 (worse).
- L2 prefetch before the wait hurts every long-K GEMM; the early-parked CTAs' traffic slows the predecessor.
### 09 (logs/gemm2_09.log) 2-CTA clusters multicasting W (CL=2), standalone: qkv 753 45.6 vs 44.3 (cl1), o 753 26.4
  vs 25.0, unpool 753 72.4 vs 71.2, pool 753 82.9 / 85.0 vs 81.4. Clock unchanged (1.60-1.72 GHz); fill +0.5 us.
### 10 (logs/gemm2_10.log) W pre-warmed in L2 (read after the flush): o 753 24.6 vs 24.9 cold, fill still 2.3 us ->
  the fill is not weight latency.
### 11 (logs/gemm2_11.log) implicit DP schedule (tile coords computed; no dependent rng/seg table loads before the
  first TMA): fill 2.05-2.3 vs 2.3-2.56 us (one 0.256 us timer step), totals unchanged.
### prof05 (logs/gemm2_prof05.log) attention trigger always on + npre 0/1 (no / one weight stage before the wait):
  o 753 eff case0 20.4 (vs refined-trigger 21.0, cuBLAS 21.2), case1 20.3-20.4 (vs 20.7, cuBLAS 21.0); spans
  within +-3 us of full17 (profile-to-profile noise is ~1-2 us per kernel, ~3 us per span).
### 12 (logs/gemm2_12.log) standalone summary, best configs (cold graph replays incl. ~4 us fixed cost; cuBLAS = the
  layer's exact call): qkv753 43.0 vs 42.0 (x0.978); o753 24.4 vs 24.5 (x1.003); pool753 79.1 vs 80.4 (x1.016);
  unpool753 69.9 vs 70.5 (x1.009); qkv3012 153.9 vs 145.2 (x0.943); o3012 73.1 vs 75.7 (x1.036);
  pool3012 301.9 vs 288.3 (x0.955); unpool3012 256.2 vs 254.8 (x0.995).
### tests04: cand_gemm_4 (o@753 + o@3012 + attn trigger) passes the 5 tests.
### layer03 (logs/gemm2_layer03.log) cand_gemm_4 vs full17, reps 12, --check pass: x1.0007 / 1.0014 / 1.0007 /
  1.0006; geomean (turn1, turn4) 1.8738 vs 1.8710.
### ltws (logs/gemm2_ltws.log) cuBLASLt with a zero-size workspace at 3012 (no memset node): o 73.3 (#0 501/7) vs
  layer 75.4; pool 287.4 (#1 = the layer's algo 500/7) vs 289.8; unpool 256.4 (#1 423/5) vs 258.7; qkv no change.
### prof06 (logs/gemm2_prof06.log) in-layer 3012, eff = from the previous non-memset kernel
| case | GEMM | full17 | g4 (o routed) | g5 (+ zero-workspace cuBLASLt pool/unpool) |
|---|---|---|---|---|
| 2 | o | 72.4 | 69.4 | 69.7 |
| 2 | pool | 281.8 | 281.7 | 275.5 (tss_256x144 ssched, no memset) |
| 2 | unpool | 248.8 | 249.5 | 244.7 (tst_128x304 ssched) |
| 3 | o | 74.5 | 69.8 | 69.8 |
| 3 | pool | 283.0 | 281.1 | 276.1 |
| 3 | unpool | 250.6 | 249.7 | 244.5 |
spans case2 1803.2 / 1799.3 / 1791.1; case3 2708.2 / 2703.9 / 2693.4.
### FINAL: dev/patch_gemm2.py defaults -> cand_gemm_6.py (o@753 + o@3012 on sk2, attn trigger on full layers /
  full grid, pool@3012 + unpool@3012 on zero-workspace cuBLASLt)
- tests06 (logs/gemm2_tests06.log): the 5 tests pass.
- layer04 (logs/gemm2_layer04.log), bench.py reps 12 --check (all pass) vs full17: x1.0030 / 0.9997 / 1.0034 /
  1.0033; geomean (turn1, turn4) 1.8686 vs 1.8624.

## Summary
- The 10% bar is not reachable for qkv/pool/unpool at 753 or 3012: the mainloop already runs at ~96-99% of the
  wgmma rate at the power-limited clock (1.60-1.72 GHz with 126-132 busy SMs; fewer SMs clock higher), cuBLAS's
  tile choices already avoid most wave quantization, and the remaining slack (fill ~2 us, final epilogue 1-4 us)
  cannot be hidden: the fill is kernel start + first-touch latency (not weight latency: warm-L2 W changed nothing),
  and weight prefetch / early triggers slow the predecessor kernels more than they save.
- o_proj (K=1024: fixed costs are a large fraction) is the one shape family that gains: in the layer 753 ~3-9%,
  3012 ~5-6% (cuBLAS's 3012 cost includes a workspace memset + launch gap).
- Dead ends (numbers above): stream-K / split-K (cold-L2 staggered starts, slow hand-offs), 2-CTA multicast,
  pingpong, TMA-store epilogue for the last tile, L2 weight prefetch, staggered prologue, finish-kernel trigger.
