# g7: dense GEMM per-shape tournament (base research/claude/stk/s_b.py), 2026-10-31

Goal: per (M, N, K) dense GEMM in the scored cases, pick the fastest of {current cuBLASLt/sk2 plan, custom DG
variants (TMA-store epilogue, PDL fill, tile/cluster), CUTLASS 3.x configs}; route each by exact shape; one candidate.

## Environment facts
- Sandbox image has CUTLASS 3.x headers (flashinfer/data/cutlass, deep_gemm/include, tilelang/3rdparty) and
  nvcc 13.0; existing kernels compile with NVRTC (cuda.bindings) at prepare (300 s budget).

## Shapes and current plans in s_b (case: kernel, ncu isolated us / TC% / GHz)
| shape (M,N,K) | case | s_b plan | us | TC% | GHz |
|---|---|---|---|---|---|
| 753,2304,8192 QKV sliding | 0 | torch/cuBLAS nvjet 112x128 1x2, 126 CTAs | 39.5 | 82.3 | 1.69 |
| 753,2304,8192 QKV full (sm_count 112) | 1 | DenseLt nvjet 128x128 2x2, 108 CTAs (kv_pack beside) | 43.1 | 70.7 | 1.76 |
| 753,8192,1024 o_proj | 0,1 | sk2 3WG 192x128 epi2, PDL from attention | 21.3 | 64.7 | 1.73 |
| 753,4608,8192 pool+router FP32 | 0,1 | LtX tn 144x192 coopB ws0 | 77.2 | 86.9 | 1.60 |
| 753,8192,4096 unpool | 0,1 | LtX nn 192x128 coopA ws0 | 67.9 | 86.8 | 1.63 |
| 3012,2304,8192 QKV sliding | 2 | DenseLt nvjet 256x104 | 146.3 | 92.7 | 1.55 |
| 3012,2304,8192 QKV full (sm_count 108) | 3 | LtX qkv_smt 128x168 2x2, 108 CTAs | 165.0 | 77.7 | 1.65 |
| 3012,8192,1024 o_proj | 2,3 | LtX tn 128x168 ssched | 65.7 | 87.5 | 1.63 |
| 3012,8192,4608 pool+router (nn) | 2,3 | LtX nn 128x168 ssched | 287.4 | 94.8 | 1.55 |
| 3012,8192,4096 unpool | 2,3 | LtX tn 128x168 2x1 ssched | 256.2 | 93.4 | 1.57 |

## Log
### Probe tools
- tp.py: per-shape cold-L2 graph probe vs s_b's exact plan (sk2 / DG / cuBLASLt list / explicit LtX / CUTLASS .so);
  --trace prints the per-CTA timeline (sk2 SK_TRACE, DG DG_TRACE); --ncu + ncu_tp.sh for real-clock ncu counters.
- Every sk2 / DG variant below is bit-identical to cuBLAS (diff-vs-base 0): same k-order FP32 accumulation.

### s1 (s1.log, s1b.log): sk2 configs vs cuBLAS, standalone (paired graph time, us; + = slower)
| shape | variant | paired | fill / main / epi (us, per-CTA median) |
|---|---|---|---|
| u753 | 3WG 192x128 epi2 dp, 2 tiles/CTA | +1.41 | 2.11 / 30.5+32.1 / 0.61(hidden)+1.38 |
| u753 | same, cluster 2 (W multicast) | +3.79 | 2.85 / 30.6+32.5 / 0.48+1.34 |
| u753 | 2WG 256x192 epi1 (1 tile, 129 CTAs) | +1.73 | 2.94 / 60.2 (1.654 GHz) / 3.07 |
| o753 | 3WG 192x128 epi2 grid 132 | +0.30 | 2.14 / 7.8+7.6 / 0.64+1.09 |
| o753 | cluster 2 | +1.50 | |
| o753 | 2WG 256x192 epi1 | +2.29 | 2.98 / 15.0 / 3.20 |
| f753 | 3WG 192x144 ns5 (FP32) | +2.54 | 2.40 / 70.6 / 3.49 |
| f753 | cluster 2 | +4.38 | |
| q753s | 2WG 128x112 ns6 | +1.33 | 2.08 / 34.9 / 0.83 |
| q753s | cluster 2 | +2.66 | |
| q753s | 3WG 192x72 | +1.82 | |
ncu (u753a): nvjet 68.4 us 1.61 GHz TC 87.1%; sk2 192x128 68.7 / 1.62 / 85.5; sk2 256x192 69.5 / 1.66 / 83.0.
=> sk2 multicast always loses (longer fill, slower mainloop). The 256x192 tile's mainloop is 3% faster (higher
clock: 28% less L2->SM traffic per FMA) but its fill (+0.8) and single 98 KB epilogue (3.1 us) eat it.

### c1/c2 (c2.log): CUTLASS 3.x (flashinfer's headers, nvcc 13.0, sm_90a) vs s_b's plan, standalone paired us
Names: qc = coop 128x128 (c1x1 / c2x1 / c1x2), qp = pingpong 64x128 / 64x256, qsk = coop 128x128 stream-K,
qs = swap-AB (tokens on wgmma N) coop 128x128 / 128x112 / pp 64x112; ns = swap coop 128x192 (c1x1, c2x1), pp 64x192,
stream-K 128x192; nc1 = coop 128x256, nsk = its stream-K, np1 = pingpong 64x256.
| shape | best CUTLASS | paired | others |
|---|---|---|---|
| q753s | qs1 swap coop 128x128 | +3.79 | qc1 +4.16, qc2 +4.30, qc3 +6.6, qsk +7.2, qp* +13..16, qs2/qs3 +47 |
| u753 | ns1 swap coop 128x192 | +1.50 | ns2 +3.2, np1 +3.3, qc1 +3.9, ns3 +6.6, ns4 +11.9, nsk +12.6, nc1 +22.8 |
| o753 (vs sk2) | np1 pingpong 64x256 | -0.08 | ns3 +0.40, ns1 +0.82, ns2 +1.38, qc1 +1.8, ns4 +5.2, nc1 +6.3, nsk +10.7 |
CUTLASS stream-K is 7-13 us worse at 753 (fix-up + workspace memset); pingpong only competitive at K=1024.
| f753 | fs1 swap coop 128x192 stream-K | +8.83 | fsk +10.6, fs2 (swap coop 128x192 dp, 144 tiles) +53 |

### s2/s4 (s2.log, s4.log): sk2 pingpong, stagger, nvjet-equivalent tiles (paired vs cuBLAS, us)
- u753: stagger 1/2 (stage 0 before the rest) +1.10/+1.09 vs +1.63 unstaggered (fill 2.08 -> 1.98, main2 -0.4);
  pingpong 128x192 / 128x128 / 64x256 per WG +3.1 / +3.2 / +5.5 (one WG alone does not keep the pipe busy);
  nvjet's own tile (128 tok x 192 feat, 2 WGs, no cluster): sk2 epi2 +1.46, direct +2.46, DG +1.66; 'n' raster +11.7.
- o753 (vs sk2 base): stagger +0.13, pingpong +1.3..+1.5, 192x64 4 tiles/CTA +1.5.
- q753s: sk2 128x112 ns7 stagger +1.02 (fill 2.05 / main 35.1 = 98% of wgmma rate at 1.66 GHz / epi 0.80); DG +1.38.
- nsys (u753, 10 reps): nvjet kernel 66.24 us, sk2 stagger 67.01, DG 66.86; launch gap after the flush 4.0-4.1 for
  all three -> the deficit is inside the kernel (fill 2.0 + epilogue 1.3 + end spread 0.8 vs nvjet's unknown split).

### c3 (c3.log): T=3012 tournament (paired vs s_b's cuBLASLt plan, us): everything custom loses
- q3012s base 146.8: sk2 stream-K 128x128 +32, 256 wide +62; CUTLASS best nsk (coop 128x256 stream-K) +18.8, nc1 +26.
- q3012f base 166.8 (108 SMs): sk2 dp 2 tiles/CTA 128x256 +4.0, 256x128 +3.5; stream-K 168 +88.
- o3012 base 68.3: CUTLASS best nc1/ns2 +12; sk2 pingpong +10.9.
- u3012 base 257.7: CUTLASS nc1 +11.6, ns2 +15.6; sk2 stream-K +114.
- f3012 base 290.9: CUTLASS fs1 +28; sk2 +65..+129.
(CUTLASS run with its default raster/swizzle; sk2's m-fastest raster thrashes L2 at 12-24 m-tiles, as the DG notes found.)

### s5/s6 (s5.log, s6.log, s6b.log): what the 2 us fill is
- One CTA alone (grid 1, cold L2) also takes 1.76-1.82 us from kernel start to its first stage landed; 132 CTAs 1.98-2.11.
  So the fill is the cold first-access latency chain (tensor-map fetch -> TLB walk -> DRAM), not bandwidth/contention.
- Raw-pointer L2 prefetch of the first stages at kernel start (sk2x.py, to overlap that chain with the tensor-map
  fetch) makes it worse: bulk prefetches queue ahead of the TMA loads in the TMA unit (1 CTA: 1.82 -> 4.61 us), LSU
  prefetch.global.L2 per line 1.76 -> 2.27 us; 132 CTAs +0.5..+2.4 us end to end. Dropped (code left off by default).

### s7 (s7.log): T=3012 with grouped (swizzled) rasters in sk2 (sk2x.py: 'gG' bands of G m-tiles, n walked inside;
'hG' bands of G n-tiles, m fastest; dealt data-parallel over that order), paired vs cuBLAS (us)
| shape (base) | m-fastest | best grouped | other |
|---|---|---|---|
| q3012s (148.0) | 192x144 +8.3 | 192x144 g4 +5.1 (g8 +6.7, h4 +6.1) | 128x144 g4/g8 +12; 128x256 stream-K g4/g8 +53/+97 |
| q3012f (167.1, 108 SMs) | 128x256 grid 108 +3.5 | g4 +2.9 (g8 +3.1, h3 +4.0) | 132 CTAs: 192x144 g4 **-13.4** (153.7), g8 -12.7 |
| o3012 (68.3) | 192x128 epi2 +7.4 | h8 +7.1 (g4/g8 +7.2) | 256x128 epi2 g4 +8.4 |
| u3012 (259.1) | 192x128 epi2 +18.2 | g8 +13.5 (g4 +16.8, h8 +16.3) | 256x128 g4 +28 |
| f3012 (290.8, FP32) | 192x144 +23.6 | g8 +20.5 (g4 +22.6, h4 +21.0) | |
The raster was worth 3-5 us, not the gap. What is left vs nvjet at 3012: wave quantization of data-parallel tiles
(f 512 tiles / 132 = 3.88 -> 4, u 7.76 -> 8: ~3%) and exposed epilogues (FP32 direct stores on all 4 f-tiles);
nvjet's 'ssched' variants balance with cheap stream scheduling. q3012f on 132 CTAs beats cuBLAS-on-108 standalone,
but in the layer QKV shares the GPU with kv_pack on the side stream (s_b gives QKV 108 SMs so kv_pack gets 24): cuBLAS's
own 132-SM plan (q3012s, 148.0) is faster still, so the question is the SM split -> in-layer test t1 (QKV_SMT sweep).

### t1 (t1.log): in-layer SM budget of the full-layer QKV (QKV_SMT sweep, nsys node trace + CUDA events, 6 reps)
Full layers run QKV on the main stream beside kv_pack on the side stream; the attention starts after both.
| case | QKV_SMT | QKV node | kv_pack#1 | attn_fwd start | event median |
|---|---|---|---|---|---|
| 1 (753 full) | 112 (s_b) | 42.9 (108 CTAs) | 49.7 | 53.3 | 712.1 |
| 1 | 104 | 46.9 (96 CTAs) | 36.9 | 55.6 | 711.5 |
| 1 | 116 / 120 | 42.9 / 43.2 | 50.7 / 54.6 | 53.7 / 59.9 | 711.6 / 716.2 |
| 1 | 132 | 39.1 (126 CTAs) | 54.5 | 58.3 | 712.6 |
| 3 (3012 full) | 108 (s_b) | 170.8 (108 CTAs) | 182.0 | 195.7 | 2500.1 |
| 3 | 100 | 220.7 | 154.9 | 237.3 | 2544.7 |
| 3 | 116 / 120 / 132 | 170.8 / 169.2 / 169.7 | 187.5 / 190.8 / 190.4 | 200.1 / 202.3 / 202.4 | 2504.8 / 2503.0 / 2505.8 |
=> s_b's split is at the balance point: in case 3 kv_pack#1 (182.0) already ends after QKV + qkv_post2 is ready
(184.4 vs 182.8), so a faster QKV on its 108 SMs buys at most ~1.6 us there, and giving QKV more SMs slows kv_pack more
than it speeds QKV (+5 us at 116-132). Case 1 is the same (kv_pack 49.7 vs QKV 42.9 + qkv_post2 7.7). The "18% idle"
in the isolated ncu view is kv_pack's share in the layer. q753f / q3012f GEMM wins are capped at ~1-2 us unless the
QKV and kv_pack schedule is co-designed (kv_pack belongs to the attention worker).

### c5 (c5.log): CUTLASS at T=3012 with explicit raster (AlongM / AlongN) x max swizzle (1 / 4 / 8), paired us
best per shape: q3012s nsk AlongN s1 +18.3 (default +18.8); o3012 nc3 (coop 128x256 c2x1) AlongM s1 +11.3;
u3012 nc3 AlongM s1 +11.1, nc1 AlongM s1 +11.5; f3012 fs1 (swap 128x192 stream-K) AlongN s4 +18.9 (default +28).
Swizzle 8 is +30..+280 us (L2 thrash); 4 helps only the FP32 stream-K f3012. CUTLASS stays 4-12% behind nvjet.

### n2 (n2.log): where nvjet's ~1 us comes from (ncu, cold caches, real clocks)
| q753s | dur us | SM active max (clk) | tensor clk max / SM | L2 bytes | L2 hit | instr |
|---|---|---|---|---|---|---|
| nvjet 112x128 1x2 cluster | 39.58 | 62904 | 57344 | 268 MB | 72.6% | 4.35M |
| sk2 128x112 ns7 | 40.93 | 64923 | 57344 | 307 MB | 66.3% | 8.39M |
| DG 128x112 ns7 | 40.54 | 64723 | 57344 | 312 MB | 62.7% | 8.67M |
u753: nvjet 68.0 us / SM-active max 106482 / L2 424 MB; sk2 68.9 / 107819 / 434 MB; DG 68.7 / 107409 / 425 MB.
Same tensor work on the busiest SM (57344 / 98304 clk); nvjet's CTA lives 1300-2000 clk less (non-tensor time 5560 vs
7579 clk at q753s) and, at q753s, moves 13% fewer L2 bytes (its 1x2 cluster multicasts the token tile).

### s8 (s8.log): sk2 CL 3 = 2-CTA cluster multicasting the token tile (nvjet's 1x2), bit-identical, paired us
q753s 128x112 +18.7 (vs +1.47 without), 128x128 +7.0; u753 192x128 +5.1, 128x192 +10.5; o753 +1.9; f753 +4.8.
Fill 3.0-3.9 us (vs 2.0) and lockstep pairs (mainloop spread up to +16 us at q753s): the pair's shared empty-barrier
handshake couples the two CTAs' jitter. Same verdict as the W-multicast CL 2. Dropped.

## TOURNAMENT RESULT (per exact shape; standalone cold-L2 graph replays, paired vs s_b; ncu real clocks, cold caches)
| shape (M,N,K) | case | winner = s_b plan | ncu us / TC% / GHz | best challenger | paired | its ncu us / TC% |
|---|---|---|---|---|---|---|
| 753,2304,8192 QKV sliding | 0 | cuBLAS nvjet 112x128 1x2 (126 CTAs) | 39.6 / 82.3 / 1.68 | sk2 128x112 ns7 stagger | +1.02 | 40.7 / 80.1 |
| 753,2304,8192 QKV full (108 SMs) | 1 | DenseLt nvjet 128x128 2x2 | 43.5 / 70.3 / 1.75 | sk2 128x128 ns6 stagger grid 108 | not paired (ncu -0.2 us; case 1 not QKV-bound, t1) | 43.3 / 71.1 |
| 753,8192,1024 o_proj | 0,1 | sk2 3WG 192x128 epi2 grid 128 (PDL) | 21.5 / 64.2 / 1.73 | CUTLASS pingpong 64x256 | -0.08 | 21.7 / 63.8 |
| 753,4608,8192 pool+router FP32 | 0,1 | LtX nvjet 144x192 coopB | 77.5 / 86.9 / 1.59 | sk2 3WG 192x144 ns5 stagger | +2.27 | 79.2 / 84.5 |
| 753,8192,4096 unpool | 0,1 | LtX nn nvjet 192x128 1x2 coopA | 68.3 / 87.3 / 1.61 | sk2 3WG 192x128 epi2 stagger | +1.09 | 69.1 / 85.5 |
| 3012,2304,8192 QKV sliding | 2 | LtX nvjet 256x104 | 148.2 / 92.7 / 1.53 | sk2 3WG 192x144 grouped g4 | +5.1 | 152.8 / 89.6 |
| 3012,2304,8192 QKV full (108 SMs) | 3 | LtX nvjet 128x168 2x2 ssched | 167.3 / 77.7 / 1.62 | sk2 2WG 128x256 g4 grid 108 | +2.9 | 170.6 / 76.6 |
| 3012,8192,1024 o_proj | 2,3 | LtX nvjet 128x168 ssched | 66.7 / 87.2 / 1.61 | sk2 192x128 epi2 h8 | +7.1 | 74.0 / 79.7 |
| 3012,8192,4608 pool+router FP32 | 2,3 | LtX nn nvjet 128x168 ssched | 290.2 / 94.6 / 1.54 | CUTLASS swap 128x192 stream-K AlongN s4 | +18.9 | 307.8 / 89.9 |
| 3012,8192,4096 unpool | 2,3 | LtX nvjet 128x168 2x1 ssched | 257.5 / 93.3 / 1.56 | CUTLASS coop 128x256 c2x1 AlongM s1 | +11.1 | 269.4 / 89.4 |
(q3012f on 132 CTAs: sk2 192x144 g4 153.2 us / 89.9% beats cuBLAS-on-108 standalone, but in the layer the QKV / kv_pack
split is at its balance point (t1): more QKV SMs = +5 us at case 3.)

VERDICT: no dense-GEMM route beats s_b's plan at any scored shape, so there is no g7 candidate; s_b's routing already
is the per-shape winner table above. Measured causes at T=753: fill = cold first-access chain (1.8 us even for one CTA,
not hidable by prefetch), epilogue = write-bandwidth burst at the end of a single-wave tile (4.0-4.6 TB/s; FP32 f753
3.5 us), mainloops at 98-100% of the wgmma rate at the power-limited clock; nvjet's remaining edge is 1300-2000 clk of
per-CTA non-tensor time and 13% fewer L2 bytes (1x2 multicast), which our cluster variants could not reproduce.
- GPU: done (no processes left on the sandbox; CUTLASS .so builds left in /root/cutl_g7, /root/cutl_g7b).

### t2 (coordinator's owed test): in-layer early fill at T=753 (mk.py candidates on s_b)
sk2 already has the mechanism: prefetch.tensormap at kernel start, npre = ns weight ring stages TMA'd before
griddepcontrol.wait, activations after it. Candidates: g7_u (unpool on sk2 3WG 192x128 epi2 with PDL + the combine
calling griddepcontrol.launch_dependents at its start), g7_u0 (same without the combine trigger), g7_f (pool+router
on sk2 3WG 192x144 FP32 with PDL; bnd3_b1 already triggers at its start), g7_uf (u + f), g7_ut2 (u + the unpool's own
trigger after its last mainloop, for bnd3_b2). QKV sliding is the first graph node (no predecessor to overlap).
Measured with gap/trace.sh (gated nsys node trace + CUDA-event replay), cases 0 and 1, 8 reps.
Result t2 (t2.log; CUDA-event replay medians, two passes; nsys medians per node; case 0 unless noted):
| candidate | case 0 event (pass 1 / 2) | case 1 event | paired case 0 vs base (8 reps) | GEMM node |
|---|---|---|---|---|
| base s_b | 539.2 / 540.2 | 713.2 / 712.9 | - | unpool nvjet 64.8 us, ends 65.0 after combine end; fused nvjet 73.8, pnr5 starts 74.4 after bnd3_b1 end |
| u (unpool sk2 + PDL + combine trigger) | 537.7 / 540.2 | 714.1 / 715.9 | -0.1 | sk2 starts 4.6 us before the combine ends; ends 64.6 after combine end (-0.4) |
| u0 (no combine trigger) | 540.1 / 542.8 | 713.8 / 714.1 | +1.6 | launched at combine end; ends 65.9 after it (+0.9) |
| f (pool+router sk2 FP32 + PDL) | 540.7 / 543.6 | 713.9 / 715.9 | +2.8 | starts 1.8 us before bnd3_b1 ends; pnr5 starts 77.0 after bnd3_b1 end (+2.6) |
| uf | 541.3 / 543.2 | 717.5 / 717.6 | +3.2 | |
| ut2 (u + unpool triggers bnd3_b2 after its mainloop) | 540.0 / 540.8 | 715.2 / 715.2 | +2.2 | bnd3_b2 starts 1.5 us early but runs 24.7 vs 21.2 |
=> The early fill is real: with the combine's trigger the unpool's fill moves under the combine's tail and sk2 gains
1.3 us in the layer (u0 -> u). That exactly cancels sk2's standalone deficit (+1.1), so u lands at parity with nvjet
(case 0 -0.1 us paired, case 1 +1..+3); the pool+router gains nothing from PDL (bnd3_b1's last wave leaves too few
whole SMs free) and stays +2.6. Next (t3): unpool variants on top of the early fill (stagger, L2 prefetch of more W
k-blocks before the wait, nvjet's 128x192 tile).
Result t3 (t3.log; paired per-rep CUDA-event deltas vs base, mean over 8 reps, two passes; pair.py):
| candidate | case 0 pass 1 / 2 | case 1 pass 1 / 2 |
|---|---|---|
| u (repeat) | -0.43 / +0.11 | -0.36 / +0.62 |
| us1 (+ stagger 1) | +0.34 / +1.25 | -0.44 / +1.52 |
| upf8 (+ L2 prefetch of 8 more W k-blocks before the wait) | +1.21 / +2.20 | -0.77 / +0.37 |
| upf16 | +4.09 / +4.82 | +1.36 / +3.58 |
| un2 (nvjet's 128x192 tile, 2 WGs) | -0.19 / +0.57 | -0.39 / +1.47 |
u over t2 + t3 (4 passes x 8 reps): case 0 -0.14 / +0.40 / -0.43 / +0.11 (mean -0.02 us), case 1 +0.79 / +0.16 / -0.36 /
+0.62 (mean +0.30 us): parity with nvjet, not a win. Extra W prefetch before the wait hurts (it competes with the
combine's own DRAM traffic); stagger and nvjet's tile shape do not help.
Replica r1 (r1.log; judge replica, panel 11,22,33, cases 0-1, 8 reps), g7_u vs s_b (us):
case 0: 529.4 -> 528.9, 535.3 -> 537.3, 524.9 -> 524.8; case 1: 711.2 -> 709.6, 768.9 -> 768.6, 711.5 -> 712.7;
RELATIVE 1.0035 / 0.9973 / 0.9992, mean 1.0000.
VERDICT (early fill): the in-layer early fill works as designed: the combine triggers, sk2's CTAs land under the
combine's tail, and the weight ring plus the tensor maps are ready at griddepcontrol.wait, worth 1.3 us. That equals
sk2's standalone mainloop/epilogue deficit vs nvjet, so the unpool reaches parity, not a win. The pool+router gains
nothing (+2.6 us). No candidate. Probe builds g7_u*.py / g7_f.py / g7_uf.py are not deliverables.
- GPU: done (no processes left on the sandbox).
