# Agent V — m = 256 (verify-32) for qkv_gate / o / pooler / unpooler (2026-11-02, phase 2)

Base to beat: claude/t3/bo1.py at m=256: qkv `torch` (F.linear, cuBLAS nvjet), o / unpooler `torchT` (cuBLAS),
pooler `cutlass` (al's CUTLASS pingpong 128x64, swap-AB). Coordinator's judge numbers for bo1 m256:
qkv 17.17, o 9.16, pooler 24.49, unpooler 23.43 (judge ~= local + 1.0).

All times local us (graph replay after a 512 MB read flush, empty-graph cost subtracted) unless noted.
Absolute times drift 2-5 % between execs / boxes; compare only within one run.

## Physical ceilings measured (this box type)
- Tensor pipe (claude/t3/V/tcprobe.py, 132 CTAs, SS wgmma from smem, random data, no memory traffic):
  clock under TC load 1.68-1.72 GHz (clock64 / globaltimer over the loop) -> ~900 TF/s for a 20 us kernel.
  SS m64nNk16 runs at 99+ % of the per-clock peak for N = 64/128/256 with 1 or 2 consumer WGs (no traffic).
  => 100 % tensor floors at m=256 (128 SMs busy / 132): qkv 10.8 / 10.5 us, o 4.8 / 4.7, pooler & unpooler 19.3 / 18.7.
- L2 -> SMEM bulk-copy ingest (claude/t3/V/l2probe.py, 132 CTAs, 4 MB per CTA, 6-8 x 16-32 KB stages):
  distinct addresses per CTA: 9.2-10.5 TB/s aggregate (70-82 GB/s/SM); all CTAs streaming the same 1-4 streams:
  16-22.5 TB/s (123-180 GB/s/SM): the L2 serves concurrent requests for the same lines much faster.
- First TMA data lands 1.0-1.5 us after kernel start (cold HBM) in every config.

## Iteration 0: library routes (same run, box 1)
- CUTLASS 3.x sweep (claude/t3/V/cs/cutgen.py + vsweep.py: 16 configs: pingpong / cooperative, swap / no swap,
  tiles 64x128 .. 256x128, clusters 1x1/2x1/1x2, persistent and stream-K scheduler with DP / split-K 2-4 /
  stream-K modes). Best per shape vs SGLang in the same run:
  | shape | SGLang | torch | torchT | best CUTLASS |
  |---|---|---|---|---|
  | qkv | 17.86 | 18.22 | 19.46 | 21.54 (coop 128x64 stream-K) |
  | o | 7.55 | 7.90 | 8.19 | 8.61 (pingpong 64x256) |
  | pooler | 28.35 | 28.54 | 26.98 | 23.68 (pingpong 64x128 swap, = al's route class) |
  | unpooler | 23.23 | 23.07 | 23.07 | 24.86 |
  CUTLASS stream-K / split-K fix-ups are slow here (qkv 128x128: DP 41 us, stream-K 26.7, split-K 3 24.7).
- nsys kernel names (cuBLAS): qkv nvjet_sm90_tst_96x64_64x10_2x4 (96 CTAs of 96x64 tiles, full K -> uses 96 of 132
  SMs), o / unpooler nvjet 128x128_64x6, pooler 128x64_64x8.

## Iteration 1: vg kernel v1 (claude/t3/V/vg.py), NVRTC, BF16 SS-wgmma, TMA ring, non-swapped (A = x, B = W)
cfg (TM, BN, LAY, S, NS, MC, RED, MAP); 1 producer WG + 2 consumer WGs, 168 regs, 0 spills, 4 HGMMA per k-block
per WG, stmatrix + 16-B store epilogue.
- First results (same run): unpooler 128x128 S1: 23.02 vs SGLang 23.42 / torchT 23.22; o 128x128: 8.02 vs 7.92;
  qkv 64x72 k-split-in-CTA (LAY2): 17.6-18.0 vs SGLang 18.06; qkv 128x72 S2 cluster DSMEM: 20.3; pooler 64x128
  LAY2: 25.36 vs SGLang 29.15 (CUTLASS ~23.9 in another run).
- ncu (full set, flushed; 128 CTAs):
  | kernel | dur | tensor pipe (hmma+hgmma) % elapsed | issue % | L2 (lts) % | DRAM read | xbar->SM | top stalls |
  |---|---|---|---|---|---|---|---|
  | qkv 128x72 S2 cluster | 22.85 | 44.5 (56.5 % of active) | 20.2 | 67 | 42 MB, 1.84 TB/s (38 %) | 212 MB, 9.3 TB/s (31 %) | long_scoreboard 54, wait 17 |
  | qkv 64x72 LAY2 | 20.06 | 52.1 | 23.9 | 61 | 42 MB | | long_scoreboard 45, wait 24 |
  | unpooler 128x128 | 24.93 | 74.6 | 17.8 | 76 | 69 MB | | barrier 37, long_sb 17, wait 17 |
  (long_scoreboard = mbarrier try_wait on full barriers; nothing is at its physical peak: latency / delivery bound.)
- Isolation (globaltimer traces, defines VG_NOMMA / VG_NOX / VG_NOW): qkv 128x72 S2 loop (16.9 us) is the same with
  NO MMA at all (17.1 us) -> pure data delivery bound; unpooler NOMMA 17.1 vs 20.7 with MMA (compute-bound, 93 %).
- **Cluster launch is the problem for qkv S2**: the same kernel without __cluster_dims__ (VG_NORED: no reduction)
  runs qkv 128x72 S2 in 14.66-14.82 (loop 12.0 us = 90 % of tensor peak) vs 19.26-20.3 with a dummy cluster of 2
  (no cluster instruction executed). 64x144 LAY1 S2: 15.0 vs 25.3. TMA multicast pairs (MC 1/2, need clusters):
  20.8 / 27.7 vs 18.2 without. => no clusters for these shapes; split-K reduction must go through global memory.
- Global split-K exchange, symmetric (RED 1: sentinel-filled fp32 slots, receiver polls ld.global.cg): exchange
  takes 2.5-3 us (polls med 2, each L2 round trip ~1.2 us at that moment) -> qkv 17.8-18.6, no gain.
  => next: asymmetric split (helpers take fewer k-blocks, publish + flag; the finisher's idle producer thread
  polls the flag and bulk-copies the partial into smem while its consumers finish the last k-blocks).

## Iteration 2: asymmetric split-K through global memory (RED 2 / 4), first win (qkv)
- Tile 128 tok x 72 cols, S = 2 without clusters (64 tiles x 2 = 128 CTAs, traffic 209 MB vs 285 MB for the
  no-split 64x72 tile that cuBLAS / bo1 uses). Bids [0, 64) are helpers: k-blocks [64 + D, 128); bids [64, 128) are
  finishers: [0, 64 + D). Helper: st.global.cg of its fp32 partial (36 KB), bar.sync, one fence.acq_rel.gpu, flag.
  Finisher's idle producer thread: after its last TMA it waits for the two ring slots k-blocks nkb, nkb+1 would use
  (RED 4; RED 2 used a separate 36 KB buffer and could only afford NS = 7), polls the flag (ld.acquire.gpu, nanosleep
  backoff), resets it, and bulk-copies both halves of the partial into those slots (complete_tx on rbar); the
  consumers add after their loop. Helpers never wait (lower bids), so no co-residency deadlock is possible.
- setmaxnreg (producer 40 / consumers 232) matters even at 168 regs: removing it costs LAY2 qkv 17.65 -> 20.03 and
  RED2 16.74 -> 17.38 (same run). Kept.
- NS 7 -> 8 is worth ~1.1 us on the no-reduction 128x72 S2 kernel (15.79 -> 14.64), hence RED 4.
- Sweep (same run, box 0): RED4 NS8 D = 2/3/4/5/6: 16.75 / 16.74 / 16.30 / 15.97 / 16.11 (SGLang 18.16). D = 5 kept.
  Symmetric-polling (RED 1) 17.8-18.6; producer-WG cp.async sentinel polling (RED 3) 18.96-20.1 (polls slow the
  finishers' main loop); bulk-store publish (VG_PUBBULK) no better than register stores (publish ~1.3 us for 36 KB,
  ~2.5 us for 64 KB either way).
- **Same-run vs bo1 (bench.py, 60 reps, box 0): qkv m256 g256 15.95 vs bo1 16.61 (SGLang 18.69; x1.171 vs x1.125).**
  Second run (all four layers, vg configs): qkv 16.29 vs 16.84; o 7.96 vs 7.75 (loss); pooler 64x128 LAY1 23.27 vs
  23.39 (tie); unpooler 128x128 NS6 23.00 vs 23.05 (tie).
- nsys in-graph (box 0, flushed): g256 qkv kernel 16.78 us vs bo1 nvjet_sm90_tst_72x64_64x12_1x1 17.09 us.
- ncu (full set, flushed, qkv RED4 D5): 18.50 us, tensor pipe 58.0 % of elapsed (72.0 % of SM-active cycles), issue
  25.8 %, eligible 0.32 warps/clk, xbar->SM 212 MB (40.8 % of peak), DRAM read 42 MB (47.5 % of peak), L2 68 %;
  stalls long_scoreboard 26 % (mbarrier waits), wait 20 %, barrier 18 %.
- Timeline (globaltimer, D4): first data 1.28 us; helpers' loop ends 12.0-12.3, publish done 13.3-13.6; finishers'
  loop ends 13.3-13.8, partial landed 14.6, end 14.9-15.6. Main loop ~90 % of tensor peak per k-block; the losses are
  start (1.3 us), the D imbalance + exchange (~1.3 us) and the epilogue (~0.6 us).
- SASS (nvcc 12.1 -arch sm_90a on the NVRTC source): 4 HGMMA.64x72x16 per k-block per WG, one WARPGROUP.ARRIVE +
  one DEPBAR.LE gsb0,1 per k-block, no per-HGMMA serialisation; 0 spills; 168 regs (launch bound 384 x 1).
- Binding unit: none at its peak; tensor pipe 72 % of active cycles. Next: shrink the exchange / imbalance, faster
  start, and the other three shapes (all at a tie or behind with plain tiles).
- Deliverable v1: claude/t3/V/g256.py (qkv only), sub_g256.py; check.py --graph qkv ALL OK (52 cases + 12 near-256
  cases); exact_all qkv --nesc 200 EXACT ALL OK.

## Iteration 3: placement (GPC) facts, compile-time grid, TMA-store epilogue, NS 9 for qkv
- **Dispatch / GPC map (claude/t3/V/smid.py)**: non-cluster launches place 8 consecutive blockIdx on one 8-SM group
  and 16 consecutive on one GPC (bids 0-7 -> SMs 124-131, 8-15 -> {0,1,16,17,32,33,48,49}, 16-31 -> GPC
  {2,3,18,19,34,35,50,51,64,65,78,79,92,93,106,107}, ...). Cluster-2 launches place cluster c on TPC c-4 in SM
  order, i.e. consecutive clusters go to DIFFERENT GPCs. The per-GPC L2->GPC traffic is a binding resource:
  the same unpooler kernel with tiles mapped "column fastest" (one x tile, 16 W tiles per GPC) runs 32 us instead of
  22.6 (pooler 40-51 vs 23). That is also why every cluster launch was slow (dummy cluster: qkv 14.7 -> 19.3).
  With a prepare-time GPC-aware permutation (probe kernels record %smid per bid and cluster-16 GPC groups,
  cfg PERM=1) the cluster kernels recover: pooler S2 DSMEM 30.1 -> 24.0, qkv cluster S2 -> 16.3; still not better
  than RED4 / S1 (DSMEM moves ~25-30 KB/us/SM: 32 KB exchange 1.3-1.5 us after the loop). TMA multicast pairs with
  GPC-aware placement: still slower (unpooler 24.1 vs 22.6, o 9.1 vs 7.8, pooler 24.4-29 vs 22.9): the pair lockstep.
- Grid constants (MT, NH) compile-time: removes parameter loads before the first TMA.
- TMA-store epilogue (128B-swizzled stmatrix sub-tiles, cp.async.bulk.tensor store, wait .read only): o x0.973 ->
  x1.027 vs SGLang in separate runs, unpooler / pooler neutral. o still loses to bo1 in same-run benches.
- L2 cache-policy hints (W evict_first, x evict_last), tensormap prefetch: neutral.
- qkv RED4 NS 9 (= exactly 227 KB smem) D 6: same run vs bo1, 120 reps: box 1 15.55 vs 16.36, box 0 15.83 vs 16.72
  (NS 8 D 5: 16.13 / 15.98).
- nsys in-graph (box 0): g256 qkv vg_qkv_gate_128_72_0_2_9_0_4_6 16.61 us vs bo1 nvjet_sm90_tst_72x64_64x12_1x1 17.26.
- ncu (full set, flushed): 17.95 us; tensor pipe 55.7 % of elapsed, 75.0 % of SM-active cycles; issue 26.7 %; eligible
  0.33 warps/clk; xbar->SM 212 MB (39 % of peak); DRAM read 42 MB (48.6 % of peak); L2 70.9 %; 168 regs, 0 spills;
  stalls long_scoreboard 25 % (full-barrier waits), barrier 20 %, wait 20 %.
  Binding: the per-GPC L2 delivery during the main loop (GPC tile 256 tok x 576 cols per k-half, ~610 GB/s per GPC)
  plus fixed costs: start 1.3 us (first TMA data), D-imbalance + exchange ~1.2 us (helper SMs idle at the end:
  that is most of the elapsed-vs-active gap), epilogue ~0.5 us. Tensor at 75 % of active cycles = 90 % in the
  steady-state loop (k-block time 0.185 us vs 0.169 ideal at 1.7 GHz).
- Other shapes, same-run vs bo1 (120 / 100 reps, both boxes; vg S1 configs with TMA store):
  | shape | bo1 box0 / box1 | vg box0 / box1 | verdict |
  |---|---|---|---|
  | o 128x128 NS6 | 7.83 / 7.68 | 7.97 / 7.82 (no TMA store run: 7.56 vs 7.80) | loss / tie: not shipped |
  | pooler 64x128 LAY1 NS9 | 23.12 / 23.13 | 23.07-23.15 / 22.43-22.91 | tie: not shipped |
  | unpooler 128x128 NS6 | 22.98 / 22.60 | 22.95 / 22.57-22.62 | tie: not shipped |
  pooler / unpooler: the per-GPC sharing is already maximal for S=1 (GPC tile 256 x 512 / 256 x 1024); S=2 halves
  GPC traffic (S2 main loop 97 % of peak) but every exchange path (DSMEM 1.3-1.5 us, global 2.5 us for 64 KB) costs
  more than it saves. 132 SMs cannot be used without split-K (8-column wgmma granularity).

## Deliverable (current): claude/t3/V/g256.py (qkv_gate m = 256 only) + sub_g256.py
- prepare_g256(weights) -> state (NVRTC compile ~0.3 s + a self-test vs F.linear incl. a one-hot probe row; any
  failure disables the route); run_g256(x, config, state) -> out, or None for every other shape / m != 256.
- check.py claude/t3/V/sub_g256.py --layers qkv_gate,o,pooler,unpooler --graph: checked 207 cases x 2 seeds + graph:
  ALL OK. exact_all --layers qkv_gate --nesc 200: EXACT ALL OK.
- Graph safety: the finisher/helper flags and the fp32 exchange slots live in state; each finisher resets its flag
  after reading it, so replays need no host work. Helpers (bids < 64) never wait, finishers only wait on helpers.

## Final table (same run, bench.py, 120 reps, local us; judge ~= local + 1.0)
| shape | SGLang | bo1 (x vs SGLang) | g256 (x vs SGLang) | int_v7 (base_v7.log) |
|---|---|---|---|---|
| qkv m256, box 1 (sub_g256.py) | 18.46 | 16.69 (x1.106) | **15.45 (x1.195)** | x1.131 |
| qkv m256, box 0 (same kernel via sub_vgy) | 18.62 | 16.72 (x1.114) | **15.83 (x1.176)** | |
| qkv m256, box 1 (same kernel via sub_vgy) | 18.50 | 16.36 (x1.131) | **15.55 (x1.189)** | |
D sweep at NS 9 (same run, both boxes): D 5-8 all within 15.68-15.94 (noise); D 6 kept.
o / pooler / unpooler: ties or losses vs bo1 -> run_g256 returns None (bo1 keeps them).
Integration: dispatch `layer == 'qkv_gate' and m == 256` to run_g256 (it returns None for anything else anyway).
