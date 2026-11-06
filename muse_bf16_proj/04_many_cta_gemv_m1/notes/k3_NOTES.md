# K3 notes: M = 128 / 256 projections (qkv_gate, o, pooler, unpooler) on H200

(Draft; final table at the end.) All times: claude/bench.py-style local us (CUDA graph replay after a 512 MB read
flush, empty-graph cost subtracted). Modal containers differ by ~3-5 % run to run; compare only same-run numbers.

## Kernel (claude/k3/k3.py)
- Swap-AB wgmma: A = W rows (64 per warpgroup), B = x^T (TN = 128 tokens, 128B-swizzled TMA 2D boxes), FP32 acc.
- 384 threads: producer warpgroup (warp 0 lane 0: W ring; warp 1 lane 0: x ring; separate rings/barriers),
  two consumer warpgroups (setmaxnreg 40 / 232).
- Weights: 12-bit lossless (K1s/K2c format: low byte + 4-bit code, raw 16x16 units ~3 %, overflow list in smem),
  decoded by the consumer warps straight into wgmma A registers (RS wgmma), double-buffered; or BF16 (SS wgmma).
- Tiling (TN, RG, TG, KG, S, COMP, NSW, NSX, CK): RG row groups x TG token groups x KG k-chunk groups = 2 warpgroups;
  S = K split across a cluster pair (DSMEM reduce-scatter), CK = 64-k chunks per ring stage.
- Epilogue: fragment-order exchanges (thread ct of every warpgroup/CTA owns the same (row, token) slots, so partials
  move as float4 per thread with no transposition): KG intra-CTA exchange through smem, cluster exchange with
  st.async + mbarrier complete_tx into the peer's W ring (after a remote 'rdy' arrive), then stmatrix.trans to a
  BF16 [token][row] staging tile and 16 B coalesced stores.
- M > 128: NTB token blocks of 128 (CTAs of a row tile share W through L2).

## Measured facts (new in this workstream)
- cuBLASLt kernels at M=128/256 (ncu, claude/k3/k3_ncu1.log): no split-K, 128 CTAs, tiles 128x32 / 128x64 / 48x64 /
  72x64 / 256x64, cluster 1x2 for some. They move 200-400 MB through L2->SM (l1tex__m_xbar2l1tex_read_bytes) in
  17-26 us, i.e. 14-18 TB/s aggregate, ~150 GB/s per SM. So the earlier "L2->SM ingress ~54-70 GB/s per SM / 7.2 TB/s"
  (claude/k2) is a latency artefact of those kernels, not a hardware limit. Tensor pipe active 29-71 %.
- Data-movement probes without compute (claude/k3/mv.py): pure 12-bit W stream (50 MB, 128 CTAs) 11.6-12.0 us,
  BF16 67 MB 15.0-15.5 us; per-CTA x-only (2 MB L2-hot) ~190 GB/s/SM with 13 x 16 KB stages; first data lands
  1.5-2.1 us after launch. Adding the x stream of a design costs 0.5-2.5 us over W alone
  (pool S2 R64 12b 14.2, S4 R128 13.15, S8 R256 12.45, unp S2 R128 12.4, o S2 R128 3.6, o S1 R64 4.5 us).
  x multicast in cluster pairs: no gain. L2 prefetch of W ahead of the ring: worse.
- DSMEM (claude/k3/dsm.py, all 132 SMs, cluster pairs): ~32 GB/s per SM for every method (st.async v4,
  cp.async.bulk smem->peer smem, st.shared::cluster + barrier): 16 KB 0.5 us, 32 KB 1.0 us, 64 KB 1.8-2 us.
  Global exchange (st.cg + threadfence + red.release flag + ld.acquire poll + ld.cg): 1.8 us for 8 KB, 3.3 us for 32 KB.
- ptxas pitfall: an `if (n > 0)` guard around the RS wgmma loop plus accumulator reads in the epilogue made ptxas
  serialize every HGMMA (WARPGROUP.DEPBAR after each; warning C7514). Fix: no guard (n >= 1 always) and an empty
  asm "+f" fence over the accumulators after each wgmma wait. Check SASS for HGMMA/DEPBAR counts.
- Consumer cost per stage (clock64 buckets, unpooler RG2 S2, 64-k stages): mbarrier wait W ~200, decode ~330,
  issue (fence + 4 RS wgmma) ~155, release/loop ~180 cycles; i.e. ~0.5 us per 64-k chunk against 0.29 us of tensor
  work -> consumer-bound. Two 64-k chunks per stage (CK=2) halves the per-chunk boundary cost: loop 16.9 -> 14.4 us.
- Arrival tracing (observer warps on the full barriers): with CK=1 rings the W ring ran just in time while x arrived
  ~2 us early (x latency ~0.6 us, W ~1.1 us): the earlier x-latency theory was wrong, the consumer was the limit.
- Epilogue phases cost 0.3-0.5 us each (smem round trips at ~128 B/clk + barriers): KG exchange (64 KB of smem
  traffic) ~0.5 us, DSMEM 16-32 KB 0.5-1.3 us incl. the peer skew, stmatrix + copy ~0.3 us.
- 12-bit decode in a separate decoder warpgroup (claude/k3/k3d.py: W12 ring -> decoder WG -> swizzled BF16 A ring ->
  SS wgmma): 1.5-2x slower (three-stage handoff latency through mbarriers). Dropped.
- L2 cache hints: W evict_first -0.3..0.4 us; x evict_last nothing.
