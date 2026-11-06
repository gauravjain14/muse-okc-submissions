# K2 notes (M = 128 / 256, BF16 layers)

## Baselines (claude/bench.py local us)
| M | layer | SGLang | Codex v2 | target |
|---|---|---|---|---|
| 128 | qkv | 16.32 | 15.63 | <= 11 |
| 128 | o | 6.24 | 6.19 | <= 5.5 |
| 128 | pooler | 24.46 | 24.44 | <= 17 |
| 128 | unpooler | 19.22 | 17.89 | <= 17 |
| 256 | qkv | 18.62 | 19.84 | <= 13 |
| 256 | o | 8.24 | 8.33 | <= 6.5 |
| 256 | pooler | 29.41 | 27.98 | <= 22 |
| 256 | unpooler | 23.70 | 24.08 | <= 22 |

## Measured hardware facts (H200, Modal)
- L2 -> SM aggregate read bandwidth ~7.2 TB/s with all 132 SMs pulling (54 GB/s per SM); up to ~100 GB/s per SM
  when few SMs read. Bulk-copy MULTICAST in a 2-CTA cluster delivers 11.4 TB/s from 5.7 TB/s of L2 reads
  (claude/k2/l2bw.py). => x re-reads (one per N tile) are the main L2 cost; W from HBM also flows through L2.
- TMA bulk reduce-add (cp.reduce.async.bulk .add.f32) to an L2-resident fp32 workspace is SLOW: ~0.8 TB/s (v3).
- W L2 prefetch (cp.async.bulk.prefetch.L2) ahead of the smem ring HURTS at every distance (8..48 stages).
- Register-A wgmma (x loaded by LDG straight into A fragments, k-permuted W) is 2-3x slower: x prefetch depth
  in registers (~2 k-blocks) cannot cover L2 latency under load (~1-2 us) (v4, k2_v4.py).
- Separate deep W ring + shallow x ring (v3) is worse than one combined stage ring: x starves.
- Timeline (globaltimer per CTA, claude/k2/trace1.py), v2 kernel: first stage lands 1.5-2.3 us after entry;
  S=1 epilogue with direct 4-byte BF16 stores costs 1.3-3.6 us; split-K tail (partial write + counter + reads)
  costs 4-7 us. Fixed in v5 (smem-staged 16-B stores; single-round parallel partial reads + smem atomics).
- ncu (o m256, S=1): DRAM 25%, L2 50%, SM 30%: latency-bound pipeline, not throughput-bound.

## Versions
- v1 (k2_v1.py): TMA 2D boxes for W and x, global split-K. Correct, slow.
- v2 (k2_v2.py): W pre-tiled + pre-swizzled in prepare (one contiguous bulk copy per stage), x TMA 2D, optional
  cluster-2 x multicast. CL=2 initially 2x slower because the remote empty-barrier arrive used
  `.release.cluster`; plain `mbarrier.arrive.shared::cluster` fixes it (then CL=2 ~= CL=1, no gain).
- v3 (k2_v3.py): split rings + TMA reduce-add. Worse.
- v4 (k2_v4.py): register-A x. Much worse.
- v5 (k2.py): v2 + relaxed cluster arrive + explicit MT per config (M=256 may use two 128-token tiles) +
  staged epilogue + parallel split-K reduction + trace/pdist/flags knobs.

## Final K2 (k2.py v5) vs SGLang / Codex v2 (claude/bench.py, 30 reps, local us; run_o.log)
| M | layer | SGLang | K2 | Codex v2 | K2 config (MT, BN, S, CL) |
|---|---|---|---|---|---|
| 128 | qkv | 16.45 | 17.10 | 15.89 | (128, 128, 7, 1) |
| 128 | o | 6.38 | 7.70 | 6.32 | (128, 64, 1, 1) |
| 128 | pooler | 24.34 | 24.45 | 24.72 | (128, 64, 2, 1) |
| 128 | unpooler | 18.90 | 20.98 | 17.77 | (128, 64, 1, 1) |
| 256 | qkv | 18.78 | 22.76 | 19.62 | (256, 128, 7, 1) |
| 256 | o | 8.22 | 10.56 | 8.51 | (128, 128, 1, 1) |
| 256 | pooler | 28.70 | 32.40 | 28.46 | (256, 128, 4, 1) |
| 256 | unpooler | 23.84 | 29.06 | 23.96 | (128, 128, 1, 1) |
K2 loses to cuBLAS on every shape except a tie on pooler m128. Do NOT integrate K2 as is.

## Where the time goes (trace1.py, per-CTA globaltimer, after a cold flush)
- first TMA stage lands 1.5-2.6 us after entry (cold DRAM + descriptor/i-cache) on every config.
- split-K tail: partial write + __threadfence + counter wait = 2.5-4 us; partial reads + stores = 1.5-2.5 us
  (qkv m128: mainloop ends 12.0, exit 16.1; qkv m256: 14.9 -> 21.5; pooler m128 BN256 S8: 17.4 -> 24.1).
- S=1 mainloops are L2->SM bandwidth bound by x re-reads (traffic = W * M/MT + x * N/BN): e.g. unpooler m128
  BN=64 moves 1.5 MB per CTA (~75 GB/s/SM) in ~18 us.
- Things that did not help: TMA reduce-add (slow), W L2 prefetch (hurts), register-A x (latency), split rings,
  cluster-2 x multicast (break-even after fixing the arrive; never a win), smem-atomic reduction (very slow).

## What would be needed to beat cuBLAS here (not done)
- Hide the split-K reduction: on-chip reduction (cluster-pair DSMEM along K for S=2 configs) or a
  schedule where partials are produced mid-run and reduced by the idle producer-WG warps during the
  remaining mainloop; the end-of-kernel global reduction alone costs 4-7 us.
- Cut x re-reads further: bigger W tiles per CTA (BN=256) only pay if the reduction is hidden.
- For M=256 (compute-bound) cuBLAS is within ~10-20% of the bound; expected gain from a hand kernel is small.
