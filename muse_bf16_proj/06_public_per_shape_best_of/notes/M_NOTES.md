# Agent M — m128 pooler / unpooler (RG2 x split-K, TMA x + bulk/LDGSTS W, RS-wgmma) — 2026-11-02

All times local us (graph replay after a 512 MB flush, empty graph subtracted; judge ~= local + 1.0). "Same run" =
one process, interleaved reps. Run-to-run (exec-to-exec) drift on the sandbox is 2-5 %, so compare within a run.

## Status (final)
- m128.py (clean, self-contained, inline NVRTC Module): `prepare_m128(weights)` / `run_m128(x, config, state)`;
  handles unpooler 97 <= M <= 128 only (returns None otherwise and if the 64 clusters of 2 cannot be co-resident).
  sub_m128.py = int_v7 + m128. prepare ~3 s.
- Correctness: `check.py claude/t3/M/sub_m128.py --layers unpooler,pooler --graph` ->
  "checked 104 cases x 2 seeds + graph: ALL OK" (rel. L2 ~1e-4, probe rows exact; M = 128 / 127 / 97 checked).
- Final same-run table (bench.py, local us, 60 / 80 reps):
  | shape | SGLang | m128 (x vs SGLang) | int_v7 (x vs SGLang) |
  |---|---|---|---|
  | unpooler m128 run 1 | 18.94 | **17.39 (x1.090)** | 17.83 (x1.063) |
  | unpooler m128 run 2 | 18.91 | **17.39 (x1.088)** | 17.89 (x1.057) |
  | pooler m128 | 25.28 | = int_v7 (not handled) 18.96 | 18.95 (x1.334) |
  nsys in-graph span: m128 18.14 | int_v7 18.19 | SGLang 19.95. ncu (flushed, full set): 19.42 us, tensor pipe
  47 % of elapsed, DRAM 52.4 MB at 2.70 TB/s, L2 65 %, issue 23 %, stalls long_scoreboard 29 % (try_wait), wait 25 %.
- Target (pooler/unpooler <= 14.0) NOT met. Unpooler gains ~0.45-0.5 us (+2.5-3 % on the shape) over int_v7.
- Breakdown (trace, median from first CTA start): first W ~1.5-1.8 | loop end ~15.1 (32 chunks, ~13.4 us; 87 % of
  tensor peak without traffic, ~70 % with) | partner data landed +1.3 (staging 0.3 + DSMEM 32 KB ~1.0) | end +0.5.

## Probe (mprobe.py, data movement only, 128 CTAs, no compute) — the requested comparison
Unpooler geometry RG2 x S2 (per CTA: x 512 KB via TMA, W 394 KB), pooler geometry RG2 x S4 (same per-CTA bytes):
| variant | time | first data / end (median) |
|---|---|---|
| x TMA + W bulk (both on the TMA engine) | 13.07-13.10 | 2.05 / 12.80 |
| x TMA + W LDGSTS, 7 loader warps | 13.07-13.09 | 1.79 / 13.06 |
| x TMA + W LDGSTS, 3 / 1 loader warps | 13.18 / 13.42 | |
| x only (512 KB/SM via TMA) | **5.31** | 1.02 / 5.38 |
| W only bulk / LDGSTS-7 / LDGSTS-3 | 12.66-12.72 / 12.67 / 12.72 | |
| W only, deeper W ring 10 / 12 / 14 slots | 12.94 / 13.04 / 13.07 (worse) | |
| RG1 (W's geometry: x 1 MB/SM): x only / x+W bulk / x+W LDGSTS | 13.41 / 15.41 / 15.82 | |
| pooler geometry x+W bulk / LDGSTS | 14.03 / 14.02 | |
=> Per-SM TMA ingress is NOT the limit at RG2/S2: x alone drains 512 KB/SM in ~4.3 us (~120 GB/s/SM), x+W 906 KB/SM
in ~11 us. Moving W to LDGSTS changes nothing (13.07 vs 13.10). The W HBM stream is the limit (~4.5 TB/s after the
first byte; deeper rings make it slightly worse). RG1 (1 MB x per SM) IS x-limited (x alone 13.4) -> halving x per SM
(RG2) is worth ~2.3 us of data time. k-order rotation: worse (14.0 vs 13.1). Progressive x prefetch (TMA prefetch):
disastrous (+6 us in the real kernel). Evict-first W / evict-last x: neutral.

## Real kernel: what mattered (unpooler, same-run deltas)
1. **Consumer loop structure (biggest lever, ~-3 us)**. With zero memory traffic the first loop ran 24-25K clk for
   32 chunks (16.4K ideal, 65-70 %). Diagnosis (ncu stalls "wait" 30 % on decode ALU chains; flags that skip decode
   or skip barrier waits each save ~3.5K clk; CUDA-13 SASS): the decode sat on the critical path between
   WARPGROUP.DEPBAR and the next HGMMA because try_wait loops / branches split the basic block, so ptxas could not
   hoist it as it does in W's wgb2 (which reproduces 98-100 % on this box, also at 32 iterations). Things that did NOT
   help: ping-pong named barriers between the 2 consumer WGs, nanosleep/spin producer waits, constant B / W slots.
   Two traps: (a) writing ANY future wgmma input register between wgmma.fence and the wait_group (decode into a 3rd
   A buffer between HGMMAs, or PRMTs right after commit) makes ptxas 13 serialize every HGMMA (C7513, each HGMMA
   gets its own ARRIVE/gsb0 + a dummy HGMMA.64x8x16); (b) a rarely-taken raw-unit (escape) fix-up block in the loop
   costs 1-2 us just by being there.
   Final loop (LOOP7): per chunk u: [x(u) waited one step early] fence, 4 HGMMA of u with the exponent bytes of chunk
   u+1 computed between them (temporaries only), commit | escape fix-up of u+1 (warp-uniform branch, rare) |
   release W(u+1) slot | LDS chunk u+2 (2 raw buffers) | x(u+1) try_wait | wait<1> | 16 PRMTs -> A buffer of u+1 |
   release x(u-1). ~60-100 SASS/chunk, 168 regs, 0 spills. Same run: 17.02 vs 18.61 (LOOP5) vs 18.85 (LOOP3)
   (no-epilogue timings); x-wait-early -0.7 us. No-traffic loop ~19.8-21.8K clk (87 % steady state, like W's).
2. **Escapes without a loop branch**: raw units now store their true low bytes in the lo array, the high bytes of
   weights 0-3 in the code slot and of weights 4-7 in the overflow list, so they decode through the same PRMTs; only
   hA/hB are overwritten in a rare warp-uniform branch (LDS of the overflow word). Correct (ALL OK).
3. **Split-K reduction**: fp32 slots (st.global.cg) + PDL converter cost +3.3-3.8 us (unpooler) / +3.0 (pooler) over
   no reduction: the 64 KB/CTA partial stores alone take ~1 us at the end (8 MB at once), an empty PDL converter
   +1.0 us, its 8 MB of reads ~2 us. Plain launch instead of PDL: 21.9 vs 22.4.
   Replaced by a **cluster (2 CTAs = the 2 k-halves) DSMEM reduce-scatter over token halves**: stage the partner's
   token half (32 KB fp32) in the freed x ring, one cp.async.bulk smem->DSMEM copy with complete_tx on the partner's
   barrier (dedicated 32 KB receive buffer), add, BF16 via stmatrix.trans, 16 B stores. No global partials, fences,
   tickets or second kernel. -3.5 us vs the converter (17.34-17.66 vs 20.96-21.2 same run).
   DSMEM moves only ~32 KB/us per SM: splitting the copy into 4/16/64 bulk copies, st.async from registers
   (17.57 vs 17.39) or a pull with ld.shared::cluster (+fence.acq_rel.cluster, 19.3-19.6) are all no better.
4. Ring split (smem 227 KB incl. 32 KB receive buffer): x ring depth is critical, W depth beyond 5-6 hurts
   (more W in flight delays x): (NSX,NSW) 6,6 17.44 | 7,5 17.50 | 5,7 17.98 | 5,8 18.62 | 4,9 20.3 | 3,10 27.8 |
   8,4 17.79. x L2 prefetch at start, evict hints, k-order rotation (+0.6-1.1): no help.
5. W via LDGSTS (8 loader warps -> 512 threads): ptxas caps the kernel at 128 regs (launch bound), the consumer is
   starved; 22.7 vs 21.3 (bulk). With 384 threads (bulk W) the consumer gets 168.

6. Unroll the loop by 6 so every ring slot / phase is a constant (needs NSX = NSW = 6): 17.25 vs 17.57 same run.
   Issue the overflow lists and the first 6 x stages / W slots from thread 0 right after the barrier init,
   interleaved x0 W0 x1 W1 ...: 17.14 vs 17.46; all x first then all W: 17.65; all W first: +0.6 (x behind W).
7. No help (same run, current kernel): W slot released right after its LDS (17.62 vs 17.52), producer nanosleep
   backoff (17.50/17.70 vs 17.39), W via 3 cp.async loader warps of the producer WG keeping 384 threads
   (18.11 vs 17.44; x still needs >= 6 stages with x alone on the TMA engine: 4 stages -> 20.35).

## pooler (S=4)
- Cluster of 4 (the 4 k-quarters): cuOccupancyMaxActiveClusters = 30 < 32 needed (GPC sizes; 1 CTA per SM) ->
  would run a second wave; m128.py checks this at prepare and does not use it.
- fp32 slots + PDL converter: 20.7 vs int_v7 18.5-18.8 (same run) -> loses.
- Cluster-2 pairs + a second global reduction level: projected ~18.5-19 (loop ~15.4 + DSMEM 1.3-2 + global 1+),
  not better than int_v7 -> not built.

## Not done / why (other layers, M = 256)
- pooler: see above (cluster-4 infeasible, converter loses, 2-level projected no win).
- qkv_gate (N 2304): 18 row blocks of 128 -> S = 7 does not divide 128 chunks, S = 8 -> 144 CTAs (2 waves),
  S = 4 -> 72 CTAs at the unpooler per-CTA time (~17 us) vs int_v7 14.75. o (K 1024): 8 chunks per CTA, the ~1.5 us
  start + ~2 us split-K epilogue dominate (projected ~7 vs int_v7 6.4). M = 256: twice the tensor work per CTA
  (n256 needs 128 accumulator registers -> > 168) -> projected >= 25 vs int_v7 23.2. Not built.
- Untested ideas (expected small): receive buffer overlaid on the x ring with a ready handshake (+32 KB of ring,
  costs the pair skew ~0.26 us); a code layout with 6 instead of 8 ALU ops per unit for the exponent bytes; 11-bit
  format (-4.5 % W bytes).

## Files
m128.py (clean deliverable), m128_dev.py (experimental superset: LOOP1-7 / U6, EPI 0 converter, EPI 1-3, WMODE 0-2,
diagnostic flags, profiling; used by tm.py / sass.py / sub_env.py), sub_m128.py (candidate),
tm.py (same-run sweep: correctness + timing + globaltimer trace / per-warp wait profile, per-spec #defines),
mprobe.py (data-movement probe), sass.py (NVRTC-13 SASS / ptxas facts without a GPU), sub_env.py (ncu candidate),
wgb2m.py (W's wgmma bench with an iteration-count knob), ncusrc.py / ncuq.py (copies of W's ncu helpers).
