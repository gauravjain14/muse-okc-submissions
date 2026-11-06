# WC: small-call latency (ALIGNED, decode q_len 1)

Shapes: sliding b1 (w46), full b1, sliding b16, sliding b32. Base dev/k3.py (copied to dev/wc/k3_base.py).

## Log
- run_base (same job): k3 check 0 failures. Bench us (sglang | k2 | k3): sliding b1 20.05 | 10.58 | 12.78;
  b16 25.62 | 13.10 | 14.05; b32 30.18 | 17.10 | 17.81; full b1 30.46 | 17.42 | 16.55.
  k3 trace sliding b1 (us): seq+pt issued 2.62, q/new 4.31 (max 5.25), tile0 4.67, tiles 5.76 (max 7.46 = new-block CTA,
  lane 0 does 2 tiles), merged 6.90, publish 7.86, released 9.86, combined 12.69 -> global combine chain ~5 us.
  full b1: seq+pt 3.01, tile0 6.45, tiles 9.34, merged 10.70, publish 11.68, released 13.02, combined 15.82.
- SASS k3 attn_1_1_1_1: inv_freq LDG results spilled with STL right after the LDG (STL waits on the load: every warp
  stalls ~1 DRAM trip before issuing gates/Q loads). STACK 104 B. seq STS issued early (warp 0 stalls).
- barrier.cluster.arrive (release) and mbarrier.arrive.release.cluster lower to MEMBAR.ALL.CTA+MEMBAR.ALL.GPU;
  st.async (STAS) + mbarrier try_wait.acquire.cluster lower without MEMBAR -> use push via st.async + mbarrier tx.
- job1 (k3c v1, same job vs k3): sliding b1 k3 12.76 | c(cl8,pf) 11.06 | noclu 12.79 | cl16 10.03 | cl8 nopf 9.49.
  Prefetches (WC_PF) HURT ~1.5 us (they delay the first barrier) -> now opt-in. full b1 k3 16.70 | c 17.97 | noanch 18.91
  | nopf 16.80. b16/b32 (pf on) k3 14.12/17.71 | cl6/cl3 12.87/16.14 | cl4/cl2 13.39/18.47.
  Trace showed merge ~1.3 us and cluster combine 2.2 us: cold spill reloads (LDL after the tile loop miss L1:
  72 KB of stack per CTA > L1 left by 219 KB smem). Root cause: values live across the segment-loop back edge.
- v2 fixes: `if (ALIGNED) break;` at the end of the segment loop + post-loop values re-derived from opaque special-reg
  reads (tid_v/ctax_v/cl_id) and smem (misc[12] = complete, vsc from the per-warp smem copy), pb[] -> scalars,
  32-bit split division, host-passed hq: ALIGNED L=1 kernels STACK 0 (32 B = sincosf slow-path array with RoPE).
- job2 (v2, same job): sliding b1 k3 12.13 | cl8 8.31 | cl16 7.78 | noclu16 9.71 | cl16+pf 8.22.
  verify-1 sliding (L=8 old path) k3 20.88 | cl16 18.79 | noclu16 18.03.
  b16 k3 13.80 | noclu 12.23 | cl6 10.24 | cl5 10.10 | cl4 10.67;  b32 k3 17.49 | noclu 14.46 | cl3 13.11/12.89 | cl2 15.72.
  full b1 REGRESSED k3 16.48 | c 22.41: my shuffle-combine rewrite made the partial loads uncoalesced (32 partials per
  warp instruction -> 32 L1TEX wavefronts each): released->combined 8.8 us. Reverting to the coalesced mapping.
  trace cl16 sliding b1: seq 1.22, qprep 2.83, issued 3.27, tile0 3.88, tiles 4.66, merged 5.70, recv 6.32, combined 8.46.
- v3 (dev/wc/k3c_dev.py, k3-based dev copy; dev/wc/build_k4c.py builds the k4-based drop-in dev/wc/k3c.py with
  run8 = W8's wgmma path, plus a no-op `if (ALIGNED) break;` in run8 (stack 72 -> 32; -DWC_R8NOBRK restores k4)):
  * mO lane-merge rows padded to 80 f32 + rotated float4 store order: the old stores were 16-way bank conflicts.
  * cluster combine fully unrolled (runtime-bound loops over 16 partials were latency-bound: ~2 us for one warp).
  * global combine: k3's coalesced mapping restored, partial loads batched 8 per round, in-warp shuffle fold.
  * seq loaded by tid 511 (issues no other cp.async), so the first barrier waits for the length only.
  * plan: sliding q_len 1 picks the largest cluster size in (16,8,6,5,4,3,2) whose clusters are all co-resident
    (cuOccupancyMaxActiveClusters; H200 1 CTA/SM: 2->66, 3->39, 4->30, 5->22, 6->17, 8->15, 16->7):
    b1 -> 16, b16 -> 6, b32 -> 3.
  * RoPE table: 32 sincosf per token (one per frequency) into smem instead of 8 per thread (128x redundant).
  * (k3c_spec) anchored full-b1: first DST K/V stages issued BEFORE the length barrier from the chunk's page-table
    rows (rows bounds-checked against the cache size, length only masks; redo path if the real range is longer).
- job3b (k4-based k3c.py, same job vs dev/k4.py): check 0 failures (56/56).
  sliding b1 k4 12.90 | c 7.56 | no-RoPE-table 7.74 | cpr8 7.89;  b16 14.14 | 10.48;  b32 17.71 | 13.11.
  full b1 16.57 | 16.07.  verify-1 sliding 17.25 | 15.66 (run8 + `if (ALIGNED) break;` only; without it 17.21);
  verify-1 full 21.62 | 20.04 (same).  Loads: decode-1 x1.635 -> x2.288, verify-1 x1.384 -> x1.515.
  trace c sliding b1 (us): seq 1.28, qprep 2.68, issued 3.14, tile0 3.75, tiles 4.54, mO-written 5.20, clwait 5.30,
  merged+pushed 5.80, recv 6.12, combine-acc 7.11, combined 7.50. Small smem-only phases still cost 0.4-1.0 us:
  suspect cold instruction fetch (the judge's L2 flush evicts the kernel code; ~80 KB SASS) -> test --mode warm.
- Rebase: team base moved k4 -> k5 (WD's QGMMA run1<> for q_len 1). dev/wc/runa1.cu = run1 rewritten for the ALIGNED
  path (batch 1 + sliding) with all WC scheduling; dev/wc/build_k5c.py injects it into dev/k5.py -> dev/wc/k5c.py
  (FLAT keeps run1, q_len 8 keeps run8 + the no-op break). New in runA1: verified length speculation at batch 1
  (coordinator ruling): guess seq = max_pages*page_size - L, issue page-table loads + first K/V stages before the
  length barrier, confirm (same first position, npast <= speculated) or drain and re-issue. -DNOSPEC disables.
  Layout: one gate segment, misc[13] = length, recv buffer sized for cluster sizes in (16,8,6,5,4,3,2) (<= 5440 B).
- job4 (k5c v1 vs dev/k5.py, same job): check 0 failures (56/56). sliding b1 k5 9.64 | c 7.32 | nospec 7.44 |
  noclu 11.68; b16 10.57 | 10.23 | 10.11 | 12.58; b32 12.62 | 13.05 | 13.02 | 14.34. full b1 15.40 | 17.88 | 18.96.
  verify-1 sliding 17.51 | 16.18, verify-1 full 21.87 | 20.46 (run8 break). decode-1 load x2.000 -> x2.190.
  Diagnosis from traces: (a) speculative K/V issue BEFORE the length barrier made every warp wait for the
  page-table rows before arriving: barrier 1.18 -> 2.16 us, eating the gain; (b) issuing the prologue K/V after
  Q^T/RoPE/new-token (my k3-era reorder) delays the gathers ~0.9 us on the QGMMA core (k5's Q prep is cheap; the
  qprep phase itself takes 1.4 us cold) -> that is why noclu/b32/full b1 lost to k5. warm-mode trace: the small
  post-loop phases do not shrink (mO stores 0.6, combine accumulate 1.0 us) -> not cold code; ncu queued.
- v2 (job5): speculative page-table rows are plain loads before the barrier (nothing consumed), K/V issued right
  after the barrier (run1's order), drain path removed.
- job5 (k5c v2 vs dev/k5.py, same job): check 0 failures. sliding b1 k5 9.63 | c 6.94 | nospec 7.09 | noclu 10.79;
  b16 10.34 | 9.66 | 9.81 | 11.56; b32 12.61 | 12.61 | 12.66 | 13.74 (cl4: 13.87). full b1 15.21 | 15.88 | 16.52.
  verify-1 sliding 17.46 | 15.89, verify-1 full 21.39 | 19.87. decode-1 load x2.059 -> x2.391, verify-1 x1.400 -> x1.529.
  trace c sliding b1: seq 1.54, issued 2.48, Q/new done 3.02, tile0 3.28, tiles 4.15, mO 4.75, merged+pushed 5.30,
  recv 5.77, combine-acc 6.78, combined 7.18. Global combine (noclu b1: released 7.54 -> combined 10.69; full b1:
  12.28 -> 16.34) is 3-4 us. runA1's non-cluster path is ~1.2 us slower than run1 (noclu b1/b16, full b1 nospec):
  not located yet -> full b1 and the non-cluster fallback route to k5's run1 by default (WC_FULLA = runA1).
  ncu (sliding b1, 95 samples only): wait 22%, short_scoreboard 21% spread over the serial prologue/combine code.
- job6 (final default, same job vs dev/k5.py): check 0 failures (56/56; WC_FULLA full cases also 0 failures).
  decode-1 sliding k5 9.56 | k5c 7.13;  full b1 15.43 | 15.52 (run1, = k5) [runA1 opt-in WC_FULLA 16.11, +K3COMB
  15.96, nospec 16.66];  verify-1 sliding 16.88 | 16.01;  verify-1 full 22.16 | 19.76;  b16 10.51 | 9.95;
  b32 12.84 | 12.91.  Loads: decode-1 x2.037 -> x2.427, verify-1 x1.393 -> x1.497.
  Full b1 is gather-bound (k5 run1 trace: tile0 6.30, tiles 9.06, merged 10.02, publish 10.99, released 11.94,
  combined 14.83): starting the gathers ~1 us earlier (speculation) does not move the end (~2.9 TB/s effective);
  the 3 us global combine is the next lever there.
- job7c (k6 = k5m + W8 round-2 run8/CORE + runA1 parallel combine; same job, 40 reps): check 0 failures.
  GEOMEAN k6 x1.8996 | k6 serial-combine x1.9029 | k5m x1.7971 | k5c x1.8545 | k5w x1.7789.
  d1 sliding 6.63 | 6.79 | 8.50 | 6.77 | 9.24;  d16 sliding 10.07 | 9.71 | 9.52 | 10.00;  d32 sliding 12.97 | 12.61 |
  11.81 | 12.52;  v1 sliding 14.87 | 14.69 | 17.13 | 15.75 | 16.32;  v16 sliding 15.07 | 15.20 | 18.21 | 15.84 | 16.89;
  v32 sliding 18.20 | 18.55 | 22.00 | 19.32 | 19.93 (W8 run8 + the single-segment break). Full shapes = k5m's.
  => k5m's run1 (non-cluster, WD_EARLY) now beats the cluster path for b16/b32: clusters only for batch 1.
  Parallel cluster combine: combine-acc 1.0 -> 0.48 us (b1 trace), net -0.16 us on b1, +0.36 on b16/b32 (now unused).
- k6 v2: length cp.async + speculative page-table LDGs moved to the top of the kernel with a host-precomputed guess
  (glo, gn1): first pt LDG at SASS 0x0aa0 instead of 0x1860 (~200 instructions earlier per warp).
- job8 (k6 v2 = dev/wc/k6.py, md5 d544651d; same job, 40 reps): check 0 failures (56/56).
  GEOMEAN k6 x1.9108 | k6 v1 (job7c) x1.9071 | k5m x1.7989 | k5w x1.7766.
  per load: d1 x2.455 (k5m 2.118) | d16 x2.307 (2.308) | d32 x2.315 (2.307) | v1 x1.527 (1.351) | v16 x1.540 (1.459) |
  v32 x1.579 (1.525). d1 sliding 6.81 (k5m 8.88; 60-rep repeat 7.01 vs 9.15, nospec 7.22); d16/d32 sliding now on
  k5m's run1: 9.59 / 11.82. Top-loaded page-table rows: ~-0.06 us on b1 (noise level).
- NOBAR variant (dev/wc/k6_nb.py, -DWC_NOBAR): no CTA barrier in the prologue (each warp fetches the length into its
  own smem slot; RoPE table per lane behind the lane barrier; the new tile zeroed/written by warp 0 only).
- job9: WC_NOBAR passes the sliding + batch-1 checks; b1 sliding 2 x 60 reps: k6 6.96 / 6.86 | NOBAR 6.90 / 6.85 ->
  within noise, not adopted (kept as dev/wc/runa1_nobar.cu + build_k6_nobar.py). dev/wc/build_k6.py + runa1.cu
  reproduce the delivered dev/wc/k6.py byte for byte (inputs: k5m md5 b0a3a4a3, w8 k2w 47a698af, run8.cu 14f001fc).
- job10 (cold-code test, k6 sliding b1, 20 reps): post-tile tail (tiles done -> output) flush 2.93 us, warm (code and
  data in L2, same SMs) 2.99 us, noflush 2.95 us. Warm shifts the whole timeline 0.3 us earlier (length load) but the
  tail does not shrink -> runA1's tail is latency/sync-bound, not instruction-fetch-bound (unlike WD's run1 finding).
  Warm per phase: tiles->mO stores 0.61, cluster wait 0.10, lane merge+push 0.44, wait for the slowest push 0.55,
  combine 0.48, sum+epilogue 0.81. Tail SASS (k6 attn_1_1_1_1c): 0x9280..0xcbe0 = 14.7 KB, combine+epilogue 10.6 KB.
- k7c (base dev/k6.py = dev/wc/k6.py; changes only inside runA1 + its host fields). Jobs 11-14:
  * per-CTA trace (k6): the new-token CTA (part 15) was the straggler, ~0.4 us late at the merge: warp 0 did the
    new-token work and every lane waited for it at a CTA barrier. Now the lane that attends the new tile does it and
    only that lane syncs (bar_sync). k6 6.69/6.72 -> 6.65/6.65 (job12).
  * ncu over 40 launches (60K samples): barrier 21.6% (idle lanes at the post-tile barrier, length barrier), wait 17%,
    short_scoreboard 13% (the l-partial shuffles 6%), no_instructions 12.8%, divisions on the path ~5%.
  * host-precomputed per-part speculative ranges (gpos/gnp, cpr <= 16): the page-table loads issue ~130 instructions
    earlier per warp; no runtime division when the guess holds; cluster-16 fast paths (no divisions) in merge/combine.
  * staging the speculative rows by one warp's cp.async (WC_STAGE) was SLOWER (+0.13 us: its wait gates the barrier).
  * job14 (2 x 40 reps, same job): k6 6.79 / 6.84 | k7c 6.32 / 6.39 us (-0.46 us, -6.8%); check 0 failures (56/56).
    trace: issued 2.46 -> 2.18, tiles 4.10 -> 3.81, output 6.87 -> 6.55 us.
  Files: dev/wc/k7c.py = build_k7c.py + runa1_k7c.cu (reproducible); k6 = build_k6_job8.py + runa1_job8.cu.
- k9 (job15) = dev/wd/k9d.py + runA1 from k7c (dev/wc/build_k9.py anchors on k9d; runa1_k7c.cu + gpos/gnp host
  fields). check 0 failures. 2 x 30 reps, all loads: GEOMEAN k9 x2.1200 / x2.1180 | k8 x2.0906 / x2.0923 |
  k9d x2.1100 / x2.1051 | k7c x1.9322 / x1.9303. d1 sliding 6.22 / 6.20 (k9d 6.73 / 6.79, k8 6.85 / 6.74);
  decode-1 load x2.642 / x2.633 (k9d 2.539 / 2.523). Other shapes = k9d's code (noise).
- job16 (all checks 0 failures): sliding b1 2 x 40 reps: k9 6.26 / 6.17 | k10c (page-table rows before the length,
  queries next, inv_freq from kernel params) 6.47 / 6.41 | + new token folded into the last past tile (cl16)
  6.99 / 6.85 | fold + cluster 8 7.51 / 7.48. Both levers LOSE. k10c trace: Q/new done 3.23 us (k9 2.69): the Q prep
  got later (param-space inv_freq indexed by lane + load order); fold adds ~0.3-0.5 us more. Reverted to k9's runA1.
- job17: WC_TAIL2 on k10's runA1 (l over the 8 g-lanes for 4 rows with 4 shuffles instead of 12; final sum split over
  half-warps + one shuffle): check 0 failures; sliding b1 2 x 40 reps: k10 6.14 / 6.24 | TAIL2 6.10 / 6.08 us.
  trace: tiles->stores 0.60 -> 0.47, sum+epilogue 0.81 -> 0.71 us. Delivered as dev/wc/k11c.py (= dev/k10.py with
  runa1_k11.cu, TAIL2 default; SASS byte-identical to the tested k11t@WC_TAIL2). Builder: dev/wc/build_k9.py
  (runA1-only replace when the base already has gpos/gnp).
- job19 (b16/b32 sliding, run1 ALIGNED, base = dev/wd/k11d.py + k11c runA1 = dev/wc/k12base.py): designated per-request
  combiner (last part keeps its partial in smem, others release-increment and exit) passes the check but LOSES:
  b16 9.01 / 8.86 -> 9.46 / 9.34, b32 10.97 / 10.97 -> 11.26 / 11.20 (the last part is not reliably the last to finish;
  its combine of 7 partials starts later). runA1 clusters for B > 1 (WC_CLUB) lose more: 10.27 / 13.6.
  b16 trace (k12d): prologue issue 2.56, tile0 4.32, tiles done 5.44, merge stores 5.92, merged 6.32, seg end 6.34,
  combiner start 7.55, combined 8.97. Publish (atomic acq_rel incl. MEMBAR) ~1 us in k12base traces.
- job20: L2::evict_first on the K/V cp.async (WC_EVF, WF's finding) for runA1 + run1: check 0 failures (sliding);
  b1 sliding 5.88 / 5.88 -> 6.11 / 6.07 (worse), b16 8.77 / 8.81 -> 8.80 / 8.78, b32 10.75 / 10.62 -> 10.79 / 10.88.
  Not adopted for the sliding shapes.
- job21: run1 register lane merge (WC_RMERGE in dev/wc/k12e.py; lanes 1..3 -> smem float4 [3][4][128], one barrier,
  lane 0 merges in registers, l via 4-shuffle transpose + xb, partial written from registers): check 0 failures,
  but b16 8.79 / 8.88 -> 8.95 / 9.01, b32 10.66 / 10.63 -> 10.85 / 10.70 (full b1 14.1 -> 14.3). Trace: tiles ->
  merged 0.78 us (k12d 0.88), but the publish after it is no faster (~1 us). Not adopted.
- job22: WC_COMB8 (fast combine loads all 8 partials before merging; b16 one round instead of 2): b16 8.83 / 8.74 vs
  8.93 / 8.87, b32 10.93 / 11.06 vs 10.74 / 10.82 -> within noise. LL end chain v1 (dev/wc/patch_ll.py -> k12f.py
  @WC_LL; self-validating 8-byte words (flag << 32 | float) in the NaN-payload space, per-slot generations in
  counters[1280 + s], relaxed arrival, no fence): check 0 failures; trace: seg end -> arrived 0.31 us (was 0.96: the
  release fence costs ~0.65 us), but combine 3.3 us -> b16 10.3-11.4, b32 12.1-13.0. Cause: the poll re-polled one
  partial at a time (1 RT each) and the newest producers' stores are still in flight behind the K/V stream.
- jobs 23-25: LL end chain (dev/wc/patch_ll.py -> dev/wc/k12f.py @WC_LL, @WC_LL,WC_LLD = ticket-designated combiner,
  no end atomic): checks 0 failures, but b16 10.2-10.6 / b32 12.0-12.3 vs k12base 8.7-8.9 / 10.7-10.9 (2 x 40 reps,
  x3 jobs). Sub-stamps: the poll never re-polls (rounds = 1: every partial is visible at the first load), yet
  combiner start -> first loads back 1.2 us, -> poll done 1.5 us, -> combined 3.4 us; coalescing the LL word layout
  (job25) changed nothing. Diagnosis: ONE combiner CTA reading all partials is bound by its own L2 load throughput
  (~30-50 GB/s effective for these strong loads); LL doubles the bytes (8.4 KB per partial). The base's combine
  (33 KB, 2 rounds) is the same effect at half the bytes. Also: the gpu-scope release fence before the arrival costs
  ~0.65 us (seg end -> arrived 0.96 vs 0.31 us with a relaxed atomic). => the combine must be spread over SMs
  (K combiners / cluster DSMEM), not made cheaper per byte.
- k12base flush vs warm (job24): warm (data + code in L2) only moves the K/V phase (issued 4.35 -> 2.94 us at b16);
  tiles -> merged -> published -> combined spans are unchanged -> run1's tail is not cold-code bound either.
- job26 STRUCTURAL (coordinator): half-size CTAs, 2/SM, one cluster per request (dev/wc/build_k12h.py: runH1 = k12's
  runA1 with NL = 2 lanes / 256 threads / DST 2 / ~92 KB smem, __launch_bounds__(256, 2), kernels attn_1_1_*_1hc,
  host _wc_half opt-in WC_HALF, HCL = cluster size). Co-residency (cuOccupancyMaxActiveClusters, 256 thr, 84-113 KB:
  2 blocks/SM): cl2 132, cl4 62, cl6 39, cl8 30, cl12 16, cl16 14 clusters (512 thr / 1 per SM: cl4 30, cl8 15,
  cl16 7). => b32 cannot have cl8 (32 needed > 30) and b16 cannot have cl16 (16 > 14).
  Smoke check 0 failures. Bench (2 x 40): b16 cl8 (128 half CTAs) 9.53 / 9.38 vs k12base 8.80 / 8.77;
  b32 cl4 13.41 / 13.27 vs 10.64 / 10.59 (HCL=8 falls back to run1: 10.64 / 10.74). Trace b16 cl8: each lane runs 2
  tiles serially (tile0 3.98 -> tiles 6.02: +0.5 us vs run1), cluster tail tiles -> done 2.6 us (merge stores 0.28,
  cl wait 0.1, push 0.42, slowest push 0.52, weighting 0.64, sum + epilogue 0.65) = no shorter than run1's global
  tail (2.5 us). Not adopted.
  K-combiner split on run1 (comb_bytes 8192 / 4096 -> K > 1 general path): b16 9.90 / 9.85, b32 12.0 / 11.9 (+1 us).
- jobs 27-28 (dev/wc/patch_cbulk.py on k12base -> dev/wc/k12g.py): run1 FASTCOMB partial reads.
  * WC_CBULK (last arrival's thread 0 TMA-bulk-copies the P partials into the idle ring on one mbarrier after its
    acquire): b16 9.18 / 9.07, b32 11.07 / 10.93 (worse): the fence.proxy.async.global it needs delays the combiner
    start by ~0.76 us (published -> combiner start 0.16 -> 0.76).
  * WC_CPLAIN (weak L1-allocating loads instead of __ldcg / LDG.STRONG.GPU; correct: after thread 0's acq_rel atomic
    (CCTL.IVALL) and the CTA barrier) + WC_COMB8 (8 partials per round): b16 8.66 / 8.46 / 8.66 / 8.36 vs k12base
    8.93 / 8.88 / 8.92 / 8.79 (4 runs x 40 reps, -0.3 us = -3.5%); b32 10.83 / 10.85 / 10.98 / 10.74 vs
    10.84 / 10.82 / 10.85 / 11.15 (noise). COMB8 alone: b16 -0.1, b32 +0.2. CPLAIN alone (4 per round): b16 -0.1.
- job29 (dev/wc/patch_pro.py -> k12p.py): WC_PRO1 = run1 prologue issues K/V stage 0 of every lane before the RoPE /
  new-token / Q^T work, the other stages right after it. With CPLAIN + COMB8: b16 8.62 / 8.58 (pl8 8.66 / 8.57,
  k12base 8.80 / 8.72), b32 10.71 / 10.61 (pl8 10.81 / 10.85, k12base 10.81 / 10.83). b32 trace: stage 0 issued
  2.94 us (was 3.84-3.98), Q^T done 4.34 (4.62-4.70); the first tile wait is then data-bound (4.88 vs 4.99).
  COMBA (8 per round only when P > 4) = COMB8 within noise.
- k12c = dev/wc/build_k12c.py dev/k12.py dev/wc/k12c.py: CPLAIN + COMB8 in run1's FASTCOMB (default; WC_NOPLAIN
  opt-out) + PRO1 for sliding only (window != 0; WC_NOPRO1 opt-out; full b1 keeps its prologue). runA1 = k12's (WD).
- job30: k12c v1 (runtime pro1 = window != 0 -> attn_1_1_1_1 128 regs, +1 LDL) on dev/k12.py: full check 0 failures,
  but b16 9.04 / 8.97 vs k12 8.70 / 8.58, b32 11.12 / 11.16 vs 10.48 / 10.67 (and verify-32 sliding, byte-identical
  run8 SASS, 15.98 / 15.95 vs 15.56 / 15.50). job31 A/A (k12 loaded 3x + k12c v1, two orders): k12 copies agree to
  ~0.1 us at b16 and <= 0.3 us at b32 (verify-32 sliding 15.5-15.8 for identical code); k12c v1 is slower in every
  position (b32 +0.4-0.5) -> a real regression of v1 (register growth from the runtime branch), and the bench's
  per-module ws/counter placement noise is ~0.1-0.3 us on the global-combine shapes: A/B gains below ~0.2 us need
  repeats with A/A controls. v2: pro1 = ROPE (compile time): 117 regs, spills as k12.
- job32 (k12c v2 vs k12 x2 vs k12c@WC_NOPRO1): b16 8.73 / 8.57 (NOPRO1 8.63 / 8.44) vs k12 8.83-8.91; b32
  10.61 / 10.53 (NOPRO1 10.85 / 10.52) vs 10.56-10.95.
- job33 DELIVERABLE dev/wc/k12c.py (v2, md5 see report; = build_k12c.py on dev/k12.py b4135bdf): full check 0 failures.
  Same job, 2 x 40 reps, all loads, k12 loaded twice (A/A): decode 16 sliding k12 8.93 / 8.83, k12_2 8.83 / 8.79,
  k12c 8.49 / 8.50 (-0.35 us, -4%); decode 32 sliding 10.65 / 10.60, 10.76 / 10.78, k12c 10.65 / 10.58 (0 .. -0.15);
  every other load within the A/A spread. GEOMEAN k12 x2.2518 / x2.2450, k12_2 x2.2507 / x2.2529, k12c x2.2531 / x2.2535.
- Coordinator: k12c integrated into dev/k14.py (= k13 + WC run1 changes + WF's k14f), 56/56, d16 sliding -0.3 us.
  Ruling: B > 1 page-table-window L2 prefetch DECLINED (window position at B > 1 only predictable from the benchmark's
  agentic spread = workload tuning; docs/PLAN.md). WC stood down.
