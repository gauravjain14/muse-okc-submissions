# XE: dedicated batch-1 sliding kernels (d1s q_len 1, v1s q_len 8) — round 2

Owner: XE. Base dev/k34.py (d1s = runA1h `attn_1_1_1_1s`, v1s = runA8h `attn_8_1_1_1h`). Write only in dev/xe/.

## Files
- `xe_mod.py`  the XE module (own NVRTC source `XE_SRC`, kernels `xe_b1s_1` / `xe_b1s_8`, host `_xe_kernel_fn`,
  predicate `_xe_ok`, kernel_fn wrapper). Appended to any base by `build.py`.
- `build.py <base.py> <out.py> [XE_X=V ...]` -> `k34xe.py` (= dev/k34.py + xe_mod.py).
- `t_xe.py <file> [--quick]` correctness: B=1 page-1 sliding cases (q_len 1/8, ctx 1..80000, window edges,
  dist edges/normal/coherent, kv_scale 1/64, speculation misses via spread mixed/agentic), 3 calls each, asserts XE ran.
- `trace_xe.py` globaltimer phase trace (K2_DEFS XE_TRACE); `probe.py` floor probe; `nsys_run.py` / `nsys_parse.py`
  (nsys of dev/wp/nsys_loop.py: mention dev/wp in --cmd!); `lc.py` local NVRTC 12.4 compile + SASS (no GPU);
  `job*.sh` Modal jobs.

## k34 critical path (job1, out/xe/job1.txt; flush traces, us from the first CTA start, median over CTAs)
d1s runA1h: prefix loads issued 0.47, page-table issued 0.82, length barrier 1.09, K/V issued 1.25, tile0 (K/V
landed) 1.86, tiles done 2.37, merge stores 2.43, pushed 2.62, owner received 2.83, out 3.18 (last stamp 3.20).
v1s runA8h: prefix loads 0.84, length barrier 1.41, K/V issued 1.57, tile0 2.17, tiles done 2.96-3.09 (max 3.5),
merged 3.42, pushed 3.99, received 4.49, out 4.92 (last 5.02).
Bench (same job, 40 reps): d1s k34a 3.04 / k34b 3.11; v1s 4.92 / 4.94 (SGLang 20.0 / 22.5).
Budget reading: k34 issues its first load ~0.5 us (d1s) / 0.85 us (v1s) after CTA start (cold prologue + constant
misses), the K/V trip lands ~0.6 us after issue, the tile phase is 0.5 us (d1s, wgmma + fences) / 0.8-0.9 us (v1s),
and the tail tiles-done -> out is 0.8 us (d1s) / 1.9 us (v1s).

## Floor probe (dev/xe/probe.py, job1): the skeleton of the XE design on the real d1s inputs
Graph time (bench-style, empty graph subtracted) and stamps (min/med/max us from the first CTA start):
- empty kernel, cluster 16 x 2 or not: -0.32 (cheaper than the empty graph's torch add_; no cluster launch cost).
- L4 (seq + guessed page-table entries at entry, K 16 B + V 8 B register LDGs per thread per key, 4 warps x 32 keys,
  32 CTAs in 2 clusters of 16): graph 1.31; K/V landed 1.12/1.26/1.41. Non-cluster launch 1.12; 8 warps x 16 keys 1.28.
- C4 (= L4 + 1-barrier CTA merge of a dummy 8 x 68 partial + st.async push to 8 owners + owner wait + store): graph 1.79;
  merge barrier 1.41, pushes issued 1.63, owner received 1.79, stored 1.88. C8 (8 warps) 1.81; 4 clusters (v1s grid) 1.89.
=> start + page table + K/V + merge + push + owner store floor ~1.8 us vs k34 d1s 3.04: the room is the cold
prologue (~0.5 us), the tile phase (~0.5) and the tail. Compute + Q prep + owner merge must fit in ~0.4-0.6 us to
reach d1s <= 2.4.

## Design v1 (xe_mod.py)
- d1s: grid 16 x 2 (cluster 16 per KV head), 160 threads (4 compute warps + 1 owner warp); v1s: grid 16 x 4 (cluster
  16 per (KV head, row group of 4 query tokens) = 32 rows per cluster), 192 threads (4 compute + 2 owner warps).
  Dynamic smem padded to 120000 B (XE_SMEM) so each CTA has its own SM.
- Keys: CTA rank r owns [r C, r C + C), C = ceil(past / 16) <= 128; warp w the keys [32 w, 32 w + 32) of them, two
  16-key blocks. Speculation: guessed seq = width - L, page-table entry per lane at entry, K/V issued when it lands,
  confirmed by the seq load (miss: true range, reload, re-issue; the same code in a 2-pass loop).
- Math: FP16 HMMA m16n8k16, FP32 accumulation. S^T = K Q^T (M = 16 keys, N = 8 rows, d permuted: thread q holds
  d 16q..16q+15 -> one 16-B LDG per key). P = 2^(s c - m_warp) in FP16, P^T B-fragments by movmatrix.trans.
  O^T = V^T P^T (M = 16 d: thread g holds d 8g..8g+7 -> one 8-B LDG per key; V^T pairs built by prmt + cvt), row
  sums by an all-ones A tile. Q: RMSNorm + RoPE (smem table of sincosf, same fp32 angle as the reference) in FP16.
- CTA merge: warp maxima + O^T fragments + sums to smem, one named barrier, the reader rescales/sums and st.asyncs
  the CTA partial (O 64 floats + (m, l)) to the row owner (d1s: CTA r < 8 owns q-head 8h + r; v1s: CTA r owns rows
  2r, 2r+1). Owner warp: q row + the new token(s) prepared by the compute warps (RMSNorm + RoPE + E4M3 into smem;
  rank 0 writes the cache rows) attended as an extra partial (causal for v1s), waits on the mbarrier, merges 16
  partials, RMSNorm x sigmoid(gate), bf16 store. No final cluster barrier.
- SASS (local NVRTC 12.4): xe_b1s_1 1264 instrs / 95 regs, xe_b1s_8 2400 / 147 regs, no spills (32-B stack =
  sincosf slow path).
- ptxas lesson: `.nc` loads are freely sunk past bar.sync and even past the mbarrier try-wait (the owner's gate load
  landed after SYNCS.PHASECHK in the first build = a DRAM trip on the critical path). Values needed after a
  barrier are stashed in smem before it.

## Log
- job1 (out/xe/job1.txt): probe + k34 traces + bench (above). nsys failed (dev/wp not uploaded: mention it in --cmd).
- job2 (v1, compute warps also did RoPE table + new tokens + Q prep): t_xe quick/full + check.py 0 failures (worst
  rel-L2 6.9% = k34's level vs the bf16 reference; bound 15%). Bench d1s 3.35 vs k34 3.12 / 3.08, v1s 5.49 vs 4.93 / 4.93.
  Trace d1s: K/V issued 0.90, Q prep done 2.10 (!), compute 2.17 -> 2.53, pushed 2.82, recv 3.07, out 3.39. The
  ~355-instruction straight-line prep chain took 1.2 us (warm L2 0.8) while the K/V trip is ~0.6 us. nsys
  (dev/xe/nsys_run.py): xe d1s dur 3.94 / k34 3.84 us, v1s 6.11 / 5.57 (flush-end -> start gap 4.29-4.48 both).
- v2: prep warps (L 1: warp 4, L 8: warps 4-7) compute the RoPE rows, the Q^T fragments (to smem), the new tokens
  and own the rows; the compute warps only issue K/V, cl_wait, bar.sync on 'Q ready', LDS the fragments. Owner q =
  the shared FP16 fragment. Tree sums (template recursion: `st >>= 1` loops did not unroll -> 256 B of stack).
  job3: 0 failures (t_xe, check). Bench (2 orders): d1s xe 2.79 / 2.83 vs k34 2.87, 2.90 / 2.93, 2.90;
  v1s xe 3.88 / 3.96 vs k34 4.71, 4.63 / 4.81, 4.70 (-0.8 us). Trace d1s: Q ready 0.83 (no longer on the path),
  K landed 1.54, P 1.76, PV 1.85, merge barrier 1.90, pushed 2.21, recv 2.43, out 2.78.
  v1s: K landed 1.69, P 2.24 (QK + softmax 0.55), PV 2.46, merge 2.69, pushed 3.65 (0.96 for 4 rows / thread!), out 3.90.
- job4: push variants (2 benches, 6-way, same job). d1s: k34 2.92 / 2.93, 2.90 / 2.91 | st.async 2.63 / 2.69 |
  st.async FP16-O 2.62 / 2.64 | bulk (staged + cp.async.bulk per owner) 2.79 / 2.84 | bulk FP16 2.81 / 2.82.
  v1s: k34 4.71 / 4.70, 4.63 / 4.63 | st 3.80 / 3.85 | st FP16 3.85 / 3.92 | bulk 4.45 / 4.42 | bulk FP16 4.31 / 4.40.
  => XE_PUSH=0 (st.async) default, FP16-O neutral (off). Bulk loses: staging STS + fence.proxy.async + barrier
  0.3 us (d1s) / 0.7 us (v1s). All four variants 0 failures (t_xe quick), check.py batch-1 cases 0 failures.
  Trace st FP16 d1s: PV 1.90, merge barrier 1.99, reader y ready 2.22 (0.23 for ~12 LDS + 4 ex2!), pushed 2.30,
  recv 2.46, owner LDS done 2.56, sums 2.75, out 2.91: the reader and the owner merge cost far more than their
  dependent latency -> job5 tests cold i-fetch by executing them twice (XE_DIAG).
- job5 (XE_DIAG: push reader and owner merge executed twice, pass 0 without side effects; d1s flush, us): reader
  pass 0 0.28 / pass 1 0.21 -> NOT i-fetch; owner first LDS block after the mbarrier 0.27 vs 0.02 in pass 1, the
  merge math 0.03 both, epilogue 0.13 both. Bench: xe 2.80 / k34 3.07, 2.97; v1s 4.03 / 4.84, 4.84.
- job6 (clock64 deltas, dev/xe/trace_xe.py "cycles" lines): reader M -> y 390 cycles (pass 1 282) = smem
  throughput: its LDS.128 of the warp O^T fragments was 4-way bank-conflicted; owner waited -> LDS 574 cycles
  cold vs 164 hot; owner idles ~2400 cycles in the mbarrier wait; 5-shuffle RMSNorm sum 176; epilogue 170.
- v4: conflict-free reader (reader thread t reads exactly the slot writer lane t & 31 wrote at j = t >> 5) and
  owner pre-warm (XE_PREW: the merge loop runs once on the receive buffer before the mbarrier wait, same code
  addresses, store only in pass 1). job7: t_xe quick (PREW 0/1), t_xe full, check.py: 0 failures.
  Bench (2 reps): d1s xe 2.63 / 2.64, no-prewarm 2.75 / 2.77, k34 3.01, 2.98 / 3.00, 2.94;
  v1s xe 3.96 / 3.94, no-prewarm 3.85 / 3.85, k34 4.84, 4.72 / 4.73, 4.70 -> pre-warm only for q_len 1 (v1s's two
  pre-warming owner warps compete with the compute warps). Trace d1s: reader 0.13 us (was 0.23).
- v5: XE_PREW=1 (q_len 1 only), XE_QBF=1: Q fragments = f16(bf16(qk_scale * rope(rms(q)))) = the reference's own
  query rounding (our 6.9% error vs the bf16 reference was the reference's query-rounding sensitivity, as k34's).
- job8 (v5 with XE_QBF=1): t_xe 2 seeds 0 failures (worst err/bound 0.008; QBF=0 0.048), check.py 0 failures.
  Bench 2 reps: d1s xe 2.55 / 2.66, QBF=0 2.54 / 2.60, k34 2.96, 2.95 / 3.07, 2.94; v1s xe 3.91 / 3.86, QBF=0
  3.76 / 3.79, k34 4.70, 4.73 / 4.71, 4.77 -> XE_QBF=0 default (precision not needed: err <= 7.2e-3 vs bound 0.15).
  nsys kernel dur: d1s 3.39 vs k34 3.87, v1s 4.77 vs 5.60.
- job9 = FINAL default (XE_PUSH 0, XE_H16 0, XE_PREW 1 = q_len 1 only, XE_QBF 0): check.py 0 failures (56/56),
  t_xe 0 failures (worst err/bound 0.048: ctx 2050 dist normal err 7.2e-3), errors 3-7e-3 (k34: 6.9e-2).
  Bench (40 reps, 2 reps, k34 controls in the same job): d1s xe 2.66 / 2.53 vs k34 2.95, 2.98 / 2.92, 3.00;
  v1s xe 3.90 / 3.82 vs k34 4.74, 4.81 / 4.63, 4.64.
  nsys (judge ~ kernel duration): d1s 3.36 vs k34 3.81 us, v1s 4.61 vs 5.52 us; event-minus-empty 2.24 / 2.70,
  3.42 / 4.35.
- Coordinator checks (2026-10-28): (1) K2_DEFS @DEFS variants are real: _XE_DEFS is read and the module compiled at
  import (job4 traces show variant-only stamps). (2) XF's prefix-load sinking: compute warps load only seq + the
  page-table entry before the K/V burst (first two LDGs in SASS); prep warps issue all their loads (seq, inv_freq,
  Q, new k/v, cache_loc, scales, gate) at entry, ~0.7 us before the burst; the gate is stashed in smem before a
  barrier. No change needed.

## State / next levers (resume here)
Deliverable dev/xe/k34xe.py = `python3 dev/xe/build.py dev/k34.py dev/xe/k34xe.py` (md5s below). Final traces
(job8, flush, us): d1s K landed 1.63, PV 1.98, merge barrier 2.11, reader 2.24, pushed 2.30, owner recv 2.63 (waits
for the slowest CTA: K-land max 1.82), out ~3.05. v1s: K landed 1.70, P 2.27, PV 2.50, merge 2.82, reader 2.98,
pushed 3.42, recv 3.74, out 4.12.
1. d1s owner tail ~0.45 us after the last push (LDS 0.09, 16 ex2 + sums 0.16, 5-shuffle RMSNorm + store 0.2) and the
   reader: unified reference = the new token's score per row (prep warps have Q fragments + new K) -> sources scale
   by 2^(m - REF), owner just sums (no max / ex2), exact fallback when m - REF > 64.
2. v1s compute: K landed -> P 0.57 us (32 MUFU ex2 + 24 shuffles per thread): ex2.approx.f16x2 on packed
   (s c - m); merge STS 0.15 + reader/push 0.45 us (4 rows / thread, 8 st.async): 8 compute warps (row halves) or
   FP16 O in the merge smem.
3. K/V landing spread (min 1.44 / med 1.63 / max 1.82 d1s): the owner waits for the slowest CTA.
Dead ends here: bulk-copy push (+0.15 d1s / +0.6 v1s), FP16-O push (neutral), owner pre-warm on v1s (+0.1),
bf16-rounded Q (v1s +0.1, only precision).
ae167cd3f999818a7497c718d7195c90  dev/xe/k34xe.py
cb820565039eb7074502141506fad95a  dev/xe/xe_mod.py
3a25f120e9ad39300ec6fe0ce6264a9b  dev/xe/build.py

## Round 2b (coordinator 2026-10-28: targets d1s <= 2.25, v1s <= 3.3; base dev/coord/k35.py = k34xe v5, md5 ae167cd3)
Controls from now on: k35a = dev/coord/k35.py, k35b = dev/xe/k35b.py (identical copy).
- v6 (xe_mod_v6.py): XE_REF (per-row reference = the row's own new-token score, computed identically in every CTA by
  the prep warps; compute warps scale slice partials to it, reader / owner plain sums, exact max-merge fallback when a
  max exceeds REF + 64; owner RMSNorm on y / L), XE_EXH (ex2.approx.f16x2), XE_W8 (q_len 8: 8 compute warps = 4 key
  slices x 2 row halves). Start stamps 19 (page-table load issued) / 20 (landed).
  job10 (2 runs, 40 reps): d1s k35 2.49, 2.56 / 2.46, 2.59 | all-on 2.69 / 2.65 | REF off 2.70 / 2.66 | EXH off
  2.62 / 2.58; v1s k35 3.71, 3.83 / 3.68, 3.76 | all-on 3.94 / 3.86 | REF off 3.91 / 3.77 | EXH off 3.91 / 3.85 |
  W8 off 3.92 / 4.07 -> all three off. The owner fast path did not shorten its post-wait phase (own-recv -> sums
  0.17 us either way: not the max / ex2 math). W8: K/V issue later (0.99 vs 0.88: duplicated loads per key slice).
  Start: page-table load issued at 0.29 us, landed 0.66, K/V issued 0.86 (12 shuffles + loads 0.2 us).
  SASS: barrier.cluster.arrive at entry = LDC c[0x0][0x140] (driver constant, another line) + branch before the
  seq / page-table loads -> XE_LATEARV (arrive after the first loads).
- v7: XE_LATEARV, XE_DPT (each thread loads the page-table entries of its own fragment keys: no shuffles; two base
  pointers + immediates), XE_D8 (q_len 1: 8 compute warps x 16 keys), XE_TW (owner spins on test_wait).
  job11 (2 runs; this container's A/A spread up to 0.23 on d1s): d1s k35 2.24, 2.38 / 2.47, 2.25 | DPT+LATEARV
  2.41 / 2.54 | LATEARV only 2.31 / 2.31 | DPT only 2.50 / 2.48 | +D8 2.34 / 2.33 | +TW 2.36 / 2.43;
  v1s k35 3.65, 3.66 / 3.72, 3.69 | DPT+LATEARV 3.74 / 3.84 | LATEARV only 3.55 / 3.54 | DPT only 3.62 / 3.71 |
  TW 3.82 / 3.87 -> DPT off, LATEARV on, TW off. Shared fp32 atomics are CAS loops on sm_90a (no smem red merge).
- Cluster size (probe.py, job11, graph us; skeleton = loads + 1-barrier merge + push + owner store):
  16 CTAs x 128 keys 1.90 (loads only 1.26) | 8 x 256 keys (8 warps) 1.84 (1.33) | 4 x 512 keys 2.35 (1.63).
  Landing gets later with fewer CTAs (1.44 -> 1.55 -> 1.92 med) while the owner merges fewer sources: 8 ~ 16,
  4 worse; > 16 needs a cross-cluster (global) merge. => no change.
- v8: + XE_XH (CTA merge buffer in FP16: half the STS / LDS; slice partials are relative to the slice max, so
  |O| <= 32 x 448 fits FP16). job12 pending.
- job12 (v8 LATEARV=1 + XH / D8, 3 bench runs, slow container): all within the A/A spread (d1s xe 2.63 / 2.73 / 2.60
  vs k35 2.63-2.87; v1s 4.06-4.15 vs 3.93-4.11). Trace: with the arrive between the page-table load and the K/V
  issue, the K/V-issued stamp moved to 2.21 us.
- job13 (v9: arrive after the K/V issue): d1s 3.53 / 3.55 / 3.46 vs k35 2.69-2.86 (!), v1s 4.97 / 4.94 / 4.82.
  SASS: ptxas tail-duplicated the 2-pass issue loop and emitted predicated copies of the LOADED registers
  (`@P2 MOV R26, R8` right after the K LDGs): each copy waits for its load, so the V loads issued only after K
  landed (two DRAM trips in series). The barrier.cluster.arrive compiles to a constant-dependent if/else
  (LDC c[0x0][0x140]); any arrive / wait placed after the loads triggered the duplication.
- job14: probe K only lands 1.09, V only 1.12, K + V 1.25 (mostly latency: issue order is worth <= 0.15; ptxas
  issues the V loads first regardless of source order, membar / syncwarp fences do not hold .nc loads).
  v10 (single issue site after the length check, branch-free) benched ~0.3 us slower on d1s (3 runs, both orders).
- METHOD CHANGE (job15): the bench has ~0.3 us placement noise between identical module copies on d1s (coordinator
  saw k35 vs an identical k34xe copy 2.72 / 2.59 vs 3.07 / 2.95). nsys kernel durations (dev/xe/nsys_run.py,
  60 replays, separate processes) agree within 0.03 us between identical copies and between passes:
    d1s / v1s nsys us:  k35 3.36, 3.39 / 4.61, 4.58 | k35b 3.38, 3.39 / 4.61, 4.61
    v10 (length check before issue) 3.74 / 3.73 | 4.90 / 4.93;  v10 + XE_KVS=1 (speculative loop) 3.49 / 3.46 |
    4.64 / 4.62;  + XE_REF 3.73 / 3.74 | 5.06 / 5.02;  + XE_D8 3.79 / 3.78 | 4.86 / 4.90.
  => issuing K/V before the length lands is worth 0.27 us; every round-2 tweak so far is neutral or negative;
  judge ~ nsys duration, so A/B decisions now use nsys (+ >= 2 copies when the bench is used).
- FROZEN round-1 deliverable (coordinator rule): dev/xe/deliver_r1/ = xe_mod.py (cb820565, = xe_mod_v5.py),
  k34xe.py (ae167cd3 = dev/coord/k35.py), build.py (rebuilds it byte for byte). Working files: dev/xe/xe_mod.py,
  dev/xe/k34xe.py (in progress; do not integrate).
- v11 (job16 pending): XE_C8 = v1s on 8-CTA clusters x 4 row groups of 2 tokens (64 CTAs, 256 keys / CTA on 8
  compute warps, 16 rows / cluster: half the per-warp compute, merge, push and owner work for ~+0.1 us later
  K/V landing); KVS loop back on; switches to revert suspected d1s costs: XE_LDPOLD (round-1 "+r" loads),
  XE_NKALL (new token only in owner CTAs), XE_ONORM (round-1 epilogue).
- job16 (nsys, 2 passes, us; xe and xe_b are two module copies): v1s XE_C8: xe 4.29 / 4.32, xe_b 4.34 / 4.30 vs
  k35 4.64 / 4.64, k35b 4.64 / 4.64 (-0.33); XE_C8=0 4.61 / 4.64. d1s: xe 3.49 / 3.46 vs k35 3.42 / 3.39
  (+0.06; LDPOLD / ONORM / NKALL switches do not recover it). t_xe full + check.py 0 failures.
  v1s C8 trace: K landed 1.73, P 2.14 (0.41, was 0.57), PV 2.30, merge 2.43, reader 2.78, pushed 2.88, owner recv
  3.26, out 3.58 (was ~3.8-4.1).
- job17: unsigned key-split math (signed (n + CL - 1) / CL and w % NKS cost sign fix-ups before the page-table
  load): d1s still 3.42-3.46 vs k35 3.33-3.39; XE_XH on C8 v1s 4.29-4.32 vs 4.32-4.35 (-0.03, off).
- Coordinator checks: mbarrier inits = 1 per owner CTA (~25 ns, XC's serialization finding does not bite).
- v13 = ROUND-2 CANDIDATE: q_len 1 runs the round-1 kernel verbatim (deliver_r1 device code inside
  `namespace xer1`, kernel xe_r1_1, SASS byte-identical to round 1's xe_b1s_1; XE_R1D1S=0 selects the generalized
  one); q_len 8 runs xe_b1s_8c8 (XE_C8): grid 8 x 8, cluster 8 (per KV head x 4 row groups of 2 tokens = 16
  rows), 384 threads = 8 compute warps (32 keys each, 256 keys / CTA) + 4 prep warps (2 build Q fragments for
  the 2 tokens, all 4 build 2 new tokens each, prep warps 0 / 1 own the CTA's 2 rows), 8-source owner merge.
- job18 first try: NVRTC 13 (image) rejected the combined source (ADL on float4* found both parts' smem_u32 ->
  ambiguous; local NVRTC 12.4 accepted it) -> XE silently fell back to k34 (_xe_ok False). Fix: round-2 code in
  `namespace xer2` too; dev/xe/nvrtc_chk.py compiles a module with the image's NVRTC and prints the log.
- job18 (round-2 candidate, md5 2555b6ea): t_xe --seeds 2 0 failures (worst err/bound 0.048), check.py 0 failures.
  nsys (2 copies x 2 passes): d1s 3.39 / 3.39, 3.42 / 3.42 vs k35 3.38 / 3.41, k35b 3.39 / 3.39 (= round 1);
  v1s 4.34 / 4.32, 4.34 / 4.32 vs k35 4.66 / 4.67, k35b 4.66 / 4.64 (-0.33).
  Bench (60 reps, 2 runs, 2 copies each): v1s xe 3.49 / 3.41, xeb 3.49 / 3.39 vs k35a 4.09 / 3.99, k35b
  4.07 / 3.95 (-0.58 local); d1s 2.76 / 2.72, 2.73 / 2.71 vs 2.75 / 2.65, 2.70 / 2.66 (noise).
  v1s C8 trace: page-table issued 0.22, landed 0.54, K/V issued 0.80, K landed 1.66, P 2.08, PV 2.26, merge
  2.43, reader 2.75, pushed 2.84, owner recv 3.20, out 3.52.
- FROZEN round-2 deliverable: dev/xe/deliver_r2/ = xe_mod.py (bdef4707), k34xe.py (2555b6ea), build.py (3a25f120;
  `python3 dev/xe/deliver_r2/build.py <base> <out>` appends deliver_r2/xe_mod.py, reproduces 2555b6ea on dev/k34.py).

## Round-2 state / next levers (resume here)
- d1s (round-1 kernel): start (page-table issued 0.22-0.26 us after the first CTA start, landed ~0.6, K/V issued
  ~0.85, landed ~1.6) is at the 2-trip floor; tail ~1.0 us (merge 0.06, reader 0.13-0.17, push 0.06, recv 0.26,
  owner 0.35-0.45). Everything tried on it in round 2 is neutral or worse under nsys (REF, EXH, D8, DPT, late
  cluster arrive, TW). Open: owner post-wait phase (first LDS block after the mbarrier 0.25 us), redux-based
  RMSNorm sum (1 instr vs 5 shuffles), DSMEM latency.
- v1s (C8): the reader is smem-bandwidth bound (8 slices; mbar -> reader 0.3 us): an FP16 merge buffer gave only
  -0.03; recv 0.36 us after the median push. Next: fewer slices with a register pre-merge between warp pairs is
  impossible (no cross-warp shuffle); REF + plain sums in the reader; 4-CTA clusters x 8 row groups (probe: 4 x 512
  keys loads land 0.4 us later: probably a loss).
- Bench placement noise ~0.3 us on d1s between identical module copies: decide on nsys (nsys_run.py, 2 passes,
  2 copies), bench only for confirmation.

## Round 3 (coordinator 2026-10-28): (A) b1 tail levers (<= 2 jobs), (B) decode sliding B > 1 (d16s / d32s)
- job19 (part A): XE_RDX (owner RMSNorm sum of squares by 2 x redux.sync: max exponent, then a fixed-point integer
  add, instead of 5 dependent shuffles; both owners) and XE_XH (FP16 CTA merge buffer); nsys 2 passes, 2 copies each,
  vs deliver_r2 x2: d1s r2 3.44 / 3.42 | 3.42 / 3.39, rdx 3.42 / 3.39 | 3.39 / 3.42, rdx+xh 3.39 / 3.39; v1s r2
  4.32 / 4.29 | 4.29 / 4.30, rdx 4.26 / 4.32 | 4.29 / 4.32, rdx+xh 4.26 / 4.32. => all within the +-0.03 nsys noise:
  NOT kept (XE_RDX removed again; the working module's b1 part is byte-identical to deliver_r2). t_xe / check.py 0
  failures. The remaining b1 tail (owner post-wait ~0.25, push -> receive ~0.26) is a chain of dependent DSMEM /
  smem latencies that the reduction form does not shorten. No deliver_r3: deliver_r2 stays the b1 deliverable.
- Part B design (namespace xem, kernels xe_dm_<CL>_<NCW>_<NPS>; the b1 namespaces untouched):
  one cluster of CL CTAs per (request, KV head), grid (CL, 2B); CTA = NCW compute warps + 1 prep warp; keys per CTA
  2048 / CL = 32 NCW NPS; warp w, pass ps owns keys 32 (NCW ps + w) .. +31 of the CTA's range (fragment-exact LDG
  K / V into registers, as b1); passes pipelined in place: K of pass ps+1 issued right after pass ps's QK, V after
  its PV (same registers), online softmax across passes (exact rescale). No length guess (ruling): seq -> page table
  -> K/V. Owners rank < 8 (q-head 8h + rank): new token (CTA 0 writes the cache), CL-source merge + new-token partial,
  RMSNorm x sigmoid(gate), bf16; st.async push + one mbarrier per owner CTA (1 init), reader conflict-free, as b1.
  No cross-cluster dependency: clusters may run in waves. Static smem 12-21 KB, so several CTAs per SM:
  16_4_1 78 regs (5 CTAs / SM by regs), 8_4_2 90 (4 / SM), 8_8_1 67 (3 / SM, 288 threads).
  Default: B <= 16 -> 16_4_1 (512 CTAs at B 16), B > 16 -> 8_4_2 (512 CTAs at B 32). Predicate: q_len 1, 2 <= B <= 64,
  tp8, page 1, 0 < window <= 2048, RoPE, contiguous qkv (2304) / page table rows / cache_loc (B entries).
- job20 (first xe_dm build): check.py 0 failures; t_xe B>1 cases that ran (B 2/4/16/32 ctx 2047) ok, err ~4.5-6.9e-3
  (the job's grep matched "relative L2 error" -> head cut the variant runs: re-run in job21 with --bonly).
  cuOccupancyMaxActiveClusters: 16_4_1 28 (clusters of 16 pack badly at several CTAs / SM), 8_4_2 62, 8_8_1 45.
  nsys (2 passes): d16s k34 7.23 / 7.23, k35b 7.26 / 7.25 | 8_4_2 6.50 / 6.50 | 8_8_1 6.69 / 6.64 | 16_4_1 8.48-8.61
  (waves: 32 clusters > 28). d32s k34 9.25 / 9.25, k35b 9.22 / 9.25 | 8_4_2 11.86-11.92 (64 clusters > 62: 2 in a 2nd
  wave), 8_8_1 11.86-11.98, 16_4_1 14.4. => d16s 8_4_2 -0.73 us vs k34 (target <= 6.3); d32s needs all 64 clusters
  co-resident -> job21: 8_4_2r (launch bounds 5 CTAs / SM: 72 regs), 8_2_4 (96 threads), plan picks the first
  variant of XE_MORDER (8_4_2 / 8_4_2r / 8_2_4) whose 2B clusters are all co-resident.
- job21 (variants): t_xe --bonly (B 2..64 incl. agentic / mixed / ctx 10 / 2047 / edges / kv 1/64) 0 failures for every
  variant, worst err/bound 0.039 (5.9e-3); check.py 0 failures. Co-resident clusters: 8_4_2 62, 8_4_2b 45, 8_4_2r 77,
  8_8_1 45, 8_2_4 92. nsys 2 passes: d16s k34 7.23 / 7.26, k35b 7.23 / 7.20 | 8_4_2 6.53 / 6.53 (copy 6.53 / 6.53),
  8_4_2b 6.59 / 6.61, 8_8_1 6.69 / 6.66, 8_4_2r 7.23 / 7.20, 8_2_4 7.94 / 7.92; d32s k34 9.28 / 9.28, k35b 9.25 / 9.28 |
  8_4_2r 9.30 / 9.28 (copy 9.25 / 9.22), 8_2_4 9.92 / 9.89 (8_4_2 / 8_4_2b / 8_8_1 run a 2nd wave: 11.8-12.3).
  => d16s -0.70 us (8_4_2); d32s = k34.
  SASS 8_4_2: ptxas hoists pass 1's K loads above the Q barrier (K two passes deep) but pass 1's V only after QK1
  (waits K1): V1's trip is serial behind K1; 8_4_2b (everything up front, 102 regs) is not faster.
  trace d16s 8_4_2 (flush, med / max over 256 CTAs): pt issued 0.61, K/V issued 1.22 / 1.70, pass-0 QK 2.67 / 3.17,
  PV done 4.32 / 4.99, merge barrier 4.40, reader 4.70 (+0.30), pushed 4.77, owner recv 5.15 / 5.50, out 5.44 / 5.79.
  trace d32s 8_4_2r: K/V issued 2.05 / 4.00 (!: page-table loads queue behind the K/V flood), PV 6.29 / 8.22, out
  7.71 / 8.99. XC's gather skeleton (128-B rows, cp.async, per request) lands d16s 3.79 / 4.21, d32s 5.76 / 6.35:
  my per-(request, head) gather fetches each 128-B cache row as two 64-B halves from different SMs at different
  times (2x the random DRAM accesses) -> job22: HB 2 = one CTA per (request, rank) holding both heads' warps
  (identical per-warp code; per-head smem, merge barrier, owner warp and mbarrier), grid (8, B).
- job22 (HB 2, per request): t_xe 0 failures. Co-resident: 8_4_2h 30, 8_4_2hr 45, 8_2_4h 62, 8_8_1h 15. nsys 2 passes:
  d16s k34 7.23 / 7.23, k35b 7.26 / 7.23 | 8_4_2 6.53 / 6.50 | 8_4_2h 6.98 / 6.98 (copy 6.98 / 6.94) | 8_8_1h 9.95
  (waves) | 8_4_2hr 8.16 | 8_2_4h 10.3; d32s k34 9.14 / 9.12, k35b 9.17 / 9.15 | 8_4_2r 9.50 / 9.47 | 8_4_2hr 10.2 |
  8_2_4h 11.7. => per-request CTAs LOSE (+0.45 at d16s): the 64-B halves were not the problem. trace d16s 8_4_2h:
  CTAs 1 per SM on 120 SMs (8 SMs with 2), K/V issued 1.23 / 1.54, PV 4.27-4.45 / 5.15, out 5.63 / 6.05.
  trace d32s 8_2_4h: K/V issued 1.34 / 2.02 (no late page-table loads with 4 CTAs of 96 threads), but PV 8.22: four
  serial passes per warp, each a loaded DRAM latency.
  bench (40 reps, same job): d16s k34a 6.57 | k35b 6.58 | 8_4_2 5.87 (x4.378 vs x3.914) | 8_4_2h 6.29; d32s (forced
  variants, waves) k34a 8.63 | k35b 8.59 | 8_4_2 11.12.
  => d32s: the per-warp chain (pass-1 V issued only after K1 lands; 8_2_4: 4 serial trips) is the limit, and an
  all-register prefetch does not fit 64 co-resident clusters -> job23: SP = passes >= NB staged in smem by cp.async
  at the first issue (each thread's own fragment-exact pieces, zero-filled when masked, read back by LDS after its own
  cp.async.wait_group: no barrier); 8_4_2s (90 regs), 8_4_2sr (72 regs, 5 CTAs / SM, 29 KB smem), 8_2_4s (96 thr).
- job23 (SP smem-staged passes): t_xe / check.py 0 failures. nsys 2 passes: d16s k34 7.23 / 7.26 | 8_4_2 6.53 / 6.53
  (copy 6.53 / 6.54) | 8_4_2s 7.07 / 7.06 | 8_4_2sr 7.62 / 7.58 | 8_4_2r 7.20 / 7.17 | 8_2_4s 9.3; d32s k34 9.34 / 9.31,
  k35b 9.33 / 9.34 | 8_4_2r 9.97 / 9.78 | 8_4_2sr 10.72 / 10.75 (copy 10.80 / 10.78) | 8_2_4s 12.8. bench: d16s k34a
  6.35 | k35b 6.40 | xe (8_4_2) 5.64 | copy 5.71; d32s k34a 8.37 | 8.46 | xe (8_4_2sr) 9.78 | 9.88.
  => LOSES: everything-in-flight makes the ISSUE stall (trace kv-iss d16s 1.22 -> 2.02 med; d32s 8_4_2sr 3.82 med /
  7.17 max, 8_2_4s 5.98 med): per-SM memory-request backpressure; the gather is throughput-bound and more
  outstanding requests only add queueing (and starve late page-table loads). 8_4_2r's d32s varies 9.22-9.97 across
  containers (k34 9.12-9.34): its launch-bounds register cap (72 regs + spills) also costs +0.7 us at d16s
  (7.17-7.23 vs 6.50-6.53) -> job24: __maxnreg__(80) (80 regs, 0 spill, 5 CTAs / SM) / 88.
- job24 (__maxnreg__): n80 (80 regs, 0 spill) / n88 still give 62 co-resident clusters: the register file is split
  over 4 SM sub-partitions, so 5 CTAs of 5 warps need <= 72 regs (7 warps per partition) -- which is what the
  launch bound did (8_4_2r: 72 regs + spills, +0.7 us). nsys 2 passes: d16s k34 7.30 / 7.26, k34c 7.33 / 7.33 | 8_4_2
  6.50 / 6.53 | n88 6.56 / 6.56 | n80 6.59 / 6.59 (copy 6.59 / 6.62) | 8_4_2 XE_MPRE=0 6.56 / 6.56; d32s k34 9.20 / 9.20,
  k34c 9.22 / 9.25 | 8_4_2r 9.74 / 9.76 | n80 / n88 / 8_4_2 11.8-12.1 (2nd wave). bench: d16s k34a 6.60 | k34c 6.71 |
  xe 5.89 | copy 5.83; d32s (8_2_4 by plan) 9.96 / 9.79 vs 8.63 / 8.68.
  => d32s: no variant beats k34 (best 8_4_2r = k34 +0.0..+0.6 depending on the container); deliverable handles
  decode sliding 2 <= B <= 16 only (XE_BMAX default 16), d32s stays on k34's runE1. Untested lever for d32s: a
  4-warp CTA without the prep warp (prep work after the K/V issue in the compute warps; owner merge by warp 0 at
  the end) would fit 5 CTAs / SM at <= 102 regs and so 64 co-resident clusters with 8_4_2's schedule.
- deliver_r4/ (frozen): xe_mod.py b1261fff (b1 device source byte-identical to deliver_r2; xem: only xe_dm_8_4_2
  compiled unless XE_MALL=1), build.py 3a25f120 (= r2's), k34xe.py 6526c8df (= build.py dev/k34.py; reproducible).
- job25 (verification of the frozen deliver_r4): image NVRTC 13 compiles (default and XE_TRACE + XE_MALL); t_xe full
  (b1 + B 2..64; B > 16 on the base kernel) 0 failures; tools/check.py 0 failures. nsys 2 passes (r4 / copy r4b vs
  deliver_r2 and k34 x2): d1s r4 3.39 / 3.39, r2 3.42 / 3.39, k34 3.81-3.84; v1s r4 4.29 / 4.35, r4b 4.32 / 4.32, r2
  4.32 / 4.32, k34 5.44-5.47; d16s r4 6.43 / 6.40, r4b 6.43 / 6.40, r2 (= k34 runE1) 7.30 / 7.26, k34 7.31 / 7.39,
  k34c 7.30 / 7.30; d32s all on k34 9.10-9.22. bench (40 reps): d1s k34a 3.13 | r2 2.76 | r4 2.74 | r4b 2.74 | k34c 3.15;
  v1s 4.90 | 3.63 | 3.52 | 3.47 | 4.86; d16s 6.64 | 6.70 | 5.76 (x4.505) | 5.82 (x4.461) | 6.67 (x3.894); d32s 8.60 |
  8.50 | 8.45 | 8.49 | 8.49.  => d16s -0.86 us nsys / -0.88 us bench vs k34; b1 shapes = r2; d32s = k34.

## Round 4 / jump mode (coordinator: r4 stacked as k41/k42; XF builds the d32s 4-warp lever; b1 targets nsys d1s <= 3.0,
## v1s <= 3.8; questions: v1s compute on the path? CTA merge removable? owner chain? launch shape? MMA clock cap?)
- job26 = measurement first: XE_DGN diagnostics in the generalized xer2 kernel (d1s with XE_R1D1S=0, v1s C8), wrong
  results, timing only, data dependencies kept: 1 no QK MMA, 2 no PV MMA, 32 no movmatrix (35 = no MMA instruction at
  all), 4 no CTA merge (reader takes its own warp's slice: no barrier, no cross-warp LDS), 8 no owner merge, 16 no owner
  RMSNorm; trace slots 26-29 = clock64 next to globaltimer stamps (compute warp 0 start -> pushed, owner recv -> out)
  -> SM clock per phase (coordinator: any MMA caps the SM at 1.80 GHz vs 1.98); XE_D1C8: d1s on 2 clusters of 8.
- job26 results. SM clock (clock64 / globaltimer over compute warp 0 start -> pushed; owner recv -> out): d1s
  1.753 / 1.767 GHz, v1s 1.768 / 1.744, d16s 1.790 / 1.812; with NO MMA instruction (DGN 35: no QK / PV HMMA, no
  movmatrix) d1s 1.905 / 1.909, v1s 1.929 / 2.1 -> the MMA clock cap is real here (~-9%).
  nsys 2 passes, deltas vs the generalized base (d1s xe_b1s_1 3.46-3.49; v1s xe_b1s_8c8 4.29-4.34; r4 = 3.39 / 4.29):
               no QK  no PV  no movm  no MMA  no CTA merge  no owner merge  no RMSNorm  all (63)
     d1s       -0.02  -0.05  -0.03    -0.25   -0.09         -0.21           -0.16       -0.53 (2.94)
     v1s       -0.04  -0.13  -0.02    -0.47   -0.34         -0.05           -0.15       -0.80 (3.47-3.52)
  => the MMA instructions themselves cost little (~0.1 / ~0.19) but their clock cap costs ~0.15 / ~0.28 across the
  whole chain. An MMA-free d1s needs ~131K FMA per CTA on 32 SMs (>= 0.27 us of HFMA2 issue per SM vs ~0.1 for the
  HMMA path): at best break-even, so not built; v1s (16 rows x 256 keys / CTA) is out of reach without MMA.
  Biggest real levers: v1s CTA-merge reader (0.32 us in the trace even at the uncapped clock: 24 LDS + 8 ex2 per
  thread, smem/MUFU-throughput bound), d1s owner merge (0.21) and RMSNorm (0.15-0.16, both shapes).
  XE_D1C8 was not applied (empty define value = off in the host) -> job27 with =1.
- job27: t_xe quick 0 failures (XE_RML, XE_D1C8=1). nsys 2 passes (vs gen base d1s 3.47 / 3.49, v1s 4.35 / 4.35; r4
  3.42 / 3.42 | 4.35 / 4.32): d1s on 2 clusters of 8 (D1C8) 3.65 / 3.63 (+0.16: LOSES; trace: 8-slice reader 0.23,
  K lands later 1.66); XE_RML (packed m,l) v1s 4.54 / 4.54 (+0.2: the 4 rows a warp reads are 128 B apart -> 4-way bank
  conflicts; padded in job28); diag 64 (reader without ex2) v1s 4.35 (0); diag 128 (one O slice) 4.26 (-0.09), 192
  4.26 / 4.29; XE_REF v1s 4.58 (+0.23), d1s 3.52; XE_PREW=0 d1s 3.49 (0); XH v1s 4.34 / 4.32 (0).
  traces (flush, med): d1s gen: K 1.54, PV 1.89, merge bar 1.95, reader 2.11, pushed 2.18 (max 2.30), owner recv
  2.43, own-sum 2.69 (!0.26 for a 16-source merge of ~160 dependent-cycle work), out 2.85; v1s: K 1.70, P 2.11, PV
  2.29, bar 2.43, reader 2.75 (0.32: smem bandwidth, 8 warps x (8 LDS.128 + 16 LDS) at once ~384 wavefronts), pushed
  2.85, recv 3.23, own-sum 3.39, out 3.55.
  SASS: EVERY kernel has `@P0 CCTL.IVALL` (full L1 invalidate) right after the owner's mbarrier.try_wait (.acquire
  .cluster) -- on the critical path -- plus one after barrier.cluster.wait (compute warps, off the path) ->
  job28: XE_OWS=1 = owner wait with CTA-scope acquire (data and mbarrier are the owner's own smem: st.async
  complete_tx), no CCTL on the owner path; all namespaces (xer1 d1s, xer2 v1s, xem d16s).
- job28: XE_OWS=1 (owner wait CTA scope: no CCTL.IVALL on the owner path; t_xe 0 failures) neutral on all three
  shapes (d1s 3.39-3.42, v1s 4.32-4.35, d16s 6.53); padded XE_RML neutral (v1s 4.32-4.35). Neither kept.
- job29 (diag 256 / 512: the owner pass and the CTA reader run twice after their wait): the SECOND pass is as slow
  as the first (d1s owner merge 0.26 -> 0.32, RMSNorm 0.19 -> 0.23; v1s 0.16 / 0.16 -> 0.19 / 0.19) -> not i-fetch;
  the tail phases are issue / dependent-latency bound as written. Same for the v1s compute phase: 8 compute warps x
  ~400 instructions on 4 schedulers ~ 0.46 us of issue (K 1.70 -> PV 2.29), and the reader ~110 instr x 8 warps.
  => v1s structural lever: XE_V1T = ONE token per 8-CTA cluster: 16 clusters x 8 = 128 CTAs (v1s used 64 of 132
  SMs): per CTA 256 keys x 8 rows (half the compute and reader work per SM), 2x redundant K/V reads (L2), owners =
  ranks 0-3 with 2 rows each (8 sources) -> job30.
- job30/31 (XE_V1T): at 1 CTA / SM only 15 clusters of 8 are co-resident (GPC sizes) -> 16 needed -> fell back
  (job30: k34 5.47). With XE_SMEM=65536 (2 CTAs / SM allowed, 30 clusters): t_xe / check.py 0 failures, but the CTAs
  land {1: 112, 2: 8} over 120 SMs: the median CTA is 0.37 us EARLIER than C8 (out med 3.14 vs 3.51: compute
  -0.14, reader -0.07, recv -0.09) but the 8 doubled SMs' clusters end ~1 us late (out max 4.13): nsys 4.70-4.74 vs
  r4 4.29-4.32, bench 3.82-3.85 vs 3.30-3.36. No 16-way split of v1s fits 15 clusters -> dead end.
- Jump-mode conclusion (jobs 26-31): nothing beats deliver_r4 on d1s / v1s; no deliver_r5. All new switches default
  off (XE_DGN, XE_RML, XE_OWS, XE_V1T, XE_D1C8; trace slots 26-29 clock, 31 smid); default SASS = r4.
  Floors: d1s with every tail phase AND the MMA clock removed (diag 63) is 2.94 nsys -> the 3.0 target is the
  architecture's floor; v1s diag 63 3.47-3.52.

## Round 5 (coordinator): decode full B > 1 (d16f / d32f) - run1f's SEGMENT SWITCH; base dev/coord/snap/k34xf.py
## (k34 + XF build_ec: XFE_PUB / RCK / PK); deliverable = builder patch dev/xe/build_sw.py on that base
- Tools: dev/xe/mk_swt.py (per-segment stamps in run1f, TR slots 18-31: segment start / rows / issued / Q^T / tile 0 /
  loop exit / merged for segments 0 and 1), dev/xe/trace_sw.py (switch breakdown over 2-segment CTAs, flush, graph
  replay), dev/xe/edge_sw.py (XF's edge2 cases + 8 switch cases: CTAs over 2-3 requests, short next segments, next
  segments with the new token, page 64, dp heads).
- job32 (base, flush, 10 reps, medians): d16f 13 of 132 CTAs have 2 segments (d32f 28.5), they finish 5.6 / 4.9 us
  before the 1-segment CTAs (OVH 1536 = 24 tiles = 6 per lane ~ 11.8 us of work, over-compensating). Per-tile 1.96 /
  1.99 us per lane. THE SWITCH (segment-0 loop exit -> segment-1 tile 0 consumed) = 6.66 us at both shapes:
      merge (lane stores + 4-lane merge + red publish)  1.43   (both shapes)
      bookkeeping -> next segment start                 0.35
      page-table rows of the next request               0.80
      K/V prologue issue (3 stages; LDGSTS stalls)      2.95 / 2.86   (the drained lane re-enters a saturated
                                                                       memory system: issue blocks ~ latency)
      Q^T / new token                                   0.32
      first-tile wait                                   0.73 / 0.77
- Design XES_PF (dev/xe/build_sw.py, knob default 0): the CTA's first segment (>= 9 tiles, next request intersects
  the range) arms; lanes with >= 3 tiles use their last 3 refill slots for the next segment: tile 0 (+ the next Q row
  staged in the new-token buffer, + gates into slot 1 by lane 0), tile 1, and for the 3rd slot tile 2's page-table row
  stashed in smem while that stage (= the lane's last consumed stage) becomes the lane's merge scratch. The segment end
  skips the ring drain; the next segment issues tile 2 into the released stage, builds Q^T from the staged Q, rotates
  its stage index by my_n_prev % 3. Removes rows + issue stall + first-tile wait from the switch (~4.5 of 6.66 us);
  OVH (host default) to be retuned down. Local NVRTC: 128 regs, 0 spill (XES_PF=1, 7152 SASS instr).
- job33 (XES_PF=1): edge_sw (2 seeds) / forced classic (W9A_REF=500) / check.py 0 failures. Trace: switch 6.66 ->
  4.06 (d16f) / 4.24 (d32f): merge 1.49, -> start 0.42, tile-2 issue + wait for tile 0's group 0.64-0.74, issue 0.22,
  Q^T 0.42, first-tile wait 0.89. BUT nsys (2 passes, 2 copies): d16f base 87.09 / 87.17 (copy 87.15 / 87.01) | XES
  87.84 / 87.82 (copy 87.73 / 87.76) | OVH 1024 87.68 / 87.70 | 640 88.43 / 88.30 | 384 89.47 / 89.58; d32f base
  165.27 / 165.01 (165.01 / 165.10) | XES 165.84 / 165.84 (165.79 / 165.76) | 1024 165.81 / 166.13 | 640 166.96 / 167.52
  | 384 167.82 / 167.76. => LOSES (+0.6..0.8; the arming barrier delays every first segment: tile 0 8.22 vs 7.84, plus
  per-tile instructions), and lowering OVH is WORSE: the kernel is HBM-throughput bound, so while one CTA idles at its
  switch the others take its bandwidth (the bubble is not lost work; OVH only keeps straddlers out of the tail).
  Recoverable at best (switch saved x switching CTAs / 132) = 0.25 / 0.55 us. Same verdict as XC on verify full.
  XES_PF dropped (dev/xe/build_sw.py kept as a record).
- Redirect (coordinator): the END chain after the last tile (not throughput-shadowed). XEL = LANE-LEVEL publish
  (dev/xe/build_lp.py on the snapshot): on the W9D path each lane red.adds its own partial straight from registers
  at its own loop exit (lane range check by bar_or; out-of-range lanes -> classic lane slots in ws at float offset 4M
  + a per-request lane bitmask (<= 17 words) in the accumulator's list area), lane barrier, leader acq_rel arrival
  (target 4P), the last of the 4P lanes combines with its 128 threads (accumulator + gates from L2 in one load round)
  -> no ring drain, no smem lane-partial store block (1.2-1.4 us), no 4-lane merge (0.5), no CTA barrier on the
  critical path. P per lane precomputed at the segment start (misc 8 / 9 / 14 / 15). Local NVRTC 128 regs, 0 spill.
- job34 (XEL=1): edge_sw (2 seeds), forced classic lanes (W9A_REF=500), mixed (W9A_REF=40), check.py: 0 failures.
  Trace (flush, 10 reps): base critical CTA tiles -> combined 3.23 / 3.22 us (tiles 82.58 / 161.57 -> published 84.91
  / 163.86 -> combined 85.86 / 164.78); XEL critical LANE exit -> combined 2.35 / 2.34 (publish = range check + reds
  + MEMBAR drain + ATOM 1.54-1.60 p50, combine 0.97-0.99), kernel end 85.94 / 163.49 (last lane exit 83.54 / 161.09).
  nsys 2 passes x 2 copies: d16f base 87.26 / 87.31 (copy 87.20 / 87.09) | XEL 86.99 / 87.10 (86.91 / 87.02): -0.2;
  d32f base 165.01 / 165.04 (164.99 / 164.96) | XEL 164.93 / 165.09 (164.83 / 164.91): -0.05. bench (30 reps): d16f
  92.47 / 92.55 vs 92.89, d32f 176.59 / 176.23 vs 176.79 (-0.4 both). => chain -0.9 us but end to end only ~-0.1..-0.4.
  Hypothesis: early lanes run the cold publish / combine code while their CTA-mates (one warp of each lane per SM
  sub-partition) still run their last tiles -> i-cache interference -> job35: XEL=2 (lane publish after a CTA barrier:
  no interference, keeps the merge removal).
- job35 (faster container; edge_sw + W9A_REF=40 0 failures for XEL=2): nsys 2 passes x 2 copies: d16f base 83.87 /
  84.05 (copy 83.97 / 83.94) | XEL1 83.49 / 83.87 (83.68 / 83.73) | XEL2 83.90 / 84.13 (83.60 / 83.89); d32f base
  158.16 / 158.13 (158.51 / 158.13) | XEL1 158.21 / 157.97 (158.38 / 158.27) | XEL2 158.37 / 158.24 (158.43 / 158.27);
  bench (40) d16f baseb 110.94 | xel1 110.88 | xel2 111.17 / 111.42; d32f 171.24 | 170.95 | 171.57 / 171.48. XEL2 trace:
  chain 2.69 / 2.64 (publish 1.79 / 1.71: reds + MEMBAR drain + ATOM = 2 L2 RTTs under load; combine 0.93 / 0.97).
  => the end-chain shortening (critical chain 3.2 -> 2.3-2.7 us in traces) does NOT show end to end (nsys -0.2 .. 0 at
  d16f, 0 at d32f; XEL2 rules out i-cache interference). Not frozen (bar: >= 1 us). Remaining chain = 3 dependent L2
  round trips (red drain, arrival atomic, combine load) + epilogue: inherent to a global (non-cluster) reduction.
- Round-5 verdict: run1f's segment switch (XES) and end chain (XEL) are both measured; neither moves end-to-end nsys
  by >= 0.3 us; no deliver_r5. Builders kept: dev/xe/build_sw.py (XES_PF), dev/xe/build_lp.py (XEL 1 / 2).

## Round 6 (coordinator, bounded <= 3 jobs): JUDGE-CONDITION CLUSTER LAUNCH DELAY (judge - local offset larger than
## SGLang's on cluster shapes: d1s +0.96 vs +0.55, d32s (runE1 C3) +1.44 vs +0.74)
- job36: dev/xe/probe_cl.py (probe kernel = entry stamps + optional spin, our 4 cluster launch shapes with and without
  the cluster attribute + a 132-CTA reference, CUDA graph, event timing minus an empty graph, conditions warm / sum
  (bench-like: flush.sum(); event; replay) / zero / sumsync), real-kernel CTA entry stamps with the flush kernel
  right before the replay (trace_xe --mode flushns), bench flush 512 MB vs 1 MB on dev/coord/k47.py vs SGLang.
- job36 results. Probe (graph, 200 reps, event us minus the empty graph): cluster vs non-cluster equal within +-0.1 in
  every condition, e.g. spin 3 us + read flush: d1s C16 2.69 vs noC 2.69, v1s C8 2.69 / 2.66, d16s C8 2.88 / 2.91,
  runE1 C3 2.88 / 2.78; CTA start spread (max - min entry) <= 0.13 us, 0 % of reps > 0.5 us, all conditions.
  Real kernels with the flush kernel right before the replay (no sync): CTA entry max d1s 0.13, v1s 0.13, d16s 0.26 us
  (= synced flush). => under the bench's judge-like sequence there is NO late cluster.
  Side finding, bench flush 512 MB vs 1 MB (k47, 30 reps): d1s 2.82 -> 4.57 (!), v1s 3.56 -> 4.92 (!), d16s 5.88 ->
  4.22, d32s 8.70 -> 7.08; SGLang 19.73 -> 19.54, 21.98 -> 21.54, 25.57 -> 24.00, 30.03 -> 29.14: our b1 kernels get
  SLOWER when the flush is tiny (L2 warm and dirty from the input copy). -> job37.
- job37: (1) flush OVERLAPPING on a side stream (SMs partly busy at launch): cluster kernels finish later than the same
  kernel without clusters: event us d1s C16 56.03 vs noC 54.56 (+1.5; spin 3 us: 58.56 / 57.63 +0.9), v1s C8 +0.6 /
  +0.5, d16s C8 +0.9 / +0.8, runE1 C3 +0.4 / +0.1 (cluster CTAs then start together: spread 0.2 vs 1.0-1.3 us for the
  trickling non-cluster CTAs) -> gang scheduling does wait for free SMs, but only when SMs are busy (not in the
  bench's same-stream sequence). (2) d1s / v1s with a 1 MB flush: kernel-internal time SHORTER (d1s out 2.83 vs 3.12,
  v1s 3.33 vs 3.58: warm L2) -> the bench's small-flush slowdown is OUTSIDE the kernel. (3) bench k47 (30 reps):
  d1s 2.80 (512 write) / 2.93 (512 rw) / 2.80 (256 read) / 4.40 (64 read) / 5.53 (16 read); v1s 3.61 / 3.67 / 3.61 /
  4.99 / 5.48; d16s 7.02 / 7.01 / 5.69 / 5.96 / 4.00; d32s 10.63 / 10.62 / 8.68 / 9.02 / 6.49; SGLang d1s 20.4-20.9.
  Hypothesis: with a short flush the GPU waits for the CPU's graph launch after event 0 (exposed launch cost,
  graph-dependent) -> job38: CPU replay time + event time per flush size, probes and real kernels.
- job38: the small-flush anomaly is HOST launch latency inside the event window, not a GPU mechanism: the EMPTY graph's
  event time grows from 4.9 us (64 / 512 MB flush) to 8.7-11.5 us (16 MB flush: the flush ends before the host has
  enqueued event + graph launch, ~3-4 us of CPU replay time, so the GPU idles inside the window). At >= 64 MB the
  real kernels are identical to 512 MB (event minus empty: d1s r4 2.37 / 2.37, k47 2.43 / 2.53, v1s 3.33 / 3.39,
  d16s 5.66 / 5.70, d32s 8.45 / 8.48) and probes cluster = non-cluster within +-0.05 us.
- Round-6 verdict: no cluster launch delay under the bench's judge-like sequence (same-stream flush, any size >=
  64 MB, read or write). Clusters DO pay a gang-scheduling penalty when SMs are busy at launch (flush overlapping on
  another stream): +1.5 / +0.9 us d1s C16, +0.5-0.9 C8, +0.1-0.4 C3 vs the same kernel without clusters. The judge's
  node-to-node spread is ~0.45-0.6 us in ABSOLUTE terms on every shape (d1s 3.59-4.03, d16s 6.90-7.48, d32s
  10.12-10.66): a per-call node-dependent constant (host latency exposure or concurrent GPU activity), not
  d1s-specific. Candidate if contention is confirmed on the judge: d1s on 2 x C8 (XE_R1D1S=0,XE_D1C8=1: +0.16 us
  locally, smaller gang penalty) as a judge A/B; a cluster-free d1s needs a global combine (3 L2 round trips, run1f
  end-chain numbers: > +1 us) -> not proposed.

## Round 7 (coordinator, fresh-eyes review, <= 3 jobs): (A) fence audit, (B) parameter-fetch probe, (C) v1s on C6
- (A) local NVRTC 12.4 SASS of every XE kernel: NO MEMBAR (cluster arrive is already .relaxed; DSMEM = st.async +
  mbarrier complete_tx); only CCTL.IVALL (after barrier.cluster.wait and the owner's try_wait.acquire.cluster; CTA-scope
  owner wait = XE_OWS, neutral in job28), FENCE.VIEW.ASYNC (fence.proxy.async), UCGABAR_ARV / WAIT. Image check in
  job39 (dev/xe/audit_sass.py).
- (B) dev/xe/probe_cl.py --pprobe: kernel `pp` stamps entry, after the first parameter is consumed (branch on it), after
  the first global load (branch on it); d1s launch shape and a 132-CTA grid; warm vs 512 MB read flush.
- (C) XE_V1C6 (opt-in kernel xe_b1s_8c6, defaults byte-identical to r4): 16 clusters of 6 = one (token, KV head) per
  cluster, 8 compute warps x 48 keys (342 keys per CTA: NBW = ceil(ceil(2048 / CL) / 128) = 3, second page-table entry
  per lane for keys 32-47), owners ranks 0-3 (2 rows each), 96 CTAs; 92 regs, 0 spill.
- job39: (A) image NVRTC 13 SASS (dev/xe/audit_sass.py): identical picture to local - no MEMBAR in any XE kernel
  (xe_r1_1 / xe_b1s_* / xe_dm_8_4_2): CCTL.IVALL x3, FENCE.VIEW.ASYNC.S x1, UCGABAR_ARV x2, UCGABAR_WAIT x1 each.
  XD's fix (MEMBAR.ALL.GPU from arrive.release -> sync_restrict fence + arrive.relaxed) has nothing to replace here.
  (B) parameter probe with globaltimer (quantum 0.256 us): entry -> parameter consumed 0-1 quantum warm AND after the
  512 MB flush (no difference); first global load 0.51 (flush) vs 0.26 us (warm). -> job40 in clock64 cycles.
  (C) C6: 17 co-resident clusters, {1 CTA / SM: 96}; nsys 4.256-4.272 vs r4 4.288-4.352; trace out med 3.39
  (C8 ~3.46-3.55) BUT WRONG RESULTS (rel-L2 0.4-0.65, also ctx 7): XT2<N> (pairwise tree) assumed a power of two ->
  the owner's sum / max over CL = 6 sources dropped one. Fixed for any N (power-of-two pairing unchanged: r4 kernels'
  SASS identical up to global-symbol table offsets) -> job40.
- job40: (B) clock64 (cycles, max over CTAs, 200 reps): entry -> first parameter consumed d1s launch 187 warm / 216
  after the 512 MB flush (132-CTA grid 244 / 269): the whole parameter fetch ~0.12 us and the flush adds ~25-30 cycles
  (~0.015 us) -> NOT a >= 0.15 us cost; first global load 413 -> 820 cycles (L2 -> DRAM). (C) C6 fixed: t_xe full 0
  failures (XE_V1C6), default quick 0, check.py 0. nsys 3 passes x 2 copies: C6 4.288 / 4.256 / 4.2725 (copy 4.288 /
  4.256 / 4.256) vs r4b 4.288 / 4.336 / 4.288, r4 4.304 / 4.32 / 4.32 -> -0.04 us; bench (40) c6 3.41 / 3.39 vs r4 3.46 /
  r4b 3.35 (noise). Below the 0.15 us v1s bar -> NOT frozen (XE_V1C6 stays opt-in). C6 needs 342 keys per CTA (3
  blocks per warp) where V1T had 256, which takes back most of V1T's -0.37 us median advantage.
- Round-7 verdict: no deliver_r5 (nothing >= the bars); 2 of 3 jobs used.

## Round 8 (coordinator: NO minimum size; stack every correct + nsys-faster change): k52 = dev/coord/k51.py + small wins
- Work only on copies: dev/xe/bundle/ (k51.py / k51b.py = copies of dev/coord/k51.py md5 c8deedb2), builder
  dev/xe/bundle/mk_k52.py: XEL=1 (build_lp.py run1f patch; k51's run1f == the snapshot's, so the job34/35 tests apply)
  + v1s variant: --v1s c6 (xe_b1s_8c6 + any-N tree, XE_V1C6 default on) -> k52.py; --v1s xh (XE_XH default 1 on the C8
  kernel) -> k52x.py. Other XE kernels' SASS == r4 (local NVRTC, up to global-symbol offsets).
- (3) XG_PFT=150 (v1f): the knob exists only in XG's LIVE dev/xg/xg_mod.py (and k34xg_r5.py); NOT in the snapshot
  dev/coord/snap/xg3_mod.py / k51's XG section nor in dev/xg/deliver_r3 -> applying it would add code to the snapshot
  -> skipped (rule: snapshots only, no other code touched).
- job41: check.py / t_xe / edge_sw (+ W9A_REF 500 / 40) on the bundle, nsys 2 passes x 2 copies k51 / k52 / k52x on
  d16f, d32f, v1s, v1f.
- job41: check.py 0 failures (k52, k52x), t_xe full 0 (both), edge_sw k52 0 (default / W9A_REF 500 / 40). nsys 2 passes
  x 2 copies (k51 / k51b | k52 / k52b | k52x / k52xb):
    d16f 86.99 / 87.07 / 86.98 / 87.10 (mean 87.04) | 87.01 / 86.90 / 87.06 / 86.91 (86.97: -0.07) | 86.93 / 86.91 /
         87.07 / 87.14 (87.01: -0.03)
    d32f 164.88 / 165.10 / 165.25 / 164.70 (164.98) | 165.28 / 164.94 / 164.88 / 165.04 (165.04: +0.06) | 165.12 / 165.02
         / 165.20 / 164.98 (165.08: +0.10)   (identical k51 copies differ by up to 0.55 -> noise; XEL d32f over jobs
         34 / 35 / 41: -0.05 / 0 / +0.06)
    v1s  4.384 / 4.352 / 4.352 / 4.352 (4.36) | C6 4.32 / 4.272 / 4.288 / 4.288 (4.29: -0.07) | XH 4.336 / 4.32 / 4.32 /
         4.304 (4.32: -0.04)  -> C6 kept (faster than XH)
    v1f  not measured by this parser (XG's kernel name); XG section untouched.
- FROZEN dev/xe/deliver_r6/: k52.py c6816120 (= mk_k52.py dev/coord/k51.py out --v1s c6, reproduced), mk_k52.py
  f9b3fdaa, build_lp.py aeffa4d0 (XEL run1f patch).
