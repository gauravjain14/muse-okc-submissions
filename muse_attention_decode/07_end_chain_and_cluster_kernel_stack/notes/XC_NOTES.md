# XC: decode sliding, batch > 1 (d16s, d32s) - new dedicated kernel (round 2)

Owner XC. Workspace dev/xc only. Base dev/k34.py (runE1 = WE's cluster-per-request kernel: d16s attn_1_1_1_1e6,
d32s attn_1_1_1_1e3). Targets: d16s <= 5.0 us local, d32s <= 6.8 (from 6.67 / 8.70).

## Files
- xc_mod.py   the module (own NVRTC program "xc.cu", kernels xc_dsl_{3,4,6,8,16}, host plan/launch); appended verbatim.
- build.py    `python3 dev/xc/build.py <base.py> <out.py>`: base + xc_mod.py + kernel_fn wrapper (predicate below).
- k34xc.py    = build.py dev/k34.py dev/xc/k34xc.py (the deliverable).
- t_xc.py     extra correctness cases for the predicate (B 2..33, short / ctx 1 / 10 / 2047 / 2049, edges, mixed,
              kv_scale 1/64), 3 calls each.
- trace.py    per-CTA phase timeline: `--kind xc` (XC_TRACE stamps, 64 slots/CTA incl. per-warp) or `--kind e1` (k34 runE1).
- nsys_run.py / nsys_loop.py / nsys_parse.py   nsys kernel durations of the judge-style graph-replay loop (any file).
- job*.sh     Modal jobs (`bash tools/gate.sh tools/modal_run.py --cmd "bash dev/xc/jobN.sh dev/xc dev/coord"`).

## Predicate (wrapper)
q_len 1, page_size 1, 16 q heads / 2 KV heads, head_dim 64, RoPE on, 0 < window <= 2048, 2 <= B <= 33, qkv width
2304, FP8 E4M3 cache; C = largest of (16, 8, 6, 4, 3) with B*C <= 132 and all B clusters co-resident
(cuOccupancyMaxActiveClusters, shape/config only). K2_DEFS XC_OFF disables, XC_C=<c> forces a cluster size.

## Design v1 (why)
Critical-path analysis of runE1 (WE NOTES, WP PROFILE): after the last tile lands, a 128-token pair step costs ~0.9 us
(QK 500 cyc, cross-warp max exchange 470, P + fence 330, PV 280: latency, not throughput), then merge / push / peer /
owner ~0.7 us. b16's tiles all land in a burst (~3.9 -> 4.1 us), so that post-landing latency is the tail.
v1 removes every cross-warp dependency from the attention phase:
- each warp owns 16-token blocks (CTA block j -> warp j % 16, <= 3 per warp), loads their page-table entries itself,
  issues their K/V rows by cp.async into private smem slots (128B-swizzled 128-B rows) and waits on its own cp.async
  groups: no mbarriers, no named barriers, no producer/consumer split;
- mma.sync m16n8k16 FP16 swap-AB per warp: S^T = K Q^T (K e4m3 -> f16 exactly via F2FP.UNPACK_B, Q = RMSNorm + RoPE
  in f16), P = 2^(s c - m) in f16, P^T B-fragments by movmatrix.trans, O^T += V^T P^T (V^T by ldmatrix.trans + one
  prmt + 2 unpacks per register); FP32 accumulation;
- per-warp lazy max: f16 P has 15 log2 units of headroom, so the block max (3 shuffles) is only re-taken when a score
  exceeds m + 15 (always on a warp's first block);
- tail: warp partial (O 64 x 16 rows, m, l) -> own smem slots -> ONE CTA barrier -> merge (warp = q-head row, 2 dims per
  lane) -> st.async row push (264 B) to the owner CTA (rows [16q/C, 16(q+1)/C)) -> owner warp combines C rows ->
  RMSNorm x sigmoid(gate) -> bf16;
- start: straight-line predicated prefix (length, q, inv_freq, gate, scales), mbarrier init + cluster arrive at entry;
  Q (head = warp) RMSNorm + RoPE (lane = frequency, precise sincosf of the float angle) -> f16 smem -> one CTA barrier
  BEFORE the K/V burst (WF start-burst rule);
- the new token: the warp whose block holds window index n1 computes RMSNorm + RoPE -> e4m3 K, e4m3 V, writes the
  cache row and its smem slot (never copied); every other slot is cp.async'ed (zero-filled when not a past token).
Precision: FP16 Q / K / V / P with FP32 accumulation is finer than FA3's FP8 Q / P (competition rules only).

## Log
- job1: tools not uploaded (dev/we, dev/wp not named in --cmd) -> self-contained copies in dev/xc.
- job2 (v1 = XC_ROLL 0, image NVRTC; local nvcc 12.1: 111 regs, 32 B stack (sincosf slow path), 0 spill, 2976 SASS
  instr, block body ~190 instr + ~84 on the raise path): t_xc 0 failures, tools/check.py 0 failures.
  bench 40 reps (us): d16s k34a 6.49 | k34b 6.59 | xc 6.81 ; d32s 8.57 | 8.51 | 8.89  -> v1 LOSES +0.25 / +0.35.
  nsys kernel duration (med): k34 d16s 7.25, d32s 9.31 | xc 7.52, 9.76 (event-minus-empty 6.18 / 8.18 | 6.50 / 8.67).
  k34 runE1 trace build (K2_TRACE, ~20 stamps, slower than the real kernel: out 7.35 vs bench 6.49) d16s med:
  pre-ld 0.45, len 0.86, entries 1.22, bar10 1.57, tile0 3.86, all copies issued 4.70, WG0 tiles 5.10, WG2 done 5.95,
  merged 6.34, pushed 6.49, recv 6.94, out 7.35; d32s: tile0 4.01, WG2 done 8.48, merged 8.90, pushed 9.03, out 9.73.
  xc trace (8 stamps + 48 per-warp, close to the real kernel: last stamp 6.86 vs bench 6.81) d16s med (per-CTA max
  over warps): pt loads issued 0.78, Q barrier 0.98, warp copies issued 2.02, first block landed 3.68 (warp medians
  3.22-3.41), warp done 4.73 (3.86-4.61), all warps + barrier 4.89, pushed 5.63 (!0.74 merge + push), recv 5.95,
  out 6.33, end 6.44 (max 6.86). d32s: copies issued 3.57 (2.56-3.46: 3 blocks per warp, issue backpressure),
  landed 4.27, done 6.58, all 6.47, pushed 7.50 (!1.03), recv 7.99, out 8.37, end 8.48.
  => the attention phase ends earlier than runE1's, but every post-landing phase is slow for its instruction count:
  per-warp 1-2 blocks take 0.6-1.2 us after landing, ~150 merge instructions 0.74-1.03 us. Hypothesis: cold
  instruction fetch (all warps reach the same never-executed code at once after the L2 flush) -> job3 codewarm test.
- job3 (cold-code test): codewarm (flush, then the same kernel on a ctx-64 problem: warm code / params path, cold
  data) vs flush: d16s end 6.42 -> 6.36, merge+push 0.71 -> 0.75; d32s identical. XC_ROLL (rolled block / copy loops,
  1904 vs 2976 SASS instr) identical. => NOT cold code. bench (40 reps): d16s k34a 6.45 | k34b 6.54 | xc 6.64 |
  xcroll 6.65; d32s 8.30 | 8.37 | 8.67 | 8.73.
  Re-read: the attention phase is ISSUE-bound (~190 instr per 16-token block without the raise path, ~270 with it;
  d16s 22 blocks per CTA -> ~5000 warp-instr / 4 schedulers ~ 0.7 us per round; warps with 2 blocks finish last), the
  merge moves 2 x 64 KB of FP32 partials through smem (~0.55 us of smem bandwidth) and the phase-1 cluster wait sits
  on the critical path (WS measured ~0.18 us for that on runA1h).
  Cross-check with the round-2 digest (XF: code first executed after the burst runs late behind in-flight K/V; XA:
  dry warm-up pass is a net loss): for THIS kernel at 8.4 / 16.8 MB the direct test (trace --mode codewarm: same kernel
  on a ctx-64 problem between the flush and the traced replay -> warm i-cache / L2 code, cold data) moves the end by
  only -0.06 us (d16s) / 0.00 (d32s), so post-burst i-fetch is not what limits it; no dry pass built. Shared inputs
  (length, q row, inv_freq, scales, gate) are ld.global.nc (L1-allocating, one miss per SM), not cp.async.cg: no
  L2 hot spot.
- job4 (v2 switches, 40 reps): d16s k34a 6.36 | k34b 6.74 | xc2 (ECW + H16P + no end barrier) 6.84 | xcv1 6.75 |
  xcnoh 6.70; d32s 8.43 | 8.51 | 9.06 | 8.90 | 8.93 -> no switch helps beyond noise. ncu d16s (dsl): issue slots busy
  21%, eligible warps / scheduler 0.83, top stall long_scoreboard 28% (prefix / length / gate loads), merge loop
  barrier 6%. => v3 redesign (below).

## Design v3 (xc_ksl, default; XC_DSL selects v1): K-first gather, deferred exact softmax, PV as V lands
Softmax needs only K; only PV needs V. Each warp issues ALL K rows of its quarters first, then all V rows, so K lands
around mid-stream and QK + the exact max + P^T are done while V streams; after the last V row only V^T + PV remain.
- 4 warpgroups; CTA tile k (64 tokens) -> WG k & 3, slot k >> 2 (<= 3); warp w' of the WG issues quarter w' (16
  tokens) of each WG tile; per tile a K and a V mbarrier (128 cp.async.mbarrier.arrive.noinc).
- QK: FP8 QGMMA m64n8k32 SS per K tile as it lands (k34 dec_tile descriptors), S^T kept in registers (<= 3 tiles).
- ONE exact max per (head, q-head) over the WG's tokens (3 shuffles + 4-warp smem exchange, one WG barrier), P =
  2^(s c - m + 8) e4m3 for all tiles -> P^T smem, l FP32; no online rescale at all.
- PV: per V tile, ldmatrix.trans + prmt V^T fragments, m64n8k32 RS into a fresh accumulator, FP32 add.
- tail: 4 WG partials (FP32) -> CTA merge (warp = q-head row) -> st.async push -> owner combine with lane-parallel
  weights (lane c = source c) -> RMSNorm x sigmoid (precomputed) -> bf16. 64 regs (local nvcc), 0 spill.
- job5 (v3 ksl first build): t_xc 0 failures, tools/check.py 0 failures. Error now ~= the reference's own FP8-query
  error (d32s 6.1e-2 vs reference-FP8 6.0e-2, bound 0.15; v1's FP16 Q gave 4.5e-3): the FP8 E4M3 Q rounding (FA3's)
  dominates; within the rules, same mechanism as k34. bench (40): d16s k34a 6.30 | k34b 6.51 | ksl 6.83 | dsl 6.78;
  d32s 8.35 | 8.39 | 9.06 | 8.95. nsys ksl 7.70 / 10.02.
  trace ksl d16s: length+pt issued 1.22 (dsl 0.8: thread 0's mbarrier-init fence after its loads + Q^T proxy fence
  after the pt loads), Q barrier 1.40, copies issued 2.06-2.30, K tiles landed 3.10-3.58 (warp medians; CTA max 3.68),
  V landed + PV done 4.10-4.70 (CTA max 4.77), all 4.93, pushed 5.34 (max 6.42), recv 6.21, out 6.66.
  => K-first did NOT make K land early: K completes ~1.2 us after issue, V ~0.8 us later; the stream end dominates.
  WE's probe (dev/we/probe.py: gather-only kernel with the real seq -> pt -> K/V chain): b16 C6 5.70 us, b32 C3 8.26
  bench time, i.e. k34 sits only ~0.7 / ~0.1 us above a gather-only kernel -> job6 measures MY gather skeleton.
- BUG found in job6 (fixed): the module compiles lazily on the first kernel_fn call, after tools/bench.py has reset
  K2_DEFS, so every `file@DEFS` BENCH variant in jobs 3-4 ran the default code (traces were right: trace.py keeps
  K2_DEFS). Those job3/4 "variants" are therefore A/A noise: identical code 6.84 / 6.75 / 6.70 (d16s), 9.06 / 8.90 /
  8.93 (d32s) in one job. XC_K2DEFS is now captured at import.
- job6 (v3.1: inits by warp 15 lane 0 before its loads, one proxy fence after the copies): start fixed (length + pt
  loads issued 0.87, Q barrier 1.09). bench (40): d16s k34a 6.51 | k34b 6.82 | ksl 6.98 | dsl 6.93; d32s 8.41 | 8.38 |
  10.05 | 9.14 (ksl d32s: everything deferred behind the WG's last K tile: K done 5.78, PV done 6.90).
  GATHER SKELETON (XC_NOCOMP: same prefix + copies, wait for every K / V tile, no math, no combine), trace:
  d16s copies issued 2.29 (CTA max), all K/V landed CTA max med 3.79 / max 4.21; d32s 2.98, 5.76 / 6.35.
  (NOCLU, plain launch: d16s 4.02 / 4.45, d32s 6.00 / 6.59: no better.) The full ksl kernel ends 6.88 / 9.76 ->
  ~2.7 / ~3.4 us AFTER the data is in smem. d16s ksl: QK done 3.81 -> PV done 4.74 (0.9 for exchange + P + 1-2 PV),
  -> all 4.87 -> pushed 5.30 (0.43 for a 4-way merge + push) -> recv 6.12 (0.82: slowest CTA of the cluster, pushed
  max 6.30) -> out 6.57 (0.45 owner) -> last 6.88. The spread across CTAs grows from 0.7 us (skeleton) to 1.5 us.
- job7 FLOOR (bench, 30 reps, --check-reps 0, same job as k34a): gather skeleton XC_NOCOMP d16s 4.09 / d32s 6.12 us
  (k34a 6.52 / 8.36). XC_TLB=1 (one L2 prefetch per 2 MB page of both caches + page-table row ends at entry,
  shape-determined) 4.07 / 6.18 -> DEAD END. XC_NOCLU (same skeleton, plain launch, no cluster) 4.09 / 6.24 ->
  the cluster launch costs nothing measurable here. => this chain (length -> pt -> all K/V issued at once on 96
  CTAs) gathers 1.6 / 2.1 us FASTER than WE's probe (5.70 / 8.26: its per-thread 24-entry preload + branchy issue),
  and k34 sits 2.4 / 2.2 us above it: there IS headroom. Targets 5.0 / 6.8 = skeleton + 0.9 / + 0.7 us.
  ksl detailed trace d16s (per-CTA max over warps, median CTA): QK done 3.79 -> P^T ready 4.38 (+0.59 exchange + P
  + proxy fence + 2 WG barriers for 1-2 tiles) -> first V wait passed 4.51 -> last V 4.66 -> PV done 4.83 -> partial
  written 5.06 -> pushed 5.44 -> recv 6.32 -> out 6.76 (last stamp 7.17). d32s: QK 5.82 -> P 6.49 -> PV 7.06 ->
  partial 7.27 -> pushed 7.66 -> recv 8.88 -> out 9.33. bench (40): d16s k34a 6.42 | k34b 6.40 | ksl 7.04 | ksl+TLB
  6.84 | dsl 6.75; d32s 8.29 | 8.34 | 9.83 | 9.80 | 9.00.
- job8 (fine per-warp stamps, flush vs warm = K/V already in L2): even WARM, ksl d16s spends 2.8 us from QK done
  (2.85) to the output (5.66): 12 shuffles of the max 0.16 us, STS + 128-thread barrier 0.19, P of 1-2 tiles (8 ex2
  + 4 cvt per thread per tile) 0.34, PV of tile 0 0.32, last PV 0.19, l shuffles + partial 0.21, merge + push 0.39,
  peer 0.30, owner 0.46. Flush adds ~+0.6 us in front (landing) and +0.6 in the peer wait (CTA spread).
  => NOT issue slots (ncu: 21% busy) and not cold code: SM-WIDE SPECIAL-PIPE THROUGHPUT. On sm_90 SHFL is 32 lanes /
  clk / SM (one warp-shuffle per clock for the whole SM), MUFU 16 / clk / SM, F2FP conversions are similarly narrow;
  all 16 warps hit the same phase together after the burst, so a 12-shuffle step costs 16 x 12 = 192 SM cycles, the
  exps of 6 tiles 384+, v1's 68 F2FP per block ~1.5 us for d16s. (That is also why runE1's steps are slow.)
  => v4: EXQ P codes (FFMA + clamp, W9's K8_EXPQ; m ceil'd to a 1/8 grid), row sums l from a ones-operand QGMMA
  (A = 0x38 x 4 = E4M3 1.0: every output row = sum_t P), no l shuffles, owner weights by broadcast LDS, copy-row
  shuffles shared by the K and V passes (24 -> 12 per warp).
- job9 (v4: EXQ + ones-QGMMA l + broadcast owner + shared row shuffles): t_xc 0 failures, tools/check.py 0 failures
  (d16s err 6.3e-2 / ref-FP8 6.2e-2, bound 0.15). bench (40): d16s k34a 6.56 | k34b 6.70 | v4 6.87 | v4 XC_EXQ=0 6.84;
  d32s 8.54 | 8.68 | 9.82 | 9.78 -> EXQ + ones-QGMMA = exactly the MUFU/F2FP path's cost: no gain. nsys 7.87 / 10.66.
  warm trace d16s (K/V in L2): QK done 3.04, max shuffles 3.20, exchange barrier 3.34, P^T ready 3.78 (+0.44 for
  ~40 FMA/ALU instr + fence + barrier), tile-0 PV 4.35 (+0.41 for 4 LDSM.T + 16 PRMT + 8 small QGMMA), last PV 4.58,
  partial 4.70, pushed 5.11, recv 5.50, out 5.73. => suspect the tensor core: m64n8k32 QGMMAs (N = 8 q-heads) issued
  by 4 WGs at once may cost close to a full-width QGMMA each -> probe_mma.py (job pmma) measures SM cycles per
  instruction for SS / RS, N = 8 / 16 / 32, 1 vs 4 active WGs, and HMMA m16n8k16.
- probe_mma.py (job pmma, 132 CTAs x 4 WGs, SM cycles): one round {fence; 4 x QGMMA m64n8k32 SS; commit; wait} =
  337 (1 WG active) / 427 (4 WGs) cycles; RS 356 / 406; N=16 383 / 465; N=32 420 / 504; 8 QGMMAs (2 tiles) 569 / 705;
  HMMA m16n8k16: 8 independent per warp 80 / 111 cycles (10-14 per HMMA). => small QGMMA rounds are LATENCY bound
  (~350-700 cycles each, N barely matters); HMMA chains are ~4x cheaper per round. That is why v3/v4's per-tile QK and
  PV rounds (0.2 / 0.35 us) dominate.
- probe_pipe.py (job ppipe2, 16 warps x 8 independent ops, warp-instr / clk / SM): FFMA 2.54, LOP3 1.77, F2FP
  e4m3x2->f16x2 1.77, F2FP f32->f16x2 1.77, PRMT 1.68, SHFL 1.00, F2FP f32->e4m3x2 0.61, MUFU ex2 0.50.
  => MUFU and SHFL are the narrow pipes; the e4m3 -> f16 unpack is NOT (v1's 68 per block cost ~0.45 us at d16s).
- Diagnosis: my gather issues everything by 2.0-2.3 us (runE1's producers still issue at 4.7 in its trace build), so
  my front end is ~2.3 us ahead, but my tail is slower: v1's per-warp attention ALREADY overlaps the landing at d32s
  (all warps done ~= last landing), yet merge + push 0.9-1.0 us (16 ex2 per lane = 8192 MUFU ops + 64 KB smem + the
  phase-1 cluster wait on the path), peer 0.5, owner 0.4.
  => v5 (xc_dsl, default again; XC_KSL selects v4): row sums by an all-ones HMMA m-tile (XE's trick; no l FADDs, no
  final shuffles), no rescale on a warp's first block, merge weights computed ONCE per (row, warp) by 256 threads
  (16-lane shuffles, 1 ex2 per thread) into smem, early phase-1 cluster wait, broadcast-LDS owner, no end barrier.
- job10 (v5 = dsl + ones-HMMA l + weights pass in smem + ECW + broadcast owner): t_xc / check.py 0 failures, error
  4.3-5.2e-3 (FP16 Q). bench (40): d16s k34a 6.38 | k34b 6.44 | v5 6.52 | v4 (XC_KSL) 6.82; d32s 8.29 | 8.48 | 8.73 |
  9.71. nsys v5 7.33 / 9.66. Closest so far; trace d16s: all warps done 4.61 -> pushed 5.63 (still ~1 us).
- job11/12 (merge sub-stamps + clock64 in thread 0): weight pass (8 warps: LDS + 4+4 shuffles + ex2 + STS) 601 SM
  cycles, merge loop 817 cycles = the smem bandwidth of reading 16 x 4 KB FP32 partials (64 KB / 128 B per clk = 512)
  + latency; same warm. globaltimer vs clock64 => SM clock ~1.2-1.7 GHz under this load.
  => v5b (XC_M2, default): partials stored as f16 O / l (32 KB), weights in-warp (lane w = partial w: 1 LDS, 5-level
  shuffles, 1 ex2, normalized to sum 1), broadcast by shuffle inside the loop, HFMA2 accumulation (convex combination
  of |v| <= 448: no overflow), row pushed normalized; owner reweights by L. No weight pass, no 2nd barrier.
- job13 (v5b = XC_M2 merge, A/B vs v5 = XC_M2=0, 2 x 40 reps, same job): d16s k34a 6.35 / 6.49, k34b 6.42 / 6.61,
  v5b 6.69 / 6.81, v5 6.46; d32s 8.25 / 8.59, 8.24 / 8.37, v5b 8.74 / 8.83, v5 8.56. merge clock 1298 cycles (v5:
  601 + 70 + 817): the 10 dependent shuffles + 16 shuffle broadcasts per warp (416 SHFL per CTA) cost what the f16
  traffic saved. check.py 0 failures (v5b). nsys v5b 7.58 / 9.79.
- job14 (XC_M3: shuffle-free weight pass (16 broadcast LDS of the row's maxima + 1 ex2 per thread, (wt, wt l) pairs,
  L summed in the merge loop); M3H: f16 O / l partials converted in the loop; t_xc 0 failures both). Weight pass 484
  cycles, merge loop 844. 2 x 40 reps: d16s k34a 6.46 / 6.44, k34b 6.54 / 6.61, m3 6.54 / 6.64, m3h 6.68 / 6.79, v5
  6.47 / 6.62; d32s 8.44 / 8.41, 8.46 / 8.43, m3 8.82 / 8.95, m3h 9.05 / 8.89, v5 8.86 / 8.71. => all merge variants
  equal within noise; a 16-partial CTA merge costs ~1300 SM cycles however the weights are computed.

## STATUS (2026-10-28 ~15:15) - no design beats k34 yet; where the time is
- Front end is NOT the difference: my gather skeleton (bench 4.09 / 6.12) lands the last K/V row at ~3.8 (median CTA)
  / 4.2 (max) us for d16s; k34's runE1 tiles also land ~3.9-4.1 (WE notes). Both kernels then spend ~2.2-2.3 us after
  the landing: k34 d16s 6.35-6.61 / d32s 8.24-8.59; my best (v5 = xc_dsl + XC_M2=0 / M3) 6.46-6.64 / 8.56-8.95.
- Measured costs that make up the post-landing tail on sm_90 (all with 16 warps busy after the burst):
  * special pipes per SM: MUFU ex2 0.5 / clk, SHFL 1.0 / clk (one warp-shuffle per clock for the whole SM), F2FP
    f32 -> e4m3 0.61; F2FP e4m3 -> f16, PRMT, LOP3 ~1.7; FFMA ~2.5 (probe_pipe.py);
  * one {fence; 4 small QGMMA; commit; wait} round 340-430 cycles, 8 QGMMAs 570-700 (probe_mma.py): QGMMA decode
    tiles pay ~0.2 us (QK) + ~0.35 us (PV) of round-trip latency per tile (v3 / v4);
  * per-warp HMMA attention (v1 / v5) is latency-cheap (HMMA ~10-14 cycles) but ~190 instructions per 16-token block,
    ~1.0 us for d16s's 22 blocks per CTA after the burst;
  * 16 per-warp partials -> CTA merge ~1300 SM cycles (~0.7-0.8 us incl. write) = smem bandwidth of 2 x 64 KB +
    dependent smem/shuffle chains; runE1 merges 3-4 WG fragments (~12 KB) instead;
  * push -> slowest peer of the cluster 0.3-0.5 us, owner combine ~0.35 us.
- Dead ends here (numbers above): TLB touching, non-cluster launch, K-first gather order (K does not land earlier:
  DRAM is not FIFO), deferred exact softmax for d32s (no overlap: +1.3 us), EXQ / ones-QGMMA row sums (= MUFU path),
  in-warp merge weights (shuffle-bound), f16 partials (conversion = saved bandwidth), rolled loops / cold code.
- Best file: dev/xc/k34xc.py (v5 family, default XC_M3=1, xc_dsl kernels): check-passing (job10 check.py 0 failures
  for v5; t_xc 0 failures for M3), error 4.5e-3 (FP16 Q / K / V / P, FP32 accumulation), d16s ~= k34, d32s +0.3 us.
  NOT for integration.
- Next levers (not built): (1) a WG-fragment tail like runE1's (4 partials, ~12 KB) on top of the per-warp HMMA
  attention: warps of a WG pair-merge their O^T in registers via one smem hop inside the WG before the CTA merge, or
  head-split warp pairs (8 partials of 8 rows: half the merge bytes, needs a per-block mbarrier because one warp of
  the pair issues the rows); (2) cut the d16s post-burst attention (~1 us) with a QK on QGMMA issued as ONE round for
  all landed tiles + per-warp HMMA PV (no per-tile QGMMA rounds); (3) the owner / peer wait (~0.7 us) needs fewer
  dependent DSMEM round trips (e.g. owners fold sources as their mbarrier bytes land, measured lost by WS at b1).
ecdcac1b4a66ade71158c160f08acd99  xc_mod.py
66b3e48bd6c0a71e28b8dacca1ce7b85  build.py
285ceae47326f2efaa67e7aedf2e7fa4  k34xc.py
36b58693753c55094da6b252de05bace  trace.py
e98d580a64bf9d3cfff92d1474c9db2c  t_xc.py
5a7004faaf13eb84a175f6c15a7d215a  probe_mma.py
e3ec5af12c088998d9a0442bfbc19059  probe_pipe.py

## Round 2 (coordinator, 2026-10-28 ~15:20): 4-/8-partial merge on per-warp HMMA attention
- deliver_r1/ = frozen round-1 hand-back (xc_mod.py ecdcac1b, build.py 66b3e48b, k34xc.py 285ceae4).
- XF's "prefix loads sunk below the burst": checked xc_dsl SASS (local nvcc): every prefix LDG precedes the first
  LDGSTS; only the new-token loads follow (by design, off the critical path).
- Design r2 = xc_hsl (default; XC_DSL = v5, XC_KSL = v4): HEAD-SPLIT warp pairs. warp w: KV head h = w >> 3, pair
  j = w & 7; CTA block k (16 tokens) -> pair k & 7, slot k >> 3 (<= 6 slots x 4 KB per pair). The pair's head-0 warp
  issues the K rows (both heads), the head-1 warp the V rows; mbarrier per (block, tensor) (32 noinc arrivals). Each
  warp attends ONE head of its pair's blocks (same per-warp HMMA code as v5, half the work per block, twice the
  blocks: same total instructions), partial = 8 rows x 64 d FP32 (2 KB; 32 KB in all = half of v5's 64 KB), written
  into the head-h halves of the pair's slot 0 (32 x 64-B pieces freed when the warp is done; the other head's warp
  only reads the other halves). Merge per HEAD: named barrier of its 8 warps (no CTA-wide barrier), warp j folds row
  8 h + j over 8 partials (lane w < 8 = partial w: 3 shuffles, 1 ex2), pushes it. 66-72 regs, ~207 KB smem.
- job15 (hsl first build): t_xc / check.py 0 failures, error 4.3-5.7e-3. bench 2 x 40: d16s k34a 6.74 / 6.71, k34b
  6.86 / 6.88, hsl 9.73 / 9.63, v5 6.68 / 6.78; d32s 8.54 / 8.69, 8.63 / 8.67, hsl 12.61 / 12.74, v5 8.99 / 8.99.
  trace: every warp finishes issuing its copies at ~4.5 us (v5 ~2.0). Cause: warp 15 lane 0 initialised 96 mbarriers
  serially before its cluster arrive (mbarrier.init ~25 ns each -> ~2.4 us), and every CTA's phase-1 cluster wait
  sits before the copies. (The same effect, 24 inits, cost v3/v4 ~0.3 us.) Fix: 3 inits per lane of warp 15, each
  lane fences its own (also applied to ksl).
- job16 (parallel init: 3 mbarrier.init per lane of warp 15): still copies issued ~4.4 us; bench 2 x 40: d16s k34a
  6.84 / 6.65, k34b 6.91 / 6.63, hsl 9.81 / 9.64, v5 6.79 / 6.56, ksl 7.10 / 6.95; d32s 8.75 / 8.53, 8.93 / 8.71,
  hsl 12.77 / 12.68, v5 9.24 / 9.02, ksl 10.42 / 9.94.
- job17 (stamps): the phase-1 CLUSTER WAIT completes at ~3.46-3.49 us in every warp (Q barrier 1.06): it needs every
  CTA's warp 15, which first runs the 96 mbarrier.init + fences. mbarrier.init SERIALIZES per SM at ~25 ns each
  whether issued by 1 lane or 32 (96 -> ~2.4 us; ksl's 24 -> ~0.5 us; runE1 inits 13). => avoid many mbarriers.
  Fix: the per-(block, tensor) mbarriers are replaced by one cp.async group per slot per warp (wait_group via a
  switch) + a 64-thread named barrier per pair (IDs 5-12) per block; only the DSMEM receive mbarrier remains.
- job18 (hsl with pair named barriers, 1 mbarrier): start fixed (cluster wait 1.15, copies issued 2.0-2.1). t_xc 0
  failures. trace d16s: first K block 2.5-2.75 (CTA max 3.12), PV done 3.84-4.48 (CTA max 4.71), head barrier 4.70,
  pushed 5.53 (+0.83), recv 6.41 (+0.88), out 6.77; d32s PV done 7.12 (v5 ~6.5: the per-block pair barrier couples
  the two warps), pushed 7.90, recv 9.50. bench 2 x 40: d16s k34a 6.64 / 6.65, k34b 6.72 / 6.75, hsl 7.29 / 7.25, v5
  6.61 / 6.55; d32s 8.54 / 8.61, 8.52 / 8.63, hsl 10.36 / 10.54, v5 8.95 / 8.90.
  => the 8-PARTIAL MERGE DOES NOT PAY: half the merge bytes, same ~0.8 us merge + push (a chain of ~15 dependent
  LDS / SHFL / MUFU ops after a barrier, ~50-100 cycles each under load, not bandwidth), and the pair coupling slows
  the attention. A 4-partial version would need either redundant QK (dims split, +70% attention instructions) or
  fewer compute warps; with the merge latency-bound, neither is expected to pay. Default restored to v5 (xc_dsl,
  XC_M3); XC_HSL selects the head-split kernel.
- job19 (deliverable r2 = default v5): tools/check.py 0 failures, t_xc 0 failures. bench 40 reps: d16s k34a 6.74 |
  k34b 6.86 | v5 6.80 | hsl 7.40; d32s 8.78 | 8.79 | 9.06 | 10.74.
- deliver_r2/ (frozen, NOT for integration: no win): md5 below. Round-2 conclusion: the post-landing tail is a chain
  of dependent latencies (attention after the burst, CTA merge, push + slowest peer, owner) that neither fewer merge
  bytes (8 partials), shuffle-free weights, f16 partials nor tensor-core row sums shortened; the only measured big
  lever was start-path hygiene (mbarrier.init serializes at ~25 ns each: keep the count tiny before a cluster wait).
66b3e48bd6c0a71e28b8dacca1ce7b85  deliver_r2/build.py
9a432b5618652a4fcccda4c68a6d420f  deliver_r2/k34xc.py
215874eadf691deb68b32ede22eca543  deliver_r2/xc_mod.py

# Round 3: verify full B > 1 END chain of XD's attn_xd_v8f (k37 = k36 + XD r1)
Scope (coordinator, final): XC owns the END chain only (last tile -> lane merge -> publish -> combine -> epilogue ->
output), XD owns start / tile loop. Work files: dev/xc/r3/{xdk.cu (edited), xdcore.cu (XD's, untouched), build.py
(XD's builder; also passes K2_DEFS entries XC_*), xcheck.py (XD's edge checker, path fixed)}; trace tool
dev/xc/trace_xd.py (XD_TRACE stamps, per-request end-chain analysis). Build: python3 dev/xc/r3/build.py dev/coord/k36.py
dev/xc/r3/k37xc.py (= k37 when xdk.cu is XD's).
- job20 (k36 = run8 control): bench v16f 98.56 / 98.77, v32f 179.55 / 179.81; nsys 97.36 / 179.33.
- job21 (XD kernel, detailed stamps), traced, us from the first CTA start (median of 12 reps):
  b32: prefix 0.90, split 2.06 (min 1.25 max 2.83: the LINEAR b_begin scan, ~100 cycles per request, + 64-bit
  divisions), pt-iss 2.62, pt-back 3.74, K/V issued 5.10, tile0 6.18; tiles-done p0/p10/p50/p90/max 162.1 / 168.0 /
  172.3 / 175.7 / 179.8; nseg-2 CTAs (straddlers) end EARLIER (mean 169.2 vs 173.0: NEWW 1536 over-compensates).
  Segment switch bubble 6.6 us (seg-0 tiles -> new block + merge + store 1.86, seg-1 start -> pt back 2.27, pt back ->
  first tile 2.30). End: tiles -> stored 1.3, fence 0.9, atom 0.36; critical CTA = a combiner (index 0 of both its
  requests) that publishes at 171.3, waits for request q0's last arrival until 181.9, combines q0 (1.7), then q1 (2.5)
  -> 186.1: per request last publish -> combined p50 3.65 / p90 6.85 / max 10.2 (HOL: serial combines of 2 requests).
  b16: same start (split max 2.08); critical request: last publish 97.46 -> combined 100.99 (combine 2.56 us: P = 9
  needs 2 rounds of 8 partial loads); tiles max 94.86 -> done ~101 = 6.2 us end chain.
- Design XC_END (dev/xc/r3/xdk.cu, #if XC_END): per request ONE combiner = the LAST end-of-kernel arrival (FINAL),
  elected by an arrival counter incremented by an idle lane-1 thread as soon as lane 1's tiles are done (before any
  store); E = P minus the (at most one) contributor whose partial is its FIRST segment of several (that one publishes
  at its segment switch: store at the segment end, fence + count before the next segment's page-table wait). Others
  (PLAIN) store an fp16 normalized partial (O / l f16 + m, l f32 = 17 KB instead of 33 KB), one fence, count, exit.
  The FINAL keeps its own partial in smem (no store / fence), lane 1 waits for the P - 1 publishes and cp.asyncs the
  other partials into the idle ring (the lane-merge exchange moves into the V^T blocks) while lane 0 merges; then one
  weights pass (row-wise max / sums), one weighted sum, epilogue. No combiner ever waits behind another request
  (each CTA has <= 1 end-of-kernel partial), no partial reload by the last arrival, half the partial bytes.
  Counters self-reset by the FINAL (cnt[b] arrivals, cnt[512 + b] publishes). P > NSL - 1 (= 8 others) -> chunked
  online merge (tests: B < 16).
  Start / switch ideas found on the way (XD's scope now, default OFF here): XC_SPL (double split + ballot b_begin:
  split max -> min; costs +30 B spill outside the loop) and XC_SW (next segment's page-table chunks + L2 prefetch of
  its Q rows right after the last pair, before the new block / merge / store of the current one).
- job22 (XC_SPL + XC_END v0): xcheck 0 failures (errors = XD's); nsys v16f k37 97.38 / 97.42 | xc 102.08 | spl-only
  100.29; v32f 180.85 / 180.72 | 181.01 | 182.37. => XC_SPL's double divisions push ptxas to spill 56-72 B (vs 24):
  reloads after the K/V stream are L2 round trips. Lesson: NO live-across-the-loop spills; checked with local nvcc
  12.1 + SASS scan (sassloop2.py in the scratchpad: spills inside the pair loop / the segment loop).
- job23 (XC_END only, 8-B stack): xcheck 0; nsys (fast container) v16f k37 95.17 / 93.98 | xc 93.31 / 93.36; v32f
  173.20 / 173.04 | 169.57 / 169.25 (-3.7).
- job24 (v2: publishes as bits, lane 1 fetches each partial as soon as it is published; own partial in the V^T blocks
  so 9 ring slots hold the others (b16 P = 9-10: one chunk); end role / first_seg out of registers -> 0 spills):
  nsys v16f k37 97.42 / 97.23 | v1 96.77 / 96.96 | v2 95.90 / 95.87; v32f 179.73 / 179.52 | 176.30 / 176.18 |
  176.43 / 176.40. Trace (traced build spills, end phases inflated): critical FINAL waits ~2-3.5 us for the LAST PLAIN
  publish (the last few arrivals are close: a PLAIN needs ~2 us from tiles to a visible publish), then loads 0.7,
  merge 1.2 (b32) / 2.3 (b16), epilogue 0.8 / 2.4.
- job25 (v3 compact FINAL code, + XC_TWICE cold-code test): the 2nd (warm) pass of the weighted sum / epilogue is
  as slow as the 1st (1.3 / 1.1 us b16): NOT instruction-cache bound; the weighted sum is smem-wavefront bound
  (10 sources x 16 KB) and the epilogue MUFU bound (2 MUFU per sigmoid). nsys: compact = unrolled (noise).
- job26 / job27 (v4: FINAL merges in rounds as partials are published (online max / sum); v6: 2 rounds; v5: wait for
  all (lane 1 fetching each on publish) + one unrolled weighted sum; all with sigmoid = 0.5 + 0.5 tanh.approx(x / 2)
  and the per-row vsc / l precomputed): job27 nsys (2 passes) v16f k37 97.30 / 97.25 | v4 96.21 / 96.26 | v6 96.13 /
  96.13 | v5 95.62 / 95.71; v32f 179.38 / 179.62 | 175.49 / 175.74 | 175.89 / 175.92 | 175.84 / 175.81.
  Rounds lose on b16 (several partials land close together: each round adds a poll + fetch trip). DEFAULT = v5
  (XC_RND 0, XC_CMP 0): v16f -1.6, v32f -3.7 us vs same-job k37.
- deliver_r3/ (frozen): xdk.cu (XC_END defaults), xdcore.cu (XD's, untouched), build.py (XD's + XC_* defines),
  xcheck.py, k37xc.py = build.py dev/coord/k36.py (k37 + XC end chain). Changes all under #if XC_END (XC_SPL / XC_SW
  start ideas default 0: XD's scope).
4b280187a46772e6aaad808654704b34  dev/xc/deliver_r3/build.py
1d92753f056edea591c0c012dd949c21  dev/xc/deliver_r3/k37xc.py
7c2188f216ba42bdd19acb9f02ef7734  dev/xc/deliver_r3/xcheck.py
2cb218f8a8612031a900c112bf7853ba  dev/xc/deliver_r3/xdcore.cu
3ac8c016ce49967384b964473dd5d8b4  dev/xc/deliver_r3/xdk.cu
- job28 (deliver_r3 on XD r1 / k36 base): tools/check.py 0 failures, xcheck 0; bench 30 reps v16f k37 96.39 / 96.63 |
  xc 94.81 / 94.98 (-1.6); v32f 179.27 / 178.99 | 175.67 / 175.38 (-3.6); nsys v16f 96.86 -> 95.71, v32f 179.26 -> 176.48.
- REBASE on XD deliver_r2 (coordinator): dev/xc/r4/ = XD r2 {xdk.cu, xdcore.cu, build.py, xcheck.py} + the XC_END
  patch (diff r1 -> deliver_r3 xdk.cu, applied with patch; 2 hunks by hand). XC_SPL / XC_SW removed (XD's start
  scope). Changes vs r3: the mid-kernel publish (fence + count of a first-segment partial) moved after the next
  segment's K/V prologue (XD r2 has no CTA barrier before the burst); the FINAL parameters (P, cf, slot, E, publish
  bit) computed by tid 256 after the burst (double divisions, exact; a 32-bit variant kept a predicate live across
  the loop -> spill); partial_cnt / pb kept only for !XC_END or traces; first_seg = (pre[b] <= c0) (b == b_begin made
  ptxas carry an induction register -> spill). Local nvcc 12.1: 0 B stack (XD r2: 24 B). Build: python3
  dev/xc/r4/build.py dev/k34.py dev/xc/r4/k34xc.py (control: dev/xd/deliver_r2/k34xd.py, reproduced md5 518c9a7d).
- job29 (r4 = XD r2 + XC_END, nsys 2 passes): v16f XD r2 96.26 / 96.27 | xc 95.49 / 95.17 (-0.9); v32f 178.80 / 179.23 |
  175.58 / 175.39 (-3.5); xc p90 v16f 96.0 vs XD 99.4-100.0 (the end-chain tail is gone). (The "XC_RND=1" runs there
  measured the default: XD's build.py passes only XD_* defines; r3's build.py also passed XC_*.)
- job30 (FROZEN deliver_r4 = k34 + XD r2 + XC_END): tools/check.py 0 failures, xcheck 0; bench 30 reps v16f k34 101.23
  | XD r2 96.24 / 97.00 | xc 95.72 / 96.09 (-0.7); v32f 182.66 | 178.31 / 178.54 | 174.83 / 175.37 (-3.3).
- REBASE on XD deliver_r3 (OVH 1024 / NEWW 512 weights; coordinator): dev/xc/r5 = XD r3 + the r2 -> r4 patch (3 hunks
  by hand: r3's per-segment u0 / u1 / complete lines and trace counters). XC logic unchanged: contributors = CTAs
  overlapping [pre[b], pre[b + 1]) (now incl. the OVH units: a CTA ending inside them gives an empty partial, m = -inf,
  l = 0, which the merge weights to 0); first_seg = pre[b] <= c0 still holds. 0 B stack (XD r3: 24 B, local nvcc).
  FROZEN deliver_r5/ = {xdk.cu, xdcore.cu (XD r3), build.py (XD r3), xcheck.py, k34xc.py = build.py dev/k34.py}.
a21d4a2dfeda76c85a792212853810bb  dev/xc/deliver_r4/build.py
adbd4505ba27b0614ee7991a0285e898  dev/xc/deliver_r4/k34xc.py
7c2188f216ba42bdd19acb9f02ef7734  dev/xc/deliver_r4/xcheck.py
4a23dba1ae8bef47fbd6adbe89fb4f99  dev/xc/deliver_r4/xdcore.cu
731ae3f1559d0fae3d42cc4d873765f4  dev/xc/deliver_r4/xdk.cu
a328af439a2ec7ef95a81fdae26e509c  dev/xc/deliver_r5/build.py
a191be6f1e3bff249a1955e60b3790a7  dev/xc/deliver_r5/k34xc.py
7c2188f216ba42bdd19acb9f02ef7734  dev/xc/deliver_r5/xcheck.py
4a23dba1ae8bef47fbd6adbe89fb4f99  dev/xc/deliver_r5/xdcore.cu
e72dd5fff23f2e7108b19ccfb2c84ea4  dev/xc/deliver_r5/xdk.cu
- job31 (FROZEN deliver_r5 = k34 + XD r3 + XC_END, fast container): check 0 failures, xcheck 0; nsys 2 passes v16f
  XD r3 92.61 / 92.83 | xc 92.55 / 92.90 (0.0); v32f 173.25 / 172.48 | 169.38 / 169.34 (-3.5); bench v16f k34 99.28 |
  XD r3 91.46 / 92.19 | xc 92.61 / 93.30 (+1.1 WORSE); v32f 179.25 | 170.91 / 171.15 | 168.71 / 168.32 (-2.5).
  => with ~10 contributors per request (B = 16) the single FINAL (~150 KB of partials through one SM + a 10-way
  weighted sum) loses to XD's 5 combiners once XD r3 rebalanced the split; with ~5 (B = 32) it wins.
- deliver_r6 (hybrid): build.py compiles the source twice (XC_END=1 / 0) and picks the XC build per call when
  6 B >= G (G = 132: B >= 22); XC_HYB=0/1 forces. Counters self-reset in both protocols, so alternating is safe.
- job32 (FROZEN deliver_r6 = k34 + XD r3 + XC_END hybrid): tools/check.py 0 failures; xcheck 0 (auto) and 0 (XC_HYB=1:
  XC end chain forced on every edge case); nsys 2 passes v16f XD r3 95.23 / 95.01 | r6 95.01 / 94.91 (same XD path);
  v32f 178.91 / 178.22 | 174.94 / 175.38 (-3.4); bench 30 reps v16f k34 100.41 | XD r3 94.87 / 94.82 | r6 94.75 /
  95.02 (0.0); v32f 182.67 | 178.49 / 178.09 | 174.57 / 174.70 (-3.65).
  Next levers (b16, ~10 contributors): XD's K-combiner protocol with XC's fp16 normalized partials (half the store /
  load bytes), tanh sigmoid epilogue and per-row 1 / l, or a 2-CTA FINAL split by rows; PLAIN publish path (tiles ->
  visible ~2 us: lane-merge barriers + store drain) is what the FINAL waits on when the last arrivals are close.
39fe1b69073ca74193ed8e0074bbf07d  dev/xc/deliver_r6/build.py
a926c8f3a93abbc5618e06fcfd3d190e  dev/xc/deliver_r6/k34xc.py
7c2188f216ba42bdd19acb9f02ef7734  dev/xc/deliver_r6/xcheck.py
4a23dba1ae8bef47fbd6adbe89fb4f99  dev/xc/deliver_r6/xdcore.cu
e72dd5fff23f2e7108b19ccfb2c84ea4  dev/xc/deliver_r6/xdk.cu

# Round 4 (XC owns XD's verify-full kernel): pipelined segment switch (dev/xc/r7, #if XC_PSW) -> MEASURED DEAD END
- Design: one ring / page-table stream across a segment boundary (persistent scur / siss; page-table ring slot =
  (local tile + so) & 7); when both segments have >= 4 DST tiles (CTA-uniform), the last DST pairs of the current
  segment issue the next segment's first DST - 2 stages and its first 2 DST - 2 page-table chunks (descriptor per lane
  in misc[48..55]); no cp_wait<0> at the switch; lane exchange in the V^T blocks (ones restored at the next setup);
  new block in a dedicated 16-KB stage (rows written at the setup, rows 8..63 zeroed at kernel start; +16 KB smem,
  230664 B); next request's Q rows L2-prefetched at the setup; first-segment partial published at the CTA end
  (XC_MIDPUB 0). Steady-state pairs run XD's exact body (loop split: main pairs + DST tail pairs, XB<> flag).
- job33: first version had the PSW smem init inside #if XC_END -> XD-protocol build (b16) read garbage flags (MMU
  faults); fixed (job34: xcheck 0 failures in XD / XC / forced modes). Traced switch (seg-0 tiles -> seg-1 first pair):
  PSW off 6.69 / 6.72 us (b16 / b32), PSW on 4.38 / 4.19 (new block 0.6-0.7, merge + store 1.1, seg-1 setup ~1.95).
- nsys: per-pair branch version (job34) v16f +1.0, v32f +0.5 vs PSW off; split-loop version (job35, 2 passes):
  v16f r6 92.67 / 92.30 | r7 92.66 / 92.66 | r7 PSW=0 92.32 / 92.13 | OVH 512 94.18 / 94.38;
  v32f 168.69 / 168.80 | 168.98 / 168.67 | 168.74 / 168.74 | 169.25 / 169.15. job33 (XC path) OVH 1024 / 512 / 256 / 0:
  175.81 / 175.95 / 176.85 / 180.29.
  => removing ~2.4 us of a straddler's bubble buys NOTHING end to end, and lowering OVH (moving work onto the
  straddlers) is worse. The kernel is HBM / power-throughput bound: while one CTA idles at its switch the others get
  its bandwidth / power; the bubble is not lost work. OVH only shapes the tail (straddlers must not be in it). Not
  frozen. (k34 run1f: same structure, presumably the same verdict; not measured.)
- Consequence for planning (throughput-bound kernel): mid-kernel bubbles are nearly free; what costs is (1) the start
  (every CTA idle at once: HBM idle until the burst at ~3.25 us), (2) the TAIL (after the median CTA finishes its
  tiles, fewer CTAs share the GPU: tiles-done p50 -> max 3.4 / 4.3 us at b16 / b32, plus the end chain), (3) total
  instructions / bytes. OVH only keeps the straddlers out of the tail.
- Next lever proposed (not built): NEIGHBOUR TAIL STEALING with no new partials. A CTA whose range ends inside request
  b's past tokens donates its last D tokens (e.g. 1024 = 8 lane-pairs) as a shared region with its right neighbour,
  when that neighbour's range lies inside b too (single segment: its partial for b is still live in registers at its
  end). Both claim 128-token lane blocks from the region through one self-resetting counter per boundary: the owner
  forward (after its core), the neighbour backward from the boundary (after its own core and its own donated tail).
  Contributor sets, P, E, partial slots and both end protocols stay unchanged (every token still lands in a static
  contributor's partial). Claims run >= 4 pairs ahead of consumption (page-table chunks are PD tiles ahead); the
  dynamic pairs use a separate loop copy (steady state keeps XD's body). Expected: tail spread / ~2 (chains of ~4 / ~8
  CTAs per request at b32 / b16) -> ~1.5-2.5 us per shape.

# Round 5 (dev/xc/r8): tail bound check, then the B = 16 combine
- job36 BOUND CHECK (timing only, #if XC_CUT, wrong output): every lane stops its tiles once XC_CUT % of the CTAs
  have finished theirs (leader polls a counter one pair ahead, flag through smem). nsys 2 passes, us:
    base (no instrumentation)   v16f 91.95 / 92.16   v32f 168.83 / 168.64
    c101 (instrumented, never)       94.64 / 94.82        171.42 / 171.68   (+2.7: the poll alone costs ~2.7 us!)
    c90                              94.66 / 94.50        171.49 / 171.52
    c75                              94.35 / 94.54        171.14 / 171.17
    c50                              94.64 / 94.61        170.48 / 170.40
  => dropping ALL work the late half still has when half the CTAs are done buys only -0.1 us (v16f) / -1.1 us
  (v32f) vs the same instrumented build. Perfect balancing redistributes that work instead of dropping it, so the
  gain of tail stealing is bounded by ~0.1 / ~1.1 us: NOT worth the build. (Side lesson: one strong global load +
  smem flag + early-exit branch per pair costs ~2.7 us: the steady-state loop must stay untouched.)
- job37 (r8: XC_XDF, B = 16 combine in XD's K-combiner protocol: fp16 normalized partials (17 KB, the XC_END format),
  one thread per item with up to 16 partials' (m, l, O / l) loads in flight, one weighted sum, tanh-sigmoid epilogue
  with vsc / l per row; store-flag epilogue for partial warps): xcheck XD mode 0 failures (XDF on and off),
  tools/check.py 0 failures; nsys 2 passes v16f r6 95.46 / 95.70 | r8 93.89 / 94.13 (-1.6) | r8 XDF=0 95.86 / 95.71;
  v32f unchanged (175.84 / 175.60 | 175.81 / 175.92 | 175.71 / 175.73: B = 32 uses the XC end chain).
- deliver_r8/ (FROZEN) = deliver_r6 + the XC_XDF blocks only (no PSW / CUT code: XB edits the start of deliver_r6 under
  XB_VSTART, so the diff vs r6 is end-chain only, 64 lines). md5s below.
acea68d0b5919ba4f5d2247d2c65e90f  dev/xc/deliver_r8/build.py
bf2ff112ee7c467fe7801a85b992c9f6  dev/xc/deliver_r8/k34xc.py
7c2188f216ba42bdd19acb9f02ef7734  dev/xc/deliver_r8/xcheck.py
4a23dba1ae8bef47fbd6adbe89fb4f99  dev/xc/deliver_r8/xdcore.cu
36a06f5a6229e32801efdf41ba2c727e  dev/xc/deliver_r8/xdk.cu
- job38 (FROZEN deliver_r8 validation): tools/check.py 0 failures; xcheck 0 failures (auto / XD forced / XC forced);
  nsys 2 passes v16f r6 92.82 / 92.51 -> r8 91.26 / 90.99 (-1.5); v32f 169.12 / 169.34 -> 169.31 / 169.18 (0).

# Round 6 (dev/xc/r9): PLAIN publish path with XG's tagged words -> MEASURED DEAD END (not frozen)
- XC_TAG (requests with P <= 7, i.e. B = 32): every 64-bit partial word carries a valid bit (O word: bit 0 of its
  first f16; ML word = (m, l | 1)), stored with st.relaxed.gpu.b64 into a separate ws region (zero at allocation,
  cleared by the FINAL after use, so a set bit can only come from this call); no fence and no count on the publish
  path (also not for the mid-kernel partial); the FINAL loads the others' words straight into registers (4 items x 6
  sources), re-polls only stale words, merges with its own partial in the V^T blocks. XC_TGATE: publishers add a
  fire-and-forget count hint and the FINAL polls only once the hint shows every other publish (XG: never poll an
  unwritten slot, its line is in DRAM).
- job39: xcheck 0 failures (auto, XC forced, XC forced + TAG 0), tools/check.py 0 failures; nsys v32f r8 174.96 /
  175.46 | tag 177.92 / 178.13 | TAG=0 176.62 / 177.07 (v16f unchanged 94.2).
- job40 (v32f, 2 passes): r8 175.02 / 175.28 | tag + gate 178.10 / 177.95 | tag, no gate 177.49 / 177.49 | TAG=0
  177.26 / 176.99.
  => the tagged path loses 0.3-0.9 us against its own TAG=0 build, and the r9 code (even with TAG=0, the tagged
  branches compiled in) costs ~1.9 us against r8: the end chain's code / register layout is that sensitive. The fence
  + count it removes is not on the critical path in a way that pays here (the FINAL is the last arrival: most partials
  were published long before; the polls of the late ones and the in-register merge are not faster than one cp.async
  staging round). Stand down on the publish path.

# Audit of dev/coord/k49.py on NEARBY shapes (coordinator task; driver dev/xc/audit.py)
- job41 / job42 correctness: 96-case grid (B 1..64 x q_len 1 / 8 x sliding / full, ctx 20000..131072, spread fixed /
  agentic / mixed, dist soft / coherent / normal, kv_scale 1 and 1/64 every 4th, 8 page-64 cases), each called twice
  on fresh copies: 0 failures (paths seen: xe, xb1, xg3, xd/xd, xd/xc, k34). Interleaved sequences (B=1 v1f / d1f at
  ctx 80000 / 20000 / 60000; verify full B 32 / 16 / 24 / 20; decode full B 32 / 16 / 8; sliding B 1 / 16 / 1 L8 / 12),
  2 rounds, every call checked: 0 failures. (job41's sliding half was swallowed by grep's binary-file detection;
  rerun in job42 with grep -a.)
- job42 perf (ctx 80000, judge distributions, 10 reps, us): k49 is never slower than k34 beyond noise (max 1.014).
  Below 2x SGLang: v-sliding B 64 (1.83x, k34's path, same as k34) and v-full B 2 (1.82x; k34 1.14x).
  Cliffs: decode sliding XE B <= 16 -> k34 at B 20 (5.68 -> 7.53 us); B = 1 -> 2 everywhere (only B = 1 has the XE /
  XA / XB / XG kernels): v-sliding 3.52 -> 9.11 (6.2x -> 2.45x), d-full 10.0 -> 19.3 (3.1x -> 2.05x), v-full 11.6 ->
  25.2 (3.0x -> 1.82x). k34's v-sliding B 16 (9.24) is faster than its B 12 (10.62) / B 20 (11.74).
- job43 perf at ctx 40000 / 100000 / 131072 (B 1 / 16 / 32): B = 1 full falls back to k34 beyond the B = 1 kernels'
  smem tile limits: v1f at ctx >= 100000 (XG3: n <= ~83K; 17.48 us, 2.19x vs 3.0x at 80000), d1f at 131072 (XB1:
  n <= ~101K; 17.41 us, 2.17x vs 2.96x at 100000). Everything else as at 80000.
  XE_BMAX probe (K2_DEFS=XE_BMAX=32): d-sliding B 20 6.24 vs k34 7.46, B 24 6.69 vs 8.07, B 32 11.25 vs 8.54 =>
  XE_BMAX = 24 removes the B 16 -> 20 cliff (-16 %), B 32 must stay on k34.
- job44: (a) k49 with K2_DEFS=XE_BMAX=24 on 15 decode-sliding boundary cases (B 17 / 18 / 20 / 22 / 24 x ctx 20000 /
  80000 / 131072, agentic / mixed): 0 failures. (b) r10 = deliver_r8 + XC_SMB (P > 16: up to 64 thinner combiners, the
  XDF combine split over thread groups + smem reduction): xcheck 0, small-B verify-full grid 18 cases 0, check.py 0;
  but nsys-free bench (10 reps) vs deliver_r8: v-full B 2 / 4 / 8 at ctx 80000 27.09 / 35.00 / 53.45 vs 25.31 / 34.09 /
  51.31 (+3..7 %), at ctx 40000 +3..12 % => dead end (more combiners only add waiters). Not proposed.
  Small-B verify full stays at 1.6-2.1x SGLang (B 2 / 4 / 8, ctx 40000-80000).

# Stacking pass (standing rule: every correct + faster change, no minimum): dev/xc/deliver_r10/
- Scan of r3-r9 for correct, faster, unfrozen pieces: only XC_RND=1 (B >= 22 FINAL merges the partials in rounds as
  they are published) measured faster on v32f (job26 -0.5 vs a non-tanh baseline, job27 -0.2 vs the frozen v5 path).
  Not faster: XC_CMP (compact code), XC_RND=2, XC_PSW (neutral +0.1), XC_TAG / XC_TGATE (+0.3..2.8), XC_SMB (+3..12 %);
  decode-sliding v5 (d32s +0.3 vs k34; d16s now XE). tanh epilogue + per-row 1 / l are already in deliver_r8's B >= 22
  path (r3 v5) and in the B = 16 path (XC_XDF).
- job45 (r11 = deliver_r8 + XC_RND default 1): tools/check.py 0 failures; xcheck 0 (auto, XC forced); nsys 2 copies x
  2 passes, us: v32f k51 175.31 / 175.02 / 175.50 / 175.25 | r8 174.82 / 175.44 / 174.95 / 175.01 | r11 174.80 / 174.87 /
  175.12 / 174.96 (mean -0.12 vs r8, -0.33 vs k51; pass a -0.29, pass b +0.06: within noise); v16f byte-identical
  kernel (XC_RND is inside #if XC_END; B = 16 runs the XC_END=0 build): 93.84 / 93.97 / 93.81 / 93.75 vs 94.05 / 94.00 /
  94.08 / 94.42 = the noise level (~0.3). B = 20 verify full: XD protocol 115.06 / 114.99 vs XC end chain 115.71 /
  115.78 -> the hybrid threshold (6 B >= G) stays.
- deliver_r10 (FROZEN) = deliver_r8 with XC_RND default 1 (one line). md5s below.
cccd6a6e3afc7251be66123a7276910a  dev/xc/deliver_r10/build.py
ae8d3682fc3cb9a3a53de739ff505d5d  dev/xc/deliver_r10/k34xc.py
7c2188f216ba42bdd19acb9f02ef7734  dev/xc/deliver_r10/xcheck.py
4a23dba1ae8bef47fbd6adbe89fb4f99  dev/xc/deliver_r10/xdcore.cu
b002101e40bdb0c71e0b2fe972a6e7ed  dev/xc/deliver_r10/xdk.cu
