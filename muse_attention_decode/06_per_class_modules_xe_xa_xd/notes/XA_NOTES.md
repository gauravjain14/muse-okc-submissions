# >>> ROUND 5 (JUMP MODE): deliver_r5/k34xa_r5.py md5 0bd38fb5: v1f nsys 13.94 / 14.00 (r4 14.21, k34 16.8); waves = XA8_DEFER 4 <<<
# >>> ROUND 4 (XA, v1f): frozen dev/xa/deliver_r4/k34xa_r4.py (md5 a546eef0), mod md5 0f039935 <<<
- v1f nsys 14.34-14.46 vs r3 14.82-14.94 vs k34 16.75-17.07: T128 (XD's 128-token step, PV retired inside the step)
  + QLATE (start barrier does not wait for Q; Q tracked by its own mbarrier) + MBP (one mbarrier per tile pair).
  bench: QLATE + T128 13.56 / 13.69 vs r3 14.17 vs k34 16.44 / 16.35.  Target 13.0 NOT reached.
- MEASUREMENT: K2_TRACE builds are +0.5..0.7 us slower than real builds under nsys and their phase deltas do NOT
  carry over (NBE, DEFER, T128 looked like -0.4..-0.7 in traces, -0.1..+0.6 in nsys). Decide on nsys only.
- nsys decomposition (timing-bound builds): default 14.88 = skeleton (start + gather + publish, no compute, no
  combine) 9.8 + exposed tile compute 3.5 + combine after the slowest arrival 1.8 (T128: 12.48 without combine).
- Dead in nsys: NBE (+0.6), DEFER (+0.25), LSH (0 / +0.3), NOF (+0.5), PROG progressive combine (+2.5: flag polls
  flood L2), DRY (+0.05), PROD NPW4 (0). d1f XA_PARE (parity load hoisted, XF audit): +0.1 (off).
# <<<

# >>> FOR XF / coordinator: v1f (verify 1 full) measured results, round 3 (XA, 2026-10-28) <<<
FINAL (r3fin, frozen dev/xa/deliver_r3/k34xa_r3.py md5 b7d8f9ac): verify 1 full 13.85 vs k34 15.92/15.97 (-2.1 us),
  nsys 14.82 vs 16.66; check.py + t_xa 0 failures. v1f ON by default (XA_NO8 opts out). + XA8_GEARLY (gate load at
  combiner entry) -0.2.
WINS (all in dev/xa/xa_mod.py, xa_b1f_8; same-job controls k34a/k34b):
- r3j4: k34 16.07/16.05 -> xa 14.12 (-1.9 us) = H16P + CLD + MONO on the j16 kernel; r3j6 14.19, r3j7 14.24 (NBE=0).
  * XA8_H16P: partial rows f16 O/l (64 f16) + float2 (M_eff, W = l 2^(M-REF)), 136 B/row: -1.0..-1.3 us (3 jobs).
  * XA8_CLD: combiner loads all issued before use. PITFALL: `for rr<5: if (q < G) {ld; use}` compiled to 5 dependent
    branches = 5 L2 round trips (~2 us). Clamp the index + weight 0 instead: -0.5 us.
  * XA8_MONO: u64 arrival counter never reset (call k = arrivals kG..kG+G-1, combiners wait for base+G): no DONE
    atomic / reset at the kernel end: -0.6 us.
KEY FINDING (r3j5): a warp has only ~16 LDGSTS (cp.async 16 B x 32 lanes) in flight; more block its issue until
  the first land. One producer warp: 2.1 us per 16 KB tile (all 10 tiles: 22 us). So the 512-thread burst (20 LDGSTS
  per warp) blocks every warp until ~4.0 us ("issued"), and tile 0 (landed long before) is consumed only then.
DEAD ENDS (bench, 40 reps, same-job controls):
- Producer warps XA8_PROD (all copies by 1/2/4 extra warps): 24.6 / 17.9 / 14.3 vs 14.5 (per-warp in-flight limit).
- XA8_DEFER (burst = tiles 0..6, tiles 7..9 issued after each WG's first PV): trace tile0 3.7 vs 4.3, loop end -0.4,
  but bench +0.2..+0.3 (d2/d3/d5) and +1.1 (d7).
- XA8_WGI (WG w issues tiles w, w+4, w+8; Q in registers): tile0 4.70 vs 4.19, bench +0.3.
- XA8_HELP (code-prefetch warp running the combine dry at the start): +0.4.  XA8_RED (red.add 128-row partials): +1.2.
- XA8_NBE (last CTA writes the new block's slots before the start barrier): trace better for the last CTA (the
  slowest CTA, tile0 5.4 vs 6.6) but bench +0.65..+0.8 in 3 jobs (not understood).
- XA8_DRY (first arrivals / early combiners run the combine dry to warm its code): +0.25 / +1.0.
OBSERVATIONS: the last CTA (c = G-1, new block) is the slowest-stored CTA in 6-10/10 reps; mbarrier.init x10 by
  warp 0 ~0.3-0.45 us (warp-0 stamp 0.9-1.06 vs 0.6 for other warps); tile ~1.0 us per 64-token tile per WG
  (compute-bound after tile 0: 4.95/5 tiles already landed); end chain loop-end -> done ~5 us (cold call of the
  noinline combiner ~0.8 us, slot loads ~0.8, warp-0 epilogue ~0.7).
# <<<

# XA (round 2): new dedicated kernels for batch-1 FULL layers (d1f q_len 1, v1f q_len 8)

Files: xa_mod.py (the module: _XA_SRC NVRTC source + host + kernel_fn wrapper), build.py (`python3 dev/xa/build.py
<base.py> <out.py>` appends xa_mod.py to any base), k34xa.py (= build of dev/k34.py), t_xa.py (B=1 full edge tests,
3 calls each: ctx 1..98000, dists, kv_scale 1/64, spread mixed/agentic = speculation misses), trace_xa.py (per-warp
globaltimer stamps XTR(k), XA_DEFS=K2_TRACE), run_nsys.py + nsys_loop.py + nsys_parse.py (copies of dev/wp's, any file,
xa_ kernels), lc.py (local nvcc 12.1 sm_90a compile: ptxas -v + SASS histogram), probe_fence.py (proxy-fence probe),
out/jN.txt (job logs).
Predicate (_xa_ok): B == 1, q_len 1, page 1, window 0, no RoPE, hq 16 / hkv 2 / d 64, width - 1 < 2^32 / 133 and
<= 47 tokens per warp (ctx <= ~99K). XA_OFF in K2_DEFS / env disables it.

## k34 baseline profile (job j1, k34 d1f / v1f, flush, us from first CTA start, dev/w9/trace9.py)
d1f (attn_1_0_1_1w): prefix 1.65 | pt-iss 1.97 | row-back 2.06 | issued 3.81 (max 4.61) | q/new 4.18 | tile0 4.74 |
  tiles med 8.49 max 9.04 | merged 9.52 / 10.08 | publish 10.53 / 11.01 | last arrival 12.11 | combined 12.58.
v1f (attn_8_0_1_1): issued 2.61 | q/new 2.94 | tile0 5.26 (max 5.97) | tiles 10.33 / 11.17 | merged 10.55 / 11.38 |
  segend 11.26 / 12.13 | publish 12.22 / 12.96 | comb0 13.30 / 13.98 | combined 14.69 / 15.33.
  warm-L2 v1f tiles still 9.28 / 10.46: v1f's tile phase is compute-chain bound, not gather bound.
nsys (job j2): k34 d1f 13.26 us (event-minus-empty 12.18). ncu (WP, k33 = same kernels): d1f DRAM 33% of peak,
barrier 22% / long_sb 16%; v1f barrier 29%, issue 37%, tensor 9%.
Critical-path budget (d1f, k34): start to issue done 3.8 (probe: 2.27) | gather/compute to tiles max 9.0 (probe CTA
completion 5.9) | end chain 3.5 (merge 1.0 + publish 1.0 + combine 1.5).

## Design v1 (job j2): xa_b1f_1, d1f
132 CTAs x 16 warps, 1 CTA/SM, 192 KB smem. Speculative split (n = width - 1, verified against seq_lens after the
burst; miss: drain own copies, redo). Warp w owns a contiguous ~38-token range (<= 48 slots: 3 groups of 16 tokens,
each group = K 2 KB + V 2 KB, XOR-swizzled 16-B chunks c ^ f(row), f(r) = (r&1)<<2 | (r>>1)&3: conflict-free K LDS.128
and ldmatrix). All copies issued at once in group order (one commit group per 16 tokens). Compute per group, FP16
mma.sync m16n8k16 (e4m3 -> f16 exact): S^T = K Q^T (A = K rows via LDS.128 + cvt, d permuted per thread; B = Q f16
with the softmax scale folded in), lazy running max (TAU 8, P <= 256 in f16), P^T by movmatrix.trans, O^T += V^T P^T
(A = V^T by ldmatrix.trans + prmt even/odd bytes + cvt). New token: last warp of the last CTA (RMSNorm k, e4m3,
cache rows, written into its slot after the zero-fill landed). End: warp partials to smem, 256-thread merge (max over
warps), fixed-reference (REF 64) red.add into NC=4 shards (call-parity double buffer), out-of-range CTAs store an
exact partial + list entry; last arrival sums shards, folds listed partials, RMSNorm x sigmoid(gate).
Result j2: t_xa 0 failures (all 21 cases x 3 calls; error 0.26% vs k34 2.7% on the scored shape), but
bench d1f k34a 12.43 / k34b 12.30 | xa 13.12 (+0.75). nsys xa 14.02 vs k34 13.26.
Trace (flush, per warp): pt landed 1.15 (max 1.94) | issued 2.98 (max 3.84) | group0 3.76 (max 6.00) | loop end
6.75 med, CTA max med 7.39 max 8.34 | CTA barrier 8.51 | arrived 10.22 (merge + red + atomic 1.7!) | last arrival
done 12.88 (epilogue 2.66!). Tile phase -0.7 vs k34 but the end chain is 4.5 us: cold code (unrolled merge / epilogue,
~40 KB kernel) is the suspect.

## v2 (jobs j3-j5): warm-up pass, row-major CTA merge, compact epilogue
- j3/j4: 32-35 us (!): Q staged with cp.async.cg by every warp of every CTA = 270K L1-bypassing 16-B requests on the
  same 2 KB of L2 -> ~18 us stall at the start (same hot-spot effect as WS's padding rule). Fixed with cp.async.ca
  (L1-allocating, the SM's 16 warps merge). Also: ptxas hoisted load consumers (new-token RMSNorm, 1/v_scale,
  min(seq, ..)) above the warm-up pass -> pinned with an empty asm volatile "+r" inside the post-pass block.
- probe_fence (fixed, 8 threads per 128-B row): fence.proxy.async right after a 20 MB burst costs 0.04 us and does NOT
  wait for the thread's other in-flight cp.async (v1: fence 4.26 = issue end, wait_all 5.47); after an mbarrier wait
  it took 0.54 us once (v3). (WF's "fence waits for the burst" does not reproduce in isolation.)
- j5 (d1f, 40 reps): k34a 12.25 / k34b 12.23 | xa (warm-up pass) 12.53 | xa XA_WARM=0 11.43 (-0.8 us).
  trace warm: s10 (after the pass) 2.34 vs 0.85 without -> the pass itself is a 1.5 us cold fetch at the start and
  delays the burst; NOT worth it. Default XA_WARM=0 from now.
  trace XA_WARM=0: start loads issued 0.56 | burst issued 3.49 med (max 5.14) | group 0 4.96 (max 7.01) | loop end
  7.50 med, CTA max med 8.58 max 9.38 | CTA barrier 9.60 | arrived 10.78 | done 11.46 (epilogue 0.7, was 2.66).
  t_xa 0 failures. Remaining: gather span first issue (~1.0) -> last landing (~9.2) = 8.2 us vs probe 5.98.
- j6 (same container): probe_shape 'a' 20.5 MB: span 5.97, first landing 2.85, issue 2.27, CTA done med 5.60 max 5.89
  ('c' 2/SM: 5.46 / 5.76). xa traces (CTA loop-end med / max): default (ZF=0) 8.35 / 9.02, XA_ZF=1 8.63 / 9.46,
  XA_NOCOMP (no attention math) 7.10 / 7.41 -> (A) my gather skeleton is ~1.5 us behind the probe (issue done med 3.46
  vs 2.27: Q wait + Q fragments sat before the burst), (B) the math adds ~1.6 us at the CTA max: the last groups of all
  16 warps land together and the SM is issue-bound (~200 SASS per 16-token group: 32 K cvt + 32 V cvt + 16 prmt + 16
  HMMA + softmax). ZF=0 (no zero-fill copies, V rows of empty slots zeroed by STS) is the default.
  v1f first smoke: t_xa q_len 8 quick 0 failures (scored 3.38e-02 = the e4m3-query reference's own 3.5%, bound 0.101).
  (@DEFS bench variants are valid: _XA_DEFS is read and the NVRTC module compiled at IMPORT, see _XA_MOD = _XAMod().)

## v3 (jobs j7-j8): burst before the Q wait; v1f kernel xa_b1f_8 (opt-in XA_V8)
- d1f: burst issued first, Q fragments / new token after it (Q staged in a separate 32 KB area, XA_QLOC=0).
  j7 (40 reps): k34a 12.53 / k34b 12.48 | xa 11.11 (-1.4). j8: k34a 12.39 / k34b 12.35 | xa 11.13, XA_EWARM=0 10.95.
  trace j8: pre-iss 0.88 | issued 3.39 (max 4.32) | grp0 4.48 | loop end med 7.55, CTA max med 8.00 max 8.80 |
  CTA barrier 9.01 | merged 9.62 | arrived 10.56 | done 11.15.
  j8 tools/check.py dev/xa/k34xa.py (d1f on xa, everything else k34): 0 failures. t_xa L=1,8 (XA_V8): 0 failures.
- v1f xa_b1f_8 (design in xa_mod.py header): j7 k34 16.39 / 16.12 vs xa 16.67 (FCP=0 16.55); j8 k34 16.04 / 16.02 vs
  xa EWARM=0 16.27. Trace j8: setup 0.83 | issued 4.02 (max 5.28) | tile0 4.32 (max 7.01) | loop end med 9.15, max
  10.19 (k34 tiles max 11.17) | parity merge 10.42 | partial rows stored max 12.00 | combiners released 12.3 | done
  15.25 (j7). WGs NEVER found their next tile already landed (test_wait at the loop top: 0.00 per warp): v1f's tile
  phase is gather-bound in this design, not compute-bound. The end chain (~5 us: 32 KB stores per CTA + release +
  128 row combiners spinning + one L2 round + epilogue) loses what the tile phase gains.
- Dead ends: XA_EWARM (non-final CTAs / waiting combiners run the final epilogue/combine dry to warm its code): d1f
  +0.2, v1f +0.9 (the dry pass is itself a cold fetch and delays the combiners' spin). Default off.

## v4 (jobs j9-j10): gather probe on the real inputs, d1f start ordering, v1f analysis
- dev/xa/probe_xa.py (j9, real decode-1 full inputs, graph us / CTA done med max from the first warp):
  WS 'a' 5.66-5.74 / 5.63 5.63 | xa skeleton 'w' 5.36 / 5.63 5.89 | 'wz' (src-size-0 fill) same | 'wbig' (229 KB smem)
  same | 'wq' (+ Q cp.async.ca, seq, scale loads before the burst) 5.82 / 6.14 6.14. => the warp-private skeleton is
  only ~0.26 us behind 'a'; the prefix loads ahead of the burst cost ~0.25; smem size irrelevant.
- XA_QDEP: the Q copies' address depends on the page-table entries via a run-time zero ((e0 ^ e1) & (qstride >> 16)):
  ptxas had put the 4 Q LDGSTS ahead of the critical page-table LDGs. (An asm "mov 0" is constant-folded by ptxas.)
  No prefix LDG is sunk below the K/V burst in either kernel (checked SASS; coordinator/XF lever 1).
- j10 d1f (40 reps): k34a 12.38 / k34b 12.40 | xa 11.02 (QDEP + EWARM=2: first 4 arrivals run the epilogue dry) |
  EWARM=0 11.15 | QDEP=0 11.26.
- v1f is COMPUTE-bound: j9 trace with per-tile stamps: the WGs found 4.49-4.58 of ~4.75 tiles already landed; tile
  starts per WG 4.29 / 5.39 / 6.43 / 7.46 / 8.48 -> ~1.04 us per tile per WG (~265 SASS per warp-tile, ~50% of the SMSP
  issue rate with 4 WGs). End chain: loop end max 10.6 -> parity merge 10.9 -> partial rows stored 12.4 (1.4) ->
  combiners released 12.7 -> done 16.1 (3.4 us: the combine's code is a cold DRAM fetch chain on 128 SMs at once).
- Dead ends: XA8_EARLYW (first 4 arrivals run the combine dry): combine 3.4 -> 1.7 us but the release came 2.2 us
  later, net +1.2 us. XA8_SKIPA (skip O rescale when no row max moved): neutral. XA8_PIPE (2-deep QK pipeline,
  unrolled x2): ptxas C7514 serializes every wgmma (even with path-independent group counting), and with 544 threads
  it spills; off.
- v5 (job j11, pending): XA8_B16 (QK accumulated in f16, W9H variant B with output-only asm for ACC 0: -16 cvt, scores
  in 16 regs; kernel 117 -> 95 regs) + XA8_HELP (a 17th warp runs the noinline xa8_comb dry at the kernel start: the
  combine's code is in L2 at the end; all CTA barriers of xa_b1f_8 are explicit 512-thread bar.sync 0, 512).
- j11 (v1f, 40 reps): k34a 16.20 / k34b 16.03 | xa (B16 + HELP) 16.72 | HELP=0 16.53 | B16=0 (+HELP: 544-thread
  register cap -> spills) 20.81. Trace: per-tile 1.07 us (B16 did not shorten the tile), combine still 3.8 us with the
  code-prefetch warp. Hypothesis: the 20 MB K/V stream evicts the prefetched code / the partial slots from L2 before
  the end chain -> XA_EVF (L2::evict_first createpolicy on every K/V cp.async, both kernels).
- XA_HELP1 (code-prefetch warp in d1f): the noinline call at 544 threads makes ptxas cap the kernel at 96 regs with
  in-loop spills -> off (default 0; host launches 544 threads only with XA_HELP1=1).
- j12 lost: the host launched xa_b1f_1 with 544 threads while the device default was XA_HELP1=0 (INVALID_VALUE).
- xa_b1f_1L (XA_L1, opt-in): lane-structured d1f (WS 'a' gather shape: 4 lanes x 3 stages of 64 tokens) with the QK
  of each tile on the lane's FP8 wgmma (m64n8k32 SS, Q^T E4M3 in smem: no per-warp K LDS / cvt / QK HMMA), per-warp
  softmax + mma.sync PV as xa_b1f_1. 106 regs, 0 spills.

## v6 (job j13): lane-structured d1f wins; v1f still behind
- j13 (40 reps, same job): decode 1 full k34a 12.21 / k34b 12.18 | xa (xa_b1f_1, EVF on) 10.73 | EVF=0 10.74 |
  **xa_b1f_1L (XA_L1) 10.19 (-2.0 us)**. verify 1 full k34a 15.89 / k34b 15.81 | xa v1f 16.57 | EVF=0 16.51 |
  HELP=0 + FCP=0 16.05 (best v1f variant) | (xa_l1's v1f = k34's: 15.73).
  Trace L1 (flush): burst issued 2.82 (max 3.44; warp-private 3.17 / 4.93) | first stage 3.55 | loop end med 6.62,
  CTA max med 7.17 max 7.89 (warp-private 8.78) | CTA barrier 8.42 | merged 9.02 | arrived 9.60 | done 10.45.
  -> the lane shape issues faster (3 page-table LDGs per thread, no Q staging before the burst), and the per-warp
  math loses the K LDS + 32 cvt + 8 HMMA per 16 tokens (QK = 4 wgmma m64n8k32 per lane-tile). Q is E4M3 (error 2.7%
  as k34, was 0.28% with the f16 Q of xa_b1f_1).
- Defaults now: XA_L1 (d1f on xa_b1f_1L), XA_EVF 1 (neutral), XA_EWARM 2, XA8_HELP 0. v1f still opt-in (XA_V8).
- v7 (next, built, untested): XA8_RED v1f end chain = d1f's model: merged 128-row partial at the fixed reference
  red.add'ed into 2 shards (call parity), last arrival runs the 128-row epilogue (4 threads per row); out-of-range CTAs
  store exact rows + list. Replaces 32 KB stores + 128 combiners (5 us of end chain in j11/j13).

## VERIFIED DELIVERABLE (job j14, dev/xa/final_job.sh '', defaults: d1f on xa_b1f_1L, v1f -> base k34)
- tools/check.py dev/xa/k34xa.py: 0 failures. t_xa (q_len 1 and 8 edge cases, 3 calls each): 0 failures.
- bench (40 reps, same job): decode 1 full k34a 12.33 / k34b 12.16 | xa 10.26 (-2.0 us); verify 1 full (unchanged
  code) 16.19 / 15.97 | 15.84.
- nsys kernel durations (judge-visible): d1f xa_b1f_1L 11.36 (p10 11.07, p90 11.62) vs k34 attn_1_0_1_1w 13.25;
  v1f attn_8_0_1_1 16.91 vs 17.02 (same kernel).
- md5 at this point: see `md5sum dev/xa/k34xa.py dev/xa/xa_mod.py` in the final report.

## Resume here (next levers, ranked)
1. d1f end chain on xa_b1f_1L (2.56 us from the slowest CTA's loop end to the output in j13): (a) write each warp's
   partial right after its loop into its own dead stage-0 K/V rows + stage-1 V rows (after the lane's last QK wgmma
   retired, the K region of the lane is dead; V rows are private to the warp) -> drops the extra CTA barrier before the
   partial stores; (b) merge with 512 threads; (c) the last arrival epilogue (~0.85 us) is cold-code bound.
2. d1f start: issued 2.82 med vs probe 'a' 1.79: Q^T/new-token work and the speculation compare sit on the path.
3. v1f: compute-bound tile phase (~1.05 us per 64-token tile per WG; fence.proxy.async = MEMBAR.ALL.CTA costs ~0.1 us
   each: XA8_FCP=0 -0.5 us end to end); end chain ~5 us with 128 combiners (cold, 3.4-3.8 us) -> XA8_RED (j15).
- j15 (v1f, 40 reps): k34a 16.64 / k34b 16.31 | XA8_RED (red.add of the 128-row partial + last-arrival epilogue)
  17.69 | 128 combiners (FCP=0) 16.50. RED trace: reds issued (barrier after) max 13.74 (L2 atomic throughput: the
  issue stalls), last arrival done 17.49 -> DEAD END for 128-row partials (as WB round 15). Default XA8_RED 0.
- Defaults after j15: v1f opt-in (XA_V8) with XA8_FCP 0, XA8_HELP 0, XA8_RED 0, XA8_B16 1; d1f xa_b1f_1L with
  XA_FCP1 1 (fence kept, as verified in j14), XA_L1P 0 (opt-in, measured in j16).
- j16 (exact file md5 262a78cb = defaults with the d1f stage fence OFF): check.py 0 failures, t_xa 0 failures
  (incl. XA_L1P=1). bench: decode 1 full k34a 12.45 / k34b 12.49 | xa 10.57 | XA_L1P=1 11.03 (worse: dead end) |
  XA_V8 10.60; verify 1 full k34 16.48 / 16.38 | xa (k34 code) 16.44 | XA_V8 (xa_b1f_8) 16.55 (par). nsys d1f
  xa_b1f_1L 11.26 (p10 10.98 p90 11.55) vs k34 13.22.
  -> FCP=0 does not speed d1f up (vs k34: j14 -1.98 with the fence, j16 -1.90 without): the FINAL build keeps the
  fence (XA_FCP1 1), i.e. the d1f code verified in j14 (check.py 0 failures there).

## FINAL STATE (round 2, first hand-off)
dev/xa/k34xa.py = python3 dev/xa/build.py dev/k34.py dev/xa/k34xa.py (md5 in the hand-off). d1f -> xa_b1f_1L
(-1.9..-2.0 us local, nsys 11.3 vs 13.2). v1f -> base k34 (xa_b1f_8 opt-in via K2_DEFS XA_V8: at par).

## Integration rule (coordinator, round 3): frozen deliverables
- dev/xa/deliver_r2/ = round-2 hand-off, read-only: xa_mod.py (= dev/coord/snap/xa_mod.py), build.py, k34xa.py
  (md5 7ccee3bc, reproduced by building the snapshot). dev/xa/k34xa.py restored to that file and is not rebuilt again.
- Working builds from round 3 on: dev/xa/k34xa_w.py (python3 dev/xa/build.py dev/k34.py dev/xa/k34xa_w.py).
  (r3j1 ran on a working build uploaded as k34xa.py md5 caeb2945 before this rule.)

## Round 3 (v1f focus)
- r3j1 (first attempt) lost: dev/coord/k36.py modified during the Modal upload (--cmd mentioned dev/coord). Controls
  now dev/k34.py + dev/xa/k34c.py (byte-identical copy, md5 1d66a1bf) -> no dev/coord upload.
- Per-tile mbarrier counts (512, + 128 only on the last CTA's new-block tiles; WG 0 used to arrive on every tile after
  issuing all its tiles): tile 0 med 4.19 (was 4.26) -> NOT the cause of the late first tile.
- r3j1b (40 reps): verify 1 full k34a 16.12 / k34b 16.03 | xa (fp32 partials) 16.25 | **XA8_H16P 15.07 (-1.0 us)**.
  XA8_H16P: partial rows = f16 O / l (|O / l| <= max |v code|) + (M_eff, W = l 2^(M - REF)), 136 B per row; combiner
  sums W (O / l). Trace (fp32 partials): combiners released 12.59 -> loads done 15.06 (2.5 us: 4.6 MB of slot reads by
  128 CTAs at once) -> reduction 15.58 -> epilogue 15.71 -> done 16.30. t_xa q_len 8 (fp32 and H16P): 0 failures.
- The late first tile (4.2 us vs ~2.6-2.8 in probe 'a'): the start waited for the Q cp.async + a CTA barrier + the Q
  fragment build BEFORE the burst, and the 512-thread all-tiles issue (20 copies + 10 mbarrier arrive-ons per thread)
  only finished at ~3.95 us. XA8_WGI: WG w issues tiles w, w + 4, w + 8 (8 copies per thread per tile; mbarrier count
  128), Q loaded to registers (issued behind the page-table loads via a run-time zero dependency), fragments built after
  the burst; the start barrier waits for no load. (XC: mbarrier.init serializes ~25 ns each per SM; the 10 inits now
  overlap the page-table load latency.)
- r3j2 (40 reps): k34a 16.60 / k34b 16.42 | WGI=1 + H16P 15.49 | **WGI=0 + H16P 15.22**. WGI trace: tile0 med 4.70
  (worse than 4.19) -> XA8_WGI DEAD END (default 0). XA8_H16P now default 1.
- r3j3 (40 reps): k34a 16.23 / k34b 15.93 | **xa (H16P) 14.80** | + XA8_HELP=1 15.21 (helper still worse).
  -> H16P: -1.0 / -1.3 / -1.3 us vs the mean control over 3 jobs.
- Coordinator (XC): mbarrier.init serializes ~25 ns each per SM. v1f start path: 10 inits by warp 0 (~0.25 us) + one
  fence.proxy.async per warp, both overlapping the Q cp.async wait (setup stamp 0.86 med). d1f xa_b1f_1L: no
  mbarriers (commit groups). Stamps 20 (mbinit) / 21 (fenced) / 22 (q-wait) added to measure it (r3j4).
- SASS of xa8_comb (r3j3): the 5 guarded load rounds (`if (q < G)`) compiled to 5 dependent branches, each
  LDG -> use: 5 L2 round trips = the 2-2.7 us "c-loads" phase. XA8_CLD (default 1): clamped index, weight 0 past G,
  all 10 loads issued first (SASS checked: 10 LDG back to back).
- XA8_MONO (default 1): u64 arrival counter never reset (call k = arrivals k G .. k G + G - 1; combiners wait for
  base + G): drops the DONE atomic round trip + counter reset at the kernel end (c-epi -> comb-done was 0.6 us).
- r3j4 (40 reps): k34a 16.07 / k34b 16.05 | **xa (H16P + CLD + MONO) 14.12 (-1.94 us)** | CLD=0 14.60 | MONO=0 14.71.
  t_xa L 1,8 quick + L 8 full: 0 failures. Trace: comb-rel 12.26 -> c-entry 12.96 (cold call) -> c-loads 13.79 ->
  done 14.61 (was 16.30 in r3j1b).
  Per CTA: the LAST CTA (c131, new block) is the slowest stored in 7/10 reps: its tile0 7.28 vs median 4.50 (WG 0 ran
  the new-block code after the burst: dependent cache_loc load + global stores + fence.proxy.async (MEMBAR.ALL.CTA)
  before its first tile). warp-0 mbinit stamp med 1.06 vs 0.58 for the other warps: 10 inits ~0.45 us on warp 0.
  KEY: "issued" med 4.05 and tile0 med 4.30 while 4.95/5 tiles were already landed when reached -> the 512-thread
  burst issue (throttled by the memory system) gates tile 0, not its data.
- XA8_PROD (default 1): warp 16 (XA8_NPW warps) = producer: all copies in tile order from a smem page table
  (X8_PT, filled by the consumers before the start barrier), mbarrier count 32 NPW; the new block's slots are written
  before the start barrier (its proxy fence covers them; no extra arrivals), its cache rows after the start;
  speculation miss: drain, barrier 5 (all threads), re-arm, reload the page table, rewrite the slots, re-issue.
  92 regs at 544 threads, 0 spills.
- r3j5 (40 reps): k34a 16.15 / k34b 16.19 | PROD NPW=1 24.57 | NPW=2 17.86 | NPW=4 14.26 | PROD=0 (r3j4 config) 14.49.
  Trace NPW=1: tile-0 copies issued only at 3.90, all at 22.0 (~2.1 us per 32 warp-LDGSTS: a per-WARP in-flight
  limit of ~16 LDGSTS instructions, i.e. ~8 KB per warp). -> the 512-thread burst (20 LDGSTS per warp) blocks each
  warp at ~16 in flight until the first copies complete: that is the "issued 4.05" wall. PROD default 0.
- XA8_DEFER (default 3): the burst issues tiles 0..6 (14 LDGSTS per warp, under the limit), each WG issues its share
  of tiles 7..9 right after its first PV. (117 regs, 0 spills.)
- r3j6 (40 reps): k34 16.03 / 16.05 | DEFER 0 14.19 | 2 14.49 | 3 14.39 | 5 14.51 | 7 15.31 -> DEFER dead end (default
  0) although the trace has tile0 3.70 vs 4.30 and issued 2.72 vs 4.05. The last CTA is the slowest in 10/10 reps.
- r3j7 (40 reps): k34 16.11 / 16.10 | NBE=1 14.97 | NBE=0 14.24 | NBE + DEFER 3 / 2: 15.11 / 15.18. Trace NBE=1:
  last CTA tile0 5.41 (was 6.64), stored 11.74. Bench says +0.7 -> NBE default 0 (not understood).
- r3j8 bench INVALID: a candidate named "base" collides with bench.py's SGLang baseline label (sglang = base column).
  (relative: GEARLY only 15.86 | + DRY 1 16.07 | DRY 2 17.10 | NBE 1 16.88 vs k34 18.48 / 18.57 on a slow container.)
  Warm-L2 trace = flush trace - ~0.3 us everywhere (cold code / data is not the end-chain problem per se).
- Host: v1f now ON by default (XA_NO8 opts out); XA_V8 in defs is harmless.
- r3j9 (40 reps): k34 16.10 / 16.06 | DRY 0 GEARLY 0: 14.25 | **GEARLY 1 DRY 0: 14.03 (-2.05 us)** | DRY 1: 14.29 |
  NBE 1: 14.90. -> defaults GEARLY 1, DRY 0 (dead end: +0.25), NBE 0 (dead end, 3rd time +0.65).
- Final defaults round 3: H16P 1, CLD 1, MONO 1, GEARLY 1, DRY 0, NBE 0, DEFER 0, PROD 0, WGI 0, HELP 0, RED 0, FCP 0.

## Round 3 hand-off (frozen: dev/xa/deliver_r3/, read-only)
- deliver_r3/k34xa_r3.py (md5 b7d8f9ac) = python3 dev/xa/build.py dev/k34.py <out> deliver_r3/xa_mod.py (md5 0681c4ef;
  rebuild verified byte-identical). d1f = xa_b1f_1L exactly as round 2 (only K2_TRACE macros differ); v1f = xa_b1f_8 ON
  by default (XA_NO8 -> base k34). Final verification: dev/xa/out/r3fin.txt (final_job3.sh).
- Working build from here on: dev/xa/k34xa_w.py.
## Resume here (round 3 -> next)
1. Last CTA (c = G-1, new block) is the slowest-stored CTA in 6-10/10 reps (tile0 6.6-7.2 vs median 4.3-4.5).
   NBE fixed it in traces but costs +0.65 in the bench (3 jobs) -> not understood; try instead: give the last CTA
   fewer old tokens (split weighting), or move the new-block work to the end of its loop (it only feeds its last tile).
2. Burst wall: ~16 LDGSTS in flight per warp -> issue ends ~4.0 us, tile 0 consumed at 4.3 (landed earlier). DEFER
   moves tile0 to 3.7 in the trace but not the bench (power cap? XD: verify full is power-bound at ~1650-1750 MHz).
3. Tile cost ~1.0 us per 64-token tile per WG: XD's T128 step (one max / alpha / fold / fence / bar per 128 tokens)
   cut resident compute 18 % in XD's kernel.
4. End chain (r3j8): stored med 10.3 -> release 12.2 (slowest CTA) -> c-entry 12.8 (cold noinline call) -> loads 13.6
   -> done 14.3.
- r3fin (final_job3.sh on md5 b7d8f9ac): t_xa quick 0 failures; tools/check.py 0 failures; t_xa L 1,8 full 0 failures.
  bench 40 reps: verify 1 full k34a 15.92 / k34b 15.97 | **xa 13.85 (-2.1 us)** | XA_NO8 15.66; decode 1 full
  12.26 / 12.22 | xa 10.21 (round-2 d1f). nsys v1f: xa_b1f_8 14.82 (p10 14.66, p90 15.17) vs k34 attn_8_0_1_1 16.66.

## Round 4 (v1f, target nsys <= 13.0 from 14.82; decide on nsys)
- XA8_LSH (last CTA carries LSH fewer old tokens): r4j1 trace LSH 128: the last CTA is now the fastest stored
  (10.62) but random other CTAs are the tail (11.3) -> comb-rel unchanged; nsys 14.88-14.96 for LSH 0..256 (k34
  16.83) -> no gain alone.
- ncu (r4j1, default): stall samples barrier 25 % (the spin/arrival barrier, loop-end WG imbalance), wait 13, long
  sb 11, short sb 10; no_instructions only 2 % -> cold code is NOT a main cost. SM 1.87 GHz, 96 regs.
- r4j2 nsys (judge-visible): default 14.91 | NBE 15.54 | DEFER 3 15.17 | DEFER 2 15.23 | NBE + DEFER 15.84 |
  LSH 128 + DEFER 3 14.78 | DEFER 4 15.14 | DRY 14.96 | GEARLY 0 15.10 | PROD NPW 4 14.94 | k34 16.77.
- XA8_T128 (T128 steps: first tile alone, then same-parity pairs (i, i+2), one softmax round / fence / barrier per
  pair, PV 4 k-steps; V^T single-buffered in the WG's 4 blocks). With the previous PV overlapping the next step
  (wait_group 1 before V^T) ptxas serializes EVERY wgmma (C7514, SASS: DEPBAR after each QGMMA). T128W 1 (PV retired
  inside the step, as XD's tile2): no C7514, 119 regs. T128W 2 (wait 0 before V^T): 128 regs, tiny spill.
- r4j3: T128W 1 t_xa 0 failures. Trace (flush): loop end 8.58 (was ~9.3), per pair ~1.4 us (= 0.7 us per 64 tokens
  vs 1.0), comb-done 13.47 (was 14.1-14.3). BUT nsys only 14.78 / 14.75 vs 14.91 (-0.15); bench 30 reps: xa 14.01 |
  T128W1 13.69 | T128W2 13.92 | LSH128 + DEFER3 14.03 (k34 15.97 / 16.13). nsys T128W1 + LSH 128: 14.70.
  -> trace gains do not carry to nsys (3rd time: NBE, DEFER, T128). Hypothesis: the trace reps run after CPU gaps
  (cool GPU, boost clock) while bench / nsys loops are back-to-back flush + replay (power cap). Added trace --mode
  soak (30 back-to-back flush + replay, the last traced) and a warp-0 clock64 stamp (MHz start -> loop end).
- XA8_NOF (no parity merge: each WG publishes its own rows as partial 2c + par; combiner merges 2G, 9 load rounds).
  With the T64 loop ptxas serializes the wgmmas (C7515); with T128W1: clean, 119 regs.
- r4j4: soak trace (30 back-to-back flush + replay) = flush trace; SM clock 1806-1837 MHz in both -> no power-cap
  difference between trace and nsys loops. nsys: NOF (T128) 15.34-15.39 (worse: dead end), T128 14.78.
- r4j5 / r4j6: TRACE BUILDS ARE NOT REPRESENTATIVE: nsys default 14.86-15.2 vs the same with K2_TRACE 15.57-15.71
  (+0.5..0.7, even the lite trace with 7 stamps outside loops); T128 14.75-14.82 vs 14.86-15.12 traced. So trace-only
  gains (NBE, DEFER, T128's -0.7) mostly vanish in the real build. Decide on nsys of non-trace builds only.
  K2_TRACE_LITE (stamps 0/1/2/4/6/7/8 only) added.
- XA8_PROG (progressive combine): every CTA st.release's its call tag into flag[c] after its arrival atomic;
  combiners skip the all-arrived wait and poll their partials' flags, loading + accumulating each partial as soon
  as it is published (bulk loads + the call's cold fetch overlap the wait).
- r4j7: XA8_PROG DEAD END: nsys 17.4 (+2.5; 128 combiners x 132 flag polls flood L2), with T128 16.9-17.0.
- r4j8 lost (my fault): an XA8_DEFER + XA8_TB_NOTILE variant deadlocks (deferred copies skipped); the orphaned
  container held a slot (app stop denied). RULE: every variant in --cmd under its own `timeout 120`; #error now
  forbids that combination; test-only knobs stay out of benchmark jobs.
- r4j8b nsys decomposition (timing-bound builds, wrong output): default 14.88 | no combine 13.07 | no tile compute
  11.42 | neither 9.79 (= start + gather + publish skeleton) | T128 14.82 | T128 no combine 12.48.
  -> combine (after the slowest arrival) 1.8 us (2.3 with T128), exposed tile compute 3.5 us (T128 -0.6 of it),
  skeleton 9.8 us (gather floor ~5.4 of it).
- XA8_QLATE: the start barrier no longer waits for Q (Q cp.async tracked by an 11th mbarrier, arrive after the
  barrier; fragments built after the burst).
- r4j9 nsys: default 14.94 | QLATE 15.07 | T128W1 14.75 | **QLATE + T128W1 14.50 / 14.54** | skeleton QLATE 9.76 vs
  9.90. t_xa (QLATE, QLATE + T128) 0 failures.
- r4j10 nsys: QLATE + T128W1 14.50 / 14.50 | + NOCOMB 12.40 (combine 2.1) | + LSH 128 14.82 | + NBE 15.50 |
  + GEARLY 0 14.78 | + SKIPA 0 14.43 | default (r3) 15.17 | k34 17.07. bench 40 reps: k34 16.44 / 16.35 | r3 xa 14.17
  | QLATE + T128 13.56 / 13.69 (two copies). -> new defaults QLATE 1, T128 1, T128W 1.
- XA8_MBP (one mbarrier per tile pair: 5 inits) built, r4j11.
- d1f (coordinator, XF audit): XA_PARE (default 1): xa_b1f_1L loads the accumulator parity (XW_PAR) in the prefix
  (pinned after the burst by an empty asm "+r") instead of after the end-chain bar.red (one L2 round trip in front
  of the red.adds). SASS: the LDG is now at 0x1c0, before the first LDGSTS.
- r4j11 nsys: d1f PARE 11.52 / 11.55 vs PARE 0 11.42 / 11.46 (k34 13.22) -> PARE off. v1f new defaults 14.46 / 14.43,
  + MBP 14.40 / 14.34 (adopted), r3 config 14.91. t_xa L 1,8 full + MBP: 0 failures.
- r4fin (final_job4.sh on deliver_r4/k34xa_r4.py md5 a546eef0): t_xa quick + full 0 failures, tools/check.py 0
  failures. bench 40 reps: verify 1 full k34 16.16 / 16.15 | r3 14.13 | **xa 13.40** | PARE 13.48; decode 1 full k34
  12.57 / 12.40 | r3 10.37 | xa 10.42 | PARE 10.67. nsys v1f: k34 17.06 | r3 15.01 | xa 14.46 / 14.37 | PARE 14.43;
  d1f: k34 13.31 | r3 11.34 | xa 11.36 / 11.38 | PARE 11.44.
## Resume here (after round 4)
1. v1f combine after the slowest arrival ~2.1 us (nsys bound): faster release detection (2 staggered polls), fewer
   partials per row (cluster DSMEM pre-reduction), or a list-driven progressive combine (counter poll only; the
   per-partial flag polling of PROG floods L2).
2. Exposed tile compute ~2.9 us with T128: the PV overlap (T128W 0) needs SS-mode PV (P via smem, W8 notes) to avoid
   ptxas's C7514 serialization, but smem is full (6 KB free).
3. Skeleton 9.8 us (start + 20.5 MB gather + publish): the gather floor is ~5.4 us.

## Round 5 = JUMP MODE (coordinator 17:00): v1f from the floor, nsys target <= 11.0
- Waves (XA8_DEFER with T128: the burst issues tiles < 10 - DEFER, the rest after each WG's step 0; XA8_W1 = start
  of a 2nd wave issued after step 1). r4j12 nsys (2 copies): default 14.43 / 14.34 | DEFER 2 14.29 / 14.24 |
  **DEFER 4 14.02 / 13.98** | DEFER 2 + NOCOMB 12.37. t_xa DEFER 2 / 4: 0 failures.
- r4j13 nsys (2 copies): DEFER 4 14.18 / 14.08 | DEFER 6 15.74 / 15.82 | DEFER 8 16.38 / 16.48 | DEFER 8 + W1 6
  17.95 / 17.98 | DEFER 6 + W1 7 16.54 / 16.62 (k34 16.77). -> DEFER 4 is the maximum for this step layout (single
  step 0, pairs (par+2, par+4), (par+6, par+8)): deferred tiles must not be needed before step 2. Default DEFER 4.
- deliver_r5 frozen (md5 see r5fin): T128 + QLATE + MBP + DEFER 4 (waves) on top of round 3.
- r5fin (final_job5.sh on deliver_r5/k34xa_r5.py md5 0bd38fb5, mod md5 6e648a42): t_xa quick + full 0 failures,
  tools/check.py 0 failures. bench 40 reps: verify 1 full k34 15.77 / 15.82 | r4 13.10 | **r5 12.85 / 12.75**;
  decode 1 full k34 12.17 / 12.01 | r4 10.11 | r5 10.14 / 10.09. nsys v1f: k34 16.75 / 16.86 | r4 14.21 / 14.21 |
  **r5 13.94 / 14.00**; d1f r5 11.23.
## Resume here (after round 5)
- v1f floor budget (nsys): skeleton (start + gather + publish) 9.8 -> floor ~8.5; exposed compute ~2.5 (waves
  saved ~0.4); combine after the slowest arrival ~1.9-2.1. Next structural candidates: (1) combine on MMA-free
  SMs / faster release (clock cap 1.80 vs 1.98 GHz with MMA, sibling 3.10); (2) waves need slack: a step layout
  whose deferred tiles are consumed later (e.g. pairs first, single last) would allow DEFER 6; (3) a smaller
  publish (the in-CTA parity merge + fold ~0.7 us).
