# W4 (round 4): per-call time the judge counts beyond kernel duration

## 1. Judge timing anatomy (from subs/profiles/*.json, no GPU) - dev/w4/rejudge.py
Fields per benchmark: runs_ns / baseline_runs_ns (60 raw event times per side), replay_overhead {one_ns, two_ns,
overhead_ns, gate_ns ~299 us, parent_one/two}, outside_ns (+3 bursts), parent_ns, paired_ratio, median_ns.
- Verified exactly (k60s_check1 d1s): ratio_i = (sgl_i - ovh) / (ours_i - ovh), ovh = overhead_ns = 2*one_ns - two_ns
  (measured PER SHAPE, 3.68..4.64 us); paired = median over pairs (2i, 2i+1) of sqrt(r_2i r_2i+1);
  median_ns (our time) = (median(sgl) - ovh) / paired.
- one_ns 4.8-5.5 = trivial graph event window (our bench "fixed" 5.66); two - one = 0.9-1.3 us. Judge raw - one_ns
  reproduces our local bench (d1s k60 2.56-2.66 vs bench 2.56; d1f 9.18-9.86 vs 9.12; v1s 3.57-3.65 vs 3.24) => the
  judge sequence ~= tools/bench.py; it just subtracts ovh (~one - K_triv) instead of one: +~1 us/call on both sides.
- The per-shape ovh estimate is NOISY (sd ~0.25 us) and is pure luck: for ours ~3.5 us vs SGLang ~20 us it moves the
  ratio ~7%/0.3 us. Recomputed with a common ovh 4.15 (rejudge.py): k55 x2.9524 / x2.8571, k60s x2.8648 / x2.8555.
  v1s with common ovh: k55 4.55 / 4.68 vs k60 4.53 / 4.73 => the judge's v1s "+0.29" is overhead-estimate noise
  (raw medians equal: 8.72/8.70 vs 8.64/8.75). d1s common ovh: k55 3.50/3.63 vs k60 3.81/3.81 (+0.25; raw +0.1 with
  SGLang raw also +0.15 in those runs). d1f: k55 10.89/10.91 vs k60 10.77/10.42 (-0.3).
- The judge's own nsys profile is NOT cold-L2 timing (d1f profile 9.5-9.7 for k55 vs local cold nsys ~10.4): "timed -
  profiled" mixes warm/cold kernel time and is not a launch-cost measure.

## 2. EVEN/ODD REP PENALTY (the unmodeled per-call cost)
Every judge run alternates: even reps slower for BOTH sides (also in the SGLang-vs-SGLang calib run, both sides equal)
=> a per-PAIR condition before the first rep of each pair (fresh inputs / check), not order.
ours even-odd (us) k55a/k55b/k60a/k60b: d1s .24/.26/.83/.29  d1f .53/.32/.86/.38  d16s .24/.32/.35/.19  d32s 2.14/1.71/2.10/1.79
v1s .61/.69/.70/.72  v1f .59/.50/.38/.69  v16s .83/.50/.90/.83  v16f 1.36/1.07/1.73/1.65  v32s ~.55  (SGLang: d1s 1.0-1.5,
d32s 0.9-1.5, v16f ~0.1-0.2). Score if OUR even penalty vanished: +1.8-1.9% (k55), +1.9-2.7% (k60), d32s alone ~0.9%.
k60a's large d1s/d1f penalties came with a larger SGLang penalty in the same run (node condition).

## 3. Probe (GPU): dev/w4/probe_jseq.py (see job log below)

## Log
### job w4_j1 (16:02, ~2.5 GPU-min; out/20261029T160440_w4_j1): window anatomy + modes gen/gens/copy/none/sleep5ms/check
- Window (nsys, steady rep): gap = flush end -> first graph kernel start: k60 4.05-4.27 us vs trivial graph 4.03-4.14
  (+0.05-0.15); tail = event - gap - span: ours 0.45-0.64 = trivial 0.46-0.64; SGLang gap 5.5-6.2 (+1.5-2 over triv),
  tail ~0.05 (PDL-overlapped multi-kernel graph, ksum > span). => our window = trivial window + (span - 1.13) + ~0.1;
  NO removable launch / dispatch / end cost on our graphs (k60, all 5 shapes).
- Judge time ~= local cold nsys span + ~0.1 + half the judge's even-rep penalty (+ node variation).
- None of gen/gens/copy/none/sleep(5 ms)/check reproduces the judge's even-rep penalty (ours d32s +0.0..+0.34, v1s
  +0.0..+0.27; SGLang d1s ~0).
### job w4_j2 (16:09, ~2.3 GPU-min; out/20261029T161112_w4_j2): modes gens / gate (300 us _sleep) / free / gatefree / idle200
- REPRODUCED by `free` = fresh inputs in NEW allocations, freed + torch.cuda.empty_cache() (cudaFree => GPU TLB
  invalidation) before each pair. rep0 - rep1 (plain / nsys): ours d1s +0.77/+0.72, v1s +0.90/+0.53, d32s +2.66/+2.35,
  v16s +1.04/+1.81; SGLang d1s +1.94/+1.89, v1s +0.94/+1.25, d32s +2.10/+3.98, v16s +1.18/+1.47; triv +0.05-0.26.
  gate alone, idle 200 ms: no penalty (<= +0.37). Judge (even-odd): ours d1s .13-.83, v1s .6-.8, d32s 1.7-2.1, v16s .5-.9.
- nsys split of rep0 - rep1 under free: gap +0.42 d1s / +0.11 v1s / +0.80 d32s / +1.07 v16s (triv +0.32..+0.77:
  launch-path walks, not ours); SPAN (in-kernel) +0.51 d1s / +0.43 v1s / +2.13 d32s / +1.44 v16s; tail -0.06..-0.40.
  rep1 is still above steady (d32s span 10.06 vs 9.12 gens): the first rep does not fully re-warm.
- => The judge's per-call cost our bench never sees = cold GPU TLBs on the first rep of each pair (the harness frees /
  allocates memory per pair). Ours pays it mostly IN the kernel (page walks on first touch of KV-cache / page-table /
  output pages); larger for big scattered caches (B 16/32: 164-328 2-MB pages).
### TP = early TLB touch (dev/w4/build_tp.py): after the page-table loads, every compute warp issues predicated,
### register-free L2 prefetches, one per 2-MB page of K cache / V cache / out (branch-free: a `w == 3` branch made
### ptxas duplicate the shfl/K-V issue code, +296 instr). TP=1 prefetch.global.L2 (CCTL.E.PF2), TP=3 cp.async.bulk.prefetch
### (UBLKPF). SASS: xe_r1_1 1232 -> 1272, xe_b1s_8c6 1800 -> 1832, regs unchanged, 0 spills.
### job w4_j3 (16:20, ~2.5 GPU-min; out/20261029T162218_w4_j3): k60 x2, tp1 x2, tp3; modes free / gens / freewarm
- check.py tp1: 56 PASS, 0 failures.
- free rep0 (pooled copies, bootstrap 90% CI): d1s plain -0.46 [-1.33, +0.51], nsys event -0.02 [-0.19, +0.24],
  nsys span 3.81 -> 3.58; v1s plain -0.29 [-0.54, +0.29], nsys +0.03, span equal. gens (steady): d1s -0.05, v1s +0.03
  (v1s gens span 4.22 -> 4.37?). freewarm (upper bound, data pages touched outside the window): d1s rep0 8.18 vs free
  8.94 (k60), v1s 8.67 vs 9.12 => data-page walks are ~half of the cold-rep penalty; TP recovers at most part of d1s.
- Rule (span AND event below both k60 copies by >= 0.10): NOT met (d1s event flat in nsys; v1s nothing).
  -> job 4: judge-faithful bench (dev/w4/bench_jt.py, tlbcold 1 / 0) with 2 copies each, d1s + v1s.
### job w4_j4 (16:30, ~2.5 GPU-min): judge-faithful bench dev/w4/bench_jt.py (pairs on one input set, empty_cache per
### pair, judge ovh 2*one - two; one 5.54-5.66, two 7.26-7.49, ovh 3.81-3.84), 2 copies each
- tlbcold 1: d1s k60 4.48 / k60b 4.25 | tp1 4.02 / tp1b 3.99  => RULE MET (both TP < both k60 by >= 0.23)
             v1s k60 5.20 / k60b 4.92 | tp1 4.95 / tp1b 4.99  => not met
- tlbcold 0: d1s k60 4.08 / 4.12 | tp1 4.04 / 4.00 (-0.08); v1s 4.83 / 4.80 | 4.85 / 4.84 (+0.03)
- d1s cold-minus-warm (pair-averaged): k60 +0.27, TP -0.02 => TP removes the d1s cold-TLB pair penalty here.
## DELIVER_R1 (frozen): dev/w4/deliver_r1/build_tp.py (md5 c8519165) `python3 build_tp.py dev/coord/k60.py out` (defaults
## TP=1, only r1) -> dev/w4/deliver_r1/k60tp.py (md5 92d088c4): TP in xe_r1_1 (d1s) ONLY; xe_b1s_8c6 / xe_dm_8_4_2 SASS
## == k60's, xe_r1_1 SASS == the measured tp1 kernel (local NVRTC 13); stripped (dev/coord/strip_cuda_comments.py)
## 1,012,282 -> 827,522 bytes, stripped SASS identical. check.py on tp1 (superset): 56 PASS / 0 failures. Job 5 = final.
### job w4_j5 (16:36): NOT RUN - "Workspace ... has exceeded its spend limit" (Modal). GPU used by W4: ~9 min (j1-j4).
- PENDING for the coordinator when GPU is back: `bash dev/w4/j5.sh` (check.py on dev/w4/deliver_r1/k60tp.py + d1s
  confirmation with bench_jt cold/warm, 2 copies). Until then correctness rests on SASS identity: xe_r1_1 == tp1's
  (check.py 56/0), xe_b1s_8c6 / xe_dm_8_4_2 == k60's.
## Resume here
- Tools: dev/w4/rejudge.py (judge JSON -> common-ovh rescoring), dev/w4/probe_jseq.py + parse_jseq.py (judge-sequence
  probe with modes, nsys gap/span/tail split), dev/w4/bench_jt.py (judge-faithful bench: pairs, --tlbcold, judge ovh).
- Next levers (untested): TP-style early page touches in the kernels with the biggest cold spans (d32s attn_1_1_1_1e3
  +2.1-2.4, v16s attn_8_1_1_1x6 +1.0-1.4, v16f xd_v8f judge even-odd 1.1-1.8); ceiling ~ the freewarm share (~half of
  the cold-rep penalty is data-page walks). Rank kernel changes with bench_jt --tlbcold 1, not tools/bench.py, for these.
