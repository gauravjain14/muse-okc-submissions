# W7 NOTES: cold-TLB first-rep penalty on d32s (attn_1_1_1_1e3 = runE1 C=3), d16s (xe_dm_8_4_2), d16f / d32f (attn_1_0_1_0)
Owner W7 (round 4), dir dev/w7 only. BASE = dev/coord/k65.py (md5 b76344ec = BEST): coordinator 19:3x REJECTED k66 (W5 TKMF hangs
on the judge). j2 / j3 files were k66-based (e3 + its launch text identical in k65 / k66); from j4 on everything is k65-based.
build.py default on k65 -> module cubin 3578cad0 == k65's (checked). GPU cap 30 min. W5 owns run1f's END chain
(dev/w5): W7 touches only kernel starts there.

## Resume here
- FROZEN dev/w7/deliver_r1/ (00:0x): build.py on k65 -> k65w7.py md5 97866cf3 (v87: e3 early real touches). bench_jt d32s
  tlbcold 1 rule met on TWO nodes (j6 -0.34..-0.70, j7 -0.39..-0.67), warm +0.03; check 56/0, t_e3 (9 C=3 cases) 0,
  t_rep 60/0 no hang. e2 (earlier pt touch) not better (-0.08). README.txt there. GPU ~18 of 30 (incl. container starts).
- d16s: no in-kernel cold penalty on the penalty node (j1 trace, every stamp warm == free) -> nothing to fix in xe_dm.
- j1 DONE (1.6 GPU-min; trace part only, the localization part crashed on SEG = freed segments, fixed). j2 READY:
  dev/w7/build.py (k66 builder, --e3 M touch mask, --var multi-variant measurement build -> dev/w7/var.py md5 3205ebd7),
  dev/w7/probe.py (variants, modes, trace), dev/w7/sassw7.py (local SASS). GPU used: 1.6 of 30.

## Judge even-odd (ours / SGLang, us; subs/profiles raw runs, 11 judge runs k49..k63s)
d32s +1.7..+2.1 (since runE1; k4/k6/k8 era 0.6-1.0) | d16s +0.2..+0.35 | d16f -0.1..+0.7 | d32f 0..+0.7 (noise ~0.5 on full).
Score value of removing ALL of it (paired = sqrt(cold x warm) => half counts): d32s ~0.95 us x 0.25 = 0.24 %;
d16s ~0.15 x 0.45 = 0.07 %; d16f / d32f ~0.2 x 0.158 / 0.087 = ~0.03 / 0.02 %. => d32s first, d16s second, full last.

## Buffers each kernel touches (code audit, k65)
- NOTE: k64+ (so k66) builds e3 with W3_TPE=2 (prefetch.global.L2 per 2-MB page of K/V/pt/out right after the page-table
  entry loads; W3 j5 'not kept' but the coordinator took section 0 from qa.py). So the e3 control already has TP mode 2.
- runE1 e3 (d32s, grid 96 x 1, cluster 3, 512 thr): prefix seq_lens[b], qkv (Q rows, new K / V, gates of owned rows),
  inv_freq, cache_loc[b], k_scale / v_scale (cp.async); then page_table row b (window entries) -> K / V cache gathers
  (128-B rows, all tiles at once, cp.async + mbarrier); new token K / V -> cache row crow (st) + tile slot; DSMEM push;
  owners: bf16 out stores; exit. No global workspace (ws only for K2_TRACE). Params (gpos / gnp) by value (const bank).
- xe_dm_8_4_2 (d16s, grid 8 x 32, cluster 8, 160 thr, ~2 CTA / SM): prep warp: seq, ifr, qkv (Q, new K / V, gate), cloc,
  ksc / vsc; compute warps: seq -> pt -> K / V register / cp.async gathers; owners: out. XE trace buf p.tr (trace only).
- attn_1_0_1_0 (d16f / d32f): FLAT run1 global-workspace combine (ws partials / counters) - W5 territory at the end.

## Job w7_j1 - DECISION RULES (written before the job; dev/w7/j1.sh; diagnostic, nothing frozen from it)
Per shape, d = median over 16 pairs of (rep0 - rep1); removed(S) = d(free) - d(free + touch S outside the window).
- S CARRIES penalty iff removed(S) >= 0.3 us (d32s) / >= 0.1 us (d16s) / >= 0.5 (full); the in-kernel fix targets those.
- removed(ALL) >= 70 % of d(free): the penalty is data-page walks (fixable in-kernel); < 50 %: mostly code / params / launch
  (not legally fixable) -> stop on that shape.
- removed(KV1) >= 70 % of removed(KV): a SHARED TLB level fills from one SM -> one touch per page per call suffices.
- removed(KVpf) < 50 % of removed(KV): prefetch.global.L2 does not fill the TLB (explains W3 / W1 TP misses) -> use real loads.
- |removed(ALL2m) - removed(ALL)| <= 0.2: 2-MB pages (2-MB touch granularity suffices).
- Trace (K2_TRACE, trace page always warm): the phase where free - warm grows = where the walks bite (start chain /
  gather / tail); fALL / fKV traces show which part a touch removes.

## Job w7_j1 RESULT (18:34-18:36, 96 s wall; dev/w7/out_j1.txt). Probe = trace build (K2_TRACE), 12 pairs, event us:
d32s: warm d +1.06 (rep0 16.69 / rep1 15.65), free d +2.51, fALL d +1.58 (removes 0.93), fKV d +1.66 (removes 0.85).
d32s e3 stamps (median over CTAs, us from the first CTA start; rep0 free vs rep1 free | warm0 | fALL0 | fKV0):
  19 pt entries landed (tid 256)   1.97 vs 1.22 | 1.23 | 1.26 | 1.76   -> PT walk ~0.75 on the chain; ALL fixes it, KV not
  17 first half-tile issued        5.09 vs 2.95 | 3.46 | 3.58 | 3.89   -> cp.async issue throttled by the K/V walks
  4  first tile landed             6.65 vs 4.23 | 4.46 | 4.72 | 5.10
  5  steps done                    9.62 vs 7.26 | 7.46 | 7.86 | 8.16
  9  out stored                   11.79 vs 9.66 | 10.46 | 10.41 | 10.70  -> in-kernel cold penalty +2.13; ALL pre-touch leaves
                                                                         +0.69 (warm-mode rep0 itself +0.38), KV leaves +1.0
  16 (seq_lens landed + pt issued) 0.98-1.0 in EVERY mode: the seq_lens / qkv loads pay no cold penalty.
=> d32s: ~1.4 of the ~2.1 us in-kernel cold penalty is data-page walks: K/V (~1.1) + page-table (~0.4-0.7, the chain).
d16s (xe_dm): free d +0.11, fALL / fKV no change; every stamp identical warm / free (pt landed 0.52-0.54, out 5.22)
=> NO in-kernel cold penalty on d16s here: nothing to fix in xe_dm (residual = launch path). d16s DROPPED.

## Job w7_j2 - DECISION RULES (written before the job; dev/w7/j2.sh)
Variants (build.py --e3 M on k66; all in dev/w7/var.py as kernels attn_1_1_1_1e3w7_M picked by env W7_E3K at capture;
SASS of w7_M == the baked --e3 M build's (names only differ); 0 spills): v1 = own pt-row pages (real cp.async 4 B),
v5 = + own out row, v7 = + every K/V page per CTA (336 touches / CTA from warps 1-7), v23 = v7 with K/V pages spread
over the grid (1 touch / page / call), v15 / v31 = v7 / v23 with prefetch.global.L2 instead of real loads.
- (b) trace: v -> job 3 (bench_jt) iff free-rep0 stamp 9 <= base free-rep0 stamp 9 - 0.3 AND warm-rep1 stamp 9 <= base + 0.1.
- (c) events: same with the rep0 event median (free) and the rep1 event median (warm).
- Both pass -> bench_jt candidate; ties (within 0.1) -> fewest touches. Real vs prefetch (v7 vs v15, v23 vs v31) answers
  whether prefetch fills the TLB; v7 vs v23 whether a shared TLB level fills from one SM.
- (a) localization (fixed probe, 14 pairs): reported; rules of j1 apply (carries >= 0.3 d32s, >= 0.5 full).

## Job w7_j2 RESULT (18:43-18:46, 143 s wall; dev/w7/out_j2.txt): THIS NODE HAD NO COLD-TLB PENALTY -> cold rules void
- base k66 e3: free d (rep0 - rep1) +0.11 (trace build) / -0.08 (events) / -0.02 (localization); j1's node: +2.51. Same
  probe logic => the per-pair free penalty is NODE-DEPENDENT (j1 GPU f6d01eb0, 128 MiB used at start; j2 GPU 5fc05ebc,
  4 MiB). The judge's nodes show it every run (d32s even-odd 1.7-2.1). Every cold measurement must first show the base
  penalty on its node (base tlbcold-1 minus tlbcold-0 >= ~1 us on d32s) or it decides nothing.
- WARM codegen cost of the in-kernel touches (events, non-trace, 20 pairs, warm rep1 vs base 13.97): v1 +0.96, v5 +0.93,
  v7 +2.24, v23 +0.89, v15 +0.11, v31 +0.05. Trace build (stamp 9 warm1 vs base 10.12): v1 -1.26, v5 -1.30, v7 +0.52,
  v23 -1.28, v15 +0.35, v31 +0.09. => e3 sits at 127 regs: ANY edit re-schedules ~1000-1350 of 2544 SASS lines (modulo
  register names) and moves the warm time by +-1 us in either direction (opposite signs in the trace / non-trace builds).
  v7 (336 real touches / CTA) also delays the pt-entry loads by ~0.45 us (LSU contention; stamp 19 1.65 vs 1.2).
- d16f localization crashed again (illegal address in a touch on the 2nd shape); full shapes are low value: dropped.
=> Next: keep e3's own SASS byte-identical (touch in a __noinline__ callee at kernel entry, or outside e3) and only then
   test cold on a node that shows the penalty.

## Job w7_j3 - DECISION RULES (written before the job; dev/w7/j3.sh)
New designs that keep the worker code: s / S = a separate 1-CTA kernel attn_w7tk (512 thr, one volatile 4-B load per
2-MB page: page table, out, K, V) on a side stream forked from / joined to the caller's stream = a parallel graph branch,
launched before (s) / after (S) e3; e3's SASS == k66's (checked). x = one extra cluster of 3 CTAs in e3's grid (same
touches, exits; but the entry test re-schedules the worker code: 594 SASS lines differ). Plus v23 / v7 / v31 / v15.
- Node gate: base free d (rep0 - rep1, events) >= 1.0, else the node has no penalty: (B), (C) skipped (warm info only).
- (A) side variant PASSES iff base free rep0 - side free rep0 >= 0.5 AND side warm rep1 - base warm rep1 <= +0.1.
- (C) bench_jt d32s (slots k66a, cand, k66b, candb, k66c-sink): KEEP iff tlbcold 1: both cand copies <= min(k66a, k66b)
  - 0.10, AND tlbcold 0: mean(cand) <= mean(k66a, k66b) + 0.05, AND check.py 56 PASS / 0 failures, t_e3 0 failures.
  KEEP -> deliver_r1 = build.py --side N on k66.

## Job w7_j3 RESULT (18:57, 43 s wall; dev/w7/out_j3.txt): AGAIN A NO-PENALTY NODE (4 MiB used at start)
base free d +0.16 -> (B), (C) skipped. Warm rep1 vs base (13.50): s +0.44, S +0.16, x +0.23, v23 +0.93, v7 +2.44, v31 -0.03,
v15 +0.18 (v23 / v7 codegen losses reproduce j2's; the side kernel's parallel graph branch costs +0.16..+0.44).
Node type survey (nvidia-smi memory.used at job start): d32s cold penalty seen on 128-130 MiB nodes (w7 j1, w3 j3, w3 j5),
absent on 4 MiB nodes (w7 j2, j3); ~40 % of today's jobs got a >= 100 MiB node. The judge shows it in every run.
=> j4+ exit at once on a < 50 MiB node (costs only the container start).

## Job w7_j4 - DECISION RULES (written before the job; dev/w7/j4.sh, base k65)
Node gate: memory.used >= 50 MiB, then (A) base free d >= 1.0. (A) probe vk.py (= build.py k65 --var --vars 23,87,31,15,0x;
87 = v23 with the cp.async drain at the exits), variants base, v31, v15, v23, v87, x, s, S; 24 pairs warm / free.
PICK = the variant with the lowest pair-mean(free) - base (pair-mean = (rep0 + rep1) / 2: the judge pair) among those
with pair-mean <= -0.3 AND warm rep1 - base <= +0.15. (C) on the pick (baked by build.py on k65): check.py 56 PASS / 0
failures, t_e3 0 failures, t_rep 60 replays (per-pair free, flush, 20-s timeout) 0 failures / no HANG, then bench_jt
decode 32 sliding (slots k65a, cand, k65b, candb, k65c sink): KEEP iff tlbcold 1 both cand copies <= min(k65a, k65b) - 0.10
(tlbcold 0 reported, sanity only). KEEP -> dev/w7/deliver_r1 = build.py <args> on k65.
- j4 attempts a / b (23:43): skipped by the memory gate (4 MiB). j5 (23:45, 43 s; dev/w7/out_j5.txt): driver 610.57.04,
  4 MiB node, base k65 d32s warm d +1.17 (rep0 15.17 / rep1 13.98), free d +2.32 (17.15 / 14.75), vfree (+ driver-VMM
  cuMemUnmap of a fresh 2-MB block per pair) +2.54 => THE PENALTY IS PRESENT on this 4 MiB node: memory.used is NOT the
  discriminator (j2 / j3 nodes: none). Gate removed from j4; the in-probe gate (base free d >= 1.0) stays.
- Coordinator 23:4x: scope = d32s + d16s only (decode full -> W10). ncu (k65, cold L2) d32s: stalls barrier (final cl_wait
  11 %, step loop 6 %), mbarrier tile waits 7.7 %, the K/V copy-issue loop ~12 % (shfl -> compare short_scoreboard, cp16
  wait/selected): the gather is issue / in-flight bound, so TLB walks stretch it (j1 trace: gather +1.67 us cold).

## Job w7_j4c RESULT (23:51, 48 s wall; dev/w7/out_j4.txt; driver 610.57.04, 72 MiB node): no pick
Multi-variant probe (8 graphs x 0.9 GB, round-robin): base free d +0.46 only, and WARM d is +0.5..+1.4 for most variants:
with 8 graphs' buffers cycling, even 'warm' rep0 is TLB-capacity-cold, so the multi-graph probe does NOT isolate the
per-pair free penalty (single-graph probes j1 / j5: +2.3..+2.5). Only same-process single-graph probes or bench_jt count.
Warm rep1 vs base: v31 -0.02, v87 -0.19, x +0.05, S +0.22, s +0.38, v15 +0.26, v23 +0.90 (v23 / v87 = same touches,
drain after the steps vs at the exits: the codegen lottery). Pair-mean(free) vs base: v87 -0.02, x +0.35, others worse.

## Job w7_j6 - DECISION RULE (written before; dev/w7/j6.sh): bench_jt decode 32 sliding, tlbcold 1, 80 reps, slots
k65a, v87a, k65b, v87b, k65c (sink). v87 = build.py k65 --e3 87 (own pt-row + out-row + grid-spread K/V pages, real 4-B
cp.async, exit drain). KEEP iff both v87 copies <= min(k65a, k65b) - 0.10; then check.py 56 / 0, t_e3 0, t_rep 60 replays
0 failures / no HANG -> deliver_r1. tlbcold 0 = sanity only.

## Job w7_j6 RESULT (23:53, 168 s wall; dev/w7/out_j6.txt; 4 MiB node, driver 610.57.04): v87 KEEP -> deliver_r1
bench_jt d32s tlbcold 1: k65a 13.19 | v87a 12.04 | k65b 12.34 | v87b 12.10 | k65c (sink) 12.48 -> both v87 <= 12.24: RULE MET.
tlbcold 0: 10.16 | 10.29 | 10.31 | 10.24 | 10.27 (v87 +0.03). check.py 56 PASS / 0 failures; t_e3 0 failures (only B 32 / 33
route to C=3 among its cases: C=3 needs ~33 <= B <= ~38 co-resident); t_rep 60 replays 0 failures, no hang.

## Job w7_j7 - DECISION RULE (written before; dev/w7/j7.sh): bench_jt d32s tlbcold 1, slots k65a, v87a, k65b, e2a, v87b,
e2b, k65c (sink). e2 = build.py k65 --e3 87 --early 2 (pt-row + out-row touches BEFORE the prefix loads, K/V after;
2696 SASS, 124 regs, 0 spills). e2 replaces v87 (deliver_r2) iff both e2 copies <= min(v87a, v87b, k65a, k65b) - 0.10 and
check.py 56 / 0, t_rep 60 / 0 no hang, t_e3 0. v87 confirmation = mean(v87) - mean(k65a, k65b), reported. t_e3 extended
(B 34..38 cases; skipped when not routed to C=3).

## Job w7_j7 RESULT (00:03, 176 s wall; dev/w7/out_j7.txt; 68 MiB node): v87 CONFIRMED, e2 not kept
tlbcold 1: k65a 12.65 | v87a 11.84 | k65b 12.03 | e2a 11.53 | v87b 11.50 | e2b 11.65 | k65c (sink) 12.09.
v87 vs k65a/b -0.67 (both copies <= 11.93: rule met again), vs k65b/c -0.39. e2 - v87 = -0.08 (< 0.10): deliver_r1 stays.
t_e3 extended on v87: 9 cases routed to C=3 (B 32..38), 27 checks, 0 failures.
