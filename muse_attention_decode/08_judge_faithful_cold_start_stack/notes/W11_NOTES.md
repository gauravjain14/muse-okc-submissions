# W11 NOTES: verify full v16f / v32f (attn_xd_v8f = _XD_SRC of dev/coord/k67.py, md5 50806e22; xd0 d5806e8f / xd1 a07f4682)
Owner W11 (round 4), write only dev/w11. GPU cap 40 min (job wall), jobs <= 12 min, timeout 120 per variant.
GPU: bash tools/gate.sh tools/modal_run.py --tag w11_jN --cmd "cd /root/p && ls dev/coord dev/w4 dev/w7 dev/w10 dev/w11 tools >/dev/null && bash dev/w11/jN.sh"

## Resume here
- J2: EVF (W11_EVF=1) KEEP (nsys v16f -1.09, v32f -2.16, all runs; bench_jt cold both copies win: -1.6 / -3.2 us).
- J3: WS correct (t_once 68/0, xcheck 24/0 auto + HYB 1) but SLOWER (+1.64 / +2.26 nsys vs EVF) -> DEAD. EVF xcheck HYB 0 / 1 24/0.
- deliver_r1/ (= k67 + EVF; build.py -> k67w11.py md5 1365d43f, cubins = e1's cc256194 / 29061c33, 864,038 B stripped).
  The coordinator already stacked it: dev/coord/k69.py's XD section = k67 + exactly this patch (verify_k69 running 02:00).
- J4 done: check.py on deliver_r1/k67w11.py CHECK_RC 0, 56 PASS, 0 failures -> deliver_r1 FROZEN (EVF). Half-row lanes DEAD.
- Report 1 sent. Coordinator FUNDED +25 GPU-min (cap 65) for the 4-lane x 1-WG verify bound (base dev/coord/k69.py).
- Premise check first (J5): XD's own mb7 (dev/xd NOTES) had the XD-geometry T128 lockstep skeleton at 150.74 us vs
  per-warp rings 149.79 (b32, same job) -> my l2x5 probe kernel (two 256-thread barriers + one-tile refill per 16-KB
  tile) may be a weak stand-in for XD's loop, not the geometry. J5 measures XD's loop (dev/w11/mbxd.py = dev/xd/mb.py
  copy) next to l4x3 in one container before any build.
- J5 done (02:26-02:28, ~2 min): premise essentially REFUTED (XD's loop skeleton is 1.4-1.6 % below l4x3, not 9 %); geometry
  ceiling 2.3 / 1.3 us < the coordinator's -4 / -2 build rule; 4-lane bound NOT built; reported. GPU used ~21 of 65 min.

## GPU ledger
- j1 00:59-~01:03 (~4 min): probe_win (out: dev/w11/j1.log)
- j2 01:07-~01:12 (~5 min, wall 200 s in-container): probe v2 + EVF xcheck / nsys / bench_jt (dev/w11/j2.log)
- j3 01:28-~01:34 (~6 min, wall 266 s): WS gates + nsys + bench_jt, EVF xcheck HYB modes (dev/w11/j3.log).
- j4 02:07-02:10 (~3.5 min, wall 153 s): check.py deliver_r1, bench_jt, probe v3 (dev/w11/j4.log). Total ~19 min.
- j5 02:26-02:28 (~2 min, wall 49 s): XD loop skeleton vs 4-lane probe (dev/w11/j5.log). Total ~21 min.

## Local analysis (no GPU) of the existing traces
- k65 trace (= k67's XD section; W10 j1 out1.tgz trx_k65.npz) with dev/w10/tl_an.py: v16f lost 19.2 = start 5.70 + steady 3.59
  (BW_s 4.187) + tail 3.86 + end 5.65; v32f 25.2 = 5.50 + 10.47 (4.102) + 4.59 + 5.58.
- START: cumulative consumer-landed bytes are ABOVE the steady line by ~9 us (the prologue burst lands at 8.7 TB/s in the
  6.5-us bin): the ramp after the first landing costs nothing; the start loss is the first landing (~5.1-5.6 us) =
  lengths trip -> ~1.1 us of setup code -> rows trip -> issue. A deeper prologue (5 tiles) has a ~0 bound -> not built.
- TAIL (dev/w11/cta_an.py): 2-segment CTAs finish 3.8 (v32f) / 5.4 (v16f) us BEFORE the median, plain CTAs +0.2..0.6; the
  latest CTAs are plain (v32f: mostly the FINALs). XC's OVH sweep (+0.14 / +1.0 / +4.5 for OVH 512 / 256 / 0) = the
  straddlers' slack is not lost work (fluid regime: the other SMs absorb the bandwidth). Lane drift = random walk
  (median +0.3..0.9 us lane 0 late, p10/p90 -2.3 / +3..5 us at the end). Bound of perfect intra-CTA lane balancing:
  max over CTAs of max(l0, l1) - max of (mean + 0.5 us granularity) = 0.47 (v16f) / 0.57 (v32f) us (k65), 0.40 / 0.43 (k63).
  Static lane swap (extra tile to lane 1 when ntp odd): 0 in 6 of 8 reps (all plain CTAs of a rep share the parity) -> dead.
- STEADY (v32f 4.10 TB/s): ncu (out/20261029T234300_roof_32_1_s/ncu/v32f.json, lines remapped: module line N = k67.py
  line 15412 + N): pair-top lane barrier 14.4 % of samples (line 1101 = issue code after bar_sync), the cp.async wait
  8.2 % (long_sb, line 1364), V^T ldsm/stsm 7.4 % (short_sb), lg_throttle 0.003 per issue (no LSU issue throttle).
  The steady shortfall vs 4.39 is the TLB: XF's probe (dev/xf/deliver_r7, decode rows) runs random rows at 4.11-4.13
  TB/s and 2-MB-page-sorted rows at 4.36-4.41 (-5.8 %); XB's in-kernel sort recovered 7-10 us of the v32f tile phase
  but every sort placement cost more at the start. The hot pair loop (458 / 463 SASS) spends ~100 instr / warp / pair
  on the K/V + page-table issue (remateralized address math at 128 regs); XD's leverage of CUDA-core work is ~0.045.

## J1 (w11_j1) - DECISION RULES (written before the job)
Gather-only probe dev/w11/probe_win.py on the REAL v16f / v32f rows (seed 4321, q_len 8), kernels l2x5 (XD geometry: 2
lanes x 256 thr x 5 stages, no hint) and l4h3 (XF's lane3h). G(o) = t(orig) - t(o), medians of 30 judge-style reps.
- Windowed in-kernel page bucketing (W rows of smem per lane, one pass, no extra traffic): OPEN iff on l2x5
  G(win1024) >= 0.6 G(lanesort) on BOTH shapes and G(lanesort) >= 2.5 us at B 32 -> next = in-kernel cost bound (timing
  arm: windowed counting sort in the pair loop). If win1024 < 0.6 but win2048 >= 0.6 on both: open at W 2048 (smem
  permitting). CLOSED iff every win<W> (W <= 2048) < 0.35 G(lanesort) on both shapes.
- Spatial footprint partition (pg<F>, TLB capacity): worth a filter design iff G(pg4) >= 0.5 G(lanesort) on both shapes.
- |G| < 0.5 us = noise.

## J1 RESULT (probe_win, seed 4321, 30 reps; gain G vs orig, % of lanesort)
| B | kernel | orig us (TB/s) | lanesort | bkt18 | win512 | win1024 | win2048 | win4096 | pg4 | pg11 |
| 32 | l2x5 | 165.31 (3.93) | -9.39 | 71 % | 8 % | 30 % | 52 % | 78 % | 4 % | 4 % |
| 32 | l4h3 | 150.64 (4.31) | -6.88 | 89 % | 29 % | 44 % | 59 % | 85 % | 5 % | -4 % |
| 16 | l2x5 | 82.62 (3.88) | -3.90 | 77 % | 49 % | 65 % | 88 % | 113 % | -1 % | -7 % |
| 16 | l4h3 | 75.20 (4.27) | -2.91 | 66 % | 30 % | 73 % | 84 % | 89 % | -19 % | -1 % |
Readings: (1) SPATIAL partition (each SM's footprint 1/4 or 1/11 of the pages, random order) gives NOTHING: the TLB loss is
not capacity / footprint; only TEMPORAL locality (runs of the same 2-MB page in an SM's request stream) helps.
(2) Windowed page sort: the gain grows with the run length per page (W / 168 pages at B 32, W / 84 at B 16): runs of ~3
-> 8 %, ~6 -> 30-49 %, ~12 -> 52-65 %, ~24 -> 78-88 %. Rule: win1024 < 0.6 at B 32 -> not OPEN at W <= 1024; not
CLOSED (B 16 win512 49 %). (3) XD-like geometry without a cache hint (l2x5) gathers 9 % slower than XF's lane3h
(4 lanes x 3 stages, evict_first): 165.3 vs 150.6 us (B 32), 82.6 vs 75.2 (B 16) -> J2 isolates hint vs lanes.

## J2 (w11_j2) - DECISION RULES (written before the job)
(a) probe v2 (kernels l2x5, l2x5h = + evict_first, l4x3 = 4 lanes no hint; orders orig, lanesort, win768, win1024):
  - per-lane windowed sort at W 768 (the largest window that fits the free smem without touching the ring): BUILD iff
    on the kernel geometry matching the kept hint choice, G(win768) >= 1.0 us at B 32 AND >= 1.0 us at B 16.
  - hint effect in the gather: t(l2x5) - t(l2x5h) at both shapes (explains (b)); lanes effect t(l2x5) - t(l4x3).
(b) EVF (W11_EVF=1: L2::evict_first policy on every K/V copy of attn_xd_v8f; hot loops unchanged 463 / 458 SASS, 0 spills;
  xd0 cc256194, xd1 29061c33): xcheck auto gate first (any FAIL -> not timed). nsys 2 copies x 2 passes vs k67a/b on
  v16f / v32f; bench_jt --tlbcold 1 (k67, e1, k67b, e1b, k67c sink; base = SGLang).
  KEEP iff nsys D <= -0.15 on >= 1 shape with all 4 runs below the same-pass control mean and the other shape D <= +0.10,
  AND bench_jt: no shape with mean(e1, e1b) > mean(k67, k67b) + 0.10. Keep -> J3 freeze gates (check.py, xcheck x2).
  Kill: D > -0.05 on both shapes.

## J2 RESULT (dev/w11/j2.log)
Probe v2 (G vs orig): B32 l2x5 win768 +2.59 (27 %), win1024 +3.33 | l2x5h (evict_first) orig 166.18 (= l2x5 165.73: the hint
does NOTHING in a pure gather) win768 +2.48, win1024 +3.63 | l4x3 (4 lanes x 3, no hint) orig 150.88 = l4h3: the 9 % gap to
the XD geometry is the LANE structure (4 x 128-thread streams vs 2 x 256-thread; same bytes in flight), not the hint.
B16: l2x5 win768 +2.32 (53 %), win1024 +2.64; l2x5h +2.03 / +2.42; l4x3 +1.92 / +2.14. -> rule met (>= 1.0 us both): BUILD W 768.
EVF (e1 = k67 + W11_EVF=1): xcheck auto 24 PASS / 0 FAIL. nsys (2 copies x 2 passes, controls k67a/b v16f 88.88 spread
0.22, v32f 167.74 spread 0.19): v16f -1.09 (-1.23..-0.95), v32f -2.16 (-2.32..-2.05) -> KEEP (+0.316 % score by nsys).
bench_jt --tlbcold 1 (order k67, e1, k67b, e1b, k67c sink; us): v16f 91.83 / 90.95 -> 90.03 / 89.54 (mean -1.61),
v32f 171.06 / 170.19 -> 167.58 / 167.32 (mean -3.18); sink k67c 91.33 / 169.32. Both copies beat both controls.
Why it helps in the kernel but not the probe: the kernel's non-stream lines (gate rows prefetched evict_last at segment
start and read in the epilogue, the published partials read by the FINAL / combiners, page-table chunks, Q / new-token
rows) are no longer evicted by 650 MB of normal-priority K/V lines (the end chain's loads hit L2 instead of DRAM).

## WS build (W11_WS=1; dev/w11/build.py, local NVRTC 13)
Per lane: 24 chunk / sorted-window slots of 68 ints (tile t -> slot t % 24), 12-slot staging, 256-bucket histogram (21.1 KB
per CTA instead of the 5-KB chunk ring; host smem 214280 -> 230792 B). Windows of 12 lane tiles (768 rows); windows 0-1 in
position order (chunks), window w >= 2: stage (pair 6w-6), count (6w-5, red.shared by 2-MB page of the K row), scan
(6w-4, warp 0 of the lane), scatter (6w-3); refills of window w run pairs 6w-2 .. 6w+4. shift kept in misc[62 + li].
SASS (with EVF): xd0 e96e9a37 0 spills, hot loop 463 -> 710 instr (the 4 step bodies; ~10-30 executed per pair); xd1
9f76d9f0, 16-B stack (28 B spill stores; in-loop: one STL / LDL pair in the 2nd refill's issue block), loop 458 -> 662.
Exactly-once debug build W11_CHK=1 (dev/w11/t_once.py: per (CTA, lane) count / sum / hash-sum of copied rows minus the
position-order rows of the lane's tiles, all must be 0). Defaults (no knobs) = k67 cubins byte for byte.

## J3 (w11_j3) - DECISION RULES (written before the job)
(a) WS gate: t_once on wsd (WS + EVF + CHK) 0 failures, then xcheck auto + XC_HYB=1 on wsa (WS + EVF) 0 failures; any
    failure -> WS not timed. EVF gate: xcheck XC_HYB=1 and XC_HYB=0 on e1a (auto was 24/0 in J2).
(b) WS timing vs the EVF base: nsys 2 copies x 2 passes (e1a/b controls, wsa/b) on v16f / v32f + bench_jt --tlbcold 1
    (e1, ws, e1b, wsb, e1c sink). KEEP WS iff nsys D(ws vs e1) <= -0.15 on >= 1 shape with all 4 runs below the same-pass
    e1 mean and the other shape <= +0.10, AND bench_jt: no shape with mean(ws, wsb) > mean(e1, e1b) + 0.10.
    Kill iff D > -0.05 on both shapes. In between: report; tune (in-loop spill, window size) before another job.

## J3 RESULT (dev/w11/j3.log; slow node: e1 v16f 91.15 / v32f 172.47 vs 87.8 / 165.5 in J2)
t_once wsd (WS + EVF + exactly-once checker): 68 PASS / 0 FAIL (17 cases x 2 seeds x 2 calls, incl. v16f / v32f scored,
ctx 200K, B 24 / 64 / 128); xcheck wsa auto 24/0, XC_HYB=1 24/0; xcheck e1 XC_HYB=1 24/0, XC_HYB=0 24/0.
nsys WS vs e1 (spread 0.18 / 0.21): v16f +1.64 (+1.49..+1.78), v32f +2.26 (+1.96..+2.48) -> KILL (rule: D > -0.05 both).
bench_jt (noisy node): v16f e1 98.22 / 102.10 vs ws 103.04 / 103.72 (sink e1c 103.33); v32f 183.70 / 182.27 vs 183.62 / 183.42.
Reading: the ~2.5 us gather-only gain of 768-row windows is eaten (and more) by ~4 us of in-loop cost: hot loop 458 -> 662
/ 463 -> 710 SASS (four step bodies, i-footprint ~11 KB), ~10-30 executed instructions + 1 smem-atomic step per pair,
one in-loop spill pair in xd1. An in-loop sort must cost <~1 us total to pay; windows >= 2048 (where the gain is 52-88 %)
do not fit next to a 5-slot ring. WS code stays in build.py (knob W11_WS, default 0; t_once.py / W11_CHK for any re-try).

## J4 (w11_j4) - DECISION RULES (written before the job)
(a) FREEZE dev/w11/deliver_r1/ (k67 + EVF) iff tools/check.py CHECK_RC 0 and 0 failures (grep -a) on k67w11.py (= d1a.py),
    xcheck auto 0 failures (J2 / J3 already: auto, HYB 0, HYB 1 24/0 on the same cubins), and bench_jt --tlbcold 1 (k67,
    d1, k67b, d1b, k67c sink): on no shape mean(d1, d1b) > mean(k67, k67b) + 0.10. No waiting / polling added -> no t_rep.
(b) probe v3 (orig order; l2x5, l4x3, h2x5 = 2 streams x 2 HALF-ROW lanes x 5 stages, h2x3): a head-split XD variant
    (each WG copies / waits / refills only its KV head's 64-B half of every row, 128-thread barriers, no lane barrier) is
    worth building iff t(h2x5) <= t(l2x5) - 0.5 (t(l2x5) - t(l4x3)) at both shapes; else the 4-lane advantage needs whole
    rows per lane (dead for XD's 2-WG lanes).

## J4 RESULT (dev/w11/j4.log)
check.py d1a (= deliver_r1/k67w11.py, md5 1365d43f): CHECK_RC 0, 56 PASS, 0 failures. -> deliver_r1 FROZEN.
bench_jt --tlbcold 1 (slots k67, d1, k67b, d1b, k67c): v16f 95.77 | 94.30 | 95.37 | 99.04 | 99.83 (slots 4-5 ~4 us slow:
the sink control k67c is the slowest, d1b in slot 4 only beats it) -> mean rule fails on v16f by slot position only
(slot-neighbour pairs: d1 -1.27 vs slots 1 / 3, d1b +1.44 vs slots 3 / 5); v32f 170.92 | 168.10 | 170.14 | 167.94 | 169.97
-> d1 / d1b -2.5 us, both beat both controls. J2's bench of the same cubins (no slow tail) won in every slot (-1.6 / -3.2).
Freeze rests on nsys J2 (all 8 runs) + J2 bench + this check; coordinator's verify_k69 (k69 = k67 + this patch) re-judges.
Probe v3 (orig order): B32 l2x5 165.50, l4x3 151.17, h2x5 174.14, h2x3 173.66; B16 82.77 / 75.71 / 86.74 / 85.82 ->
half-row lanes (each 128-thread lane copies one KV head's 64-B half of every row, independent rings) are 5 % SLOWER
than the XD geometry: split-line requests from drifting lanes cost DRAM efficiency. Head-split XD redesign DEAD.

## Dead / closed (W11)
WS windowed page sort (correct, +1.6 / +2.3 us), spatial page partition (probe 0), half-row lanes (probe +5 %), deeper
prologue (bound ~0), static lane swap (bound 0), EVF=2 (per-tile policy: +12 loop instr, not timed; EVF=1 kept).
## Findings worth handing on
- The 2 x 256-thread-lane geometry is ~9 % slower than 4 x 128-thread lanes in a pure random gather (l2x5 165.5 vs
  l4x3 151.2 us at B 32, J1/J2/J4 reproduce) with the same bytes in flight; whole-row lanes are required (half rows lose).
  XD's 2-WG lanes cannot be split; a 4-lane verify kernel would need 1 WG per (lane, 2 heads) = > 128 regs (XD notes).
- The v32f steady-state shortfall is TLB temporal locality: only same-2-MB-page runs in an SM's stream help; the in-kernel
  price of building runs (>= 2048-row windows) exceeds the gain with a 5-slot ring.

## J5 (w11_j5) - DECISION RULES (written before the job)
Same container: dev/w11/mbxd.py variants 3_5_2_1 (XD T128 lockstep loop, lean issue, DST 5, no compute = gather
skeleton of the real geometry), 3_5_2_0 (+ real QV2 tile compute), 3_5_2_2 (resident: compute, no gather), + the real
k69-XD kernel (d1a.py) on the same inputs; probe_win l4x3 / l2x5 (orig). Compare in TB/s (inputs differ by seed).
- Geometry premise HOLDS iff skeleton TB/s <= l4x3 TB/s x 0.966 at B32 (>= ~5 us slower) and x 0.97 at B16 -> build the
  4-lane x 1-WG bound (Step 1). REFUTED iff skeleton TB/s >= l4x3 TB/s x 0.987 at B32 (<= ~2 us) -> no build; the 9 % was
  the probe kernel. In between: report numbers, coordinator decides.
- Coupling (explains): t(3_5_2_0) - t(3_5_2_1); the real kernel vs 3_5_2_0 = start + end chains.

## J5 RESULT (dev/w11/j5.log; slow container: l4x3 157.7 us vs 151 in J1/J2/J4)
mbxd (XD's own T128 lockstep loop, 2 lanes x 2 WGs, DST 5, lean issue; seed 4242 + B rows) | probe (seed 4321):
| | XD skeleton 3_5_2_1 | XD + compute 3_5_2_0 | resident 3_5_2_2 | real k69-XD (same inputs) | l4x3 probe | l2x5 probe |
| B32 | 158.29 us = 4.062 TB/s (643 MB) | 164.26 | 110.16 | 168.56 | 157.68 = 4.120 TB/s | 172.58 = 3.764 |
| B16 | 82.74 = 4.006 TB/s (331.5 MB) | 87.87 | 59.94 | 92.37 | 78.67 = 4.078 TB/s | 86.14 = 3.724 |
-> XD's real loop geometry gathers at 0.986 (B32) / 0.982 (B16) of 4 x 128-thread lanes: the 9 % of l2x5 was MY probe
   kernel (two 256-thread barriers + a one-tile refill per 16-KB tile), not XD's geometry. Rule: between REFUTE (>= 0.987)
   and HOLD (<= 0.966), but the whole geometry ceiling is ~2.3 us (v32f) / ~1.3 us (v16f), below the coordinator's build
   rule (-4 / -2) even if fully captured. Compute coupling = 5.97 / 5.13 us (3_5_2_0 - 3_5_2_1); start + end of the real
   kernel = 4.3 / 4.5 us over 3_5_2_0. A 4 x 1-WG kernel could only pass by removing >= ~30 % of that coupling with the
   same instructions and a 2x longer per-WG chain (XD-M decoupling lost; XD: ~3.2 us of coupling exists with only QK + PV
   + barriers). Not built.
