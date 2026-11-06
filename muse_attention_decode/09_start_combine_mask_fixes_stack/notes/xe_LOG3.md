# xe round 3: cost-model split for d16f / d32f (base run1f FLAT q_len 1), XF section on top of k77

Files: work/xe/cand3.py (= runs/k77.py + `_XF_SRC` built by mkxf3.py + xf_py3.txt), collect.py (per-CTA loop-end
times of a graph replay after a flush, 5 runs each, + seq lens), fit/fit1.py (model fit), fit/sim2.py (offline split
simulator), base_ends.json (31 cases: B16 x10, B32 x8, B8 x3, B24 x3, B48 ctx 60K x3, B16 ctx 40K x2, B32 ctx 120K x2;
agentic spread, coherent).

## Model fit (per-CTA end time, k6 = after the loop, base split with ovh 1536)
end_c = a_case + t * tokens_c + s * (segments_c - 1) (+ new-token / tiny-last-segment terms, both ~0):
- pooled: t = 5.62 ns/token, s = 4.15 us (= ~740 tokens); per case: B16 5.59 ns / 4.17 us (746 tok), B32 5.82 / 4.30
  (740), B8 6.07 / 4.75 (783), B48 5.01 / 3.26 (651). (t < the average 8.3 ns/token: CTAs that finish early hand
  their bandwidth to the others; the slope is the marginal cost.) new token: -0.11 us, tiny last segment: -0.46 us.
- residual std 0.86 us (1-segment CTAs 0.81, 2-segment 1.09). The MAX of each case is set by the residuals (max-p50 of
  the residuals ~= max-p50 of the ends: 2.4 vs 2.5 us): the systematic part only explains the early finishers.
- Expected gain of a split with equal predicted cost (residuals unchanged): B16 +0.59..0.75 us, B32 +0.67..1.33 (mean
  ~1.0), B24 ~0.95, B8 ~0.4, B48 mixed (one outlier case).
- The base's two systematic errors: (1) every request (also request 0) carries a 1536-unit zone; a CTA boundary inside
  a zone gives both neighbours virtual units and nobody restarts; (2) 1536 units overcharge a real restart (~740).

## Split implemented in xf_run1f (XF_ZC, XF_S = 768 units)
Zones of XF_S before every request but the first; the CTA grid is fixed by the all-charge total NU0 = sum(seq + 1) +
(B - 1) XF_S; a zone that a grid point would cut is shortened to end exactly at that point (the next CTA starts at the
request's first token; nobody restarts there, the previous CTA wastes < XF_S). Decisions in exact greedy order (a round
fixes the first cut zone and shifts the later positions). Offline (fit/sim2.py): predicted max-mean 0.03-0.1 us (base
0.6 / 1.2), predicted max -0.55 (B16), -1.08 (B32), -1.6 (B48), 0-4 rounds.

## Results
- first version (iterate "decide all -> rescan" until stable): does not converge in ~1/3 of the cases (tiny segments):
  d16f +1.46, d32f +2.32 (XF_S 768); 1024 +1.52/+2.80; 1536 +2.50/+2.77.
- greedy rounds with 64-bit integer divisions: correct but the length barrier moves 0.85 -> 1.6-2.8 us and the whole
  end distribution with it: d16f +1.17, d32f +2.19 (XF_S 768); 1024 +1.52/+2.45; 576 +1.15/+2.67.
- greedy rounds with FP64 multiplies + incremental shifts: barrier 1.0-1.34 us (XF_S=0: 1.06, 768: 1.36, 4000: 2.24):
  each round still ~150 ns; the start delay is directly visible in the end times.
- final: float estimate + exact 32-bit integer fix-up (64-bit fallback when NU0 G >= 2^32), branch-free round body, one
  scan + incremental shifts: length barrier 0.85 (k77) -> 1.06 (XF_S 0, no cut) / 1.54 us (768); rounds ~220-400 ns
  each (dependent conversion/multiply/shuffle chain, k20..k24 stamps), no spills (128 regs, LOCAL 0).
  robust_f.py (132 full-layer cases B 2..64 through XF): 0 failures.

## Did the split equalize? (collect.py, seeds 1000-1011, split features recomputed for each split: fit/fitxf.py)
- refit on the XF-split ends: t 5.60 ns/token, s 4.63 us (base split on the same seeds: 5.52 / 3.68), residual std 0.78
  (base 0.76): the model transfers; the predicted max-mean drops 0.6 -> 0.05 (B16), 1.2 -> 0.08 (B32).
- measured end max-mean: B16 2.72 -> 2.41 (-0.31), B32 3.73 -> 2.68 (-1.05): the systematic part IS removed.
- but the duration net of the start phase did not move. Paired traced comparison (cmp_split.py, held-out seeds
  2000+, 7 runs each, per run dur = max k6 - min k0, net = dur - median k9), d = XF(768) - k77:
    B16 (12): d dur +0.61, d k9 +0.47, d net +0.13 +- 0.13, d (max - p50) +0.27
    B32 (12): d dur +0.77, d k9 +0.59, d net +0.19 +- 0.17, d (max - p50) +0.32
    B24 (4):  d dur +0.66, d k9 +0.50, d net +0.28 +- 0.03
    B48 (3):  d dur +1.26, d k9 +0.65, d net +0.63 +- 0.42
  ab.py (interleaved graphs, 24 instances, seeds 71-76) vs k77: XF_S 768 d16f +0.87 +- 0.12, d32f +0.93 +- 0.10;
  XF_S 1024 +1.02 / +0.89. (Earlier versions with the slower split: +1.15..+2.80.)
  The 12 seeds of collect.py give the same (B16: end shift +1.45 vs k9 shift +1.48; B32 +1.25 vs +1.37).

## Why a static split cannot win here (the evidence)
1. Early finishers are free. The marginal cost of a token (5.6 ns) is 2/3 of the average per-CTA time per token
   (8.4 ns): CTAs share a saturated HBM; when a 2-segment CTA (overcharged by the 1536 zone) stops ~4 us early, the
   others absorb its bandwidth. Equalizing moves its work onto CTAs that were already bandwidth-limited.
2. The loss is asymmetric (consistent with round 2's OVH sweep): undercharging a restart makes the CTA LATE 1:1
   (OVH 512: +1.17/+0.88 us), overcharging only makes it early (1024/1280/1536/1792 all within +-0.1). k77's 1536 sits
   on the free side; XF_S 768 is slightly below the refit cost (4.63 us = ~830 tokens) and loses a little (+0.1..0.3 net).
3. The max is set by run-to-run noise, not by systematic per-CTA cost: per-CTA run-to-run std 0.5-1.2 us; the argmax CTA
   changes every run (top-5 overlap 0.3-1.1 of 5, fit/rep.py); per-run max-p50 2.4-4.7 us vs max-p50 of the 5-run
   median ends 2.0-2.9. The "systematic" between-CTA spread (std 1.8-2.5 us) is the early tail (point 1).
4. A per-CTA-index (hardware position / launch order) component exists (std 0.29 us, split-half corr 0.59, adjacent CTA
   pairs alike) but a leave-one-out per-index correction lowers the per-case max by 0.00 us on average (fit/resid_idx.py).
=> XF stays OFF by default (K2_DEFS XF_ON enables it). Nothing to merge from round 3.

## What could still move d16f / d32f (not done here)
- The recoverable part is the TAIL under-utilization: once fewer than ~20-30 CTAs are still running (each keeps ~4 lanes
  x 3 stages x 16 KB in flight), HBM is no longer saturated. That is noise-driven, so only a dynamic scheme fixes it:
  e.g. static ranges for ~95 % of the work plus a stealable tail per request, claimed by one red/atom per CTA issued a few
  tiles before its static end (latency hidden), or a "finished CTAs" counter read asynchronously that lets the last CTAs
  deepen their prefetch. Round 2 showed the two traps (atomic latency in the lane path, interleaved tiles: ~6 us each).
- Upper bound guess: half of the p50 -> max interval, ~1-1.5 us at d16f/d32f.

## Mapping onto the XD verify kernels (same zone scheme: u_b = s_b + NEWW + OVH, OVH at each request start)
- Model: cost_c = t * tokens_c + w * newblock_c + s * (segments_c - 1), fitted per build (XD OVH 1024 / XC OVH 256) from
  XTR stamps the same way (collect.py with the XD workspace + kernel; per CTA k6/k4 vs features; features from the XD
  split: u0/u1 as in lines `u0 = max(max(c0, P0) - P0 - OVH, 0)`, `u1 = min(c1, P0 + OVH + NUb) - P0 - OVH`).
- FIRST check the bandwidth-sharing test above: if the fitted marginal t is well below the average per-CTA time per unit
  (G * T / total units), early finishers are absorbed and a static split gains ~nothing (as here). If t ~= average (each
  CTA limited by its own SM: q_len > 1 verify tiles do more MMA per byte), early finishers are wasted capacity and an
  equal-cost split pays off. The XC_OVH 1024 -> 256 win (v32f -0.8, B24 -0.8, B64 -1.4) says OVERcharging costs there,
  i.e. the XD/XC kernels are NOT in the free-early-finisher regime: the model is worth fitting for them.
- Split code (drop-in for warp 0's scan in xd_*): zones S = round(s / t) before every request but the first, unit
  u_b = s_b + NEWW (NEWW keeps its role as the new-block weight; refit w and set NEWW = w / t), grid fixed by
  NU0 = sum(s_b + NEWW) + (B - 1) S, greedy rounds exactly as xf_run1f's XF_ZC block (first cut zone -> zone shortened to
  end at the grid point, later positions shifted; 0-4 rounds). Then:
  * `const long long NU = pre[B];` -> NU0 (grid; the last CTA's range runs to NU0 >= pre[B]), also `NUu = (u32)run_` in
    the XB_VS2 block -> NU0;
  * every `P0 + OVH` / `- OVH` / `OVH + NUbj` (XB_VS2 lane block, XD_PTP == 3 prefetch, the segment loop u0/u1) -> the
    per-request zone ovb_b = pre[b+1] - pre[b] - sq[b] - NEWW (XF uses `#define XF_OVB(b)`), pre[0] = 0 and ovb_0 = 0.
  * cost: ~0.2 us fixed + ~0.25 us per round on warp 0 before the length barrier (measured here); hide it by running
    the rounds while the other warps issue the page-table L2 prefetch, or skip the rounds (plain zones of S, zone 0
    removed): a cut zone then wastes <= S units, which is harmless in the bandwidth-bound case only.
