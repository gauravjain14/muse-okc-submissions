# xe worker log (XE section: d1s xe_r1_1, v1s xe_b1s_8c6, d16s xe_dm_8_4_2)

## Tools (work/xe/)
- `mkslim.py IN OUT`: XE-only slim copy (compile ~1 s) for timing d1s/v1s/d16s.
- `ab.py A.py B.py ... [--env 'DEFS_A;DEFS_B']`: interleaved A/B over many graph instances (seeds x captures),
  judge-like (pairs with rotated order). Per-instance "sticky" bimodal launch mode => per-instance noise ~0.3 us;
  A/A over 20 instances: d = -0.001 +- 0.066 (d1s), -0.017 +- 0.043 (v1s), +0.050 +- 0.028 (d16s).
- `trace.py CAND SHAPE`: %globaltimer stamps (existing XT() hooks, K2_DEFS=K2_TRACE), 32 ns resolution.

## Baseline k73 d1s timeline (ns from first CTA entry, median over CTAs)
page-table hop -> K/V issued 640 | K arrives (first S block) 1408 (1248..1600) | softmax 1648 | PV 1760 |
CTA bar 1888 | CTA combine+push 2080 (max 2272) | owner mbarrier complete 2432 | recv loaded 2528 |
merge 2688 | store issued 2880.

## Ablations (d1s, 24 inst, raw replay delta vs k73 7.667)
- owner skips merge+norm after the wait: -0.364 +- 0.021  => owner tail is ~0.36 us of critical path
- owner stores before waiting: -0.275
- no K/V loads (compiler also drops the page-table load): only -0.176 (owner prologue + pre-warm become critical)
- page-table load replaced by synthetic rows: -0.28 +- 0.09

## Experiments
### Measurement notes
- The %globaltimer/XT trace build distorts heavily (stamps in the K/V issue loop): v1s trace build is +0.5 us slower and
  shows a 930 ns issue phase that is ~4x shorter in the real build. Use replay A/B for decisions; ablations (non-trace) for budgets.
- Per-instance noise: need >= 32 instances (8 seeds x 4 captures) for +-0.015-0.02 us stderr on d1s/v1s.
- No measurable i-cache effect for k73 (warm-code vs cold-code and a 256 KB i-cache thrash kernel before each replay: same time).
  But code first executed at the critical moment (data-dependent branches) costs ~0.4-0.8 us per cold block (why XE_PREW exists).
- mbarrier.test_wait spinning on barriers that receive remote st.async complete_tx slows the arrivals (OM=1 16 barriers,
  TW1 spin: +0.05, OG2 2-barrier poll: +0.094 +- 0.015). Keep try_wait.
- Kernel-start constant-bank misses on different 64-B lines overlap (microbenchmark: no cost); 2nd dependent DRAM hop +0.3 us.
- mma.sync e4m3 (m16n8k32) on sm_90 is emulated (F2FP + 2 HMMA): no FP8 mma.sync gain. F2FP unpack ~2.2 cyc/warp-instr,
  MUFU.EX2 8, MOVM 4, HMMA issue 4.3 (dependent latency 24.4).

### Ablations (replay deltas)
- d1s: math (S..PV) 0.48 us, owner merge+norm 0.37, CTA combine 0.14.
- v1s (with CPA): math 0.46, owner merge+norm 0.25.

### Experiments (A/B vs k73, d = candidate - k73, +-stderr)
- d1s owner online merge, 16 mbarriers + test_wait poll (XE_OM=1): +1.2 us (in-kernel iterations ~0.8 us each).
- d1s owner online merge, flag-in-data (NaN sentinel, weak st.shared::cluster, volatile LDS poll) (XE_OM=2): ~+0.5 us
  (release arrive -> MEMBAR.GPU +0.3 us; fixed with fence.cta; still slow polls / cold merge blocks).
- d1s owner 2 mbarrier groups (merge first-landed 8 while the other lands) (XE_OG2): +0.094 +- 0.015.
- d1s split-K independent HMMA accumulators (XE_SPLITK): +0.03; owner wait CTA scope (XE_W0CTA): +0.04 (noise level).
- d1s K loads issued before V (XE_KF1 4/6: needs coherent loads + bar fence, ptxas hoists .nc loads): +0.06.
- d1s owner weights by ex2.f16x2 (XE_OWH): +0.11; test_wait spin (XE_TW1): +0.05.
- d1s FP8 wgmma S in xe_r1_1 (XE_QG, W21 layout): -0.01 +- 0.02; + cp.async staging (XE_CPA1): +0.03..0.055.
- d1s generalized kernel xe_b1s_1: +0.155; +CPA +0.095; +QG1 +0.098; +both +0.012 (still not better than xe_r1_1).
- d1s ablate F2FP (replace by PRMT+SHL): -0.02/-0.004 (conversion not the limiter).
- v1s **cp.async K/V staging (XE_CPA, 4 lanes x 16 B per key row, 1 shuffle per 8 keys, K group then V group): -0.089 +- 0.014** (KEEP)
- v1s clusters of 5: +0.29; 7 / 8 (one token): only 15 clusters co-resident (not usable); C8 (2 tokens): +0.21; CL16: +0.64.
- v1s ex2.f16x2 for P (XE_EXH): +0.05 +- 0.04.
- d16s: all passes in flight is worse: 8_4_2b +0.45, 8_8_1 +0.45, 8_4_2s +1.1, new coalesced cp.async all passes (8_4_2c) +0.49,
  per-request both heads 8_4_2h +0.56. (DRAM random-row rate bound; 2 sequential passes pipeline better.)
- d1s owner merge lane-parallel weights (XE_OLP: lane s computes source s's ex2, shuffled; FFMA accumulation):
  -0.056 +- 0.038 (40), -0.041 +- 0.029 (64)  => KEEP (default on)
- v1s the same for its 6-source owner (XE_OLP2): +0.031 +- 0.036 (dropped)
- d1s/v1s CTA-merge weights lane-parallel (XE_CLP): +0.022 / +0.011 (dropped)
- d1s early row-max exchange (bar 5 after softmax, CTA-merge weights computed during PV) (XE_EM): -0.039 +- 0.026 then
  -0.012 +- 0.024 (100 inst): not significant (dropped)
- d1s 8 compute warps x 16 keys in the round-1 kernel (xe_r1_8): +0.096 +- 0.015 (dropped); generalized+QG+CPA+D8: +0.08
- d1s FP16 pushed partials (XE_H16): +0.075; owner pre-warm off (XE_PREW=0): +0.058 (pre-warm is worth keeping)
- v1s owner pre-warm (XE_PREW=2): 0.000 +- 0.026
- d1s keep the s-dependent miss-path arithmetic below the K/V issue (XE_NOHOIST): +0.03 +- 0.03
- d1s sub-ablations: no max shuffles -0.005, no S HMMA -0.011, no PV HMMA -0.045 (vs whole math region -0.46!):
  the math region is not one dominant latency chain; removing it entirely also removes ~200 instr/warp of issue.
- per-graph-instance timing is sticky (+-0.3 us, split halves agree), not output-address or CTA-placement related
  (placement identical across instances); plus a slow upward drift during a run (HBM heating?) -> always interleave.

## Final candidate (work/xe/cand.py = k73 + XE_OLP (xer1) + XE_CPA (xer2)), A/B vs k73 same session
- slim (XE only), 100 instances: d1s -0.048 +- 0.024, v1s -0.108 +- 0.026, d16s -0.023 +- 0.017 (d16s code unchanged)
- slim, 96 instances: d1s -0.041 +- 0.025, v1s -0.089 +- 0.022
- full files, 64 instances: d1s -0.098 +- 0.023, v1s -0.052 +- 0.027, d16s -0.059 +- 0.018 (unchanged code: noise/bias ~0.05)
- tests 16/16 pass (full file); robust.py 145 nearby XE cases (B=1 ctx 1..1M L=1/8, soft/edges/normal, kv_scale 1/64,
  B=2..24 decode) 0 failures; robust2.py speculation-miss path (B=1 mixed/agentic spreads) 96/96 pass.
- cand_dev.py keeps every experimental variant (XE_OM/OG2/SPLITK/KF1/QG/CPA1/OWH/TW1/W0CTA in xer1, QG1/OLP2/V1CL/NT1 in
  xer2, 8_4_2c in xem) behind macros (all default off except CPA/OLP) for reference.
- d16s pass 0 by coalesced cp.async at entry + pass 1 at the register path's issue point (8_4_2d, same schedule as 8_4_2):
  +0.226 +- 0.029 (cp.async staging hurts d16s; it only pays off where the issue phase is the bottleneck, v1s).
