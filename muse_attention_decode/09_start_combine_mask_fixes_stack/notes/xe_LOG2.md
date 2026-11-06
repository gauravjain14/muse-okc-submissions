# xe round 2: d16f / d32f (decode FULL q_len 1, base run1f `attn_1_0_1_0g` / `attn_1_0_1_0`)

Work copy: work/xe/cand2.py = runs/k75.py + an `_XF_SRC` section built by work/xe/mkxf.py (+ xf_py.txt):
`xf_run1f` = a modified copy of the base run1f compiled against the base SRC prefix (everything before run1f:
`SRC[:rindex("template", 0, index("void run1f(const Params& p)"))] + _XF_SRC`, own NVRTC module, own workspace
copy, own `kernel_fn` override: q_len 1, full, page 1, tp8, 2 <= B <= 64, page-table width >= 4096, FLAT plan).
All XF changes are behind macros (XF_START, XF_PTB, XF_KPF, XF_ZC).

## Measurements of the base kernel (K2_TRACE TR stamps, trace_base.py; ns from the first CTA entry)
d16f (median CTA): length barrier (seq_lens + prefix) 880 | c0/c1 (FP64 div, every warp) 1184 | b_begin ballot 1344 |
first page-table rows issued 1712-1888 | rows arrived 2240-2320 | 3 ring stages issued 5.7 us | first tile ready 6.2-7.5 us |
segment loop end (k4): min 74.8, p10 79-81, p50 83-84.5, p90 84.7-86, max 85.6-88.3 | combine done (k6) max ~87.
d32f: rows issued 2.07 us, first tile 8.3 us, k4 p50 161-162, max 164.4-166.6.
- Start: ~1 us of avoidable work between the length barrier and the first row load: the FP64 split redone by all 16 warps
  (~300 ns), the b_begin ballot loop (~160 ns), and ptxas hoists 1/k_scale (new-token path) above the row loads, so every
  warp waits ~450 ns for the k_scale DRAM load right there.
- End spread is SYSTEMATIC per CTA for given data (per-CTA end stdev over runs 0.9-1.3 us, spread of per-CTA medians 11 us /
  9 us), not per SM (CTA->SM changes every run; no smid pattern). Sources: (a) the ovh (1536 virtual units per request)
  attribution: CTAs whose range starts in a request's zone (or CTA 0) get up to 1536 fewer real tokens and finish 5-10 us
  early; 2-segment CTAs (8542 vs 10111 tokens) end ~5 us early; (b) pure single-segment CTAs still vary ~3 us (83.4-86.7):
  data-dependent per-CTA throughput.
- No-compute ablation (ABL_NOCOMPUTE): ends ~1.5-2 us earlier (p50 82.1, max 84.6): the tile compute is not fully hidden.

## Experiments (A/B vs k75, interleaved, d us)
- XF exact copy (XF_START=0, XF_PTB=1): d16f -0.10 +- 0.06, d32f +0.13 +- 0.04 (module-level noise).
- XF_START (scales at entry, split by warp 0 with a host 1/G + exact floor fix): first rows issued 0.3-0.5 us earlier but
  d16f -0.06, d32f +0.21 (no gain: the bandwidth is busy with the page-table L2 prefetch at that time).
- page-table L2 prefetch all after the first rows (XF_PTB=0): d16f +0.36, d32f +1.32 (the early prefetch matters).
- K/V L2 prefetch of tile it+NPRO+1/2 (deeper DRAM queue than the 3-stage ring): d16f +7.3, d32f +49 (catastrophic).
- Restart-zone collapse (zone charged only if no CTA boundary falls in it, iterated): d16f +4.0, d32f +3.2 (creates tiny
  segments: the zones are what protects against them).
- OVH (base knob): 1024 -0.12/-0.11 (12 inst), 768 +0.16/+0.18, 512 +1.17/+0.88; XF_OVH 1152 / 896: within +-0.1;
  32 instances: 1024 +0.08/-0.10, 1280 -0.02/-0.13, 1792 -0.02/+0.10 (all within +-0.09): a flat optimum.

## Dynamic groups (xf_dyn, work/xe/xf_dyn.cu): 4 x XF_GS lanes take a group slice's 64-token tiles in order (4 static,
## then one atomic claim per tile issued one tile ahead), per-lane publish on request switch, token-count completion,
## new tokens by lane 0 of CTA b mod G. Correct (robust_f.py: 132 full-layer cases B 2..64, ctx 5K..120K,
## agentic/mixed, coherent/soft: 0 failures) but much slower:
- XF_GS 12: d16f +17.2, d32f +29.1; XF_GS 6: +13.8 / +22.6; XF_GS 33: +28.3 / +60.1 us.
- Diagnostics without atomics: static interleaved tiles (lane L: L + k NLG) +9.0 / +10.3; static contiguous per lane
  +3.2 / +3.5.
- Lessons: (1) a global atomic whose result a lane needs costs ~ the loaded memory latency (~6 us = 25 MB in flight /
  4.15 TB/s): claims must be issued >= 3-4 tiles ahead and the publish counter must not be waited on mid-stream (the base
  only publishes once, at the end, when the load has dropped); (2) interleaving tiles across a group's lanes costs ~6 us
  by itself vs contiguous ranges; (3) even contiguous static per-lane ranges lose ~3 us (switch publishes with a blocking
  atomic, Q reloads, lane 0 of 16 CTAs straggling after the new-token work, no restart-cost model).
- Group-balancing potential measured on the base end times (perfect balance inside groups of k adjacent CTAs, no
  overhead): k=2: -0.7 / -1.2 us, k=4: -1.3 / -2.5, k=8: -2.0 / -2.5, global: -3.0 / -3.8 us.

## Status
cand2.py = k75 + the XF section, DISABLED by default (needs K2_DEFS XF_ON): nothing to merge from round 2.
