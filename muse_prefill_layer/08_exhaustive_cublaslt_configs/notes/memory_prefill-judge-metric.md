---
name: prefill-judge-metric
description: muse_prefill_layer judge score = 387403.9 us constant / paired speedup vs SGLang; absolute local latency is the wrong metric; judge noise ~1.4%
metadata:
  node_type: memory
  type: project
  originSessionId: <session>
  modified: 2026-10-30T10:09:58.394Z
---

Verified 2026-10-30 from 16 judge result JSONs in muse_prefill_layer/research/*official*.json:
- `baseline_us` is the constant 387403.9 in every run; score_us = 387403.9 / speedup. Speedup = geomean over 3 loads of
  sum(w*sglang_med)/sum(w*sglang_med/paired_ratio), paired ratio = median over 6 reps.
- SGLang's own absolute load geomean varies 376.9-398.1 ms between judge runs; ours 195.5-201.3 ms. Codex's "197.4 ms local"
  was our absolute latency, so it never compared to the judge (194.1 run had ours = 199.5 ms abs, SGLang = 398 ms).
- Same code twice: bundle14 1.9748 vs 1.9475; bundle12 1.9561 vs 1.9500. The 1.9955 best is probably a favourable draw.
- Per-us score sensitivity (Codex harness): case0 0.040%, case1 0.014%, case2 0.012%, case3 0.004%, case5 0.0025%,
  other 16K cases 0.0003%. Turn-1 (T=753) is the lever: MoE 284.5 us streaming ~906 MB at ~3.2 TB/s.

- Judge draws a new router seed every run (reference.weights_fields). Per-shape ratios move 3-5% between draws and
  candidate ranking flips (case 5: b23 2.054 fixed seed / 1.935 fields 11 / 2.013 fields 22). Codex's fixed
  ROUTER_SEED 728034927264498030 is a favourable draw, so its tiny "gains" were partly overfit.
- research/judge/replica.py (2026-10-30): SGLang-paired, judge overhead formula (2*one - two), --panel of fields
  seeds. Single fixed-seed calibration: b23 local 1.9963 vs judge 1.9955; judge ~ local x 0.989 +- 1.1%.

- Position bias (found 2026-10-30): timing [sglang, c1, c2] then reversed keeps c1 in the middle; the last of three
  back-to-back long graphs runs slow, so c1 gains 0.4-1.1%. replica.py now pairs each candidate with its own SGLang
  replays (AB/BA, judge-faithful) and rotates candidate order; replica_v1_biased.py is the old one.
- AGENTS.md: final ranking = medians of organiser reruns (held-out data, nearby shapes) -> only expected value counts.

**How to apply:** compare candidates by paired speedup vs SGLang baseline.py (judge method), never by absolute us,
and over a panel of judge-style router draws (--panel 11,22,33), never one seed.
Related: [[goal-prefill-rank1]].
