# WP-B1P: bs1 (moe1t_4) per-stage consumer cost, phase B (down projection)

Worker WP-B1P, 2026-11-01. Namespace v4000-v4099, dev/wpb1p_*. Base line B dev/moe_v1513.py, line A dev/moe_v3770.py
(moe1t_4 SASS identical in both and in v1504: b3ca7bbf93c3, 136 registers, 620-instruction stage loop). Logs:
scratchpad wpb1p/logs/run*.log; bench JSON, trace dumps and ncu CSVs in dev/remote_out/wpb1p_*.

## 1. Files and switches

- `dev/wpb1p_mk4000.py BASE OUT` (applies to both lines). Outputs: dev/moe_v4004.py (line B, from v1513), dev/moe_v4005.py
  (line A, from v3770). Default builds: all 16 kernels SASS-identical to the base (sass13.sh, MOE_S5R_FIT).
  (dev/moe_v4000.py / v4001.py: first draft, does not compile with the PTR define order fixed later -- not used;
  dev/moe_v4002.py / v4003.py: an intermediate switch set, default builds also identical, superseded by v4004 / v4005.)
- Switches (all off by default, moe1t only):
  - `MOE1T_BL4B` phase-B fragment reloads by lanes 0-3 only (exact: only B column 0 = lanes 0-3 is read back).
    Forms: `=1` (C++ `if (lane < 4)`, 116 registers), `=2` / `=3` (predicated inline asm, 128 registers),
    `=5` (lane limit through a run-time value: 116 registers) with its twin control `=6` (same SASS up to one immediate,
    all lanes load).
  - bounds (timing only, wrong output): `MOE1T_PBNL=2` (phase-B reload loads compiled but skipped at run time; twin
    `=3` executes them, same code up to a branch sense), `MOE1T_PBNC` (no expert change at all), `MOE1T_NSTG=2` (per-stage
    partial staging skipped at run time; twin `=3`).
  - trace `MOE1T_PTR`: shared-memory stamps only (producer issue, warp 9's per-stage bar.sync return = the CTA's
    slowest consumer warp finishing that stage, fetch go / done, out), flushed after the output store. The consumer stage
    loop stays byte-identical to the bench build (base and every variant checked). The older TRLITE / TRS stamps are
    global stores, which WP-B1c found delay the stamping warp's following shared loads by 0.4-0.7 us; warp 0's B(q)
    stamp sits exactly between the phase-B fragment reload and the stage's loads.
- Tools: dev/wpb1p_trace.py (PTR dumps), dev/wpb1p_tana.py (per-stage periods), dev/wpb1p_ncu.py + dev/wpb1p_ncuana.py
  (ncu source-counter regions), dev/wpb1p_period.py / dev/wpb1p_bper.py (old TRLITE dumps), dev/wpb1p_run*.sh.

## 2. Why phase B: what the old traces already said (v1511 TRLITE dumps, same moe1t_4)

(to be filled)

- v1511 TRLITE (issue stamps, warp 0 B(q) stamps): phase A ~1.05-1.08 us/stage with 1.2-1.28 spikes at each expert's
  first stage (fetch start); B(1)->B(2) 2.11-2.14 us vs the last expert's two stages 1.86-1.92 -> ~0.2-0.25 us per change.

## 3. Run 1 (dev/wpb1p_run1.sh, dev/moe_v4004.py, 2026-11-01 15:02-15:14 local)

Correctness: BL4B =1/=2/=5/=6, PTR, line A (v4005) default + BL4B=5: check_mt M 1 2 0 failures, 60 sequential bs1 calls 0
failures, outputs bit-equal to the base 60/60. hang1 bs 1-4 ok (bounds: wrong bs1 output by design, no hang). Idle GPU 0%.

PTR traces (us, p50 over CTAs, seeds 73235 / 72563 ne 3, 70055 ne 2; out = output stored):
| build | out ne3 | out ne3 | out ne2 | phase-A done | B change / B second period |
|---|---|---|---|---|---|
| base | 24.10 | 24.06 | 17.60 | 15.42 | 1.205 / 0.98 |
| BL4B=5 | 23.74 | 23.68 | 17.92 | 14.94 | 1.19 / 1.00 |
| BL4B=6 (twin, all lanes) | 23.71 | 23.62 | 17.63 | 15.01 | 1.195 / 0.97 |
| PBNL=2 (no reload) | 23.42 | 23.42 | 17.57 | 15.01 | 1.15 / 0.93 |
| NSTG=2 (no staging) | 23.78 | 23.81 | 17.60 | 15.33 | 1.18 / 0.98 |
- Phase-B expert-change stage ~0.22 us slower than the second stage; the 116-register schedule (BL4B / PBNL family) ends
  phase A ~0.4 us earlier in the trace. BL4B=5 = its twin: fewer wavefronts do not shorten the change.
- warp 9's done stamps in phase A lag the consumers at expert boundaries (per 1.8-1.9 then 0.4 us).

Bench (bs1 jm 100 reps, vs geomean of 4 middle controls, passes A/B/C; control spread 0.33/0.51/0.54%):
| variant | A | B | C |
|---|---|---|---|
| BL4B=5 (4 copies, mean) | ~1.0013 | ~0.9990 | ~1.0036 |
| BL4B=6 twin | 0.9994 | 0.9993 | 1.0035 |
| **PBNL=2 bound (no phase-B reload)** | **1.0199** | **1.0137** | **1.0218** |
| PBNL=3 twin | 1.0048 | 1.0017 | 1.0056 |
| NSTG=2 bound (no staging) | 0.9946 | 0.9901 | 0.9899 |
| NSTG=3 twin | 0.9853 | 0.9814 | 0.9896 |
=> reload bound +1.45% at equal schedule (n2/n3), +1.85% vs base; staging (boundary part) bound +0.5% at equal
schedule; BL4B gains ~0 (the reload cost is per instruction, not smem wavefronts). bs8 placebo 0.996-1.003.
(passes B/C JSON lost to a script bug; ratios from the logged median speedups.) ncu failed: option renamed in 2025.3.

## 4. Pending at hand-back
- run 2 (dev/wpb1p_run2.sh, v4006, launched 15:21): ncu source counters (base / PBNL=2 / PBNL=3), DPH=300/500/700 de-phase
  vs twin DPHC, PBNL repeat. Output: scratchpad wpb1p/logs/run2.log, dev/remote_out/wpb1p_run2_*.json, wpb1p_tr2_*.pt,
  wpb1p_ncu_*_sass.csv.
- run 3 prepared, not launched (dev/wpb1p_run3.sh, v4008/v4009): MOE1T_BLD=1 (reload spread through the change stage,
  loads verified interleaved in SASS) x4 vs base x4, twin BLD=2 x2.
- Chained (scratchpad wpb1p/logs/chain34.sh, log chain34.log): when run 2 ends, run 3 then run 4 = the same BLD A/B
  (dev/wpb1p_run3.sh with WPB1P_TAG=run3 / run4; variants f1 f2 c l1 l2 cb l1b n2 cc l1c l2b cd l1d d5(DPH=500) f3 f4;
  l = MOE1T_BLD) through dev/modal_gate.sh, one job at a time. Results: scratchpad wpb1p/logs/run3.log / run4.log,
  dev/remote_out/wpb1p_run{3,4}_{A,B,C}.json and wpb1p_run{3,4}_tr_*.pt. Summary lines at the end of each log
  (wpb1x_sum vs c,cb,cc,cd and wpb1p_pairs: l1* vs controls, l1* vs twin l2*).
