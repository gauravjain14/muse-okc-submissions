# QC: shallow-queue streaming (RV's in-flight cap) on the bs1 kernel moe1t_4 -- worker notes (2026-11-03)

## Status (top) -- CLOSED 2026-11-03: the in-flight cap does not pay at bs1; nothing banked
- 3 jobs (box 1, box 1/0, box 0), knob build with identical SASS for every cap pattern + compile-time forms, cold-ordered
  qcinj (t3/B1/b1inj.py + 'qc' knob), 100 reps each, 10-13 interleaved moe_i2 controls per job.
- Cap 2 / 3 anywhere (everywhere, start only, phase A, exchange windows, phase B / expert changes): within noise of the
  twin (cap code, no cap): ratio of means cap 2 vs twin -0.6..+0.6%, never >= 1% vs moe_i2 (best RoM +0.94% one copy
  job 1, +0.60 job 2, +0.29 job 3). Cap 1 in ANY window loses: start -6.7%, phase B -1.4%, last exchange + first B change
  -4.4..-4.8%, everywhere -6.4% (RoM).
- Mechanism check (producer stamps): with cap 2 every milestone moves < 0.2 us; from stage 4 on the slot release (consumer
  + warp 9) paces the stream, not landed data or a loaded round trip. With 1 copy per SM in flight a CTA's stream is
  ~1.1-1.7 us per 36 KB stage (> the consumer's ~0.95), so cap 1 turns the call stream-bound wherever it is applied.
- Twin code effect: the QC source form (opaque try_wait loop, phase-A producer loop rolled x2) reads +0.5..+1.2% by
  medians but -0.5..+0.5% by ratio of means: placement-level noise, not a lever.
- Agrees with RV's main-kernel result (t3/RV/NOTES.md IMPLEMENTATION: every cap lost).

## Base
t3/INT/moe_i2.py, moe1t_4 sha 1c34fcc62359 (116 regs, local NVRTC 13), consumer stage loop 642 instr [1173..1814],
sched a1d08bdfeb (t3/B1/loopcmp.py). Producer code (warp 8, MOE_WPN_1TNB loop) sits AFTER the consumer loop in the
SASS (instr 1815-2171), so a producer edit does not move the consumer loop's address.

## Bound (before building; measured inputs from B1 / SU / RV)
- RV (rv_q*.py, 132 CTAs x 36 KB): round-trip latency follows in-flight BYTES; 2 copies/SM keep the DRAM rate
  (4.56 vs 4.57 TB/s), 1 copy/SM loses 7.6% (4.22 TB/s) if DRAM binds.
- bs1 time ~ start (first slot release 4.8 us) + phase A (DRAM-bound once the initial ring drains, ~1.03 us/stage)
  + phase B (consumer-bound: landed slots wait, little in flight) + end chain (~2 us, stream over = unloaded).
- Start window: the producer issues row + stages 0-3 at 0.74-1.67 us = 128 x 4 x 36 KB = 18.9 MB queued (~4 us of DRAM)
  while the consumers run their cold path (dry pass, row -> fragments, cold first stage) to the first release at 4.8 us.
  SU (run 8/9): an earlier / bigger burst moves that release LATER (E=1 +0.56 us, E=4 before the barrier +3 us); a
  later burst does not move it earlier (the stream then lags 1.4:1). The cap changes the queue DEPTH, not the start
  time: with cap 1 in the burst (4.7 MB in flight) each cold round trip is ~0.5 us instead of ~2-3 us. Stages 1-3 still
  land before the consumers need them (~5.7 / 6.6 / 7.5 us) at 1 copy/SM (~1.1 us per stage). Upper bound: the
  release path's queueing part, <= ~0.5-1 us (2-4%); unknown share.
- Phase A: rate unchanged at cap 2 (RV); latency-sensitive parts are slack-protected (exchange: XNOX bound on RF +1.16%,
  ~0.5% left after XHC/XLEAN, B1 job 11), so ~0.1-0.3 us at most.
- Phase B / end: ~0 (little in flight).
=> central guess ~0.3-0.5 us (1-2%) at bs1, almost all from the start window; risk: cap 1 or 2 in phase A starves a CTA
whose share of DRAM dips (no ring slack in flight).

## Build: t3/QC/mk_qc.py BASE OUT (BASE = t3/INT/moe_i2.py -> t3/QC/moe_qc_k.py)
MOE1T_QC: before issuing stage it (after the slot's named-barrier wait), warp 8 also waits until stage it - cap(it) has
landed: try_wait on full[(it - c) % NS], parity ((it - c) / NS) & 1; no extra barrier (the slot of stage it - c cannot be
re-armed before stage it is issued, so no parity aliasing). cap(it) = 2-bit field min(it, 31) of a 64-bit word:
MOE1T_QCK = run-time word from P.trc[32760..32761] (loaded at the producer's start, first used at stage 1), MOE1T_QCW =
constant word. MOE1T_QCF selects the wait's source form (SASS search below).

## SASS (local NVRTC 13, dev/wpb1c_sass.py + t3/B1/loopcmp.py)
| form | moe1t_4 | consumer loop |
|---|---|---|
| base i2 | 2488 instr 1c34fcc62359 | 642 [1173..1814] sched a1d08bdfeb |
| QCF=0 (mbar_sleep) / 1 (mbar_wait) / 4 (test_wait) | 2576 / 2568 / 2552 | 646 [1188..], sched 051071233f: FLIPPED (the partial-store `if (tig == 0)` becomes a BSSY branch instead of predication) |
| QCF=2 (one opaque PTX try_wait loop) QCK | 2432 ebd89ea787f7 | 642 [1173..1814] sched a1d08bdfeb (registers renamed, sim-exact 0.992) |
| QCF=3 (= 2 + suspend hint) QCK | 2440 b65b75166fe6 | same: schedule-identical |
| QCF=3, QCW=0xAA..AA (cap 2) | 2440 06cf30194c21 | same |
QCF 2/3: the producer's phase-A k loop is unrolled x2 instead of x4 (fewer instructions); 116 regs, no spills.

## Job 1 (box 1, t3/QC/job1.sh + job1_cfg.py, qcinj 100 reps, jm bs1 seeds, cold ordering: every setting follows an
## i2 control replay; 12 interleaved i2 controls, medians 23.76-23.90 us; % = speed vs the per-rep control mean, + = faster)
Smoke: hang1 bs 1,2 ok for QCW cap 1 / 2 / 3 and the knob build. Correctness checked on reps 0-1 for every setting.
| setting (knob build qk / qkb, one SASS) | vs i2 | ne2 / ne3 / ne4 / ne5 delta us |
|---|---|---|
| fA / k0 / k0b / fB (word 0: no cap) | +1.25 / +1.24 / +0.97 / +0.68 | ne3 -0.31 / -0.31 / -0.23 / -0.07 |
| c3 (cap 3 everywhere) | +0.40 | ne3 -0.15 |
| c2 / c2b (cap 2 everywhere) | +1.03 / +1.59 | ne3 -0.26 / -0.37 |
| s2 (cap 2 for stages 0-3 only) | +1.25 | ne3 -0.28 |
| c2A (cap 2 in phase A only) | +1.08 | ne3 -0.28 |
| c1 (cap 1 everywhere) | -6.02 | ne3 +1.70, ne5 +2.73 |
| s1 (cap 1 for stages 0-3 only) | -6.43 | ne2 +1.78, ne3 +1.50 |
| s1c2 / s1c2b (cap 1 start, 2 after) | -4.37 / -4.06 | ne3 +1.18 / +1.11 |
Reading: vs the no-cap setting of the SAME SASS (k0 mean +1.04%), cap 2 is +0.0..+0.5 (noise), cap 3 -0.6, cap 1 -5..-7%.
The start window is not helped by a shallower burst (s2 = k0), and cap 1 there costs ~1.5 us: the stream at the start is
per-CTA latency-bound with one copy in flight. Phase A / B: no change at cap 2 (as RV's rate data predict) and no
latency-chain gain visible.

## Job 2 (box 1, t3/QC/job2.sh, job2.log): producer stamps (qcprobe, MOE1T_QCTS knob build, 8 seeds x 4) + code-effect A/B
Stamps (ne 3, us after the first CTA's producer start, CTA medians). NOTE the slot-wait stamp follows a
BAR.SYNC.DEFER_BLOCKING, so it records the producer's ARRIVAL (as B1 found for warp 9); 'issued' (after the cap's
TRYWAIT, which blocks) = max(slot release, landing of stage it - cap).
| setting | stage 0 / 1 / 2 / 3 issued | stage 4 (slot 0 released) | stage 11 | stage 17 (last) | event (gross) |
|---|---|---|---|---|---|
| k0 (no cap) | 0.32 / 0.48 / 0.74 / 1.00 | 4.10 | 11.47 | 18.06 | 29.20 |
| c3 | 0.33 / 0.50 / 0.77 / 2.23 | 4.20 | 11.58 | 18.08 | 29.12 |
| c2 | 0.32 / 0.49 / 2.24 / 3.15 | 4.23 | 11.64 | 18.09 | 28.98 |
| s2 (cap 2 in stages 0-3) | 0.32 / 0.52 / 2.19 / 3.09 | 4.08 | 11.44 | 18.00 | 28.91 |
| s1 (cap 1 in stages 0-3) | 0.32 / 1.87 / 3.61 / 5.00 | 5.41 | 11.91 | 18.62 | 31.18 |
| c1 | 0.33 / 1.87 / 3.61 / 4.93 | 6.00 | 13.82 | 20.61 | 31.39 |
- First copies: stage 0 lands 1.55 us after issue with one copy per SM in flight (c1), stage 0 / 1 land 1.9 / 2.7 us
  after issue with two (c2). With cap 1 the steady stream is ~1.1 us per stage per CTA (stream-bound, > the consumer's
  ~0.95): c1 = -6%. s1: slot 0 is released 1.3 us LATER although stage 0 landed earlier: the first release follows stage
  1's landing (3.61) by 1.8 us, i.e. the consumers' start path is not what a smaller burst speeds up.
- Cap 2 / 3: every milestone within 0.2 us of k0 (stage 4 +0.1, last stage +0.03): the slot release, not the data,
  paces the stream from stage 4 on; nothing on the path waits on a loaded round trip that the cap shortens.
bench (qcinj 100 reps, 10 interleaved i2 controls 23.81-23.90 us; + = faster than i2):
| build | copies |
|---|---|
| twin k0 / k0b / k0c (knob, word 0) | +0.64 / +0.57 / +0.69 |
| knob c2 | +1.18 |
| w2 / w2b / w2c (compile-time cap 2 everywhere, sha 06cf30194c21) + fillers fA / fB (same SASS) | +0.95 / +0.75 / +1.01, +0.68 / +0.97 |
| w2s (compile-time cap 2 for stages 0-3 only, 508a78a12a2a) | +1.15 |
| w3 (compile-time cap 3, 8891323abe2e) | +0.54 |
Reading: every QC-form build is +0.5..+1.2% vs i2, the cap value matters <= ~0.4% (cap 3 lowest). With job 1 (twin
+0.68..+1.25, 4 copies) the producer-code change itself (opaque try_wait loop: phase-A k loop rolled x2, 2440 vs 2488
instructions, consumer loop registers renamed at the same address) is the larger part. Not separable further without
a no-cap form of the same code.

## Job 3 (box 0, decisive per the coordinator; t3/QC/job3.sh / job3_cfg.py / job3.log, qc_job3.json)
Windows (per routing, from ne): pB = it >= 4 ne - 2 (last expert's exchange fetch, A -> B transition, all phase-B expert
changes), lx = it in [4 ne - 2, 4 ne + 3] (last exchange + first B change), x = it >= 5 (every exchange window
[4q + 5, 4q + 9]). Two module copies of each knob setting (qk, qkb) + compile-time cap 2 (w2, w2b); 13 i2 controls
(medians 23.57-23.78 us, means 23.14-23.44). % = speed vs i2 (+ = faster): median-ratio / ratio of means (each-vs-each).
| setting | copy 1 | copy 2 |
|---|---|---|
| twin k0 (word 0) | +0.84 / +0.32 (-0.34..+0.99) | +0.17 / -0.34 |
| pB2 | +0.55 / +0.29 | -0.04 / -0.25 |
| pB1 | -1.34 / -1.50 (-2.15..-0.84) | -1.07 / -1.36 |
| lx1 | -5.02 / -4.39 | -5.44 / -4.80 |
| x2 | +0.47 / -0.38 | -0.12 / -0.64 |
| w2 (compile-time cap 2) | +0.80 / +0.09 (-0.57..+0.76) | +0.76 / +0.03 |
Stamps (ne 3): pB2 = k0 within 0.25 us at every stage; pB1 / lx1 issue phase-B stages 1.1-1.5 us apart (stage 10 at
10.99 vs 10.34, stage 12 13.75 vs 12.31): the B phase becomes stream-bound at 1 copy in flight.
=> below the coordinator's >= 1% bar in every window; CLOSED.

## Ratio of means over all jobs (qc_job1/2/3.json; % speed vs the control mean, each-vs-each range)
job 1: twin +0.46 / +0.19, c2 +0.25 / +0.94 (+0.38..+1.74), c3 -0.04, s2 +0.46, c2A +0.46, c1 -6.35, s1 -6.71, s1c2 -4.65.
job 2: twin -0.29 / -0.49 / -0.29, knob c2 +0.60 (+0.12..+1.30), w2 +0.20 / +0.31 / +0.37, w2s +0.40, w3 -0.40.
job 3: see above.
Control-copy spread inside one job: 1.3% (job 3 means), so single copies under ~0.7% are not separable.

## Files (t3/QC/)
- mk_qc.py BASE OUT: MOE1T_QC (+ MOE1T_QCK / MOE1T_QCW / MOE1T_QCB / MOE1T_QCF form / MOE1T_QCTS stamps); default build =
  base SASS (QCW=0 also = base, 1c34fcc62359). moe_qc_k.py (job 1 build) and moe_qc_2.py (+ QCTS) = mk_qc.py(moe_i2.py).
- qcinj.py (b1inj.py + 'qc' knob words in trc[32760..32761]), qcprobe.py (producer stamps under cap patterns).
- job1..3 .sh / _cfg.py / .log, job2_probe.py, job3_probe.py, qc_job1..3.json.
- Not run (nothing banked): check_mt full list, hang1 16 sizes. Smoke: hang1 bs 1, 2 for compile-time caps 1 / 2 / 3, the
  knob build and the stamp build; reference check on reps 0-1 of every cap setting in qcinj (no failures).
