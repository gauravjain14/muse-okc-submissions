# B1 notes: bs1 (moe1t_4) -8% push, base dev/moe_v4111.py (moe1t_4 sha b3ca7bbf93c3, 136 regs, 620-instr stage loop)

## STATUS (2026-11-03, final): goal (-8% bs1) NOT reached; best real gain +1.0..1.6% bs1 (candidate 2), bound analysis below
- **Candidate 2 = t3/B1/moe_b1_2.py** (v4111 + MOE1T_RF + MOE1T_XHC + MOE1T_XLEAN=2, baked; escapes MOE1T_NO_RF / _NO_XHC /
  _NO_XLEAN give v4111's moe1t): bs1 vs v4111, every candidate copy above every v4111 copy in 4 jobs:
  job 8 (knob-twin bench, cold ordering) +1.49 / +1.64 / +1.63%; job 9 (bench2 recipe, 3 vs 4 copies) ratio of means +0.97%,
  pairs +0.41..+1.72%; job 10 (bench2, all shapes, 2 vs 2) +0.38% RoM, pairs +0.53..+0.94%; job 11 +1.18 / +1.69 / +1.26%.
  Other shapes (job 10, main kernels byte-identical): bs8 -0.04, bs16 -0.37 (v4111 copies differ 0.4% there), bs32 +0.03,
  bs128 -0.05, bs256 +0.09%. Projection (v1504 anchor, job 10): 5105.6 / 5099.4 us vs v4111 5106.4 / 5108.3.
  hang1 1..512 ok, check_mt full list 0 failures; outputs bit-identical to v4111 (30 / 30 routings).
- Candidate 1 = t3/B1/moe_b1_1.py (v4111 + MOE1T_RF only): bs1 +0.95% (13 copies, 5 jobs), bench2 job 9 +1.51%, job 10
  +0.40%; validated the same way. Candidate 2 vs 1: +0.65% in jobs 8 / 11 (every c2 copy above every RF copy), -0.5 / 0 in
  jobs 9 / 10 -> +0.3% pooled (not significant on its own).
- Patch script: `python3 t3/B1/mk_b1.py BASE OUT [MOE1T_RF MOE1T_XHC MOE1T_XLEAN=2]` (reproduces both candidates from
  dev/moe_v4111.py; needs BASE's moe1t = v4111's).

## Tools (all new files in t3/B1/)
- `mk_inj.py` (v4111 -> `moe_b1_inj.py`): MOE1T_INJ run-time knobs read from P.trc[32752..32767] at entry, so every knob
  setting runs the same SASS. Sites, each removable by INJX_<site>: P producer spin / S consumer start spin / R no phase-B
  reload / K skip stage math (mask) / C consumer per-stage spin / X no polling / F fetch-done spin / W warp-9 per-stage spin /
  E end spin. With all sites removed the build is byte-identical to v4111 (sha b3ca7bbf93c3).
- `b1inj.py CONFIG`: one graph per build (shared weights as B2_SHARE_EQ), per rep (jm bs1 seeds) every setting = (build,
  knobs) timed as a cold-L2 graph replay in alternating order; `pre` / `post` untimed replays control the instruction-cache
  state; ratios vs the mean of the v4111 controls of the same rep.
- `loopcmp.py`: consumer stage-loop span / exact hash / schedule hash (registers stripped) / similarity between SASS dumps.
- `mk_smr.py` (-> `moe_b1_smr.py`): MOE1T_SMR=<consumer regs> setmaxnreg per role (consumers inc, others dec).
- Local SASS: NVRTC 13.0 + cuobjdump 13 from an old scratchpad (env.sh), `dev/wpb1c_sass.py` reproduces the judge's
  moe1t_4 hash b3ca7bbf93c3 for v4111.

## Register / schedule facts (local NVRTC 13, measured)
- Non-consumer sites (P, X, F, W, E) keep moe1t_4 at 136 regs and the stage loop's schedule identical (sched hash 019c8697e0).
- Any consumer-loop site flips the allocation: S 148, K 146, C 148, R 116 regs; combos 118 / 120 / 128 / 151. Stage loop
  620 -> 627..672 instructions, schedule similarity 0.28-0.57.
- setmaxnreg (MOE1T_SMR 168..200): launch count becomes 168, the stage loop 609 instructions (a different schedule per value);
  216 spills. Under SMR=184 the non-consumer sites keep the loop schedule identical too; consumer sites still change it.

## Job 1 (box 0, 100 reps, 30 settings; CONFOUNDED by same-build adjacency)
- v4111 controls 23.78 / 23.81 / 23.78 us (spread 0.2%).
- Cold zero-knob builds vs v4111: nc (loop identical, warp-9 loops re-rolled) -2.7..-3.1%, c -3.4%, k -4.2%, s -2.3%,
  r (116 regs) **+0.5%**; r with the reload skipped (r_nr, cold) **+1.4%** vs v4111.
- Robust despite the bias: consumer +100 ns per phase-A stage = +1.4 us at ne 3 (~1:1); skipping all phase-B stage math
  -3.3 us at ne 3, the B experts' first stages -2.3 us, the last 2 stages -2.4 us; skipping all phase-A math +0.4 us (slower).

## Job 2 (box 0, 100 reps, cold ordering: every timed replay preceded by a v4111 replay; 29 interleaved v4111 controls,
## control medians 24.03-24.19 us; ratios vs the per-rep control mean; ne-3 deltas in us)
| setting | vs ctl | ne3 delta | reading |
|---|---|---|---|
| nc0 / nc0b (non-consumer sites compiled, knobs 0; loop identical) | -2.3 / -2.7% | +0.64 / +0.69 | warp-9 code re-rolled by the W site: -2.5% code presence |
| nc_prod200 (+200 ns per producer issue) | -3.1% | +0.72 (+0.05 vs nc0) | producer has >= 200 ns slack per stage |
| nc_w9_200 (+200 ns warp 9 per stage before the slot release) | -4.4% | +1.02 (+0.35) | 10% conversion: ring slack |
| nc_fd1000 (fetch done +1 us) | -5.2% | +0.81 (+0.15) | slack ~0.85 us at ne 3; ne 2 +1.3, ne 4/5 +1.0/+1.4 |
| nc_end500 (+0.5 us before the last FXSS red) | -4.8% | +1.22 (+0.55) | end chain 1:1 on the critical path |
| nc_nx (no polling, timing only) | -0.8% | +0.25 (-0.42) | exchange bound +1.5% vs nc0 (= WP-B1X XNOX) |
| c0 (consumer spin site, 148 regs) | -2.6% | +0.74 | |
| cA100 (+100 ns per phase-A stage) | -8.0% | +2.13 (+1.4) | 1.2 : 1 (12 stages x 0.1) -> consumer on the path in A |
| cB200 / cB600 (per phase-B stage) | -6.7 / -13.1% | +1.79 / +3.75 (+1.1 / +3.0) | 0.9 / 0.84 : 1 -> consumer on the path in B |
| k0 (skip site, 146 regs) | -3.4% | +0.99 | |
| k_last1 / k_last2 (skip the last 1 / 2 stages' math) | +0.5 / +4.2% | -0.95 / -1.87 vs k0 | 1.1 : 1 (0.86 us per stage): the end is consumer-bound |
| k_B (skip every phase-B stage's math) | +6.8% | -2.44 vs k0 | only 47% of the 5.2 us removed: the next path (data) is 2.4 us behind |
| k_Bf (skip each B expert's first stage) | +2.8% | -1.46 vs k0 | 56% |
| k_q4 (every 4th stage: 3 A + 1 B at ne 3) | -2.0% | -0.34 vs k0 | A removals barely convert |
| k_first2 / k_A | -3.5 / -7.4% | -0.52 / +0.65 vs k0 | start lag ~0.5 us; all-A removal slower (data / exchange bind) |
| r0 / r0b (R site, 116 regs) | **+1.69 / +0.55%** | -0.29 / -0.18 | schedule lead |
| r_nr (R site, reload loads skipped) | **+2.83%** | -0.69 | reload bound ~+1.1..1.5% on the R schedule |
| smr184 / smr168 (setmaxnreg) | -0.05 / +0.51% | | no schedule gain |
| w_same / w_code (v4111 right after itself / after its own code with another state) | +2.9 / +3.3% | -0.61 / -0.65 | cold instruction fetch costs ~0.6 us per call (code, not buffers / TLB) |

Critical-path reading: the consumer path binds the call (delays convert ~1:1 in both phases); phase A is balanced with
the data / exchange paths (A removals do not convert), phase B is consumer-bound with ~2.4 us of data slack at ne 3
(B removals convert 50-100%). So bs1 time ~ A_end (balanced) + phase-B consumer work + end chain. The -8% goal needs
~1.9 us off that chain at ne 3: phase-B work + end chain + start.

## Job 3 (box 0): cold / warm PTR traces (v4111 + smem stamps, loop identical) + schedule replication
- Replication (cold ordering, 13 interleaved controls): r0 +0.68, rb0 +0.80, r0x +0.75, rbx +1.07% (with job 2: 6 copies,
  mean **+0.92%**, min +0.55); r_nr +2.33 / rb_nr +2.33 / r_nrx +2.59%; smr168 +0.07 / +0.42, smr176 -0.86, smr200 -0.70%.
- Identical-SASS module copies (x0 / fillers) read -0.34..+0.64%: single-copy noise floor ~0.6%.
- PTR (ne 3, p50 over CTAs): stage 0 done 4.77 (warm 4.48), A done 15.6, last done 22.05, out 24.1; B change stage period
  1.18-1.31 vs second B stage 0.90-0.96; warm instruction caches move every milestone ~0.3 us earlier.
## Job 4 (box 0): TLB warm of the later weight pages (MOE1T_TLBW, fetcher warp, loop identical): -0.7..-1.8% (dead).
  NO_OLATE escape: -0.09 / +0.03% (neutral). End-chain stamps (median CTA, ne 3; ne 2 similar):
  last stage done -> partials read 0.70 us (ne 2: 0.90), -> red 0.13, red -> all words seen 0.93 (0.45 after the slowest
  CTA's red), seen -> output value ready 0.38 (the 8-iteration serial combine: 2 SHFL + branch + MUFU.RSQ each).
  Phase A: warp 9's partial reads take +0.7-0.9 us after its bar.sync once a global store preceded them (stage 0-1 +0.13).

## Job 5 (box 0): end-chain fixes + RF (cold ordering, 14 interleaved controls)
- OFAST (parallel final combine, warp 9 only, loop identical, bit-identical): -0.98 / -1.19 / -0.66% -> rejected (its
  seen->ready got slower, 0.54 vs 0.38 us, despite 50 vs 130 instructions: the combine's cost is not its instruction chain).
- OLATE0 (output-row touch one stage earlier): +0.70 / +0.66% (job 6: -0.04 / +0.57; on RF: rfol - rf ~ +0.1%) -> neutral.
- RF (clean form `if (P.local_start != -77777)` around the reload loop = WP-B1P's PBNL=3 twin; 116 regs, loop = the R-site
  schedule up to one ISETP): +0.87 / +1.05%.
- Bitwise: OFAST, OLATE0, RF+OFAST+OLATE0 equal to v4111 on 40 / 40 routings.
- Trace correction: the PTR "stage done" stamp follows BAR.SYNC.DEFER_BLOCKING, i.e. it records warp 9's ARRIVAL; the
  "done -> partials read" gap (0.7-0.9 us) is warp 9 waiting for the consumers (diagnostic MOE1T_WKST, weak item stores,
  leaves it unchanged, job 6). So at the end: consumers done ~ 'partials' stamp; then red +0.13, all words seen at the slowest
  CTA's red + 0.45, output value +0.38.
## Job 6 (box 0): RF + OLATE0: +1.24 / +0.74 / +0.73 / +1.50% (all above every control); RF +1.06%; PRL (fragment reload as
  16 predicated LDS inside the stage block, 143 regs) -4.4 / -4.9 / -5.5% -> rejected.
## Mechanism 2 result: RF pins the allocation
- On the RF build (116 regs) the WP-B1X fetcher switches XLEAN=1, XLEAN=2, XHC, XHC+XLEAN, and FLATE / F0DLY / FDLY
  retunes all keep the consumer stage loop byte-for-byte schedule-identical (sched a1d08bdfeb, 642 instructions); on v4111
  (136 regs) XLEAN flips to 128 regs and re-schedules the loop (WP-B1X). BL4 / S0DLY change the loop slightly (sim 0.98-0.99).
## Job 7 (box 0, 13 controls): RF +0.83 / +1.24 / +0.83; RF+XLEAN=1 +1.10 / +0.72 / +1.12; RF+XLEAN=2 +1.61; **RF+XHC +2.01**;
  RF+BL4+FLATE2 +0.10 / -0.48; RF+BL4+FLATE1 -0.55; RF+XLEAN1+BL4 -2.49. Bitwise: all equal to v4111 (30/30).
## Validation of t3/B1/moe_b1_1.py (box 1): hang1 1..512 (16 sizes) ok; check_mt 1 2 4 8 9 16 17 32 33 64 65 128 129 200 256
  257 512: FAILURES 0 (max rel-L2 3.49e-3 = v4111's). SASS (local NVRTC 13): moe1t_4 de4cc3c3af82 (116 regs, 2480 instr), the
  11 main kernels byte-identical to v4111; MOE1T_NO_RF gives v4111's b3ca7bbf93c3.

## Job 8 (box 0): replication on RF (12 controls): RF+XHC+XLEAN=2 +1.49 / +1.64 / +1.63% (all above every control); RF+XHC
  +0.61 / +0.94 / +1.02 (job 7 +2.01); RF+XLEAN=2 +1.01 / +0.88 (job 7 +1.61); RF+XHC+XLEAN=1 +0.94 / +1.34; RF +1.19 / +0.61.
## Job 9 (box 0, bench2 recipe, bs1, 100 reps): cand 2 x3 vs v4111 x4 +0.97% RoM (pairs +0.41..+1.72, all >);
  cand 1 x2 +1.51% (pairs +0.47..+1.65, all >). Per ne (speedup over SGLang, v4111 mean / cand 2 mean): ne3 3.313 / 3.359.
## Job 10 (box 0, bench2, all six shapes, 60 reps, v1504 anchor): see STATUS. JSON t3/B1/b1_job10.json (also job 8 / 9).
## Job 11 (box 1): exchange headroom on RF: XNOX bound (no exchange; wrong output) +1.76 / +2.33 / +1.62% vs v4111,
  RF +0.74 / +0.74, cand 2 +1.18 / +1.69 / +1.26 -> the exchange costs +1.16% on RF, cand 2 recovers ~0.64 of it, ~0.5% left.

## Bound / mechanism summary (all numbers measured; bs1 = 24.0-24.2 us bench net at the jm median, ne 3 block 23.9-24.1)
- Floors: pure 4 x 36 KB TMA streamer 18.7 us net for 82.7 MB (gj, ~19.2 us for ne 3's 84.9 MB); moe1t without any stage
  math (MOE1T_NOCOMP, gj) 21.94 us = -9.2%. The -8% target (22.2 us) needs ~85% of all consumer math hidden.
- Critical path (job 2 knob twins): consumer delays convert 1.2:1 (phase A) and 0.9:1 (phase B); A-math removal does not
  convert (k_q4 -0.34 us for 3.4 us removed, all-A slower), B-math removal converts 47-56% (k_B -2.44 us), the last stages
  1.1:1, the end 1:1. bs1 ~ A_end (consumer = data = exchange, balanced) + phase-B consumer work + end chain.
- Mechanism 1 (fewer stage boundaries): the boundary is at most ~0.05-0.1 us/stage (second phase-B stage period 0.90-0.96 us
  vs 0.86 us math; the 0.2 us 'boundary' of the old TOBS traces was stamp overhead); only phase-B boundaries convert (6 at
  ne 3), so pairing stages bounds at <= ~0.3 us (~1.2%) while doubling the stage-loop body (cold-code dry pass +~1.2 us,
  register re-allocation). Its sub-lever, hiding the phase-B reload inside the stage block (MOE1T_PRL): -4.4..-5.5%.
  Reload bound (R-site knob, loads skipped): +1.5% on the R schedule; not realized.
- Mechanism 2 (register pinning): setmaxnreg (MOE1T_SMR 168-200) does not pin the consumer loop against consumer-side
  edits and gives no schedule gain (-0.9..+0.5%). The RF form (116 regs) does pin it against fetcher / warp-9 edits and is
  itself +0.95%; it made XHC + XLEAN (WP-B1X's +0.45%-at-equal-schedule lever, lost to a 136->128 flip on v4111) bankable.
- Mechanism 3 (DSMEM 2-CTA split fetch): remaining exchange headroom over cand 2 ~0.5% (job 11), the cluster attribute costs
  ~0.12 us (~0.5%, gj): net <= ~0: not built.
- Other measured, not convertible here: cold instruction fetch 0.6 us / call (w_code; no in-kernel prefetch path for code),
  TLB warm of later weight pages -0.7..-1.8%, OFAST parallel combine -0.7..-1.2%, OLATE0 ~0, BL4 + FLATE on RF -0.5..+0.1%.
