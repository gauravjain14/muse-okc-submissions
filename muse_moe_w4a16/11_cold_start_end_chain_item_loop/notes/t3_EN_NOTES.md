# EN: bs128 / bs256 item transitions + end of call (moe_8_8_1_3_4 / moe_8_8_1_3_8), base dev/moe_v4111.py

## STATUS (2026-11-03): mission 1 (item transitions) CLOSED after 2 jobs, nothing banked; mission 2 (bs65-256 token
## combine) CLOSED after 3 jobs: nothing >= 0.8%; borderline candidate EN_TOK,EN_TOKM=2 (t3/EN/moe_en_5.py, bs256
## ~+0.23% over 8 copies / 3 jobs, only moe_8_8_1_3_8 changes). Details: sections M2.0-M2.4 at the end.
## Mission-2 files: t3/EN/mk_en3.py BASE OUT -> moe_en_3.py (TOK / TOK2), moe_en_4.py (+ TOK2_ORD), moe_en_5.py
## (+ TOKM / TOKA / TOKW / INA; the benched file), moe_en_6.py (+ ESIG, not benched); en_run3-5.sh / .log,
## bench/en_run3-5.json. (t3/EN/moe_en_3.py.new is a stray intermediate copy of moe_en_4.py, kept per the no-delete rule.)
## Mission 1 (below):
- Item transition (consumers): per warp per item ~1260 (bs256) / ~1340 (bs128) cycles of non-wait latency chain
  (ncu, zero perturbation). A delay inserted at every item end of ALL consumer warps converts at 0.49 / 0.57 (small,
  ~106 ns) and 0.71 / 0.64 (large, ~510 ns) per unit (bs128 / bs256; in-situ calibrated sleeps). Warp 0 alone 0.38 /
  0.54, warps 1-7 alone 0.41 / 0.36. Producer first-stage issue 0 (data is not critical at the item top); epilogue
  wake (held slot) 0.16 / 0.18.
  => BOUND if the whole transition were free: ~0.5 x 0.8 us x 9.9 / 11.1 items per CTA = ~4.1 / ~5.1 us = ~2.5% /
  ~2.6%; the irreducible part (MMA drain, stores, fence, arrive, wait, record, dispatch) leaves ~1.5-1.8% reachable
  only by a rewrite. A cut of X cycles of every warp's latency gives ~0.55 X per item: <= 50-cycle cuts are <= 0.06%
  (why WP-W0 CS/SA and WP-CGS DZ read 0). No loop-exact form removes >= 300 cycles (the chain is ~20 elements of
  10-75 cycles: i-cache misses at the loop exit / end / preamble / loop head, 2 x S2R TID rematerialisation, 3 uniform
  modulo chains, LDS record -> R2UR -> compare -> branch chains, LDL tsplit, MEMBAR + FENCE, the peeled MMA drain).
  Code layout is not steerable: EN_LAY 2/4/8 compile to v4111 byte-for-byte, bit 1 re-schedules 16/16 stage loops.
- End of call: bs256 the token combine is throughput-bound after the items (combine-start delay converts 0.61,
  down-item signal delay 0.00); bs128 the last tile's signal chain binds (signal 0.63, combine start 0.37). Bounds
  stay WP-TAIL's NO_COMBINE +1.15 / +1.39%; the end-of-call counter reset is already off the path (MOE_PAR).
- Files: t3/EN/mk_en1.py -> moe_en_1.py (probes), t3/EN/mk_en2.py -> moe_en_2.py (layout knobs); both default builds =
  v4111 SASS in all 16 kernels (wpcgs --full EXACT 67/67 stage, 106/106 producer loops). No candidate to stack.


## 0. Zero-perturbation decomposition (ncu SourceCounters of the unmodified v4111, ST's st_ncu3 full / st_ncu4
compute-only CSVs, jm seeds bs128 71637 / bs256 73011; t3/EN/en_regions.py, t3/EN/en_stalls.py; MEASURED)
Calibration c = samples / (kernel cycles x 12 warps x 132 SMs) = 3.08-3.18e-4 samples per warp-cycle.
Consumer item loop, per warp per real item (bs256 10656 / bs128 8064 warp-items per call; ~10.1 / ~7.6 items per CTA):

| region (cycles per warp-item) | bs256 full | bs256 compute-only | bs128 full | bs128 compute-only |
|---|---|---|---|---|
| item top, non-wait (loop head .. path dispatch; 84 / 75 instr) | 437 | 343 | 306 | 159 (+ long_sb) |
| item-top first-stage data wait (TRYWAIT -> BRA long_sb) | 535 | 111 | 1132 | 435 |
| path preamble + peeled last-stage tail ("mid", 48 / 46 instr) | 340 | 390 | 618 | 416 |
| item end (staging, fence, hand-off; 67 / 63 instr) | 486 | 352 | 415 | 366 |
| stage loops' in-loop data waits (per item) | 972 | 505 | 1400 | 645 |
| stage loops total (per item) | 21000 | 21300 | 21700 | 21600 |
Stall mix of the non-wait transition (bs256 full, ~200 instr per warp-item, ~1260 cycles): issue (selected) ~210,
fixed-latency dependency 'wait' ~490, no_inst (i-cache) ~220, branch/membar/barrier/mio ~100, not_selected ~90.
=> the per-warp transition is a ~1260-cycle LATENCY chain (uniform-datapath index math, 2 x S2R TID rematerialization,
LDS info -> R2UR -> compare -> branch chains, MEMBAR + FENCE, i-cache misses at the loop exit / end / preamble), not
issue: consistent with WP-CGS DZ (32-61 fewer instructions per item path, loop-exact) measuring 0.
The item-top data wait (535 / 1132 per warp-item vs 114 / 132 per warp-STAGE mid-item) says the next item's first
stage lands ~1000 cycles later relative to the consumers than a mid-item stage.

## 1. Job 1 (t3/EN/en_run1.sh, box 0, log t3/EN/en_run1.log): item-transition delay slopes (MEASURED)
Probes: t3/EN/mk_en1.py BASE OUT -> t3/EN/moe_en_1.py (default build = v4111 SASS, sha 595ff80197a9 / 571003e3fd87).
Codegen: any warp-dependent BRANCH at the item end or a probe at the item top re-schedules all 16 stage loops (wgmma
loops 588 -> 645 instr); the branch-free form (nanosleep of a register = mask bit x N, EN_DCF=7) keeps bs256 8/8 and
bs128 6/8 stage loops exact (2 mma.sync loops: one UMOV hoisted), producer / epilogue / signaler / combine probes keep
16/16. Every probe build of a job has identical loops (only immediates differ) -> slopes are vs the twin (all sites, N~0).
bench2 jm 60 reps, fillers both ends, 3 v4111 controls, 2 copies each; Delta vs twin (us), per item (bs128 7.6 / bs256
10.1 items per CTA):

| probe (nanosleep N, isolated-calibration length) | bs128 copies vs ctl | Delta us / per item | bs256 copies vs ctl | Delta us / per item |
|---|---|---|---|---|
| controls a1/a2/a3 | +0.09 / +0.10 / -0.18 | | -0.01 / +0.03 / -0.04 | |
| twin z (warp 0 N 1, others 0; producer 0; epilogue 0) | +0.49 / +0.46 | +0.79 vs v4111 | +0.22 / +0.35 | +0.60 vs v4111 (loops identical!) |
| c: warp 0 item end N 200 (~155 ns) | +0.81 / +0.67 | +0.40 / 53 ns | +0.53 / +0.76 | +0.63 / 62 ns |
| r: warps 1-7 item end N 300 (~410 ns) | +1.19 / +0.99 | +0.98 / 129 ns | +0.72 / +0.81 | +0.97 / 96 ns |
| p: producer, after every item's first-stage slot wait, N 200 | +0.54 / +0.47 | +0.04 / ~0 | +0.30 / +0.27 | -0.06 / ~0 |
| e: epilogue warps at every item wake, N 300 | +0.80 / +0.70 | +0.39 / 51 ns | +0.57 / +0.42 | +0.48 / 48 ns |
Readings: (1) the next item's first-stage data is NOT on the critical path (producer slope ~0: warp 0 reaches the item
top >= 155 ns after the data, as WP-W0's stamps said: rdy 474-565 cycles after warp 0's item end start); (2) neither
warp 0 (slope ~0.4-0.46 if the delay is really ~135 ns) nor the other warps (~0.25-0.34) converts 1:1: the ring and
the SMSP pairs absorb part (a late warp 0 gets SMSP0 to itself while warp 4 waits); (3) the held slot (epilogue) ~0.12;
(4) the TWIN, i.e. ~zero-length sleeps at 3 sites per item, costs +0.29% / +0.48%: a NANOSLEEP is not free in situ
(calibration in job 2).

## 2. Job 2 (t3/EN/en_run2.sh, box 0, log t3/EN/en_run2.log, json t3/EN/bench/en_run2.json; MEASURED)
(a) In-situ length of the probe's nanosleep (t3/EN/en_cal.py, EN_CAL build: clock64 / globaltimer around the sleep,
lane 0 of every warp, 6 jm seeds, cold L2): N 0 -> 63-68 ns (isolated calibration said ~26), N 200 -> 171-173 ns,
N 300 -> 303-306 ns (not 410), N 600 -> 558-588 ns. Item ends per CTA (= real items): bs128 9.94, bs256 11.09.
Job 1 re-read with these: c (warp 0, +106 ns) 0.38 / 0.54; r (warps 1-7, +240 ns) 0.41 / 0.36; e (+240 ns) 0.16 / 0.18;
p 0 / 0.
(b) bench2 jm 60 reps, fillers both ends, 3 v4111 controls, 2 copies each; twin z = every probe site, N 0:

| variant | bs128 copies vs ctl (%) | Delta vs twin, us (per unit) | bs256 copies vs ctl (%) | Delta vs twin, us (per unit) |
|---|---|---|---|---|
| controls a1/a2/a3 | +0.12 / +0.16 / -0.26 | | +0.14 / -0.10 / -0.02 | |
| z twin (all 8 warps nanosleep 0 at every item end, ~65 ns) | +0.41 / +0.17 | +0.22 vs v4111 (2 loops differ) | +0.30 / +0.11 | +0.45 vs v4111 = 41 ns/item (0.63) |
| g all warps N 200 (+106 ns per item end) | +0.60 / +0.45 | +0.52 (52 ns/item, 0.49) | +0.51 / +0.67 | +0.67 (60 ns/item, 0.57) |
| h all warps N 600 (+510 ns per item end) | +2.50 / +2.46 | +3.58 (360 ns/item, 0.71) | +2.16 / +2.16 | +3.62 (326 ns/item, 0.64) |
| k combine start, every CTA, N 600 (+510 ns once) | +0.36 / +0.43 | +0.21 (0.37 of 0.575 us) | +0.30 / +0.51 | +0.35 (0.61) |
| s every down-item counter release, N 600 | +0.62 / +0.28 | +0.36 (0.63) | +0.25 / +0.28 | +0.01 (0.00) |

## 3. What was ruled out / not built
- Producer-side item transition (first-stage issue latency, readiness, claim): slope 0 -> no lever (also explains why
  ring/L2-prefetch style ideas cannot help at the item top).
- Held slot (epilogue wake -> TMP copy -> release): slope ~0.17; the full release-at-wake bound is WP-IT2's BH +0.2%.
- Consumer micro-cuts: per-element latency 10-75 cycles at slope ~0.55 -> <= ~0.05% each; history (W0 CS/SA/QM/HP,
  DZ, AQW, ESP, GC) agrees. Not built.
- Code layout of the dispatch (i-cache of the end / preamble): mk_en2.py EN_LAY knobs: bits 2/4/8 (case order,
  __builtin_expect) give v4111's exact SASS (ptxas normalises), bit 1 (if/else swap) re-schedules all 16 stage loops.
- End: the signaler hop for the last down items (~0.21 us on the bs128 chain x 0.63 = ~0.08%) and token-combine
  rewrites (inline edits re-schedule the loops; WP-E2's TOKF frame measured -0.4..+0.3%) not built.

## 4. Structural levers left (for the coordinator; each re-schedules every stage loop = +-1% lottery on top)
1. Per-path item loop: each consumer path (1-group, 2-group, wgmma n) runs consecutive same-path items in its own
   code region (end + top adjacent to its stage loop: i-cache; lane offsets and ring / queue indices kept in registers
   inside the path: no S2R / modulo chains; a successor of the same path only reads its record and re-enters the loop).
   Estimate: per-warp transition ~1260 -> ~600 cycles -> ~0.55 x 660 cyc x 11 items = ~2.5 us = ~1.2% (bs256),
   ~1.1% (bs128), before the schedule draw.
2. Two-row-block items for 1-group tiles (32-stage items alternating rb, 16 accumulator floats): halves the 1-group
   transitions (bs128 ~65%, bs256 ~50% of items) -> ~0.6-0.9%, plus a coarser tail.
3. bs256 combine throughput (k slope 0.61): spreading the last tokens / an earlier combine start; bounded by
   NO_COMBINE +1.39% and by the last tile's chain; bs128: the last tile's epilogue -> signal -> poll chain (s 0.63).

## Files
- t3/EN/en_ncu_dump.py, en_regions.py, en_stalls.py: ncu SourceCounters CSV analysis (ST's st_ncu3 / st_ncu4 CSVs).
- t3/EN/mk_en1.py BASE OUT -> t3/EN/moe_en_1.py: probes EN_DC/EN_DCM/EN_DCF (item end), EN_DT/EN_DTM (item top;
  re-schedules 7 loops even branch-free: not used), EN_DP (producer first stage), EN_DE (epilogue wake), EN_DK
  (combine start), EN_DS (down-item signal), EN_CAL (in-situ sleep calibration into __device__ en_cal[16]).
- t3/EN/mk_en2.py moe_en_1.py OUT -> t3/EN/moe_en_2.py: EN_LAY layout knobs.
- t3/EN/en_cal.py (calibration driver), t3/EN/en_tab.py (bench2 table vs controls and vs twin).
- t3/EN/en_run1.sh / en_run1.log, en_run2.sh / en_run2.log, t3/EN/bench/en_run1.json, en_run2.json.

# MISSION 2 (coordinator, 2026-11-03): the bs65-256 end-of-call token combine

## M2.0 What runs today (v4111, moe_8_8_1_3_4 / _3_8)
Inline token combine (no MOE_END_RO for MW 4/8, no MOE_E2_ORD2, no TOKF frame): every CTA, after its item loop, claims
tokens in INDEX order from CB + 4 (ECL: the next claim's atomic issued after the current token's polls); a token =
the whole CTA (256 consumer threads x 16 columns), warp 0 polls the token's tiles (one lane per pair: relaxed spin,
then a closing acquire = one more L2 round trip), CTA barrier, CG = 2 pairs per round (LDG.64 BF16 rows / LDG.128 FP32
KRED rows via __ldcg, post-norm weights in registers), one barrier per round, BF16 store. Claimed by finished CTAs
only (dynamic), so "every finished CTA" already participates.
Codegen (wpcgs_search, both ITK kernels): MOE_E2_ORD2, ORD2+FUN/LND, TOKF, TOKF+FORD, TG, FUN, LND: 0/16 stage loops
exact (every inline combine edit re-schedules) -- EXCEPT MOE_NO_COMBINE (16/16 stage + 4/4 producer loops = v4111:
the NO_COMBINE bound is codegen-clean) and EN_INA (below).

## M2.1 Job 3 (t3/EN/en_run3.sh, box 0, log en_run3.log, json bench/en_run3.json; t3/EN/mk_en3.py -> moe_en_3.py)
EN_TOK = WP-E2's __noinline__ e2_tok frame for the ITK kernels only (other kernels byte-identical); inside it
EN_TOK2 = two independent 4-warp combiners per CTA (named barriers 12 / 13, 32 columns per thread). Frame variants
have stage loops identical to each other (TOK2: one 24-UBLKCP producer pseudo-loop register-renamed).
hang1 7 sizes ok, check_mt (M 64 65 66 100 127 128 129 160 200 255 256 257, normal + skewed, 3 seeds, 2 calls)
0 failures for TOK, TOK2, TOK2 CG1. bench2 jm 60 reps, % vs 3 v4111 controls (+ = slower):

| variant | bs128 copies | bs256 copies |
|---|---|---|
| controls | +0.02 / +0.04 / -0.07 | -0.00 / +0.11 / -0.11 |
| NO_COMBINE (bound; loops = v4111) | -1.08 / -1.20 (-1.83 us) | -1.49 / -1.50 (-2.92 us) |
| EN_TOK frame (draw) | +0.29 / +0.27 | -0.18 / -0.26 (all copies below all controls) |
| EN_TOK2 (2 combiners) | +1.69 / +1.50 (+2.0 us vs frame) | +0.83 / +0.90 (+2.0 us vs frame) |
| EN_TOK2 CG 1 | +0.77 / +0.64 | +0.04 / +0.16 |
=> the end is NOT combine-throughput-bound: halving the threads per token (2 combiners) makes every token slower and
the end 2 us later at both shapes. The binding quantity is the latency of the critical (last) tokens' chain:
release -> poll seen -> (closing acquire) -> barrier -> row loads -> barrier -> store. Direction (a) as "more
combiners / spread wider" is refuted; what can pay is fewer round trips per token.

## M2.2 Job 4 (t3/EN/en_run4.sh, box 1, log en_run4.log, json bench/en_run4.json; t3/EN/moe_en_5.py)
New loop-exact form: EN_INA = v4111's INLINE combine with the first poll as an acquire spin (no relaxed spin + closing
acquire): all 16 stage + 4 producer loops = v4111's (the only non-trivial inline combine edit found exact). Frame
variants with EN_TOKM=2 (frame on the bs256 kernel only; the bs128 kernel = v4111 byte-for-byte): t = frame,
q = + EN_TOKA (acquire polls), w = + EN_TOKW (every warp polls, no barrier before the loads), o = + EN_TOKA +
MOE_E2_FORD (claims in last-pair order); all have the frame's stage / producer loops. hang1 7 sizes, check_mt 12 sizes
(normal + skewed, 3 seeds, 2 calls): 0 failures for i, q, w, o. % vs 3 v4111 controls (box 1, bs256 ran at 200.8 us):

| variant | bs128 copies | bs256 copies | bs256 vs frame |
|---|---|---|---|
| controls | +0.14 / +0.24 / -0.31 | +0.05 / -0.01 / -0.10 | |
| i EN_INA (inline, loop-exact) | -0.12 / +0.10 | -0.07 / +0.03 | |
| t frame (bs256 only) | +0.06 / +0.17 (placebo) | -0.33 / -0.03 | |
| q frame + acquire polls | placebo | -0.42 / -0.04 | +0.06 |
| w frame + per-warp polls | placebo | +0.53 / +0.52 | +0.59 |
| o frame + acquire + ready order | placebo | +0.60 / +0.66 | +0.67 |
Frame at bs256 over jobs 3-4: -0.18 / -0.26 / -0.33 / -0.03% (4 copies, 2 boxes; a codegen draw of the bs256 kernel's
16 loops + e2_tok); bs128 frame +0.27 / +0.29% (job 3) -> frame restricted to MW 8 (EN_TOKM=2).
Readings: one fewer round trip on every token's poll (INA, TOKA) is worth nothing measurable; polling from every
warp (8x acquire spins: L1 invalidations, more L2 polling) and ready-order claims cost 0.6%. With TOK2 (+1%): the
per-token cost is not the poll round trip but the token's load / reduce work per SM, and the late-finishing CTAs
still claim tokens at bs256 (256 tokens > 132 CTAs): anything that lengthens one token's time lengthens the end.
EN_ESIG (down-item counter released by the epilogue warps, no signaler hop; bound ~0.2 us x s-slope 0.63 at bs128
= ~0.08%): every form (5) changes the main producer loop (sim 0.77-0.83): not benched.

## M2.3 Job 5 (t3/EN/en_run5.sh, box 1, log en_run5.log, json bench/en_run5.json): validation, all shapes
Candidate EN_TOK,EN_TOKM=2 on t3/EN/moe_en_5.py: only moe_8_8_1_3_8 changes (571003e3fd87 -> 8c74cc1ec539; its 8 stage
loops re-scheduled by the call boundary, worst sim 0.37; producer loops sim 0.965; spills 16/44 -> 44/64 B, stack 40);
every other kernel = v4111's (wpcgs --full). hang1 16 sizes (1 .. 512) ok; check_mt M 1 2 4 8 9 16 17 32 33 64 65 128
129 200 256 257 512 (normal + skewed, 3 seeds, 2 calls): 0 failures. bench2 jm 60 reps, 4 + 4 copies interleaved,
fillers both ends, % vs the 4 v4111 copies:

| shape | v4111 copies | frame copies | mean |
|---|---|---|---|
| bs1 (placebo) | -0.45 / -0.92 / -0.47 / -0.42 | -0.09 / -0.90 / -0.52 / -1.37 | noise |
| bs8 (placebo) | +0.34 / +0.14 / +0.12 / -0.27 | +0.06 / -0.33 / -0.24 / -0.75 | -0.31 |
| bs16 (placebo) | +0.13 / +0.03 / -0.06 / +0.12 | +0.27 / +0.21 / +0.32 / +0.05 | +0.21 |
| bs32 (placebo) | +0.10 / +0.04 / -0.03 / -0.17 | -0.04 / -0.14 / +0.03 / -0.09 | -0.06 |
| bs128 (placebo) | -0.13 / +0.02 / +0.06 / +0.04 | -0.04 / +0.04 / +0.20 / -0.24 | -0.01 |
| bs256 (target) | -0.06 / +0.03 / -0.03 / +0.07 | -0.26 / -0.28 / -0.04 / -0.48 | -0.27 |
bs256 over jobs 3-5: -0.18 / -0.26 / -0.33 / -0.03 / -0.26 / -0.28 / -0.04 / -0.48% (8 copies, 3 jobs, both boxes; mean
-0.23%, i.e. ~+0.23% faster); every copy beats every control only in job 3 (jobs 4 / 5: one copy each inside the
control band). Placebo shapes with identical kernels read -0.31..+0.21 (module-level placement noise). => borderline
+0.2% at bs256 (~+0.04% geomean), a schedule draw rather than a combine mechanism: offered for stacking only with
a replication on the coordinator's stack (its bs256 loops depend on everything else in that kernel).

## M2.4 Conclusion (mission 2)
- Ceiling: NO_COMBINE -1.14% (bs128) / -1.49% (bs256) with all stage + producer loops = v4111's (codegen-clean).
- The end is NOT combine-throughput-bound: 2 combiners per CTA (TOK2) +1.0..+1.6%; per-warp polls +0.6%; ready-order
  claims (FORD) +0.6%. One fewer round trip per token (INA loop-exact, TOKA in the frame): 0.
- (a) more throughput / wider spread: refuted (TOK2). (b) bs128 last tile's signal chain: the one hop that can go
  (signaler wake, ~0.2 us x 0.63) is ~0.08% and every form re-schedules the main producer loop (sim 0.77-0.83): not
  run. (c) second kernel with PDL: not built. Bound: a separate combine kernel can only use SMs that a finished main
  CTA left (one 384-thread CTA holds the whole register file), so it adds no combine capacity during the tail; with
  griddepcontrol.wait it starts after the main grid completes (end = main_end_nocombine + wait + one token ~1.5 us +
  second-kernel teardown, vs today's +1.83 / +2.92 us: <= ~+0.3% at bs256, <= 0 at bs128); without the wait it needs
  a formally-ordered hand-off of the call's counter-set parity (MOE_PAR flips it at the main kernel's start) and of the
  routing tables, for the same critical chain as today. Not worth its risk at these bounds.
- Banked: nothing >= the 0.8% bar. Candidate (borderline): EN_TOK,EN_TOKM=2 bs256 ~+0.23%.
