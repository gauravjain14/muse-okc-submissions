# SU: compact cold startup path for the M <= 32 kernels (bs8 moe_4_1_1_4_1, bs16 moe_4_2_1_4_1, bs32 moe_8_4_1_4_1)

## MISSION 2 status (bs1, moe1t_4, base t3/INT/moe_i2.py) -- CLOSED, nothing converts (details at the end)
- Every producer-side start change loses or is neutral at bs1. Earlier first copies: row + stage 0 -0.42% mean
  (4 copies, 2 jobs); row + 4 stages -2.2 / -0.6%; before the init barrier -2.0..-2.3% (1 stage), -11% (4 stages);
  row only early -0.07% (3 copies). Later first copies: 200 / 400 / 800 ns cost -0.5 / -2.3..-2.8 / -4.5..-5.0%
  (b1inj, negative = slower).
- Stamps: the consumers release slot 0 (stage-4 refill) at 4.8-4.9 us whenever the producer starts in its base window or
  later. Earlier bursts push the release later (5.2-5.4 us): the consumers' warm-up path (MOE1T_WARM dry pass, row, cold
  first stage) runs under the burst. Later bursts shift the stream behind the release. The bs1 start is consumer-bound,
  and the producer's start (stage 0 issued ~1.0 us) is at the optimum.

## Status (top) -- 2026-11-03 ~11:15 PT: DONE. Candidate t3/SU/moe_su_8.py (O)
- O = v4111 + SU_EP (1 prologue stage) + SU_EPP at bs16 + WP-FAST MOE_FT_PAR at bs8 (baked; switches source moe_su_5.py).
  bs8 -1.11/-1.57 (run 6), -1.53/-1.10 (run 7, all-shape): mean -1.33% time; bs16 -1.30/-0.98, -1.34/-1.18: mean -1.20%;
  every copy below every control in both jobs. Other shapes: identical kernels (placebo rows within +-0.3, bs1 +-0.6).
  Projection (run 7, PJ_ANCHOR v1504): O 5086.8 / 5084.1 vs v4111 5106.0 / 5107.5 us (~-21 us, ~+0.42% geomean).
  hang1 16 sizes ok, check_mt full list (1..512, 17 sizes) 0 failures. Only moe_4_1_1_4_1 / moe_4_2_1_4_1 change; every
  stage loop of the module byte-identical.
- Fallback D = t3/SU/moe_su_7.py (2 stages): bs8 kernel = the 14-copy one (mean -1.03%), bs16 -1.02% (10 copies).
- bs32: no form converts (SU_F32 +0.6..0.9%; the earlier FST burst delays the routing's parity atomic and the gather).

## Budget (measured, from WP-FAST run 1 on v1514 = v4111's bs8/bs16 kernels; bs32 v4111 = + GSD; cold, us after the first
## CTA's entry, per-CTA medians; stamp builds)
| step | bs8 | bs16 | bs32 |
|---|---|---|---|
| init barrier | 0.91 | 0.97 | 0.33 (GSD) |
| producer first copy (stage 0 issued) | 2.49 | 2.86 | 2.82 (FST) |
| stage 0 landed | 4.30 | 4.61 | 4.52 |
| consumers at the first wait | 4.99 (dead NW1S copy) | 3.89 | 4.41 |
| first stage passed | 5.09 | 4.56 | 4.62 (FST CTAs) |
| stage 2 issued (first steady-loop pass) | 5.95 | 6.40 | (FST issues 4) |
| first refill (stage NS) issued / landed | 8.20 / 10.14 | 8.08 / 10.44 | 8.00 / 9.72 |
Floor for the first copy: ids from DRAM ~0.6-0.75 us after entry + a few hundred instructions -> ~0.9-1.0 us.
Delay slopes (ST run 5): producer first copies 0.84 / 0.70 / 0.57; ring fill 0.38 / 0.18 / 0.32; consumers' first wait
0.51 / 0.13 / ~0 (upper bounds on what an advance converts).

## SASS map of the startup path (local NVRTC 13, MOE_S5R_FIT; dev/moe_v4111.py)
moe_4_1_1_4_1 (6112 instr, 98 KB): entry+init 0x0-0xb30 (2.8 KB, init barrier BAR.SYNC 0 at 0xb30), routing 0xb40-0x3b00
(12 KB, warps 0-7), consumers 0x3b30-0xd040 (setmaxnreg.inc at 0x3b30), producer warpgroup 0xd060 (setmaxnreg.dec), warp 8
role 0xd5e0, early path 0xd5e0-0xf2a0 (ids LDG 0xd700, 32 unrolled VOTE ballots + scan, first UBLKCP at 0xed00 = 5.9 KB
after the role dispatch), steady producer loop 0xf510 / 0xfa20 (340 / 252 instr). Warp 8's path from the init barrier to
its first copy: BRA 0x3b00 (12 KB far), BRA 0xd050 (38 KB far), setmaxnreg.dec (warpgroup-collective), LATE9 test, BRA
0xd5e0, 5.9 KB early path: 3 far jumps + ~350 instructions, while warps 0-7 fetch the 12 KB routing (one per-SM miss path).

## Files
- su_mk1.py BASE OUT -> moe_su_1.py: SU_EP (+ SU_EPN, SU_EPU rolled, SU_EPF noinline form), SU_EPM mask.
- su_mk2.py moe_su_1.py -> moe_su_2.py: + SU_NW0 (dead NW1S copy skipped, NW0K=0 form, SU_NW0M mask).
- su_mk3.py moe_su_2.py -> moe_su_3.py: SU_E32 bs32 separate-function form (re-schedules all 4 bs32 stage loops: dropped).
- su_mk4.py moe_su_2.py -> moe_su_4.py: + SU_F32 (bs32 FST stages through wp32_fst's FDRY call form, ids mode).
- su_run1..3.sh jobs, su_run1..3.log outputs, su_tab.py (copy of ST's table script).
Every default build = v4111 SASS in all 16 kernels (wpcgs --full EXACT).

## SU_EP mechanism (su_mk1.py)
bs8/16 (EARLY kernels): warp 8's ids hoisted at entry (lane t < M: token t), mbarrier inits on tid 352 (no loads in
flight), and right after the init barrier warp 8 works out its first item with a closed form (T >= M in both configs, so
every expert has <= 1 tile: active-expert mask by one REDUX.OR, the tile-th set bit by one ballot, its token bitmap by one
ballot, the pairs before it by one REDUX.SUM: ~50 instructions instead of the old path's 32 unrolled ballots + scan) and
issues the first SU_EPN stages (records, expected bytes, WBLK block, x rows), arrives eis, then takes the usual path; the
old early path is compiled out, the producer loop starts at `early` = SU_EPN when blockIdx.x < n1 (same rule).
SASS: only moe_4_1_1_4_1 / moe_4_2_1_4_1 change: 6112 -> 5280 / 6104 -> 5264 instructions (E=2, the old early path's
BRA.DIV fallbacks gone too); every stage loop byte-identical (ctl bits incl.); the producer steady loop 252 -> 250 /
340 -> 334 instructions (register renames, one BSSY level less); no spills, 168 regs. EP block 0xb70-0x19e0 (E=2 unrolled,
2.1 KB to the first UBLKCP), the routing warps branch over it (3.6 KB; rolled SU_EPU=1: 3 KB).
SU_EPF=1 (noinline form) renames the bs8 stage loop: not used.

## Run 1 (box 0): stamps (wpfast_probe, cold, 4 jm seeds x 2, us after the first CTA's entry, medians) + bench2 bs8/16
| bs8 | base | E=2 | E=4 |
|---|---|---|---|
| producer first copy (iss0) | 2.54 | 1.41 | 1.41 |
| stage 0 landed | 4.35 | 3.52 | 4.29 |
| parity atomic back (tid 32) | 2.27 | 3.38 | 4.69 |
| tables / consumers' inc | 2.96 / 3.26 | 3.84 / 4.14 | 5.26 / 5.55 |
| consumers' first stage passed / released | 5.22 / 6.91 | 4.83 / 6.67 | 6.26 / 7.96 |
| first loop stage issued (stage E) / rtd -> it | 6.11 / 2.42 | 5.02 / 0.90 | 8.33 / 2.64 |
| first refill (stage 4) issued / landed | 8.44 / 10.59 | 7.45 / 9.36 | 8.36 / 10.11 |
| second item seen | 15.14 | 13.98 | 15.13 |
bs16: first copy 3.08 -> 1.45 / 1.40; landed 4.79 -> 3.32 / 3.39; tables 2.75 -> 3.85 / 5.49; first stage passed 4.71 ->
4.43 / 6.07; refill landed 10.62 -> 9.40 / 9.57; second item 14.98 -> 13.80 / 14.69.
Reading: the first copy moves 1.1-1.6 us earlier (floor-like: ids at ~0.85 us, ~1.0 item, 1.4 issued), but any global round
trip or cold code fetch issued while the burst is in flight waits for it: the routing's parity atomic (the tables barrier
waits for it) returns 1.1 / 2.4 us later (E=2 / E=4), so the consumers start later; with E=2 the producer's first loop pass
after rtd takes 0.9 us instead of 2.4 (the burst has drained by then).
bench2 (jm 60 reps, % time vs mean of a1/a2, negative = faster; each-vs-each controls in brackets):
| | bs8 | bs16 |
|---|---|---|
| controls a1 / a2 | +0.33 / -0.33 | -0.04 / +0.04 |
| E=2 e2 / e2b | +0.13 (-0.51/+0.33) / -0.27 (-0.68/-0.38) | -0.86 (-0.99/-0.88) / -0.89 (-0.84/-0.86) |
| E=4 e4 / e4b | +1.91 / +1.69 | +0.05 / +0.03 |
| E=4 rolled r4 / r4b | +1.52 / +1.73 | -0.15 / +0.04 |

## Run 2 (box 1): + MOE_FT_PAR (PARM=3: the parity atomics by the signaler, off the routing) and SU_NW0
Stamps (q = SU_EP + FT_PAR + NW0, cold): bs8 tables 2.38, consumers' inc 2.67, first wait 2.89, stage 0 landed 3.63, first
stage passed 3.20 (base 5.22), released 5.80; producer rtd seen 4.05 (its post-prologue path: dealloc, role code, rtd wait,
cold under the burst), first loop stage 5.42, refill issued 7.41 / landed 9.48, second item 14.00. bs16: first stage
passed 3.37 (base 4.71), loop stage 5.43, refill landed 9.63, second item 13.96.
bench2 (controls a1/a2/a3):
| | bs8 | bs16 |
|---|---|---|
| controls | +0.17 / -0.11 / -0.16 | -0.04 / +0.22 / -0.00 |
| p = SU_EP + FT_PAR, p1 / p2 | -1.12 (-1.19/-0.83/-1.06) / -1.51 (-1.53/-1.38/-1.37) | -0.42 / -0.51 (all < every control) |
| q = p + SU_NW0, q1 / q2 | -0.56 / -0.71 | -0.53 / -0.56 (bs16 kernel = p's) |
| e = SU_EP alone | -0.81 (-1.02/-0.60/-0.58) | -0.65 (-0.72/-0.84/-0.57) |
NW0 (consumers start earlier still) costs ~0.7% at bs8 (as WP-FAST's v4110 set); FT_PAR helps bs8 (~-0.5% on top of EP),
not bs16 (consumers then start their cold first stage under the burst; the stream is gated by the producer's path).

## Run 3 (box 0): E=2 vs E=4 with FT_PAR; bs32 SU_F32 (su_mk4: FST stages from warp 8's own ids at the init barrier)
Stamps p4 = SU_EP + SU_EPN=4 + FT_PAR (cold): bs8 stages 0-3 land 4.30 / 5.31 / 5.61 / 5.96 (DRAM-bound burst), but the
consumers' first wait moves to 5.70 (inc 2.71: their first-stage code / item top fetched under the 4-stage burst) and the
producer sees rtd only at 5.64 (its post-prologue path cold under the burst) -> refill issued 7.98, second item 14.63
(E=2: 14.00). bs16: first stage passed 3.18 but released 7.36 (cold first stage under the burst 4.2 us), second item 14.36.
bs32 SU_F32 (N=2) vs base (same job): FST stage 0 issued 3.03 vs 2.81 (the call + its ids load + cold function code),
bitmaps barrier 1.25 vs 1.12 and tables 4.66 vs 1.92 (warp 9's parity atomic under the burst; FT_PAR at bs32 re-schedules
all 4 bs32 stage loops, so not usable), first data later everywhere.
bench2 (controls a1/a2/a3; c1 = SU_EP+FT_PAR(3)+SU_F32 N=2, c4 = SU_EPN=4+FT_PAR(3)+SU_F32 N=1, e1 = SU_EP alone):
| | bs8 | bs16 | bs32 |
|---|---|---|---|
| controls | -0.08 / +0.19 / -0.16 | -0.02 / +0.06 / -0.15 | -0.04 / -0.11 / -0.03 |
| c1 / c1b | -0.96 (-1.26/-1.03/-0.72) / -0.67 (-0.60/-0.78/-0.58) | -0.99 / -0.95 (all < every control) | +0.55 / +0.65 |
| c4 / c4b | +0.89 / +0.94 | -0.44 / -0.13 | +0.86 / +0.86 |
| e1 | -1.23 (-1.13/-1.33/-0.72) | -0.99 (-1.22/-1.06/-0.91) | -0.01 (bs32 kernel = base) |
=> E=4 loses at bs8 in every form; SU_F32 loses at bs32 (+0.6..0.9%): dropped. E=2 holds.

## Tally for E=2 over runs 1-3 (% time vs that job's control mean; every copy vs every control in its job)
- bs8 SU_EP + FT_PAR: -1.12, -1.51 (run 2), -0.96, -0.67 (run 3): mean -1.07%, 4/4 copies below every control.
- bs8 SU_EP alone: +0.13, -0.27 (run 1), -0.81 (run 2), -1.23 (run 3): mean -0.55%, not robust.
- bs16 SU_EP alone: -0.86, -0.89 (run 1), -0.65 (run 2), -0.99 (run 3): mean -0.85%, 4/4 below every control.
- bs16 SU_EP + FT_PAR: -0.42, -0.51 (run 2), -0.99, -0.95 (run 3): mean -0.72%.
Candidate C = SU_EP + MOE_FT_PAR + MOE_FT_PARM=1 (PAR at bs8 only): its bs8 kernel = the measured SU_EP+FT_PAR kernel
(ef712d6e2456), its bs16 kernel = the measured SU_EP kernel (14e2e190ce43); every other kernel = v4111's.

## SASS notes (local NVRTC 13, MOE_S5R_FIT, wpcgs --full vs dev/moe_v4111.py)
- Candidate t3/SU/moe_su_6.py (= moe_su_5.py baked with SU_EP,MOE_FT_PAR,MOE_FT_PARM=1): only moe_4_1_1_4_1 (bf4da21e9674 ->
  ef712d6e2456, 6112 -> 5296 instr) and moe_4_2_1_4_1 (4a3b020976a2 -> 14e2e190ce43, 6104 -> 5264) change; all 67 stage
  loops of the module byte-identical incl. control bits; producer steady loop register-renamed (252 -> 250 / 244 -> 242
  instr, outer 340 -> 334 / 331 -> 325: one BSSY level and two moves less); FT_PAR moves the two parity atomics into the
  signaler's role start (WP-FAST's loop-exact form); 168 regs, no LDL/STL. Size 604192 B.
- bs8 layout: init barrier 0xb60; prologue 0xb70-0x19e0 (REDUX.OR 0xe60, REDUX.SUM 0x11d0, first UBLKCP 0x13b0 = 2.1 KB
  after the barrier, ~140 instructions); routing warps branch over it (0xbb0 -> 0x19e0).
- Not loop-exact (dropped): SU_EPF (noinline prologue: bs8 stage loop renamed); SU_E32 separate function (bs32: all 4
  stage loops re-scheduled, even the ids hoist alone); MOE_FT_PAR at bs32 (4 loops renamed). Loop-exact but losing:
  SU_EPN=3/4, SU_NW0, SU_F32.

## Run 4 (box 1): C = SU_EP + FT_PAR(bs8 only) vs X = C + SU_EPP (prologue at the producer section start)
Stamps X (cold): bs8 first copy 2.08 (C 1.41; two far branches first), rtd seen 2.68, first loop stage 4.53 (C ~5.0-5.4),
but consumers' first wait 4.65 (their dead NW1S copy now under the later burst); bs16 first copy 2.24, rtd 3.65, loop
stage 5.07, consumers' first wait 3.30, second item 13.46 (C-like E=2 runs: 13.8-14.0).
bench2: bs8 C -1.31 / -1.19, X +0.29 / +0.17; bs16 C -0.94 / -0.65, X -1.00 / -1.10 (all below every control) -> EPP at
bs16 only (SU_EPPM=2).

## Run 5 (box 0): C vs D = C + EPP(bs16) vs T = C with 3 stages
bs8: C -0.99 / -1.11, D (same bs8 kernel) -0.42 / -0.99, T -0.19 / -0.03; bs16: C -0.98 / -0.89, D -1.49 / -0.80,
T -0.94 / -0.77. => 3 stages lose at bs8 (E=2 -1.1, E=3 -0.1, E=4 +0.9).

## Run 6 (box 1): D vs G = EPP at bs8 too + SU_NW0 vs O = D with 1 stage
bs8: D -1.33 / -1.24, G -1.05 / -0.90, O -1.11 / -1.57; bs16: D -1.01 / -1.00, G (= D's kernel) -0.72 / -0.89,
O -1.30 / -0.98.

## Run 7 (box 0, final, all shapes, jm 60 reps; a1/a2 = v4111, z1 = v1504, fillers v4111)
| % vs v4111 | bs1 | bs8 | bs16 | bs32 | bs128 | bs256 |
|---|---|---|---|---|---|---|
| D d1 / d2 | -0.71 / -0.45 (same kernel) | -1.01 / -0.50 | -1.08 / -1.15 | +0.23 / +0.01 | -0.17 / -0.10 | +0.04 / -0.00 |
| O o1 / o2 | -0.57 / -1.15 (same kernel) | -1.53 / -1.10 | -1.34 / -1.18 | +0.18 / -0.03 | +0.02 / -0.28 | +0.07 / -0.06 |
| v1504 z1 | -0.54 | +1.67 | +0.81 | +1.09 | +1.11 | +1.15 |
| fillers fA / fB | -0.30 / -1.22 | -0.08 / +0.31 | -0.14 / -0.48 | +0.02 / -0.21 | -0.40 / -0.50 | -0.34 / -0.54 |
(bs8/16 each-vs-each: O -1.22..-1.35 / -1.13..-1.51, D -0.39..-0.87 / -1.11..-1.23.) Projected: D 5090.3 / 5092.2, O 5086.8 /
5084.1, v4111 5106.0 / 5107.5, v1504 5157.1 (anchor). check_mt (17 sizes, normal + skewed, 3 seeds, 2 calls): 0 failures
for moe_su_7.py and moe_su_8.py; hang1 1..512 (16 sizes) ok for both.
SASS of O (moe_su_8.py): moe_4_1_1_4_1 bf4da21e9674 -> ed8193934dc4 (6112 -> 5224 instr), moe_4_2_1_4_1 4a3b020976a2 ->
655dc0db46b2 (6104 -> 5200); all 67 stage loops identical, producer steady loop register-renamed as above, no spills.

## What the budget says (measured) and what is left
- First copy: 2.5 / 2.9 us -> 1.4 us at the init barrier (floor ~0.9-1.0: ids from DRAM + ~140 instructions).
- Every global round trip or cold instruction fetch issued while the first burst is in flight waits until it drains
  (DRAM queue): the routing's parity atomic (+1.1 us with 2 early stages, +2.4 with 4), the producer's post-prologue path
  (role code, rtd wait, loop top: +1.5-2.5 us), the consumers' first-stage code (cold first stage 1.8 -> 2.5-4.2 us). Hence
  the optimum is a SMALL early burst (1 stage) plus moving the dependent round trips off the routing path (FT_PAR at bs8).
- Left: the producer's first steady-loop pass still lands after the burst (stage 2 issued ~4.5-5.4 us); bs8 consumers'
  first stage cold (~2.5 us) under the burst; bs32's FST path (fragile kernel: every form but the FDRY call site
  re-schedules all 4 stage loops; FT_PAR there renames them).
- Ruled out (measured): 3-4 early stages (bs8 +0.9..1.9%), NW0 with the early prologue (bs8 ~+0.4..0.7% vs without),
  EPP at bs8 (+0.2..0.3%), SU_F32 (bs32 +0.6..0.9%), noinline prologue (bs8 stage loop renamed), separate-function bs32 form
  (all 4 bs32 stage loops re-scheduled).


# MISSION 2 (bs1, moe1t_4): the cold-startup prologue on the one-token kernel -- CLOSED, no gain
Base t3/INT/moe_i2.py (moe1t_4 1c34fcc62359, 116 regs, consumer stage loop 642 instr, sched a1d08bdfeb).
## Files
- su_mk6.py moe_i2.py -> moe_su_9.py: SU1_EP (warp 8 issues the token row + stages [0, SU1_EPN) of the first local route's
  expert right after the init barrier; closed form: the first local route in slot order = the list's q = 0; the producer
  loop skips them), SU1_EPW=1 (the same before the init barrier, warp 8 initialising the row / full barriers), SU1_PD=ns +
  SU1_PDR (diagnostic: the producer's copies held back ns, before / after the row), SU1_TS (stamps P.trc[24576 + 32 blk + k]).
  Default build = moe_i2's SASS in all 16 kernels (wpcgs --full EXACT).
- su1_probe.py: bs1 stamp probe (cold code: each timed replay follows another build's replay; cold L2).
- su_b1cfg1..3.py + su_run8..10.sh / .log (b1inj cold-ordered benches, 100 reps; su_run10.json).
## SASS (local NVRTC 13)
Every SU1 variant keeps the consumer stage loop's schedule identical (loopcmp sched a1d08bdfeb, 642 instr; the instruction
text is identical, the scoreboard numbers shift by one in the SU1_EP forms); producer loops change; 116 regs, no spills.
SU1_EP: row copy at 0x860, stage 0 at 0xbb0 (the init barrier at 0x660), vs the base's far branch 0xd10 -> 0x7180 first.
## Run 8 (box 1): stamps (cold, 6 jm seeds x 4, us after the first CTA's entry, CTA medians) + b1inj (100 reps)
| event | base | E=1 | E=4 | EPW=1, E=1 | EPW=1, E=4 |
|---|---|---|---|---|---|
| warp 8 past the init barrier | 0.11 | 0.11 | 0.11 | 0.53 | 0.73 |
| row issued | 0.74 | 0.39 | 0.40 | 0.21 | 0.22 |
| stage 0 / 1 / 3 issued | 1.02 / 1.19 / 1.67 | 0.60 / 1.21 / 1.63 | 0.67 / 0.73 / 0.86 | 0.47 / 1.19 / 1.41 | 0.50 / 0.55 / 0.65 |
| stage 4 issued (slot 0 released) | 4.82 | 5.38 | 5.37 | 5.68 | 7.79 |
| stage 11 issued | 11.95 | 12.58 | 12.26 | 12.85 | 14.98 |
b1inj (% vs the 10 interleaved moe_i2 controls; negative = slower): E=1 -1.11 / -0.51, E=2 -1.52, E=4 -2.16 / -0.59,
EPW=1 E=1 -2.34 / -2.03, EPW=1 E=4 -10.92 / -10.96; fillers +0.39 / +0.93.
## Run 9 (box 1): the producer's first copies held back (diagnostic slope)
Stamps: 400 ns before the row: stage 0 1.47 (base 0.99), stage 4 4.85 (4.83), stage 11 12.23 (11.97); 800 ns: 2.01 / 4.98 /
12.39; after the row 400 / 800 ns: stage 4 4.76 / 5.00. b1inj: 200 ns -0.54; 400 ns -2.36 / -2.32 (after the row -2.55 /
-2.80); 800 ns -4.71 / -5.04 (after the row -4.47 / -4.67) => ~1.4 us per us of delay, while the slot-0 release does not move.
## Run 10 (box 1): row only early (SU1_EPN=0) + E=1 replication
Stamps: row 0.41 (base 0.79), stage 0 0.98 (1.06), stage 4 4.88 (4.88). b1inj (6 controls, spread -0.03..+0.53): row only
-0.16 / +0.21 / -0.26, E=1 -0.30 / +0.23.
## Reading
The bs1 start is set by the consumers' own path to their first release (dry pass of the stage code on the empty ring,
row -> B fragments, cold first stage: done ~4.8 us). The producer's base timing (row 0.7, stage 0 1.0 us) sits at the
optimum: any earlier traffic lands on that path's cold fetches (released 0.3-0.9 us later; with the copies before the
init barrier all warps wait for warp 8). Any later traffic delays the stream behind it (1.4 : 1). A start gain at bs1
needs the consumer path itself (the WARM dry pass placement or a smaller first-stage code path). That is consumer-side,
it flips the 116-reg schedule per B1, and it is not built here.
