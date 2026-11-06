# IL: per-path consumer item loop for the (8,8,1) bs65-256 kernels (moe_8_8_1_3_4 / moe_8_8_1_3_8), base t3/INT/moe_i3.py

## STATUS (2026-11-03 ~13:00 PT): candidate t3/IL/moe_il_6.py (= mk_il3.py(moe_i3) baked with MOE_IL, MOE_IL_M=2,
## MOE_IL_K=53248; escape MOE_NO_IL = moe_i3 SASS in all 16 kernels). Only moe_8_8_1_3_8 (bs129-256) changes; bs128
## stays on the base kernel (IL forms there read 0..+0.3%: bs128's 1-group phase is stream-bound). bs256: every IL
## copy beat every control in 4 jobs (-0.16..-0.88%; K53248 -0.72 / -0.88 in job 3); final all-shape validation in
## section 8. Mechanism (ncu): per-warp transition 1338 -> 865 cycles (bs256), 1358 -> 822 (bs128); the gain is
## capped by the stage-loop re-draw (+-3% per mma.sync loop) that every form of the new loop nest causes.

## 0. Bound before building (offline, MEASURED inputs: EN mission 1, HV qsim routings)
- EN: per-warp item transition ~1260 (bs256) / ~1340 (bs128) cycles, delay slope 0.5-0.7; EN's per-path estimate
  1260 -> ~600 cycles = ~+1.1-1.2% IF every transition took the fast path.
- NEW: how many transitions can stay on one path. t3/IL/il_paths.py replays HV's calibrated queue simulator
  (t3/HV/qsim.py, mix5 order, dynamic claims, 60 H200 jm routings) and records each CTA's item sequence:

| | items / CTA | P(next item same ntl) | 1->1 | 2->2 | >=3 same class | top classes |
|---|---|---|---|---|---|---|
| bs256 | 11.45 | 0.417 | 0.348 of all transitions | 0.054 | 0.069 (same ntl 0.015) | ntl1 54%, ntl2 20%, ntl3 10%, ntl8 6% |
| bs128 | 10.34 | 0.654 | 0.624 | 0.027 | 0.005 | ntl1 75%, ntl2 15% |

  => a same-path-only fast path converts ~42% (bs256) / ~65% (bs128) of the transitions: bound ~0.42 x 1.2 = ~0.5%
  (bs256), ~0.65 x 1.1 = ~0.7% (bs128) before any codegen draw, unless the cross-path transitions also get cheaper
  (record-only dispatch: no modulo chains / no rebuilt indices, but the target path's entry is a cold i-cache region).
  The +1.0% goal at bs256 is out of reach for the same-path mechanism alone.

## 1. Design (t3/IL/mk_il1.py BASE OUT; switch MOE_IL, kernel mask MOE_IL_M (1 = MW 4, 2 = MW 8), knobs MOE_IL_K)
`if constexpr (ILK) { new loop } else <old loop>`: every other kernel instantiates the old loop unchanged.
New loop: slot s / parity ph / q_epi % NS in registers (incremented); record read once by whoever finished the
previous item; dispatch on ntl only; per path K an inner loop: stage loop K (same stage functions and ring protocol as
MOE_RUN_ITEM_TOP), its own item end (exactly K groups of 8 STS, no per-group branches; no barrier for K <= 2 (XS items),
bar.sync 1, 256 for K >= 3; fence.proxy.async; aq_slot / aq_full hand-off), the next item's slot wait + record; same
ntl -> straight back into stage loop K, else -> dispatch with the record already read. Warm-up stage (bs256): a
1-stage item of its path without the item end. Codegen lesson 1: the item end must zero ALL NT accumulator groups
(stage_compute_wg's pin_acc makes every group opaque); zeroing only K groups made ptxas serialize every wgmma
(WARPGROUP.DEPBAR after each HGMMA, non-in-place accumulators, loops 1081-1413 instr).

SASS (default form K=0, local NVRTC 13): the 1-group same-path transition is ~75 instructions in two contiguous
chunks (loop exit -> 4-HMMA drain -> item end: 1 S2R TID, 8 STS, MEMBAR + FENCE, aq_slot STS, ARRIVE -> TRYWAIT next ->
LDS.128 + LDS record -> 4 R2UR -> same-path test -> nst (tsplit in a uniform register) -> BRA to the path entry, 16
CS2R, loop) vs ~180 in the base (2 S2R TID, 3 modulo-3 chains, LDL tsplit, 48 CS2R, 4 far code regions).
Cross-path: exit -> end-check block (placed after all paths) -> outer head (LDL tsplit) -> dispatch -> path entry.
Stage loops: re-scheduled in every form (opcode similarity ~0.35-0.63 vs base, wgmma loops MW8 544-545 instr vs
586-588); spills 32/88 (MW4) / 40/68 (MW8) vs 16/40 / 16/44 (outside loops; LDL 0 in loops).

## 2. Codegen search (dev/wpcgs_search.py, 161 sampled forms of MOE_IL_K bits 1..1024 + targeted ones; MEASURED)
- 0 of 161 forms keep any stage loop of either kernel identical to moe_i3 (worst opcode similarity 0.21-0.44): the new
  loop nest re-schedules all 16 stage loops in every form (as EN predicted). Non-serialized forms fall into ~6 clusters
  (MW4 sim 0.39-0.44, spills 32/88 .. 48/80; MW8 sim 0.37-0.42, spills 32/92 .. 40/68; base 16/40, 16/44 -- every
  added spill is outside the stage loops: LDL 0 in every loop). Forms with bit 1024 (zero only K groups) mostly
  serialize the wgmma (12/40 spill cluster): never use 1024.
- Within the IL family some knobs keep the stage loops exact relative to the K=0 form: K=16 = K=0 byte-for-byte (copy
  control); K=8192 (pinned staging offset): all 16 stage loops exact vs K=0; K=4096: MW8 6/8 exact vs K=0.
  -> IL-vs-IL A/Bs isolate transition mechanisms without the loop lottery.
- wgmma loop codegen (MW8, base vs IL): base wgmma loops (586-588 instr) carry 12 x (UIADD3 + UIADD3.X) + ~24 UMOV for
  the 64-bit descriptor adds and one S2R TID per stage; IL K=0 / 4096 / 8192 / 12288 wgmma loops are 544-545 instr
  (12 x UIADD3.64, no S2R). The twin K=2048 and K=110 keep base-like wgmma loops (585-594, no UIADD3.64; K110 without
  the S2R). MW4: K0 585-586 (no UIADD3.64), K4096 / K12288 / K20480 547-548 (UIADD3.64).

## 3. Job 1 (t3/IL/il_run1.sh, box 0, log t3/IL/il_run1.log; moe_il_2.py; MEASURED)
hang1 16 sizes (K0) / 8 sizes (others): all finished + ok. check_mt K0 (M 64 65 128 129 200 256 257): 0 failures.
bench2 jm 60 reps, fillers both ends, 3 moe_i3 controls (a1-a3), 2 copies per form; % time vs control mean (- = faster):

| form | bs128 copies | bs128 each-vs-each ctl | bs256 copies | bs256 each-vs-each ctl |
|---|---|---|---|---|
| controls a1/a2/a3 | +0.09 / +0.14 / -0.19 | -0.42..+0.42 | -0.05 / +0.06 / +0.01 | -0.12..+0.12 |
| x K0 (per-path loops, lean MW8 wgmma loops) | +0.00 / +0.05 | -0.18..+0.17 | **-0.61 / -0.64** | -0.71..-0.56 |
| t K2048 twin (no path loops; base-like wgmma loops) | +0.01 / +0.02 | -0.13..+0.25 | -0.43 / -0.69 | -0.77..-0.25 |
| y K110 (wgmma paths not looping; base-like wgmma loops) | -0.21 / -0.16 | -0.35..-0.03 | -0.44 / -0.44 | -0.53..-0.35 |
| z K539 | +0.06 / -0.25 | -0.29..+0.31 | -0.51 / -0.44 | -0.53..-0.41 |
| fillers fA / fB (base, end positions) | -0.21 / -0.30 | | -0.32 / -0.59 | |
Readings: bs256 every IL form beats every control (-0.44..-0.64% means); the twin (no same-path loop, base-like wgmma
loops) already gets ~-0.56% -> most of the bs256 gain is the per-path item end + record-only top/dispatch, not the
same-path loop and not the wgmma-loop codegen (<= ~0.1-0.2%). bs128: nothing (0.00..-0.18%) although 65% of its
transitions are same-path: bs128's 1-group phase is stream-bound (1-group stage 1460 cyc ~ 5.5 TB/s demand > ~4.5
TB/s DRAM), the ring keeps streaming during a transition, so a shorter consumer transition turns into a longer
first-stage data wait (EN: bs128 item-top data wait 1132 cyc per warp-item full vs 535 at bs256).

## 4. Tools
- t3/IL/il_paths.py: per-CTA item sequences from HV's qsim (P(same path) table in section 0).
- t3/IL/il_loops.py dump.sass kernel [ref.sass]: stage loops + best match vs a reference dump.
- t3/IL/il_feat.py WPCGS_DIR TAG REF.sass: per-candidate loop features (UIADD3.64, S2R, exact vs ref, LDL/STL).
- t3/IL/il_cc.py: compile one version's target kernels with NVRTC 13 (ptxas log + SASS).
- t3/IL/il_ncu.py src.csv det.csv: item-region decomposition of an ncu SourceCounters CSV that works for both loop
  structures (item region = smallest backward-branch body containing all stage loops; out-of-line TRYWAIT retry
  blocks attributed by the address they return to; warp-items = FENCE.VIEW.ASYNC executions). On ST's zero-
  perturbation v4111 CSVs (t3/ST/st_ncu3, same (8,8,1) kernels as moe_i3): bs256 transition 1385 cyc / warp-item
  (221 instr), item-top wait 611, stage loops 23288, in-loop waits 1147; bs128 transition 1361 (183 instr), item-top
  wait 1328, stage 22401, in-loop waits 1525 (EN: 1263-1340 / 535 / 1132: same picture).
- t3/IL/il_ncu.sh "label:file:defs ...": the ncu job for any builds (box /tmp/il_ncu.tgz).

## 5. Job 2 (t3/IL/il_run2.sh, box 0, log t3/IL/il_run2.log; moe_il_3.py = mk_il2.py; MEASURED)
hang1 (10 sizes) ok for K8192 / K4096 / K12288 / K20480. check_mt (M 65 128 129 256): K20480 0 failures; **K12288
printed nothing = the 600 s timeout hit: a HANG** -> the blocking early record (bit 4096 without 16384) can deadlock:
the consumers wait for the next item's first stage BEFORE handing the held slot to the epilogue warps, while the
producer may be waiting for that item's tile readiness, which needs this very item's epilogue (a down item right after
the gate/up row block it depends on). K4096 / K12288 are UNSAFE (never use); 16384 (test_wait: early only if already
complete, else after the hand-off) is deadlock-free by construction (no wait before the hand-off).
bench2 jm 60 reps, fillers both ends, 3 controls, 2 copies each; % time vs control mean:

| form | bs128 copies | bs256 copies | bs256 each-vs-each ctl |
|---|---|---|---|
| controls a1/a2/a3 | -0.06 / +0.19 / -0.07 | +0.04 / +0.08 / -0.14 | -0.24..+0.24 |
| x K0 | -0.11 / -0.21 | -0.48 / -0.35 | -0.64..-0.23 |
| p K8192 (pinned staging offset; ALL 16 stage loops = K0's) | -0.04 / -0.08 | -0.41 / -0.40 | -0.54..-0.22 |
| e K4096 (blocking early record; UNSAFE) | +0.02 / +0.14 | -0.81 / -0.71 | -0.86..-0.59 |
| q K12288 (4096 + 8192; UNSAFE) | +0.20 / +0.25 | -0.23 / -0.48 | -0.62..-0.08 |
| **w K20480 (non-blocking early record)** | +0.32 / +0.13 | **-0.77 / -0.73** | -0.90..-0.57 |
| fillers fA / fB | -0.25 / -0.19 | -0.25 / -0.34 | |
Readings: (1) identical stage loops -> identical time (p = x at bs256: -0.41 / -0.41): the S2R removal at the item end
is worth 0. (2) K4096 vs K12288 differ only in the 2-group stage loop's schedule (7/8 MW8 loops exact) and read -0.76
vs -0.36: one loop draw = ~0.4% at bs256. (3) bs128 IL forms 0..+0.3% (no gain): bs128 stays on the base kernel
(MOE_IL_M=2).

## 6. ncu decomposition (t3/IL/il_ncu.sh, box 1, log t3/IL/il_ncu1.log, CSVs t3/IL/ncu1/; one jm routing per
shape, zero perturbation; t3/IL/il_ncu.py, t3/IL/il_ncu_loops.py; MEASURED)
Cycles per consumer warp-item (bs256 seed 73011: 9.09 items per CTA; bs128 seed 71637: 7.64):

| | base trans | IL K0 trans | twin K2048 trans | transition instr executed (base / K0 / twin) |
|---|---|---|---|---|
| bs256 | 1338 | **865 (-35%)** | 1089 | 221 / 138 / 140 |
| bs128 | 1358 | **822 (-39%)** | 875 | 183 / 99 / 113 |
(item-top waits: the base's shared top wait site also holds the CTA's first (startup) wait, IL's does not, so the
wait_top column is not comparable; excluding the startup site the base's per-item top wait is ~300 cyc at bs128,
IL's 135.)
Stage loops (compute cycles per warp-stage, waits excluded):

| loop | bs256 base / K0 / twin | bs128 base / K0 / twin |
|---|---|---|
| 2-group mma.sync (64 HMMA) | 1767 / 1822 (+3.1%) / 1767 | 1834 / 1832 / 1823 |
| 1-group mma.sync (32 HMMA) | 1492 / 1535 (+2.9%) / 1473 (-1.3%) | 1515 / 1524 / 1539 |
| wgmma n24 / n32 / n40 / n48 / n56 | 2034/2110/2412/2618/2764 -> K0 1996/2088/2365/2655/2790 | n56 / n64: 2682/3088 -> 2745/3099 |
=> the mechanism works: the per-path loop cuts the per-warp transition by ~470-540 cycles (35-39%; EN's design
estimate was -660). It converts at bs256 only, and there the re-scheduled stage loops move by +-3% per loop (K0's
mma.sync loops +3% = +~460 cyc per warp-item, eating the transition gain on this routing: kernel -0.33%; twin -0.98%).
The loop draw is as large as the mechanism; only bench A/Bs of complete builds decide.
Repeatability of the per-loop ncu metric: ST's independent run of the same base kernel (t3/ST/st_ncu3, bs256 73011):
2-group 1772 / 1-group 1491 / n32 2096 cyc per warp-stage vs this job's 1767 / 1492 / 2110 (<= 0.7%): the +3% of the
K0 mma.sync loops is outside the run-to-run spread.

## 7. Job 3 (t3/IL/il_run3.sh, box 0, log t3/IL/il_run3.log; moe_il_4.py = mk_il3.py, MOE_IL_M=2: bs256 only; MEASURED)
hang1 (1 129 160 200 256 257) ok for all; check_mt K86016 (M 129 200 256): 0 failures.
bench2 jm 60 reps, bs256, fillers both ends, 3 controls, 2 copies each; % time vs control mean:

| form (MW8 kernel; spills; MW8 loops vs K20480) | copies | each-vs-each ctl |
|---|---|---|
| controls a1/a2/a3 | -0.02 / +0.10 / -0.03 | -0.17..+0.17 |
| w K20480 (job 2: -0.77 / -0.73) | -0.64 / -0.54 | -0.76..-0.46 |
| t K22528 = twin + early record (36/76) | -0.42 / -0.49 | -0.62..-0.35 |
| e K20488 = wgmma paths not looping (16/52, producer loops sim 0.998) | -0.63 / -0.71 | -0.76..-0.56 |
| **c K53248 = K20480 + carried stage count (32/92; all loops re-drawn)** | **-0.72 / -0.88** | -0.94..-0.76 |
| g K86016 = K20480 + goto end (40/112) | -0.16 / -0.27 | -0.32..+0.03 |
| fillers fA / fB | -0.11 / -0.39 | |
K20480 over jobs 2-3: -0.77 / -0.73 / -0.64 / -0.54 (4 copies, all beat all controls; mean -0.67%).
Every IL form so far beats every control at bs256 (19 of 19 IL copies over 3 jobs, -0.16..-0.88%); the spread
between forms (-0.2..-0.8%) is the stage-loop draw (section 6).

## 8. Final validation (t3/IL/il_val.sh, box 0, log t3/IL/il_val.log; MEASURED)
Candidates: c = t3/IL/moe_il_6.py (K53248), w = t3/IL/moe_il_5.py (K20480); both MOE_IL_M=2 (only moe_8_8_1_3_8).
hang1 16 sizes (1 .. 512): c 16/16, w 16/16 ok. check_mt c full list (M 1 2 4 8 9 16 17 32 33 64 65 128 129 200 256 257
512, normal + skewed, 3 seeds, 2 calls): 0 failures; w (M 129 200 256): 0 failures.
bench2 jm 60 reps, ALL shapes, fillers both ends, 3 moe_i3 controls, c x3, w x2, one v1504 copy; % time vs controls:

| shape | controls a1/a2/a3 | c K53248 copies | c each-vs-each | w K20480 copies |
|---|---|---|---|---|
| bs1 (placebo) | +0.26 / -0.21 / -0.44 | -0.54 / -0.05 / -0.82 | -0.67..+0.62 | -0.16 / -0.44 |
| bs8 (placebo) | -0.17 / -0.22 / +0.24 | +0.15 / -0.34 / +0.20 | -0.68..+0.29 | +0.05 / -0.04 |
| bs16 (placebo) | +0.04 / +0.06 / -0.03 | +0.21 / +0.06 / -0.05 | -0.18..+0.48 | -0.12 / +0.30 |
| bs32 (placebo) | -0.13 / +0.00 / +0.03 | -0.20 / +0.12 / +0.02 | -0.30..+0.28 | +0.06 / -0.01 |
| bs128 (placebo: base kernel) | -0.13 / -0.03 / +0.11 | -0.04 / +0.05 / -0.15 | -0.25..+0.14 | +0.00 / +0.03 |
| **bs256 (target)** | +0.06 / +0.06 / -0.10 | **-1.00 / -0.92 / -1.09** | **-1.23..-0.81** | -0.76 / -0.72 |
Projection (PJ_ANCHOR=v1504, p1): controls 5068.4 / 5068.1 / 5068.3; c 5063.3 / 5067.1 / 5060.4; w 5069.4 / 5065.8.
c over 2 jobs (job 3 + this): bs256 -0.72 / -0.88 / -1.00 / -0.92 / -1.09% (5 copies, mean -0.92% time = +0.93%
speed; every copy beats every control in both jobs). w over 3 jobs: -0.77 / -0.73 / -0.64 / -0.54 / -0.76 / -0.72
(6 copies, mean -0.69%). Geomean: ~+0.15% (~-5..-8 us projected).

## 9. Conclusion
- BANKED candidate: t3/IL/moe_il_6.py = python3 t3/IL/mk_il3.py t3/INT/moe_i3.py X && python3 t3/IL/bake_il.py X OUT
  "MOE_IL_M=2,MOE_IL_K=53248". Only moe_8_8_1_3_8 changes (b2f5fb05aecf, 10896 instr vs 10560); MOE_NO_IL = moe_i3
  in all 16 kernels (wpcgs --full EXACT 67/67 stage, 108/108 producer loops).
- SASS of moe_8_8_1_3_8: all 8 stage loops re-scheduled (opcode similarity vs base 0.35-0.69; 1-group 511 -> 509,
  2-group 551 -> 550, wgmma 586-588 -> 544-545 instr: 12 UIADD3.64 instead of UIADD3 + UIADD3.X + UMOV pairs and no
  S2R per stage); producer loops sim 0.998; registers 168 (launch bound) unchanged, consumer setmaxnreg 216; spills
  32/92 B (base 16/44), LDL / STL 21 / 5 (base 11 / 3), stack 32 (16), none inside a stage loop.
- No loop-exact form exists (0 of ~300 compiled forms keep any stage loop); IL-family knobs do give loop-exact pairs
  (K0 = K8192, K20480 = K20482) used to isolate mechanisms.
- Mechanism (ncu, zero perturbation): transition 1338 -> 865 cyc per warp-item at bs256 (221 -> 138 instructions
  executed), 1358 -> 822 at bs128 (183 -> 99). bs256 converts (+0.6..+1.1% over 5 forms, 4 jobs); bs128 does not:
  its 1-group phase is stream-bound and the next item's first-stage data wait absorbs the saving (every MW4 IL form
  0..+0.3% slower, so bs128 keeps the base kernel).
- Ruled out: blocking early record (bit 4096 / 12288: deadlock, check_mt timeout), zero-only-K accumulators (1024:
  wgmma serialization), pinned staging offset (8192: 0, same loops), goto end (65536: -0.2% draw), bs128 IL.
- For the coordinator's stack: the bs256 gain is ~half transition mechanism, half stage-loop draw (K-forms spread
  -0.2..-1.0%); any other change inside moe_8_8_1_3_8 re-draws its loops -> re-measure; backups with the same
  mechanism: K20480 (moe_il_5, -0.69% over 6 copies), K20488 (-0.67%, spills 16/52).
