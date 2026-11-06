# S5: cold code behind branches on the batch-1 sliding kernels xe_r1_1 (d1s) and xe_b1s_8c6 (v1s) (worker S5, 2026-10-29)
Base dev/coord/k56.py (md5 427cc112). Controls dev/s5/k56a.py = k56b.py (byte copies). Cap: 10 GPU-min total (J1 <= 4, J2 <= 6).

## Resume here
- DONE: win FROZEN as dev/s5/deliver_r1/ (build.py <base> <out>: defaults VOTE=1 SINCALL=2, ZERO=1 = k56; k56_s5r1.py md5 7e7a2c33bce6c1efc43e28673f9775fd =
  build(dev/coord/k56.py md5 427cc112); nsys_table.txt, out_j1..3.txt, md5s.txt).  v1s nsys -0.096 .. -0.111 us in 4/4 runs (copies x passes) vs k56a/k56b;
  d1s and d16s untouched (xe_r1_1 SASS identical modulo label numbers, xe_dm identical) and flat; t_xe FULL 0 failures (worst err/bound 0.543 = the base
  kernel's), tools/check.py CHECK_RC=0, 56 PASS, 0 failures.  GPU used: J1 3.3 + J2 3.0 + J3 2.4 = 8.7 of the 10-min cap.
  Score: v1s 2.17 %/us x 0.10 us ~ +0.2 % local.  Stack with the other workers' wins by applying build.py to their file (it patches only
  XE_SRC namespace xer2 prep code + one owner `if`; needs the anchors of k56's XE section).
- Next lever candidates (not funded here): d1s (xer1) has ONE sincosf (row 0) before its Q chain: SINC on d1s was +0.19 (a pre-existing scheduling
  sensitivity, not understood) so the same change is not free there - test SINCALL=2-style on xer1 only with an A/B; v1s: the nk/owner path after the Q-ready arrive.

## Files
- build.py <base> <out> [MISS=1 EDGE=1 NK=1 SINC=1 VOTE=1 ALL=1]: patches XE_SRC in place (string replace with asserts). All 0 = base file
  byte for byte (md5 427cc112, cubin identical to k56's: cmp of the NVRTC-13 cubin dev/s5/sass_off/mod.cubin vs sass_base/mod.cubin).
- cfg.py <kernel.s> [--min N]: forward-branch inventory (target, skipped instructions); sass_li.py + blk.py: same with source lines (-lineinfo,
  nvdisasm -g) so each skipped block is named; regs.py <file>: regs / spills for xe_r1_1 and xe_b1s_8c6.
- v_MISS.py v_EDGE.py v_NK.py v_SINC.py v_VOTE.py v_ALL.py = builds with that knob; j1.sh = the bound job.

## Local SASS facts (NVRTC 13.0, sm_90a, default build = k56): xe_r1_1 1232 instr 82 regs, xe_b1s_8c6 1912 instr 92 regs, 0 spills (32 B stack = sincosf)
Forward branches with a skipped block >= ~9 instrs (instruction counts; "taken" = the common case in the benchmark shapes):
d1s xe_r1_1 (compute warps 0x00e0-0x1e80 = 475 instrs, prep/owner warps 0x1e90-0x48d0 + out-of-line tails to 0x4c80):
  compute warps (straight-line otherwise; no loop, no call):
    0x0b80 @!P0 BRA  skip 21  = length-speculation MISS path (lo/n/C/base/cnt recompute + page-table LDG + the 2-pass loop back edge 0x0cd0), taken on a hit.
    0x1060 @!P0 BRA  skip 15  = `if (edge)` mask block (keys past the CTA range; taken for 3 of 4 warps in 15 of 16 CTAs, executed by warp 3 of rank 15).
  CTA reader (0x1a50 BAR.SYNC 1 .. EXIT 0x1e80, ~100 instrs): no forward branch at all.
  prep/owner warps: 0x1ed0 / 0x1ef0 skip 17 / 14 (mbarrier init block, owner CTAs 0..7 only, before Q-ready); 0x2350 / 0x2370 skip 64 / 60 (sincosf
    Payne-Hanek slow path, always skipped); 0x35d0 / 0x3640 skip 20 / 13 (`if (L==8||lane<16)` smem store, `if (cw)` cache-row write: CTA 0 only);
    0x3d40 skip 185 = the owner's mbarrier try_wait RETRY loop is out of line at 0x48e0 (after EXIT 0x48d0), the merge code (185 instrs) lies between;
    0x47a0 skip 14 (`if (dp)` store); 0x4950-0x4a30 (skips 11/41/34/30) = __frcp_rn slow path (called from the nk block only when kv scale is not 2^k, never here).
  There is NO exact-fallback merge in xe_r1_1 (round-1 kernel: only the classic max merge exists).
v1s xe_b1s_8c6 (compute warps 0x00e0-0x2f70 = 746 instrs, prep/owner 0x2f80-0x7210 + tails to 0x7680):
  compute warps: 0x10b0 @!P0 BRA skip 29 (MISS path), 0x17d0 skip 9 + 0x1860 skip 42 (`if (edge)` mask: sliding left edge of the 8 tokens + range end).
  CTA reader: straight-line (0x26e0 BAR.SYNC 1 .. 0x2f70), no forward branch.
  prep/owner: mbarrier init skips 17 / 14; sincosf slow-path skips 64 / 60 / 63 / 59 / 64 / 60 + a 96 (3 sincosf: rows pw, pw+4 and the Q token's row);
    0x4790 skip 117 and BRA.DIV 0x4a00 -> 0x7220 (divergence check of the shuffles, never taken); 0x5550 skip 14 (cw cache write);
    8 x `@!P BRA` skip 9 (new-token partial `if (i <= jo)`, BSSY/BSYNC each; pre-wait, off the critical path); the owner's mbarrier retry loop out of
    line at 0x7300 (skip 135 over the merge); OWNER MERGE: `VOTE.ALL P0, !PT` + `@P0 BRA` (never taken) at 0x6c20/0x6c50 + a taken `BRA` at 0x6f50 over an
    18-instr DEAD block (the XE_REF plain-sum path): `if (__all_sync(0xffffffffu, allref))` with allref = XE_REF && ... = constant false that
    NVVM/ptxas do not fold through the vote.  The reader's `XE_REF && __all_sync(same)` is guarded by the macro and IS folded.
  => the "fallback merge" family of the brief does not exist as inline code in k56's SASS except this dead vote on v1s.
Total cold instructions behind compute-warp forward branches: d1s 21 + 15 = 36 of 475 (7.6 %); v1s 29 + 51 = 80 of 746 (10.7 %); S2's v1f had 504 (+ 92 x 3).
Static sizes: d1s compute path 7.6 KB, v1s 11.9 KB (post-landing straight-line 256 / 449 instrs, P1); the cold blocks are 336-1030 B ahead of their branch.

## Knobs (build.py) and what each deletes (SASS check, instrs d1s / v1s, regs d1s / v1s; base 1232 / 1912, 82 / 92)
- MISS (timing-only; correct whenever the guess hits, which it does in the benchmark shapes): compute warps verify nothing and never re-issue: 1200 / 1856, 82 / 94.
- EDGE (timing-only): `edge = false`: 1216 / 1856, 82 / 90.
- NK (timing-only): the new-token block (RMSNorm, RoPE, E4M3, cache-row write, rcp slow path) deleted from the prep warps: 1064 / 1728, 82 / 92.
- SINC (timing-only): sincosf -> __sincosf (no Payne-Hanek, no 32 B stack): 1128 / 1632, 84 / 92.
- VOTE (CORRECT, no output change): `XE_REF && __all_sync(...)`: v1s 1888 (-24: vote, dead branch, 18-instr dead block); d1s unchanged (1232, identical).
- ALL = all five: 904 / 1304, 84 / 90; the compute-warp and reader paths of both kernels have NO forward branch left.

## Decision rules (written before J1)
Measure: nsys_go per file, shapes "1,1,sliding;1,8,sliding;16,1,sliding", 2 passes; per pass the files load in the order k56a, k56b, ALL, ALLb,
MISS, EDGE, NK, SINC, VOTE.  Delta(run) = variant median - mean(k56a, k56b medians of the same pass) per shape (also look at means; the timestamps
quantize at 0.032 us).  ALL has 2 copies x 2 passes = 4 runs, every family 1 copy x 2 passes = 2 runs (budget: J1 <= 4 GPU-min).
1. KILL: if ALL's 4 deltas have mean > -0.05 on d1s AND mean > -0.08 on v1s (bound below the build thresholds), nothing can pay: STOP after J1,
   report (J2 not run).  Nothing else is built; a family cannot beat the umbrella.
2. FAMILY QUALIFIES for a correct fix iff its 2 runs are both <= -0.05 on d1s (v1s: both <= -0.08) and ALL's 4 runs do not contradict
   (all 4 <= -0.03 / -0.05).  The fix is then built per the brief: the hot path carries no rare-path code; the rare path is a separate
   instance / out-of-line block reached by one NOT-TAKEN branch (miss: `goto` to a block placed after the hot path's EXIT, the whole post-landing body
   instantiated a second time there; edge: a masked instance used only by the warps that need it, S2 MFREE style; new-token / sincos: they are on the
   prep path, only worth it if NK / SINC show the prep chain is critical).
3. VOTE is correct and free: it goes to J2 as a rider iff its v1s delta is <= -0.02 in both runs (J2 then re-measures it with 2 copies x 2 passes), and
   is kept iff it is not slower than the controls in any run and passes t_xe.  No regression allowed on d1s / d16s (> +0.03 in any run kills it).
4. J2 (<= 6 GPU-min): the built fix(es) + k56 controls, 2 copies x 2 passes; keep iff >= 0.03 faster on d1s or >= 0.04 on v1s in EVERY run and no
   regression elsewhere; t_xe FULL 0 failures then tools/check.py 56 PASS on the exact built file in the same job -> freeze dev/s5/deliver_r1/.
5. Smoke: t_xe --quick on v_ALL only (proves the XE path runs; the wrong-output builds are expected to FAIL value checks, only crashes / "NOT XE" matter);
   the other bound builds are proven by their nsys kernel names (xe_r1_1 / xe_b1s_8c6 rows).

## Job s5_j1 RESULT (2026-10-29 14:42-14:46, 3.3 GPU-min; dev/s5/out_j1.txt; table: python3 dev/s5/table.py dev/s5/out_j1.txt)
nsys medians (us), pass 1 | pass 2; controls (mean of k56a, k56b): d1s 3.328 | 3.328, v1s 4.152 | 4.192, d16s 6.032 | 6.016.  Deltas vs controls (med):
  file     d1s              v1s               d16s            SASS instrs d1s/v1s
  ALL x2   +0.16 +0.13      +0.31 +0.26 (copy b +0.26 +0.22)   -0.08 -0.06   904 / 1304
  MISS     +0.16 +0.16      +0.36 +0.38                        -0.03  0.00   1200 / 1856
  EDGE     -0.03  0.00      +0.04 -0.03                        -0.02 -0.03   1216 / 1856
  NK       +0.08 +0.05      +0.02 -0.03                        -0.05 -0.05   1064 / 1728
  SINC     +0.19 +0.18      -0.09 -0.13 (means -0.12 -0.14)    -0.02  0.00   1128 / 1632
  VOTE      0.00 +0.03 (xe_r1_1 SASS identical = noise)  -0.02 -0.03 (means -0.01 -0.06)   -0.03 -0.03   1232 / 1888
Noise floor: +-0.03 per run (d1s file with identical SASS: 0.00 / +0.03; control copies +-0.016).
Verdicts (rule 1/2 of the plan):
- The umbrella ALL is NOT a clean bound: MISS removes the 2-pass issue loop and ptxas then issues K FIRST (KKKK VVVVVVVV; default is V first, K last in
  every XE kernel, P1 found K-first +0.02 / +0.09 and here +0.16 / +0.36), so ALL / MISS measure the load order flip, not the removed cold block.
  The miss path itself cannot be removed cleanly: a goto-based placement of the miss block behind the hot path (MISSFIX, built, in build.py) was
  re-laid out by ptxas (block placed before its sole successor), +112 / +136 instrs, v1s 111 regs: dropped.
- EDGE (the edge mask: 15 / 51 cold instrs behind forward branches on the compute warps): 0.00 +- 0.03 => DEAD (bound < 0.05 d1s, < 0.08 v1s).
- NK (new-token block): d1s slower (+0.05 +0.08), v1s 0 => DEAD.
- Cold code behind the compute-warp / reader branches therefore carries NO measurable cost on d1s / v1s (36 / 80 instrs, one-shot code per warp, no loop).
- SINC: v1s -0.09 / -0.13 in both runs = meets the v1s bar (<= -0.08 both).  d1s +0.19 (SINC is NOT applied to d1s).  VOTE: v1s -0.02 / -0.03.
- Where SINC's gain comes from (SASS, dev/s5/sass_off/xe_b1s_8c6.s + sass_li): libdevice sincosf loads its coefficient table with an LDG.E.CONSTANT
  (a global read: a cold DRAM trip after the L2 flush) and the v1s prep warp 0 runs THREE sincosf in a row (rows pw, pw+4, then the Q token's row) before
  the Q-ready barrier, each behind a `@!P BRA` that skips ~124 instrs of Payne-Hanek (no overlap between them); the other three prep warps do two and
  all four must arrive at barrier 3.  The prep warps' ~200 instrs also share the issue slots with the compute warps' K/V issue window (page-table
  landing 0.54 -> K/V issued 0.80, 2 compute warps + 1 prep warp per scheduler).  Both are CORRECT-fixable without touching the cold code:
  QF    prep: the Q chain (rtq sincosf -> RMSNorm/RoPE -> qf, cks) and bar.arrive(3) first; the two new-token rows' sincosf after the arrive;
        warps pw >= NT arrive at once.  Bit-identical numerics.
  GATE  named barrier 7: compute warps bar.arrive after their K/V issue, prep warps bar.sync (after their loads are issued) before any ALU work.
        The prep loads must be PLAIN ld.global (ptxas sank all the .nc prep loads below the bar.sync in the first build: v1s SASS checked, now
        every prep LDG precedes BAR.SYNC 7; d1s already had that order).
  SINCQ the Q chain's sincosf -> table-free 2-term Cody-Waite + Cephes polynomials (|x| < 1e5), `sincosf` in a __noinline__ function on the rare
        branch (CALL block 4 instrs, 0 spills, 56 B stack); CPU emulation over 48k RoPE-like angles: worst abs err 8.6e-8 (libdevice-class 5.4e-8).
        New-token rows / K-cache rows keep libdevice sincosf (K cache bit-identical).
  VOTE  rider (correct, SASS -24 instrs).   GATE1: the same gate on d1s (xe_r1_1, count 160).
  All arms: compute-path load order unchanged (V first, K last), 0 spills, regs 92 / 82 as the base, d16s kernel SASS identical.

## Job s5_j2 plan (written BEFORE the job; ~5.5 GPU-min incl. the freeze checks; total after J2 <= 9.5 of 10)
Files (2 identical copies each, controls k56a/k56b loaded FIRST in each of the 2 passes): armA = VOTE QF GATE SINCQ GATE1, armB = VOTE QF SINCQ,
armC = VOTE QF GATE.  Smoke first: t_xe --quick on armA (timeout 90): a failure or a hang drops the GATE arms (A, C) from the nsys loop.
Decision (dev/s5/decide.py, runs inside the job on the nsys output; delta = median of a copy minus the mean of the two controls' medians of the same pass):
- v1s config X in {A, B, C} is OK iff all 4 of its v1s deltas <= -0.04 and all 4 d16s deltas <= +0.06.  Among OK arms take the best mean delta; within
  0.02 of the best prefer C (bit-exact numerics) over B over A.
- GATE1 (d1s) is OK iff all 4 d1s deltas of arm A <= -0.03 (arms B and C leave xe_r1_1 untouched: they are the d1s noise reference).
- flags = VOTE + the chosen arm's flags + GATE1 if OK; NONE -> no freeze, report.  The two kernels are independent (separate .text sections), so the
  components stack; the job builds the final file from build.py with the flags, prints its md5, and runs t_xe FULL + tools/check.py (56 PASS, 0
  failures) on that exact file.  Freeze = dev/s5/deliver_r1/ (build.py with the winners as defaults, built file, md5, nsys table).
- Fail-safe: any t_xe/check failure on the final file = no freeze (report the numbers only).

## Job s5_j2 RESULT (2026-10-29 15:00:34-15:03:31, 3.0 GPU-min; dev/s5/out_j2.txt; decide.py output inside): NO WIN
Smoke armA t_xe --quick: 0 failures.  nsys medians (copy a / copy b), pass 1 | pass 2.  Controls: d1s 3.392 3.392 | 3.392 3.424, v1s 4.192 4.192 | 4.192 4.192, d16s 6.048 | 6.048.
  armA (VOTE QF GATE SINCQ + GATE1): v1s 4.512 4.512 | 4.512 4.480 (+0.29 .. +0.32), d1s 3.392 3.424 | 3.424 3.456 (0 .. +0.05), d16s 0
  armC (VOTE QF GATE)             : v1s 4.480 4.512 | 4.512 4.512 (+0.29 .. +0.32), d1s 0
  armB (VOTE QF SINCQ, no gate)   : v1s 4.160 4.192 | 4.161 4.192 (deltas -0.032 -0.031 0.000 0.000), d1s 3.360 3.361 | 3.392 3.392 (xe_r1_1 SASS identical to base: noise -0.03..0)
Verdicts: GATE (prep ALU work after the compute warps' K/V issue) is a +0.3 us LOSS on v1s: the Q-ready chain (loads landed -> sincosf with its table trip
-> RMSNorm/RoPE -> qf) is NOT slack; pushing it behind the K/V issue (0.8 us) delays the first QK by ~0.3.  So contention in the K/V issue window is not the
SINC mechanism.  GATE1 on d1s: no gain (0 .. +0.05).  QF + SINCQ (Q-ready chain shortened: one sincosf first, table-free) + VOTE: mean -0.016 on v1s, below the
VOTE-alone J1 numbers (-0.024 / -0.032) => QF / SINCQ add nothing measurable: the Q-ready time is not what SINC bought.  decide.py: FLAGS=NONE.
What is left of the SINC bound (-0.09 / -0.13): the two NEW-TOKEN ROW sincosf (+ the owner / nk path after the Q-ready arrive) were not touched by B.
## Job s5_j3 plan (written BEFORE the job; last job, cap: <= 3.5 GPU-min => total <= 9.8)
armE = VOTE + SINCALL=2: the three prep sincosf (rows pw, pw+4 and the Q row) -> table-free reduction (j = rint(x 2/pi) by the magic-number trick, 3 fmaf of
pi/2 split in three constants, quadrant by the low bits of j) + __sinf / __cosf (MUFU) on |r| <= pi/4, abs error <= ~3.6e-7 (FP16 Q / E4M3 K quantization
are 1e3 / 1e5 times coarser); |x| >= 1e5 or NaN -> libdevice sincosf in a __noinline__ function (called only there, 0 spills).  SASS 1800 instrs (base 1912),
regs 92, load order V first / K last unchanged; d1s and d16s code untouched.  Controls k56a / k56b first, armE + copy, 2 passes (8 nsys runs ~ 55 s).
Rule (decide3.py): E qualifies iff all 4 v1s deltas <= -0.04 and all 4 d16s deltas <= +0.06 (the brief's >= 0.04 us on v1s in every run).  Then the job runs
t_xe FULL + tools/check.py on dev/s5/armE.py (the exact file, md5 printed) and I freeze deliver_r1 if both are clean.  Otherwise NONE: report, stop.

## Job s5_j3 RESULT (2026-10-29 15:06:57-15:09:18, 2.4 GPU-min; dev/s5/out_j3.txt): WIN, FROZEN as dev/s5/deliver_r1/
armE (VOTE + SINCALL=2; md5 7e7a2c33; two copies E / Eb) vs controls k56a / k56b, nsys medians (copy a / copy b), pass 1 | pass 2:
  controls v1s 4.192 4.192 | 4.192 4.192; E v1s 4.081 4.096 | 4.096 4.096  => deltas -0.111 -0.096 | -0.096 -0.096  (means 4.192 4.131 | 4.116 4.140 vs 4.221 | 4.248)
  d1s (xe_r1_1 SASS unchanged) controls 3.392 3.392 | 3.392 3.376, E 3.392 3.360 | 3.392 3.392 (noise 0 .. -0.03); d16s 6.080 6.080 | 6.048 6.080 vs 6.048-6.112 (noise).
  decide3.py: E ok -> job ran t_xe FULL on armE.py (rc 0, 0 failures, worst err/bound 0.543 at B 33 = base kernel) and tools/check.py (CHECK_RC=0, CHECK_PASS=56,
  0 failures).  The 4 v1s runs are all <= -0.096 (rule: <= -0.04 each).
What the win is (SASS, dev/s5/sass_deliver/xe_b1s_8c6.s vs sass_base): 1912 -> 1800 instrs, 92 regs, 0 spills, load order V first / K last unchanged, forward branches
into cold code on the prep path: 3 sincosf slow-path skips (64/60 each x3 sites) gone from the hot instance (the Payne-Hanek fallback is a __noinline__ function
called on `|x| >= 1e5 or NaN` only, a 4-instr CALL block skipped by a 4-instr branch), libdevice's coefficient-table LDG (a global read: cold DRAM trip after
the L2 flush, behind each angle) gone, the dead `VOTE.ALL` + never-taken branch + taken BRA over the 18-instr dead plain-sum block gone from the owner merge.
Numerics: x = fp32 angle as before; reduction by the magic-number rint + 3-constant Cody-Waite (|x| < 1e5 < 105615), quadrant from the low bits of j,
__sinf / __cosf on |r| <= pi/4 (abs err <= 3.6e-7 + reduction 6e-8; CPU emulation of the polynomial variant: worst 8.6e-8 over 48k RoPE-like angles).  K cache rows
use the same sin/cos (E4M3-quantized: a flip needs a 1e-7 perturbation to cross a 6 % bin edge).
Which part carries it (J1/J2 evidence): NOT the Q-ready chain alone (QF + SINCQ, only the Q row table-free + reorder: -0.016), NOT a gate (GATE +0.3), NOT the nk block,
NOT the edge mask or the miss path (bound ~0): it is the three prep sincosf calls together (SINC bound -0.09 / -0.13, E -0.10), i.e. the long libdevice sincosf
(table trip + slow-path skips, 3 in a row, 12 prep warps share 4 schedulers with the compute warps) in the first microsecond; the exact split is not resolved.

## Lessons / dead list (S5)
- Cold blocks behind forward branches in xe_r1_1 / xe_b1s_8c6 compute warps + reader + owner: 36 / 80 static instrs; EDGE bound 0.00 +- 0.03, NK +0.05..+0.08 (slower), MISS flips the
  K/V issue order (K first, +0.16 d1s / +0.36 v1s: a measurement artifact, not a bound); a goto-placed miss block is re-laid out inline by ptxas (MISSFIX, dropped).
- A bar.sync does not hold ld.global.nc loads (ptxas sank all prep loads below the gate): use plain ld.global (ldq_*) when a barrier must follow a load issue.
- Delaying the prep chain behind the compute warps' K/V issue (GATE) costs +0.3 us on v1s: Q-ready is close to critical, not slack.
- d1s (xe_r1_1): SINC +0.19, GATE1 0 .. +0.05, NK +0.05..0.08: any change to its prep warp so far costs or is flat; its single sincosf stays.
