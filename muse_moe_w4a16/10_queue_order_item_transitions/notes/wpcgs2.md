# WP-CGS2: loop-identical forms of the bs16 (4,2,1) KRS k-split last round, base v1508 (2026-11-01)

Worker WP-CGS2. Base dev/moe_v1508.py (judged build: MOE_S5R_FIT). Files: dev/moe_v2700-v2714.py (each made by
dev/wpcgs2_mk27NN.py from the previous one; every default build = v1508's SASS in all 16 kernels, checked with
dev/wpcgs_search.py --full), scripts dev/wpcgs2_*, logs kda_ws/runs/wpcgs2_run1-5.log (tables: dev/wpit2_tab.py <log>
'^v1508[b-y]$'). Search JSON / SASS in the session scratchpad wpcgs2/.

**Result: no loop-identical form of the split exists with the pieces on hot code; the split itself, in v1605's form
(stage loops re-scheduled), is a robust +0.9% win at bs16 on v1508 and is baked: dev/moe_v2714.py (MOE_E2_KSXNT2 2 on,
escape MOE_NO_CGS2KS).**

## 1. Tools

- dev/wpcgs2_search.py: dev/wpcgs_search.py (same arguments) + two loop classes per base loop in the JSON and a summary:
  SCHED = a candidate loop with the same opcode sequence AND the same control words (stall / yield / barriers / reuse),
  i.e. the same schedule with other register numbers; SCHEDSB = the same up to a scoreboard permutation. It also uses
  quick_ratio for the ~1300-instruction "producer loop" regions (their O(n^2) difflib ratio made the main process the
  bottleneck: 2048 candidates went from ~2 h to ~25 min).
- dev/wpcgs2_loops.py KERNEL a.sass [b.sass ...] (loop table) / --diff I J / --pdiff I J (unified diff of a stage /
  producer loop with control words) / --cmp base.sass cand.sass ... (per base loop: EX / TXT / REN / SCHED / SCHEDSB /
  OPS (same opcodes only) / similarity).

moe_4_2_1_4_1 in v1508: 5336 instructions, stage loops 558 (2-group) / 512 (1-group), 5 "producer loops" (one real
244-instruction copy loop and four overlapping 1258-1465-instruction regions that contain the producer's item decode).
Its whole-item stage count is the compile-time 8 (ST1 = 16 / KW = 8, down 16 / (KW KS2) = 8, no warm-up): the loop exit
is `ISETP.GE.U32 ... R, 0x8`.

## 2. Port

dev/moe_v2700.py = v1508 + v1605's Python wiring (with MOE_E2_KSXNT2 = 2 | 4 the (4,2,1) kernel gets state["ykz"] as
ypart: KRS red.add pieces, shared (4,1,1) parity / record / zeroing protocol) + MOE_S2_SPLW2 = f (float, the split amount
in waves; without it v1508's MOE_SPLW = 0.75 waves for KW > 1 applies; v1508 has no MOE_E2_KSPL). v2700 KSXNT2=2:
moe_4_2_1 6104 instructions sha 3e42fc6b99cb, stage loops 552 / 511 (sim 0.45 to v1508's), producer regions 0.81.

## 3. What re-schedules the (4,2,1) loops (code-presence bisection, dev/moe_v2701.py MOE_CGS2_NX + MOE_WZ_KSXR; invalid builds)

| change (only this one) | stage loops | producer regions |
|---|---|---|
| consumer stage count `f.tile >= tsplit ? 4 : 8` (KSXR 2) | 0 / 2 (sim 0.37) | 1 / 5 |
| the same count read from the stage record (v2713 MOE_CGS2_NR, no tsplit in the consumer, all else neutral) | 0 / 2 (0.50 / 0.46) | SCHED with a laundered count |
| combine split path (KSXR 8, KSXM) | 0 / 2 (0.24) | 1 / 5 |
| SPLFV (split tokens on the fast units) | 0 / 2 | 1 / 5 |
| epilogue warps' split path (KSXR 4) | 2 renamed | 0 / 5 |
| producer decode (KSXR 1), KRSV protocol, routing nspl, epilogue tsplit read | exact | changed |

=> A variable trip count of the item loop's stage loop re-schedules both loops by itself; the only way to keep them is
to run the split pieces through OTHER stage-loop code. The producer regions contain the item decode, which must change.

Probes (dev/moe_v2705.py / v2707.py MOE_CGS2_PRB, never-taken or empty edits, (4,2,1) only): after the item loop a
never-taken store, a store of lane / tid / a record field, or a never-taken bar.sync keep the stage loops exact; an
empty `asm volatile("")`, `it` or kw / rw live after the loop, a never-taken stage_compute re-schedule them. At the item
top a never-taken `break` keeps them (stage), a never-taken store does not.

## 4. Forms searched

- SPX (v2702): the piece on its own stage loop inside the item loop (constant whole-item count kept): 64 forms
  (tsplit from shared memory, laundered condition / fields, one 2-group loop, bottom-tested, laundered count): 0 exact,
  stage sim 0.37-0.47. SPXF (v2703): the piece in a __noinline__ function: ptxas inlines it; even an empty dummy call
  re-schedules (0.40 / 0.47). SPY (v2704): a second loop after the item loop with an exit test at the item top: the
  exit test alone re-schedules; the second loop alone (no exit test) too.
- FX (v2706-v2710, MOE_CGS2_FX): the producer marks a split piece's record item = -3, so the consumers leave the item
  loop through its EXISTING end path (no edit of the item loop); the pieces run in a second loop after it; the last
  piece of each (tile, row block) (static, ks = KSX-1) acquires the others' fp32 rows (ykz2, self-cleaning), writes the
  final BF16 y + slot sums, and releases cnt2 once: epilogue warps and combine compiled as without the split.
  Without the second loop: both stage loops EXACT (producer regions changed). With it: re-scheduled until the second
  loop stopped reviving item-loop values: FXK 16384 (ring position from scal[10], stored by the producer at the CTA's
  first split stage), 32768 (thread offsets from a laundered tid), 65536 (shared-memory pointers from a laundered smem
  base). Full product of 11 bits (2048 forms, t28): 84 SCHED for both stage loops (same opcodes + control words; the
  registers are v1508's shifted down by 4: 4 fewer long-lived registers), 64 + 36 partial, 0 byte-exact.
- FY (v2711-v2712, MOE_CGS2_FY): v1605's own epilogue (KRS red.add) and combine for the pieces, with FX's consumer
  structure (the epilogue warps take the item loop's end record as a phase end once when scal[10] says split pieces
  follow; the second loop hands pieces over through the SEPI buffer exactly like the item loop). 128 forms: 0 SCHED
  (stage sim 0.38-0.47). (v2712 FYK 4: the combine re-reads tsplit / psplit; a shadowing declaration instead of an
  assignment re-scheduled ALL kernels with a combine, even with nothing else changed.)

## 5. GPU results (bench2 jm 100 reps x 2 passes, B2_SHARE_EQ + B2_NOVERIFY (correctness separately), v1508 fillers at
both ends, 3-4 middle v1508 controls; % vs the middle-control mean; * = above every middle control)

Correctness: every build (FX 5 rules, FY 5 rules, v2700 6 rules, v2714): hang1 smoke ok, check_mt M 9-16 (v2714: 2 8 9-17)
normal + skewed x 3 seeds x 2 calls 0 failures, wps2_mix 120 interleaved bs 1-33 calls (v2714: 200) 0 failures, worst
rel-L2 3.3e-3.

| build | bs16 | note |
|---|---|---|
| FX K2 / K2w5 / K2w10 / K4w5 / K4w10 (run 1, schedule-identical stage loops) | -11.9 / -10.3 / -13.7 / -14.6 / -21.1% | static fix-up waits (convoys) + slow second loop |
| FY K2 / K2w5 / K2w10 / K4w5 / K4w10 (run 2) | -7.8 / -7.3 / -8.1 / -6.5 / -8.0% | |
| FY K2 diag (run 3): as is / no stage compute in the second loop / 1 split tile | -8.2 / **+3.7** / -5.4% | the second loop's compute is the loss |
| v2700 K2w10 (v1605 form), runs 2 / 4 / 4 | +1.06* / +0.80* / +0.78* | |
| v2700 K4w5, runs 2 / 4 / 4 | +0.76* / +0.72* / +0.81* | |
| v2700 K2 (0.75 waves), runs 3 / 4 / 4 | +1.08* / +0.90* / +0.76* | |
| v2700 K2w5 / K4 (0.75) / K4w10 (run 4) | +0.36* / +0.08 / +0.13* | |
| **v2714 (bake = v2700 K2) x4, run 5** | **+0.75* / +1.08* / +1.07* / +0.82*** | escape build -0.22% (control) |
| middle controls (runs 1-5) | -0.24 .. +0.18% | |
| bs8 / bs32 placebos (identical kernels) | -0.27 .. +0.37% | noise in every job |

Why the separate-loop forms lose although their whole-item loops keep v1508's schedule: the pieces' own stage-loop
copy runs only in the last round, cold in the instruction cache, on every CTA's critical path, and its schedule is worse
(static stall sums 1130 / 944 vs 842 / 766 for the item loop's 2- / 1-group loops in the FY build); one split tile
(~1 piece per CTA on 64 CTAs) already costs 5.4%, removing only the second loop's compute turns -8.2% into +3.7%. FX
adds waits (a block's last piece holds its CTA until the earlier pieces, queued behind other items elsewhere, are done).
v1605's form runs the pieces on the hot item-loop code: its re-scheduled loops (WP-S2's +8% 1-group item time on v1002)
cost less than the tail they remove on v1508.

## 6. Banked

**dev/moe_v2714.py** = v2700 with MOE_E2_KSXNT2 2 on by default (escape **MOE_NO_CGS2KS** = v1508's SASS in all 16
kernels, checked; default build = v2700 KSXNT2=2's SASS in all 16, checked). Only moe_4_2_1_4_1 changes: sha
3e42fc6b99cb (v1508 aef9ae76c3f5), 6104 instructions, 0 spills, 168 registers (sass13.sh hashes, MOE_S5R_FIT).
bs16 over 7 copies / 3 jobs (runs 3, 4, 5): +1.08, +0.90, +0.76, +0.75, +1.08, +1.07, +0.82% (mean **+0.92%**), every copy
above every middle control; geomean over the six shapes ~+0.15%. Alternatives at the same level: K2 + MOE_S2_SPLW2=1.0f
(+0.88% mean of 3), K4 + SPLW2=0.5f (+0.76% mean of 3).

## 7. Lessons for loop-rule work on this kernel

- The item loop's stage-loop schedule depends on the trip count being a compile-time constant; any runtime count (even
  a record field) re-schedules it. A different stage count needs different loop code, which is cold when it runs only
  at the tail: a loop-identical split costs far more than the re-schedule it avoids.
- Code after the item loop re-schedules it when it revives values the loop's region holds (it, kw, rw, pointer
  registers); rederiving them (shared-memory ring position, laundered tid / smem base) gave schedule-identical loops.
- Leaving the item loop through its own end path (a record marker set by the producer) changes no item-loop code.
- SCHED (same opcodes + control words, registers renumbered) is the closest class reached; byte-identity also needs the
  same set of long-lived registers across the loop.
- Declaring a shadowing `const int tsplit = ...` in the combine changed every kernel's SASS (spills in four); an
  assignment under `if constexpr` changed none.
