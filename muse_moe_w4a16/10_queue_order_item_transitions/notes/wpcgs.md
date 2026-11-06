# WP-CGS: codegen-search harness for loop-preserving forms of shelved ideas (2026-11-01)

Worker WP-CGS. Base dev/moe_v1507.py (judged build: MOE_S5R_FIT). Namespace v2600-v2699, defines MOE_CGS_* (plus the
ported MOE_END_RO / MOE_END_ROK, MOE_IT2_TMG, MOE_IT_DB, WP-COLD's MOE1T_*), scripts dev/wpcgs_*, logs
kda_ws/runs/wpcgs_runN.log. Local NVRTC 13.0 (= judge toolchain) through ctypes, cuobjdump 13.

## 1. The harness: dev/wpcgs_search.py (+ dev/wpcgs_paths.py)

    python3 dev/wpcgs_search.py --ver dev/moe_v2605.py --kernels moe_8_4_1_4_1 --sem MOE_END_RO \
        --bits "MOE_END_ROK=1,2,4,8" --bits "MOE_CGS_ROK=1,2,4,8,16,32,64,128,256,512" --bits "MOE_E2_KN1=2,16,32" \
        --knob "none|MOE_E2_FUN|MOE_E2_LND" --sample 400 --seed 1 --tag ro1
    python3 dev/wpcgs_search.py --full --ver dev/moe_v2605.py --tag dflt          # default build == v1507 in all kernels?
    python3 dev/wpcgs_search.py --kernels moe1t_4,moe_8_8_1_3_8 --selftest         # reduced module == full module?

- Inputs: base file (default v1507) + base defines (default MOE_S5R_FIT), candidate file(s) (`--ver`, repeatable), the
  semantic change (`--sem`), knob axes: `--knob "A|B,C|none"` (alternatives, each a define list) and `--bits NAME=b1,b2,..`
  (every OR-subset). Full product, or `--sample N --seed S` (+ the all-first combination).
- Compiles only the target kernels (`--kernels`): a reduced module with only their INST() lines and the moe1 / moe1t entry
  points that are targets; `--selftest` checks it gives the full module's SASS (it does: moe1t_4, moe_8_4_1_4_1,
  moe_8_8_1_3_8 OK). One process per candidate (`-j`, default nproc - 1 = 11): ~2 s per bs32-kernel compile, ~10 s for
  both (8,8,1) kernels, ~1 s moe1t; 400 candidates in 2 minutes.
- Per target kernel and candidate: stage loops = innermost backward-branch bodies with HMMA / HGMMA (< 1500 instructions),
  producer loops = innermost bodies with UBLKCP / UTMALDG; each base loop is matched in the candidate by
  (1) EXACT: same instruction text (branch targets made loop-relative) AND the same control words (stall / yield /
  scoreboards / reuse = bits 41-63 of the second encoding word), (2) scoreboard-permuted: same text, control words equal up
  to a permutation of the 6 dependency barriers (schedule-identical), (3) ctl-bits only, (4) register-renamed, else the
  best opcode-order similarity. Spill signature: LDL / STL counts, cuobjdump -res-usage REG / STACK, ptxas -v spill bytes
  (absent when the driver's compute cache served the cubin: NVRTC 13 uses ~/.nv/ComputeCache). EXACT = every stage and
  producer loop of every target kernel exact and the spill signature unchanged. Kernel sha = wpc_sass.py's (comparable
  with every earlier worker's hashes).
- Output: a line per candidate, a summary (EXACT first, then by identical loops), <tag>.json in the session scratchpad
  wpcgs/ (override WPCGS_DIR); the .sass of EXACT candidates is written for inspection (--keep: every candidate's). The
  harness never deletes a file (one cubin per worker process, overwritten). (Note: the first version of this session's
  harness removed its own temporary cubins / non-exact .sass in the scratchpad; fixed before the end of the work.)
- GPU jobs: dev/wpcgs_run1-5.sh (hang1 smoke -> check_mt -> bench2 jm, fillers at both ends, v1507 controls in the middle),
  logs kda_ws/runs/wpcgs_run1-5.log, tables by dev/wpit2_tab.py <log> '^v1507[b-y]$'.
- dev/wpcgs_paths.py dump.sass kernel [dump2.sass]: static per-path cost from the item-loop head to each stage loop's entry
  (CFG over the SASS, BRX jump tables resolved): instructions and CS2R zeroings per path, and from each loop's exit back
  to the head. Used to check what a loop-identical form actually removes from the executed path.

Codegen facts found with it (all local NVRTC 13):
- A never-executed `cp.async.bulk.global.shared::cta.bulk_group` store anywhere in the bs17-32 kernel changes the control
  bits of all 4 stage loops (text identical; 3 of 4 a pure scoreboard permutation: wb / rb / wait-mask fields only, stall
  counts identical) and re-schedules a producer loop (sim 0.80); a never-executed mbarrier bulk load global -> shared in
  the epilogue re-schedules the producer loops (sim 0.79) but not the stage loops.
- LLVM `undef` (an uninitialized local) as the value of dead accumulators is folded away by NVVM (the build = v1507's SASS);
  an empty `asm volatile("" : "=f"(x))` defines a register with no instruction and is kept.
- The bs17-32 combine's live register set after the stage loops steers the loops' allocation: dropping the 16 w4s weight
  registers (re-reading P.nw1 per round) is what makes the RO rounds loop-identical (section 2a).

## 2. Ideas (files: dev/wpcgs_mk2600-2606.py -> dev/moe_v2600-2606.py, a chain; each default build = v1507's SASS in all
16 kernels, checked with --full; v2606 contains every switch below)

### 2a. bs17-32 ready-order token combine (WP-END's MOE_END_RO) -- LOOP-IDENTICAL FORM FOUND, +0.2% at bs32, ACCEPTED

Port: dev/wpcgs_mk2600.py ports v1200's MOE_END_RO (warp 0 polls the token's pending tiles, publishes the ready mask in
scal[15] (unused in v1507), the CTA reduces the ready pairs two per round while the late split tile still runs, then polls
again) and v1202's poll knobs MOE_END_ROK (1 lane from %laneid, 2 laundered poll address, 4 precomputed lane bit, 8 want
outside warp 0's branch) onto v1507, plus new body knobs MOE_CGS_ROK (1 tid from %tid.x, 2 post-norm weights re-read from
P.nw1 per round instead of the 16 w4s registers, 4 memory clobber, 8 pair-major butterflies, 16 laundered mask, 32 laundered
pair indices, 64 rbuf as two float4, 128 laundered r, 256 accv zeroed inside the branch, 512 bottom-tested loop with the
np = 0 guards) and WP-END's PB (MOE_END_PB) for reference.
Search (moe_8_4_1_4_1; 4 stage loops 569/514/551/593, 3 producer loops):
- v1200's form (MOE_END_RO): stage 0/4, opcode sim 0.43 (as WP-END saw on v1002); PB + KN1=64: 4/4 register-renamed only.
- 400 sampled of ROK x CGS_ROK x KN1{2,16,32} x {-, FUN, LND}: 18 EXACT, all with CGS_ROK bits 2 + 512. Minimal:
  **MOE_END_RO,MOE_CGS_ROK=514** -> moe_8_4_1_4_1 c555b2934b1f (7808 instr; v1507 6d55575471cc 7768): all 4 stage + 3
  producer loops byte-identical incl. control bits, 0 spills, every other kernel = v1507's. Bit 2 alone: stage loops exact,
  one producer loop renamed; bit 512 alone: re-scheduled (0.43). Other exact forms: CGS_ROK=578 (7b38437bb43f),
  ROK=1 + CGS_ROK=707 (3bb0a5b0bcf9), ROK=14 + CGS_ROK=871 (e025606b878b).
- Control MOE_CGS_W4G (dev/wpcgs_mk2605.py: v1507's own CG = 2 rounds, no ready order, weights re-read from P.nw1 like
  RO's body): EXACT too (e391178bed7b).
- MOE_END_RO_MW=8 on the (8,8,1) kernels: 0 of 24 forms exact (sim 0.42); not pursued (WP-END: no end-chain prize there).
Correctness: check_mt 0 failures: run 1 RO + DZ M 17 18 20 23 24 25 28 31 32 33 64 65 100 127 128 129 200 255 256 257; run 2
RO / RO578 / RO707 M 1 8 16 17 18 19 21 22 26 27 29 30 32 33 (normal + skewed, 3 seeds, 2 calls); hang1 all sizes ok.
Bench (bench2 jm 100 reps x 2 passes, verified, fillers at both ends, v1507 controls in the middle; % vs the mean of the
middle controls, dev/wpit2_tab.py; bs16 = placebo, identical kernel):

| job / section | RO copies (CGS_ROK=514) | other forms / controls | v1507 middle controls | bs16 placebo |
|---|---|---|---|---|
| run 1 | +0.23* / +0.09 / +0.24*; with DZ +0.13* / +0.23* | ro0 (v1200 form) **-0.32**; tmg1 +0.03 | -0.08 / +0.15 / -0.02 / -0.05 | RO -0.06..+0.08 |
| run 2 a | +0.22* / +0.22* / +0.21* / +0.09 / +0.13 | 578: +0.05 / +0.12; 707: +0.22* / +0.12 | +0.04 / +0.21 / +0.03 / -0.23 / -0.04 | -0.22..+0.24 |
| run 2 b | +0.40* / +0.24* / +0.13* / +0.09 / +0.17* | 578: -0.06 / +0.02; 707: +0.09 / +0.19* | +0.08 / +0.16 / -0.11 / -0.13 / +0.01 | -0.22..+0.10 |
| run 3 | +0.20* / +0.28* / +0.17* / +0.25* | W4G control: -0.10 / -0.16 / -0.07 / -0.06 | +0.02 / -0.12 / +0.04 / +0.06 | -0.13..+0.10 |
(* = above every middle control of that section.)
=> RO (514) over 3 jobs / 4 sections: 17 copies, mean **+0.20%** at bs32, every copy above the control mean, 13 of 17
above every control; the code-form control W4G is -0.10% (the extra weight reads per round cost a little), so the gain is
the ready-order mechanism. The re-scheduled v1200 form (same mechanism) reads -0.32% in the same job: the loop schedule
alone is worth ~0.5% here, as WP-END measured. Geomean ~+0.035% (bs32 is one of six loads).
Banked candidate: **dev/moe_v2602.py (or any later v26xx) with MOE_END_RO,MOE_CGS_ROK=514** (only moe_8_4_1_4_1 changes).

### 2b. (8,8,1) per-path accumulator zeroing (WP-DEAD's MOE_DZ) -- LOOP-IDENTICAL FORM FOUND, no gain, REJECTED

dev/wpcgs_mk2601.py, MOE_CGS_DZ (MOE_CGS_DZ_M: 1 the XSF (8,8,1) kernels, 2 bs17-32) with form knobs MOE_CGS_DZF: the
groups an item does not touch get an OPAQUE value (an empty `asm volatile("" : "=f"(x))`, no instruction) instead of zero,
which kills the loop-carried value like v1507's zero does (WP-DEAD's forms left it loop-carried or re-zeroed it), and each
path zeroes its own groups (cdz_zero<K>) before its stage loop; bits 1 undefs at the item top, 2 one asm per 4 registers,
4 MOE_DEADACC's re-zeroing after the mma paths -> opaque defs, 8 runtime `if (b < ntl)` top loop, 16 LLVM undef, 32 warm-up
undef, 64 zeroing per case, 128 pinned zeros. Correct: no code reads a group >= ntl (XSF stores, TMP copies n tokens,
wgmma n(8K) uses groups < K, ITE skips the x64 on these kernels).
- All 256 forms on moe_8_8_1_3_4 + _3_8: 24 EXACT; non-trivial: **MOE_CGS_DZ,MOE_CGS_DZF=4** (= 6, 68, 70): moe_8_8_1_3_4
  7b9d43d5f5ca (10832 -> 10736 instr), moe_8_8_1_3_8 a38022cb23ba; all 8 + 8 stage and 2 + 2 producer loops exact, spills
  unchanged (16/40, 16/44). DZF 36 / 38 / 100 (+ warm-up undef) also exact; DZF 16 (LLVM undef) = v1507's SASS (folded).
  Form 0 (DEADACC removed) re-schedules (1/8 loops): the dead groups must be re-defined right after the mma path's loop.
- dev/wpcgs_paths.py on moe_8_8_1_3_8: item-loop head -> stage loop entry, v1507 vs DZF=4: mma 2-group path 119 -> 87
  instructions (CS2R 60 -> 28), 1-group 102 -> 70 (46 -> 14), wgmma paths 124 -> 76, 152 -> 92, 163 -> 127, 191 -> 159,
  193 -> 132, 193 -> 145: 32-61 fewer instructions on every item's path (the 32-CS2R item-top block is gone).
- bs17-32 (DZ_M=2): 0 EXACT of 256 DZF forms and 0 of 301 crossed with KN1 / FUN / LND (best 3/4 loops, sim 0.77).
Bench run 1 (5 copies incl. 2 with RO): bs128 +0.04 / -0.04 / +0.08 / +0.09 / +0.01%, bs256 -0.05 / -0.15 / -0.02 / -0.05 /
-0.07% vs controls -0.14..+0.12 / -0.09..+0.13: the removed item-top instructions do not convert (the item top is not
issue-bound). Rejected; the form stays available as MOE_CGS_DZ,MOE_CGS_DZF=4.

### 2c. bs17-32 held-slot TMG (MOE_IT2_TMG) -- NO LOOP-IDENTICAL FORM, REJECTED

dev/wpcgs_mk2602.py, MOE_CGS_TMK bits (bs17-32 kernel only): 1 one call site for the room's chunk and the reloaded chunks,
2 `#pragma unroll 1` on the bulk-store loop, 4 nn laundered, 8 v1507's direct-from-slot call kept under a runtime-false
guard (code presence), 16 scratch pointer laundered, 32 polling reload waits, 64 one bulk store when the rest fits one chunk;
x {-, MOE_WPA_TMG2}: 0 of 256 exact. Best: TMK=1 (7320 instr, -448: one inlined wpa_g1 / g2 copy fewer): stage loops
text-identical with other control bits (3 of 4 a pure scoreboard permutation), producer loops re-scheduled (sim 0.32).
Probes (scratchpad wpcgs/scr/p_bulk.py, never-executed pieces): the bulk-group store alone = the scoreboard renaming, the
bulk reload alone = the producer re-schedule; both are intrinsic to TMG (the bulk engine reads the slot, the room is
refilled by bulk loads). Exploratory bench (run 1) tmg1: bs32 +0.03% (bound +0.24% per WP-IT2). Not pursued.

### 2d. moe1t (bs1) cold-code warm-up forms (WP-COLD's EDISP / EDISP1 / PLAST / DRYF) -- NO LOOP-IDENTICAL FORM

dev/wpcgs_mk2603.py applies dev/wpcold_mk2200.py's patches unchanged to v2602 (all WP-COLD switches) plus MOE_CGS_EDK
(1 each role re-loads the ids by a volatile load after its dispatch, 2 no laundering, 4 entry prefetch.global.L1 of the ids
line, 8 non-unrolled list loop). moe1t_4 has 5 innermost MMA loops (620 + four ~1040-1110) and 8 innermost copy loops.
EDISP (all roles): stage 0/5 (sim 0.87), 2 fewer registers (SPILL-DIFF); EDISP1: 1/5 (sim 0.96), producer 0/8; DRYF: all
renamed only (sim 0.999); PLAST: 0/5 (0.32). 96 forms (EDISP / EDISP1 / EDISP1X x EDK 0-15 x PLAST): 0 exact. moe1t's
allocation reacts to any change before the role dispatch. WP-COLD's measured losses (EDISP1 -1.4%, DRYF -1.2%) were
explained by more distinct cold lines / a later dry pass, not by the schedule, so a loop-identical form would not have
been enough. Closed.

### 2e. (extra) WP-IT's MOE_IT_DB (BF16 stmatrix staging of the (8,8,1) full-k down items), shelved by the loop rule

In v1507 but never benched; WP-IT: "9 formulations re-schedule all 8 loops". On v1507 (dev/wpcgs_mk2604.py MOE_CGS_DBK
consumer forms: 1 XSF control structure, 2 laundered scale, 4 laundered address, 8 two x2 stmatrix, 16 products first,
32 volatile scale read, 64 nested in the XSF branch, 128 laundered condition; MOE_CGS_DBE generic-copy epilogue):
- Bug found: since WP-IT2's MOE_IT2_S6E, ITE (= ITD) also covers the bs17-32 kernel, where DB's layout is wrong: NaN at
  bs 17-32 (wpcgs run 3 smoke). dev/wpcgs_mk2606.py -> dev/moe_v2606.py restricts ITD to ITK (the (8,8,1) kernels).
- Codegen: consumer half alone: MW4 6 loops differ by ONE uniform operand (the wgmma descriptor's high-word constant
  `UMOV UR19, 0x40000040` -> `UMOV UR23, UR18`) + 2 register-renamed, MW8 sim 0.91; epilogue half alone (bulk copies):
  stage loops exact, producer loops sim 0.63, spills 16/40 -> 44/68 B. 0 of 256 DBK forms exact.
- Bench run 4 (DB on (8,8,1), loops / spills as above; check_mt M 9 12 16 17 32 64 65 100 128 129 200 256 257: 0 failures):
  db bs128 -0.31 / -0.43 / -0.52%, bs256 -0.56 / -0.60 / -0.61%; dbe (generic copy, 4 x 16 B per thread: spills 60/100 B)
  bs128 -0.68 / -0.64 / -0.47%, bs256 -0.44 / -0.45 / -0.32%; bs32 placebo -0.07..+0.02 (controls -0.12..+0.17). Raw DB
  loses ~0.5%.
- Narrow epilogue: dev/wpcgs_mk2607.py -> dev/moe_v2607.py MOE_CGS_DBMW (DB's MW mask) + MOE_CGS_DBEW (generic-copy width):
  epilogue-only probe with width 1 keeps every stage loop exact, producer sim 0.997, spills +4 B. DB on MW4 only
  (MOE_IT_DB,MOE_CGS_DBMW=4,MOE_CGS_DBE,MOE_CGS_DBEW=1, moe_8_8_1_3_4 af381abde5ff): 6 stage loops differ from v1507's by
  the one uniform operand, 2 register-renamed, producer 0.992, spills 20/44 B; no DBK form (256) makes it exact.
  (Side effect: any DBE code also flips moe_4_2_1_4_1's bistable producer loop, sim 0.997; DBK=64 alone avoids it.)
- Bench run 5 (check_mt M 9 12 16 65 70 80 97 100 110 127 128: 0 failures): dbn bs128 -0.33 / -0.36 / -0.49 / -0.29%
  (controls -0.08..+0.18), bs256 placebo -0.03..+0.12, bs16 -0.11..+0.21. With the codegen damage nearly gone DB still
  loses ~0.37%: the mechanism itself costs (the consumers' scale + cvt + stmatrix at the item end is on their critical
  path; the epilogue warps that save the work are not). REJECTED (not a schedule problem).

## 3. Summary (WP-CGS)

- Harness: dev/wpcgs_search.py (codegen search: knob product / sampling, reduced-module parallel NVRTC 13 compiles,
  per-loop EXACT / scoreboard-permuted / renamed / similarity classification incl. control bits, spill signature,
  --full default-build identity check, --selftest), dev/wpcgs_paths.py (static per-path instruction / CS2R cost from the
  item head to each stage loop). Throughput ~3-5 candidates / s for the bs32 kernel, ~1 / s for both (8,8,1) kernels.
- (a) bs17-32 ready-order combine: loop-identical form found by search (MOE_END_RO,MOE_CGS_ROK=514: the RO rounds read
  the post-norm weights from global instead of 16 live registers, bottom-tested ready loop); **+0.20% at bs32** over 17
  copies / 3 jobs (13 above every control), the re-scheduled v1200 form -0.32% in the same job, the code-form control
  -0.10%. check_mt 0 failures. ACCEPTED: dev/moe_v2602.py (.. v2607) with `MOE_END_RO,MOE_CGS_ROK=514`
  (moe_8_4_1_4_1 c555b2934b1f, every other kernel = v1507's). Geomean ~+0.035%.
- (b) (8,8,1) per-path zeroing: loop-identical form found (MOE_CGS_DZ,MOE_CGS_DZF=4: opaque empty-asm values for the
  dead groups; 32-61 fewer instructions per item path), bench neutral (bs128 +0.04%, bs256 -0.07% over 5 copies):
  rejected. bs17-32: no exact form.
- (c) bs17-32 TMG: no loop-identical form (bulk-group stores rename the scoreboards, bulk reloads re-schedule the producer;
  both intrinsic); exploratory TMK=1 +0.03%. Rejected.
- (d) moe1t cold-code forms (EDISP / EDISP1 / PLAST / DRYF x EDK knobs, 96 forms): none loop-identical. Closed.
- (extra) MOE_IT_DB: was broken on v1507 at bs17-32 (fixed in v2606); raw -0.3..-0.6%, near-identical MW4 form still
  -0.37%: rejected on mechanism.
- Lesson: the loop rule is necessary but not sufficient. Where the schedule was the only obstacle (RO: v1200's form
  -0.3..-0.5% from the schedule) the search converts the mechanism's prize; where the edit's own execution cost dominates
  (DZ's item top is not issue-bound, DB's item end is on the consumers' path) an exact form shows that the prize is not there.
