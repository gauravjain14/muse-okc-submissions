# P1 (round 3): batch-1 sliding d1s (xe_r1_1) + v1s (xe_b1s_8c6); B>1 sliding only where a bet applies (d16s xe_dm_8_4_2)

Owner: P1. Write only in dev/p1/. Base = dev/coord/k55.py (= BEST.py). No okc. Modal only via tools/gate.sh, tag p1_jN.

## Status / resume here
- 2026-10-29 PHASE 1 DONE (phase 2 H1/H2 done: see below) (no GPU used): dev/p1/PORTFOLIO.md = 4 bets ranked by EV per GPU minute
  (H1 cluster dims, H2 K-before-V, H3 compute-path pre-execution, H4 v1s wgmma rewrite).
- 08:50 FUNDED H1 + H2 (+ H1b): built and SASS-gated locally, job p1_j1 launched (bash dev/p1/j1.sh; output
  dev/p1/out_j1.txt). Builder dev/p1/build.py (all options 0 = k55 byte for byte, checked with cmp).
  Variant files (md5): k55a / k55b = dev/coord/k55.py f0bff1b7; k55cd 582e7931 (CD=1); k55kf 49bb9959 (KF=1 KFP=1);
  k55cdkf 7b2f6c85 (CD KF KFP); k55sm 761b6c05 (CD KF KFP SM).
  SASS gate results (NVRTC 13): CD removes LDC c[0x0][0x140] + branches in xe_r1_1 / xe_b1s_8c6 / xe_dm_8_4_2.
  KF with .nc loads FAILS (ptxas hoists BAR.SYNC 6 above every load and still issues V first). KF with plain
  ld.global (KFP=1) PASSES: every K LDG.128, then BAR, then V LDG.64, in all three kernels. No new LDL/STL;
  instruction counts xe_r1_1 1248 -> 1240, xe_b1s_8c6 1920 -> 1912 (CD+KF).
  NOTE: k55's own XE_KFIRST=1 (`__syncwarp()` fence) was ineffective. SASS shows V first.
- p1_j1 RESULT (out_j1.txt, ~2.5 min wall): t_xe --quick 0 failures (cdkf, sm). Placement sm: 1 CTA/SM in 20/20 reps
  (d1s 32 CTAs, v1s 96). nsys medians, 2 passes; controls k55a / k55b (4 runs):
      shape  ctl    | cd (H1)        | kf (H2, plain) | cdkf           | sm (H1b on cdkf)
      d1s    3.352  | 3.312 (-0.040) | 3.376 (+0.024) | 3.360 (+0.008) | 3.360 (+0.008)
      v1s    4.200  | 4.176 (-0.024) | 4.288 (+0.088) | 4.256 (+0.056) | 4.288 (+0.088)
      d16s   6.528  | 6.160 (-0.368) | 6.488 (-0.040) | 6.032 (-0.496) | 6.040 (-0.488)
  (means agree: cd -0.039 / -0.043 / -0.395, cdkf d16s -0.503). => H1 WIN (d16s big: xe_dm's start path, 256 CTAs; b1 small);
  H2 LOSES on v1s (+0.09) and d1s (~+0.02): K-first through a barrier + plain loads does not land K earlier on b1, but
  helps d16s on top of CD (-0.13). H1b (32 KB smem + SPREAD) = no gain (smem padding costs nothing) -> dropped.
- FROZEN dev/p1/deliver_r1/: build.py (987dee6f; no options = CD=1 KFM=1 KFP=1) -> k55p1.py (22361eb9) from
  dev/coord/k55.py (f0bff1b7). Per-kernel SASS identity (local NVRTC 13): xe_r1_1 / xe_b1s_8c6 == the measured cd
  kernels, xe_dm_8_4_2 == the measured cdkf kernel. Expected: d1s -0.04, v1s -0.02, d16s -0.50 nsys.
- p1_j2 RESULT (out_j2.txt, ~2.5 min wall): tools/check.py 56 PASS / 0 failures; full t_xe 0 failures (worst err/bound
  0.543 at B 33 = base kernel). nsys 2 copies x 2 passes vs k55 x2: d1s 3.412 vs 3.424 (-0.012), v1s 4.224 vs 4.224
  (0.000), d16s 6.072 vs 6.528 (-0.456; means 6.106 vs 6.582, -0.476). => deliver_r1 = a d16s win (~+0.2% score at
  0.45 %/us); the b1 effect of CD is noise-level (j1 -0.04 / -0.02, j2 -0.01 / 0.00).
- GPU used: j1 + j2 ~5 min total (each nsys run ~7 s: one process per file, all shapes).
- 09:00 FUNDED (coordinator): H3 dry pass (cap 10 GPU-min) + H4's kill probe only. Both on top of deliver_r1.
  H3 = build.py PW1=1 (xer1 / d1s) PW2=1 (xer2 / v1s): the post-landing body (QK .. reader) runs in a `#pragma unroll 1`
  2-pass loop on zero operand registers Ko / Vo; SEL copies K -> Ko / V -> Vo at the end of pass 0; pushes `if (xp)`.
  SASS gate PASSED: HMMA counts unchanged (18 / 27: body once), no new LDL/STL, loop label after BAR 3, pass-0 body
  reads Ko (R77-R92 d1s / R100+ v1s), not the K load targets (R32-R47 / R44-R67), and 32 / 48 SELs end the loop body.
  k55pw.py (= k55pwb.py) = CD=1 KFM=1 KFP=1 PW1=1 PW2=1.
  H4 probe dev/p1/probe_h4.py: 2 x C16, 128 threads, after a cluster barrier 64-row FP16 pushes to 16 owners x 4 rows,
  4 owner warps fold 16 sources; clock64 barrier -> arrived -> folded; D1S calibration mode (8 rows FP32, 8 owners).
  Job p1_j3 = t_xe quick + nsys (k55p1 x2 vs k55pw x2, 2 passes) + H4 probe + check.py / full t_xe on k55pw.
- p1_j3 RESULT (out_j3.txt, ~2.5 min wall): k55pw correct (t_xe quick + full 0 failures, check.py 56 PASS) but
  H3 LOSES BADLY. nsys 2 copies x 2 passes vs k55p1 x2: d1s 3.768 vs 3.368 (+0.40), v1s 5.000 vs 4.176 (+0.82),
  d16s 5.960 vs 5.956 (unchanged code). The dry pass is not hidden: pass 1 is no faster, so the post-landing code is not
  i-fetch-bound, and pass 0 runs past the landing. H3 CLOSED (builder options PW1 / PW2 kept as a record, default off).
  H4 probe (clock64 at 1.9 GHz, owner side, 190 reps, flush / warm identical):
    D1S calibration (8 rows FP32 -> 8 owners): all bytes arrived 614 cyc, fold 571 -> push -> folded 0.62 us
    H4 (64 rows FP16 -> 16 owners x 4 rows):    all bytes arrived 1913 cyc, fold 618 -> push -> folded 1.33 us
  -> fails the <= 0.30 us rule (DSMEM receive of 8.7 KB per owner ~1.0 us). H4 CLOSED. No deliver_r2.
- GPU used by P1 in total: j1 + j2 + j3 ~7.5 min.
- Resume here (09:05): deliver_r1 is the only deliverable. H2-b1, H3 and H4 are closed. Nothing is open in P1's portfolio.
- Resume here: deliver_r1 is frozen and handed back. H3 (compute-path pre-execution) / H4 are not started and wait for
  funding. H2 on b1 is closed: forced K-first (bar.sync + plain loads; SASS-verified) is +0.09 on v1s and +0.02 on d1s.

## Files
- build.py (phase 2), j1.sh (job 1), nsys_multi.py + nsys_go.py (one process per file, all shapes, NVTX ranges per shape,
  durations parsed per range), place.py (CTA-per-SM histogram from trace slot 31), t_xe.py (copy of dev/xe/t_xe.py).
- sass.py: local SASS tool (no GPU). `<venv>/bin/python dev/p1/sass.py <file.py> <SRC_VAR> <outdir> [-DX=Y]`.
  venv = <scratch>/nvrtc13
  (NVRTC 13.0 = the judge image's). Writes <outdir>/<kernel>.s and prints regs/spills and instruction counts.
  The k34 monolith (SRC) takes several minutes to compile locally.
- PORTFOLIO.md: phase-1 deliverable.

## SASS facts (k55 XE_SRC, NVRTC 13.0, sm_90a; scratch outputs in <scratchpad>/p1xe, p1cd)
- xe_r1_1: 1248 instr (19.97 KB), 82 regs. xe_b1s_8c6: 1920 instr, 92 regs. xe_dm_8_4_2: 1472 instr, 89 regs.
  Every one has 32 B of stack (sincosf) and 0 spills.
- Compute-warp entry, xe_r1_1: 13 ALU, `@P0 BRA` (prep warps leave), `LDC R2, c[0x0][0x140]` + ISETP + BRA +
  UCGABAR_ARV + BRA (runtime cluster check of barrier.cluster.arrive), then param ULDC/LDC c[0x0][0x210..0x288], seq LDG
  at 0x02b0, page-table LDG at 0x0350 (instr 54). xe_b1s_8c6 and xe_dm_8_4_2 have the same pattern.
- With `__cluster_dims__(16,1,1)` / `(6,1,1)` (test file <scratchpad>/p1_cd.py): no c[0x0][0x140] and no branches;
  UCGABAR_ARV is inline at 0x0180; the param ULDC is at instr 15; pt LDG at 0x0310; xe_r1_1 1232 instr, xe_b1s_8c6 1912.
  The cl_wait check also disappears (it was off the critical path). The prep warp's arrive moves after its loads.
- Load order: V first, K last, in every XE kernel.
  - xe_r1_1: 8 x LDG.64 (V) at 0x06f0..0x0ad0, then 4 x LDG.128 (K) at 0x0b60..0x0ba0.
  - xe_b1s_8c6: 12 x LDG.64 at 0x08a0..0x0e70, then 6 x LDG.128 at 0x0f10..0x10d0.
  - xe_dm_8_4_2: pass 0 is 8 x LDG.64 then 4 x LDG.128; pass 1's K (4 x LDG.128) is hoisted; pass 1's V comes after QK0.
- Post-landing straight-line code on the compute warps:
  - xe_r1_1 from the first K cvt (~0x0f00) to EXIT 0x1f00: ~256 instructions, 4 KB.
  - xe_b1s_8c6 from 0x13e0 (first F2FP of K) to EXIT 0x2ff0: 449 instructions, 7.2 KB, with 165 F2FP/HADD2 in the kernel.
- CCTL.IVALL follows UCGABAR_WAIT (compute warps, off the critical path) and the owner's try_wait.acquire.cluster
  (XE_OWS tested it: neutral).
- k34 monolith B > 1 sliding kernels (attn_8_1_1_1x6 v16s, attn_1_1_1_1e3 d32s, attn_8_1_1_1w4 v32s) issue their first
  LDG at instructions 11 / 97 / 20 with no c[0x140] check, so H1 does not apply to them.

## Evidence used (not re-measured)
- d1s trace (XE job8 / job27): pt issued 0.22-0.29, landed ~0.6, K/V issued 0.85, K landed 1.44 / 1.63 / 1.82
  (min / med / max), PV 1.98, merge barrier 2.11, reader 2.24, pushed 2.30, owner recv 2.63, out ~3.05; nsys 3.39.
- v1s C6 trace (out/xe/job39.txt, buggy-output build, timing representative): see PORTFOLIO section 0. nsys 4.256-4.288.
- XE job40: entry -> first param consumed 187 warm / 216 cycles after the flush; first global load 820 cycles (DRAM).
- XE job14 probe: d1s K-only lands at 1.09, V-only 1.12, K+V 1.25.
- XE job5: reader 0.28 cold / 0.21 warm; owner first LDS block 0.27 cold / 0.02 warm (-> PREW, d1s -0.12).
- XE job26 diags: removing every math / merge phase leaves d1s at 2.94 nsys. The C4 probe graph is 1.79, about 2.45
  nsys-equivalent.
- dev/xd: DSMEM moves ~18-30 GB/s per SM (push or pull). WE (runE1 comment): whole 128-B rows gather 6-13% faster than
  64-B halves at B > 1.
- XE job11 (generalized kernel, bench): LATEARV (arrive after the pt load) v1s -0.13, d1s ~0. It was later turned off
  because the constant-dependent if/else made ptxas tail-duplicate the loop (v9/v10).

## Phase 2 plans (build only what the coordinator funds)
### H1 + H2 (one job, build first)
- Builder dev/p1/build.py <base> <out> [XP1_CD=0/1] [XP1_KF=0/1] patches the XE section of the base file. Defaults
  reproduce k55's SASS byte for byte.
  - CD: `__cluster_dims__(16,1,1)` on xe_r1_1, `(6,1,1)` on xe_b1s_8c6, `(8,1,1)` on xe_dm_8_4_2 (via the XE_DM macro).
    Host `_XeMod._cfg`: numAttrs = 0 for these kernels. Check that cuOccupancyMaxActiveClusters still works with
    compile-time dims; if not, keep the attribute with the same value.
  - KF: in each compute-warp issue block, K loads stay in the 2-pass speculation loop; V loads move after `cl_wait()`
    as plain `ld.global` (no `.nc`, so ptxas cannot hoist them above the acquire) from the verified range (lo, base, cnt
    after the loop). The V re-issue on a miss is no longer needed. xem: the same for pass 0 only.
    Fallback: a named bar.sync among the compute warps between K and V.
  - SASS gate (local, before Modal): CD has no c[0x0][0x140] before the first LDG. KF has every K LDG.128 before the
    first V load, no MOV/SEL of loaded registers, no new spills, and regs <= 120.
- Job p1_j1 (~12-14 min, every variant under `timeout 120`):
  - t_xe --quick on CD, KF and CD+KF (b1 + --bonly for B > 1).
  - nsys via dev/xe/nsys_run.py (mention dev/wp and dev/coord in --cmd), 2 passes, files k55a, k55b (copy), CD, KF,
    CD+KF; shapes '1,1,sliding;1,8,sliding;16,1,sliding'.
  - Optional H1b: XE_SMEM=20480 + CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE=SPREAD, with a smid trace
    (trace slot 31) to prove 1 CTA/SM.
  - Rule: freeze any variant with a mean nsys gain >= 0.03 us on some shape and no shape >= 0.03 worse. Kill an arm if
    every shape is within +-0.02. For a freeze candidate: tools/check.py (grep -a "0 failures") + full t_xe.
  - Deliverable: dev/p1/deliver_r1/ (build.py + the built file + md5s).
### H3 (after H1/H2 freeze, on top of the winner)
- XP1_PW=1: `#pragma unroll 1` dp loop around [Q LDS .. push]. Operand registers Kop/Vop: zero in pass 0; copied from
  the K/V load targets at the top of pass 1 through a uniform branch (asm `@p bra.uni`, so ptxas cannot if-convert it to
  SEL). st.async predicated on dp. bar.sync(1) runs in both passes.
- SASS gate: nothing in pass 0 reads a K/V load-target register (otherwise pass 0 stalls until landing), no tail
  duplication, regs <= ~110.
- Job: d1s only first (t_xe quick + nsys 2x2 vs controls). Rule: keep if >= 0.05; port to v1s (pass 0 after Q-ready) if
  >= 0.08; kill if <= 0.02.
### H4 (only if funded; kill probe first)
- Extend dev/xe/probe.py (copy into dev/p1/) with a C-stage variant: 2 x C16, each CTA pushes 64 FP16 rows (136 B each)
  to 16 owners (4 rows each), and 4 owner warps fold 16 sources. Stamps: merge -> pushed -> owner folded.
- Rule: stop unless push -> folded <= 0.30 us. Next, a timing-only wgmma skeleton (borrow wgmma / descriptor helpers
  from _XD_SRC) must beat C6's nsys by >= 0.4 us.

## Dead ends I relied on (do not retry): see dev/xe/NOTES.md
REF, EXH, D8, DPT, TW, RDX, XH, OWS, RML, V1T (co-residency), D1C8, bulk push, H16, 2-CTA/SM, MMA-free, WS_INC
(16-mbarrier online fold), T2 (warp pushes), L2::128B / evict_first (WF, full layers), TLB touching, carveout pref 100.
