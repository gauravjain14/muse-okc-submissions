# EC: end chain of bs1 (moe1t) and bs8 / bs16 / bs32 -- worker notes (2026-11-03)

## Status (top) -- CLOSED ~11:55 PT
- BANKED (bs1): **t3/EC/moe_ec_13.py** = t3/INT/moe_i1.py + `python3 t3/EC/mk_ec_final3.py BASE OUT` (EC_OF + EC_UP1 +
  EC_R4 baked on, escape EC_NO_B1END = moe_i1's moe1t SASS). Only moe1t_* change; all 8 main kernels and moe1_* are
  byte-identical to moe_i1 (local NVRTC 13, with and without MOE_S5R_FIT). moe1t_4 sha 0b459162f760, 2256 instr,
  116 regs, 0 spills, consumer stage loop schedule-identical to moe_i1 (sched a1d08bdfeb, registers renamed).
  - bs1 vs moe_i1: job 13 (box 1) +2.79% RoM, pairs +1.66..+2.39; job 14 (box 0, all-shape recipe) +1.96% RoM, pairs
    +1.73..+2.63; every new copy above every base copy in both. vs v4111 +1.93% (pairs +2.22..+2.71). Other shapes
    placebo (identical kernels): -0.13..+0.15%. Projection (v1504 anchor, job 14): 5084.2 / 5080.4 / 5086.8 vs moe_i1
    5098.3 / 5105.4 / 5099.5 (-17 us, ~-0.34% geomean); bs1 judge-scale ratio 1.035 vs 1.015.
  - Correctness: outputs bit-identical to moe_i1 (60/60 bs1 routings, also the escape build); hang1 1..512 (16 sizes)
    ok; check_mt 1 2 4 8 9 16 17 32 33 64 65 128 129 200 256 257 512: FAILURES 0.
  - Mechanism (trace, cold, 16 seeds x 2): seen -> stored 0.384 -> 0.064 us per CTA; slowest red -> first CTA seen
    0.384 -> 0.256; slowest red -> last store 0.976 -> 0.512 us.
- Earlier banked forms (superseded, kept): moe_ec_12.py (EC_OF + EC_OFS2 + EC_R4: +1.03 / +1.60 / +1.72%, 3 jobs) and
  moe_ec_8.py (EC_OF + EC_K1 + EC_R4: +0.85 / +0.72 / +1.20 / +1.39%, 4 jobs); moe_ec_13 beats moe_ec_12 by +0.97% (every
  copy) in job 14.
- bs8 / bs16 / bs32: nothing banked. The analogous fix in the (4,*,1) fast units (EC_UP8, warp-uniform tile poll; loops
  exact) is neutral (bs8 +0.18, bs16 0.00%, job 13). Bounds and why below (section 6).

## Files (t3/EC/)
- mk_ec_final3.py BASE OUT -> moe_ec_13.py (FINAL). mk_ec_final2.py -> moe_ec_12.py, mk_ec_final.py -> moe_ec_8.py.
- Probe chain (each default build = moe_i1's moe1t SASS): mk_ec1 (EC_OF, EC_R4, EC_PAD=N) -> moe_ec_1; mk_ec2 (EC_FDLY=N,
  EC_NOWAIT) -> moe_ec_2; mk_ec3 (EC_XK=K replicated FXSS words) -> moe_ec_3; mk_ec4 (EC_TR per-CTA stamps + %smid) ->
  moe_ec_4; mk_ec5 (EC_DRY end-block dry pass) -> moe_ec_5; mk_ec6 (EC_TR2 dependency-forced stamps) -> moe_ec_6;
  mk_ec7 (EC_K1, EC_K2) -> moe_ec_7; mk_ec9 (EC_OFS) -> moe_ec_9; mk_ec10 (EC_OFS2) -> moe_ec_10; mk_ec11 (EC_UP1,
  EC_UP8) -> moe_ec_11 (contains every switch).
- ec_trace.py TRACED.py DEFS OTHER.py out.pt [reps] [nseeds] [cold|warm]: judge-like replays (other build -> 512 MB flush
  -> traced), per-CTA end stamps with %smid, summary (seen -> store, exchange, imbalance, slow-SM exclusion bound).
- job1..job15 .sh / .log; ec_job1.json, ec_job13.json, ec_job14.json (bench2), ec_tr_box1.pt (trace dump).

## 1. bs1 end chain, decomposition (EC_TR trace on moe_i1's code, both boxes agree; us after the slowest CTA's red)
| piece | base | final (moe_ec_13) |
|---|---|---|
| slowest red - median red (CTA imbalance) | 0.29-0.37 (p90 0.7-1.0) | same |
| slowest red -> first CTA sees all words | 0.384 | 0.256 |
| -> median CTA seen | 0.448-0.480 | 0.352 |
| seen -> output stored (every CTA) | 0.384 | 0.064 |
| slowest red -> last store | 0.96-0.98 | 0.512 |
Bounds measured with bench2 (job 2): post-poll delay +500 ns -> +0.52 us (1:1, EC_FDLY twins, same layout); no wait for the
other CTAs' sums (EC_NOWAIT, timing only) -0.55 us = +3.2% (pairs +1.94..+2.63) = the whole exchange part of the chain.
- Slow SMs (trace, 72 replays per box): CTAs 8-23 sit on smids 14/15, 30/31, 46/47, 62/63, 76/77, 90/91, 104/105, 118,
  122 (blk -> smid map identical on both boxes); they average +0.12..+0.23 us late, but the slowest red moves between
  CTAs: removing the 4 / 8 / 12 slowest SMs (in sample and out of sample) moves the slowest red by only 0.01-0.06 us.
  Static rebalancing of moe1t's work by %smid: bound ~0.05 us, dead.
- Hot-line contention of the FXSS word: EC_XK=4 (4 copies, each CTA reds into all, polls one; 32 pollers per line):
  +0.00% vs its layout twin (job 3): not contention.
- Instruction fetch of the end block: EC_DRY (the end block runs once dry before the last expert, one code copy, SASS
  checked) leaves seen -> stored at 0.384 us; a warm previous replay also leaves it at 0.384: not i-fetch.

## 2. What the 0.38 us after the poll was (jobs 5-8, 11, 13, 15)
- The base's final combine is 8 serial branch-separated blocks {SHFL x2, FFMA, FSETP, MUFU.RSQ, 3 FMUL, FFMA} after the
  poll (~0.14-0.2 us of issue at ne 3); EC_OF (lane q computes f_q right after its poll, then 8 SHFL + an FFMA chain,
  ~0.06 us) measured SLOWER: seen -> stored 0.48 us, bench -1.0% (job 6, every copy below every base copy).
- Dependency-forced stamps (EC_TR2) and EC_K1 (a shared load + never-taken branch on it after the poll) made EC_OF
  convert: K1 + OF +1.45% / +0.85% (jobs 8 / 9); K1 alone (base combine) -0.12%.
- Shuffle-free forms: EC_OFS (factors via shared memory after the poll) +2.38 / +0.99%; EC_OFS2 (the polling lane stores
  its factor inside the poll branch) +1.76 / +1.03 / +1.60%, trace seen -> stored 0.096 us.
- EC_UP1 (the poll as ONE warp-uniform loop, exit by __all_sync, every lane loads; lanes >= ne load word 0) with EC_OF:
  +2.79 / +1.96%, seen -> stored 0.064 us, and the words are seen 0.13 us sooner.
  Reading: the cost sits in the lane-divergent spin loop `if (lane < ne) do { ld } while (...)` and what follows it
  (in the SASS the poll is BSSY / LDG.STRONG / YIELD / BRA loop / BSYNC, then BRA.DIV-guarded shuffles whose fallback
  WARPSYNC.COLLECTIVE code sits at the end of the function); with a warp-uniform loop the warp never diverges and
  both the observation and the post-poll code run at issue speed. Exact hardware cause not isolated; the forms above
  are the measured evidence.
- EC_R4 (last red's chain: post-norm weight SHFL precomputed, 4-level butterfly): ~0.03 us by count, not separately
  significant (OF+K1 with / without R4: +1.45 / +1.17%).

## 3. Placement / layout noise at bs1 (useful for every bs1 A/B)
- Identical moe_i1 copies (separate module + state instances) differ by up to 0.23 us (1%) in one job (job 1) and
  0.1-0.15 us typically. EC_PAD=40 (40 never-executed STGs after the output store, consumer loop moved +46 instr) read
  +0.50 / +0.54 / -0.03% in three jobs: noise. Use >= 3 copies per build for bs1 effects below 1%.
- moe1t_4 sits after every main kernel in the cubin (.text sections, 128-B aligned, reverse definition order), so main-
  kernel edits shift its address; moe1t edits shift only moe1t_3 and moe1_*.

## 4. GPU jobs (all bench2 jm B2_SHARE_EQ, fillers fA / fB = moe_i1, 100 reps bs1 unless noted; % = ratio of means)
| job (box) | content | result |
|---|---|---|
| 1 (0) | OF+R4 +-PAD76, PAD40, v4111 | b +0.03, c -0.18, e +0.50, v4111 -0.48 |
| 2 (0) | 21 variants OOM; rerun 14: FDLY 1/500, NOWAIT, b/e/g | FDLY slope 1:1 (+0.52 us), NOWAIT +3.19%, b -0.13, e +0.54, g +0.77 |
| 3 (0) | XK=4 / XK=2 at the PAD40 layout | XK4 +0.00 vs e, XK2 +0.46 (noise), e -0.03 vs a |
| 4 (0+1) | EC_TR trace, smid | section 1 |
| 5 (1) | trace cold / warm, base / OF | seen->stored 0.384 / 0.48 (both modes) |
| 6 (1) | EC_DRY, DRY+OF; trace | DRY -0.57%, DRY+OF -0.99% (every copy below), trace unchanged |
| 7 (1) | EC_TR2 trace | base o-compute 0.256 us, OF 0.064 us |
| 8 (1) | TR2 builds, OF+K1(+K2) | OF+R4+K1 +1.45%, +K2 +1.41% (every copy above) |
| 9 (0) | moe_ec_8 x3, K1 alone, OF+K1 no R4; check_mt | +0.85% (every copy), K1 alone -0.12, OF+K1 +1.17; check_mt 0 fail |
| 10 (0) | all shapes 60 reps: moe_ec_8 x3, OFS x2 | ec8 bs1 +0.72%, OFS +2.38%; other shapes +-0.15 |
| 11 (1) | OFS, OFS2, OFS+K1, ec8 | +0.99 / +1.76 / +1.70 / +1.39% (all every copy) |
| 12 (0) | all shapes: moe_ec_12 x3, ec8 x2; check_mt | ec12 +1.03%, ec8 +1.20%; check_mt 0 fail |
| 13 (1) | UP1 (+OF+R4), UP8, ec12; bs1/8/16 | UP1 +2.79%, ec12 +1.72%; UP8 bs8 +0.18 bs16 0.00 |
| 14 (0) | all shapes: moe_ec_13 x3, ec12 x2, v4111 | ec13 bs1 +1.96% (every copy; +0.97% over ec12); check_mt 0 fail |
| 15 (1) | trace final form | section 1 table |

## 5. SASS notes (local NVRTC 13 = judge toolchain)
- moe_ec_13 moe1t_4: 2256 instr (moe_i1 2488), 116 regs, no stack, stage loop 642 instr schedule-identical (moved from
  instr 1173 to 1088). End block: 8 LDS of the y rows, then the uniform poll loop (LDG.E.64.STRONG.GPU, SHF, ISETP,
  VOTE.ALL, BRA: no BSSY/BSYNC, no YIELD), then the per-lane factor as PREDICATED code (@!P1 I2F.U64 / FMUL / FFMA /
  MUFU.RSQ / FMUL, no branch), 8 SHFL.IDX, 8 predicated FFMA, F2FP, STG. (moe_i1: BSSY + divergent loop with YIELD +
  BSYNC, then BRA.DIV-guarded serial blocks.)
- Main kernels: byte-identical to moe_i1 (8/8 sha). The (4,*,1) EC_UP8 form: stage loops exact, producer loops exact or
  register-renamed (not banked: neutral).

## 6. bs8 / bs16 / bs32 end chain: bounds and verdict (no code banked)
Measured on today's base by ST (lean stamps): K - latest last item 2.69 / 2.80 / 2.56 us; NO_COMBINE -3.0 / -2.4 / -1.2%.
- Fast-unit SASS (moe_4_1_1, after warp 0's poll): CCTL.IVALL, WARPSYNC.ALL, BAR.SYNC, then ~74 instructions of address
  arithmetic (16 LDC.64 of kernel params) before the first data load (~0.05-0.08 us), 8 slots x 5-level interleaved
  butterflies (~0.1 us), 8 serial branch-guarded rsqrt blocks (~50-60 cycles per live slot), then the split part
  (16 KB fp32 row per split slot per unit, L2->SM bandwidth/latency bound: +0.64 us when split slots are present, 7/10
  seeds at bs8). Removable without a cross-CTA exchange: ~0.15-0.25 us at bs8/16 (0.25-0.4%), each change in the
  (4,*,1) combine region at risk of re-scheduling the stage loops.
- The bs1 lesson (lane-divergent spin loop) does not transfer: the fast units' poll is followed by a CTA barrier;
  EC_UP8 (uniform poll, loops exact) bs8 +0.18 / bs16 0.00%.
- "Last arriver combines": replaces signaler wake (~0.19) + red.release (~0.38) + poll (~0.45) by fence (~0.35) + atom
  round trip (~0.4, at the very end the ring has drained so this is the unloaded cost: the bs1 trace shows ~0.26-0.38 us
  red -> seen at the end) and keeps the same data loads; per-token completion needs a second atomic or per-token
  counters, and the combine moves onto the 2 epilogue warps or needs a wake of the consumers. Net bound ~0.2-0.3 us
  at bs8 (0.3-0.5%), below the 1% goal; the split part (the bs8 critical unit in 7/10 seeds) is unchanged by it. Not built.
- Split-row SSQ alternatives (quarter-SSQ exchange between the 4 units, last-piece SSQ, token-wide units): each needs a
  cross-CTA hop (~0.45 us) or more bytes per SM than the 12 KB per split slot it saves (~0.26 us): <= 0.
- Leads for others (not on the end chain): BRA.DIV-guarded __syncwarp / vote after lane-divergent readiness polls in the
  main producers (moe_4_1_1 SASS 4179 / 4217, moe_8_4_1 5369 / 5386 / 5491); per-item, so warm code; untested.
