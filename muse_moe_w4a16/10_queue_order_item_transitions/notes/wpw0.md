# WP-W0: the consumers' (warp 0's) item-end path in all main kernels (2026-11-01)

Worker WP-W0. Base dev/moe_v1510.py (best build, projected ~5119 us). Namespace v3100-v3199, defines MOE_W0_*,
scripts dev/wpw0_*, logs kda_ws/runs/wpw0_runN.log. Brief: map warp 0's (and the other consumers') item-end
instruction path (last stage done -> next item's first stage) per kernel, remove / move instructions (WP-SEPI32:
~14 never-blocking try_wait instructions cost ~0.5% at bs32), keep every stage loop and producer loop byte-exact.

## Files
- dev/wpw0_mk3100.py -> dev/moe_v3100.py: v1510 + MOE_W0_* switches (all off; default build = v1510 SASS in all 16
  kernels, scratchpad wpend/sass13.sh, re-checked after every patcher change). Switch list in the patcher docstring.
- dev/wpw0_map.py: per-path instruction map (role: all-lane / lane-0 / warp0-lane0 / never-executed (@!PT); kind) of a
  list of SASS address ranges (the executed path of a 1-group item, branch decisions applied by hand).
- dev/wpw0_ts.py: consumer item-end / item-head timeline from MOE_W0_TS clock64 stamps (warps 0 and 7, lane 0;
  CUDA-graph replay after a 512 MB flush; the stamps need the enlarged ctr the script allocates: never run a
  MOE_W0_TS build under hang1 / check_mt, its stamps would write past the normal ctr).
- dev/wpw0_run1.sh .. dev/wpw0_run5.sh; logs kda_ws/runs/wpw0_run1.log, wpw0_run2b.log (wpw0_run2a.log: the first
  attempt, stopped at the smoke because it ran the TS build under hang1), wpw0_run3.log .. wpw0_run5.log.
- Loop identity: dev/wpcgs_search.py (base v1510 + MOE_S5R_FIT). Note for the (4,*,1) kernels: of the harness's 5
  "producer loops" only the 244 / 252-instruction one is the producer's copy loop; the four 1275-1480-instruction ones
  are pseudo-loops closed by the out-of-line try_wait retry stubs at the end of the function (backward branches from
  the stubs into the epilogue / consumer code), so any change of a wait's registers "changes" them. Accepted forms
  below have every stage loop and every real producer loop byte-identical incl. control bits (hash hc checked).

## 1. Item-end / item-top instruction map (v1510, local NVRTC 13; dominant 1-group item, dev/wpw0_map.py)

Roles: lane-0 / warp0-lane0 instructions are predicated, so EVERY warp of the 8 issues them (predicated off); only
branch-skipped code is not issued (the (8,8,1) MW8 queue arrival is a divergent branch block, BSSY / BSYNC).

| kernel (shape) | item end (last stage -> back to head) | item top (head -> stage loop entry) |
|---|---|---|
| moe_4_1_1_4_1 (bs8, SEPI) | 48: x64 FMUL 8, slot-release arrive (lane 0) + 64-bit generic address math, buffer-free try_wait (SYNCS + BRA, parity / address 9 ALU), staging STS 8, record STS.128 x3 (warp 0 lane 0), queue arrive (lane 0); 35 all-lane, 9 lane-0, 4 warp0-lane0 | 33: try_wait (64-bit address), record LDS.128 x3, CS2R 4, dispatch; then the peeled first stage |
| moe_4_2_1_4_1 (bs16, SEPI) | 54: x64 FMUL 16 (8 of them dead for 1-group items), 4 dead predicated STS + 8 STS, release / try_wait / arrive as bs8, record: S2R CgaCtaId -> MOV -> LEA -> STS.128 x3 (warp 0 only latency chain); 39 / 8 / 7 | 36: S2UR CgaCtaId, try_wait, LDS x7 (record + constants), CS2R 8 |
| moe_8_4_1_4_1 (bs32, XS 1-group) | 48: S2R TID -> 6-op staging address, XS decision (bar.sync predicated off), STS 8, MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S, aq_slot store (warp 0 lane 0: 5 instr + warp/lane predicate 2), queue arrive with 64-bit generic address (5); 37 / 6 / 5 | 61: S2R CgaCtaId, try_wait, record LDS, CS2R 20, S2R TID + 6-op stage offsets |
| moe_8_8_1_3_4 (bs128, XS 1-group) | 53: S2R TID + address, STS 8, 4 x `@!PT LDS RZ,[RZ]` (never executed, issued), MEMBAR + FENCE, aq_slot (uniform q % 3), arrive (64-bit uniform address); uniform 14 | ~100 (as MW8) |
| moe_8_8_1_3_8 (bs256, XS 1-group) | 55 + 8 (lane-0 arrive block, divergent): as MW4 plus q % 3 by IMAD.HI chain for aq_slot (warp 0 lane 0: 6 instr) and a second uniform multiply-high chain (8 uniform ops) for the arrival address inside BSSY / BSYNC | 102: try_wait, record, R2UR dispatch, LDL spill reload, CS2R 46, S2R TID + offsets, uniform 29 |

## 2. Where warp 0's item transition goes (dev/wpw0_ts.py, v3100 default + MOE_W0_TS, 2 jm seeds x 2 reps)

Segments (cycles, medians; each includes one stamp ~= the clock read + an STG, so absolute values are inflated by
~50-100 per segment; compare warps / builds): e01 item end start -> after the slot release (SEPI: x64 FMULs + release)
/ before the stores (AEPI), e12 SEPI buffer wait / AEPI staging stores, e23 SEPI stores + record / AEPI fence, e34
queue arrival (AEPI incl. aq_slot), h loop back + next head's first-stage try_wait, d record + dispatch + zeroing.
lag = warp 0's item end start - warp 7's; rdy = warp 7's next first-stage data-ready - warp 0's item end start.

| shape, class | warp 0: e01 e12 e23 e34 h d (tot) | warp 7: same (tot) | lag | rdy |
|---|---|---|---|---|
| bs8 G1n1->G1n1 | 124 181 165 124 242 233 (1138) | 119 160 125 97 176 229 (935) | 831 | -106 |
| bs16 G1n1->G1n1 | 127 151 176 96 171 259 (1008) | 123 143 157 88 156 241 (940) | 380 | 354 |
| bs32 G2n1->G2n1 | 119 273 150 170 285 240 (1286) | 108 197 117 122 240 235 (1058) | 914 | -6 |
| bs128 G2n1->G2n1 | 112 224 122 147 205 272 (1133) | 107 209 118 138 196 248 (1054) | 328 | 474 |
| bs256 G2n1->G2n1 | 111 220 122 227 200 348 (1248) | 106 204 117 218 197 300 (1190) | 304 | 565 |
| bs256 + MOE_W0_CS | 109 219 112 184 198 344 (1184) | | 296 | 540 |
| bs16 + RA | 129 152 141 96 166 246 (963) | | 262 | 462 |

- Warp 0 ends every item 300-900 cycles after warp 7 (its stage compute shares SMSP0 with the producer) and then
  spends ~1000-1300 cycles (stamps included) until its next first stage; the next item's data was there earlier
  (rdy -106..565). The per-warp boundary trace (dev/wpit_edge.py on v1510, run 1) agrees: warp 0 is the slowest warp
  at 86-99% of the 1-group boundaries; its own transition 626 (bs8) / 714 (bs16) / 820-874 (bs32) / 860-966 (bs128) /
  976-1078 (bs256) cycles vs first-warp data-ready -1127 / -586 / -108..87 / 266-614 / 438-756; boundary excess per
  CTA 1.2% (bs8), 3.1% (bs16), 4.0% (bs32), 4.2% (bs128), 4.8% (bs256) of the span.
- The (8,8,1) MW8 queue arrival (e34 220 cycles vs MW4's 140) is the q % 3 multiply-high chains; MOE_W0_CS takes it
  to 184.

## 3. Candidates (dev/moe_v3100.py; every listed form: all stage loops + the real producer loop byte-identical)

| switch (defines) | kernel: what leaves warp 0's path | form / sha |
|---|---|---|
| cs32: MOE_W0_CS,MOE_W0_CS_M=4 + MOE_W0_SA,MOE_W0_SA_M=4,MOE_W0_SA_S=4 | bs32: no per-item aq_slot store (all bs32 items have a stage count = 0 mod 4, so the last slot is always 3: entries preset at init) + 32-bit aq_full address; item end 40..78 -> 29..67 instr (-11) | 69fa4b353ab2 |
| sa128: MOE_W0_SA,MOE_W0_SA_M=8,MOE_W0_SA_S=4 | bs128 MW4: 32-bit aq_full address (-4) | 796f92b75b39 |
| cs256: MOE_W0_CS,MOE_W0_CS_M=16 (CSW=2: the producer re-presets the entries after a warm-up stage) | bs256 MW8 (item stage count 16 = 1 mod 3: last slot of item q = (w + q) % 3): no aq_slot store, no q % 3 multiply-high chain (-9, TS e34 220 -> 184 cycles) | 323e963536eb |
| et16: MOE_W0_ET,MOE_W0_ET_M=2 | bs16: buffer-free try_wait issued before the slot release (and, by the new schedule, 1-group items skip the 8 x64 FMULs of the dead second token group) | 7504158c4262 |
| ra16: MOE_W0_RA,MOE_W0_RA_M=2 | bs16: record stores on 32-bit shared addresses: warp 0's S2R CgaCtaId -> MOV -> LEA chain gone (-8 instr; TS e23 176 -> 141) | d76a16f208b6 |
| raet16: both | bs16 item end 54 -> 45 instr on the 1-group path | 4a3b020976a2 |
| hp (MOE_W0_HP, position 0 bs8 / 1 bs16) | the next item's first-stage try_wait at the item end, the head waits only if it failed (h 240 -> 164 bs8, 171 -> 130 bs16) | bs8 72f4550434ce, bs16 e8cafa235c93 |
| hp2 (MOE_W0_HP,MOE_W0_HPP=2) | the same test right after the queue arrival | bs8 279e757819a6, bs16 c62165dea32a |

No loop-exact form exists (searched, dev/wpcgs_search.py) for: MOE_W0_S6E (x64 into the SEPI epilogue's scale; 120 forms
x ET / RA / RW / SA / HP knobs, bs16 always sim 0.74), MOE_W0_SX (x64 at the store), MOE_W0_RW (record by warp 1-7;
with MOE_W0_RWB (branch) the bs8 loop is register-RENAMED only, 192 forms), MOE_W0_RH (bs8 record written at the item
head into the free 8 KB gap before the ring; 16 forms, sim 0.53-0.58), MOE_W0_EB (buffer test at the item head;
bs16 renamed only, bs8 sim 0.45-0.92), MOE_W0_ET / RA at bs8 (0.84-0.92), MOE_W0_SA at the item head (all kernels
re-scheduled), MOE_W0_CS at bs128 (0.908; exact only with a runtime-false store that costs the same), MOE_W0_QM
(q % NS counter; renamed), MOE_W0_HP at bs32 (0.63) and (8,8,1) (renamed only).
Formal-ordering note: moving the AEPI consumers' fence.proxy.async to the epilogue thread that issues the bulk copy
would remove MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S (+ the 4 never-executed LDS at (8,8,1)) from every consumer item end,
but the PTX memory model text states "a proxy fence along the base causality path" only for alias proxy fences, and
the CUDA guide's TMA pattern fences in the writing threads: not done (rule: formally correct ordering).

## 4. Measurements (bench2 jm verified, B2_SHARE_EQ, 100 reps x 2 passes, fillers at both ends, v1510 controls in
## the middle; % vs the mean of the middle controls, dev/wpit2_tab.py; * = above every middle control)

| variant | target shape: copies | placebo shapes | middle controls (target) |
|---|---|---|---|
| cs32 (run 1) | bs32: +0.01 / +0.01 / -0.02 | bs16 +0.01 / -0.06 / -0.09, bs128 +0.08 / -0.04 / +0.08 | -0.09 .. +0.11 |
| sa128 (run 1) | bs128: -0.15 / +0.09 | bs16 +0.01 / -0.02, bs32 -0.09 / -0.13 | -0.14 .. +0.11 |
| cs256 (run 3) | bs256: -0.21 / -0.07 / -0.22 | bs16 +0.04 / -0.03 / +0.08, bs128 0.00 / -0.03 / -0.02 | -0.08 .. +0.10 |
| et16 (runs 1, 2) | bs16: +0.24* / +0.12* / +0.10 / +0.25* | bs32 -0.02 / +0.06, bs128 +0.05 / +0.01, bs8 -0.01 / -0.04 | -0.12 .. +0.25 |
| ra16 (runs 2, 3) | bs16: +0.47* / +0.27* / +0.09 / +0.15 / -0.06 / +0.08 | bs8 +0.02 / -0.08, bs128 / bs256 -0.16 .. +0.19 | -0.17 .. +0.26 |
| raet16 (runs 2, 3, 5) | bs16: +0.10 / +0.25* / +0.29* / +0.22* / +0.19 / +0.13* / +0.21* / +0.22* (mean +0.20) | bs8 0.00 / -0.15 / -0.13 / -0.24 / -0.12 / -0.02, bs32 -0.01 / -0.08 / +0.12 | -0.17 .. +0.26 |
| etf16 = ET with test_wait (run 2) | bs16: +0.09 | bs8 +0.12 | |
| hp / hpra / hpall (run 4) | bs8 (same kernel in all three): -0.25 / -0.15 / -0.23 / -0.33 / -0.35 / -0.65 / -0.18 / -0.40 / -0.39; bs16: hp -0.24 / -0.12 / -0.27, hpra -0.12 / -0.15, hpall -0.15 / -0.28 / +0.09 / +0.01 | bs32 -0.10 .. +0.10 | bs8 -0.11 .. +0.11, bs16 -0.19 .. +0.16 |
| hp2 / raethp2 (run 5) | bs8: -0.04 / -0.34 / -0.46, -0.56 / -0.25 / -0.60; bs16: hp2 -0.30 / -0.17 / -0.10, raethp2 -0.22 / -0.11 / +0.06 | bs32 -0.13 .. +0.03 | bs8 -0.11 .. +0.16 |

Correctness: hang1 (bs 1 .. 512 subsets) all finished / ok for every variant; check_mt 0 failures: cs32 (M 16-33, 64),
et16 (M 8-17), sa128 (M 64-129), ra16 / raet16 / etf16 (M 8-17), cs256 (M 128-257), hp + hpall (M 2-16), raethp2 (M 2-16).

## 5. What the measurements say

- bs32 / bs128 / bs256 (asynchronous-epilogue kernels): removing 4-11 bookkeeping instructions (incl. the MW8 q % 3
  multiply-high chains, -36 cycles of warp 0's item end in the stamp timeline) converts to nothing (bs32, bs128) or
  reads slightly negative (bs256, 3 copies -0.07..-0.22). The instructions removed were outside the latency chain
  that binds there (WP-IT3's co-critical held-slot turnaround at bs32; bs256 is at the power-limited clock).
- bs8 / bs16 (SEPI kernels: no held slot, the queue arrival hands the item to warps 10-11): the item end is
  HAND-OFF critical. Shortening the chain from the last stage to the queue arrival gains (record chain without the
  S2R, the buffer test issued earlier: +0.2% at bs16, 8 copies / 3 jobs), lengthening it loses (the next item's
  first-stage try_wait moved into the item end before the stores: e23 +40..60 cycles, bs8 -0.33% mean over 9 copies,
  bs16 -0.15%), and moving the head's try_wait to right after the arrival still loses (hp2: bs8 -0.04..-0.60, bs16
  -0.10..-0.30) although the stamp timeline shows the head segment 40-76 cycles shorter: warp 0's own next-item start
  is not what binds; an extra / earlier SYNCS on warp 0 costs.
- WP-SEPI32's "each instruction ~0.03-0.04%" does not generalise: what costs is SYNCS round trips and latency on the
  hand-off chain, not issue slots (cs32's 11 instructions: 0; hp's extra try_wait at the item end: -0.3%).

## 6. Bake: dev/moe_v3101.py (dev/wpw0_mk3101.py) = v3100 with MOE_W0_RA + MOE_W0_ET on the bs9-16 kernel only

Default on, escape MOE_NO_W0 (= v1510 SASS in all 16 kernels). Default build: only moe_4_2_1_4_1 changes
(4a3b020976a2; both stage loops (hc aee3651d0909 / f11d59a3794d) and the producer loop (f0b012bf038e) byte-identical to
v1510's); every other kernel = v1510's. The 1-group item end (bs9-16) is 54 -> 45 instructions: warp 0's record S2R
CgaCtaId / MOV / LEA chain gone (stores on UR5, the head's uniform shared base), the buffer try_wait issued before the
slot release, the dead second token group's 8 x64 FMULs moved into the 2-group block.
Validation (run 6, kda_ws/runs/wpw0_run6.log): hang1 20 sizes (1 .. 512) ok; check_mt M 1..17, 20, 24, 32, 33, 64, 65,
128, 129, 256, 257 (normal + skewed, 3 seeds, 2 calls) 0 failures; outputs bitwise equal to v1510's at M 9 / 12 / 16
(normal + skewed, 2 seeds: 100% equal elements). bench2 jm all six shapes (v1510 middle controls):
bs16 +0.11 / +0.02 / +0.27* / +0.14 (controls -0.19 .. +0.26); placebos bs1 -0.42 .. +0.16 (controls -0.19 .. +0.25),
bs8 -0.50 .. +0.07 (controls -0.27 .. +0.25), bs32 +0.01 .. +0.10, bs128 +0.02 .. +0.09, bs256 -0.07 .. +0.08.
Over runs 2, 3, 5, 6 the bs16 kernel 4a3b020976a2 measured +0.10 / +0.25 / +0.29 / +0.22 / +0.19 / +0.13 / +0.21 / +0.22 /
+0.11 / +0.02 / +0.27 / +0.14% (12 copies, 4 jobs, mean +0.18%; 7 of 12 above every middle control of their job).
Run 7 (kda_ws/runs/wpw0_run7.log) added 6 more copies of the same bs16 kernel (4 as rl8's bs16 kernel, 2 v3101):
+0.32* / +0.34* / +0.06 / +0.35* / +0.29* / +0.15* (controls -0.22 .. +0.17).
=> 18 copies in 5 jobs, mean +0.20% at bs16 (12 of 18 above every middle control of their job, none below the
control mean); ~+0.03% geomean (~1.7 us on the judge scale). Below the 0.3% per-shape bar, but consistent; offered
for stacking at the planner's discretion (only moe_4_2_1_4_1 changes).

## 7. Run 7: MOE_W0_RL (dev/moe_v3102.py = v3101 + MOE_W0_RL, off by default; default build = v3101's SASS)

The ring-slot release arrival after the queue arrival (SEPI): loop-exact only at bs8 (moe_4_1_1_4_1 f87d1161ffca:
stage loop, producer loop and the peeled first stage identical; bs16 forms all sim 0.74). hang1 12 sizes ok, check_mt
M 2-9, 16: 0 failures. Timeline: e01 124 -> 92, e34 124 -> 172 (both arrivals now in one divergent lane-0 block after
the stores), total unchanged. bs8 rl8 -0.27 / +0.08 / +0.31* / -0.09 (controls -0.16 .. +0.33): no gain.

## 8. Summary (WP-W0)

- Map (section 1): per kernel the 1-group item end is 45-63 SASS instructions and the item top 33-102; per role 35-50
  all-lane, 5-9 lane-0 (predicated, issued by every lane), 2-9 warp-0-lane-0 (predicated, issued by every warp), 4
  never-executed `@!PT LDS` in (8,8,1); the latency elements are S2R TID (AEPI, end and top), the fence's
  MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S (AEPI), up to three SYNCS (SEPI: release, buffer try_wait, queue arrival) and the
  head's try_wait + record LDS. Stamps (section 2): warp 0 ends items 300-900 cycles after warp 7 and spends ~1000-1300
  stamped cycles to its next first stage, with the next data already there.
- Measured (section 4): instruction removal outside the binding chain converts to nothing (cs32 bs32 0.00%, sa128
  bs128 -0.03%) or slightly negative (cs256 bs256 -0.17%); SYNCS placement matters in the SEPI kernels: an extra or
  earlier try_wait on warp 0 loses (hp / hp2 bs8 -0.04 .. -0.65%, bs16 -0.10 .. -0.30%), a shorter hand-off chain wins
  a little (bs16 RA + ET +0.20% over 18 copies); the release-after-arrival reorder (rl8) is neutral at bs8.
- Banked candidate: dev/moe_v3101.py (default build; = dev/moe_v3100.py with
  `MOE_W0_RA,MOE_W0_RA_M=2,MOE_W0_ET,MOE_W0_ET_M=2`), only moe_4_2_1_4_1 -> 4a3b020976a2, loops byte-identical,
  escape MOE_NO_W0 = v1510 SASS; validated (hang1 20 sizes, check_mt 27 sizes 0 failures, outputs bitwise = v1510).
- Rejected / not loop-exact: everything else in section 3 (no form found by dev/wpcgs_search.py), the epilogue-side
  proxy fence (formal-ordering doubt).
- For later workers: (1) the (4,*,1) kernels' 1275-1480-instruction "producer loops" of dev/wpcgs_search.py are
  retry-stub pseudo-loops: check the real producer loop (244 / 252 instructions) by hash instead; (2) the bs<=8 kernel's
  bottom-tested stage loop re-schedules under almost any item-end edit (only MOE_W0_HP and MOE_W0_RL were exact);
  (3) MOE_W0_TS / dev/wpw0_ts.py gives per-segment item-end timings for any kernel (never run a TS build under hang1:
  its stamps need the enlarged ctr).

## 9. Two integration lines (planner, 2026-11-01): the winner as a patch for v1504 (line A) and v1510 (line B)

dev/wpw0_mk3103.py BASE OUT [--bake [--etf N]]: a self-contained patch of the SEPI consumer item end (the text is the
same in v1504 and v1510) with MOE_W0_RA / MOE_W0_ET (+ MOE_W0_ETF form bits), off unless --bake. (Header fix: every
WP-W0 file now keeps `# okc: problem=muse_moe_w4a16` as its FIRST line; dev/moe_v3100-3102.py were regenerated with
only that header order changed, SASS unchanged.)
- dev/moe_v3103.py = v1504 + switches (off): all 16 kernels = v1504's SASS.
- dev/moe_v3104.py (line A winner) = v1504 + `--bake --etf 2` (MOE_W0_RA,MOE_W0_RA_M=2,MOE_W0_ET,MOE_W0_ET_M=2,
  MOE_W0_ETF=2 baked, escape MOE_NO_W0 = v1504 SASS in all 16): only moe_4_2_1_4_1 changes, 8fb6104136ab ->
  4ac5d75ade40, EXACT (both stage loops, the producer loop and the four retry-stub pseudo-loops byte-identical).
  On v1504 the measured v1510 form (ETF=0) re-schedules one stage loop (sim 0.985); ETF=2 (slot release, then the
  try_wait, wait only on failure) and RA alone (8f8c691ebd61) are exact. The v1504 form keeps RA's record-chain
  removal and the relocation of the dead second group's 8 x64 FMULs into the 2-group block, but not the earlier
  try_wait (not benched by WP-W0 beyond run 8's short A/B).
- dev/moe_v3105.py (line B) = v1510 + `--bake` (ETF=0): moe_4_2_1_4_1 4a3b020976a2, the kernel measured in runs 2-7
  (= dev/moe_v3101.py's); MOE_NO_W0 = v1510 SASS in all 16.
- Run 8 (kda_ws/runs/wpw0_run8.log): hang1 13 sizes ok for both v3104 and v3105; check_mt M 8-17 (normal + skewed,
  3 seeds, 2 calls) 0 failures for both; v3104's outputs bitwise equal to v1504's at M 9 / 12 / 16 (12 cases, 100%).
  Short A/B on line A (v1504 middle controls -0.14 .. +0.08): v3104 at bs16 -0.13 / -0.04 / +0.01 / +0.30* (mean +0.04%);
  bs8 placebo -0.20 .. +0.24 (controls -0.13 .. +0.22). => the loop-exact v1504 form (ETF=2) shows no gain in 4 copies:
  the earlier try_wait (ETF=0, exact only on v1510) seems to be the part that converts. Line A alternative: RA alone
  (dev/moe_v3103.py + MOE_W0_RA,MOE_W0_RA_M=2, 8f8c691ebd61, exact; on v1510 ra16 measured +0.17% mean over 6 copies),
  not benched on v1504.
