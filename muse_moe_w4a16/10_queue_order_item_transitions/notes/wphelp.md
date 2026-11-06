# WP-HELP: helper-warp issue interference in the (8,8,1) kernels (bs128 / bs256), 2026-11-01

Worker WP-HELP. Base line B dev/moe_v1513.py, line A dev/moe_v3770.py. Namespace dev/moe_v3900-v3999.py, scripts
dev/wphelp_*, logs kda_ws/runs/wphelp_runN.log. Target kernels moe_8_8_1_3_4 (bs65-128, MW 4) and moe_8_8_1_3_8
(bs129-256, MW 8). Warp layout (12 warps, SMSP = warp % 4): consumers 0-7 (two per SMSP), producer 8 (SMSP0),
signaler 9 (SMSP1, lane 0), epilogue warps 10 / 11 (SMSP2 / SMSP3).

## Files
- dev/wphelp_roles.py: per-role / per-kind executed warp instructions of a kernel from ncu SourceCounters TSVs
  (dev/wpdead_csv.py format) + a local `nvdisasm -c -g -gi` dump of the same SASS (role = first line of the inlining
  chain inside a role region of moe.cu; innermost HMMA / HGMMA loops = STAGE); `--loops ROLE` lists a role's loops.
- dev/wphelp_trace.py: MOE_WPN_TRACE per-stage per-warp trace with pacing statistics (slowest-warp period of mid-item
  stages, per-warp lateness, share of stages whose last warp is on SMSP s).
- dev/wphelp_mk3900.py (MOE_HELP_PSW), dev/wphelp_mk3901.py (+ MOE_HELP_ED), dev/wphelp_mk3902.py (+ MOE_HELP_P2):
  patch scripts BASE OUT for both lines; outputs dev/moe_v3900/3901/3902.py (line B), dev/moe_v3950/3951/3952.py
  (line A). Every default build = the base's SASS in all 16 kernels (NVRTC 13, MOE_S5R_FIT, checked for each file).
- dev/wphelp_run1.sh (ncu + traces), dev/wphelp_run2.sh (P2 / ED correctness, traces, bench2 screening).

## 1. Per-role instruction map (run 1: ncu SourceCounters on v1513, 4 jm seeds per shape, one cold call each)

Warp instructions executed per CTA per call and per ring stage (stages per CTA: bs256 145.0, bs128 122.2):

| role (SMSP) | bs256 per CTA | /stage | bs128 per CTA | /stage | what it is |
|---|---|---|---|---|---|
| consumer stage loops (8 warps) | 625.2k | 4312 | 514.2k | 4209 | 86-87% of all; ~540 per warp per stage |
| signaler w9 (SMSP1) | 26.5k | 182.6 | 21.5k | 176.4 | 97% spin: try_wait + NANOSLEEP.SYNCS 200 ns wakes (SYNCS 89, BRA 45, NANOSLEEP 44 per stage); real work ~0.8k |
| epilogue w10-11 (SMSP2+3) | 22.9k | 157.8 | 17.2k | 140.7 | spin 12.1k / 10.1k (1 ms hint wakes on every CTA mbarrier event: SYNCS 42, NANOSLEEP 21, BRA ~20 per stage); real work ~10.8k / 7.1k: FMUL 12.8, MUFU 4.9, LDS 3.5, F2FP 2.4, STG 2.0 and ~35 integer / address / control per stage |
| producer w8 (SMSP0) | 13.1k | 90.2 | 10.6k | 86.6 | no spin: IMAD 17, R2UR 14, ISETP 8, BRA 7.5, VIADD 4, BSSY/BSYNC 6, SEL 3, UBLKCP 2.8, UIADD3 2.4, UMOV 2, down-item readiness polls (LDG.STRONG.GPU + CCTL.IVALL + YIELD, 222 / 136 polls per CTA) |
| consumer item top / end | 19.1k | 132 | 13.2k | 108 | |
| combine / routing / gather / setup | 20.0k | 138 | 13.0k | 106 | |

Kinds (bs256, per CTA): signaler spin 25.7k of 26.5k; epilogue spin 12.1k, sync 0.26k, smem 0.5k, glob 0.3k, math
3.3k, alu 5.4k, ctl 1.0k; producer sync 0.85k (BAR, ELECT, SYNCS arrive), copy 0.40k, glob 0.26k (polls), smem 0.17k,
alu 9.0k (R2UR operand moves for the UBLKCPs, address / slot arithmetic), ctl 2.4k.
Epilogue inner loops (bs256 SASS): gate/up wgmma layout 113 instr per 8 outputs per lane (MUFU 16, FMUL 32, FADD 8,
LDS 8, F2FP 4, STG 1, ~39 integer for the swizzle and addresses), gate/up row layout 95, down full-k 99 per 4 float4
(LDS 4, FMUL 16, F2FP 8, STG 8 = two 32-bit stores per float4, ~60 integer: signed o / 64 and 64-bit addresses per
token), down split-k 83 (REDG).
=> Issue count does not predict the interference: SMSP1 carries the most helper instructions (183 / stage of spin
wakes) and its consumers are the fastest; the producer's 90 / stage (R2UR / uniform-datapath / UBLKCP / BAR wake) and
the epilogue's ~75 real instructions / stage are what slow their SMSP pairs.

## 2. Who paces the stages (run 1: dev/wphelp_trace.py, MOE_WPN_TRACE builds, 2 seeds per shape x 3 cold calls)

Mid-item stages (the previous record is the same item's previous stage); P = slowest-warp stage period (max done_i -
max done_{i-1}), median; "slowest SMSP" = share of mid stages whose last-finishing warp sits on SMSP s. BE = MOE_IT_BE
(timing only: the epilogue warps do no work), PSW = MOE_HELP_PSW (producer on warp 9 / SMSP1, signaler on warp 8).
Trace builds: PSW's stage loops differ slightly from the v1513 trace build's (opcode similarity 0.975), BE's are equal.

| routing | class | v1513 P / slowest SMSP0-3 % | BE | PSW | PSW + BE |
|---|---|---|---|---|---|
| bs256/73011 | G1_1 (1-group) | 1656 / 69 1 24 6 | 1653 / 95 1 2 2 | 1679 / 1 82 12 4 | 1674 / 1 97 1 1 |
| bs256/73011 | G1_m (wgmma) | 2090 / 45 10 34 11 | 2054 / 74 11 8 7 | 2083 / 18 44 28 10 | 2047 / 18 67 9 6 |
| bs256/73011 | G2_m | 2103 / 55 10 25 10 | 2069 | 2098 | 2075 |
| bs256/71497 | G1_1 | 1664 / 78 2 14 5 | 1658 | 1679 | 1675 |
| bs256/71497 | G1_m | 2019 / 59 9 24 8 | 2007 | 2017 | 2010 |
| bs256/71497 | G2_m | 2043 / 52 10 28 9 | 2026 | 2056 | 2045 |
| bs128/71637 | G1_1 | 1661 / 79 2 13 6 | 1660 | 1676 | 1675 |
| bs128/72808 | G1_1 | 1652 / 86 2 9 3 | 1649 | 1673 | 1672 |
| bs128/72808 | G1_m | 2127 / 44 17 24 15 | 2118 | 2100 | 2084 |

Per-warp lateness (done - earliest done, mean, bs256/73011 G1_1): v1513 [572 115 344 248 330 107 321 228] (w0, w4 on
the producer's SMSP0 late by 330-570 cycles, w2 / w6 next to epilogue warp 10 by 320-345, SMSP1 ~110); PSW moves the
lateness to w1 / w5 (701 / 653). The warps drift by hundreds of cycles inside the ring's slack, so a stage's period is
set by the slowest SMSP pair's average load, not by per-stage peaks.
- 1-group stages (~40% of bs256 stage time, ~70% at bs128) are paced by the producer's SMSP in 69-97% of the stages;
  removing the epilogue's work (BE) does not move them (1649-1665 either way), moving the producer to SMSP1 (PSW)
  makes them 15-25 cycles slower.
- wgmma stages: SMSP0 (producer) 45-59%, SMSP2 (epilogue warp 10, which also runs the et == 0 bulk copies / TMG
  stores) 24-34%; BE saves 9-36 cycles per stage (0.4-1.7%) and leaves SMSP0 pacing 74%.
=> The producer's SMSP pair is the bottleneck in every phase; the epilogue's pair is the second one in wgmma stages.
Moving either alone (PSW, BE) changes little; what can pay is splitting the producer's bursts over two SMSPs and
cutting the epilogue's work at the same time.

## 3. Candidates (dev/wphelp_mk3905.py = the latest patch; all switches off by default, kernel mask MOE_HELP_KM)

| switch | what changes | codegen (NVRTC 13, MOE_S5R_FIT; both lines) |
|---|---|---|
| MOE_HELP_PSW | producer role on warp 9 (SMSP1), signaler on warp 8, GCP gather on warps 8/10/11 (role branches; the first tid-remap form re-scheduled all 8 stage loops, sim 0.49) | 8/8 stage loops identical, producer loop renamed |
| MOE_HELP_ED | epilogue diet: the same thread -> element map and the same math, strength-reduced loops (gate/up wgmma layout per 64-k chunk with tokens stepped by 8: the swizzle's rows hoisted; row layout tokens stepped by 4; down rows tokens stepped by 1, one STG.64 per 4 BF16 values); epilogue inner loops 113 -> 77 (gate/up wgmma), 95 -> 73, 99 -> 42 (down full-k), 83 -> 34 (split-k REDG); outputs bitwise = v1513 (wpit_cmp: differences only at v1513's own KRED run-to-run level) | 8/8 stage loops identical, spills unchanged; MW 8 producer loop register-renamed |
| MOE_HELP_P2 | two producers: in each LEAN4 steady sub-loop warp 8 issues the even and warp 9 the odd stages (descriptor at smem 64..127, hand-off mbarrier at 48, issued-stage words at 56 / 60 with st.release / ld.acquire, a wait before a stage whose slot's previous stage was the other warp's); the signaler role moves into the epilogue (warp 11 lane 0 releases the item's counter after bar 15 of both epilogue warps); epi / sfree / sinfo no longer initialized | 8/8 stage loops identical, producer loops new, +5..8 LDL/STL in the producer's per-item path (none in a loop < 200 instr) |
| MOE_HELP_P2S / P2V / P2O | (with P2) signal move only / volatile issued-stage words / slot ownership: warp 9 issues the stages k % 3 < 2 of a sub-loop, warp 8 k % 3 == 2 (each producer keeps its slots, no per-stage waits; 8 + 8 stages per 16-stage item) | 8/8 stage loops identical |
| MOE_HELP_EL | the epilogue's single-thread duties of an item (TMP bulk copy, TMG bulk stores + waits, scratch reloads) alternate per item between warp 10 lane 0 and warp 11 lane 0 | alone: every stage and producer loop identical; with ED as ED |

## 4. Run 2 (kda_ws/runs/wphelp_run2.log; line B, dev/moe_v3902.py)
Correctness: hang1 bs 8 32 65 72 100 128 129 160 200 256 257 512 ok for P2, ED, P2 + ED; check_mt (M 32 65 66 80 100 127
128 129 150 160 200 255 256 257, normal + skewed, 3 seeds, 2 calls) 0 failures for all three; wpit_cmp vs v1513 (M 128
200 256, normal + skewed): 99.99-100% bitwise-equal elements, every difference at the KRED atomics' own run-to-run level
(a - a' column); wpmix_calls P2 + ED 120 calls 0 failures (worst 3.29e-3).

bench2 screening (jm seeds, 100 reps x 2 passes, fillers at both ends, 3 v1513 controls + 3 copies of each candidate
interleaved; % vs the mean of the controls, per copy; * = above every control):

| | bs32 (placebo) p1 / p2 | bs128 p1 / p2 | bs256 p1 / p2 |
|---|---|---|---|
| controls band | -0.08..+0.10 / -0.04..+0.09 | -0.10..+0.09 / -0.11..+0.09 | -0.03..+0.04 / -0.07..+0.08 |
| P2 | -0.14 / -0.07 | -0.29 (-0.33 -0.43 -0.21) / -0.35 | -0.32 (-0.36 -0.37 -0.23) / -0.26 |
| ED | -0.03 / -0.01 | +0.01 / -0.10 | +0.07 (+0.01 +0.17* +0.05*) / +0.16 (+0.16* +0.11* +0.14*) |
| P2 + ED | -0.07 / +0.02 | -0.28 / -0.32 | -0.15 / -0.14 |

Trace (P2's trace build re-schedules the stage loops slightly, sim 0.975: only qualitative): with P2 the SMSP0 pair is
still the slowest in 67-85% of the 1-group stages and w0's lateness is unchanged (~550 cycles), while w1 / w5 (SMSP1,
now with warp 9's stages) get 240-300 cycles late. ED: wgmma stages G2_m -14 / -20 cycles (bs256), G1_m ~0.
=> The alternating second producer loses 0.3% at bs128 / bs256: moving ~20% of the producer's instructions (half of the
steady stages) to SMSP1 does not relieve SMSP0 and loads SMSP1, plus a release store (MEMBAR.ALL.CTA) and an acquire
poll per stage on both warps. ED is a small, consistent bs256 gain (+0.07 / +0.16%).

## 5. Run 3 (kda_ws/runs/wphelp_run3.log; line B, dev/moe_v3905.py): EL, ED + EL, P2O (slot ownership) + ED + EL
Correctness: hang1 (same 12 sizes) ok for EL, ED + EL, P2O + ED + EL, P2O; check_mt (14 sizes, normal + skewed) 0 failures
for ED + EL and P2O + ED + EL; wpit_cmp vs v1513: only KRED-level differences; wpmix_calls P2O + ED + EL 120 calls 0
failures.

Trace (bs256/73011, 71497, bs128/72808; P2O's trace build has its own stage-loop schedule): P2O does what it was built
for -- the 1-group stages' slowest warp is now split between SMSP0 and SMSP1 (35-40% / 39-45%, was 73-86% / 1-2%),
w0's lateness 313-354 (was 452-509), w1's 327-350 (was 99-122) -- but the slowest-warp period does not move:
1-group P 1662 / 1667 / 1660 vs 1658 / 1661 / 1657, wgmma 2088 / 2013 / 2116 vs 2095 / 2024 / 2130.
ED + EL: 1-group unchanged, wgmma -1..-20.

bench2 screening (3 controls, 3 copies each). Pass 1 was disturbed at bs32 / bs128 (every control's median 4-5 us above
pass 2's, control bands +0.34..+0.86 / -0.06..+1.17: not usable); bs256 and pass 2 are clean:

| | bs32 p2 | bs128 p2 | bs256 p1 | bs256 p2 |
|---|---|---|---|---|
| controls band | -0.09..+0.07 | -0.14..+0.20 | -0.06..+0.05 | -0.03..+0.07 |
| ED | -0.09 | +0.05 (0/3) | +0.04 (+0.10 +0.07 +0.08, 3/3) | +0.12 (+0.14 +0.06 +0.19, 2/3) |
| ED + EL | -0.01 | +0.02 (0/3) | +0.07 (+0.11 +0.14 +0.15, 3/3) | +0.08 (+0.05 +0.18 +0.11, 2/3) |
| P2O + ED + EL | -0.05 | -0.11 (0/3) | -0.16 (0/3) | -0.13 (-0.13 -0.29 -0.10, 0/3) |

=> Splitting the producer's stages over two SMSPs (P2 alternating, P2O balanced 8 + 8) moves the per-warp lateness but
not the period, and costs 0.1-0.35%. The per-SMSP issue interference the per-warp traces show is absorbed by the
ring's slack (the warps drift); the stage period is not the slowest pair's issue-limited rate. What does move the
wgmma period (BE: -9..-36 cycles) is the epilogue's work as a whole (its shared-memory traffic: TMP copy, TMG stores /
reloads, room reads ~190 KB per 64-token item, and its global stores), not its issue slots; ED (fewer instructions,
same traffic) recovers ~+0.1% of it at bs256.
ED over runs 2-3 at bs256 (12 copy-passes): +0.01 +0.17 +0.05 +0.16 +0.11 +0.14 +0.10 +0.07 +0.08 +0.14 +0.06 +0.19 (mean
+0.11%, 10 / 12 above every control); bs128 0 (+0.01 / -0.10 / +0.05). Baked for the bs129-256 kernel only:
dev/moe_v3910.py (line B = v3905 + MOE_HELP_ED,MOE_HELP_KM=2; only moe_8_8_1_3_8 changes: ee25b69d8408 -> 571003e3fd87)
and dev/moe_v3960.py (line A = v3955 + the same; d8adaacb5a69 -> fee834eea09b). Confirmation: run 4.

## 6. Run 4 (kda_ws/runs/wphelp_run4.log): confirmation of the ED bake, both lines, formal protocol
dev/moe_v3910.py (line B) vs v1513, dev/moe_v3960.py (line A) vs v3770: hang1 bs 1 8 16 32 33 64 65 100 128 129 200 256 257
512 ok (both); check_mt M 1 8 16 32 33 64 65 66 100 127 128 129 160 200 255 256 257 300 512 (normal + skewed, 3 seeds, 2
calls) 0 failures (both); wpmix_calls 160 calls 0 failures (worst 3.29e-3, both). bench2 jm, fillers 2 + 2 at the ends,
4 base + 4 candidate copies interleaved, 100 reps x 2 passes (bs100 / bs200 bench2's own seeds; bs32 / bs100 / bs128 run
unchanged kernels = placebos):

| line, pass | bs32 | bs100 | bs128 | bs200 | bs256 |
|---|---|---|---|---|---|
| B p1 (controls band) | -0.08 (-0.02..+0.10) | +0.09 (-0.06..+0.15) | +0.00 (-0.07..+0.10) | +0.06 (+0.10 +0.01 -0.12 +0.21; -0.04..+0.03) | +0.09 (+0.03 +0.15 +0.05 +0.14; -0.05..+0.12) |
| B p2 | -0.03 | +0.05 | +0.02 | +0.02 (-0.07 +0.05 +0.02 +0.20) | +0.13 (+0.03 +0.10 +0.16 +0.26; -0.06..+0.04) |
| A p1 | +0.01 | +0.03 | -0.01 | +0.19 (+0.24 +0.12 +0.07 +0.30, 4/4; -0.16..+0.05) | +0.14 (+0.02 +0.15 +0.10 +0.23, 3/4; -0.05..+0.03) |
| A p2 | -0.01 | -0.04 | -0.00 | +0.14 (4/4) | +0.09 (+0.10 +0.11 +0.15 +0.26, 4/4; -0.03..+0.06) |

ED at bs256 over runs 2-4: 28 copy-passes, mean +0.11% (line B +0.11, line A +0.12), 22 / 28 above every middle control;
bs200 (the same kernel) +0.02..+0.19; placebo shapes inside their control bands. ~+0.02% geomean: below the 0.2%
per-shape bar, consistent on both lines, outputs bitwise = the base's (up to KRED run-to-run noise).

## 7. Summary (WP-HELP)
- Per-role issue map (section 1): per ring stage the 8 consumer warps issue ~4300 instructions; the helpers add the
  signaler's 183 (97% spin wakes, SMSP1), the epilogue's 158 (half spin wakes, half real work, SMSP2/3) and the
  producer's 90 (all real: R2UR / UBLKCP / BAR / per-item decode; SMSP0). Producer per CTA (bs256): steady LEAN4
  sub-loop 41%, the non-steady (LEAN3) stage path 27%, per-item work 20%, the startup gather wait 12%.
- Issue count does not predict the slowdown (SMSP1 has the most helper instructions and the fastest consumers), and
  per-SMSP lateness does not set the period: the warps drift by hundreds of cycles inside the ring's slack. Moving the
  producer (PSW), splitting its stages over two SMSPs (P2 alternating: -0.26..-0.35%; P2O slot ownership, 8 + 8 stages
  per item: -0.11..-0.16%) or alternating the epilogue's lead thread (EL: 0) moves the lateness but not the period.
- The only lever that moved wgmma stages is the epilogue's total work (BE bound -9..-36 cycles per wgmma stage at
  bs256); ED, the strength-reduced epilogue (same values, ~35-55% fewer epilogue instructions), banks +0.11% at bs256.
- Rejected: PSW, P2, P2O, P2 + ED, P2O + ED + EL, EL (no gain over ED). Untested: P2S (signal move alone) / P2V.

Deliverables (both lines; every default build = its base's SASS in all 16 kernels):
- Patch: dev/wphelp_mk3905.py BASE OUT (switches MOE_HELP_PSW, MOE_HELP_ED, MOE_HELP_P2 [+ P2S / P2V / P2O],
  MOE_HELP_EL, kernel mask MOE_HELP_KM: 1 = bs65-128 kernel, 2 = bs129-256 kernel, default 3). Line B dev/moe_v3905.py,
  line A dev/moe_v3955.py (earlier forms mk3900-3904 -> v3900-3904 / v3950-3954).
- Banked candidate: MOE_HELP_ED,MOE_HELP_KM=2 -> dev/moe_v3910.py (line B, only moe_8_8_1_3_8: ee25b69d8408 ->
  571003e3fd87) and dev/moe_v3960.py (line A, d8adaacb5a69 -> fee834eea09b); 8 / 8 stage loops byte-identical, spills
  unchanged, no C7511 / C7512; the MW 8 producer loop is register-renamed (same instructions).
