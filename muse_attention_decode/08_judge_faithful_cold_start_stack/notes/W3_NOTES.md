# W3: B16/32 sliding END chains (v32s attn_8_1_1_1w4 global, v16s attn_8_1_1_1x6 C6 pull), base dev/coord/k60.py (md5 152a09bb)
Cap 30 GPU-min. Write only here. Decide on nsys 2 copies x 2 passes vs same-job k60 controls (k60a/k60b = byte copies).

## Resume here
- FROZEN dev/w3/deliver_r1/ (17:2x): build.py <k61|k60> <out> (defaults --lay 1 --ea 1 --tpg 1) -> k61w3.py (md5 6291469d): check.py 56/0,
  t_v8w 0 failures, nsys v32s -0.264 / v16s -0.084 vs k61, bench_jt verify 32 cold-pair -0.9 / warm -0.30. README.txt + md5s.txt there.
- j5 DONE: TPE (d32s) dead in both placements; TPC (v16s) cold-pair -0.28 / warm +0.11 -> option only. GPU total ~25.2 of 30: stopped.
- Knob map (build.py): LAY w4 slot layout (KEPT), EA x6 early exit arrive (KEPT), TPG w4 TLB touch (KEPT), TPC x6 TLB touch (j5),
  TPE e3 TLB touch (mode 1 dead in j3; mode 2 in j5), CO compact fold (dead, +0.08), PAR no-reset parity slots (dead, +0.10; needs
  dev/w3/t_par.py, t_v8w's all-sentinel check does not apply), NR bound (never run).

## End-chain anatomy (from code + existing traces; k44 trace XF r6 = same w4 end code as k60 except WH_NAP 64)
v32s (128 CTAs, P = 4, 1 CTA/SM): last unit -> stream merge (smem swap + 256-bar) -> publish (FP16 rows, plain st, no fence) ->
owner poll (round 1: probe 1 word / source, then rows of landed sources; later rounds nanosleep + full rows; every word
sentinel-validated) -> fold -> epilogue (gates in regs) -> bf16 out -> RESET of every consumed slot (sentinel stores) -> exit.
Trace (k44, med / max): part 3 merged 9.75 / 10.27, published 10.30 / 10.93; all-in 12.0 / 12.6 (EVERY part, incl. part 3 whose
sources published 9.3); out 12.38 / 12.9; nsys 13.78 => ~0.8 us after the last out (v16s cluster kernel: ~0.1).
Dependent trips on the critical path: store one-way + >= 1 load RTT (0.6 under load) per round, 2-3 rounds (median 2) + the
reset drain (2 MB of stores) before the kernel can complete. Minimum: 1 one-way + 1 RTT (~0.9-1.0 us) + epilogue.
Byte/transaction audit of the slot path (k60 code): row stride 288 B (9 sectors, 5 used); rows rr = 1,2,3 mod 4 straddle two
128-B lines; each thread's 32-B sector is written by TWO 16-B stores in two different warp instructions (half-sector writes,
8 rows x 4 sectors per instruction); same for the reset; PFX prefetches 36 KB / owner (4.6 MB DRAM at the start, 2.6 MB of it
unused sectors / own-source slots).
v16s (96 CTAs, C6, owner pull): merged -> stage own rows (st.shared) + restricted release fence + arrive -> gate ldg -> cluster wait
-> 12 DSMEM loads / thread (6 src x (8 B data + 8 B m,l)) -> fold -> RMSNorm + gate -> bf16 store -> arrive -> cluster wait -> exit.
XD vs20 trace (pull, med): merged 6.8-7.5, staged+arrived 7.2-7.9, all-in (wait done) 8.1-8.2, out 9.0-9.1: the DSMEM pull +
epilogue is ~1.0 us; ~15 KB remote per SM at the measured 18-30 GB/s/SM DSMEM rate = 0.5-0.8 us => near its bytes floor.
Only latency piece left on top of it: the final arrive after the epilogue (the barrier latency is serial with it).

## Bets (decision rules written before each job)
### Job w3_j1 - DECISION RULE (written before the job; script dev/w3/j1.sh)
Files (build.py on k60; local NVRTC 13: base0 = all knobs off -> monolith cubin md5 7d71fb51 == k60; w4 / x6 128 regs, 0 spill):
 v1 = LAY + EA (w4: line-aligned full-sector slots; x6: early exit arrive) | v2 = NR (w4: no reset, host refill: BOUND) |
 v3 = LAY + NR (BOUND) | v4 = LAY + CO + EA (w4: + compact rolled fold, 2376 vs 2592 SASS, poll loop 250 vs 471; x6 == v1's).
Smoke: t_v8w seeds 1 on v4a (WH_ALLB: x6 + w4) and (WH_ALLB, WH_CLU=0: w8 / w4 global), v1a (WH_CLU=0); a FAIL drops the file.
nsys 2 copies x 2 passes, k60a / k60b first in each pass. D = mean(4 medians) - mean(4 control medians); per-run vs same-pass ctrl.
 v32s LAY (v1): KEEP iff D <= -0.08 and all 4 runs below their same-pass control mean.
 v32s LAY + CO (v4): KEEP iff the same AND v4 - v1 <= -0.05 (else LAY alone if it passes).
 v32s reset bound: NR (v2) and LAY + NR - LAY (v3 - v1): <= -0.15 -> build D2 (parity-tagged slots + per-(region, WG) epoch words,
   no reset ever) in job 2; > -0.10 -> the reset is free, D2 dead.
 v16s EA (v1 + v4 = 8 runs of identical x6 code): KEEP iff mean D <= -0.05 and >= 7 of 8 runs below their same-pass control mean
   (the whole x6 register allocation moved: lottery risk, see S4).
 Traces (K2_TRACE, B = 32, 12 reps): base0 (= k60 + stamps 28 reset-issued, 29 rows-back (data-dependent), 30 fold-done) and v4.
A KEEP -> job 2: combined build, tools/check.py (0 failures) + t_v8w seeds 2 (both defs) + nsys confirm -> deliver_r1.
### Job w3_j2 (= j1 retry, re-planned to fit the remaining cap; script dev/w3/j2.sh) - DECISION RULE (written before the job)
Files: k60a/b controls; v1 = LAY + EA; v4 = LAY + CO + EA; v5 = LAY + CO + PAR + EA (PAR = the correct no-reset build instead of the
NR bound: v5 - v4 IS the reset's cost). Smoke: t_v8w seeds 1 on v4a (WH_ALLB: x6 + w4; WH_ALLB,WH_CLU=0: w8 + w4) must be 96 PASS;
dev/w3/t_par.py seeds 1 x 3 interleaved rounds on v5a (WH_ALLB,WH_CLU=0: P = 8 and P = 4 regions mixed) must be 48 PASS.
nsys "16,8,sliding;32,8,sliding" 2 copies x 2 passes, controls first. D, per-run as in j1.
 v32s LAY (v1): KEEP iff D <= -0.08 and all 4 runs below their same-pass control mean.
 v32s LAY + CO (v4): KEEP iff the same and v4 - v1 <= -0.05.   PAR (v5): KEEP iff the same and v5 - v4 <= -0.05.
 v16s EA (x6 identical in v1 / v4 / v5: 12 runs): KEEP iff mean D <= -0.05 and >= 10 of 12 runs below their same-pass control mean.
 Any KEEP -> freeze job: per-kernel combined build, tools/check.py 0 failures + t_v8w seeds 2 (both defs) (+ t_par for PAR) + nsys confirm.
### Job w3_j3 (TP, script dev/w3/j3.sh) - DECISION RULE (written before the job)
Coordinator 16:4x: W4 found the judge's per-pair cold-TLB first rep (d32s even-odd 1.7-2.1 us, v16s 0.5-0.9, v32s ~0.55) and an
early per-2-MB-page L2 prefetch fixes it on d1s. W3_TP mode 1 (build.py --tpc/--tpg/--tpe 1): in the straight-line prefix right
after the first loads, slot c + nc tid (warp 0 of every CTA) issues one predicated prefetch.global.L2 per 2-MB page of the
K cache, V cache, page table and output (all addresses inside the allocations; nothing read back). Local NVRTC 13: x6 2032 -> 2096,
w4 2592 -> 2648, e3 2464 -> 2552 SASS; 128 / 128 / 127 regs, 0 spills; TP knobs 0 == k60 / k61 monolith cubin (7d71fb51).
Rule (W4's, per shape d32s / v16s / v32s): KEEP iff BOTH tp copies beat BOTH k61 copies by >= 0.05 us under --tlbcold 1 AND the
tp mean is within +0.03 of the k61 mean under --tlbcold 0. bench_jt's check reps must print no FAIL.
### Job w3_j2 RESULT (16:55:51-17:00:23, 4.5 GPU-min; dev/w3/out_j2.txt; GPU total ~12.5 of 30)
Smokes: v4a t_v8w WH_ALLB 48 PASS / 0, WH_ALLB,WH_CLU=0 48 / 0; v5a t_par WH_ALLB,WH_CLU=0 48 / 0 (P = 8 + P = 4 mixed, 3 rounds).
nsys medians (pass1 a, b | pass2 a, b), controls k60a/b:
| shape | control | v1 LAY+EA | v4 LAY+CO+EA | v5 LAY+CO+PAR+EA |
| v32s | 13.680 13.632 13.664 13.632 (13.652) | 13.280 13.216 13.232 13.264 = -0.404 (runs -0.38..-0.44) | 13.344 13.344 13.312 13.312 = -0.324 | 13.408 13.472 13.376 13.456 = -0.224 |
| v16s | 9.504 9.536 9.472 9.536 (9.512) | 9.424 9.504 9.376 9.472 = -0.068 | 9.472 9.440 9.376 9.472 = -0.072 | 9.488 9.472 9.472 9.424 = -0.048 |
Rules: v32s LAY KEEP (D -0.404, all 4 runs below by >= 0.376). CO NOT kept (v4 - v1 = +0.080: the rolled fold + early sigmoids are slower
than the unrolled fold once the slots are line-aligned). PAR NOT kept (v5 - v4 = +0.100: removing the reset does not pay for the epoch
cp.async + mbarrier waits; the reset is off the critical path). v16s EA KEEP: 12 of 12 runs below their same-pass control, mean -0.063.
=> freeze candidate = v1's kernels: w4 LAY + x6 EA (on k61). The traces did not run (a sed '&' broke the two trace lines; fixed in j2.sh).
### Job w3_j3 RESULT (16:53:09-~16:55:40, ~3 GPU-min incl. start; dev/w3/out_j3.txt; GPU total ~15.5 of 30)
bench_jt (judge pairs, judge ovh, paired-ratio time us), k61a / k61b vs tpa / tpb (TP mode 1 on x6, w4, e3):
| shape | tlbcold 1: k61a k61b / tpa tpb | tlbcold 0: k61a k61b / tpa tpb | rule |
| d32s | 12.42 11.78 / 11.75 11.80 | 10.24 10.35 / 10.31 10.30 | NO (tpb 11.80 >= k61b 11.78; the k61 copies differ by 0.64) |
| v16s | 11.96 11.46 / 11.04 11.15 | 10.53 10.60 / 10.65 10.74 | cold: YES (-0.31 worst case, ~-0.6 mean); warm +0.13 > +0.03: NO |
| v32s | 16.51 16.31 / 15.86 15.91 | 14.59 14.69 / 14.50 14.58 | YES: cold -0.40 worst case (~-0.5 mean), warm -0.10 |
=> TP KEPT on w4 (v32s) only. x6 (v16s) TP is a coordinator option (judge-mode ~-0.6, warm +0.13: fails the pre-registered warm guard);
d32s: the cold penalty (+1.8 us, cold - warm) is NOT recovered by touching K / V / page-table / out pages in the prefix (next lever:
the qkv / cache_loc / seq_lens pages, or mode 2 after the page-table loads; not run).
### Job w3_j4 (freeze gate, script dev/w3/j4.sh) - DECISION RULE (written before the job)
cand.py = candb.py = build.py dev/coord/k61.py --lay 1 --ea 1 --tpg 1 (md5 6291469d; local NVRTC 13: x6 == v1's x6 (EA), e3 == k60's,
w4 = LAY + TP 2648 SASS; 128 / 128 / 127 regs, 0 spills; stripped size 836,818 B with every W3 knob on).
FREEZE (deliver_r1 = builder with these defaults) iff tools/check.py 56 PASS / 0 failures AND t_v8w seeds 1 (WH_ALLB; WH_ALLB,WH_CLU=0)
0 FAIL AND nsys (2 x 2 vs k61a/b): v32s D <= -0.08 with all 4 runs below their same-pass control, v16s D <= 0 with >= 3 of 4 runs below.
TP part: bench_jt verify 32 tlbcold 1: both cand copies <= both k61 copies - 0.05 and tlbcold 0 mean <= k61 mean + 0.03; if the TP part
fails but the rest passes -> freeze with --tpg 0 (w4 = v1's measured LAY machine code; no new job).
### Job w3_j4 RESULT (17:15:11-~17:21, ~6 GPU-min; dev/w3/out_j4.txt; GPU total ~21.7 of 30): FREEZE PASSED -> deliver_r1
check.py dev/w3/cand.py: 56 PASS, 0 failures. t_v8w seeds 1: WH_ALLB 48 / 0, WH_ALLB,WH_CLU=0 48 / 0.
nsys (k61a/b controls): v32s 13.600 13.632 13.648 13.648 (13.632) vs cand 13.360 13.344 13.408 13.361 = -0.264 (runs -0.24..-0.29);
  v16s 9.568 9.585 9.504 9.536 (9.548) vs cand 9.472 9.472 9.440 9.472 = -0.084 (runs -0.05..-0.10).
bench_jt verify 32: tlbcold 1 k61a 16.60 k61b 16.19 | cand 15.59 candb 15.39 (-0.9 mean); tlbcold 0 14.95 14.80 | 14.63 14.52 (-0.30).
  (cross-job: LAY alone was -0.404 nsys in j2, LAY + TP -0.264 here: TP may cost ~+0.1 in warm nsys; judge-mode cold pairs: -0.9 total.)
Traces (B 32, 8 reps, med parts 0-2 / part 3; us from the first CTA start), base0 (= k60) -> cand:
  part 3 merged 9.80 -> 9.69, published 10.30 -> 9.96 (merged -> published 0.50 -> 0.27: full-sector stores issue twice as fast);
  probe RTT 0.60-0.67 -> 0.42-0.47; rows of the last round back 11.42-11.65 -> 10.97-11.02; fold of that round done 11.82-12.02 ->
  11.38-11.42 (0.4 us; part 3's single cold fold 0.67-0.77); all-in 11.90-12.09 -> 11.48-11.57; out 12.33-12.50 -> 11.95-12.05 (epilogue
  0.4-0.47); reset issued 12.58-12.75 -> 12.26-12.36. Poll rounds parts 0-2: 2.30 / 2.45 / 2.53 -> 1.96 / 2.18 / 2.27, part 3 1.16.
### Job w3_j5 (TP options on top of the freeze, script dev/w3/j5.sh) - DECISION RULE (written before the job)
qa = qb = deliver_r1/build.py dev/coord/k61.py --tpe 2 --tpc 1 (md5 ca76e9e4): e3 TP after the page-table entry loads (d32s), x6 TP in the
prefix on top of EA (v16s); w4 == cand. check.py on qa first (56 / 0 needed for any keep).
 d32s TPE2: KEEP iff both q copies <= min(k61a, cand) - 0.05 under tlbcold 1 and mean(q) <= mean(k61a, cand) + 0.03 under tlbcold 0.
 v16s TPC: KEEP iff both q copies <= cand - 0.05 under tlbcold 1 and mean(q) <= cand + 0.03 under tlbcold 0.
 A KEEP -> deliver_r2 = deliver_r1 builder with that knob on (the measured qa file if both keep).
### Job w3_j5 RESULT (17:30:04-~17:33, ~3.5 GPU-min; dev/w3/out_j5.txt; GPU TOTAL ~25.2 of 30 -> STOPPED)
check.py qa 56 / 0. bench_jt (paired-ratio us): d32s tlbcold 1 k61a 12.58 cand 11.96 | qa 12.03 qb 11.81; tlbcold 0 10.16 10.29 | 10.40 10.35
 -> TPE mode 2 NOT kept (qa above min control - 0.05; warm +0.15). d32s cold penalty (~+1.8) is not a K/V/pt/out page-walk effect here
 (both TP placements fail; the two d32s controls differ by 0.6 cold: noisy). v16s tlbcold 1 cand 11.29 | qa 10.94 qb 11.07 (-0.28);
 tlbcold 0 cand 10.45 | 10.53 10.58 (+0.11) -> TPC NOT kept by the pre-registered warm guard (2nd time: j3 cold -0.6 / warm +0.13).
 Coordinator option: deliver_r1/build.py --tpc 1 (x6 TP on top of EA, check.py 56/0 on qa): judge-mode (cold pairs) -0.3..-0.6 us v16s,
 warm +0.1. deliver_r1 unchanged.
