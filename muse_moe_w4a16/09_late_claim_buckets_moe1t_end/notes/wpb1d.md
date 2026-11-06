# WP-B1d: BCPS x early in-step start at bs1 (moe1t), base dev/moe_v1502.py

Worker WP-B1d, 2026-10-31/11-01. Namespace v2000-v2099, dev/wpb1d_*. 5 Modal jobs (one H200 via dev/modal_gate.sh).
Logs: scratchpad wpb1d/logs/run1..5.log; bench JSON and trace dumps dev/remote_out/wpb1d_*.

**Bottom line.**
- The planner's synergy hypothesis is refuted. It said BCPS makes the consumers ~10% faster than HBM, and an early,
  in-step start (ZA fast token-row load + S0DLY stage-0-first) then lets them run ahead of the stream. Every BCPS
  build is slower than the v1502 controls in both swapped-order passes. BCPS alone is 0.990 (mean of 6 passes). The
  BCPS + early-start builds are 0.959..0.979. With FLATE 2 they reach 0.992..0.994. The same early start without BCPS
  (BL4 + S0DLY + FLATE 2) is 0.990.
- Why: the per-stage gain is real. In valid traces the consumers release their slots 0.67 us earlier by the end at
  ne 3, and the bench gains +1..2% at ne 4/5. But the cross-CTA exchange (the fetchers' polls of the other CTAs'
  intermediate words) eats it. With faster median CTAs, the first poll of an expert more often runs before the slowest
  CTA's items are out, so about one extra round is needed (ne 2: 3 rounds vs 2). Each round costs 1.3-2 us under the
  weight stream. That puts the exchange on the critical path: ne 2 -5%, ne 3 flat. The scored median sits in the ne-3
  block.
- Fixes tried:
  - Retiming the first polls (F0DLY 500 + FDLY 800) removes the extra round and recovers about half of the loss
    (0.996).
  - A 3-slot ring (fewer bytes in flight, MOE1T_NS3) does not shorten the rounds and loses 2.2% by itself.
- Nothing banked. dev/moe_v2000.py / v2001.py default builds are SASS-identical to v1502 in all 16 kernels.

## 1. Files

- `dev/moe_v2000.py` = v1502 + dev/moe_v1700.py's MOE1T_BCPS (=1 both phases, =2 phase B only) and MOE1T_BCPS_ZA.
  - Merged with v1310's FXSS / CTRE / OLATE. FXSS's max |h| is taken from the raw fetched words, before the 2^48
    scaling of the stored intermediate. v1309's BL4 / S0DLY / FLATE / F0DLY / FDLY switches stay available.
  - With BCPS, MOE1T_BL4 is refused by #error. BL4's lanes-0-3 rule does not match the BCPS columns (the data lanes
    are 0, 9, 18, 27 = g == 2 tig).
  - MOE1T_BCPS_ZA is the BCPS analog of BL4: the other lanes read a 256-byte zero area. That makes the token-row load
    conflict-light, so a CTA starts at its stage-0 landing.
  - #error also for BCPS with XN / COLS / PIPE.
- `dev/moe_v2001.py` = v2000 + MOE1T_NS3 (host switch: the bs1 call runs on the 3-slot ring kernel moe1t_3).
- SASS (judge NVRTC 13 wrapper sass13.sh):
  - Default builds of v2000 and v2001 = v1502 in all 16 kernels (moe1t_4 b3ca7bbf93c3).
  - BCPS=1 moe1t_4 6f8cf7f674bc: 2600 instructions, 155 registers, no spills.
  - BCPS=1 + ZA: ce6b0bd056d5. + ZA + S0DLY 400 + FLATE 2: e86bf474845c. BCPS=2: 3056 instructions / 164 registers.
  - Stage loop (HMMA loop span): v1502 620 instructions (128 HMUL2); BCPS 590, with no HMUL2 in the stage body (its
    64 HMUL2 are the once-per-call token-row scaling) and 16 FFMA.
  - MOE1T_NO_FDLY,MOE1T_FDLY=500 reproduces the baked SASS exactly (the escape is a clean override).
- Scripts:
  - `dev/wpb1d_run1..5.sh`: Modal jobs.
  - `dev/wpb1d_sum.py`: ratios vs the two controls, per ne.
  - `dev/wpb1d_tana.py`: TRLITE + TOBS dumps. Per-stage landing p10/p50/max, copy latency,
    hold = issue(it + 4) - land(it), and the consumers' lag behind the data front.
  - `dev/wpb1d_tcmp.py`: TRLITE dumps side by side (milestones, per-stage issue p50/max).

## 2. Correctness (Modal H200, dev/wpb1c_check.py)

- Builds checked: c (v1502); bc, bcs4, za, zas4, zasf, blsf (run 1); n3, bcn3, zasfn3, bs4n3 (run 4); bcF5D8, zasF5D8,
  cF5, bcF5D11 (run 5).
- check_mt M 1 2 (normal + skewed, 3 seeds, 2 calls): 0 failures. Max rel-L2 = v1502's (M1 2.73e-3 / 2.58e-3, M2
  3.23e-3 / 3.27e-3).
- 40 sequential bs1 calls: 0 failures (max 2.62e-3).
- BCPS outputs vs v1502: max |diff| 3.9e-3, one BF16 ulp at 0.5-1 (15/40 bit-equal). The non-BCPS switches are
  bit-equal (40/40).
- hang1 smoke ok for every new build, trace builds included. The register-pinned TOBS build (416 threads x 152
  registers) fails to launch (CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, run 2).

## 3. bench2 (jm seeds, B2_SHARE_EQ, 100 reps; pass A bs1 + bs8 placebo, pass B bs1 in reversed order)

Ratio = median paired ratio (> 1 = faster). "vs ctl" = the ratio vs c divided by the geometric mean of the two v1502
copies' ratios. Builds are dev/moe_v2000.py (v2001 for NS3) with the listed defines.
D8 / D11 = MOE1T_NO_FDLY,MOE1T_FDLY=800 / 1100 (the baked FDLY is 500).

| run | variant | defines | A vs ctl | B vs ctl | mean | ne 2 / 3 / 4 / 5 vs ctl (A; B) |
|---|---|---|---|---|---|---|
| 1 | cb | v1502 copy | 0.9957 | 0.9990 | 0.9973 | |
| 1 | bc | MOE1T_BCPS=1 | 0.9856 | 0.9910 | 0.9883 | .953 .997 1.005 1.014; .953 1.003 .992 1.009 |
| 1 | bcs4 | BCPS + S0DLY=400 | 0.9800 | 0.9925 | 0.9862 | .952 .986 1.014 1.023; .962 1.006 .996 1.004 |
| 1 | za | BCPS + ZA | 0.9550 | 0.9631 | 0.9590 | .907 .977 .988 .998; .884 .970 .980 .961 |
| 1 | zas3 | BCPS + ZA + S0DLY=300 | 0.9786 | 0.9788 | 0.9787 | .923 .990 .998 1.004; .938 .998 .996 .967 |
| 1 | zas4 | BCPS + ZA + S0DLY=400 | 0.9815 | 0.9734 | 0.9774 | .943 .993 .998 1.024; .932 .988 .997 .982 |
| 1 | zas6 | BCPS + ZA + S0DLY=600 | 0.9665 | 0.9826 | 0.9745 | .906 .979 .982 1.021; .944 .998 1.000 .998 |
| 1 | zas8 | BCPS + ZA + S0DLY=800 | 0.9718 | 0.9705 | 0.9712 | .934 .982 1.009 1.004; .919 .986 .979 .995 |
| 1 | zasf | BCPS + ZA + S0DLY=400 + FLATE=2 | 0.9919 | 0.9956 | 0.9938 | .952 .997 1.017 1.012; .955 1.001 1.009 1.000 |
| 1 | zaf | BCPS + ZA + FLATE=2 | 0.9922 | 0.9914 | 0.9918 | .953 .997 1.016 .999; .953 1.001 .999 .988 |
| 1 | blsf | BL4 + S0DLY=400 + FLATE=2 (no BCPS) | 0.9907 | 0.9886 | 0.9896 | .983 .985 1.000 1.000; .983 .995 .995 .982 |
| 4 | cb | v1502 copy | 1.0040 | 1.0023 | 1.0032 | |
| 4 | n3 / n3b | MOE1T_NS3 (2 copies) | 0.9780 / 0.9752 | 0.9785 / 0.9802 | 0.978 | all ne -1.5..-4% |
| 4 | bc | MOE1T_BCPS=1 | 0.9966 | 0.9930 | 0.9948 | .961 1.003 1.019 1.023; .962 .999 1.009 1.001 |
| 4 | bcn3 | BCPS + NS3 | 0.9656 | 0.9672 | 0.9664 | .940 .976 .981 .997; .944 .976 .970 .980 |
| 4 | bs4n3 | BCPS + S0DLY=400 + NS3 | 0.9654 | 0.9677 | 0.9665 | |
| 4 | zasfn3 | BCPS + ZA + S0DLY=400 + FLATE=2 + NS3 | 0.9989 | 1.0023 | 1.0006 | .956 1.009 1.012 1.014; .960 1.013 1.010 .998 |
| 5 | cb | v1502 copy | 1.0013 | 1.0013 | 1.0013 | |
| 5 | bc | MOE1T_BCPS=1 | 0.9872 | 0.9856 | 0.9864 | .951 1.001 .985 .986; .948 1.002 1.007 .974 |
| 5 | bcF3 | BCPS + F0DLY=300 | 0.9961 | 0.9942 | 0.9951 | .967 1.008 1.007 .969; .972 .998 1.009 .986 |
| 5 | bcF5 | BCPS + F0DLY=500 | 1.0002 | 0.9883 | 0.9942 | .966 1.005 1.016 .953; .966 .995 1.007 1.028 |
| 5 | bcD8 | BCPS + D8 | 0.9949 | 0.9937 | 0.9943 | .951 1.003 1.007 .962; .972 .994 1.011 1.027 |
| 5 | bcF5D8 | BCPS + F0DLY=500 + D8 | 1.0020 | 0.9908 | 0.9964 | .982 1.003 1.007 1.006; .964 .994 1.006 1.032 |
| 5 | bcF5D11 | BCPS + F0DLY=500 + D11 | 0.9902 | 0.9980 | 0.9941 | .955 1.003 .999 1.003; .969 1.002 1.004 1.025 |
| 5 | cF5 | F0DLY=500 (no BCPS) | 1.0047 | 0.9934 | 0.9990 | 1.005 1.003 .992 1.008; .989 .994 .990 1.022 |
| 5 | zasF5D8 | BCPS + ZA + S0DLY=400 + F0DLY=500 + D8 | 0.9824 | 0.9773 | 0.9799 | .939 .984 .982 .999; .955 .991 1.006 1.017 |

- Control spread: cb vs c is 0.991..1.008 per pass, so single-pass differences under ~1% are noise.
- BCPS alone over 6 passes vs ctl: 0.9856, 0.9910, 0.9966, 0.9930, 0.9872, 0.9856 (mean 0.990).
- bs8 placebo (moe1t is bs1-only): every build 0.992..1.003 vs c.
- Per-ne times (run 1 pass A, us):
  - c: 17.89 / 23.68 / 30.34 / 37.31 (ne 2 / 3 / 4 / 5).
  - bc: 18.75 / 23.94 / 30.05 / 37.07.
  - zasf: 18.78 / 23.90 / 29.87 / 36.56.
  - The BCPS signature is the same in every BCPS build, with or without an early start: ne 2 +0.9 us, ne 3 +-0.2,
    ne 4/5 -0.3..-0.8.
- The early start (ZA) adds WP-B1c's known BL4-type ne-2 loss on top. FLATE 2 removes that part. S0DLY 300..800
  does not.
- zasfn3 (1.0006, one job) is the only BCPS build at the control level. It is not a +1% win and was not pursued
  (its NS4 twin zasf is 0.994 over two passes).

## 4. Traces

### 4a. Trace-build validity
- TOBS (the landing observer, warp 12) makes moe1t 416 threads. ptxas then caps the kernel at 128 registers (it rounds
  the block to 512 threads).
- BCPS needs 155 registers. Its run-1 TOBS builds spilled and ran +1.8 us slower than the bench build (bc ne 3 out
  25.6 vs 23.8 us), so they are not valid.
- MOE1T_MAXREG=152 compiles without spills but fails to launch (out of resources).
- Valid BCPS traces are therefore TRLITE-only: 384 threads, 160 registers, producer issue times + milestones, no
  landing stamps (runs 3-5, 9 reps x seeds 73235 / 72563 (ne 3), 70055 (ne 2)).
- The v1502 control's TOBS build (118 registers, no spills) is usable.

### 4b. v1502 control, TRLITE + TOBS (run 1; seed 73235 ne 3; medians over CTAs and reps)
- Stage 0 lands at p10 / p50 / max 2.27 / 2.69 / 3.71 us. After that the data front lands about every 1.03 us (the
  last stage at 19.81 p50 / 20.06 max).
- Hold grows from 1.9 to 2.9 us.
- c-done is 22.05 (p50) / 22.66 (max).
- c-done - land(last stage) is 2.30 us for the median CTA and 2.72 us for the CTA that arrives last.
- So the consumers, not HBM, trail the data front by 2.3-2.7 us at the end in every CTA. That is the room the
  hypothesis wanted to use.

### 4c. Where BCPS's gain goes (TRLITE, run 3; ne 3 seed 73235, c vs bc vs zasf)
- The consumers do run ahead:
  - Stage 17's copy (slot release) is issued at 17.89 vs 18.59 us.
  - p-done 18.05 vs 18.72; c-done 21.57 vs 22.05.
- The exchange gets slower: each expert's fetch from first poll to done, c vs bc:
  - q0 2.05 vs 3.62 us; q1 2.78 vs 3.87 us; q2 1.22 vs 2.24 us.
- The last expert's exchange now gates its B stages: f2ok max 19.74 = B2fr 19.74.
- The CTA tail widens: arrive max - c-done p50 is 1.31 vs 1.12 us.
- So 'out' improves by only 0.32 us in the trace build, and the bench build shows ne 3 flat.
- zasf: same c-done (21.60), but arrive max 23.23 (the early start widens the spread, as BL4 did), and out 24.06 = c.

### 4d. ne 2 (seed 70055), c vs bc
- The A(0) items are out at the same time (A0st max 7.81 / 7.78).
- But f0ok is 9.76 vs 11.01 (max 11.74), and B(0) now waits on it (B0fr 11.74; c had 1.9 us of slack).
- B1fr 14.18 vs 15.07, c-done 16.13 vs 17.06, out 17.66 vs 18.59 (+0.93 us = the bench's +0.9).

### 4e. Mechanism: fetch rounds (slot 26 counter)
- The fetchers' total poll rounds, median CTA:
  - c: ne 2 2 (per rep 2-4), ne 3 3-6.
  - bc: ne 2 3 (2-4), ne 3 4-5.
  - zasf (FLATE 2): 2 / 3, but its first polls are ~1.6 us later.
- A round of 16 relaxed LDG.128 per lane costs 1.3-2 us under the weight stream (one-round fetches: 1.2-2.0 us).
- With BCPS the median CTA stores its A(q) items ~0.3 us earlier than the slowest CTA (0.2 in v1502) and polls
  right away. The first round then misses the slowest CTAs' words, and a whole extra round trip follows.

### 4f. Shallower ring (run 4, MOE1T_NS3; the "fewer bytes in flight per SM" hypothesis)
- The fetch durations barely change: n3 q0 1.92 vs c 2.05; bcn3 3.2-3.8 vs bc 3.5-3.8.
- The CTA spread widens (n3 ne 3 c-done max 22.56-22.88 vs 22.40-22.53).
- So the round latency is not set by the ring depth, and NS3 loses 2.2% by itself.

### 4g. Poll retiming (run 5, bcF5D8 = BCPS + F0DLY 500 + FDLY 800)
- The extra round disappears: ne 2 median rounds 2, f0ok 10.53 vs bc 11.01.
- ne 2 out is 17.92: still +0.26 vs c (B1fr 14.46 vs 14.18; arrive max - c-done p50 0.61 vs 0.51).
- ne 3 out 23.81-23.94 vs c 24.06-24.13 in the trace, flat in the bench.
- The retimed first poll costs what it saves, and the faster CTAs still widen the tail.
- F0DLY 500 alone on the base (cF5) is neutral (0.999).

## 5. Conclusions for the next bs1 attempt

- Faster consumers alone do not convert at bs1. Neither does a fast token-row load or stage-0-first ordering, nor
  BCPS combined with either.
- The call is gated by the cross-CTA exchanges: the fetch of each expert's intermediate, whose 1.3-2 us poll round
  trips under the stream are the expensive unit, and the end exchange. It is also gated by the slowest CTAs.
- Any speed-up of the median CTA moves its polls ahead of the slowest CTA's stores and costs a round. Moving the
  polls later costs the same time back.
- The lever this points at is the exchange itself: one-round fetches without timing guesses. For example:
  - a per-expert arrival count (one relaxed counter per expert, polled with a single load) that tells the fetcher
    when every CTA's items are out, so it issues exactly one data round;
  - or pushing the intermediate into consumers' buffers instead of pulling it.
- BCPS's per-stage gain (+1..2% at ne 4/5) only becomes reachable once the exchange is cheaper than one round. The
  per-SM / per-TPC tail (WP-BCPS 6c) remains on top of that.
- Tools for that: dev/wpb1d_tcmp.py (TRLITE milestone / issue comparisons). For BCPS builds use TRLITE without TOBS
  (TOBS + BCPS spills or does not launch).
