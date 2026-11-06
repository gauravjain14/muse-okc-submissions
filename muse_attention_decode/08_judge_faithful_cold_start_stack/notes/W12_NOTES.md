# W12: START of the B16/32 sliding kernels (d16s xe_dm_8_4_2, v16s attn_8_1_1_1x6, d32s attn_1_1_1_1e3, v32s attn_8_1_1_1w4)
Base dev/coord/k67.py (md5 50806e22). Write only dev/w12. GPU cap 35 min (one slot via tools/gate.sh). Local NVRTC 13 venv for SASS.

## Resume here
- FROZEN dev/w12/deliver_r1/ (build.py defaults --r8s 3 --e3cd 1 on k67 -> k67w12.py 7396fd3a; README.txt): v16s x3
  (-0.46 / -0.49 nsys, bench_jt verify 16 ~-0.5) + d32s E3CD (-0.064 nsys). j5 PASSED (check.py 56/0 on the
  final file, t_rep v16s / d32s / v32s 60/0 each); g3 (R8SG 3 on w4) +0.356 -> dead. GPU used ~29.5 of 35. DONE.
- Dead here: R8Q (prefix-before-length fence, +0.30 / +0.16), E3S (length first in e3, 0.00), q0 (WH_QEARLY 0),
  p4 (WH_PH1_4 4), x4 (R8S 4, -0.10 < x3), EVF (x6 -0.02, w4 +0.01, e3 +0.91), split on w4 (R8SG 2 +0.40).
- Tools: dev/w12/build.py (research knobs), trace.py (W12 clock64 stamps; traced builds overstate A/B), ifp.py (i-fetch
  probe: code is NOT cold after the flush), nsparse.py, nsys_go.py / nsys_multi.py, sassv.py; local scratch SASS tools
  pre.py / loopdiff.py / kdiff.py in <scratchpad>/w12.

## Local SASS facts (k67, NVRTC 13, sm_90a)
- x6 (v16s): seq LDG at instr 10, FIRST USE of seq at instr 106 (SHF sign-extend), i.e. 6 of 8 Q loads, kraw/vraw,
  ksc/vsc and all 6 page-table loads (instr 240-251) are issued only after the length lands; qprep (sincos, bar 14,
  Q frags, wg_setup/ones, fence; ~270 SASS) sits between the pt loads and the first LDGSTS (instr 542).
- w4 (v32s): seq LDG at 22, first use at 95; 5 Q loads + kraw/vraw/crow + pt loads after it.
- e3 (d32s): seq LDG at instr 120 (after warp 15's mbarrier-init block, the gnp indexed LDC, Q/new-token address math).
- xe_dm (d16s): compute warps' seq LDG at instr 42, used at 44, pt LDGs 66/74, first K LDG 115: lean already.
- Old traces (WH r2, k26-era run8w): start 0.13, prefix end 0.70, length landed 1.34, Q frags 2.27, pt landed <= 2.34,
  K/V issued 2.99, tile0 3.97 (v16s); d16s (W7 j1, k65): pt issued 0.53, K/V issued 1.12, QK0 2.18, out 5.22.

## Knobs (dev/w12/build.py; 0 = base cubin, checked: e61d0c0f == k67 monolith)
- R8Q (run8w x6 / w4): prefix loads before the first length use. 1 = never-taken trap branch on p.B after the prefix
  (SASS: all Q / scale / new-block loads before the first seq use, but the seq LDG drifts to instr 122 / 91);
  2 = cp.async + smem read-back (ptxas SINKS every Q load below the DEPBAR + BAR: dead); 3 = source order only (= base
  schedule: useless); 4 = seq load at the kernel top + trap fence after it (p.seq_lens == 0) + fence 1's p.B trap after
  the prefix: x6 seq LDG 9, Q loads 84-151, fence 225, pt LDGs 277-282 (base 240-251 after the length landed), first
  LDGSTS 551; w4 seq 10, Q 99-159, pt 277-295. Hot unit loop: x6 320 = 320 instr (12 opcode reorders), w4 identical.
- E3S (runE1 e3): length load first: 1 = + trap fence, 2 = no fence (ptxas keeps it at instr 20 anyway; base 120).
  Step loop 666 instr, 26 opcode reorders.
- --tr 1: W12 stamps (K2_TRACE only): run8w w2 length consumed (dep pos0), w4 Q frags done, w3 pt landed (dep ent[0],
  issue start), w5 issue done, w6 first tile landed, w7 first unit done, w8 WG done (tid 0); e3 w2 length consumed
  (tid 256), w3 pt landed, w4 first half-tile issued, w5 all issued, w6 Q^T ready (tid 0), w7 first pair landed,
  w8 steps done, w9 out stored. Clock64 per CTA, calibrated by globaltimer (slots 0/1 entry, 30/31 end).

## Job w12_j1 - DECISION RULE (written before the job; dev/w12/j1.sh)
c4 = R8Q 4 + E3S 2 (both copies identical). nsys D = mean(4 medians) - mean(4 k67 medians), per-run vs same-pass ctrl.
- R8Q KEEP per shape (v16s, v32s) iff D <= -0.08 AND all 4 runs below their same-pass control mean AND bench_jt
  tlbcold 1: both c4 copies <= min(k67a, k67b) on that load.
- E3S KEEP (d32s) iff D <= -0.05 AND >= 3 of 4 runs below AND bench_jt tlbcold 1 both c4 copies <= min(k67a, k67b).
- Trace = diagnostic (where the chain lands; no decision). Any KEEP -> j2 freeze gate (check.py, t_v8w, t_e3, t_rep).

## Job w12_j1 RESULT (00:44-00:49, 247 s job wall, ~5.5 GPU-min; dev/w12/out_j1.txt; 4 MiB node, driver 610.57.04)
nsys (k67a/b vs c4a/b, 2 passes): v16s k67 9.416 | c4 9.716 (+0.300, 0/4 below) -> R8Q 4 KILLED; v32s 12.812 | 12.972
(+0.160, 0/4) -> KILLED; d32s 9.292 | 9.288 (-0.004, 2/4) -> E3S not kept (neutral). bench_jt tlbcold 1 (k67a c4a k67b c4b
k67c): d32s 12.43 11.79 11.56 11.69 11.52; v16s 11.02 10.89 10.68 11.09 10.86; v32s 15.45 15.53 15.06 15.34 15.33 (noise).
Trace (us from the first CTA start, median CTA, warm rep1; base b | R8Q4+E3S2 c):
- v16s x6: prefix end (t17) 0.80 | 0.86, length consumed (w2) 1.30 | 1.32 (!: the length lands no earlier when issued at
  instr 9), Q frags (w4) 2.25 | 2.26, pt landed / issue start (w3) 2.36 | 2.38, issue done (w5) 3.10 | 3.12, first tile
  landed (w6) 4.07 | 4.07, first unit done (w7) 4.86 | 4.83, WG done (w8) 6.55 | 6.54, out (t9) 9.11 | 9.02.
- v32s w4: t17 0.98 | 1.25, w2 1.19 | 1.45, w4 2.67 | 2.95, w3 2.81 | 3.08, w5 4.85 | 5.13 (issue of 128 KB/CTA takes
  2.0 us: DRAM backpressure), w6 5.25 | 5.50 (tile 0 lands 2.4 us after the issue starts), w7 5.94 | 6.22, w8 8.62.
- d32s e3: prefix (t20) 0.40 | 0.38, length consumed by producer warp 8 (w2) 0.70 | 0.67, pt landed (w3) 1.42 | 1.39,
  first half-tile issued (t17) 2.74 | 2.74, first pair landed (w7) 4.23 | 4.22, steps done (w8) 6.90 | 6.95, out 8.78.
  (w4/w5 not recorded: stamps sat in the inactive non-WS_PD copy loop.)
- Clock 1.77-1.82 GHz. Free (cold-TLB) rep0 ~ warm on this node (no penalty node).
READING: run8w's start does not follow the DRAM trips. Base x6 has ~440 SASS between the first length use and the first
LDGSTS, executed by all 16 warps, and ~540 from entry (8.7 KB); a ~0.25 us per KB cold-code cost (WH job22's number)
reproduces t17 / w2 / issue start (0.8 / 1.3 / 2.3 us). Hypothesis H-IF: the run8w start is instruction-fetch /
issue bound after the L2 flush, not trip bound -> test with a microbenchmark (ifp) and code-order variants.

## Job w12_j2 - DECISION RULE (written before the job; dev/w12/j2.sh)
Variants (local SASS: no new local-memory ops; first LDGSTS x6 542 -> 309 for q0 / s2): q0 = WH_QEARLY 0 (existing knob:
Q prep after the K/V issue); s2 = R8S 2 (tiles 0-1 issued, then Q prep, then tiles 2..T-1); p4 = WH_PH1_4 4 (v32s:
tiles 4-7 issued by their stream one tile ahead). nsys 2 copies x 2 passes vs k67a/b on v16s, v32s.
- q0 / s2 KEEP per shape iff D <= -0.08 AND 4/4 runs below their same-pass control mean; both pass -> the larger gain,
  q0 (no new code) if within 0.05. p4 KEEP (v32s) iff D <= -0.08 AND 4/4 below; p4 + (q0|s2) both kept -> combine in j3.
- H-IF (ifp) confirmed iff (cold - warm) >= 0.15 us per KB for the >= 8 KB bodies.
- Any KEEP -> j3: freeze gate (check.py, t_v8w both defs, t_rep v16s/v32s 60, bench_jt tlbcold 1 with a sink slot).

## Job w12_j2 RESULT (01:2x, 310 s job wall, ~6.5 GPU-min; 122 MiB node; dev/w12/out_j2.txt). GPU total ~12.5 of 35.
- ifp probe: H-IF REFUTED. Straight-line FFMA bodies 2 / 8 / 32 / 64 KB: cold (after the 512-MB flush) - warm =
  +0.04 / +0.03 / +0.04 / -0.11 us (<= 0.02 us per KB): code is not re-fetched cold after the flush. Cold single LDG at
  entry 816 cycles median (0.47 us) / p90 1248 (0.71); L2 hit 390 (0.22).
- nsys 2 x 2 vs k67a/b (9.392 / 12.896): v16s s2 9.136 (D -0.256, 4/4 below) KEEP; q0 +0.088; p4 +0.024.
  v32s s2 +0.400 (0/4), q0 +0.608, p4 +0.128: all KILLED on w4.
- Trace (traced builds, warm rep1, base | s2): v16s issue start w3 2.36 | 1.58, first tile w6 4.12 | 3.29, first unit w7
  4.88 | 4.03, WG done 6.59 | 5.83, out 9.31 | 8.32; v32s w3 2.83 | 1.95, w6 5.36 | 4.97, out 12.35 | 11.74. (Traced builds
  overstate: production v16s -0.26, v32s +0.40.) No-flush mode: base v16s w2 1.00 / w3 2.04 / w6 3.03 (seq consumed at
  ~1.0 us even with L2-warm inputs: the length is not the start limiter in run8w; issue-bound prefix + Q prep are).
READING: issuing the streams' first tiles before the Q prep (all 16 warps run ~440 SASS of prefix + Q prep before the
first LDGSTS in k67) moves the x6 first-tile landing ~0.8 us earlier in the trace; production v16s -0.26 us. Issuing ALL
tiles first (q0) makes the Q prep's LDS wait behind the LDGSTS: worse. w4 loses with the same edit (+0.40): kept off.
- R8S made per kernel: W12_R8S (x6) / W12_R8SG (w4). x2 = --r8s 2: only attn_8_1_1_1x6 / attn_8_1_0_1x6 differ from k67
  (local NVRTC 13, raw SASS per kernel), and x2's x6 == the measured s2's x6 (identical text).

## Job w12_j3 - DECISION RULE (written before the job; dev/w12/j3.sh)
- FREEZE x2 iff tools/check.py 56 PASS / 0 failures AND t_v8w seeds 1 (WH_ALLB; WH_ALLB,WH_CLU=0) 0 FAIL AND t_rep v16s
  60 replays 0 failures / no hang AND nsys v16s D <= -0.08 with 4/4 runs below (confirmation) AND bench_jt tlbcold 1
  verify 16: mean(x2a, x2b) <= mean(k67a, k67b) + 0.05.
- x1 / x3 replace x2 iff their nsys D <= D(x2) - 0.05 with 4/4 below (then t_v8w on it in the next job).

## Job w12_j3 RESULT (01:3x, 344 s job wall, ~6.6 GPU-min; 4 MiB node; dev/w12/out_j3.txt). GPU total ~19 of 35.
- x2 gates: tools/check.py CHECK_RC 0, 56 PASS, 0 failures; t_v8w seeds 1 WH_ALLB 48/0, WH_ALLB,WH_CLU=0 48/0; t_rep v16s
  60 replays 0 failures (no hang).
- nsys v16s 2 x 2 vs k67a/b (9.408): x1 -0.256, x2 -0.248, x3 -0.456 (runs -0.42..-0.50), all 4/4 below.
  -> x3 (R8S 3: tiles 0-2 before the Q prep) replaces x2 (beats it by 0.21 >= 0.05); needs t_v8w (j4).
- bench_jt tlbcold 1 verify 16: k67a 11.25 | x2a 11.21 | k67b 11.56 | x2b 11.14 | k67c 11.35 -> x2 -0.23 vs k67a/b.
- Coordinator 01:3x: add an L2::evict_first variant (v16s / v32s / d32s) -> j4 (WH_EVF 1 + new E3EVF).

## Job w12_j4 - DECISION RULE (written before the job; dev/w12/j4.sh). Stack base x3 (R8S 3, x6 only).
ef = x3 + WH_EVF 1 (run8w x6 + w4 K/V copies L2::evict_first) + E3EVF 1 (e3 K/V copies evict_first; e3 step loop
re-scheduled: 509/667 normalised-equal); cd = x3 + E3CD (e3 __cluster_dims__(3), no launch attribute: the 5 runtime
cluster checks (LDC c[0x140] + branch) gone; step loop 636/666); x4 = R8S 4 (v16s screen only).
- x3 FREEZE (v16s) iff t_v8w seeds 1 both defs 0 FAIL AND nsys v16s D <= -0.08 4/4 below (re-confirmation). (check.py /
  t_rep on x2 cover the same code path; check.py re-run on the final stacked file.)
- x4 replaces x3 iff D(x4) <= D(x3) - 0.05 with 4/4 below.
- EVF per kernel: x6 KEEP iff ef - x3 (nsys v16s) <= -0.05 with >= 3/4 runs below the same-pass x3 mean; w4 KEEP iff
  ef - k67 (v32s) <= -0.05 >= 3/4; e3 KEEP iff ef - k67 (d32s) <= -0.05 >= 3/4; AND for each, bench_jt tlbcold 1 mean(ef)
  <= mean(k67a, k67b) (+ the x3 part on verify 16: compare ef with cd there, cd = x3's x6).
- E3CD KEEP iff t_e3 0 failures AND nsys d32s D <= -0.05 >= 3/4 below AND bench_jt tlbcold 1 decode 32 mean(cd) <= mean(k67a, k67b).

## Job w12_j4 RESULT (01:5x, 371 s job wall, ~7 GPU-min; 4 MiB node; dev/w12/out_j4.txt). GPU total ~25.5 of 35.
- t_v8w x3a seeds 1: WH_ALLB 48/0, WH_ALLB,WH_CLU=0 48/0. t_e3 on cd: 0 failures (C=3 cases; 3 SKIP = routed elsewhere).
- nsys 2 x 2 vs k67a/b (v16s 9.404 | v32s 12.848 | d32s 9.388):
  x3: v16s -0.488 (4/4; confirms j3 -0.456), v32s -0.020, d32s +0.004 (w4 / e3 code == k67). x4 (R8S 4): v16s -0.104.
  cd (x3 + E3CD): d32s -0.064 (4/4; vs x3 -0.068 4/4), v16s = x3, v32s +0.032.
  ef (x3 + WH_EVF + E3EVF): v16s -0.024 vs x3 (3/4, below the 0.05 bar), v32s +0.012, d32s +0.908 (0/4!).
- bench_jt tlbcold 1 (k67a ef_a cd_a k67b ef_b cd_b k67c): decode 32 12.69 11.75 11.92 11.76 11.48 11.59 11.75;
  verify 16 11.07 10.36 10.24 10.68 10.13 10.41 10.68 (x3 part ~ -0.5); verify 32 15.04 14.77 14.86 14.73 14.71 14.79 15.03.
DECISIONS: x3 FROZEN (v16s). E3CD KEPT (d32s: nsys -0.064 4/4, bench_jt cd mean 11.76 <= k67a/b 12.23, t_e3 0).
EVF (coordinator's side test) NOT kept anywhere: x6 -0.024 (< 0.05), w4 +0.012, e3 +0.91 in nsys (bench_jt read it
-0.3: the two disagree; nsys rule fails). x4 not kept.
- FROZEN dev/w12/deliver_r1/: build.py (defaults --r8s 3 --e3cd 1) on dev/coord/k67.py -> k67w12.py (md5 7396fd3a); its
  monolith cubin == the measured j4 cda (6afc93a9; cda.py md5 aecab221 rebuilt locally); stripped k67w12s.py 863,324 B,
  stripped monolith cubin identical. Only attn_8_1_*_1x6 (R8S) and attn_1_1_1_1e3 (E3CD) differ from k67 (+ host _config).

## Job w12_j5 - DECISION RULE (written before the job; dev/w12/j5.sh)
- deliver_r1 final gate: tools/check.py 56 PASS / 0 failures AND t_rep (v16s, d32s, v32s) 60 replays 0 failures / no hang.
- g3 (R8SG 3, w4 only): KEEP iff nsys v32s D <= -0.08 with 4/4 below; then the gate runs on fin2 (= deliver + g3) +
  t_v8w WH_ALLB,WH_CLU=0 0 FAIL -> deliver_r2 = deliver_r1 + --r8sg 3.

## Job w12_j5 RESULT (02:15-02:19, 232 s job wall, ~4 GPU-min after ~20 min queued; 4 MiB node). GPU total ~29.5 of 35.
- fin.py == deliver_r1/k67w12.py (7396fd3a). tools/check.py CHECK_RC 0, 56 PASS, 0 failures. t_rep 60 replays each on
  v16s / d32s / v32s: 0 failures, no hang.
- g3 (R8SG 3, w4 only) nsys v32s 12.852 -> 13.208 (+0.356, 0/4): KILLED (w4 loses with every split tried: R8SG 2 / 3).
