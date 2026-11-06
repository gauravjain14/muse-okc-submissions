# W2 NOTES: verify full v16f / v32f (attn_xd_v8f; xd0 = XC_END=0 = v16f XD K-combiner protocol, xd1 = XC_END=1 = v32f FINAL)
Owner W2 (round 4, docs/ROUND4.md), dir dev/w2 only. Base dev/coord/k60.py (md5 152a09bb). GPU cap 30 min (job wall).
k60's XD module cubins (local NVRTC 13): xd0 68f4ec61..., xd1 a76db5a3... (= k56's; S4 notes).

## Resume here
- FINAL ROUND 2 (18:10): FROZEN dev/w2/deliver_r3/ = k64 + TP=1 (XD section = deliver_r2 + the early TLB touch):
  `python3 dev/w2/deliver_r3/build.py dev/coord/k64.py <out>` -> k64w2.py md5 b76344ec (== measured t64a.py; outside the
  XD section byte-identical to k64; xd0 d5806e8f, xd1 a07f4682; 853,388 B stripped, cubins unchanged). check.py 56 PASS /
  0 failures, xcheck auto 24 / 0 (job 11). bench_jt --tlbcold 1 (job 11, position sink): v32f -0.98 (both copies win),
  v16f -0.40 (mean). Round-2 GPU ~17 of 20 (j7 6, j8 3, j9 2.9, j10 2.5, j11 2.7). Round-2 dead: FF2, VB (invalid,
  hot-loop realloc), PL, SL, XD protocol at B 32.
- ROUND 2 (17:35, coordinator): deliver_r2 stacked in dev/coord/k63.py (md5 4c4ed5f0; its XD section == deliver_r2's,
  checked); +20 GPU-min; base k63. build.py now SPLICES: `python3 dev/w2/build.py dev/coord/k63.py <out>
  --xdfrom=dev/coord/k61.py <knobs>` patches k61's pristine XD section and replaces k63's (with the r2 knobs it gives k63's
  cubins: xd0 1a72f202, xd1 7d3f7fbb). Job 7 prepared (rules below); arms r<name>a/b.py = r2 knobs + the named knob.
- FINAL (17:27): GPU used ~25 of 30 min (j1 3.9, j2 6.5, j3 0 (spend limit), j4 6.0, j5 5.1, j6 3.7). Base k61.
- FROZEN dev/w2/deliver_r2/ (USE THIS): `python3 dev/w2/deliver_r2/build.py dev/coord/k61.py <out>` (defaults XD_W2PB=1
  XD_W2EE=1 XD_W2LT=1) -> k61w2.py md5 996fed8e (in the dir); xd0 cubin 1a72f202, xd1 7d3f7fbb (local NVRTC 13, 0 spills),
  839,585 B after strip_cuda_comments (cubins unchanged). check.py 56 PASS / 0 failures, xcheck auto / HYB 0 / HYB 1
  24 / 0 each (job 5). nsys vs k61 (job 4, 2 copies x 2 passes, all runs negative): v16f -1.59, v32f -0.40 (~+0.25 %).
- FROZEN dev/w2/deliver_r1/ (k61 + PB + EE1, superseded by r2): check 56 / 0, xcheck 0 x3 (job 4); v16f -0.62, v32f -0.19.
- TP (W4's cold-TLB touch) NOT kept: job 5 cold bench -0.8 / -1.3 us for lttp but its warm gates failed; job 6 (noisy slow
  node: cold v16f 100-103 vs 94 in job 5) did not confirm (TPKEEP no). cand_r4 (S1LT + TP=1) kept unfrozen for a re-test.
- Files: build.py (knobs, default == k60 cubins byte for byte, checked), sassw2.py (local NVRTC 13 compile + SASS +
  ptxas -v via a PTX nonce, md5 with judge options), trace.py (GPU: W2 stamps -> npz), tan.py (local analysis of the
  npz), decide.py (nsys table: D = mean of 4 variant medians - mean of 4 control medians; per-run vs same-pass ctrl),
  pick.py (in-job freeze pick), mk_deliver.py (deliverable dir from knobs), j1.sh / j2.sh / j3.sh, out_j1..3.txt.
  Arm files now in dev/w2: k60a/b (= k60), k60w (trace stamps, default build == k60), s1*, sp* (job-3 arms); job-1/2 arm
  files were removed (rebuild: python3 dev/w2/build.py dev/coord/k60.py <out> <knobs as in the job tables>).

## W2 trace stamps (build.py, compiled only with -DXD_W2TR; p.tr[c * 64 + k], pair tops at p.tr + 16384)
0 entry | 1 lengths scanned (warp 0) | 2 prefix barrier | 3 split / b_begin | 4 first-seg row loads issued | 5 Q / new
work done (= K/V prologue start) | 6 tile 0 copies issued | 7 prologue issued | 8 tile0 data (pair 0 top) | pair tops
per lane (all segments, running index) | 9-12 loop end WG 0..3 (last segment) | 13 new block done | 14 merge barrier |
15 election atomic returned (XC) | 16 role known (XC) | 17 FINAL lane-1 round-0 fetch issued | 18 segment end | 19 PLAIN
publish done | 20 FINAL round 0 | 32+r FINAL round r | 21 FINAL merged | 22 FINAL / chunked done (out issued) |
23 XD publish (fence, atom, fence) | 24/25 XD wait start q0/q1 | 26/27 XD combine start | 28/29 XD combined |
30 mid-kernel publish (XC). Values: 40 endrole | endb, 41 FINAL rounds, 42 XD flags | pb, 44 smid, 46 P / cf / E /
r_old / fin, 47 round-0 published mask, 48-51 segment info, 56-63 pair index / npair per segment.

## Job 1 (w2_j1) - DECISION RULES (written before the job)
Arms (nsys_go, shapes "16,8,full;32,8,full", per pass k60a k60b za zb la lb pma pmb xha xhb; 2 passes). D = mean of
the arm's 4 medians - mean of the 4 control medians (same job); "every run" = each arm median vs its same-pass control mean.
- Z (XD_W2Z: every segment ends after tiles + new block; no lane merge / partial / publish / combine / output): -D(Z) =
  nsys value of the whole end chain. If -D(Z) < 1.0 on a shape: NO structural end-chain work on that shape.
- L (XD_W2L: last pair step of each lane not computed, data still waited for): -D(L) = exposure of the last-pair
  compute. Pursue a split of the last pair (lanes / WGs) only if -D(L) >= 0.8 (a split recovers at most ~half).
- PM (XD_W2PM + XC_END forced on both shapes: the FINAL fetches + merges ONE of the partials published before its
  election, the later ones as today; counters unchanged = no hang risk): ideal pre-merge (collector) bound.
  v16f: build the collector protocol iff D(PM) <= -1.0 vs k60 (XD protocol) AND every run <= -0.5.
  v32f: iff D(PM) <= -0.6 AND every run <= -0.3.
- XH (HYB=1: XC_END forced, correct, = XC's measured-slower v16f option): reference. If XH v16f <= -0.3 in all 4 runs,
  it becomes a freeze candidate (hybrid threshold change) for job 2 (check.py + xcheck there).
- Controls: copies / passes disagree by > 0.5 on a shape -> that shape's cells are noise-flagged (no build on them).
- Traces (k60 both shapes 12 reps; xh v16f 8 reps): budget + end-chain anatomy; they explain, never decide.
- Kill: an arm that hangs / times out is dropped and reported.

## Job 1 RESULT (nsys medians us, 2 copies x 2 passes; D vs the 4 k60 controls)
| arm | v16f (ctl 90.50, spread 0.26) | v32f (ctl 168.86, spread 0.27) |
| Z (XD_W2Z=1) | -10.09 (all runs -10.0..-10.1) | -11.17 (-10.9..-11.4) | INVALID: the compiler deleted the tile math (no QGMMA left in the loop; the partial was dead) = a no-compute skeleton. Redone as Z2 (math kept live) in job 2.
| L (last pair not computed) | -1.72 (runs -1.2..-2.1) | -1.80 (-1.6..-1.9) | -> >= 0.8: last-pair lever open
| PM (XC_END forced + ideal pre-merge) | +0.51 (+0.41..+0.58) | -1.20 (-1.06..-1.39) | -> v16f NO (even an ideal collector loses to the XD protocol at B 16); v32f passes (<= -0.6, every run <= -0.3)
| XH (XC_END forced, correct) | +1.83 | -0.04 (same module) | -> no hybrid change
## Traced budget of k60 (W2 stamps, 12 reps, judge-style flush; us from the earliest CTA entry; min / p10 / med / p90 / max over CTAs, median over reps). Trace build runs ~2-2.5 us slower than nsys.
| phase | v16f | v32f |
| entry | 0 / .03 / .13 / .19 / .26 | 0 / .03 / .18 / .22 / .24 |
| lengths scanned (warp 0) | 1.09 / 1.09 / 1.12 / 1.15 / 1.18 | 1.10 / 1.14 / 1.15 / 1.17 / 1.20 |
| prefix barrier | 1.12 / 1.12 / 1.17 / 1.20 / 1.23 | 1.34 / 1.36 / 1.39 / 1.42 / 1.46 (VS2 split work) |
| split / b_begin | 1.47 / 1.50 / 1.55 / 1.62 / 1.65 | 1.44 / 1.46 / 1.49 / 1.52 / 1.55 |
| row loads issued | 2.16 / 2.22 / 2.27 / 2.34 / 2.40 | 2.08 / 2.10 / 2.16 / 2.19 / 2.24 |
| Q / new-token work done | 2.98 / 3.10 / 3.39 / 3.54 / 3.87 | 2.85 / 3.04 / 3.36 / 3.58 / 3.84 |
| tile-0 copies issued | 3.09 / 3.23 / 3.49 / 3.63 / 3.98 | 2.94 / 3.15 / 3.44 / 3.68 / 4.00 |
| prologue issued | 3.62 / 3.76 / 4.08 / 4.34 / 4.54 | 3.49 / 3.81 / 4.30 / 4.66 / 4.88 |
| tile0 data (pair 0) | 4.61 / 5.30 / 6.19 / 6.70 / 7.28 | 4.38 / 5.02 / 5.95 / 6.62 / 7.12 |
| steady pair step (per lane, 64 KB / CTA) | 1.97 (= 4.39 TB/s: at the gather ceiling) | 1.98 |
| last landing per CTA (med / max) | 83.07 / 86.46 | 159.33 / 165.12 |
| last-pair compute (loop end - last landing) | 1.60 | 1.38 |
| tiles done (any WG) | 75.8 / 80.3 / 84.5 / 86.1 / 87.9 | 152.7 / 156.7 / 160.5 / 163.1 / 166.3 |
| merge barrier | med 84.59, max 87.95 | med 160.64, max 166.38 |
| segment end (lane merge + partial) | +1.1 after the merge barrier | +1.1..1.6 |
| publish (XD: fence, atom, fence) | med 87.02 max 90.06 (+1.0 after segend) | PLAIN pub med 161.9 |
| combine done / FINAL done | XD combiners max 92.4..93.0 | FINAL done max 170.8 |
| kernel end (last stamp) | 92.91 | 170.82 |
| end chain (kend - tiles_max) | 4.69 | 4.38 |
- Lane drift: the two lanes of a CTA run independent rings; lane 0 lags lane 1 by +0.45 (pair 2), +0.6..0.9 (mid),
  +0.8..1.1 (end) median, p90 3.2..3.8 us at the end; |l0 - l1| of the last landings median 1.66 us.
- v32f critical CTA = the FINAL of its request in 12 / 12 reps (tiles -> done 5.3 us median): from its last landing:
  last pair +1.0..1.5 -> merge barrier -> election returned +0.1..0.6 (exposed only when lane 1 finishes last) ->
  role +0.12 -> lane-1 round-0 fetch issued +1.35 (ld.acquire cntP trip + lane barrier + issue) -> round 0 +0.64
  (fetch landed) -> merged +1.5 (weights + weighted sum, ~5 sources, serial unroll-1 loops) -> done +0.45 (epilogue).
- v16f (XD protocol) critical request: last publisher tiles -> its publish 2.1 (lane merge 1.1, fence + atom + fence
  1.0); waiting combiners done +1.8..2.1 after that publish, the last arrival's own combine +2.2..2.8 (its publish ->
  combine start gap 0.4..0.9 us: pb[] lives in LOCAL memory (dynamically indexed int pb[2], 8-B stack in xd0) and its
  LDLs after the MEMBAR feed both the publish atomics' addresses and the combine's request id).

## Job 2 (w2_j2) - DECISION RULES (written before the job)
Correct pieces (built, local NVRTC 13: 0 spills each; default knobs == k60 cubins):
- PB (XD_W2PB, xd0 only): pb[] -> smem misc[24 + q]; combine parameters (double divisions) at segment start. xd0 0 B stack.
- EE1 (XD_W2EE=1, xd1 only): S4's mask-free hot pair loop + peeled masked last pair without refill code; the FINAL
  election atomic issued before the last pair (result consumed after it: checked in SASS).
- FF (XD_W2FF + EE1, xd1 only): one counter word (low 16 = publish bits / count, high 16 = arrivals); the early election
  returns rank + published snapshot; the FINAL's lane 1 fetches the snapshot right after the merge barrier (acquire fence).
- FU (XD_W2FU, xd1): FINAL round loops unrolled over the NSL + 1 slots.  LB8 / LB16 (XD_W2LB, both): CTA barrier every
  8 / 16 pair steps (caps the lane drift).  ALL = PB + EE1 + FF + FU + LB16.
Bounds: Z2 (end chain removed, math kept live), FFUPM (= FFU + ideal pre-merge: fetch / merge one pre-election partial).
Order: xcheck (dev/xb/deliver_r15/xcheck.py) on ALL in auto mode and with K2_DEFS=XC_HYB=1 (XC_END on every case,
incl. B = 2 count-mode chunked path); any failure / timeout -> the FF / FU / EE arms are skipped (pb, lb, z2 still run).
nsys per pass: k60a k60b z2 pb ee1 ff ffu ffupm lb8 lb16 all (a and b copies), 2 passes.
- KEEP a piece iff D <= -0.10 on the shape whose module it changes, all 4 runs < the same-pass control mean, and the
  other shape's D <= +0.15 (byte-identical module there: placement noise only).
- Freeze candidate for job 3 = the kept pieces combined (ALL if every piece is kept and ALL <= each kept single on its
  shape + 0.10); job 3 = check.py + xcheck x3 + nsys 2 x 2 vs k60, then deliver_r1.
- Collector (pre-merge) build at v32f only iff D(FFUPM) - D(FFU) <= -0.6 with every run of FFUPM below FFU's same-pass mean.
- Z2: reported as the end chain's nsys value (no decision).

## Job 2 RESULT (nsys medians, 2 copies x 2 passes; xcheck of ALL: auto 24 PASS / 0 FAIL, XC_HYB=1 24 PASS / 0 FAIL)
| arm | v16f D (ctl 90.36, spread 0.16) | v32f D (ctl 168.57, spread 0.32) | verdict |
| Z2 (end chain removed, math live) | -5.98 (-5.86..-6.06) | -5.46 (-5.27..-5.66) | the whole end chain (lane merge -> output) is worth ~6 / ~5.5 us in nsys |
| PB | **-0.65** (-0.51..-0.78) | -0.20 (module byte-identical: noise level) | KEEP (v16f) |
| EE1 | -0.04 (identical) | **-0.64** (-0.45..-0.97) | KEEP (v32f) |
| FF (+EE1) | +0.07 | +0.54 (+0.36..+0.80) | DROP: +1.2 vs EE1. The early fetch puts the acquire fence (MEMBAR.ALL.GPU, ~0.9 us like the publish fence in XC job21) + a lane barrier + the fetch issue BEFORE the lane exchange, i.e. on lane 0's merge path |
| FFU (+FU) | +0.14 | +0.24 | FU -0.30 vs FF -> re-test FU on EE1 (job 3) |
| FFUPM (ideal pre-merge on FFU) | -0.11 | +0.04 (-0.20 vs FFU) | collector rule (<= -0.6) FAILS: pre-merge DEAD on both shapes |
| LB8 / LB16 | +1.20 / +1.02 | +1.41 / +0.88 | DROP: coupling the lanes costs more than their drift |
| ALL (PB+EE1+FF+FU+LB16) | +0.60 | +8.95 | DROP; also had a real bug: the LB barrier at pp = npmin - 1 is in lane 1's PEELED last pair (no LB line) but in lane 0's hot loop when lane 0 has one more pair -> barrier-0 mismatch (xcheck did not catch it). Fixed in build.py (LB only for pp < npmin - 1). |

## Job 3 (w2_j3) - DECISION RULES (written before the job)
Stacks on S1 = PB + EE1 (the two job-2 keepers; S1's xd0 = PB's, xd1 = EE1's cubin, byte-identical):
S1, S1FU (+FU), S1LT (+LT: last pair as two T64 steps, both builds, xd0 hot loop mask-free), S1LT2 (+LT, xd0 hot loop
keeps its runtime mask: MF=2), S1SP (+XD_SP=1: K/V prologue before the Q / new-token work), S1X (+FU+LT+SP).
Gate: xcheck auto on S1 and S1X first; if S1X fails, only S1 is timed. nsys 2 copies x 2 passes vs k60.
PICK (dev/w2/pick.py --base=s1 --margin=0.01): a candidate qualifies iff D <= +0.15 on both shapes and D <= -0.10 with all 4
runs below the same-pass control on at least one; a stack beyond S1 is picked only if its score-weighted gain
(0.135 %/us v16f + 0.078 %/us v32f) beats S1's by >= 0.01 %. The pick then runs tools/check.py (56 PASS, 0 failures) and
xcheck XC_HYB=0 / =1 (0 failures) in the same job -> freeze dev/w2/deliver_r1/ (build.py + built file + md5 + tables).

## Job 3: NOT RUN (Modal spend limit, 16:36:18). Nothing measured; cand_r1 prepared instead (see Resume here).

## Jobs 4 / 5 - DECISION RULES (written before; READY, not run)
Files (local NVRTC 13, all 0 spills, md5 from `python3 dev/w2/build.py dev/coord/k60.py <out> <knobs>`):
| file | knobs | xd0 (v16f) cubin / hot loop | xd1 (v32f) cubin / hot loop |
| s1 = cand_r1/k60w2.py | PB EE=1 | 69067fc9 / 675 | 9323c5dd / 459 |
| s1fu | + FU | 69067fc9 | 8b6c034a / 459 |
| s1lt | + LT | 1a72f202 / 463 | 7d3f7fbb / 458 |
| s1lt2 | + LT MF=2 | e535ad16 / 674 | 7d3f7fbb |
| s1sp | + SP | ff5d9a4c / 676 | a0fe2363 / 459 |
| s1x | + FU LT SP | 2dfec837 / 463 | f064b8e0 / 458 |
| s1tp | + TP=1 (K, V, page table, qkv, out pages) | fa200a03 / 729 (!) | 106cf3ed / 459 |
| s1tp2 | + TP=2 (K, V pages) | 67e174cb / 675 | 37679cc0 / 459 |
- Job 4 (a) FREEZE cand_r1 -> dev/w2/deliver_r1/ (copy, never edit) iff CHECK_PASS=56 and 0 failures, xcheck 0 failures in
  all three modes, and its nsys D <= -0.10 on BOTH shapes vs k60 (job 2: PB v16f -0.65, EE1 v32f -0.64). Any check
  failure: no freeze; report which knob (rebuild pba.py / ee1 to bisect).
- Job 4 (b) stacks: PICK by dev/w2/pick.py --base=s1 --margin=0.01 (qualify: D <= +0.15 both shapes, D <= -0.10 with all 4
  runs below the same-pass control on one; beat S1's score-weighted gain by >= 0.01 %). A pick != s1 needs its own
  check.py + xcheck x3 (next job) before it replaces cand_r1 as deliver_r2.
- Job 5 (TP, W4's rule adapted): keep TP variant v on a shape iff under --tlbcold 1 BOTH v copies beat BOTH s1 copies by
  >= 0.10 us, under --tlbcold 0 mean(v) <= mean(s1) + 0.05, and warm nsys D(v vs s1) <= +0.10; between tp and tp2 take
  the larger cold gain (tp2 preferred on a tie: its hot loops are S1's). Then check.py on S1 + TP before freezing.
- Expected: LT -0.3..-0.8 per shape (job-1 L bound -1.7 / -1.8 = the whole last pair; LT removes the T128 chain after
  the last landing but keeps a T64), FU 0..-0.3 (v32f, it was -0.30 on top of FF), SP 0..-0.4 (start order, never timed),
  TP judge-only (v16f judge even-odd 1.1-1.8 us per W4; ceiling ~half: the data-page walks).

## Job 4 RESULT (w2_j4, 17:00:33-17:06:21, ~6 GPU-min; out/20261029T170621_w2_j4/; base k61, controls k61a / k61b)
- FREEZE GATE cand_r2 (S1 = k61 + PB + EE1, md5 27a93c4d): tools/check.py CHECK_RC 0, 56 PASS, 0 failures; xcheck
  auto / XC_HYB=0 / XC_HYB=1: 24 PASS / 0 FAIL each. => FROZEN dev/w2/deliver_r1/ (= cand_r2 + nsys_j4 / dec_j4 / check tail).
| arm (vs k61: v16f ctl 93.66 spread 0.18, v32f ctl 175.12 spread 0.24) | v16f D (runs) | v32f D (runs) |
| s1 (PB + EE1) | -0.62 (-0.74..-0.51) | -0.19 (-0.43..+0.07) |
| s1fu (+FU) | -0.74 (same xd0 as s1) | -0.11: FU no gain -> DROP |
| **s1lt (+LT)** | **-1.59** (-1.35..-1.74) | **-0.40** (-0.30..-0.56) | PICK (score +0.246 %) |
| s1lt2 (+LT, xd0 hot loop masked) | -1.42 | -0.30 | below s1lt |
| s1sp (+XD_SP) | -0.31 (+0.31 vs s1) | +0.05 | DROP: K/V prologue before the Q work is slower (the Q trip then queues behind the burst) |
| s1x (+FU+LT+SP) | -1.35 | +0.20 | DROP |
- LT reading: the last pair's tile A (issued one pair step before B) is computed as it lands; only a T64 chain remains
  after the last landing (bound L was -1.7 / -1.8). v16f gains -0.96 beyond S1, v32f -0.21.
- Note: EE1 on v32f was -0.64 in job 2 (fast container, k60 168.6) but -0.19 here (slow container, k61 175.1).

## Job 5 (w2_j5) - DECISION RULES (written before the job)
(a) FREEZE dev/w2/cand_r3/ (S1LT = k61 + PB + EE1 + LT; k61w2.py md5 996fed8e == the measured s1lta.py; 839,585 B after
    stripping, cubins unchanged) as dev/w2/deliver_r2/ iff check.py 56 PASS / 0 failures and xcheck 0 failures x3.
(b) TP on top of S1LT (lttp2 = K / V pages, lttp = + page table, qkv, out; hot loops unchanged: 463 / 458): keep variant v
    on a shape iff (bench_jt --tlbcold 1) both v copies beat both s1lt copies by >= 0.10 us, (--tlbcold 0) mean(v) <=
    mean(s1lt) + 0.05, and warm nsys D(v vs s1lt) <= +0.10 (dev/w2/jt_decide.py; TPPICK = largest cold gain). A TPPICK
    gets check.py + xcheck in the same job (if wall < 560 s) and then becomes deliver_r3 (= cand_r3 + TP).

## Job 5 RESULT (w2_j5, 17:10:30-17:15:03, ~5.1 GPU-min; out/20261029T171503_w2_j5/, copies in dev/w2/out_j5/)
- FREEZE GATE cand_r3 (S1LT, md5 996fed8e): check.py CHECK_RC 0, 56 PASS, 0 failures; xcheck auto / HYB 0 / HYB 1 24 PASS /
  0 FAIL each => FROZEN dev/w2/deliver_r2/ (= cand_r3 + job-4 nsys table + check / xcheck tails). deliver_r2 supersedes
  deliver_r1 (it contains PB + EE1 + LT).
- TP (bench_jt, us per call, copies a / b; judge pair structure, sglang base paired ratio):
| | s1lt | lttp2 (K/V) | lttp (5 regions) |
| v16f tlbcold 1 | 94.81 / 94.25 | 94.57 / 94.17 | 93.79 / 93.70 |
| v32f tlbcold 1 | 179.75 / 179.29 | 178.18 / 178.06 | 178.57 / 177.79 |
| v16f tlbcold 0 | 93.54 / 93.28 | 93.02 / 93.48 | 93.66 / 93.57 |
| v32f tlbcold 0 | 175.04 / 174.85 | 175.25 / 174.27 | 175.48 / 174.51 |
| warm nsys D vs s1lt (v16f / v32f) | 0 | +0.04 / +0.13 | -0.10 / +0.12 |
  Rule -> TPPICK none (lttp: v16f warm +0.20 > +0.05, v32f nsys +0.12 > +0.10; lttp2: v16f cold not both >= 0.10, v32f
  nsys +0.13). The cold (judge-like) gain is clear for lttp on both shapes (-0.8 / -1.3 us).
## Job 6 (w2_j6) - DECISION RULE (written before the job): TP=1 confirmation, pooled with job 5 (dev/w2/tp_pool.py)
KEEP lttp (cand_r4 = S1LT + TP=1, md5 5815447f == lttpa.py) iff (cold) in BOTH jobs, on >= 1 shape both lttp copies beat
both s1lt copies by >= 0.10 and on no shape mean(lttp) > mean(s1lt) + 0.10; (warm) pooled tlbcold-0 mean diff <= +0.10 and
pooled nsys D (8 + 8 medians) <= +0.10 on both shapes. On keep: check.py + xcheck auto in-job -> deliver_r3.

## Job 6 RESULT (w2_j6, 17:22:54-17:26:11, ~3.7 GPU-min; out/20261029T172611_w2_j6/): TPKEEP no
- tlbcold 1: v16f lttp 103.28 / 102.76 vs s1lt 100.14 / 103.22; v32f 185.37 / 190.39 vs 185.94 / 186.42 (this node ran
  ~6-9 us slower than job 5's and with large copy spread). Pooled warm diff +0.105 / +0.150, pooled nsys D -0.06 / -0.07.
  => cold win not reproduced; TP stays out (lever plausible: job 5 cold -0.8 / -1.3; re-test on a quiet node with more copies).

## Job 7 (w2_j7, base k63) - DECISION RULES (written before the job)
Arms (all on top of deliver_r2 = PB + EE1 + LT; local NVRTC 13, 0 spills): rvb1 (VB=1: XD combiners load ONE partial =
ideal pre-loading while they wait), rvb2 (VB=2: + the last arrival does not combine when P > K), rvb2h0 (VB=2 with the XD
protocol forced at B 32 too), rh0 (XD protocol forced at B 32, correct, reference), rff2 (FF=2, correct: snapshot from the
early election + acquire fence by tid 256 before the merge barrier; the FINAL's lane 1 fetches after the role barrier with
no ld.acquire trip), rtp (TP=1). xcheck rff2 (auto + HYB=1) gates its timing.
- Pre-loading XD combine (bits + combiners merging each partial as published; last arrival not a combiner): BUILD iff
  D(rvb2) <= -0.8 on v16f with all 4 runs <= -0.4 (vs k63). Also for v32f iff D(rvb2h0) <= D(k63 = 0) - 0.8 (it would then
  replace the XC FINAL at B 32).
- FF2: keep iff v32f D <= -0.10 with all 4 runs < 0 and v16f D <= +0.15 (xd0 identical).
- TP (coordinator's rule: judge pair = cold x warm): D_shape = mean of the tlbcold 1 and tlbcold 0 deltas (2 copies each,
  k63 controls); keep iff 0.135 D16 + 0.078 D32 <= -0.01 % and no shape D > +0.10 (dev/w2/tp_judge.py).
- Kept pieces -> job 8: check.py + xcheck x3 on the stack, freeze dev/w2/deliver_r3/ (builder on k63).

## Job 7 RESULT (w2_j7, 17:34:58-17:40:35, ~6 GPU-min; out/20261029T174035_w2_j7/; controls k63: v16f 91.93, v32f 174.17)
| arm | v16f D | v32f D | verdict |
| rvb1 (bound: combiners load 1 partial) | +0.82 | +0.25 | INVALID bound: ptxas re-allocated xd0 and its hot pair loop grew 463 -> 517 instr |
| rvb2 (+ last arrival not a combiner) | +0.44 | +0.16 | same contamination (517-instr hot loop); rule (<= -0.8) not met |
| rvb2h0 / rh0 (XD protocol forced at B 32) | +0.52 / +0.15 | +2.74 / +2.01 | the XC FINAL stays at B 32 |
| rff2 (FF=2, correct; xcheck 24/0 auto + HYB=1) | +0.02 | +0.51 | DROP (FF=1 was +0.54: the combined-counter / snapshot path itself costs) |
| TP (bench_jt, judge pair cold x warm) | cold -0.25, warm +0.22 -> D -0.02 | cold +1.28 (one copy 189.2), warm +0.13 -> D +0.70 | TPJ no -> TP dead |
## Job 8 (w2_j8) - DECISION RULE (written before): direct test of the correct PL build (rpla.py; xd0 hot loop 468 instr,
0 spills; xd1 == k63's). xcheck auto + HYB=0 gate. KEEP PL iff v16f D <= -0.30 with all 4 runs below the same-pass k63
mean and v32f D <= +0.15; then check.py + xcheck HYB=1 in-job -> deliver_r3 (= k63 + PL, builder on k63). Plus a W2
trace of k63 (8 reps) for the post-r2 budget.

## Job 8 RESULT (w2_j8, 17:41:47-17:44:30, ~3 GPU-min): PL correct (xcheck auto / HYB 0: 24 / 0) but v16f D +0.68 (runs
+0.40..+0.88), v32f +0.11 -> PLKEEP no, DEAD: the waiting combiners' poll + round replace the last arrival's immediate
combine (loads of P partials are one trip in parallel anyway: pre-loading saves issue, not latency).
Post-r2 trace (k63, 8 reps): v16f last-pair compute 0.83 (was 1.60), end chain (tiles max -> end) 4.46; v32f last pair 1.04
(1.38); v16f critical chain after the last landing: merge barrier +0.8..1.3 -> segment end +1.0..1.5 (lane merge + fp16
store) -> publish +1.0..1.1 -> combined +2.2..2.7.
## Job 9 (w2_j9) - DECISION RULE (written before): split lane merge SL (rsl: both builds, xd0 hot loop 476 instr; rsl2:
XC_END build only, xd0 == k63's). xcheck auto gate (failure: nothing timed). PICK by dev/w2/pick.py (D <= +0.15 both shapes,
D <= -0.10 with all runs below the same-pass k63 mean on one; largest score-weighted gain); the pick then gets check.py
+ xcheck HYB 0 / 1 in-job -> deliver_r3 (builder on k63).

## Job 9 RESULT (w2_j9, 17:49:25-17:52:00, ~2.9 GPU-min): SL correct (xcheck 24 / 0) but slower: rsl v16f +0.15 / v32f +0.40,
rsl2 v16f 0.00 (xd0 identical) / v32f +0.34 -> PICK none, DEAD (half-row exchanges + the extra CTA barrier for the FINAL
cost more than the halved per-lane merge / store).
## Job 10 (w2_j10, base dev/coord/k64.py md5 f693d5bb; its XD section == deliver_r2's) - DECISION RULE (coordinator 17:50:
bench_jt --tlbcold 1 alone reproduces the judge's pairs): t64 = k64 + TP=1 (cubins == job 7's rtp: xd0 d5806e8f, xd1
a07f4682). KEEP iff on >= 1 shape both t64 copies beat both k64 copies by >= 0.10 us at --tlbcold 1 and on no shape is
mean(t64) > mean(k64) + 0.10 (dev/w2/tp_cold.py); --tlbcold 0 printed as a sanity check only. Keep -> check.py 56 / 0 +
xcheck auto / HYB 0 / HYB 1 in-job -> freeze dev/w2/deliver_r3/ (builder on k64).
Prior tlbcold-1 data for the same TP code: job 5 (k61 base) v16f -0.79 / v32f -1.34 (both copies win both shapes); job 6
(noisy node) mixed; job 7 v16f both copies win (-0.17..-0.32), v32f mixed (one copy 189.2).

## Job 10 RESULT (w2_j10, 17:52:45-17:54:52, ~2.5 GPU-min): TPC no. tlbcold 1: v16f t64 100.82 / 100.61 vs k64 100.38 / 101.32
(mean -0.14, copies interleave); v32f t64 179.60 / 186.90 vs k64 181.17 / 180.11 (t64b 186.9: the LAST --cand slot, as
in jobs 6 (190.4) and 7 (189.2) where TP's b copy also sat last). tlbcold 0 sanity: v16f +0.01, v32f -0.10.
## Job 11 (w2_j11) - DECISION RULE (written before): same rule as job 10 (dev/w2/tp_cold.py) with a position sink: order
k64, t64, k64b, t64b, k64c (k64c not judged). Keep -> check.py + xcheck auto in-job -> deliver_r3. Last job (cap).

## Job 11 RESULT (w2_j11, 18:05:21-18:07:30, ~2.7 GPU-min): TPC keep. tlbcold 1 (order k64, t64, k64b, t64b, k64c):
v16f t64 95.41 / 99.45 vs k64 95.70 / 99.95 (mean -0.40; positions 3-5 ran ~4 us slower than 1-2, the sink k64c 99.63),
v32f t64 179.33 / 179.76 vs k64 180.50 / 180.55 (-0.98, both copies win). check.py 56 / 0, xcheck auto 24 / 0 -> FROZEN deliver_r3.
