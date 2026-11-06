# S1 (round 3b): d1f END CHAIN (xb_d1f in k56's _XB1_SRC). Base dev/coord/k56.py md5 427cc112. Cap 20 GPU-min.

## Resume here (ROUND 3 appended 15:01; rounds 1-2 text follows)
- ROUND 3 DONE, NO deliver_r3: merge-free end chain (XB1_MF=1) LOSES. J6 (14:56:39-15:00:32 = 3.9 GPU-min; lineage 27.1 of 34), nsys d1f medians
  (copy a/b, pass 1 | pass 2): controls 10.752/10.688 | 10.688/10.688; deliver_r2 ref 10.240/10.176 | 10.208/10.160 (same-pass means
  10.208 / 10.184); mf 10.432/10.368 | 10.368/10.369 = +0.188 (runs +0.16..+0.22); hab (timing-only: no hand-off barrier, same red volume
  as today, results wrong by design) 9.984 | 10.017 = -0.196 vs ref. Correctness of MF was fine (t_d1 FULL 60/60 plain and with FORCEWO 3 / 2 / 1000).
  Reading: even a free hand-off+merge is worth only ~0.2 us now (J1's 0.18+0.04+0.33 shrank under PW2/PWB: the parity LDG left the merge's
  path); the MF mechanics cost ~0.4 us more than that bound: 2x red bytes / ops (66 reds per shard word, MEMBAR.GPU waits for 2x the acks),
  one extra unconditional 256-thread pair barrier, one shfl exchange, 70 regs. A cheaper route to the 0.2 bound would have to keep the
  4 KB merged red volume (e.g. merge through a lane-pair smem exchange of only the half that the partner needs, no CTA/pair barrier).
  Code stays in build.py (default off); deliver_r2 remains the last frozen hand-back.

- ROUND 2 DONE: dev/s1/deliver_r2/ FROZEN = deliver_r1 + XB1_PWB=1 (two 256-thread head-pair barriers for the lane hand-off, per-head exact
  lists). nsys (J5, 2 copies x 2 passes): pw2 ref 10.081/10.064 | 10.112/10.048 (mean 10.076); +a 10.017/10.000 | 10.048/10.016
  (mean 10.020, -0.056 vs ref, runs -0.032..-0.072; vs controls 10.560/10.528 | 10.496/10.496 = -0.50). b (parity at the last tile
  iteration) +0.204 LOSS, c (EWARM=1) +0.156 LOSS, a+b +0.083 LOSS (screen). Same-job suite on the exact built file (md5 ac52fa63):
  t_d1 FULL + FORCEWO=3 FULL 60/60, check.py 56 PASS / 0 failures; delivered cubin == measured variant a. J5 = 14:18:02-14:23:34 =
  5.55 GPU-min; lineage total 23.2 of 26. The 0.03 stack margin is met by all 4 runs but the smallest is -0.032: modest win.
  Apply: `python3 dev/s1/deliver_r2/build.py dev/coord/k56.py <out.py>` (default --pw 2 --pwb 1; --pwb 0 = deliver_r1; --pw 0 = k56).
- Why b/c lost: PARL puts a predicated LDG + an extra live register (72 regs) into the tile loop (codegen shift, +0.2); EWARM=1 makes
  every non-final warp run the whole dry epilogue after its arrival (delays their exit; +0.16).

- DONE 2026-10-29 14:11 (17.65 of 20 GPU-min used): dev/s1/deliver_r1/ FROZEN = k56 + XB1_PW=2 (per-warp arrival), d1f nsys -0.43 us
  (10.444 -> 10.01; J2 -0.416, J3 -0.432, 8/8 runs below both controls). check.py 56 PASS / 0 failures, t_d1 FULL 60/60 (+ FORCEWO=3 60/60)
  on the delivered file itself (J4). Apply: `python3 dev/s1/deliver_r1/build.py dev/coord/k56.py <out.py>` (default --pw 2; --pw 0 = k56).
  Only differences vs k56: the _XB1_SRC section and one host line (-DXB1_GC, unused by the winner).
- Killed: MONO (+0.08), GSM (+0.08), GCONST/NC2 on top of PW (+-0.00 / +0.04), PAR0 alone (-0.056 border), PW=1 (-0.21, half of PW=2).
  Alive but subsumed by PW (their code is in build.py, default off): EPI_INLINE -0.20, EPI_V4 -0.28, GCONST -0.15, NC2 screen -0.19.
- Next lever (needs a PW trace, ~2 GPU-min with dev/s1/trace_s1.py adapted: PW emits XTE 3/5/6, 7-10 and XTR 7 only for the finisher
  warp): (1) the lane hand-off bar.red.or is still a 512-thread barrier although the merge of head hr needs only lanes (hr, hr + 2):
  two 256-thread named barriers (one per head pair, with a per-pair OR vote and a per-pair exact-path barrier) remove the skew
  between the pairs; (2) PW=2 vs PW=1 (-0.2) shows the parity LDG position matters: try issuing it at the start of the lane's last
  tile iteration; (3) the two serial L2 RTTs (atom 0.65 + shard loads 0.5) are what remains of the 3.0 us; (4) EWARM=1 for PW.

## What the end chain is in k56 (read from the source + SASS, no GPU)
Loop end (`xa_wg_wait<0>`, cp.async.wait_all, XTR 4) -> HD lane partial to xh_lp (XTE 1) -> bar.red.or 512 (XTR 5; parity
LDG.STRONG right after it) -> tid<256 merge: 2 lane partials + 4 l-sums, red.global.add.v4 into shard c%4 (XTE 3) ->
tid>=448 clear their slice of the OTHER parity's shards (runtime division by G inside: ~25 SASS) -> X1SYNC (XTR 11) ->
tid 0: MEMBAR.CTA/GPU + ATOMG.ADD acq_rel on XW_CNT, STS xl_last (XTE 5) -> X1SYNC (XTR 6) -> non-finishers EXIT, first 4
arrivals (EWARM 2) run xa1_epi dry -> finisher: CALL xa1_epi (noinline): 8 LDG.STRONG shard loads + LCNT + gate LDG,
FADD sums, rcp, ss shuffles x5, rsqrt, sigmoid (hidden in the shuffle shadows by the scheduler), STG bf16 (XTE 7..10)
-> RET -> X1SYNC -> tid 0 stores CNT=0, LCNT=0, PAR=par^1 (3 STG) -> EXIT (XTR 7).
SASS facts: epilogue sigmoid MUFU.EX2/RCP already sit between the SHFL.BFLY steps; gate LDG.E issues in the same batch as the
shard loads, so XB1_GSM can only remove an L2-hit LDG that overlaps the shard RTT -> expect ~0 (XG_GSM was -0.03 noise).

## Knob design (all default 0 => k56 SASS byte for byte; -D via XB_DEFS or build.py defaults)
- XB1_MONO: arrival counter that wraps in hardware: `atom.acq_rel.gpu.global.inc.u32 old,[CNT],2G-1` (counter cycles 0..2G-1,
  call k uses [kG mod 2G, +G-1]); parity = (value read early >= G) replaces XW_PAR (same early ld.relaxed at the same place);
  finisher iff old%G == G-1 (old == G-1 || old == 2G-1); EWARM-warm iff old%G < 4. LCNT becomes LCNT[2] indexed by parity,
  the tid-448.. clear step also zeroes LCNT[par^1] (CTA 0) -> the finisher needs NO reset stores and NO closing X1SYNC.
- XB1_GSM: sigmoid(gate) per (row, d-pair) computed into registers... (see below once built)
- XB1_EPI_INLINE: ...

## J1 RESULT (2026-10-29 13:21:30-13:23:07 = 1.6 GPU-min; dev/s1/out_j1.txt): the k56 end chain, finisher CTA, clock64 us
(14 reps each; flush = 128 MB, judge = 512 MB flush enqueued before the replay; they agree to 0.03)
| step (from the previous stamp)                         | flush | judge |
| slowest warp's loop end -> lane partials stored (HD)   | 0.177 | 0.178 |
| -> bar.red.or done (512-thread lane hand-off barrier)  | 0.044 | 0.045 |
| -> merge + reds issued (warps 0-7; warp 0 alone 0.33)  | 0.334 | 0.356 |
| -> clear block (warps 14-15, 0.35 from bar.red!) + X1SYNC | 0.071 | 0.070 |
| -> arrival atomic RETURNED (MEMBAR.GPU + ATOMG RTT)    | 0.647 | 0.650 |
| -> X1SYNC after atomic (tid 0 STS -> bar -> LDS)       | 0.076 | 0.078 |
| -> xa1_epi entry stamp (LDS, branch, CALL.REL.NOINC)   | 0.154 | 0.154 |
| -> shard loads back (8 x LDG.STRONG + LCNT + gate)     | 0.579 | 0.549 |
| -> shuffles + rsqrt (+ sigmoid, hidden in the shuffles) | 0.226 | 0.229 |
| -> output store issued                                 | 0.107 | 0.112 |
| -> done (RET + X1SYNC + 3 reset STG)                   | 0.155 | 0.160 |
| SUM (global last loop end -> done)                     | 2.572 | 2.579 |
Context (globaltimer us from first warp start, medians): burst issued 2.53 | median CTA loop end 6.66 | last loop end 7.66
(flush) / 7.76 (judge) | finisher = the CTA with the LAST loop end in most reps (rank 132; min 128) and the last arrival in
most reps | last arrival 8.98 | done 10.16 / 10.38 (nsys kernel duration 10.43-10.46, so the trace is not slower this time).
Clock 1.87-1.89 GHz in the finisher. nsys k56a / k56b, pass 1: 10.464 / 10.432, pass 2: 10.432 / 10.432 (mean 10.52-10.60).
Reading: 3 pieces: A = loop end -> atomic issue 0.63, B = atomic 0.65, C = after the atomic 1.30 (C - the loads RTT = 0.72 of
sync / call / math / store / exit; the AB8 bound -1.03 matches C + part of B).  Two surprises: (1) the merge is 0.33 us for
~25 instructions (the clear block of warps 14-15 is as slow, 0.35: runtime-G integer division, 100+ SASS in a dependent chain);
(2) atomic return -> first epilogue stamp = 0.23 us (X1SYNC 0.076 + 0.154 for LDS / branch / CALL to a function 0x500 bytes
away: cold instruction fetch is the only thing that makes sense of it; the XB WARMEC attempt to warm it lost for other reasons).
Also: the start has a runtime division by G (cstart) before the first page-table LDG (SASS 0x110..0x540).

## Decision rules (written BEFORE each job)
### J1: measurement only (done). Follow-up rules it set: build/measure MONO iff (X1SYNC after atomic + done) >= 0.10 [0.23: YES],
EPI_INLINE iff epi-entry >= 0.08 [0.154: YES], GSM iff rsqrt/gate step >= 0.15 [0.226: YES by the rule, but the SASS shows the
sigmoid already hidden -> expect ~0], V4 epilogue (float4 shard loads, 8 warps x 2 rows, 4 shuffle rounds) [loads 0.58: YES],
new riders from the trace: GCONST (compile-time G: removes the cstart division at the START and the `per` division in the
clear block, 0.35 -> ~0.1) and NC=2 (halves the finisher's shard bytes 17 KB -> 8.7 KB; NC 8 was +0.19 so bytes matter).
### J2 (smoke + nsys, one job, <= 9 min): knobs MONO, EPI_INLINE, EPI_V4, GSM, GCONST, NC2, each alone, 2 copies (k56s, k56sb)
x 2 passes, controls k56a, k56b FIRST in each pass; t_d1 --quick per knob first (a knob that fails its smoke is not timed);
MONO also under XB1_FORCEWO=3 (full t_d1: every 3rd CTA takes the exact list path -> exercises the per-parity list counts).
KEEP a knob iff every one of its 4 runs is below BOTH controls' same-pass medians AND its mean over the 4 runs is >= 0.05 below
the mean of the 4 control runs (rule: >= 0.05 faster in both copies/passes); KILL iff |mean delta| < 0.03 or any run is slower than
both controls; in between (0.03-0.05, consistent sign): keep only if free (no extra code), else re-measure once in J3 stacked.
Stack: all kept knobs together form J3 (t_d1 full, nsys 2 copies x 2, check.py 56 PASS) -> deliver_r1.
J2 AMENDED 13:33 (before it started; still queued): files k56p / k56pb (build.py with the PW knob added; default cubin == k56's,
MONO SASS == k56s's MONO) replace k56s / k56sb; added knobs XB1_PAR0=1 (load the parity at the kernel start: with k56's HD
the merge step's red addresses wait on the parity LDG issued after bar.red: 0.33 us for ~25 instructions; XB-era PAR0 was a
loss in another structure) and XB1_PW=1 (per-warp arrival: warps 0-7 reduce rows 2w, 2w+1, each its own release + atom.inc
counter (8 counters, 128 B apart) + shfl broadcast + float4 epilogue on its own rows; warps 8-15 clear and exit right after
the lane hand-off bar.red; no CTA barrier / smem broadcast / closing sync / reset; rare exact path adds one X1SYNC), PW+PAR0.
gsm / nc2 run as one-copy one-pass SCREENS (nominate for J3 only if <= -0.08 vs the same-pass controls).
FORCEWO=3 tests: quick base, FULL t_d1 for MONO and for PW (every 3rd CTA takes the exact list path).

## J2 RESULT (2026-10-29 13:38:54-13:46:11 = 7.3 GPU-min; dev/s1/out_j2.txt; summary: python3 dev/s1/summ.py dev/s1/out_j2.txt)
Smokes: t_d1 --quick 6/6 PASS for mono inl v4 gc par0 pw1 pw2 gsm nc2 (the whole t_d1 suite takes 2-4 s, so FULL runs are cheap);
FORCEWO=3 (every 3rd CTA takes the exact list path): base quick 0 failures, MONO FULL 60/60, PW=1 FULL 60/60.
nsys d1f medians us (pass 1 copy a, copy b | pass 2 a, b); controls k56a/k56b: 10.592 10.560 | 10.560 10.592 (mean 10.576;
this container ~0.13 slower than J1's 10.44):
  mono  10.688 10.656 | 10.656 10.624  mean +0.080  KILL (all four runs slower)
  inl   10.400 10.352 | 10.400 10.336  mean -0.204  KEEP (epilogue inlined: no CALL)
  v4    10.304 10.304 | 10.288 10.288  mean -0.280  KEEP (inline + 8 warps x 2 rows, float4 shard loads, 4 shuffle rounds)
  gc    10.464 10.432 | 10.400 10.416  mean -0.148  KEEP (G compile-time: start's cstart division + clear block's division)
  par0  10.480 10.544 | 10.496 10.560  mean -0.056  BORDER (runs -0.096..-0.016; mean-col -0.024) -> not kept alone
  pw1   10.384 10.368 | 10.368 10.336  mean -0.212  KEEP (per-warp arrival, parity read at the kernel start)
  pw2   10.160 10.176 | 10.144 10.160  mean -0.416  KEEP  <== all four runs -0.40..-0.43; p10 9.89 (parity read right before bar.red)
  gsm   10.656 (screen)  +0.080 KILL (as the SASS reading said: nothing to remove; +4 KB smem + the early gate load cost)
  nc2   10.384 (screen)  -0.192 (one run: nominated for J3)
Verdicts: MONO and GSM die; INL / V4 / GC / PW are real. PW=2 is the biggest single lever (-0.42): it deletes the three CTA
barriers + smem broadcast + reset of the old chain (warps 8-15 clear and leave; warps 0-7 each release + atom.inc their own
counter, shfl the rank, run the float4 epilogue for their own two rows). PW=1 (parity at the start) is half as good as PW=2.
Why MONO lost although it deletes the closing sync + 3 stores: the atom.inc + parity-from-counter changes the post-loop schedule
(sassnorm: instruction order shifted in the merge region) -- same family as ERST (+0.15); PW did not hit it.

## J3 plan / decision rule (written BEFORE the job): stack PW=2 with GC and NC
Candidates (baked files, md5 in the job log): c1 = PW2 + GCONST, c2 = PW2 + GCONST + NC2, c3 = PW2 + NC2, c6 = PW2 + GCONST + NC1
(screen, one copy), plus pw2 alone (k56p + XB_DEFS=XB1_PW=2) as the SAME-JOB reference (J1/J2 containers differ by 0.13).
Each: t_d1 FULL 0 failures (c2 also under XB1_FORCEWO=3); nsys 2 copies x 2 passes vs k56a/k56b. check.py (56 PASS / 0 failures)
on c2. KEEP rule: a component stays iff the stack with it is >= 0.03 below the same stack without it (means over the 4 runs, all
four runs below the lower stack's mean) -- e.g. gc: c1 vs pw2, nc2: c2 vs c1 and c3 vs pw2. The best passing stack is frozen
as deliver_r1 (build.py defaults = the winners). If c2 is not the best, J4 (<= 4 min) runs check.py + a 2x2 confirm on the winner.

## J3 RESULT (2026-10-29 13:49:03-13:56:06 = 7.05 GPU-min; dev/s1/out_j3.txt; python3 dev/s1/summ.py dev/s1/out_j3.txt)
t_d1 FULL 60/60 PASS for c1 c2 c3 c6; FORCEWO=3 FULL 60/60 for c2 and c6; tools/check.py on c2: CHECK_RC=0, 56 PASS, 0 failures.
nsys d1f medians (copy a, b | pass 1, 2), same job, controls k56a/k56b 10.448 10.432 | 10.464 10.432 (mean 10.444):
  pw2 (k56p -DXB1_PW=2)   10.000 9.984 | 10.016 10.048   mean -0.432 (runs -0.456 -0.440 -0.400 -0.432)
  c1 = pw2 + GCONST       10.048 10.016 | 9.952 10.016   mean -0.436  -> GCONST adds -0.004: KILL on top of PW (alone it was -0.148)
  c2 = c1 + NC2           10.112 10.032 | 10.048 10.016  mean -0.392  -> NC2 on c1 is +0.044 worse: KILL
  c3 = pw2 + NC2          9.984 10.016 | 10.016 10.016   mean -0.436  -> NC2 on pw2 adds -0.004: KILL
  c6 = c1 with NC1 (1 copy) 10.176 | 10.176             mean -0.268  -> worse
KEEP rule result (component stays iff the stack is >= 0.03 below the stack without it): none of GC / NC2 qualifies on top of PW=2
(PW removed the clear block and the epilogue from the critical CTA barriers; the division / shard bytes they cut are no longer
on the critical path). FREEZE = PW=2 alone (-0.43 us, 8/8 runs of J2 + J3 below both controls: J2 -0.416, J3 -0.432).
deliver_r1/ = build.py (default --pw 2; --pw 0 = k56 cubin-identical) + k56_s1r1.py (cubin == the measured pw2 cubin, cmp'd with
NVRTC 13 locally). check.py was run on c2 (not the final stack) in J3 -> J4 runs it on the delivered file itself.

## J4 plan / decision rule (written BEFORE; ~3.5 GPU-min, the last job: 15.95 + ~3.5 ~= 19.5 of 20)
deliver_r1/k56_s1r1.py: rebuild from dev/coord/k56.py == delivered file; t_d1 FULL 0 failures (+ FORCEWO=3 FULL); check.py
CHECK_RC=0, 56 PASS, 0 failures. All pass -> deliver_r1 is final (MD5SUMS + NSYS_TABLE.txt written from J2/J3, never edited
after). Any failure -> deliver_r1 is NOT handed back as a win (report the failure); no GPU left to iterate.

## J4 RESULT (2026-10-29 14:09:43-14:11:24 = 1.7 GPU-min; dev/s1/out_j4.txt): deliver_r1 file itself
REBUILD (deliver_r1/build.py on dev/coord/k56.py) == delivered file (md5 f371f670...); t_d1 FULL 60/60 PASS 0 failures; t_d1 FULL
XB1_FORCEWO=3 60/60 0 failures; tools/check.py: CHECK_RC=0, CHECK_PASS=56, 0 failures.  => deliver_r1 final.

## ROUND 2 (funded by the coordinator after the first hand-back; cap +6 GPU-min, lineage 26, 17.65 used): J5 plan / decision rule
Riders on deliver_r1's PW=2 (all default off in build.py; PW=2 build == deliver_r1 cubin, cmp'd): (a) XB1_PWB: the lane hand-off
by two 256-thread named barriers (ids 5 + hh, lanes hh and hh + 2; per-pair OR vote and per-pair exact-path barrier; the exact path
becomes per head: own list (XW_LIST + 256 hh), own count (XW_LCNT2 + 2 par + hh), own slot rows); (b) XB1_PARL: the parity LDG
issued at the start of the lane's last tile iteration (hn == 0 lanes load after the loop) -- still before every bar.red; (c)
XB1_EWARM=1 (every non-final warp runs the dry epilogue). Variants: pw2 ref x2 copies (r2, r2b), a x2, b x2, c x2, a+b x1
(screen), nsys 2 passes vs k56a/k56b (controls first), 1,1,full; t_d1 FULL and FORCEWO=3 FULL per knob first (the suite takes 5 s).
RULE (coordinator): a rider STACKS iff every one of its runs is >= 0.03 below the pw2 reference's same-pass mean and none is above
either control (same pass); the stacking measured variant with the lowest mean is frozen as deliver_r2/ ONLY IF its delivered
cubin == the measured variant's cubin (baked vs -D cubins cmp'd locally for a b c ab: identical) and, in the SAME job, t_d1 FULL,
FORCEWO=3 FULL and tools/check.py (56 PASS / 0 failures) pass on the exact built file (dev/s1/decide_r2.py picks the winner in-job,
build.py builds /tmp/r2_built.py with the winner's flags; I rebuild locally afterwards and compare md5). Otherwise: report numbers, stop.

## ROUND 3 (funded; cap +8 GPU-min, lineage 34, 23.2 used): J6 plan / decision rule (written BEFORE the job)
Bet: remove the lane hand-off store + CTA-side merge (J1: stores 0.18 + hand-off bar 0.04 + merge/reds 0.33 on the critical path) under
PW=2 + PWB. XB1_MF=1 (build.py --mf 1; needs --pw 2 --pwb 1): every HD lane keeps its fixed-reference partial in registers
(hs[2] = (hacc[e] f, hacc[e+2] f)), votes out-of-range with a 128-thread LANE barrier (bar.red.or on hbid), and in range red.adds
its OWN partial into the shards: one shfl exchange with lane^4 gives each thread a float4 (row 2t+odd, d 16wl+4(g>>1)..+3) -> one
red.v4 per thread (16 B x 512 threads = 8 KB per CTA vs 4 KB merged: 2x bytes, 2x v4 ops, 66 instead of 33 reds per shard word at
NC4); the l-sums are summed over the lane's 4 warps by its first 8 threads (smem xh_lw) and red.add'ed. No xh_lp stores, no merge.
Exact path per LANE: its own slot (c*4+li; O[8][64] | M[8] at 512 | L[8] at 528 = 544 words, slots reach word 296448 so the arrival
counters moved to 400000 and the per-head lists (264 entries > 256) to 440000 + 512 hh), entry c*4+li into the head's list; the epilogue
fold reads that layout. One UNCONDITIONAL 256-thread pair barrier (warps 8-15 bar.arrive, warps 0-7 bar.sync) before the warp-0..7
arrivals, since a row pair now receives contributions from 8 warps. Everything else (counters, parity before the lane barrier, epilogue) = deliver_r2.
Variants: deliver_r2 reference (r3/r3b, build --pw 2 --pwb 1, cubin == deliver_r2's, cmp'd) x2 copies, mf x2 copies, hab x1 copy
(XB_DEFS=XB1_HDAB=1: timing-only upper bound, no hand-off barrier, own registers red'ed at today's volume, results wrong by design),
nsys 2 passes vs k56a/k56b (controls first), 1,1,full. Smokes first: t_d1 FULL for MF plain, FORCEWO=3 (mixed lanes), FORCEWO=2, FORCEWO=1000.
RULE: KEEP iff every one of the 4 mf runs is >= 0.05 below the reference's same-pass mean and none is above either control
(dev/s1/decide_r3.py). If KEEP: build --pw 2 --pwb 1 --mf 1 in-job, t_d1 FULL + FORCEWO=3 FULL + tools/check.py (56 PASS / 0 failures)
on that exact file, then freeze deliver_r3/ (build.py defaults = deliver_r2 + --mf 1, built file == the in-job md5, cmp cubin vs the
measured -DXB1_MF=1 variant: checked locally, identical). Otherwise report the numbers and stop.

## GPU ledger (job wall minutes from the gate's "[gate] slot" stamp to the end)
- J1: 13:21:30 -> 13:23:07 = 1.6 min.
- J2: 13:38:54 -> 13:46:11 = 7.3 min.
- J3: 13:49:03 -> 13:56:06 = 7.05 min.
- J4: 14:09:43 -> 14:11:24 = 1.7 min.   TOTAL 17.65 / 20.
- J5 (round 2): 14:18:02 -> 14:23:34 = 5.55 min.   LINEAGE TOTAL 23.2 / 26.
- J6 (round 3): 14:56:39 -> 15:00:32 = 3.9 min.    LINEAGE TOTAL 27.1 / 34.
