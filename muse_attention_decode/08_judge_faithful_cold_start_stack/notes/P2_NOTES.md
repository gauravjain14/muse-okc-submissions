# P2 (round 3): batch-1 FULL shapes d1f (xb_d1f, k55 _XB1_SRC) and v1f (xg_v8f, k55 _XG5_SRC)

## Resume here
- Phase 1 DONE (2026-10-29): dev/p2/PORTFOLIO.md (4 bets, budgets, probes, decision rules). No GPU used yet.
- P2-J1 DONE: B1 fired, B3 did not, NC 8 dropped. B1 (a) + diagnostic DONE: B1 closed per the funded rules; HD4+PV candidate (-0.37) in cand_r1/ stacked by the coordinator as k56. P2-J2 done: no v1f stagger win (see below). Planned order: P2-J1 (B1 + B3 bounds, d1f) -> B1 build if it passes
  -> P2-J2 (B2 stagger probe, v1f) -> B4 build -> B3 build.
- Scope rules: write only dev/p2/; Modal only via `bash tools/gate.sh tools/modal_run.py --tag p2_jN --cmd "..."`
  (mention dev/p2 in --cmd), every variant under `timeout 120`, job <= 15 min, nsys decides (2 copies, controls
  first: dev/xe/nsys_run.py or dev/wp/nsys_loop.py pattern), no okc, no edits outside dev/p2.

## Files
- PORTFOLIO.md: the phase-1 portfolio (bets B1-B4).
- sass_of.py: local no-GPU SASS of a module string inside a k-file:
  `$SP/nvrtc13/bin/python dev/p2/sass_of.py dev/coord/k55.py _XB1_SRC <out.sass> [-DXB1_AB=5 ...]`
  ($SP = the scratchpad path in docs/ROUND3.md; NVRTC 13 + nvdisasm 13 = judge image). Prints ptxas -v + C75xx.
- toy_rsa.py: compile check of FP8 RS-A wgmma m64n8k32 with A = V^T from 2 LDSM.T + 8 PRMT (B1's PV primitive).

## Local SASS facts (NVRTC 13)
- xb_d1f: 106 regs, 0 spills; TORD burst = 12 unrolled tile blocks 0x0c90-0x2b50 (~7.9 KB, ~36 SASS per tile incl.
  BSSY/BRA + 2 LDGSTS + ARRIVES.LDGSTSBAR); stage loop 0x3e90-0x5250 = 324 SASS; post-loop to EXIT ~450 SASS;
  parity LDG.STRONG.GPU right after BAR.RED (overlaps the merge); arrival = MEMBAR.ALL.CTA + MEMBAR.ALL.GPU + ATOMG;
  epilogue = CALL.REL.NOINC into xa1_epi (8+ LDG.STRONG.GPU shard loads).
- xg_v8f: 127 regs, C7511 (harmless per XG r2i/r4b), 4000 SASS (64 KB); QK = QGMMA.64x64x32.F16, PV =
  QGMMA.64x72x32.F32 (ones column = row sums); 42 MUFU total (softmax is f16x2 FFMA/HMNMX2 based).

## P2-J2 (B2 v1f soft stagger) RESULT (2026-10-29 09:16-09:20, ~4 GPU-min, dev/p2/out_j3.txt): nothing staged
Build: dev/p2/mk_k55g.py <base> <out> [--stag-default N] (XG_STAG ns: parity-1 WGs start their step chain later,
globaltimer + 64-ns nanosleeps; XG5 section of k55 == k56's); k55g.py = k55gb.py md5 c9d97777; STAG 0 cubin = k55's.
t_xg quick 0 failures (DEFER 0 + STAG 800, STAG 800). nsys v1f medians (pass 1 / pass 2, copies g / gb):
  k55c 12.30 / 12.35 | DEFER0 13.46, 13.46 / 13.50, 13.34 (~13.44, +1.1 vs DEFER 4) | D0 + STAG 400 13.31, 13.18 /
  13.30, 13.18 (-0.20 vs DEFER0) | D0 + STAG 800 13.57, 13.46 / 13.47, 13.55 (+0.07: an 800-ns delay of half the WGs
  is ~absorbed -> real convoy) | DEFER 4 + STAG 400 12.29, 12.32 / 12.32, 12.38 (0.00 vs k55) | + STAG 800 12.35,
  12.32 / 12.42, 12.35 (+0.03).
Rules: vs k55 no knob wins (coordinator's >= 0.15 staging rule: NO). Probe rules on DEFER0: soft stagger -0.20 and
the 800-ns delay absorbed -> the convoy is real, but on the shipped DEFER 4 the stagger also delays half of every
deferred tile (each warp issues 2 rows of each deferred tile), so the gain does not survive. Untested port (needs
funding): parity-0 WGs issue the whole deferred wave (deferred-tile mbarrier count 256) + STAG ~400 (est. <= -0.2).
## B1 diagnostic job RESULT (2026-10-29 09:06-09:11, ~5 GPU-min, dev/p2/out_j2b.txt): B1 rules NOT met; HD4+PV -0.37
k55h.py = k55hb.py md5 09d0852a (mk_k55h.py now has XB1_HD8 / XB1_HDPV / XB1_HDAB; default cubin = k55's).
t_d1 full 60/60 PASS for HD8, HD8+PV, HD4+PV (scored err 3.70e-2 / 3.70e-2 / 3.80e-2).
nsys d1f medians (pass 1 / pass 2; copies k55h, k55hb): control k55c 10.77 / 10.72 (~10.745)
  HD4 10.67 / 10.69 (-0.07) | HD4 no-merge (HDAB 1) 10.40, 10.34 / 10.30, 10.34 (-0.40; AB5-equivalent = ctrl - 1.08
  = 9.67 -> 0.68 above: the HD tile phase costs ~0.7 more than k55's) | HD4 + PV in flight 10.37, 10.34 / 10.40, 10.38
  (-0.37, all 4 runs below both controls) | HD8 11.20, 11.23 / 11.23, 11.26 (+0.47) | HD8 + PV 11.20, 11.23 / 11.18,
  11.26 (+0.46) | HD4 no-math (HDAB 2): illegal-instruction crash (timing-only knob; not used, not debugged).
Coordinator rules: no-merge within 0.3 of AB5? NO (0.68). HD8 >= 0.3 faster? NO (slower). -> B1 as funded is CLOSED.
Standing user rule (stack every correct nsys win): HD4 + PV is correct (t_d1 full 0) and -0.37 in 2 copies x 2
passes -> prepared (NO GPU) as dev/p2/cand_r1/ (build.py = mk_k55h.py with HD 1 + HDPV 1 defaults, applies on
dev/coord/k55.py; k55_b1.py md5 d6268c12; cubin == the measured HD4+PV variant). NOT frozen: needs tools/check.py
(0 failures, grep -a) + a confirming nsys run before it becomes dev/p2/deliver_r1/ (one <= 6-min job, needs funding).
## B1 job (a) RESULT (2026-10-29 08:58-09:01, ~3 GPU-min, dev/p2/out_j2a.txt): KILL RULE FIRED (no gain)
Build: dev/p2/mk_k55h.py <base> <out> [--hd-default 0|1] (XB1_HD in place on the _XB1_SRC section); k55h.py = k55hb.py
md5 fcf73dea (HD default 0: cubin byte-identical to k55). HD 1: 64 regs, 0 spills, no C75xx, 12160 B static smem,
~110 SASS per warp per tile (k55 324 per warp per stage). PV operand mapping verified on CPU (dev/p2/emu_hd.py).
Correctness HD 1: t_d1 quick 6/6 and full 60/60 PASS (scored case err 3.80e-2 vs bound 0.108; xa_b1f_1L E4M3-Q
baseline 3.7e-2: FP8 P adds ~nothing; worst normal-dist case ctx 1000 0.124 vs bound 0.150).
nsys d1f medians (slow container: controls 10.82): k55c 10.82 / 10.82 | HD 0 10.80 / 10.82 | HD 1 k55h 10.78 /
10.82, k55hb 10.83 / 10.82 -> HD 1 = control (-0.01). Kill rule (>= 0.4 faster) fired: (b)/(c) not spent.
Inference (not measured): HD gives each lane 5 sequential tiles (only 2 tiles in flight per SM vs 4) with a per-tile
chain of 3 barriers + 2 proxy fences + 2 waited wgmma groups; at ~0.5 us/tile the lanes fall behind the parity
landing rate (~0.58 us/tile) and the tail eats the ~1.1 us the merge removal is worth (J1 AB5).
Next if funded (diagnostic <= 6 GPU-min): HD + no-merge and HD + no-math ablations (where the time went), PV left in
flight across tiles, and an 8-lane 1024-thread HD (64 regs x 1024 = the register file exactly; 4 tiles in flight).
## P2-J1 RESULT (2026-10-29 08:39-08:43, ~5 GPU-min, dev/p2/out_j1.txt; durations out/20261029T084342_p2_j1)
nsys d1f medians, 120 replays, pass 1 / pass 2 per copy:
  k55c (control) 10.69 / 10.69 | default k55p 10.72 / 10.75, k55pb 10.72 / 10.72 (= k55, base ~10.73)
  AB5 (no in-CTA exchange, same red volume) 9.63 / 9.63, 9.60 / 9.66 -> -1.10
  AB6 (AB5 + no math)                       8.64 / 8.64, 8.67 / 8.67 -> -2.07 (math exposure with no merge ~1.0)
  AB8 (nothing after the arrival atomic)    9.70 / 9.70, 9.66 / 9.73 -> -1.03
  NC 8 (real)                              10.94 / 10.94, 10.88 / 10.91 -> +0.19 (t_d1 quick 0 failures) -> DROP
Rules: B1 FIRES (AB5 <= base - 0.6; also AB6 <= base - 1.3). B3 does NOT fire (base - AB8 = 1.03 < 1.2; implied net
~0.2). Caveat: AB5 lets warps 0-7 red without waiting for lanes 2-3 (slightly optimistic vs B1's head-pair hand-off).
Next (needs coordinator funding): B1 build behind a default-off knob XB1_HD (default SASS stays = k55), ~3 jobs.
## P2-J1 build details
Files: mk_k55p.py (builder: k55 -> k55c.py byte copy md5 f0bff1b7 = control; k55p.py = k55pb.py md5 47798431 with the
XB1_AB 5/6/8 knobs), j1.sh (job), nsys_run.py / nsys_loop.py / t_d1.py (copies; nsys capped 110 s).
Local NVRTC 13 check: default k55p xb_d1f cubin byte-identical to k55's; AB5 105 regs (all 8 HMMA + 4 QGMMA kept,
the 32 accumulators folded into the red so nothing is dead-code-eliminated; no merge STS.64 / BAR.RED), AB6 58 regs,
AB8 104 regs (ATOMG then EXIT), NC8 106 regs. (First AB5 build kept only o[0][0][*] live -> 63 regs: ptxas deleted
the other HMMAs; fixed before the job.)
### original plan (kept for reference)
Build a copy of k55 whose XB1 module gets new timing-only knobs; default build must be SASS-identical to k55's
(check with sass_of.py diff before the job). Variants, nsys d1f, 2 copies x 2 passes, k55 controls first:
- base (k55) x2 | AB=5 x2: after the loop skip X1SYNC/store_partial/bar.red/merge; tid < 256 red.v4 of 4 own
  register values (scaled o[0][0][*]) at today's shard addresses + c2==0 L red (same volume as the real merge), then
  X1SYNC -> arrival -> epilogue unchanged | AB=6 (AB 1 + AB 5) x1 | AB=8 x2: last arrival exits right after the
  atomic (no epilogue, no reset; next call's counter = garbage -> timing only, run it last per process or reset the
  ws between variants) | XB1_NC=8 x2 (real, correct; t_d1 quick smoke first).
- Decisions: B1 build iff AB5 <= base - 0.6 (or AB5 <= base - 0.35 and AB6 <= base - 1.3). B3 build iff
  base - AB8 >= 1.2. NC 8 kept iff faster by >= 0.05 in both copies and t_d1 0 failures.

## P2-J2 plan (v1f, ~11 GPU-min) - not built yet
Copy of k55's XG5 module + XG_STAG (ns): parity-1 WGs __nanosleep(XG_STAG) right before step 0 (after the start
barrier), default 0 (SASS-identical). Variants on XG_DEFER=0: DEFER0 x2, STAG 400 x2, STAG 800 x2, plus k55 x2.
Decisions: best STAG <= DEFER0 - 0.15 -> freeze soft stagger, port to DEFER 4 (parity-0 WGs issue the deferred wave);
STAG800 - DEFER0 <= +0.25 -> build the ping-pong hand-off; >= +0.5 -> close B2.

## Key numbers used (sources)
d1f: deliver_r7 nsys 10.69-10.85 (XB r7), k51 10.78 (XB job61); AB=1 no-math 9.50 (XB job35); AB=3 -0.2 (job42);
AB=4 +5.5..+15 (job60); finisher end chain XB j25 / j39 (store 0.61-0.82, bar.red 0.06, merge 0.47, atomic 0.56-0.66,
call 0.13, loads 0.37-0.48, rsqrt 0.21-0.23, store 0.11, sync/reset 0.22); TORD trace j38 (loop end med 6.88 max
7.41, tiles lane 0 3.62/4.90/6.18, stages 0.55-0.65 latency-bound). Gather probe 'a' CTA done 5.60/5.89, 'c' (32
warps/SM) 5.46/5.76 (XA j6). WB r26: 4-KB publish incl. release, red into 4/8/16/32 copies 1.34/1.25/1.15/0.96 us.
v1f: deliver_r5 12.22-12.30 (XG r5fin); TB3 7.74 (g3), TB2 8.10 / TB1 10.56-10.72 (r4a/r4c); ablations no V^T -0.46,
no P pack -0.27, no clamp -0.17, all CUDA-core -1.47 (r4a); r2g trace (step0 med 2.98 max 4.85; loop end med 8.06 max
8.78; publish med 8.90 max 9.71; gate max 10.29; bulk 10.93; out 11.73; publish - loop end 0.93). XB job 7 wgmma
microbench 537/566/994 cycles per iteration per WG at 1/2/4 WGs per SM.
