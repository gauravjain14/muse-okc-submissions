# Agent Q5 — m = 128 (verify-16) qkv_gate W[2304,8192] and o W[8192,1024], BF16; base bo14 / LN/bo14_lean.py

## RESULT (read first)
- SHIPPED: claude/t3/Q5/q5.py (13.5 KB, readable, no options / trace code) — prepare_q5(weights) / run_q5(x, config,
  state): layer o at 97 <= m <= 128 only (None for everything else, incl. qkv_gate). Kernel = Q2's o kernel (64 x 128
  tiles, 9 x 24 KB stages, packed W at a baked address) with W's bulk copy issued before x's TMA in every stage.
  Candidates: sub_q5.py (LN/bo14_lean.py + q5 first), sub_q5_only.py (q5 + torch). prepare ~0.3 s, never raises.
  Integration: `if layer == 'o' and 97 <= m <= 128: out = run_q5(x, cfg, st)` before the Q2 route.
- o m=128 vs bo14_lean, box 0: same-run x-clone tables -0.06 / -0.10 / -0.08 us (x1.112-1.121 -> x1.124-1.141 vs
  SGLang); alternating single-candidate processes x1.129 -> x1.173 (3 pairs, -0.25..-0.31 us); nsys in-graph 6.34 ->
  6.24; ncu 7.46 -> 7.33. Correctness: check.py --graph ALL OK (103 cases, NaN-poisoned), exact_all --nesc 200 / 3000
  EXACT ALL OK.
- qkv_gate m=128: NOT shipped. Q3's techniques (baked placement, solo phase 2 with the exchange pushed into unused ring
  space, ring depth 12/13) and four start/tail variants all tie with Q2's kernel (11.1-11.4 same run; nsys 12.85 vs
  12.37 for the best-on-bench variant). Reason (measured): at m128 the loop is DRAM/L2-bound (0.133 us per stage =
  4.43 TB/s of W, L2 ~65 % / DRAM ~61 % of peak over the whole ncu duration), not per-SM-ingress-bound as at m256, so
  every phase-2 stage costs a full ring cadence (Q2/2 x 0.13 us) — as much as the exchange it hides.
- Facts for others: (1) __syncthreads at kernel entry costs ~0.45 us; (2) the TMA unit issues ~75-90 ns per stage
  (x TMA + W bulk) and a second issuing warp does not help; (3) never de-synchronise the k order across CTAs (rotated
  starts: qkv +3.3 us, o +0.2-0.3); (4) releasing each stage right after its own MMAs (wait<0>) is much worse (qkv
  +1.3, o +0.15); (5) NVRTC 13 C7514 (all wgmma serialised) from `__float_as_uint(acc)` asm operands — local nvcc 12.1
  is silent, check the box's NVRTC log (nvc.py); (6) box 1 died ~15:00, all later numbers are box 0.

Base route at m = 97..128 (both layers): agent Q2's g128q (lean copy LN/g128q_lean.py, SASS-identical to bo14):
qkv_gate = 64 tok x 72 col tiles, 2 consumer warpgroups alternating k-blocks (in-CTA k-split), 2-way split-K as a cluster
pair with a DSMEM reduce-scatter after the loop (RED 9), 12 x 17 KB stages, GPC placement read from a param array;
o = 64 tok x 128 col tiles (warpgroup g owns 64 W rows), 9 x 24 KB stages.
Files: q5k.py (dev kernels: qkv_body with solo phase 2 / options, o_body), q5t.py (same-run harness: SGLang + bo14_lean's
route + q5 configs, x at XCL random 4 KB-aligned addresses per candidate (P5's method), NaN-poisoned correctness incl.
one-hot rows, per-CTA clock64 traces), probe_start.py (first-landing / TMA issue probe), gensrc.py + nvc.py (source
dump; NVRTC-only compile on the box to read ptxas C75xx warnings).
Times: local us (graph replay after a 512 MB flush, empty graph subtracted; judge ~= local + 1.0). Same-run A/Bs only.
Box 1 died mid-session (2026-11-02 ~15:00); everything after that is box 0 (user limit: one box, 4 agents).

## iteration 1 — Q3's techniques ported to the qkv m128 kernel (q5k.qkv_body)
Design (cfg = NS, Q2, CMB, PERM): tile / split as Q2; rank r finalises token rows [32r, 32r+32) (warps 2r, 2r+1 of
each warpgroup). Placement measured at build time and BAKED as immediates (Q3's compare chain; no param read before the
first copy). Solo phase 2: NA = 64 - Q2/2 full stages, then Q2 half stages (x = own 32 rows, 4 KB box, + W) computed by
the finisher warpgroup F alone; the other warpgroup P combines the partner rows (F hands its partner rows over in F's
last, never-refilled ring slot), pushes 9 KB with st.async into the unused x half of the partner's phase-2 slots (ready
handshake from the partner's producer), receives the partner's push, adds and hands its own rows to F, which adds and
stores after phase 2. CMB 0 = both warpgroups push (18 KB). Correct on the first build (NaN-poisoned, one-hot exact).
- NVRTC 13 trap (local nvcc 12.1 silent): ptxas C7514 "wgmma serialized ... non wgmma instructions reading
  accumulator registers" -> 12.90 us. Cause: `"r"(__float_as_uint(acc[i]))` operands of st.async made NVVM carry the
  accumulators as integers through the phase-2 loop phi; the int->float movs sat inside the wgmma pipeline. Fix: "f"
  operands (st.async .b32 accepts f32 registers) and float LDS. Bisected with nvc.py (NVRTC-only, no GPU).
- Same run (box 1, XCL 4, 30 reps): SGLang 15.69 | bo14 11.13 | Q2 8/6/4, NS 12/13: 11.10-11.16 | CMB 0 11.16 | no
  placement (PERM 0) 15.49 (smoke). => tie.
- Trace (per-CTA clock64, us from CTA start; Q2 = 8): full stages end 9.75 (F) / 9.88 (P), phase 2 9.84 -> 10.85
  (0.126 us per 13 KB stage = the SAME cadence as a 17 KB full stage: the ring is release-driven, every stage costs one
  cadence whatever its bytes), P push issued 10.46, partner rows landed 10.78, F end 11.29 (max 11.56).
  Q2 = 0 (same code, exchange after the loop): loop end 10.07 / 10.14, push 10.40, landed 10.71, end 11.06 (max 11.36).
  => the solo phase adds Q2/2 cadences (0.07 us x Q2) of stream and the DSMEM chain (push ~0.1 + transit 0.3 +
  add/handoff 0.1-0.25) is only partly hidden; at m256 (Q3) the loop was per-SM-ingress-bound, here it is DRAM/L2-bound.

Analysis block (iteration 1): nsys / ncu were run once for this kernel family at its final form (iteration 4-5 block:
q5 qkv 12.85 nsys / 14.50 ncu vs kg_qkv 12.37 / 14.34, DRAM 60.6 % / L2 64 % / SM 37 %, long_scoreboard 54 %); per
iteration: clock64 traces above, SASS (168 regs, 0 spills, LOCAL 0; per stage per warpgroup TRYWAIT + WARPGROUP.ARRIVE +
4 HGMMA.64x72x16 + DEPBAR gsb0 1 + SYNCS.ARRIVE; push 9 STAS.128; producer UTMALDG.2D + UBLKCP per stage; no LDG).
Binding unit: W from DRAM at 4.43 TB/s (92 % of 4.8) in the loop; next: the start (iteration 2).

## iteration 2 — start: what the first landing is (probe_start.py, 128 CTAs x 384 thr, 210 KB smem, after the flush)
| variant | copies issued by | stage 0 landed (us after CTA entry, mean / max) | all 12 landed |
|---|---|---|---|
| stage 0 alone (x TMA 8 KB + W bulk 9 KB), issued before any sync | 0.21 | 0.89 / 1.22 | |
| W bulk 9 KB alone / 1 KB alone / x TMA 8 KB alone | 0.16 | 0.82 / 0.58 / 0.82 | |
| prefetch.tensormap first / L2 prefetch of the W page first | | 0.89 / 0.93 (no change) | |
| __syncthreads first, then stage 0 | 0.66 | 1.34 / 1.64 | |
| 12 stages by thread 0 after a sync | 1.51 | 1.64 / 2.13 | 3.20 / 3.45 |
| 12 stages split over 2 warps (x / W, or even / odd stages) | 1.42-1.49 | 1.67-1.79 | 3.18-3.21 |
| stage 0 before the inits + sync, then 11 stages (= Q2's fast prologue) | 1.49 | 0.96 / 1.24 | 3.17 / 3.46 |
Facts: a __syncthreads at kernel entry costs ~0.45 us; the TMA unit issues ~75-90 ns per stage (2 copies) and a second
issuing warp does not help (per-SM unit limit); the descriptor fetch of a 2D TMA costs nothing extra vs a baked bulk
copy; the first landing is ~0.65-0.75 us after issue, size-dependent (1 KB 0.42, 9 KB 0.66).
In the kernels thread 0 issues all NS prologue stages before __syncthreads -> consumers start at 1.45 us although stage
0 lands at ~0.96. ES option (sync after stages 0-1, producer issues the rest): consumers start 0.60, data0 1.03 / 1.18
(qkv) and 1.45 (o, vs 1.57) — but the loop end does not move (qkv loop end 10.40-10.51 in that run): the stream is
DRAM-bound, earlier consumers only wait later. Bench: tie (qkv 11.37 vs 11.39, o 5.37 vs 5.41 / 5.30 vs 5.30).

## iteration 3 — release policy, final hop, k-order stagger (same run, box 0, XCL 3-4)
| variant | qkv | o |
|---|---|---|
| bo14 (Q2) | 11.39 / 11.25 | 5.41 / 5.35 |
| baked placement, exchange after the loop (Q2 = 0) | 11.54 (one graph 11.97) / 11.25 | 5.38 (NS 9, = Q2's kernel) |
| + ES | 11.37 | 5.37 / 5.31 |
| + ES + ST (partner rows stored by P's own-row warps: one smem hop less) | 11.35 / 11.25 | |
| + REL (release each stage right after its own MMAs: wait<0>) | **12.71** | 5.53-5.57 |
| + STG 1/2/5 (column tile nt starts its k loop at nt*STG: x reads de-synchronised) | **14.57-15.06** | 5.50-5.61 |
- REL: more slots in flight earlier -> deeper DRAM queues -> later landings (same as P5's in-flight sweet spot).
- STG: the lockstep k order (all column tiles stream the same k-block at once) is worth 3.3 us at qkv m128: x and the
  two token tiles' W hit L2 together. Never de-synchronise the k order across CTAs here.

## iteration 4 — issue order: W bulk copy before x TMA in every stage (WF), o and qkv
Same run (box 0, XCL 3-4, 24-30 reps), local us:
| run | bo14_lean (Q2) | o WF (9,0,0,0,1) | o ES+WF | o NS 8 WF | o no WF (same code) |
|---|---|---|---|---|---|
| 1 | 5.31 | **5.18** | 5.19 | | 5.34 (ES) |
| 2 | 5.46 | **5.34** | 5.44 | 5.37 | |
| 3 (with the lean q5.py) | 5.31 | **5.26**, q5.py 5.25 | | | |
| WF 2 (prologue: all W copies of stages 2-8 first, then their x) | 5.31 | 5.70 | | | |
qkv with WF: (12,0,1,1,1,0,0,0,1) 11.35 vs bo14 11.62 (bo14 had a 12.42 placement-outlier graph; per-graph medians
11.20-11.47 vs 11.28-11.42) and 11.45 (one 12.26 graph) vs 11.24; solo phase + WF 11.62-11.66 -> qkv stays a tie.
Reading: o's loop is DRAM-bound on W (16 KB of the 24 KB stage); issuing W first takes the x descriptor fetch / TMA
issue (~40 ns) out of the W request path in every stage (O2 measured -0.10..-0.12 for the same change in K's kernel).
Bursting all prologue W first delays the x tiles behind 112 KB of W per SM (+0.4 us; Q3 saw +0.8 at m256).

## iteration 5 — deliverable q5.py (o only; qkv returns None: no qkv variant beat bo14 robustly)
q5.py = the o kernel (9 stages, fast prologue, W first) with no options, no trace code: 13.5 KB, readable, NVRTC at
prepare (~0.3 s incl. W pack 16.8 MB + self-test with one-hot rows at m 128/100/97/128). Local nvcc: 168 regs, 0 spills,
440 SASS instructions, same instruction mix as the dev kernel q5k o:9,0,0,0,1 (+-1 scalar setup instruction).
Correctness (box 0): check.py sub_q5_only.py --layers o --ms 128,100,97 -> ALL OK (smoke); check.py sub_q5.py
(bo14_lean + q5) --layers o,qkv_gate --graph -> "checked 103 cases x 2 seeds + graph: ALL OK" (NaN-poisoned);
exact_all.py sub_q5_only.py --layers o --nesc 200 --ms 97,100,127,128 -> EXACT ALL OK; --nesc 3000 --ms 97,113,128 ->
EXACT ALL OK.
Timing (box 0 only; local us, judge ~= local + 1.0; int_v7 o m128 x1.003 from base_v7.log):
| o m=128 | SGLang | bo14_lean (x vs SGLang) | q5 (x vs SGLang) | delta |
|---|---|---|---|---|
| same run, q5t.py XCL 4 x 30 reps, run 1 (seed 21) | 5.90 | 5.31 (x1.112) | **5.25 (x1.124)** | -0.06 |
| run 2 (seed 31) | 5.97 | 5.33 (x1.119) | **5.23 (x1.141)** | -0.10 |
| run 3 (seed 41) | 6.00 | 5.35 (x1.121) | **5.27 (x1.139)** | -0.08 |
| alternating single-candidate bench.py processes (sub_q5.py vs LN/bo14_lean.py, 60 reps), pair 1 | 6.32 / 6.53 | 5.75 (x1.135) | **5.44 (x1.162)** | |
| pair 2 | 6.43 / 6.46 | 5.73 (x1.129) | **5.48 (x1.173)** | |
| pair 3 | 6.32 / 6.34 | 5.64 (x1.124) | **5.34 (x1.183)** | |
| mean | | x1.129 | **x1.173** | x vs SGLang +3.9 % |
qkv_gate m=128 in the same processes (q5 returns None there, i.e. identical code): x1.393 / 1.371 / 1.387 vs
x1.392 / 1.379 / 1.416 (process noise +-0.2 us on qkv). Same-run x-clone runs also had the dev kernel without WF at
5.34 / 5.30 (= bo14_lean): the gain is the W-first issue order, not the rewrite.
Projected: verify-16 load o 6.3 -> 6.2 judge (-0.08 same-run .. -0.29 process pairs) = +0.02..+0.07 % of the score.

### analysis block (iterations 4-5, shipped kernel q5_o vs bo14_lean's kg_o; qkv variant vs kg_qkv for reference)
1. nsys in-graph (box 0, one process, nsys_shapes.py lean / q5 / q5k): o kg_o 6.34 | q5_o 6.24 | dev q5_o_9_0_0_0_1
   6.22 us; qkv kg_qkv 12.37 (lean) / 12.40 (q5 = same route) | dev q5_qkv_gate_12_0_1_1_1_0_0_0_1 12.85.
2. ncu (claude/t3/ncu_run.sh + --page details, full set, flushed, clock-control none):
   | kernel | dur | DRAM % / bytes / TB/s | L2 % | SM busy % | issue % (slots) | gmma inst % act | occ % | elig/clk | top stalls |
   |---|---|---|---|---|---|---|---|---|---|
   | q5_o (shipped) | 7.33 | 48.6 / 17.05 MB / 2.33 | 59.6 | 30.3 | 14.7 | 5.6 | 14.7 | 0.26 | long_sb 28, wait 20, barrier 19 |
   | kg_o (bo14_lean) | 7.46 | 47.8 / 17.05 MB / 2.29 | 58.7 | 30.5 | 15.0 | 5.5 | 14.7 | 0.27 | long_sb 29, wait 20, barrier 18 |
   | q5 qkv ES+WF (not shipped) | 14.50 | 60.6 / 39.86 MB / 2.91 | 64.0 | 37.2 | 15.4 | 5.1 | 13.7 | 0.24 | long_sb 54, wait 19, selected 8 |
   | kg_qkv (bo14_lean) | 14.34 | 60.7 / 39.86 MB / 2.91 | 65.3 | 36.8 | 14.7 | 5.1 | 13.9 | 0.22 | long_sb 50, wait 17, selected 11 |
   (long_scoreboard = mbarrier full-waits on TMA data; ncu durations are ncu-inflated, ratios hold)
3. SASS (local nvcc 12.1 -arch sm_90a on the lean source; NVRTC 13 on the box: 168 regs, LOCAL_SIZE_BYTES 0, no C75xx):
   q5_o 440 instructions, 0 spills; hot loop per stage per warpgroup: SYNCS.PHASECHK.TRYWAIT, WARPGROUP.ARRIVE,
   4 HGMMA.64x64x16.F32.BF16, WARPGROUP.DEPBAR.LE gsb0 1, 1 SYNCS.ARRIVE (release); producer per stage UBLKCP (W, 16 KB)
   then UTMALDG.2D (x, 8 KB) — the order is the change; no LDG anywhere (no sunk loads possible); epilogue 16 F2FP +
   16 SHFL.BFLY + 4 STG.128 per thread.
4. Binding unit: o = the W stream from DRAM: inside the loop 16 KB/stage x 64 column tiles per 0.225 us = 4.66 TB/s
   (97 % of 4.8); averaged over the kernel only 48.6 % because ~30 % of the time is outside the loop: first landing
   1.55-1.66 us after CTA start (TMA issue 0.25 + ~1.3 us first-stage latency under the 27.6 MB prologue burst),
   epilogue 0.30 us (16 KB of BF16 per CTA at the ~30 B/clk per-SM store path), CTA end skew ~0.35 us. Tensor busy
   30 % (MMA need 0.16 us per stage vs 0.225 us cadence). Trace with / without W-first: producer's last refill 3.49 /
   3.60, loop end 5.19 / 5.26, CTA end 5.49 / 5.56 us (data0 1.66 / 1.55: x now lands after W). Next (not done):
   the start is the largest fixed cost (stage 0 alone lands 0.96 us after entry, 1.55 inside the burst), but issuing
   fewer stages up front starves the DRAM pipe and issuing W first in bulk delays x (+0.4); a smaller first stage
   (k32) or an x-only first copy are the remaining candidates.
   qkv = DRAM/L2: loop 4.43 TB/s of W (92 % of 4.8) + 64 MB of x L2->SM; ring depth saturated at 12 stages; start and
   exchange tail (~0.9 us) are the only slack and the solo phase trades one for the other at this shape.
