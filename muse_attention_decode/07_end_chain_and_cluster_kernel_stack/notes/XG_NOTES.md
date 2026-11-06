# >>> XG LATEST: dev/xg/deliver_r5/ (r3 + ASK + PFT 150; k51xg5.py md5 9256d8f0): v1f nsys 12.30 / 12.22 / 12.22 vs r3 12.32 / 12.38 / 12.38; check.py + t_xg full 0 failures <<<
# >>> XG ROUND 2 HAND-BACK: dev/xg/deliver_r3/ (xg_mod.py md5 a60d31e5, k46xg3.py md5 2d2ecc86 = k46 + r3): v1f nsys 12.35 / 12.38 / 12.37 vs
#     k46 13.15 / 13.12 / 13.09 and r2 12.45 / 12.56 / 12.48 (same job); check.py + t_xg full 0 failures. (r2: deliver_r2/, 12.50 / 12.59.) <<<
# >>> XG ROUND 1 HAND-BACK: dev/xg/deliver_r1/ (xg_mod.py md5 b2c1bdfd, k44xg.py md5 c3c192ac): v1f nsys 13.25 / 13.28
# vs r5 14.05 / 14.14 and k44 14.19 / 14.11 (same job); bench 12.27 / 12.35 vs 13.07; check.py + t_xg full 0 failures <<<
# XG: verify-1 full (v1f) after XA (jump mode). Owner of v1f only; d1f stays XA/XB's.

## Files
- xg_mod.py: own NVRTC module (_XG_SRC, kernel `xg_v8f`) + `_xg_ok` predicate (B 1, q_len 8, full, page 1, tp8,
  fits 10 tiles) + kernel_fn wrapper appended last. Knobs via XG_DEFS (or K2_DEFS), read at import.
- build.py `<base> <out>`: appends xg_mod.py (stack on dev/coord/k44.py; for A/B jobs I build on k34: k34xg.py).
- ref/: frozen copies: k34.py, k44.py (coordinator base, verified 0 failures), r5.py (= XA deliver_r5/k34xa_r5.py,
  md5 0bd38fb5), xa_mod_r5.py (md5 6e648a42).
- t_xg.py (XA's t_xa edge tests, q_len 8, 3 calls each), run_nsys.py / nsys_loop.py / nsys_parse.py (XA's, + xg_),
  lc.py (local nvcc 12.1 sm_90a: regs, spills, SASS histogram), gN.sh job scripts, out/gN.txt logs.

## Starting point (XA r5, nsys = judge-visible): v1f 13.94-14.00 us (k34 16.75). XA decomposition: skeleton
(start + gather + publish) 9.8, exposed compute ~2.5, combine after the slowest arrival ~2.0.

## Design v1 (xg_v8f): tile phase = XA r5 exactly (T128, QLATE, MBP, DEFER 4); NEW END CHAIN (tagged slots)
XA's publish path per CTA: parity merge -> 17 KB of partial-row stores -> bar -> atom.acq_rel.gpu (MEMBAR.GPU waits
for the store acks, then an L2 round trip) -> the last 128 arrivals spin on the counter -> one bulk L2 round of slot
loads in a cold noinline combine -> reduce -> epilogue. Everything after the slowest CTA's stores is serial.
xg: every 64-bit word of a partial row carries a 1-bit call tag (call parity) and is stored with st.relaxed.gpu.b64
(single-copy atomic). No fence, no atomic round trip on the publish path: tid 0 adds 1 to a u64 publish counter with
a fire-and-forget red.relaxed (a hint only). Combiner = CTA c < 128 (row c); it waits until the counter shows
>= G - XG_GATE publishes (XG_GATE 8), loads row c of every partial (one bulk round, code now warm), keeps the words
whose tag matches and re-polls ONLY the stale words (the slowest CTAs) until they flip. After the slowest CTA's stores
only one poll round trip + the reduction/epilogue remain.
- Call index k = floor(counter / G) from a cp.async copy of the counter taken at the kernel start (tracked by the Q
  mbarrier): every CTA of call k reads k G .. k G + G - 1 whatever the others did (robust to start skew). Tag = (k+1)&1
  (call 0 tag 1: the zeroed workspace reads as stale). Each call writes every slot word exactly once, so a slot holds
  either this call's or the previous call's tag.
- O word = 4 f16 (O / l), tag in bit 0 (1 f16 ulp); ML word = (M_eff | W << 32), tag in W's mantissa LSB (masked off
  before use). Rare exact path (row max out of [REF - 90, REF + 40]): tag-spinning loads, as XA's slow merge.
- Knobs: XG_GATE (8; 0 = wait for all; >= G = no wait), XG_NOF (each WG publishes its own rows, no parity merge,
  2 G partials), XG_PF (pairs first, single tile last), XG_DEFER (4), XG_PNS (32 ns), XG_TB (1 no combine, 2 + no
  tile compute: timing-bound, wrong output).
- Local compile (nvcc 12.1): 125 regs, 0 spills, 16 GMMA / 10 DEPBAR (same as xa_b1f_8: no serialization).
  REDG.E.ADD.64.STRONG.GPU for the counter, STG.E.64.STRONG.GPU slot words, LDG.E.64.STRONG.GPU polls.
- Observation (also in XA r5): ptxas schedules the 2 Q LDGSTS per thread ahead of the page-table LDGs (start path).

## Jobs (nsys = median kernel duration, 60 reps, dev/xg/run_nsys.py; same container within a job)
- g1 (out/g1.txt, mod 7e660ad3): smoke 0 failures for GATE 0 / none, NOF, PF, PF + D2 (the default's smoke printed
  nothing: the first cold import exceeded `timeout 120`; nsys of it ran fine). nsys: r5 14.11 / 14.11 | xg (GATE 8)
  13.63 / 13.70 (-0.45) | GATE 0 13.78 | no gate 16.74 (early bulk round + ~66 stale partials polled by 128 combiners
  floods L2: ~100K sectors/us) | NOF 14.99 (2x partial bytes, 4.6 MB stores + loads: dead, as XA's NOF) | PF 15.39,
  PF + DEFER 2 14.34 (dead: step 0 = a pair delays the deferred wave) | TB1 (no combine) 11.68 | TB2 (no compute, no
  combine) 9.34. => compute exposure 2.34 us, combine after the slowest publish still ~2.0 us: the slowest CTA itself
  (or the combiners) does gate poll RTT -> bulk round RTT (+2.3 MB) -> stale poll in series.
- g2 (out/g2.txt, mod 072d9755, XG_BF 1 was the default there): smoke 0 failures (default, BF 0, GATE 4 / 16, DEARLY,
  DEARLY + D6). nsys: r5 13.94 / 13.95 | **BF 0 GATE 8 13.38 (-0.56)** | BF 1: 16.91 / 16.91 (GATE 4 / 16 / 32 and
  DEARLY all ~16.9 = confounded by BF 1) | TB1 11.55.
  Trace (XG_TRACE, us from the first warp, min / med / max): BF 0: q-ready 2.43 | step0 2.96 (max 4.58) | loop end
  7.81 (max 8.78) | publish 8.99 (max 10.35) | gate + bulk done 11.49 (max 11.90) | all fresh 11.54 (11.95) | out
  11.81 (12.62). BF 1: bulk done med 12.86 (!), out max 15.73.
  KEY FINDING: a load of a slot NOT yet written in this call misses L2 (last call's lines were evicted by the judge's
  flush) and goes to DRAM behind the K/V stream (~4 us while it runs, ~1 us after). BF 1's early bulk round and any
  early polling pay it; with GATE 8 the <= 8 stale slots still make the bulk round ~1 us.
  -> XG_PFS (default 1): each CTA L2-prefetches its own 136 slot lines right after its burst.
- g3 (out/g3.txt, mod 48b32391): smoke 0 failures (default, BF 1, LS, DEARLY, LS + DEARLY). nsys: r5 14.05 / 14.02 |
  PFS 1 (GATE 8) 13.66 / 13.68 | PFS 0 13.50 | PFS + GATE 16 13.63, GATE 32 13.92 | PFS + BF 1 15.52 | LS 14.11 |
  DEARLY 13.97 | LS + DEARLY 14.43 | TB1 11.74 | **TB3 (no compute, no publish) 7.74 | TB3 + DEFER 0 7.65**.
  => LS (2 singles instead of the last pair), DEARLY (deferred wave at step 0's QK) and PFS (slot prefetch after the
  burst) are dead ends; waves vs all-at-once does not change the gather skeleton (7.7). The publish alone (TB2 9.34
  - TB3 7.74) costs 1.6 us in the no-compute skeleton: cold code? (post-loop publish SASS 0xb2a0-0xc540 = 4.8 KB,
  combine 0xc540-0xf190 = 11 KB). Trace: loop end med 8.16 max 9.14 | publish med 9.28 max 10.69 | bulk 11.84 |
  fresh 11.89 | out max 12.69. Device code cannot take a code address usable by prefetch.global (function pointers
  are code-segment offsets / descriptors).
- g4 (out/g4.txt, mod b667f921): **COLD CODE IS LINEAR: ~0.15 us per KB** of never-executed straight-line SASS after
  the tile loop: TB3 7.74 | + 1 KB 7.92 | + 4 KB 8.35 | + 8 KB 8.98 | + 16 KB 10.11 (all 512 threads execute it).
  Publish split: TB4 (publish code, stores suppressed) 8.35 = TB3 + 0.61 (its ~4.7 KB of code) | TB2 (with the 2.3 MB
  of slot stores) 9.34 = TB4 + 1.0 (stores) | TB2 + PFS 1 9.76, PFS 2 (prefetch before the burst) 9.31: not L2 misses.
  Full: PFS 0 13.50 / 13.55, PFS 2 13.98; r5 14.11 / 14.02. -> PFS default 0.
  => the post-loop path (publish 4.7 KB + combine ~7-8 KB executed) costs ~1.8 us of instruction fetch alone, and the
  partial stores ~1 us. Levers: compact one-shot code, fewer partial bytes.
- Coordinator note (XB d1f TORD: tile-ordered burst, per-tile mbarriers): v1f's burst already is tile-ordered (all
  512 threads issue tile 0, then tile 1, ...; cp.async.mbarrier.arrive.noinc), but with one mbarrier per tile PAIR
  (XA's MBP) -> XG_MBP 0 (per-tile mbarriers, 10 inits) added for g5.
- g5 (out/g5.txt, mod: CP / CC / MBP knobs): smoke 0 failures (all). nsys: r5 13.98 / 13.98 | base 13.34 / 13.34 |
  CP 13.22 | CC 13.25 | CP + CC 13.28 / 13.22 | MBP 0 13.18 | **CP + CC + MBP 0 12.75** | SV 1 (weak stores) 13.33 |
  TB2 9.22 | TB2 SV 1 9.34 | TB2 ML-only stores 8.42 (store VOLUME costs ~0.8 us) | TB2 CP 8.86 (coalesced CP stores
  -0.36) | TB4 CP 8.48. Trace (CP + CC, MBP 1): s0 wait 2.61 -> go 2.94 | s1 wait 4.59 go 4.72 | s2 wait 6.30 go 6.45 |
  loop end 7.95 (max 9.15) | publish 9.76 (max 10.85; per CTA publish - loop end 1.76!) | fresh 11.68 | out max 12.59.
- CODE SIZE (SASS, CP + CC + MBP 0 build: 65 KB): prologue + burst 15 KB (the burst alone: 359 instrs for 12 copies,
  unrolled), single step A 8.8 KB, pair step B 13 KB, single step C 8.2 KB (unused at nt 10), publish 3 KB, combine
  9 KB. A and B are first executed AFTER the burst (i-fetch behind the K/V stream).
  -> XG_CB (rolled burst / deferred loops): 4040 -> 3280 instrs; XG_US (one pair-step instance in one rolled loop,
  step 0 and an odd last tile = pairs with a fully masked dummy 2nd tile): -> 2912; + XG_DRY (combine runs once with
  all memory ops predicated off when >= XG_DRYM (32) CTAs are still unpublished; mode from tid 0's first counter
  read, which is also the gate check): CB + US + DRY 2488 instrs, 40 KB. All 0 spills, 8 GMMA / 4 DEPBAR (no
  serialization).
- New defaults: CP 1, CC 1, MBP 0 (PFS 0, BF 0, GATE 8).
- g6 (out/g6.txt, mod 75f9bbf0): smoke 0 failures (base, CB, US, CB + US, CB + US + DRY); t_xg full printed nothing
  (timeout 600 too short). nsys (2 copies): r5 14.14 / 14.05 | **base (CP + CC + MBP 0) 13.34 / 13.26** | CB 13.34 /
  13.41 | US 14.14 / 14.16 | CB + US 14.30 / 14.27 | CB + US + DRY 15.02 / 15.01, DRYM 16 15.07 | TB1 (CB + US) 12.16.
  => g5's 12.75 was a one-copy outlier; CP / CC / MBP 0 are within noise of the g2 base. CB (rolled burst) neutral:
  the start code runs before the stream. US loses 0.8 (the dummy half-step = 2 QK + 2 PV + V^T on step 0, and the
  deferred wave goes out later). DRY loses 0.7 (combiners publish within ~0.6 us of each other: the dry pass delays
  them). The g4 cold-code probe (all CTAs hit the same cold code at once) overstates the cost in the real kernel,
  where early CTAs pull the code into L2 for the late ones.
  STABLE RESULT jobs 2-6: xg (tagged end chain) 13.26-13.50 vs r5 13.94-14.14 in the same job: -0.55..-0.8 us.
  Trace (CB + US + DRY): the slowest CTAs start late (step 0 at 3.6-4.4 vs median 2.9-3.2) and stay ~1 us late;
  last publish -> combiners all-fresh ~1.7 us (= nsys full - TB1).
- g7 (out/g7.txt, mod dd5643c1): nsys r5 14.08 | base 13.25 / 13.28 | PNS 0 13.34 | GATE 4 13.28 | GATE 0 13.47 |
  TB1 11.36 (combine after the slowest publish ~1.9 us). The trace and t_xg full printed nothing -> g8.
- g8 (out/g8.txt): t_xg FULL on the base: 60/60 PASS, 0 failures (also with CUDA_LAUNCH_BLOCKING); r5's t_xa: 0
  failures. The g6 / g7 "stop at the 21st case line" was MY job-script bug: every PASS line contains "error", so
  `grep error | head -20` closed the pipe after 20 lines and SIGPIPE killed t_xg (use tail, never head, on t_xg).
  The XG_TRACE build (g7 stamps) hangs (rc 124): trace-only __reduce_max_sync after the divergent poll loop (now
  replaced by lane 0's own count); not present in non-trace builds.
- deliver_r1 (frozen, dev/xg/deliver_r1/): xg_mod.py md5 b2c1bdfd (defaults: tagged end chain, CP 1, CC 1, MBP 0,
  PFS 0, BF 0, GATE 8, DEFER 4), k44xg.py = build(ref/k44.py) md5 c3c192ac, k34xg.py = build(ref/k34.py) md5 80b93e0e.
  r1fin (out/r1fin.txt, `bash dev/xg/deliver_r1/final.sh dev/xg/deliver_r1`): t_xg quick + FULL 0 failures,
  tools/check.py dev/xg/deliver_r1/k44xg.py **0 failures**. bench 40 reps (same job): verify 1 full k34 16.02 | r5 13.07
  | k44 13.08 | **xg 12.27 / 12.35**; decode 1 full k44 10.27 | xg 10.32 / 10.30 (unchanged: xa_b1f_1L). nsys v1f: r5
  14.05 / 14.14 | k44 14.19 / 14.11 | **xg 13.25 / 13.28 (-0.83 us)**; d1f 11.33 = k44 11.33.

## Resume here (after deliver_r1)
State: xg_v8f nsys ~13.25-13.34 vs r5 13.94-14.14 (same job, 6 jobs): -0.55..-0.8 us, all from the end chain.
Decomposition of the current kernel (nsys, timing-bound builds): TB3 (start + gather, no publish) 7.74 | TB2 (+ CP
publish) 8.86 | TB1 (+ tile compute, no combine) 11.36-11.7 | full 13.25. So: publish ~1.1, exposed compute ~2.5,
combine after the slowest publish ~1.8 (floor ~0.8: store visibility + one poll RTT + warm reduce / epilogue).
Measured facts worth keeping:
1. Cold code on a one-shot path costs ~0.15 us per KB when every SM hits it at once (TB3 + CW probe), but in the full
   kernel early CTAs pull it into L2 for the late ones: compacting (CB, US) and dry passes (DRY) bought nothing or lost.
2. Partial-store VOLUME costs (~0.8 us for 2.3 MB in the skeleton); CP's 128-B-coalesced (row, word) layout -0.36.
   L2 prefetch of the slot lines does not help (PFS). Weak vs relaxed.gpu stores: same.
3. Loads of not-yet-written slots miss L2 (last call's lines are flushed) and wait behind the K/V stream (~4 us while
   it runs): never poll or bulk-load early (BF 1: +3 us; no gate: +3 us).
4. Tile phase: WGs barely wait for data (trace: 0.13-0.15 us per step) -> compute-bound; a pair step executes ~310
   instructions per warp (QK + V^T 120, softmax ~96, PV ~80) + 232 for the deferred issue on step 0; ~1.2-1.5 us per
   pair step with 4 WGs.
Next levers (ranked):
1. Combine (~1.0 us above its floor): the gate -> bulk round -> stale poll chain after the 124th publish. Try a
   gate at G - 16 with the stale-slot polls throttled by a CTA-wide pending count (only threads with stale words
   poll, others sleep on a bar), or halve the partial bytes (cluster-pair DSMEM pre-reduction to 66 partials).
2. Tile compute: instruction count per step (softmax ~96 + V^T ~20 + addressing): e.g. compute the f16 P codes for
   two tiles with fewer max / prmt ops, hoist descriptor math out of the step (empty asm "+l" pins), smaller
   deferred-issue block (it costs step 0 ~232 instructions; a rolled loop CB made no difference end to end).
3. Trace build (XG_TRACE) hung in g7 / g8 (rc 124) after the 32-stamp change; non-trace builds are fine. Fix before
   relying on traces (suspect: trace-only code in the CC poll loop).

## Round 2 (coordinator: r1 stacked as dev/coord/k46.py; target v1f nsys <= 12.3)
- XG_RK (rank-ordered combiners): tid 0 replaces the publish red with atom.add.relaxed.gpu.u64 (old = kG + arrival
  rank; old + 1 also replaces the combiner's first counter read = one RTT fewer); ranks < XG_NCMB combine rows rank,
  rank + NCMB (NCMB 128: 1 row, 64: 2 rows per CTA, 16 partial groups x RR 9); the last arrivals (incl. the slowest
  CTA, which was a combiner 97 % of the time with the fixed c < 128 mapping: its own counter RTT + bulk RTT + cold
  reduction sat on the critical path) exit right after their publish.
- XG_D32 (descriptor low-word adds) + XG_PIN (ldmatrix / stmatrix addresses pinned): -40 static instrs (~3 %).
- r2a (out/r2a.txt, mod 04f09ece): smoke 0 failures (all), t_xg FULL 0 failures (RK, RK NCMB 64). nsys: r1 13.25 /
  13.28 | RK 13.73 / 13.74 | RK NCMB 64 14.37 / 14.37 | RK + D32 + PIN 13.66 / 13.66 | RK + DRY 14.91 | RK + GATE 16
  13.65 | RK TB1 11.41 (= r1's) | D32 + PIN (no RK) 13.22 | r5 14.11. => RK +0.47 (the whole loss is after the last
  publish: full - TB1 2.32 vs r1 1.89). Premise error: CTA 131 (new block, a non-combiner under c < 128) is the
  usual slowest CTA (XA), so r1's critical path was already the pollers' detection.
- r2b (out/r2b.txt): r1 13.12 / 13.14 | RK + PFS 1 13.79 / 13.79 | RK + PFS 2 13.97 / 13.89 | RK + PFS 1 + D32 + PIN
  13.86 / 13.82 | PFS 1 13.50 / 13.52 | PFS 2 13.57 / 13.41 -> slot L2 prefetch costs ~0.3 and does not rescue RK.
- r2c (out/r2c.txt): XG_TRACE works again (rc 0). r1 timeline after the slowest publish (10.48; CTA 131 slowest in
  5/8 reps): gate passed med 10.43 (max 10.98) | bulk round returned 11.26 (max 11.74: one round = ~0.8 us) | all fresh
  11.31 (0 poll iterations: nothing stale by then) | out med 12.11, max 12.45 = **0.8 us for the reduction +
  epilogue** (should be ~0.15 warm): cold code, every combiner runs it for the first time at the same moment.
  RK: everything ~0.35 later (gate 10.77, bulk 11.71, out max 12.75). Per-CTA publish - loop end 1.76 (r1) / 2.11 (RK).
  CTA 131 (new block): step 0 at ~5.0 vs 3.1-4.3 for the others (its WG 0 runs the new-block code after the burst).
- XG_WS (warm start): the kernel top level becomes a 2-pass loop; pass 0 runs only the combine body dry (every load /
  store predicated off, mode 0) while the prologue's page-table / Q / new-token loads are in flight (before the start
  barrier); pass 1 = the real kernel -> the combine's code (one copy) is warm in L1.5 / L2 at the end. A noinline
  dry call was rejected: saving the in-flight load registers across the call ABI would stall on them. Also: the new
  block's RMSNorm / e4m3 code is pinned after the Q wait (empty asm on kraw / vraw): in the WS build ptxas had hoisted
  it above the start barrier (spilling 4 x 16 bit), which would stall the last CTA's start on the new-token loads.
- r2d (out/r2d.txt, mod cf7fb094): smoke + t_xg full (WS) 0 failures. nsys: r1 13.12 / 13.15 | current default 13.31
  / 13.30 | **WS 15.74 / 15.74 (dead: +2.6)** | WS + D32 + PIN 15.68 / 15.68 | WS + GATE 16 / 32 15.78 / 15.78 |
  **D32 + PIN 13.12** (= r1; the current default build is +0.17 vs r1: codegen shift from the non-const ldsm / stsm +
  unused descriptor locals -> D32 + PIN restores it). WS trace: burst issued 4.46 (normally ~2.0: the dry pass at the
  start costs ~2.5 us of cold fetch, far more than the page-table wait), and all-fresh -> out STILL ~1.1 us with the
  combine code warm -> the slow combine tail is NOT cold code. Suspect: the epilogue gate load (gwe, issued at combine
  entry; relies on the start-time L2 prefetch: a DRAM round trip if evicted) -> XG_GSM.
- XG_GSM: combiner row c's 128-B gate line cp.async'ed into smem with the Q copies (Q mbarrier); the epilogue reads
  smem. XG_CLE: the new block's cache_loc loaded in the prologue (was a cold dependent load after the Q wait on CTA 131,
  the usual slowest CTA: its step 0 at ~5.0 vs ~3.1-4.3).
- r2e (out/r2e.txt, mod 4d09489c): smoke + t_xg full (D32 + PIN + GSM + CLE) 0 failures. nsys: r1 13.22 / 13.25 |
  D32 + PIN 13.12 / 13.15 | + GSM 13.09 / 13.09 | + CLE 13.15 / 13.18 | + GSM + CLE 13.15 / 13.15 -> GSM -0.03 (noise
  level), CLE makes CTA 131 stop being the tail (trace: slowest ids random) but someone else takes over (as XA's LSH).
  Trace: issued 1.86 | step0 2.94 | loop end 7.86 | **publish 9.74 (per CTA publish - loop end 1.74 us)** | gate 10.38
  | bulk returned 11.22 | all fresh 11.28 | out 12.05 / max 12.24.
  Publish analysis: CP moves ~190 KB through smem (dump 90 KB STS + 98 KB LDS ~ 0.8 us at 128 B/clk); the old merge
  78 KB but scattered stores into 136-B (sector-misaligned) rows.
- XG_MF (f16 partner exchange: WG (h, 1) writes O / l f16 + (l, M) = 80 B per thread = 20 KB, WG (h, 0) folds in
  registers and stores its 2 rows x 4 words + ML) and XG_AL (slot layout: 128-B O rows = one line, ML words in their
  own array; CP and MF stores become full-sector writes; combiner loads via xg_ow / xg_mlw).
- r2f (out/r2f.txt): smoke + t_xg full (MF + AL) 0 failures. nsys: r1 13.25 / 13.18 | base (D32 + PIN + GSM) 13.28 /
  13.28 | + AL 13.09 / 13.02 | + MF 13.18 / 13.22 | **+ MF + AL 12.74 / 12.61 (-0.54 vs r1)** | TB1: MF + AL 11.30,
  base 11.38 -> the gain is in the combine after the last publish (1.37 vs 1.90 us): with 128-B O rows every MF store is
  a full 32-B sector and every combiner O load one line (no partially written sectors). The XG_TRACE build (128 regs)
  prints C7511 (wgmma serialized): trace timings of this config are pessimistic; the real build has 123 regs, no C75.
- New defaults: MF 1 (CP 0), AL 1, D32 1, PIN 1, GSM 1 (CC 1, MBP 0, GATE 8, DEFER 4).
- deliver_r2 (frozen, dev/xg/deliver_r2/): xg_mod.py = the current module with UNIQUE Python globals (_XG2_* /
  _xg2_* / _PREV_KFN_XG2; CUDA source byte-identical to dev/xg/xg_mod.py's) so it can be stacked on k46 (which
  already contains r1's module: same names would make r1's wrapper call itself through r2's globals). k46xg2.py =
  ref/k46.py (coordinator k46, md5 cad4c213, verified 0 failures) + r2; k34xg2.py = k34 + r2.
  r2fin (out/r2fin.txt, `bash dev/xg/deliver_r2/final.sh`): t_xg quick + FULL 0 failures, tools/check.py
  dev/xg/deliver_r2/k46xg2.py **0 failures**. bench 40 reps: verify 1 full k34 16.30 | k46 12.45 | **xg2 11.63 /
  11.80**; decode 1 full k46 10.02 | xg2 9.98 / 9.99 (unchanged). nsys v1f: k46 13.22 / 13.20 | r1 13.18 / 13.22 |
  **xg2 12.50 / 12.59 (-0.66 vs k46)**. (d1f nsys parse n=0: k46's d1f kernel is XB's, not matched by the name filter.)
- XG_BFI (branch-free burst / deferred issue; predicated cp.async for the new-block slots; hoisted offsets): burst 233
  -> 204 instrs, step-0 deferred block 142 -> 92; but 128 regs (watch C7511 on NVRTC 13).
- r2g (out/r2g.txt): nsys r2 12.46 / 12.48 | base (= r2 code) 12.42 / 12.45 | **BFI 12.38 / 12.30** | GATE 4 12.46 |
  GATE 16 12.54 | PNS 0 12.46 | CLE 12.51. Trace (MF + AL): per-CTA publish - loop end 0.93 (CP: 1.74) | slowest publish
  9.71 -> gate max 10.29 -> bulk returned 10.93 -> all fresh 10.99 -> bar.red 11.18 -> 16-warp sum 11.41 -> epilogue
  store 11.66 -> out 11.73 (a serial one-warp tail of ~0.75 us after the last word is fresh).
- **ptxas C7511 on the image (NVRTC 13)**: "wgmma.mma_async instructions are serialized due to insufficient register
  resources" for the r2 / default build (local nvcc 12.1 does not report it). r2h (out/r2h.txt, compile only, XG_LOG +
  --ptxas-options=-v): every MF=1 config has C7511 (127 regs; PIN 0 + D32 0: 124 regs, still C7511); MF 0 + CP 1:
  126 regs, NO C7511; r1-like config 123 regs, no C7511. => MF's publish block (o[36] + 20 staged registers) breaks
  the wgmma pipeline's register allocation: r2's end-chain gain was paid partly with a serialized tile loop.
- XG_MF 2: low-register MF (partner stores pairs as converted; the folding WG loads l1 / M1 per row and the 2 partner
  words per output word just in time).
- r2i (out/r2i.txt): MF 2 still C7511 in every variant (123-127 regs) -> C7511 is triggered by the MF code shape, not
  the register count; and it costs nothing measurable (TB1: MF + AL 11.30 vs CP 11.38). nsys: r2 12.58 / 12.58 | MF 2
  12.74 / 12.62 | MF 2 + BFI 12.64 / 12.54 | MF 2 + PIN 0 + D32 0 12.94 / 12.85 | others 12.64-12.77. BFI: -0.07..-0.1
  in 3 comparisons (r2g, r2i x2).
- deliver_r3 (candidate, dev/xg/deliver_r3/): r2 + XG_BFI 1 (unique Python names _XG3_*; stacks on k46 or on k46 + r2);
  r3fin (out/r3fin.txt): t_xg quick + FULL 0 failures, tools/check.py dev/xg/deliver_r3/k46xg3.py **0 failures**.
  nsys v1f (3 copies each, same job): k46 13.15 / 13.12 / 13.09 | r2 12.45 / 12.56 / 12.48 | **r3 12.35 / 12.38 /
  12.37** (-0.13 vs r2, -0.75 vs k46; p90 12.70-12.93 vs r2 13.02-13.28). (The bench step died silently after its
  2nd import in that job; re-run separately: out/r3bench.txt: verify 1 full k34 16.30 | k46 12.57 | **r3 11.83 /
  11.89**; decode 1 full k46 10.17 | r3 10.05 / 10.08 = unchanged.) => deliver_r3 FROZEN = the round-2 hand-back.

## Resume here (after round 2 = deliver_r3)
- v1f nsys ~12.37 (k46 13.12). Phases (MF + AL build, trace): publish 0.93 us per CTA after its loop end; after the
  slowest publish: gate (+0.6) -> bulk round (+0.64) -> all fresh -> bar.red (+0.19) -> 16-warp sum (+0.23) ->
  epilogue (+0.25) -> out. Exposed tile compute ~2.4 us (WGs compute-bound, ~0.15 us data waits per step).
- Dead this round: RK (rank-ordered combiners +0.47), NCMB 64 (+1.1), RK + DRY (+1.2), PFS with RK (+0.3..0.6), WS
  warm start (+2.6: the dry combine at the start costs ~2.5 us of cold fetch and the end was not code-bound), GSM /
  CLE (noise), GATE 4 / 16, PNS 0 (noise), MF 2 (no gain; C7511 stays and is harmless).
- C7511 (wgmma serialized) appears on NVRTC 13 for every MF build but has no measurable cost (TB1 equal).
- Next levers: (1) the one-warp serial tail after the last fresh word (~0.4-0.7 us: bar.red + 16-warp sum +
  epilogue): spread the final sum / epilogue over warps; (2) cluster-pair DSMEM pre-reduction (66 partials: half the
  publish stores and bulk-round bytes, costs a pair exchange); (3) tile-step instruction count (softmax ~96 / step).

## Round 3 (coordinator: r3 stacked as dev/coord/k49.py; target v1f nsys <= 11.8)
- XG_EG (early gate read): tid 0 issues ld.relaxed of the publish counter right after its tile loop (the RTT hides
  behind the ~0.9 us publish); a CTA that is already late (count + 1 >= G - GATE) skips the gate's dependent counter
  read and goes straight to its bulk round (the gate only exists to keep EARLY CTAs off flushed slot lines).
- XG_PR (parallel final reduction): after bar.red each warp w sums the 16 warps' partials of its own 4 d with a 4-step
  shuffle tree (was: 16 threads x 16 serial adds), one more smem round for the RMSNorm sum of squares, each warp's
  lane 0 stores its 4 outputs.
- Working module now has XG_BFI 1 by default (= r3).
- r3a (out/r3a.txt, mod 2c15f0b4): smoke + t_xg full (EG + PR + CLE) 0 failures. nsys (2 copies): r3 12.48 / 12.42 |
  base (= r3 code) 12.51 / 12.42 | **EG 12.70 / 12.74 (+0.27)** | **PR 12.82 / 12.75 (+0.3)** | EG + PR 12.99 / 12.86 |
  EG + PR + CLE 12.85 / 12.93 -> both DEAD. Trace (EG + PR): bar.red 11.06 -> 4-step shuffle sums 11.38 -> second
  smem round + stores 11.71 -> out 11.79: the parallel tree is not faster than the 16 serial adds here (trace build is
  C7511-distorted; nsys says +0.3). EG: a late CTA skipping the gate read did not shorten the chain (+0.27; cause not
  isolated - the extra live register / ordering of the early relaxed load around the publish).
- Not built (estimates from team data): cluster-pair DSMEM pre-reduction to 66 partials: a ~17 KB DSMEM exchange per
  pair costs ~0.8-1.2 us on the critical path (WB / XB / XE measurements), more than the ~0.6 it saves (half the
  publish stores + bulk-round bytes) -> not worth it. 2-MB-page-ordered gather: ~6 pages per tensor at B = 1, the TLB
  covers them -> no effect expected. FP8 partials (would halve the bytes) rejected: coarser than BF16 where SGLang
  keeps FP32 partials.
- Round-3 hand-back = deliver_r3 unchanged (v1f nsys ~12.37-12.45 across jobs).
## Resume here (after round 3)
- The end chain is now: publish ~0.9 us per CTA, then after the slowest publish ~1.1-1.4 us (gate + bulk round + one
  warp's reduction / epilogue). Every attempt to restructure the tail (RK, EG, PR, DRY, WS, GSM, CLE) was neutral or
  worse. Biggest remaining phases: exposed tile compute ~2.4 us (compute-bound steps, ~310 instrs / warp / pair step)
  and the CTA-to-CTA spread of the loop end (median 7.9 vs max 8.66 in traces: random gather service, the end waits
  for the slowest of 132): dynamic tail balancing (a global pool of last tiles claimed by early finishers, merged into
  their own running partial: no extra partials) is the untested big lever.

## Final bounded round (coordinator: exposed tile compute; freeze only on >= 0.3 us)
- r4a (out/r4a.txt): TIMING-BOUND tile ablations (XG_ABL, wrong output), nsys: base (= r3) 12.32 / 12.35 | r3 12.29 |
  **no V^T transpose -0.46 (11.87)** | no P byte packing -0.27 (12.06) | no clamp -0.17 (12.16) | no row max / alpha
  (ABL 2) and all-ablations break the combine (83 us: slow path) | no-combine: TB1 10.56 vs all ablations 9.09 (-1.47) |
  TB2 (no compute) 8.10. => at B = 1 CUDA-core step work DOES convert (unlike XD's power-capped b32): ~1.47 us of the
  ~2.46 us exposed compute is CUDA-core work, ~1.0 us is the wgmma / barrier chain.
- r4b (out/r4b.txt): NVRTC 13: C7511 for r3 (MF), for XG_ABL 1 (118 regs) and for XG_CP 2 (f16 dump + rolled fold,
  125 regs); not for CP 1 (123). nsys: base 12.54 / 12.54 | CP 2 13.42 / 13.38 (TB1 11.65 vs 10.72: the rolled
  smem fold publish is ~0.9 us slower than MF's register fold) | CP 1 13.34 / 13.38. t_xg full CP 2: 0 failures.
  => C7511 is ubiquitous and NOT the cost; the V^T transpose's 0.46 is real and inherent to FP8 PV wgmma (B must be
  K-major; the register-A alternative O^T = V^T P^T trades it for a P smem round trip + CUDA-core row sums).
  Remaining realistic cut: skip the alpha / bias recompute when the row max does not move (~20 of ~140 softmax
  instructions per pair step, ~0.07 us) -> below the 0.3 us bar. STOOD DOWN (no deliver_r4); deliver_r3 stays.

## Coordinator follow-up: FP16 PV wgmma (MN-major V, f16 P as register A) - evaluated, NOT built (probe says it loses)
Idea: convert V e4m3 -> f16 in its natural [token][d] layout (FP16 wgmma accepts an MN-major B), feed P as f16x2
straight from the softmax (RS mode), PV = 4 x k16 per 64 tokens: removes the V^T transpose (ablation -0.46) and the
P byte packing (-0.27). Accounting per pair step per warp: removed V^T 24 + pack 16 + clamp 32; added V conversion
~44 (2 LDS.128 + 16 cvt.f16x2.e4m3x2 + 4 STS.128 per thread per tile, ON the same QK -> PV chain), 32 ex2.approx.f16x2
for real f16 P (the 1024 + code trick yields e4m3 codes, not f16), PV wgmmas 4 -> 8 (+ a ones n8 wgmma per k-step for
the row sums, or 32 HADD2), V buffers 4 x 16 KB (fits only by reusing the dead 16 KB Q area: 227 KB limit).
r4c (out/r4c.txt, TIMING-BOUND TB1 builds = tile phase without the combine; XG_ABL bits 16 / 32):
  base TB1 10.59 / 10.72 / 10.66 | ideal (no V^T, no pack) 10.03 (-0.63) | **f16 P via MUFU ex2.f16x2 instead of
  clamp + pack: 10.82 (+0.16)** | conversion workload instead of V^T: 14.37 (+3.7, NOT representative: the probe's LDS
  mapping had 16-way bank conflicts) | full FP16 mix 14.56 / 14.69.
=> The f16-P half alone is net +0.16 (MUFU costs more than the clamp + PRMT packing it replaces), so the best case is
   -0.46 + 0.16 = -0.30 BEFORE the conversion (>= the transpose's instruction count on the same chain) and the 2-4x PV
   wgmmas. No realistic path to >= 0.3 us; not built, no deliver_r4. (Porting note for verify-full / run8w: same
   conclusion applies - the FP8 V^T transpose is cheaper than an e4m3 -> f16 V conversion + MUFU-exp P.)

## Coordinator follow-up: priority first tile (fresh-eyes #2) - DEAD (r5a, out/r5a.txt)
XG_PFT = gap (ns): tile 0's copies issued alone, a globaltimer spin of XG_PFT ns, then tiles 1 .. T0-1 (DEFER 4 kept).
smoke 0 failures. nsys (2 copies): base (= r3 code) 12.53 / 12.45 | r3 12.45 / 12.53 | PFT 150 12.48 / 12.42 | PFT
300 12.48 / 12.48 | PFT 600 12.70 / 12.70. Trace (PFT 300): step 0 at 3.10 (base traces 2.94-2.98: tile 0 does NOT
land earlier) while the first wave is issued later (2.14 vs ~1.9): tile 0's latency is the loaded DRAM round trip,
not queue position. Default XG_PFT 0; deliver_r3 stays.

## Standing rule (user): stack every correct + faster change -> deliver_r5 (FROZEN)
Scan of every dropped v1f variant vs r3's config: faster and cheap = PFT 150 (r5a: -0.05 / -0.03) and the alpha/bias
skip (estimated, now built as XG_ASK); already in r3 = GSM, D32, PIN, MBP 0, BFI; at or above r3 (excluded) = CLE (r2e /
r2g), GATE 4 / 16, PNS 0, MF 2, EG, PR, RK, DEARLY, LS, PFT 300 / 600.
- XG_ASK: warp vote `any(tile max * sc > mr)`; if no row max moves, alpha = 1 and the persistent bias is reused (skips
  ceil / fmax / ex2 / cvt per row half). Exact: mr is always a multiple of 1/8, so the skip never changes a result.
- deliver_r5 (dev/xg/deliver_r5/): xg_mod.py md5 78485972 = working module with XG_ASK 1, XG_PFT 150 defaults, unique
  Python globals _XG5_* / _PREV_KFN_XG5 (stacks on k51, which holds r3 as _XG3_); k51xg5.py md5 9256d8f0 = ref/k51.py
  (coordinator k51) + r5; k34xg5.py md5 8361c597; final.sh.
- r5fin (out/r5fin.txt): t_xg quick + FULL 0 failures, tools/check.py dev/xg/deliver_r5/k51xg5.py **0 failures**.
  nsys v1f: r3 12.32 / 12.38 / 12.38 | **r5 12.30 / 12.22 / 12.22 (-0.11)** | ASK only 12.35 / 12.32 | PFT only 12.35 /
  12.30 | neither (r5 module, both off) 12.40 / 12.32. Per piece vs neither: ASK -0.03, PFT -0.04, both -0.11.
