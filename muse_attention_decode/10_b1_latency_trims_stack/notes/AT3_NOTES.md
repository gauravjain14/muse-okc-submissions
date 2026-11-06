JUDGE-CHECK: cdl/AT3/stack_fr2.py (stack_fr1 + XG_DORD: deferred K/V wave issued par-0 tiles first (6, 8, 7, 9); 893,583 B; 16/16 tests, nearby 56/56, 7-call smoke; same-run A/B vs stack_fr1, 9 inst x 2 copies: v1f 15.01 -> 14.96 (-0.05 us, marginal: only worth it bundled with a later change))
JUDGE-CHECK: cdl/AT3/stack_fr1.py (cand4 + AT2 XE_FR from cdl/AT2/k8x_fr1.py: head = k8x_fr1 up to _XG5_SRC, tail = cand4 XG5 section; 893,179 B; 16/16 tests, nearby 56/56, 7-call smoke v1f + d1s; same-run A/B 2 copies each vs k80: d1s 7.353 -> 7.226 (-0.13), v1f 15.43 -> 15.055 (-0.375))
JUDGE-CHECK: cdl/AT3/cand4.py (k80 + v1f/xg_v8f-only changes: NC2 ring-polled combine with per-warp smem slots + mbarrier + warp-0 epilogue, ALB=2, SRL seq-reload, XG_GCONST compile-time G; XB1/d1f byte-identical to k80; 890,737 B; 16/16 tests, nearby 56/56, 7-call smoke both seeds+; same-run A/B 2 copies each: v1f 15.55 -> 15.09 us (-0.46, copies -0.475/-0.370 vs k80 copies 0/+0.071), d1f unchanged)
# AT3: muse_attention_decode B=1 full-context shapes (d1f = xb_d1f, v1f = xg_v8f), base k80.py

## STATUS (top) -- 2026-11-03 ~22:00 PT
- Scope: v1f (xg_v8f) only; xb_d1f belongs to AT4 (my d1f measurements below are for AT4).
- Submitted (coordinator): cand4.py (v1f -0.46 us). JUDGE-CHECKs: stack_fr1.py (cand4 + AT2 XE_FR d1s; d1s -0.13,
  v1f -0.375), stack_fr2.py (stack_fr1 + XG_DORD, v1f a further -0.05: marginal).
- What the v1f gain is made of (slim A/Bs): NC2 combine (ring-polled stale words, per-warp smem slots, mbarrier,
  warp-0 epilogue) -0.1..-0.2 | XG_GCONST (compile-time G, no runtime division before the pt loads) ~-0.15..-0.35 |
  SRL (avoid ptxas hoisting the uniform min(seq_lens) above the burst) + ALB=2 ~-0.05 each | DORD -0.05.
- C7511 (serialized wgmma) is NOT the lever: unserialized builds save ~330 cycles per pair step but the SM clock drops
  from ~1.88 to ~1.63 GHz (on-die power limiter), so the step takes the same ns (step anatomy table below).
- The loop is power / latency limited; ALU-instruction ablations are the only things that moved it (-0.17..-0.19 us per
  removed clamp / P-packing / V^T block), but no correct way to drop those instructions was found.
- End chain (s13, ns): publish 830-860 | -> gate 830 | gate -> bulk back 740 | poll 32 | final 450. Publish time does
  not scale with its work (MF6 halves it: same ns; no stores: same; warm second pass: same); bulk phase is not
  volume bound (ML-load ablation neutral); gate polling more often hurts (GRING +0.2).
- Remaining ideas (not done): N=128 QK wgmma for the regular pairs (needs a K / V split smem layout with tiles 3<->4,
  7<->8 swapped into adjacent slots; -2 serialized wgmma per pair step, but may be eaten by the clock limiter);
  understanding the ~1000-cycle receiver phase of the publish (latency source unknown; ncu sees it as cheap).

## Setup
- Copies of AT2's tools (bench, ab, nearby, trace_bg, reference, checks, nvk, mb_*). ab.py also passes per-candidate
  defs as XB_DEFS (XB1 module) besides K2_DEFS (XG5 module).
- slim80.py = k80 header + XB1 + XG5 sections only (mkslim.py): fast compile for d1f / v1f timing. Tests need the full file.
- at3tr.py = slim80 + extra XG trace stamps in the publish (23 after B1, 24 pre B2, 25 after B2, 26 stores issued).

## Iteration log
### 1. Is cold instruction fetch a factor? (exp_warm.py, box 0, median of 40, event us around the real replay)
| shape | base (flush->real) | warm (flush->tiny-ctx same kernel->real) | thrash i-cache first | thrash + warm |
|---|---|---|---|---|
| d1f s7/s8 | 14.00 / 14.03 | 13.89 / 13.95 | 14.18 / 14.16 | 13.84 / 13.92 |
| v1f s7/s8 | 15.58 / 15.71 | 15.55 / 15.58 | 15.62 / 15.79 | 15.52 / 15.65 |
=> code in L2 / i-caches is worth only ~0.1 us (d1f) / ~0.07 us (v1f). Not the lever.

### 2. Timing-bound ablations (ab.py, 2 seeds x 2 captures, raw us)
- d1f: XB1_HDAB=2 (no tile math) -0.92 +- 0.17; HDAB=1 (no lane-partial exchange) -0.15.
- v1f (vs MFREE=0,KF=0 control, which itself is +0.72 vs k80): XG_TB=1 (no combine) -1.91; XG_TB=2 (no combine, no
  tile math) -4.21 => tile math ~2.3 us, combine ~1.9 us in that build. XG_GATE=0 (wait for all 132) +0.43.
- v1f ping-pong token ring (XG_PP, all on MFREE=0): PPR1 +0.41, PPR2 +1.02, PPR3 +2.37, PPR4 +3.50, PPN2/PPR2 +0.13.
  XG_ABL=1 (no V^T) -0.29.
- v1f knobs on k80: XG_PFT 300 0.00, PFT 600 +0.25, XG_STAG 300 +0.08, STAG 700 +0.43, STAG 700 + BD +6.3,
  XG_DEFER 6 +1.37. => k80's XG5 is at a local optimum for these knobs.

### 3. Half-row gather (mb_g64.py: CTA = one KV head, 64-B halves of the K/V rows, same bytes), box 1, median of 3
empty 4.94 | full G132 11.30 | half G132 12.16 | quarter 13.09 (G128 alike). => 64-B random gathers cost +0.86 us:
splitting CTAs by KV head (to halve v1f's combine volume) is a dead end.

### Trace (trace_bg.py, k80 slim, box 1, ns from the first CTA start, per-CTA max over warps: med / max)
- d1f: burst issued 2928 / 3616 | loop done 6848 / 7552 | lane partials 7248 / 7936 | epilogue done 9392 / 9424.
- v1f: burst issued 2048 / 2272 | Q loaded 2816 | s0 data 3168 / 3712 | deferred issued 4480 | s1 data 4800 |
  s2 data 6400 / 6832 | loop done 7744 / 8128 | published 8672 / 9072 | gate passed 9568 / 9760 |
  bulk loads back 10208 / 10384 | poll done 10272 / 10448 | accumulated 10416 / 10592 | bar.red 10480 / 10656 |
  final sum 10720 / 10880 | stored 10976 / 11136 | end 11040 / 11200.

## Restart session (17:00-) -- tools added
- cc.py: compile-only NVRTC check (no GPU, `--nolock`): regs, spills, C7511 (wgmma serialization), SASS + count of
  GMMAs followed by a full wait. sass.py: SASS of both modules from a candidate. smoke.py: one call + check, both seeds.
- percta.py: per-CTA systematic lateness from a trace (split-half correlation). xbchain.py: d1f end-chain clock64 deltas.
- trace_bg.py: TSEED env selects the input seed.

### Fine v1f trace of k80 (at3tr.py, box 0, ns from first CTA start, per-CTA max over warps; med / max)
loop done 7856 / 8192 | after B1 7872 | par WGs exchange written (pre B2) 8272 / 8576 | after B2 8288 |
!par merge + 9 STG issued 8912 / 9264 | published 8944 / 9264 | gate passed 9792 / 9968 | bulk back 10464 / 10624 |
poll done 10512 / 10688 | accumulated 10656 | bar.red 10720 / 10880 | final sum 10960 / 11104 | stored 11216 / 11360 |
end 11296 / 11456. => publish 1.07 us, gate+bulk+poll 1.25 us after the last publish, final 0.78 us (1 warp, serial).

### ncu PC sampling of k80 v1f (SourceCounters, ~10K samples): share of samples per phase
prologue 10.7 % (imc 303 + no_inst 183 of 1087: cold code at the start) | burst 13 % | Q/new token 8.4 % |
step 0 8.7 % | pair loop 20.8 % | masked pair 4 % | post-loop B1 3.6 % | merge+store 1 % | gate 10.8 % |
bulk poll 8.5 % | final 5.6 % (barrier: 15 warps wait for warp 0's serial epilogue).
Pair step = 330 SASS instr / warp: HMNMX2 45, FMUL 42 (36 = ALB rescale), PRMT 34, HFMA2 32, IMAD/VIADD/LOP3/SHF/
LEA/ISETP/R2UR ~86 (address + descriptor), WARPGROUP 16, QGMMA 8.

### wgmma serialization (C7511) in xg_v8f
k80's xg_v8f: 128 regs, EVERY wgmma serialized (ARRIVE / QGMMA / DEPBAR 0 per instruction). Triggered by the
MF=1 publish (arithmetic on the accumulators); MF=3/MF=4 (f32 exchange) or XG_US compile unserialized.
But unserialized builds are SLOWER: MF=3 +0.73, MF=4 (my symmetric f32 exchange) +0.13, MF=4 + QS +0.51, US +1.1..1.9.
Explicit serialization of MF=4 (XG_SER 1/2/3: commit + wait 0 after each wgmma): +0.24 / +0.36 / +0.11. => dead end.

### A/B results (ab.py, v1f, 2 seeds x 3 captures, 40 reps; k80 control copy each run: -0.05..+0.07)
| variant | d vs k80 (us) |
|---|---|
| XG_QS=1 (FP8 Q from smem: A-desc wgmma, -8 regs; still serialized) | -0.04 |
| XG_GATE=132 (no gate) | -0.03 |
| XG_PR=1 (parallel final) | +0.16 |
| XG_GATE=132 + PR | +0.23 |
| XG_PNS=0 | -0.02 |
| XG_ALB=2 (o rescale inside the ASK raise branch: -36 FMUL / step when no max moved) | -0.04 / -0.06, -0.11, +0.005 |
| NC2 (ring-poll K=2 + per-warp slots + mbarrier + warp-0 epilogue) | +0.51 (main loop 0.3-0.6 us slower: SASS of the WHOLE kernel changed) |
| NC2 with smem f32 atomics | +2.7 (red.shared.add.f32 = ATOMS.CAST.SPIN CAS loop on sm_90!) |

d1f knobs (same method): XB1_PARS (parity at kernel start) +0.11, PARL +0.09, NC=8 +0.14, NC=2 +0.18 (all within the
~0.15 module bias; nothing).

### d1f per-CTA lateness (percta.py, 20 replays): loop-done spread median->max ~0.65 us, split-half correlation 0.81 for
a fixed input, but the late CTAs differ between seeds 7 and 8 => data-dependent (which rows a CTA gathers), not SM
position. Static rebalancing cannot fix it.
### d1f end chain (xbchain.py, final CTA, clock64): lane partial -> reds issued 916 cyc | reds -> atom.inc back 1186 |
-> epilogue start 82 | shard loads 633 | sum/norm 327 | gate/store 164.

### KEY: code-shape sensitivity explained (ncu long_sb + seqchk.py) -- 2026-11-03 ~18:00
Why tail edits slowed v1f by 0.3-0.5 us: prologue trace (mk_tr.py stamps 27 "tile0 issued", 28 "PFT done"):
tile0 issued med k80 1312 | n0 1344 | s6 1712 ns. ncu: s6 start->bar1 has long_sb 497 samples (k80 ~0) at
`R2UR UR20, R2` = the seq_lens LDG result: ptxas hoisted the UNIFORM `n = min(seq, p.seq_g)` of the (cold)
speculation-failure path above the first X8SYNC, so the whole CTA waits for the seq_lens DRAM load before the burst.
k80 itself computes that min right after the barrier (VIMNMX), also before tile 0 (same arrival time as pt, so ~free).
Fix (mk_srl.py, "SRL"): the failure path re-reads seq_lens with an asm-volatile load => the early register is only
used by the `seq == n` check after the burst. seqchk.py (static: first consumer of the seq_lens load vs first
BAR.SYNC) now runs on every build. d1f k80 has the same pattern (min at instr 140, before BAR.SYNC 195).
Other A/B: n0 (CC combine as __noinline__ function) +0.17 (main-kernel SASS still changes: interprocedural regalloc).
Step layouts (v1f): MFREE=0/KF=0 +0.54 | +LS +0.93 | +LS+DEFER=2 +1.23 | DEFER=2 +0.42 => final-single-step idea dead.
d1f per-tile arrival (tr_s2.py, lane tiles it=0..4): 3.46 / 4.54 / 5.25 / 5.92 / 6.56 med (max last 7.33), loop done
7.04 / 7.73: data arrives in order ~0.65 us apart per lane; last-tile compute ~0.45 us; DRAM spread med->max 0.77 us.
d1f PARS (9 inst, 2 copies each): PARS=1 +0.141 / -0.006, PARS=0 (= k80 XB1 code) +0.111 / +0.151, k80 copy -0.034
=> module-instance bias ~0.15 us even for identical code (likely ws placement / L2 slice of counters); PARS ~ -0.06.
Final-CTA chain with PARS (xbchain): lane partial -> reds 470 cyc (was 884).

### v1f progress (ab.py v1f, 2 seeds x 3 captures x 40 reps, k80 = mean of 2 k80 module copies)
| build | what | d vs k80 |
|---|---|---|
| k80srl | k80 + SRL | -0.05 |
| s7 | NC2 (ring-poll K=2, per-warp smem slots, mbarrier, warp-0 epilogue) + ALB=2 + SRL | -0.24 (2 copies -0.107/-0.167..-0.160) |
| s8 | s7 + XG_GCONST (compile-time G: no runtime division before the pt loads) (+TLBP knob) | -0.37 (TLBP=0) |
| s8 GCONST=0, TLBP=8 | | +0.0 (TLBP=8 alone: noise / codegen) |
| s9 | s8 + epilogue trims (sigmoid before the wait, LDS.128 L sums, tree sums) | -0.40 (2 copies -0.386 / -0.370) |
Trace s7 vs k80 (separate processes): final phase poll done -> end 0.56 us (k80 0.74); warp-0 epilogue
(mbarrier -> end) 0.38 us. Forced slow path (cand1_fs.py: even CTAs not "fast") still exact (rel-L2 2.381e-02).
TLBP (TLB warm-up prefetch of the K/V 2 MB pages): no measurable gain on v1f or d1f; default off.
cand1.py = mk_full.py k80.py (s1 QS knob off, PARS on, SER knob off, ALB=2, NC2 RING=2, SRL, TLBP off, GCONST,
s9 trims), 891,414 B. cand1np.py = same with XB1_PARS=0.

### 18:50-19:30: coordinator: v1f only (AT4 owns d1f); cand4 submitted scored; build stack with AT2 XE_FR
- stack_fr1.py = AT2/k8x_fr1.py head (XE_FR d1s change) + cand4 XG5 tail (k8x_fr1 tail == k80 tail, cand4 head ==
  k80 head, checked). Validation running.
- For AT4 (d1f): cand1np (k80 XB1 + SRL) vs cand3 (+ XB1_PARS=2: parity cp.async'ed to smem with tile 0) vs k80,
  9 inst x 2 copies: k80 14.002 / 13.866, cand1np 14.024 / 13.911, cand3 13.980 / 13.919: all noise. PARS=1
  (register load at start) made ptxas move the seq_lens load from instr 39 to 147 (seqchk.py) and lost ~0.2.
- Per-phase trace (xgdelta.py, within-kernel, 2 alternating runs each) s9 (serialized wgmma) vs s9 + MF=4
  (UNserialized): step 0 + deferred issue 1344-1376 vs 1472 ns | pair step 1472 vs 1536-1568 | last pair 1344 vs
  1344 | loop done -> published 832 vs 960 | poll done -> end 448 vs 480. => pipelined wgmma does not shorten the
  steps: the pair step (330 SASS / warp, 4 warps / SMSP = 1320 issue cycles; 32 wgmma / SM ~1100 tensor cycles) is
  issue + tensor bound, not wgmma-latency bound. C7511 is not the lever.
- s9 phases (ns, per-CTA medians): start->tile0 issued 992 | tile0->s0 data 1920 | step0+def 1344 | step1 1472 |
  last step 1344 | publish 832 | ->gate 832 | gate->bulk back 736 | poll 32 | poll done->end 448 | start->end 10368.
- Gate under NC2 (s9): GATE 8 (default) -0.445 | 132 -0.380 | 32 -0.384 | 16 -0.409 | 132 + RING 1 -0.290 => keep 8.
- Next: XG_PFM (mk_pfm.py, s10): MFREE "pairs first, single last" + masked-single instance; with DEFER=2 the
  deferred (last-arriving) tile is processed alone => tail 1 tile instead of a pair (~-0.6 us if the data order holds).

### 19:30-20:30 v1f tile-math / tail experiments (slim A/B vs s9/s13; per-phase traces with xgdelta.py)
| variant | result |
|---|---|
| XG_PFM (pairs first, single last; masked-single instance) + DEFER 2 | +0.33 vs s9 (last step 832 vs 1376 ns but head +0.86: 8-tile burst, step 0 = pair) |
| XG_PFM, DEFER 4 | +1.3 |
| DSPLIT (deferred pair (6,7) for par 0, (8,9) for par 1; extra step instances) | +1.4 (GMMA 60, 6000 SASS lines) |
| XG_DORD (deferred issue order 6, 8, 7, 9: par-0 tiles first) | -0.06 vs s9 (2 copies each) => kept (s13) |
| XG_DPAR (par-0 tiles 6, 8 in the burst, only 7, 9 deferred) | +0.37 vs s13 (tile0 -> s0 data +0.29) |
| XG_UNI (ballot-uniform WG index for descriptors) | pair loop 284 -> 281 SASS: nothing |
| ablations on s9 (timing only): ABL=1 no V^T -0.17, ABL=4 no clamp -0.19, ABL=8 no P packing -0.18 | |
Key: with DPAR the par-1 WGs ran their last pair with par 0 already done, yet it still took 1216 ns (vs 1344 with all
4 WGs): the pair step is LATENCY-bound inside the WG (dependent chain), not SM-issue-bound. Clock64 step-phase
stamps (mk_stp.py / stpan.py) running to find the chain.

### Step anatomy (mk_stp.py clock64 stamps in the step, stpan.py; per warp medians, cycles) -- WHY C7511 is not the lever
| phase | s13 last pair (serialized wgmma) | s13 middle pair | s13 + MF4 (UNserialized) | s16 (VTI) |
|---|---|---|---|---|
| data ready -> QK issued (4 QK wgmma) | 454 | 467 | 205 | 508 |
| -> V^T built (4 LDSM.T + 16 PRMT + 4 STSM) | 468 | 580 | 533 | 75 (moved) |
| -> QK done | 64 | 83 | 82 | 53 |
| -> softmax / P packed | 959 | 847 | 954 | 1233 |
| -> WG barrier | 99 | 123 | 109 | 102 |
| -> PV issued (4 PV wgmma) | 425 | 530 | 164 | 430 |
| -> PV done | 54 | 47 | 145 | 47 |
| total | 2523 | 2677 | 2192 | 2448 |
The unserialized build saves ~330 cycles per pair step in CYCLES, yet its steps take the same NANOSECONDS (xgdelta: last
step 1344 ns both): 2192 cyc / 1344 ns = 1.63 GHz vs 2523 / 1344 = 1.88 GHz. Pipelined wgmma raises tensor power and
the on-die limiter drops the SM clock to ~1.63 GHz: the gain is cancelled. Serialized wgmma keeps ~1.88 GHz.
=> the loop is power / clock limited; what helps is less work (ALU instruction ablations: -0.17..-0.19 us per ~16-32
instructions removed per pair step), not more tensor overlap.
V^T interleaved into the P batches (XG_VTI, s16): -75 cycles per step, A/B neutral (-0.525/-0.527 vs s13 -0.658/-0.477).
XG_EXS (par-0 early sender over named barriers; s15 DPAR=0): -0.04 vs s13 (noise).

### 20:30-21:30 end-chain experiments (A/B vs s13 = s9 + DORD, 2 copies each)
| variant | d vs s13 |
|---|---|
| XG_GRING (gate counter: 3 polls in flight, 100 ns stagger) | +0.20 (GRD 200: +0.45): more polling of the counter line hurts |
| XG_PNS 100 / 200 / 400 (gate sleep) | +0.07 / +0.06 / +0.15 | XG_RD 256 (ring stagger) +0.21 |
| XG_MF=6 (symmetric f16 exchange, keeps C7511 serialization) | -0.02 (neutral) |
Publish anatomy (mk_pub.py clock64, s13 MF=1, cycles per warp): B1 wait 7 | after B1 until B2 (par-1 sender convert +
STS 492; par-0 just LDS cnt + callk 584!) | B2 wait 38-66 | par-0 receiver merge + 9 STG 1146 | B3 32. Yet xgdelta
"loop done -> published" = 864 ns for BOTH MF1 and MF6 (MF6 halves the per-thread work): the phase is not bound by the
exchange arithmetic. (Ablation of the ML-word bulk loads running to size the bulk phase's volume sensitivity.)
- XG_DEARLY=1 +0.25 | XG_BD=1 broken (+24 us) | XG_DEFER 3 +0.24, 5 +0.63 (vs s13) => DEFER 4 after step-0 PV stays.
- XG_SV=1 (weak st.global for the slot words): neutral. ML-word bulk loads removed (timing-only XG_ABLB): neutral =>
  the bulk phase is not volume / request bound.
- Pair-step SASS (s13, common path ~225 dynamic instr): ptxas already interleaves the max tree and the V^T LDSM/STSM with
  the serialized QK wgmmas, and the HFMA2/clamp/PRMT of P with the serialized PV wgmmas; fence.proxy.async compiles to
  MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S after the V^T stores.
