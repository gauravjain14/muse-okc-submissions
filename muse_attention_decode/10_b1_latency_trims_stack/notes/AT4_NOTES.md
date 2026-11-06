JUDGE-CHECK: cdl/AT4/cand_b2.py (AT3 stack_fr1 + XB1_BAT=2 on d1f only: the HD lane loop does 2 landed tiles per pass (4 QK wgmmas / 1 wait / 2 P^T buffers / 1 fence / 1 bar.red / 4 PV wgmmas), separate code for 1- and 2-tile passes; same-run A/B 2 copies each vs stack_fr1: d1f 13.780/13.735 -> 13.541/13.431 (-0.27 us), d1s control +-0.06; 16/16 tests, nearby 86/86 x 2 seed sets (incl. 30 extra B=1 full decode cases), 7-call smoke; 898,655 B, NVRTC source)
# AT4: muse_attention_decode d1f (xb_d1f in k80.py)

## STATUS (top) -- 2026-11-03 night
- JUDGE-CHECK cand_b2.py (-0.27 us d1f, XB1_BAT=2 batched HD lane loop). 14 further structural / end-chain variants
  measured and rejected (sections 1-5). Best next bets: per-pass instruction diet of the BAT=2 loop (code-shape
  sensitive: re-check SASS + 2 copies per arm), BF16 reds (halve the red ops), smem parity (PS, -0.07 noise).

## Setup
- Copies of AT3's tools (ab, bench, cc, checks, nearby, reference, sass/ctl, smoke, trace_bg, xbchain, mb_g1) + k80.py.
- mkslim4.py: header + XB1 section only (no XG5): fast compile for d1f. s4_k80.py = mkslim4(k80.py).
- mk_tr4.py: extra globaltimer stamps (15 after bar0, 1 pt arrived (dep), 14 tile 0 issued, 8+it lane tile data,
  3 published, 6 atom back / poll done); trace4.py = trace_bg.py with these names.
- ctl.py: decodes sm_90 SASS control bits (stall / yield / write+read scoreboard / wait mask) from a cubin.

## Baseline trace k80 d1f (tr4_k80, box 2, 20 replays, ns from first CTA start; per-CTA max over warps: med / p90 / max)
start 144/224/256 | after bar0 608 | pt arrived 672/736/768 | tile0 issued 1120/1248/1344 | burst issued 2832/3278/4032 |
it0 data 3424/3968/4640 | it1 4512 | it2 5184 | it3 5888 | it4 6512/6941/7552 | loop done 6976/7421/7984 |
lane partials (pair bar.red) 7408/7902/8496 | reds issued 7488/7998/8624 | atom back 8304/8702/9376 | epi done 9968.
- Final CTA (median over its warps): loop done 7632 -> pair bar 8192 -> reds 8400 -> atom back 9120 -> epi done 9952.
- Per pair: min bar.red - max loop done of the pair's 8 warps = 288 ns (median).
- Per-CTA loop done correlates with tile-0 issue time (0.73) and burst issued (0.77), not with pt arrival (0.20).
- SASS (ctl.py): the end-chain parity LDG (SB5) is consumed only after the pair BAR.RED; DEPBAR SB0 is the
  cp.async wait_all. xb_d1f has no C7511 serialization (4 GMMA, the 2 QK GMMAs are waited by design), 71 regs, no spills.

## Iteration log
### 1. XB1_FIN v1 (mk_f1.py -> f1.py / s4_f1.py)
Design: every CTA publishes its 16 x 64 merged partial as BF16 + (m, L) per row with plain stores to per-(row,
source) slots whose words hold a sentinel (0xFFFFFFFF) until written; CTA r < 16 finalizes row r by polling its
row's 132 slots (512 threads: chunks of 16 B), accumulating as they land, resetting to the sentinel; fast path all
m = REF, exact path otherwise (never taken on seeds 7-10). No reds, atomics, counters or parity.
- Correct (seeds 7-10). ab: +3.08 us (!). Trace: last publish ~8.1-8.35 us, polls done 8.1-9.5, then the CTA
  barrier after the poll passes at ~12.2: some warps spend ~5000 cycles between "poll done" and the barrier
  (clock64): the global-store issue is blocked (the sentinel reset stores right after each successful poll load?).
- FIN v1 root causes: (a) lanes exiting the divergent poll loop early (lane 0 stamped "poll done" while its warp
  mates still polled; a __syncwarp after the loop cut 2.4 us); (b) each poll iteration costs ~1.17 us in the real
  kernel (2-3 iterations; mb_lat: 200-260 ns unloaded, mb_poll with writers ~380-750 ns): pick-up p50 800 ns,
  row complete - last publish med 672 / max 1312 (tr9). FIN v2 (4 poll warps x 9 chunks, 128-thread finalize,
  on top of BAT=2): +1.1 us vs b2; + L2 bulk prefetch of the slots at kernel start: +0.6 vs b2. DEAD (in this form).
- Prefetching k80's accumulator shards / counters into L2 at kernel start (cp.async.bulk.prefetch.L2): +0.22..+0.29.

### 2. The lane loop is compute-latency bound (register clock stamps, no stores in the loop: mk_tr12 / mk_tr13)
- Tiles 1-4 of a lane have ALWAYS landed when the lane reaches them (98.7 % of passes, mbarrier test before the
  wait); per tile ~1050 cycles: steady-state chain (it 2) prev PV committed -> mbar wait 177 | fence 21 | lane bar 40
  | QK 143 | V^T LDSM 96 | P^T + fence 207 | bar.red(need) 143 | PV issue 51. Tile 0: 1351 (lazy-max raise ~713).
- Per-tile landing times (trl_k80, HDAB=2, warp 0 polls all 12 mbarriers): tiles land in order: 0/1 3.17 us med
  (max 4.6-4.9), 2 3.30, 3 4.10, 4 4.67, 5 4.90, 6-9 5.50-5.60 (max 6.37). Out-of-order processing would not help.
- Note: "dep" stamps whose asm does not use the dependency register do NOT wait for it (no SASS consumer): mk_tr4's
  "pt arrived" and AT3's XTD_ clock stamps are not reliable. Fixed with a predicated timer read (XGD in mk_trc.py).
- ab: BAT=2 single code path (tile-1 work unconditional) +0.33; BAT=2 with separate 1/2-tile code -0.22..-0.27 (b2);
  BAT 3 / 4 with redux.min + smem agreement +0.17 / +0.15 (code size / agreement cost); VEARLY (retire PV first,
  LDSM under QK) +0.17 vs b2; MBT (test_wait before try_wait) noise.
### 3. b2 end chain (mk_trc.py, final warps, globaltimer ns from first CTA start, 10 replays)
loop exit 7072 | lane partial stored +256 (cp.async.wait_all: the other-parity tiles of this thread's copies) |
pair bar.red +384 (partner lane) | reds issued +128 | atom.inc.acq_rel back +832 | epilogue shard loads +416 | out +224.
### 4. Dead ends after b2 (ab.py d1f, 2 seeds x 3 captures, 2 copies per arm unless noted; d vs b2)
| idea | file | result |
|---|---|---|
| FIN v2 (lean 4-warp poll finalizer) / + L2 bulk prefetch of the slots | mk_f2.py, mk_pf.py | +1.1 / +0.6 |
| k80 chain + L2 bulk prefetch of shards / counters at start | mk_pf.py (s4_b2p) | +0.22..+0.29 |
| XB1_NC 8 / 16 (more accumulator shards; counters moved) | s4_b2nc | +0.13 / +0.40 |
| LR: per-lane reds + per-head counters (no pair merge), NC 4 / 8 | mk_lr.py | +0.55 / +0.48 |
| DYN: the 2 lanes of a head grab tiles from a shared smem cursor | mk_dyn.py | +0.68..+0.83 |
| FRS: first pass skips the P^T write before its certain max exchange | mk_b6.py | +0.22 (code-shape; 94 -> 88 regs) |
| PRI/PRID: burst tiles 0..1 (or 0..3) first, then a 200/400/800 ns pause | mk_pri.py | noise (PRID 800: tile-0 landing max 4.6 -> 3.7 us, median unchanged, last landing +0.25) |
| HB: lanes = tile residues mod 4, both heads per pass (+ parity at start) | mk_hb.py / mk_hb2.py | +0.52 / +0.34 (lanes become data-bound, 4-lane CTA merge waits for the slowest lane) |
Poll physics (FIN): in-kernel poll iterations ~0.8-1.2 us each although a single L2 probe right after costs ~165 ns
(311 cycles): polls of lines being written by other SMs are slow; mb_poll (idle GPU) row complete - last publish
~350 ns. The red / acq_rel-atom / load chain (3 round trips, ~1.6 us after the last pair merge) stays.
| PARS4: PW=2 parity read by ALL threads at kernel start | s4_b2s | +0.38..+0.51 (2112 same-line loads at start: L2 hot spot delays the pt / burst) |
| PS: parity cp.async'ed (4 B, thread 0) at start, tracked by tile 0's mbarrier, read from smem at the end | s4_ps | -0.07 (copies spread +-0.07: noise) |
| NP: no parity: per-warp counters wrap at G-1, the last arriver zeroes the shard words it read, per-row-pair exact lists | mk_np.py | +0.17..+0.26 (extra stores at the very end of the kernel) |
| SRL (coordinator item 1) on xb_d1f: retry path re-reads seq_lens (s4_srl) / + seq==n check as volatile asm after the burst (s4_srl2) | s4_srl, s4_srl2 | +0.16..+0.20 / +0.17 (seqchk: the seq consumer moves past bar0 but still precedes the first burst LDGSTS in SASS) |
Coordinator items checked for d1f: (2) XB1_GCONST (compile-time G, no division before the pt loads) is already on in
k80's xb_d1f; (4) every A/B above uses >= 2 copies per arm (3 when close); (6) xb_d1f has no C7511/C7520 (b2: GMMA 12,
only the QK groups are waited by design).
### 5. b2 lane-exit -> pair-bar clock breakdown (trk2_b2, register clocks, no trace stores; all warps, cycles)
loop exit -> wait_all 47 | -> lane partial (3+3 SHFL, ex2, STS) 415 | -> parity LDG issued 290 | -> pair bar 104 (p90 616).
Spare-warp landing tracer (trs2_b2, 17th warp polling all tile mbarriers) shows per-CTA lane exits differ by up to
0.5 us between the two lanes of a head; the later lane decides the pair bar. Landing stamps from the spare warp are
coarse (its poll loop is slow) and not trustworthy.

## Summary for the coordinator (2026-11-03 night)
- Banked: cand_b2.py = stack_fr1 + mk_b1.py (XB1_BAT=2). d1f -0.27 us same-run vs stack_fr1 (2 copies each),
  tests 16/16, nearby 86/86 x 2 seed sets. The patch touches only the _XB1_SRC section (mk_b1.py applies to any file
  whose XB1 section equals k80's, e.g. the decode-layer copy of xb_d1f).
- Physics: gather lands as in the floor kernel (mb_g1t: CTA all-landed 5.8-6.0 us vs ours 5.6-6.4); after the
  first tile each HD lane is compute-latency bound (~1000 cycles/pass); the end chain after the last pair merge is
  3 dependent L2 round trips (reds+release 128+832 ns, shard loads 416, epilogue 224) and every protocol tried to
  shorten it (polling finalizers, per-lane reds, no parity, prefetch, more shards) was slower in this kernel.
