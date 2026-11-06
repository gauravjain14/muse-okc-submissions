# WP-E2: the end-of-call phase of the main kernel (bs 2-512)

Worker WP-E2, 2026-10-27. Base dev/moe_v730.py. Files: dev/moe_v750.py (v730 + MOE_E2T trace stamps; default build
SASS-identical to v730), dev/moe_v751.py (v750 + the end-phase switches below, all off by default; default build
SASS-identical to v730), dev/wpe2_probe.py (decomposition from the stamps), dev/wpe2_run*.sh (Modal jobs). Logs:
scratchpad wpe2/run*.log.

## 1. Decomposition of v730 (runs 1-3; trace build, judge protocol: graph replay, L2 flush read, 6-10 jm seeds)

Stamps (globaltimer): per CTA (start, last item's last stage, combine entry/exit, every warp's exit), per tile (atomic
max over its down items: last stage done, staged, epilogue warp woke, epilogue stores issued, signaler woke, red.release
issued, signal count), per combine unit/token (claim, every slot's poll done + the counter value seen, reduced, stored).
Medians over seeds, us:

| | bs8 | bs16 | bs32 | bs128 | bs256 |
|---|---|---|---|---|---|
| end phase (last down stage -> last warp exit) | 3.1-3.5 | 3.4-4.3 | 3.7-4.7 | 3.0-3.7 | 5.5-6.4 |
| last tile: staging | 0.13 | 0.16 | 0.19 | 0.16 | 0.35 |
| + epilogue warp wake | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 |
| + epilogue stores | 0.50 | 0.54-0.96 | 0.22 | 0.22 | 0.26 |
| + signaler wake | 0.22 | 0.26 | 0.19 | 0.19 | 0.22 |
| + red.release (MEMBAR.GPU) | 0.38 | 0.40 | 0.42 | 0.42 | 0.43 |
| critical unit: its tile's red -> its last poll done | 0.40 | 0.38 | 0.43 | 0.5-5 | 17-21 (claimed late) |
| critical unit: last poll -> output stored | 1.30 | 1.73 | 1.76-2.3 | 2.2-2.5 | 3.0-3.1 |
| stored -> CTA exit (= kernel end) | 0.06 | 0.08 | 0.43 | 0.46 | 0.37 |

- The kernel end is always the CTA that stores the last output; its exit follows the store by 0.06 us (bs8/16, static
  units) or ~0.4 us (bs32+: the next claim's atomic round trip before the exit).
- bs8/16 (fast SSQ units, pinned to CTA u): the unit's CTA is free well before the data in almost every seed; after the
  polls, ~1.3-1.7 us go to the CTA barrier, then each pair's shuffle-reduction chain in its own warp-uniform branch,
  serialized (np 2-5).
  Not cold code (no-flush runs: same), not L1 invalidation by the acquire polls (relaxed polls: same).
- bs32/128 (claimed tokens): the CG = 2 pair groups serialize a barrier plus an L2 round trip each (np 3-5).
- bs256: throughput-bound: 105-154 of 256 tokens start their combine after the last down stage; the token finishing last
  was claimed 16-21 us after its data was ready (index-order claims, ~4 us per token per CTA: claim round trip, 2
  barriers, 2-3 serialized pair groups).
- Measurement note: a globaltimer read right after `bar.sync` (BAR.SYNC.DEFER_BLOCKING) issues before the barrier
  completes; per-thread stamps after the wait itself are needed (the per-slot poll stamps here).
- v730's combine keeps `plist[np]` / `wl[np]` in local memory (dynamic register-array index): the LDL/STL and 64-byte
  stack frame of every main kernel come from there.

## 2. Mechanisms (dev/moe_v751.py)
- MOE_E2_SSQW (bs 2-16 fast units): per-slot arrays (no local memory), warp 0 polls one lane per slot (relaxed spin,
  one acquire), one barrier, all 8 slots' sum-of-squares shuffles interleaved.
- MOE_E2_CMB (one-k-warp configs, bs 17-512): per token, warp 0 polls (one lane per slot), one barrier, every pair's row
  in one load round trip (BF16 words; a split pair's fp32 KRED row rounded to BF16 as every full-k pair's y is), one
  barrier for all 8 slots' sums of squares; the next claim's atomic is issued right after the polls and read after the
  stores (one more barrier), so the CTA exits right after its last store; nw1 loaded once per CTA.
- MOE_E2_ORD: claim index i -> the i-th token ordered by its last pair index (pairs are in tile order, about the order
  tiles finish): deterministic bitmap + popcount ranking in shared memory at the combine entry.
- MOE_E2_ESIG: the second epilogue warp of a down item posts the signal itself when the CTA's end marker is already
  queued (its last item), skipping the signaler hand-off; MOE_E2_ENW1 / MOE_E2_EUNR: the SEPI epilogue's post-norm
  weight load hoisted, its store loop unrolled.
- Dead end measured: every warp polling the token's tiles itself (8 warps x np lanes per CTA, relaxed spin) floods the
  counters' L2 lines: the signals themselves arrive 7-13 us late at bs32-256 (run 4). One polling lane per slot per CTA.

## 3. Measurements (bench2, jm seeds, same-job controls; ratios = control-copy-1 time / variant time)

| run | variant | bs8 | bs16 | bs32 | bs128 | bs256 |
|---|---|---|---|---|---|---|
| 5 | v730b control | 1.0014 | 0.9998 | 0.9943 | 0.9967 | 0.9971 |
| 5 | SSQW v1 (x3 builds sharing the kernel) | 1.0105..1.0134 | 1.0056..1.0069 | | | |
| 5/6 | CMB (unrolled 8-slot token body) | | | 0.947..0.967 | 0.945..0.953 | 0.974..0.982 |
| 8 | CWARM / SWARM dry warm-up passes | 0.979 | 0.982 | 0.927..0.936 | 0.953..0.960 | 0.966..0.969 |
| 9 (80 reps) | SSQW / +ESIG / +ECL on v730's stack loop | 1.0087 / 1.0084 / 1.0117 | 1.0077 / 1.0057 / 1.0068 | 0.997 / 0.998 / 1.001 | 0.998 / 0.998 / 0.991 | 0.995 / 0.994 / 0.999 |
| 10 | v730b control | 1.0005 | 0.9997 | 0.9968 | 0.9977 | 0.9966 |
| 10 | SHFL+SSQW x2 | 1.0068 / 1.0103 | 1.0072 / 1.0085 | 1.0032 / 1.0038 | 1.0037 / 1.0032 | 1.0063 / 1.0066 |
| 10 | SHFL+SSQW+ECL | 1.0069 | 1.0092 | 1.0071 | 1.0047 | 1.0116 |
| 11 | v749b control | 0.9994 | 0.9954 | 0.9980 | 0.9989 | 0.9967 |
| 11 | **v752 x2 (vs v749)** | **1.0132 / 1.0127** | **1.0060 / 1.0070** | **1.0069 / 1.0072** | **1.0020 / 1.0015** | **1.0095 / 1.0125** |

bs1 (moe1t, byte-identical in every variant): 0.991-1.007, noise. End phase after SHFL+SSQW+ECL (traces, run 10 / 6):
bs8 ~2.8, bs16 ~3.1, bs32 3.1, bs128 2.9, bs256 3.95 us (v730: 3.1-3.5, 3.4-4.3, 3.7-4.7, 3.0-3.7, 5.5-6.4).

Lessons:
- The v730 combine's cost was mostly its local-memory pair list (planner's ncu: 72-79% L1 misses on those lines; the
  polls' acquire invalidates L1): register-resident lists fix it.
- An unrolled 8-slot body (~1100 straight-line instructions) loses to cold instruction fetch (one token per CTA).
- Any extra combine work (dry passes, poll floods) on the SM whose epilogue warps still carry the last item stretches that
  item's signal chain: keep the combine light until its tiles are seen.
- ESIG (epilogue warps post the last signal themselves), ORD/ORDL (readiness-ordered claims), CG4: no gain.

## 4. Result: dev/moe_v752.py (BAKED, base v749)
v749 + MOE_E2_SSQW + MOE_E2_SHFL + MOE_E2_ECL on by default; MOE_NO_WPE2 = v749's kernels (SASS-verified). Only the
main kernels' combine changes; moe1t / moe1 byte-identical; no main kernel keeps a stack frame for the combine (the
remaining LDL/STL in the bs33-512 kernels are v749's lean-producer per-item reloads); producer stage loops spill-free; no
C7510/C7511/C7513; (8,8,1) hot loops identical in order to v749. hang1 16 shapes ok, check_mt 17 sizes (1-512) 0 failures
(max rel-L2 3.303e-3). Gains vs v749 (run 11, vs the control mean): bs8 +1.3%, bs16 +0.9%, bs32 +0.8%, bs128 +0.2%,
bs256 +1.3%; projected geomean ~+0.75%. Recommendation: integrate.

## 5. Task 2: k-split last round at bs8/16 (dev/moe_v800.py, baked dev/moe_v801.py; base v764)
Prize (ncu): SMs idle 19% (bs8) / 15% (bs16) in the wave-quantization tail; WP-S's TAILB bound +4.9 / +3.7%. WP-K's
KREDS lost x0.93 because split tokens made v730's slow combine body live ((4,1,1) 5712 -> 7592 instructions).
- Mechanism: MOE_KSXK2-style k pieces (existing producer / consumer / epilogue support; fp32 partial slices, no zeroing
  needed) with MOE_E2_KSPL (only the last round's ceil(r / 32) tiles, and only when ceil(KSX r / grid) < KSX: halves
  help only when r <= grid / 2; queue model on 100 jm routings: KSX 2 splits 41-45% of calls, saves 0.21-0.23 rounds;
  KSX 4 splits 71-74%, saves 0.31-0.33), MOE_E2_SPLF (split tokens stay on the SSQW fast units: a compact lane-held
  loop over split slots loads every slice of the full row, rounds to BF16, one CTA barrier; the slow token body stays
  dead at compile time), v657's early-start epilogue tsplit fix, MOE_E2_KSXNT1 / NT2 (KSX per config).
- Per routing (run 13, 48 seeds x 9 cold replays): bs8 K4 +2.8% where it splits (35 seeds), -0.7..-1% where it does
  not (the build's code cost), +1.7% overall; bs16 K2 +2% where it splits (27 seeds), -0.65% elsewhere, ~+0.8% overall
  vs a control that read -0.6% (bench2: ~0).
- Code cost: no single role (producer / consumer stage count / epilogue) carries it; any KSX-aware code re-schedules the
  (4,*,1) consumer loop (same 1059 instructions); readings swing +-1% between jobs (codegen lottery).
- bench2 (jm 100 reps, v764 x2): bs8 K4 +1.9 (run 12), +1.6 (14), +1.2/+1.3 (15), +1.0/+1.1% (16, baked v801);
  bs16 K2 0..+0.3%, K4 0 -> bs16 left unsplit.
- dev/moe_v801.py (BAKED): v764 + MOE_E2_KSXNT1 4 + KSPL + SPLF on by default; MOE_NO_WPE2S = v764 (SASS-verified);
  only the (4,1,1) kernel differs (5880 instructions, no spills, no C75xx, producer loop 244 instructions spill-free,
  re-scheduled 0.81; consumer loop same size, re-scheduled 0.41). check_mt 21 sizes 0 failures (max rel-L2 3.32e-3).
  Gain bs8 +1.0..1.9% (four jobs), geomean ~+0.2%. Recommendation: integrate.
- What would unlock bs16 (+2% on half its routings): a split build whose consumer stage loop stays byte-identical
  (the runtime stage count is what moves it); a second loop instance with a constant 4-stage count would be cold code on
  the end-of-call critical path, so it needs warming off that path.

## 6. Task 3: a bs16 split with the (4,2,1) consumer loop byte-identical (dev/moe_v820.py) -- not found
Local search (compile + loop comparison vs v801, ~2.5 s per build; scratchpad wpe2/lc.sh):
- Neutral (consumer loop byte-identical): the split rule, the early-start tsplit fix, the epilogue warps' split stores,
  the producer's split decode (changes the producer loop only: 0.811, spill-free), and a split marker folded into the
  consumer's existing end-marker branch (MOE_E2_SPL3P).
- Each of these alone re-schedules the consumer loop (order similarity): the consumers' runtime stage count 0.407; the
  live slow combine 0.325; the SPLF combine 0.398; a second constant-4-stage loop instance inside the item loop
  (MOE_E2_SPL2) 0.386; a tail loop for the pieces after the item loop (MOE_E2_SPL3) 0.424 even when never reached; the
  tail as a __noinline__ call (MOE_E2_SPL3N) 0.431; a one-line exit condition in the item loop 0.533.
  -> any register-hungry code anywhere in the (4,2,1) kernel moves its loop; a loop-identical split does not exist in
  this structure.
- Measured anyway (run 17, bs16, v801 x2, control 0.9982): SPLF x0.9994 / 1.0026, slow combine x0.9603, tail loop
  x0.9566 (cold register-hungry code on the end-of-call path). All correct (check_mt M 9-16). bs16 split: closed.
- Follow-up (run 18, base v814): bs16 K4 (MOE_E2_KSXNT2=4 with v801's KSPL + SPLF; only (4,2,1) changes) x0.9947 /
  0.9938 vs control 0.9963 (~-0.3%); bs8 placebo identical kernel 0.9950 / 0.9993 vs 0.9980. KSX 3 impossible (8 stages
  per item). bs16 k-splitting closed (K2 runs 12/15/17 ~0, K4 runs 12/13/18 ~0 / -0.3%).

## 7. Task 4: the bs8 end-of-call chain on v837 (dev/moe_v890-v894.py)
Planner bound: timing-only MOE_NO_COMBINE bs8 +4.0% (~2.5 us).
- Decomposition (trace build dev/moe_v890.py = v837 + MOE_E2T; run 19, 12 jm seeds): bs8 end phase 3.71 us = last
  tile's chain 1.17 (staging 0.13, epilogue wake 0.10, stores 0.24, signaler wake 0.19, red.release 0.35) + red -> last
  poll 0.32 + last poll -> stored 2.35 + exit 0.08. The critical unit has split slots (v801's K4 tokens) in 9 / 12
  seeds and its split part takes 2.88 us: per split slot 4 slices x the full row (64 KB per unit).
- MOE_E2_KRS (dev/moe_v892.py): the k pieces red.add into one fp32 row per split pair (persistent ykz[2][256][4096],
  own parity word flipped by the last CTA to start; each call records its split rows and its idle-at-start epilogue
  warps zero the previous (4,1,1) call's rows in the other half: no in-call ordering needed; check_mt + 150 interleaved
  calls over configs 2-64 clean). The row offset rides in tiles[].w, so the (4,1,1) consumer and producer loops stay
  v837's modulo register names (every other formulation of the read re-scheduled the loop: 0.464). Data /4, but the
  split part only 2.88 -> 2.34 us: fine stamps (run 21) show a chain of dependent round trips after the poll (non-split
  part 0.90, split lane list + top-k load 0.38, first slot 0.45, second slot 0.54, barrier 0.13), unchanged without the
  L2 flush (not cold code).
- MOE_E2_SPE (dev/moe_v893.py, with KRS): lane list, top-k weights and post-norm rows before the polls; the first two
  split slots' rows in the same round trip as the non-split data -> split part 1.70 us, end phase (per-seed median)
  3.73 -> 3.31 us.
- bench2 (jm 100 reps, v837 x2): KRS bs8 +0.25..0.4% (run 20), +0.7% (run 22); KRS+SPE +0.4 / +0.5% (run 22),
  +0.1 / +0.35 / +1.05% (run 23, three copies, control 0.9966) -> ~+0.5% bs8 (~0.08% geomean), borderline against the
  control spread; bs16 placebo (identical kernel) within noise.
- Baked as dev/moe_v894.py (KRS + SPE on; MOE_NO_WPE2K = v837's kernels; only moe_4_1_1 differs).
- What remains of the ~3.3 us is synchronization: the last item's signal (staging, epilogue wake, stores, signaler wake,
  release barrier ~1.2), the poll (~0.35), one CTA barrier + one load round trip (~1.0) and the split slots' extra
  reduction (~0.6). A last-arriver combine would replace the signaler hop + poll (~0.6) with an acq_rel atomic of the
  same cost on the epilogue warps (plus their per-item MEMBAR under streaming), so it was not built. A combine-code warm-up
  is moot here (no-flush runs are identical) and would need the consumer warps (the 72-register epilogue warps cannot
  run the consumers' instructions).

## 8. Task 5: end chains of bs16 / bs32 / bs128 / bs256 on v894 (dev/moe_v910-v913.py)
Planner bounds (timing-only MOE_NO_COMBINE on v837): bs16 +1.1%, bs32 +1.3%, bs128 +1.3%, bs256 +1.1%.
- Decomposition (dev/moe_v910.py = v894 + MOE_E2T stamps in the SHFL token combine; run 25, 10 jm seeds):
  - bs16 (4,2,1) SSQW units: end 3.14 us = last tile's chain 1.92 (staging 0.13, epilogue wake 0.10, epilogue stores
    0.94, signaler wake 0.26, red.release 0.38) + red -> poll seen 0.45 + seen -> reduced 1.06 + exit 0.06; no splits.
  - bs32 (8,4,1): end 3.89 us; the critical token waits on its last tile (red -> seen 0.40), then two CG=2 pair rounds
    finish 1.23 / 2.37 us after the last poll (stored 2.51).
  - bs128 / bs256: end 5.30 / 5.23 us, throughput-bound: the critical token is claimed 12-14 us after its tile's red;
    ~2.4 us per token (two rounds).
- Loop identity: every non-trivial edit of the inline (8,*,1) token combine re-schedules the consumer stage loops
  (order similarity 0.33-0.74 vs v894: CG4, TG, ORD2, laundered counter base, inputs re-read from shared memory; 64
  combinations of 8 neutral codegen knobs land in 2 alternative schedules). Cause seen in the PTX: NVVM re-derives the
  pre-loop index arithmetic (or vs add, sign-extension hoisting) from the combine's uses, and ptxas' allocation is
  global. Constant-only edits keep the loops (dev/moe_v911.py).
- MOE_E2_TOKF (dev/moe_v912.py): the token combine in the __noinline__ e2_tok (call + EXIT, nothing live after it).
  Every callee variant then has byte-identical consumer and producer loops (checked for FCG4 / FTG / FLD / FTW / FPA /
  FPF and combinations; FORD moves the producer loop and was dropped). The TOKF loops themselves are a new schedule vs
  v894 (similarity 0.43-0.66), spill-free; only the five (8,*,1) kernels change.
- Run 26 (jm 100, v894 x2; bs16 placebo identical kernels 0.999-1.003): tokf x0.9946 / 0.9931 (bs32), 1.0033 /
  1.0034 (bs128), 0.9962 / 0.9972 (bs256) vs v894b 0.9976 / 0.9988 / 0.9957. FCG4 (4 pairs per round) loses 0.7-1.1%,
  FTG (pairs by tile, per-round polls) ~0 / -0.7%. Traces: a round's loads already issue back to back; a 4-pair round
  takes 1.62 us vs 0.86 for 2 pairs: the token combine is bound by one SM's L2 bandwidth (~0.35 us per extra pair,
  8 KB per BF16 pair, 16 KB per fp32 split row) plus ~0.5 us latency; fewer rounds do not help.
- Run 27 (jm 100, v894 x2): inside the frozen combine FTW (own top-k weight load, no shuffle wait before the polls)
  and FPA (one acquire first, spin only if needed) give nothing over the tokf control (bs32 0.9933-0.9943 vs 0.9935,
  bs128 1.0023-1.0032 vs 1.0031, bs256 0.9946-0.9955 vs 0.9968). Closing the relaxed spin with fence.acq_rel.gpu
  instead of an acquire load is slower: red -> seen 0.35 -> 0.64 us (bs32), and as MOE_E2_FPF4 in the bs2-16 fast
  units (loops byte-identical) 0.45 -> 0.80 us at bs16, bench bs16 x0.9946. TOKF's own codegen draw reproduced: bs32
  -0.6%, bs128 +0.3%, bs256 -0.3% (two jobs) -> not bakeable as is.
- bs8 / bs16 epilogue: the last tile's SSQ epilogue (2 warps, one 64-column group per thread) runs nn tokens as nn
  dependent iterations (~60 ns each: 16 at bs16 -> 0.94 us). Hoisting its post-norm weight load changes nothing (the
  compiler already does); `#pragma unroll 2` (MOE_E2_EU2) / 4 for the 16-token tiles (MOE_E2_EUM, dev/moe_v914.py)
  keep the (4,1,1) / (4,2,1) consumer and producer loops byte-identical to v894 (EU4 on (4,1,1) moves its producer
  loop). No shared memory left for a post-norm weight copy in (4,2,1) (229376 of 232448 bytes).
- Timing bounds on the TOKF platform (run 30, invalid outputs, vs tokf): half / a quarter of the combine's bytes
  bs32 +0.34 / +0.34%, bs128 +0.31 / +0.50%, bs256 -0.33 / +0.18%; no combine +1.43 / +1.56 / +1.18%. The bytes are
  worth <= ~0.5%, so spreading a token over 2-4 SMs (SSQ-style units for (8,*,1), which would also need KRED under SSQ
  and the SPLF split slots on KRED rows) is bounded there; the rest is fixed per-token latency (signal chain, poll,
  barriers, two L2 round trips). No-flush traces (run 31): the critical token's combine is the same (2.10 / 2.14 us
  at bs32: no cold-code cost on it); earlier tokens run 2x faster without the flush, which shortens the bs128 tail.
- bs16 epilogue (run 31): the consumers' combine polls (acquire: L1 invalidated) run while the epilogue warps store the
  CTA's last tile, so that tile's plain nw1 load went to L2. Through the read-only path (MOE_E2_EWG): epilogue stores
  0.99 -> 0.50 us (no-load bound 0.38), last tile's chain 2.02 -> 1.84 us; bs8 unchanged (0.27). EUM / EU2 (unrolled
  store loop): no change (0.98 -> 1.01 us).
- Result: dev/moe_v916.py (BAKED; run 33: hang1 16 shapes both modes, check_mt 25 sizes 1-512 0 failures, max rel-L2
  3.321e-3, 280 interleaved calls clean) = v915 with MOE_E2_EWG2 ((4,2,1) only) on by default, escape MOE_NO_WPE2G = v894's
  kernels; only moe_4_2_1 differs, its stage loops byte-identical to v894's. bs16 vs own-job control means: +0.29 /
  +0.62% (run 31), -0.07 / +0.30 / +0.17% and +0.01 / +0.23 / +0.37% (run 32): mean +0.24% (~+0.04% geomean),
  borderline against the control spread (0.07-0.37%).
- Not bakeable / closed: TOKF (loops frozen, but its own schedule is -0.6% bs32 / +0.3% bs128 / -0.3% bs256), FCG4,
  FTG, FTW, FPA, FPF, FPF4, EU2 / EUM, EWG on (4,1,1). What is left at bs32-256 is the fixed latency of one token's
  combine (~2 us after the last poll: two rounds of L2 round trip + barrier) and, at bs128 / 256, the claim order of the
  throughput-bound tail (ORD-style claims move the producer loop).
- Follow-up (planner): TOKF for the (8,4,1) MW 4 kernel only on v944 (dev/moe_v917.py, MOE_E2_TOKF4, escape
  MOE_NO_WPE2T; every other kernel byte-identical to v944). Run 34 (jm 100, v944 x2): bs128 -0.4 / -0.5% in two passes
  (3 copies each, controls 0.9987 / 0.9982); placebos bs32 / bs256 within noise. v912's +0.3% on v894 was a codegen draw
  that does not survive v944's MOE_WGSN kernel. Rejected, not baked (checks were clean: check_mt 15 sizes, 180
  interleaved calls).
