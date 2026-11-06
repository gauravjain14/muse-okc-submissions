# WP-S findings (bs8 / bs16 / bs32), worker S

## S1 (2026-10-26 PT) baseline timelines of v200 (dev/wps_run1.sh, Modal H200, trace_cta MOE_TRACE+MOE_TRN)

Per-CTA item timelines (us from the first kernel event), dumps in dev/remote_out/wps_tc_<bs>_<seed>.json:

| bs | seed | experts | items | items/CTA | first item start | first item dur | steady item dur (G1/G2 med) | gap med (p90) | CTA end min / med / max |
|---|---|---|---|---|---|---|---|---|---|
| 8  | 70186 | 7  | 672  | {5:120, 6:12}  | 5.9 | 9.9  | 8.77 / 8.99 | 0.74 (1.28) | 53.2 / 54.8 / 62.5 |
| 8  | 70610 | 9  | 864  | {6:60, 7:72}   | 4.4 | 9.8  | 8.74 / 8.74 | 0.61 (1.09) | 62.2 / 70.6 / 71.9 |
| 8  | 72887 | 11 | 1056 | {8:132}        | 4.6 | 10.2 | 8.74 / 8.70 | 0.32 (0.80) | 78.5 / 80.4 / 82.0 |
| 16 | 71062 | 10 | 960  | {7:96, 8:36}   | 4.7 | 11.6 | 8.80 / 8.67 | 0.32 (1.06) | 71.5 / 72.9 / 81.4 |
| 16 | 70877 | 12 | 1152 | {8:36, 9:96}   | 4.9 | 10.8 | 8.96 / 9.02 | 0.35 (0.96) | 84.0 / 90.0 / 93.5 |
| 16 | 72054 | 14 | 1344 | {10:108, 11:24}| 4.8 | 10.0 | 8.83 / 8.80 | 0.32 (0.96) | 98.2 / 101.4 / 107.4 |
| 32 | 72520 | 17 | 1248 | 7..11          | 8.1 | 17.7 | 17.15 / 4.80 | 0.29 (0.83) | 122.4 / 123.7 / 127.8 |

- Wave quantization is the visible tail: e.g. bs8 seed 70186 (7 experts) runs 12 sixth items alone for ~9 us
  (end 54.8 med -> 62.5 max, ~11% of the call); bs16 71062 8.5 us (36 CTAs with an 8th item).
- Items are ~8.8 us steady (8 stages of ~1.07-1.1 us = the bare stream's per-stage time), the first item 10-11.6 us
  (ring fill of all 132 CTAs at once), plus a 0.3-0.7 us gap at every item boundary.
- MOE_TIMING (step1_timing) cycle counts do NOT add up to the wall time (8 items x (8 x 1520 + 822) cycles = 60 us vs
  75 us of trace): the timer reads in the stage loop are moved around by the compiler; use traces/bench for wall time.

## S2 producer at item boundaries (dev/trace_bnd.py on v350, which adds trace event 27 after the x rows)

| bs | seed | gap (ready - prev end) | slot acquired before prev end | weights issued before | rows issued before | ready after weights issue |
|---|---|---|---|---|---|---|
| 8  | 70610 | 0.90 | 0.51 | 0.19 | -0.13 | 1.15 |
| 16 | 70877 | 0.64 | 0.61 | 0.32 | -0.03 | 1.09 |

The producer gets the next item's first slot only ~0.5 us before the previous item ends (in a 4-slot ring it frees
~3 stages earlier) and needs ~0.3 us per stage for 4 weight copies, another ~0.3 us for 1-2 x rows (trace events
themselves slow the starved producer, so these are upper bounds).

## S3 tail-split design (v351, MOE_WPS_SPL) -- paired early/late k pieces

Queue: [G1 items][A pieces of the first h down items (stages 0..a-1)][remaining whole down items][B pieces (stages
a..7)]. A leaves fp32 partial rows (ypart) and posts flag ctr[6144 + item]; B's producer polls it (set ~a round
earlier, so no wait) and B's epilogue adds it, writing final y + 64-row sums of squares (fast combine unchanged).
(h, a) per tile count from a list-scheduling simulation (dev/wps_sim.py: 1.07 us stages, 0.5 us epilogue, 0.35 us
gap, +1.5 us first item, 5% noise, A->B dependency); splits kept only where the simulation gains >= 1.5%.
Simulated gain over the judge-matched tile-count distribution: bs8 ~+2.9%, bs16 ~+1.7%.

## S4 producer copy count (dev/wps_run2b.sh, dev/wps_run5.sh; jm seeds, 100 reps, 2 controls)

| variant | bs8 | bs16 | bs32 | note |
|---|---|---|---|---|
| NOX (no gate/up x-row copies; timing only) | +1.13% | +1.30% | -0.1% | valid upper bound for a 1-copy x path (x is L2-resident either way) |
| W1T (one WB+SB copy from the kw0 source; timing only) | +4.97% | +4.82% | +3.53% | ARTIFACT: the contiguous 35 KB read overlaps the neighbour item's rows -> L2 hits, fewer DRAM bytes |
| v352 MOE_WPS_WBLK (real 1-copy blocks) | +0.04 / -0.12% | +0.59 / +0.84% | n/a | controls 0.5% (bs8) / 0.09% (bs16) apart |

So the weight copy count is worth <= ~0.7% (bs16), consistent with the planner's microbenchmark.

WBLK block format (for anyone reading it): wb13[e][rb 0..63][st 0..7], wb2[e][rb 0..31][st 0..7], each WB+SB bytes =
[kw0 weights: row groups rb*8..rb*8+7 of k chunk 2*st, 8 x 2048 B][kw1 weights: same rows, chunk 2*st+1][kw0 scales 8 x SRG]
[kw1 scales 8 x SRG]; SRG 192 with 6-bit scales (s13c6/s2c6), else 256 -- byte-for-byte the (4,*,1) ring slot's first WB+SB
bytes. Built in prepare() from w13p/w2p and the scale tensors the kernel uses.

## S5 per-item time split (dev/wps_run4.sh: trace full vs MOE_NO_COMPUTE vs MOE_NO_LOAD, v200)

| bs seed | full item / gap | data-only item / gap | compute-only item / gap |
|---|---|---|---|
| 8 70610  | 8.70 / 0.64 | 7.36 / 0.99 | 7.78 / 0.29 |
| 16 70877 | 8.86 / 0.35 | 7.44 / 0.90 | 8.00 / 0.29 |
| 16 71062 | 8.93 / 0.32 | 7.39 / 0.86 | 7.97 / 0.32 |

Per item cycle: compute ~8.1-8.3 us, data ~8.3-8.4 us, full ~9.2-9.3 us -> item-level C/M ~0.97, right at the planner's
knee (P1): the coupling costs ~11%. Compute-side fixed cost per item (epilogue ~0.5 + transition ~0.3 us) is ~10% of C.

## S6 light probe (v356 MOE_WPS_PROBE + dev/wps_probe.py: thread 0 clock64 per item, producer lane 0 per stage; no timers in the stage loop)

Per item (us; 3 bs8 seeds / 2 bs16 seeds, medians over CTAs and 5 reps):

| build | loop (8 stages) | epilogue | gap | producer issue / stage | producer slot wait / stage |
|---|---|---|---|---|---|
| v200 full | 7.71-8.27 | 0.51-0.56 | 0.18-0.20 | 1130-1170 cyc | 255-320 cyc |
| MOE_NO_LOAD (compute only) | 7.11-7.24 | 0.47-0.53 | 0.18-0.20 | - | - |
| MOE_NO_COMPUTE (data only) | 6.44-6.59 | 0.39-0.44 | 0.98-1.23 | 760-860 | 570-650 |
| WBLK | 7.69-8.10 | 0.50-0.55 | 0.19-0.20 | 1040-1060 | 310-390 |

The producer spends ~0.65 us per stage inside its copy issue (TMA backpressure: the memory system accepts the
stage's requests at about the per-SM bandwidth share), not waiting for slots; copy count barely changes it.
Compute per item 7.8 us vs data 8.0 us (C/M 0.97); the consumer-side fixed cost per item (epilogue 0.5 + gap 0.18)
is ~8.5% of C.

## S7 rejected (details in candidates.jsonl)
- v351/v353 MOE_WPS_SPL paired tail split: -1.6..-2.6% (code path alone -1.5%; splits ~0: the A pieces advance the
  queue so the last tile's whole down items start before its G1 rows are signalled -> 12 us items).
- v354 MOE_WPS_XG gathered gate/up rows: -3.3..-4.5% (with WBLK -1.2..-1.5%).
- v355 MOE_WPS_ETID epilogue work off SMSP 0/1: neutral.
- v200 MOE_WSC_NT=4 (bs32/128 one-copy stages): neutral.

## S8 two-pool scheduler hooks (for WP-P's CUDA-core stage), design note
- Routing (warp 0 lane 0, after the tile table): classify tiles A (T <= 2, CUDA-core) / B (tensor); give pool A its
  own item range: queue A = [A G1 items][A G2 items], queue B likewise, two head words (CB[0] and a free word, e.g.
  CB[7]); pool sizes from a cost model per call (items_A * cA vs items_B * cB, cA/cB ~0.89 from P5), CTA b < NA in
  pool A. First wave static per pool (item = b or b - NA), then claims from the pool's head.
- Down items stay in their tile's pool, so the G1->G2 dependency polls are unchanged; each pool has its own tail.
- Lesson from S7: every runtime branch added to the consumer's per-item code (dispatch/epilogue) cost ~1.5% at
  bs8/16 even when unused (SASS: item loop +100 instructions, a BSSY in the 2-group stage loop). Pool A's stage
  must therefore be a separate compile-time path (template / constexpr on blockIdx-independent config, or a second
  kernel name) rather than a runtime switch inside the existing consumer loop.
- Wave quantization inside a small pool is worse than in one pool of 132; pool A needs the tail balance more.

## S9 SEPI (barrier-free item epilogue on warps 10-11) and the startup nw1 copy
- v357 (bs<=8 only): bs8 -1.0% x2 (jm 100 reps); with WBLK +0.4%. Probe: warp 0's item cycle -3.5% (epilogue 0.51 ->
  0.15 us, gap 0.175 -> 0.145 us) but the NO_LOAD stage loop is +1.4% (the epilogue warps issue on SMSPs 2/3) and the
  kernel did not end earlier: without the per-item barrier the warps drift and the slowest (SMSP 2/3) sets the end.
- v358 (also bs9-16; one buffer in the gap, NW1GAP copy dropped): bs16 +1.0/+1.4%, bs8 -0.7/-0.9%. bs16's first item
  ready 5.1 -> 4.3 us: the consumers' cold 16 KB nw1 copy (NW1S/NW1GAP) sits on the startup path.
- v359 MOE_WPS_NWL moves that copy to warps 10-11 (mbarrier before the consumers' first read): pending (wps_run13).
- v359 NWL (wait inside the G2 epilogue) loses 1.0-1.4% at bs8/16 in bench2 although its probe shows first item ready
  -0.8 us and kernel end -0.5 us; Modal SASS: a BSSY appears in the bs16 2-group stage loop, epilogue +0.07 us. Probes
  are not a substitute for the paired bench (the planner's consumer-side nw1 fix lost ~1% the same way).
- NWL+SEPI: bs16 +1.2%, +WBLK +1.8% (one copy each), bs8 ~0.
- Probe of SEPI's epilogue warps (run14): they wake 0.75-1.35 us after the consumers' staging (suspend-hinted
  try_wait, MOE_SUSPEND = 1 ms) and work 1.1-1.4 us per item -> each item's signal lands ~2-2.7 us after its last
  stage (v200: ~0.5 us). The same wake-up latency applies to the signaler and producer waits: tested in run16
  (MOE_SUSPEND=300).

## S10 run16 (jm 100 reps, one job, controls 0.1% apart)
| variant | bs8 | bs16 |
|---|---|---|
| v360 NWL (wait before the consumer loop) | -1.04% | -0.70% |
| v360 W2=1 (2 slots per wait, release between) | -3.60% | (n/a) |
| v360 W2=2 (2 slots per wait, release after both) | -8.90% | (n/a) |
| v200 MOE_SUSPEND=300 | -0.39% | +0.15% |
| v360 SEPI + WBLK | +0.11% | +1.88% |
=> SEPI+WBLK is the bs16 win (+1.76..+1.88% in two jobs); NWL, W2 and shorter suspend hints are out. W2's probe: the
consumer idles on the second slot while the first is ready (W2=2 also holds a slot a stage longer).

## S11 kept: v363 (v601 BF16-only + SEPI + WBLK + ESUS/SSUS=200), wps_run19 (jm 100 reps, v601 x2, v363 x2)
bs16 +1.67/+1.88%, bs8 -0.25/-0.15%, bs32 -0.1%, bs1/128/256 within the control spread; geomean 2.3874/2.3852 vs
2.3837/2.3800. hang1 13 sizes ok; check_mt M 1..40, 48..248 step 8, 255..257: 0 failures. Earlier on v200 (run17):
bs16 +2.40/+2.52%, bs8 +0.39/+0.51%. Extra memory: WBLK blocks 881 MB (prepare()).

## S12 (2026-10-27 00:45 PT) after v365
- v365 = v475 (BF16 + S6O/S6X) + SEPI + WBLK + ESUS/SSUS 200: kept (run22: bs16 +3.1/+3.2% vs v601, +1.9-2.0% over v475;
  bs8 +0.8% (+0.4% over v475), others = v475). FP16 define-text pitfall in comments reworded in v363/v365/v366.
- Rejected: ESUS/SSUS 50 / 1000 ns (noise), v366 MOE_WPS_SPL2 late-placed tail split on SEPI (bs8 -0.6%, bs16 -0.5%;
  traces: kernel end +2-3 us, item gaps x2). Three tail-split designs have now lost in measurement while the queue
  model (dev/wps_sim2.py) predicted +3..8%: the model is missing a real cost (item-transition gaps grow with more,
  shorter items; the B pieces' SEPI epilogue latency ~2 us lands on the end).

## S13 end phase: consumers run the CTA's last epilogue (v367 MOE_WPS_LASTE) -- rejected
run25 (jm 100 reps): bs16 -1.7%, bs8 -0.4..-0.7%, bs32/128 flat; check_mt M 1..257 clean. Traces: the kernel end moved
2-3 us LATER and the G1 stage loops +0.1 us/item: making the staged epilogue path live again next to SEPI's costs
codegen, and the last item's epilogue on the consumers (0.5 us of barriers) does not shorten the end. Not tried for the
(8,4,1) AEPI end (the precondition, a win here, failed).

## S14 tail bound, side-by-side halves, SEPI on (8,4,1)
- v368 TAILB (timing only: the last R = items mod 132 down items run half their stages): bs8 +4.9/+5.2%, bs16 +3.7%;
  the code path alone (TAILB_OFF) -0.7% / -0.4%. The prize is real (part of it is removed work).
- v369 SPL3 (real: the last round's down items as two side-by-side k halves, A's fp32 rows + flag, B's SEPI epilogue
  warps wait and add): bs8 +0.35/+0.64%, bs16 -0.5%; gated to short last rounds (RMAX 36/48/60): noise. Traces: the
  tail shrinks 2.5-3.5 us on R <= 36 seeds but the median CTA ends ~1-1.5 us later. Rejected.
- v370 SEPI8 (barrier-free, slot-free hand-off of the (8,4,1) config's 1-group epilogues to the AEPI warps): bs32 -0.7%,
  bs128 -2.1%. Rejected.
