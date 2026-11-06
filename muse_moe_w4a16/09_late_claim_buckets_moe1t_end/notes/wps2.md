# WP-S2: bs16 (4,2,1) 1-group items and bs32 (8,4,1) MW1 multi-group startup, base v1002 (2026-10-31)

Worker WP-S2. Namespace dev/moe_v1600-v1699.py, dev/wps2_*. Base dev/moe_v1002.py.

## 0. Setup
- Local SASS: NVRTC 13.0 (libnvrtc.so.13 from an earlier session's scratchpad) through dev/wpc_sass.py with
  cuda.bindings blocked (local cuda.bindings is NVRTC 12.1).
- jm seeds bs16: 43/100 routings have no expert with > 8 tokens (light: only 1-group tiles), 52 have one 9-16-token
  expert, 5 have two; none > 16. bs32: 9 have none > 8, mean 1.43 experts > 8, 0.79 > 16, 0.06 > 24.

## 1. bs16: is the (4,2,1) 1-group stage loop the same as (4,1,1)'s?
SASS (NVRTC 13, v1002): (4,1,1) 1-group loop 510 instructions (bottom-tested MOE_RUN_ITEM_BOT), (4,2,1) 1-group loop
512 (top-tested MOE_RUN_ITEM_TOP; MOE_BOT1 off). Same op mix (LOP3 171/172, HMUL2 128, IMAD 72, SHF 68, HMMA 32, LDS 19),
same LDS / SYNCS placement; the instruction order differs (ptxas schedule), not "identical".

## 2. Run 1 (dev/wps2_run1.sh; bench2 jm100 bs16 + bs8 placebo, v1002 x2; item + stage traces, 12 / 5 jm seeds)
dev/moe_v1600.py = v1002 + MOE_S2_B411 (Python: bs 9-16 on the (4,1,1) kernel) + MOE_S2_BOT1 (bottom-tested 1-group loop
in (4,2,1) only; the global MOE_BOT1 no longer compiles: DS DEF static_assert in the (8,*,1) kernels). Default = v1002 SASS.

| vs v1002 (median paired ratio) | bs16 all (100) | light reps, max expert <= 8 (43) | heavy reps (57) | bs8 placebo |
|---|---|---|---|---|
| v1002b (control) | 1.0007 | 1.0023 | 0.9967 | 0.9997 |
| B411 x2 ((4,1,1) kernel) | 0.9817 / 0.9804 | **1.0165 / 1.0132** | 0.9639 / 0.9592 | 0.9991 / 1.0037 |
| S2_BOT1 | 0.9988 | 0.9977 | 0.9989 | 1.0033 |

check_mt B411 M 9 12 16: 0 failures. So on v1002 the (4,1,1) kernel is still ~1.5% faster on light bs16 routings
and ~4% slower on heavy ones (old v198 numbers: +2.7 / -3.1%).

Where the light-rep difference is (item traces, same 6 light seeds, per CTA us; stage traces cycles):
- tail (latest CTA's last item end - own): (4,2,1) 1.7-6.3 (mean ~4.5) vs (4,1,1) 1.5-2.4: the (4,1,1) kernel's k-split
  last round (MOE_E2_KSXNT1 4, KSPL). The (4,2,1) kernel has none (WP-E2 measured KSXNT2 2 / 4 at ~0 / -0.3%).
- startup (first data): (4,2,1) 3.9-4.1 vs (4,1,1) 5.0-5.4 us. Cause: the (4,1,1) consumers' 16 KB MOE_NW1S
  post-norm-weight copy (cold global loads) before their first stage wait. With SEPI that copy is DEAD: the SEPI
  epilogue warps read P.nw1 from global and nothing reads the smem copy (only the non-SEPI consumer epilogue did).
  -> candidate MOE_S2_NONW1 for bs 2-8 (dev/moe_v1602.py).
- 1-group stage compute (consumer, middle stages): (4,2,1) 1440-1455 vs (4,1,1) 1380-1404 cycles (+3.5-5%, the TOP vs
  BOT loop schedule), but the stage period is memory-paced (median 1866-1875 vs 1840-1864, C/M ~0.77), so it is
  hidden; BOT1 in (4,2,1) measures 0.9988.
- So the "(4,2,1) 1-group items are slower" effect on v1002 = the missing tail split (+ a few % of hidden compute);
  the 1-group items themselves run at the ring rate in both kernels.

## 3. Run 2 (dev/wps2_run2.sh; bench2 jm100 bs8 / 16 / 32, v1002 x2)
dev/moe_v1601.py = v1600 + MOE_S2_SWZ ((4,2,1) SEPI staging buffer XOR-swizzled: the consumers' FP32 staging stores
were 4-way bank conflicted, SEPSP = RB = 128 floats with no padding in the 16 KB gap; only moe_4_2_1 changes, its
consumer loops re-scheduled, similarity ~0.3). dev/moe_v1602.py = v1601 + MOE_S2_NONW1 (no dead nw1 copy in the SEPI
(4,1,1) kernel; only moe_4_1_1 changes, every stage loop byte-identical). MOE_WGS_MW=1 (existing switch: WP-WG's WGSN
wgmma stage loop also in the bs17-64 kernels; their mma.sync loops are re-scheduled too).
hang1 15 sizes ok, check_mt (NONW1 M 2-5, 8; SWZ 9 12 15 16; WGS1 17 20 24 25 32 33 48 64): 0 failures.

| vs v1002 | bs8 | bs16 | bs32 |
|---|---|---|---|
| v1002b (control) | 0.9956 | 0.9954 | 0.9988 |
| NONW1 x2 | 0.9988 / 0.9973 | (placebo) 0.9966 / 0.9989 | (placebo) 0.9988 / 0.9986 |
| SWZ | (placebo) 0.9977 | **1.0008** | (placebo) 0.9994 |
| WGS_MW=1 | (placebo) 0.9958 | (placebo) 0.9982 | 0.9994 |

Placebo spread (identical kernels) ~0.995-0.999: copy a reads ~0.2-0.3% high. NONW1 trace (wp32_probe, 6 bs8 seeds):
first data 5.11 -> 4.09 us, trace-build kernel 60.36 -> 59.45 us, but bench2 +0.1 / -0.05% vs the control mean: the
startup DRAM fill hides the consumers' 1 us (same as WP-EDGE's cold first stage). Not converting (repeat in run 3).
WGS_MW=1 at bs32: +0.05% vs the placebo mean: rejected. SWZ +0.3% vs the bs16 placebo mean: repeat in run 3.

## 4. bs16 heavy reps: the 2-group gate/up stages are producer-bound (run 1 stage traces, seeds 71001 / 72054)
| (4,2,1), cycles mean/median | 1-group gate/up middle | 2-group gate/up middle | 2-group down middle |
|---|---|---|---|
| stage period | 1920-1945 / 1861-1866 | 2444-2710 / 2337-2686 | 2257-2315 / 2246-2247 |
| consumer compute | 1444-1449 | 1806-1811 | 1774-1777 |
| producer copies issued - slot acquired (piss) | 1107-1137 | 1902-2259 | 1142-1153 |
| producer slot acquired - slot freed (turn) | 625-940 | 2720-4072 | 721-732 |
The 2-group gate/up stage takes 1 WBLK copy + one 1 KB bulk copy per token row (9-16, from x) and the producer needs
~2000-2260 cycles to issue them: it reaches each freed slot 2700-4000 cycles late, and the stage runs ~30-50% over its
compute. The 2-group down stage (intermediate: ONE n x XP copy) is not producer-bound. Prize (if the gate/up stage ran at
compute + gap ~2100): ~64 items x 8 stages x ~500 cycles per heavy expert = ~1 us per heavy call (~1.2% of heavy reps,
~0.7% of bs16). Bounds in run 3 (v1603, timing only): XB1 (the rows as one contiguous copy, same bytes), XB8 (8 rows).

## 5. Run 3 (dev/wps2_run3.sh; bench2 jm100 bs16 + bs8, v1002 x2; ratios vs control copy v1002b)
dev/moe_v1603.py = v1602 + timing bounds MOE_S2_XB1 / MOE_S2_XB8 (only moe_4_2_1 changes, consumer loops
byte-identical to v1002).

| vs v1002b | bs16 all | light (43) | heavy (57) | bs8 |
|---|---|---|---|---|
| v1002 (copy a) | 1.0007 | 1.0000 | 1.0019 | 0.9995 |
| XB1 (2-group gate/up x rows as one copy) | 1.0000 | 0.9993 | 1.0006 | (placebo) 0.9982 |
| XB8 (only 8 of the rows) | 1.0005 | 1.0000 | 1.0010 | (placebo) 0.9966 |
| SWZ x2 | 1.0027 / 1.0014 | 1.0016 / 1.0008 | 1.0038 / 1.0016 | (placebo) 0.9960 / 0.9989 |
| NONW1 | (placebo) 0.9979 | | | 1.0005 |

Stage traces (71001 / 72054): both bounds bring the 2-group gate/up middle stage to 2291-2302 cycles (from 2752 / 2426),
the producer's issue to ~1100 (XB1) / ~1560 (XB8), and its slot lag (turn) from 4045 / 2617 to 607 / 562. Yet bench2
does not move on the heavy reps (+0.06 / +0.10%): the producer-bound 2-group stages are not on the wall-time path
(the first wave's DRAM share goes to the other CTAs, the queue rebalances; the trace build's polling producer also
exaggerates producer-boundness). Closed: no x-copy work for the (4,2,1) 2-group tiles.
NONW1: third reading +0.05% at bs8 -> rejected (no gain; the dead copy stays).
SWZ over runs 2+3 (3 copies): +0.31 / +0.24 / +0.10% vs the control means; kept as a weak candidate, repeat in run 4.

## 6. Run 4: bs32 wgmma first-wave CTAs, prize bound (dev/moe_v1604.py; bench2 jm100 bs32 + bs16, v1002 x2)
MOE_S2_FWXB (TIMING BOUND, with MOE_WP32_FSTW): wp32_fst issues the ~27 wgmma first-wave CTAs' first 4 stages complete
(weights + scales + the 4 gathered x chunks, read before the all-CTA gather is done: stale) at ~2.5 us, as the FST CTAs.
Consumer loops byte-identical to v1002 in all builds.

| vs v1002b | bs32 | bs16 (placebo for FSTW / FWXB) |
|---|---|---|
| v1002 (copy a) | 1.0019 | 1.0033 |
| FSTW (WP-32's: weights early, x after the gather) | **0.9881** | 0.9998 |
| FWXB x2 (everything early, stale x) | **0.9930 / 0.9927** | 1.0029 / 1.0005 |
| SWZ x2 | (placebo) 0.9994 / 1.0000 | 1.0015 / 1.0023 |

Trace (6 seeds, wp32_probe): the wgmma CTAs' first data p90 10.1 -> 6.9 us, but every CTA's cold startup path gets later:
consumers' setmaxnreg.inc (c15) 4.45 -> 5.42 us, producer loop entry (p44) 4.56 -> 6.50, first claim 6.70 -> 7.79;
trace kernel 106.3 -> 106.9 us. The extra 27 x 4 stages of early DRAM fill delay the cold instruction fetches of all
132 CTAs (as WP-32's FSTW / WQ1 and WP-RT found). Even the ideal (an x that is ready at 2.5 us for free) loses 0.7%,
so no real per-tile early gather can win: the bs32 first-wave direction is CLOSED.
SWZ after 4 runs (5 copies): +0.31, +0.24, +0.10, -0.02, +0.06% vs control means: noise. Rejected.

## 7. Run 5: bs16 k-split last round WITH the KRS buffer (dev/moe_v1605.py; bench2 jm100 bs16 + bs8, v1002 x2)
v1605 (Python): with MOE_E2_KSXNT2 = 2 | 4 the (4,2,1) kernel gets state["ykz"] as ypart, so its pieces red.add into one
fp32 row per split pair (MOE_E2_KRSV is SEPIC && KSX > KS2 && PARV, not NT-gated) and share the (4,1,1) kernel's parity /
record / zeroing protocol. (WP-E2's KSXNT2 runs predate KRS: fp32 slices and an L2-bound combine.) KSPL rule unchanged
(K4 splits 74 of 100 jm calls, K2 45). Correct: check_mt M 9 10 12 13 15 16 (K4, K2), dev/wps2_mix.py 120 interleaved
bs1-33 calls (K4) 0 failures, worst rel-L2 3.30e-3. moe_4_2_1 6224 / 6160 instructions, no spills; loops re-scheduled.

| per rep / control mean | bs16 all | light (43) | heavy (57) | bs8 placebo |
|---|---|---|---|---|
| v1002 / v1002b | 1.0012 / 0.9988 | 1.0009 / 0.9991 | 1.0013 / 0.9987 | 0.9996 / 1.0004 |
| K4 x2 | 0.9973 / 0.9998 | 0.9968 / 0.9988 | 0.9990 / 0.9999 | 1.0025 / 1.0001 |
| K2 | 1.0033 | 0.9992 | 1.0039 | 0.9986 |
| K4 + SWZ | 0.9970 | 0.9929 | 1.0002 | 0.9981 |
| SWZ | 1.0023 | 0.9996 | 1.0042 | 1.0027 |

Trace (12 seeds, K4 vs run 1's v1002): tail 4.80 -> 2.13 us (as the (4,1,1) kernel's 1.8), but startup 4.00 -> 4.57,
end 2.75 -> 3.00, item gaps +0.55, and the 1-group gate/up items 8.29 -> 8.99 us each (x 2.8 items): trace kernel
85.61 -> 85.82. The split build's codegen (consumer loop re-scheduled, more cold producer code on the startup path) eats
the tail gain, as WP-E2 found without KRS. Run 6 tries a few more builds (K2 x2, K4 / K2 + BOT1, K2 + 0.5-wave split).

## 8. Run 6: more bs16 split builds (v1605; bench2 jm100 bs16 + bs8; per rep vs the v1002 x2 control mean)
| | bs16 all | light | heavy | bs8 placebo |
|---|---|---|---|---|
| v1002 / v1002b | 1.0006 / 0.9994 | 1.0000 / 1.0000 | 1.0017 / 0.9983 | 1.0000 / 1.0000 |
| K2 (KSPL) x2 | 0.9988 / 1.0050 | 0.9989 / 1.0076 | 0.9986 / 1.0031 | 0.9979 / 1.0015 |
| K4 + S2_BOT1 | 0.9821 | 0.9846 | 0.9786 | 0.9962 |
| K2 + S2_BOT1 | 0.9755 | 0.9808 | 0.9710 | 1.0009 |
| **K2 + SPLW2 0.5** (always ceil(0.5 x 132 / 32) = 3 tiles) | **1.0091** | 1.0070 | 1.0107 | 0.9983 |
check_mt M 9 12 13 16 0 failures for all three new builds. Same-build copies differ by up to 0.6% (K2), so one copy
is not evidence. K2 (KSPL) over runs 5-6 (3 copies): +0.33, -0.12, +0.50%. BOT1 with the split: -1.8 / -2.5%.
K2 + 0.5-wave split: +0.9% in one copy -> repeat + neighbours (1.0 / 0.3 waves, K4 0.5) in run 7.

## 9. Run 7 (v1606; bench2 jm100 bs16 + bs32; per rep vs the v1002 x2 control mean)
| | bs16 all | light | heavy | bs32 |
|---|---|---|---|---|
| v1002 / v1002b | 1.0008 / 0.9992 | 1.0007 / 0.9993 | 1.0018 / 0.9982 | 1.0011 / 0.9989 |
| K2 + SPLW2 0.5 x2 | 1.0003 / 1.0015 | 0.9996 / 0.9988 | 1.0003 / 1.0018 | (placebo) 1.0002 / 0.9990 |
| K2 + SPLW2 1.0 (5 tiles) | **1.0084** | 1.0076 | 1.0092 | (placebo) 1.0012 |
| K4 + SPLW2 0.5 | **1.0061** | 1.0077 | 1.0055 | (placebo) 1.0004 |
| K2 + SPLW2 0.3 (2 tiles) | 0.9945 | 0.9796 | 0.9961 | (placebo) 1.0003 |
| MG1=1 bound (bs16 2-group items on the 1-group loop) | 1.0069 | 1.0005 | 1.0154 | (placebo) 0.9988 |
| MG1=2 bound (bs32 multi-group items on the 1-group loop) | (placebo) 1.0009 | | | **0.9769** |
check_mt M 9 12 13 16 (K2 w1.0, K4 w0.5, K2 w0.3) and the 120-call mix (K2 w0.5): 0 failures.
- K2 + SPLW2 0.5 did not repeat run 6's +0.9% (+0.03 / +0.15%). All K2 builds have byte-identical consumer loops (the
  split rule is routing-only code), so the spread between them is the rule plus noise (same-build copies up to 0.6%).
- Multi-group compute bounds: bs16 2-group compute is worth at most ~0.7% (all on heavy reps, and the bound also
  re-schedules the 1-group loop); at bs32 dropping the multi-group loops makes the kernel 2.3% SLOWER (the 1-group
  loop's new schedule): no positive prize visible, so no multi-group stage-code work for bs32. Closed.

## 10. Final summary (2026-10-31; stopped at the planner's request, no run 8 queued)
Nothing beat the controls by more than the noise (~±0.3% per copy, up to 0.6% between copies of the same build).
NOTHING IS BANKED. Every default build is SASS-identical to v1002 in every kernel (local NVRTC 13 hashes).

Per-shape facts for others:
- bs16 (4,2,1): 1-group items run at the ring rate, as in (4,1,1). Compute per middle stage is 1440-1455 cycles vs
  1380-1404 in (4,1,1), against an ~1870-cycle memory-paced period. That is hidden. The "(4,1,1) is faster on light
  routings" effect (+1.5% on v1002) is the (4,1,1) kernel's k-split tail: trace tail ~1.8 us vs ~4.5 us. The
  (4,1,1) kernel's dead NW1S copy gives back about 1.2 us of startup. On heavy routings (4,2,1) wins by ~4%.
- bs16 2-group gate/up stages look producer-bound in the WPN trace build (16 per-token row copies). Bounds that remove
  that cost do not move bench2, and removing all 2-group compute is worth at most +0.7%. Not a lever.
- bs16 tail: K2 / K4 pieces now use the KRS buffer (v1605; correct across interleaved bs2-16 calls), and the trace
  tail drops 4.8 -> 2.1 us. In bench2, every split rule tried is within noise (-0.55 .. +0.9% single copies). The
  split build costs startup (+0.6 us) and first-item time in traces. If the tail is revisited, the next step is a
  multi-copy job: K2 + SPLW2 1.0 and K4 + SPLW2 0.5, x3 each, 150 reps.
- bs32: the ~27 wgmma first-wave CTAs cannot be started earlier for a gain. Even a free, already-gathered x at 2.5 us
  (FWXB bound) loses 0.7%: the extra early DRAM fill delays every CTA's cold startup code (c15 +1.0 us, producer loop
  entry +1.9 us). WP-32's FSTW now measures -1.2% on v1002. WGSN for MW 1 is neutral. The multi-group compute bound
  is negative (loop re-schedule). Steady state is at the DRAM rate. What is left is item edges (WP-EDGE2) and
  tail / end (WP-END).
- bs8 (4,1,1), side finding: the 16 KB MOE_NW1S copy is dead under SEPI (the SEPI epilogue reads P.nw1). Removing it
  moves first data 5.11 -> 4.09 us but gains nothing in bench2: the startup DRAM fill hides it.
- Harness note: with B2_SHARE_EQ, state["ykz"] (zeros, > 1M elements) is SHARED between variants while each variant
  has its own ctr / parity. KRS variants in the same bench2 job dirty each other's rows. Timing is valid; outputs
  after rep 2 are not. Use B2_NOVERIFY or separate jobs for correctness.

Files: dev/moe_v1600.py (B411, S2_BOT1), v1601 (SWZ), v1602 (NONW1), v1603 (XB1 / XB8 bounds), v1604 (FWXB bound),
v1605 (KRS for the (4,2,1) split, SPLW2), v1606 (MG1 bound), v1607 (SPLW1, untested); dev/wps2_run1-7.sh,
dev/wps2_mix.py; dumps dev/remote_out/wps2_*.json; logs in the session scratchpad logs/run*.log.
