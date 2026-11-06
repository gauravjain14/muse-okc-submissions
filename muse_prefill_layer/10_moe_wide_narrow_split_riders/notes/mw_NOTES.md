# mw worker: tile plan, per-item overheads, wide items and tail of the fused NVFP4 MoE kernel (2026-10-31)

Base: research/claude/stk/s_b.py (sha256 55689a16...). cu/sb_moe.cu / cu/sb_glue.cu = its _pm__CUDA_SRC / _pm_GLUE verbatim.
Scope (split with mx): tile plan, per-item overheads (first stage, hand-off, cold epilogues, d256 vs g256), wide main
loop / epilogue, tail. mx owns the narrow-item execution path.
Tools: build.py (variants as text patches, one function per lever; v_<name>.py overlays on s_b.py), r.sh (MoE-only
paired bench = mx/mb.py, STK_FUSE=0, item traces kept), ana.py (= mx/ana.py), sass.sh / sass_loops.py (nvcc 13 SASS).

## 0. Where the overhead time goes (s_b item traces; mx logs r1_11_22.json, r2_11_22_2_3.json; means per item)
Epilogue of the first item of a width class on an SM vs a repeat of that class on the same SM (us):
| T | very first item (g256) | g224 first / repeat | g192 | g160 | g128 | d256 |
| 3012 | 4.60 (vs 1.12) | 3.07 / 0.98 | 3.18 / 0.84 | 2.49 / 0.68 | 1.97 / 0.68 | 1.49 / 1.27 |
| 753  | 4.71 (vs 1.06) | 1.21 / 1.08 | 1.26 / 1.32 (g208) | 0.90 / 0.73 | 0.95 / 0.64 | 1.65 |
At 3012 (tile-major order) the CTAs run in lockstep waves: a width class with one tile has its 64 gate items start
within ~0-3.5 us of each other, and their epilogues fall from 3.5 to 1.9 us in order of start time (g160, d11 c2): a
shared cache (L2) warming up, i.e. cold instruction fetch of the class's straight-line epilogue code by many SMs at once
after the judge's L2 flush. At 753 the MAT_SMALL interleaved order spreads a class over time: first-of-class costs only
+0.1..0.3 us. Cost of cold epilogues: ~7.8 us per CTA at 3012 (very first item ~3.5, fine classes ~4), ~1.9 at 753.

## Log
### r1 (logs/r1.json, d11 cases 0-3, 6 reps): s_b vs pipe_all (= mx p1: pipelined fetch + smem tile entry)
| case | base | p1 | p1 vs base | trace: sbt -> p1t per CTA |
| 0 | 260.4 | 259.9 | +0.18% | |
| 1 | 260.4 | 262.3 | -0.72% | |
| 2 | 812.8 | 815.4 | -0.33% | firstwait 14.1 -> 6.6, handoff 3.0 -> 0, epi 27.6 -> 29.9, gap 7.0 -> 8.1; span 771.1 -> 764.7 |
| 3 | 751.6 | 754.3 | -0.36% | span 707.7 -> 704.7 |
The trace builds favour p1 (+0.5..0.7%), the real builds do not: the non-pipelined loop's trace hooks sit on its
critical path. p1's first stage is 0.13 us/item (0.42-0.48 before), but d256 epilogue 1.27 -> 1.50 (moe16 saw the same)
and fine-class epilogues rise. p1 also changed the 16K (BIG) kernel's allocation (LDL 4 -> 11, STL 2 -> 3): p1b keeps
the BIG source as it was.

### r2 (logs/r2.log/json, d11+d22 cases 0-3, 8 reps; MoE-only, us; sb = s_b's source through the overlay = control)
| draw/case | base | sb | p1 | p1b | p2 |
| 11/0 | 260.5 | +0.23% | -0.15% | +0.03% | -0.17% |
| 11/1 | 261.4 | -0.17% | -0.93% | -0.47% | -0.72% |
| 11/2 | 818.8 | -0.47% | -1.96% | -1.94% | -1.57% |
| 11/3 | 759.5 | -0.02% | -0.29% | -0.21% | 0.00% |
| 22/0 | 285.7 | -0.10% | -0.70% | -0.59% | -1.04% |
| 22/1 | 314.3 | -0.43% | -0.45% | -0.47% | -0.80% |
| 22/2 | 780.6 | -0.03% | -0.10% | -0.21% | +0.12% |
| 22/3 | 870.1 | 0.00% | +0.01% | +0.03% | +0.32% |
p1b = p1 with the 16K (BIG) source untouched; p2 = p1b + the plan's tile list cached in shared memory at kernel start
(the producer's per-item tile entry without a dependent global load before publishing). Control noise ~0.3-0.5%.
Verdict: the pipelined fetch does not pay at 753 / 3012 (753: -0.5..-1%), with or without the tile cache. The trace's
-10 us/CTA of first-stage + hand-off does not show in wall time.

### r3 (logs/r3.*, d11+d22 cases 0-3, 8 reps): lever 1, tile plan (tplan.py; table from mktab.py over the 3012 costs)
Above 256 rows, an expert's last 256 + r (512 + r) rows are re-split into 2-3 tiles of >= 128 (tp128) / >= 160 (tp160)
rows by a 16-row-bin table (min modeled cost, plan_dp.py / mktab.py) instead of 256 tiles plus a narrow remainder.
e.g. 552 -> [224, 208, 128] (was [256, 256, 64]); 2155 -> 6 x 256 + [224, 208, 192] (B = 512 + 107 = 619; was
8 x 256 + 128). Model: +0.6..2% of the MoE call. Measured (vs base, same run):
| draw/case | sb | tp128 | tp160 |
| 11/0 (plan unchanged) | -0.21% | -0.49% | -0.40% |
| 11/1 | -0.24% | -0.21% | -0.22% |
| 11/2 | -0.03% | -0.71% | -0.72% |
| 11/3 | +0.09% | +0.16% | -0.06% |
| 22/0 | +0.02% | -0.27% | -0.29% |
| 22/1 | +0.02% | 0.00% | -0.37% |
| 22/2 | +0.09% | +0.06% | -0.09% |
| 22/3 | -0.02% | -0.07% | +0.03% |
Outputs: rows that change tile class change accumulation order where NACC differs (N <= 128 grouped tiles use 2-4
accumulator sets): rel 3e-5..1.3e-4, max abs 0.004 (same class as s_b's run-to-run A/A). Verdict: no gain.
Trace comparison across runs (d11 c1): the 552-expert's tiles drop 73.6 -> 70.4 us/CTA as modeled, but every other
class is 2-3% slower in that run; total item cycles per CTA identical (395.2K vs 395.3K). Same-run traces in r4.

### r4 (logs/r4.*, d11 cases 1-3, 10 reps, same-run traces): tile plan vs base; dry-pass warm-up (wm)
| case | base | tp128 | wm (dry first item) |
| 1 | 260.8 | -0.18% | -1.7% |
| 2 | 816.0 | -0.42% | -1.2% |
| 3 | 777.9 | +0.20% | -0.8% |
Why the tile plan does not pay (same-run traces, per CTA): case 2 main loops -7.5 us (713.8 -> 706.3, as modeled) but
epilogues +6.0 (28.2 -> 34.2) and tail +3.7; case 1 main -2.7, epilogue +1.8. The extra epilogue time is all in the
first item of each width class on each SM (cold code): first-of-class epilogue sum per CTA 13.7 -> 23.1 us (case 2),
11.7 -> 20.2 (case 3), 7.6 -> 9.6 (case 1); repeats unchanged (~1.0 us each). The plan adds tiles to the fine classes
(224 / 208 / 192), and each such class's first wave pays 2-4 us per SM. => cold epilogue code is the blocker; fix that
first (lever 2), then the tile plan's main-loop gain (~7 us/CTA at 3012, ~3 at 753) can land.
Base cold-epilogue cost from the same traces (first-of-class minus repeat): 3012 ~7.8 us/CTA (very first item 3.7 of
it), 753 ~2 us/CTA.
wm (each CTA dry-runs its first item's consumer_item: epilogue only, same call site) does warm the first real
epilogue (4.83 -> 1.12 us at 3012, 1.93 -> 0.83 at 753), but the dry pass itself takes ~6.8 us (cold, during the
initial HBM burst), and the `dry` flag live across the main loops made ptxas spill (3012 kernel LDL 93 / STL 29 vs
2 / 1): every item's first-stage wait +0.15..0.33 us. Rejected.

### Lever 2 attempts: shared wide-epilogue code (wide.py) - ptxas will not have it
consumer_item_wide: one consumer for N = 160..256 (shared accumulator array, one main loop per width in a switch, one
epilogue with runtime-width guards), so all wide classes share their epilogue code. Every form compiles to C7511/C7512
(all wgmma of the kernel serialized) or spills, on nvcc 13 (sass.sh; logs/sass_wd*.log, sass_wf*.log):
| variant | 753 / 3012 kernels |
| wd (lambda main loop, 5 widths) / wdp (width packed in ntile) | C7512 |
| wd3 (256/224/192) | C7512 + LDL 14-23 |
| wdf (224/208/192/160, 256 untouched) | C7511 |
| wdc (per-width accumulators copied into the shared one) | C7512 |
| wf (main loop as a template function, no lambda) | C7512 |
| wf1 / wf1c (only 256 through the new function; wf1c: compile-time width in the epilogue) | C7512 + LDL 21-23 |
| cs1 (control: only the call site restructured, consumer_item<256> unchanged) | clean |
So even a one-width copy of the consumer with the incumbent's main loop and epilogue tips ptxas over: the consumer
side has no register slack at all. Closed.
### Lever 2b: spread item order (so, logs/r5.*) - right idea, wrong place
mw_spread: 256-row tiles tile-major, the other tiles in chunks of 8 n-tiles round-robin (3012 gate + all down items).
In the trace the cold storms go (g192 first-of-class epilogue 3.3 -> 1.3 us), but the consumers computing the mapping
(n256 + divisions, live across the item) spill (3012 kernel LDL 81 / STL 9): -6..-10% everywhere. Rejected; redone with
the mapping in the producer (handoff): the producer maps each item and hands (tile, ntile) to the consumers in shared
memory, n256 travels in ntiles' upper 16 bits from plan_warp (ho = handoff only, same order; ho2 = handoff + spread;
ho2tp = + tile plan). SASS: ho / ho2 / ho2tp 753 + 3012 kernels LDL 0 / STL 0 (base: 0 / 0 and 2 / 1), no C75, 16K
kernel unchanged (LDL 4 / STL 2, 21224 lines).

### r6 (logs/r6.*, d11+d22 cases 0-3, 8 reps): handoff, spread (all non-256 tiles), + tile plan
| draw/case | base | ho | ho2 | ho2tp |
| 11/0 | 262.1 | +0.36% | -4.4% | -4.4% |
| 11/1 | 262.2 | +0.04% | -2.3% | -3.0% |
| 11/2 | 820.4 | +0.37% | -0.9% | -0.8% |
| 11/3 | 759.9 | +0.18% | -1.9% | -1.0% |
| 22/0 | 289.1 | +0.22% | -2.8% | -3.1% |
| 22/1 | 316.5 | -0.45% | -2.2% | -2.3% |
| 22/2 | 783.7 | +0.17% | -1.4% | -1.0% |
| 22/3 | 861.7 | +0.27% | -0.9% | -2.0% |
ho (consumers take (tile, ntile) from the producer, same order): +0.15% average, 7/8 > 0 (noise ~0.3%): the consumers'
item-start arithmetic is gone. ho2: spreading every non-256 tile breaks the shortest-last end of each phase: tail
4.6-5 -> 8.6-15.9 us/CTA, last-end spread 10.4 -> 16-29 us. Next (r7): spread only the 128..224-row tiles (the classes
with cold epilogues), narrow tiles stay tile-major at the end; the plan packs (tiles, 256-row tiles, >= 128-row tiles)
as 12 / 10 / 10 bits in ntiles.

### r7 (logs/r7.*, d11+d22 cases 0-3, 8 reps): restricted spread (ho2: only 128..224-row tiles), tile entry handoff
| draw/case | base | ho | ho2 | ho2tp | ho3 (entry handoff) |
| 11/0 | 261.9 | +0.23% | +0.05% | -0.52% | -6.7% |
| 11/1 | 261.6 | -0.25% | -0.55% | -0.23% | -7.2% |
| 11/2 | 834.8 | +0.38% | +0.20% | -0.19% | -6.4% |
| 11/3 | 774.0 | +0.27% | +0.17% | +0.34% | -7.3% |
| 22/0 | 288.2 | -0.01% | -0.20% | -0.40% | -8.6% |
| 22/1 | 316.7 | +0.02% | +0.20% | -0.22% | -7.5% |
| 22/2 | 808.0 | +0.35% | +0.88% | +1.04% | -4.9% |
| 22/3 | 878.7 | +0.31% | -0.03% | -0.65% | -6.6% |
ho again +0.16% average (r6 + r7: 14/16 >= 0, all four 3012 cells +0.17..0.38% both runs). ho2 = ho within noise:
the restricted spread trims the first-of-class epilogue sum only a little (d11 c2 13.7 -> 12.1 us/CTA; g128 3.0 ->
1.5, g160 2.7 -> 1.8, g192 3.3 -> 2.6) and the tile plan on top still does not pay. ho3 (the producer also hands the
4-int tile entry, from a shared-memory tile cache) spills: 3012 kernel LDL 177 / STL 16, -5..-9%. Every consumer-side
addition hits the register cliff.
Orchestrator: wrap up with ho as the candidate (mw_ho.py, mkfinal.py ho); ho2 measured alongside in the replica.

## Candidate: research/claude/mw/mw_ho.py (python3 mkfinal.py ho mw_ho.py)
= s_b.py with _pm__CUDA_SRC = cu/ho.cu (build.py handoff(base(), False, False)); 907,944 bytes,
sha256 7cef6a1544d17fa3dd10d970add37e7216430f6e24c6752e7279360b36eff61f.
Change (753 / 3012 / every T < 8192 instantiation; the 16K BIG kernel compiles as before: LDL 4 / STL 2, 21224 lines):
the producer thread maps each claimed item to (tile, weight n-tile) (mw_map: the incumbent order, MW_SPREAD_* = 0) and
stores the pair in shared memory next to the item index before publishing it; the consumers read it there (before
they release the hand-off slot) instead of each recomputing the MAT_SMALL / tile-major mapping (divisions by the tile
count) at every item start. plan_warp packs (tile count | 256-row tiles << 12 | >= 128-row tiles << 22) into ntiles
(only the fused kernel reads it; 16K's plan_warp_pg is unchanged), the kernel unpacks it. Same items, same order, same
tiles: MoE outputs bit-identical to s_b (mb.py errs 0 / 0 on all 16 draw x case runs of r6 + r7).
SASS (nvcc 13): 753 kernel 22456 lines LDL 0 / STL 0 (s_b 22632, 0 / 0); 3012 kernel 22968 lines LDL 0 / STL 0
(s_b 23016, 2 / 1); no C75 warnings.
Checks: gate CHECK_RC=0 PASS=42 CASES=21 (logs/gate_ho.json, gate_ho.log); whole layer vs s_b (stk/cmp.py cases 0-3,
2 reps): rel 4.1e-5..1.5e-4, inside the A/A envelope (s_b vs itself 5.4e-5..1.75e-4), topk_ids / caches exact.
MoE-only (r6 + r7, d11 / d22): +0.16% average; 3012: +0.17..+0.38% (8/8 > 0); 753: -0.45..+0.36% (noise).
Replica panel 11,22,33, cases 0-3, 8 reps (logs/rep1.log; each candidate in its own AB/BA pair with SGLang), layer us:
| case | d11 base / ho / ho2 | d22 base / ho / ho2 | d33 base / ho / ho2 |
| 0 (753 sliding) | 521.1 / 521.9 / 524.2 | 527.4 / 528.4 / 529.1 | 517.9 / 519.0 / 518.1 |
| 1 (753 full) | 706.2 / 701.2 / 706.7 | 763.1 / 761.1 / 762.1 | 702.7 / 703.7 / 705.1 |
| 2 (3012 sliding) | 1746.4 / 1735.0 / 1735.5 | 1756.1 / 1751.6 / 1752.3 | 1739.2 / 1734.7 / 1741.4 |
| 3 (3012 full) | 2474.8 / 2477.1 / 2474.3 | 2574.9 / 2579.2 / 2579.9 | 2453.5 / 2460.0 / 2450.9 |
RELATIVE ho/base 1.0007 0.9989 0.9971 (mean 0.9989); ho2/base 0.9987 0.9982 0.9992 (mean 0.9987): both inside the
per-draw noise (~+-0.25%); case 2 is -4.5..-11.4 us for ho in all three draws, case 3 +2..+6.5 us. ho2 does not win
on panel 1 -> dropped (orchestrator rule: include only if it wins on both panels).
Replica panel 44,55,66, cases 0-3, 8 reps (logs/rep2.log), layer us:
| case | d44 base / ho / ho2 | d55 base / ho / ho2 | d66 base / ho / ho2 |
| 0 | 497.5 / 497.3 / 497.3 | 530.8 / 532.3 / 532.0 | 510.4 / 509.9 / 511.2 |
| 1 | 693.8 / 692.0 / 692.2 | 740.7 / 737.2 / 738.5 | 709.0 / 708.7 / 709.7 |
| 2 | 1708.3 / 1715.6 / 1714.7 | 1767.5 / 1765.1 / 1769.0 | 1721.8 / 1721.3 / 1727.1 |
| 3 | 2452.9 / 2452.0 / 2459.2 | 2488.8 / 2486.1 / 2484.8 | 2459.6 / 2460.1 / 2464.1 |
RELATIVE ho/base 0.9985 1.0009 1.0001 (mean 0.9998); ho2/base 0.9964 0.9995 0.9978 (mean 0.9979).
Both panels, ho - base per case (6 draws, us): case 0 +0.6 mean, case 1 -1.9, case 2 -2.7 (5/6 <= 0), case 3 +1.7.

## Bottom line (2026-10-31, wrap-up on the orchestrator's request; GPU released)
- mw_ho.py is correct (gate 42/42, MoE bit-identical to s_b, layer inside the A/A envelope) and consistently a hair
  faster in the MoE-only bench (+0.16%, 3012 +0.17..0.38%), but in the layer replica it is neutral: panel RELATIVE
  0.9989 / 0.9998. Not a submission on its own; it costs nothing and can stack under other MoE changes.
- What blocks the bigger items in this lane is the consumer warpgroups' register budget (232 at the limit):
  pipelined fetch (p1/p2): no wall gain; tile plan: real main-loop gain (-7.5 us/CTA at 3012) eaten by cold
  first-of-class epilogues (+6..9 us/CTA); shared wide epilogue: C7511/C7512 in every form; dry warm-up pass, spread
  mapping in the consumers, tile-entry handoff: spills. The cold-epilogue cost itself (~8 us/CTA at 3012, ~2 at 753;
  very first item 3.7 of it) remains the main per-item overhead after the epilogue math.
- Not done: in-graph ncu before/after (ho's change is an item-start bookkeeping move; SASS main loops are unchanged).
