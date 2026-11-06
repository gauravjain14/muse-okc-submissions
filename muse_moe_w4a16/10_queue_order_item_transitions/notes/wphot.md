# WP-HOT: the bs256 (bs128) hot expert's continuation tiles, 2026-11-01

Worker WP-HOT. Base line B dev/moe_v1511.py, line A dev/moe_v1504.py. Namespace v3200-v3299, defines MOE_HOT_*,
scripts dev/wphot_*, logs kda_ws/runs/wphot_runN.log, per-rep samples dev/remote_out/wphot_runN_pP.json.
RESULT: CLOSED, nothing banked (section 6).

Files (every patch script applies to both lines: `python3 dev/wphot_mkNNNN.py <base> <out>`; every default build = the base's
SASS: line B all 16 kernels, line A 15 of 16, moe_4_2_1_4_1 being bistable (8fb6104136ab / aef9ae76c3f5 for v1504 itself)):
| file (B / A) | script | defines | what |
|---|---|---|---|
| dev/moe_v3200.py / v3201.py | dev/wphot_mk3200.py | MOE_HOT_BND=1 / 2 / 3 | decode bound (no-decode continuation tiles; never taken; odd tiles = pair model) |
| dev/moe_v3211.py (B) | dev/wphot_mk3211.py | MOE_HOT_BND4=1 / 2 | bytes bound (continuation weights from one L2-hot block) |
| dev/moe_v3202.py / v3203.py | dev/wphot_mk3202.py | MOE_HOT [, MOE_HOT_MIN=n] | real pair path, first form |
| dev/moe_v3204.py / v3205.py | dev/wphot_mk3204.py (on v1511 / v1504) | + MOE_HOT_G1ONLY | second form (weights once per k chunk) |
| dev/moe_v3208.py / v3209.py | dev/wphot_mk3208.py (on v3204 / v3205) | + MOE_HOT_2S | third form (bulk-reloaded second entry) |
| dev/moe_v3206.py, v3210.py (B) | dev/wphot_mk3206.py (on v3204 / v3208) | + MOE_HOT_TR | per-item trace twins |
| dev/moe_v3207.py (B) | dev/wphot_mk3207.py (on v3204) | MOE_HOT | descriptor-form experiment (SASS only) |
Tools: dev/wphot_split.py, dev/wphot_chk.py, dev/wphot_trace.py; jobs dev/wphot_run1..7.sh.

Target: an expert with > 64 tokens runs as 2+ 64-token tiles of the (8,8,1) kernel; every tile streams and DECODES the
expert's 48M weights again (the bytes of concurrent tiles mostly come from L2: tile t and t+1 of an expert are adjacent
in the queue, so their row blocks run in the same wave). Planner estimate: one decode per row block instead of two saves
~644 SM-us = ~4.9 us at bs256 (~-2.4%).

## 0. Routing facts (jm seeds, local CUDA generator; the H200's numbers differ slightly: bench2 reports max_tok med 96)

| bs | busiest expert tokens min / med / mean / max | experts > 64 tokens (mean) | tiles (mean) | tiles with 128-token tiles | reps with a continuation tile |
|---|---|---|---|---|---|
| 128 | 16 / 42.5 / 52.6 / 128 | 0.26 | 23.31 | 23.05 | 24 / 100 (H200 jm: 62 / 200) |
| 256 | 30 / 99.5 / 122.1 / 256 | 0.91 | 26.57 | 25.44 | 72 / 100 (H200 jm: 138 / 200) |

bs256 H200 jm reps by busiest expert (200 reps = 2 passes x 100): <= 64: 62, 65-80: 18, 81-96: 22, 97-128: 26,
129-192: 26, > 192: 46 (an expert with 3-4 tiles in 23% of the reps).

## 1. Bound build (dev/wphot_mk3200.py -> dev/moe_v3200.py line B, dev/moe_v3201.py line A)

MOE_HOT_BND=1: in the (8,8,1) MW 4 / 8 kernels, the wgmma items (>= 17 tokens) of an expert's continuation tiles
(tile != ftile[e]) run a stage body with the raw weight words as A (no decode; same LDS, same wgmma count, same copies,
same epilogue; outputs wrong). MOE_HOT_BND=2: same code, condition false at run time (code-presence control).
MOE_HOT_BND=3: only the odd continuation tiles (tile - ftile[e] = 1, 3): what a pair design ((t0, t1), (t2, t3) in one
item, one decode per pair) saves. Default build = base SASS (line B all 16 kernels; line A all but moe_4_2_1_4_1, whose
hash flips between 8fb6104136ab and aef9ae76c3f5 with unrelated source / define changes: bistable, also seen by WP-CGS).
Bound builds: 6 new 144-instruction no-decode loops per kernel (32 HGMMA, no decode), every existing stage loop
byte-identical, same spills.

### run 1 (kda_ws/runs/wphot_run1.log): INVALID for BND=1 (warm-up bug), BND=2 valid
bench2 B2_NOVERIFY jm 100 reps x 2, fillers at both ends, 4 middle v1511 controls. Bug: the warm-up stage's record has
e = 0, so `tile != ftile[0]` sent most CTAs' warm-up through the no-decode loop and the real first stage ran cold:
bnd1 at bs256 reps WITHOUT continuation tiles -1.03 / -1.11 / -1.18% (bs128 has no warm-up stage: -0.01..-0.19).
Read with that offset, bs256 by busiest expert (bnd1 copies, % vs the middle controls): 81-96 tokens -0.04..+0.05,
97-128 +0.31..+0.56, 129-192 +1.02..+1.56, > 192 +2.82..+3.16 (i.e. ~+1.1 / +1.5 / +2.4 / +4.0 over the warm-up offset);
bs128: 81-96 +1.14..+1.36, 97-128 +2.12..+2.19, <= 80 ~0. BND=2 (code presence of the 6 small loops, never taken):
bs256 -0.24 / -0.35% (all reps), bs128 -0.03 / -0.14%. Fixed in run 2 (`!warm`).

### run 2 (kda_ws/runs/wphot_run2.log): THE BOUND (warm-up fixed)
Smoke ok; dev/wphot_chk.py bs256 10 jm routings: BND=1 / 3 FAIL (NaN) exactly on the routings with a > 80-token expert, ok on
the others; BND=2 all ok. bench2 B2_NOVERIFY jm 100 x 2, fillers at both ends, 4 middle v1511 controls (median of per-rep
ratios vs the controls' mean, the judge's statistic; dev/wphot_split.py):

| variant | bs256 all reps (3 copies) | bs256 by busiest expert: <=64 / 65-80 / 81-96 / 97-128 / 129-192 / >192 | bs128 all | bs128 81-96 / 97-128 |
|---|---|---|---|---|
| BND=1 (no decode on any continuation wgmma tile) | +0.95 / +0.73 / +0.67 (mean +0.78%) | -0.2 / -0.1 / +0.8 / +1.1 / +1.9 / +3.3 | -0.01 / +0.02 / -0.02 | +1.0 / +1.9 |
| BND=3 (pair model: odd continuation tiles) | +0.67 / +0.56 / +0.61 (mean +0.61%) | -0.4 / -0.3 / +0.7 / +1.0 / +1.1 / +2.2 | +0.11 / +0.18 / +0.03 | +1.2 / +1.9 |
| BND=2 (code, never taken) | -0.18 / -0.11 | -0.2 everywhere | +0.02 / -0.08 | ~0 |

=> The decode of the continuation tiles is worth +0.8% at bs256 in the judge's median statistic (+0.9% net of the bound
code's own -0.14%), the pair design's share +0.6% (+0.75% net). Far below the planner's 2.4%: the saving sits in the 69% of
reps with a > 64-token expert (+1..3%), and the median of per-rep ratios moves by much less than the mean. bs128: ~0 in the
median (only 31% of its reps have a continuation tile). Not pursued at bs128.

## 2. Real path, first form (dev/wphot_mk3202.py -> dev/moe_v3202.py B / dev/moe_v3203.py A; define MOE_HOT, MW 8 only)
Pairs (2 j, 2 j + 1) of the rank-0 expert's tiles (full first tile, >= MOE_HOT_MIN = 17 tokens in the second, below
tsplit) become one queue item (queue tile q -> real tile q < np ? 2 q : q + np; every other table keeps the real numbering).
A pair item = 2 nst half-stages of 128 k (per half: the k chunk's 35 KB weight block + 2 x 64-k chunks of each tile's
activations = 67 KB); consumers decode once per k16 step (8 A registers double buffered) and issue an n64 wgmma per row
group into acc (tile t) and accq (tile t + 1, padded to 64 tokens); acc goes through the normal item end; accq to a per-CTA
global scratch (P.wc13h, unused by this kernel) and a second epilogue entry (aq_slot = NS, record in the header's free 48
bytes at 272, scratch reuse guarded by an mbarrier at 760); the epilogue warps run wpa_g1 / wpa_g2 straight from the scratch.
Codegen: accq at function scope (re-zeroed per pair) or 2-step decode groups made ptxas serialize EVERY wgmma of the kernel
(C7512, "insufficient register resources"); accq scoped to the pair block + 1-step groups: no C75xx, 694-instruction pair
loop (64 HGMMA), but all 8 stage loops and both producer loops re-scheduled (wpcgs_search: 0/8, 0/2, similarity 0.64),
spills 16/44 -> 60/88 B.

### run 3 (kda_ws/runs/wphot_run3.log): correct, much SLOWER
hang1 8 sizes ok; wphot_chk 24 jm bs256 routings ok; check_mt M 129 160 192 200 224 240 255 256 257 300 512 (normal + skewed,
3 seeds, 2 calls) 0 failures; outputs 100% bitwise-equal to v1511 (dev/wpit_cmp.py M 200 / 256: differences = v1511's own
KRED run-to-run noise). bench2 jm 100 x 2: MOE_HOT -5.41 / -5.50 / -5.35% at bs256 (by busiest expert: 81-96 -5.0, 97-128
-7.1, 129-192 -8.4, > 192 -8.0), MOE_HOT_MIN=999 (the code, no pairs) -0.87 / -0.55%; bs128 (kernel unchanged) 0.
Reading: one pair costs as much as two (a per-call critical-path effect, ~16 us): (1) a half-stage re-copied the 35 KB
weight block, so one SM had to fill 134 KB per 256-k chunk (above the per-SM fill rate of ~35-40 KB/us); (2) a pair DOWN
item is twice as long as a single one and starts at the gate/up -> down boundary of the queue: it becomes the tail.

## 3. Second form (dev/wphot_mk3204.py -> dev/moe_v3204.py B / dev/moe_v3205.py A)
The weight block copied once per k chunk (half 0 = "A": weights + chunks 0-1 of both tiles' rows; half 1 = "B": chunks 2-3
only); the consumers read chunks 2-3's words from the A slot and release A and B together at the k chunk's end (99 KB per
chunk into one SM instead of 134). MOE_HOT_G1ONLY: only gate/up items paired (down items stay single, in parallel).

### run 4 (kda_ws/runs/wphot_run4.log): correct, still slower; code presence -3.5% (lottery)
Correctness as run 3 for both (check_mt 0 failures, 24 hot routings ok, bitwise = v1511). bench2 jm 100 x 2 (bs256, % vs
middle controls): MOE_HOT -8.10 / -8.07, G1ONLY -3.95 / -3.97, MOE_HOT_MIN=999 (code, no pairs) -3.63, G1ONLY+MIN=999 -3.34.
=> this build's mere presence costs -3.3..-3.6% (run 3's -0.6..-0.9%: same design, a ptxas lottery on the re-scheduled
loops); the mechanism on top: pairs everywhere -4.5%, gate/up pairs -0.6%. Fewer bytes did not help.

### run 5 (kda_ws/runs/wphot_run5.log): per-item trace (dev/wphot_mk3206.py MOE_HOT_TR, dev/wphot_trace.py; 8 jm routings
with a > 80-token expert, judge protocol; "single" = the same binary with MOE_HOT_MIN=999)
| | span (us) | G1 ntl8 single | G1 pair | G2 ntl8 single | G2 pair | G1 ntl1 single |
|---|---|---|---|---|---|---|
| single | 185.6 | 27.61 | - | 27.79 | - | 16.13 |
| pairs G1 + G2 | 202.7 (+17.1, every routing +11..23) | 27.66 | 51.89 | 28.29 | 46.76 | 16.76 |
| pairs G1 only | 189.3 (+3.7) | 27.42 | 52.26 | - | - | 16.95 |
- A pair gate/up item (51.9 us) saves only 6% against the two singles it replaces (2 x 27.6): in the pair loop decode and
  tensor work are additive (pair chunk 5525 cycles ~= decode 1500 + 4096 tensor; the single ntl8 chunk 2940 overlaps ~600).
- Single items run 4-5% slower in the same binary once pairs run (16.13 -> 16.8-17.0 us), and the down pairs add ~14 us:
  the second epilogue entry ran wpa_g1 / wpa_g2 straight from the global scratch, i.e. 16 (gate/up) / 64 (down) dependent
  L2 round trips per entry on the epilogue warps, which hold the slots the CTA's next items need.
- The pair loop: 64 HGMMA, 16 commit groups per k chunk, and ptxas builds the two tiles' 64-bit descriptors in vector
  registers and moves them with 2 R2UR per HGMMA (the base loops keep them in uniform registers).

## 4. Third form (dev/wphot_mk3208.py -> dev/moe_v3208.py B / dev/moe_v3209.py A; trace twin dev/moe_v3210.py)
= the second form + the pair's second epilogue entry bulk-reloaded through the room (MOE_F32T tokens per copy, as TMG's
reloads; fence.proxy.async.global after the aq_full acquire) + activation addresses shuffled from lane 0 (ptxas still keeps
the descriptors in vector registers: 2 R2UR per HGMMA) + MOE_HOT_2S (2-step commit groups, A 2 x 16 registers: compiles
without C75xx now that accq is block-scoped). dev/wphot_mk3207.py (32-bit descriptor adds) did not remove the R2URs either.

### run 6 (kda_ws/runs/wphot_run6.log): correct; the mechanism itself is negative
MOE_HOT, + MOE_HOT_2S, + MOE_HOT_G1ONLY: hang1 8 sizes, 24 hot routings, check_mt 11 sizes (normal + skewed): 0 failures,
outputs bitwise = v1511. bench2 jm 100 x 2 (bs256, % vs the 4 middle controls): MOE_HOT (1-step) -5.45, 2S -5.32 / -5.23,
2S + G1ONLY -4.02, MOE_HOT_MIN=999 (code presence) -3.65. Against the never-taken control, by busiest expert:

| vs MOE_HOT_MIN=999 | <=64 | 65-80 | 81-96 | 97-128 | 129-192 | >192 |
|---|---|---|---|---|---|---|
| pairs G1 + G2 (2S, 2 copies) | +0.0 | -0.2 | -2.4 | -2.5 | -3.4 | -3.2 |
| pairs G1 + G2 (1-step) | -0.2 | -0.5 | -2.3 | -2.4 | -3.4 | -3.2 |
| pairs G1 only (2S) | +0.2 | -0.1 | -0.6 | -0.5 | -1.0 | -1.2 |

Trace (8 hot routings): span single 186.3 / pairs 2S 192.6 (+6.3; run 5 +17.1: the bulk reload removed most of the
epilogue stall) / 1-step 194.1 us; pair gate/up item 50.4 us vs 2 x 27.7 for the two singles (-9%), pair down item 46.3 vs
2 x 27.9 (-17%); the last pair down item ends 26 us before the last item end (not the tail); but every other item is
slower while pairs run (G1 ntl1 16.02 -> 16.82 us, ntl2 18.78 -> 19.22, +2..5%), which more than eats the pairs' saving.

## 5. Bytes bound and clock (dev/wphot_mk3211.py -> dev/moe_v3211.py, MOE_HOT_BND4; run 7, kda_ws/runs/wphot_run7.log)
MOE_HOT_BND4=1 (timing only): every stage of a continuation tile copies its expert's first weight block (always an L2 hit
after the first) instead of its own; =2 the same code never taken. Consumer stage loops byte-identical; producer loops
re-scheduled. bench2 B2_NOVERIFY jm 100 x 2: =1 -0.16 / -0.06%, =2 -0.21 / -0.29% vs the middle controls (taken vs not:
+0.05..+0.2% all reps, ~+0.3% on reps with a > 64-token expert). => the continuation tiles' weight bytes are mostly L2
hits already (their row blocks run in the same wave as tile 0's): the brief's design (c) (co-scheduling for L2 reuse) is
bounded by ~+0.2%. Not pursued.
Energy / in-kernel clock (dev/wpe_energy.py judge mode, MOE_CLK, 8 jm routings, bs256): v1511 198.25 us, 180.1 mJ, 1.672 GHz;
third form never taken (MOE_HOT_MIN=999) 203.93 us (+2.9%), 178.0 mJ, 1.695 GHz; with pairs (2S) 206.83 us (+4.3%), 178.4 mJ,
1.697 GHz. => the losses are not power / clock (same or higher clock, slightly less energy): code presence (instruction
schedule) plus the pairs' own cost.

## 6. Conclusion (WP-HOT CLOSED: nothing banked)
- Bound (bs256, judge statistic = median of per-rep ratios, net of the bound code): decoding every continuation tile once
  ~+0.9%, the pair design's share ~+0.75%, the continuation tiles' weight bytes ~+0.2%; bs128 ~0 (31% of its reps have a
  continuation tile). The planner's -2.4% used the mean saving over reps with a hot expert; the median moves far less.
- Real path (three forms, all correct: check_mt bs129-512 0 failures, 24 hot jm routings ok, outputs bitwise = v1511):
  every form loses. Code presence alone (path compiled, MOE_HOT_MIN=999): -0.6..-0.9% (first form) and -3.3..-3.7% (second
  / third: a ptxas lottery). Loop identity (dev/wpcgs_search.py vs v1511's moe_8_8_1_3_8): first form 0/8 stage + 0/2
  producer loops (worst similarity 0.64 / 0.65, spills 16/44 -> 60/88 B); third form 0/8 + 0/2 for MOE_HOT, + 2S and
  + 2S + G1ONLY (stage worst 0.34-0.35, producer 0.97 / 0.75; spills 44/72, 20/48 B); no C75xx in any final build (the
  function-scoped accq form had C7512 = every wgmma serialized). The mechanism
  on top (vs that control): pairs everywhere -2.2..-3.7% on hot reps (-1.4..-4.7% all reps), gate/up pairs only -0.5..-1.2%.
- Why the mechanism loses (trace, run 5 / 6): a 128-token half-k stage is tensor-bound and decode does not overlap it (pair
  chunk ~ decode + 4096 tensor cycles), so a pair item saves only 9% (gate/up) / 17% (down) of the two singles' SM time;
  every other item runs 2-5% slower while pairs run, at the same clock (run 7). Most likely cause (not isolated): the second
  epilogue entry (64 x 1 KB FP32 per pair item through the global scratch: 6 bulk round trips after the fix, 16-64
  dependent L2 loads before it, when the slowdown was larger) keeps the epilogue warps busy, so the CTA's next items wait
  on their held slots; down pairs are also 2x longer items.
- What it would take: a pair loop with real decode / tensor overlap (acc 128 registers leave ~60 for A, words, scales and
  addresses; 1- and 2-step groups compile, a full-chunk double buffer was not tried; a function-scoped second accumulator
  serialized every wgmma of the kernel, C7512), a second-entry epilogue that does not occupy the epilogue warps (e.g.
  consumer-side direct stores), AND an
  implementation that leaves the existing loops byte-identical (none of the forms tried did). Each is hard; together they
  chase a <= +0.75% bs256 (~+0.12% geomean) prize. Recommend closing this bucket.
- Reusable: dev/wphot_split.py (per-rep split of bench2 --out JSON by the busiest expert), dev/wphot_chk.py (per-routing
  check with its busiest-expert counts), dev/wphot_trace.py + MOE_HOT_TR (per-item durations by class and the tail).
