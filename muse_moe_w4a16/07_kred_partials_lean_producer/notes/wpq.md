# WP-Q: fresh-eyes review, new levers (2026-10-27)

Reviewer WP-Q. Base: dev/moe_v567.py (judge check 69f9e304: 5523.8 us). #2 at 5204.1 needs x1.061 geomean, i.e. the
per-shape ln(gain) must sum to 0.357 (e.g. +6% on every shape, or +12% on three).
Read: TASK.md, baseline.py, reference.py, draft.md, planner P1-P14 and all worker files, candidates.jsonl (173 entries),
memory okc-moe-findings / okc-moe-versions, the v567 source (stage_compute, MOE_RUN_ITEM, producer x path, configs).
New files: dev/wpq_place.py, dev/wpq_ntl1.sh, dev/wpq_b411.sh, dev/wpq_cap8.py. Four Modal jobs, run one at a time, each
with nvidia-smi state before and after (logs: scratchpad wpq_place_run1.log, wpq_ntl1_run1.log, wpq_b411_run1.log,
wpq_cap8_run1.log).

## 0. Ranked list

| # | lever | shapes | expected | evidence | first step |
|---|---|---|---|---|---|
| 1 | PDL two-kernel hybrid: tiles with >= 2 token groups on the one-k-warp binary first, 1-group tiles on the lean NT=1 (4,1,1) binary as SMs free up, combine in the second kernel | bs32, bs128, bs256 (bs16) | bs32 +4..5%, bs128 +2..4%, bs256 +1..3% (geomean +1.2..2%) | **measured prize**: bs32 routings capped at 8 tokens/expert (same experts, pairs, bytes): (4,1,1) **x1.069 / x1.072** vs (8,4,1) (control x0.998). Per stage (same bs8 data): 1470+85 cyc in (4,1,1) vs 1580+268 (8,4,1), 1597+197 (8,8,1). The plain bucket change loses (bs32 x0.958, bs16 x0.973-0.982): split multi-token experts | prototype at bs32 (section 2 #1) |
| 2 | the next item stalls at stage NS-1 behind the ring slot held by the previous (large-tile) item's async epilogue | bs256, bs128 | bound ~2% bs256, ~1% bs128 | **diagnosed** (section 2 #2): first epilogues 8941 vs 1387 cyc steady at bs256; not x, not cold code, not sync; warm-ups (v641, v644) and stagger (v642) rejected | 3-warp epilogue for >= 4-group tiles, or NS 4 at bs256 via a separate x ring |
| 3 | freeze the hot loops' codegen: one `__noinline__` item function per (NT, NTL) (or a separately compiled cubin linked with cuLink) | all; unlocks rejected mechanisms | indirect: XR +1.6..1.9% (bs128/256), KSX8=2 +0.9% (bs256), SEPI8, half-item tails (TAILB +4.9% bs8 / +3.7% bs16 bound) | XR_NONE bisection (code presence alone -3.1% / -1.8%, hot-loop instruction counts unchanged, only re-scheduled); ncu: not i-cache | SASS identity check + neutral bench of v567 with the 1-group item loop as a noinline function |
| 4 | bs128 tail k-split partials: the last-arriving piece of a split row block sums the others into BF16 y (combine reads one BF16 row instead of 4 FP32 partials) | bs128 (bs32) | +0.8..1.5% bs128, +0.5..1% bs32 | WP-L KSFREE bound +1.65..1.76% bs128, +1.3% bs32 (2-way) | build (WP-L's suggestion, not built) |
| 5 | bs32 start without the all-CTA gather barrier (first item per CTA fed by its own raw x rows) | bs32 (in the hybrid: kernel A only) | +1..2% bs32 without #1, less with it | P7/P9: first data bs32 7.2 us vs 4.3-4.6 us at bs8/16; the barrier's lag is the writers' red.release behind TMA loads | after #1 (B's 1-group tiles need no gather; only A's first items wait) |
| 6 | PDL tail kernel at bs8/16 (last R items as 4x finer row items in a second kernel, no code in the main kernel) | bs8, bs16 | +0..2% (TAILB bound +4.9% / +3.7%, minus 2-4 us tail-CTA start + cold code) | TAILB, the three tail-split losses (codegen + epilogue latency) | only after #1/#3; measure PDL CTA start latency first |
| 7 | 2 CTAs per SM (Chord): 4 consumer warps per CTA, two producers / rings, per-item barriers decoupled | bs8-32 | 0..3% | Chord report (WP-R item 5); not tried | expensive; after 1-3 |

Dead (probed here): weight / scratch placement autotune (x1.0004); bs16 / bs32 on the (4,1,1) binary as a bucket change
(x0.973..0.982 / x0.958). The coordinator's two-kernel question is answered in section 3.

## 1. Probe results

### Q1. Placement autotune is dead (dev/wpq_place.py, Modal H200, 5 own prepare() states x 2 graphs, jm seeds, 50 reps x 2 rounds)

Earlier notes put the weight-placement effect at 2.2-3.6% between prepare() copies. If it were real and stable within
a process, a prepare()-time best-of-K choice would be worth +1..1.5% everywhere. Measured on v567: graph-to-graph
spread within a round bs1 1.019, bs8 1.010, bs16 1.002, bs32 1.005, bs128 1.0025, bs256 1.0035; between-state spread
(weights) 1.010 / 1.004 / 1.002 / 1.001 / 1.001 / 1.002; round-1 -> round-2 correlation 0.48 (bs1) and -0.3..0.4
elsewhere. Selecting the best state on round 1 and measuring it on round 2: x1.0012 bs1, x1.0003 bs8, x1.0006 bs16,
x0.9999 bs32, x1.0006 bs128, x1.0000 bs256, **geomean x1.0004**. Not a lever. Side result: for A/B work, own-state
copies of the main kernel agree within ~0.3% (bs1 ~1%), so placement is no longer a noise source to design around.

### Q2. The 1-group stage is 7-9% slower in the NT>1 configs on identical data (dev/wpq_ntl1.sh: step1_timing, MOE_TIMING, v567, first jm seed per shape)

Warp 0's cycles per 32 KB stage (comp + data wait). The bucket table was forced so that bs8's routing (seed 70261,
all experts <= 8 tokens: every tile 1 group) runs on each config:

| config (binary) | NT | 1-group comp | wait | total | vs (4,1,1) |
|---|---|---|---|---|---|
| (4,1,1) bs8 routing (default) | 1 | 1470 | 85 | 1555 | 1 |
| (8,4,1) bs8 routing (forced) | 4 | 1580 | 268 | 1848 | +18.8% (comp +7.5%) |
| (8,8,1) bs8 routing (forced) | 8 | 1597 | 197 | 1794 | +15.4% (comp +8.6%) |
| (4,1,1) bs32 routing (forced) | 1 | 1440 | 86 | 1526 | 1 |
| (8,4,1) bs32 routing (default), ntl1 G1 items | 4 | 1577 | 180 | 1757 | +15.1% |
| (8,4,1) bs128 (default), ntl1 G1 | 4 | 1553 | 123 | 1676 | +9.8% vs 1526 |
| (8,8,1) bs256 (default), ntl1 G1 / G2 | 8 | 1581 / 1571 | 103 / 99 | 1684 / 1670 | +9.4..10.4% |

Caveats: MOE_TIMING builds perturb codegen (P14 used the same tool); step1_timing's per-(phase, ntl) split divides by 16
stages per item, so it reads half for the (4,*,1) configs' 8-stage items (their rows use warp 0's median) and a quarter
for the 4-stage k-split pieces (the (8,4,1) G2 ntl1 numbers at bs32 / bs128, 387 / 446, left out). (P14 has (4,2,1) at 1519 + 90 on bs16: +3.5% comp.) The stage function is the same template (stage_compute<NT, 1, S6>
with the same S6O 6-bit decode in every config: MOE_S6X makes S6 true for KW == 1 && NT >= 4), so the difference is in
how the NT>1 kernels run it: their 1-group items use the top-tested loop (MOE_RUN_ITEM: only NT == 1 gets the
bottom-tested form), a heavier producer on SMSP0 (warp 0 is the measured warp), a bigger kernel (codegen), and in the
one-k-warp configs a 600 / 475 / 750-cycle staged epilogue per item (bs32 / bs128 / bs256) where (4,*,1) has SEPI.
The penalty grows with NT (+3.5% at NT 2, +7.5% at 4, +8.6% at 8). MOE_BOT1 (bottom-tested 1-group loop inside an
NT>1 kernel) lost 1.4-1.7% twice, so it is not the loop form alone: it is the 1-group loop living inside an NT>1 kernel.
(8,1,1) does not compile on v567 (NVRTC error), so a lean one-k-warp NT=1 binary needs work.

Share of each shape's consumer cycles in 1-group stages (stage mix from WP-E, per-ntl costs above): bs32 ~89%, bs128
~64%, bs256 ~39%. At (4,1,1)'s cost per stage that is -13% / -5.7% / -3.7% of consumer cycles before the epilogue
saving.

### Q3. bs16 / bs32 on the (4,1,1) binary end to end (dev/wpq_b411.sh, bench2 jm 100 reps, 2 copies each, one job)

| variant (shared weights) | bs8 (placebo) | bs16 | bs32 |
|---|---|---|---|
| a v567 (bs16 (4,2,1), bs32 (8,4,1)) | 65.4 us, x2.243 | 85.3 us, x2.214 | 116.6 us, x1.809 |
| b v567 copy | x0.9967 vs a | x0.9988 | x0.9996 |
| c BUCKETS bs9-32 -> (4,1,1) | x1.0010 | **x0.9726** | **x0.9575** |
| d copy of c | x0.9997 | x0.9823 | x0.9576 |

Strata (speedup over SGLang, a vs c): bs32 e<=15 1.811 vs 1.676 (-7.5%), e>17 1.803 vs 1.767 (-2%), hot>19 1.821 vs
1.700; bs16 hot>9 2.264 vs 2.149 (-5%). The bucket change is dead: experts with > 8 tokens become 2-3 tiles that each
decode the expert again (stages x1.14 on the jm mix, more on light / hot reps), and the item count doubles (8-stage
items).

### Q4. The lean binary wins on 1-group work alone (dev/wpq_cap8.py, bs32 jm seeds with every local expert capped at 8 tokens, 80 reps, 2 copies each)

The cap moves each over-full expert's excess pairs onto the same call's other active experts with the fewest tokens
(expert count 16.40, pairs 58.6, max 8 tokens: the same experts and bytes as the judge mix, every tile one 8-token
group in both binaries, no duplicated decode). check_implementation passes on the capped inputs.

| variant | us | vs a841 |
|---|---|---|
| a841 v567 (8,4,1) | 116.3 | 1 |
| b841 copy | 116.3 | x0.9981 |
| c411 (4,1,1) | 108.2 | **x1.0686** |
| d411 copy | 107.9 | **x1.0721** |

So the 1-group stage advantage of Q2 is real in wall time: the same 1-group work is ~7% faster on the NT=1 binary,
and (8,4,1) takes the same 116 us whether the call's pairs sit in a few multi-group tiles or are spread over 1-group
tiles (uncapped 116.6 us). This is the prize lever #1 captures without duplicating any decode.

## 2. The levers in detail

### #1 Lean NT=1 binary for 1-group tiles

Mechanism: Q2. The (4,1,1) binary (moe_4_1_1_NS_1) already serves every M <= 32 (MW = 1), so moving bs9-32 onto it is a
bucket-table change: no new code. Its costs at bs32: experts with > 8 tokens become 2-3 tiles that each stream (mostly
from L2, they sit next to each other in the rank order) and decode the expert's weights again (stages x1.14 on the jm
mix: 0.913 + 2 x 0.039 + 3 x 0.044 + 4 x 0.004), and raw x rows (one bulk copy per token row per stage, ~2-3 rows per
1-group tile at bs32). Back of envelope per (8,4,1)-equivalent stage on the jm mix: (8,4,1) 0.913 x 1757 + 0.039 x 2116
+ 0.044 x 2377 + 0.004 x 2350 = 1800 cycles; (4,1,1) 1.139 x 1526 = 1738 (-3.4%), plus no 600-cycle staged epilogue
per item (SEPI), i.e. about -5% of consumer work at C/M ~0.85-0.9.
Differs from: v603 ((8,2,1), -1.2%), v606 / "(4,2,1) at bs32" (-0.4..-3.7%): both are NT>1 binaries whose 1-group loop
carries the penalty in Q2; WP-P pools (static split, wave quantization); MOE_BOT1 (loop form inside an NT>1 kernel).
Measured (Q3): the plain bucket change loses (bs32 x0.958, bs16 x0.973-0.982) because the split tiles duplicate the
decode; on 1-group-only work (Q4) the same binary is x1.07 faster. So the version that pays keeps every multi-group tile
on its NT>1 binary and moves only the 1-group tiles.

PDL two-kernel hybrid (bs32, bs128, bs256; bs16 optional):
- Graph = two kernel nodes on one stream, the second launched with cuLaunchKernelEx and
  CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION (cuda.bindings.driver; verify in the Modal image). Kernel A = the
  existing (8,4,1) / (8,8,1) binary restricted to the tiles with >= 2 token groups (a runtime filter where the queue is
  built: outside the hot loops), KSX = 1 (its tiles run first, never in the tail), no combine; it executes
  griddepcontrol.launch_dependents at entry. Only its first n_G1(A) CTAs stay (the others exit at once): otherwise CTAs
  would claim A's down items at start and spin on their G1 dependency for a whole item while B could use the SM. Kernel B = the (4,1,1) binary (MW 4 / 8 builds for bs128 / 256) restricted
  to the 1-group tiles, plus the combine for every pair after a done-counter that covers A's down items.
- No static partition: A's CTAs with no work exit at once and B's CTAs take those SMs; the rest of B's CTAs land as A's
  CTAs exit (heavy-first order = LPT across the two kernels). No duplicated decode (each tile runs once), no code
  presence (each binary keeps its own codegen), no cross-kernel dependency except the combine (A's tiles are other
  experts than B's).
- Costs: B's routing / cold start on SMs that free up late (~2-3 us of one SM each, while the others stream), one shared
  y / sum-of-squares format for the combine (A must write the (4,1,1) fast-combine format, or B's combine reads both),
  one more node per graph (PDL hides its launch).
- Expected: bs32 +4..5% (Q4: x1.07 on the 1-group work, which is ~89% of bs32's consumer cycles; minus ~1% for B's late
  starts and the combine hand-off), bs128 +2..4% (1-group stages 64% of consumer cycles at -9.8% each; B needs the MW 4
  (4,1,1) build, which has no SEPI), bs256 +1..3% (39% share, compute-bound at 1.68 GHz). bs16 optionally the same with
  (4,2,1) as A (its 1-group penalty is +3.5%: ~+1..2%). Geomean ~+1.2..2%.
- Risk: medium-high engineering (two builds share routing, ctr layout and the combine); PDL CTA-placement latency
  unmeasured. Order of work: (1) a 10-minute PDL check: cuLaunchKernelEx with the programmatic attribute under
  torch.cuda.graph capture in the Modal image, and the gap from a primary CTA's exit to a secondary CTA's first
  instruction on that SM (globaltimer stamps); (2) the bs32 prototype: A = the (8,4,1) binary with the 1-group tiles
  filtered out of its queue, B = the (4,1,1) binary with the multi-group tiles filtered out, combine in B; check_mt
  M 17..32 and bench2 jm against v567 (Q4 says the ceiling is ~+6% at bs32); (3) MW 4 / 8 B builds for bs128 / 256.
- Rules: no change of precision, layout or routing; the split is by the call's own token counts (shape-generic, so the
  nearby-shape reruns see the same code path).

### #2 Gate/up stages with 2-4 token groups wait for data: DIAGNOSED (2026-10-27, planner request)

Tools: dev/moe_v640.py (= v567 + timing-only MOE_WQ consumer counters / MOE_WQP producer counters), v643 (+ MOE_WQE:
the async-epilogue warps' wake -> slot-release time per epilogue ordinal), dev/wpq_wait.py (judge protocol, 6 jm seeds
x 3 calls per shape), jobs dev/wpq_wait.sh, wpq_wait2-5.sh (logs scratchpad wpq_wait*_run1.log).

Not steady state, not the x path, not cold code:
- Waits at stage index >= 4 are 80-125 cycles everywhere. The excess sits at stage NS-1 of the item that follows a
  large-tile item: bs256 (ring NS = 3: the 68 KB (8,8,1) slot reserves x room for 64 tokens) 5000-6600 cycles per item
  on each CTA's 2nd item, 1200-2300 on its 3rd; bs128 (NS = 4) 2300-4300 on the 2nd. Plus the first item's top wait
  (startup). The earlier per-(phase, ntl) table showed it on 2-4-group G1 items only because those are the second wave
  in LPT order.
- MOE_NO_X: unchanged (not the x copies). No judge flush: unchanged (not L2-cold code). Odd CTAs started 2 us later
  (v642): unchanged (not cross-CTA synchronization).
- MOE_WQE: the async epilogue's wake -> slot-release time for epilogue 0/1/2/3+ of a CTA is 8941 / 4856 / 2794 / 1387
  cycles at bs256 and 6790 / 2603 / 1378 / 1549 at bs128; the producer's slot wait per stage on 2nd items is 1000-1350
  cycles vs 520-750 later. The item's last ring slot holds the staged accumulators until the 2 epilogue warps finish
  SwiGLU + stores, and the next item's stage NS-1 needs that slot. The first items are the largest tiles (LPT order),
  whose epilogue (up to 64 tokens x 128 values on 2 warps that share SMSPs 2/3 with 4 busy consumer warps) takes ~5 us.
  Dry warm-ups through a separate copy (v641) or the loop's own call site (v644) shorten epilogue 0 by only 9-20%.
- Correction: c_wait does include each item's first-stage wait (taken at the item top); step1_timing's per-(phase, ntl)
  split divides by 16 stages per item (it reads low for 8-stage (4,*,1) items and 4-stage k-split pieces).

Bound: bs256 ~3 us per CTA after the first item + ~1 us after the second (~2%), bs128 ~1.4 us (~1%).

Fix attempts (all bench2 jm 100 reps vs v567 x2, check_mt 0 failures, rejected):

| build | bs32 | bs128 | bs256 |
|---|---|---|---|
| v641 MOE_AEPI_WARM (3 dry inlined epi_stores calls at entry) | 0.9765 / 0.9743 | 0.9841 / 0.9821 | 0.9851 / 0.9901 |
| v642 MOE_SDELAY_ODD 1500 / 3000 ns | - | 1.0007 / 0.9989 | 0.9965 / 0.9998 |
| v644 MOE_AEPI_DRY=3 (dry loop iterations, same call site) | 0.9797 / 0.9837 | 0.9845 / 0.9874 | 0.9875 / 0.9865 |
| controls (v567 b) | 0.9985 / 0.9974 | 0.9990 / 0.9981 | 0.9964 / 0.9981 |

What would fix it (not built): shorten the large-tile epilogue or give the next item more slack.
(a) The signaler warp (9, idle ~5 us between signals) joins epi_stores for tiles with >= 4 token groups (3 warps: -33%;
the slot's 8 release arrivals split 3/3/2), helper-warp code only. (b) NS = 4 at bs256: move the wgmma x (up to 33 KB)
out of the weight slot into its own 2-slot x ring, so the held slot gates stage 3 instead of 2 (+1 stage of slack, and
the general NS 3 -> 4 gain); structural. (c) The consumers run the epilogue themselves for the largest tiles only
(4x the threads; AEPI stays for the rest): a runtime branch in the consumer epilogue, codegen risk. Expected at most
the bound above (~2% bs256, ~1% bs128).

### #3 Freeze the hot loops' codegen

Every added code path has cost 1-5% at some shape with the hot loops' instruction counts unchanged (XR_NONE bisection:
the loops are only re-scheduled, opcode-order similarity 0.4-0.7; HSPL / HSPL2 -5..-8% with no halves; SPL -1.5%;
SEPI8 -0.7..-2.1%; KSX8=2 -1.8% from code alone). ncu rules out the i-cache. The common factor is ptxas scheduling the
inlined stage loops differently when the rest of the kernel changes. A `__noinline__` device function per (NT, NTL)
that runs a whole item (all stages: waits, compute, arrives, the accumulators to the staging buffer) is register-
allocated and scheduled on its own, so its SASS does not depend on the rest of the kernel; the ABI cost (callee-saved
registers, a few dozen instructions) is paid once per 8-16-stage item (< 0.5%). Risks: setmaxnreg (the callee must be
allocated within the consumer budget; ptxas may assume the launch's 168), and the call boundary may cost the
compiler's 2x cross-stage unroll at the item edges. If it holds, the measured-but-eaten gains come back: XR +1.6..1.9%
at bs128/256, KSX8=2 +0.9% at bs256, half-item tails (TAILB +4.9% / +3.7% bound at bs8/16), and the coordinator's
hybrid could become a single kernel. Probe: v567 with MOE_RUN_ITEM(1)'s loop in a noinline function for the (4,1,1)
config; check (a) its SASS is byte-identical with and without MOE_XR-style unused code, (b) bench2 within +-0.3% of v567.

### #4 bs128 split-partial path

WP-L measured the partial path (FP32 ypart stores + the combine's 4-partial reads at the very end) at 1.65-1.76%
(bs128, KSFREE timing bound) and 1.3% at bs32 (2-way). Instead of the combine reading 4 FP32 partials per row at the
end: each split piece does atom.add.acq_rel on its row block's counter after its partial store; the last one reads the
other pieces' partials (L2, ~1 us, off the critical path except for the very last block), adds its own registers,
writes BF16 y + the 64-row sums of squares, so the combine sees whole rows only. Expected +0.8..1.5% bs128,
+0.5..1% bs32 (if bs32 stays on (8,4,1)). Differs from v189/v190 (fixups on the consumers / signaler with global
fences: -2..-7%): no fence on the consumer path, one atomic per piece. Risk: the last piece's extra epilogue (~1 us) on
the last wave.

### #5 bs32 start without the gather barrier

First data at bs32 is ~7.2 us vs 4.3-4.6 at bs8/16; the lag is the all-CTA gather barrier whose writers' red.release
waits behind the SM's in-flight TMA loads (P9). P9's fix (the first item per CTA fed by direct x-row copies, mma.sync
tiles only) was blocked by MOE_PS's FP16 gathered x, which the BF16 rule removed. Differs from MOE_E32 / E32X (weights
or rows during routing, -1.5..-3.4%: they moved the routing barrier) and GATHER_MIN=2. Expected +1..2% at bs32 without #1; with #1 only kernel A's
first items wait on the gather (B's 1-group tiles read raw x rows), so most of it is captured by #1 already.

### #6 PDL tail kernel (bs8/16)

The main kernel stops its queue R = items mod 132 items early; a second kernel (PDL, same graph) runs those items as
(1,*,1)-geometry row items (32 rows x 4096 k, 2 stages of 32 x 2048, 8-way k reduction in shared memory: full K, so no
partial sums or fences) on SMs as the main kernel's CTAs exit. Keeps the main kernel's code untouched (the reason
HSPL / SPL lost). TAILB's measured bound is +4.9% bs8 / +3.7% bs16; the second kernel's CTA placement + routing
(or reading the first kernel's tables) + cold code is ~2-4 us at the very end, i.e. most of the 3-4 us it saves.
Expected 0..2%. Only worth it if a PDL CTA starts within ~1 us of the SM freeing (measure first) and its code can be
made warm.

### #7 2 CTAs per SM

Chord's H200 W4A16 MoE closed its 8-tokens-per-expert gap only with 2 resident CTAs per SM. Here: 2 x (4 consumer
warps, 64 rows each = 4 row groups, B fragments reused 4x; producer + helpers; ~113 KB: NS 3 x 35 KB) per SM. Per-item
barriers and epilogues of one CTA overlap the other's stages; two producers halve the per-producer copy rate. Against:
NS 3 per CTA (-4.5% measured at bs8/16 for one CTA, though each CTA's stage takes 2x as long to consume here), two
producers on SMSP0/SMSP1, setmaxnreg budgets for 2 x 192 threads. Not tried; template change (RW x KW = 4).
Expected 0..3%, cost high.

## 3. The coordinator's question: two kernels per tile class

- Concurrent kernels in parallel graph branches with fixed grids: no. The per-call share of multi-group work varies a
  lot at bs32 (0-3 multi-token experts per routing; in seed 73736 two of ten tiles carry 27% of the consumer cycles),
  so any fixed SM split quantizes per pool; WP-P measured exactly this with two pools in one kernel (-8% at bs8/16
  although every item got faster). Work stealing across kernels needs each binary to run the other class (the code
  comes back) or the multi-group tiles split into 1-group tiles (the x1.14 duplicated decode).
- The version that pays is sequential with PDL (#1): heavy class first on the one-k-warp binary, the 1-group class on
  the lean binary as SMs free up, combine in the second kernel. No static partition, no duplicated decode, no code
  presence, no cross-kernel G1 -> G2 dependency (the classes are disjoint experts). The zero-code variant (move all of
  bs16 / bs32 to the lean binary) was measured and loses (Q3: x0.958 at bs32) because multi-token experts split into
  duplicated tiles; the lean binary's advantage on 1-group work alone was measured at x1.07 (Q4). For bs128 the prize
  is smaller (1-group stages are 64% of consumer cycles, -9.8% each) but real: ~+2..4%.

## 4. Considered and not recommended

- Energy at bs256: in judge mode the in-kernel clock is 1.68 GHz (Q2) vs the 1.79 tensor cap. WP-E's decomposition
  leaves no sink beyond the activation copies (XR already built and eaten by codegen). Cycle cuts (#1, #2) are the
  energy lever: same instructions, fewer stall cycles.
- Tile-shape changes to cut decode/MMA issue: the 1-group (4,1,1) stage is at its isolated floor (1470 + 85 vs WP-D's
  1615 with the ring boundary in isolation); HMMA count is fixed by rows x k x tokens; post-scale in BF16 needs FFMA
  (+12.5%); decode sharing across tiles is bounded at +1.6% (SKIPB); larger N per tile only helps where tensor-bound
  (ntl8, 12% of bs256 stages).
- Faster stream: the ring is at 95% of 4.8 TB/s; placement does not matter (Q1).
- Cold i-cache warming via code prefetch: code addresses are not readable (v203 note); dry passes were measured (WARM,
  WWARM, CWARM).
