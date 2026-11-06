# WP-L findings (bs128 / bs256: startup, tail, coupling)

Worker L, started 2026-10-26. Base: dev/moe_v200.py (== submission.py). Files: dev/moe_v30x.py, dev/wpl_*.

## Tools written
- dev/wpl_trace.py: raw per-CTA trace dump (MOE_TRACE + MOE_TRN) of one cold call per (bs, seed) + routing counts.
- dev/wpl_ana.py: offline analysis of the dumps (startup events, per-CTA compute / epilogue / gaps, tail, per-(phase, ntl) stage times).
- dev/wpl_routes.py: per-seed local-expert counts of the judge-matched seeds.

## Log
- run1 (v200, Modal H200, 6 jm seeds/shape traced, 60-rep bench): bs128 175.6 us (control spread 0.24%), bs256 213.3 us
  (0.04%); NO_COMPUTE 161.4 / 182.0 us. NOTE: trace builds give wrong outputs (rel-L2 ~1: the static __shared__ trace
  counter shifts the dynamic smem base off 1 KB alignment -> swizzled wgmma operands misread); timing still representative.
  bs256 per-CTA budget (traces, cold L2): first real item med 13.3-14.5 us (gated by the cold warm-up stage: warm-up
  starts after the gather at ~7.4 us and takes 6.5-6.8 us cold vs 2.5 us with L2 warm; the first item's stages are
  already in smem at ~10.9 us), compute 148-205, epilogue 4.1-6.0 (0.51-0.7 us/item), gaps 2.0-3.2 (0.32 us/item),
  TAIL IDLE mean 8.4-12.8 us (last items are G2 ntl1, 16.8 us each), end after last item 2.5-5.2 us.
  bs128: first item med 8.7-13.0 (CTAs whose first item is a wgmma tile wait for a 6 us cold warm-up, mma.sync ones 2.4 us),
  epilogue 3.6-4.8, gaps 4.3-6.5 (mostly G2n1->G2n1 transitions of the 4-stage k-split tail items, 0.56-0.86 us each),
  tail idle 2.2-3.7, end ~4-5 us.
  Producer trace (bs256 CTA 0): the first item's 3 stages were issued by 10.94 us (weights of stage 0 at 5.3, x after the
  gather barrier at 9.09); consumers start at 14.69 after the warm-up. Item transition on the producer ~1.5 us
  (claim 0.5, decode 0.3, slot 0.4, issue 0.5: the starved producer warp), stage issue 0.4-0.6 us each.
- Model: bs256 per-CTA compute at in-kernel stage costs = 170 us mean over the 100 jm seeds vs HBM 161 us: with only
  scheduling fixes the bs256 floor is ~178 us (Modal 213 now, ~-16% max); -10% needs compute reduction as well.
- run2 (diagnostic, bs256, 60 reps): MOE_KSX8=2 tail split -0.88%; with SPLW 0 (code only, no split) -1.82% ->
  the split itself is +0.9% (tail idle 3.5-6.1 us), the KSX!=KS2 build costs 1.8% (codegen). Next: why (SASS).
- run3 (planner request): MOE_NO_COMPUTE per-CTA timelines (6 jm seeds/shape). bs256 data-only: first item 9.4-10.8 us
  (gather barrier ~9 + x copy; no cold-code wait), per CTA data-wait "compute" 113-160, epilogue 3.2-4.6 (0.45 us/item),
  gaps 6.8-10.0 (0.7-0.9 us/item: the starved producer's item transition), tail idle 4.0-7.2, end 2.4-3.3 us. With the
  bare ring at ~155 us (planner) the 27 us data-path excess = startup ~8 + item boundaries ~11 + tail ~6 + end ~3.
  bs128 data-only: first item 7.8-9.5, gaps 14.5-21.8 (!) per CTA: the 4-stage k-split tail items (KSX 4) cost ~1 us of
  transition each in data-only mode (0.95 us in full mode: epilogue 0.35 + gap 0.58 per 4.2 us item).
- Zero data (planner/WP-R): hidden_states = 0 -> bs256 +2.08%, bs128 +1.48% (all tensor B operands zero). Padded
  columns are 20% / 34% of the tensor columns -> zeroing them is worth <= ~0.4-0.5%; they currently read stale smem
  (mma.sync tiles: whatever the slot held, often fp32 staging) or stale xs/inter rows (wgmma tiles). Parked (small).
- SASS (Modal NVRTC 13.0; local nvcc 12.1 does NOT reproduce it): v200 bs256 wgmma stage loops have 604 (n32, 10 UMOV,
  64-bit UIADD3) vs 632-642 instructions (n24/n40/n48/n56/n64: 26-30 UMOV + UIADD3/UIADD3.X pairs); with MOE_KSX8=2
  the n24..n64 loops grow to 652-658 (46 UMOV: descriptor copies UR12:13 -> UR18:19 before every HGMMA). The bs128
  kernel's n32 loop has 46 UMOV (654 instr) vs its n24 loop 10 (598). => the -1.8% of the KSX8=2 build is ptxas
  uniform-register allocation (likely tsplit/psplit kept live in uniform registers through the stage loops).
  The descriptor formulation is a codegen lottery worth ~16-50 instructions per wgmma stage (2.5-8%).
- run4: v300 MOE_GCP (gather on warps 9-11 with cp.async into the ring's free x regions; consumers warm up right after
  the pair list): correct; bs256 first real item 14.5 -> 11.5 us (warm-up 3.8 -> ~10.5 overlaps the gather) but CTA end
  times unchanged and bench -0.07/+0.01% (2 copies); bs128 -0.4%, bs32 -0.7% (the 96-thread gather ends 1.5-2 us later
  and gates the first item). REJECTED. Reading: at bs256 (in-kernel clock 1.68-1.70 GHz, power-capped) idle startup
  time is largely absorbed (same as MOE_SDELAY 5 us costing 2.5 us): bs128/256 levers are energy/instructions, the tail
  (+0.9% measured), not startup idle.
- run4/6 SASS: KSCLEAN alone has 46 UMOV/stage (v200 26-30) -> REJECTED. Descriptor formulations (stage_compute_wgh):
  v200 bs256 [n32 604, n24 636, n56 634, n64 642, n40 632, n48 634], bs128 [n24 598, n32 654];
  WDF=2 (lo = (xaddr>>4)|1<<16 per stage + immediates, mov.b64 with const hi): bs256 618 everywhere (n64 624), bs128 612/612;
  WDF=3 (u64 base + u64 immediates): the FIRST loop of each kernel 582-588 (4 UMOV), the others 622-654;
  WDF=6 (4 hoisted kk descriptors) = v200; WD32 = 618-630. ptxas gives the first-emitted wgmma loop the best allocation.
- run5 (MOE_TIMING, per-warp stage-compute cycles / CTA mean): warps 0/4 (SMSP0 + producer) +2.0..3.8%, warps 1/5
  (signaler SMSP) -1.4..-2.1%, others ~0: the producer steals ~4-5% of SMSP0's issue; the consumers' per-item barrier then
  makes every warp wait for warps 0/4 (max/mean 1.028-1.039). Producer SASS: ~140 instructions per G1 wgmma stage, ~60
  of them the 4 x-chunk copies (ELECT + 5 R2UR per copy, one lane at a time).
- run7: descriptor formulations measured: WDF 2/7/8 within +-0.35% of v200 on all three shapes (60 reps; WDF8 +0.35% at
  bs128/256 in one copy, inside the 0.1-0.4% control spread). The 16-56 extra uniform instructions per wgmma stage are
  ~free (uniform datapath). So the -1.8% of the KSX8=2 (SPLW 0) build is NOT the UMOVs; remaining suspect: code layout /
  i-cache (the bs256 kernel is ~11k instructions = 180 KB; GCP+KSCLEAN+KSX8=2 measured -0.13% while KSCLEAN+KSX8=2
  measured -1.0%: +-1% swings from unrelated code changes). Promotion needs > 1% or two separate jobs.
- run8: producer diet (MOE_PX1 one x copy per wgmma stage, MOE_PWSC, MOE_WSC_NT=4): producer stage loop 274 -> 249
  instr, but bench 0 to -0.6% (2 copies) -> REJECTED. MOE_SKIPB upper bound (odd tiles of multi-tile experts skip all
  compute, timing only): bs256 +1.64%, bs128 +0.75%, bs32 -0.6% -> sharing the decode between an expert's tiles
  (MOE_PAIR) can win at most ~1% at bs256. KSX8=2 on two more builds: -0.35% / -0.7% (the split's +0.9% keeps
  being eaten by a build-dependent loss).
  bs128 tail (MOE_TRT trace, 6 seeds, 576 4-stage split items each): producer decode -> slot 0.32 us, slot -> weights
  issued 0.22, weights -> x issued 0.55 (the cnt1 acquire poll), x -> consumer start 0.35-0.4; the producer decodes
  the next split item only ~0.4 us before the consumers finish the current one, so the consumers wait 1.0-1.06 us
  (median) per split item (4.2-4.35 us compute, 0.35 epilogue). Suspect: the producer cannot issue an item's last
  stage until the AEPI warps release the previous item's held slot, and those can be held up by the signaler, whose
  red.release takes ~4.4 us under streaming (> the 4.2 us split item). Next: signaler trace (MOE_TRS) and a batched
  signaler (one fence.acq_rel + relaxed reds for every pending signal).
- run9: v301 MOE_PAIR (tile pairs on 128-k half stages; one decode for two tiles; new lossless 18 KB half-block layout):
  correct (check_mt 0 fail), pair loop 384 instr per half stage (two tiles: 630), but bench bs32 -2.1%, bs128 -0.4%,
  bs256 -0.25% (spills 24/92 B, two held slots per pair item). REJECTED (the SKIPB bound was only +1.6%).
  Signaler trace (MOE_TRS): red.release ~1.0-1.2 us median, 0.5 us in the last 25 us, and ~5 us idle between signals
  at the bs128 tail -> not the bottleneck. Tail timeline (bs128): the producer decodes split item k+1 exactly when the
  consumers finish computing item k: the producer warp gets almost no issue slots while the two consumer warps of
  SMSP0 compute; per stage it then needs slot->weights 0.22 + weights->x 0.55 (the cnt1 acquire poll) + x->data 0.35 us.
  Next: MOE_TRDY (skip the poll for tiles already seen ready; 96-bit mask in shared memory, no producer registers).
- run10: smoke hang at bs8 (v301 bug: the queue's unit count was read before the early-start configs' routing tables;
  fixed). New rule (user): BF16 only (no MOE_F16 / F16_NT4) -> new base v601; WP-L work moved to dev/moe_v302.py
  (= v301 minus the two defines; all WP-L switches behind #ifdef). run11 measures TRDY / SIGB / PNEXT / KSX8=2 / GCP on it.
- run11 (v601 base, 80 reps): v302 without defines is bitwise = v601. MOE_TRDY -0.2..-0.5% (tail trace: weights->x
  still 0.61 us: the poll was not the cost), +SIGB -0.5..-1.6%, +PNEXT -0.5..-1.2%, KSX8=2 neutral (-0.15% bs256),
  MOE_GCP: bs256 +0.89% (one copy), bs128 -1.0%, bs32 -1.1% -> next: GCP for the 64-token config only (MOE_GCP_NT=8).
  Producer starvation reading: ~70 producer instructions between an item's last copy and the next item's decode take
  1.4-1.5 us while the consumers compute, and run fast as soon as they reach the epilogue barrier -> the scheduler
  favours the consumer warps; MOE_WREMAP (helpers on physical warps 0-3, consumers 4-11, same SMSPs) tests whether warp
  age/ID decides it.

## STATUS (2026-10-27, for the planner)
- No WP-L switch has a verified >= 1% gain yet. Best so far: MOE_GCP + MOE_GCP_NT=8 (bs256 only): v601 base +0.24..+0.36%
  (3 copies, 100 reps; one earlier copy +0.89%), bs128 untouched. Ported in dev/moe_v303.py (= v365 + switches).
- run13 (in flight, base v365): MOE_WREMAP x2, GCP_NT=8 x2, WREMAP+GCP8, plus bs128 tail-piece sizes (SPLW 0.5, KSX1=2),
  WREMAP per-warp skew (MOE_TIMING) and tail trace.
- Measured dead ends (details above): GCP at bs32/128, KSCLEAN, WDF (descriptor formulations), PX1/PWSC/WSC_NT=4
  (producer diet), PAIR (tile pairs, one decode), TRDY, SIGB, PNEXT, KSX8=2 (bs256 tail split; +0.9% intrinsic on FP16,
  neutral on BF16). v601 SASS: no C7511/C7513; descriptor UMOVs measured free.
- run12 (v601 base, 100 reps): MOE_GCP_NT=8 bs256 +0.25/+0.24/+0.36% (3 copies; spread 0.38%), bs128 untouched;
  GCP everywhere bs256 +0.46% / bs128 -1.1%. v601 SASS: no C7511/C7513; BF16 wgmma loops 571 (first) / 621 instr
  (48 UMOV) at bs256, 565 / 615 at bs128 (same descriptor-copy pattern as FP16, measured free in run7).
- run13 (v365 base, 80 reps): MOE_WREMAP REJECTED (bs128 -0.5%, bs256 -0.8%, per-warp skew unchanged: warp ids do not
  set scheduler priority). GCP_NT=8 bs256 +0.34/+0.37% (spread 0.21%). bs128 tail pieces SPLW 0.5 / KSX1=2: +0.06 /
  +0.13% (noise). WREMAP+GCP8 bs256 +0.56% (one copy).
- run14 (in flight): GCP_NT=8 on the integration base v496 (dev/moe_v304.py), 3 copies, 100 reps, bs32/128/256.
- run14 (v496 base, 100 reps, 3 copies): MOE_GCP + MOE_GCP_NT=8 bs256 +0.15/+0.31/+0.52% (spread 0.25%); bs128/bs32
  kernels identical (noise -0.1..-0.4%); bitwise = v496 (bs 8-257), check_mt 0 failures. Over 8 copies in 3 jobs on
  3 bases: +0.32% mean at bs256. KEPT as a small item (dev/moe_v304.py, defines MOE_GCP,MOE_GCP_NT=8).
- run15 (in flight): planner's light-first start at bs256 (MOE_RANK_SF=8: first wave from the smallest experts, 1-2
  group mma.sync tiles with a 2.4 us instead of 6 us cold warm-up), +GCP8, and KSX8=2 (SPLW 1 / 0.5) on v496.
- run15 (v496, bs256, 100 reps): planner's light-first start MOE_RANK_SF=8 -0.93/-1.03% (first item 14.5 -> 10.1 us but
  heavy items later, tail idle up to 15 us), +GCP8 -0.42%; KSX8=2 -0.94% (SPLW 0.5: -1.35%). All REJECTED.
- MODEL (dev/wpl_sched_sim.py, planner's static schedule): a static LPT per-CTA plan equals the dynamic queue at the
  same per-item overhead (the queue is already LPT; the bs256 tail is item granularity) and loses under realistic
  duration noise (+0.5% at 4%, +2.8% at 8%). What pays in the model, inside the dynamic queue: per-item overhead
  0.85 -> 0.4 us (bs256 -2.0%, bs128 -3.1%) and ~66 row-split tail halves at bs256 (-1.3%; -2.5% with 0.6 us overhead).
  The per-item overhead is mostly the consumers' end-of-item barrier waiting for warps 0/4 (SMSP0 + producer, +2-4%)
  plus the starved producer's ~1.4 us item transition; the cheap fixes (PX1/PWSC, TRDY, PNEXT, SIGB, WREMAP) did not
  move it -> it needs producer work moved off SMSP0 (x copies + dependency polls on warp 9) or a per-warp barrier-free
  staging (as WP-S's SEPI for bs <= 16).

## Item 1 (planner, base v610): producer side of the per-item overhead
- dev/moe_v305.py = v610 + MOE_XH: for (8,4,1)/(8,8,1) warp 9 (SMSP1) issues every stage's activation copy and does
  the down items' dependency polls (and the one gather-barrier wait), interleaved with its signals (non-blocking
  try_wait on both; every wait loop also services signals, so a dependency wait cannot hold back this CTA's own
  releases); the producer only waits for slots, writes records, posts expected bytes, issues weights and hands the
  slot over (xhq[NS] mbarriers past the ring, +128 B smem). Producer stage loop 274 -> 155/139 static instructions.
- First build hung at (8,8,1): the warm-up stage's record raced (fixed, see candidates). run16 re-running.
- run16: MOE_XH (fixed twice: warm-up record race, slot-0 parity) correct (bitwise = v610, check_mt 0 fail) but -5% on
  bs32/128/256 (2 copies each). Per-warp skew moved to SMSP1 (warps 1/5 +2.9..4.2%, warps 0/4 ~0..+1.4%) and the data
  wait per stage rose 150 -> 245-266 cycles: the x warp's polling loop (try_wait + shuffles every ~200 ns) and the
  hand-off latency cost more than the moved work. So the producer's instructions DO slow SMSP0's consumers (the skew
  follows the work), but splitting it this way does not pay. run17: 2000 ns suspend variant, and RSPL (item 2).
- run17: MOE_XH with a 2000 ns suspend: -4.7% bs128 / -5.3% bs256 (as bad as 200 ns) -> item 1 REJECTED.
  MOE_RSPL (item 2, dev/moe_v306.py): bitwise = v610, check_mt 0 fail, but bs256 -2.5% (f 0.5) / -3.3% (f 1.0).
  Trace: a 128-row half takes 13.6-14.2 us vs 15.7-16.6 us for a whole item (85%): a tail stage costs ~0.85 us
  regardless of its bytes (per-stage latency/overhead floor), so row halves don't shorten the tail; only fewer stages per
  piece (k split) would, and that path's partial sums / codegen cost measured -0.9% on v496. REJECTED.
- run18 (planner's last task, bound first): MOE_KSFREE (timing only) makes k-split pieces store and combine as full-k.
  bs256: KSX8=2 real -1.1%, with free partials +0.43/+0.47% -> below the 1.5% bar, the 2-CTA cluster/DSMEM version is
  NOT built (stop rule). bs128 (its existing KSX1=4 tail split): free partials +1.65..+1.76% -> the partial-sum path
  (fp32 ypart stores + the combine's 4-partial reads at the very end) costs bs128 ~1.7%; a cheaper partial path there
  (e.g. DSMEM-reduced 4-way pieces, or the last piece of a split row block summing the others into BF16 y while the
  combine waits) is the one remaining bs128 lever with a measured bound.
- run19/20 (bs128 cluster k-split, bound first): free partials with 2-way pieces bs128 +1.33%, bs32 +1.27%; 4-way
  +1.82% / +1.91% (but 4-CTA clusters cannot place 33 co-resident clusters on 132 SMs: 9-TPC GPCs leave 2 SMs);
  KSX1=2 with real partials -0.05%. Decomposition (4-way): store-side alone -0.2%, combine-side alone +0.6%, both +1.6%.
  Pairing model (dev/wpl_pair_sim.py): both halves of a split item starting together on a 2-CTA cluster costs ~0.4%
  vs independent pieces -> expected real net ~+0.5..+0.9% before the DSMEM handshake / receive buffer (8 KB of free
  smem: tiles <= 8 tokens only) / code-layout cost; the global-memory fixups (v189, v190) lost 2-7%. NOT BUILT (at or
  below the planner's +0.8% bar); reported with the numbers.

## CLOSED (planner, 2026-10-27)
WP-L closed by the planner after the bs128 cluster k-split bound. Kept: MOE_GCP + MOE_GCP_NT=8 (in v610, ~+0.3% bs256).
Everything else measured and rejected (see candidates.jsonl). Key facts for later work: the tail's per-stage floor
(~0.85 us whatever the bytes), the scheduler does not favour warps by id, a static schedule equals the dynamic queue,
and code presence / layout moves results by up to +-1%.
