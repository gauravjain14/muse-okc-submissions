# Fresh-eyes review, round 2: structural ideas not yet measured (2026-10-28)

I wrote this without running anything on a GPU. I read TASK.md, AGENTS.md, docs/ROUND2.md, docs/PLAN.md (rulings, final
sections), docs/research_r2.md, dev/x{a..g}/NOTES.md and dev/coord/SNAPSHOTS.md. Every "measured" number below is taken
from those notes and names its source. Every gain is my estimate, and each estimate has a probe that can falsify it.

## 0. Bottom line

- Score ~x2.85 on the judge-equivalent times in the brief. Getting to #2 needs about +6.2%, and #1 about +6.9%.
- Sensitivity (% score per us saved, at the current judge times): d1s 2.14, v1s 1.82, d1f 0.74, v1f 0.63, d16s 0.45,
  v16s 0.385, d32s 0.25, v32s 0.22, d16f 0.157, v16f 0.134, d32f 0.086, v32f 0.077.
- I found no single idea worth 7%. Each of the eight below is worth roughly 0.1-1.3% if it works. The sum of expected
  values is about 2-2.5%. If every idea worked, the total would be about 5-6%.
- Two probes cost minutes and gate the most value. Run them first:
  - #1, PDL completion latency: the script exists (dev/xb/pdl_test.py) but was never run.
  - #5, the cold kernel-parameter fetch at kernel start.
- Two ideas are cheap knob-level builds: #2 (v1f first wave) and #4 (v1s on C6). The rest are real builds.

| # | idea | shapes | est. gain (us) | est. score | effort | P(win) |
|---|---|---|---|---|---|---|
| 1 | Programmatic-dependent combine kernel (PDL) replaces in-kernel arrival/poll end chains | d1f v1f v16f v32f v32s (+v16s via #6) | d1f -0.4..-0.8, v1f -0.2..-0.5, v16f -1..-2, v32f -0.5..-1.5, v32s -0.8..-1.3 | +0.6..1.3% | probe: script exists; build medium | ~40% |
| 2 | v1f "priority first tile": tile 0 alone, then Q prep, then the rest | v1f | -0.4..-0.9 | +0.25..0.55% | low (one knob) | ~45% |
| 3 | d1f lane partials kept in FP8-wgmma accumulators (XQA-style RS PV), reds straight from registers, no warp-to-CTA smem merge | d1f | -0.6..-1.1 | +0.45..0.8% | high (bound probe cheap) | ~40% |
| 4 | v1s: one (token, KV head) per C6 cluster (16 x 6 = 96 CTAs, all co-resident) | v1s | -0.15..-0.3 | +0.27..0.55% | medium | ~45% |
| 5 | Cold kernel-parameter fetch on every start path; if real, compile start-path pointers as immediates | all 8 small shapes | 0 .. -0.2 each | 0 .. +1.3% | probe 10 min; build medium plus a ruling | ~20% |
| 6 | Get verify sliding past cluster co-residency: P8 global + #1 for v16s; C4 x 30 + overflow slivers for v32s | v16s v32s | v16s -0.4..-0.8, v32s -0.8..-1.2 | +0.3..0.55% | high | ~25% |
| 7 | Length-free ("anchored") first burst inside the existing full-layer kernels | d16f d32f v16f v32f | -1..-2 each | +0.45..0.9% | high | ~20% |
| 8 | Fixed-point `red.async` accumulation into cluster owners | d1s (d16s, v1s) | d1s -0.05..-0.15 | +0.1..0.3% | medium | ~30% |

## 1. Combine as a programmatically dependent grid (PDL), replacing the in-kernel end chains

**Mechanism.** kernel_fn launches two kernels:

- **The existing main kernel, minus its end chain.**
  - Each CTA runs `griddepcontrol.launch_dependents` at entry.
  - It writes its partial with plain stores: FP16 O/l plus (m, l), or the fixed-reference `red.add` that d1f already uses.
  - Then it exits. There is no fence, no arrival atomic, no counter reset, no tag bit and no poll.
- **A small combine kernel**, launched with `cuLaunchKernelEx` and `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`.
  - The hardware schedules its CTAs onto SMs as main CTAs exit.
  - Each combine CTA preloads its gate rows, fetches its own code (cold, but off the critical path) and then executes
    `griddepcontrol.wait`.
  - The wait returns once the main grid has completed and all its writes are visible.
  - Then it does one bulk load round, the merge, RMSNorm x sigmoid and the bf16 store.
- **Correctness.** Hardware grid completion orders everything, so the self-resetting counters, XG's call-parity tags and
  the MONO counters all go away. The combine kernel zeroes the shards it read, or uses a parity buffer that the main
  kernel clears at its start.

**What it replaces, per shape (measured chain pieces):**

- **d1f** (XB r3 j25, clock64, finisher). These go: the MEMBAR drain plus acq_rel arrival (0.56-0.66), the
  return/sync/reset (0.22) and the cold epilogue entry (0.13). The shard-load round trip (0.37-0.48) stays.
- **v1f** (XG r2e trace, after the slowest publish). The gate poll (+0.6) goes. The bulk round (+0.64) and the
  sum/epilogue (~0.5) stay. The tag bits disappear from the stores.
- **v16f** (XC r3/r5). The PLAIN publish path goes: "~2 us from tiles to a visible publish", meaning lane-merge
  barriers, store drain, fence and count. With it go the combiners' wait and the round of partial loads. XDF cut the
  combine by 1.6 us, but the publish fence and count remain today.
- **v32f** (XC_END). The FINAL's lane-1 wait for P-1 publishes is replaced by grid completion.
- **v32s** (XF r6, run8w global). The publish to all-in hop is 1.7-2.4 us ("store visibility ~1 us + poll detection
  ~0.9"). It becomes completion plus one load round. XF's poll-schedule variants (PP, NAP, LC) all failed to shorten it.
- **d16f/d32f.** Probably nothing: XE r5 showed that shortening run1f's end chain does not show in nsys. I count 0.

**Why the notes suggest headroom.**

- Every global combine in the notes costs 2-3 dependent L2 round trips (red drain, arrival atomic, combine load) plus
  polls. The polls run under contention: XF r6, XG g2 ("loads of not-yet-written slots miss L2 and wait behind the
  stream"), and sibling MOE ("publish -> seen ~6 us under streaming").
- On b1 and verify-full shapes, end-chain changes do show in nsys: XA H16P/CLD/MONO -1.9, XG r1/r2 -0.8/-0.66,
  XC_END -3.5 on v32f.

**Why this is not a measured dead end.**

- PDL was only considered as a sort pre-pass (XF r8, XB r11). There it was impossible because the main kernels use
  ~230 KB smem and leave no room to co-reside.
- Here the dependent grid needs no co-residency: it runs on SMs that exiting main CTAs leave behind.
- research_r2 §2.1 rejected the FA3/FlashMLA separate-combine pattern as "an extra kernel boundary". That is exactly
  what the probe has to price.
- XB wrote dev/xb/pdl_test.py (round 11), but its job j57 was cancelled, so it never ran.

**Gain.** Let X = the time from the last main CTA's exit to the dependent's `griddepcontrol.wait` returning.

- If X ≤ 0.4 us: d1f -0.6, v1f -0.3, v16f -1.5, v32f -1.0, v32s -1.0 → about **+1.0..1.3%**. Add #6a (v16s) on top.
- If X ≥ ~1 us (comparable to a normal kernel boundary), the gain is about 0.

**Cheapest falsification.** Extend dev/xb/pdl_test.py:

1. Kernel A: 132 CTAs x 512 threads, ~200 KB dynamic smem (so 1 CTA/SM, like d1f). Spin a staggered 5-6 us, store a
   4 KB partial, stamp, exit.
2. Kernel B: 128 CTAs x 256 threads, launched with PDL. Stamp at entry, load a gate line, `griddepcontrol.wait`, stamp,
   load 132 rows, store.
3. Capture both into a torch graph and replay after a 512 MB flush.
4. Report:
   - X;
   - the total time against the same work done by an in-kernel atomic last-arrival combine;
   - an empty 1-node graph against an empty 2-node PDL graph.
5. **Build only if X ≤ 0.4 us and the extra node costs ≤ 0.2 us.** Build d1f first: highest sensitivity, and the
   combine is just a shard read.

**Main risk.**

- X may simply equal a kernel boundary (~1 us).
- The judge subtracts its own empty-graph cost, so any per-node overhead is ours.
- Kernel completion still waits for the main CTAs' final stores (same as today).

## 2. v1f "priority first tile": issue tile 0 alone, build Q in the gap, then issue the rest

**Mechanism.** In XG's xg_v8f, after the speculated page-table rows land:

1. Issue only tile 0's K/V: 16 KB per CTA, 2 LDGSTS per thread, 2.1 MB across all CTAs.
2. Build the Q fragments (Q is already in). This leaves a natural 0.4-0.6 us gap.
3. Issue tiles 1-5. Keep DEFER 4 for tiles 6-9.

During the gap, tile 0's lines have the DRAM queues to themselves.

**Why the notes suggest headroom (XG g5/r2e traces).**

- The work-groups (WGs) reach the step-0 wait at 2.61 us and tile 0 lands at 2.94. After that:
  - step 0 takes 1.65 us; step 1 starts at 4.72 and takes 1.6;
  - step 2 starts at 6.45 and the loop ends at 7.86-7.95;
  - the last tile lands around 6.2-6.5.
- WGs "barely wait for data" (0.13-0.15 us per step), so the tile phase is compute-bound from step 0 on. The whole
  chain therefore shifts with step 0's start, until it reaches the landing frontier (last landing plus one step
  ≈ 6.8).
- With ~2.1 MB of tile 0 alone in flight, tile 0 should land around 1.8-2.0 instead of 2.94.

**Why this is not a measured dead end.**

- "In-order issue ≠ in-order landing" (XC K-first, XB TORD) is about issue order inside one burst. A time gap is
  different, and nobody tried one.
- XA's DEFER and XG's DEARLY delayed the last tiles. Nothing moved tile 0 earlier.
- XA8_WGI changed which WG issues, not when.
- QLATE moved Q work after the burst. This moves it into the gap instead.

**Gain.** v1f -0.4..-0.9 us → **+0.25..0.55%**. It also applies to v16s/v32s, whose first unit sits at 3.8-4.6 us in
the k34 traces. But XB's early-start kernel transferred only about -0.2..-0.3 nsys there, so I count nothing for them.

**Probe.** One knob in dev/xg/xg_mod.py: a tile-0-only first wave, plus an optional nanosleep δ of 0, 200 or 400 ns
before tiles 1-5. Measure nsys with 2 copies each, and trace tile 0's landing. **Keep if ≥ 0.3 us in nsys.**

**Risk.**

- Trace gains have not transferred to nsys three times before (XA r4).
- The Q-prep code becomes cold code ahead of the main burst (~0.15 us/KB).
- DRAM may still interleave tile 0 with the rest.

## 3. d1f: lane partials kept in FP8-wgmma accumulators, reds from registers, no warp-to-CTA smem merge

**Mechanism.** Replace the per-warp mma.sync PV (which produces 16 warp partials) with one wgmma per lane and head.

- **The PV.** O^T[64 d][8 rows] += V^T[64 d][64 keys] · P^T[64 keys][8 rows], as 2 x m64n8k32 instructions.
  - A = V^T, in registers. It is built by `ldmatrix.trans` + `prmt`, as XQA's `loadVTileTransposed` does; research.md
    §2.4 already describes this register-A swap-AB form.
  - B = P^T in E4M3: 512 B per head per tile, in smem. FA3 also uses FP8 P.
- **The max.** Switch to a lane-level lazy max. The lane barrier that is needed anyway before the PV becomes a
  `bar.red.or` raise vote; maxima are exchanged only when some warp raises.
- **The end.**
  1. Each lane scales its accumulator by 2^(m_lane - REF): 8 floats per thread.
  2. It `red.add`s them into NC=16 shards. Four lane contributions per CTA instead of one, so 16 shards keep each
     address's add count at today's level.
  3. CTA barrier, then one arrival as today.
- **Side effect.** Each stage also loses the V e4m3→f16 cvt (32 per warp), the movmatrix and the HMMA. The stage is
  328 SASS per warp today (XB j19).

**Why the notes suggest headroom.**

- The d1f finisher's end chain is ~3.0 us (XB j25). Of that, "store partials 0.61-0.82 + merge 0.47 + bar" is the
  warp-to-CTA merge.
- XB's SP2 measured the store step at 1.04 on the first pass and 0.45 on the second. So at least 0.45 is intrinsic MIO
  work (16 warps x (12 SHFL + 16 STS.64)), and WARMEC ruled out cold code for the rest.

**Why this is not a measured dead end.**

- XB's AB=3 bound (-0.2) only shrank the merge. It kept the smem store step.
- NL=2 halved the merge but slowed the tile phase (2 CTAs/SM).
- XEL removed the merge in run1f, where the end chain is invisible in nsys. d1f's end chain is not invisible.

**Gain.** About -0.6..-1.1 us d1f → **+0.45..0.8%**. It stacks with #1: d1f's end chain would become
reds → exit → X → shard load → epilogue.

**Probe.** A timing-only `XB1_AB` build of xb_d1f that skips the warp-partial smem store and the CTA merge entirely.
Each CTA reds its warp-0 registers; the output is wrong. The nsys delta bounds the gain. **Build only if the bound is
≥ 0.6 us.**

**Risk.**

- The lane-max exchange adds a sync per tile, though only the last stage is exposed.
- The wgmma PV round (~350-430 cycles, XC probe_mma) may lengthen the last stage's chain.
- If the 0.6 us "first-execution" part of the store step is really a drain of the loop tail's MIO traffic, it may also
  delay the reds.

## 4. v1s: one (token, KV head) per C6 cluster

**Mechanism.** Use XE's XE_V1T layout (8 rows per cluster, owners holding 1-2 rows, the cluster folding its sources)
on 16 C6 clusters (96 CTAs, 342 keys per CTA). Today it runs on C8: 4 row groups x 2 heads, 256 keys x 16 rows per CTA.

**Why the notes suggest headroom.**

- XE_V1T on C8 x 16 (128 CTAs) finished its median CTA 0.37 us earlier: compute -0.14, reader -0.07, recv -0.09.
- It lost only on placement. Just 15 C8 clusters fit at 1 CTA/SM, so with smem capped, 8 SMs ran 2 CTAs and their
  clusters ended about 1 us late (nsys 4.70 vs 4.29).
- C6 co-residency is 17 (XB job 2), so 16 clusters fit in one wave.
- The CTA reader is smem-bandwidth bound: XE's "no CTA merge" diagnostic gives -0.34 on v1s. That cost halves with
  8 rows per CTA.

**What differs from V1T.** C6 instead of C8. Per CTA, keys go up 33% and rows halve.

**Gain.** v1s -0.15..-0.3 us → **+0.27..0.55%**.

**Probe.** Run V1T with CL as a template parameter set to 6. Check `cuOccupancyMaxActiveClusters` ≥ 16, then nsys
with 2 copies.

**Risk.**

- 33% more K/V per CTA lands later; XE's probe shows landing moving with keys per CTA.
- Each head's K/V is read 8x from L2 (once from DRAM).
- The co-residency margin is 17 vs 16; the existing plan logic falls back if needed.

## 5. The cold kernel-parameter fetch on every start path (probe first)

**Mechanism.**

1. Probe the gap between a kernel's first instruction (`%globaltimer` into a register, no parameter access) and the
   first LDG whose address comes from a kernel parameter (LDC c[0x0][..]), in a graph replay after the flush.
2. If the gap is ≥ 0.2 us, compile the start-path pointers and guesses (page_table, seq_lens, qkv, the cache bases,
   XE's glo/gseq) as NVRTC immediates, once per buffer set:
   - key the cache on `data_ptr()`;
   - fall back to today's generic module for any buffers not seen before.
   The first LDG would then issue right after the first instruction fetch.

**Why the notes suggest headroom.**

- Every trace stamp needs the trace-buffer pointer, which is a parameter. So no trace can see this fetch.
- The notes find nsys time outside the stamped window, but the evidence is mixed:
  - XB j10: "~0.5 us of v6 lies outside its stamped window" (unresolved);
  - XE: nsys d1s 3.36 against the last stamp ~3.05;
  - but k34t's v16s last stamp equals its nsys.
- After a 512 MB flush, the graph's parameter buffer is in DRAM unless the launch preloads it into the constant cache.

**Gain.**

- If the fetch costs 0.2 us on all 8 small shapes: about **+1.3%** (d1s 0.43, v1s 0.36, d1f 0.15, v1f 0.13, d16s 0.09,
  v16s 0.08, d32s 0.05, v32s 0.04).
- If parameters are preloaded at launch: 0.

**Probe.** A 10-minute kernel taking stamps S0 (first instruction), S1 (after the LDC) and S2 (after the dependent
LDG), plus the nsys duration of the same kernel. Do not build anything unless S1 - S0 ≥ 0.15 us.

**Risk.**

- Needs a coordinator ruling: code specialized to buffer addresses. Output stays correct for any input because of the
  fallback.
- The module has to be compiled before graph capture. If the judge's eager calls use different buffers from the
  capture, the timed graph gets the generic kernel and the gain is 0.
- Compile time per buffer set counts against the 15-minute run.

## 6. Verify sliding: get past the cluster co-residency ceiling

**Mechanism (two variants).**

- **(a) v16s.** Move from run8w's C6 DSMEM path (96 SMs) to run8w's global path with P8: 128 CTAs and 4.1 units per
  CTA, so about 2 serial unit chains per WG instead of 3. Combine through #1's PDL kernel.
- **(b) v32s.** Use C4 clusters for 30 requests (all 30 co-resident) with a DSMEM combine.
  - Spread the 2 overflow requests as thin slivers over all 120 CTAs (about 0.55 units each).
  - Compute the slivers first, and combine them by fixed-reference `red.add` plus a last-arrival epilogue that
    finishes long before the cluster owners.

**Why the notes suggest headroom.**

- v16s's tile phase is chain-bound: +1.7 us over the 6.0 gather skeleton, of which ~0.8 is the extra chain of the
  6-tile parts (XD r4).
- v16s is capped at C6 because 16 C8 clusters do not fit (15 co-resident).
- v32s pays a ~2.4 us global hop because 32 C4 clusters do not fit (30 co-resident; XF r6, which named the 2-kernel
  hybrid as untested).

**Why this is not a measured dead end.** The global path lost to clusters only because its combine was 2-2.8 us. Both
variants change that: (a) depends on #1 making the global combine cheap, and (b) keeps DSMEM for 30 of the 32 requests.

**Gain.** v16s -0.4..-0.8 (+0.15..0.3%) and v32s -0.8..-1.2 (+0.18..0.26%).

**Probe.** Timing-only bounds, all with wrong output:

- v16s: P8 global vs C6, both with the combine removed.
- v32s: C4 x 30 with 2 requests dropped, against run8w.

**Build only if the bound is ≥ 0.8 us (v16s) or ≥ 1.2 us (v32s).**

**Risk.**

- High complexity.
- Exact-fit co-residency (30/30), and GPC harvesting varies per die, so keep the occupancy-query fallback.
- Red contention on the slivers.

## 7. Length-free ("anchored") first burst inside the existing full-layer kernels

**Mechanism.**

1. At t=0, each CTA loads seq_lens and `page_table[c mod B, A_c .. A_c + 768)`. These positions are shape-determined.
   Rows past a short request's length are masked once seq lands. The PLAN ruling allows shape-determined addresses at
   any B.
2. It issues that K/V at about 0.8 us.
3. The FLAT split, Q^T and the new block run while this burst is in flight.
4. The anchored tokens become the CTA's first segment. The FLAT split covers [A_b, n_b) per request, so the FLAT
   ranges stay contiguous.
5. Anchored lengths are staggered across CTAs so the switches from anchored to FLAT do not coincide. Each switch goes
   through refills, with no ring drain.

**Why the notes suggest headroom.**

- Full kernels are 1:1 sensitive to a uniform start shift: VDELAY +1.0 → +1.1 us (XB j48).
- The earliest burst goes out at 2.75-3.25 us, against the probe's first issue at 1.54.
- By 14 us, run1f has landed 17 MB less than the probe (3.7 us of DRAM time, XF job 1).
- Every start variant that keeps the chain lengths → split → rows (VSTART 2/3/5, PF1, PTP) was zero-sum.

**What differs from the measured dead end.** XF's xf_dfull (+2.7..+4 us) was a whole new kernel: 528 free lanes,
claims, and +0.26 us/tile of bookkeeping. This keeps run1f's and XD's tight steady-state loops untouched and only adds a
first segment.

**Gain.** -1..-2 us on each of d16f, d32f, v16f and v32f → **+0.45..0.9%**.

**Probe.** Gather-only first, using dev/xf/probe.py: anchored first tiles against a length-dependent start. Measure the
onset and the ramp. XF saw the anchored phase land at only 3.3 TB/s with 22-24 MB in flight. That has to be explained
before building.

**Risk.**

- High: code first executed after the burst runs 3-8 us late (XF).
- Each CTA gets an extra segment and partial, inside a 10K-line kernel.

## 8. Fixed-point `red.async` accumulation into cluster owners

**Mechanism.** This is research_r2 idea D, never tried.

- Each CTA does `red.async…add.u64` of its fixed-point partial into one owner accumulator. The completion bytes land
  on the owner's mbarrier.
- The owner reads one slot instead of C slots.
- A common reference is required: the max over the new token plus a shape-determined key sample, with the existing
  exact fallback for large deltas.

**Why the notes suggest headroom.**

- XE's "no owner merge" diagnostic gives -0.21 us on d1s.
- XE_REF used a unified reference but still read 16 slots, and its post-wait phase stayed at 0.17 us. So the cost is
  the 16-slot read chain, which `red.async` removes.

**Gain.** d1s -0.05..-0.15 → **+0.1..0.3%**. Smaller on d16s and v1s.

**Probe.** A timing-only owner that reads one slot, with garbage `red.async` from 16 sources.

**Risk.**

- `red.async` is integer-only, so the fixed-point range and fallback logic are needed.
- Remote smem atomic throughput has not been measured.

## 9. Cheap audits (not structural)

- **MEMBAR on cluster tails.** `barrier.cluster.arrive` with the default `.release`, and `fence.acq_rel.cluster`, both
  compile to MEMBAR.ALL.GPU (XD r5). Fixing that was most of v16s's -0.45 us.
  - Grep the SASS of runE1 (d32s), xe_dm (d16s) and xer1/xer2 (d1s/v1s) for MEMBAR on their cluster tails.
  - Fix any hit with `fence.release.sync_restrict::shared::cta.cluster` + `arrive.relaxed`.
- **Co-residency margins on the judge GPU.** v16s runs C6 at 16 of 17 co-resident, and GPC harvesting differs per die.
  Make sure every cluster plan keeps its occupancy-query fallback.

## 10. Considered and rejected

- **NUMA / L2-partition-local gather (research_r2 B).** Streaming reads top out at 4.33-4.54 TB/s (sibling MOE),
  against 4.3 random and 4.38 page-sorted. So the upside is ≤3-5%, and it depends on an unverified H200 address hash.
- **Page-order sorting in any form.** Closed by XF r8-9 and XB r10-11 (0.46-2.2 cycles/key; ≤0.15 needed).
- **FA3 intra-WG overlap, ping-pong, or a producer WG with setmaxnreg.** ptxas C7513-C7515, the per-warp ~16-LDGSTS
  in-flight limit (W8, XA r3, XB), and XB's PP gave no per-unit gain.
- **FP16 PV with MN-major V.** XG r4c: net +0.16 for the f16-P half alone, before conversion.
- **Fixed-reference softmax with no max in v1f or verify.** E4M3/E5M2 overflow means redo paths (XB's xr: 1-5% of
  CTAs, +3-5 us each).
- **TMA multicast for v1s row groups.** It only recovers the ~0.1 us of L2 read amplification.
- **Tail stealing for the b1 full shapes.** A stolen tile needs a fresh loaded DRAM trip, which is ≥ the 0.5-0.8 us
  CTA spread. XC bounded verify full at ≤1.1 us.
- **Code prefetch via function addresses.** Infeasible (XG), and it reads code memory.
- **More precision cuts.** int8 partials were rejected (coarser than SGLang's FP32). FP16 red accumulation overflows
  without a known normalizer.
