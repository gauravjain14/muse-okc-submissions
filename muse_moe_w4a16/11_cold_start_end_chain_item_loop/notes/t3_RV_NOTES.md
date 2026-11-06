# RV: independent structural design review (muse_moe_w4a16 top-3 push), 2026-11-03

## IMPLEMENTATION (coordinator mission, 2026-11-03 PM): in-flight cap in the main kernels' producer -- REJECTED
Files: t3/RV/mk_rv1.py (patch BASE -> OUT), t3/RV/moe_rv_1.py (= t3/INT/moe_i2.py + mk_rv1.py), rv_run1/2/3.sh + .log,
bench/rv_run1..3.json.
- Switch RV_CAP. Before issuing stage it, the producer waits for stage it - C's full barrier (slot (it-C) % NS, parity
  ((it-C)/NS)&1), after its slot wait, at all 4 steady sites: LEAN4 inner sub-loop, LEAN3 per-stage loop, LEAN (KW 1)
  path, and the full path. C is run-time data: a `__constant__ int rv_cfg[5][4]` whose initializer comes from RV_C1 /
  RV_C0 / RV_N0 (+ _K<b> per-kernel overrides). It is defined last in CUDA_SRC so no other constant's bank offset moves.
  RV_CAP_M is a compile-time kernel mask (default 31 = bs8/16/32/128/256).
- SASS (wpcgs_search --full vs moe_i2):
  - The default build is EXACT in all 16 kernels.
  - In the RV_CAP build, the consumer stage loops are byte-identical (67/67). Only the producer loops of the 5 target
    kernels change. moe1t, moe1 and the bs33-64 / bs257+ kernels keep the base hashes.
  - Every C value gives the same SASS.
  - The (8,8,1) kernels have a different spill signature (ldl/stl 9/2 vs 10/3, 10/2 vs 11/3).
- hang1 16 sizes ok for C=1 and C=2 (box 0) and C=2 (box 1). check_mt was not run because nothing was banked. The cap
  only adds waits.
- bench2 jm, 60 reps, fillers at both ends, controls a1/a2 = moe_i2. Values are % time vs the control mean (+ = slower);
  z = twin (cap code present, RV_C1=0, i.e. no cap).

| shape | twin z | C=1 | C=2 (two copies) | C=3 | C=2 + startup C0=1 for it<4 |
|---|---|---|---|---|---|
| bs8 (box 0) | +0.95 | +31.4 | +0.91 / +1.44 | +0.54 | +2.87 |
| bs16 (box 0) | +0.23 | +30.8 | +0.58 / +0.42 | +0.44 | +1.69 |
| bs32 (box 0) | +0.84 | +24.6 | +1.43 / +1.35 | +1.12 | +1.45 |
| bs128 (box 0, run 3) | +0.71 / +0.45 | (box 1: +6..+7) | +0.79 / +0.70 | = no cap (NS 3) | - |
| bs256 (box 0, run 3) | +0.23 / +0.09 | (box 1: +3.7..+3.9) | +0.45 / +0.40 | = no cap | - |

  (Run 2 on box 1 was perturbed: its control means were 179.5 / 212.1 us against 160.9 / 196.7 normally. Only its C=1
  each-vs-each ratios are quoted.) Every copy is slower than every control copy at every C, and no C beats moe_i2 on any
  shape. Against the twin, the cap itself is -0.4..+0.6%, i.e. noise to slightly worse.
- Why the microbenchmark result does not transfer:
  1. The kernel's slot cycle contains the producer's per-stage issue path (~1200 of ~1900 cycles under consumer load on
     SMSP0, per ST). So any cap below today's ~3.2 stages in flight costs per-SM stream rate: C=1 gives +25..31% (the
     ring benchmark lost only 7.6%), and C=2 roughly cancels whatever latency it saves.
  2. The control round trips that get faster are not on the steady critical path. The startup window gets worse with
     C0=1 (+1.5..+2.9%).
  3. Code presence alone (a few lane-0 instructions per stage) costs +0.1..+0.95%, which matches the producer's
     per-stage sensitivity (PDLY: ~0.03% per ns).
  This also re-reads ER: its bs32 loss was codegen (as ST said), not control latency. The Little's-law physics (section 4) stands as
  a measured fact, but the #1 bound in section 3 was wrong for this kernel.
- Byte-based (8,8,1) cap: not built. C=2 already equals the twin there (NS = 3), so a byte cap between C=2 and no cap
  has no room.

## ADD-ON (coordinator, from EC's moe1t finding): warp-uniform producer polls (RV_WUP) -- NEUTRAL, not banked
Files: t3/RV/mk_rv2.py (IN = moe_rv_1.py -> OUT), t3/RV/moe_rv_2.py (= moe_i2 + mk_rv1 + mk_rv2; both patches also apply
cleanly to t3/INT/moe_i3.py), rv_run4.sh / rv_cmt.sh, rv_run4.log / rv_run5.log, bench/rv_run4.json / rv_run5.json.
- What the main producers had: lane-0-only ld.acquire spins, i.e. LDG.E.STRONG.GPU + CCTL.IVALL + YIELD inside
  BSSY/BSYNC, followed by the __syncwarp's BRA.DIV. In bs32's SASS these are 0x14f00 (down readiness),
  0x15020 / 0x156b0 (gather barrier CB+3), and the lane < 16 ballot poll at 0x14dc0. This matches EC's moe_8_4_1 SASS
  lines 5369 / 5386 / 5491.
- RV_WUP sites (RV_WUP_M bits; default 30):
  - 16 = W1, down-item readiness spin. Every lane acquire-loads every chunk counter (broadcast), loops on __all_sync, then
    every lane fences. Applied in the bs8/16/128/256 kernels only: in bs32 it changes one stage loop's control bits
    (bit 32 forces it).
  - 2 = W2, gather barrier.
  - 4 = W3, the early-stage gather wait.
  - 8 = W5, the routing mbarrier.
  - 1 = W4, ballot poll without the lane branch. Off by default: it adds spills in (8,8,1) and re-schedules (4,*,1)
    loops.
  All sites use `if constexpr (RVWK)`, so only the 5 scored kernels change.
- SASS (wpcgs_search --full vs moe_i2):
  - The default build is EXACT.
  - The RV_WUP build has 67/67 stage loops identical. Only the producer loops of moe_4_1_1, moe_4_2_1, moe_8_4_1,
    moe_8_8_1_3_4 and moe_8_8_1_3_8 change. There are no spill changes.
- hang1 16 sizes ok (both boxes). check_mt was not run (nothing banked).
- bench2 jm, 60 reps, 3 RV_WUP copies vs 3 moe_i2 controls, fillers at both ends; % time vs the control mean:

| shape | RV_WUP copies | control copies |
|---|---|---|
| bs8 | +0.20 / +0.25 / +0.05 | +0.38 / -0.20 / -0.32 |
| bs16 | -0.04 / -0.11 / +0.23 | -0.03 / +0.08 / -0.08 |
| bs32 | +0.32 / +0.21 / +0.01 | -0.11 / +0.03 / +0.04 |
| bs128 (box 1) | +0.02 / -0.11 / -0.40 | +0.20 / -0.17 / -0.08 |
| bs256 (box 1) | -0.07 / -0.09 / -0.11 | +0.02 / +0.07 / -0.07 |
  All are inside the control spread, and no shape has every copy beat every control. VERDICT: neutral. In the main
  kernels these polls fire about once per CTA (gather) or rarely (a down item whose chunk is not ready after the
  ballot), so they are off the critical path, unlike moe1t's per-occurrence exchange polls.

## Status (review, earlier today)
- DONE. Review + 1 experiment family (3 short jobs, box 0, ~5 min GPU total; standalone microbenchmarks, no kernel edits).
- Finding 1 (correction): HV's bare-ring times in STATE.md are GROSS event times (t3/HV/l2b.py does not subtract the
  ~5.3 us empty-graph cost that bench2 subtracts). On the bench2 protocol the bare ring is 5.3 us faster, so the
  stream-bound shapes are 11-19% above a bare ring of their bytes, not 5-7%.
- Finding 2 (new mechanism, validated): every global round trip issued while the GPU streams waits behind the global
  in-flight weight bytes (Little's law). In-flight bytes set this latency, and copy count does not. DRAM is already saturated with 2 x 36 KB in
  flight per SM (9.7 MB). Today's 4-slot ring keeps about 3.2 stages in flight when stream-paced (~15 MB). Capping
  in-flight at 2 cuts atomics, flag publish->seen, small TMA copies and dependent misses by 40-58% with no stream-rate
  loss (measured with consumers that hold slots and hiccup every 16 stages).
- #1 candidate: shallow-queue streaming. The producer issues stage it only when stage it-2 has landed
  (in-flight cap K = 2, a separate setting from the NS = 4 slot buffer), and uses K = 1 in startup windows. The change is
  producer-only, so the consumer stage loops stay untouched. Expected +0.6..+1.8% geomean (central ~+1.1%).

## 1. Correction: bare-ring headroom (measured, same protocol as bench2)
t3/RV/rv_q.py (bare ring, 132 CTAs, 36 KB stages, no consumers): bench2 net (gross - fixed) = in-kernel span
(median first issue -> last landed) + 0.1..0.3 us (K=1 457 MB: 108.67 vs 108.38; K=2 701 MB: 152.32 vs 152.00).
fixed = 5.25-5.44 us here, 5.0-5.5 in every t3 bench2 json. So there is no hidden launch cost in our kernel either
(ST's K at bs16/32 = bench2 net within noise).

| bytes | HV l2b c1 (4 x 36 KB) gross | that - 5.3 | RV K=4 span (net - 0.3) | kernel bench2 net (v4111) | kernel over ring |
|---|---|---|---|---|---|
| 238 MB (bs8) | 58.6 us | 53.3 | 52.2 (234 MB) -> ~53.0 | 62.58 | +18% (STATE said +6.8%) |
| 457 MB (bs32 ~450) | 105.2 | 99.9 | 99.1 (453 MB) | 110.46 | +10.5-11% (STATE +5%) |
| 701 MB (bs256) | 157.5 | 152.2 | 152.1 | 196.76 | +29% (STATE +25%) |
Ring rates (net): 4.47 TB/s at 234 MB, 4.57 at 453 MB, 4.61 at 701 MB. With P-GAP1511 data-only (bs8 59.6, bs16 78.4,
bs32 102.5) the kernel's DATA PATH alone (startup + end + items, no compute) is +6.3 / +4.0 / +3.0..4.5 us over the
ring. Compute coupling is +3.0 / +5.8 / +8.0 us on top.

## 2. Where the time is (measured inputs from STATE / ST / SU / EN / HV / PW / B1, corrected ring)
| shape | net us | ring / floor us | gap | dominant measured pieces |
|---|---|---|---|---|
| bs1 | 24.3 | 18.7-19.2 (4x36 KB streamer) | +26% | consumer-bound 18 stages (~1.0 us each), first data ~3.9 us, end chain 2.05 us (1:1) |
| bs8 | 62.6 | ~53.0 | +18% | first data 4.8 us (floor 2.4), K - median last item 4.26 (tail 1.5 + chain 2.7) |
| bs16 | 84.2 | ~74.5 | +13% | first data 4.9, end 5.0, coupling 5.8 (NO_DEQ ceiling 3.4%) |
| bs32 | 110.5 | ~99.5 | +11% | first data 6.2, end 4.8, 17-32-token wgmma tiles push C/M to ~0.9 |
| bs128 | 160.9 | ~140 | +15% | first data 8.7, end 5.8, coupling 8.8 (DRAM-bound in principle: decode 116 + tensor 11 < 140) |
| bs256 | 196.8 | 152.4 | +29% | sum of stage costs (decode 126 + tensor extra ~38 us at 1.68 GHz) + edges + power |
A clean design at ~1.05x ring on all six shapes gives ~4650 us. The leader's 4865 is about halfway there. Our
edges (startup 4.8-11 us to first data, 4.3-6.5 us end) are the obvious difference: a lean start (~2.5 us) and
end (~2.5 us) alone would give ~+5% geomean.

## 3. Ranked structural candidates (expected gain x feasibility in ~24 h)

### #1 Shallow-queue streaming: separate the in-flight cap from the slot buffer (VALIDATED key assumption, section 4)
(Superseded: built and measured in the kernel, REJECTED. See IMPLEMENTATION at the top. The bound below assumed the
producer issue path is not in the slot cycle, and it is.)
- Mechanism: latency of any control access during the stream ~ unloaded + in_flight / BW. Today: ~3.2 x 36 KB per SM in
  flight when stream-paced (consumer releases a slot ~350 cyc before the next landing -> 4 in flight, then 3), so every
  atomic, flag, record read, cold code/constant miss, x / first-stage copy waits ~2-3 us. The producer would issue stage it
  only when (a) slot it % NS is released (as today) AND (b) stage it-K has landed (try_wait/test_wait on full[(it-K)%NS],
  parity ((it-K)/NS)&1; no extra barrier). K = 2 in steady state. The buffer stays NS = 4: during a consumer hiccup
  the ring still fills all 4 slots before the stream stalls, so jitter absorption is unchanged. NS = 3 lost 4.5% because it
  cut the buffer. K = 1 applies in startup windows (until the consumers' first stage), in bs32 FST, in the bs8/16 early path
  (SU's E = 2 > E = 4 already shows this) and in moe1t around the phase-A -> B exchange. At bs128/256 heavy tiles, cap
  bytes (~72 KB) rather than stages, since x per stage ~ weights there.
- Why it is not closed: ring depth NS changed the buffer and in-flight together. ER / WP-RING BRC / L2 prefetch / v81
  early weights / SU E = 4 all RAISED in-flight, and ALL lost:
  ER bs16 -0.6, bs32 -3.3..-5.0%; BRC -0.2..-1.0%; L2 prefetch -6..-15%; SU E = 4 vs 2 bs8 -1.7..-1.9% (parity atomic
  +1.3 us later); bs256 warm-up i-fetch "queues behind ~14 MB of weights" (= 132 x 3 x 36 KB). Nobody LOWERED in-flight
  while keeping the buffer.
- Bound (measured deltas x measured critical paths, conversions from ST/EN/SU slopes): per loaded round trip -0.8..-1.4 us
  (K 3.2 -> 2). Startup chains at bs8-32 (parity atomic, cold producer loop pass 2.3-2.9 us, cold consumer first
  stage) give -0.5..-1 us. The item-top first-stage excess (WP-EDGE +0.3 us/item bs8, +0.85 bs32; EN +0.25..0.41 bs128/256,
  conversion ~0.5) trimmed ~40% gives bs8 ~-0.4, bs32 ~-1.0, bs128/256 ~-0.6 us. End chain under the tail gives -0.2..-0.5 us.
  bs1 gives ~-0.3 us. If ER's measured loss for +~0.8 stage is roughly symmetric, the inverse is +0.5% bs16 and +2..3% bs32.
  Per shape: bs1 ~1%, bs8 1-1.5%, bs16 0.8-1.2%, bs32 1.5-3%, bs128 ~0.5%, bs256 ~0.3%, giving a geomean of +0.6..+1.8%
  (central ~1.1%). Stream-rate cost: 0 (section 4).
- Feasibility: HIGH. This is a producer-loop edit in the (4,*,1), (8,4,1), (8,8,1) and moe1t producers. B1/WP-N showed
  producer-site edits keep the consumer loops schedule-identical (verify with dev/wpcgs_search.py + dev/kern_sass.py). Risks:
  one more satisfied try_wait per stage on SMSP0 (~35-50 cyc, producer ~1200/1900 busy; PDLY slope 0.3); per-SM share
  variability under the dynamic queue. Plan: a switch RV_K = {2, 3, 4 (= today)} + RV_K0 (startup) on v4111/i1; bench2 jm
  2+ copies with controls; then re-run ST/EN's delay slopes, which this change rescales.

### #2 End-of-call hop removal: the CTA completing a token's last pair normalizes + combines it inline
- Mechanism: today latest G2 item -> SEPI epilogue -> signaler wake -> red.release -> combine unit poll -> loads -> norm
  -> store (2.6-2.8 us, ST E2). An acq_rel arrival count per token (or expert) lets the last arriver run the 4096-col
  norm + weighted sum from L2 (BF16 y rows as SGLang, FP32 ss partials). This drops the wake and poll hops (~0.6-0.8 us)
  and the split-pair FP32 row reloads at bs8.
- Bound: NO_COMBINE -3.0 / -2.4 / -1.2 / -1.15 / -1.39% (bs8..256, measured) = ceiling ~1.5% geomean. Realistic
  -0.6..-0.9 us/call gives bs8 ~1.2%, bs16 0.9%, bs32 0.7%, bs128/256 0.4%, so +0.5..+0.8% geomean. With #1, each remaining
  hop is ~40% cheaper. Feasibility MEDIUM (SEPI epilogue + combine units of 4 kernel families, ~1 day).

### #3 Cross-item consumer pipelining (item transitions under the next item's first stage)
- Mechanism: the per-warp item transition is a ~1260-1340 cycle latency chain (EN: drain, staging, fence, record ->
  R2UR -> branch, S2R, i-cache), not issue. Records for item i+1 would be prefetched into registers during item i, the
  accumulators double-buffered on the <= 16-token paths (8-16 regs/lane), and the drain of set A interleaved into the
  peeled first stage of item i+1.
- Bound: whole transition free = 2.5 / 2.6% (bs128 / bs256, EN measured slopes), HV qsim "no item edges" +4.3% bs32,
  +2.5..2.9% bs128/256; realistic +0.4..+1% geomean. Feasibility LOW in 24 h: it re-schedules every stage loop (+-1%
  lottery) and needs wpcgs_search for loop-exact forms. #1 already shrinks the data-wait part of each transition.

## 4. Experiment: control-plane latency vs bytes in flight (t3/RV/rv_q.py, rv_q2.py, rv_q3.py; box 0)
Command (each): `python3 t3/sbx.py exec "timeout 120 python3 t3/RV/rv_qN.py --quick ... && timeout 600 python3
t3/RV/rv_qN.py --reps 9 --out /tmp/rv_qN.json" --box 0`. 132 CTAs, per-CTA contiguous 36 KB TMA stages, K = in-flight
cap, NS = slots. Probes ran on CTA 0 (and the poller on CTA 1) every 4 us inside the streaming window: a 4-hop dependent
ld.global.cg chain on cold 4 KB lines (TLB-warm 4 MB region), an atom.global.add round trip (L2-warm line), a 4 KB bulk
copy from a cold buffer, and red.relaxed publish -> ld.relaxed seen on another SM. The flush.sum() before every graph
replay followed the bench2 protocol. Medians over 9 reps x up to 16 probes. Rate = bytes / in-kernel span. Net times
are inflated by the probe chain when latencies are long, so the span is the rate metric.

Bare ring (rv_q.json), 457 MB, probe SMs streaming (unloaded: chase 336 ns/hop, atom 256, 4 KB TMA 448, pub->seen 320):
| K x 36 KB | in flight | TB/s | chase/hop | atom | 4 KB TMA | pub->seen (ns) |
|---|---|---|---|---|---|---|
| 1 | 4.9 MB | 4.220 (-7.6%) | 512 | 320 | 800 | 448 |
| 2 | 9.7 MB | 4.564 | 1176 | 608 | 1600 | 1376 |
| 3 (~today) | 14.6 MB | 4.568 | 2028 | 1376 | 2608 | 2544 |
| 4 (ER-like) | 19.5 MB | 4.568 | 3108 | 3072 | 3840 | 4960 |
| 6 | 29.2 MB | 4.557 | 4816 | 4960 | 5824 | 6624 |
In-flight BYTES set the latency, not the copy count: 18 KB stages x 4 (9.7 MB) give 4.545 TB/s, 1296 / 736 / 1760 / 1568 ns
(the same as 36 KB x 2). 18 KB x 2 (4.9 MB) give 4.360 TB/s, which matches K=1. Probe SMs idle (smode 2) at K=3: 1672 / 1056 / 2048 / 1952,
so a streaming SM's own copies add 0.35-0.6 us (head-of-line), rising to ~1.1-1.6 us at K=4. Sizes: K=2 vs 3 vs 4 at
234 MB 4.470 / 4.489 / 4.472, 453 MB 4.559 / 4.555 / 4.568, 701 MB 4.610 / 4.611 / 4.606 TB/s (K=2 costs <= 0.4%).

Coupled ring (rv_q3.json): NS = 4 slots, one consumer warp per CTA holds each stage H cycles (+1300 every 16th
stage = item hiccup), clock 1.98 GHz (no MMA), stream-paced:
| H (C/M) | K | TB/s | chase/hop | atom | 4 KB TMA | pub->seen | net incl. probe chain |
|---|---|---|---|---|---|---|---|
| 1100 (~0.55) | 4 (= today's NS=4 producer) | 4.460 | 2280 | 2080 | 2976 | 3680 | 119.3 us |
| 1100 | 3 | 4.446 | 1888 | 1408 | 2464 | 2400 | 116.8 |
| 1100 | 2 | 4.456 | 1180 | 640 | 1712 | 1312 | 107.0 |
| 1400 (~0.7) | 4 | 4.359 | 2036 | 1536 | 2688 | 2528 | 116.4 |
| 1400 | 3 | 4.346 | 1760 | 1440 | 2336 | 2496 | 116.0 |
| 1400 | 2 | 4.408 (+1.1%) | 1144 | 640 | 1632 | 1280 | 106.5 |
| 1400, 234 MB | 4 / 2 | 4.236 / 4.236 | 2184 / 1224 | 1184 / 736 | 2432 / 1728 | 2400 / 1568 | 79.4 / 74.9 |
At equal stream rate, the latency-bound 16-step probe chain (~7 round trips per step) finishes 9.5-12.3 us sooner with K=2.
That is the effect expected on the kernel's startup, item-edge and end chains.
rv_q2.json (H = 1720 / 1900: C/M ~1, consumer-bound): K = 2/3/4 give the same rate (4.33-4.37 / 4.02-4.04), probes are near
unloaded (landed slots wait, so little is in flight), K = 1 gives -4.7%, and NS = 3 gives -0.6% (buffer). The cap is harmless
when consumers bind (bs128/256 heavy phases, moe1t phase B).
VERDICT: key assumption holds. Control latency follows Little's law in in-flight bytes, and K = 2 keeps the DRAM stream (bare
and coupled). Today's NS = 4 producer runs at the K ~3-4 latency point for no bandwidth benefit.

## 5. Considered and rejected (why, with numbers)
- 4-bit / entropy-coded scales (lossless; scale entropy ~3.9-4.0 bits): WP-S5 timing bound +1.35% geomean decode-free,
  ~4.3 bits achievable (escapes ~1.2% of scales: 51 per stage). Consumer decode costs more than the bytes at bs16/32 (S5R:
  +4 ops/stage = -0.35 / -1.5%). An expander warp would need ~75-300 issue slots per stage on SMSPs at C/M 0.8-0.9.
  Net <= 0.
- x / h activation reuse: XR built and lost (WP-X). TMA multicast of x within clusters is bounded by XHALF (-2.0% bs256,
  -1.0% bs128), i.e. <= +0.5% geomean, and adds cluster-coupled slot releases.
- Heavy-stage decode + tensor overlap (HV, every arrangement worse). Decode op count is provably minimal. CUDA-core,
  TF32, FP8 and INT8 paths are out by the rules or by >= 16-29% more issue per weight. FP32-FFMA bs1 needs ~3.1 instr/weight vs 1.6.
- Expert-local norm in 16-SM clusters (DSMEM): 9 experts on 8 clusters at bs8 gives cluster quantization. K-split
  phase B for bs1 would need red traffic ~7% of the stream (WP-FUSE / moe1 history).
- Routing-independent speculative first wave: 28-78% of first copies are useful (bs8..256), and it breaks LPT order
  (-1.1..-1.5%).
- PDL pre-kernel (plan / gather): adds a node, and the main kernel's own cold path gates next (NB1 bound +0.72 / +0.30%).
- TLB at item switches: P-PATTERN random single stages cost only -1.5% in a bare ring. Launch overhead: none (net ~ span + 0.2).

## 6. How a 4865 us competitor can be ~5% faster legally (consistent with the numbers)
Lean edges: our first data at 4.8-11 us vs a ~2.4 us floor, and our end at 4.3-6.5 us. Each 2 us of start + end is ~3-5%
at bs8-32 and ~1.5% at bs128/256. Shallower pipelines are a second route: a cp.async / TMA design with ~2 x 32 KB in flight per SM
gets ~2x faster control paths at no bandwidth cost (section 4), without its authors ever tuning for it. Our 4 x 36 KB
ring keeps ~15 MB queued.

## Files (all in t3/RV/)
- rv_q.py: bare-ring in-flight sweep + probes + ring gross/net (`--quick` smoke). rv_q.json: its results.
- rv_q2.py: coupled ring (NS slots, cap K, consumer hold H + hiccup J). rv_q2.json (H 1720/1900, consumer-bound).
- rv_q3.py: rv_q2 + SM clock readout, stream-paced H 1100/1400. rv_q3.json.
