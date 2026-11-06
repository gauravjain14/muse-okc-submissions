# xd round 2: warp-specialized q_len-8 full kernel (attn_xw_v8f), log

Base: runs/k76.py -> work/xd/cand2.py. Same sandbox GPU, ab.py interleaved A/B vs k73's XD section (identical XD
build to k76).

## Step 1: 1 producer WG + 2 consumer WGs (one per KV head, every tile), 384 threads
- Producer WG: page-table rows by __ldg 4 tiles ahead, K/V rows via cp.async (16 B, 8 lanes per 128-B row) into an
  NS-stage ring, `cp.async.mbarrier.arrive.noinc` on full[s] (count 128); empty[s] (count 2) released by one thread of
  each consumer WG after the tile's internal WG barrier (K read by QK wgmma done, V read by ldmatrix done).
- Consumers: the k73 tile math per head (xd_tile2x / xd_tile), no li-merge (each WG sees every tile of its head),
  new-token tile from a dedicated zeroed 16-KB stage; publish/combine = XD protocol on 256 consumer threads.
- Correct (v16f/v32f rel. err 2.754e-2 / 2.829e-2 vs 2.747e-2 / 2.807e-2 for attn_xd_v8f: accumulation order).
- setmaxnreg: `illegal instruction` when setmaxnreg.dec runs after loads were issued (ksc/vsc __ldg in flight);
  works when the register split is the first thing the kernel does (minimal kernels always worked).
- Timing: v16f 112.0 us (+14.4 vs 97.6), v32f 208.1 (+30.6). Ring depth 6/8/10/11: no change -> consumer-bound:
  each head's WG walks every pair serially (~1.4 us per 128-token pair) = ~91 tok/us/CTA < the ~125 the DRAM feeds.
  Needs two pairs in flight per head (2 consumer WGs per head, or intra-WG QK/softmax/PV software pipelining).

## Step 2: 1 producer WG + 4 consumer WGs (2 groups x 2 heads, as attn_xd_v8f), 640 threads
- setmaxnreg rules learned (also from the base worker): the per-WG counts after dec/inc must fit the launch pool
  (640 x 96 = 128 x 32 + 512 x 112); `setmaxnreg` illegal-instruction/hang when that is exceeded. Placing the split
  at entry with common code after it is also unsafe (that code is compiled for the launch count).
- Bug found: with two groups alternating tiles by segment-local parity and an odd ring (NS 5), a stage alternated
  between groups and a consumer could wait on a full[] parity two phases ahead -> passes at once on stale data ->
  extra empty arrival -> hang. Fix: groups take tiles by CTA-global parity and NS is even (static_assert).
- Correct, but slow: v16f NS 4/6/8: 157.5 / 122.2 / 114.8 us; v32f NS 6/8: 225 / 210 (k73 97.4 / 177.8).
  Producer at 32 registers (STACK 64 spills incl. the consumers at 112): deeper page-table lookahead made it worse
  (LA 1/2/4/6: 114.8 / 117.1 / 119.5 / 124.7). Register-starved design; dropped.

## Step 3: XH = 1 producer WG + 2 consumer WGs, each consumer WG = one tile group for BOTH heads (384 threads)
- Registers 56 (producer) / 224 (consumers) = 384 x 168 launch pool. No spills.
- Producer: 4 warps, half-tile (32 tokens) each, page-table entries one per lane (lookahead LA half-tiles), 16-B cp.async
  with the evict-first policy, 2 rows x 8 lanes per op; full[s] count 64 (2 warps).
- Consumer pair schedule (FA3-like): QK h0, QK h1 issued back to back (f16 accum), V^T of both heads, wait<1> ->
  softmax h0 (O and row sums rescaled in place) -> PV h0 accumulated in place -> wait<1> (QK h1 done: release the
  stages) -> softmax h1 -> PV h1 -> wait<0>. Last pair / new-token tile: k73 xd_tile per head.
- Merge buffer (36 KB) aliases the V^T buffers (extra barrier; V^T ones rows rewritten after) -> ring 10 stages.
- Correct (rel err 2.757e-2 v16f / 2.860e-2 v32f; in-place accumulation rounding).
- v16f: NS 4/6/8/10 = 149.2 / 118.8 / 100.9 / 101.2 us; final config +3.1 us vs k73 (seeds 7, 8);
  v32f +4.2..+5.8.
- Trace (v16f): first K/V issue 3.5-3.7 us (k73 4.1), first pair 5.95 (6.18) -> start ~0.3 us better; ring end
  median 87.1 (k73 85.0); end phase from the last ring end to the combine end ~6 us (k73 ~3.5): consumer-only combine
  with 256 threads (two items per thread now) and the XD protocol. v32f ring end median 168.6 vs 163.5.
- Steady-state consumption ~4.1-4.3 TB/s in both kernels (pair stamps): the stream is at the random-gather rate in
  k73 already; the earlier "math costs 4-5 us of stream rate" (W2Z=1 vs W2Z=2) was mostly start / end differences.
- Producer-only limit (XH_NOMATH: consumers skip the math, still release; end phase kept): v16f 99.25 (+2.4 vs k73),
  v32f 179.6 (+1.4); trace: ring end median 84.2 / max 87.0 (k73 85.0 / 87.9) -> the producer feeds slightly faster
  than k73's loop; with math XH's ring end is 87.6 (consumers cost ~3 us); XH's end phase (last ring end -> combine
  end) ~6 us vs ~3.5 for k73.
- Symmetric end (new-token tile: group li takes head li; per-head merge so both groups write one head each): no
  measurable change (+3.0..3.4 v16f, +5.0..5.7 v32f vs k73).
- Consumer WGs use up to R211 (212 registers) -> a 3-consumer-WG layout (160 regs each) would spill.
- Odd observation in both kernels: all combiners of many requests finish within ~60 ns of each other (k73 at 91.3 us
  for b9-b15 although their last publishes differ by ~0.8 us; XH all at 96.3) -> some common stall at the end
  (instruction fetch of the cold combine code under load is the leading suspect; not proven).
STATUS: XH correct on v16f/v32f but slower than k73 (+3 / +5 us); not dispatched (env XD_XH=1 only).
- XH on other shapes (K2_DEFS XD_XH=1): batch 2 ctx 100000 page 1 correct; batch 4 ctx 6000, B24, B64 ctx 20000,
  B128 ctx 8000 hang or fail (not debugged; likely the many-small-segment paths). NOT safe to dispatch.
- The XW/XH sources are now compiled only on request (`#if defined(XD_XW) || defined(XD_XH)`): with them in the
  default XD build import+compile went from ~5 s to ~39 s.

## Round 2 conclusion (for now)
- The steady-state K/V stream of attn_xd_v8f is already at the random-gather rate (4.1-4.3 TB/s by pair stamps in
  both kernels); warp specialization cannot raise it. What is left is the start (~1 us over 3 dependent hops; XH's
  producer starts ~0.4 us earlier), the per-run imbalance tail (~2-3 us) and the latency-bound end phase (~3.5-4 us).
- XH (best WS layout found) is correct on v16f/v32f but +3 / +5 us: its 2 consumer WGs (212 registers, both heads,
  QK/softmax/PV overlap across heads) are ~3 us slower than the producer can feed; its end phase is ~2 us longer
  (256 consumer threads for the combine). 4 consumer WGs need 640 threads -> 112-register consumers (spills) and a
  32-register producer (slow).
- If WS is pursued further: per-stage metadata from a producer that claims tiles dynamically would give the tail
  balancing cheaply (consumers only read per-stage valid counts), but XH first needs ~3 us more consumer throughput
  and a faster end phase to be competitive. My recommendation is to not invest further in WS for these shapes.
