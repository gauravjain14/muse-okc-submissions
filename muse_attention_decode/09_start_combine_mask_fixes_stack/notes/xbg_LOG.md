# xbg log (XB1 = d1f xb_d1f, XG5 = v1f xg_v8f)

## RESULT (work/xbg/cand.py = runs/k73.py + 2 one-line default changes; 16/16 tests, nearby 56/56)
- `#define XB1_GCONST 1` (was 0): grid size compile-time -> no runtime u32 division (+ p.G param load) between the kernel
  start and the page-table loads; the whole d1f pipeline starts earlier. d1f -0.16 us (7/7 interleaved rounds slim,
  full-file 5 rounds x 3 seeds: k73 13.90 -> 13.73).
- `#define XG_CLE 1` (was 0): the new block's cache_loc load moves to the prologue. It was issued after the burst and
  waited ~1.9 us behind ~20 MB of queued K/V requests -> WG0 of CTA 131 started step 0 1.9 us late and CTA 131 was always
  the last publisher. v1f -0.2 us (k73 15.84 -> 15.62, same 5-round run).

## Tools (work/xbg)
mkslim.py (slim copy = header + XB1 + XG5 sections only), trace.py (per-CTA %globaltimer/clock64 timeline; XB_TRACE /
XG_TRACE builds), ab.sh (interleaved A/B, one process per run), abx.py / abx2.py (replay-level A/B in one process),
ana_*.py, mb_*.py (microbenchmarks), dumpsass.py, c7511.py. cand_experimental.py = my working copy with all the
experiments below behind knobs (default off) + better trace instrumentation: stamps buffered in smem (no global stores on
the timed path), XB1_OBS=1 observer warp that records each tile mbarrier's completion time.

## Timelines (ns from the first CTA start)
d1f: pt arrives ~0.7 us; burst issue done 2.8 (med) / 3.9 (max) (cp.async back-pressure: ~73 KB in flight per SM);
tile data (observer) 3.1 .. 6.2 us (max over CTAs); BUT loop done 7.9 us max: per CTA the loop ends ~1.4 us after its
last tile landed -> the lanes are compute(latency)-bound, not data-bound, from the 3rd tile on.
 - lanes start the first tile ~0.7 us after it lands (they sit in the back-pressured burst issue, then X1SYNC)
 - per-tile chain ~920 cycles steady (wait/fence 220, lane bar 55, QK wgmma 180, softmax + P^T 260, bar.red 95, PV ~100),
   first tile ~1600 (the max raise from -inf always takes the rare path: +600)
 - tail after the last tile's data: last-tile math ~450, lane-pair exchange + bar.red ~400, merge+reds ~200,
   release + atom.inc ~650-850 (microbench: atom.acq_rel after reds = 978 cyc single CTA, 1931 with 132 CTAs),
   epilogue ~700 (L2 RTT 360 + RMS 170 + sigmoid/store 90)
 - ablations (timing only): no tile math (XB1_HDAB=2) -1.1..-1.4 us; no wgmma (new HDAB=3) -0.48; no V^T build (HDAB=4)
   -0.16; no lane-pair exchange (HDAB=1) -0.42.
v1f: s0/s1/s2 data 3.1/4.8/6.4 (med); each pair step ~1.5 us (QK 4 serialized QGMMA 256 ns, V^T ldmatrix/stmatrix 400,
softmax 600 (issue-bound: 310 instr/thread/pair step), PV 4 serialized QGMMA 290); loop done 7.9 med / 8.3 max (WGs do not
wait for data at steps 1, 2: compute-bound); publish ~1 us after the last loop (par-1 convert 320-576 ns cold / 176 warm,
par-0 fold + 9 stores ~550 ns warm or cold); gate +0.4-0.7; bulk loads +0.6; reduce/epilogue +0.6-0.8; end ~11.3.

## Methodology notes (important for anyone A/B-ing small effects)
- Code executed for the first time after the K/V burst is issued is fetched from DRAM BEHIND the burst queue: e.g. a dry
  tile chain started at 0.8 us (while producers issued the burst) took ~5 us; cold code even with idle DRAM ~0.5 us
  (kernel start -> page-table loads issued ~450-600 ns, incl. param constant-cache misses).
- Replaying the same graph back-to-back keeps the kernel's code warm in the SM instruction caches across the flush; in an
  alternating A,B order position matters (a graph replayed twice in a row is ~0.1-0.2 us faster). Different module
  instances (own ws buffers) also differ by ~0.1 us. => only trust multi-process interleaved rounds (ab.sh) for < 0.2 us.

## Experiments (A/B vs k73 or vs cand, same session)
- (my mistake) `flock gpu.lock ncu ... prof.py` deadlocked the GPU lock ~6 min (prof.py locks itself); coordinator killed it.
- XG_CLE=1: v1f -0.2 us. KEEP.
- XB1_GCONST=1: d1f -0.16 us. KEEP.  (XG5 analogue, G compile-time in xg_v8f: no gain, not kept.)
- XG_GATE 132 / 40 / 24(+PNS 200): no gain. XG_NOGATE (new: no gate, no counter read after the publish red): +2.4 us
  (stale-word polling of 128 combiners floods L2; also +2.5 with 300/800 ns poll sleeps).
- XG_PR=1 +0.35; XG_ALB=0 +0.1; XG_DEFER=2 +0.4; XG_PFT=0 +0.1; XG_MF=3 (fp32 exchange) +0.7; XG_MF=2 ~0; XG_SV=1 (weak
  stores) -0.08 (noise-level, not kept: weak stores); XG_DRY=1 (DRYM 8/32/96) +0.6..+1.0.
- XG_MF=4 (new: symmetric split publish, both parities fold half the columns, f32 exchange, pair barrier): +0.4.
  (first version put o[] on the stack: dynamic index -> 29 us.)
- XG_MF=5 (new: writer WG hands O/l through its own dead V^T buffer, folder = the WG with the last tile, no CTA barriers
  before the publish): +0.3. The publish stays ~1.1 us whatever its structure (not icache: a warm 2nd pass of the MF=1
  publish still takes 0.7 us for the fold; the stores themselves did not test cheaper).
- XB1_HWARM (new: whole HD end chain run dry before the tile loop, same code addresses): +3.3 us (the dry pass fetches its
  code behind the burst), and the real end chain was not faster.
- XB1_REL (atom.release instead of acq_rel + address dependency on the result): 0 (reverted, kept acq_rel).
- XB1_PAIR (new: dynamic two-tile chains when the next tile has landed, lane AND vote): 0 (pairs taken often; the pair
  chain incl. its rare path is not cheaper in practice).
- XB1_MMS (new: tile math on per-warp mma.sync m16n8k32 e4m3, no wgmma/proxy fences/lane bar): correct, +0.35 (fp8
  mma.sync ~100 cycles latency on H100; PV no longer async).
- XB1_PROD (new: 4 producer warps issue the whole burst, compute warps free from t=0): +1.7 us — compute warps then run
  the tile-loop code for the first time while the DRAM queue is full (instruction fetch behind the burst); + XB1_DRYT dry
  chain before the first tile: still +1.8.
- XB1_F0 (first tile: max exchange before a single put_p, saves ~220 cycles once per lane): ~0 / -0.05 (noise), not kept.
- XB1_NB (no tile-start lane barrier): 0. XB1_FCP1=0 (no proxy fence after the tile wait): -0.06..-0.16 but formally
  needed (cp.async generic-proxy writes -> wgmma async-proxy reads), not kept. XB1_HDPV=0: +0.38.
  HD8 + PW0: +1.0 (PW0 alone +0.48).
- C7511 (xg_v8f wgmma serialization) persists for every knob combo tried, even at 115 registers (XG_ABL=1): structural.

# ROUND 2 (base runs/k74.py = k73 + XB1_GCONST 1 + XG_CLE 1)
Result of round 2: no change kept (work/xbg/cand.py == runs/k74.py). Findings:

## v1f / xg_v8f: C7511 (wgmma serialization)
- Found the trigger: the PV wgmma's 36-register accumulator (m64n72: 64 d + 8 "ones" columns for the row sums) in
  this function. Compile-only probes (c7511.py, NVRTC cache defeated with a source nonce):
  PV as m64n64 (o[0..31] only, row sums dropped; probe E4) -> NO C7511 (126 regs, SASS: 4 QK QGMMA back to back,
  DEPBAR.LE 1 then 0; 4 PV QGMMA back to back, one DEPBAR). Any form of the row-sum wgmma brings it back: n64 + n8 in
  one window (E6), a separate n8 array (E10), n8 in its own later window (E9), n32 + n40 (E12); also with Q^T from
  smem (E7, -8 regs), operand fences on o / pa around the window (E14, the CUTLASS recipe), DEFER 0, KF 0, ABL 1
  (115 regs). A standalone n72 kernel compiles pipelined (78 regs): it is this function's register picture.
  Other layouts that avoid it: XG_MFREE=0 + XG_US=1 (one pair step instance) or XG_MFREE=0 + XG_PF=1 (pairs first).
- BUT un-serializing does not make v1f faster: the steps are ALU / issue bound, not wgmma-latency bound.
  E4 (real MFREE layout, pipelined, timing-only: row sums garbage): +0.13 us; per-step times unchanged (1.52 / 1.54 /
  1.54 us vs 1.47 / 1.49 / 1.52). US=1: +0.9 us (step 0 = dummy pair: 2.25 us). PF=1: +1.8. (MFREE=0 alone +0.35.)
- Step ablations (timing only, k74): no V^T transposition (XG_ABL=1) -0.22 us; no clamp + no P byte packing
  (XG_ABL=12) -0.36 us => even large ALU cuts buy only 0.2-0.4 us.
- XG_TAU (new: lazy max raise, raise only when a score exceeds the running max by > TAU; P codes referenced to
  max + TAU): correct (3.32e-2), 16/16, nearby 56/56; first session -0.1 us (TAU 2/4/6 alike, 7 rounds), but a clean
  5-round same-file A/B (TAU 2 vs 0) -0.03 and vs k74 file +0.01 => noise, not kept (cand_r2_tau.py).
- Device function pointers are NOT readable as global data (illegal address): no way to L2-prefetch the kernel code.

## d1f / xb_d1f
- XB1_QP (new: static software pipeline, QK(it+1) issued right after QK(it) retires, double-buffered V^T fragments,
  P^T and QK accumulators, unrolled x2): correct, +0.58 us (cand_r2_qp.py).
- (round 1, same theme) XB1_PAIR 0, XB1_MMS +0.35, XB1_F0 ~0, XB1_PROD +1.7: the tile-chain latency resists
  restructuring; lanes starting tile 0 earlier only moves the first-execution instruction fetch of the loop code under
  the full DRAM queue (code fetch latency ~ queued bytes / bandwidth, several us).
