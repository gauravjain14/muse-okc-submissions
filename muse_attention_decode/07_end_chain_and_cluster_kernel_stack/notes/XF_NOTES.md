# XF (round 2): decode full, batch > 1 (d16f, d32f)

STATUS (2026-10-28, job 12): xf_dfull (dev/xf/xf.cu v7) is correct (check.py 0 failures, edge + forced fallback 0)
but SLOWER than k34: bench d16f 89.2 vs 86.5, d32f 167.9 vs 165.1; nsys 91.0 vs 88.1, 169.4 vs 165.4. The module is
OPT-IN (K2_DEFS=XF_ON) so dev/xf/build.py is harmless in a stack. Measured budget, findings, next levers: bottom.
Old versions: xf_v1.cu .. xf_v6.cu (see the job sections). Trace any build with
`python3 dev/xf/tl.py --file dev/xf/k34xf.py --defs XF_ON --shape 32,1,full` (k34: --file dev/xf/k34t.py).

Files: xf.cu (the kernel `xf_dfull`, own NVRTC module), build.py (`python3 dev/xf/build.py <base> <out>`: appends the
module + kernel_fn wrapper; predicate q_len 1, window 0, page 1, 16/2 heads, hd 64, 2 <= B <= 64), k34xf.py (= build of
dev/k34.py), edge.py (edge tests of the predicate, 2 calls each; `--defs XF_REF=500.f` forces the exact fallback path),
tl.py (timeline: per-tile issue/land stamps, BW per us, per-CTA / per-warp end stamps; works on k34t.py and k34xf.py),
probe.py (gather floor on the REAL inputs, run1f-like lanes + per-tile stamps), mk_k34t.py -> k34t.py (k34 + per-tile
stamps in run1f, -DXT_TILE), nsys_run.py (+ nsys_loop.py copy: per-replay kernel durations), lc.py (local compile),
job*.sh / job*.log (Modal jobs; mention dev/coord in --cmd for the k34b control).

## Job 1: where k34's time goes (same container; traced builds; us from the first CTA start)
Gather probe on the real inputs (probe.py, no math, 132 x 4 lanes x 3 x 16-KB stages = run1f's skeleton):
  d16: lane3 74.98, lane3h (evict_first) 74.78, lane2h 74.83, ring_8_2 (probe_gather best) 74.61 us
  d32: lane3 149.65, lane3h 149.97, lane2h 150.62, ring_8_2 151.07 us   (649.7 MB: 4.33 TB/s average)
  probe d32 timeline: first issue p50 1.54 us, first tile landed p50 4.2, steady landing ~4.6 TB/s (bins of 1 us; every
  10th bin dips to ~3.3: probably a globaltimer artifact, also in k34), in flight 17.7 MB (~134 KB/SM), issue->land
  latency ~3.9 us under load; last tile p10/p50/max 143.9/147.2/150.9 (the probe has its own ~3-7 us tail).
k34 (k34t, d32): first K/V issue p50 3.58 (probe 1.54), first landed p50 7.42 (probe 4.22), then a deeper start bubble
  (landing 1.25 / 1.34 / 2.18 TB/s in bins 8-10: first tiles slow = cold tile code + first-tile max exchange); by 14 us
  k34 has landed 17 MB less than the probe (= 3.7 us of DRAM time). Steady state identical (~4.6 TB/s, 20.6-21 MB in
  flight, 4.6 us latency). Last tile p50/max 150.66/155.14 (probe 147.2/150.9). End chain (critical CTA): tiles 155.36
  -> mstore 156.00 -> merged 157.15 -> segend 157.22 -> publish 158.45 -> comb0 158.69 -> combined 159.46 (4.1 us).
  d16: first issue 3.33 (probe 1.41), first land 6.91 (4.10); last tile p50/max 76.54/79.62 (probe 73.34/75.01);
  critical CTA tiles 80.02 -> combined 84.16 (4.1 us). Last stamp med 84.16.
  No-compute k34 d32: last stamp 156.02 (with compute 159.46): compute delays the tile end by ~3.5 us (lanes are 96 %
  busy with compute, wait share 0.039 vs 0.376 without compute; the probe 0.007 because its warps sit in LDGSTS issue).
BUDGET k34 vs the lane probe (traced): d32 ~159.5 vs 150 = start ~3.7 + end chain ~4.1-4.3 + compute in the tail /
  last tile ~1-1.5 (the tail spread p50->max is ~4.5 vs the probe's 3.7). d16 ~84.2 vs 74.8: start ~3, end 4.1, tail ~1.5.
(nsys and bench of job 1 failed: dev/wp and dev/coord not uploaded. Fixed: dev/xf/nsys_loop.py copy; --cmd names dev/coord.)

## Design of xf_dfull (dev/xf/xf.cu) - targets the start (anchored issue) and the end chain (warp-level arrivals)
- 132 CTAs x 512 threads = 528 INDEPENDENT 128-thread lanes; no CTA-wide barrier anywhere.
- Each lane streams a continuous tile sequence through its own 3-stage ring; tiles of different requests follow each
  other (descriptor FIFO in registers: b | nv << 8 | start / end / anchored flags), so a segment switch never drains.
- ANCHORED first segment (no length dependency): lane L takes request L % B, positions [(L / B) * 192, +192). Its three
  tiles are issued right after ONE page-table round trip (rows outside the cache zero-filled; masked past the length
  once seq is known; V rows past the length zeroed in smem in that rare case). Allowed by the PLAN.md rulings (shape-
  determined addresses; nothing derived from a guessed length).
- FLAT remainder: request b's positions [min(A_b, seq_b), seq_b) concatenated and split evenly over the 528 lanes
  (exact double floor division), computed REDUNDANTLY by every warp from the 64 lengths (no smem, no barrier).
- New token: lane b (anchored lane 0 of request b) normalizes k_new, writes both cache rows and folds the single key into
  its softmax state directly (m = its score, P = 2^8, o = 256 v): no new tile.
- Tile math = WD's QGMMA dec_tile (refill at the vote barrier).
- Partials: at each segment end every WARP red.add's its O columns and row-sum partial at a fixed reference
  2^(m - 50) into the request's FP32 accumulator (fire and forget); maxima outside [-74, 110] go to an exact classic slot
  (index lane + b for FLAT pieces, 528 + B + lane for anchored: unique) + list entry. At the lane end each warp fences
  once and arrives on every request it touched (lane j of the warp: arrival j, atom.add of its units; target
  4 x (anchored lanes + FLAT units)); the completing warp combines alone (32 threads: O 4 KB + L + gates in one load
  round, RMSNorm x sigmoid, bf16 store, reset).

## Job 2: v1 (dev/xf/xf_v1.cu) - correct, slower
check.py 0 failures; edge.py 0 failures (incl. forced fallback XF_REF=500: every partial through the classic slots).
bench (20 reps, same job): d16f k34 86.12 / 85.98 | xf 90.71 / 90.83;  d32f k34 165.36 / 165.50 | xf 169.99 / 170.22.
timeline d32 (traced): first K/V issue p50 1.79 (k34 3.58) BUT tiles 1-2 were issued only after the cold prologue
(stamp p50 5.2, max 6.9): DRAM ramp slow (landing 0.9-3.8 TB/s in bins 3-13); steady landing ~4.35 TB/s (k34/probe
~4.6, other containers though); last landing p50/max 157.3/163.2; end: loop exit max 164.1 -> combined max 169.1
(5 us: the globally last warp completes EVERY request it touched -> 2-3 serial combines; 4x the end-time reds).

## Job 3: v2 = all 3 anchored tiles at once (XF_ISS3), lane FLAT page-table bulk L2 prefetch (XF_PTP), anchored
arrival mid-stream as a release-only atomic 2 tiles after the segment (XF_AEARLY), paired red.v4 (XF_RV4)
check / edge (+ fallback, + AEARLY=0,RV4=0) 0 failures. bench (20 reps):
  d16f k34 86.45 / 86.32 | xf 90.54 | PTP=0 91.02 | PTP=2 (+ whole-table slice) 90.05 | AEARLY=0 91.16 | RV4=0 89.62
  d32f k34 163.86 / 164.58 | xf 167.26 | PTP=0 169.08 | PTP=2 167.52 | AEARLY=0 168.77 | RV4=0 167.07
  => page-table prefetch -1.8 us (d32), mid-stream anchored arrival -1.5; red.v4 pairing neutral.
timeline d32: the 3-tile burst issued by 4.35 (p50), but the prologue AFTER it runs on cold code behind 25 MB of K/V:
  prologue-end stamp p50 8.76 / max 11.5 us; DRAM landing starts at bin 6, idle-ish 6-9 us, steady only from ~14 us
  (by bin 19 ~30 MB = 6.4 us of DRAM time lost vs saturating from t = 0). End: last landing max 163.84 -> loop exit
  164.69 -> atomics 165.54 -> combined 168.21 (the single combine itself ~2.7 us).

## Job 4: v3 = anchored segment of NA tiles (3 into smem at entry + NA-3 L2-prefetched, refilled from L2),
host-reciprocal split, gate-row L2 prefetch (XF_GPF). check / edge (NA 3/5/6, fallback) 0 failures. bench (20 reps):
  d16f k34 86.46 / 86.09 | NA5 91.13 / 91.24 | NA3 89.29 | NA6 92.28 | NA5 no gate prefetch 93.95
  d32f k34 164.26 / 163.93 | NA5 169.08 / 169.23 | NA3 167.00 | NA6 169.81 | NA5 no gate prefetch 171.50
  => L2 prefetch-ahead of anchored tiles LOSES (+2 us); gate prefetch WINS (-2.4..-2.8 us: the end-chain combine's
     gate load was a cold DRAM read).
timeline (same container, traced): the prologue-end stamp stays at p50 ~9 us in "codewarm" mode too (flush, then a
  tiny replay of the same kernel so only its code is in L2) => NOT cold code. SASS: ptxas sank the "straight-line"
  predicated prefix loads (seq_lens, Q, k/v-new, scales, cache_loc, even page-table rows 1..5) BELOW the K/V LDGSTS
  burst: they queue behind 25 MB of K/V (~4-6 us). (Same risk in every kernel with a load prefix + burst; WS's
  "predicated asm loads" recipe is not enough under ptxas scheduling.)
  INTRA-CTA lane imbalance: the 4 lanes of a CTA finish p50 4.4 / p90 7.2 / max 10.9 us apart (lanes of one SM run
  at different speeds); CTA last landing p50 162.2 vs lane p50 160.0. k34t in the same container: steady landing BW
  4.25 TB/s (xf 4.23: no steady-state deficit), intra-CTA spread p50 2.8, tiles-done max 162.24 -> combined 166.14.

## v4 (dev/xf/xf.cu; v1-v3 kept as xf_v1.cu / xf_v2.cu / xf_v3.cu)
- XF_DEP: the first K/V copy's source address carries an opaque dependency ((xor of every prefix load) & p.zero,
  p.zero = 0 at run time) -> ptxas must complete all prefix loads before the burst (they land together with the
  page-table rows the burst waits for anyway). SASS check: no prefix LDG after the first LDGSTS (DEP=0: 6).
- The FLAT split (needs the lengths) and Q^T of the CTA's first two FLAT requests now happen before / right after the
  burst issue, so nothing after the burst waits on memory.
- FLAT tiles are split per CTA (132 ranges); the CTA's 4 lanes claim tiles from a shared-memory counter (claim for
  iteration i+1 made at the top of iteration i by the lane's thread 0 into a 2-slot mailbox, read after the lane
  barrier). Lane-local smem copy of the CTA tile map (incl. tile prefix, piece length, first position per request).
- Segments are detected in the consumed stream (request change); a lane has <= 1 segment per request. Mid-stream
  FLAT segment ends: one deferred release-only arrival; further ones (CTA spanning >= 3 requests: tests only) use a
  blocking acq_rel arrival and leave the combine to the lane end (smem bit mask). Fallback slot of a FLAT piece:
  4 (c + b) + li; anchored: 4 (G + 64) + lane.

## Job 5 (v4): first run failed edge (B=33, B=16: NaN) = bug: the deferred mid-stream FLAT arrival's atomic result
lived in lane 0 but was evaluated by lane 1 at the lane end (lane 1 arrived a second time -> count overshoot -> never
combined). Fixed (shfl from lane 0). Rerun: check / edge / fallback 0 failures. bench (slow container: probe 157.9):
  d16f k34 86.41 / 86.38 | xf 91.93 / 91.13 | DEP=0 91.80;  d32f k34 164.50 / 165.15 | xf 170.56 / 170.18 | DEP=0 169.48
timeline: intra-CTA spread FIXED (p50 1.66 / max 3.2, was 4.4 / 10.9) but the burst now starts at p50 ~5-6 us: the
  split before the burst put 631 SASS instructions (~10 KB) ahead of the first LDGSTS -> cold fetch ~0.4 us / KB even
  with DRAM idle. End: last landing max 168.58 -> loop exit 169.87 -> atomics 170.78 -> combined 172.30.
## Job 6 (v5 = tiny pre-burst path: prefix loads + dep + burst at SASS instr 202; split / claims / Q^T after it)
check / edge 0 failures. bench (fast container: probe lane3h 74.98 / 150.82):
  d16f k34 83.60 / 83.94 | xf 89.05 / 89.14 | DEP=0 90.70 | AEARLY=0 90.18
  d32f k34 158.28 / 157.86 | xf 164.18 / 163.93 | DEP=0 164.63 | AEARLY=0 164.89     (xf still +5.5 / +6 us)
timeline d32: burst issued p50 3.7 (p10 2.3), prologue end p50 5.7, first tile observed p50 7.55 (probe 4.35);
  bins 7-19 land only ~3.3 TB/s with 22-24 MB in flight (k34 ~4.7, probe ~4.6); steady 4.35-4.6 later.
  CTA last landing p10/p50/p90/max 153.2/156.0/160.0/162.7 (probe same container 147.5/149.3/150.3/151.2: tight);
  end: last landing max 162.69 -> loop exit 164.16 -> atomics 165.09 -> combined 166.53.

## Job 7 (diagnostics, same container): code-warm vs flush identical for xf AND k34t (first land 7.68 both modes;
xf CTA last landing p50 161.7 / max 167.9 vs k34t 158.5 / 161.7). Skipping the anchored reds (XF_DIAG=1): no gain
(+1 us). No page-table prefetch: +1 us. => not cold code, not red contention, not the prefetch.
## Job 8: claim modes all equal (d32: dynamic 169.2, hybrid RR + 2-tile tail pool 169.9, static RR 170.9, tail pool
1 / 4 ~170, anchored arrival 8 tiles later 170.1; k34 164.4 / 164.6). Stamps: the mid-stream anchored arrival (release
atomic) stalls each warp ~0.8 us (MEMBAR waits for its fresh reds).
## Job 9: per-tile consumer timing (globaltimer quantum ~0.26 us; medians over lanes): in steady state BOTH kernels
NEVER wait for data (L(k) - W(k) = 0): per-tile time W(k+1) - L(k) = k34 1.79 us (d32) / 1.54 (d16), xf 2.05 / 1.79.
Landing -> refill issued (QK + softmax + vote) 1.02 us in both. k34's 1.79-1.88 us per tile per lane = 16 KB x 528 /
4.6 TB/s: k34 sits AT the DRAM limit; any extra per-tile cost (xf's bookkeeping, ~25 instrs) shows up directly.
Early tiles: xf k=2 (anchored end: publish + Q^T of 2 FLAT requests) 3.07, k=4 (anchored arrival) 3.33 us; k34 k=0 3.58.
## Job 10: XF_PTA (next refill's page-table row one iteration ahead) LOSES: d16f 90.4 vs 88.6, d32f 165.5 vs 163.5
(PTA off; k34 83.2 / 158.2 on that fast container). Refill issue time unchanged (1.02): the row is not on the critical
path (same verdict as WD_PTA in k34). Default PTA off.
## Job 11: XF_OVH (virtual units at each request's FLAT start: the boundary CTA pays an extra segment) sweep, d32 / d16:
  OVH 0 169.81 / 91.20 | 128 169.53 / 91.31 | 256 168.37 / 89.85 | 512 168.19 / 89.80 | 1024 167.92 / 89.95
  (k34 164.84 / 164.71, 86.30 / 86.51). CTA last landing p50 / max with 1024: 161.15 / 164.86 (tail 3.7 us, was 6+).
## Job 12 (FINAL, v7 = xf.cu: OVH 512, CLAIM 2, TAILT 2, PTA 0, DEP 1, AEARLY 2, GPF 1, PTP 1, RV4 1)
check.py (K2_DEFS=XF_ON) 0 failures; edge.py + forced fallback 0 failures. Errors d16f 3.71e-2, d32f 4.20e-2 (= k34's).
bench (20 reps, same job): d16f k34 86.66 / 86.29 | xf 89.37 / 88.97 | xf off (= k34 path) 86.04
                           d32f k34 165.03 / 165.18 | xf 167.64 / 168.06 | xf off 164.93
nsys per-replay kernel duration (40 reps): d16f k34 88.06 (p10 86.46, p90 89.34) | xf 91.02 (89.57, 91.94)
                                           d32f k34 165.39 (164.48, 166.78) | xf 169.39 (168.38, 170.37)
=> xf_dfull does NOT beat k34 (+2.7..+4 us). build.py makes the module OPT-IN (K2_DEFS=XF_ON); without it nothing is
   compiled and kernel_fn is the base's, so stacking dev/xf/build.py is harmless.

## Conclusions (what the measurements say about decode full B > 1)
1. Budget of k34 vs the lane-structured gather probe (same container, traced): start ~3-3.7 us (first tile consumed
   at ~7.4-7.7 us; the probe at 4.35), end chain ~4-4.4 us (last landing -> combined), tail p50 -> max ~3.2 us (probe
   ~1.9-2.3); steady state ~at the probe (k34's per-tile 1.79-1.88 us == DRAM limit, lanes never wait for data).
2. The anchored (length-independent) start gets the first K/V request out at ~1.0-1.3 us (k34 3.1-3.6) but does NOT
   move the first consumed tile earlier (xf 7.7 vs k34 7.4-7.7): after a 25-MB burst everything the SM does next
   (code fetch, loads) waits in the SM-side memory queue; code already in L2 does not help (codewarm == flush).
3. Independent 128-thread lanes inside a CTA drift 4-11 us apart by the end (warp scheduling): share the CTA's tiles
   among its lanes (smem claims; round robin + small dynamic tail is as good as full claiming).
4. ptxas sinks "straight-line" predicated asm prefix loads BELOW a later LDGSTS burst (they then wait ~4-6 us behind
   it). Fix: an opaque data dependency of the first copy's address on every prefix load ((xor ...) & p.zero). Check
   the SASS of every start path (k34's run1f is fine; compact new kernels are exposed).
5. Gate rows for the end-chain combine must be in L2 (xf: -2.4 us); a mid-stream release arrival costs ~0.8 us of
   MEMBAR per warp; boundary CTAs need ~512-1024 virtual units (OVH) of compensation.
6. Lost / neutral: L2 prefetch-ahead of extra anchored tiles (+2 us), page-table row one iteration ahead (+2 us),
   red.v4 pairing (0), claim granularity (0), anchored-arrival timing beyond 2 tiles (0).
## Next levers (not built)
- End chain "known-last fast path": before its last tile a lane acquire-reads its final request's counter; if every
  other contributor already arrived, it prefetches the accumulator + gates during the last tile and folds its own
  registers in (no red / fence / atomic / reload): ~-1.5..-2 us on the critical lane IF its CTA-mates arrived first
  (they finish within ~1 tile with shared claims, so maybe half the time; a CTA-level variant after a 4-lane barrier
  would make it deterministic). Applies to k34's run1f end chain as well (its critical CTA: tiles -> combined 4.0 us).
- Start: a smaller post-burst critical path (tile 0 only in the first burst, then the prologue, then tiles 1-2) to
  shorten the SM-side queue the post-burst code fetch waits in; not measured in isolation.

# ROUND 2 (coordinator): levers inside k34's run1f (attn_1_0_1_0) via dev/xf/build.py <base> <out> (patches run1f in
the base's SRC only; applies to dev/coord/k36.py and dev/k34.py; old xf_dfull builder = dev/xf/build_xfd.py).
Knobs (all default OFF = byte-identical run1f): XFR_KL (known-last end path), XFR_PF1=N (page-table rows of the first
N refills loaded before the burst; the burst's tile-0 row carries an opaque dependency on them, p.pad_ == 0), XFR_W
(first-wave tiles per lane; the rest at tile 0's vote). `--trace` adds dev/xf/mk_r1t.py stamps (per tile + start path:
TR 18 gates, 1 burst issued, 19 new token, 20 Q^T+fence, 21 new tile, 10 iteration-0 rows, 3 tile0, 22 / 23 first
refill entered / issued, 24 iteration 0 done, 25 iteration 1 barrier). dev/xf/edge2.py = run1f edge tests (B 2..64,
page 1 / 64, dp heads, kv_scale 1/64, lengths 1..300K; --defs W9A_REF=500.f forces every partial to classic slots).
## R1 (k36t, d32 medians, us): prefix 0.79 | page-table rows issued 2.10 | rows back 2.97 | gates 3.10 | burst issued 6.78
(3.7 us of LDGSTS issue under backpressure) | new token 6.84 | Q^T + fence 7.29 | iteration-0 rows requested 7.53 |
tile0 7.65 | first refill entered (vote) 8.22 | first refill ISSUED 10.44 (it waits 2.2 us for its page-table row, loaded
at 7.53 behind the start burst) | iteration 0 done 11.22. First tile 3.3 us (steady 1.79); landing 1.5-2 TB/s in
bins 8-10. SASS: run1f's Q / new-token loads precede the burst loop (no ptxas sinking there). Probe same container:
first land 4.35. d16 the same shape (first refill issued 9.83).
## R2 (KL + PF1=3 on k36; check.py 0 failures; edge2 + forced classic slots 0 failures). bench (20 reps):
  d16f k36 86.58 / k34 86.67 | both 87.61 / 87.86 | KL 87.37 | PF1 86.88 | none (= k36) 86.52
  d32f k36 164.65 / k34 164.78 | both 165.30 / 165.23 | KL 165.15 | PF1 165.18 | none 164.97
  KL: only 1 of 32 requests takes the fast path (combined count 31): the CTAs of one request finish within ~2 us
  of each other, so when the last one starts its merge the others have not published yet; and the decision barrier
  (thread 0's acquire sample -> CTA) made EVERY last-segment merge ~0.9 us slower (mstore -> merged 1.14 -> 2.05 us;
  critical end chain 3.9 -> 5.0 us). PF1=3: first refill issued 0.74 us after the vote (was 2.2), but the stall moves
  to iteration 3 (refill 1.79 us) and the bench does not move.
## R3 (fast container: k36 83.07 / 157.78, k34 83.12 / 157.53): none 82.94 / 157.79 | W2 83.28 / 157.77 | W1 84.03 /
  158.58 | W2 + PF1 83.88 / 158.23 | KL 84.10 / 158.71. W2 lands tile 0 1.1 us earlier (first land 6.66 vs 7.42) but
  its deferred tile + first refill stall in iterations 0-1; bytes landed by 13 us ~22 MB in every variant.
## R4 (k36 86.66 / 164.82, k34 86.46 / 164.80; dflt = all off 86.66 / 164.35): PF1 3 / 6 / 9 = 86.77 / 86.72 / 86.74,
  164.76 / 165.80 / 165.06; PF1 6 + WS_PTP 6 (superset prefetch after the prologue) 87.77 / 166.61; PF1 9 + WS_PTP 0
  87.40 / 165.92; PF1 6 + W2 86.76 / 164.95. PF1=6 removes the start bubble (landing 3.3-4.2 TB/s in bins 7-11) but its
  extra pre-burst loads delay the burst issue ~0.8 us (issued 7.53 vs 6.73) and the stall reappears at iteration 6.
## Round 2 verdict: both levers are DEAD ENDS in run1f (numbers above). The start is zero-sum in DRAM time: the first
  ~15 us deliver the same bytes whichever refill waits; WS's mode-10 page-table superset placement is the local
  optimum (moving it costs +1..+2 us). The known-last end path needs the other CTAs of the critical request to have
  published before the critical CTA merges, which the correlated finish times of neighbouring CTAs prevent; sampling
  later exposes the acquire latency on every CTA. Gate rows: run1f already stages them into smem at segment start.
  Remaining small item (not built): the merge's redundant range check (every item thread scans 16 rows x 4 lanes of mM,
  ~60 LDS) -> one per-row check + __syncthreads_or, est. -0.3 us on the end chain.

# ROUND 3a (dropped by the coordinator): independent v1f kernel xf_v1f (dev/xf/xf8.cu + xf8core.cu = XD's xdcore,
build: dev/xf/build8.py <base> <out>, tests dev/xf/edge8.py, debug dev/xf/dbg8.py). Design: 2 lanes x 2 WGs, one
resident stage per tile (<= 10 per CTA), T128 xd_tile2 pairs + T64 xd_tile for the odd tile / causal new block (first,
lane 1 of the owner), XA's d1f start recipe (speculation, prefix loads + opaque dependency, lean issue, evict_first),
H16P partials + monotonic counter + last-128-arrival combiners. 127 regs, 0 spills, 3040 SASS. FIRST RUN WRONG
(rel-L2 ~1.3 on every case incl. ctx 1; finite, call-stable); the debug job (per-CTA slot check vs torch) did not run
(no output) before the coordinator dropped v1f (XA landed 13.85 vs k34 15.95). Not usable as is.

# ROUND 3b: END-CHAIN AUDIT (coordinator: apply XA's measured combine fixes (CLD clamped loads, MONO counter, FP16
partials, early gate load) to every other shipped kernel; base dev/coord/k39.py = k34 + xe2 + xa3 + xd)
Tools: dev/xf/endsass.py <cubin> <kernel> (memory / sync / branch ops after the last tensor op, backward branches
marked); local NVRTC 12.4 cubins of k39's three SRC strings. Builder: dev/xf/build_ec.py <base> <out> [KNOB=V..]
[--trace] (knobs XFE_PUB / XFE_RCK / XFE_NPL, default 0 = base; --trace = mk_r1t stamps + end stamps 26-29).
## SASS audit: dependent L2 round trips (RTT ~0.4-0.7 us under load) on each critical end path
- run1f W9D (d16f/d32f, attn_1_0_1_0): merge (smem) -> red.v4 into the fixed-reference accumulator -> bar ->
  thread 0: MEMBAR.GPU (drains the reds: 1 RTT) + acq_rel ATOM (1 RTT) -> bar -> last arrival: flag / O / L ldcg
  issued together (1 RTT; CLD-clean, the flag branch comes after) -> epilogue (gates already in smem) -> store.
  = 3 RTT, BUT the publish is a `#pragma unroll 1` loop over the CTA's <= 2 partial requests in thread 0: each
  iteration = 2 double divisions (P), MEMBAR, ATOM, then the misc store waits for the atomic -> a CTA with 2 partials
  (most CTAs at d16/d32) arrives for its LAST request 1 RTT late = 4 RTT. Fix XFE_PUB (warp q publishes request q).
  No DONE atomic on the K == 1 W9D path (counter reset after the output) = MONO-clean. FP16 n/a (red.add
  accumulator). Also: the merge's range check scans all 16 rows x 4 lanes per item thread (XFE_RCK: once per row +
  __syncthreads_or; 16 LDS x 4 iterations, est. ~0.1 us). Rare out-of-range list path: serial list loop (2 RTT per
  listed CTA) - rare, left. Trace (round 2, d32 critical CTA): tiles -> mstore 0.97 -> merged 1.87 (!) -> publish 1.2
  -> comb0 0.2 -> combined 0.7.
- run8w global (v32s, attn_8_1_1_1w4): non-owners: FP16 slot rows by plain stores, no fence / atomic (sentinel-
  validated), exit. Owner: gate LDGs issued before the poll (OK); round 1 = one probe word per source (1 RTT), then
  the landed sources' rows, 4 sources by PREDICATED loads issued together (CLD-clean) (1 RTT); later rounds nanosleep
  128 + full rows. Slot resets after the output (MONO-clean). FP16 already (WB_H16G). Critical: part 3 (new block,
  merges last, every source landed) pays probe + rows = 2 RTT for its own rows; owners 0-2 wait for part 3's stores
  (+ <= one poll period). Fix XFE_NPL: part P-1 skips the probe (1 RTT). (WH job11 "no probe" for ALL parts was worse.)
- run8w cluster (v16s, attn_8_1_1_1x6), runE1 (d16s/d32s, attn_1_1_1_1e6/e3), XE (d1s xe_r1_1, v1s xe_b1s_8*):
  st.async pushes into the owner's smem + mbarrier wait; gates by cp.async / LDG before the wait. 0 L2 RTT on the
  end path; nothing of XA's list applies (XE's tail ~1 us is on-chip DSMEM + owner LDS / shuffles, XE NOTES).
- xa_b1f_1L (d1f, XB owns: report only): `par = __ldcg(ws + XW_PAR)` is loaded AFTER the bar.red.or (SASS: LDG
  right after BAR.RED.OR) and the red.add addresses depend on it -> the reds wait 1 RTT before issue; then MEMBAR
  drain (1) + ATOM (1) + xa1_epi's loads (shards + L + list count + gate together: 1) = 4 RTT; hoisting the par load
  into the prefix (safe: the last arrival rewrites it only after every CTA arrived) -> 3 RTT. The other-parity
  accumulator clear (tid 256..319) is drained by the same MEMBAR (parallel, no extra RTT).
## Measurements (Modal H200; same-job controls = two k39 copies)
- e1 (PUB / RCK / NPL; check all-on 0 failures): nsys d16f k39 87.81 / k39b 88.26 vs all 87.52; d32f 165.92 / 165.74 vs
  165.36; v32s 13.76 / 13.76 vs NPL 13.86 (bench v32s npl 13.04 / nplb 13.25 vs 13.14 / 13.13): NPL NEUTRAL -> off.
- e2 trace (d32 critical CTA, base): tiles 160.67 -> mstore 161.52 -> lsum 161.92 -> rck 162.59 (the redundant range
  scan = 0.67 us!) -> merged 162.72 -> pre-atom 163.15 (P / K divisions 0.43) -> atom-ret 163.79 (MEMBAR drain + ATOM
  0.64) -> comb0 164.10 -> combined 164.82. XFE_DIAG (double pass): the smem store block pass 0 1.37 / pass 1 0.75 us
  (median CTA; cold-code share ~0.6), the merge loop dry pass 0 1.08 = pass 1 1.05 (NOT i-fetch bound: the scan).
- e3 (PUB + RCK + PK, check 0 failures): bench d16f 85.46 / 85.44 vs k39 86.12 / 86.41, d32f 163.60 / 163.72 vs
  164.72 / 164.75; nsys d16f 87.23 / 87.18 vs 88.08 / 87.90, d32f 165.20 / 165.25 vs 165.81 / 165.95. Trace: segend
  -> pre-atom 0.16 (was 0.35-0.43), rck 0.0 (was 0.67); remaining: tiles -> mstore 1.44, lsum 0.54, drain + ATOM 0.90,
  publish -> comb0 0.26, combine 0.65.
- e4 FROZEN dev/xf/deliver_r3/k39xf.py (= build_ec.py k39 XFE_PUB=1 XFE_RCK=1 XFE_PK=1; 45 diff lines; check.py 0
  failures): bench d16f 82.49 / 82.53 vs 82.98 / 82.91 (-0.44), d32f 157.14 / 157.09 vs 157.71 / 158.13 (-0.8); nsys
  (60 reps) d16f 83.82 / 83.78 vs 84.46 / 84.64 (-0.75), d32f 158.05 / 157.98 vs 158.80 / 158.70 (-0.74); d16s
  7.26 / 7.28 vs 7.26 / 7.26; v32s 13.76 / 13.76 vs 13.63 / 13.66 (attn_8_1_1_1w4 SASS byte-identical to k39's:
  module-layout noise).
## Verdict / next lever
- WIN (small): run1f end chain -0.75 us nsys on d16f and d32f (XFE_PUB parallel arrivals, XFE_RCK one-shot range
  check, XFE_PK P / K off the end path). Below the 1.5 us full-shape bar.
- Remaining run1f end chain ~3.9 us: lane-partial smem store block 1.2-1.44 (cold first block of the end code: ~0.6
  of it), 4-lane merge 0.54, MEMBAR drain + ATOM 0.9 (2 RTT, inherent to red + release arrival), combine 0.9 (1 RTT +
  cold combine code: no CTA runs it before the last arrival). Next lever: lane-ordered pre-warm (early-finishing lanes
  of the CTA run the store block / merge / combine code dry, stores predicated off, while waiting for the slowest
  lane; the last lane skips it) - XE_PREW / XA_EWARM pattern; est. -0.4..-0.8 us. d1f: hoist XW_PAR (XB).

# ROUND 4 (coordinator): FLAT verify sliding B > 1 (v16s / v32s) from XD's verify-full kernel - KILLED AT BUDGET
Budget tool: dev/xf/mbs.py = copy of dev/xd/mb.py (XD's tile-phase microbench: equal flat slices over 132 CTAs, 2 lanes x
2 WGs, T128 xd_tile2x QV2 pairs, DST 5 x 16 KB, lean issue; NO lengths / split / page-table indirection (rows given as a
flat array), no RoPE / new block, no lane merge / publish / combine) fed with every request's window rows
(page_table[b, seq - 2047 : seq]); core = dev/xf/xdcore_mb.cu (snapshot of dev/xd/xdcore.cu md5 4a23dba1). Control =
dev/coord/k39.py kernel_fn (run8w w4 / x6) on the same data, same rep loop (graph replay after a 512 MB flush minus the
empty graph). Job s1 (40 reps):
  B=32 (65504 rows, 17 MB): T128 full 10.05 (CTA p10/50/max 7.8/8.3/9.2 us, 1815 MHz) | skeleton 7.20 | resident
       6.85 | run8w 13.04
  B=16 (32752 rows, 8 MB):  T128 full 7.62 (CTA 5.8/6.2/6.5) | skeleton 5.15 | resident 5.18 | run8w 9.17
Budget of a real FLAT kernel = mb + lengths -> split DRAM trip (~0.8-1.0; mb loads the rows directly) + RoPE / new
block (~0.3-0.5) + lane merge (~0.5-1) + global end chain (store / drain + arrival + combine RTT + 128-row epilogue,
>= 2-2.8 us; run8w's own global chain merged max -> out max 2.75, run1f W9D ~3) = >= 13.4 (v32s, vs 13.04) and
>= 10.9 (v16s, vs 9.17). Cannot beat run8w by 1 us -> killed before building (coordinator's criterion).
Why: at 2 (B=32) / 1 (B=16) T128 pairs per lane the tile phase is start-latency + compute bound and T128 buys
nothing over run8w's T64 streams (FLAT CTA p50 8.3 vs run8w WG done med 8.35, WH k32 trace); FLAT's only structural
gain (no part-3 new-block imbalance, ~0.9 us at the max) is smaller than its extra start trip + global end chain, and
at B=16 run8w's cluster DSMEM combine is far cheaper than any global one.

# ROUND 5 (coordinator, JUMP MODE): DECODE SLIDING B = 32 (d32s) in ONE wave - no win (best +0.2 us nsys vs k34)
Module (own NVRTC): dev/xf/xfd_help.cu (snapshot of dev/xe/deliver_r4 xer2 helpers, namespace xfh) + dev/xf/xfd.cu
(namespace xfm); builder `python3 dev/xf/build_ds.py <base> <out>` appends the module + wrapper (predicate: q_len 1,
17 <= B <= 33, tp8, page 1, sliding <= 2048 + RoPE, all clusters co-resident; XF_DS_K = dr (default) | ds | drb | dq |
dh; XF_DS_OFF). Tests dev/xf/edgeds.py (9 cases B 17..33, ctx 1..80000, mixed / agentic / edges / kv 1/64, 2 calls),
timelines + SM clock dev/xf/trace_ds.py (graph replay after flush; clock64 / globaltimer), dev/xf/mk_e1t.py (runE1 clock).
Every variant: edgeds 0 failures, tools/check.py 0 failures. d32s (us; same-job k39 controls ~8.3-8.6 bench, 9.22-9.31 nsys):
- ds = xf_ds_8_4_2: XE's xe_dm<8,4,2> with the prep warp removed (each warp builds its own Q^T in registers after its
  K/V issue, warp 0 owner); 94 regs, 128 thr, 77 co-resident clusters (one wave). bench 9.27 / 9.13 (XF_MPRE=0 8.78),
  nsys 10.18 / 10.08 (MPRE=0 9.76): +0.4..+0.9.
- dr = xf_dr_16: one cluster of 16 per REQUEST (both KV heads in every CTA: each 128-B row fetched by one SM), warp =
  (head, slice), rank r owns q-head r; 96 regs, 35 clusters. bench 8.74 / 8.71 / 8.59 / 8.54, nsys 9.47-9.57: BEST,
  +0.2..+0.25 vs k34. Trace (graph): pt issued 0.80, K/V issued med 2.08 / max 3.84 (late CTAs: per-SM in-flight
  capacity), qk0 4.29 / 6.13, loop 6.66 / 8.85 (pass 1 = 2nd serial DRAM trip), pushed 7.41 / 9.30, out 8.68 / 9.74.
- drb (both passes in registers): 124 regs -> 4 CTAs / SM -> 28 clusters of 16 < 32: second wave, not run.
- dq (one full-row cp.async burst into a 32-KB swizzled stage, both passes from smem): 78 regs, 40 KB smem; data lands
  like XC's skeleton (qk0 max 6.30) but the 20 warps / SM then compute at once (qk0 -> loop 1.4 / 2.3 us) and the
  200-KB carveout leaves a 28-KB L1: bench 10.11 / 9.95, nsys 10.85 (+1.6).
- dh (dr + pass 1 by full-row cp.async + prep before the issue): the cp.async burst behind the half-row LDGs deepens
  the LSU backlog (K/V issued med 4.24 / max 5.82): bench 9.55 / 9.41, nsys 10.38 / 10.43 (+1.1).
- Carveout (XF_DS_CARVE % on dr): 37 / 44 / 58 / 72 = default (9.49-9.57); 100 (L1 28 KB) nsys 11.55, and the traced
  CTA timeline itself is +2.2 us (out max 11.90 vs 9.74): register-direct gathers need L1 capacity (not a launch cost).
- MMA clock (coordinator's lever): MEASURED: SM clock with HMMA 1.69-1.78 GHz (runE1 1.72-1.77), timing stub without
  MMA (XF_NOMMA: HMMA -> dependent FFMA, wrong output) 1.89-1.90 GHz and -0.5 bench / -0.64 nsys on ds. A real
  CUDA-core QK + PV is ~2.1 M MAC per (request, KV head) = ~16 k HFMA2 warp-instructions per SM (+ conversions /
  reductions ~3.5 us of issue at 20 warps / SM) vs ~0.25-0.5 us estimated: it costs more than the clock returns. Not built.
Verdict: d32s stays on k34's runE1. The limit is the per-SM in-flight request capacity x the loaded DRAM latency with
two serial trips per warp (or a compute burst when everything lands at once); every one-wave layout that fits the
register file (<= 96 regs at 5 CTAs / SM for 64 / 32 clusters) keeps a 2nd trip. Next lever (not built): dq with
pass-pipelined cp.async groups (pass 0 first, compute while pass 1 lands) AND <= ~164 KB smem per SM (e.g. FP8 stage
for one pass + register pass) - est. at best parity with k34 given dh's issue backlog.

# ROUND 6 (coordinator): VERIFY SLIDING B = 32 (v32s), base dev/coord/k44.py (run8w global path attn_8_1_1_1w4)
Own module: `python3 dev/xf/build_vs.py <base> <out>` appends a block that, at import, patches a copy of the base's
SRC (run8w end chain; asserts every anchor) + `xf_vs_w4` (= run8w<1, 4>), compiles it with the base's _Mod, and
redirects only the base kernel_fn's launch of attn_8_1_1_1w4 (17 <= B <= 33) to it (same grid / smem / args / slots).
Knobs XFV_LC / XFV_NAP / XFV_PP / XFV_PPS / XFV_NONEW (timing only), XFV_OFF. Tests dev/xf/edgevs.py (8 cases,
asserts xf_vs_w4 ran), timeline dev/xf/trace_vs.py (run8w TR slots per part).
- v1 k44 timeline (graph after flush, med / max over requests): tile0 parts 0-2 4.5, part 3 5.70 (new block first);
  merged 8.7-9.0 / part 3 9.75 (max 10.27); published 9.3 / part 3 10.30 (10.93); all-in 12.0 / 12.6 for EVERY part
  (incl. part 3's own owners, whose sources published <= 10.0); out 12.4 / 12.9; nsys 13.78.
- v2 knobs: WH_PFN=1 (fence after the publish): published 9.35 -> 10.55 (parts 1-2), 10.23 -> 11.13 (part 3): the
  publish stores take 0.9-1.2 us to be acked; all-in unchanged. WH_LDK=1, WH_NAP=0, WH_PROBE=0: unchanged; WH_PFX=0
  +0.5. => the owner chain = store visibility ~1 us + poll detection ~0.9 us (2 rounds median).
- v3 XFV_LC (last part owns all 128 rows, one-round loads, tight retry): correct (edge / check 0 failures) but nsys
  15.33 / 15.42 vs 13.73 (bench 14.94 vs 13.1): one round of 3 x 128 rows into ONE SM took 2.6 us, its 512-thread
  epilogue 1.1 us (per-SM in-flight limit, as d32s) = WH's LOWN verdict confirmed. Off.
- v4 XFV_NONEW (timing only: part 3 skips the new block = upper bound of moving it): part 3 balanced (merged 8.94,
  published 9.34) but all-in still 11.7 (2.4 us after the publishes): nsys 13.44 / 13.50 vs 13.65-13.70 = -0.2 only.
  => the cost is the global hop itself, not part 3's lateness.
- v5 XFV_PP (pipelined owner poll after round 1: one pending source at a time, PP staggered full-row loads in flight
  per thread): correct (edge / check 0 failures) but nsys PP4 15.04 / 15.06, PP2 13.89, PP4 stagger 200 ns 15.95,
  50 ns 14.88 vs k44 13.66 / 13.79 (bench PP4 14.29 vs 12.86 / 12.94); trace PP4: all-in 12.5-12.9 (WORSE than the
  base's 12.0) with 1 round: denser polling of the slot lines delays the publishers' stores becoming visible.
- v6 base WH_NAP sweep (less polling): 300 13.82, 600 14.29, 1000 14.27 vs 13.73 / 13.76: the base's 128 ns is optimal.
Verdict round 6: NO WIN; v32s stays on k44's run8w. The ~2.4 us publish -> all-in hop is L2 store visibility under
contention (acks 0.9-1.2 us + one detection round) and neither the poll schedule (PP / NAP), nor moving the combine to
one CTA (LC: per-SM in-flight limit), nor balancing part 3 (NONEW bound -0.2) shortens it; target 12.0 needs a DSMEM
(cluster) combine, but only 30 C4 clusters co-reside (B 32 needs 32; XD). Untested structural idea: hybrid launch
(30 requests on C4 clusters, the remaining 2 on the global path with more parts on the 12 leftover SMs); needs a C4
cluster layout that fits smem (T 8 stages + FP16 receive ~225 KB), a P = 6 global variant and two concurrent kernels
in the graph (fragile CTA placement).
Frozen: dev/xf/deliver_r6/ (build_vs.py, edgevs.py, trace_vs.py, jobv1-v6 .sh / .log); nothing to integrate.

# ROUND 7 (coordinator): physical-row-ordered gathers for FULL layers - PROBE (dev/xf/probe_sort.py, frozen deliver_r7)
Gather-only, dev/xf/probe.py's run1f-like lane3h / lane2h (132 CTAs x 4 lanes, 16-KB stages, evict_first), real d16f /
d32f rows (page_table[b, :seq] flattened), graph replay after the 512 MB flush minus an empty graph, 30 reps.
- p1 (TB/s = K+V / time): d16f 320.8 MB: orig 78.02 us 4.11 | lanesort 73.41 4.37 | ctasort 73.39 | global 73.44;
  d32f 649.7 MB: orig 157.31 4.13 | lanesort 148.29 4.38 | ctasort 148.48 | global 148.03 (lane2h the same).
  => -5.7..-5.9 % (d32f -9 us, d16f -4.6 us); a globally coordinated tight band adds NOTHING over per-lane order.
- p2 granularity (each lane's slice stably bucketed by row >> K): d32f orig 157.41 | full sort 147.36 | K 6 147.47 |
  K 10 147.15 | K 14 (2 MB of K rows) 147.18 | K 18 (32 MB) 149.39; d16f 78.00 | 73.54 | 73.62 | 73.55 | 73.46 | 74.78.
  => the gain is PAGE (TLB) locality within a lane: 2-MB buckets give all of it, 32-MB ~80 %; no exact sort needed.
Budget (run1f, not built): per segment (one request's positions; full layers: no RoPE / no past mask -> order free),
counting sort by row >> 14 (~170 buckets at d32f's 2.75 M cache rows) into an L2-resident workspace (~77 KB / CTA,
10 MB total; KV loads are evict_first so it stays in L2; written back during the next flush), done by each CTA right
after it issues its first DST tiles from the original order (first ~768 tokens per CTA unsorted), i.e. in the ~4 us
start bubble; the tile loop then reads rows from the workspace. Costs: all page-table entries read up front (10 MB:
~2.3 us of DRAM inside the idle start), histogram + scan + scatter ~1-2 us of SM time under the first tiles' latency.
Est. net d32f -7..-9 us, d16f -3..-4 us. Verify full uses the same gather (XC / XD kernels): same lever, maybe more
under the power cap. A separate pre-pass kernel (read + write 10 MB + kernel boundary ~5-7 us) only pays at d32f.

# ROUND 8 (coordinator): run1f with physical-row bucket order - BUILT, MEASURED, KILLED (bound says no)
Builder `python3 dev/xf/build_srt.py dev/coord/snap/k34xf.py <out>` (frozen deliver_r8; Params.xsr / xsrf appended,
workspace int32 [hp][G][cap] host-allocated, cap = ceil(B (max_pages page + ovh) / G) + 64; XF_SRT 0 default = off,
1 in-kernel sort, 2 pre-pass kernel xf_srt_pre). Tests dev/xf/edgesrt.py (11 cases: B 2..128, ctx 1..100K, page 64,
mixed / agentic, kv 1/64): 0 failures; tools/check.py 0 failures with XF_SRT=2 and default (job s14).
- s8 / s9 (mode 1: 2-MB buckets of each CTA's first 4 segments after its first 768 tokens, sorted after the prologue
  burst into the workspace): nsys d16f 105.2-105.6 vs 87.1-87.3, d32f 202.7-204.9 vs 164.9-165.4 (XF_SRT=0 = base).
  s10 / s11 trace: the sort block runs 7.0 -> 47.7 us: pass 1 (page-table loads behind the 25 MB start burst +
  histogram) 8.8 us, pass 2 (19K scattered 4-B global stores per CTA into the workspace) 31.5 us. CH 4/8/16 the same.
- s12 / s13 (mode 2: pre-pass kernel, all rows bucketed in smem, coalesced copy-out; run1f reads every sorted
  segment from the workspace): nsys d16f 94.4-94.6 vs 87.2, d32f 175.4-175.6 vs 164.9; per kernel pre-pass 10.8 /
  15.8 us + run1f 83.4 / 159.3 (= the sorted order's REAL in-kernel gain: -3.8 / -5.6 us, vs -4.5 / -9.7 gather-only).
  Pre-pass anatomy: ~6 us fixed (lengths + page-table DRAM trips + launch) + ~0.8 cycles / key / SM (smem atomics
  with conflicts + scatter) -> it can never be < ~6-8 us at d32f. (s13 also fixed: launch smem must be constant.)
- p3 gather-only ceiling of a cheap coarse partition (row >> K buckets): d32f orig 157.02 | K14 147.33 | K18 148.96 |
  K19 (6 buckets) 151.87 | K20 (3) 154.35 | K21 (2) 156.00; d16f 78.11 | 73.65 | 74.69 | 75.98 | 77.07 | 78.11.
- Bound (with XB's verify-full numbers: start delay costs ~1:1): PDL cannot overlap a pre-pass (run1f fills every
  SM's smem); lazy in-loop sorting has no slack (lanes ~96 % busy) and its scattered workspace stores compete with the
  K/V stream; a ballot-cheap coarse partition before the burst keeps <= ~half the gain (-5.2 / -2.1 gather-only,
  ~-3 / -1.8 realized) minus >= ~1 us of sort -> net <= ~-2 / -0.8 us. KILLED (target was -7 / -3).
- Sliding (coordinator's side probe, s13): B 32 x 2047 window rows (16.8 MB over the 705 MB cache): orig 5.76 us,
  lanesort / ctasort / global / bkt14 5.68-5.74: <= 0.08 us -> nothing for XD / XE (2.4 % density: ~0.7 rows per
  2-MB page per CTA).
- s15 (mode 3, coordinator: bucket in smem BEFORE the burst; every key by one 4-B cp.async into the idle ring, per
  segment per-warp private histograms [16][256] + block scan + scatter in smem, coalesced copy-out to the workspace,
  prologue rows read from smem, one barrier, then the burst): edge / check 0 failures. nsys d16f 89.54 / 89.54 vs
  87.09 / 87.22 (+2.4), d32f 167.38 / 167.31 vs 165.26 / 165.06 (+2.2); bench +2.9 / +2.6..+3.1. Trace d32f: the sort
  runs prefix 0.79 -> 13.92 us (pt-iss), burst issued 16.88 (base 6.93), tile0 18.51 (8.22): +10.3 us start, but
  tiles-done med 160.16 vs 158.46 (+1.7): the bucket order RECOVERS ~8.6 us of tile phase at d32f (= the probe's
  -9.7). Blocker = sort cost ~0.8 cycles / key / SM with smem atomics + scattered smem stores (~13 us at 19.4K keys /
  CTA, same as the pre-pass). It would pay only below ~2 us (<= ~0.15 cycles / key). KILLED (rule: >= 1 us on d32f).

# ROUND 9 (coordinator): mode 3 with an atomic-free one-pass block rank - MEASURED SLOWER, KILLED
xf_bucket32<NT>(A, B, E, cnt, scr, dig, tid) in build_srt.py's TABLE_FN (self-contained, portable): CUB-style counter
lanes (digit d -> row d & 15, half d >> 4 of packed 16-bit counters [16][NT]; thread-private columns, bank = lane),
raking exclusive scan over the linear (row, thread) order + warp-shuffle block scan + high-half fix-up = digit-major
ranks, one smem store per key. Digits: (segment << (5 - nsb)) | (row >> shb), 32 (smem: keys 2 x 78 KB + counters
32 KB in the 192-KB ring). Knob XF_SRT_RK (1 rank, 0 per-warp histograms + atomics). edge / check 0 failures (s16).
- s16 nsys: d16f 90.80 / 90.80 vs 87.26 / 87.06 (+3.6), d32f 169.66 / 169.58 vs 165.14 / 165.14 (+4.5); bench +3.8 / +4.8.
- s17 anatomy (d32f, med CTA, us): prefix 0.80 -> table 2.62 -> keys landed (19.4K 4-B cp.async) 6.53 (+3.9) ->
  bucketed 13.72 (+7.2; RK 0: 11.48, +5.1) -> copy-out (coalesced 4-B STG) 15.63 (+1.9) -> burst issued 19.8.
  The rank's per-thread counter RMW chain (LDS -> add -> STS on its own column) is latency-bound with 4 warps / SMSP:
  ~0.65 cycles / key / SM, slower than the contended atomics (0.46). The bucketing alone would need ~0.1 cycles / key.
  Cheap fixes (16-B key loads ~-2.6 us, vector copy-out ~-1.4) leave a ~9-10 us start delay vs ~7.3 us of tile gain
  (16 buckets / segment): est. d32f ~-1.3, d16f ~+0.5 -> below the keep rule (d32f <= -2 and d16f <= -1). KILLED.
- s18 (coordinator's final bounded attempt; mode 3 = 16-B cp.async key loads of each segment's aligned page-table
  superset (XF_SRT_V16) + warp multisplit bucketing xf_msplit32 (XF_SRT_RK 2: __match_any_sync per 32-key group, the
  leader does one plain RMW on its warp's private counter row [16][32], digit-major block scan, rank = base (shfl
  from the leader) + popc(mask & lt), one STS per key) + 16-B copy-out): edge / check 0 failures. nsys d16f 98.40 /
  98.05 vs 87.12 / 87.46 (+11), d32f 183.34 / 183.59 vs 165.07 / 164.93 (+18.4); bench +11.4 / +18.9.
  Anatomy d32f: table 2.38 -> keys landed 5.04 (+2.7: 16-B loads DID help, was +3.9) -> bucketed 28.50 (+23.5 !)
  -> copy-out 30.18 (+1.7, was +1.9); d16f bucketing 4.13 -> 16.72 (+12.6). __match_any_sync is very slow on this
  part (~1.2K matches per CTA cost ~20 us at d32f). Best remaining combination (16-B loads + atomic bucketing 5.1 us)
  projects to ~+0.8 us at d32f -> the lever is CLOSED: the d32f tile phase gain of the page order (-8.6 us) cannot be
  bought with any in-kernel bucketing measured here (atomics 0.46, counter lanes 0.65, multisplit ~2.2 cycles / key /
  SM; needed <= ~0.1-0.15).

# ROUND 10 (coordinator, new user rule: stack every correct change that is faster in nsys, however small), base k51
Scan of rounds 1-9 for unfrozen variants that ever measured faster: round-2 run1f start levers XFR_PF1 (3 / 6 / 9) and
XFR_W 2 (within noise on one shape), XFE_NPL (v32s bench -0.1 / nsys +0.1), run8w owner-poll knobs (WH_NAP / WH_LDK /
WH_PROBE: traced only or back-off >= 300 slower). Everything else in rounds 1-9 measured slower or was timing-only.
dev/xf/build_r2nokl.py = the round-2 builder without its XFR_KL patches (anchors changed by XFE_RCK; KL lost anyway).
- r10a screen (nsys 60 reps, same job): run1f d16f / d32f k51 87.26 / 165.02, k51b 87.31 / 164.86, no-op patched
  87.31 / 164.91 | PF1 3 87.78 / 165.31 | PF1 6 87.57 / 165.42 | W 2 87.62 / 165.22 -> all slower, dropped.
  v32s k51 13.78 / 13.73 | XFE_NPL 13.84 | WH_NAP 0 13.63 | WH_NAP 64 13.65 (p90 13.86 vs 14.82) | WH_LDK 1 13.86 |
  WH_PROBE 0 13.95 -> only the smaller back-off is faster.
- r10b verify (k51x = build_r10.py k51: WH_NAP 128 -> 64): check.py rc 0, edge2 0 failures, edgevs 0 failures;
  nsys 2 copies x 2 passes: v32s k51 13.73 / 13.73 / 13.76 / 13.76 (mean 13.745) vs x 13.65 / 13.73 / 13.63 / 13.66
  (mean 13.67, -0.08 us); WH_NAP 0 13.70 / 13.68 (-0.06); d16f 87.16 vs 87.20, d32f 165.13 vs 165.07 (run1f SASS
  unchanged: noise). STACKED: WH_NAP 64 (frozen dev/xf/deliver_r10/: build_r10.py + k51x.py; independent of XE's XEL).
