# base worker, round 4: make w8_pair pay off end to end (start: work/base/t/cand3_pairs.py)

Changes tried in w8_pair / the run8w loop (work/base/t/cand4_pairs.py):
- (2) tile 0's PV issued before tile 1's P is built (two commit groups, each with its own warpgroup.arrive).
- (3) entry barrier only after a single-tile step (a pair's exit barrier already freed both V^T buffers).
- (4) static decision, no runtime check: pairs only in kernels whose streams have >= 4 tiles (T >= 8: the v32s
  global path); v16s (C6, <= 3 tiles per stream) is unchanged (no pair code executed).
- (1) spills: with pairs in every step the w4 kernel stays at STACK 48 / 12 LDL-STL; pairing only the stream's first
  two tiles (BZ_PR=1 below) brings it back to the k80 count (5 LDL/STL, no reload at the combine entry).

A/B (ab.py, slim files, 6 seeds x 30 reps), v32s: k80 16.71; pair only the first two tiles of each stream +0.84; pair
every step (all improvements above) +2.05; pair code compiled but off +0.05.

Diagnosis (trace build, v32s, same session):
- The first tile of stream 0 lands on time with pairs (slot 4 4.61 vs 4.51 us). (A clock64-instrumented build showed
  it 2 us late: that was an instrumentation artefact - register allocation changes - and is not real.)
- clock64 around the calls (warpgroup 0): a pair costs 1650-1894 clk = 825-947 per tile vs 1098 per single tile, and
  WG0's loop span is shorter (4734 vs 5268 clk).
- Yet the streams finish later: median warpgroup-done stream 0 7.65 vs 7.36-7.49 us, stream 1 8.16-8.19 vs 7.57-7.62,
  and the slowest CTA 9.9 vs 8.4 us. Stream 1 (the early-starting Q-prep warpgroups, which in the last part also
  attend the new block) is hit hardest. Static pairing makes a stream wait for its second tile and adds a full tile of
  compute after the last landing; the back-to-back QK/PV bursts of a pairing warpgroup also seem to delay the other
  warpgroups' wgmmas on the shared tensor pipe (per-WG clock64 shows the pairing WG faster, the other stream
  slower). Even a single pair per stream (no tail effect) loses 0.84 us, so the per-WG instruction savings do not
  turn into SM throughput: the 4 warpgroups' tensor/issue contention, not one warpgroup's instruction count, sets
  the pace.

Result: no change kept (work/base/cand4.py == runs/k80.py). The cluster-of-3 v32s path was not revisited (it needs a
faster end-to-end tile loop).
