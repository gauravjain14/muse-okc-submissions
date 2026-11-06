# base worker, round 2 (combine phase), start = runs/k77.py (work/base/cand2.py)

## v32s global combine anatomy (trace build, per owner warp, 10 replays; times in us from the first CTA start)
P = 4 CTAs per (request, head pair); each CTA owns 2 tokens x 16 heads = 32 rows, done by its 4 owner warps
(warp (mt, s) with j = 2 mt + s, j >> 1 == part); 12 warps per CTA only publish. 4 partials per row (3 remote).
Protocol: non-owners write FP16 O/l rows + (m, l) to the owner's slot region (plain stores, 0xffffffff sentinels),
owners poll (round 1: one probe word per source, later rounds: full rows validated word by word, nanosleep 64),
fold, normalize x sigmoid(gate), store, reset their slots to the sentinel. No fence / atomic anywhere.
| event (median / p90) | |
|---|---|
| owner WG done | 7.78 / 8.45 |
| owner stream merge done (xch swap waits for the partner stream WG) | 8.35 / 8.96 |
| last of the 3 sources published (max over sources) | 9.02 / 9.47 |
| source: merged -> published (FP16 convert + stores) | 0.33 |
| owner: last source published - own merge (imbalance) | 0.74 / 1.34 |
| owner: folded - max(last publish, own merge) (protocol: L2 propagation + detection + loads) | 1.76 / 2.30 |
| owner: rounds (probe round included) | 2 (max 4) |
| owner: stored - folded (norm, 16 sigmoids, stores) | 0.61 |
| owner: slot reset issued | +0.38 |
| kernel end (last CTA) | 12.50 |

## v16s cluster combine anatomy (C = 6, pull)
| event | median |
|---|---|
| WG done -> stream merge | 0.45 |
| merge -> staged + cluster arrive | 0.38 |
| acquire -> all DSMEM loads landed | 0.96 (p90 1.12) |
| landed -> stored | 0.29 |

## DSMEM microbenchmarks (H200, 96 CTAs)
- pull, cluster 6, 352 threads x 6 sources, row-contiguous 8-B loads (the v16s pattern): 1208 clk for the O loads,
  641 clk for the (m, l) broadcast loads; scattered 8-B loads (144-B stride) 6839 clk.
- push, cluster 6, 3168 B to each of 5 peers: st.async v4 from 512 threads 1266 clk until every incoming byte landed,
  one bulk copy per peer 1386 clk. Cluster 3, 6192 B per peer: 1031 / 1152 clk.
- => DSMEM moves only ~12-14 B/clk per SM here: the v16s exchange (~14.5 KB per CTA) is bandwidth-bound at ~0.6-0.9 us.

## v32s combine changes (ab.py, 6 seeds x 30 reps, slim files)
- WH_PROBE=0 (full-row poll from round 1, no probe round): -0.21 / -0.29 (two runs).
- BZ_PP (kept): global-path owners (P = 4) keep two full-row poll batches in flight (the next one issued before the
  current one is consumed), gate sigmoids computed while the first batch is in flight, slot resets issued before the
  output epilogue: v32s 17.08 -> 16.62 (-0.46); stagger 0 / 160 / 320 ns: -0.45 / -0.46 / -0.50. 16/16 tests.
  Re-run: 17.04 -> 16.56 (-0.48).
- Tried: W3_CO=1 +0.22, WH_NAP=0 +0.04, XFE_NPL=1 +0.24, owner gate loads issued before the stream swap +0.5 (vs PP).
- BZ_SPL (token split 512 | 544 | 544 | 447 + new block, so the last part, slowest because it also attends the new
  block (trace: its stream-1 WG ends ~0.6 us after the others, last source in 52 % of the cases), owns fewer past
  tokens; parts 1/2 use the new-block tile buffer for a 9th tile): v32s +0.32 (X = 32) / +0.17 (X = 48) vs PP alone
  -> removed.
- v16s: one DSMEM (m, l) load per row + shuffle (BZ_ML1): +0.28 -> removed (microbenchmark agrees: divergent
  single-lane DSMEM loads 837 clk vs 644 for the 16-lane broadcast). 16-B loads x 8 lanes vs 8-B x 16 lanes: same
  (1229 vs 1305 clk).

## d32s combine anatomy (runE1, C = 3, trace build; us after the last WG of the CTA finished)
| step | median |
|---|---|
| WG0: wait bar 7 + merge of the 4 WGs' fragments | 0.48 |
| WG0: push to the owners (st.async) | 0.16 |
| owner: receive mbarrier complete after the cluster's last push | 0.19 |
| owner: combine 3 sources, RMSNorm x sigmoid, store | 0.42 |
- d32s ablations (ab, invalid results): skipping the merge math of the 4 warpgroups' fragments (still waiting on the
  barrier) -0.17; skipping wait + merge +0.19 (?). The 4-WG merge math costs ~0.17 us; pushing per WG instead would
  quadruple the DSMEM bytes (bandwidth-bound) -> not pursued.
- BZ_LA (v32s): last-arrival finisher (every CTA publishes all 128 rows, an atomic per (request, head pair) picks the
  fourth arrival, which folds the 3 other partials of every row; counter in the slot buffer's tail, reset by the
  finisher): v32s +1.78 vs k77 (PP alone -0.35 in the same run) -> removed. One SM loading 3 x 17 KB + the atomic
  round trip is far slower than 4 owners polling 1/4 each.
- v16s push mode anatomy (XDV_PULL=0, trace): merge -> pushes issued 1.06 us (st.async issue throttled by DSMEM
  bandwidth: ~17 KB per CTA incl. its own rows), receive complete +0.19 after the cluster's last push, fold + store
  +0.64 -> 1.9 us after the last CTA's merge vs pull: stage 0.38, DSMEM loads 0.93 after the acquire, store 0.29.
  Both are bound by the ~12-14 B/clk DSMEM rate; the exchange volume (~14.5 KB / CTA, FP16 O + (m, l)) cannot shrink
  without lower-precision partials or a different decomposition (per KV-head split = half-row gathers, slower).
- v16s gate sigmoids computed while the DSMEM loads are in flight (BZ_SG): +0.06 vs without -> removed.
- Re-run of BZ_PP (6 seeds): v32s 17.02 -> 16.58 (-0.44).
- Store / load flavours for the v32s slot protocol (on top of PP): WH_PST=1 +0.09, WH_PST=2 -0.02, WH_LDK=1/2 +0.02.
  Cross-CTA visibility microbenchmark (writer stores, reader polls ld.relaxed.gpu, 128 CTAs): ~1.5 us from store to
  observed (rounds ~0.65 us) -> the protocol latency is mostly L2 round trips, not the protocol itself.

## Result round 2 (work/base/cand2.py = runs/k77.py + BZ_PP)
- ab.py full files, same session, 6 seeds x 30 reps: v32s k77 16.95 -> 16.44 (-0.51); v16s 13.22 -> 13.12 (-0.11,
  kernel unchanged = noise); d32s 13.91 -> 13.92.
- bench.py: 16/16 tests; d32s 13.20, v16s 12.72, v32s 16.09.
- Nearby shapes (near.py, each twice to check the slot sentinels are restored): verify sliding B 17/20/24/32/33 x
  (ctx 80000 agentic, 3000 mixed, 1500 fixed), B 2/9/16, B 24 edges, B 20 kv_scale 1/64: all pass.
- Merge: the diff `diff runs/k77.py work/base/cand2.py` (95 lines): BZ_PP / BZ_PPD defines before w8_tile's BZ_FL
  block, and an `#if BZ_PP` block at the start of run8w's W3_CO=0 owner poll (`#else` branch after the W3_CO path).
  No change inside w8_tile.
