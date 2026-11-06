# WH: verify sliding batch 16 / 32 (q_len 8, window 2048, RoPE) — base dev/k19.py

Deliverable: dev/wh/k20h.py, built by `python3 dev/wh/mk_k20h.py [dev/k19.py] [dev/wh/run8w.cu] [dev/wh/k20h.py]`
(anchor asserts; NVRTC program k20h.cu). New kernels attn_8_1_P1_1w{8,4} = run8w<P1, P> (dev/wh/run8w.cu); every
other kernel's source unchanged. K2_DEFS: WH_V8W=0 restores k19; WH_ALLB forces run8w for every eligible B (tests).
Tools: t_v8w.py (extra q_len-8 B>1 cases, forced run8w, 3 calls each, checks the slot buffer is all-sentinel after
every call), trace.py (WS's, my stamp names), t_fence.py (fence vs in-flight cp.async microbenchmark), job*.sh.

## Open question (settled): no duplicate K/V bytes
WB round 13 (dev/wb/NOTES.md): b32 16.8 MB, b16 8.4 MB; part 0 had one extra tile from the TSPLIT split.

## run8w design (dev/wh/run8w.cu header)
- grid (B * P, hkv / 2), P = 8 (B <= 16) / 4 (B <= 33); part i = whole 64-token tiles [64 T i, 64 T (i+1)) of the
  window (T = 32 / P), all T stages resident (no ring, no refill), the last part also builds + attends the new block.
- 4 independent warpgroups WG (h, s): head h, stream s = tiles s, s + 2, ...; tile math = run8's wg_tile (unchanged).
- start: straight-line predicated loads, mbarrier per tile, after the length every thread loads its token's
  page-table entry of all T tiles at once, RoPE table, then issues its K / V chunk of every tile.
- tail: stream merge by a register swap of row halves (one 256-thread barrier per head), then each merged row goes
  to its owner part as FP32 (O, m, l) in a self-resetting sentinel slot buffer (no flags: complete = no word is
  0xffffffff); owners poll, fold, reset, epilogue (WS's side-by-side RMSNorm form).

## Log
- job1 (k19 fence ablation, 40 reps x2, us): v16 k19 11.68/11.77, dup 11.92/12.09, K8_FENCE_CP=0 11.85/12.03,
  K8_FENCE_VT=0 11.76/11.83, both 11.53/11.73; v32 k19 14.22/14.17, dup 14.22/14.53, CP0 14.25/14.37, VT0 13.97/13.93,
  both 13.73/13.82. WS run8g (dev/wh/k20s_ws.py: balanced split + new tile first): v16 12.00/11.82, v32 14.22/14.29
  (no gain). => run8's lane step (~1.1-1.25 us) is compute, not a fence waiting for refills.
- job2 (t_fence.py, 132 x 512 threads, random rows of a 2 GB buffer): fence.proxy.async does NOT wait for the
  thread's in-flight cp.async (wait_all after it still 2.6k-5.7k cycles); bar.sync and membar.cta neither;
  fence.acq_rel.cta DOES (op = whole burst). Issue of 16 x 16-B cp.async per thread took ~43k cycles there (TLB-bound
  on 2 GB; not representative of the 350 MB caches).
- job3 (first run8w: every thread issues every tile; owners poll full rows): correct (check 0 failures) but
  v16 15.0 / v32 17.7 (k19 11.9 / 14.5). Trace: owners all-in 6.5 us after publish (polling every pending
  source's full rows each round saturated L2).
- job4 (lane-parallel probe word, then rows): v16 13.3, v32 15.3. job5: publish-store variants (st.relaxed.gpu,
  st.global.cg, + fence.acq_rel.gpu): all ~equal; the fence shows the publish stores take ~2.3 us to be acked.
- job6 (slot rows 288 B = whole 32-B sectors, (m, l) in a full sector, resets after the poll): v16 12.0-12.2,
  v32 14.15-14.23 (k19 11.7 / 14.0). Partial-sector writes + reads of the same sectors were costing ~1.3 us.
- job7 (WH_PFX: the owner prefetches its 36 KB slot region into L2 at kernel start, cp.async.bulk.prefetch.L2):
  v16 11.80/11.70 vs nopfx 12.24/12.38, k19 11.92/11.94; v32 14.42/14.24 vs nopfx 14.66/14.53, k19 14.35/14.37.
  t_issue.py (fixed): cp.async 64 KB/CTA (132 CTAs, random 128-B rows of 352 MB): 1 warp issues in 5.2 us,
  2: 2.8, 4: 1.6, 8: 1.0, 16: 1.0; landed 5.7 / 3.5 / 3.1 / 3.0 / 2.9 us. 128 KB: 4 warps 5.1, 16 warps 5.0 us.
  TMA (cp.async.bulk.tensor, box 128 B x 1 row, 128B swizzle; placement verified correct): 64 KB takes 13.7 us from
  one warp => row-granular TMA gathers are DEAD.
- job8: warp-specialized producers (640 threads, setmaxnreg 32 / 112, WG4 issues): v16 12.2-12.3, v32 14.6-14.7 (LOSS:
  length barrier 1.57 vs 0.8 us - setmaxnreg.inc stalls the consumers until the producers dealloc; 4-warp issue
  slower). 512-thread (consumer-issued) with the 2-RTT poll + resets after output: v16 11.35/11.32 vs k19 11.92/11.88.
- job9/10: staged issue (later tiles issued one step ahead by their stream): worse than issuing every tile at once
  (v16 p8_4 11.48/11.41 vs staged 12.01/11.79; v32 all-at-once 13.86/13.93 vs 14.14/14.17). Probe RTT under load
  ~0.65-0.7 us (stamps around one relaxed load): the global end chain is ~1 store + 2 RTTs ~ 2 us after the last
  publish (merged max -> out max ~3.5 us).
- job11 (40 x2): Q fragments + RoPE table BEFORE the K/V burst (WH_QEARLY): v16 11.26/11.58, v32 13.63/13.88 vs
  k19 11.79/12.01 (dup 11.87/12.02), 14.13/14.37 (dup 14.18/14.33); after the burst 11.77/12.00, 14.44/14.72.
  No probe (rows from round 1) 12.36 / 14.93 (worse); nanosleep 0 / 32 ~ equal to 128. check.py 0 failures.
- job12-14: per-request CLUSTER variant for B <= 16 (run8w<P1, 6, 6>: one cluster of 6 CTAs per request, 96 CTAs,
  whole-tile split 5/5/6/5/5/6 tiles + new block in part 0 on stream 1, merged rows pushed by st.async from registers
  to owner CTAs (head-major rows R = 64 h + 8 j + g, 21-22 rows per owner), owner folds 6 partials + epilogue).
  v16: 10.68/10.52 (job12, new block on stream 0) -> 10.25/10.40 (job13, new block on stream 1) vs k19 11.85/11.89.
  Bulk push (rows staged per head in the swap area, one cp.async.bulk per (head, owner), 2 extra barriers):
  10.86/10.67 vs st.async 10.39/10.28 -> st.async kept (WH_BULK=0). Global path b16 with Q early: 11.20-11.45.
  Trace (b16 C6): length barrier 1.4 us (0.8 on the global path; not the mbarrier-init fence, not the L2 prefetch),
  tile0 3.7, WG done med 6.5-6.9 / max 8.1-8.4, push 0.85 us med, all-in 9.3-9.8, out 9.9-10.3.
- job15 (FINAL k20h = dev/k23.py + run8w; B <= 16 -> C6 clusters, 16 < B <= 33 -> global owner path):
  t_v8w 2 seeds 0 failures, global-only 0 failures, check.py 0 failures.
  verify sliding 40 x2: v16 k23 11.73/11.70 (dup 11.78/11.69) -> k20h 10.44/10.48 (dup 10.27/10.25);
  v32 k23 13.97/14.18 (dup 14.01/14.05) -> k20h 13.79/13.78 (dup 13.85/13.81). WH_CLA=1: v16 10.31/10.22 (noise).
  All loads 20 reps: geomean x2.6700 -> x2.6850 (judge-proj x2.4925 -> x2.5060); other shapes noise.
- job16: run8p (dev/wh/run8p.cu: one cluster of C per (request, KV head), 64-B half rows with a 64B-swizzle stage and
  its own tile function wh_tile64 = wg_tile NB 2 EXQ; WG pairs merge through their idle V^T buffers; st.async to
  owners). Correct (t_v8w 0 failures for C2 and C3 forced on every B). v32 C2: 14.96/15.02 vs k20h global 14.17/14.04,
  k23 14.38/14.34 (LOSS: issuing 16 half-row tiles takes 4.5 us after the Q fragments - the cp.async issue rate per
  64-B segment is the same as per 128-B row, so half rows halve the issue bandwidth; tile0 7.1 us). v16 C3: 10.82/10.70
  vs C6 per request 10.52/10.43. Off by default (WH_PH16 / WH_PH32 = 0).
- job17: WH_NEW32 (new block attended as a 32-token tile, wg_tile NB 1 on the same stage layout): v32 (global path)
  13.82/13.77 (dup 13.86/13.82) vs 14.15/14.24 (dup 13.98/14.01) -> kept for the global path; v16 (C6 cluster)
  10.93/10.84 vs 10.67/10.63 -> not for the cluster kernel (N32 = WH_NEW32 && CL == 0). Start stamps (traced):
  cluster b16 pre-B0 0.67, B0 0.74, cluster arrive 0.83, length visible 1.33; the global path shows the same ~1.3 in
  this traced build (the length load itself, ~1.1 us after issue).
- job18 (on dev/k25.py): split first tile (each stream's first tile as two 32-token halves with their own mbarriers,
  WH_SPLIT0_G / _C): v16 10.84/10.73 vs default 10.54/10.72, v32 14.32/14.36 vs 13.81/13.86 -> LOSS, off.
  Default (C6 b16, global + NEW32 b32) vs k25: v16 10.54/10.72 (dup 10.63/10.67) vs 11.96/11.97; v32 13.81/13.86
  (dup 13.73/13.63) vs 14.33/14.43.

## Deliverable (round 1): dev/wh/k20h.py = `python3 dev/wh/mk_k20h.py dev/k25.py` (base/out args: works on k19/k23/k25)
- B <= 16 (scored v16): attn_8_1_P1_1x6 = run8w<P1, 6, 6>: one cluster of 6 CTAs per request (96 CTAs), host checks
  16 x C6 co-residency by the occupancy query (else the global path). B 17..33 (scored v32): attn_8_1_P1_1w4 =
  run8w<P1, 4> global owner path (sentinel slots + L2 prefetch + probe), new block as a 32-token tile.
  Defaults: WH_QEARLY 1, issue every tile up front, WH_BULK 0, WH_SPLIT0 0, run8p off (WH_PH16 / WH_PH32 = 0).
- Next levers: (1) the C6 kernel's length barrier sits at ~1.3 us vs 0.8 on the global path (traced; cause not found);
  (2) b16 C6 tile phase: 6-tile parts give WGs 3 tiles (+0.9 us vs the 5-tile parts); (3) b32 global end chain
  ~3.5 us (merged max -> out max): an L2 round trip under load is ~0.7 us and the chain needs ~2 + 1 store.
- job19 (FINAL, dev/wh/k20h.py on dev/k25.py): check.py 0 failures; t_v8w 2 seeds 0 failures, global-only 0 failures.
  verify sliding 40 x2 (k25 | k20h | k25 dup | k20h dup): v16 12.02 | 10.66 | 12.06 | 10.72 ; 11.99 | 10.51 | 12.05 |
  10.68 ; v32 14.30 | 13.73 | 14.47 | 13.89 ; 14.70 | 14.04 | 14.49 | 14.35. All loads (20 reps): x2.6001 -> x2.6049
  (judge-proj x2.4320 -> x2.4365); every other shape runs identical code (noise).
e3a2c6ccb4fdff273f1b4ec2afb66be2  dev/wh/k20h.py
2d7d7d7e4cfd2c2963db2f0b4fa57056  dev/wh/run8w.cu
1daf9d65579a172177278e5323482aaa  dev/wh/run8p.cu
76a8afac108eeb67601044d644b944f3  dev/wh/mk_k20h.py
6236c11794a70a9f95bd24fb464560bf  dev/k25.py

## Round 2 (coordinator: b16 levers 1 and 2), base dev/k26.py (= dev/k27.py minus run8w)
- Lever 2 (split the last tile of 6-tile parts across WGs): moot. A C6 part with 6 tiles has 12 (tile, head) units for
  4 WGs = 3 each, and the 5-tile parts already have a 3-tile stream (tiles 0, 2, 4); every cluster holds two 6-tile
  parts, so the per-CTA max stays 3 units for any split. Only a cheaper tile or more WGs changes it.
- Lever 1 (job20, data-dependent stamps TRD: the stamp asm takes the loaded value as an input): the cluster kernel
  and the global path start the same way - start 0.13, end of the straight-line prefix 0.70 / 0.77, init barrier
  0.77 / 0.85, length landed 1.34 / 1.38, Q fragments done 2.27 / 2.18, first page-table entry landed 2.34 / 2.24
  (stamped after the Q fragments, so <=), K/V issued 2.99 / 2.78, tile0 3.97 / 3.89 (b32 global: 1.31, 2.27, issued
  4.35, tile0 4.67). The earlier "length at 0.8 on the global path" stamps had no data dependency (artifact).
  The chain is param/constant fetch (~0.5) -> length DRAM trip (~0.6-0.9) -> page-table DRAM trip (~0.9) -> issue
  (0.65 for 96 KB) -> first landing (~1). WH_PTW (touch page_table[b * max_pages] in the prefix; ruled allowed):
  v16 10.45/10.55 vs 10.41/10.47 (dups 10.40/10.38, 10.37/10.33), v32 13.52/13.50 vs 13.69/13.68 (dups 13.63/13.56,
  13.52/13.50): noise -> off. k26 control: v16 11.76/11.79 (dup 11.78/11.98), v32 14.13/14.19 (dup 14.18/14.10).
dab21002a4c8420d94eaed6a296ca64a  dev/wh/k28h.py
2e6e4a26228885fde9cf300b25d77fc6  dev/wh/run8w.cu
76a8afac108eeb67601044d644b944f3  dev/wh/mk_k20h.py
a0b382b9cc63556e365db1fb0041dfaa  dev/k26.py
7c97d1924a5a58ab9da7a6067bf631c0  dev/k27.py
- dev/wh/k28h.py = `python3 dev/wh/mk_k20h.py dev/k26.py dev/wh/run8w.cu dev/wh/k28h.py`: in default mode
  attn_8_1_1_1x6 / attn_8_1_1_1w4 SASS are identical to dev/k27.py's (local NVRTC); the round-2 additions are
  trace-only stamps (TRD) and opt-in knobs (WH_PTW, WH_SPLIT0_*). No performance change vs k27: keep k27.
- Remaining b16 levers (none cheap): per-unit tile cost (W9's QK-folded softmax lands in wg_tile, which run8w calls
  unedited); the C6 DSMEM push is bandwidth-bound (~28 KB per CTA, ~1 us); the start chain is three dependent DRAM
  trips (params, length, page table) before the K/V issue.
- Judge k27 (coordinator): x2.4488 / 1538.0 us; v16 sliding 13.02 -> 11.59, v32 sliding 15.86 -> 14.77 (no extra
  judge offset on the C6 cluster kernel). Standing down until W9's QK-folded softmax lands; next: b32 global end chain.

## Round 3 (coordinator: verify-32 sliding global end chain), base dev/k32.py (W9H FP16 softmax tile, WB FP16 slots)
- Policy change (docs/PLAN.md POLICY CHANGE 15:30): FP16 math / partials allowed within tolerance.
- job21 (k32 trace, flush, 30 reps): b32 tile0 4.64, WG done med 8.35, merged med 8.86 / max 10.21, publish med 9.45 /
  max 10.78, probe RTT 0.65, all-in med 11.90 / max 12.61, out med 12.30 / max 12.96. Controls v32 13.1-13.35,
  v16 9.3-9.5. Per part (percta): parts 0-2 tile0 ~4.55, merged ~8.8-9.0; part 3 (new block) tile0 5.74, merged 9.76
  (+0.9 us) -> every owner waits for part 3.
- job22 WH_NISS (dev/wh/mk_k33h.py v1: the new-block stream issues no K/V, the other stream issues every tile):
  v32 14.00/13.96 (dup 13.94/14.11) vs k32 13.08/13.04 (b 13.07/13.15), v16 9.96/9.76 vs 9.40/9.59: LOSS. Part 3 got
  later still (one stream issuing 128 KB) and every CTA's Q fragments ran 0.8 us later (q-done 3.01 vs 2.21): the extra
  issue-loop code (+3.4 KB before the first QGMMA; the prologue is already 13 KB of SASS up to the last LDGSTS) costs
  ~0.25 us per KB on the start path. Dropped.
- run8w3.cu = k32's run8w region + WH_NXB: spare CTAs (#SM - B P = 4 at b32) attend the 32 new blocks (one 32-token
  wg_tile per (request, KV head) job, 4 jobs per WG) and publish them as slot source P (P + 1 sources per owner); the
  parts carry past tiles only. Builder dev/wh/mk_k33h.py v2 (swaps the run8w region of the base).
- job23 WH_NXB (spare CTAs attend the new blocks): parts balanced (merged 9.0-9.24 for all parts) but the 4 spare
  CTAs (16 serial jobs each: load latency per job) published at 13-19 us: v32 20.4-20.6 vs k32 12.8 -> dropped
  (kept as dev/wh/run8w3_nxb.cu).
- WH_ONB (run8w3.cu re-extracted from k32 + ONB): on the global path no part attends the new block as a tile; stream 0
  of every part builds the new e4m3 K / V into smem (last part writes the cache rows; one 512-thread barrier after the
  build); every owner thread folds its row's 8 new keys (FP32 scores from the E4M3 q codes in its Q fragments and the
  new E4M3 K, P = 2^(s c - m + 8) in FP32, V e4m3) as a source before polling. The 4 parts carry 8 tiles each.
- job24 (ONB v1: stream 0 of every part builds the new K/V after the K/V issue + a 512-thread barrier; owner fold
  fully unrolled): v32 14.37/14.38 (dup 14.48/14.42) vs k32 13.04/13.10 (b 13.14/12.98), ONB=0 13.10/13.08: LOSS.
  Trace: parts balanced (merged 9.4-9.6) but tile0 +0.4 (build + barrier on the start path) and all-in +1.8 us: the
  unrolled fold is ~14 KB of one-shot SASS on the owner path (cold code). -> v2: owner warps build (2 / 4 units each),
  128 / 64-thread barrier, online fold over t <= j in a rolled loop (~1-2 KB).
- job25 (ONB v2, compact): v32 16.49/16.49 (dup 16.29/16.37) vs k32 13.11/12.96: LOSS. The owner block took 2.5-4.5 us
  (own-st 11.6 / 12.2 / 12.8 / 13.7 for parts 0-3): the build's qkv / cache_loc loads are serial DRAM trips inside a
  rolled loop, and each online iteration over t <= j cost ~0.3 us. ONB off (WH_ONB 0). Lesson: the new block must not
  add work or one-shot code to any critical path; 33 tile units over 8 streams leave part 3 at 4.5 units.
- WH_RIL (rolled K/V issue loop, each entry re-read by ld.global.nc after the up-front prefetch load): first QGMMA at
  0x3370 instead of 0x3f20 (-3 KB of start-path SASS) on attn_8_1_1_1w4; attn_8_1_1_1x6 0x33d0.
- job26 WH_RIL (rolled issue loop, entry re-read per iteration): v32 15.78/15.81 vs k32 13.06/12.97, v16 11.11/11.15
  vs 9.35/9.43: LOSS (kv-iss 6.7 vs 4.3): the per-iteration LDG queues behind the thread's pending LDGSTS in the LSU,
  so tiles issue one load latency apart. The entries must be in registers before the burst. RIL off.
- WH_LOWN: the last part (new block, merges last) owns all 128 rows; parts 0..P-2 publish every row and exit; every
  thread of the last part polls its own row chunk's P - 1 sources (slot row = (request, source, r)).
- job27 WH_LOWN: v32 14.79/14.88 (dup 14.76/14.85) vs k32 12.98/13.30 (b 13.07/13.13), LOWN=0 13.11/13.14: LOSS.
  Last part: probe RTT 0.62, all-in 2.1 us after the probe, epilogue 0.95 us: one SM loading P - 1 partials of all
  128 rows (~72 KB) is per-SM-bandwidth bound; the distributed owners spread that over P SMs. Off.
- WH_NEWLAST: the new block's tile attended after the stream's past tiles with the (warm) 64-token wg_tile instead of
  first with the separate 32-token instantiation (cold code on part 3 only).
- job28 WH_NEWLAST (new tile last, 64-token warm code): v32 13.51/13.37 (dup 13.47/13.32) vs NEWLAST=0 13.08/12.99,
  k32 13.32/13.06: LOSS (part 3 merged 10.07 vs 9.76: a full 64-token tile costs more than the cold 32-token one). Off.
- WH_NXB v2: spare-CTA jobs with all inputs prefetched by cp.async (one group per job, 218 KB smem layout, P = 4 only).
- job29 WH_NXB v2 (prefetched jobs): v32 28.3-28.6 (!) vs k32 12.96-13.11 (NXB=0 build 13.18/13.41); v16 9.19-9.31 vs
  9.09-9.24. The inlined spare-CTA routine sits ahead of the regular path (tile0 5.5 vs 4.55 for every part) and the
  spare CTAs published at 18-28 us. Dead.
- Round 3 verdict: no win on the verify-32 sliding end chain. The b32 critical path is part 3 (new block: 33 tile units
  over 8 streams leave one stream at 4.5) merging ~0.9 us after the others, then store -> poll RTT (~0.65 us under
  load) -> fold -> epilogue (~2.5 us from the last merge to the last output). Every attempt to move the new block or the
  combine either added one-shot / start-path code (0.25-0.5 us per KB here), serialized DRAM trips, or concentrated
  L2 traffic on one SM. dev/wh/k33h.py = `python3 dev/wh/mk_k33h.py dev/k32.py` is k32 unchanged (SRC identical, NVRTC
  name k33h.cu); the experiments build with `python3 dev/wh/mk_k33h.py dev/k32.py <out> dev/wh/run8w3.cu` (knobs off).
1eb64914df52f11f2f9c3361f00b530e  dev/wh/k33h.py
ca72e2b9c2a83a5e263c3fc84d8755db  dev/wh/mk_k33h.py
708390778088309e8b09cd0c482a98a5  dev/wh/run8w3.cu
752d7dec0bc2bcae27ccc6affba49c22  dev/k32.py
- Coordinator: round 3 accepted; standing down until a zero-critical-path-code way to rebalance the new block appears.
