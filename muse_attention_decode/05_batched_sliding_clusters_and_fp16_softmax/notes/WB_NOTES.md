# WB: full-attention decode (L=1) b16/b32, FLAT mode, toward the gather floor

Files (final):
- k3b.py = k3b_j4.py: DELIVERABLE drop-in (dev/k4.py with my L=1 run<>, run8 verbatim), check.py 56/56; WD merged
  its scheduling into dev/wd/k5m.py. k3b_dev.py = working copy (PEEK/EARLYG/NO_CUNROLL experiments, not validated).
- k5b.py (+ snapshots k5b_j6/j7/j8.py): dev/wd/k5m.py snapshot (k5m_snap.py) + opt-in FLAT experiments on run1<>
  (WB_FASTSTART on by default, WB_LATESCALE on, WB_TLBW, WB_PEEK, WB_FR, OVH not on request 0): all neutral.
- trace2.py (per-CTA phase trace: segment count, smid, prologue stamps, --mode warm), regs.py (image NVRTC 13
  regs/spills + tile-loop LDL/STL + SASS dump), job*.sh (the Modal jobs as run).

## Changes in k3b.py (all switchable by -D via `file@DEFS` in bench.py)
- PTQ=n (default 2): page-table rows prefetched n tiles beyond the cp.async issue point, kept in a register queue.
  k3 loads row(it+2) at the top of iteration it and needs it right after the barrier, so the page-table latency
  (HBM, cold) sits between "slot free" and "slot refilled".
- Fast issue path for full tiles (NO_FASTISSUE disables): per-thread constant smem offsets and source base, no
  bounds selects: 5 SASS instrs per cp.async instead of 10.
- L=1 PV mma with discarded rows g+8 (mma8h; NO_HALFMMA disables): o[d][2..3] no longer live across tiles.
- Combine: all loads of 4 partials issued before merging (was one dependent L2 round trip per partial).
- OVH=n (python-side, K2_DEFS "OVH=256" or env K3_OVH): FLAT split charges n virtual units per request (a
  per-segment restart cost), so CTAs that hold a request boundary get fewer tokens.
- LAZY: FA4-style lazy rescale (threshold 8 in log2 units, P <= 256).
- Trace: TRV(15) = %smid, TRV(14) = segments processed.

Static (local nvcc 12.1 PTX -> ptxas 12.8, sm_90a; the image's NVRTC 13 may differ): attn_1_0_1_0 main loop
594 -> 661 static instrs (both issue paths present), no spills inside the tile loop in any variant.

## Round 3 (2026-10-28): q_len 8 ALIGNED path (verify sliding b1/b16/b32, full b1) in run8
Build: `python3 dev/wb/mk_k6v2.py` -> dev/wb/k6v.py = dev/wb/k5m_v.py (snapshot of dev/wd/k5m.py) + W8's CORE (k2w.py)
+ dev/wb/run8v.cu (W8's dev/w8/run8.cu + my ALIGNED changes) + cluster launch plumbing (mk_k6v.py does the k5w-style
swap). Owned by me: run8's prologue, ALIGNED split, lane merge, combine/epilogue; W8 owns the tile loop.
- CLU (cluster/DSMEM combine, ALIGNED sliding): the cpr CTAs of a request form a cluster; after the lane merge every
  CTA st.async-pushes each partial row (64 O + m + l = 264 B) into the owner CTA's receive buffer (rows split evenly
  over ranks; buffer [cl][ceil(128/cl)][68] f32 <= 132 rows: cl=6 needs 6x22 - the first build sized it 128 rows and
  faulted with Xid 13 "SM Cga Exception: OOR Addr"); one mbarrier (complete_tx) per CTA; owner merges and runs the
  epilogue. Gates: one slot (16 KB) for CLU. Cluster size: host occupancy query (cuOccupancyMaxActiveClusters),
  largest of (16,8,6,5,4,3,2) with B*cl <= 132, all clusters co-resident, and cl >= k5's cpr.
- job9: check 56/56. job10 bench (local us, same job; km = k5m_v, w = W8's k5w):
    verify1 sliding km 16.86 | w 16.20 | CLU(cl16) 12.87 | NOCLU 15.95 | +OUTW 12.63 | cl8 14.32
    verify1 full    km 21.51 | w 19.94 | 20.88 | 20.31 | OUTW 19.41
    verify16 sliding km 17.77 | w 16.49 | CLU(cl6) 16.73 | NOCLU 16.65 | cl4 17.70
    verify32 sliding km 21.41 | w 19.52 | CLU(cl3) 21.58 | NOCLU 19.64 | cl2 25.77
  => clusters only when they do not cut the CTA count (b1: 8 -> 16 CTAs); b16/b32 keep the global combine.
  Trace (cl16 b1, flush): tiles 6.58, merged 9.05, rows landed 10.9, merge math done 13.68 (ALL CTAs at the same
  instant), epilogue done 14.98. Warm mode: landed 10.0, done 11.2 -> ~3 us of the combine is COLD INSTRUCTION FETCH
  (all CTAs miss the same lines; the judge's L2 flush evicts the code). Same in the global path (nc b1: released
  12.65 -> combined 16.02 flush vs 12.12 -> 13.74 warm).
- WB_RTAB (RoPE table, sliding): sincosf(float(pos_j) * inv_freq_i) for the 8 tokens x 32 freqs, one per thread
  (tid < 256), into smem [8][32] float2, read by the Q fragments and the new-token RoPE (bit-identical args); was
  16 precise sincosf per thread. Removes the ifr[8] registers held across the length barrier.
- OUTW: one dummy ld.global of the CTA's first output row at entry (translation warm for the epilogue stores).
- WARMC: the cluster combine runs once as a dry pass (stores predicated off, epilogue_store_x executes the whole
  instruction stream) while the CTA waits for its peers' rows (skipped if they all landed): one code copy
  (#pragma unroll 1 pass loop), so the real pass finds the code in the I-caches / L2.
- job11 (same job, local us): km = k5m_v | w = W8 k5w | j10 | a = j11 (cl rule + RTAB + OUTW) | b = a + WARMC |
  bnr = b without RTAB.  check 56/56 (b).
    verify1 sliding   17.35 | 17.21 | 13.35 | 9.95  | 10.50 | 13.89
    verify1 full      21.48 | 19.84 | 21.12 | 20.20 | 19.73 | 19.94
    verify16 sliding  18.37 | 16.73 | 16.97 | 14.73 | 14.72 | 18.41
    verify32 sliding  21.57 | 19.88 | 21.78 | 18.14 | 17.84 | 21.16
    verify16/32 full  141.2/255.0 | 141.9/253.0 | ... unchanged (W8's FLAT)
    verify-load geomean km x1.459, w x1.495, a x1.703, b x1.695. decode loads: km vs b within +-0.2 us.
  => RTAB is worth 3.3-3.9 us on every sliding shape (16 precise sincosf per thread on pos~80K arguments ->
  slow path). WARMC: +0.5 us (the real combine is already ~1.1 us after the rows land in this build; the job10
  cold-combine 4 us is gone) -> opt-in (WB_WARMC). Trace a, b1 sliding cl16 (us): issued 2.46, Q/new 3.27,
  tile0 3.59, tiles 5.17 (max 6.59), lane merge+push 7.64, rows landed 8.42, combined 9.54.
  b16 sliding (global path): tiles 6.9 (max 8.5), merged 8.84, publish 9.79, released 10.94, combined 12.65.
  full b1: tiles 11.29 (max 12.6), merged 13.41, publish 14.38, released 15.73, combined 18.78 (tail ~7.5 us).
- Post-tile SASS size (local NVRTC 12.4, tile-loop end to last EXIT, all branches): cluster kernel 14.6 KB vs
  global-combine kernels 28-30 KB.
- job12 (k6v_j12 = final k5m-based, WARMC opt-in): check 56/56. Bench (same job, local us) km | w | v | v NOCLU8:
    verify1 sliding 17.38 | 16.74 | 10.02 | 12.87; full 21.50 | 20.10 | 19.88 | 19.76; v16 sliding 18.14 | 16.93 |
    14.77 | 14.92; v32 sliding 21.80 | 20.19 | 18.08 | 18.25. Verify geomean km x1.448, w x1.488, v x1.693.
    Decode loads km vs v: within +-0.2 (sliding b1 8.77/8.93, b32 12.09/12.19).
- DELIVERABLE (round 3): dev/wb/k7v.py = dev/k6.py with run8 -> dev/wb/run8v.cu (+ WC's ALIGNED break), built by
  `python3 dev/wb/mk_k7v.py`. job13: check 56/56. Same job vs dev/k6.py (local us):
    verify1 sliding 15.51 -> 10.35 | full 19.45 -> 19.59 | v16 sliding 15.23 -> 13.58 | v16 full 141.77 -> 141.68 |
    v32 sliding 18.39 -> 17.24 | v32 full 254.49 -> 256.62; decode loads unchanged (566.9/1893.9/3269.2 ->
    566.2/1896.5/3282.0). Loads verify-1 x1.536 -> x1.993, verify-16 x1.527 -> x1.568, verify-32 x1.579 -> x1.586.
    GEOMEAN x1.9117 -> x2.0054 (judge-projected x1.8735 -> x1.9596).
  W8 merged its round-3 tile-loop patch onto run8v.cu -> dev/w8/k8.py (56/56, x2.095 same job).
- job14 (k7v_j14, opt-in A/Bs, check 56/56 with both on). Same job, local us: k6 | k7v | WB_MU | WB_GPRE | both
    verify1 sliding 14.96 | 10.17 | 10.55 | 10.22 | 10.55;  full 19.85 | 19.54 | 20.49 | 19.35 | 19.60
    v16 sliding 15.21 | 13.71 | 14.15 | 13.63 | 14.33;  v16 full 141.6 | 141.4 | 145.2 | 141.6 | 145.4
    v32 sliding 18.43 | 16.85 | 17.71 | 16.87 | 17.80;  v32 full 253.1 | 255.4 | 261.1 | 254.8 | 261.2
  => WB_MU (lane-merge loop unrolled x4) is slower everywhere, also on FLAT (shared merge loop): more cold code.
  WB_GPRE (gate prefetched before the combiner's wait, global-gate epilogue): neutral. Both stay opt-in.
- Open (not reached: v16/v32 sliding 13.6/16.9 vs ~8/10, verify-1 full 19.5 vs ~10):
  * b16/b32 sliding cannot use one cluster per request without cutting CTAs: 16 clusters of 8 > 15 co-resident,
    32 of 4 > 30. Remaining option: cluster-of-2 pairwise pre-reduce (64 clusters fit) + global combine of half the
    partials, or two launches.
  * Tail on the global path (job12/14 traces): tiles max -> combined max 4.8 us (b16/b32), 6.5 us (full b1):
    lane merge ~2 us (also in warm mode, so not code; the merge stores after the last tile take ~0.6 us for
    thread 0 alone), publish ~1, wait ~1.2-1.5, combine ~1.7 (sliding) / ~3 (full b1, 128 one-row combiners
    over 132 partials). Full b1: 2-level cluster combine (cpr 120 = 15 clusters of 8, reduce-scatter in the
    cluster, global combine over 15 partials) is the next thing to try.

## Round 4 (2026-10-28): run8 lane merge, on dev/k8.py
- job15 (dev/wb/k8p.py = k8 + merge sub-stamps; ncu source counters, b16 sliding): tiles -> first wg_store_row
  0.07 us -> second wg_store_row +0.53 us (thread 0 alone) -> mM/mL + barrier 0.07 -> first merge-loop pass +0.42
  -> remaining 3 passes + barrier +0.9. Warm mode identical (not cold code). ncu: the top stall line of the whole
  kernel is wg_store_row's STS.128 (mio_throttle 43 samples). Root cause: smem traffic - 512 threads write both
  lanes' O rows (64 KB, 2-way conflicts at stride 68) in one burst (MIO/smem pipe saturated), then 4 dependent
  passes re-read 64 KB + mM/mL.
- WB_RMERGE (dev/wb/mk_k9v.py on dev/k8.py, default on; the cpr == 1 `complete` case keeps the old path): lane
  merge in registers. Both lanes hold the same (row, d) set per thread; lane 1 writes its 36 values (32 O + 2 m +
  2 l) k-major [9][256] float4 (conflict-free, 36 KB), one barrier, lane 0 reads them in its own register layout,
  merges, and writes ws with wg_store_row(dst = ws) (natural d order = ws layout) or pushes cluster rows by
  st.async from registers.
- job16: check 56/56. Same job, local us: k8 | k9v | k9v WB_RMERGE=0
    verify1 sliding 9.90 | 9.85 | 10.00;  verify1 full 18.92 | 18.99 | 19.33
    v16 sliding 12.96 | 12.52 | 13.04;   v16 full 122.74 | 119.08 | 123.38
    v32 sliding 16.00 | 15.75 | 16.13;   v32 full 222.41 | 218.35 | 222.32
    verify-load geomean x1.879 -> x1.911 (judge-projected x1.844 -> x1.875). NVRTC 13: attn_8_0_1_0 spills
    76/112 -> 64/64 B. Trace b16 sliding: tiles -> merged 0.92 us (was ~2.0).
  Root cause + fix sent to WD (run1 has the same lanes -> mO -> barrier -> multi-pass merge structure).
- DELIVERABLE (round 4): dev/wb/k9v.py (= k9v_j16.py), `python3 dev/wb/mk_k9v.py` (anchored on dev/k8.py; edits
  only the run8 merge block, after the tile loop).
- Not done this round: (2) verify-1 full 2-level cluster combine, (3) b16/b32 pairwise clusters. Estimates from
  the traces: full b1's tail after the slowest tile is merged +1.0, publish +0.85, combine ~2.9 us (128 one-row
  combiners; per combiner one 4-partial load batch, a 32-group sequential smem reduction by 16 threads ~0.5 us,
  the epilogue). A 2-level cluster version also loses 12 CTAs of streaming (+~0.7 us), net ~0-1 us; a
  warp-shuffle group reduction (mapping tid = c4*32 + group) is the cheaper ~0.4 us. Pairwise clusters for
  b16/b32 halve the global-combine loads only (~0.3-0.5 us est.).
- dev/wb/clred.cu (for WD): compact two-level cluster combine helpers between `// WB-CLRED-BEGIN/END` (ClRed,
  clred_init/ready/push/merge/exit; level 2: clred_slot_write/publish/gmerge). Compile-checked only (NVRTC 12.4,
  test kernel 50 regs, no spills). Level-2 slot rows are 68 floats (272 B) so float4 stores stay aligned.

## Round 5 (k10v): `python3 dev/wb/mk_k10v.py [base]` (default base dev/k10.py; its run8 region == k9v's)
- job17 (first cut, base k9v; same job, local us; k9v | v | v WB_SHRED=0):
    v1 sliding 9.59 | 9.45 | 9.38;  v1 full 19.28 | 26.13 | 19.14;  v16 sliding 12.34 | 12.30 | 12.29
    v32 sliding 15.26 | 15.00 | 15.20;  v16 full 120.0 | 122.2 | 120.3;  v32 full 221.3 | 221.2 | 219.6
  The first WB_SHRED (threads = (item, group) with the group index in the low lane bits, xor butterfly) made
  verify-1 full +7 us: a warp then reads 32 partials' float4 of one (row, c4) = 32 sectors per load instead of
  2 x 256 B (trace: released -> combined 11.5 us). The coordinator's "tid = c4*32 + group" mapping has the same
  problem. Replaced by the coalesced mapping + one xor-16 shuffle (groups 2w, 2w+1) when nitem == 16.
  Clusters again (job17): b16 sliding cl7 12.13 vs 12.30 global (-0.17, noise level), cl6 14.14; b32 cl3 19.05
  vs 15.00. Cluster b32 trace: st.async push of 32 KB/CTA takes 1.6 us (global ws stores 0.7), and the cluster
  combine's first item 1 us (cold code). Not worth it; plan_clu8 unchanged.
- End chain, b16 sliding (job17 trace, global path, slowest CTA): tiles 8.34 -> xch 8.40 -> barrier 8.53 ->
  merged + ws written 9.23 -> publish (acq_rel atomic: store acks + round trip) 10.08 -> released max 10.51 ->
  combined max 12.11, then each combiner's "done" atomicAdd round trip before exit. Merging partials "as they
  land" saves nothing: every combiner already loads all P partials of its rows in one round after the last
  publish; per-partial flags keep the same number of trips after the last publish. A single last-arrival
  combiner would pull 7 x 32 KB alone (~1.2 us vs ~0.7).
- Changes (all after the tile loop):
  * WB_RMERGE=2 (ALIGNED only): split lane merge - lane li keeps row half li and sends the other half (4 float4 +
    (m, l)) to its partner; both lanes merge and store 64 rows each. Halves per-thread smem + math + ws stores on
    the critical path (was lane 0 alone: 9 LDS.128 + 8 STG.128). FLAT keeps the k9v merge (the split one added
    prologue spills to attn_8_0_1_0 in local NVRTC 12.4).
  * WB_GEN: arrival counter = (generation << 8) | count at cnt[512 + b] (a region no other path uses); the last
    arrival adds 256 - P right after its arrival (count 0, generation + 1; red.release, part of the release
    sequence); combiners wait for the generation to move. No end-of-kernel "combiners done" atomicAdd round trip
    and no reset stores. Guard: B <= 512, G <= 255.
  * WB_CPF5: combine loads in predicated batches of 5: verify-1 full (P = 132, 32 groups) needed a second
    dependent load round for 4 of the 32 groups.
  * WB_SHRED (coalesced): xor-16 shuffle merges 2 groups when nitem == 16 (16 entries instead of 32 in the
    sequential smem reduction).
  * WB_DSPLIT: ALIGNED split by exact double floor (neutral in job17).
  * Receive buffer for clusters sized 140 rows (any cl <= 16); rtab behind it.
- Start (c): the ALIGNED prologue already issues the length load first, then Q / gate cp.async / new-token loads
  (length-independent), one barrier, and the page-table loads right after the split arithmetic (PAGE1: no
  divisions). Nothing left to reorder there.
- Local NVRTC 12.4 spills (st/ld B): k10 vs k10v - 8_1_1_1c 4/8 -> 0/0, 8_1_0_1 4/4 -> 0/0, FLAT unchanged.
- job18 (base dev/k10.py; check 56/56; same job, local us; k10 | v (split merge + GEN on) | GEN=0 | RMERGE=1 |
  SHRED=0,CPF5=0):
    v1 sliding 10.19 | 9.84 | 9.69 | 9.61 | 9.71;   v1 full 19.26 | 18.91 | 18.98 | 19.05 | 19.12
    v16 sliding 12.54 | 12.67 | 12.39 | 12.47 | 12.82;   v32 sliding 15.59 | 15.53 | 15.54 | 15.48 | 15.70
    v16 full 119.68 | 119.74 | 120.24 | 119.77 | 119.84;   v32 full 219.59 | 219.89 | 220.59 | 220.47 | 220.02
    geomean x1.8866 | x1.9025 | x1.9069 | x1.9105 | x1.9007
  Split merge: no gain (b16 sliding trace: s12 -> merged 0.85 us vs 0.70 in job17, and merged -> publish 1.23
  vs 0.85: both lanes storing ws at once does not finish sooner) -> opt-in. GEN: neutral (the done atomic is not
  on the visible tail) -> opt-in. CPF5 + SHRED: -0.2 us on verify-1 full -> kept.
  Trace verify-1 full (flush): publish max 16.29 -> released med 16.45 -> combined med 19.74: 3.3 us for a
  one-row combine. The epilogue there (ALIGNED, no RoPE: GSM false) loads the gate from global = one more
  dependent DRAM trip after the last partial -> WB_GPF (job19).
- job19 (check 0 failures; local us; k10 | v | GPF=0 | GEN=1 | SHRED=0,CPF5=0): v1 sliding 10.21 | 9.90 | 10.03 |
  9.74 | 10.00; v1 full 19.71 | 18.83 | 18.79 | 18.82 | 19.03; v16 sliding 12.73 | 12.71 | 12.69 | 12.57 | 12.62;
  v32 sliding 15.70 | 15.50 | 15.60 | 15.82 | 15.62; v16 full 120.9 | 121.6 | 120.9 | 121.1 | 121.0; v32 full
  221.3 | 222.4 | 221.9 | 221.5 | 221.6; geomean x1.8687 | x1.8878 | x1.8867 | x1.8978 | x1.8857.
  GPF neutral in the bench (trace released -> combined 2.95 vs 3.19 us). Verify-1 full moved to WD (scope update):
  WD keeps CPF5 / SHRED / GPF / DSPLIT and edits only under !GSM.

## Round 6 (verify sliding b1/b16/b32 end + start chains; verify-1 full is WD's): same builder, k10v_j20.py
- Finding: in every b16/b32 sliding trace the slowest CTAs are last parts (b16 job17 rep0: CTAs 47, 23, 55, 63, 95,
  71 = part 7; b32 cluster: all part 2). The last part owns the new block: new-token prep (+0.5 us to tile0) and
  one extra tile (window 4096 -> 4095 past tokens = 64 tiles + the new tile = 65 over 2 x cpr lanes: the token-even
  split puts the 65th on the last part). b1 sliding cl16 (job16): CTA 15 tiles 6.75 vs ~5.1 for the rest.
- WB_TSPLIT (ALIGNED RoPE): tile-aligned split, part i gets tiles [ceil(i nt/cpr), ceil((i+1) nt/cpr)) of
  nt = past tiles + 1 (new); the leftover tile goes to part 0 (no new-token prep) and the last part keeps an even
  lane count. Empty parts (short requests) publish m = -inf partials (handled by the merges).
- WB_BULK (CLU): b1 sliding barrier -> merged was 2.0 us for ~2176 16-B st.async per CTA (vs 0.7 us for the same
  bytes as global stores). Lane 0 now writes its merged rows to local staging [R][68] (ring + 40 KB, same layout as
  the receive slots), fence.proxy.async, and after the merge barrier thread q < cl issues ONE
  cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx (rows of owner q, 272 B/row). Receive expect_tx
  272 B/row.
- WB_EVF (A/B, off): WF's L2::evict_first createpolicy on the run8 ALIGNED RoPE K/V cp.async.
- Start chain: unchanged (length -> barrier -> page-table loads already first; RTAB sincos after the K/V issue).
- job20 (check 0 failures; local us; k10 | j19 | v = TSPLIT+BULK | TSPLIT=0 | BULK=0 | EVF=1):
    v1 sliding 9.89 | 9.76 | 9.12 | 9.80 | 9.13 | 9.33;  v16 sliding 12.38 | 12.45 | 12.20 | 12.54 | 12.28 | 12.43
    v32 sliding 15.50 | 15.45 | 15.57 | 15.17 | 15.78 | 15.49;  (v1 full 19.21 | 18.79 | 18.20 | 18.21 | 18.45 | 18.26)
    geomean x1.8897 | x1.8955 | x1.9309 | x1.9054 | x1.9224 | x1.9204
  TSPLIT: v1 sliding -0.68, v16 sliding -0.34, v32 sliding +0.40 (17 tiles/part: bandwidth-bound, a 9th gather
  tile on part 0 costs more than the new tile + prep) -> TSPLIT only when parts have <= 12 tiles (job21).
  BULK: neutral in the bench. Trace b1 (flush): barrier -> staged 0.6 us (was 2.0 with st.async), but the bulk
  issue -> publish stamp +1.2 and released 7.91 vs 8.46: the DSMEM transfer itself (~31 KB out of every CTA of a
  16-CTA GPC-local cluster, ~0.5 MB per cluster) takes ~1.5-2 us either way (GPC network bound, not instruction
  bound). Kept (fewer instructions). EVF (L2 evict_first on K/V): +0.2 on v1/v16 sliding -> not adopted.
  b1 trace: slowest CTA now part 0 (the extra tile; 5.92 vs ~4.8 tiles), start: issued 2.05, q/new 3.14, tile0 3.42.
  b16 trace: the 8 slowest CTAs are all part 0 now (was part 7): one lane per request has 5 tile steps (65 tiles
  over 16 lanes) - inherent with 64-token tiles and cpr 8. DST_S=4 does not fit (228 KB global, 250 KB cluster).
- WB_SPEC (b1 sliding): verified speculation of the first DST-1 tiles' page-table rows for seq_g = max_pages *
  page_size - L (reference.py sizes the table to seq + q_len at batch 1) with the TSPLIT range, issued before Q /
  gates / new-token loads; hit iff same pos0 and npast <= guessed (entries past npast zeroed). WB_LENR: thread 0
  keeps the length in a register until right before the barrier (its warp no longer stalls ~1 us on it before
  issuing its other loads).
- job21 (check 0 failures; j20 | v = SPEC regs + LENR + TSPLIT<=12 | SPEC=0 | LENR=0 | BULK=0 | v dup):
    v1 sliding 9.28 | 9.89 | 9.43 | 9.91 | 9.70 | 9.86;  v16 sliding 12.42 | 13.21 | 12.85 | 12.98 | 12.94 | 13.28
    v32 sliding 15.57 | 16.69 | 16.03 | 16.69 | 16.49 | 16.64;  geomean x1.9541 | x1.9158 | x1.9273 | x1.9221 | ...
  Everything j21 is slower than j20 (0.4-1.1 us on every sliding shape, also with SPEC=0 / LENR=0). Image NVRTC:
  attn_8_1_1_1c spill st/ld 20/48 B (SPEC=0: 12/36). Trace b1: issued 2.60 (SPEC=0 2.33; j20 2.05): the guessed
  rows held in registers across the barrier spilled and came back from local memory on the critical path. ->
  job22: the speculation staged through smem by cp.async (ptq slots 0..DST-2), the length by cp.async (WB_LCP),
  the guessed (pos0, npast) through misc[8..9]; nothing held in registers across the barrier.
- job22 (check 0 failures; image NVRTC attn_8_1_1_1 0/0 spills in all variants; j20 | v = staged SPEC + LCP +
  TSPLIT<=12 | SPEC=0 | LCP=0 | WARMG=1 | WARMC=1):
    v1 sliding 9.14 | 9.38 | 9.39 | 9.06 | 9.21 | 9.71;  v16 sliding 12.06 | 12.62 | 12.54 | 12.34 | 12.98 | 12.61
    v32 sliding 15.34 | 16.01 | 15.74 | 15.48 | 16.88 | 16.01;  geomean x1.9213 | x1.9091 | x1.8982 | x1.9273 | ...
  Speculation does not pay here: b1 trace issued 2.42 us with the staged guess (j20 2.05). Any change to how the
  length arrives (register LENR 2.33, cp.async LCP 2.42) made the first K/V issue later, not earlier; not
  understood (would need stamps right after the barrier). Dry warm-up passes (WARMG global, WARMC cluster)
  lose 0.5-1.5 us. TSPLIT threshold for b32: within noise across jobs (job20 -0.40, job22 +0.14).
- DELIVERABLE (round 6): dev/wb/k10v.py = default build of `python3 dev/wb/mk_k10v.py` on dev/k10.py; compiled
  code identical to k10v_j20.py (local SASS diff 0 on all attn_8 / attn_1 kernels checked). Experiments (SPEC /
  LCP, WARMG) only with WB_EXP=1 (dev/wb/k10v_exp.py); opt-in macros WB_GEN, WB_RMERGE=2, WB_EVF, WB_WARMC.
- job23 (final; check 0 failures; same job, local us; k10 | k10v | j20 | k10 dup):
    v1 sliding 9.75 | 9.23 | 9.36 | 9.86;  v1 full 18.96 | 18.35 | 18.28 | 18.77
    v16 sliding 12.36 | 12.25 | 12.39 | 12.45;  v32 sliding 15.55 | 15.66 | 15.67 | 15.67
    v16 full 119.56 | 120.28 | 121.05 | 120.45;  v32 full 220.27 | 221.03 | 219.93 | 220.58
    verify geomean x1.8900 | x1.9150 | x1.9086 | x1.8820 (judge-projected x1.8071 | x1.8289 | x1.8231 | x1.7997)

## Log
- job1 (wb_base): dev/k3.py check FAILED on L=8 FLAT (NaN, then a hang; coordinator fixed it: K <= P combiners,
  ported into k3b and k3_base). Bench lost (bench.py exits on the ABL_NOCOMPUTE candidate's failed check: run
  ablations with --check-reps 0). k3 trace, b32 full (us from first CTA start, median over reps of min/med/max
  over CTAs): first cp.async issue 6.2 / 10.5 / 176 (2nd segments overwrite); tile0 7.6 / 11.9; tiles done
  167.1 / 173.2 / 186.6; merged 170.6 / 176.0 / 188.9; publish 172.4 / 177.5 / 191.5; combined (23 CTAs)
  186.9 / 192.3 / 197.8. b16: first issue 6.1 / 8.5; tiles 85.5 / 89.0 / 101.9; combined max 113.3.
  => startup ~10 us before the first gather, CTA spread ~13 us (med -> max), tail ~11 us after the slowest CTA's
  last tile (merge ~2.5, publish ~2.5, some CTAs combine 2 requests one after the other, combine ~2.5).
  Judge (coordinator's okc check of k3): full b32 195.6, b16 111.9 (local 190.8 / 107.5).
- RELFIRST (L=1 only; NO_RELFIRST disables): k3's loop is wait(it) -> bar -> refill slot(it-1) -> compute(it), so
  during every wait only DST-1 = 2 of the 3 slots are in flight (the slot freed by compute(it-1) is refilled
  only after the wait). New order: bar -> refill -> wait(it) -> bar -> compute(it): 3 slots in flight while
  waiting (24 KB per lane, 192 KB per SM), 2 barriers per tile. Added after job2 was launched (job2 = without).
- job2 (wb_job2, k3b before RELFIRST): k3b check 56/56 PASS (defaults); OVH=512+CPS2 on all "layer full" tests +
  sliding b32 PASS. BENCH NUMBERS INVALID: a candidate labelled "base" collides with bench.py's SGLang key
  ("base"), so the SGLang column mixes both. Relative order among candidates (same broken denominator), b32 full:
  b 202.9, p0 (PTQ=0) 194.2, ns (no SPEC) 202.7, o256 201.1, o768 198.3, lz 200.8, h2 (CPS2) 205.8, h2o 201.7.
  => PTQ=2 HURTS (~8 us): the register queue shift rq[0]=rq[1] waits on the LDG issued one iteration earlier
  (loaded latency ~4 us > one iteration). SPEC: no effect. CPS2: no gain. OVH helps, more is better.
  Trace (b32 full, k3b): prefix ready 1.1-1.3 us, first page-table row back 4.5 / 6.6 us (min/med) -> the
  page-table load takes 3-5 us at start; tiles done 142 / 167 / 183; nseg=1 CTAs mean 162.9 (max 176), nseg=2
  mean 174.9 (max 186): a request boundary costs ~12 us (merge ~2 + page-table hop + gather hop at loaded latency
  ~4 us each). Spread among nseg=1 CTAs is 142..176 us: SM-dependent speed (fastest CTAs repeatedly on SMs 33-42,
  84-87, 126-127). Combined max 191, last stamp ~193. b16: tiles 74 / 89 / 104, nseg 1 mean 86.8 vs nseg 2 99.4.
  globaltimer resolution 32 ns (fine).
- PTS (L=1): page-table entries for lane-iteration t+DST are cp.async'ed (4 B each, lanes 0-15) into a small
  smem ring inside tile t's group; the issue of tile t+DST reads them with one LDS. No LDG in the tile loop.
- job3 (wb_job3, k3b_j3 = RELFIRST + PTS by default): smoke PASS but the full check printed nothing within
  timeout 1500 (hang or crash somewhere in the 16 tests; job4 locates it). Bench (valid labels), local us:
    full b16: k3 108.2 | j2 (k3 order, PTQ=0: fast issue + half mma + fast combine + combine-unroll) 104.8 |
              rf (RELFIRST, PTQ=0) 111.1 | pts (RELFIRST+PTS) 114.9 | +OVH1024 108.1 | +OVH1536 105.0 |
              +OVH2048 105.2 | +LAZY+OVH1536 105.8
    full b32: k3 191.0 | j2 183.4 | rf 193.4 | pts 196.2 | o1k 191.7 | o15 189.8 | o2k 189.5 | lz 189.7
    no-compute: b16 k3 94.0 pts 93.3 o15 90.3; b32 k3 172.4 pts 171.7 o15 168.9
    other L=1 (k3 / pts): sliding b1 12.44 / 11.66, b16 13.90 / 13.27, b32 17.64 / 16.39; full b1 16.74 / 18.37.
  => RELFIRST and PTS are slower (+10 / +3 us at b32). Trace of pts: tiles-done -> merge-stores 6.9 us (was
  0.6 us in j2): most likely local-memory spills (smem 213 KB leaves ~40 KB L1, so spill reloads go to L2/DRAM
  at loaded latency). OVH ~1536-2048 balances nseg=1 vs nseg=2 (tiles mean 166.1 vs 169.3) and is worth ~6 us
  at b32 / ~10 at b16. Startup: prefix 1.2 us, code reaches the page-table LDGs at 4.9 us (pre-ld) and the
  first row is back at 6.3 us: ~3.7 us of the startup is before the LDGs are even issued (code/i-cache or
  the Q/gate/new-token load issue), only ~1.4 us is the page-table latency.
- Local NVRTC 12.4 (pip) and ptxas 12.8 disagree on spills (e.g. k3b_j3 attn_1_0_1_0: 0 vs 124/136 B);
  dev/wb/regs.py reports the image's NVRTC 13 numbers + tile-loop LDL/STL counts.
- Rebased on dev/k4.py (coordinator): dev/wb/k3b.py = k4 with my run<> (L=1) and run8 verbatim; RELFIRST/PTS/
  SPEC now opt-in (RELFIRST_ON, PTS_ON, SPEC_ON); OVH default 1536 for FLAT L=1. k3b_k3.py = the k3-based copy.
- job4 (wb_job4, k3b_j4 = k4 base + my L=1 run<>: k3 order, fast issue, half mma, fast combine, combine
  unroll, OVH=1536): check.py 0 failures (56/56). NVRTC 13: attn_1_0_1_0 spills 124/144 B but none in the tile
  loop (686 instrs, LDL 0); RELFIRST_ON puts 7 LDL in the loop (5 with PTS_ON) -> explains its slowdown.
  Bench (local us, same job):        k4      j4(b)   OVH=0   OVH2048  OVH3072  RELFIRST  LAZY
                   decode full b16   107.35  97.76   103.73  97.92    98.42    102.56    97.35
                   decode full b32   189.47  178.30  182.57  178.30   179.54   187.65    177.67
  other L=1 (k4 / j4): sliding b1 12.29 / 11.28, b16 13.90 / 13.74, b32 17.63 / 16.92; full b1 16.66 / 17.31 (!)
  GEOMEAN of the decode loads x1.788 -> x1.896. The j3 hang did not reproduce (j3 --only tests: 0 failures).
  Trace j4 b32: prefix 1.04, pre-ld 4.61, first row 5.90, tile0 9.48; tiles 155.9 / 169.3 / 174.0;
  merged med 172.0, publish 173.7, combined med 177.0 max 179.6, last stamp 181.5. b16: tiles 76 / 88.6 / 91.9,
  combined max 97.9. Spread is much tighter than with RELFIRST+PTS.
- job5 (wb_job5, k3b_j5 = j4 + PEEK + EARLYG): check 56/56. Bench (local us):
                 k4      j4      j5      NO_PEEK  NO_EARLYG
    full b16     107.98  97.65   101.73  98.05    101.49
    full b32     190.00  178.68  184.83  179.42   184.27
  => PEEK costs +4..6 us: it keeps peekv/peekP/peek_ok live across the tile loop -> 3 LDL in the NVRTC-13 tile
  loop (j4: 0). EARLYG neutral. Both now opt-in. decode 1 full A/B (WC's area): k4 16.61 | j5 18.56 | NO_CUNROLL
  18.02 | NO_FASTISSUE 18.33 | NO_HALFMMA 19.01 (j5 includes PEEK's spills; in j4 the regression was 16.66->17.31).
  Trace warm vs flush: prefix->pre-ld gap 3.4 us in BOTH -> not cold code. Local SASS shows the k_scale LDG is
  followed at once by R2UR (uniform-register move) => the warp stalls on k_scale before the warp-0 prefix.
- Coordinator: base is now dev/k5.py / dev/wd/k5m.py (WD's QGMMA run1<> with j4 scheduling). My FLAT work goes
  in as patches to run1<>: dev/wb/k5b.py = dev/wd/k5m.py snapshot (k5m_snap.py) + (a) OVH not charged to
  request 0 (it has no restart), (b) WB_LATESCALE: FLAT k/v-scale loads moved after the first gather issue,
  (c) fine prologue stamps TR18..21 (c0c1, b_begin, segment top, inputs issued), TR16/17/10.
- job6 (wb_job6, k5b_j6 = k5m_snap + OVH-not-request-0 + LATESCALE + stamps): check 56/56. NVRTC 13: run1
  attn_1_0_1_0 has 0 spills (tile loop 484-488 instrs, 8 QGMMA, LDL 0). Bench (local us):
                 k5m_snap  j4      k5b_j6  LATESCALE=0  OVH=1024  OVH=2048
    full b16     89.96     97.91   90.32   90.29        91.07     90.62
    full b32     169.98    178.50  170.67  170.12       171.57    170.62
    no-compute: b16 km 88.83 / b 88.58; b32 km 166.66 / b 167.69  => run1's compute costs only 1-3 us; the
    rest of the gap to the floor (b16 76, b32 152) is scheduling: ~12.6 / ~15 us.
  LATESCALE: no effect. OVH 1536 stays best for run1. Trace k5b b32 (us, med): prefix 0.93 | c0c1 1.25 (64-bit
  divisions 0.3) | b_begin 2.15 (linear dependent-LDS search: med +0.9, max 2.88) | seg top 2.65 | inputs issued 2.85
  | page-table LDG issued 3.15 | first row back 4.10 | prologue issued 5.28 | Q/new 7.18 (Q load ~4 us at start) |
  tile0 8.33 (first gather ~4.2 us). End: tiles 152.0 / 161.0 / 164.9, merged max 167.0, publish max 168.2,
  combined max 170.1, last stamp 171.8. nseg1 mean 161.7 vs nseg2 157.9. b16: tile0 8.05, tiles 73.7 / 82.6 /
  85.3, combined max 90.3.
- FASTSTART (default on in k5b_j7): c0/c1 by double floor (exact: c*NU < 2^53, a non-integer quotient is >= 1/G
  from an integer; 2M random cases verified in Python), b_begin by ballot+popc over pre[] (pre strictly increasing).
- WB_TLBW (opt-in): every 2 MB page of the K and V caches is touched once across the grid at entry (CTA c: pages
  c, c+G, ..., warps 1-15 lane 0, result unused) while warp 0 waits for seq_lens. Needs the cache size: Params
  int tlbp = k_cache.numel() >> 21 (was pad_). Targets the ~4 us first-gather latency.
- WB_PEEK (opt-in, run1): thread 0 reads the arrival counter (ld.acquire) right after the tile loop (nothing
  loop-carried), decision broadcast with the merge barrier; if all P-1 others published, the merge pass folds
  their partials and stores the final output; counter reset; no ws write, no publish atomic.
- job7 (wb_job7; this container ran ~6 us faster on b32 than job6's, so compare within a job only): check
  56/56 (defaults); "layer full" PASS with WB_TLBW+WB_PEEK. Bench (local us):
                 k5m_snap  k5b_j6  k5b_j7  FASTSTART=0  TLBW    PEEK    TLBW+PEEK
    full b16     87.84     88.35   87.69   88.04        88.25   88.29   88.40
    full b32     163.78    163.44  163.84  164.25       163.41  163.09  163.46
    sliding b32  11.54     -       11.88   -            -       -       11.78
  => all within noise (+-0.5 us) on full; sliding b32 +0.34 (job8 A/Bs it).
  Trace k5b_j7 b32: prefix 0.92, c0c1 1.16, b_begin 1.36 (FASTSTART: -0.8 us on this stretch), page-table LDGs
  issued 2.1, first row back 3.4, prologue issued 4.5, tile0 7.8 (first gather ~4.4 us after issue = the start
  burst with 16 MB in flight; TLBW does not change it, so it is not the TLB). Streaming tile0 -> tiles (med 154.7)
  = 147 us = 4.45 TB/s: the steady state runs at HBM speed. What is left: the start (~3.4 us of idle HBM before
  the first gathers) and the end (spread med->max 4.5 us + merge 1.2 + publish 1.2 + combine 2.2 + exit ~1.5).
  b16: stream 72 us (4.54 TB/s), tile0 7.7, tiles max 82.4, end ~88.
- WB_FR (opt-in, run1): request-boundary fast restart for a CTA whose range continues into request b+1 (without
  its new token). In seg 0's prologue the CTA cp.asyncs seg 1's page-table entries for its first NPRO tiles (smem
  frrow), its Q row (frq), position and gates. At the boundary (after cp_wait_all) seg 1's first NPRO tiles go
  into stages 0..NPRO-1 BEFORE seg 0's lane merge (lane partials move to each lane's stage DST-1); seg 1 skips its
  input loads and prologue. Removes the page-table hop and overlaps the merge with seg 1's first gather.
- job8 (wb_job8, k5b_j8): check 56/56 with WB_FR=1; "layer full" PASS with WB_FR=1,OVH=512. Bench (local us):
                 k5m_snap  k5b_j8  FR      FR+OVH1024  FR+OVH512  FR+OVH0
    full b16     88.01     87.95   88.14   87.90       90.89      93.56
    full b32     163.47    163.03  163.82  164.22      167.31     168.26
    sliding b32 (ALIGNED in k5m: G=128, cpr=4; not FLAT): km 11.49 | k5b 11.83 | LATESCALE=0 11.87 |
    FASTSTART=0 11.73 | FR 11.95 | OVH512 11.66 | OVH0 11.73 (OVH does not apply to ALIGNED: +-0.2 is noise;
    k5b vs k5m ~+0.3 from codegen of the shared run1 template).
  => FR is correct but buys nothing, and lowering OVH under FR still hurts: the ~10-12 us boundary cost is not
  the page-table hop or the merge; it is the drain/refill (the CTA's in-flight bytes go to zero at the boundary
  and the first gather of the next segment takes the loaded latency again). Only true cross-segment flow
  (issuing seg 1's tiles during seg 0's last iterations, with the lane-merge area outside the live ring slots)
  removes it; after rebalancing that is worth <= ~2 us at b32 / ~1 us at b16.
- CONCLUSION for FLAT on run1 (k5m): the steady state runs at HBM speed (b32 4.45 TB/s, b16 4.5 TB/s from tile0
  to the median CTA's last tile). The rest is a fixed start (~7.8 us to tile0: 0.9 prefix, ~1.2 params, ~1.2
  page-table hop, ~4.4 us first gather inside the 16 MB start burst) and the end (SM-speed spread med->max ~4.5
  us, then merge 1.2 + publish 1.2 + combine 2.2 on the slowest CTA). None of FASTSTART, LATESCALE, TLBW,
  PEEK (post-loop; it fired for only ~1 of 32 requests: the second-slowest CTA's publish lands ~2.5 us after its
  tiles), FR, OVH-not-on-request-0 moved the end-to-end time beyond +-0.5 us. Not recommended for merge.

## Hand-off note (WD, after round 6)
- WD parked verify-1 full (context exhausted). Its k13v1 experiments there (dry warm-up pass, L2 evict_first on K/V)
  were a dead end and neutral, so nothing of WD's needs merging into k10v. Next owner of verify-1 full: start from
  dev/wb/k10v.py (job23) and first try WB_TSPLIT on the !GSM path (it is gated on ROPE today); see dev/wd/NOTES.md.
- WD's final finding, relevant to the open run8 start-chain question (job21/22: "any change to how the length
  arrives made the first K/V issue later"): a cold load whose result is used right away stalls the warp for a DRAM
  trip and can gate a barrier. run8's RTAB prologue does `tpos = (float)__ldg(p.positions + ...)` for tid < 256
  BEFORE the length barrier (I2F right away -> warps 0-7 stall there). Untested lever for b1 sliding: keep the
  position as an int (or cp.async it to smem) and convert it after the barrier, then retry WB_EXP's SPEC/LCP.

## Round 7 (coordinator: half-size CTAs, 2/SM, one cluster per request for verify sliding b16/b32) - evaluation
- job24 (dev/wb/occ.py, cuOccupancyMaxActiveClusters, H200 132 SMs): 256-thread CTAs at <= 113 KB smem run 2/SM;
  co-resident clusters cl2 132 | cl4 62 | cl6 39 | cl7 32 | cl8 30 | cl16 14 (114 KB+: 1/SM, 66/30/17/15/15/7, same
  as 512-thread CTAs).
- Fits: b16 cl8 (16 clusters, 128 half-CTAs), b32 cl4 (32 x 4 = 128) and b32 cl7 (32 x 7 = 224). Does not fit:
  b16 cl16 (needs 16, 14 co-reside), b32 cl8 (needs 32, 30).
- Smem of a 1-lane CLU run8 at DST_S 3: ~147 KB as laid out today. To reach <= 113 KB: size V^T for 2 warpgroups
  (-18 KB), and alias the 38 KB receive buffer into the idle ring. Aliasing forces a cluster barrier after every
  CTA's tile loop before any push, so all DSMEM traffic starts after the slowest CTA. Result ~109 KB. DST_S 4 would
  need ~125 KB.
- Why b16 cl8 / b32 cl4 would lose: they halve the lanes, from 256 lanes today (128 CTAs x 2) to 128. At 3 x 16 KB
  stages per lane that is 6 MB in flight instead of 12 MB; Little's law at 4.3 TB/s and ~2 us loaded latency wants
  ~8.6 MB. Measured precedent (job17, 1 CTA/SM, 192 lanes): b16 cl6 +1.8 us, b32 cl3 +4.0 us. The b32 cl3 trace
  had a tile phase +2.9 us vs 256 lanes; 128 lanes is worse. The per-lane tile count doubles (b16: 8-9 instead of
  4-5). The end chain saves at most ~1.3-2 us (3.8 us global chain -> DSMEM exchange ~1.6 us at the ~0.5 TB/s per
  GPC measured on b1 cl16 + merge/epilogue ~0.8).
- Only plausible variant: b32 cl7 (224 lanes): ~+0.5-1 us tile phase vs ~-1.3 us end chain = maybe -0.3..-0.8 us
  on b32 sliding. Not built: it needs a run8 NW=8 instantiation (Shape<8, 8>, no lane merge, 1 lane), the smem
  re-layout above, the post-tile cluster barrier, and host plan/launch for 256-thread cluster launches.

## Start-chain stamps (job25: dev/wb/stamp.py on k10v / k10v_exp / k10v_j21; flush; b1 sliding cl16, medians, us)
  pre-ld = thread 0 at the length barrier, st0 = after it, st1 = split/hit done, it0 = first page-table row landed
                         pre-ld  st0   st1   it0   issued tile0
  A  default (j20)        1.15  1.29  1.72  1.92  2.26   3.38
  B  cp.async len + spec  1.34  1.43  1.88  2.11  2.30   3.40
  B0 cp.async len         1.31  1.41  1.82  2.05  2.38   3.47
  C  LENR + reg spec      1.89  1.94  2.38  2.56  2.78   3.99
  C0 LENR                 1.50  1.56  1.98  2.18  2.55   3.84
  b16 sliding A / B0: pre-ld 1.26 / 1.36, st0 1.36 / 1.48, it0 2.08 / 2.26, issued 2.58 / 2.77.
- Closed: (1) the page-table trip after the length is only ~0.2 us (it0 - st1 in every variant), so speculation can
  save at most ~0.2 us, and its extra pre-barrier code costs more than that. (2) The barrier time is the length
  load's issue point plus its latency. Default A issues the length __ldg as the kernel's first global load; the
  cp.async version issues it later in program order (B0 +0.16). LENR lets the compiler sink the register load
  toward its use right before the barrier (C0 +0.35). The register spec spills on top (C +0.74).
  The prologue code is cold after the L2 flush, so anything placed ahead of the length load delays it.
- New lever (not tried): after the barrier, ~1 us of the start chain is code, not memory: split/hit arithmetic
  st0 -> st1 0.43 us (TSPLIT runtime int divisions + cold fetch of ~250 instrs / 4 KB between the barrier and the
  first LDGSTS), page-table rows 0.2, issue_rows it0 -> issued 0.34. Compacting that path (division by cpr via a
  magic multiplier prepared before the barrier, no dead split variants in the instantiation, a lean PAGE1 issue)
  could take ~0.3 us off every verify sliding shape.

## Round 8: post-barrier start-chain compaction (dev/wb/mk_k13v.py on dev/k12.py; WB_LEAN, run8 ALIGNED RoPE only)
- Steps (builder STEP(n); WB13_SKIP=list to drop steps):
  1+2. TSPLIT without division after the barrier: part and the magic m = floor((2^32-1)/cpr)+1 are prepared before
       it, t = umulhi(n, m) (exact for n*cpr < 2^32); 32-bit window start.
  3.   ptrow1 base pointer for the page-size-1 first rows.
  4.   RTAB inputs (positions, inv_freq) by cp.async into the thread's own slots before the barrier. The compiler had
       sunk the LDG.64 + I2F.S64 into the post-barrier span.
  5.   32-bit npast / has_new / complete.
  7.   part/magic stored to misc[10..11] by warp 1 before the barrier (the compiler had sunk the uniform division).
  8.   asm predicated page-table loads from one base address per tile.
  9.   prologue K/V issue with 32x32->64 unsigned row offsets (issue_rows0; the tile loop keeps issue_rows).
- Local SASS barrier -> first K/V LDGSTS: k12 250 / 246 instrs (attn_8_1_1_1 / _1c) -> all steps 130 / 124.
- job26 (all steps; check 0 failures): the traced build starts faster (b1 issued 2.30 -> 2.00, tile0 3.43 -> 3.20;
  b16 2.62 -> 2.34 / 4.30 -> 3.92; b32 2.98 -> 2.75). But the bench of the untraced build is SLOWER:
  v1 sliding 9.56/9.47 -> 10.17/10.25, v16 12.54/12.55 -> 13.42/13.69, v32 15.74/15.97 -> 16.63/16.61.
  Image NVRTC attn_8_1_1_1: spill 0/0 -> 4/8 B (a long-lived prologue value spilled, reloaded after the tile loop
  on the end chain).
- Local ablation (NVRTC 12.4 as a proxy): step 7 alone causes the cluster-kernel spill, step 3 the attn_8_1_1_1
  spill. Skipping 3, 7, 8 leaves both spill-free with spans 196 / 195. job27 tests v0 (all), v1 (skip 3,7,8),
  v2 (skip 3,8), v3 (skip 7) with image regs.
- job27 (image NVRTC; barrier -> first K/V instrs; spill st/ld B), attn_8_1_1_1 | attn_8_1_1_1c:
    k12 253, 0/0 | 248, 0/0;  v0 132, 4/8 | 124, 20/48;  v1 198, 0/0 | 196, 0/0;  v2 182, 4/8 | 178, 24/44;
    v3 147, 4/8 | 154, 0/0.  check v1 0 failures.  Bench 40 reps (k12 | v0 | v1 | v2 | v3 | k12 dup):
    v1 sliding 9.45 | 10.20 | 9.55 | 10.00 | 9.30 | 9.51;  v16 sliding 12.38 | 13.64 | 13.68 | 13.75 | 13.56 | 12.70
    v32 sliding 15.54 | 16.50 | 16.58 | 16.55 | 16.34 | 15.84
- job28 bisect (verify 16/32 only): w1 (steps 1+2 only: the umulhi split and the 32-bit window start, same split,
  no spills, span 234) is already +1.0 / +1.1 us; w2 (+4) and w3 (+5,9) the same; v1 with WB_LEAN=0 == k12.
  So any LEAN edit in attn_8_1_1_1 (the b16/b32 kernel) costs ~1 us without spills and without a different split.
  The local 12.4 tile-loop SASS is near identical (455 vs 456 instrs); presumably the image compiler's schedule of
  W8's tile loop moves. The traced build (TR stamps) did not show it (job26: b16 combined 11.40 vs 11.49) -> do
  not trust traced builds for small prologue edits.
- Deliverable candidate (job29): LEAN only on the cluster instantiation (CLU, b1 sliding) with step 7 off
  (= v3 for that kernel: 154 instrs, spill-free, 9.30 vs 9.45/9.51 in job27). All other kernels are SASS-identical
  to k12 (local diff 0).
- job29 (k13v = LEAN on the cluster kernel only; image: attn_8_1_1_1c span 248 -> 154, 0/0 spills; attn_8_1_1_1
  identical span 253, 0/0; check 0 failures). Bench 40 reps x2 (k12 | k13v | k12 dup | k13v dup):
    v1 sliding 9.26 | 9.43 | 9.16 | 9.51;  v16 sliding 12.10 | 12.28 | 12.18 | 12.30;  v32 sliding 15.37 | 15.62 |
    15.42 | 15.54;  geomean x1.9404 | x1.9312 | x1.9421 | x1.9285.
  Verdict: no gain. b1 +0.25 here vs -0.18 in job27, so the effect is within run-to-run variance. Do not merge
  k13v. Cold-code bytes in the post-barrier span are not a measurable cost at this size: the traced 0.2-0.3 us
  earlier first issue does not survive in the untraced bench. The tile-loop schedule is fragile to prologue edits
  (+1 us on b16/b32).
- WC told the recipe + caveat (runA1 span 377 instrs with two 32-bit divides and a 64-bit division CALL; run1
  sliding 264).

## Round 9: verify-1 sliding on two 16-CTA clusters (X2) - dev/wb/mk_k14v.py on dev/k13.py -> dev/wb/k14v.py
- Host: B == 1 with a 16-CTA cluster plan -> cpr 32, G 32 (2 clusters co-reside; 7 allowed). Disable with NOX2.
  Device: the cluster size comes from %cluster_nctarank. The in-cluster owner combine is unchanged. Then a 2-way
  cross-cluster merge per owner rank: each owner writes its 8 merged rows to its own ws slot; one acq_rel atomic
  per (request, rank) at counters[512 + rank]; the last of the two loads the partner's rows (ld.cg), merges and
  stores. Other kernels are SASS-identical to k13.
- job30 (image: attn_8_1_1_1c 0/0 spills; check 0 failures). Bench 40 reps
  (k13 | k14v | k14v NOX2 | k13 dup | k14v dup): v1 sliding 9.44 | 9.82 | 9.52 | 9.39 | 9.76. X2 is +0.35 us.
  v16/v32 sliding and the full shapes are unaffected (noise).
- Trace (flush, cluster 0): tiles med 4.29 / max 4.98 (k13 ~4.9 / 6.0: the tile phase is ~1 us shorter as
  intended); merged max 5.79; bulk push -> released (DSMEM exchange) 6.78 -> 7.55, the same ~1.2-1.7 us as one
  cluster; in-cluster merge s9 7.70; the cross-cluster merge (write + release atomic + partner load + merge +
  epilogue) s10 7.78 -> combined med 8.90 / max 9.38 = +1.1 us. It costs as much as the tile phase saves, plus a
  little.
- Why the end chain cannot shrink this way: the reduce-scatter moves (cl-1)/cl x 33 KB per CTA whatever the tree
  shape, and the exchange of ~0.5 MB per 16-CTA cluster is DSMEM-bound. A second cluster only adds a global
  round-trip pair. Splitting the two clusters by output rows instead of tokens would need no cross merge (the
  extra K/V read is ~1 MB, trivial), but it changes the tile loop's row slabs (W8's code).
- Verdict: not for merge.

## Round 10: verify-1 sliding split by OUTPUT ROWS over NRS 16-CTA clusters (dev/wb/mk_k15v.py on dev/k14.py)
- Each cluster covers all tokens (the same 16-part split as today) and owns RR = 128 / NRS output rows: cluster rs
  takes KV head rs % 2 and query tokens [j0, j0 + 16/NRS). Lane 0 stages only those rows under a compact index
  s = (j - j0) * 8 + g. The bulk push / mbarrier / owner combine run over RR rows, so the in-cluster exchange is
  NRS times smaller. The epilogue maps s back to j * 16 + head * 8 + g. The other rows are computed and dropped
  (the duplicated K/V read is ~1 MB per extra cluster). No core or tile-loop function is touched; non-cluster
  kernels are SASS-identical to k14. W9 was told the touched lines.
- job31 (runtime NRS from Params.pad_; check 0 failures, RS8=4 smoke 0 failures). Verify-1 sliding, 40 reps
  (k14 | rs2 | rs4 | rs1 = new code, 1 cluster | k14 dup | rs2 dup | rs4 dup):
    9.48 | 9.09 | 9.08 | 9.78 | 9.51 | 8.98 | 9.10 -> -0.46 us (rs2), -0.40 (rs4). The runtime-NRS code costs
  +0.3 at NRS 1. v16/v32 sliding and the full shapes: noise.
- job32: NRS as a template parameter (dedicated kernels attn_8_1_P_1r = 2 clusters, _1q = 4; _1c == k14).
- job32 (template NRS; image: attn_8_1_1_1r / _1q / _1c all 0/0 spills; full check 0 failures; RS8=4 smoke 0
  failures). Verify-1 sliding, 40 reps (k14 | rs2 | rs4 | j31 runtime-NRS | k14 dup | rs2 dup | rs4 dup):
    9.18 | 8.50 | 8.86 | 8.62 | 9.14 | 8.59 | 8.64  -> rs2 -0.62 us (verify-1 load geomean x2.19 -> x2.29).
  Control on the identical b16/b32 kernels (k14 | k15v | dups): v16s 12.06 | 12.20 | 12.21 | 12.08 (noise);
  v32s 15.31 | 15.59 | 15.39 | 15.56 (+0.2 in both copies here and +0.15 in job31 with an SASS-identical kernel:
  bench order or noise); full shapes noise.
- DELIVERABLE (round 10): dev/wb/k15v.py (= k15v_j32.py), `python3 dev/wb/mk_k15v.py` on dev/k14.py, RS8=2 default
  (RS8=4 and RS8=1 via K2_DEFS).

## Round 11: verify sliding b16 / b32 (ALIGNED B > 1, global combine), base dev/k15.py
- job33. Traced vs untraced (the same file, K2_TRACE on/off, 30 reps x2): b16 12.48/12.38 -> 12.90/12.76; b32
  15.93/16.02 -> 16.25/16.36. The traced build is +0.35-0.4 us, so phases are representative to ~0.4.
- Trace k15 (flush, medians / max, us):
    b16: issued 2.49 / 2.94, tile0 4.30 / 4.94, tiles 6.90 / 8.42, merged - / 9.31, publish - / 10.16,
         released 9.74 / 10.62, combined 11.49 / 12.27
    b32: issued 2.69 / 3.30, tile0 4.40 / 5.18, tiles 10.06 / 11.65, merged - / 12.54, publish - / 13.41,
         released 12.67 / 13.86, combined 14.43 / 15.55
  Budget b16: first issue 2.5 + first data 1.8 (the 12 MB start burst: 128 CTAs x 2 lanes x 3 stages x 16 KB) +
  slowest tile phase 3.5 (part 0 has 5 steps on one lane) + end chain 3.85 (merge 0.9, publish 0.85,
  release-observe 0.45, combine 1.65). The coordinator's 10.4 estimate misses 0.5 (first data), 0.4 (5th step),
  1.35 (end chain). b32: 1024 tokens per CTA = 8 lane-tiles (not 4); the tile phase (~7 us from tile0 for the
  slowest) is at the bandwidth floor (33.5 MB / 4.3 TB/s = 7.8 us); end chain 3.9.
- FLATS (FLAT split for sliding B > 1, all SMs): b16 22.0, b32 24.9 (k15 12.4 / 15.7). The FLAT RoPE kernel
  (attn_8_1_1_0, spills) with 2-segment CTAs is far slower -> dead.
- job34: warm-mode traces (does cold code cost the combine 1.65 us?) and WF's COMB8 hook for the combiner count.
- job34 (warm-mode traces, k15): b16 released -> combined 8.96 -> 10.72 = 1.76 us (flush 1.75); b32
  12.23 -> 14.02 = 1.8 us. The combine is NOT cold-code bound. All 128 combiners read every partial of their rows
  at the same instant right after the last release (b16: 8 x 33 KB per request x 16 = 4.2 MB L2 burst; b32 the
  same), i.e. L2-bandwidth bound plus latency. COMB8 hook (fewer combiners): b16 1 combiner 20.1, b32 2
  combiners 17.4 (k15 12.4 / 15.6) -> more combiners are better; K is already capped at P.
- WB_MAL (dev/wb/mk_k16v.py on dev/k16.py; run8 ALIGNED RoPE global path only): merge as partials land. Each CTA
  combines a fixed row block; it publishes with one st.release of a per-(request, part) flag = generation + 1
  (generation from an early relaxed atomic on a (gen << 8 | count) word; the last arrival bumps it), then polls the
  P flags (warp 0 ld.acquire + ballot) and folds newly landed partials, 4 per batch, into a running (m, l, O) per
  item. After the slowest partial lands, each combiner reads only that partial's rows (4-8 KB).
  Local NVRTC: attn_8_1_1_1 spill 0/0 -> 8/8 (early atomic) or 4/8 (atomic at publish), all outside the tile loop.
  job35 tests both on the image.
- job35 (on k16; check 0 failures; image attn_8_1_1_1 spill 0/0 -> 8/8 early, 4/8 late). 40 reps
  (k16 | mal | late-atomic | WB_MAL=0 | k16 dup | mal dup):
    v16 sliding 12.99 | 12.67 | 12.79 | 12.96 | 12.82 | 12.72;  v32 sliding 15.55 | 14.70 | 15.06 | 15.68 | 15.72 | 14.88
  -> MAL: v32 -0.85 us, v16 -0.2. The early atomic beats the late one. Full shapes are noise.
  Trace b16: publish (flag release) 0.54 after merged (old atomic 0.85), but released -> combined max still ~1.9 us.
  With P = 8 and 4-wide batches the slowest CTA, which finds all its 8 partials ready at once, needs two
  dependent load rounds. b32 (P = 4) has one round, hence the larger gain.
- Base moved to k17 (W9's XU-free exp). job36: k16v on k17 with 8-wide load rounds (WB_MALB=8 default, 4 as A/B).
- job36 (on k17; image attn_8_1_1_1 0/0 for the 8-wide build; check 0 failures). 40 reps
  (k17 | MALB 8 | MALB 4 | k17 dup | 8 dup | 4 dup): v16 sliding 12.06 | 12.09 | 11.81 | 11.95 | 12.04 | 11.87;
  v32 sliding 14.73 | 14.85 | 14.25 | 14.81 | 14.80 | 14.19 -> 4-wide rounds win, 8-wide gives nothing.
  Poll-loop SASS: LDG.STRONG.GPU + CCTL.IVALL per lane (no membar); flag store MEMBAR.ALL.GPU + STG.STRONG.
- job37 FINAL (k16v = k17 + WB_MAL, MALB 4; image attn_8_1_1_1 spill 4/4 outside the loop; full check 0
  failures). 40 reps x2 (k17 | k16v | k17 dup | k16v dup):
    v1 sliding 8.66 | 8.69 | 8.81 | 8.70;  v1 full 18.25 | 17.90 | 18.20 | 17.83;  v16 sliding 12.06 | 12.02 |
    12.04 | 11.95;  v32 sliding 15.04 | 14.32 | 14.86 | 14.28;  v16 full 112.45 | 112.20 | 112.39 | 111.97;
    v32 full 208.44 | 207.98 | 208.26 | 208.50;  verify geomean x2.0572 | x2.0701 | x2.0532 | x2.0717.
  -> v32 sliding -0.65 us, v16 sliding ~-0.07 (noise level), +0.7% verify geomean.
- DELIVERABLE (round 11): dev/wb/k16v.py (= k16v_j37.py), `python3 dev/wb/mk_k16v.py` (default base dev/k17.py).
- Open for b16: with P = 8 the slowest CTA's own combine needs two 4-wide rounds after its flag (8-wide is not
  faster). The other combiners need only the slowest partial. Remaining b16 end chain ~3 us; candidates: WE's
  per-(request, KV head) clusters, or the slowest CTA pre-merging the others' partials before its own publish.

## Round 12: verify sliding b16/b32 with the combine inside a cluster (base dev/k18.py)
- Co-residency wall (job24 + WE's probe): 512-thread CTAs at 1 CTA/SM fit only 15 x C8 or 30 x C4 (and 66 x
  C2), so the scored shapes cannot get one cluster per request (b16: 16 x C8) or per (request, KV head)
  (b16: 32 x C4; b32 needs 32 x C4 per request). WE measured that 32 x C4 really runs a second wave (11.9 vs 8.0 us
  in its gather probe). Fitting full-row per-request clusters: b16 C6 / b32 C3 (96 CTAs). 64-B half rows per head
  cost +0.4 us (b16) / +1.0 us (b32) in WE's gather probe, and computing one head per CTA needs a
  tile-loop/staging change (W8/W9's code).
- WB_MALP (dev/wb/mk_k17v.py on k18; lane 1 pre-folds landed partials during lane 0's merge; check 0 failures):
  v16 sliding 11.92/12.06 vs k18 11.72/11.94 (MALP=0 11.72) -> no gain. Trace: merged max 10.24 (vs ~8.9): the
  pre-fold outlasts lane 0's merge and the merge barrier delays the publish. Off by default.
- job39: per-request cluster plan through the CL8S16 / CL8S32 hooks on k18 (C6 / C3, and C5 / C2).
- job39 (k18 hooks; 40 reps; k18 | C6/C3 | C5/C2 | k18 dup | C6/C3 dup):
    v16 sliding 11.76 | 12.07 | 12.87 | 12.04 | 12.13;  v32 sliding 14.19 | 17.55 | 21.03 | 14.12 | 17.38
  b16 C6 trace: the tile phase is balanced (max - med 0.7 vs 1.4) but ends later (max 8.29 vs ~7.9; 96 CTAs). Bulk
  push -> all rows received 1.3 us (DSMEM exchange of ~27 KB per CTA), in-cluster merge of 6 partials 0.9,
  epilogue 0.45: an end chain of ~2.7 us, about the global MAL chain, plus a longer tile phase. b32 C3 (96 CTAs,
  11 tile steps per lane) loses 3.3 us.
- Verdict round 12: no cluster layout beats k18's global MAL path at B > 1 without a single-head tile (64-B half
  rows, a tile-loop change), and that also needs co-residency, which C4 lacks for 32 clusters. dev/wb/k17v.py
  = k18 + WB_MALP opt-in (default off; equals k18 in effect, malp0 11.72 vs k18 11.72/11.94). Not for merge.

## Round 13 (A): reconciliation - the sliding window is 2048 (reference.py SLIDING_WINDOW), not 4096
- My round-11 tile counts and bytes were 2x too high. job40 (debug counter: valid 16-B K/V cp.async per CTA,
  dev/wb/k18_cnt.py): b32 part 0 9216 (576 tokens, 9 tiles), other parts 8192 (512 tokens, 8 tiles); b16 part 0
  5120 (5 tiles), others 4096 (4 tiles). That is exactly 16 copies = 256 B per token, no duplicates; totals b32
  16.8 MB, b16 8.4 MB (floors ~3.9 / 2.0 us).
- So the b32 tile phase (tile0 med 4.61 -> tiles med 9.64 / max 10.93; 4 lane-steps, 5 for part 0's lane 0) runs
  ~1.25 us per lane-step at ~2.4 TB/s effective: latency/issue-limited, not at the bandwidth floor (W8's resident
  compute is 0.84 us per lane-tile). Levers: more bytes in flight per SM (DST 4 does not fit; WE: cp.async issue
  capped per warp at ~13 KB in flight, so spread the issue over more warps) and part 0's extra tile.

## Round 13 (B): verify-1 full end chain (base dev/k18.py)
- WF (decode-1 full, run1): the cluster-pair (C2) two-level end chain is slightly slower than k18 (14.32/14.00 vs
  13.94/13.94; C2 off 13.87). The one-row global combine is latency-bound, not bytes-bound: halving the partials
  removed nothing and the pair exchange adds 0.1-0.2 us. Not rebuilt for verify-1 full.
- WB_MALF (dev/wb/mk_k18v.py, merge-as-landed generalised to P = 132 with coalesced (item, group) folds, warps 0-4
  polling 132 flags; check 0 failures): verify-1 full 20.52/20.29 vs k18 18.46/18.03 (MALF=0 18.09) -> +2.2 us.
  Sub-stamps (job42), flush (warm): last publish 14.77 -> all landed + folded 16.07 (+1.3) -> gate wait + barrier
  16.21 -> 32-entry cross-group reduction 19.74 (+3.5 us flush, +1.0 warm) -> epilogue 20.88 (+1.1 flush, +0.4
  warm). The tail is cold-code bound: the reduction loops are unrolled (several KB of first-executed code). Even
  with a compact reduction the poll/fold (+1.3 after the last publish) only matches k18's
  publish -> release -> combine. MALF off by default; k18v == k18 in effect. Not for merge.
- k18's own verify-1 full tail is the same kind (post-loop code 21.7 KB for attn_8_0_1_1; WD's earlier finding).
  The lever left is code size and latency: rolled one-pass reductions after one xor-16 shuffle, gate sigmoid
  precomputed while spinning (WF's / WS's recipe), not bytes or clusters.

## Round 14: verify-1 full post-loop code diet (base dev/k20.py)
- Measured first (job43, k20, flush vs warm, medians): merge (tiles -> merged) 0.91 vs 0.96, publish 0.83 vs 0.81,
  released -> combined 2.93 vs 2.96. The verify-1 full tail is NOT cold-code bound in k20 (unlike MALF's own
  unrolled 32-entry reduction in round 13). The flush-only cost sits before the tile loop (tile0 max 6.38 vs 4.43).
  WB_GEN=1 (no end-of-kernel done atomic): 18.47/18.21 vs 18.36/18.25 -> neutral.
- Combine sub-stamps (job44, k20, flush / warm): released -> loads + merges (132 partials, one row) 1.26 / 1.30;
  -> gates + red barrier 0.13 / 0.12; -> 16-entry reduction by 16 threads 1.14 / 1.12 (a serial dependent chain);
  -> epilogue 0.43 / 0.42.
- WB_PRED (dev/wb/mk_k21v.py, !RoPE only): the 16 entries item-major, 256 threads x 4 xor-shuffle merges, warp 0
  epilogue. Check 0 failures. Bench (job45, 40 reps): verify-1 full 18.54/18.55 vs k20 18.34/18.05 (PRED=0 18.17)
  -> +0.3. Trace: released -> combined 2.8 (k20 2.9), so the reduction gain did not appear end to end. The tile loop
  SASS changed (register allocation, 467 vs 468 instrs) and tiles max was 12.99 vs ~12.4-12.8. Off by default:
  k21v (default build) is SASS-identical to k20 for attn_8_0_1_1 / attn_8_1_1_1.
- Conclusion: verify-1 full's ~6 us end chain (merge 0.9 + publish 0.83 + release-observe ~0.6 + combine 2.9) is
  latency, not code bytes: the L2 burst of the 132-partial reads (1.26), serial reductions, the epilogue. Every
  post-loop edit so far has perturbed the tile loop's allocation. A structural change (fewer partials per row:
  split-K tree inside the CTA lanes, or larger tiles per CTA) is needed, not a code diet.

## Round 15: verify-1 full end chain by FP32 atomic accumulation (unified reference max)
- Microbench (job46, dev/wb/redbench.py; 132 CTAs at once, kernel time minus an empty graph):
    128 rows x 64: plain 32 KB store per CTA 1.54 us; red.global.add.v4.f32 into one shared ACC 3.90 us; 2 copies
    2.85; 4 copies 2.43; 8 copies 2.05; scalar float atomics 19.2 (1 copy) .. 6.8 (8 copies).
    16 rows (decode): plain 1.18; red.v4 1 copy 1.86 / 4 copies 0.96 / 8 copies 0.77.
- WB_ACC (dev/wb/mk_k21u.py): WS's fixed reference REF = 64 log2 units (RMS-normalized q / keys bound |m| < 64),
  no per-row M_ref. Lane 0's merged rows: slot store (exact fallback) + red.v4 of 2^(m - REF) (O, l) into ACC
  copy c % NC ([row][68], 16-B aligned rows). A row max outside [REF - 90, REF + 40] sets a per-head flag.
  Combiners (unchanged selection) read the flag after the acquire: clear -> out = sum of the NC copies, then zero
  their rows (after a barrier: the first version raced and gave NaN on the second call); set -> the exact combine.
  The last done combiner clears the flag. ACC sits behind the counters buffer.
- job47b (on k20; check 0 failures; forced fallback REF=500 and NC=8 smokes 0 failures). 40 reps:
  v1 full k20 18.34 / 18.28 | NC4 17.98 / 18.20 | NC2 18.19 | NC8 18.28 | forced fallback 18.57 -> about -0.2.
  Trace: the combine shrinks (released -> combined ~1.2 us vs 2.9), but merged moves +1.9 us (merged max 15.25 vs
  ~13.3): the 2560 reds per CTA (338K for the launch) are L2-atomic-throughput bound, and the release fence waits
  for them. With 128 rows per CTA the atomics cost about as much as the combine they remove. WS's decode version
  wins (-0.9) with 8x fewer rows.
- job49: final A/B on k21 (ACC behind WS's run1w sums).
- job49 (on k21; check 0 failures, forced fallback 0 failures). 40 reps (k21 | NC4 | NC2 | k21 dup | NC4 dup |
  NC2 dup): v1 full 17.97 | 18.80 | 18.68 | 18.03 | 17.62 | 18.61 -> neutral to worse (NC4 copies 18.80 / 17.62:
  large variance). Decode-1 and verify-1 sliding are unchanged (their kernels are untouched).
- Verdict: for verify-1 full (128 rows per CTA) the atomic accumulation trades a ~1.7 us faster combine for a
  ~1.9 us slower publish (L2 atomic throughput + release fence on the reds). WB_ACC is off by default in
  dev/wb/k21u.py, so the default build is SASS-identical to k21 for attn_8_0_1_1 / attn_8_1_1_1. Kept as opt-in.

## Round 16: FA3-style warp-specialized consumer microbench - stopped before any run
- Plan (not built): extend dev/w8/t_ws.py 'cons' to 3 consumer WGs @160 regs (setmaxnreg.inc) + 1 producer WG @24
  taking (tile, KV head) units round-robin. Each consumer then needs two O / m / l sets (units alternate heads):
  ~72 regs of accumulators + s[32] + qa[2][2][4], tight at 160 for lockstep and likely spilling for the SS
  pipelined variant. Plus 2 consumers @232 pipelined as the reference.
- Closed by the coordinator before any job: WS's producer probe (dev/ws/NOTES.md Round 5) shows bulk / TMA
  1-row copies issue-bound and 20-30% slower than the cp.async ring (328 MB: cp.async 4.09 TB/s vs bulk 3.36 / TMA
  3.28 with 16 issuing warps; 1.0-1.6 TB/s with 1 warp per lane), and W8's cp.async producers plateaued at
  3.6 TB/s. A <= 1-WG producer cannot feed a warp-specialized verify-full kernel. No dev/wb/t_fa3.py.

## Round 17: W9's FLAT-full instruction diet ported to the ALIGNED full b1 kernel (dev/wb/mk_k23v.py on dev/k22.py)
- Gate change: FD = !ROPE && !CLU && (!ALIGNED || WB_FDA) (was !ALIGNED && !ROPE): F1 window = 0, F2 predicated
  K/V cp.async issue, G1 next rows before the lane barrier, G3/G4 V^T ldmatrix / fence order now also in attn_8_0_1_1
  / attn_8_0_0_1. W9's S items (FD0) stay FLAT-only: S3 reads the FLAT split before the length barrier; in ALIGNED
  the page-table rows already follow the length barrier directly.
- Image NVRTC (job50): attn_8_0_1_1 0/0 spills, tile loop 470 -> 405 instrs with 0 LDL / 0 STL, barrier -> first
  K/V 330 -> 276 instrs, no C75xx. Every other kernel is SASS-identical to k22 (local diff 0: FLAT full, sliding,
  cluster, run1).
- Bench 40 reps x2: job50 v1 full k22 17.73 / 17.79 -> k23v 17.24 / 17.19 (WB_FDA=0 17.78); job51 k22
  18.51 / 18.54 -> 17.48 / 17.52. -0.5 .. -1.0 us (3-6%); the other verify shapes are noise. Full check 0
  failures (both jobs).
- A/B, off: WB_ISPLIT (32-bit division for the ALIGNED split instead of the double floor) 17.75 (+0.5 vs k23v;
  barrier -> first K/V grows 276 -> 324). WB_CLC (constant cluster size in the row-split cluster kernels, WE's runE1
  finding: 3224 -> 2880 instrs): no effect, because verify-1 sliding now runs WS's runA8h, not those kernels.
- DELIVERABLE: dev/wb/k23v.py (WB_FDA on, ISPLIT / CLC off).

## Round 18: verify-1 full START (dev/wb/mk_k26v.py on dev/k25.py)
- k25 timeline (job52, flush, medians, us): thread 0 at the length barrier 0.80 -> split 1.25 -> first page-table row
  1.66 -> first K/V issued 2.29 -> tile0 4.10 (max 6.48) -> tiles 10.07 (max 12.46). Warm: 0.62 / 1.07 / 1.50 /
  2.11 / 2.98 / 8.94. The bulk of the stream starts ~2.3 us and takes ~8 us for 20.5 MB (~2.6 TB/s): per-lane tile
  steps (~4.7 per lane), not bandwidth. WS's faithful probe: first stage lands 2.6-2.85 us after CTA start, issue
  done 2.27, stream 5.9-6.2 us.
- WB_KSPEC (verified batch-1 speculation): the length goes to smem by cp.async (its own oldest group); the CTA runs
  the unchanged split / page-table / K/V-issue code for seq_g = max_pages * page_size - L before the length
  barrier, then waits the length group (cp.async.wait_group DST-1), takes the barrier, and on a miss drains
  (wait_all) and re-runs the same code with the true length (backward goto, one code copy). WB_KSPEC_MISS=1 forces
  misses (test).
- job53 (check 0 failures; forced-miss smoke 0 failures; image attn_8_0_1_1 0/0 spills, loop 405 -> 412 instrs,
  0 LDL / 0 STL). 40 reps: v1 full k25 17.78 / 17.56 | KSPEC 17.47 / 17.21 | KSPEC=0 (same file) 17.55 | forced
  miss 19.81 -> about -0.3 us. The other shapes are unchanged (other kernels SASS-identical to k25).
- job54 sub-stamps (flush, medians): rows landed (first issue_rows) 1.47 (KSPEC=0: 2.15) -> all speculative copies
  issued 2.30 (0.83 us to issue 3 tiles x 2 lanes = 96 KB per SM: cp.async issue backpressure, WE ~60 GB/s per SM)
  -> length landed 2.34 -> barrier 2.37 -> tile0 4.22 (KSPEC=0 4.93 same job). Speculation hides the length trip
  (~0.7 us at tile0), but the first tile still lands ~1.9 us after the issue completes (WS's probe: ~0.5). The
  remaining start gap is between the issue and tile 0's data (DRAM queueing behind the 12.7 MB start burst plus
  the lane barrier on all 8 warps' copies), not the length / page-table chain.
- DELIVERABLE: dev/wb/k26v.py (WB_KSPEC on). Not built: the full warp-specialized producer / consumer start (WS's
  runA8h). With issue done at ~2.3 us like WS's probe, its gain would have to come from the issue -> tile0 gap.

## Round 19: verify-1 full sum-only combine (fresh-eyes IDEA 2; dev/wb/mk_k27v.py, base dev/k26.py, KSPEC off)
- Coordinator 2026-10-28: KSPEC did not hold up in a combined A/B (k25 17.55 / 17.27 vs k26v 17.85); base is dev/k26.py
  (k25 + W9S decode-1 full start). run8 is byte-identical in k25 / k26.
- WB_SUMC: lane 0 stores each merged row pre-scaled, O * 2^(M - REF), l * 2^(M - REF), M := REF (REF = 64, range
  [REF - 90, REF + 40] or M = -inf), else the classic (O, M) with l NEGATED (per-row range flag in the sign). Plain
  stores, no atomics, no barrier. A scaled partial is a valid classic partial with max REF, so the classic combine
  (reading |l|) is exact on any mix. Combiner (one row, 512 thr = 32 groups x 16 float4): one round of 5 predicated
  (O float4, l) loads, plain adds, one xor-16 shuffle, [16 warps][16] in smem, __syncthreads_and(all l >= 0), warp 0
  adds 16 entries (unrolled), unchanged epilogue; any l < 0 in the row -> the classic combine for that row.
- WB_SPB (default on): pb[] index constant for the one-partial ALIGNED list. Without it the image NVRTC puts pb[] on
  an 8-B stack frame AND perturbs the tile loop (attn_8_0_1_1 405 -> 406, attn_8_0_0_1 565 -> 569 instrs).
  With it: image attn_8_0_1_1 / attn_8_0_0_1 0 stack, 0 spills, tile loops 405 / 565 = k26.
- job55 (on k26v): check 0 failures; smokes: forced fallback (REF = 500) 0, mixed (REF = 25, LO = 0) 0 failures.
- job56 (on k26; check 0 failures; smokes default / SPB=0 / SEPI / fallback / mixed all 0 failures). Trace flush,
  medians (released -> loads+adds -> red visible -> reduction -> combined):
    SUMC   13.77 -> 14.67 (0.90) -> 14.80 (0.13) -> 14.94 (0.14) -> 15.39 (0.45)   = 1.62; last stamp 16.10
    classic 13.66 -> 14.94 (1.28) -> 15.01 (0.07) -> 16.10 (1.09) -> 16.51 (0.41)  = 2.85; last stamp 17.33
  -> released -> combined -1.23 us (kill rule 0.6: passed); warm (job55) -1.09.
  Bench (40 reps, paired with SGLang): k26 17.22 / 17.74 | sum 17.08 / 17.11 | SPB=0 16.74 | SEPI 16.87 |
  SUMC=0 17.34. job55 (k26v base): k26v 18.20 / 17.72 | sum 17.58 / 16.97 | SUMC=0 17.63 | forced fallback 19.00.
  Untraced the gain is ~-0.4..-0.5 us (SUMC builds vs controls, both jobs), less than the traced -1.2.
- WB_SEPI (opt-in, off): sigmoids computed before the barrier from the thread's own 8-B gate copy, __fdividef
  epilogue: epilogue 0.45 -> 0.21 but loads + barrier +0.23 (same released -> combined 1.62); image spills 4/8 B.
- WB_SE1 (default on): the sum path uses run1's compact epilogue (__fdividef; epilogue_store1's code). Image: 0 stack,
  tile loops 405 / 565. Trace flush: epilogue 0.45 -> 0.25 (released -> combined 1.63, loads phase noisier).
- job57 untraced raw medians (bench --no-base, 100 reps, cold L2, interleaved): v1 full k26 17.26 / 17.42 |
  SUMC 16.96 / 16.88 | SUMC + SE1 16.56 / 16.75 -> SUMC -0.42, + SE1 -0.68 us vs k26.
- job58 FINAL (dev/wb/k27v.py defaults SUMC + SPB + SE1 on dev/k26.py): image attn_8_0_1_1 / attn_8_0_0_1 0 stack,
  0 spills, tile loops 405 / 565 (= k26), 0 LDL / STL. Full check 0 failures; smokes default / forced fallback
  (REF = 500) / mixed (REF = 25, LO = 0) 0 failures.
  v1 full, raw medians (100 reps): k26 17.17 / 17.09 | k27v 16.19 / 16.42 | SE1=0 16.67  -> -0.82 us
  v1 full, paired with SGLang (40 reps): k26 17.17 / 16.96 | k27v 16.29 / 16.18           -> -0.83 us
  v1 sliding unchanged (4.74-4.78; other kernels untouched).
- Left: loads phase ~0.9-1.06 us (4.9 MB of L2 sector reads by 128 combiners, ~5 TB/s; only fewer bytes or
  fewer partials per row help), barrier + 16-entry sum 0.3, epilogue 0.25. Ports: WB_MAL's folds (v16s / v32s) and
  WS's runA8h owner fold can use the same 2^REF stores (adds instead of max / ex2 merges).

## Round 20: sum-only owner folds in the sliding cluster kernels (dev/wb/mk_k29v.py, base dev/k28.py)
- WB_FSA1 (runA1h, decode-1 sliding) / WB_FSA8 (runA8h, verify-1 sliding): pushers send 2^(m - REF)-scaled rows
  (m := REF; out of range: unscaled, l negated), owner warp: __all_sync(l >= 0) -> 16 plain adds (no 4-step shuffle
  max chain, no ex2, no 16 weight shuffles), else the classic fold with |l|. These owner epilogues normalize as
  y rsqrt(mean(y^2) + eps (L / vsc)^2) without dividing first, so the range is narrower than v1f's: REF = 16,
  [REF - 48, REF + 28] (sum y^2 <= 2^(63.6 + 2 (m - REF)) worst case; eps (L / vsc)^2 normal for m - REF >= -48).
- job59: check 0 failures; batch-1 smokes default / forced fallback (REF = 500) / mixed (REF = 20, HI = 0 and
  REF = 0, LO = 0) 0 failures. Image: 0 spills; the regs8 "loop" region of attn_1_1_1_1s / attn_8_1_1_1h grows
  1120 -> 1199 / 1758 -> 1868 instrs with the folds on (that region spans the whole body); off = k28 exactly.
  Trace flush (owner received -> out): d1s 0.35 vs 0.37, v1s 0.36 vs 0.42 (pushes +0.03). Bench raw medians
  (100 reps): d1s k28 3.28 / 3.18 | on 3.28 / 3.31 | off 3.30; v1s k28 5.04 / 5.10 | on 5.04 / 5.07 | off 5.06
  -> NEUTRAL (<= 0.06 us traced, nothing untraced). Defaults OFF: dev/wb/k29v.py == k28 in effect.
- Not built: runE1 (decode sliding B > 1, C = 3 / 6) and run8w (v16s C6 cluster, v32s P = 4 global) owners: C <= 6
  sources, no shuffle chain (WE measured the whole E1 owner combine at 131-173 cycles); dropping the max pass and
  C ex2 saves <= ~50 cycles minus the pusher's ex2 + scaling, i.e. <= 0.02 us. run8w's P = 4 path folds as rows land
  (one merge after the last arrival).

## Round 21: half-precision partial exchange (user policy change 15:30; dev/wb/mk_k30v.py, base dev/k29.py)
- (1) WB_H16, verify-1 full (sum-only path): lane 0 stores O / l as FP16 (|O / l| <= 448) over the first half of the
  slot's O area + FP32 W = l 2^(M - REF) (or -l, classic); sum path acc += W float(O / l); classic fallback
  x = float(O / l) |W|. 5 L2 sectors per (row, partial) instead of 9.
  job60: check 0 failures; errors identical to k29 to 3-4 digits (1M-ctx case 9.168e-02 vs 9.166e-02, bound 0.150);
  smokes forced fallback / mixed 0 failures. Image attn_8_0_1_1 loop 405 -> 404 (attn_8_0_0_1: 8-B stack, 4/4 spill).
  Trace flush: released -> loads 0.76 vs 0.89 (-0.13 only), publish 12.50 vs 12.66, combined 14.91 vs 15.14,
  last stamp 15.73 vs 15.92. Bench raw (100 reps): k29 16.51 / 16.26 | H16 16.35 / 16.26 | H16=0 16.16; paired:
  k29 16.42 / 16.09 | H16 16.41 / 16.21 -> NEUTRAL. Halving the bytes takes only 0.13 us off the one-round load
  phase: it is latency-bound (one L2 round trip under 128-CTA load), not bandwidth-bound. WB_H16 off by default.
- (2) WB_H16W run8w C6 cluster (v16s): O / l as FP16, thread-contiguous halves (thread tig's 16 at bytes 32 tig: two
  16-B st.async per row instead of four), (m, l) FP32 at the old offset: 136 B per row (was 264), expect_tx to match.
  (3) WB_H16A8 runA8h (v1s, BULK): staged rows [4][34 floats] = halves[64] | (m, l): one 544-B bulk copy per owner
  (was 1088); owner weight 2^(m - M) l. job61 (on k29; check 0 failures, errors = k29 to 3-4 digits, worst rows
  +-3e-4): raw medians (100 reps) v16s k29 10.40 / 10.43 | FP16 9.57 / 9.79 | off 10.40 -> -0.74 us; v1s k29
  4.90 / 4.85 | FP16 4.86 / 4.70 | off 4.86 -> ~-0.1. Trace: v16s received 9.46 -> 9.02, last stamp 10.72 -> 10.24;
  v1s received 4.58 -> 4.43, out 5.01 -> 4.86. Image: attn_8_1_1_1x6 loop region 374 -> 364 instrs, 0 spills.
- Coordinator: base moved to dev/k31.py (k29 + W9's FP16 softmax tile + WS's refined run1f prefetch).
- (4) WB_H16G run8w global path (v32s, sentinel L2 slots): halves at floats [8 tig, 8 tig + 8) (2 x 16-B stores; 5
  sectors per row, was 9), owner validates the 8 words (0xffffffff never arises from cvt.rn.f16x2 of finite values),
  folds float(O / l) l_q; the reset writes only the used words.
- (5) WB_H16E runE1 (d16s / d32s) and (6) WB_H16S runA1h (d1s): each warp's l covers only its 16 tokens (O is a full
  d-slice over all tokens), so O / l would need a cross-warp l; instead O x 2^-13 as FP16 (|O| <= 448 * 2^8 * 705
  tokens = 2^26.4 -> 2^13.4 < 65504; typical O ~ 2^8..2^14 -> 2^-5..2^1, FP16 normal), (l, m) FP32: 160-B rows
  (was 288). E1: 8-B st.async (was 16); A1h: 4-B st.async (was 8). Owners scale the fold back by 2^13.
- job62 (k30v on k31, all exchanges FP16 except v1f): check 0 failures; every sliding case's error within +-5e-5 of
  k31 (d1s 7.200e-02 vs 7.195e-02; worst rows +-1e-4). Image: 0 spills everywhere; regs8 loop regions x6 319 -> 317,
  w4 320 -> 322, e6 / e3 667 = 667, (whole-body regions) 1s 1120 -> 1164, 1h 1758 -> 1856.
  Raw medians (100 reps; k31 a / b | FP16 a / b | all off):
    d1s  2.98 / 3.01 | 2.94 / 3.01 | 3.07   -> ~0 (runA1h, 4-B messages: same count)   => WB_H16S OFF
    d16s 6.46 / 6.54 | 6.40 / 6.37 | 6.51   -> -0.11 (runE1 C6)
    d32s 8.38 / 8.35 | 8.35 / 8.32 | 8.38   -> -0.03 (runE1 C3)
    v1s  4.70 / 4.83 | 4.58 / 4.51 | 4.70   -> -0.22 (runA8h, 544-B bulk per owner)
    v16s 9.70 / 9.70 | 9.15 / 9.09 | 9.54   -> -0.58 (run8w C6, 2 x 16-B st.async per row)
    v32s 13.01 / 12.96 | 12.77 / 12.83 | 12.86 -> -0.18 (run8w global, 5 sectors per row)
- DELIVERABLE dev/wb/k30v.py (dev/wb/mk_k30v.py on dev/k31.py): WB_H16A8, WB_H16W, WB_H16G, WB_H16E on; WB_H16S and
  WB_H16 (v1f) off (their kernels are then SASS-identical to k31). Lesson: halving bytes pays where the exchange is
  DSMEM-bandwidth / message bound (run8w C6: bytes AND message count halved); the L2 combines are latency-bound.

## Round 22: verify-1 full START experiments (dev/wb/mk_k33v.py on dev/k32.py)
- Knobs (all V1F = run8 ALIGNED, no RoPE, no cluster): WB_FW (first wave: only FW of the DST-1 = 3 stages at the
  start, the rest issued once tile 0 lands), WB_SORD (CTA barrier after every thread's stage-0 issue), WB_MB0 (tile 0
  waited on a per-lane cp.async.mbarrier.arrive.noinc mbarrier instead of cp.async.wait + the lane barrier).
- Note on (2): a tile's wgmma reads K / V of all 64 tokens of both heads, which all 8 warps of the lane copy (8
  tokens each), so no per-warp mbarrier lets tile 0 start before the slowest warp's copies land; MB0 only removes the
  wait for the other warpgroup to reach the barrier. Splitting the tile compute by token half would be tile code.
- job63 (image attn_8_0_1_1; k32 loop 356 instrs): FW=1 kept the deferred page-table rows in registers -> 44/72 B
  spills, loop 607 instrs with 8 LDL -> 18.32 us; FW=2 36/52 B spills, loop 431 -> 16.42; SORD 0 spills, loop 356
  -> 16.06; MB0 loop 387 -> 16.10. Controls k32 16.19 / 15.97 (raw medians, 100 reps). Smokes 0 failures.
  Trace flush (k32 | SORD): tile0 med 4.88 | 4.06, max 5.94 | 5.71; tiles 10.13 | 9.25; combined 14.47 | 14.27;
  last stamp 15.17 | 14.98 -> SORD helps the trace's start but the end only -0.2 (untraced: neutral).
- Fix for FW: deferred rows stashed in their own (unused until then) ring slots, read back before a pre-loop lane
  barrier; the deepening happens before the tile loop (loop body unchanged): local NVRTC 0 spills.
- job64 (fixed FW; image attn_8_0_1_1 0 spills, loop 353 instrs (k32 356); smokes 0 failures). Raw medians (100 reps):
  k32 15.68 / 16.13 | FW=1 15.36 | FW=2 15.42 | SORD 15.87 | FW=2 + SORD 15.94 | MB0 16.00.
  Trace flush (k32 | FW=1 | FW=2): issued 2.77 | 2.29 | 2.62; tile0 med 4.62 | 4.13 | 4.62, max 6.46 | 4.56 | 5.81;
  tiles 10.14 | 9.48 | 9.82; combined 14.88 | 14.08 | 14.27; last stamp 15.63 | 14.70 | 15.01.
  -> FW=1 (a 4.2 MB first wave: tile 0 only) shortens the issue (0.5 us), lands tile 0 0.5 us earlier (max -1.9) and
  the whole tile phase follows. SORD and MB0: neutral.
- job65 FINAL A/B (FW=1 build; check 0 failures, every verify-full error identical to k32 to 4 digits; image loop 353,
  0 spills): raw medians (100 reps) k32 15.87 / 15.55 | FW=1 15.60 / 16.19 | FW=0 15.74; paired (40 reps) k32
  15.87 / 15.78 | FW=1 15.68 / 16.01. Over jobs 64 + 65 FW=1 averages 15.77 vs k32 15.81 -> NEUTRAL untraced, although
  the traces show -0.6..-0.9 at the last stamp (tile0 max 6.46 -> 4.56). Same traced-vs-untraced gap as before.
- DELIVERABLE dev/wb/k33v.py (dev/wb/mk_k33v.py on dev/k32.py): all three knobs OFF by default; the default build's
  attn_8_0_1_1 / attn_8_0_0_1 SASS is instruction-identical to k32's (local NVRTC cuobjdump diff). Verdict: the
  verify-1 full start (issue order, first-wave size, tile-0 wait primitive) is not the untraced bottleneck.

## Round 23: verify full FLAT (attn_8_0_*_0; v16f / v32f) - gate buffer out of smem, FLAT-only DST (dev/wb/mk_k34v.py on k32)
- WB_FGG: GSM = ROPE || (!ALIGNED && !WB_FGG): no 32-KB smem gate buffer; the combiner rows' gates are cp.async'd into
  the idle ring (+64 KB) right after the publish and waited before the combine epilogue; complete requests read them
  from global (epilogue_store_g). K8_DSTF / WB_DSTF: FLAT-only stage count (ALIGNED keeps K8_DST = 4); host
  smem_bytes drops the gate term under WB_FGG: DST 5 = 231,784 B (fits).
- job66 (check 0 failures, errors identical to k32; image attn_8_0_1_0 k32 16-B stack 4/8 B spills, loop 362 |
  FGG + DST 5 24-B stack 12/16, loop 363). Raw medians (100 reps; k32 a / b | FGG + DST 5 a / b | FGG, DST 4):
    v16f 99.54 / 99.57 | 104.19 / 104.46 | 99.87;   v32f 181.57 / 181.63 | 187.87 / 187.87 | 184.34
  paired (40 reps): v16f 99.83 / 99.78 | 104.30 / 104.61 | 99.94; v32f 182.53 / 182.55 | 189.36 / 189.22 | 185.44.
  -> step 1 alone: v16f neutral, v32f +2.8 (the gates now arrive from DRAM after the publish, on the combine path);
     step 2 (DST 5): +4.3 (v16f) / +3.5 (v32f) on top: MORE bytes in flight per lane makes FLAT full slower.
- job67: FLAT DST 3 (gates back in smem; image loop 362 = k32, smoke 0 failures). Raw medians (100 reps; k32 a / b |
  DST 3 a / b): v16f 98.74 / 98.38 | 101.41 / 101.70 (+3.0); v32f 179.71 / 179.78 | 188.82 / 189.04 (+9.2).
  -> DST 4 is the optimum for FLAT full on both sides (3: too few bytes in flight; 5: worse queueing).
- DELIVERABLE dev/wb/k34v.py (dev/wb/mk_k34v.py on dev/k32.py): WB_FGG off and K8_DSTF = K8_DST by default; the
  default build's attn_8_0_1_0 / attn_8_0_0_0 / attn_8_0_1_1 SASS is instruction-identical to k32 (local cuobjdump).
  Opt-ins kept: WB_FGG=1, WB_DSTF=n (DST 5 needs WB_FGG for smem).

## Round 24: verify-1 full end chain (dev/wb/mk_k35v.py on dev/k32.py)
- (1) WB_LP, lane publish: each lane stores its own partial into its own workspace slot (slot li; ALIGNED used only
  slot 0) in the FP16 sum-only format; no lane exchange / register merge; combiners add 2 P = 264 slots in one round
  (9 loads per thread); classic fallback walks the 2 P slots. job68: check 0 failures (errors = k32), smokes
  default / forced fallback / mixed 0 failures; image loop 356 = k32. Raw medians (100 reps): k32 16.06 / 16.10 |
  LP 16.10 / 16.22 | LP=0 16.00; paired: k32 16.28 / 16.25 | LP 16.39 / 16.34 -> NEUTRAL (off).
  Trace: tiles -> merged 0.88 (LP) vs 0.94: the ~0.9 us "merge" phase is NOT the exchange / ex2 merge - it is the
  partial store phase itself (4.3 MB of stores from every SM while the slowest CTAs still saturate DRAM), and
  merged -> publish (~0.95) is their drain under the release atomic.
- (2) WB_EAR, as-landed sum-only combine: per-CTA st.release flag (counters[hp][1600 + c]) before the arrival atomic;
  combiners (the last K arrivals) skip the all-arrived spin and add each partial as its flag lands (warp-uniform
  poll rounds, every landed partial's loads in flight together, 64-ns back-off); the last done combiner zeroes the
  flags. Expected: after the last flag only one partial's load + reduction + epilogue (vs a full 132-partial round).
- job69 (EAR; check 0 failures incl. call 1 after the flag reset, errors = k32; smokes default / forced fallback /
  mixed / EAR + LP 0 failures; image loop 355 vs 356). Raw medians (100 reps): k32 16.02 / 16.13 | EAR 18.02 / 17.98
  | EAR=0 15.97; paired: k32 16.19 / 16.10 | EAR 18.14 / 18.54 -> +2.0 us. Trace: combine start (7) 13.22 vs 13.17,
  loads done (9) 15.42 vs 14.02, combined 15.98 vs 14.57: each poll round is an acquire round trip (up to 5
  ld.acquire per thread from 65K threads) followed by a data round trip, and the polling traffic competes with the
  late CTAs' stores. Same failure mode as round 13's MALF. Off.
- DELIVERABLE dev/wb/k35v.py (dev/wb/mk_k35v.py on dev/k32.py): WB_LP and WB_EAR off; default attn_8_0_1_1 /
  attn_8_0_0_1 SASS instruction-identical to k32 (local cuobjdump). Kill rule: neither passes.
- What the end chain is: tiles max -> its partial stores (~0.9) -> their drain under the release (~0.9) -> the
  combine (~1.5: one L2 round of 132 partial rows, latency-bound). Bytes (round 21), the lane merge (LP) and the
  all-arrived wait (EAR) are not the cost; the store / drain phase under the tail of the DRAM stream is.

## Round 25: verify full FLAT cheaper decoupling (dev/wb/mk_k36v.py on dev/k33.py)
- WB_MTR (lane-level mid-tile refill): the loop top no longer waits / barriers / refills; in tile it's PV window (the
  W9F_PVW hook, after the PV wgmma issue, before its wait) each thread waits for its own copies of tile it+1, the
  lane barrier makes tile it+1 visible and proves both warpgroups are past tile it's stage reads, and tile it+DST
  is issued into tile it's slot (pt entries DST-1 ahead as before). Tile 0's wait + tile DST-1's issue before the
  loop; V^T double-buffered by tile parity (no barrier between a WG's PV(it) wait and its tile it+1 V^T writes).
  Full 128-B row copies kept (W9 round 14: half-row layouts are line-transaction bound).
- job70: check 0 failures (errors = k33). Image attn_8_0_1_0: k33 16-B stack 4/8 B spills, loop 362 | MTR 48-B
  stack, 56/88 B spills, loop 408 with 3 LDL (the issue state is live across the PV window next to the PV
  accumulators). Raw medians (100 reps; k33 a / b | MTR a / b | MTR=0): v16f 100.32 / 100.11 | 112.86 / 113.41 |
  99.68; v32f 181.87 / 182.13 | 203.50 / 204.30 | 181.94. Paired: v16f 100.15 / 99.99 | 113.55 / 113.36; v32f
  181.82 / 182.05 | 204.29 / 204.58 -> +13 / +22 us. No-compute skeleton (ABL_NOCOMPUTE, 40 reps): v16f 93.14 /
  93.20 | 94.67 / 95.25; v32f 173.14 / 173.22 | 174.66 / 174.72 -> the skeleton gets 1.5 us WORSE too (no
  coupling relief from moving the refill), and with compute the spills dominate. Killed (rule: < 1.5 us on v32f).
- (2) cheaper issue path, SASS (local NVRTC, attn_8_0_1_0, WB_MTR=0): per tile per thread ~25 instructions of slot /
  predicate / row setup before the lane barrier (the row LDS.128 already sits under it: W9F_RL), ~25 after it
  (4 IMAD.WIDE row x rowbytes + base, 4 xor'd destinations, 4 LDGSTS with src-size predicates), and a ~30-instruction
  page-table cp4 path executed by 8 lanes per warp (divergent). The branch-free / predicated / no-select variants
  are W9's F2 / F3 / F5 (F2 on; F3 +3 us; F5 neutral); the page-table rows already land DST-1 tiles ahead and are
  read under the barrier. What is left (precomputed per-slot destinations: dst_j(s) = D_j + s STG since s STG does
  not touch bits 4-5; incremental pt pointers) needs 4+ more live registers in a loop that already spills 4/8 B
  at the 128-register cap - MTR just showed what added live state costs. Not built.
- DELIVERABLE dev/wb/k36v.py: WB_MTR off; default attn_8_0_1_0 / attn_8_0_0_0 SASS instruction-identical to k33.

## Round 26: publish microbench, end to end incl. the release (dev/wb/redbench2.py, job71)
132 CTAs x 512 threads at once; graph replay after a 512-MB L2 flush minus an empty graph (4.83 us), median of 30.
(a) STG.128 from registers + __syncthreads + atom.add.release; (b) red.global.add.v4.f32 into copy c % NC + release;
(c) st.shared staging + fence.proxy.async.shared::cta + barrier + S threads' cp.reduce.async.bulk.global.shared::cta
.bulk_group.add.f32 + commit + wait_group 0 + fence.proxy.async.global + barrier + release; (d) as (c) with a plain
cp.async.bulk store into the CTA's own slot. (c) verified: acc = sum of the 132 CTAs' values.
   size     (a) STG | (b) red 1 / 2 / 4 copies | (c) bulk red 1 / 2 / 4 copies (1 issuer) | (d) bulk st 1 / 4 / 16 issuers
   4 KB     1.57    | 2.46 / 1.57 / 1.34       | 2.62 / 1.89 / 1.66 (4 issuers, 4 copies: 1.60) | 1.15 / 1.15 / 1.66
   33 KB    2.02    | 4.26 / 3.62 / 2.94       | 4.96 / 4.22 / 3.30 (16 issuers, 4 copies: 3.23) | 2.40 / 2.59 / 2.66
- Decision rules: (c) for verify-1 full needs <= ~1.6 us at 33 KB: best 3.23 -> NO (the bulk reduce is slower than
  red.v4 at every copy count: the L2 reduction rate, not the issue, is the limit). (d) at 33 KB loses to STG by 0.38
  -> no port for verify-1 full. At 4 KB (decode-sized partials, 16 rows) the single-thread bulk store beats STG by
  0.42 us -> candidate for the decode partial publishes (run1 FLAT d16f / d32f; run1w's slot path), not built.
- Coordinator GO: port (d) to run1f FLAT's partial publish (decode full b16 / b32). Checked first: run1f's DEFAULT
  publish is W9D's fixed-reference accumulation (w9d = B <= 64 && hp < 4: every in-range CTA red.add.v4's its
  4-KB partial into the request's accumulator; STG slots only for out-of-range CTAs, the rare classic fallback). So
  the bulk equivalent of the default path is (c) bulk reduce, not (d).
- job72 (rerun + the decode-full regime: 4 KB, one accumulator per request -> 8 / 16 / 32 copies):
    4 KB  (a) STG 1.28 | (d) bulk store 1 issuer 1.31  -> job71's -0.42 for (d) did NOT replicate (now +0.03)
    4 KB  (b) red.v4 8 / 16 / 32 copies 1.25 / 1.15 / 0.96 | (c) bulk reduce 1.47 / 1.25 / 1.25 -> (c) +0.1..+0.3
    33 KB as before: (a) 2.02, (d) 2.27, (b) 4 copies 3.04, (c) 4 copies 3.26..3.30.
  -> no passing case: run1f's red.add publish beats its bulk-reduce equivalent in its own regime, and the 4-KB
  bulk store is a tie with STG. Nothing ported (k37v not built). Bulk async copies lose or tie at these sizes.

## Round 27: run1f (decode full FLAT, attn_1_0_1_0) page-table smem ring (dev/wb/mk_k38v.py on dev/k33.py)
- WP's ncu: the __shfl_sync of the directly __ldg'd page-table row feeding the next refill's cp.async (k33 L8988)
  = 26-28 % of stall samples. WB_PTR (page size 1): lanes 0..15 of each warp cp.async their 16 entries of tile j into
  ring slot j % 4 ([NL][4][64] ints, 4 KB after the unused RoPE area; host smem 226,888 -> 231,000 B), riding in
  tile j-3's commit group (issued at the loop top of tile j-6); read by one LDS after tile j-3's lane barrier.
- job73: check 0 failures (decode-full errors unchanged); image 128 regs, 0 spills, loop 588 -> 641 instrs.
  ncu d32f (k33 -> PTR): L8988 32.8 % -> 4.3 % of samples; long_scoreboard 34.9 % -> 10.3 %; BUT total samples
  15,631 -> 17,221 and the stalls move to mio_throttle (1.1 -> 11.9 %), not_selected (6.6 -> 14.6 %),
  short_scoreboard (4.5 -> 12.0 %) on the cp4 / LDS / shuffle / cp.async lines.
  Raw medians (100 reps; k33 a / b | PTR a / b | PTR=0): d16f 86.50 / 86.29 | 87.47 / 87.81 | 86.30;
  d32f 164.77 / 165.10 | 166.06 / 166.08 | 164.85. Paired (40): d16f 86.92 / 86.44 | 88.00 / 88.10;
  d32f 164.33 / 164.65 | 166.66 / 166.13 -> +1.1..+2.0 us. Killed (rule: < 1 us gain on d32f).
- Reading: the row wait is where warps park while DRAM runs at 80-85 % of peak, not the limiter. Removing it moves the
  stall to the MIO queue (the ring adds a cp4 + LDS per warp-tile on top of the shuffles and 8 cp.async), the same
  conclusion as WS's mode 4 (in-loop L2 prefetch of rows: +0.1..+0.6).
- DELIVERABLE dev/wb/k38v.py: WB_PTR off; default attn_1_0_1_0 / attn_1_0_0_0 SASS instruction-identical to k33.

## Round 28: are the verify sliding tails code-bound? (measurement first; dev/wb/trace3.py = trace2 + all 32 stamps)
- job74 (k33, traced builds, 10 reps, median CTA; flush = cold L2 incl. code, warm = replay once then trace with L2
  warm: code in L2, only the per-SM instruction caches cold). Tail = last warpgroup's tile end -> the final stamp:
    v1s  (runA8h, attn_8_1_1_1h): tile end 2.96 | 2.59 -> fold 3.29 | 2.85 -> pushed 3.86 | 3.39 -> received 4.44 | 3.98
         -> out 4.86 | 4.39: tail 1.90 flush vs 1.80 warm (+0.10; per phase +0.07 / +0.03 / -0.01 / +0.01).
    v16s (run8w C6, attn_8_1_1_1x6): tile end 6.50 | 5.73 -> merged 6.97 | 6.10 -> pushed 7.71 | 6.82 -> received
         8.70 | 7.76 -> out 9.23 | 8.28: tail 2.73 vs 2.55 (+0.18; +0.10 in the first post-loop phase, the rest <= 0.05).
    v32s (run8w global, attn_8_1_1_1w4): tile end 8.47 | 8.19 -> merged 8.89 | 8.54 -> owner folded 11.90 | 11.33 ->
         out 12.32 | 11.76: tail 3.85 vs 3.57 (+0.28, of which +0.22 is the owner's wait for the slowest part, whose
         tile end is itself 0.5-0.8 us later under flush: data, not code). Folded -> out 0.42 vs 0.43.
- Verdict: none of the three tails is instruction-fetch bound. The cold-minus-warm cost of ALL tail code is 0.10 /
  0.18 / <= 0.06 (code part) us, an upper bound for any compaction (rolled loops, one `fin` for runA8h's two
  instantiations: its fold + stage + push phases differ by 0.10 us in total between cold and warm code). Kill rule
  < 0.2 us per shape: nothing built (no k39v). v32s's ~3 us tail is the owner waiting for the slowest of its 4 parts
  (tile-end spread med -> max ~1.4 us) plus that part's publish, not code.
