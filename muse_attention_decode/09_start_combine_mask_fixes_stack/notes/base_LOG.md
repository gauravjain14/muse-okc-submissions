# base worker log (d32s / v16s / v32s / d16f)

Tools (sandbox `/work/base/t/`, local `work/base/t/`):
- `mkslim.py IN OUT`: base section only (cuts at the XE section) -> fast compile, all 4 of my shapes still route to base.
- `ab.py --shapes .. --seeds .. --reps .. name=path[:K2_DEFS] ...`: interleaved A/B (variants take turns per rep, flush
  before each replay, sync per rep). NOTE: its absolute numbers are ~0.7-0.9 us higher than bench.py on sliding shapes
  (bench enqueues all reps without syncing); they match the judge's raw numbers better (judge raw = yours + 3.58:
  d32s 14.21, v16s 13.33, v32s 16.73). Same-code A/A noise: up to ~0.2 us on v16s with 3 seeds x 30 reps.
- `tr.py CAND SHAPE`: per-CTA %globaltimer trace (K2_TRACE build). Tracing adds ~1 us; per-slot timings are
  indicative only.
- `mb_slide.py SHAPE`: sliding gather floor (page-table hop + all 2048 K/V rows per request, cp.async, no math).

## Measurements (k73)
- Gather floor (bench-style event time): B32 (d32s/v32s bytes) ~11.3 us (rows all landed ~6.4 us after the first CTA
  starts), B16 ~9.2 us (landed ~4.3 us). 48-64 SMs already saturate it; 8 issuing warps > 16; half-row (64 B)
  gathers are 0.5-1 us slower; TMA bulk 128-B copies are much slower (TMA issue rate); ld.global to registers is
  terrible. TLB touch / L2 prefetch of the 2 MB pages: no gain in the microbenchmark (W7_E3=0 on d32s: +-0).
- Ablations (ab.py, results invalid, timing only):
  - d32s 14.12: no cross-CTA combine -0.94, no tile math -1.04
  - v16s 13.70: no cross-CTA combine -1.31, no tile math -1.55
  - v32s 17.20: no cross-CTA (global) combine -2.86, no tile math -1.26
- Trace (v32s, WG0): each 64-token tile costs ~0.8 us of warpgroup time (4 tiles per WG -> compute-bound from the
  first tile at ~4.5 us to ~8.5 us); v16s same ~0.8 us per tile (3 tiles per WG). The data has all landed by ~6.5 us.
  Tile loop = 192 SASS instructions per warp (4 warps per SMSP -> ~770 issue slots) + 544 clk of QGMMA per SMSP per
  tile-step; observed ~1600 clk -> the 4 warpgroups run in lockstep (tensor phase / ALU phase not overlapped).
- Cluster occupancy (1 CTA/SM): max clusters of 2: 66, 3: 39, 4: 30, 5: 22, 6: 17, 7: 15, 8: 15, 16: 7.
  (2 CTAs/SM allowed: 4 -> 62, 8 -> 30, but 128 CTAs land on only 120 distinct SMs.)

## Tried, no gain
- L2 prefetch of the epilogue gate rows at entry (run8w CL + global): v16s +0.09, v32s +0.35 (worse).
- run8p (cluster per (request, KV head), half rows): WH_PH32=2 on v32s +3.1 us, WH_PH16=2 on v16s +4.8 us.
- Lazy O rescale in w8_tile (PV accumulating in place, warpgroup vote via bar.red.or): the vote almost always says
  rescale (64 rows per WG) -> v16s +0.25, v32s +0.36. First version (two accumulator paths) made ptxas serialize every
  QGMMA (+0.4 / +1.2 us).

## Progress log (appended 2026-10-30 late)
Scope note: d16f/d32f dropped (coordinator). Focus v16s / d32s / v32s.

### More measurements (k73)
- run8w per-warpgroup tile loop (clock64, v32s WG0): ~1265 clk per 64-token tile, waits for data only ~100 clk per tile
  once started -> the q_len-8 kernels are compute-bound from the first landed tile (~4.5 us traced) to the end.
- ncu warp sampling (v32s tile loop, 3.1K samples): mbarrier try-wait 17 %, LDSM->PRMT latency (V^T transpose) 11 %,
  fence.proxy.async (MEMBAR.ALL.CTA) 7 %, PV DEPBAR 7.5 %, WG barrier 4 %, S2R(tid) recompute chains ~4 %.
- Tile-part ablations (ab.py, invalid results): no softmax arithmetic +0.1 (noise), no V^T transpose -0.34/-0.15
  (v16s/v32s), no PV wgmma -0.10/-0.32, no tile compute at all -1.47/-1.27 -> no single dominant part (latency chain).
- In the RS (role-split) v32s path the issuer warpgroups (WG0/1) cannot start computing before every copy is issued
  (issue back-pressured until ~4.2 us); WG2/3 also start late.

### Tried, no gain (cont.)
- w8_tile: both V^T ldmatrix before the QK wgmma (BZ_VT) +0.2 / +0.1; fence.proxy.async moved right before the PV
  barrier (BZ_FL) -0.11 / -0.07 (noise level, not kept).
- v32s on run8w's cluster variant with P = 3 (cluster of 3, DSMEM combine instead of the global slot combine; needed a
  compact smem layout: new block in part 0's spare tile buffer, swap/stage areas aliasing the tiles after a CTA
  barrier, 8-dim owner items): correct (16/16) but v32s 17.73 vs 16.27 (bench): 11 tiles per CTA of the
  compute-bound tile loop cost more than the global combine saved.
- WH_WS=1 (run8w warp-specialized producer) has bit-rotted (does not compile with the RS path); not pursued.
- New kernel runV8 (verify sliding B > 1, warp-specialized cp.async producer + FA3-ordered consumers, cluster DSMEM
  pull combine, gates prefetched to smem):
  - v1: 2 consumer WGs (one per KV head, 128-token iterations, setmaxnreg 208/40): correct after fixing the PV-column
    -> head-dim permutation of the V^T layout (col 8i+2q+e = dim 16(i>>1)+4q+2e+(i&1)); v32s 18.15, v16s 14.72 (bench)
    -> 1.0 us per 128-token iteration: only 2 consumer warps per SMSP, latency-bound softmax (~770 clk) and V^T
    (~470 clk).
  - ptxas serialized every QGMMA until the loop followed FA3's order exactly (P written only after the PV that reads
    it retired; no conditional wgmma; iteration 0 peeled) - double-buffering P/S alone did not help.
  - v2: 4 consumer WGs (head x stream, 64-token tiles) + 1 producer WG (640 threads; setmaxnreg must fit the CTA's
    launch pool 640 x 96: consumers 112 / producer 32, else setmaxnreg.inc deadlocks): correct 16/16 but v32s 19.26,
    v16s 14.83 (bench) - slower still. Not kept (default path unchanged).
- runV8 v2 trace: producer (4 warps, 32 regs after setmaxnreg) finishes issuing only at ~9.3 us for C = 3 (vs 3.9 us
  with 8 producer warps in v1); per 64-token tile ~1630 clk of warpgroup time (softmax ~620 clk, V^T ~410) -> slower
  than run8w's 1265. Conclusion: these tile loops are ALU-issue bound (~200 SASS instr per warp per 64-token tile,
  4 warps per SMSP -> ~800 issue slots vs 544 clk of QGMMA per tile-step); FA3-style reordering cannot help an
  issue-bound loop, and fewer consumer warps make it latency-bound. DSMEM pull of the owner rows also costs
  ~0.9-1.5 us in every cluster variant measured.
- runE1 (d32s) knob sweep (ab, 4 seeds): base 13.83; WE_NP=1 +1.12, WE_NP=0 +0.27, WE_PAIR=0 +1.45, WE_EXACT=0 +0.55,
  WS_E1U=1 +0.12, WE_CI2=1 +1.33, WE_NS=1 +1.03 -> the shipped settings are the best.
- run8w cluster combine variants for v16s (ab): XDV_PULL=0 (st.async push) +0.53, XDV_PULL=0 WH_BULK=1 +1.43;
  8 owner items per row with 16-B DSMEM loads and one shuffled (m, l) per row (BZ_PL8) +0.19 vs without -> the pull
  combine is not DSMEM-request bound; reverted.
- BZ_CMP=3 (compact smem layout for the C6 cluster kernel too, CTA barrier before the swap): v16s +0.15.

### Kept (work/base/cand.py = runs/k73.py + BZ_FL)
- w8_tile (run8w tile, used by v16s attn_8_1_1_1x6 and v32s attn_8_1_1_1w4): fence.proxy.async of the V^T stores moved
  from right after the stores (where its MEMBAR.ALL.CTA stalled before the QK wait) to right before the PV barrier.
  Same ordering guarantee (stores -> fence -> barrier -> wgmma). 16/16 tests.
  ab.py, same session, 6 seeds x 30 reps, full files: v16s k73 13.40 -> 13.32 (-0.07), v32s 17.06 -> 16.99 (-0.08),
  d32s 13.95 -> 13.94 (untouched kernel); earlier 5-seed runs: v16s -0.10/-0.11, v32s -0.02/-0.07. Small, at noise level.
- Experimental code (all default-off, not in cand.py): work/base/cand_exp.py (runV8 kernel attn_v8_<C>, run8w P = 3
  cluster variant with the compact layout, K2_DEFS BZ_V8=1 / BZ_X3=1), work/base/t/cand_pl8.py (BZ_PL8).
