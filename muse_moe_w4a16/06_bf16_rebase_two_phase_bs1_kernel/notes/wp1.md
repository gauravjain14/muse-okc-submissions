# WP-1 findings: bs1 one-token kernel (moe1_4)

Files: dev/moe_v400.py (v200 + switches below), dev/wp1_trace.py (per-CTA timeline, MOE_TRACE1 + TR2 stamps),
dev/wp1_check.py (60-call bs1 correctness incl. graph replays), dev/wp1_cc.py (local nvcc compile of moe1_4),
dev/wp1_run*.sh (Modal jobs).

## 1. Baseline timeline (v400 == v200 kernel, Modal H200, judge-style flush, medians of 7 calls)

ne=3 (seed 73235), us from the first CTA start:
- start 0.03 (max 0.10), x ready 0.45, producer first weight copy issued 0.64 (waits for topk_ids ~0.6 us)
- stage 0 ready 2.54 med / 3.30 max (1.9 us after issue: 4 stages x 128 CTAs = 18.9 MB requested at once)
- stage intervals (per unit of 6 stages: 4 gate/up, 2 down): ~1.0 / 1.0 / 1.0 / 1.23 / 1.25 / 1.44-1.7
  -> gate/up stages stream at ~4.7 TB/s, the down stages and the unit transition are slower: the 18 stages
  take 21.6 us (3.9 TB/s); at 1.0 us/stage they would take 18.
- last stage ready 22.2, compute done 23.4, reds issued 23.6, arrival median 24.3 / max 25.2
- last CTA: compute done 24.3, pre-atom 24.6, atom 25.2 (0.64 us: atom.acq_rel waits for its 16 KB of reds),
  combine loads 26.3 (1.0), reduce+zeroing 27.0 (0.77), end 27.5 (0.42) -> 2.24 us combine after the last arrival.
- ne=2: end 19.9-20.1; ne=4: 34.3-34.4; ne=5: 42.3. Same stage pattern for every ne.
- MOE_NO_RED (no down reds): same stage pattern and same end within noise -> the red traffic is not the limiter.

Control A/B (run1, 100 judge-matched reps): v200 27.7 / v200b 27.9 / v400 28.1 / v400b 27.8 us (x2.850 / 2.840 /
2.825 / 2.836): control-to-control spread ~0.5%.

## 2. Stage compute vs wait (run2, TR2 stamps: per stage, median over CTAs; ne=3 seed 73235)

- compute (warp 0, ready -> computed): gate/up 0.90-0.96 us, down h0 1.06, down h1 1.12 us
- wait before stage (computed -> next ready): gate/up 0.13, before down0 0.35 (k-warp reduction + SwiGLU, two CTA
  barriers), before down1 0.19 (yl add), before the next unit 0.32 (0.6-0.8 at an expert change: the flush)
- per unit: compute 5.9 us + gaps 1.25 us = 7.1 us vs ~6.3 us of data at 1.05 us/stage -> consumer-bound, matching
  the planner's coupling numbers (P1/P4).
- MOE_F_NOCOMP (no stage math): 18 stages land by 18.8 us (vs 22.5 with compute); stage 0 still ready at 2.45.
- Startup is data-bound: ids at ~0.6 us gate the first copy (0.64), stage 0 needs 1.9 us more (128 CTAs x 4 stages
  = 18.9 MB queued at once, served ~FIFO). Not improvable without a smaller first stage (= a layout change).

## 3. Candidates (bench2, judge-matched seeds, 100 reps, 2 copies, same job)

| switch | what | result |
|---|---|---|
| MOE_W1YDB v1 | two yacc buffers by call parity, idle one zeroed by all CTAs while streaming | combine reduce 0.77 -> 0.13 us, but -3.7% overall: parity load behind the weight stream stalled stage 0 (~2.3 us). Fixed (v2: load with the routing ids) |
| MOE_W1AF | arrival before the last expert's adds; last CTA keeps its sums in smem | -11..-12%: rejected |
| MOE_W1DN | (stage math, info for WP-1C) w2 k-permutation, one scale pair per row-group pair | alone noise (-0.3 / +1.6%) |
| MOE_W1SW1 | (stage math, info for WP-1C) one-barrier SwiGLU via shuffles | alone noise (+0.2 / -0.1%) |
| DN + SW1 | | +1.5 / +1.4% (placebo W1PAD -0.8%, control -0.5%); not stacked, WP-1C scope |

Trace-build caution: some TRACE1 builds (YDB, DN+SW1) show a 2.3-3.3 us stage-0 "compute" that the bench builds
do not always have: the per-unit `gs13[e]` load is issued behind the weight stream and some schedules consume it
early -> MOE_W1GSE (one warp-wide load of the 32 local global scales at the start, shuffled per unit).

## 4. Scope note (coordinator, 2026-10-27): WP-1C owns moe1's stage math and the weight repack

Functions/regions of dev/moe_v400.py my switches touch (all inside `moe1_body` / host M == 1 path):
- kernel start (before the producer/consumer split): MOE_W1YDB parity load, MOE_W1GSE gate/up global-scale load,
  MOE_W1CL extra mbarrier inits + `barrier.cluster.arrive`.
- producer: MOE_W1CL `barrier.cluster.wait` before its return (nothing else).
- consumer unit loop: the per-unit `gs1` (GSE), the yl add after each down stage (MOE_W1EFL: reds of each half at
  the expert's last unit), the end-of-expert flush (EFL disables it), YDB zeroing slice after the first unit.
- end/arrival + combine: MOE_W1AF (rejected), MOE_W1CL cluster flag exchange + distributed combine, YDB (no
  zeroing in the combine, parity flip in the reset).
- host: `_Module.launch_cl`, `_w1cl()`, the cluster-occupancy check in `prepare()`, the M == 1 launch in `kernel_fn`.
- stage math / repack (WP-1C's area, kept only as info switches): MOE_W1DN (`_repack_w2_slices_dn`, the down
  stage's scale loads), MOE_W1SW1 (SwiGLU section), MOE_W1PAD (placebo copy of w2f/s2f).

## 5. Runs 4-6: combine variants, cold start

| stack (bench2 jm 100 reps, ratio vs v200, 2 copies) | run4 | run5 |
|---|---|---|
| control v200b | 1.0006 | 0.9981 |
| GSE | 1.0015 | |
| GSE+YDB(v2) | 0.9652 / 0.9707 | |
| +EFL | 0.9829 / 0.9762 | |
| +CL=2 | 0.9839 / 0.9819 | 0.9828 / 0.9870 |
| +PRD | | 0.9621 / 0.9607 |

- 4-CTA clusters: cuOccupancyMaxActiveClusters = 30 on the H200 (GPC sizes) -> 120 CTAs max; 2-CTA: 66.
- DSMEM handshakes at the end cost 0.5-0.8 us each (CL2 flag exchange, PRD pair handshake ~1 us); the CL2 split
  combine loads 24 KB per CTA in 0.76 us (latency-bound under the red burst), so CL2 gains nothing over one CTA.
- run6 (dev/wp1_time.py, judge protocol vs "warm code": flush, then a decoy call on disjoint local experts, then
  the timed call; 50 seeds x 2): v200 27.90 -> 26.75 us: **cold code/constants cost ~1.1 us per call**. YDB / EFL /
  CL2 lose 2-3% in BOTH modes -> their loss is not cold code. Suspect: the parity word was read with a strong
  (L1-bypassing) load by all 288 threads of all CTAs (~1150 requests on one L2 line at the start, a hot spot in
  front of the first stages' data). YDB v3: one weak .cg load by thread 0, published via smem (run7).
