# WP-L2: bs128 (moe_8_4_1_4_4) and bs256 (moe_8_8_1_3_8), 2026-10-31 top-3 push

Worker WP-L2. Base dev/moe_v1002.py. Namespace v1400-v1499. Judge (v969 scored): bs128 164.7 us (2.773x), bs256 202.7 us (3.081x).

Files:
- dev/moe_v1400.py = dev/moe_v1010.py (v1002 + WP-RT's MOE_RT_TR stamps, off by default) + WP-L2 switches (all off by
  default; default build = v1002 SASS in all 16 kernels, checked with local NVRTC 13.0). MOE_L2_B881 (Python only):
  bs65-128 on the (8, 8, 1) 64-token config, as the planner's dev/moe_v1033.py.
- dev/wpl2_probe.py: copy of dev/wprt_probe2.py + c8 (consumers done with the queue), tail statistics (per-CTA queue-done
  p10 / p50 / p90 / max, kernel end, end phase), item classes and gaps; always dumps (WPRT_DUMP).
- dev/wpl2_ifetch.py: cold instruction-fetch microbenchmark (planner's staggered-entry idea).
- dev/wpl2_runN.sh; logs kda_ws/runs/wpl2_runN.log.

## Log
### run 1 (v1002 timeline, judge protocol: graph replay after a 512 MB flush, 6 jm seeds; trace build MOE_WP32_TR,MOE_TRN,MOE_RT_TR)
Means over seeds of per-CTA medians (us after the first CTA entry):

| | bs128 (8,4,1) MW4 | bs128 B881 (8,8,1) MW4 | bs256 (8,8,1) MW8 |
|---|---|---|---|
| tables (c11) | 2.25 | 2.29 | 2.55 |
| producer loop top / first claim / first slot | 2.81 / 3.45 / 5.19 | 3.46 / 4.45 / 5.57 | 3.66 / 4.70 / 5.70 |
| first stage-0 weights issued (r12) | 5.40 | 5.94 | 6.15 |
| gather done / GCP release returned | c13 4.42 | r24 7.36 | r24 9.00 (GCP conversion 1.0-1.15 us per row) |
| all-CTA gather barrier seen (p26) | wg 6.58 | 8.27 | 10.56 |
| warm-up end (c16) | (no warm-up) | 7.30 | 10.63 |
| first data (data0) | 7.30 (wg 7.36, ms 6.76) | 9.17 (wg 10.22, ms 8.93) | 11.59 |
| first item per-stage vs later same-class items | 1340 vs 1104 ns (+3.8 us / CTA) | 1150 vs 1055 | 1161 vs 1152 |
| CTA queue done c8 p10 / p50 / p90 / max | 159.6 / 161.3 / 163.0 / 163.8 | 155.0 / 157.0 / 160.8 / 162.1 | 193.7 / 196.0 / 198.6 / 200.8 |
| mean tail idle (max c8 - c8) | 2.5 | 4.7 | 4.8 |
| end phase (max c9 - max c8) | 3.0 | 2.5 | 2.5 |
| item gaps per CTA (end -> next item's first data) | ~4.5 (G1->G1 0.49 us) | ~3.2 | ~3.4 (0.30-0.33 us each) |
| kernel end (trace build) | 166.8 | 164.6 | 203.3 |

Item durations bs256 (us, 16 stages): G1 ntl1 16.3, ntl2 19.4, ntl3 22.7, ntl8 28.5 (1.78 us / stage: 64-token stages are
tensor-bound, 2048 tensor cycles per SMSP per stage at peak = 1.2 us); G2 ntl1 15.9, split pieces (4 stages) 4.39.
No flush (4 seeds): bs256 first data 11.13 (-0.46 us; warm-up 7.0 vs 10.6 but the gather barrier gates), bs128 6.46
(-0.84); kernel end NOT faster at bs256 (matched seeds +1.3..+4.4 us slower) or bs128 (+-1.5 us).
bench2 jm 80 reps (v1002 x2, v1033 x2): bs128 v1002b 0.9966, v1033 1.0042 / 1.0071; bs256 0.9964 / 0.9963 / 1.0014
(placebo); bs32 0.9985 / 0.9974 / 1.0010 (placebo). v1033 confirmed (+0.6% vs the control mean, as the planner's b881).
The cold i-fetch microbenchmark printed nothing (NVRTC log with a NUL byte made grep treat the output as binary): rerun.

Tail mechanism (new): the producer claims item k+1 at the TOP of item k (atom at the item start, ~NS stages before the
consumers finish item k-1), so a CTA running a 16-29 us item holds the next claimed item for the whole item. At the end of
the queue the last pieces sit behind full items on a few CTAs while the others go idle: e.g. bs256 seed 70058 the last CTAs
ran G2n1[170.8-186.8] then their pre-claimed piece [187.1-191.6] while the median CTA was done at ~185.8.
dev/wpl2_claimsim.py replays the traced items (queue order, measured durations) with claim lead (nst + NS) stages vs
(min(nst, D) + NS): bs256 max end -2.0 / -1.1 / +1.5 / -6.2 / -0.2 / -4.1 us over the 6 seeds (mean -2.0 us = ~1%), idle
4.7 -> ~2.7 us; bs128 ~-0.3 us (its 576 pieces already cover the tail).

### run 2 (wpl2_run2.sh)
- Cold i-fetch microbenchmark (dev/wpl2_ifetch.py; 132 CTAs x 8 warps, straight-line LOP3 bodies, 512 MB flush before the
  launch): cold first pass 1.38 / 2.43 / 4.51 us for 64 / 128 / 256 lines vs warm 1.15 / 2.18 / 4.29 us, i.e. a constant
  ~0.23 us cold penalty whatever the size, flush or not. Staggered entry (16 sections, SM groups enter at different
  sections) only shortens pass 0 by running less code (max over CTAs unchanged). => straight-line code without load is NOT
  fetched line by line at HBM latency; the planner's per-line mechanism is not reproduced in isolation (run 3 repeats it
  under an HBM stream and with cold constant-bank lines, the two things the real warm-up has and the microbenchmark lacked).
- First stages (MOE_WPN_TRACE, dev/wpedge_st0.py, 3 seeds x 2 reps, cycles): bs256 warm-up stage 6791 cyc with the flush,
  3547 without, steady wgmma stages ~2050 => ~4700 cycles (~2.5 us) of cold cost inside the warm-up stage, half of it
  flush-dependent; bs128 (8,4,1) first stage 1611 vs 1440 later (no warm-up there; the first item runs at the HBM rate:
  gaps 240-600 cyc / stage). Per first-item ntl (run 1 dumps): warm-ups of 40-64-token tiles end at 10.6-12.5 us vs
  6.6-7.8 us for <= 32-token tiles; B881 bs128: CTAs whose first tile has 40-64 tokens see first data at 11.5-12.9 us
  vs ~9.0 for the others (barrier 8.3): their warm-up gates them by 3-4 us.
- MOE_L2_CLD (late claim; bench2 jm 100 reps, v1002 x2 controls 1.0000 / 0.9964 bs256, 0.9995 bs128, 0.9965 bs32):
  bs256 CLD2 0.9946, CLD3 0.9958; bs128 CLD2 0.9977, CLD3 0.9983; CLD2+B881 bs128 1.0044 vs v1033 1.0026; bs32 placebo
  0.9957 / 0.9957. Not a gain. Trace (CLD2, bs256): tail spread max-p50 4.8 -> 2.9 us but the end phase grows 2.5 -> 3.2:
  CTAs that finish early run the token combine (c9 - c8 median ~4.2 us at bs256) during what looked like idle tail, so a
  compressed tail moves combine work behind the last item. SASS: ptxas warp-aggregates the claim (SHFL of the result right
  after the ATOMG), so the late claim stalls the producer mid-item for the round trip; MOE_L2_CLDI (atom.inc) removes that
  (run 3). check_mt 64/65/128/129/256/257 0 failures (CLD2, CLD2+B881); hang1 ok.

### run 3 (wpl2_run3.sh)
- Planner's staggered-entry (Duff) idea for the cold warm-up: REJECTED by microbenchmark (dev/wpl2_ifetch.py, 132 CTAs,
  10 reps, judge flush). Straight-line code: cold pass 1.34 / 2.40 us for 64 / 128 lines vs warm 1.15 / 2.18 (+0.2 us
  whatever the size); under a concurrent HBM stream (4 extra warps per SM streaming 1 GB) 1.44 / 2.62 vs warm 1.28 / 2.46
  (still +0.16 us); staggered entry (8 SM groups at 8 sections) changes only how much code pass 0 runs (max over CTAs
  unchanged). The instruction prefetcher streams straight-line code; there is no per-line HBM wait to spread over SMs.
  What IS slow cold: constant-bank lines, ~130 ns each, serialized (16 cold lines: pass 3.30 us vs 1.18 warm, 2.66 without
  the flush; stagger 2.90) - but the bs128 / bs256 kernels reference only ~8 constant lines (bank 0 params, bank 3, one
  jump table), all at entry. The real warm-up's extra ~4 us (bs256: 6791 vs ~2050 cycles warm per stage; two warm-up
  stages (MOE_RT_W2): the second ends 2.5 us after the first at bs256, 2.2 us for B881 bs128 wgmma CTAs) is therefore in
  branchy cold code (taken branches to cold targets are not prefetched) and the first wgmma/tensor use, not in
  straight-line fetch; it is hidden behind the gather barrier at bs256 (warm-up end 10.6 vs barrier 10.56) and costs B881
  bs128's 40-64-token-first CTAs ~2-3 us.
- MOE_L2_XD8 (XDIR in the GCP kernels): WRONG (check_mt M 257: 6 failures, rel-L2 up to 0.26) - the direct rows land in the
  x regions of ring slots 1..NS-1, which the GCP warps use as their gather scratch until their release; and slower
  (bs256 0.9941, B881 bs128 0.9912 vs controls 0.9963 / 1.0029). Rejected (fixing the race means waiting for the GCP
  release, which removes the point).
- MOE_L2_CLD=3 + MOE_L2_CLDI (late claim, atom.inc): bench2 jm 100 reps, controls v1002 1.0000 / v1002b 0.9963 (bs256),
  0.9999 (bs128); identical-kernel placebos b881-at-bs256 0.9950 -> non-first identical copies 0.995-1.0005 over runs 2-3
  (noise +-0.3%): ci3 bs256 0.9995, ci3b (same bs256 kernel) 1.0041 (mean +0.5% over the non-first controls); bs128 ci3
  1.0020 (+0.2%), ci3b (on B881) 1.0039 vs v1033 1.0029 (+0.1%); bs32 placebo 0.9969 / 0.9976. check_mt 0 failures.
  (CLD without CLDI lost: the aggregated claim's result wait moved into the stage loop.) Confirmation in run 4.

### run 4 (wpl2_run4.sh): late claim confirmation + item-boundary decomposition
bench2 jm 100 reps, 2 passes, three v1002 copies (first-position copy = 1.0000 reference; the non-first identical copies
v1002b / v1002c are the honest control: bs256 0.9956 / 0.9967, 0.9960 / 0.9952; bs128 0.9982 / 0.9978, 0.9988 / 0.9976):

| variant | bs256 pass 1 / 2 | bs128 pass 1 / 2 |
|---|---|---|
| CLD=3 + CLDI (ci3) | 0.9988 / 0.9983 | 1.0030 / 1.0024 |
| CLD=3 + CLDI copy (ci3x) | 1.0035 / 1.0016 | 1.0029 / 1.0020 |
| CLD=2 + CLDI | 0.9973 / 0.9963 | 1.0008 / 1.0002 |
| CLD=4 + CLDI | 0.9972 / 0.9964 | 1.0001 / 0.9986 |
| CLD=100 + CLDI (inc claim at the top, code present) | 0.9955 / 0.9974 | 1.0015 / 0.9996 |
B881 bs128 (v1033 copies 1.0000 / 0.9970 / 1.0004): CLD3+CLDI+B881 1.0022 / 1.0014, CLD2+CLDI+B881 1.0031.
=> CLD3 + CLDI over runs 3-4 (6 bs256 copies): 0.9995 1.0041 0.9988 0.9983 1.0035 1.0016 (mean 1.0010) vs the 6 non-first
identical controls 0.9950-0.9967 (mean 0.9958): +0.5% at bs256, every copy above every control; bs128 +0.45% ((8,4,1)
config, 5 copies) / +0.3% (on B881). D = 2 / 4 and the top-claim inc are +0.05..0.25% (inside the noise). check_mt
(CLD3+CLDI+B881, M 64 65 100 128 129 200 256 257, normal + skewed, 3 seeds, 2 calls): 0 failures; hang1 ok for every D.

Item boundaries (dev/wpl2_edge.py, MOE_WPN_TRACE, 3 jm seeds x 2 reps, cycles, v1002):
| item that ends | warp skew at its last stage | last warp done -> next item's first data | next item's first copies issued vs last done | copy latency | next first-stage compute vs mid-item |
|---|---|---|---|---|---|
| bs256 G1 ntl1 | 1075 | 1428 | -980 | 2407 | 1747 vs 1358 |
| bs256 G1 ntl8 | 847 | 1703 | -2938 | 4641 | 2077 vs 2365 |
| bs256 G2 ntl1 | 772 | 1468 | +106 | 1361 | 1461 vs 1367 |
| bs256 G2 piece | 524 | 1407 | -319 | 1726 | 1455 vs 1381 |
| bs128 G1 ntl1 | 1371 | 985 | -3325 | 4310 | 1504 vs 1393 |
| bs128 G2 piece | 654 | 929 | -1992 | 2920 | 1465 vs 1399 |
| B881 bs128 G1 ntl1 | 1024 | 1462 | -1189 | 2651 | 1576 vs 1358 |
| B881 bs128 G2 ntl1 | 851 | 1435 | +3 | 1432 | 1451 vs 1366 |
Mid-item the next stage's data is there 80-620 cycles BEFORE the slowest warp finishes (consumer-paced); at an item
boundary it arrives 930-1830 cycles AFTER. The next item's first copies go out only ~1000 cycles (G1) or 0 cycles (G2,
whose first stage also polls the tile's gate/up counters) before the item ends, although their slot is free ~2 stages
(~2700 cycles at NS 3) earlier: the producer's item transition (claim + decode + setup + first-stage generic path) is the
boundary cost, ~1.4-1.8k cycles (~0.8-1.0 us) per item at bs256, ~0.55 us at bs128: ~5% of bs256's time, ~4% of bs128's.

### run 5 (queued; run 6 folded in per the planner's one-job queue rule)
- Planner request: power / energy / in-kernel clock at bs128 / bs256 (dev/wpe_energy.py, judge mode = flush + one replay per
  call with the flush-only energy subtracted, and back to back; MOE_CLK in-kernel clock), and the x-traffic energy bound:
  MOE_L2_XHALF (new, timing only: the lean-path stages copy half their activation bytes) and MOE_NO_X (no activation copies;
  also drops the producer's gather wait) with their clocks and mJ / call; bench2 time bounds of both (B2_NOVERIFY).
- MOE_L2_PREP (new, with CLD + CLDI): MOE_L2_PPD stages after the late claim the producer decodes the next item and, for a
  down item, loads its tile's 16 gate/up chunk counters with ld.relaxed; at that item's top a ballot of the loaded values +
  fence.acq_rel.gpu (acquire pattern) + fence.proxy.async replaces the first stage's polled ld.acquire round trip (edge data:
  G2 items' first copies go out ~1100 cycles later than G1 items'), and the whole item takes the lean path. Consumer loops:
  one wgmma loop re-allocated (register names), producer spills 20 -> 56 B (bs256 kernel, per-item code, not stage loops).
- Edge decomposition now also reports the producer side (last stage's copies -> next slot acquired, when that slot was
  freed by the consumers, slot freed -> acquired, acquired -> copies issued) for v1002, CLD3+CLDI, +PREP.

### run 5 results (wpl2_run5.sh, run 6 folded in)
Power / energy / clock (dev/wpe_energy.py, 8 jm routings, 2 rounds x 2.5 s; judge = 512 MB flush + one replay + sync per call,
flush-only energy subtracted; MOE_CLK in-kernel clock = median over CTAs, and by 20-us bin of the call):

| bs | mode | variant | us / call | mJ / call | avg W | NVML SW power cap | in-kernel clock (GHz) |
|---|---|---|---|---|---|---|---|
| 256 | judge | v1002 (2 copies) | 197.8 / 197.9 | 177.0 / 172.6 | 661 | 0.00 | 1.698 / 1.703; by bin 1.83 (0-20 us) -> 1.57-1.66 (60-120 us) -> 1.74 |
| 256 | judge | XHALF (half the lean-stage x bytes) | 194.9 (-1.5%) | 169.9 (-4%) | 654 | 0.00 | 1.733 (+2%) |
| 256 | judge | NO_X (no x copies) | 192.7 (-2.6%) | 168.2 (-5%) | 640 | 0.00 | 1.753 (+3%) |
| 256 | b2b | v1002 | 224.6 | 155.4 | 697 | 1.00 | 1.417 |
| 256 | b2b | XHALF / NO_X | 220.0 (-2.0%) / 210.0 (-6.5%) | 146.7 / 146.1 | 693 / 697 | 1.00 | 1.433 / 1.509 |
| 128 | judge | v1002 (2 copies) | 164.9 / 164.9 | 144.5 / 145.0 | 641 | 0.00 | 1.714 |
| 128 | judge | XHALF / NO_X | 163.6 (-0.8%) / 159.4 (-3.4%) | 142.0 / 128.0 (-11%) | 636 / 599 | 0.00 | 1.715 / 1.764 |
| 128 | judge | B881 / B881 + XHALF | 164.3 (-0.2%) / 162.5 (-1.3%) | 140.8 / 138.5 | 625 | 0.00 | 1.699 / 1.717 |
| 128 | b2b | v1002 / XHALF / NO_X | 187.0 / 185.9 / 173.5 (-7.2%) | 130.0 / 126.3 / 118.3 | 700 / 696 / 697 | 1.00 | 1.396 / 1.403 / 1.523 |

Reading: under the judge protocol the SW power cap never engages (NVML 1980 MHz, ~640-660 W averaged over flush + call),
but the in-kernel clock sits at 1.70 (bs256) / 1.71 (bs128) GHz, below the 1.80 GHz tensor cap, and sags to ~1.6 GHz in the
middle of the call: a fast (sub-ms) power / current limiter that NVML's 2 ms samples do not see. Less on-chip x traffic does
raise it (bs256 XHALF +2%, NO_X +3%), so x energy is a real but small lever. Time bounds on bench2 (jm 60 reps, B2_NOVERIFY,
v1002 x2 = 1.0000 / 0.9979 at bs256, 0.9981 at bs128): XHALF bs256 1.0062, bs128 1.0048; NO_X bs256 1.0133, bs128 1.0255.
=> halving the activation bytes is worth <= ~0.7% at bs256 (NO_X, which also drops the gather wait, 1.4%): 512-row items /
x reuse would recover at most that before their own costs (WP-X measured 1.8-3.1% code-presence cost for XR) - not pursued.

MOE_L2_PREP: correct (check_mt 0 failures, hang1 ok) but slower: bench2 jm 100 x2 passes, bs256 0.9899 / 0.9895 (copy 0.9957 /
0.9961), PPD with D=4 0.9890 / 0.9901; bs128 0.9888 / 0.9895, 0.9871 / 0.9888, copy 0.9907 / 0.9912 (controls bs256 0.9953 /
0.9955, 0.9947 / 0.9969; bs128 0.9959 / 0.9971, 0.9980 / 0.9984). Edge trace: the G2 first stage's acquired -> copies time
drops 1925 -> 237 cycles as intended, but slot freed -> acquired rises 1414 -> 3056 (and 1106 -> 1955 for G1 items): the
producer's mid-item extra work (the shfl of the claim result, decode, relaxed loads, more producer spills 20 -> 56 B) delays
it more than the poll it saves. REJECTED.

CLD3 + CLDI in this job: bs256 0.9961 / 0.9963 and copy 0.9964 / 0.9979 vs controls 0.9953 / 0.9955 / 0.9947 / 0.9969 (+0.1%);
bs128 1.0001 / 1.0000, copy 0.9985 / 1.0012 vs 0.9959 / 0.9971 / 0.9980 / 0.9984 (+0.26%). wpe_energy (8 routings): bs256
197.6 vs 197.8 us, bs128 164.9 vs 164.9. Over 3 jobs: bs256 +0.5 / +0.5 / +0.1%, bs128 +0.2 / +0.45 / +0.26%.

Producer side of the item boundary (dev/wpl2_edge.py, cycles; v1002 / CLD3+CLDI / +PREP):
| item that ends | last copies -> next slot acquired | of which: slot freed after the last copies, then freed -> acquired | acquired -> copies |
|---|---|---|---|
| bs256 G1 ntl1 | 2667 / 2276 / 3114 | 1104 + 1563 / 1170 + 1106 / 1159 + 1955 | 859 / 960 / 248 |
| bs256 G2 ntl1 | 2817 / 2576 / 4189 | 1122 + 1695 / 1163 + 1414 / 1133 + 3056 | 1823 / 1925 / 237 |
| bs128 G1 ntl1 | 2542 / 2196 / 2551 | 1044 + 1498 / 1168 + 1028 / 1108 + 1442 | 538 / 563 / 398 |
The next item's first slot is free ~1.1k cycles after the producer's last copy of the current item; the producer then needs
another 1.0-1.7k cycles (item transition: claim wait, decode, setup in a warp that shares SMSP0 with two consumer warps) plus
0.25-1.9k (record, expect_tx, copies; G2: the dependency poll) before the next item's first copies go out. CLDI removes ~450
cycles of claim wait (data gap at the boundary only -57: copy latency grows 2441 -> 2767 when issued earlier, the SM's TMA
stream is busy in order). Moving work into the item (PREP) costs more than it saves.

## Summary (so far)
Where bs128 / bs256 time goes on v1002 (judge protocol, per CTA): startup to first data bs256 11.6 us (gated by the GCP gather
barrier at 10.6 and, for 40-64-token first tiles, by a ~7 us cold warm-up), bs128 7.3 us; item boundaries ~0.8-1.0 us each at
bs256 (~5% of the call) / ~0.55 us at bs128 (~4%): the producer's item transition leaves the next item's first slot unused for
1.0-1.7k cycles plus 0.25-1.9k cycles of first-stage issue work; tail idle bs256 4.8 us (bs128 2.5) but much of it is filled
by the token combine (c9 - c8 ~4.2 us per CTA); end phase 2.5-3.0 us. 64-token stages are tensor-bound (1.78 us / stage).
Clock: 1.70 / 1.71 GHz in judge mode (no SW power cap; a fast limiter sags it to ~1.6 GHz mid-call); x traffic is worth
<= 0.7% (XHALF) / 1.4% (NO_X) at bs256.
Kept: MOE_L2_CLD=3 + MOE_L2_CLDI (late non-aggregated claim): bs256 +0.1..0.5%, bs128 +0.26..0.45% (+0.1..0.3% on B881);
ported to the integration base as dev/moe_v1401.py (= v1500 + switches, default = v1500 SASS; with the defines only the three
(8,8,1) kernels change, byte-identical to the measured v1400 builds).
Rejected: late claim without CLDI, PREP (early G2 poll), XD8 (XDIR in GCP kernels: race), Duff i-cache stagger (no per-line
cost to spread), x-reuse redesigns (bound <= 0.7%).

### run 7 (wpl2_run7.sh): late claim on the integration base (dev/moe_v1401.py = v1500 + MOE_L2_CLD / CLDI)
hang1 65-512 ok, check_mt 64 65 100 128 129 200 256 257 300 512 (normal + skewed, 3 seeds, 2 calls) 0 failures (and with
SPLW8 0.8f at 129 / 256 / 257). bench2 jm 100 reps, 2 passes, v1500 copies 1.0000 (first) and non-first controls:
| | bs128 (v1500 = B881: (8,8,1) MW4) pass 1 / 2 | bs256 pass 1 / 2 |
|---|---|---|
| v1500b / v1500c (controls) | 0.9976 / 0.9979, 0.9967 / 0.9983 | 0.9947 / 0.9942, 0.9949 / 0.9949 |
| CLD3+CLDI x3 | 1.0008 / 1.0002, 1.0018 / 0.9993, 1.0020 / 1.0025 (+0.35%) | 0.9958 / 0.9965, 0.9950 / 0.9954, 0.9978 / 0.9969 (+0.15%) |
| CLD3+CLDI + SPLW8 0.8f | 1.0011 / 1.0044 (+0.5%) | 1.0009 / 1.0002 (+0.6%) |
Late claim over 4 jobs: bs256 +0.5 / +0.5 / +0.1 / +0.15%, bs128 +0.2 / +0.45 / +0.26 / +0.35%. A deeper tail split (SPLW8
0.8: 7 split tiles instead of 5 at bs256) on top of it looks better (+0.6% / +0.5%, 2 samples each; SPLW8 0.8 alone on v969
was +0.16%): run 8 sweeps SPLW8 0.8 / 1.0 / 1.2 with the late claim and 0.8 alone.

### run 8 (wpl2_run8.sh): tail split depth with the late claim, on v1401
hang1 ok, check_mt (65 100 128 129 200 256 257 512) 0 failures for CLD3+CLDI with SPLW8 0.8 / 1.0 / 1.2.
bench2 jm 100 x2 passes (non-first v1500 controls bs256 0.9947 / 0.9953, 0.9949 / 0.9951; bs128 0.9985 / 0.9968, 0.9967 / 0.9973):
| variant | bs256 | bs128 |
|---|---|---|
| CLD3+CLDI (cl) | 1.0021 / 1.0007 (+0.65%) | 1.0041 / 1.0027 (+0.6%) |
| + SPLW8 0.8 (2 copies) | 0.9962 / 0.9944, 0.9940 / 0.9944 (0) | 0.9985 / 0.9989, 0.9977 / 0.9969 (+0.1%) |
| + SPLW8 1.0 / 1.2 | 0.9946 / 0.9937, 0.9918 / 0.9943 | 0.9978 / 0.9962, 0.9931 / 0.9938 |
| SPLW8 0.8 alone | 0.9966 / 0.9985 (+0.25%) | 1.0027 / 1.0009 (+0.45%) |
=> the run-7 SPLW8 0.8 + late-claim result does not replicate; deeper splits lose; SPLW8 0.8 alone is within noise (+0.16% on
v969 earlier). Late claim CLD3+CLDI over 5 jobs: bs256 +0.5 / +0.5 / +0.1 / +0.15 / +0.65% (mean +0.38%), bs128 +0.2 / +0.45 /
+0.26 / +0.35 / +0.6% (mean +0.37%); never below its controls.

### run 9 (queued): MOE_L2_PAIR on v1401
The direct test of the boundary bucket: with CLD + CLDI, the late claim takes 2 queue items (atom.add 2 through an address
ptxas cannot prove uniform, so not warp-aggregated) while the claiming item is more than MOE_L2_PAIRM waves before the end
of the gate/up items; when the second item is row block rb + 1 of the same tile (gate/up only), it runs inside the first's
lean stage loop: weights source + one 256-row block, activations back to stage 0, a new stage record at its stage 0, the late
claim re-armed in it; held-slot bookkeeping (nb_le / nb_was) unchanged. If the pair cannot run that way, the second item is
simply the next one (no claim in the first). Consumers unchanged (they see two items). SASS: producer stage loops spill-free;
per-item producer code 3 -> 16 STL (72-register warpgroup); one consumer wgmma loop of moe_8_8_1_3_4 re-allocated.
Expected: ~half of each CTA's gate/up boundaries (~2.5 at bs256) lose their ~0.85 us data gap -> up to ~1% at bs256 / bs128.

### run 9 results: MOE_L2_PAIR rejected; the boundary metric re-read
hang1 65-512 ok, check_mt 64-512 0 failures. bench2 jm 100 x2 passes (v1500 non-first controls bs256 0.9963 / 0.9960, 0.9972 /
0.9961; bs128 0.9982 / 0.9970, 0.9973 / 0.9983): PAIR bs256 0.9931 / 0.9932, copy 0.9921 / 0.9918, PAIRM 3 0.9938 / 0.9936;
bs128 0.9905 / 0.9908, 0.9913 / 0.9919, PAIRM 3 0.9969 / 0.9965 => -0.3..-0.65%. CLD3+CLDI in the same job: bs256 0.9972 /
0.9958, copy 1.0008 / 1.0019 (+0.2%), bs128 1.0001 / 1.0022, copy 1.0032 / 1.0058 (+0.4%).
Edge trace with PAIR: G1 boundaries unchanged (last warp done -> next stage's stamp 1440 vs 1399 cycles, freed -> acquired
1046 vs 1099) although half of them no longer have a producer transition. Correction to the run-4 reading: the consumer's
per-stage "data-ready" stamp (MOE_WTR0) sits inside the item's stage loop, AFTER the item-top wait, the record read and the
dispatch, so "last warp done -> next item's first stamp" (~1400 cycles at bs256) is mostly CONSUMER item-transition work:
the epilogue staging + consumer_sync (code 2 -> 3: 0.57-0.85 us at bs256 incl. ~1000 cycles of warp skew), acc zeroing,
the item top (record, ntl dispatch, per-item setup) and only ~0.3 us of real data wait (probe code 3 -> 1); plus +390 cycles
in the next item's first stage. The producer is not the bottleneck at boundaries (its first copies go out 1-3.6k cycles
before the consumers need them for G1), which is why CLDI / PREP / PAIR move the producer without moving the boundary.
Remaining lever (not attempted): a leaner consumer item transition in the (8,8,1) kernels (bs256: ~1 us per item, ~5%);
every consumer-side edit re-schedules the stage loops (+-0.5..1%, WP-X / WP-DS / WP-EDGE), so it needs a call-boundary or
byte-identical-loop formulation.

## Final summary (supersedes "Summary (so far)" where they differ)
Kept (bank): dev/moe_v1401.py (= planner base v1500 + WP-L2 switches; default build = v1500 SASS in all 16 kernels) with
defines MOE_L2_CLD=3,MOE_L2_CLDI: the (8,8,1) kernels' producer claims its next queue item 3 stages before the current item's
last stage, with a bounded atom.inc (not warp-aggregated by ptxas). Only moe_8_8_1_3_4 / _3_8 / _3_16 change (consumer stage
loops byte-identical; producer spills 20 -> 16 B). Same kernels measured on v1002 (dev/moe_v1400.py, same defines; there the
bs65-128 (8,4,1) kernel changes too). bench2 jm100 vs the non-first identical control copies, 6 jobs (runs 3, 4, 5, 7, 8, 9):
bs256 +0.5 / +0.5 / +0.1 / +0.15 / +0.65 / +0.2% (mean +0.35%), bs128 +0.2 / +0.45 / +0.26 / +0.35 / +0.6 / +0.4%
(mean +0.38%; runs 7-9 on v1500 = B881 bs128); never below its controls. Correctness: hang1 65-512 ok; check_mt 64 65 100 128
129 200 256 257 300 512 (normal + skewed, 3 seeds, 2 calls) 0 failures. Projected ~+0.12% geomean (~6 us).
Measured and rejected: late claim with the aggregated atom.add (stall), D = 2 / 4, inc-at-top alone, PREP (early down-item
poll), PAIR (2-item claims, in-loop next row block), XD8 (XDIR in GCP kernels: race), SPLW8 0.8 / 1.0 / 1.2 with the late claim,
Duff staggered i-cache warm-up (no per-line cost), x-reuse redesigns (XHALF bound <= 0.7% at bs256).
Facts for the planner: judge-mode clock 1.70 / 1.71 GHz without the SW power cap (fast limiter, sags to ~1.6 GHz mid-call), x
traffic worth <= 0.7 / 1.4% (half / all) at bs256; bs256 boundaries cost ~1 us per item, almost all consumer-side (staging +
warp skew + item top + slower first stage), not the producer; tail idle 4.8 us is mostly filled by the token combine.
