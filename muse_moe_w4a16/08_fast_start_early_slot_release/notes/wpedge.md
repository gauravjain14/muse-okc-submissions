# WP-EDGE: time outside the steady stage loop at bs8 ((4,1,1)) and bs32 ((8,4,1) MW1), base v1002 (2026-10-27)

Worker WP-EDGE. Files: dev/wpedge_mk1020.py -> dev/moe_v1020.py (v1002 + timing-only edge bounds, all off by default:
default build SASS-identical to v1002 in every kernel), dev/wpedge_ana.py (per-CTA budget from a dev/wp32_probe.py
dump), dev/wpedge_bw.py (bytes / steady window, streaming-CTA time series), dev/wpedge_run*.sh. Logs: scratchpad
wpedge/run*.log; dumps dev/remote_out/wpedge_tr1_{8,32}.json.

## 1. Accounting (wpedge_run1: MOE_WP32_TR + MOE_TRN item trace, graph replay after a 512 MB L2 flush, 12 jm seeds each;
per-stage MOE_WPN_TRACE on 3 seeds each)

Per CTA, us (sums to the trace build's kernel time; trace build ~+1-2% over v1002):

| bucket | bs8 (63.84) | bs32 (113.96) | notes |
|---|---|---|---|
| steady stage loops (first data -> compute done) | 50.2 | 98.7 | bs8: 7.6 items/CTA; bs32: 9.1 items/CTA (3.6 G1 1-group, 4.3 KRED pieces, 0.7 G2 whole, 0.5 multi-group) |
| startup (kernel start -> first item's first data) | 5.37 | 6.01 | bs32: 27 CTAs/call start on 3-4-group (wgmma) tiles: first data 9.8-10.3 us vs 4.6 for 1-group CTAs |
| end (last hand-off -> last combine done) | 2.83 | 2.89 | |
| item end (compute done -> handed off), consumer warp 0 | 2.02 (0.25 us/edge) | 2.27 (0.27 us/edge) | bs8 SEPI hand-off; bs32 bar.sync + staging + fence + hand-off |
| gaps (handed off -> next item's first data) | 2.43 (0.32 med) | 2.83 (0.29 med) | mid-item consumer data wait is ~0.13-0.23 us |
| tail (latest CTA's last hand-off - own) | 1.67 | 2.46 | last items are 2-stage (bs8) / 4-stage (bs32) KRED pieces |

DRAM view (wpedge_bw.py): weight+scale bytes over the median-first-data -> median-last-hand-off window: bs8 4.35 TB/s,
bs32 4.25 TB/s (bare ring 4.23-4.5). bs8's steady phase runs at the ring rate, so its per-CTA edge time is absorbed
(as WP-N's CLM / X1 found); its excess is startup + tail + end (~9.7 us of 63.8). bs32's steady phase is ~2.4% below
bs8's rate (multi-group CTAs + item edges).

Per-stage trace (cycles, mean): first stage of an item vs middle stages:
- bs8 gate/up: period 2407-2567 vs 1968-1987; consumer gap 786-924 vs 418-425; producer reaches the slot late
  (pidle 1747-2335 vs 538-586); producer per-stage copy issue (piss) ~1050-1240.
- bs32 gate/up 1-group: period 3233-3668 vs 1817-1896; gap 1455-1896 vs 276-349; pidle 1909-2055 vs 659-702.
- bs32 KRED pieces: period 3018-3149 vs 1776-1786; gap 1290-1407 vs 233-251. Down whole items: period 3887-3913.
- consumer compute per 1-group stage: 1390 (bs8) / 1430-1440 (bs32) cycles vs a ~1800-1900 period (C/M ~0.75-0.80).

## 2. Timing-only bounds (dev/moe_v1020.py; bench2 jm100, B2_NOVERIFY, same job, v1002 x2)

| bound | bs32 | bs8 |
|---|---|---|
| control v1002b | 0.9994 / 0.9985 | 0.9954 |
| BS: item end without bar.sync + staging stores (hand-off and fence kept) | 1.0071 / 1.0061 | (identical kernel) |
| BSB: without the bar.sync only | 1.0035 | |
| BSF: without the consumers' fence.proxy.async only | 0.9964 | |
| BH: held slot released at the epilogue's wake | 1.0008 | |
| BS + BH | 1.0091 | |
| B8: SEPI item end without the staging stores | | 0.9987 |
| MOE_NO_COMPUTE | 1.0708 | 1.0566 |
| MOE_NO_COMBINE | | 1.0302 |

The consumers' whole item-end work at bs32 is worth <= ~0.7% (the bar.sync ~0.4%, the stores ~0.3%, the fence 0); the
held slot (after TMP) nothing. At bs8 the SEPI item end is free already; its excess is startup / tail / end.
(Bounds that drop the stores must keep the accumulators live: an empty asm use or an `f.n < 0` guard lets the compiler
delete every HMMA; a runtime-false `M < 0` guard keeps the stage loops byte-identical.)

## 3. First use of code (planner question; dev/wpedge_first.py, dev/wpedge_st0.py, wpedge_run3/4)

Item trace, per CTA (us/stage of the item loop): the CTA's first item bs8 1.210 vs later gate/up 0.968 (+1.9 us per
CTA), bs32 1.117 vs 0.987 (+2.1 us); the first 1-group item of a bs32 CTA that started on a wgmma tile (code cold, DRAM
already streaming) +46 ns/stage and a +0.4 us gap (~1.1 us once); first down item / first piece <= +0.1 us.
Per-stage trace of the first stages (graph replay after the L2 flush, cycles): bs8 stage 0 compute 2705 vs ~1400 later
(cold code ~1300 cycles = 0.73 us), but stage 1's data lands just as stage 0 ends (gap 249): the cold compute hides behind
the ring's fill. bs32 stage 0 compute 1809 vs ~1555 (+250 cycles). The CTA's second item's first stage waits 2046 cycles
at bs32 (later item starts ~1300-1500): ~0.3 us once per CTA (first run of the producer's item path).
Candidate MOE_EDGE_WE=1 (dev/moe_v1021.py: a warm-up record in slot 0 ahead of the (4,1,1) early stages; consumer loop
re-scheduled 0.971 but identical to its code-presence control WEN): the warm stage takes the cold fetch (2811 cycles) and
the first real stage runs warm (1439), yet bench2 bs8 0.9917 / 0.9973 vs control 0.9987, WEN 1.0037: no gain, rejected.
First-use code fetch is not a wall-time bucket at bs8 / bs32: the first item's excess is the all-CTA DRAM ramp.

## 4. bs17-32 barrier-free item end (dev/moe_v1022.py MOE_EDGE_PWS, MOE_EDGE_PWSNS)

Each consumer warp stages its FP32 accumulators into its own 4 KB of weight bytes of the held slot (warp rw reads only
row groups 2 rw, 2 rw + 1), [32 tokens][32 rows]; no bar.sync 1, 256; the fence and hand-off are kept. The epilogue warps
read that layout (pidx), and WP-A's TMP bulk copy becomes 8 per-region copies of nb x 128 B. PWSNS (no swizzle: 4-way
conflicted staging stores) keeps both consumer stage loops byte-identical to v1002 and changes only moe_8_4_1_4_1; the
swizzled form re-schedules them (similarity 0.51). Default build = v1002 in all kernels.

wpedge_run5 (bench2 jm100, all six shapes, v1002 x2, B2_SHARE_EQ):

| | bs1 | bs8 | bs16 | bs32 | bs128 | bs256 |
|---|---|---|---|---|---|---|
| v1002b (control) | 0.9974 | 0.9981 | 0.9994 | 0.9982 | 0.9993 | 0.9953 |
| PWS x2 | 0.9948 / 0.9970 | 0.9991 / 1.0011 | 0.9981 / 0.9987 | **0.9908 / 0.9907** | 0.9985 / 0.9983 | 0.9945 / 0.9939 |
| PWS swizzled | 0.9948 | 1.0009 | 0.9967 | 0.9880 | 0.9989 | 0.9945 |
| WE=1 | 0.9937 | **0.9859** | 1.0007 | 0.9990 | 1.0005 | 0.9962 |
| WE=1 + PWS | 1.0000 | 0.9915 | 1.0004 | 0.9925 | 0.9998 | 1.0016 |

(bs1 / bs16 / bs128 / bs256 and, for PWS, bs8 run byte-identical kernels: their spread is noise.)
check_mt (PWS) 15 sizes 1-257 0 failures; outputs match v1002 up to the KRED atomic-order noise.
PWS loses ~0.75% at bs32 against a bar.sync bound of +0.35%: what it adds costs more than the barrier. That is the
conflicted staging stores on the consumers, which pace the ring, plus 8 small bulk copies and the extra index math on the
issue-starved epilogue warps. It joins CSO / BSTG / KDIR: at bs32 the consumers' item end cannot be traded for more work
anywhere on the item-end path. Rejected.

## 5. Conclusion
- Nothing to integrate: no candidate beat the controls (v1029 not baked).
- At bs32 the consumers' entire item end is bounded at +0.6-0.7%, and the barrier part at +0.35%. Reformulations
  cost more than that, and the held slot is worth nothing after TMP.
- First-use code fetch is not a wall-time bucket at bs8 / bs32. The cold first stage hides behind the all-CTA DRAM fill,
  and a warm-up stage loses.
- bs8's steady phase streams at the ring rate (4.35 TB/s), so its edges are absorbed. What remains is startup (first data
  5.4 us), the tail (~1.5 us: 2-stage pieces) and the end chain (NO_COMBINE bound +3.0%).
- bs32's tail (2.5 us) comes from its 4-stage KRED pieces. KSX1 = 8 (2-stage pieces) does not compile with the MW >= 2
  LEAN producers (static_assert: items of >= NS stages); it would need a per-MW KSX for the MW 1 kernel. Not tried.
