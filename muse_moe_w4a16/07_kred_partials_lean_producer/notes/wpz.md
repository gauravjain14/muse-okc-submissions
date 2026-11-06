# WP-Z findings (bs256 specialist)

Worker Z, 2026-10-27. Base: dev/moe_v655.py (okc check 5427.8 us; bs256 kernel moe_8_8_1_3_8 byte-identical to v567's).
Files: dev/moe_v690.py.. (candidates), dev/wpz_run*.sh (Modal jobs), dev/wpz_ana.py (per-CTA tail / budget from a
dev/wpl_trace.py dump), dev/wpz_start.py (startup event medians from the same dump), dev/wpz_sim.py (queue model).
Logs: scratchpad wpz/run*.log; trace dumps in dev/remote_out/wpz_tr*.json.

## 1. Time budget of v655 at bs256 (run 1: 8 jm seeds, judge flush before the traced call, MOE_TRACE + MOE_TRN)

Mean per CTA over the 8 routings (trace build: kernel end 222.9 us mean; bench2 of the same version ~213 us):

| bucket | us | share | notes |
|---|---|---|---|
| first real stage (startup) | 11.5 | 5.2% | see the chain below |
| compute (item start -> item compute end, incl. in-item data / slot waits) | 190.3 | 85% | ntl1 42%, ntl2 23%, ntl3-7 26%, ntl8 8% of it |
| epilogue (compute end -> item end, consumer staging) | 5.3 | 2.4% | 0.55 us / item |
| gaps (item end -> next item's first stage) | 2.9 | 1.3% | 0.30 us / item, 9.6 items / CTA |
| tail idle (CTA's last item end -> last item end of the call) | 9.8 | 4.4% | |
| end phase (last item end -> kernel end, combine) | 3.1 | 1.4% | |

Per-stage time (us, median over items, trace build): G1n1 1.081, G1n2 1.364, G1n3-7 1.587, G1n8 1.832, G2n1 1.024,
G2n2 1.262, G2n3-7 1.533, G2n8 1.864 (G1 runs earlier at a lower clock than G2).

Startup chain (medians over CTAs, then over runs; us after the first CTA's entry): bitmaps 1.82, ranked 2.83, pair list
3.70, consumers' setmaxnreg 3.87, warm-up stage start 4.24, producer has its item 4.21, slot 5.22, stage-0 weights issued
5.68, producer starts waiting for the all-CTA gather 6.53, GCP gather done med 7.99 / max 9.28, warm-up end med 8.86 /
max 11.14, gather barrier seen 10.11, stage-1 slot 11.14 (the producer serializes: stage 0 x waits for the barrier
before stage 1's weights go out), first real stage 11.25 (x copy ~1.1 us after the barrier), stage-2 weights 13.0.

Tail: the last item of every CTA is a G2n1 item (~16 us). CTAs run in lockstep rounds near the end: e.g. seed 71497
last-item start p10/50/90/max 186.2 / 190.2 / 202.8 / 203.7: 80-90% of the CTAs finish at ~203-207, 10-20% claim one
more G2n1 item and finish 12-14 us later. Mean tail idle 7.4-10.9 us per routing.

Clock (wpe_energy judge mode, MOE_CLK, K=8): 213.1 us, 190.6 mJ / call, in-kernel 1.72 GHz; by 20-us bin 1.87 (0-20),
1.73, 1.76, 1.69, 1.66, 1.70, 1.70, 1.71, 1.71, 1.76, 1.76; NVML power cap active in only 7% of the samples (GPU at
56 C): the mid-call dip below the 1.80 GHz tensor clock is not the board power cap alone.
End spread (plan_spread, MOE_PROBE, 8 seeds): producer exit p10/50/90/max 191.7 / 195.5 / 205.9 / 206.8, CTA done
202.1 / 203.6 / 208.4 / 212.4, kernel end 212.4 us.

### Queue model (dev/wpz_sim.py)
Replays the kernel's queue (G1 tile-major largest first, then G2; static first wave; next item claimed when the
current one starts; G2 after its tile's G1) with the traces' per-(phase, ntl) stage, epilogue and gap medians.
Reproduces the traced last-item end within 1-3 us on all 8 routings. Results (mean over the 8 routings):
- static k-split of the last X down items into K pieces, free pieces: best +1.5..2.4% (X 24-132); with 0.5 us extra
  per piece +1.0..1.7%; with 1 us extra <= +1.3%. Per-routing spread -0.9..+4.4%. => the tail split is a ~1-2% lever
  before any code-presence cost, consistent with WP-L's KSFREE bound (+0.45% measured on a KSX8=2 build).
- first real stage 3 us earlier: +1.4% (an upper bound: the measured SDELAY response at bs256 is ~half of the delay).
- per-item gap -0.4 us: +1.85% (epilogue / slot hold: WP-W's area).

## 2. Log
- run1: budget above.
- run2 (dev/moe_v690.py, bs256 jm 100 reps, v655 x2; control b x0.9954): MOE_WZ_WDELAY (the producer's first weights
  wait for this CTA's own GCP release) x0.9953 / x0.9959, WDELAY + MOE_XPF x0.9951, XPF alone x0.9957: neutral,
  rejected. Traces (4 seeds): gather done med 8.0 -> 7.1 / max 9.3 -> 8.2 us, but the all-CTA barrier is still seen
  at 10.15 us (the last releases take ~1 us longer while the other CTAs' weight bursts land) and the first real stage
  moves 11.18 -> 11.01 only. The barrier crossing (release + poll under load, 0.8-2 us) is the gate, not the weights.
- run3 (WP-K's dev/moe_v656.py KRED-GCP split, decomposed; v655 x2, control b x0.9952):

  | variant | bs256 vs v655 |
  |---|---|
  | KSX8=2 SPLW=0 (split code compiled, no split items) | **x0.9764** |
  | KSX8=2 (X=144 down items in 2 k pieces) | x0.9892 |
  | KSX8=4 SPLW=0.45 (X=64, 4 pieces) | x0.9885 |
  | KSX8=2 SPLW=0.45 (X=64, 2 pieces) | x0.9864 |

  => the split is worth +1.2..1.3% over its own build (the queue model said +1.0..1.95%), the build costs 1.9%.
  Traces (dev/wpz_cls.py, 4 seeds), code-only build vs v655: first real stage 11.43 -> 12.70 us (GCP gather done
  med 8.0 -> 9.65: v656 zeroes the split rows before the gather's cp.asyncs and adds two 64-bit division calls -- a
  cold subroutine at the end of the kernel -- on the startup path; the producer's first weights 5.7 -> 6.4), G2
  per-stage +2.4% (G1 unchanged), tail idle +0.7, end phase +0.3. X64K4 vs code-only: tail idle 9.8 -> 3.75 us, end
  phase 3.45 -> 5.26 us (split tokens' combine waits for 64 pieces' red.adds and their releases).
  check_mt M 129 200 256 257 384 512 (normal + skewed, 3 seeds, 2 calls): X64K4 and X64K2 0 failures, rel-L2 3.26e-3.
- run4 (dev/moe_v691.py; v655 x2, control b x0.9962): MOE_WZ_GDIV32 (GCP pair range in 32-bit math, no 64-bit
  division calls on the cold path) x0.9965 / x0.9969: noise, rejected (trace: gather done med 8.0 -> 7.1 us, first
  real stage 11.43 -> 11.21). Split with the zeroing moved after the gather + 32-bit math: code only x0.9856 (v656's
  0.9764), X64K4 x0.9939. Traces: X64K4 tail idle 9.1 -> 3.6 us but the multi-group stages +2.5..4% and the end phase
  +2 us; the last item ends at the same time as without the split on 3 of 4 routings.
- run5 (dev/moe_v692.py MOE_WZ_SDEC, division-free item decode; v655 x2, bs32 / 128 / 256): SDEC changes only the
  (8,4,1) kernels: bs32 x1.0053 / 1.0055 / 1.0070 / 1.0082 over four builds sharing that kernel (controls 1.0000 /
  0.9969), bs128 x1.0009..1.0017 (controls 1.0000 / 0.9980). bs256 split with SDEC: code only x0.9841, X64K4 x0.9907,
  X80K4 x0.9983 (control b x0.9954). check_mt 0 failures.
- run6 (dev/moe_v693.py MOE_WZ_KSXR role mask, no split items; control b x0.9969): none 0.9973, producer 0.9989,
  consumer stage count 0.9929, epilogue warps 0.9873, combine 0.9949, GCP zeroing 0.9948, all 0.9882. The cost sits
  in the epilogue warps' store loop (full_k branch inside the loop) and the consumers' runtime stage count.
- run7 (dev/moe_v694.py = v693 + MOE_WZ_EHOIST: the epilogue warps' full-k down store loop hoisted out of the split
  branch; all six shapes, v655 x2):

  | variant | bs1 | bs8 | bs16 | bs32 | bs128 | bs256 |
  |---|---|---|---|---|---|---|
  | v655b (control) | 1.0000 | 1.0013 | 0.9983 | 1.0013 | 0.9995 | 0.9958 |
  | SDEC | 0.9972 | 1.0029 | 0.9994 | 1.0105 | 0.9994 | 0.9966 |
  | SDEC + GDIV32B (x2) | 0.9968 / 1.0000 | 0.9983 / 0.9998 | 1.0002 / 0.9995 | 1.0117 / 1.0111 | 1.0015 / 1.0040 | 0.9957 / 0.9953 |
  | + EHOIST | 0.9974 | 1.0043 | 1.0032 | 1.0113 | 1.0083 | 0.9968 |
  | k5h = + GDIV32 + KSX8=4 + SPLW8=0.6f | 1.0000 | 1.0008 | 1.0022 | 1.0126 | 1.0095 | **1.0085** |

  bs1 / 8 / 16 run byte-identical kernels in every variant (noise). With EHOIST the bs256 split pays for the first
  time. hang1 13 shapes ok; check_mt k5h M 1..257 (15 sizes) 0 failures, max rel-L2 3.30e-3.
- run8 (confirmation, fresh copies): bs32 k5h x1.0101 / 1.0086, bs128 x1.0060 / 1.0061, bs256 x1.0038 / 1.0038
  (controls 1.0004 / 0.9985 / 0.9963); split amount at bs256: SPLW8 0.45f x1.0018, 0.75f x1.0058. bs1 / 8 / 16
  identical kernels (bs1 0.9875..0.9960 vs control 0.9963: noise on a byte-identical one-token kernel).
- baked: dev/moe_v695.py (v694 + the k5h defines on by default; MOE_NO_WPZ = v655's kernels; SASS checked locally:
  default == v694 + defines, NO_WPZ == v655). run9 validates it.
- run9 (baked dev/moe_v695.py, v655 x2; controls bs32 0.9977, bs128 0.9994, bs256 0.9965): v695 bs32 x1.0080 /
  1.0081, bs128 x1.0079 / 1.0062, bs256 x1.0058 / 1.0013; with SPLW8 0.75f x1.0084 / 1.0111, 1.0066 / 1.0101,
  1.0046 / 1.0069; bs1 / 8 / 16 identical kernels within noise. hang1 13 shapes ok; check_mt M 1..512 (17 sizes)
  0 failures, max rel-L2 3.30e-3.
- run10 (planner request, dev/moe_v696.py = v695 + MOE_WZ_TGCP + MOE_WZ_TG41: per-tile gather readiness; v655 x2,
  v695 x2): vs the v695 copies bs32 +0.17%, bs128 +0.3%, bs256 0 (x1.0033 / 1.0034 vs 1.0025 / 1.0032); bs1/8/16
  identical kernels. Trace bs256: the first real stage does not move (10.78 -> 10.88 us): with the barrier gone the
  cold warm-up of the n5-n8 wgmma loops (end med 9.7, max 12.7-13.5 us) gates. check_mt 15 sizes 0 failures.
  Rejected (noise-level; not worth the code). v695's own bs256 traces: tail idle 9.1 -> 2.9 us, end phase 3.2 -> 5.7.

## 3. Summary (2026-10-27)
Kept: dev/moe_v695.py (integrated by the planner as dev/moe_v730.py with WP-W's FSW): MOE_WZ_SDEC, MOE_WZ_GDIV32,
MOE_WZ_GDIV32B, MOE_WZ_EHOIST, MOE_KSX8 4 + MOE_SPLW8 0.6f. Over four all-shape jobs vs v655: bs32 +0.9..1.2%,
bs128 +0.7..1.0%, bs256 +0.3..1.1% (mean ~+0.6%); bs1/8/16 identical kernels. check_mt M 1..512 0 failures.
Lessons: (1) the tail split was always worth ~1.2% at bs256; its losses were build costs -- 64-bit / 16-bit
division subroutine calls on the cold startup path and on the starved producer per down item, zeroing before the
gather's cp.asyncs, and a full_k branch inside the epilogue warps' store loop (-1.1% alone); (2) at bs256 startup is
gated by latency chains (gather barrier crossing ~2 us under load, then the cold warm-up of the heavy wgmma loops),
not by bandwidth: neither delaying the weights, prefetching x, 32-bit math nor per-tile readiness moved the first
real stage by more than 0.4 us.
Remaining bs256 levers (not built): the cold warm-up of the n5-n8 wgmma loops (6-8 us vs 3 for n3/n4; gates 25-45%
of the CTAs at the start), the end phase after the split (5.7 us: the split tokens' combine waits for 64 pieces'
red.adds and releases), the ring depth (NS 3 at bs256: turnaround ~4800 cycles vs ~4600 lead).
