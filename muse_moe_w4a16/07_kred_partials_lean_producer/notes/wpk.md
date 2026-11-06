# WP-K findings: wpq.md levers #4 (bs128 split-partial fixup) and #5 (bs32 gather-free start)

Worker K, 2026-10-27. Base: dev/moe_v567.py (judge check 5523.8 us). Files: dev/moe_v650.py (MOE_KFIX),
dev/moe_v651.py (MOE_XDIR), dev/moe_v652.py (MOE_KRED), dev/wpk_cc.py (local nvcc compile check),
dev/wpk_sass.py (NVRTC / nvcc SASS summary: per kernel instructions, spills, and every HMMA/HGMMA loop's size and
opcode hash, so two builds' hot loops can be compared), dev/wpk_run1.sh, dev/wpk_run2.sh. Logs: scratchpad
wpk_run1.log, wpk_run2.log. Every mechanism is behind a define; with the define off each file's source is
token-identical to v567 (cubin byte-identical with local nvcc, all main-kernel SASS identical on Modal NVRTC).

## Codegen note (applies to every A/B here)
Declaring an unused `constexpr` local in moe_body (e.g. `constexpr int KSXC = KFIX ? KS2 : KSX;`) re-schedules the
(4,2,1) bs16 kernel (same instruction count, different order) even when the value equals an existing one. All my
switches therefore use macros (`#define MOE_KFIXV (...)`) inside `#ifdef`, which leaves the other configs' kernels
byte-identical. Recommend the same for every worker's define.

## #4 attempt 1: v650 MOE_KFIX (last-arriving piece sums the partials) -- REJECTED
- Mechanism: (8,4,1) tail pieces store fp32 partials as before; epilogue warps 10-11 bar.sync(5, 64), one thread does
  atom.add.acq_rel.gpu on ctr[set + 6144 + tile * 16 + rb] (zeroed by MOE_PAR's other-set clear: no reset, no epoch),
  bar.red.or broadcasts "last"; the last piece reads the 4 partials (L2, 2 tokens per pass) and writes BF16 y; only
  it posts the down signal (combine waits for NRB2 per tile and reads one BF16 row per pair). Combine pair loop kept
  at `#pragma unroll 1` (the simpler body was otherwise unrolled 4x: +536 instructions).
- wpk_run1 (jm 100 reps, v567 x2): bs128 x0.9852 / x0.9857 (control x0.9990); bs32 x0.9989 / x1.0006 (control
  x0.9987); other shapes are the same kernels (x0.995..1.001). check_mt 15 sizes 0 failures (max rel-L2 3.30e-3).
- SASS (Modal NVRTC 13): (8,4,1) 7608 vs 7792 instructions, 12 B spills, stage loops same sizes, re-scheduled.
- Reading: the release-atomic is synchronous (a MEMBAR.GPU before the ATOM) and sits on the epilogue warps once per
  piece, and the fixer adds atomic + L2 round trips to the end of the call: the same chain that sank v189/v190.
  Moved on to a fence-free form (attempt 2).

## #4 attempt 2: v652 MOE_KRED (pieces add into one fp32 row) -- WIN in run 2, confirmation in run 3
- Mechanism (fence-free): the (8,4,1) tail pieces `red.relaxed.gpu.global.add.v4.f32` their scaled fp32 rows into
  ypart slice 0 (one accumulator row per split pair) instead of storing 4 slices; the split pairs' rows
  [psplit, P) are zeroed by the consumers during the gather (this CTA's slice, < 4 KB at bs128), which the gather's
  consumer barrier + red.release(CB[3]) orders before every piece (the pieces' CTAs acquired CB[3], or cnt1 of a
  G1 item whose CTA did); signals and counts unchanged (want = 16 x 4), so the cnt2 acquire covers the adds; the
  combine reads one fp32 row per split pair (MOE_KSMC = 1) instead of 4. FP32 accumulation throughout; the 4 pieces'
  add order varies per call (FP32 rounding only, ~1e-7 relative).
- wpk_run2 (jm 100 reps, v567 x2, one job): bs32 x1.0068 / x1.0128 (control x0.9979), bs128 x1.0046 / x1.0072
  (control x0.9987); bs1/8/16/256 run byte-identical kernels: x0.9955..1.0033 (control spread up to 0.5% there).
  check_mt 15 sizes 0 failures, rel-L2 = v567's. NVRTC: (8,4,1) +40 instructions, no spills (stage loops re-scheduled,
  same sizes). Local nvcc even kept the MW 2/4 stage loops hash-identical to v567.
- wpk_run3 (confirmation, same protocol): bs32 x1.0107 / x1.0080 (control x1.0003), bs128 x1.0022 / x1.0041
  (control x0.9988). Over both jobs: bs32 +0.8..1.3%, bs128 +0.3..0.8%, other shapes byte-identical kernels.
  probe (5 routings): end bs32 110.1 vs 110.3 us, bs128 167.9 vs 168.2. KEPT.
- MOE_SPLW 1.5 with KRED (1.5 waves of pieces): bs32 x1.0027, bs128 x1.0020 (one copy): no better than 1.0, rejected.

## #5: v651 MOE_XDIR (first item's x rows straight from x) -- marginal in run 2
- Mechanism: in the (8,4,1) gathering config, a CTA's first item (item == blockIdx.x) that is a gate/up item of an
  mma.sync tile takes its x rows by per-row 512-B bulk copies from x (as bs 2-16 do) after the pair list (mbarrier
  plr, arrived by the consumers before they gather); its expected bytes n * 512; later items keep the gathered rows.
- wpk_run2: bs32 x1.0031 / x1.0016 (control x0.9979: +0.2..0.4% vs the control mean), bs128 x1.0006 / x1.0005.
- Suspect: consumer thread 0's gather red.release (MEMBAR.GPU) now waits behind the 3 extra stages of TMA loads that
  XDIR issues at ~2.5 us, blocking consumer warp 0 (P9's mechanism, now on the consumer). v654 MOE_XREL moves that
  release to the signaler warp (idle until the first item ends): consumers arrive rdy after their gather barrier,
  warp 9 waits rdy and releases CB[3] (cumulative via the CTA-scope mbarrier).
- wpk_run3: v653 (XDIR + KRED) bs32 x1.0150 / x1.0142 vs KRED alone x1.0107 / x1.0080 in the same job (XDIR adds
  +0.4..0.6% on top), bs128 x1.0021 / x1.0017 (= KRED). check_mt 15 sizes 0 failures.
- MOE_PROBE (v567 vs v651 XDIR, bs32, 5 routings): pair list 2.59 / 2.66 us, gather done med 2.98 / 3.01, first
  weights issued 4.80 / 5.31 (the producer's cold code, not the gather), first data at consumer thread 0 med
  7.04 / 7.17. XDIR does not move the consumers' first stage: thread 0 runs gather -> red.release(CB[3]) (MEMBAR) ->
  rdy -> setmaxnreg.inc -> the cold warm-up stage -> slot 1, which ends ~7 us either way. What XDIR buys is a full
  ring at that point (stages 1-3 loaded instead of waiting for the all-CTA barrier).

## Run 4: MOE_XREL and dropping the (8,4,1) warm-up stage (dev/moe_v654.py)
- v654 MOE_XREL: consumers arrive rdy after their gather barrier; the signaler warp (lane 0) waits rdy and does the
  red.release(CB[3]); consumer thread 0 no longer executes a gpu-scope release at startup.
- MOE_GATHER_MIN0=5 (an existing switch) turns the instruction-cache warm-up stage off for NT 4 only, i.e. the
  (8,4,1) kernels; (8,8,1) keeps it (SASS unchanged). With XDIR the first item's ring is full by the time the
  consumers finish the gather, so the warm-up's idle-consumer window is gone.
- wpk_run4 (jm 100 reps, v567 x2; control x0.9981 bs32 / x1.0001 bs128):
  | variant | bs32 | bs128 |
  |---|---|---|
  | v653 XDIR+KRED | x1.0139 | x1.0034 |
  | v654 XDIR+XREL+KRED | x1.0129 / x1.0126 | x1.0052 / x1.0057 |
  | + GATHER_MIN0=5 (no warm-up) | **x1.0186 / x1.0207** | **x1.0068 / x1.0080** |
  bs1/8/16/256 identical kernels (x0.994..1.003 vs control x0.992..1.003). check_mt 15 sizes (and the no-warm-up
  set on 7 sizes) 0 failures. Probe (5 routings): bs32 first data med 6.27 us without the warm-up vs 6.85 with it.
- run 5 decomposes it: no-warm-up alone on v567, KRED + no-warm-up, XDIR + KRED + no-warm-up (no XREL), full set x2.

## Run 5: decomposition (jm 100 reps, v567 x2; control x0.9988 bs32 / x0.9976 bs128)
| variant | bs32 | bs128 |
|---|---|---|
| v567 + GATHER_MIN0=5 (no warm-up only) | x1.0060 | x0.9963 |
| KRED + no warm-up | x1.0188 | x1.0007 |
| XDIR + KRED + no warm-up (no XREL) | x1.0152 | x1.0017 |
| XDIR + XREL + KRED + no warm-up | **x1.0171 / x1.0209** | **x1.0046 / x1.0064** |
The full set is the only one that wins on both shapes; XREL is the bs128 part, the warm-up removal pays only with a
ring that is already full when the consumers start. Over runs 4+5 (4 copies): bs32 +1.8..2.2%, bs128 +0.6..0.9%
vs the control means; bs1/8/16/256 kernels byte-identical to v567 (NVRTC), measured within the control spread.
Projected geomean +0.45% (~5499 us from 5523.8). Baked as dev/moe_v655.py (MOE_NO_WPK = v567).

## Run 6: baked dev/moe_v655.py validated (integration candidate)
- NVRTC SASS: v655 == v654 + (MOE_XDIR, MOE_XREL, MOE_KRED, MOE_GATHER_MIN0=5) for every kernel; v655 with
  MOE_NO_WPK == v567 for every kernel; only moe_8_4_1_4_{1,2,4} differ from v567.
- hang1 7 shapes ok; check_mt 15 sizes normal + skewed, 3 seeds, 2 calls: 0 failures, max rel-L2 = v567's.
- bench2 jm 100 reps, v567 x2 vs v655 x2: bs32 x1.0199 / x1.0215 (control x0.9990), bs128 x1.0040 / x1.0049
  (control x0.9971). bs1 / bs8 / bs16 / bs256 run byte-identical kernels: 0.998/0.993, 0.991/0.995, 0.999/1.003,
  0.996/0.999 vs controls 0.996, 0.998, 1.001, 0.997 (bs8's copies sit ~0.4% under the control mean in this job,
  +0.0/-0.3% in runs 4/5: noise on an identical binary).
- Recommendation: integrate v655's four switches (or take dev/moe_v655.py whole: its main kernels for other buckets
  are v567's). Six copies over three jobs: bs32 +1.8..2.2%, bs128 +0.5..0.9%, geomean ~+0.45%.

## Not pursued / closed
- KFIX-style fixups with a returned atomic (any gpu-scope release on the epilogue path): measured -1.45% bs128.
- XDIR for wgmma first items: the wgmma x layout permutes k inside each 64-k chunk (gather does it), so direct row
  copies cannot produce it; bs128's first wave is mostly wgmma tiles.
- MOE_SPLW 1.5 with KRED: no gain.

## Follow-up 1 (planner, after v655's okc check 5427.8 us): bs256 (8,8,1) -- all rejected
- XREL: nothing to port (the GCP gather already releases CB[3] on warp 9, off the consumers). XDIR: not applicable
  (GCP uses ring slots 1..NS-1's x regions as gather scratch until the release; bs256's first wave is wgmma tiles,
  whose k-permuted x layout only the gather produces).
- dev/moe_v656.py: KRED for the GCP config (split pairs' rows zeroed by warps 9-11 before their release, after a
  96-thread barrier) + MOE_DSPL8 f (DSPL's last-round rule for the 64-token config only). Defaults = v655 SASS.
- wpk_run7 (jm 100 reps, v655 x2, bs256 control x0.9967): KSX8=2 (one wave) x0.9914, KSX8=2 last round x0.9846,
  KSX8=4 last round x0.9846, warm-up off (GATHER_MIN0=9) x0.9839; check_mt M 129..512 0 failures for all. The tail
  split stays a loss even with the cheap partial path (bound was +0.45% with free partials: per-stage tail floor
  ~0.85 us plus the split code's schedule).

## Follow-up 2: bs8/bs16 SEPI configs, KRED-style 2-way split of the last round -- rejected
- dev/moe_v657.py MOE_KREDS (+ MOE_KSXK2 2, MOE_DSPLK2 f = DSPL's last-round rule for the two-k-warp configs only).
  The SEPI configs are not gathering configs, so there is no in-call grid release to order a zeroing: the halves
  red.add into a persistent self-cleaning fp32 buffer (state["ykz"], zero at prepare; the slow combine zeroes each
  split row after reading it; the kernel boundary orders that before the next call).
- Found a latent bug for any KSXK2 > 0 build: with the early start (bs <= 16) warps 8-11 skip the routing and keep
  tsplit = 0, so the SEPI epilogue warps treated every down item as split (bs2 rel-L2 0.91, NaN). Fixed (warps
  10-11 wait rtd and read scal[4] when KSX > KS2); the default build is unaffected (KSX == KS2 never reads it).
- wpk_run8b (jm 100 reps, v655 x2): bs8 x0.9314 / x0.9342, bs16 x0.9346 / x0.9365 (DSPLK2 0.5: x0.9314 / x0.9399;
  plain partials: x0.9424 / x0.9332); check_mt 16 sizes 0 failures. NVRTC: (4,1,1) 5712 -> 7592 instructions: the
  slow combine body (dead in v655, where every down item is full k) becomes live, plus the split decode. The loss is
  the same with half the splits, so it is the code / combine path, not the split itself. Not retried on WP-C's
  MOE_NI_RDC (not cleared by the planner).
