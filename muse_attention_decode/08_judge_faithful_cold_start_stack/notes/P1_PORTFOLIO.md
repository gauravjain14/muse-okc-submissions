# P1 portfolio: batch-1 sliding (d1s xe_r1_1, v1s xe_b1s_8c6), round 3, phase 1 (no GPU used)

Sources: docs/ROUND3.md table (judge us, % per us), dev/xe/NOTES.md traces (jobs 8, 26-29, 39-40), out/xe/job39.txt
(v1s C6 trace), dev/xe/probe.py (C4 skeleton), and my own local SASS of k55's XE_SRC (NVRTC 13.0 + nvdisasm 13; tool
dev/p1/sass.py). Score % per judge us: d1s 2.28, v1s 1.83, d16s 0.45.

## 0. Minimum dependent chain vs what runs now

d1s (trace build, us from the first CTA entry; local nsys 3.39, judge 3.54):

| phase | measured | first-principles floor | what the gap is |
|---|---|---|---|
| entry -> page-table LDG issued | 0.22-0.29 | ~0.10 (one constant-bank miss for the params) | **two serialized constant lines**: SASS `LDC c[0x0][0x140]` + 2 branches (runtime "is cluster" check of `barrier.cluster.arrive`) before the param ULDCs (H1) |
| page-table trip | 0.37 | 0.37 (one cold DRAM trip) | floor |
| pt landed -> K/V issued | 0.25 | ~0.14 (256 half-line requests per SM at ~1 line/clk) | SASS: **all 8 V LDG.64 issue first, the 4 K LDG.128 last** (H2) |
| K/V trip, med / max CTA | 0.78 / 0.97 | ~0.5 + return of 8 KB of K | K data comes back behind V (H2) |
| QK + softmax + PV | 0.35 | ~0.2 (dependent HMMA / SHFL / MUFU chain at 1.75 GHz) | ~256 instructions (4 KB, 31 i-cache lines) run for the first time after K lands (H3) |
| CTA merge + reader + push | 0.32 | ~0.15 | explored (XE diag 4: -0.09 bound) |
| push -> owner recv -> out | 0.33 + 0.42 | ~0.1 + ~0.25 | explored (REF, RDX, OWS, TW, H16, PREW, WS_INC) |

v1s C6 (out/xe/job39.txt trace, medians; local nsys 4.26-4.29, judge 4.70): entry 0.10 (max 0.22) | pt issued 0.32 |
pt landed 0.62 | K/V issued 0.99 (0.37 us for 684 half-line requests per SM: 342 V, then 342 K) | K landed 1.95 (0.96 after
the issue: 43 KB per SM comes back, K last) | P 2.34 | PV 2.53 | merge barrier 2.59 | reader 2.82 | pushed 2.88 | owner
recv 3.16 | out 3.39. The compute warps' post-landing path is **449 straight-line instructions (7.2 KB)** in SASS.

Skeleton gap: XE's C4 probe (same loads + merge + push + owner store) is graph 1.79 us, about 2.45 nsys-equivalent,
against 2.94 nsys for the real kernel with every math / merge phase removed (diag 63). About 0.4 us of the real start /
issue / landing chain is not in the probe. H1-H3 target three concrete pieces of it, each visible in SASS.

## H1. Start path: compile-time cluster dims remove a serialized constant miss before the first DRAM load
- **Bottleneck.** Page-table LDG issued 0.22-0.29 us after entry (d1s, XE job 8 / job27 traces) and 0.22 per CTA on v1s
  (job39: entry 0.10 -> pt issued 0.32). XE job40 measured one cold param fetch at 187-216 cycles (~0.1 us). SASS
  (k55, NVRTC 13): xe_r1_1, xe_b1s_8c6 and xe_dm_8_4_2 all begin the compute-warp path with
  `LDC R2, c[0x0][0x140]; ISETP; @!P0 BRA; UCGABAR_ARV; BRA`. The param ULDCs that feed the page-table address come only
  after that branch, so two constant lines are fetched in series. On xe_r1_1 the pt LDG is instruction 54.
- **Mechanism.** Add `__cluster_dims__(16|6|8, 1, 1)` to the three kernels; the host omits the CLUSTER_DIMENSION launch
  attribute for them and keeps NON_PORTABLE_CLUSTER_SIZE_ALLOWED. Checked locally: the check and both branches disappear,
  UCGABAR_ARV becomes one inline instruction, and the param ULDC moves to instruction 15 (pt LDG at 0x0310 vs 0x0350;
  xe_r1_1 1248 -> 1232 instructions). It also makes XE's old LATEARV idea safe: the if/else that triggered ptxas tail
  duplication in v9/v10 is gone. In job11, LATEARV gave v1s -0.13 bench. Optional arm H1b, same job:
  XE_SMEM=20480 with CLUSTER_SCHEDULING_POLICY_SPREAD instead of the 120 KB padding. This tests the open "large
  dyn-smem / carveout after the flush" hint from PLAN 18:30. Keep it only if the smid trace shows 1 CTA/SM.
- **Expected.** -0.05..-0.10 judge us on each of d1s, v1s and d16s. That is +0.11..0.23% + 0.09..0.18% + 0.02..0.05% =
  **+0.22..0.46%**. H1b: 0 to -0.1 more, P 0.1.
- **P = 0.5.** It fails if the c[0x140] line and the param line are fetched in parallel anyway, for example because the
  front end preloads bank 0.
- **Probe.** One Modal job, shared with H2, ~12-14 min. A builder applies the patch to k55. Run t_xe --quick
  (b1 + B > 1), then nsys 2 copies x 2 passes on d1s, v1s and d16s against 2 k55 copies, each variant under `timeout 120`.
  **Rule:** freeze if the mean nsys gain is >= 0.03 us on any shape and no shape gets >= 0.03 worse. Kill if every shape is
  within +-0.02.

## H2. Issue K before V (SASS shows V first, K last)
- **Bottleneck.** QK needs K first, but ptxas puts every V load ahead of every K load. In xe_r1_1, 8 x LDG.64 (V) sit at
  0x06f0-0x0ad0 and 4 x LDG.128 (K) at 0x0b60-0x0ba0. In xe_b1s_8c6, 12 x LDG.64 sit at 0x08a0-0x0e70 and 6 x LDG.128 at
  0x0f10-0x10d0. xe_dm_8_4_2 pass 0 has the same order. On v1s the issue phase is 0.37 us for 684 half-lines per SM (job39),
  and K's 342 lines go out last. K's 22 KB then returns behind V's 22 KB, and K lands 0.96 us after the issue. XE job14
  probe: d1s K-only lands at 1.09, K+V at 1.25, so issue order is worth up to ~0.15 there. XE never achieved K-first:
  "ptxas issues the V loads first regardless of source order; membar / syncwarp do not hold .nc loads".
- **Mechanism.** Pin the order with a real ordering point. K loads (the speculative 2-pass loop as now) -> `cl_wait()`
  (barrier.cluster.wait, which already exists right after the issue and passes immediately) -> V loads, as plain
  `ld.global` (not `.nc`) from the verified range, so V needs no re-issue on a length miss. ptxas cannot hoist a plain
  load above an acquire barrier, or sink one below it. Fallback: a named bar.sync among the compute warps between K and V,
  which also forces SM-wide K-first. Gate before any GPU time: local SASS shows every K LDG.128 before the first V load,
  with no new waits.
- **Why it doesn't regress.** V now comes back where K comes back today, and PV starts after QK + softmax: +0.2 us on d1s,
  +0.39 on v1s. So V is still early. XC's d16s and XB's d1f K-first losses were DRAM-saturated gathers (8-20 MB). Here it
  is 0.5 MB, and the limit is per-SM issue and return.
- **Expected.** v1s -0.08..-0.18 (+0.15..0.33%), d1s -0.04..-0.08 (+0.09..0.18%), d16s -0.03..-0.06 (+0.01..0.03%).
  Total **+0.25..0.54%**.
- **P = 0.45.** For d1s alone, 0.4.
- **Probe.** The same job as H1: KF alone, CD+KF, controls. **Rule:** keep if v1s nsys is >= 0.05 faster or d1s >= 0.03
  faster, with no regression. Kill if the SASS cannot be made K-first or nsys is within +-0.02.

## H3. Pre-execute the cold post-landing code during the K/V flight (compute-warp PREW)
- **Bottleneck.** The compute warps stall at their first K use, from K/V issued at 0.85 us to K landing at 1.44-1.82
  (d1s). After that they run, for the first time, ~256 instructions on d1s (4 KB: QK, softmax, PV, merge, reader, push)
  and 449 on v1s (7.2 KB). Cold code measurably costs time right after a long wait. XE job5: the owner's first LDS block
  took 0.27 cold vs 0.02 warm, and the reader 0.28 vs 0.21. Owner PREW gave d1s -0.12 end to end. The compute path was
  never pre-warmed. The d1s compute phase is 0.35 us against a ~0.2 us dependent-chain estimate.
- **Mechanism.** A `#pragma unroll 1` 2-pass loop around the Q-LDS -> push code. Pass 0 runs on zero operand registers
  during the flight. The K/V load targets are copied into those operand registers only at the top of pass 1, via a uniform
  `bra`, so pass 0 never waits on a load scoreboard. In pass 0 the st.async is predicated off and the merge smem gets
  garbage that pass 1 overwrites; the mbarrier and the recv buffer are untouched. Same code addresses, so the L0/L1
  i-caches are warm. Costs about +24 registers (82 -> ~106 of 255). Gate: local SASS shows no SEL/MOV of K/V registers
  ahead of pass 0, so ptxas has not if-converted the copy, and no tail duplication.
- **Expected.** d1s -0.08..-0.2 (+0.18..0.46%), v1s -0.05..-0.2 (+0.09..0.37%). The v1s pass 0 has to fit between
  Q-ready (1.02-1.18) and K-land (1.95), less any H2 gain.
- **P = 0.3** on d1s, **0.2** on v1s. Against it: XE job29 found the owner and reader second passes no faster after PREW.
- **Probe.** d1s first, built on the H1/H2 winner: one job ~8-10 min (t_xe quick + nsys 2x2 vs controls), plus ~8 min
  for the v1s port. **Rule:** keep if d1s >= 0.05 faster. Port to v1s only if d1s >= 0.08. Kill if <= 0.02.

## H4. v1s structural: one warpgroup with wgmma holds all 64 rows of a KV head, so no token-amplified gather and no CTA merge
- **Bottleneck.** The C6 layout loads each head's keys 8 times, once per token cluster: 43 KB and 684 requests per SM,
  against d1s's 16 KB and 256. Its K lands at 1.95 vs 1.63 on d1s. It also merges 8 key slices (reader 0.23), and its
  compute is issue-bound: 8 warps x ~449 instructions, about a third of them FP8 -> FP16 cvt/prmt. Cluster co-residency
  (C8 15, C7 15, C16 7) blocks the 128-SM HMMA layouts (V1T died on this).
- **Mechanism.** 2 clusters of 16 (one per KV head), so 16 KB of K/V per SM as on d1s. K/V go to smem by cp.async.
  - QK: FP8 wgmma m64n128k32 x 2, SS mode, Q as E4M3, M = 8 tokens x 8 q-heads.
  - Softmax: on the accumulators, quad-local rows, ex2.f16x2 for the 8192 exps.
  - PV: FP16 wgmma m64n64k16 x 8, P from registers, V converted to FP16 in smem by the prep warps.
  - The warpgroup's accumulator is the CTA partial: no CTA merge.
  - FP16 push of 64 rows to 16 owners (4 rows each), then the existing owner fold and RMSNorm.
- **Expected.** Landing -0.35, merge -0.25, push about +0.2: DSMEM moves ~25-30 GB/s per SM (dev/xd), so 8.7 KB per owner
  is ~0.3 us vs ~0.1 now. Compute about 0. Net **v1s -0.2..-0.4 (+0.37..0.73%)**.
- **P = 0.12.** Big build; push bandwidth; wgmma/ptxas C75xx hazards. k34's wgmma v1s tile phase was 0.8-0.9 us.
- **Probe (kills it first).** Extend probe.py's C stage on the 2 x C16 launch: 64-row FP16 partials from 16 CTAs to 16
  owners, timed from the merge until the owner has folded. ~4 GPU min. **Rule:** stop unless push -> folded is <= 0.30 us.
  Then a timing-only wgmma skeleton (~15 min) must beat C6's nsys by >= 0.4 before a correct build (~3 jobs).

## Ranking by expected score gain per GPU minute

| bet | EV (P x mid gain) | GPU min to decide | EV / min |
|---|---|---|---|
| H1 cluster dims (+ H1b smem arm) | 0.5 x 0.34% = 0.17% | ~7 (shared job) | ~0.024 |
| H2 K-before-V | 0.45 x 0.40% = 0.18% | ~7 (shared job) | ~0.026 |
| H3 compute-path pre-execution | 0.3 x 0.32% + 0.2 x 0.23% = 0.14% | ~16 | ~0.009 |
| H4 v1s wgmma rewrite | 0.12 x 0.55% = 0.07% | 4 to kill, ~55 to win | ~0.0013 |

Honest total: EV about +0.55%; about +1.2% if H1-H3 all land (d1s -0.25, v1s -0.3, d16s -0.12). That is roughly a
quarter of the +5% the team needs. These shapes are near their architectural floor (XE diag 63 = 2.94 nsys on d1s).

**Build first: H2 + H1 together.** They change the same ~40 lines (the compute warps' prologue and issue code in xer1,
xer2 and xem). Both are verified in local SASS before any GPU minute: no LDC c[0x0][0x140] before the first LDG, and K
LDG.128 ahead of V. One <= 15 min job answers both, plus the combination and the smem arm. They have the best EV per
minute and attack measured, SASS-visible defects rather than tuning. H3 goes next, on top of the winner, because H2
shortens the window it must fit in.
