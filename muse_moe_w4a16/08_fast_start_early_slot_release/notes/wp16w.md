# WP-16W: 16 consumer warps for the (4,*,1) kernels (bs 2-16), 2026-10-27

Worker WP-16W. Base dev/moe_v969.py. Files: dev/wp16w_occ.py (probe), dev/wp16w_csrc.py (raw-source NVRTC 13.0.88
compile + SASS stats), dev/wp16w_sass.py (dev/wpc_sass.py that emits INST16 for MOE_W16 configs), dev/moe_v990.py
(K16 kernel build, MOE_W16 mask, off by default = v969 SASS for every kernel), dev/wp16w_run1.sh / run2.sh.
Logs: scratchpad w16/run1b.log, run2.log.

## 1. Probe (dev/wp16w_occ.py, Modal H200, run1b)

v969's exact consumer stage body (stage_compute<NT, NTL, S6> with the S6O 6-bit scales: the probe's A8 loop is 519 SASS
instructions per 32-word stage vs the kernel's 510) on (4,*,1)-layout stages (32 KB weights + 3 KB 6-bit scales +
8 NT x 1040 B activations), 132 CTAs, 256 (and 512) 32 KB-weight stages per CTA. Producer warp: one 35840 B bulk copy
(weights + scales, evict_first) + T x 1 KB x rows (L2) per stage, as the (4,1,1) producer. Designs:

| design | consumer warps | per warp per stage | threads / regs |
|---|---|---|---|
| A8 (control = v969) | 8 (RW 4 x KW 2) | 2 row groups x 256 k of a 512-k stage | 384, consumers 216 / helpers 72 |
| K16 | 16: A8's 8 + a k-half partner each | 2 row groups x 128 k (same stage, same ring) | 640, 104 / 64 |
| R16 | 16: A8's 8 x 2 row groups | 1 row group x 256 k (x fragments read 2x) | 640, 104 / 64 |
| D16 (the brief's design) | 16 (RW 4 x KW 4) | 2 row groups x 256 k of a 1024-k stage (NS 2) | 640, 104 / 64 |

Modes: load (TMA streaming), sync (ring handshake, no copies), free (no waits, resident slots).

Cycles per 32 KB weight stage (warp 0, median CTA) and wall-time ratio vs A8 (x > 1 = faster):

| case | A8 | K16 | R16 | D16 |
|---|---|---|---|---|
| NTL1 free | 1429 | 1468 (x0.993 wall, x0.973 cyc) | 1505 (x0.971 / x0.949) | 1390 (x1.048 / x1.028) |
| NTL1 sync | 1557 | 1594 (x0.983) | 1664 (x0.948) | 1444 (x1.080) |
| NTL1 load | 1776 (1143 ns) | 1784 (x0.992) | 1787 (x0.984) | 1747 (x0.990) |
| NTL1 load, 512 stages | 1804 (1111 ns) | x0.999 | x0.997 | x0.998 |
| NTL2 free | 1787 | 1788 (x1.002) | 1823 (x0.983) | 1735 (x1.033) |
| NTL2 load | 2226 (1385 ns) | 2287 (x0.979) | 2218 (x0.991) | 2050 (x1.054) |
| NT2 stage, NTL1 load | 2030 | 2125 (x0.964) | 2008 (x1.001) | 1986 (x1.007) |

Findings:
- The 1-group loop (all bs8 work, 95% of bs16's) is memory-paced under streaming in this harness (free 882 ns vs load
  1143 ns per stage, C/M ~0.77, 4.14 TB/s). No consumer-side design changes its wall time (x0.98-1.00).
- Splitting a stage over 16 warps (K16, R16) costs more than the extra latency hiding buys: the per-warp fixed work
  per stage (~46-52 SASS instructions: full/empty handshake, scale decode, weight/x LDS, loop and address control) is
  paid twice per stage, +8% instructions (K16 loops 284 + 278 per 32 words vs 519). Issue efficiency rises only enough
  to break even in free mode (x0.993 wall) and they lose in sync / load.
- Pure occupancy (D16: same per-warp body, 4 warps per SMSP) gains ~3% compute (free), ~8% with handshakes (half as many
  handshakes per byte), but under streaming only on 2-group stages (x1.054). On jm seeds 0% of bs8 experts and 5.1% of
  bs16 experts have more than 8 tokens: D16 is worth at most ~0.3% at bs16 and nothing at bs8, against a refactor of the
  stage layout (1024-k stages, NS 2, new WBLK blocks, x rows of 2 KB, 4-way k reduction).
- setmaxnreg moves registers inside the CTA's own pool (SASS: USETMAXREG.TRY_ALLOC.CTAPOOL): the helpers' release must
  cover the consumers' increase. At 640 threads (96 per thread at launch) the limit is consumers 104 / helpers 64
  (4 x 32 = 16 x 8); 112 / 64 deadlocks (the first probe container hung in TRY_ALLOC).

Verdict: NO-GO for the 16-consumer-warp refactor of the (4,*,1) kernels. The old occ_bench "+5% with 16 warps" is
reproduced only as D16's ~3% compute-only gain, which streaming hides on 1-group stages.

## 2. K16 kernel build (dev/moe_v990.py, MOE_W16 mask 1 = (4,1,1), 2 = (4,2,1))

Built before the probe result (to have the real-kernel answer): physical warps 0-11 keep their roles, warps 12-19 are
k-half partners of consumer warps 0-7 (same SMSP). Each stage: warp w decodes 64-k chunks 0-1 (scale words 0-1), its
partner chunks 2-3 (word 2 + spare bits) of the same 2 row groups (stage_half<NT, NTL, KH>). Partners run only the stage
loop; at an item's end they write their partial sums into their own (already read) weight bytes of the item's last
slot, bar.arrive on their SMSP group's named barrier (8 + 2 smsp + item parity, 128 threads: two alternating barriers
are needed because a partner can run one item ahead), and release the slot; warp w bar.syncs, adds the partials and
continues with v969's SEPI epilogue. Ring, stage layout, producer, epilogue warps, combine and shared-memory layout are
unchanged; empty barriers count 16. Default build: every kernel's SASS hash = v969's. MOE_W16=3 SASS: (4,1,1) consumer
loop 272 + partner loop 275 instructions per 32 words (v969 510); (4,2,1) 294 / 271 (+296 / 276); launch 96 registers,
no spills in (4,2,1), (4,1,1) 20 B spill in the routing prologue and the once-per-CTA combine only.

## 3. Real kernel (wp16w_run2, Modal H200, bench2 jm100, B2_SHARE_EQ, v969 x2 controls)

| shape | v969b (control) | W16=1 ((4,1,1) only) | W16=3 (both) |
|---|---|---|---|
| bs8 (62.4 us) | x0.9954 | x0.9617 | x0.9625 |
| bs16 (84.2 us) | x0.9980 | x1.0005 (untouched) | x0.9585 |
| bs32 placebo (111.7 us) | x0.9970 | x0.9994 | x0.9988 |

Correctness of MOE_W16=3: hang1 1-257 ok, check_mt M 1 2 3 4 5 7 8 9 12 15 16 17 (normal + skewed, 3 seeds, 2 calls)
0 failures, wpa_mix 60 interleaved calls 0 failures (worst rel-L2 3.3e-3). REJECTED: -3.8% bs8, -4.2% bs16 (worse than
the probe's -0.8% / -2..-4%: in the kernel the partner hand-off adds a per-item barrier + partial loads on the item-end
path, and the routing / producer code runs at the 96 / 64-register launch split).

## 4. Bare-ring floor (wp16w_run3: consumers wait + release only, no compute)

| stage | bare ring | 8-warp loop under streaming | over the ring |
|---|---|---|---|
| (4,1,1) stage, 1 group, 256 stages/CTA | 1119 ns | 1138 ns | +1.7% |
| same, 512 stages/CTA | 1086 ns | 1106 ns | +1.8% |
| (4,2,1) stage, 1 group | 1258 / 1231 ns | 1260 / 1213 ns | ~0 |
| (4,2,1) stage, 2 groups | 1259 / 1231 ns | 1387 / 1333 ns | +8-10% |

The steady-state 1-group stage loop of the bs8/16 kernels is at the TMA ring's streaming floor: no consumer-side change
(more warps, fewer instructions) can buy more than ~2% there. Only the rare 2-group stages (5% of bs16 experts) are
coupled. The bs8/16 gaps to the floor (P3: full 65.6 vs bare ring 56.6 us at bs8) are outside the stage loop: startup,
item edges (first stage of each item), wave quantization / tail, end chain and combine.

## 5. Conclusion
Do not integrate. The 16-consumer-warp lever has no headroom left on bs8/16: on 1-group stages the loop already runs at
the ring floor, and splitting stages across more warps only adds per-warp overhead. v990 stays as a measured, opt-in
reference (MOE_W16 off = v969).
