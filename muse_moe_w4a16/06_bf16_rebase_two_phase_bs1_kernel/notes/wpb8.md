# WP-B8 findings: two-phase balanced kernel for M = 2..8 tokens (bs8 "verify 1", bs4 test), BF16 operands / FP32 acc

Files: dev/moe_v580.py (= v560 + moe8t, on by default; `MOE8_NO_TWO` = v560's main-kernel path for M 2..8),
dev/wpb8_trace.py (MOE8T_TRACE timeline), dev/wpb8_run*.sh (Modal jobs).

## 0. Routing at bs8 / bs4 (jm seeds, 100 each)

| bs | experts med (p10/p90/max) | pairs med (max) | tokens per expert histogram |
|---|---|---|---|
| 8 | 9 (6 / 11 / 14) | 19 (25) | T=1 600, 2 160, 3 49, 4 20, 5 12, 6 6, 7 9, 8 34 |
| 4 | 6 (5 / 7 / 11) | 8 (15) | T=1 449, 2 95, 3 29, 4 38 |

So 2/3 of the experts carry one token; the stream is ~9 experts x 27.6 MB = ~250 MB per call.

## 1. Design (v580 moe8t)

- moe1t's stage structure unchanged (same repacked wa/wb blocks: 16 rows x K 4096, 36 KB per stage, CTA b owns
  gate/up items 4 b .. 4 b + 3 and down rows [32 b, 32 b + 32) of every local expert, 128 CTAs, NS 4).
- Phase A: the M tokens are the n8 MMA columns (column g = token g, lanes with g >= M hold zeros), so the B fragments
  (the x values of the lane's k slice of token g) are loaded once (16 LDG.128 per lane) and stay resident for the whole
  phase: no reload at expert changes. Tokens an expert does not route are computed and dropped (free in the n8 MMA).
- Partials: each lane leaves its float4 (rows g / g+8 x columns 2 tig, 2 tig + 1) = 4 KB per stage, 2 slots; warp 9
  reduces 8 LDS.128 per lane, applies SwiGLU per (row, token), transposes with 8 shuffles and stores one 16-byte chunk
  per routed (item, token) into ib[pair] (8 KB per pair, pre-swizzled: chunk m of lane (g, tig) at m ^ ((g & 1) << 2 | tig)).
- Publication: warp 9 arrives on a per-expert mbarrier; warp 10 fences once (fence.acq_rel.gpu) for every expert
  finished meanwhile and adds 1 to each one's per-call counter ctr[32 + 32 parity + e] (no epoch tags: half the bytes).
- Phase B: warp 11 polls an expert's counter (ld.acquire, 128 = all CTAs), fence.proxy.async, then bulk-copies the
  expert's T pairs into an 8-slot ring of 8 KB chunks (pair p -> slot p & 7, freed when every consumer has loaded
  pair p - 8). Consumers load column j = pair P[q] + j (conflict-free LDS.128); other columns keep stale values (their
  outputs are dropped). Warp 9 keeps y rows per pair (FP32, 8 KB) and red.adds each pair's sum of squares.
- Combine: monotonic arrival counter as moe1t, then out[t][32 b + r] = sum over t's pairs of
  w * rsqrt(ss / 4096 + 1e-8) * y. CTA 0 zeroes the other parity's sums and expert counters.
- Shared memory 231424 B (bars at 0, tables, 8 KB partials, 8 KB y rows, 64 KB chunk ring, 4 x 36 KB ring).
- SASS (local nvcc 12.1): moe8t_4 116 registers, no spills; stage loop = moe1t's (32 HMMA, 121 HMUL2, 150 LOP3).

## 2. Runs 1-4 (Modal H200, jm seeds, bench2 100 reps x 2 passes, same-job v560 controls)

| build | bs8 us (pass 1 / pass 2) | v560 same job | ratio |
|---|---|---|---|
| v580 counters + fence | 75.1, 75.1 / 74.9, 74.9 | 64.8-65.1 | x0.869-0.875 |
| v582 sentinel fetch (flag-in-data, no fence) | 73.9, 73.2 / 74.3, 73.1 | 65.4-66.1 | x0.884-0.894 |
| v582 + WARM / ER+DP / WARM+ER+DP | 76.9 / 74.7 / 79.7 | | all slower (WARM+ER+DP also failed one check_mt case) |
| v583 x by bulk copy + rotated wa8 | 72.4, 72.0 / 71.8, 71.5 | 63.1-64.6 | x0.885-0.891 |
| v584 (x load inside the loop) / + WARM | 74.4 / 74.0; 75.2, 75.1 / 74.7, 74.3 | 65.3-65.5 | x0.871-0.887 |

bs4 (fresh routings, 60 reps): v583 51.1 vs v560 46.9 us (x0.943). check_mt M 2..8 normal+skewed: 0 failures for every
default build, rel-L2 2.5e-3 (same as the main kernel's).

What the traces showed (per CTA, median over CTAs):
1. v580's per-expert fence (fence.acq_rel.gpu on warp 10) completes only when the SM's in-flight TMA loads drain: every
   expert got published at the A->B boundary when the ring stalled (4.2-4.8 us stall). Fixed in v582 by sentinel
   words (0xFFFFFFFF cannot be two cvt.rn BF16 values), copy + verify + retry on warp 11, refill of the other parity's
   buffer by warp 10: no counter, no fence (the same idea as moe1t's epoch tags, without doubling the bytes).
2. 128 CTAs x 4096 LDG.128 of the same 64 KB of x delayed everything at startup (first copy issued 3.8 us after entry,
   x in registers 10.5 us, stage 0 ready ~11 us). v583: M bulk copies of 8 KB at entry into rows of 8 KB + 16 B in the
   (not yet used) chunk ring; a lane's 16-k blocks rotated by its tig in the repack (wa8: block i of lane (g, tig) =
   k-block (i + tig) & 7) make the LDS conflict-free -> first copy 0.4 us, x 0.67 us, stage 0 ready 2.2 us.
3. Stage compute 0.86-0.90 us at an SM clock of 1.70-1.76 GHz (~1500 cycles, the decode floor); the slowest consumer
   warp drifts 0.035 us per stage behind warp 0 (up to ~2 us, bounded by the 4-slot ring) and sets the stage interval
   (1.1-1.18 us vs the 1.0 us HBM interval): phase A is consumer-bound by ~10-15%, unlike moe1t at bs1. Warp 9 itself
   is not slow (0.16 us after the slowest consumer, 0.10 us processing per stage).
4. Phase-B expert starts cost +0.3 us each; T = 8 experts wait 2-3 us for their 8-pair fetch (the chunk ring must
   drain first).
5. End: last stage -> output ~5.5 us (moe1t 3.5): warp 9's lag behind the slowest warp, the arrival release behind
   ~19 x 128-256 red.adds on one or two L2 lines, then the sums' reload.

## 3. Run 5 (v586 switches) and close-out

| build (dev/moe_v586.py defines) | bs8 us pass 1 / 2 | v560 same job |
|---|---|---|
| none | 73.2 / 73.3 | 65.4, 65.5 / 65.0, 65.5 |
| MOE8T_HSUS | 72.6 / 73.2 | |
| MOE8T_HSUS,MOE8T_REL1 (best) | 72.2 / 71.7 (x0.905-0.910) | |
| + MOE8T_SSPAD (2 copies) | 72.2, 72.3 / 72.1, 72.6 | |
| + MOE8T_EPI2 | 73.7 / 72.6 | |

check_mt M 2..8 normal + skewed: 0 failures for every build. The slowest-warp drift stays at ~0.03 us per stage with
HSUS (so it is not warp 9's spins); EPI2 halves it in phase A but does not shorten the call; SSPAD is neutral.

Per-bucket budget, v586 HSUS+REL1+SSPAD, seed 71516 (ne 8, 18 pairs), per CTA from its own entry stamp:

| bucket | moe8t | floor / reference | what it would take |
|---|---|---|---|
| entry -> first weight copy issued | 4.7 us | moe1t ~0.5 us | find the cause (ids load behind the 64 KB x copy? cold code / TLB?): issue the routing loads first, x after the first weight copy or by cluster multicast; ~3-4 us |
| first copy -> stage 0 ready | 1.8 us | ~1.7 us (plan_start) | - |
| stage 0 compute (cold code) | 3.3 us | 0.9 us | WARM-style warm-up that does not cost the loop (WARM -1..-1.5% in v584): ~2 us |
| phase A, 32 stages | 34.3 us (1.07/stage) | 32.0 us (1.0 us HBM interval) | slowest-warp drift: ~2 us |
| phase B, 16 stages | 19.5 us (1.22/stage) | 16.0 us | expert-start reload +0.3 us x 8 (WP-B1's BSPLIT lost at bs1), T = 8 fetch stalls (descending-T expert order, MOE8T_ORD, not built): ~3 us |
| last stage -> output | 5.5 us | moe1t 3.5 us | ~2 us |
| total | ~68.4 us (bench2 jm median 72) | v560 main kernel 65.0-65.5 | ~10-12 us of edges to remove to reach ~60 us |

Conclusion: rejected for now. At bs8 moe8t's stage costs ~1500 cycles at 1.70-1.76 GHz, the same as the main kernel's
(4,1,1) stage; only the main kernel's ~4 us end spread is there to win, and moe8t's own edges cost more. bs16 with two
n8 groups: 128 resident B-fragment registers (over the 168-register cap at 384 threads unless the helpers are merged),
two MMAs per k step whenever an expert's tokens span both halves (31% of bs16 experts), and T > 8 experts (5%) need 16
chunk slots (128 KB, no room): phase A would be more consumer-bound than at bs8, so not feasible as designed.

## 4. Reopened on v752 (planner, after the v730 ncu profile: bs8 SMs idle 19%) — runs 6-8

Files: dev/moe_v780.py (v752 + moe8t, HSUS + REL1 defaults; FENT / PEL / WARM switches), v781 (+ORD, BPRE), v782
(+FENT2), v783 (+ARR8), v784 (+ER2, HWARM), v785 (BAKED best, moe8t opt-in via MOE8_TWO); dev/wpb8_sass.py (entry path
and waterfall report from a dev/wpc_sass.py dump, judge NVRTC 13). Bench: bench2 jm seeds 100 reps x 2, v752 x2 same job.

| build | bs8 us (pass 1 / 2) | v752 same job | vs v752 |
|---|---|---|---|
| v780 (v586 moe8t on v752) | 72.3 / 72.4 | 64.2-64.5 | x0.891 |
| + FENT | 68.5 / 69.3; 68.9 / 69.1 | | x0.916-0.938 |
| + FENT + PEL | 70.0, 69.7 / 69.5, 69.9 | | x0.920-0.924 |
| + FENT + FENT2 | 68.7 / 69.0; 69.1 / 69.1 | 62.9-64.5 | x0.920-0.932 |
| **+ FENT + FENT2 + ARR8 (best, v785 opt-in)** | **68.5 / 68.6** | 63.7-64.5 | **x0.940-0.947** |
| FENT2 + EPI2 / + ORD / + BPRE / + ORD + BPRE | 68.8, 69.7 / 71.0, 70.7 / 70.9, 71.3 / 71.0, 71.7 | | x0.876-0.913 |
| FENT2 + HWARM / + ER2 / ER2+HWARM+ARR8 / ER2+EPI2+ARR8 | 70.1, 70.2 / 71.2, 70.3 / 73.1, 73.3 / 70.2, 70.0 | | x0.879-0.919 |

check_mt M 2..8 (normal + skewed, 3 seeds, 2 calls): 0 failures for every build listed.

Edge by edge (per CTA from its own entry stamp, ne 8 seed 71516 unless noted):
1. Entry -> first weight copy 4.7 us: SASS showed ~96 lane-indexed mbarrier inits on warp 0 compiled to serialized
   R2UR / FENCE.VIEW.ASYNC.S / SYNCS.EXCH / BRA.U.ANY waterfall iterations (10 fence sites, 13 waterfalls before the
   first BAR.SYNC) and the ld.global.nc id loads sunk below the barrier. FENT (ids as volatile loads first, inits spread
   over the 12 warps' elected lanes with uniform addresses) -> 2.1-2.4 us; FENT2 (x at entry, elected lane, rows rotated
   per CTA) keeps x at 2.7 us (x behind the weights had landed at 5.65 us: 128 CTAs x the same 64 KB take ~3.5 us).
   Gain FENT x0.891 -> x0.93.
2. Phase-A drift: the slowest consumers are the SMSP1 warps (w1/w5, beside warp 9's epilogue: +0.9-1.5 us by the end
   of phase A); w2/w6 (beside the idle warp 10) run ahead. PEL (elected-lane producer / fetcher copies, 0 waterfalls in
   the kernel) -1%; EPI2 balances SMSP1 / SMSP2 (lags 0.5-0.96 on both) but the interval stays 1.088 us; ER2 (slot
   released after the loads) makes it 1.136: the interval is the stage's decode + n8-MMA floor (~1500 cycles at
   1.70-1.76 GHz, the main kernel's (4,1,1) stage cost) plus helper interference, not ring lookahead.
3. Cold stage 0 (2.9-3.3 us compute vs 0.9): WARM (dry iteration, x loaded inside the loop) -2.5..-3%, HWARM (warp 10
   runs the dry iteration) -1.5%: both slow the loop or the start more than they save.
4. End (c-done -> out ~5.3 us): ARR8 (8 arrival counters) +0.6-1.5%. Remaining: arrive -> seen ~2.1 us, seen -> out ~1.7.
5. Phase-B transitions (+0.26 us per expert start; T = 8 experts 2.5-3.4 us): ORD (descending-T order) costs ~1.2 us of
   ranking before the first copy and moves the 8-pair fetch earlier but not off the path (-3%); BPRE (next expert's
   fragments loaded inside the previous stage, a second stage body) raises phase-B compute 0.96 -> 1.07 us (-3..-4%).

Budget of the best build (FENT2 trace, ne 8): entry -> first copy 2.4 us, stage 0 ready 4.2, cold stage 0 2.9, phase A
32 x 1.088 us, phase B 16 x 1.2 us, end 5.3 us -> ~65.7 us per CTA vs ~62 for v752 at that routing. Closing it needs the
consumer stage below ~1.0 us (helper-free SMSPs) plus ~4 us of the edges; none of the five attempted edge fixes beyond
FENT / FENT2 / ARR8 paid. moe8t stays rejected; v785 keeps it as an opt-in (MOE8_TWO).
