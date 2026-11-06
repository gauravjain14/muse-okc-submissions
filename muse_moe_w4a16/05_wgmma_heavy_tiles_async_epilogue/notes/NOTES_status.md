# muse_moe_w4a16 — status and technical notes (2026-10-24)

## Where we stand

- Best scored: **5817.0 µs (v119, 2.312×), rank 6.** First is 5724.2 µs (2.349×): we need about 1.6–2.2% real improvement. The user's bar for the next submission is a projected score under 5.69 ms.
- Judge per shape for v119 (paired ratio vs SGLang):

  | shape | ratio |
  |---|---|
  | bs1 | 2.967 |
  | bs8 | 2.189 |
  | bs16 | 2.074 |
  | bs32 | 1.695 |
  | bs128 | 2.480 |
  | bs256 | 2.697 |

  Repeated runs of the same code vary by about ±1% per shape (bs1 ±3%).
- Current candidate: v130 = v119 plus four changes:
  - direct epilogue for the bs32 config;
  - 2-way tail k-split at bs256 (judge-side lab: +1.4% at bs256);
  - instruction-cache warm-up stage for the gathering configs (+1% at bs128);
  - knobs.

  Projected about 5810 µs from same-machine H200 runs (dev/project.py, anchored on v119 and v122's six judge measurements).

## Problem recap

- Per call: SGLang's routed-expert MoE for M tokens on one EP8 GPU (32 local experts of 256, top-8).
- Weights are NVFP4: E2M1 codes, an E4M3 scale per 16 values, and a per-expert global scale. Each expert is w13 [8192 × 4096] plus w2 [4096 × 4096], about 27 MB with scales.
- Activations are BF16 and must stay BF16. The output is a post-expert RMSNorm per (token, expert) times (1 + w), weighted by the top-k weight and summed.
- Scoring: geomean of paired ratios over bs 1, 16, 32 (decode) and bs 8, 128, 256 (verify), on an H200 with L2 flushed before each call, CUDA graphs.
- Expected experts touched: about 2.5 at bs1, 8.4 at bs8, 12 at bs16, 17 at bs32, 22 at bs128, 25 at bs256. There's one hot expert: about 10 tokens at bs32, 55 at bs128, 100 at bs256.

## Kernel design (current)

- **bs1:** a separate fused one-token kernel.
  - 132 CTAs × 288 threads, with an NS1 = 4-stage TMA (bulk-copy) ring.
  - Work units are 32 intermediates: the 64 w13 rows, then those intermediates' w2 columns. Partials go into fp32 yacc with red.add.v4.
  - The last CTA to arrive (atomic counter) does the combine.
- **bs ≥ 2:** one persistent kernel per call, 132 CTAs × 384 threads.
  - **Warps:** 8 consumer warps (setmaxnreg 216); warp 8 is the producer (bulk copies into a 3–4 stage shared-memory ring); warp 9 is the signaler (red.release publishes); warps 10–11 are idle.
  - **Routing:** every CTA computes it redundantly (bitmaps, rank, pair list). For 32/64-token tiles the activations are gathered tile-major into global memory, followed by a grid-wide count.
  - **Work queue:** gate/up items (tile × 256 or 128 rows × all of K), then down items. The tail tiles' down items are split in k. The first wave is static, then an atomic queue.
  - **Stage:** 32 KB FP4 weights, 4 KB scales, the activation tile (or the intermediate for down).
  - **Decode:** 8 codes per 32-bit word into 4 BF16x2: 5 LOP3, 2 SHF, 2 IMAD (shift-as-multiply on the FMA pipe), 4 HMUL2 (×scale, also fixes the exponent bias). The weights are pre-permuted (lossless repack) to the mma fragment order.
  - **MMA:** mma.sync m16n8k16 for 1–4 token groups of 8; wgmma m64nNk16 (A from registers, B from 128B-swizzled shared memory) for 5–8 groups at bs256.
  - **Epilogue:** stage the accumulators into the item's last ring slot, two CTA barriers, coalesced stores (SwiGLU → BF16 intermediate; or BF16 y plus per-64-row sums of squares; or fp32 partials for split items).
  - **Combine:** per token, once its tiles are done. Dynamic claims, or static for single-group configs.
- **Tile configs** (RW, NT, KS2):

  | batch size | config |
  |---|---|
  | ≤16 | (4,1,1): 2 k-warps |
  | ≤32 | (8,2,1) |
  | ≤128 | (8,4,1) |
  | larger | (8,8,1) |

## Measured facts (H200 via Modal unless noted)

- **HBM ceiling** with this access pattern (bulk-copy ring, no compute): 4.13–4.18 TB/s at 2 MB per CTA; 3.2 TB/s for a 69 MB call including ramp (dev/bw_bench.py). SM clock under load is about 1.76 GHz.
- **Gap to that floor per shape:**

  | shape | gap | bound by |
  |---|---|---|
  | bs1 | about 0–10% | memory |
  | bs8 | about 4% | memory |
  | bs16 | about 4% | memory |
  | bs32 | about 3–5% | memory |
  | bs128 | about 20% | compute |
  | bs256 | about 35% | compute |

- **Stage compute, 8 consumer warps, cycles per 32 KB stage:**

  | token groups | mma.sync | wgmma |
  |---|---|---|
  | 1 | 1,459 | 1,565 |
  | 2 | 1,804 | 1,708 |
  | 4 | 2,441 | 1,963 |
  | 8 | 4,012 | 2,641 |

  - mma.sync costs about 365 cycles per extra group (about 70% of wgmma peak); wgmma about 150 cycles per extra group, but with a higher base.
  - Isolated decode: about 22 cycles per warp-word against 13 instructions. That's issue/latency-bound with 2 warps per SMSP; the line-level profile shows the decode lines as the top cost, `o3` (4 ops) alone at 17.6% of samples.
- **The decode is at its minimum for this BF16 bit placement:** an exhaustive search of all 96 valid nibble layouts finds 7 ALU + 2 IMAD minimum; no 4-rotation tiling exists.
- **Stage boundary** (mbarrier try_wait, lane-0 arrive, then LDS of the next stage): +174 cycles per stage at 1 group in isolation (1,618 vs 1,444).
  - Two stages per wait (one 64 KB wait) gives 1,508.
  - Register-pipelined variants are worse (1,669); in-stage release ties (1,624).
  - In the real kernel, paired stages are −4 to −15% (they halve prefetch depth); hand prefetch is −10%.
- **bs128/256 consumers are compute-bound:**
  - The producer waits for free slots 47–66 µs per call; consumer data waits are only 10–11 µs.
  - bs256 per CTA: decode about 119 µs, extra MMA about 27, boundaries about 14–25, epilogues about 12, waits about 11, startup and tail about 20. Data floor about 165 µs; we're at about 223.
- **Cold instruction cache** at call start: the first stage of a CTA's first item costs 8,000 cycles against 2,800 steady at bs256 (code is fetched from HBM after the L2 flush).
  - A warm-up dummy stage fixes it where the gather leaves idle time (bs128/256): +1% at bs128.
  - It hurts at bs8–32 (no idle window; memory-bound shapes recover anyway).
- **Tail quantization at bs8/16:** items are equal size (8 stages, 288 KB). CTAs that drew one item more finish about 7–8 µs later, partly self-correcting because the last round runs faster. Using fewer CTAs for equal rounds gave only +0.06%.
- **Header layout rule:** the ring mbarriers must be at dynamic shared-memory offset 0. In a sweep of 6 layouts, every one with them at offset 0 performed the same; with them at offset 192, 256 or 704, bs8/16/32 were about 3% slower and bs128 0.7%, wherever the other fields went. That was the v85 regression. Mechanism unknown.

## Tried and rejected (same-machine, shared weights, 2 copies each)

| change | result |
|---|---|
| global-barrier elementwise combine | −3 to −9% at bs32–256 |
| static combine after a barrier | −2% |
| wgmma for 3–4 group tiles | −2 to −3% at bs128: faster wgmma stages, but the kernel's mma.sync stages get 4–11% slower |
| interleaved tile order | 0 |
| cross-stage operand prefetch, 3 variants | −2 to −10% |
| hand-unrolled stage loops | −1.5% |
| balanced item rounds | +0.06% |
| start-up pacing of the first ring fill | −0.6 to −1.7% |
| early weights before the gather | −2% |
| bs8/16 early start | −1.5% |
| direct (staging-free) epilogue for 32/64-token tiles | −1.4 to −4% |
| run-time direct/staged switch | worse |
| 8-way tail split | −2 to −6% |
| one wgmma body for 5–8 groups (code size) | −1.3% |
| decode instruction-mix variants | −0.6 to −3.7% |
| test-wait spin, predicated arrive | mixed |
| ring depth NS=3 for the main kernel | −4.5% at bs8/16 |
| NS=5 | doesn't fit the header layout |

Configs, NS1 and the other knobs were re-swept on shared weights: current values are best.

## Open questions worth asking an expert

1. Any faster Hopper (sm_90a) FP4(E2M1)→BF16 decode than about 13 instructions per 8 values, given BF16 MMA is required (activations BF16)?
   - Hopper has no e2m1 cvt.
   - Is there a trick using cvt.rn.f16x2.e4m3x2, or byte-permute lookup tables, that ends in BF16 cheaply?
   - Or applying the block scale after the MMA instead of with HMUL2?
2. How to hide mbarrier wait/arrive cost (about 174 cycles per stage) in a warp-specialized consumer that isn't tensor-bound but ALU/issue-bound, without losing ring depth?
3. At bs256 the hot expert (about 100 tokens) gets two 64-token tiles, so its weights are streamed and decoded twice. Shared memory (227 KB) caps the stage size. Any known layout that serves about 128 tokens per weight decode on Hopper without blowing registers?
4. Is wgmma with A from registers known to make neighbouring mma.sync code slower (register allocation) when both are in one kernel? Any way to isolate them?
5. bs8/16 tail: what stream-K style fix-up has the lowest latency when the partials must feed a per-row RMSNorm (sum of squares over 4096 columns) before the combine?


## Update 2026-10-24 evening (v138-v148)

Current best code: dev/moe_v148.py (check 5753.7 us, 2.337x; per shape bs1 2.965, bs8 2.181, bs16 2.045, bs32 1.710, bs128 2.570, bs256 2.806). Not submitted (bar: projected < 5690 us).

What changed since v130, all judge-lab A/B (same check, shared weights, 16-24 routings):
- v138 asynchronous epilogue: consumers stage accumulators and hand the slot to warps 10-11, which store, free the slot and signal (bs256 +1.3%; bs128 +0.5% once enabled there in v147).
- v141/v142 no code for wgmma/mma.sync paths the WGMIN choice rules out (+0.7%), WGMIN 3 at bs256, no k-split tail at bs256 (KSX8 1): +1.6%.
- v148 wgmma for 3-4 token groups at bs128 (MOE_WG4 now on): +2.2% at bs128.

Rejected today: L2 prefetch of upcoming stages (-8%: it shares the copy engine the producer waits on), early weights before the gather (-1.3%), L2 sharing of split experts (redundant: HBM reads already equal unique bytes), paired ring slots for bs256 (-1.6%), 32-token tiles at bs256 (-7%), wgmma width merging (-0.3..-0.8%), mixed big/small tile order (0), combined weight+scale copy (0), split mma.sync accumulators (0).

Measured limits (dev/wg_bench.py, dev/roofline.py, ablations): see ROOFLINE.md. Decode and tensor work do not overlap on Hopper even across warpgroups; bs256 is bound by its compute path (199 us compute-only vs 177 data-only vs 221 full).

## Next on my list

1. Variable-size ring slots, so small tiles get deeper rings at bs128/256 (about +0.4%).
2. bs8/16 tail split with the reduction folded into the combine (+0.5–1% if the reduction is cheap).
3. Asynchronous epilogue on the idle warps 10–11 at bs128 (+0.5%).
4. A per-range config for the bs128 hot expert (64-token tile when the hot expert has 33–64 tokens).
