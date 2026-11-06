# WP-R: published techniques for the H200 NVFP4 W4A16 MoE kernel (research only)

Date: 2026-10-26. No GPU work was run. Context taken from the team memory (okc-moe-findings): decode is issue-bound
at 9 int + 4 HMUL2 per 8 weights; removing the HMUL2s entirely saves 17-23% of stage time; cross-CTA signals cost
4-6 us while the kernel streams; bs8/16 lose 4.6-7.2% to wave quantization; the bs256 tail spreads 17 us; tensor-core
use caps the clock at ~1.79 GHz, and bs256 is also capped at 700 W.

Legend: **[NEW]** = not in the team notes; **[DONE]** = we already do this or have measured it; **[TRIED]** = the team
tried something close and lost.

---

## Ranked list (most promising first)

### 1. A fused slice kernel for bs8/16, with independent units and no G1->G2 dependency [NEW for bs>1]
- **Source:** HazyResearch "Look Ma, no bubbles" megakernel, https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles.
  Its MLP intermediate is handed off in chunks with per-chunk counters, so the down projection waits only for its own
  chunk. Our own one-token kernel (v82+) is the extreme form: units of 32 intermediates (64 w13 rows plus their w2
  columns), `red.add.v4` into yacc, and the last CTA combines.
- **What it does:** each unit is G1 rows -> SwiGLU -> the matching w2 column slice -> an fp32 red.add of a
  `T x 4096` partial. The units are independent, so a dynamic queue of equal-sized units has a tail of at most one
  small unit, and the 3-5 us G1->G2 waits per wave disappear.
- **Applies to:** our exact format and shapes. At bs8/16 the judge has ~9-12.5 experts with ~1-2 tokens each.
- **Cost:** L2 red traffic is about `(4096/U) * 4096 * 4 B * T` per expert. With U = 32 that is ~2 MB per token per
  expert (~20 MB at bs8, against 243 MB of weights). U = 64-128 cuts it 2-4x but puts quantization back (a 64-unit is
  ~12 us of one SM's bandwidth). Measure L2 `red.v4.f32` throughput under streaming before committing.
- **Estimate:** bs8/16 time is roughly 4.7 + waves x 8.8 + 3 us. Removing the dependency waits and the
  up-to-one-item tail could give **4-8% on bs8/16**, and maybe bs32. This assumes the red traffic does not saturate L2.

### 2. DSMEM handoff inside a 2-CTA cluster for tail K-splits and fixup [NEW]
- **Sources:**
  - PTX ISA `st.async` / `red.async` with `.mbarrier::complete_tx::bytes` on `shared::cluster` (sm_90),
    https://docs.nvidia.com/cuda/parallel-thread-execution/
  - Luo et al., "Benchmarking and Dissecting the Nvidia Hopper GPU Architecture", https://arxiv.org/abs/2402.13499:
    SM-to-SM (DSMEM) latency 180 cycles vs 263 for L2; 3.27 TB/s at cluster size 2, 2.65 TB/s at size 4.
- **What it does:** the partner CTA pushes its fp32 partial accumulators into the owner's smem with `st.async` and
  completes the owner's mbarrier (tx bytes). The owner waits on its local mbarrier.
  - No GPU-scope fence.
  - No L2 round trip. Traffic goes over the GPC SM-to-SM network, not the L2/HBM path that is clogged with TMA
    streams.
- **Why it fits us:** our splits lost to exactly these costs: `__threadfence` ~4 us per CTA, `red.release` blocking
  4.4 us, publish->seen ~6 us under streaming.
- **Use:** pair the last-wave G2 items (bs256 tail 205-222 us; bs128/32 split items) and the bs8/16 last-wave items
  between the two CTAs of a cluster.
- **Risks:** `cudaOccupancyMaxActiveClusters` for cluster size 2 at our smem/regs must still give 66 clusters on 132
  SMs. A persistent kernel can only pair cluster partners, not arbitrary CTAs.
- **Estimate:** bs256 2-5% (tail), bs8/16 1-3%. It also makes the item-1 fixup cheaper if units are split per cluster.

### 3. Group scale applied after the MMA: FP16-accumulator HMMA + HFMA2 for 1-token-group tiles [NEW for A16]
- **Sources:**
  - Humming `include/humming/arith/mainloop_arith.cuh`, `may_apply_as_and_bs_on_mma_c` (https://github.com/inclusionAI/humming).
    With an f16 accumulator, each MMA writes a fresh `c0`, then `c1 = __hfma2(c0, bs_pair, c1); c0 = 0`. Humming uses
    this only for W4A8; with 16-bit activations it pre-scales like we do.
  - DeepSeek-V3 report, section 3.3.2 (https://arxiv.org/abs/2412.19437): promote low-precision partials to FP32 at
    an interval and apply the scale at promotion.
- **Instruction effect per 8 weights on an NT=1 (n8) tile:**

  | | FMA-pipe ops per 8 weights |
  |---|---|
  | Today (pre-scale) | 4 HMUL2 |
  | Post-scale, f16 accumulator | 2 HFMA2 (`mma.sync m16n8k16 .f16` with C = RZ; D is 2 half2 regs, rows r / r+8; HFMA2 `.H0_H0` / `.H1_H1` selectors pick the row scale from one (s_r, s_r+8) register) |
  | Promotion to fp32 every 4 groups | ~1 more (cvt + FADD) |

  Net: about one FMA-pipe op saved per 8 weights. Post-scale with an f32 accumulator saves nothing (4 FFMA). For
  NT >= 2 it costs more, so use it only for NT=1. The FMA pipe also competes with the tensor core (XOR in place of
  HMUL2 saved ~200 cycles at NTL8).
- **Hard constraint:** the 2^14 E2M1 bias currently lives in the scale. With post-scaling, the raw decoded values are
  x 2^-14, so the bias must move into x.
  - Per-token power-of-two prescale of x so that `16 * 6 * max|x'| < 65504`, and partial sums of <= 4 groups
    (divide by a per-row max scale) do not overflow fp16.
  - Check with dev/err_budget.py. Expected error is ~5e-4 to 1e-3, inside the 1.5e-2 tolerance.
- **Estimate:** 2-5% of NT=1 stage time, which covers bs8-32 and more than half of the bs128/256 work. Medium risk:
  accuracy, register pressure, and the gather has to emit scaled x.

### 4. Data-dependent tensor-core power at bs256 (keep tile padding exactly zero, avoid padded math) [NEW]
- **Sources:**
  - Gregersen, Patel, Choukse, "Input-Dependent Power Usage in GPUs" (https://arxiv.org/abs/2409.18324).
    - GEMM power changes by up to ~38-40% with the same shapes.
    - Sparsity/zeros, sorted values, high bit similarity, and zeroed LSB/MSB bits all lower power. FP16 tensor-core
      math draws the most of the dtypes tested.
    - Measured on A100; trends also held on H100.
  - Horace He, "Strangely, matrix multiplications on GPUs run faster when given predictable data"
    (https://www.thonking.ai/p/strangely-matrix-multiplications): zeros run 15% faster than randn on a power-capped
    A100 (295 vs 257 TFLOPS). Nvidia's MLPerf notes describe moving power from L2 to the SMs.
- **Use at bs256** (700 W, 1.68 GHz vs the 1.79 GHz tensor cap):
  - Make sure every padded token column/row in mma.sync B fragments and wgmma smem tiles is exact 0.0, not stale ring
    data or duplicated tokens.
  - Prefer narrower N when the padding exceeds ~25%.
  - Optionally sort tokens within an expert tile by norm, which keeps neighbouring B rows similar.
- **Estimate:** at most ~6% at bs256 (the gap between 1.68 and 1.79 GHz). Realistic 0-2%. Cheap to check.

### 5. Chord: 2 resident CTAs per SM for small-token tiles, plus a semi-static token-tile guard [NEW / partly TRIED]
- **Source:** Novita "Chord" (https://github.com/novitalabs/chord, docs/design.md; blog
  https://vllm.ai/blog/2026-09-15-novita-chord-w4a16-moe).
  - H200 EP8 decode uses swap-AB `mma.sync m16n8k16` with weights on M, block-M 8, and 3-4 CTAs/SM.
  - The masked grouped path runs 2 CTAs/SM when `block_m <= 24`. This "hides the single TMA producer's latency" and is
    what closed their 8-tokens-per-expert gap; none of 91 single-CTA layouts matched it.
  - Semi-static token tiles: tile j=0 is unconditional and only j>=1 is guarded. This is -7% cycles vs fully static,
    and 186 vs 216 us vs fully dynamic.
- **Fit:** our producer is starved by its SMSP's consumers (0.4-1.7 us per stage issue).
  - Two half-size CTAs per SM give two producers and two independent rings.
  - Against that, a ring depth of 3 instead of 4 cost 4.5% at bs8/16, and 12/16 consumer warps gained nothing.
- **Estimate:** 0-3% at bs8-32. Worth one A/B at bs8/16 with a ring of 2 x (NS=4, 16 KB stages).

### 6. Stream-K "data-parallel + K-split tail" with FP32 partials and a serial-chain fixup [TRIED; depends on item 2]
- **Sources:**
  - Chord/Humming `scheduler.cuh`. Peel a tail of mn-tiles for K-splitting and keep the rest data-parallel. Partials
    go to fp32 scratch; slice 0 sums them in CTA order (deterministic); there is one lock word per stream-K tile,
    left at 0 so it needs no reset.
  - Chord uses it only when the last wave is < 90% full. Stream-K is off for mid-K down projections.
  - Osama et al., Stream-K, PPoPP'23, https://arxiv.org/abs/2301.03598.
- **Fit:** our KSX splits lost because of the global fence and slow combine, not because of the decomposition. Re-try
  only on top of item 2's DSMEM fixup, or with the fixup on the signaler warp (v190 direction).
- **Estimate:** included in item 2.

### 7. NCCL LL / LL128-style "flag in the data" handoff for cross-CTA dependencies [NEW, uncertain]
- **Source:** NCCL LL protocol: 8-byte stores of {4 B data, 4 B flag}, and the reader polls the data itself.
  See "Demystifying NCCL", https://arxiv.org/abs/2507.04786.
- **What it does:** no release fence and no separate flag. Each 8-byte (or 16-byte vector) store is single-copy
  atomic, so a reader that sees the epoch flag also sees the data.
- **Fit:** removes the `red.release` / MEMBAR.ALL.GPU writer stall (4.4 us). It does not fix queueing: even relaxed
  red took 3.8 us. Doubles handoff bytes (small).
- **Estimate:** 0-2% (bs1-32 dependency waits). Lower priority than item 2.

### 8. Scale decode: Marlin's S0E5M3 pre-conversion or Hopper `cvt.rn.f16x2.e4m3x2` [likely DONE]
- **Source:** vLLM Marlin `csrc/quantization/marlin/dequant.h`, `dequant_fp8_scales<half2, kFE4M3fn>`.
  - prepare() re-encodes the positive E4M3 scale as unsigned S0E5M3, with the FP4 2^14 bias folded in.
  - The kernel then needs only `(q & 0xFF00FF00) >> 1` = 2 int ops per 2 scales for FP16.
  - The BF16 variant is 3 ops: (sign >> 1) | (mag >> 4).
  - Hardware alternative: `cvt.rn.f16x2.e4m3x2` (sm_89+), 1 instruction per 2 scales.
- **Related bug:** vLLM PR #34577 shows S0E5M3 underflows for small scales (global_scale = 1 models) unless the scales
  are rescaled.
- **Estimate:** 0% if already done. Otherwise ~1-2% (2 scales per 8 weights per thread).

---

## Dequant op counts: we are already at or below every published sequence [DONE]

Counts are int ops (2 cycles each on H200) plus FMA-pipe ops, per 8 values. No sm_90 hardware E2M1 cvt exists:
`cvt.rn.f16x2.e2m1x2` is sm_100a+.

| Source | Format | Per 8 values | Notes |
|---|---|---|---|
| **Ours** | E2M1 -> FP16 | 7 LOP3/SHF + 2 IMAD + 4 HMUL2 | Team exhaustive search: no lossless repack gets below 9 int. My own shift/rotate exact-cover search agrees: no 4-shift tiling of the {15,11,10,9}x2 FP16 field exists. |
| vLLM Marlin `dequant<half2,kFE2M1f>` | E2M1 -> FP16 | ~14 int + 4 HMUL2 | `(q & 0x80008000) \| ((q & 0x70007000) >> 3)`, `q <<= 4`, repeat, per 4 values; bias folded into the scale (`skip_flop`). |
| CUTLASS `numeric_conversion.h` `_e2m1_to_half_x4` / `_bf16_x4` | E2M1 -> FP16 / BF16 | ~18 / ~24 int | PRMT LUT: LUT 0x464442403E3C3800 (FP16) or 0x03020100FFFEFC00 + upper 0xc0804000 (BF16). Used by SM90 mixed-input E2M1. |
| Humming `fp_to_fp` | E2M1 -> 16-bit | ~3-4 per pair + outer shifts | |
| Humming `fused_dequant_for_mxfp4` | E2M1 -> FP8/INT8 | 2 PRMT LUT per 8 codes | Folds the E8M0 (power-of-two) scale into the LUT via an exponent offset. Not possible for E4M3 scales, which have a mantissa. |
| Chord / Humming INT4 | INT4 -> BF16 | 3 SHF + 4 LOP3 + 4 HSUB2 + 4 HMUL2 | Magic 0x43004300, subtract 0x43084308, then the group scale. Linear INT4 only; E2M1 is not linear and cannot be transcoded to INT4 losslessly (2x values reach 12). |
| LiquidGEMM / QServe | UINT4 -> INT8 (W4A8) | 2 ops per 4 (IMAD + XOR) after unpack, ~7 per 8 total | https://arxiv.org/abs/2509.01229. Needs INT8 activations. FP8/INT8 activations fail our tolerance (1.8-3.8e-2). |

A Chord micro-detail worth a SASS check: extract the two scales of a 32-bit pair with
`__low2bfloat162/__high2bfloat162` (`mov.b32 {lo,hi}`). Indexing through a pointer made NVVM emit SHF.R.U64 funnel
shifts in the hot loop, which cost them 1-2% on decode. If our decode loop has SHF.R.U64 or extra PRMTs on the scale
path, that is free.

## Chord vs Humming (requested), H200 numbers
- **Chord:** INT4 group-32 BF16-scale W4A16 MoE (Kimi K2.5/K2.6 shapes).
  - "Indexed" path forked from Humming; "grouped masked/contiguous" SM90 path forked from DeepGEMM (TMA producer WG
    plus RS-wgmma consumers, scales lane-permuted so rows r/r+8 form one BF16 pair).
  - Decode: swap-AB `mma.sync` with 8-24 token blocks, 256x64 N/K tiles, 3-4 CTAs/SM, cp.async rings of 4-6 stages.
  - Prefill: wgmma with `wait<1>`, keeping one group in flight over the next load + dequant (3-6% gate/up, 1-5% down).
  - Separate gate_up and down launches. The combine is not fused.
- **Effective bandwidth, H200 EP8 decode:**

  | Kernel | gate_up | down |
  |---|---|---|
  | Chord | 3.2-3.6 TB/s | 3.2-3.4 TB/s |
  | Humming | 3.1-3.5 TB/s | 3.0-3.3 TB/s |

  Our per-call effective bandwidth is 3.3-3.8 TB/s with the fused combine, so Chord gives no bandwidth or decode lead
  to copy. Its transferable pieces are items 5 and 6 and the scale-extraction detail above.
- **Not reported:** neither project reports clocks or power.
- **Note:** Humming's group scale on C with an f16 accumulator is the precedent for item 3.

## Hopper clocks with tensor cores [DONE: measured by the team; no published workaround]
- I found no published FP16 vs BF16 tensor power measurement on Hopper. Gregersen et al. did not test BF16.
- Luo et al. (above), on H800 PCIe:
  - `mma.sync` averages ~63% of peak; wgmma reaches >95% with zero inputs.
  - wgmma drops to ~88% with random inputs as power nears the limit and clocks fall. This is the item-4 effect.
- Supporting arithmetic (my calculation, not a quoted statement): NVIDIA's H100 SXM dense FP16 spec of 989.4 TFLOPS
  equals 132 SM x 4096 flop/clk x 1.83 GHz. The datasheet assumes ~1.83 GHz, not 1.98 GHz, for tensor work, which
  matches our ~1.79 GHz in-kernel cap. Avoiding tensor cores to keep 1.98 GHz was measured as a dead end (CUDA-core
  path 2x slower).

## Unpublished idea (no source; flagged speculative)
- **Cold instruction fetch at startup** cost bs256 ~5 us (team trace).
  - An early `prefetch.global.L2` / `cp.async.bulk.prefetch.L2` over the kernel's own code range would turn I-cache
    misses into L2 hits. This assumes the SASS VA is readable from the data path; it can be found via a device
    function pointer or the module's cubin load address.
  - Separately, shrinking hot-path code (fewer unrolled variants per launch) helps.
  - Test with an isolated probe only.

## Not useful / dead ends
- **FLUTE-style smem LUT dequant:** bank conflicts on random 4-bit indices; LDS throughput is below our ALU path.
- **FP8 (E4M3) tensor path:** FP4 x E4M3 products are not representable in E4M3, and the activations would fail the
  tolerance.
- **TRT-LLM / Hazy "CUDA cores for matvec":** keeps 1.98 GHz but needs >= 2.25 instructions per weight, against 1.75
  on our tensor path.
