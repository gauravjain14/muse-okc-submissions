# WP-RES-DECODE: can the scaled BF16 A operand (NVFP4 -> BF16) be made cheaper on sm_90? (research, 2026-10-27)

Research only. No GPU runs, no okc/Modal, nothing uploaded. CPU emulation scripts are in the session scratchpad
(`.../scratchpad/wpres/tile.py`, `bpc.py`); every formula they check is restated below.

Baseline (ours), per 32-bit word of 8 E2M1 codes = one m16n8k16 A fragment:
9 int ops (5 LOP3 + 2 SHF on ALU, 2 IMAD-as-shift on FMA-heavy) + 4 HMUL2.BF16 (FMA-heavy) + about 0.25-0.5 scale op
+ 1 HMMA.16816.F32.BF16 (about 4.4-4.6 issue cycles). Pipe load per word: ALU 7-7.5 ops (about 15 cycles),
FMA-heavy 6 ops (12 cycles). WP-D mix table: 21.1 cycles/word with HMMA, 17.1 without.

## Verdict

- **No published E2M1 -> BF16 decode beats our 9 int ops.** I re-checked Marlin, Machete, CUTLASS, TRT-LLM/FlashInfer
  sm90 mixed-input, Triton triton_kernels (Hopper), TileLang, FLUTE, LUT-GEMM, LiquidGEMM, QServe and MixQ. The best
  of them uses 11 int + 4 bias HMUL2 + 4 scale HMUL2. The PRMT-LUT, shared-memory-LUT, I2F/F2F and exponent-add
  routes all lose or break the rules (details below).
- I also checked multi-word decoding, i.e. funnel shifts across 2 or 4 words: no tiling of Z64 or Z128 by translates
  of the output bit set S = {6,7,8,15,22,23,24,31} exists (tile.py; Z32 also has none, which confirms the team's
  result). Decoding words jointly therefore cannot reach 7 int ops per word either.
- **What can be beaten is the 4 HMUL2 per word, and only for tiles with T <= 2 tokens.** Technique #1 below is my own
  derivation; I found no published source for it. It uses the n8 columns that are idle at T = 1-2 as per-block
  partial-sum lanes, and applies the E4M3 scale after the MMA with T/2 FFMA per word instead of 4 HMUL2.
- Removing the HMUL2s frees the FMA-heavy pipe. That makes technique #2 possible: move one rotation onto
  IMAD.WIDE.U32, which brings the ALU load to about 6.5 ops/word.
- For T >= 3 I found nothing that beats our sequence.

---

## 1. Block-per-column post-scaling ("spare-N block lanes") for T <= 2 tokens [derived, not found published]

### Idea
At T = 1 an m16n8k16 tile wastes 7 of its 8 columns, and at T = 2 it wastes 6. The tensor work per word is fixed
anyway: 2048 MACs = about 4 cycles per SMSP at peak, whatever is in B.

Put a **different 16-k scale block in each column**: B[k, n] = x_tau[k] if block(k) == block(n), else 0. Then
D[m, n] is the *unscaled* partial sum of one (row, block). The block scale is applied once per partial in FP32:
`acc += D * s`, instead of once per weight pair (HMUL2).

The plain post-scale (WP-D V12, one block per MMA, 4 FFMA per word) lost +12.5%. This variant amortizes one FFMA over
8/T words of MMAs per accumulator element.

Related prior art, none of which is this trick:
- DeepSeek-V3 / DeepGEMM per-128-k FP32 "promotion": an FFMA by the block scale, but one block per MMA chain.
- MAGMA batched small HGEMM: packs independent small GEMMs into one tensor-core tile.
- WP-F's column trick: 8 row tiles share one accumulator. It saves registers, not scale work.

### Layout (one weight layout for every T; it is the moe1t W1DN-style "lane owns the block" order)
PTX m16n8k16 fragments, with lane = 4g + t:
- A regs: (row g, k 2t..2t+1), (row g+8, k 2t..2t+1), (row g, k 2t+8..2t+9), (row g+8, k 2t+8..2t+9).
- B regs: (k 2t..2t+1, col g), (k 2t+8..2t+9, col g).
- D: c0 = D[g, 2t], c1 = D[g, 2t+1], c2 = D[g+8, 2t], c3 = D[g+8, 2t+1].

Group J = 4 consecutive k-steps j = 0..3, covering blocks 4J .. 4J+3:
- Position p of k-step j belongs to block beta(p) = (p >> 1) & 3, at block offset 4j + 2(p >> 3) + (p & 1).
- So lane t's 8 A values of every word in the group are rows g and g+8 of block 4J+t. Its 4 words cover that
  block completely, with one scale pair per 4 words.
- The HMUL2 path (T >= 3) can run on the same layout, with one scale pair per 4 words.

Columns: n = 2*beta + tau (tau = token 0/1).
- Lane (g, t) loads B only when g>>1 == t and (g&1) < T. Its two B regs are then x_(g&1)[16(4J+t) + 4j + {0,1}]
  and [... + {2,3}]. Over the 4 k-steps that is 32 contiguous bytes, i.e. 2 LDS.128.
- All other lanes load from a 32-byte zero buffer (broadcast, conflict-free). Register count and B-load count are
  the same as today; moe1t's 64 resident B registers stay 64.
- After the group, lane (g, t) holds:
  - c0 = (row g, block 4J+t, token 0), c2 = the same for row g+8.
  - c1 / c3 = token 1 (T = 2 only).
- These need exactly the lane's own scale pair (s(g, 4J+t), s(g+8, 4J+t)), the same scales its A block uses. Each
  scale is now used by one lane: the 4x quad redundancy of today's scale handling is gone.

### Instruction sequence
Per word j of group J:
```
9-op decode (unchanged)  -> a0..a3 = BF16(E2M1) * 2^-126  (exact; 0.5 is BF16-subnormal, -0 kept)
HMMA.16816.F32.BF16 D, a, b_j, (j == 0 ? RZ : D)
```
Per group (4 words), with 2 scales:
```
sg  = LOP3((w << k0) & 0x0FF00000 | 0x40000000)   # 1 IMAD.SHL (or none) + 1 LOP3 per scale, see "scales"
sg8 = ...
FFMA acc0 = c0 * sg  + acc0 ;  FFMA acc2 = c2 * sg8 + acc2          # T = 1
FFMA acc1 = c1 * sg  + acc1 ;  FFMA acc3 = c3 * sg8 + acc3          # T = 2 only
```
Per item, once: 2 SHFL.BFLY + 2 FADD per row pair (quad reduction over t), then x 2^115 x global scale in the
epilogue.

### Cost per word (NTL1 tile)

| | ALU | FMA-heavy (FMA-lite) | HMMA | issue slots |
|---|---|---|---|---|
| today | 7 + 0.25-0.5 | 2 IMAD + 4 HMUL2 | 1 | ~13.5 + HMMA |
| #1, T = 1 | 7 + 0.5 | 2 IMAD + 0.5 IMAD + 0.5 FFMA | 1 | ~10.5 + HMMA |
| #1, T = 2 | 7 + 0.5 | 2 + 0.5 + 1 FFMA | 1 | ~11 + HMMA |
| #1, T = 3-4 | needs n16 = 2 HMMA per word: loses | | | |

### Scales
FFMA needs an FP32 scale whose exponent is shifted by a large power of two. D is about 2^-126 * sum, so a plain
(E4M3 << 20) placement (= s * 2^-120) would underflow the product.

E4M3 subnormal scales break any "OR an exponent offset" trick. The exact options are:

- (a) **Lossless re-encode in prepare()** (preferred). It is the same value set, which the "compress losslessly"
  rule allows.
  - Every positive E4M3 value (subnormals included) becomes an all-normal code u = ((E + 10) << 3) | m.
    E is in [-9, 8], so u <= 151.
  - Zero-scale blocks: zero their codes (as `_repack` already does) and give them any nonzero u.
  - Then the FP32 bits are (u << 20) | 0x40000000, which equals s * 2^11 exactly.
  - Cost: 1 LOP3, plus 1 IMAD.SHL when u is not already at bits [27:20]. That is about 0.5 ALU + 0.5 IMAD per word.
- (b) **Keep the E4M3 bytes.** Use shift + mask (s * 2^-120, exact including subnormals), then FMUL by 2^127.
  That is 3 ops per scale.

### Numerics
CPU emulation of the exact fragment mapping (bpc.py; random NVFP4 rows, real E4M3 value set, FP32 partials,
K = 1024): rel-L2 vs float64 is 6.1e-8 (T = 1), 4.2e-8 (T = 2) and 4.2e-8 (T = 4 variant). Each lane uses 2-4
distinct scales per group, as derived.

The scale must not be s * 2^126 in FP32: that overflows. The emulation hit Inf, which is why the exponent is split
(2^11 in the scale, 2^115 in the epilogue).

About 7% of block partials are FP32-subnormal for x ~ N(0, 1) (|sum| < 1 with the 2^-126 coding):
- The absolute error is about 2^-23 in sum units, which is harmless.
- WP-D V12 measured 1.7e-7 with the same coding, so the tensor core does not flush subnormal FP32 results. If it
  did, the error would have been ~3e-2.
- Where x passes through the kernel anyway (gather / staging), pre-scale it by 2^64 (exact). That keeps the
  partials normal and protects tiny-x test inputs.

### Rules
- BF16 x BF16 MMA with FP32 accumulation; no FP16 or FP8 anywhere. A holds exact BF16 values (E2M1 * 2^-126) and
  the scale is applied in FP32.
- This is the same rule status as WP-D's V12, which the team classified as "valid, just slower".

### Expected gain
- Additive model from the WP-D mix table: 21.1 -> about 20 cycles/word, i.e. -5%. It removes ~2.7 cycles of HMUL2
  interference and adds ~1 ALU + 0.55 FFMA.
- The team's in-kernel "no HMUL2" ablation (-17..-23%) suggests up to about -12%.
- Range for T <= 2 stages: **-5..-12% of stage cycles** (combined with #2: -10..-18%).
- It applies to bs1 (moe1t, whose phase A is data-bound, so the gain is small), bs8/16 (compute-bound, T mostly
  1-2) and part of bs32.
- Test first in dev/wpd_stage.py as V14 (T = 1, T = 2), with the >= 4% port threshold.

### Risks
- One more stage-loop variant per T class (i-cache / cold code at startup).
- The main kernel's weights need the W1DN-style k order (WP-1 measured that order alone as noise).
- The bs8/16 raw-x B path needs per-lane zero-buffer addresses.

## 2. After #1: move one rotation from the ALU to IMAD.WIDE.U32 [derived, conditional]

- With the HMUL2s gone, the FMA-heavy pipe carries about 2.5-3 ops per word (5-6 cycles) against the ALU's 15.
- IMAD.WIDE.U32 {lo, hi} = q * 2^r gives lo = q << r and hi = q >> (32 - r), so rot(q, r) & M = LOP3(lo, hi, M)
  with the function (a | b) & c.
- When the consumer of an SHF.L.W rotation is a 2-input mask, the pair SHF + LOP3 (2 ALU ops) becomes
  IMAD.WIDE (FMA, 2 passes, ~4.4 cycles) + LOP3 (1 ALU op).
- A right shift whose wrapped bits are masked away becomes IMAD.HI; a left shift becomes IMAD.SHL.
- Result: ALU about 6.5 ops/word (13.4 cycles), FMA about 10 cycles.
- Converting both SHFs makes the FMA pipe the binding one (~14.4 cycles), so convert only one.
- The team's earlier "mul.wide rotations are slower" result was measured with 4 HMUL2s on the FMA pipe: re-test it
  only on top of #1.
- If the rotation's consumer LOP3 already uses 3 inputs, this gains nothing. Re-run dqsearch with the cost
  max(2 * ALU, 2 * IMAD + 4 * IMAD.WIDE + FFMA) instead of the int-op count.

## 3. Published / other techniques (none beat ours)

| technique (source) | per 8 E2M1 -> scaled BF16 | pipes | exact? | rules | verdict |
|---|---|---|---|---|---|
| vLLM Marlin `dequant<bf16, kFE2M1f>` | Out = (q & 0x80008000) \| ((q & 0x70007000) >> 6); q <<= 4; repeat; twice per word: ~14-15 ALU + 4 HMUL2 scale (bias 2^126 folded with skip_flop, else +4 HMUL2) | ALU | yes (same 2^-126 coding) | ok | worse |
| Triton triton_kernels Hopper `_unpack_fp4_to_bf16_triton` | masks 0x81C081C0 on q, q<<3, q<<6; 4th output assembled: 6 and + 3 shl + 2 shr + 2 or = 13 int, + 4 mul.bf16x2 by 0x7e807e80 (2^126), + 4 scale mul (E8M0) | ALU + FMA | yes | ok | worse (comment: "14 SASS" for the dequant) |
| TRT-LLM / FlashInfer sm90 mixed-input `psx_cvt_triton_fp4x8_to_bf16x8_interleaved` | same twiddle, ~11 int (2 LOP3) + 4 bias HMUL2 + 4 scale HMUL2; PRMT-LUT variant present but commented out | ALU + FMA | yes | ok | worse |
| CUTLASS `lookup_table_convert` (mixed_input_utils) | 3 PRMT per 4 codes with the scale baked into a per-group LUT, **8-bit outputs only** (W4A8 FP8/INT8); fused E8M0 only for MXFP4 x FP8 | ALU | for FP8 | violates (FP8 MMA) | n/a |
| PRMT register LUT for BF16 (derived) | the 16-B LUT of s*{0,..,6} needs separate lo-byte and hi-byte tables + interleave: >= 8 PRMT + 2 sign LOP3 per word + LUT build (4 HMUL2 + 4 PRMT per block-row; one lane owns at most 2 block-rows per 4 words) | ALU | yes | ok | >= 10 ALU: worse |
| FLUTE smem LUT (A100/A6000 only) | 256-entry BF16x2 pair table (1 KB): 4 LDS.32 + >= 4 LOP3 (mask \| base) + 3 shifts + 4 HMUL2 | ALU 7 + MIO 4 | yes | ok | worse: one table copy = 3.2 wavefronts per LDS.32 (sim: uniform 3.15, NVFP4-Gaussian 3.19) -> ~13 smem cycles per word-warp, ~100 per SM word-round vs ~42 issue cycles: 2.5x over smem bandwidth; 32 bank-private copies (32 KB) = 1 wavefront, but ALU unchanged and smem ~100% busy with the ring reads |
| Machete (vLLM) | u4 / u4b8 / u8 / u8b128 only (no E2M1); LOP3 magic 0x43004300 | | | | n/a |
| INT4-magic (Marlin/CUTLASS/Chord) | 5 ALU + 4 HFMA2 per 8 | | E2M1 is nonlinear, cannot transcode | | n/a |
| LiquidGEMM (2 instr / 4 elem), QServe, MixQ | W4A8 / INT8 IMMA paths | | | violate (INT8 activations/MMA) | n/a |
| LUT-GEMM | BCQ weights, CUDA cores, T = 1 | FMA | E2M1 not BCQ | FP32 acc per weight ~1 FFMA/weight | worse |
| cvt.rn.{f16x2,bf16x2}.e2m1x2 | sm_100a+ only; f16 output | | | FP16 temp | n/a on sm_90 |
| cvt.e4m3x2 -> f16x2 (F2FP UNPACK_B) for scales | 1 ALU per 2 scales | | | FP16 temp | not allowed |
| I2F / I2FP.F32.U32 / cvt.bf16.u32 | 1 value per op (8 per word, ALU / XU) | | | | worse |
| F2FP.BF16.F32.PACK_AB from FP32-placed codes | +4 ALU per word | | | | worse |
| exponent-add scaling (IADD on BF16 exponent) | only for E8M0 (MXFP4: Humming, TRT-LLM fp8 path); E4M3 has 3 mantissa bits, and code 0 / 0.5 break exponent adds | | no for E4M3 | | n/a |
| per-16-k post-scale, one block per MMA (DeepGEMM-style) | 4 FFMA per word (3 register sources, no co-issue) + 2 scale cvt per word | FMA | yes | ok | measured +12.5% (WP-D V12); #1 fixes it for T <= 2 |
| sign carried in the HMUL2 multiplier (derived) | magnitudes tile {6,7,8} but each output's multiplier needs a sign LOP3: 7 + 4 = 11 ALU | ALU | yes | ok | worse |
| uniform datapath (ULOP3 / USHF) | weights are per-lane, not warp-uniform | | | | n/a |

Fixed floor per word: 2048 MACs per m16n8k16 = about 4 cycles per SMSP at the BF16 dense peak, for any mma.sync or
wgmma shape. HMMA time per word cannot shrink; only the CUDA-core work around it can.

## Sources
- vLLM Marlin dequant.h: https://github.com/vllm-project/vllm/blob/main/csrc/quantization/marlin/dequant.h
- vLLM Machete generate.py: https://github.com/vllm-project/vllm/blob/main/csrc/quantization/machete/generate.py
- Triton triton_kernels Hopper value layout / unpack: https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/tensor_details/layout_details/hopper_value.py
- FlashInfer (TRT-LLM cutlass_extensions) mixed_input_utils.hpp: https://github.com/flashinfer-ai/flashinfer/blob/main/csrc/nv_internal/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/detail/collective/mixed_input_utils.hpp
  and ..._mixed_input_prescale.hpp in the same tree (fused E8M0 only for MXFP4 x FP8)
- CUTLASS mixed_input_utils.hpp: https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/detail/collective/mixed_input_utils.hpp
- TileLang mxfp (twiddling decode): https://www.tilelang.com/autoapi/tilelang/quantize/mxfp/index.html
- FLUTE: https://arxiv.org/abs/2407.10960 (HTML v4: 256-entry pair table, duplicated copies, A100/A6000 only)
- LiquidGEMM: https://arxiv.org/abs/2509.01229 ; QServe: https://arxiv.org/abs/2405.04532 ; MixQ: https://pacman.cs.tsinghua.edu.cn/~zjd/projects/mixq/ ; LUT-GEMM: https://arxiv.org/abs/2206.09557
- DeepSeek-V3 (FP32 promotion with block scale): https://arxiv.org/abs/2412.19437 ; DeepGEMM: https://github.com/deepseek-ai/DeepGEMM
- MAGMA batched small HGEMM packing: https://netlib3.cs.utk.edu/files/publications/2019/icl-utk-1234-2019.pdf
- CUDA FP4 conversion API (sm_100+ hardware cvt): https://docs.nvidia.com/cuda/archive/13.4.1/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP4__MISC.html
- Team notes used: findings/wpd.md (pipe costs, V12), wpr.md, wpb1.md (moe1t layout), wpf.md (column trick), okc-moe-findings.

---

## Build status: gates 1-2 prepared, NOT measured (2026-10-27 evening)

**Why nothing is measured:** the planner asked for Modal runs. The Claude Code permission classifier denied the Modal
upload (`bash dev/modal_gate.sh dev/modal_run.py --cmd "bash dev/wpres_run1.sh"` -> "Data Exfiltration"). This task
had also started with "do not run okc or Modal". No Modal job was started and nothing was uploaded.

What is done:
- The harnesses below.
- Local NVRTC-13 sm_90a compiles with SASS counts.
- Local correctness on the RTX 4060 Ti (sm_89, nvcc 12.1). The mma.sync BF16 path is the same instruction there;
  V0/V1 use sm_90-only mul.bf16x2, so they cannot run locally.

Gate 3 (dev/moe_v1050.py) was not started: it is gated on gates 1 and 2.

### Files
- `dev/wpres_stage.py`: gate 1, the stage microbenchmark.
  - Same 32 KB stage image, harness and FP64 error check as dev/wpd_stage.py; base dev/moe_v1033.py.
  - `BPC_SRC` holds the reusable device code: stage_bpc<TT, DQW, SC6>, s6o_f32, dec_bp, mulw.
  - Variants:
    - V0 = today's stage_compute on 8-bit scales; V1 = today's stage_compute on S6O scales (the (4,1,1) stage).
    - V14 / V15 = #1 at T = 1 / 2 on 8-bit scales; V16-V19 = the same plus #2 (one or both rotations on
      IMAD.WIDE.U32).
    - V20 / V21 = #1 on the S6O words (same stage bytes as today); V22 / V23 = V20 / V21 plus #2 (one rotation).
- `dev/wpres_occ.py`: gate 2, the streaming harness.
  - dev/wp16w_occ.py's A8 harness for the (4,1,1) 1-group stage: 132 CTAs, NS 4, cold HBM weights, TX x rows from L2.
    Absent-token rows are zeroed once.
  - Designs: A8 / A8b (today, two controls), P1 / P2 (#1 T = 1 / 2 on S6O words), P1W / P2W (#1 + #2).
  - Modes: load, free (compute only), ring.
- `dev/wpres_run1.sh` (gate 1) and `dev/wpres_run2.sh` (gate 2): each does a smoke step (timeout / 20 s watchdog per
  kernel) and two passes, and records nvidia-smi before and after.
- Run them with:
  `bash dev/modal_gate.sh dev/modal_run.py --cmd "bash dev/wpres_run1.sh"`
  and then the same with wpres_run2.sh.

### Correctness (local sm_89; random NVFP4 experts from nvfp4.random_weights, seed 5252; generate_input x, 16 stages = k 4096)

| variant | w13 T=1 | w13 T=2 | w2 T=1 | w2 T=2 |
|---|---|---|---|---|
| V14 / V16 / V17 (8-bit scales, #2 none / 1 / 2) | 7.09e-8 | | 1.05e-7 | |
| V15 / V18 / V19 | | 7.00e-8 | | 1.13e-7 |
| V20 / V22 (S6O words) | 7.09e-8 | | 1.05e-7 | |
| V21 / V23 | | 7.00e-8 | | 1.13e-7 |

These are rel-L2 vs FP64, with no nonfinite values. The S6O FP32 conversion `(f << 20) | 0x74000000` reproduces the
8-bit results bit for bit. For comparison, WP-D measured V0 (today) at 1.0e-6 / 2.9e-6 on the judge's weights: the
post-scale is about 10x more accurate, because block partials and scale products are FP32.

### SASS per warp-stage (local NVRTC 13.0 sm_90a, the bench loop = 32 words)

| V | ALU (LOP3 + SHF + PRMT) | FMA-heavy (IMAD, HMUL2, FFMA, IMAD.WIDE) | HMMA | total | regs |
|---|---|---|---|---|---|
| V0 today 8-bit | 161 + 64 + 16 = 241 | 83 + 128 HMUL2 | 32 | 506 | 56 |
| V1 today S6O | 171 + 67 = 238 | 71 + 128 HMUL2 | 32 | 492 | 56 |
| V14 (T1) | 177 + 68 = 245 | 79 + 16 FFMA | 32 | 393 | 80 |
| V15 (T2) | 245 | 79 + 32 FFMA | 32 | 409 | 86 |
| V16 (T1, #2 x1) | 177 + 36 = 213 | 79 + 32 WIDE + 16 FFMA | 32 | 393 | 90 |
| V17 (T1, #2 x2) | 177 + 4 = 181 | 79 + 64 WIDE + 16 FFMA | 32 | 393 | 78 |
| V20 (T1, S6O) | 181 + 67 = 248 | 80 + 16 FFMA | 32 | 399 | 84 |
| V21 (T2, S6O) | 248 | 79 + 32 FFMA | 32 | 413 | 90 |
| V22 (T1, S6O, #2 x1) | 181 + 35 = 216 | 80 + 32 WIDE + 16 FFMA | 32 | 399 | 78 |

In the gate-2 harness (168-register cap) the consumer loops compile to 502 (A8, today) and 407 (P1) instructions per
stage, with no spills.

Codegen finding: writing the #2 rotation in C, `(u64)q * c`, makes ptxas add a VIADD (hi + 0) per rotation, i.e. +32
ALU ops per stage. The PTX `mul.wide.u32` form (`mulw`) is clean. v1033's own MOE_DQW switch uses the C form, which
may be part of why the team measured "mul.wide rotations are slower".

### Prediction (pipe model from WP-D's tables; NOT a measurement)
Baseline: V0 measured 1450 cycles/stage (WP-D), and the model attributes ~190 of those cycles to HMUL2 interference.

| variant | ALU work | predicted cycles/stage | change |
|---|---|---|---|
| V20 (T = 1) | 248 ops | ~1340-1360 | -6..-8% |
| V22 (+ #2) | 216 ops | ~1210-1250 | -12..-16% |

Gate 1 (>= 5% at T = 1, compute-only) is therefore expected to pass, but only the H200 run decides it.

### Applicability caveat for gate 3 (bs2-8 kernel moe_4_1_1_4_1)
#1 needs **T <= 2 tokens** in the tile, not just one token group. In the bs2-8 kernel an expert can have up to 8
tokens, so the candidate must branch per item: T <= 2 -> stage_bpc, otherwise today's stage_compute.

The weight words, scale bytes and x layout are the same for both paths: stage_bpc reads today's S6O words and the
_POS words unchanged, so prepare() and the repack do not change.

Per item, the kernel needs:
- the x row choice (lane (g, t) reads token g & 1 when g >> 1 == t, else a zero row);
- the quad reduction of acc (2 SHFL.BFLY + 2 FADD per row pair);
- a 2^-22 output factor instead of the S6O tiles' 2^-14, i.e. the item's accumulators x 2^-8 relative to S6O.

---

## Gates 1-2: measured by the planner (2026-10-27; the planner runs every Modal job)

**Gate 1, dev/wpres_run1.sh: PASSED.** H200 at 1.80 GHz, compute-only cycles per stage; the two passes were identical.

| variant | cycles/stage | vs today |
|---|---|---|
| V1, today S6O | 1423.4 | |
| V20, #1 T = 1 on today's S6O bytes | 1261.8 | **-11.4%** |
| V21, #1 T = 2 | 1342.0 | -5.7% |
| V22, V20 + #2 | 1333.6 | worse than V20 |
| V23, V21 + #2 | 1419.4 | worse than V21 |
| V0, today 8-bit | 1450.4 | |
| V14, #1 T = 1 on 8-bit | 1265.6 | -12.7% |

**#2 loses on top of #1: dropped.**

**Gate 2, dev/wpres_run2.sh: MISSED (gate >= 2%).** 132 CTAs, cold HBM.
- Today A8 load runs at 1121.5 ns/stage (control A8b x1.003); the bare ring is 1097 ns/stage.

| design | load, 256 stages/CTA | load, 512 stages/CTA | free (compute only) |
|---|---|---|---|
| P1 (#1 T = 1) | x1.0093 | x1.0032 | x1.093 |
| P2 (#1 T = 2) | x0.997 | x1.000 | x1.055 |
| P1W / P2W (+ #2) | no better | no better | |

- Reading: the steady 1-group stage already runs within 2% of the ring floor, and #1 recovers less than 1% of it.
- The planner still wants ONE kernel measurement: item edges and the last stage's compute after the last byte may
  gain a little more. Scope: the bs2-8 kernel only, T = 1 items only.

## Gate 3 candidate: dev/moe_v1050.py (MOE_BPC), built by dev/wpres_mk1050.py from dev/moe_v1033.py, local only

- **Scope:** moe_4_1_1_4_1 (bs 2-8). Items with f.n == 1 take the V20 body (`bpc_stage1`); every other item runs
  today's stage loop. The token-group switch is unchanged.
- **No prepare() change.**
- **Zero lanes:**
  - B lanes outside their block (g even and g >> 1 != tig) read 1 KB of zeros at the end of dynamic shared memory.
  - The host's `_smem_bytes` adds 1024 B for (4,1,1, mw 1) under the define: 222464 B, still NS 4.
  - The zeros are written at kernel start by tid < 64, before the mbarrier-init `__syncthreads`.
- **Scale:** the FP32 scale is `(f << 20) | 0x78000000` = s * 2^112. acc is then E2M1 * s * x * 2^-14, the S6O
  tiles' scale, so the item end's x64 and the epilogues are unchanged.
- **Item end:** after the item's stage loop, 2 SHFL.BFLY + 2 FADD per value (rows g / g + 8 of both row groups) give
  every quad lane the row total.
- **SASS (local NVRTC 13.0, dev/wpc_sass.py, every kernel of the module):**
  - Default build: all 16 kernels byte-identical to v1033.
  - MOE_BPC build: 15 of 16 byte-identical; only moe_4_1_1_4_1 changes (6128 -> 7000 instructions, 168 registers,
    0 spills, 0 stack).
  - Today's 1-group stage loop in moe_4_1_1_4_1 has the same instructions (510, same opcode multiset), registers
    renamed and re-scheduled: opcode-order similarity 0.46 (wpc_cmp-style).
  - For calibration, earlier work recorded that a reschedule like this (same counts, similarity 0.24-0.83) costs
    <= 0.5%.
- **Correctness (local sm_89, $SCRATCHPAD/wpres/lbpc.py):**
  - Test: the v1050 BPC device code extracted verbatim, 16 stages (k 4096), S6O words, plus the item end's quad
    reduction and x64. The absent tokens' x rows hold stale non-zero data, as they can in the kernel.
  - Result: rel-L2 vs FP64 7.9e-8 / 8.0e-8 (w13, experts 1 / 3) and 1.14e-7 / 1.12e-7 (w2).
  - All four quad lanes are equal; no nonfinite values.
  - The full kernel (TMA / mbarrier / setmaxnreg) cannot run locally: hang1 / check_mt are the planner's job.
