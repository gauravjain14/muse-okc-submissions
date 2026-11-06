# muse_prefill_layer

One full MoE transformer layer at prefill shapes on one H200, timed against SGLang running the same layer. The layer has attention, 32 local routed NVFP4 experts out of 256, and BF16 dense GEMMs. It was an Open Kernel Challenge problem; submissions closed on 2026-11-04. This tree is the technical record of the iterations that moved the score: what the profiles showed, what that decided, what changed, and what it landed.

## The problem

`tools/TASK.md` and `tools/reference.py` define the computation. Per call, with T = batch × q_len new tokens:

```
x[T,8192] -> QKV+gate GEMM (N = 2304: 16 q heads + 2 kv heads x 64, + the gate) -> QK-norm, RoPE, FP8 K/V into a paged cache
-> attention (sliding window 2048, or full context; causal inside the new block) -> rmsnorm * sigmoid(gate) -> o_proj
-> boundary 1 (FP32 residual stream s, two RMSNorms) -> router (256 experts, top-8; FP32 weights as a BF16 hi/lo pair)
   + pooler (8192 -> 4096, BF16) -> pre-expert norm -> 32 local NVFP4 experts (W13 8192x4096, W2 4096x4096 each)
-> combine -> unpooler (4096 -> 8192) -> boundary 2
```

SGLang's side (`tools/baseline.py`) uses cuBLAS for the dense GEMMs, FA3 FP8 paged attention, JIT/Triton fused norms, and **Marlin W4A16** for the experts.

The rules that shaped the design (`tools/TASK.md`, "Rules"):
- Weights stay NVFP4 / BF16.
- Activations stay BF16 wherever SGLang keeps BF16. This rules out FP8 activations into the experts, because Marlin is W4A16.
- q may be FP8 E4M3, and every key must be attended.
- The layer is replayed as a CUDA graph with fresh inputs, so nothing may sync or allocate.
- The submission file is capped at 1,000,000 bytes. The milestone 10 build is 974,299 B.

## The score

There are 8 scored cases in 3 "loads" (`tools/TASK.md`, "Shapes"). A load's time is Σ weight × case time. **Speedup = the geometric mean over the 3 loads of SGLang's load time over ours.** The board shows score_us = 387,403.9 / speedup (`muse-opt-codex/muse_prefill_layer/research/claude/NOTES.md:4-7`).

| case | shape | load (weight) | SGLang µs (one judge run) | SOL µs | % score per µs saved |
|---|---|---|---|---|---|
| 0 | b1 ctx 80K **sliding** q753 | turn 1 (46) | 1088.6 | 257.0 | **0.042** |
| 1 | b1 ctx 80K full q753 | turn 1 (16) | 1483.7 | 496.7 | 0.0146 |
| 2 | b4 ctx 80K sliding q753 (T = 3012) | turn 4 (46) | 3357.7 | 1036.3 | 0.0126 |
| 3 | b4 ctx 80K full q753 (T = 3012) | turn 4 (16) | 4886.9 | 1992.4 | 0.0044 |
| 4 | b1 ctx 0 full q16384 | first turn (16) | 17394.7 | 5518.4 | 0.0003 |
| 5 | b1 ctx 32K sliding q16384 | first turn (138) | 17050.0 | 5663.4 | 0.0027 |
| 6 | b1 ctx 32K full q16384 | first turn (16) | 21222.9 | 7710.0 | 0.0003 |
| 7 | b1 ctx 64K full q16384 | first turn (16) | 25134.5 | 9847.7 | 0.0003 |

- SGLang times are from `10_moe_wide_narrow_split_riders/evidence/judge_after_s3_70739638.txt`.
- Sensitivities are from `muse-opt-codex/muse_prefill_layer/research/harness/NOTES.md:138-141`.
- SOL is max(bytes / 4.8 TB/s, FLOP / 989 TF/s) (`tools/TASK.md`).

**One µs at case 0 is worth 15 µs at case 5.** Sliding cases are 67-70 % of every load.

**How noisy one judge reading is.** Each judge run draws a fresh router seed, so the expert work itself changes between runs. Per-shape ratios move 3-5 % between draws (`muse-opt-codex/muse_prefill_layer/research/claude/NOTES.md:8-10`), and SGLang's own times move too. Five pairs of same-code judge runs in milestones 06-10 differed by 0.16-0.77 % (computed from `okc-retro/index/judge_runs.tsv`).
- Every milestone gives all judge readings of its build. The **real level** is their mean.
- The real effect comes from a same-run local A/B with `tools/research/judge/replica.py`: judge-faithful AB/BA pairing against SGLang on a panel of router draws.

## Milestones that moved the score

"real" = mean of the judge readings of that exact build.

| # | milestone | judge readings → real | what drove it |
|---|---|---|---|
| 01 | [own FP8 attention on full layers (attn8)](01_fp8_attention_full_layers/) | start 1.5855 → **1.6369** | FA3 reached ~55 % of the MUFU exp floor on full layers (~9 of 16 scores/clk/SM) → own FP8 QK/P·V wgmma attention with 3 consumer warpgroups, full layers only; polynomial-exp offload lost |
| 02 | [native expert path v7, f16 softmax, PDL](02_native_moe_f16_softmax_pdl/) | **1.7359** | roofline: experts and attention are the two big gaps; ablation: the softmax chain is ~50 % of attn_fwd (379 → 201 µs) → native expert kernel v1 → v7, f16 scores, unit-mode sliding attention, PDL (stack) |
| 03 | [EXPQ: the FP8 exponential in one HFMA2](03_expq_fp8_exponential/) | **1.8310** | P is consumed as an E4M3 code and MUFU was still 64 % busy → EXPQ writes the code with one HFMA2 (8S + 1080 − 8m), a clamp and a PRMT: attn_fwd −25..−33 % on full layers |
| 04 | [expert-kernel vN chain (v9 → v24b)](04_moe_vn_expert_chain/) | **1.8329** | roofline: the expert kernel at 44 % of its roofline at T = 753; ncu: issue-bound with descriptor rebuilds on every wgmma → routing / gather / issue trims, each kept on a same-run layer win |
| 05 | [stream-K-only attention module, fused 16K unpool + boundary](05_sk_only_attention_fused_unpool/) | 1.8455 (full25a) … **1.8866** | ablation: combine paths attn_fwd never runs cost it 11 % (code presence) → stream-K-only module; plus SM-shared QKV + kv_pack, sk2 o_proj, zero-workspace cuBLASLt plans, fused 16K unpool + boundary |
| 06 | [public okc-2026 expert kernel at T ≥ 2048](06_public_okc2026_expert_kernel/) | 1.9232, 1.9201 → **1.9217** | at T ≥ 2048 the native expert kernel lost time around its mainloop; the public 128-row × 256-token kernel with a publisher warp was ×1.03-1.06 MoE-only → routed there; exact early row-max vote in full attention |
| 07 | [exact micro-gain bundles](07_exact_micro_gain_bundles/) | **1.9955** (one reading; replica 1.959) | every kernel within a few % of its own bound → SASS wgmma counts, stall PCs, excess sectors and item traces found ~20 exact 0.03-0.6 % items, stacked as bundles b3-b23 |
| 08 | [exhaustive cuBLASLt configs + judge-faithful replica](08_exhaustive_cublaslt_configs/) | 1.9831, 1.9901 → **1.9866** | the old local metric was absolute latency; a judge-faithful replica became the yardstick; exhaustive cuBLASLt search: workspace configs add memset nodes → memset-free picks + finer 16K boundary tails |
| 09 | [attention latency (attn1, c2) + 16K MoE without L2 prefetch](09_attention_latency_and_16k_moe/) | 1.9970, 2.0125 → **2.0048** | traces: full-layer attn_fwd ends spread by segment count; case-0 attention a latency chain; 16K expert loop paying a per-stage L2 prefetch → cost-aware stream-K split, fused 2-CTA sla_fwd, no prefetch at 16K |
| 10 | [expert kernel split into wide W + narrow N, with riders](10_moe_wide_narrow_split_riders/) | 2.0047, 2.0143 → **2.0095** | narrow expert items are decode-issue bound; a wgmma rider costs +26 cyc/k16 in a microbenchmark but +77 in production → kernel W with riders + kernel N, fused 3012 router+gather, attention probe / mask trims |

`dead_ends.md` lists the failures that changed direction.

## Where the time goes in build 10 (`s3`, judge job 70739638)

Shares of summed kernel time from the judge's per-kernel profile (`10_moe_wide_narrow_split_riders/evidence/judge_after_s3_70739638.txt`; top-12 rows per case, the remainder ≤ 2.4 %). Overlapped kernels (PDL, side stream) include their waits, so the shares are of summed time, not wall time.

| case | ours µs / SOL | MoE experts | dense GEMMs (cuBLAS + sk2) | 16K fused D6 + UB | attention + K/V | boundaries | route / norm / combine |
|---|---|---|---|---|---|---|---|
| 0 | 520.4 / 257.0 (2.03×) | **51.6 %** (W 183 + N 118 µs) | 34.7 % | – | 3.0 % | 5.9 % | 3.9 % |
| 1 | 726.9 / 496.7 (1.46×) | 37.3 % | 26.3 % | – | 28.6 % (attn_fwd 196, kv_pack 13) | 4.3 % | 1.1 % |
| 2 | 1711.2 / 1036.3 (1.65×) | **41.8 %** | 41.5 % | – | 5.3 % | 8.1 % | 3.4 % |
| 3 | 2446.8 / 1992.4 (1.23×) | 30.1 % | 28.0 % | – | 34.2 % (attn_fwd 738, kv_pack 118) | 5.1 % | 2.3 % |
| 4 | 8794.6 / 5518.4 (1.59×) | 40.0 % | 25.7 % | 25.4 % | 6.0 % | – | 2.8 % |
| 5 | 8640.8 / 5663.4 (1.53×) | **41.9 %** | 26.5 % | 25.8 % | 2.7 % | – | 3.1 % |
| 6 | 10451.8 / 7710.0 (1.36×) | 34.5 % | 21.9 % | 21.4 % | 19.7 % | – | 2.4 % |
| 7 | 11885.4 / 9847.7 (1.21×) | 29.5 % | 19.5 % | 19.1 % | 29.7 % | – | 2.1 % |

D6 (`ub2_2_192_4_4`) is the 16K o_proj GEMM with the post-attention boundary fused in; UB (`ub2_2_192_3`) is the 16K unpool GEMM with the post-FFN boundary (milestone 08). "Attention + K/V" includes `qkv_post2` and `kv_pack`; "route / norm / combine" is pnr5 / `stk_pnrg` / pnr2y, the gather and the combine kernels.

**What binds each kernel** (measurements from milestones 09-10; `RC` = `muse-opt-codex/muse_prefill_layer/research/claude/`):

| kernel | measured | limiter |
|---|---|---|
| MoE kernel W (wide items + riders), T = 753 | n256 host loop 256.4 cycles per k16 against the 256 floor in the consumer-loop microbenchmark; a wgmma rider adds +26 there and +77..+81 in production (milestone 10, §2) | tensor on wide items. A running wgmma halves the int/fp16 pipes on its SMSP, so NVFP4 decode beside it is not hidden (`RC/moe2/NOTES.md`); ptxas' function-wide register allocation sets the rider cost |
| MoE kernel N (narrow items) | IPC 0.69-0.82 at 2 warps per SMSP; applying the scale after the MMA costs +24 % (N = 16) / +43 % (N = 32) | decode issue (`RC/ms/NOTES.md`, `RC/mn/NOTES.md`) |
| MoE at 16K | 90.5 % tensor-active (milestone 09) | tensor; per-item overheads are a few % of 3.5-3.6 ms |
| dense GEMMs | cuBLASLt mainloops at 96-99 % of the power-capped wgmma rate; at T = 753, ncu tensor-active QKV 82 %, fused pool + router 87 %, unpooler 87 %, sk2 o_proj 63 % (milestone 09, §2b) | at 753: fill, epilogue and idle SMs; at 3012 / 16K: the power-capped clock (`muse-opt/muse_prefill_layer/logs/gemm2_notes.md`) |
| full attn_fwd | case 1: 198 µs, tensor 72.8 %, ALU 42.8 %, issue 47.2 %; the ~25-instruction max / vote chain alone is 16 % of attn_fwd as latency (milestone 10, §2e) | the serial softmax chain of each tile |
| sliding attention (unit module) / sla_fwd | sla_fwd 17.4 µs (case 0), attn_fwd 41.7 µs (case 2) | a latency chain (milestone 09); 2-3 % of a sliding case |
| bnd3 (753) | 55-58 % DRAM (milestone 09 ncu); in-layer headroom to the floor 2.7 µs (b1) / 1.5 µs (b2) | latency-bound per-row chain (`RC/b3/NOTES.md`) |
| 16K D6 / UB | boundary row sums in the UB epilogue cost +1.2-1.5 µs per tile | boundary rows time-sliced inside a GEMM that owns all registers and smem; per-SM write port ~63 GB/s (`RC/bnd/NOTES.md`) |
| graph edges | case 0 idle 2.2-5.4 µs | a cross-stream join costs ~2 µs wherever it sits (`RC/gap/NOTES.md`) |

**The remaining time** sits in two places:
- the T = 753 / 3012 expert path, 30-52 % of cases 0-3;
- the full-layer attention, 20-34 % of the full cases.

A roofline estimate made on milestone 10's base build (`s_b`): with the experts at the measured ~824 TF/s, cases 0-3 would drop 265 → 167, 272 → 157, 786 → 667 and 753 → 676 µs of MoE time, a layer projection of ×2.172 (`RC/NOTES.md:237-239`).

## Tools

`tools/` holds the harness as it was at milestone 10, laid out the way the scripts expect: run them from `tools/`.

| file | what |
|---|---|
| `TASK.md`, `reference.py`, `checks.py`, `baseline.py` (SGLang's layer), `nvfp4.py`, `dev/public_cases.json` | the problem as published |
| `research/harness/check.py` | correctness gate: the judge's 5 tests, the 8 scored shapes and nearby shapes, 2 graph replays each |
| `research/claude/gate_check.py` | `check.py` plus 4 extra nearby shapes |
| `research/judge/replica.py` | judge replica: AB/BA paired against SGLang with the judge's overhead formula and score, on a panel of judge-style router draws |
| `research/claude/attn/front.py` | front-of-layer probe (QKV → K/V pack → attention → o_proj) with a cold L2, AB/BA |

```bash
cd tools
python research/claude/gate_check.py ../NN_x/build.py --json /tmp/gate.json
python research/judge/replica.py --candidate a=../MM_y/build.py --candidate b=../NN_x/build.py --cases 0,1,2,3 --panel 11,22,33 --reps 8
python research/claude/attn/front.py --candidate a=../MM_y/build.py --candidate b=../NN_x/build.py --cases 1,3 --reps 30
```

Every `build.py` is the judged file; where it contains third-party code, attribution comments mark each borrowed block in place and the code is unchanged (the judged md5 is in each README). Each one compiles its CUDA with NVRTC at import, and some also build a load_inline extension. Milestones 06-10 contain the routed-expert kernel `okc_moe::moe_fused_persistent` adapted from vjeux/okc-2026 (Apache-2.0); see the root `NOTICE`.
