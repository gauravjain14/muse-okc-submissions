# 01: attn8, our own FP8 attention kernel for full-attention layers (plus dense-path tuning)

| | before (Codex start) | after (01) |
|---|---|---|
| build | `evidence/base_codex_candidate_native_parallel_lt.py` (Codex's `candidate_native_parallel_lt.py`, md5 acae6549, 207,155 B) | `build.py` = `cand_att8c.py` (md5 04b318ac, 257,402 B) |
| judge | 1.5855 (check, job 134a1dec36614beb) | **1.6369** (check, job e4e4ea4469c94941) |
| real effect | | ≈ **+3.0 %** (same-run local bench: dense tuning +1.45 %, attention +1.5 %); the judge's +3.2 % is about half SGLang-side drift, see Result |
| cases changed | | attention: full-layer cases 1, 3, 4, 6, 7; dense GEMM picks and the fused pooler+router GEMM: all 8 cases |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: code copied or adapted from other projects is marked in place with `SPDX-SnippetBegin` / `SPDX-SnippetEnd` comments (source, copyright, license, our changes); see `NOTICE` and `LICENSES/` at the repository root. The code is unchanged. The judged file has md5 `04b318ac`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

**The problem, in one paragraph.** The task (`../tools/TASK.md`) is one transformer prefill layer of an MoE model on an H200: QKV+gate projection, QK-norm/RoPE and an FP8 paged KV cache, attention (sliding window 2048 or full context), output norm and gate, o_proj, residual boundaries, router + pooler, 32 local NVFP4 experts, unpooler. The judge times 8 shapes against SGLang and scores the geomean over 3 loads of Σw·t_SGLang / Σw·t_ours. Cases: 0 = 753 tokens sliding (turn 1, weight 46), 1 = 753 full (turn 1, w16), 2 = 3012 sliding (turn 4, w46), 3 = 3012 full (w16), 4 = 16K tokens ctx 0 full, 5 = 16K ctx 32K sliding (w138), 6 = 16K ctx 32K full, 7 = 16K ctx 64K full (all w16 in the first-turn load). One µs saved on case 1 is worth ~0.015 % of score, on case 3 ~0.004 %, on the 16K full cases ~0.0003 % (`muse-opt-codex/muse_prefill_layer/research/harness/NOTES.md:138-141`).

## 1. Starting point

On 2026-10-24 we took over Codex's best layer, `candidate_native_parallel_lt.py` (judge 1.5855). It used SGLang's own attention path (FusedQKNorm, cache store, **FA3** `flash_attn_with_kvcache` in FP8, FusedRMSNormGate), cuBLAS for the dense GEMMs and a native NVFP4 expert kernel (`moe_8_16_1_2_*`). Judge per case (`evidence/judge_before_codex_candidate_134a1dec.txt`):

| case | ours µs | SGLang µs | ratio | FA3 kernel µs | FA3 share of the case |
|---|---|---|---|---|---|
| 0 sliding 753 | 651.6 | 1060.2 | 1.627 | 25.8 | 3.7 % |
| 1 full 753 | 1037.4 | 1454.1 | **1.402** | 424.3 | **38.0 %** |
| 2 sliding 3012 | 2023.9 | 3397.7 | 1.679 | 65.6 | 3.1 % |
| 3 full 3012 | 3512.9 | 4926.8 | **1.402** | 1639.5 | **45.1 %** |
| 4 full 16K ctx 0 | 11041.9 | 18845.4 | 1.707 | 906.4 | 8.1 % |
| 5 sliding 16K | 10015.0 | 17064.4 | 1.704 | 305.6 | 3.0 % |
| 6 full 16K ctx 32K | 14232.9 | 21138.0 | 1.485 | 4529.0 | 31.9 % |
| 7 full 16K ctx 64K | 18003.2 | 24965.0 | **1.387** | 8151.0 | **46.0 %** |

The full-layer cases had the worst ratios, and FA3 was the largest kernel in three of them.

## 2. What we saw

**(a) FA3 sat at ~55 % of the exponential-unit floor.** Every attention score needs one `exp2`. The H200's MUFU unit issues 16 `ex2` per clock per SM, which caps a softmax that runs its exponentials there at 16 × 132 × 1.98 GHz = 4.18 T scores/s (`evidence/dev_roofline.py:20`, the `EXP` constant; the FP8 tensor cores would allow 32 scores/clk/SM at 256 FLOP per score). Case 1 has 753 × 16 × ~80,377 = 968 M scores (247.9 GFLOP in the roofline tool's count, `evidence/dev_roofline.py:65-69`), so its MUFU floor is 231.6 µs and its FP8 tensor floor 125 µs. FA3 took 424.3 µs: ~55 % of the MUFU floor, about 9 scores/clk/SM (the takeover analysis, (private notes, not included):15-16`). Cases 3/6/7 sit at the same 56-57 % (FA3 1639.5 / 4529.0 / 8151.0 µs against MUFU floors of 926 / 2568 / 4622 µs, computed with the same constants).

**(b) The softmax chain, not the matmuls, sets the time of a straightforward FP8 kernel.** Our first own kernel (`dev/attn.py`, `evidence/abl_nosoft.log`, `evidence/abl_nopoly.log`, case 1 attention only): 538 µs against SGLang's 437 µs path. With the softmax removed it ran in 315 µs (−41 %). With the polynomial exponential replaced by MUFU it ran in 476.5 µs. The second design, attn8 (`evidence/a8_01.log`), confirmed both: 592.7 µs with the FMA-pipe polynomial `exp2`, 392.5 µs without it (`cw3nopoly`), and two consumer warpgroups instead of three were slower (653-666 µs).

**(c) With the exponential on MUFU, the kernel is co-limited by the exp unit and the ALU, not by the tensor cores** (`evidence/ncu_a8.log`, `evidence/ncu_poly.log`, case 1, unmodified clocks):

| variant | µs | XU (MUFU) pipe | ALU | FMA | issue active | note |
|---|---|---|---|---|---|---|
| POLY 0 (all exps on MUFU) | 378.7 | **73.0 %** | 42.4 % | 13.8 % | 48.8 % | shipped |
| POLY 2 | 431.0 | 56.1 % | 39.7 % | 19.0 % | 51.5 % | |
| POLY 1 (FMA polynomial) | 475.0 | 43.3 % | 38.3 % | 23.0 % | 52.6 % | |

`ncu_a8.log` (394 µs, SM 1.79 GHz): issue slots 47.8 % busy, ALU the busiest pipe at 43.5 %, 3.23 active / 0.76 eligible warps per scheduler, no spills, DRAM 1.5 %. Moving exponentials off MUFU onto the FMA pipe only adds instructions to an issue stream that is already half busy; every polynomial split lost (`evidence/a8_03.log`: POLY 1/2/3 at 435-481 µs against 386 µs).

**(d) Sliding layers at 753 tokens are too small for the kernel.** attn8 lost there (`evidence/a8_02.log`, `evidence/a8_05.log`: case 0 attention path 0.86-0.89× SGLang's) but won on every full layer: case 1 1.09×, case 3 1.06×, case 6 1.17× (`evidence/a8_05.log`).

**(e) The dense path had measurable slack** (`evidence/dense_gemms_01.log`, `evidence/dense_section_01.log`, `evidence/dense_router_01.log`):
- The cuBLAS heuristic choice was not the fastest measured cuBLASLt algorithm on several shapes: 16K o_proj 382.2 → 343.0 µs (1.11×), 16K unpooler 1429.6 → 1361.8 µs (1.05×), 3012 QKV 150.7 → 148.3 µs.
- The router (two BF16 GEMMs on a hi/lo split plus an add) can ride in the pooler GEMM as extra output columns (one FP32-output GEMM of N = 4608) with the pre-expert norm and the hi+lo sum in one pass: 753 tokens 105.2 → 90.0 µs, 3012 334.1 → 321.5 µs, 16K no gain (0.998×, so only up to 4096 tokens).

## 3. Diagnosis and projection

- Full-attention layers are bound by the per-score softmax work, and FA3 reaches only ~55 % of the MUFU floor.
- A kernel that keeps everything in FP8 (Q·Kᵀ and P·V on wgmma, P quantized to E4M3 as FA3 does) and keeps MUFU fed could approach that floor: case 1 424 → ~232 µs, case 7 8151 → ~4622 µs (projection, at 1.98 GHz; the real clock under load is lower).
- Projected score effect (inferred from the sensitivities above): ~+2 % if the full-layer attention reached the MUFU floor. The ncu numbers in 2(c) already said the first version would land well short of that, because the ALU and the issue stream are half busy too.
- Dense tuning: +1 to +2 % per case from algorithm picks and the fused router (projected from the isolated GEMM timings).

## 4. Change

`change.diff` (+1,055 lines) appends two sections to Codex's file and touches two lines of `kernel_fn`:

1. **Dense path** (`DenseLt`, `DENSE_ALGOS`, `FUSED_ALGOS`, `_pooled_norm_route`): explicit cuBLASLt algorithms (tile id + custom option) measured per scored shape; for up to `FUSED_ROUTER_MAX = 4096` tokens, one FP32-output GEMM over [pooler; router_hi; router_lo] followed by a Triton kernel doing the pre-expert RMSNorm and the router's hi+lo sum.
2. **attn8** (`ATT_CUDA_SRC`, `PrefillAttention`; composed by `evidence/dev_build_cand.py` from `dev/attn8.py`), used for full-attention layers only (`ATT_SLIDING = False`; sliding layers keep FA3). Three kernels:
   - `qkv_post`: per-head RMSNorm of q and k, RoPE, the new tokens' K/V written to the FP8 paged cache, q to FP8.
   - `kv_pack`: copies the paged FP8 cache into contiguous 128-key tiles: K as `[128 keys][64 B]` (64-byte swizzle), V transposed as `[72 rows][128 keys]` whose key order is permuted to match the order in which a thread's FP8 P fragment holds the score columns, so P feeds the second wgmma straight from registers. Rows 64-71 are constant FP8 1.0 ("ones rows"): the P·V wgmma (m64n72k32) then returns the softmax row sum l in its last 8 columns for free. Part 1 (keys already in the cache) runs on a side stream beside the QKV GEMM.
   - `attn_fwd`: 132 persistent CTAs × 512 threads. One producer warpgroup streams K / Vᵀ tiles with `cp.async.bulk` into an 8-stage mbarrier ring; three consumer warpgroups (`ATT_CW 3`, `setmaxnreg` 160) each own 8 tokens × the 8 q-heads of one KV head = 64 rows. Per tile: S = Q·Kᵀ (wgmma m64n128k32 e4m3, FP32 accumulators), online max, `exp2` on MUFU (`ATT_POLY 0`), P → E4M3, O += P·Vᵀ. `ATT_PP 1`: the consumer warpgroups issue their wgmmas in turn (ping-pong). Work is split stream-K: every request's tiles are divided evenly over the CTAs, and causal query blocks are dealt short/long alternately (0, n−1, 1, n−2, …) so each CTA's share costs the same. The epilogue fuses SGLang's output RMSNorm × sigmoid(gate).

How the kernel converged (attention-only microbenchmark `evidence/dev_attn_test.py`, case 1 = 753 tokens, ctx 80K, full; "SGLang" = its whole attention path, ~437-441 µs):

| build | log | case 1 attention µs | vs SGLang | what we learned |
|---|---|---|---|---|
| `dev/attn.py`, f32 softmax, polynomial exp | `abl_nosoft.log` | 538.0 | 0.78× | without the softmax: 315 µs |
| attn8 CW3, POLY 1 | `a8_01.log` | 592.7 | 0.71× | without the polynomial: 392.5 µs; CW2 653-666 µs |
| attn8 CW3, POLY 0, PP 1 | `a8_02.log` | 394.8 | 1.05× | case 6 1.05×; sliding 753 0.86× |
| POLY 1/2/3 sweep | `a8_03.log`, `ncu_poly.log` | 435-481 | | XU 73 % at POLY 0; offloading to FMA loses |
| CW2, PP 0-3 | `a8_04.log` | 557-566 | | dead |
| shipped (full layers only) | `a8_05.log`, `a8_06.log` | 381.0 | **1.09×** | case 3 1.06×, case 6 1.17×; case 0 sliding 0.89× |

## 5. Result

| check | result |
|---|---|
| local layer bench, same run, fixed router seed (`evidence/dev_bench.py`; 6 reps, paired with SGLang, cold L2) | Codex → dense-tuned: geomean 1.6270 → 1.6506, **+1.45 %**, all 8 cases +0.5 to +2.0 % (`evidence/dense_bench_01.log`). Dense-tuned → att8c: 1.6225 → 1.6470, **+1.5 %**: case 1 ×1.0135, 3 ×1.0112, 4 ×1.0016, 6 ×1.0502, 7 ×1.0936 (`evidence/bench_att8c.log`) |
| controls in that run | sliding cases run identical code: case 0 ×1.0013, case 2 ×1.0006, but case 5 ×1.0225: the 16K cases carry ~2 % of run-order and power noise in this bench (the bench's ordering bias was found and fixed on 10-25, (private notes, not included):52`) |
| correctness | judge 5/5 tests; our bench `--check` on all scored cases; attention error vs a BF16-q reference: case 1 6.96e-2 (SGLang's FA3 7.13e-2), case 3 7.25e-2 (7.34e-2) (`evidence/a8_01.log`, `evidence/a9_01.log`) |
| judge | 1.5855 → **1.6369** (`evidence/judge_after_att8c_e4e4ea44.txt`). Attention kernels: case 1 FA3 424.3 → attn_fwd 376.6 µs (+ kv_pack 28.2 µs, mostly hidden on the side stream), case 3 1639.5 → 1482.6, case 4 906.4 → 895.6, case 6 4529.0 → 3847.7, case 7 8151.0 → 6714.1 |

Judge per case, Codex → att8c:

| case | ours µs | SGLang µs | ratio |
|---|---|---|---|
| 0 sliding 753 | 651.6 → 652.7 | 1060.2 → **1138.7** | 1.627 → 1.745 |
| 1 full 753 | 1037.4 → 1014.3 | 1454.1 → 1476.9 | 1.402 → 1.456 |
| 2 sliding 3012 | 2023.9 → 1993.5 | 3397.7 → 3411.6 | 1.679 → 1.711 |
| 3 full 3012 | 3512.9 → 3447.8 | 4926.8 → 4841.6 | 1.402 → 1.404 |
| 4 full 16K | 11041.9 → 10290.9 | 18845.4 → 17248.8 | 1.707 → 1.676 |
| 5 sliding 16K | 10015.0 → 10158.5 | 17064.4 → 17483.3 | 1.704 → 1.721 |
| 6 full 16K 32K | 14232.9 → 13268.2 | 21138.0 → 20554.2 | 1.485 → 1.549 |
| 7 full 16K 64K | 18003.2 → 16234.4 | 24965.0 → 24465.6 | 1.387 → 1.507 |

**Read honestly:**
- Case 0's ratio jumped +7.3 % with our time flat (651.6 → 652.7 µs): SGLang ran 7.4 % slower in this run.
- The judge draws a new router for every run, so our unchanged expert kernel moved too: `moe_8_16_1_2_512` 4801 → 4142 µs on case 4 and 4342 → 4521 µs on case 5 (`evidence/judge_*`). Case 4's −751 µs is mostly that.
- Recomputing the score with att8c's times and the Codex run's SGLang medians gives 1.6072 (+1.4 %, computed with the judge's load formula); that number still carries the router-draw noise in our own times.
- The judge CIs (1.562-1.609 and 1.612-1.661) barely separate. The real effect is the same-run bench's **≈ +3.0 %**; the judge's +3.2 % agrees within its ±2-3 % run-to-run spread ((private notes, not included):32`).

**Against the projection.** The attention gained 10-18 % on full layers (in-layer), not the ~1.8× the MUFU floor allowed. 2(c) predicted the gap: with MUFU at 73 % and the ALU at 42 %, the per-tile chain (max, subtract, exp, convert, rescale) runs serially per warpgroup, and half the issue slots are empty for lack of eligible warps.

## 6. Takeaways, and what came next

- **Count the exponentials before the FLOPs.** For FP8 attention at head dim 64, the MUFU `ex2` rate (16/clk/SM) is a tighter floor than the tensor cores (32 scores/clk/SM). FA3 reached ~55 % of it.
- **An FMA-pipe polynomial is not free relief for MUFU** when the ALU and the issue stream are already half busy: every split lost here.
- **Ones rows in Vᵀ give the row sum from the tensor core**, and a permuted Vᵀ lets P go from the accumulators to the second wgmma without shuffles.
- Next, [02](../02_native_moe_f16_softmax_pdl/) stacked the expert-kernel rewrite with two attention changes aimed at the softmax chain (f16 scores, a partial f16 polynomial) and extended attn8 to sliding layers; [03](../03_expq_fp8_exponential/) then removed the exponential from MUFU altogether.

## Reproduce

All you need is torch and an H200. `../tools/` holds the problem spec, `reference.py`, `baseline.py` (SGLang), the correctness gate and the judge-faithful A/B harness.

```bash
cd ../tools
python research/harness/check.py ../01_fp8_attention_full_layers/build.py --edges        # judge tests + 8 scored shapes + nearby edges
python research/judge/replica.py --candidate codex=../01_fp8_attention_full_layers/evidence/base_codex_candidate_native_parallel_lt.py \
       --candidate att8c=../01_fp8_attention_full_layers/build.py --panel 11,22,33 --reps 8          # paired A/B vs SGLang, 3 router draws
```

`evidence/dev_bench.py` and `evidence/dev_attn_test.py` are the era's own bench and attention microbenchmark. They expect the layout of `muse-opt/muse_prefill_layer/` (`reference.py`, `dev/public_cases.json`, `dev/attn8.py`) and were run on Modal through `dev/mrun.sh`.

Originals: `muse-opt/muse_prefill_layer/cand_att8c.py` (= `checks/att8c.py`), `codex_best.py` (= Codex's `candidate_native_parallel_lt.py` in `muse-opt/backups/codex-okc-h200-20261023-20261024-161555.tar.gz`), `dev/attn8.py`, `dev/attn_test.py`, `logs/`. Judge results: `muse-opt/muse_prefill_layer/checks/profiles/att8c.e4e4ea4469c94941.*`; Codex's in the backup tarball (`work/muse_prefill_layer/profiles/candidate_native_parallel_lt.134a1dec36614beb.txt`).
