# okc: problem=muse_prefill_layer
# Third-party code: each segment copied or adapted from another project is marked in place with SPDX-SnippetBegin /
# SPDX-SnippetEnd comments that give its source, copyright, license and our changes. Full license texts are in
# LICENSES/ and the third-party notices in NOTICE, both at the repository root.
# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: SGLang contributors <https://github.com/sgl-project/sglang>
# SPDX-License-Identifier: Apache-2.0
# Adapted from the OKC starter's muse_prefill_layer/tools/baseline.py (SGLang's MuseSpark1xDecoderLayer, sgl-project/sglang @ a55ab7d), module docstring through `kernel_fn`.
# Changes: `_gemm` calls our dense GEMM (`_layer_dense`) instead of `_bf16_gemm_dispatch_impl`; `prepare` is renamed `_sglang_prepare` and no longer builds the RoPE cos/sin table; `_experts` is renamed `_marlin_layer_experts`; `kernel_fn` runs our attention module instead of the QK norm, cache store, FA3 call and output norm gate, and our `_layer_router` instead of the two router GEMMs.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
"""SGLang's decoder layer for Muse Spark Open, as one H200 runs it under TP8/EP8, at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f (MuseSpark1xDecoderLayer.forward), with the same modules:

  qkv_gate projection              bf16_gemm_dispatch (cuBLAS on H200)
  FusedQKNorm (Triton)             per-head RMSNorm of q and k, read from the fused projection,
                                   with the interleaved RoPE fused in on sliding layers
  cache store, q to FP8            as FlashAttentionBackend and set_kv_buffer do for an FP8 cache
                                   (cast, then the store_cache JIT kernel at the step's rows)
  flash_attn_with_kvcache (FA3)    paged, causal, window (2047, 0) on sliding layers
  FusedRMSNormGate (JIT)           per-head output norm x sigmoid(gate)
  o projection                     bf16_gemm_dispatch; then the all-reduce (identity here)
  FusedPostNormResidualPreNorm     post-attention norm + residual gate + pre-FFN norm (JIT);
                                   it also writes the FP32 router input up to 8 tokens
  router                           FP32 F.linear up to 8 tokens, above that two BF16 GEMMs
                                   on a hi/lo split of the weight (MuseSpark1xRouter); on a
                                   side stream, overlapping the pooler
  FusedMuseSparkRouting (JIT)      sigmoid, biased top-8, renormalize (on the side stream)
  pooler                           bf16_gemm_dispatch, replicated on every GPU
  pre-expert norm                  MuseSpark1xRMSNorm (F.rms_norm)
  experts                          moe_align_block_size, two moe_wna16_marlin_gemm (W4A16),
                                   silu_and_mul, Triton post-expert norm weighted sum
                                   (MuseSpark1xFp4Experts._forward_marlin); then the
                                   all-reduce (identity here)
  unpooler                         up to 8 tokens, this GPU's 1/8 slice of the weight, then
                                   the all-gather (identity here: zeros in the other columns);
                                   above that, the whole weight (bf16_gemm_dispatch)
  FusedPostNormResidualPreNorm     post-FFN norm + residual gate + the next layer's pre-attention
                                   norm (JIT)

Differs from SGLang:
  * The step's positions, cache rows, FA3 lengths and FA3 scheduler metadata come in with the
    inputs; SGLang builds them once per step for all layers.
  * All-reduces are the identity; the all-gather is a zero pad of the unpooler's output."""

from typing import TypedDict

import types

import torch
import torch.nn.functional as F
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sgl_kernel.scalar_type import scalar_types
from torch import nn

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor            # bfloat16 [2304, 8192]: q 16x64, k 2x64, v 2x64, output gate 16x64
    o: torch.Tensor                   # bfloat16 [8192, 1024]
    attn_residual_gate: torch.Tensor  # bfloat16 [8192], the post-attention residual gate's weight
    pre_ffn_norm: torch.Tensor        # bfloat16 [8192] (applied as 1 + w)
    router: torch.Tensor              # float32 [256, 8192]
    router_bias: torch.Tensor         # float32 [256], added to the scores only to choose experts
    pooler: torch.Tensor              # bfloat16 [4096, 8192]
    pre_expert_norm: torch.Tensor     # bfloat16 [4096] (applied as 1 + w)
    w13: torch.Tensor                 # uint8 [32, 8192, 2048]: this GPU's experts, gate then up rows, packed E2M1
    w13_sf: torch.Tensor              # float8_e4m3fn [32, 8192, 256]
    w13_gs: torch.Tensor              # float32 [32]
    w2: torch.Tensor                  # uint8 [32, 4096, 2048]
    w2_sf: torch.Tensor               # float8_e4m3fn [32, 4096, 256]
    w2_gs: torch.Tensor               # float32 [32]
    post_expert_norm: torch.Tensor    # bfloat16 [4096] (applied as 1 + w, before normalizing)
    unpooler: torch.Tensor            # bfloat16 [8192, 4096]; up to 8 tokens only rows 0-1023 (this GPU's slice)
    post_ffn_norm: torch.Tensor       # bfloat16 [8192] (applied as 1 + w, before normalizing)
    ffn_residual_gate: torch.Tensor   # bfloat16 [8192]
    next_pre_attn_norm: torch.Tensor  # bfloat16 [8192], the next layer's input norm (applied as 1 + w)


class Config(TypedDict):
    batch: int            # B sequences
    q_len: int            # L: 1 for decode, 8 for a verify block (causal inside it); 200-16384 on muse_prefill_layer
    tokens: int           # T = B * L, request-major
    hidden: int           # 8192
    num_q_heads: int      # 16 per GPU under TP8
    num_kv_heads: int     # 2
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers (interleaved RoPE); 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)
    n_experts: int        # 256
    top_k: int            # 8
    local_experts: int    # 32: this GPU computes experts local_start .. local_start + 31


class Inputs(TypedDict):
    stream: torch.Tensor      # float32 [T, 8192], the residual stream
    attn_input: torch.Tensor  # bfloat16 [T, 8192], the normalized stream (the previous layer's boundary wrote it)
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    positions: torch.Tensor   # int64 [T], each new token's position
    cache_loc: torch.Tensor   # int64 [T], each new token's flat cache row
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule on full-attention decode steps
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 []
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    local_start: torch.Tensor # int32 [1], this GPU's first expert (a multiple of 32; its weights are w13[0], ...)
    config: Config


HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
# MuseSpark1xRouter switches to the split BF16 GEMMs above this many tokens.
ROUTER_FP32_MAX_TOKENS = 8
# Up to this many tokens the unpooler runs on this GPU's slice and is all-gathered.
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0


def _gemm(x, w):
    return _layer_dense(x, w)


def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def _norm(cls, w, size, eps, **kw):
    """A Muse norm module holding w (applied as 1 + w), as the checkpoint loads it."""
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = _param(w)
        norm.precompute_effective_weight()
    return norm


def _residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
    gate.gate = _param(w)
    gate.precompute_coefficients()
    return gate


def _marlin_bank(weights):
    """The local experts in Marlin's layout, via SGLang's own loader step."""
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def _sglang_prepare(weights):
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    pre_ffn = _norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], HIDDEN, RMS_EPS)
    next_pre_attn = _norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], HIDDEN, RMS_EPS)
    post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, POST_NORM_EPS, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], HIDDEN, POST_NORM_EPS,
                     is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    post_expert = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_expert_norm"], D_EXPERT, POST_NORM_EPS,
                        is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    state = {
        "dense": {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
        "router": router.contiguous(),
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64),
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_norm": _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], D_EXPERT, RMS_EPS),
        # MuseSpark1xFp4Experts binds the post-expert norm's effective (1 + w) weight.
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "bank": _marlin_bank(weights),
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
    }
    torch.cuda.synchronize()
    return state


def _marlin_layer_experts(x, topk_weights, topk_ids, state):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize."""
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = bank["expert_map"][topk_ids].contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


def kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    dense = state["dense"]
    # Attention: MuseSpark1xAttention.forward.
    qkv = _gemm(data["attn_input"], dense["qkv_gate"])
    o = optimized_attention(data, qkv, custom_window=CUSTOM_WINDOW).reshape(tokens, hq * d)
    attn_output = _gemm(o, dense["o"])  # all-reduce: identity

    # Post-attention boundary, router and MoE block: MuseSpark1xMoE._forward_experts.
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
    # The router and routing run on a side stream, overlapping the pooler on the main one.
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    expert_input = _gemm(ffn_input, dense["pooler"])
    with torch.cuda.stream(side_stream):
        if split_router:
            logits = _layer_router(ffn_input, state)
        else:
            logits = F.linear(router_input, state["router"])
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)
    # This GPU's experts start at local_start: as SGLang's expert_map on that EP rank.
    local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
    # all-reduce: identity
    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = _gemm(expert_output, dense["unpooler"])

    # Post-FFN boundary, with the next layer's pre-attention norm.
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
# SPDX-SnippetEnd

import torch
import triton
import triton.language as tl


@triton.jit
def _projection_gemv(
    X, W, Y,
    N: tl.constexpr, K: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Stream contiguous weight rows, reusing each input tile across rows."""
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for start in range(0, tl.cdiv(K, BLOCK_K)):
        kk = start * BLOCK_K + cols
        x = tl.load(X + kk, mask=kk < K, other=0).to(tl.float32)
        w = tl.load(
            W + rows[:, None] * K + kk[None, :],
            mask=(rows[:, None] < N) & (kk[None, :] < K),
            other=0,
        ).to(tl.float32)
        acc += w * x[None, :]
    out = tl.sum(acc, axis=1)
    tl.store(Y + rows, out, mask=rows < N)

@triton.jit
def _router_partials(
    X, HI, LO, PART,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    STRIDE_XM: tl.constexpr, STRIDE_XK: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Two BF16 dot products in one launch, sharing activations and addressing."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    part = tl.program_id(2)
    kk = part * BLOCK_K + tl.arange(0, BLOCK_K)
    acc_hi = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_lo = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        x = tl.load(
            X + rows[:, None] * STRIDE_XM + kk[None, :] * STRIDE_XK,
            mask=(rows[:, None] < M) & (kk[None, :] < K),
            other=0,
        )
        hi = tl.load(
            HI + kk[:, None] * N + cols[None, :],
            mask=(kk[:, None] < K) & (cols[None, :] < N),
            other=0,
        )
        lo = tl.load(
            LO + kk[:, None] * N + cols[None, :],
            mask=(kk[:, None] < K) & (cols[None, :] < N),
            other=0,
        )
        acc_hi = tl.dot(x, hi, acc_hi)
        acc_lo = tl.dot(x, lo, acc_lo)
        kk += BLOCK_K * SPLIT_K
    tl.store(
        PART + part * M * N + rows[:, None] * N + cols[None, :],
        acc_hi + acc_lo,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )

@triton.jit
def _router_reduce(
    PART, Y,
    SIZE: tl.constexpr, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr,
):
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLIT_K)
    partials = tl.load(
        PART + splits[:, None] * SIZE + indices[None, :],
        mask=indices[None, :] < SIZE,
        other=0,
    )
    tl.store(Y + indices, tl.sum(partials, axis=0), mask=indices < SIZE)

"""Metadata-selected split-K BF16 GEMM, for small token batches."""

import torch
import triton
import triton.language as tl


@triton.jit
def _dense_partials(X, W, PART,
                    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                    SXM: tl.constexpr, SXK: tl.constexpr,
                    SPLIT_K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    part = tl.program_id(2)
    kk = part * BK + tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, tl.cdiv(K, BK * SPLIT_K)):
        a = tl.load(X + rows[:, None] * SXM + kk[None, :] * SXK,
                    mask=(rows[:, None] < M) & (kk[None, :] < K), other=0)
        b = tl.load(W + cols[None, :] * K + kk[:, None],
                    mask=(cols[None, :] < N) & (kk[:, None] < K), other=0)
        acc = tl.dot(a, b, acc)
        kk += BK * SPLIT_K
    tl.store(PART + part * M * N + rows[:, None] * N + cols[None, :], acc,
             mask=(rows[:, None] < M) & (cols[None, :] < N))


def splitk_dense(x, weight, *, block_n=64):
    m, k = x.shape
    n = weight.shape[0]
    split = 8 if k >= 4096 else 4
    part = torch.empty((split, m, n), dtype=torch.float32, device=x.device)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _dense_partials[(triton.cdiv(m, 16), triton.cdiv(n, block_n), split)](
        x, weight, part, m, n, k, x.stride(0), x.stride(1), split, 16, block_n, 64,
        num_warps=4, num_stages=2,
    )
    _router_reduce[(triton.cdiv(m * n, 256),)](part, out, m * n, split, 256, num_warps=4)
    return out

"""Attention building blocks embedded into each standalone submission."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _fused_qkv_cache(
    QKV, POS, LOC, KC, VC, KS, VS, Q,
    HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
    STRIDE: tl.constexpr, THETA: tl.constexpr,
    QUERY_FACTOR: tl.constexpr, QUERY_FP8: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    dim = tl.arange(0, D)
    is_query = head < HQ
    source_head = tl.where(is_query, head, head - HQ)
    start = tl.where(is_query, source_head * D, HQ * D + source_head * D)
    x = tl.load(QKV + token * STRIDE + start + dim).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + 1.0e-5)
    x = x * inv
    if THETA != 0:
        # libdevice has argument reduction for long contexts; tl.cos/sin's
        # fast approximate instructions are unsuitable for large positions.
        partner = tl.load(QKV + token * STRIDE + start + (dim ^ 1)).to(tl.float32) * inv
        position = tl.load(POS + token).to(tl.float32)
        angle = position * tl.exp(-tl.log(THETA) * ((dim // 2) * 2).to(tl.float32) / D)
        x = x * libdevice.cos(angle) + tl.where((dim & 1) == 0, -partner, partner) * libdevice.sin(angle)
    if is_query:
        # With FA3, preserve its unit-RMS FP8 query and fold the model's
        # multiplier into softmax_scale. The custom path retains BF16 queries.
        if not QUERY_FP8:
            x *= QUERY_FACTOR
        tl.store(Q + (token * HQ + head) * D + dim, x)
    else:
        slot = tl.load(LOC + token)
        k_scale = tl.load(KS)
        v_scale = tl.load(VS)
        # SGLang materializes normalized keys as BF16 before FP8 conversion.
        x = x.to(tl.bfloat16).to(tl.float32)
        v = tl.load(QKV + token * STRIDE + (HQ + HK) * D + source_head * D + dim).to(tl.float32)
        tl.store(KC + (slot * HK + source_head) * D + dim, x / k_scale)
        tl.store(VC + (slot * HK + source_head) * D + dim, v / v_scale)


@triton.jit
def _fused_qkv_cache_grouped(
    QKV, POS, LOC, KC, VC, KS, VS, Q,
    HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
    STRIDE: tl.constexpr, THETA: tl.constexpr,
    QUERY_FACTOR: tl.constexpr, QUERY_FP8: tl.constexpr, BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    heads = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    dims = tl.arange(0, D)
    valid = heads < HQ + HK
    x = tl.load(QKV + token * STRIDE + heads[:, None] * D + dims[None, :],
                mask=valid[:, None], other=0.0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 1) / D + 1.0e-5)
    x *= inv[:, None]
    if THETA != 0:
        partner = tl.load(QKV + token * STRIDE + heads[:, None] * D + (dims[None, :] ^ 1),
                          mask=valid[:, None], other=0.0).to(tl.float32) * inv[:, None]
        position = tl.load(POS + token).to(tl.float32)
        angle = position * tl.exp(-tl.log(THETA) * ((dims // 2) * 2).to(tl.float32) / D)
        x = x * libdevice.cos(angle)[None, :] + tl.where((dims[None, :] & 1) == 0, -partner, partner) * libdevice.sin(angle)[None, :]
    query = x
    if not QUERY_FP8:
        query *= QUERY_FACTOR
    tl.store(Q + (token * HQ + heads[:, None]) * D + dims[None, :], query,
             mask=(heads[:, None] < HQ))
    kv_head = heads - HQ
    valid_kv = valid & (heads >= HQ)
    slot = tl.load(LOC + token)
    k_scale, v_scale = tl.load(KS), tl.load(VS)
    key = x.to(tl.bfloat16).to(tl.float32)
    value = tl.load(QKV + token * STRIDE + (HQ + HK) * D + kv_head[:, None] * D + dims[None, :],
                    mask=valid_kv[:, None], other=0.0).to(tl.float32)
    tl.store(KC + (slot * HK + kv_head[:, None]) * D + dims[None, :], key / k_scale, mask=valid_kv[:, None])
    tl.store(VC + (slot * HK + kv_head[:, None]) * D + dims[None, :], value / v_scale, mask=valid_kv[:, None])


@triton.jit
def _attention_norm_gate(A, QKV, OUT, HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, STRIDE: tl.constexpr):
    head_row = tl.program_id(0)
    token = head_row // HQ
    head = head_row % HQ
    dim = tl.arange(0, D)
    x = tl.load(A + head_row * D + dim).to(tl.float32)
    gate = tl.load(QKV + token * STRIDE + (HQ + 2 * HK) * D + head * D + dim).to(tl.float32)
    x *= tl.rsqrt(tl.sum(x * x, 0) / D + 1.0e-5)
    tl.store(OUT + head_row * D + dim, x / (1.0 + tl.exp(-gate)))


@triton.jit
def _attention_norm_gate_grouped(A, QKV, OUT,
                                 TOKENS: tl.constexpr, HQ: tl.constexpr, HK: tl.constexpr,
                                 D: tl.constexpr, STRIDE: tl.constexpr, BLOCK_H: tl.constexpr):
    heads = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    dims = tl.arange(0, D)
    valid = heads < TOKENS * HQ
    x = tl.load(A + heads[:, None] * D + dims[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
    gate = tl.load(QKV + (heads[:, None] // HQ) * STRIDE + (HQ + 2 * HK) * D + (heads[:, None] % HQ) * D + dims[None, :],
                   mask=valid[:, None], other=0.0).to(tl.float32)
    x *= tl.rsqrt(tl.sum(x * x, 1) / D + 1.0e-5)[:, None]
    tl.store(OUT + heads[:, None] * D + dims[None, :], x / (1.0 + tl.exp(-gate)), mask=valid[:, None])


@triton.jit
def _paged_window_partials(
    Q, KC, VC, KS, VS, TABLE, POS, PART, LSE,
    HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
    LENGTH: tl.constexpr, PAGE: tl.constexpr, TABLE_STRIDE: tl.constexpr,
    WINDOW: tl.constexpr, SM_SCALE: tl.constexpr,
    SPLITS: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_Q: tl.constexpr,
):
    # One program shares a KV head across its grouped query heads and verify
    # tokens. Splits cover the entire union of their causal windows.
    batch = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    group: tl.constexpr = HQ // HK
    rows = tl.arange(0, BLOCK_Q)
    q_token = rows // group
    q_head = kv_head * group + rows % group
    dims = tl.arange(0, D)
    first_pos = tl.load(POS + batch * LENGTH)
    positions = first_pos + q_token
    start = tl.maximum(0, first_pos - WINDOW + 1)
    extent: tl.constexpr = WINDOW + LENGTH - 1
    split_width: tl.constexpr = triton.cdiv(extent, SPLITS * BLOCK_S) * BLOCK_S
    q = tl.load(Q + ((batch * LENGTH + q_token[:, None]) * HQ + q_head[:, None]) * D + dims[None, :],
                mask=q_token[:, None] < LENGTH, other=0.0)
    k_scale = tl.load(KS)
    v_scale = tl.load(VS)
    maximum = tl.full((BLOCK_Q,), float("-inf"), tl.float32)
    denominator = tl.zeros((BLOCK_Q,), tl.float32)
    accumulator = tl.zeros((BLOCK_Q, D), tl.float32)
    for offset in range(0, split_width, BLOCK_S):
        logical = start + split * split_width + offset + tl.arange(0, BLOCK_S)
        valid_key = logical <= first_pos + LENGTH - 1
        pages = tl.load(TABLE + batch * TABLE_STRIDE + logical // PAGE, mask=valid_key, other=-1)
        valid_key &= pages >= 0
        slots = tl.maximum(pages, 0) * PAGE + logical % PAGE
        key = tl.load(KC + (slots[None, :] * HK + kv_head) * D + dims[:, None],
                      mask=valid_key[None, :], other=0.0).to(tl.bfloat16)
        value = tl.load(VC + (slots[:, None] * HK + kv_head) * D + dims[None, :],
                        mask=valid_key[:, None], other=0.0).to(tl.bfloat16)
        scores = tl.dot(q, key).to(tl.float32) * (SM_SCALE * k_scale)
        visible = valid_key[None, :] & (logical[None, :] <= positions[:, None])
        visible &= logical[None, :] > positions[:, None] - WINDOW
        visible &= q_token[:, None] < LENGTH
        scores = tl.where(visible, scores, float("-inf"))
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        finite_max = tl.where(next_max == float("-inf"), 0.0, next_max)
        alpha = tl.exp(maximum - finite_max)
        probabilities = tl.exp(scores - finite_max[:, None])
        denominator = denominator * alpha + tl.sum(probabilities, 1)
        # FA-style BF16 probability operands, FP32 accumulation. Every key
        # participates; no probability or block pruning is used.
        accumulator = accumulator * alpha[:, None] + tl.dot(probabilities.to(tl.bfloat16), value)
        maximum = next_max
    norm = tl.where(denominator > 0, denominator, 1.0)
    result = accumulator * (v_scale / norm[:, None])
    base = ((batch * HK + kv_head) * SPLITS + split)
    tl.store(PART + (base * BLOCK_Q + rows[:, None]) * D + dims[None, :], result,
             mask=q_token[:, None] < LENGTH)
    logsum = tl.where(denominator > 0, maximum + tl.log(denominator), float("-inf"))
    tl.store(LSE + base * BLOCK_Q + rows, logsum, mask=q_token < LENGTH)


@triton.jit
def _window_reduce_gate(
    PART, LSE, QKV, OUT,
    HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
    LENGTH: tl.constexpr, STRIDE: tl.constexpr, SPLITS: tl.constexpr, BLOCK_Q: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    batch = token // LENGTH
    q_token = token % LENGTH
    group: tl.constexpr = HQ // HK
    kv_head = head // group
    row = q_token * group + head % group
    split = tl.arange(0, SPLITS)
    dims = tl.arange(0, D)
    base = ((batch * HK + kv_head) * SPLITS + split)
    logsum = tl.load(LSE + base * BLOCK_Q + row)
    weights = tl.exp(logsum - tl.max(logsum, 0))
    part = tl.load(PART + (base[:, None] * BLOCK_Q + row) * D + dims[None, :])
    x = tl.sum(weights[:, None] * part, 0) / tl.sum(weights, 0)
    # Match the BF16 attention activation boundary before output RMSNorm.
    x = x.to(tl.bfloat16).to(tl.float32)
    x *= tl.rsqrt(tl.sum(x * x, 0) / D + 1.0e-5)
    gate = tl.load(QKV + token * STRIDE + (HQ + 2 * HK) * D + head * D + dims).to(tl.float32)
    tl.store(OUT + (token * HQ + head) * D + dims, x / (1.0 + tl.exp(-gate)))


def optimized_attention(data, qkv, *, custom_window=False):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hk, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = qkv.reshape(tokens, -1)
    use_window = custom_window and cfg["window"] > 0 and length <= 8
    query_dtype = torch.bfloat16 if use_window else torch.float8_e4m3fn
    q = torch.empty((tokens, hq, d), device=qkv.device, dtype=query_dtype)
    preprocess = _fused_qkv_cache_grouped if tokens >= 32 else _fused_qkv_cache
    preprocess_grid = (tokens, triton.cdiv(hq + hk, 32)) if tokens >= 32 else (tokens, hq + hk)
    preprocess_extra = {"BLOCK_H": 32} if tokens >= 32 else {}
    preprocess[preprocess_grid](
        qkv, data["positions"], data["cache_loc"], data["k_cache"], data["v_cache"],
        data["k_scale"], data["v_scale"], q,
        hq, hk, d, qkv.stride(0), cfg["rope_theta"], cfg["qk_scale_factor"], not use_window,
        **preprocess_extra, num_warps=4,
    )
    result = torch.empty((tokens, hq, d), device=qkv.device, dtype=torch.bfloat16)
    if use_window:
        # Plenty of independent CTAs for one request without flooding large
        # batches with tiny partitions. This is metadata-only dispatch.
        splits = 8 if batch <= 4 else 4
        block_q = max(16, triton.next_power_of_2(length * (hq // hk)))
        partials = torch.empty((batch, hk, splits, block_q, d), device=qkv.device, dtype=torch.float32)
        lse = torch.empty((batch, hk, splits, block_q), device=qkv.device, dtype=torch.float32)
        _paged_window_partials[(batch, hk, splits)](
            q, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
            data["page_table"], data["positions"], partials, lse,
            hq, hk, d, length, cfg["page_size"], data["page_table"].stride(0),
            cfg["window"], cfg["sm_scale"], splits, 64, block_q,
            num_warps=4 if block_q <= 32 else 8, num_stages=2,
        )
        _window_reduce_gate[(tokens, hq)](
            partials, lse, qkv, result, hq, hk, d, length, qkv.stride(0), splits, block_q,
            num_warps=4,
        )
    else:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
        descale = {}
        if cfg["kv_scale"] != 1.0:
            descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hk),
                           v_descale=data["v_scale"].reshape(1, 1).expand(batch, hk))
        window = cfg["window"]
        a = flash_attn_with_kvcache(
            q=q, k_cache=data["k_cache"], v_cache=data["v_cache"],
            page_table=data["page_table"], cache_seqlens=data["cache_seqlens"],
            cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=length,
            softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
            window_size=(window - 1, 0) if window else (-1, -1),
            scheduler_metadata=data["scheduler_metadata"], **descale,
        )
        if isinstance(a, tuple):
            a = a[0]
        if tokens >= 32:
            _attention_norm_gate_grouped[(triton.cdiv(tokens * hq, 16),)](
                a, qkv, result, tokens, hq, hk, d, qkv.stride(0), 16, num_warps=4)
        else:
            _attention_norm_gate[(tokens * hq,)](
                a, qkv, result, hq, hk, d, qkv.stride(0), num_warps=4)
    return result.reshape(batch, length, hq * d)

"""NVFP4 expert kernels. Original packed weights and scales stay unchanged."""

import torch
import triton
import triton.language as tl


@triton.jit
def _e2m1(code):
    mag = code & 7
    value = tl.where(mag < 4, mag.to(tl.float32) * 0.5,
                     tl.where(mag == 4, 2.0, tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0))))
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def _small_expert_gate_up(
    X, IDS, W, SF, GS, H,
    WIDTH: tl.constexpr, INNER: tl.constexpr, TOPK: tl.constexpr,
    LOCAL_COUNT: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    route = tl.program_id(0)
    rows = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    expert = tl.load(IDS + route)
    if (expert >= 0) & (expert < LOCAL_COUNT):
        cols = tl.arange(0, BLOCK_K)
        ag = tl.zeros((BLOCK_N, BLOCK_K), tl.float32)
        au = tl.zeros((BLOCK_N, BLOCK_K), tl.float32)
        for offset in range(0, tl.cdiv(WIDTH, BLOCK_K)):
            kk = offset * BLOCK_K + cols
            x = tl.load(X + (route // TOPK) * WIDTH + kk, mask=kk < WIDTH, other=0.0).to(tl.float32)
            base = expert * (2 * INNER) * (WIDTH // 2)
            gate = tl.load(W + base + rows[:, None] * (WIDTH // 2) + kk[None, :] // 2,
                           mask=(rows[:, None] < INNER) & (kk[None, :] < WIDTH), other=0.0)
            up = tl.load(W + base + (rows[:, None] + INNER) * (WIDTH // 2) + kk[None, :] // 2,
                         mask=(rows[:, None] < INNER) & (kk[None, :] < WIDTH), other=0.0)
            shift = (kk[None, :] & 1) * 4
            sg = tl.load(SF + expert * (2 * INNER) * (WIDTH // 16) + rows[:, None] * (WIDTH // 16) + kk[None, :] // 16,
                         mask=(rows[:, None] < INNER) & (kk[None, :] < WIDTH), other=0.0).to(tl.float32)
            su = tl.load(SF + expert * (2 * INNER) * (WIDTH // 16) + (rows[:, None] + INNER) * (WIDTH // 16) + kk[None, :] // 16,
                         mask=(rows[:, None] < INNER) & (kk[None, :] < WIDTH), other=0.0).to(tl.float32)
            ag += _e2m1((gate >> shift) & 15) * sg * x[None, :]
            au += _e2m1((up >> shift) & 15) * su * x[None, :]
        scale = tl.load(GS + expert)
        gate_value = (tl.sum(ag, 1) * scale).to(tl.bfloat16).to(tl.float32)
        up_value = (tl.sum(au, 1) * scale).to(tl.bfloat16).to(tl.float32)
        result = gate_value / (1.0 + tl.exp(-gate_value)) * up_value
        tl.store(H + route * INNER + rows, result, mask=rows < INNER)
    else:
        tl.store(H + route * INNER + rows, 0.0, mask=rows < INNER)


@triton.jit
def _small_expert_down(
    H, IDS, W, SF, GS, Y,
    WIDTH: tl.constexpr, INNER: tl.constexpr, LOCAL_COUNT: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    route = tl.program_id(0)
    rows = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    expert = tl.load(IDS + route)
    if (expert >= 0) & (expert < LOCAL_COUNT):
        cols = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_N, BLOCK_K), tl.float32)
        for offset in range(0, tl.cdiv(INNER, BLOCK_K)):
            kk = offset * BLOCK_K + cols
            x = tl.load(H + route * INNER + kk, mask=kk < INNER, other=0.0).to(tl.float32)
            packed = tl.load(W + expert * WIDTH * (INNER // 2) + rows[:, None] * (INNER // 2) + kk[None, :] // 2,
                             mask=(rows[:, None] < WIDTH) & (kk[None, :] < INNER), other=0.0)
            sf = tl.load(SF + expert * WIDTH * (INNER // 16) + rows[:, None] * (INNER // 16) + kk[None, :] // 16,
                         mask=(rows[:, None] < WIDTH) & (kk[None, :] < INNER), other=0.0).to(tl.float32)
            code = (packed >> ((kk[None, :] & 1) * 4)) & 15
            accumulator += _e2m1(code) * sf * x[None, :]
        result = tl.sum(accumulator, 1) * tl.load(GS + expert)
        tl.store(Y + route * WIDTH + rows, result, mask=rows < WIDTH)
    else:
        tl.store(Y + route * WIDTH + rows, 0.0, mask=rows < WIDTH)


@triton.jit
def _expert_norm_scales(Y, IDS, NW, SCALES,
                        WIDTH: tl.constexpr, LOCAL_COUNT: tl.constexpr, BLOCK: tl.constexpr):
    route = tl.program_id(0)
    expert = tl.load(IDS + route)
    dims = tl.arange(0, BLOCK)
    valid = (expert >= 0) & (expert < LOCAL_COUNT)
    y = tl.load(Y + route * WIDTH + dims, mask=(dims < WIDTH) & valid, other=0.0).to(tl.float32)
    weight = tl.load(NW + dims, mask=dims < WIDTH, other=0.0).to(tl.float32)
    # Effective norm weights have the same BF16 rounding as SGLang.
    y = y * weight
    # The post-expert norm materializes its scaled input in BF16.
    y = y.to(tl.bfloat16).to(tl.float32)
    scale = tl.rsqrt(tl.sum(y * y, 0) / WIDTH + 1.0e-8)
    tl.store(SCALES + route, tl.where(valid, scale, 0.0))


@triton.jit
def _expert_weighted_sum(Y, IDS, RW, NW, SCALES, OUT,
                         WIDTH: tl.constexpr, TOPK: tl.constexpr, LOCAL_COUNT: tl.constexpr,
                         BLOCK: tl.constexpr, ROUTE_BLOCK: tl.constexpr):
    token = tl.program_id(0)
    dims = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slots = tl.arange(0, ROUTE_BLOCK)
    route = token * TOPK + slots
    expert = tl.load(IDS + route, mask=slots < TOPK, other=-1)
    valid = (slots < TOPK) & (expert >= 0) & (expert < LOCAL_COUNT)
    y = tl.load(Y + route[:, None] * WIDTH + dims[None, :], mask=valid[:, None] & (dims[None, :] < WIDTH), other=0.0).to(tl.float32)
    nw = tl.load(NW + dims, mask=dims < WIDTH, other=0.0).to(tl.float32)
    y = (y * nw[None, :]).to(tl.bfloat16).to(tl.float32)
    scale = tl.load(SCALES + route, mask=valid, other=0.0)
    weight = tl.load(RW + route, mask=valid, other=0.0)
    value = tl.sum(y * (scale * weight)[:, None], 0)
    tl.store(OUT + token * WIDTH + dims, value, mask=dims < WIDTH)


def small_experts(x, local_ids, route_weights, packed, norm_weight, *, block_n=4, block_k=512):
    m, width = x.shape
    topk = local_ids.shape[1]
    inner = packed["w2"].shape[2] * 2
    local_count = packed["w2"].shape[0]
    h = torch.empty((m * topk, inner), device=x.device, dtype=torch.bfloat16)
    y = torch.empty((m * topk, width), device=x.device, dtype=torch.bfloat16)
    _small_expert_gate_up[(m * topk, triton.cdiv(inner, block_n))](
        x, local_ids, packed["w13"], packed["w13_sf"], packed["w13_gs"], h,
        width, inner, topk, local_count, block_n, block_k, num_warps=4,
    )
    _small_expert_down[(m * topk, triton.cdiv(width, block_n))](
        h, local_ids, packed["w2"], packed["w2_sf"], packed["w2_gs"], y,
        width, inner, local_count, block_n, block_k, num_warps=4,
    )
    return finish_experts(y, local_ids, route_weights, norm_weight, local_count)


def finish_experts(y, local_ids, route_weights, norm_weight, local_count):
    m, topk = local_ids.shape
    width = y.shape[-1]
    scales = torch.empty((m, topk), device=y.device, dtype=torch.float32)
    out = torch.empty((m, width), device=y.device, dtype=torch.bfloat16)
    _expert_norm_scales[(m * topk,)](y, local_ids, norm_weight, scales,
                                    width, local_count, triton.next_power_of_2(width), num_warps=4)
    _expert_weighted_sum[(m, triton.cdiv(width, 256))](
        y, local_ids, route_weights, norm_weight, scales, out,
        width, topk, local_count, 256, triton.next_power_of_2(topk), num_warps=4,
    )
    return out


@triton.jit
def _grouped_gate_up(
    X, SORTED, EXPERTS, PADDED, W, SF, GS, H,
    M: tl.constexpr, WIDTH: tl.constexpr, INNER: tl.constexpr, TOPK: tl.constexpr, LOCAL_COUNT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    block = tl.program_id(0)
    expert = tl.load(EXPERTS + block)
    if (block * BLOCK_M < tl.load(PADDED)) & (expert >= 0) & (expert < LOCAL_COUNT):
        slots = tl.load(SORTED + block * BLOCK_M + tl.arange(0, BLOCK_M))
        rows = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        kk = tl.arange(0, BLOCK_K)
        gate_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        up_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for offset in range(0, tl.cdiv(WIDTH, BLOCK_K)):
            cols = offset * BLOCK_K + kk
            x = tl.load(X + (slots[:, None] // TOPK) * WIDTH + cols[None, :],
                        mask=(slots[:, None] < M * TOPK) & (cols[None, :] < WIDTH), other=0.0)
            base = expert * (2 * INNER) * (WIDTH // 2)
            pg = tl.load(W + base + rows[None, :] * (WIDTH // 2) + cols[:, None] // 2,
                         mask=(rows[None, :] < INNER) & (cols[:, None] < WIDTH), other=0.0)
            pu = tl.load(W + base + (rows[None, :] + INNER) * (WIDTH // 2) + cols[:, None] // 2,
                         mask=(rows[None, :] < INNER) & (cols[:, None] < WIDTH), other=0.0)
            scale_base = expert * (2 * INNER) * (WIDTH // 16)
            sg = tl.load(SF + scale_base + rows[None, :] * (WIDTH // 16) + cols[:, None] // 16,
                         mask=(rows[None, :] < INNER) & (cols[:, None] < WIDTH), other=0.0).to(tl.float32)
            su = tl.load(SF + scale_base + (rows[None, :] + INNER) * (WIDTH // 16) + cols[:, None] // 16,
                         mask=(rows[None, :] < INNER) & (cols[:, None] < WIDTH), other=0.0).to(tl.float32)
            shift = (cols[:, None] & 1) * 4
            # E2M1 * E4M3 is exactly representable in BF16. The original FP32
            # expert-wide scale is applied after FP32 dot accumulation.
            wg = (_e2m1((pg >> shift) & 15) * sg).to(tl.bfloat16)
            wu = (_e2m1((pu >> shift) & 15) * su).to(tl.bfloat16)
            gate_acc = tl.dot(x, wg, gate_acc)
            up_acc = tl.dot(x, wu, up_acc)
        scale = tl.load(GS + expert)
        gate = (gate_acc * scale).to(tl.bfloat16).to(tl.float32)
        up = (up_acc * scale).to(tl.bfloat16).to(tl.float32)
        value = gate / (1.0 + tl.exp(-gate)) * up
        tl.store(H + slots[:, None] * INNER + rows[None, :], value,
                 mask=(slots[:, None] < M * TOPK) & (rows[None, :] < INNER))


@triton.jit
def _grouped_down(
    H, SORTED, EXPERTS, PADDED, W, SF, GS, Y,
    M: tl.constexpr, WIDTH: tl.constexpr, INNER: tl.constexpr, TOPK: tl.constexpr, LOCAL_COUNT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    block = tl.program_id(0)
    expert = tl.load(EXPERTS + block)
    if (block * BLOCK_M < tl.load(PADDED)) & (expert >= 0) & (expert < LOCAL_COUNT):
        slots = tl.load(SORTED + block * BLOCK_M + tl.arange(0, BLOCK_M))
        rows = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        kk = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for offset in range(0, tl.cdiv(INNER, BLOCK_K)):
            cols = offset * BLOCK_K + kk
            h = tl.load(H + slots[:, None] * INNER + cols[None, :],
                        mask=(slots[:, None] < M * TOPK) & (cols[None, :] < INNER), other=0.0)
            p = tl.load(W + expert * WIDTH * (INNER // 2) + rows[None, :] * (INNER // 2) + cols[:, None] // 2,
                        mask=(rows[None, :] < WIDTH) & (cols[:, None] < INNER), other=0.0)
            sf = tl.load(SF + expert * WIDTH * (INNER // 16) + rows[None, :] * (INNER // 16) + cols[:, None] // 16,
                         mask=(rows[None, :] < WIDTH) & (cols[:, None] < INNER), other=0.0).to(tl.float32)
            w = (_e2m1((p >> ((cols[:, None] & 1) * 4)) & 15) * sf).to(tl.bfloat16)
            accumulator = tl.dot(h, w, accumulator)
        value = accumulator * tl.load(GS + expert)
        tl.store(Y + slots[:, None] * WIDTH + rows[None, :], value,
                 mask=(slots[:, None] < M * TOPK) & (rows[None, :] < WIDTH))


def grouped_experts(x, local_ids, route_weights, packed, norm_weight, *, block_m=32, block_n=64, block_k=64):
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    m, width = x.shape
    topk = local_ids.shape[1]
    inner = packed["w2"].shape[2] * 2
    local_count = packed["w2"].shape[0]
    local_ids = torch.where((local_ids >= 0) & (local_ids < local_count), local_ids, -1).to(torch.int32)
    sorted_ids, block_experts, padded = moe_align_block_size(local_ids, block_m, 256)
    h = torch.empty((m * topk, inner), device=x.device, dtype=torch.bfloat16)
    y = torch.empty((m * topk, width), device=x.device, dtype=torch.bfloat16)
    # Capacity comes from metadata. The GPU's n_padded scalar masks inactive
    # blocks, so no routing-dependent host synchronization is needed.
    blocks = block_experts.numel()
    options = dict(num_warps=4, num_stages=2)
    _grouped_gate_up[(blocks, triton.cdiv(inner, block_n))](
        x, sorted_ids, block_experts, padded, packed["w13"], packed["w13_sf"], packed["w13_gs"], h,
        m, width, inner, topk, local_count, block_m, block_n, block_k, **options,
    )
    _grouped_down[(blocks, triton.cdiv(width, block_n))](
        h, sorted_ids, block_experts, padded, packed["w2"], packed["w2_sf"], packed["w2_gs"], y,
        m, width, inner, topk, local_count, block_m, block_n, block_k, **options,
    )
    return finish_experts(y, local_ids, route_weights, norm_weight, local_count)

def _fused_layer_prepare(weights):
    packed = {
        name: weights[name].clone()
        for name in ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs")
    }
    state = _sglang_prepare(weights)
    state["packed"] = packed
    return state


def _native_layer_dense_without_lt(x, w):
    m, k = x.shape
    n = w.shape[0]
    if m == 1 and x.stride(1) == 1:
        out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        _projection_gemv[(triton.cdiv(n, 4),)](x, w, out, n, k, 4, 1024, num_warps=4)
        return out
    if DENSE_SPLITK and 1 < m <= 32:
        return splitk_dense(x, w)
    return _bf16_gemm_dispatch_impl(x, w, None)


def _layer_router(x, state):
    m, k = x.shape
    n = state["router"].shape[0]
    if m > 256:
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        return acc + torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    split = 16 if m <= 32 else 4 if m <= 128 else 2
    part = torch.empty((split, m, n), device=x.device, dtype=torch.float32)
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    _router_partials[(triton.cdiv(m, 16), triton.cdiv(n, 32), split)](
        x, state["router_hi_t"], state["router_lo_t"], part,
        m, n, k, x.stride(0), x.stride(1), split, 16, 32, 64, num_warps=4, num_stages=2)
    _router_reduce[(triton.cdiv(m*n, 256),)](part, out, m*n, split, 256, num_warps=4)
    return out


def _fallback_layer_experts(x, topk_weights, topk_ids, state):
    if x.shape[0] <= 8:
        local_ids = state["bank"]["expert_map"][topk_ids].contiguous()
        return small_experts(x, local_ids, topk_weights, state["packed"], state["post_expert_w"])
    if GROUPED_EXPERTS and x.shape[0] >= 128:
        local_ids = state["bank"]["expert_map"][topk_ids].contiguous()
        return grouped_experts(x, local_ids, topk_weights, state["packed"], state["post_expert_w"], block_m=GROUPED_M)
    return _marlin_layer_experts(x, topk_weights, topk_ids, state)

CUSTOM_WINDOW = False
DENSE_SPLITK = False
GROUPED_EXPERTS = False
GROUPED_M = 32

# Parallel fresh route metadata and independent activation gather on large prefill.
# Separate fresh per-token normalization and route-weighted combine after native expert GEMMs.
# 128-token WGMMA experiment with bounded two-pass epilogue storage.
# Wider 96-token WGMMA prefill experiment; exact NVFP4 storage and BF16 activations.
# Large-prefill experiment: separate per-call GPU routing; native arithmetic retained.
# Derived from the user's W4A16 submission; SHA256 fe79ea76dea5291511ebdae7d69372072514d6db258a3bcc135dcf18ea6fe9bd.
"""Routed NVFP4 experts on Hopper in one persistent CUDA kernel (compiled in-process with NVRTC).

prepare() permutes each local expert's packed E2M1 codes and E4M3 block scales into the order the
kernel's TMA stages and mma.sync fragments read them: a lossless bit and byte permutation, the
same NVFP4 values in the same number of bytes. Nothing is dequantized ahead of time.

One launch per call, one CTA per SM:
  routing   every CTA builds the same per-expert pair lists and token tiles (largest experts
            first) in shared memory from topk_ids, deterministically.
  experts   a dynamic queue of items: (token tile, block of weight rows, k-split) of the gate/up
            GEMM, then of the down GEMM. A producer warp streams an item's weights, scales and
            activation rows into a shared-memory ring with cp.async.bulk; eight consumer warps
            unpack E2M1 to BF16 in registers (exact: E2M1 x E4M3 fits BF16), run mma.sync
            m16n8k16 with FP32 accumulation and apply the FP32 global scale after. Gate/up items
            end with SwiGLU and write the BF16 intermediate; a down item streams its weights,
            waits for its tile's gate/up items, then writes FP32 partial outputs.
  combine   once every down item is done: per token, each local expert output's RMSNorm
            (scaled by 1 + w first), weighted by its routing weight and summed, in BF16.
Work counters live in a small workspace that the last CTA to finish resets for the next call.
"""

import ctypes
from typing import TypedDict

import torch
import triton
import triton.language as tl


class NativeExpertWeights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class NativeExpertConfig(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class NativeExpertInputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: NativeExpertConfig


H = 4096
I = 4096
TOPK = 8
NLOCAL = 32
MAX_M = 16384        # larger shapes remain supported by bounded fresh chunks
ONE_TOKEN = True     # M = 1: the fused one-token kernel
NS_MAX = 4           # ring stages: as many as shared memory holds, at most this
CTR_WORDS = 131072    # workspace counters: [0] queue head, [1] down items done, [2] CTAs done, [3] gathered, [8..] per tile x k chunk
SMEM_MAX = 232448

CUDA_SRC = r"""
#define MOE_NO_COMBINE

// the main kernel streams weights with plain bulk copies unless MOE_TMAP_MAIN (tensor maps)
#ifndef MOE_TMAP_MAIN
#define MOE_NO_TMAP
#endif
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned long long u64;
#define DEVI __device__ __forceinline__

struct __align__(16) StageInfo { int item, phase, tile, rb, ks, st, last, n, p0, e, pad0, pad1; };

DEVI u32 smem_addr(const void* p) {
  u32 r;
  asm volatile("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(r) : "l"(p));
  return r;
}
DEVI float u2f(u32 x) { float f; asm("mov.b32 %0, %1;" : "=f"(f) : "r"(x)); return f; }
DEVI void mbar_init(u32 bar, u32 count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
DEVI void mbar_arrive_tx(u32 bar, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}
DEVI void mbar_arrive(u32 bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory"); }
DEVI bool mbar_try(u32 bar, u32 parity) {
  u32 ok;
  asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
// try_wait with a suspend-time hint (ns): the thread sleeps until the phase completes instead of
// re-polling, so a waiting warp does not take issue slots from the consumers on its scheduler.
#ifndef MOE_SUSPEND
#define MOE_SUSPEND 1000000
#endif
DEVI bool mbar_try_s(u32 bar, u32 parity) {
  u32 ok;
  asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3; selp.u32 %0, 1, 0, p; }"
               : "=r"(ok) : "r"(bar), "r"(parity), "n"(MOE_SUSPEND) : "memory");
  return ok != 0;
}
#ifdef MOE_SUSPEND_ALL
DEVI void mbar_wait(u32 bar, u32 parity) { while (!mbar_try_s(bar, parity)) {} }
#else
DEVI void mbar_wait(u32 bar, u32 parity) { while (!mbar_try(bar, parity)) {} }
#endif
// for the producer and signaler warps, which share schedulers with consumer warps
DEVI void mbar_sleep(u32 bar, u32 parity) {
#ifdef MOE_NO_SUSPEND_PS
  while (!mbar_try(bar, parity)) {}
#else
  while (!mbar_try_s(bar, parity)) {}
#endif
}
DEVI void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar, u64 pol) {
#ifdef MOE_NO_HINT
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
#else
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
#endif
}
// One tiled tensor-map copy (4-D box) into shared memory, completing on an mbarrier.
DEVI void tma_4d(u32 dst, u64 tmap, int c0, int c1, int c2, int c3, u32 bar, u64 pol) {
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
               :: "r"(dst), "l"(tmap), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(bar), "l"(pol) : "memory");
}
DEVI void prefetch_l2(const void* src, u32 bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(src), "r"(bytes) : "memory");
}
DEVI u64 policy_evict_first() { u64 p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p; }
DEVI u64 policy_evict_last() { u64 p; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p)); return p; }
DEVI int ld_acquire(const int* p) {
  int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
DEVI void red_release_add(int* p, int v) {
  asm volatile("red.release.gpu.global.add.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
DEVI u32 gtimer() { u32 t; asm volatile("mov.u32 %0, %globaltimer_lo;" : "=r"(t)); return t; }
#ifdef MOE_TRACE
// event trace: per CTA, consumer thread 0 at [0, 256) and producer lane 0 at [256, 512) words of
// ctr + 32768; word 0 of each half counts its events, then (code << 24 | item, globaltimer) pairs
__shared__ int tr_n[2];  // events so far per half (shared memory: no global round trip per event)
DEVI void trace_ev(int* ctr, int half, int code, int item) {
  int* b = ctr + 32768 + blockIdx.x * 512 + half * 256;
  const int n = code == 0 ? 0 : tr_n[half];
  if (code == 0) tr_n[1] = 0;
  // the producer half's words from 192 on hold its startup stamps
  if (n < (half ? 95 : 127)) { b[1 + 2 * n] = (code << 24) | (item & 0xFFFFFF); b[2 + 2 * n] = (int)gtimer(); b[0] = n + 1; tr_n[half] = n + 1; }
}
#define MOE_TR(half, code, item) trace_ev(P.ctr, half, code, item)
#else
#define MOE_TR(half, code, item)
#endif
DEVI u32 ctimer() { u64 t; asm volatile("mov.u64 %0, %clock64;" : "=l"(t)); return (u32)(t >> 4); }
DEVI void consumer_sync() { asm volatile("bar.sync 1, 256;" ::: "memory"); }
DEVI void group_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
#ifdef MOE_HMUL_NONE
DEVI u32 bmul2(u32 a, u32 b) { return a; }
#elif defined(MOE_NO_HMUL)
DEVI u32 bmul2(u32 a, u32 b) { return a ^ b; }
#elif defined(MOE_BFMA)
// a * b as a fused multiply-add with a -0 addend: the same bits as the product (zeros keep their sign)
__constant__ u32 c_nz2 = 0x80008000u;
DEVI u32 bmul2(u32 a, u32 b) { u32 d; asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c_nz2)); return d; }
#else
DEVI u32 bmul2(u32 a, u32 b) { u32 d; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
#endif
DEVI u16 f2bf(float f) { u16 r; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f)); return r; }
DEVI void mma16816(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__constant__ u32 c_s16 = 16u;
// Scale bytes w = [s_g(kk), s_g8(kk), s_g(kk+1), s_g8(kk+1)]: BF16 pair (s, s) = s * 2^118.
#ifdef MOE_FAST_SCALES
// Every local scale is zero or a normal, positive E4M3 (checked in prepare): its exponent and
// mantissa bits land in BF16's with the exponent offset by 238 = 2^118 / 2^-120 rebiased.
// IM: the shift-and-add as a multiply-add on the FMA pipe (pays off when few MMAs share it)
template <int K, bool IM = false> DEVI u32 scale_of(u32 w) {
#ifdef MOE_NO_SCALE
  return w + K;
#endif
  u32 x;
  asm("prmt.b32 %0, %1, 0, %2;" : "=r"(x) : "r"(w), "n"(0x4040 | (K << 8) | K));
  return x * (IM ? c_s16 : 16u) + 0x77007700u;
}
#else
DEVI u32 scale_pair(u32 b);
template <int K, bool IM = false> DEVI u32 scale_of(u32 w) { return scale_pair((w >> (8 * K)) & 0xFF); }
#endif
// E4M3 block scale -> BF16 pair (s, s) holding s * 2^118 (exact; subnormal scales included).
DEVI u32 scale_pair(u32 b) {
#ifdef MOE_NO_DEQ
  return b * 0x00100010u;
#endif
  float f = u2f(((b & 0x80u) << 24) | ((b & 0x7Fu) << 20));  // s * 2^-120
  f = f * 6.6461399789245794e35f;                             // * 2^119
  f = f * 6.6461399789245794e35f;                             // * 2^119
  u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(r) : "f"(f)); return r;
}

// One packed word = 8 E2M1 codes = one m16n8k16 A fragment. Each code's bits sit where the
// shift-and-mask below moves them to a BF16 sign bit, exponent's two low bits and mantissa's
// top bit: the BF16 is the E2M1 value * 2^-126 (subnormal for 0.5), times scale_pair's
// s * 2^118 gives E2M1 * s * 2^-8, exact in BF16.
#define M_BF 0x81C081C0u
DEVI u32 rotl(u32 x, u32 r) { u32 d; asm("shf.l.wrap.b32 %0, %1, %1, %2;" : "=r"(d) : "r"(x), "r"(r)); return d; }
#ifndef MOE_DEQ_IM
#define MOE_DEQ_IM 2
#endif
// 16 and 64, read from constant memory so the multiplies stay multiplies (IMAD, FMA pipe)
__constant__ u32 c_shl[2] = {16u, 64u};
DEVI void dequant(u32 q, u32 sg, u32 sg8, u32& a0, u32& a1, u32& a2, u32& a3) {
#ifdef MOE_NO_DEQ
  a0 = q; a1 = q ^ sg; a2 = q + sg8; a3 = q; return;
#endif
  // 8 codes from 7 ALU ops and 2 multiplies. The masked bits of the rotations by 4 and 6 never wrap,
  // so those two are left shifts: multiplies by 16 and 64 on the FMA pipe, beside the ALU.
  u32 o0 = q & M_BF;
#if MOE_DEQ_IM >= 1
  u32 o1 = (q * c_shl[0]) & M_BF;
#else
  u32 o1 = rotl(q, 4) & M_BF;
#endif
  u32 o2 = rotl(q, 10) & M_BF;
#if MOE_DEQ_IM >= 2
  u32 o3 = ((q * c_shl[1]) & 0x80C080C0u) | (rotl(q, 14) & 0x01000100u);
#else
  u32 o3 = (rotl(q, 6) & 0x80C080C0u) | (rotl(q, 14) & 0x01000100u);
#endif
  a0 = bmul2(o0, sg);
  a1 = bmul2(o1, sg8);
  a2 = bmul2(o2, sg);
  a3 = bmul2(o3, sg8);
}

// ---- wgmma (m64nNk16, A from registers, B from 128B-swizzled K-major shared memory) ----
DEVI u64 wg_desc(u32 saddr) {
  // start address >> 4, leading-dimension offset 1 (unused for swizzled K-major), stride offset
  // 1024 B (8 rows of 128 B) >> 4, 128-byte swizzle
  return (u64)((saddr & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)(1024 >> 4) << 32) | ((u64)1 << 62);
}
DEVI void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
DEVI void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> DEVI void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
template <int NT> DEVI void wgmma_n8(float (&acc)[NT][4], const u32* a, u64 desc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p, 1, 1, 0; }"
               : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int NT> DEVI void wgmma_n16(float (&acc)[NT][4], const u32* a, u64 desc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, p, 1, 1, 0; }"
               : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int NT> DEVI void wgmma_n24(float (&acc)[NT][4], const u32* a, u64 desc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, p, 1, 1, 0; }"
               : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int NT> DEVI void wgmma_n32(float (&acc)[NT][4], const u32* a, u64 desc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1, 0; }"
               : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int NT> DEVI void wgmma_n40(float (&acc)[NT][4], const u32* a, u64 desc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19}, {%20,%21,%22,%23}, %24, p, 1, 1, 0; }"
               : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int NT> DEVI void wgmma_n48(float (&acc)[NT][4], const u32* a, u64 desc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23}, {%24,%25,%26,%27}, %28, p, 1, 1, 0; }"
               : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int NT> DEVI void wgmma_n56(float (&acc)[NT][4], const u32* a, u64 desc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27}, {%28,%29,%30,%31}, %32, p, 1, 1, 0; }"
               : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int NT> DEVI void wgmma_n64(float (&acc)[NT][4], const u32* a, u64 desc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 0; }"
               : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]), "+f"(acc[7][2]), "+f"(acc[7][3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int NT> DEVI void wgmma_n72(float (&acc)[NT][4], const u32* a, u64 desc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n72k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, {%36,%37,%38,%39}, %40, p, 1, 1, 0; }"
               : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]), "+f"(acc[7][2]), "+f"(acc[7][3]), "+f"(acc[8][0]), "+f"(acc[8][1]), "+f"(acc[8][2]), "+f"(acc[8][3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int NT> DEVI void wgmma_n80(float (&acc)[NT][4], const u32* a, u64 desc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n80k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39}, {%40,%41,%42,%43}, %44, p, 1, 1, 0; }"
               : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]), "+f"(acc[7][2]), "+f"(acc[7][3]), "+f"(acc[8][0]), "+f"(acc[8][1]), "+f"(acc[8][2]), "+f"(acc[8][3]), "+f"(acc[9][0]), "+f"(acc[9][1]), "+f"(acc[9][2]), "+f"(acc[9][3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int NT> DEVI void wgmma_n88(float (&acc)[NT][4], const u32* a, u64 desc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n88k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43}, {%44,%45,%46,%47}, %48, p, 1, 1, 0; }"
               : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]), "+f"(acc[7][2]), "+f"(acc[7][3]), "+f"(acc[8][0]), "+f"(acc[8][1]), "+f"(acc[8][2]), "+f"(acc[8][3]), "+f"(acc[9][0]), "+f"(acc[9][1]), "+f"(acc[9][2]), "+f"(acc[9][3]), "+f"(acc[10][0]), "+f"(acc[10][1]), "+f"(acc[10][2]), "+f"(acc[10][3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int NT> DEVI void wgmma_n96(float (&acc)[NT][4], const u32* a, u64 desc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47}, {%48,%49,%50,%51}, %52, p, 1, 1, 0; }"
               : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]), "+f"(acc[7][2]), "+f"(acc[7][3]), "+f"(acc[8][0]), "+f"(acc[8][1]), "+f"(acc[8][2]), "+f"(acc[8][3]), "+f"(acc[9][0]), "+f"(acc[9][1]), "+f"(acc[9][2]), "+f"(acc[9][3]), "+f"(acc[10][0]), "+f"(acc[10][1]), "+f"(acc[10][2]), "+f"(acc[10][3]), "+f"(acc[11][0]), "+f"(acc[11][1]), "+f"(acc[11][2]), "+f"(acc[11][3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int NT> DEVI void wgmma_n104(float (&acc)[NT][4], const u32* a, u64 desc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n104k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51}, {%52,%53,%54,%55}, %56, p, 1, 1, 0; }"
               : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]), "+f"(acc[7][2]), "+f"(acc[7][3]), "+f"(acc[8][0]), "+f"(acc[8][1]), "+f"(acc[8][2]), "+f"(acc[8][3]), "+f"(acc[9][0]), "+f"(acc[9][1]), "+f"(acc[9][2]), "+f"(acc[9][3]), "+f"(acc[10][0]), "+f"(acc[10][1]), "+f"(acc[10][2]), "+f"(acc[10][3]), "+f"(acc[11][0]), "+f"(acc[11][1]), "+f"(acc[11][2]), "+f"(acc[11][3]), "+f"(acc[12][0]), "+f"(acc[12][1]), "+f"(acc[12][2]), "+f"(acc[12][3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int NT> DEVI void wgmma_n112(float (&acc)[NT][4], const u32* a, u64 desc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n112k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55}, {%56,%57,%58,%59}, %60, p, 1, 1, 0; }"
               : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]), "+f"(acc[7][2]), "+f"(acc[7][3]), "+f"(acc[8][0]), "+f"(acc[8][1]), "+f"(acc[8][2]), "+f"(acc[8][3]), "+f"(acc[9][0]), "+f"(acc[9][1]), "+f"(acc[9][2]), "+f"(acc[9][3]), "+f"(acc[10][0]), "+f"(acc[10][1]), "+f"(acc[10][2]), "+f"(acc[10][3]), "+f"(acc[11][0]), "+f"(acc[11][1]), "+f"(acc[11][2]), "+f"(acc[11][3]), "+f"(acc[12][0]), "+f"(acc[12][1]), "+f"(acc[12][2]), "+f"(acc[12][3]), "+f"(acc[13][0]), "+f"(acc[13][1]), "+f"(acc[13][2]), "+f"(acc[13][3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int NT> DEVI void wgmma_n120(float (&acc)[NT][4], const u32* a, u64 desc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n120k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59}, {%60,%61,%62,%63}, %64, p, 1, 1, 0; }"
               : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]), "+f"(acc[7][2]), "+f"(acc[7][3]), "+f"(acc[8][0]), "+f"(acc[8][1]), "+f"(acc[8][2]), "+f"(acc[8][3]), "+f"(acc[9][0]), "+f"(acc[9][1]), "+f"(acc[9][2]), "+f"(acc[9][3]), "+f"(acc[10][0]), "+f"(acc[10][1]), "+f"(acc[10][2]), "+f"(acc[10][3]), "+f"(acc[11][0]), "+f"(acc[11][1]), "+f"(acc[11][2]), "+f"(acc[11][3]), "+f"(acc[12][0]), "+f"(acc[12][1]), "+f"(acc[12][2]), "+f"(acc[12][3]), "+f"(acc[13][0]), "+f"(acc[13][1]), "+f"(acc[13][2]), "+f"(acc[13][3]), "+f"(acc[14][0]), "+f"(acc[14][1]), "+f"(acc[14][2]), "+f"(acc[14][3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int NT> DEVI void wgmma_n128(float (&acc)[NT][4], const u32* a, u64 desc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, %68, p, 1, 1, 0; }"
               : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]), "+f"(acc[7][2]), "+f"(acc[7][3]), "+f"(acc[8][0]), "+f"(acc[8][1]), "+f"(acc[8][2]), "+f"(acc[8][3]), "+f"(acc[9][0]), "+f"(acc[9][1]), "+f"(acc[9][2]), "+f"(acc[9][3]), "+f"(acc[10][0]), "+f"(acc[10][1]), "+f"(acc[10][2]), "+f"(acc[10][3]), "+f"(acc[11][0]), "+f"(acc[11][1]), "+f"(acc[11][2]), "+f"(acc[11][3]), "+f"(acc[12][0]), "+f"(acc[12][1]), "+f"(acc[12][2]), "+f"(acc[12][3]), "+f"(acc[13][0]), "+f"(acc[13][1]), "+f"(acc[13][2]), "+f"(acc[13][3]), "+f"(acc[14][0]), "+f"(acc[14][1]), "+f"(acc[14][2]), "+f"(acc[14][3]), "+f"(acc[15][0]), "+f"(acc[15][1]), "+f"(acc[15][2]), "+f"(acc[15][3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int NT> DEVI void pin_acc(float (&acc)[2][NT][4]) {
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int b = 0; b < NT; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) asm volatile("" : "+f"(acc[a][b][c]) :: "memory");
}

// Position of original k (0..63 within a 64-k chunk) in the mma k order the weights use: step jj,
// position p, with k = 16 * ((p % 8) / 2) + 4 * jj + (p % 2) + 2 * (p >= 8).
DEVI int kperm_pos(int k) {
  const int tg = k >> 4, jj = (k >> 2) & 3, r = k & 3;
  return 16 * jj + 2 * tg + (r & 1) + ((r >> 1) << 3);
}
// Element offset, inside a tile of T rows, of row i's activation k (0..4095) in the wgmma B layout:
// [k chunk of 256][4 chunks of 64][T rows][64 positions, 16-byte groups swizzled by row % 8].
DEVI int wg_xoff(int i, int k, int T) {
  const int q = kperm_pos(k & 63);
  const int pc = (q >> 3) ^ (i & 7);
  return ((k >> 6) * T + i) * 64 + pc * 8 + (q & 7);
}

// wgmma stage (all 8 consumer warps on the same 256 k): per 64-k chunk, unpack the warp's 2 x 16
// rows to A fragments (double buffered), then one m64n(8*NTL)k16 per row group and 16-k step; the
// warpgroup's 4 warps supply the 64 rows. B is the stage's activations, already laid out for wgmma.
template <int NT, int NTL>
DEVI void stage_compute_wg(float (&acc)[2][NT][4], const unsigned char* wbase, const unsigned char* sclb,
                           u32 xaddr, int T) {
  uint4 qw[2][4];
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    qw[0][kk] = *reinterpret_cast<const uint4*>(wbase + kk * 512);
    qw[1][kk] = *reinterpret_cast<const uint4*>(wbase + 2048 + kk * 512);
  }
  const uint2 sv0 = *reinterpret_cast<const uint2*>(sclb);
  const uint2 sv1 = *reinterpret_cast<const uint2*>(sclb + 256);
  u32 A[2][2][4][4];
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    const uint4 q0 = qw[0][kk], q1 = qw[1][kk];
    const u32 sw0 = kk < 2 ? sv0.x : sv0.y, sw1 = kk < 2 ? sv1.x : sv1.y;
    constexpr bool SIM = NTL == 1;
    const u32 sg0 = (kk & 1) ? scale_of<2, SIM>(sw0) : scale_of<0, SIM>(sw0);
    const u32 sg80 = (kk & 1) ? scale_of<3, SIM>(sw0) : scale_of<1, SIM>(sw0);
    const u32 sg1 = (kk & 1) ? scale_of<2, SIM>(sw1) : scale_of<0, SIM>(sw1);
    const u32 sg81 = (kk & 1) ? scale_of<3, SIM>(sw1) : scale_of<1, SIM>(sw1);
    u32 (&Ab)[2][4][4] = A[kk & 1];
    dequant(q0.x, sg0, sg80, Ab[0][0][0], Ab[0][0][1], Ab[0][0][2], Ab[0][0][3]);
    dequant(q0.y, sg0, sg80, Ab[0][1][0], Ab[0][1][1], Ab[0][1][2], Ab[0][1][3]);
    dequant(q0.z, sg0, sg80, Ab[0][2][0], Ab[0][2][1], Ab[0][2][2], Ab[0][2][3]);
    dequant(q0.w, sg0, sg80, Ab[0][3][0], Ab[0][3][1], Ab[0][3][2], Ab[0][3][3]);
    dequant(q1.x, sg1, sg81, Ab[1][0][0], Ab[1][0][1], Ab[1][0][2], Ab[1][0][3]);
    dequant(q1.y, sg1, sg81, Ab[1][1][0], Ab[1][1][1], Ab[1][1][2], Ab[1][1][3]);
    dequant(q1.z, sg1, sg81, Ab[1][2][0], Ab[1][2][1], Ab[1][2][2], Ab[1][2][3]);
    dequant(q1.w, sg1, sg81, Ab[1][3][0], Ab[1][3][1], Ab[1][3][2], Ab[1][3][3]);
    wg_fence();
    const u64 d0 = wg_desc(xaddr + kk * T * 128);
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        if constexpr (NTL == 1) wgmma_n8<NT>(acc[r], Ab[r][jj], d0 + 2 * jj);
        else if constexpr (NTL == 2) wgmma_n16<NT>(acc[r], Ab[r][jj], d0 + 2 * jj);
        else if constexpr (NTL == 3) wgmma_n24<NT>(acc[r], Ab[r][jj], d0 + 2 * jj);
        else if constexpr (NTL == 4) wgmma_n32<NT>(acc[r], Ab[r][jj], d0 + 2 * jj);
        else if constexpr (NTL == 5) wgmma_n40<NT>(acc[r], Ab[r][jj], d0 + 2 * jj);
        else if constexpr (NTL == 6) wgmma_n48<NT>(acc[r], Ab[r][jj], d0 + 2 * jj);
        else if constexpr (NTL == 7) wgmma_n56<NT>(acc[r], Ab[r][jj], d0 + 2 * jj);
        else if constexpr (NTL == 8) wgmma_n64<NT>(acc[r], Ab[r][jj], d0 + 2 * jj);
        else if constexpr (NTL == 9) wgmma_n72<NT>(acc[r], Ab[r][jj], d0 + 2 * jj);
        else if constexpr (NTL == 10) wgmma_n80<NT>(acc[r], Ab[r][jj], d0 + 2 * jj);
        else if constexpr (NTL == 11) wgmma_n88<NT>(acc[r], Ab[r][jj], d0 + 2 * jj);
        else if constexpr (NTL == 12) wgmma_n96<NT>(acc[r], Ab[r][jj], d0 + 2 * jj);
        else if constexpr (NTL == 13) wgmma_n104<NT>(acc[r], Ab[r][jj], d0 + 2 * jj);
        else if constexpr (NTL == 14) wgmma_n112<NT>(acc[r], Ab[r][jj], d0 + 2 * jj);
        else if constexpr (NTL == 15) wgmma_n120<NT>(acc[r], Ab[r][jj], d0 + 2 * jj);
        else wgmma_n128<NT>(acc[r], Ab[r][jj], d0 + 2 * jj);
      }
    }
    wg_commit();
    wg_wait<1>();
  }
  wg_wait<0>();
  pin_acc<NT>(acc);
}

// One stage of one consumer warp: 2 row groups x 256 k of weights against NTL token tiles. The
// stage's weight words and scales go to registers first, so their shared-memory latency overlaps;
// activation fragments are fetched one 64-k chunk ahead.
template <int NT, int NTL>
DEVI void stage_compute(float (&acc)[2][NT][4], float (&acc2)[2][NT][4], const unsigned char* wbase, const unsigned char* sclb,
                        const unsigned char* xbase, int XP) {
  uint4 qw[2][4];
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    qw[0][kk] = *reinterpret_cast<const uint4*>(wbase + kk * 512);
    qw[1][kk] = *reinterpret_cast<const uint4*>(wbase + 2048 + kk * 512);
  }
  const uint2 sv0 = *reinterpret_cast<const uint2*>(sclb);
  const uint2 sv1 = *reinterpret_cast<const uint2*>(sclb + 256);
  uint4 bx[2][NTL][2];
#pragma unroll
  for (int nt = 0; nt < NTL; ++nt) {
    const unsigned char* xp = xbase + nt * 8 * XP;
#ifdef MOE_NO_BLDS
    bx[0][nt][0] = qw[0][nt & 3]; bx[0][nt][1] = qw[1][nt & 3];
#else
    bx[0][nt][0] = *reinterpret_cast<const uint4*>(xp);
    bx[0][nt][1] = *reinterpret_cast<const uint4*>(xp + 16);
#endif
  }
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    if (kk < 3) {
#pragma unroll
      for (int nt = 0; nt < NTL; ++nt) {
        const unsigned char* xp = xbase + nt * 8 * XP + (kk + 1) * 128;
#ifdef MOE_NO_BLDS
        bx[(kk + 1) & 1][nt][0] = qw[1][(kk + nt) & 3]; bx[(kk + 1) & 1][nt][1] = qw[0][(kk + nt) & 3];
#else
        bx[(kk + 1) & 1][nt][0] = *reinterpret_cast<const uint4*>(xp);
        bx[(kk + 1) & 1][nt][1] = *reinterpret_cast<const uint4*>(xp + 16);
#endif
      }
    }
    const uint4 q0 = qw[0][kk], q1 = qw[1][kk];
    const u32 sw0 = kk < 2 ? sv0.x : sv0.y, sw1 = kk < 2 ? sv1.x : sv1.y;
    constexpr bool SIM = NTL == 1;
    const u32 sg0 = (kk & 1) ? scale_of<2, SIM>(sw0) : scale_of<0, SIM>(sw0);
    const u32 sg80 = (kk & 1) ? scale_of<3, SIM>(sw0) : scale_of<1, SIM>(sw0);
    const u32 sg1 = (kk & 1) ? scale_of<2, SIM>(sw1) : scale_of<0, SIM>(sw1);
    const u32 sg81 = (kk & 1) ? scale_of<3, SIM>(sw1) : scale_of<1, SIM>(sw1);
    u32 A[2][4][4];
    dequant(q0.x, sg0, sg80, A[0][0][0], A[0][0][1], A[0][0][2], A[0][0][3]);
    dequant(q0.y, sg0, sg80, A[0][1][0], A[0][1][1], A[0][1][2], A[0][1][3]);
    dequant(q0.z, sg0, sg80, A[0][2][0], A[0][2][1], A[0][2][2], A[0][2][3]);
    dequant(q0.w, sg0, sg80, A[0][3][0], A[0][3][1], A[0][3][2], A[0][3][3]);
    dequant(q1.x, sg1, sg81, A[1][0][0], A[1][0][1], A[1][0][2], A[1][0][3]);
    dequant(q1.y, sg1, sg81, A[1][1][0], A[1][1][1], A[1][1][2], A[1][1][3]);
    dequant(q1.z, sg1, sg81, A[1][2][0], A[1][2][1], A[1][2][2], A[1][2][3]);
    dequant(q1.w, sg1, sg81, A[1][3][0], A[1][3][1], A[1][3][2], A[1][3][3]);
#pragma unroll
    for (int nt = 0; nt < NTL; ++nt) {
      const uint4 b0 = bx[kk & 1][nt][0], b1 = bx[kk & 1][nt][1];
      const u32 B[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
      for (int jj = 0; jj < 4; ++jj)
#pragma unroll
        for (int r = 0; r < 2; ++r)
#ifdef MOE_MMA_SINK
          asm volatile("" :: "r"(A[r][jj][0]), "r"(A[r][jj][1]), "r"(A[r][jj][2]), "r"(A[r][jj][3]), "r"(B[2 * jj]), "r"(B[2 * jj + 1]));
#elif defined(MOE_NO_MMA)
          acc[r][nt][jj] += u2f(A[r][jj][0] ^ A[r][jj][1] ^ A[r][jj][2] ^ A[r][jj][3] ^ B[2 * jj] ^ B[2 * jj + 1]);
#else
#ifdef MOE_SPLITACC
          if (jj & 1) mma16816(acc2[r][nt], A[r][jj][0], A[r][jj][1], A[r][jj][2], A[r][jj][3], B[2 * jj], B[2 * jj + 1]);
          else
#endif
          mma16816(acc[r][nt], A[r][jj][0], A[r][jj][1], A[r][jj][2], A[r][jj][3], B[2 * jj], B[2 * jj + 1]);
#endif
    }
  }
}

// Work-queue order: every gate/up item (tile-major), then every down item (tile-major).
DEVI void decode_item(int item, int ntiles, int B1, int B2, int& phase, int& tile, int& idx) {
  const int n1 = ntiles * B1;
  if (item < n1) { phase = 1; tile = item / B1; idx = item - tile * B1; }
  else { phase = 2; tile = (item - n1) / B2; idx = item - n1 - tile * B2; }
}
// Down items of tiles from tsplit on are split KSX ways in k instead of KS2: the queue's last
// wave is short, so CTAs finish together.
DEVI void decode_item2(int item, int ntiles, int tsplit, int NRB1, int NRB2, int KS2, int KSX,
                       int& phase, int& tile, int& rb, int& ks, int& kss) {
  const int n1 = ntiles * NRB1;
  if (item < n1) { phase = 1; tile = item / NRB1; rb = item - tile * NRB1; ks = 0; kss = 1; return; }
  phase = 2;
  int j = item - n1;
  const int na = tsplit * NRB2 * KS2;
  if (j < na) { kss = KS2; tile = j / (NRB2 * KS2); j -= tile * NRB2 * KS2; }
  else { kss = KSX; j -= na; const int t2 = j / (NRB2 * KSX); j -= t2 * NRB2 * KSX; tile = tsplit + t2; }
  rb = j / kss; ks = j - rb * kss;
}

struct Params {
  const u16* x; const int* ids; const float* topk_w; int M; int local_start;
  const u32* w13; const unsigned char* s13; const float* gs13;
  const u32* w2; const unsigned char* s2; const float* gs2; const float* nw1;
  u16* inter; float* ypart; u16* out; int* ctr; int pmax;
  u16* xs;  // NT >= 2: the gate/up activations gathered per tile, stage-major like the intermediate
  float* ssq;  // [pmax][64]: per pair and 64-row slot, the sum of squares of y * (1 + w), full-k items
  // tensor maps of this config's stage boxes (RGS row groups x KW k chunks): w13, s13, w2, s2
  const unsigned char* tmap;
  u16* yh;  // BF16 down output, unchanged arithmetic boundary
  unsigned char* routes; int route_sms;
};
DEVI void st_bf4(u16* p, float4 v) {
  const u32 lo = (u32)f2bf(v.x) | ((u32)f2bf(v.y) << 16), hi = (u32)f2bf(v.z) | ((u32)f2bf(v.w) << 16);
  *reinterpret_cast<uint2*>(p) = make_uint2(lo, hi);
}
DEVI float4 ld_bf4(const u16* p) {
  const uint2 u = __ldcg(reinterpret_cast<const uint2*>(p));
  return make_float4(u2f(u.x << 16), u2f(u.x & 0xFFFF0000u), u2f(u.y << 16), u2f(u.y & 0xFFFF0000u));
}
DEVI float4 bf_round4(float4 v) {  // v as stored in BF16
  return make_float4(u2f((u32)f2bf(v.x) << 16), u2f((u32)f2bf(v.y) << 16), u2f((u32)f2bf(v.z) << 16), u2f((u32)f2bf(v.w) << 16));
}

// Register split after the role dispatch: the launch gives 168 per thread (384 threads), so the
// producer warpgroup's release must cover the consumers' increase: 128 (168 - PREG) >= 256 (CREG - 168).
#ifndef MOE_PREG
#define MOE_PREG 40
#endif
#ifndef MOE_CREG
#define MOE_CREG 232
#endif
#define MOE_STR2(x) #x
#define MOE_STR(x) MOE_STR2(x)
#ifndef MOE_WGMIN
#define MOE_WGMIN 5
#endif
// MOE_NO_COMPUTE: consumers wait for and release every stage without decoding it (timing of the
// data movement alone; the outputs are garbage)
#ifdef MOE_NO_COMPUTE
#define MOE_NOC true
#else
#define MOE_NOC false
#endif
// RW row-warps x KW k-warps = 8 consumer warps, each owning 2 row groups (32 rows) and 256 k of
// every stage. A stage: RB = 32*RW rows x KW*256 k of weights (32 KB) + scales (4 KB) + up to
// 8*NT activation rows.

// One CTA builds fresh routing tables for the whole prefill. Its kernel boundary
// publishes them before any persistent expert CTA starts reading them.
template <int RW, int NT, int KS2, int MW>
__device__ __forceinline__ void global_route_body(const Params& P) {
  constexpr int KW=8/RW, RB=32*RW, T=8*NT;
  constexpr int NRB1=8192/RB, NRB2=4096/RB;
  constexpr int KSX=KW==1 ? 4 : KS2;
  constexpr int MAXP=MW*32*8, MAXT=32+MAXP/T;
  constexpr bool EARLY=false;
  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, M=P.M;
  int* scal=reinterpret_cast<int*>(P.routes);
  int* ftile=scal+16;
  int* off=scal+48;
  u32* bm=reinterpret_cast<u32*>(P.routes+512);
  int* wpre=reinterpret_cast<int*>(bm+32*MW);
  int4* tiles=reinterpret_cast<int4*>(wpre+32*MW);
  u16* pair_tok=reinterpret_cast<u16*>(tiles+MAXT);
  signed char* lid=reinterpret_cast<signed char*>(pair_tok+MAXP);
  if (tid<16) scal[tid]=0;
  // Token bitmaps per expert without atomics: a warp takes 32 tokens (one bitmap word), each lane
  // its token's local-expert mask, and 32 ballots give the word of every expert.
  for (int t0 = warp * 32; t0 < MW * 32; t0 += (EARLY ? 8 : 12) * 32) {
    const int t = t0 + lane;
    u32 em = 0;
    if (t < M) {
      const int4 v0 = reinterpret_cast<const int4*>(P.ids)[t * 2], v1 = reinterpret_cast<const int4*>(P.ids)[t * 2 + 1];
      const int ev[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int e = ev[j] - P.local_start;
        const bool loc = (unsigned)e < 32u;
        lid[t * 8 + j] = loc ? (signed char)e : (signed char)-1;
        em |= loc ? 1u << e : 0u;
      }
    }
    u32 mine = 0;
#pragma unroll
    for (int e = 0; e < 32; ++e) {
      const u32 b = __ballot_sync(0xffffffffu, (em >> e) & 1u);
      if (lane == e) mine = b;
    }
    bm[lane * MW + (t0 >> 5)] = mine;
  }
  if constexpr (EARLY) consumer_sync(); else __syncthreads();
  if (tid == 0) MOE_TR(0, 10, 0);
#ifdef MOE_TIMING
  if (tid == 0) reinterpret_cast<u32*>(P.ctr)[14420 + blockIdx.x * 4 + 2] = gtimer();
#endif
  if (warp == 0) {
    int c = 0;
    for (int w = 0; w < MW; ++w) {
      wpre[lane * MW + w] = c;
      c += __popc(bm[lane * MW + w]);
    }
    const int nt = (c + T - 1) / T;
    // Largest experts first (load balance), except with single-group tiles, where every tile is
    // about the same size and index order saves the ranking's latency at startup.
    int rank = lane;
#ifndef MOE_NOSORT
    if constexpr (NT > 1) {
      rank = 0;
      for (int j = 0; j < 32; ++j) {
        const int cj = __shfl_sync(0xffffffffu, c, j);
        rank += (cj > c) || (cj == c && j < lane);
      }
    }
#endif
    // pairs and tiles before this expert in rank order: an exclusive scan in rank space (the
    // counts move to their rank through shared memory)
    int2* rk = reinterpret_cast<int2*>(tiles);  // tiles[] (>= 32 entries) is written only after this
    rk[rank] = make_int2(c, nt);
    __syncwarp();
    const int2 mine = rk[lane];  // lane = rank here
    int sc = mine.x, st = mine.y;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int a = __shfl_up_sync(0xffffffffu, sc, d), b = __shfl_up_sync(0xffffffffu, st, d);
      if (lane >= d) { sc += a; st += b; }
    }
    __syncwarp();
    rk[lane] = make_int2(sc - mine.x, st - mine.y);
    __syncwarp();
    const int2 ex = rk[rank];
    const int o = ex.x, to = ex.y;
    __syncwarp();
    ftile[lane] = to;
    off[lane] = o;
    for (int k = 0; k < nt; ++k) tiles[to + k] = make_int4(lane, o + k * T, min(T, c - k * T), 0);
    int tot_t = nt, tot_p = c;
    for (int s = 16; s > 0; s >>= 1) {
      tot_t += __shfl_xor_sync(0xffffffffu, tot_t, s);
      tot_p += __shfl_xor_sync(0xffffffffu, tot_p, s);
    }
    __syncwarp();
    if (lane == 0) {
#ifndef MOE_SPLW
#define MOE_SPLW 1.0f
#endif
      // about MOE_SPLW waves of down items, taken from the last (smallest) tiles, split KSX ways
      const int nspl = KSX > KS2 ? min(tot_t, (int)ceilf(MOE_SPLW * P.route_sms / (NRB2 * KS2))) : 0;
      const int tsplit = tot_t - nspl;
      scal[0] = tot_t;
      scal[1] = NRB1 * tot_t;
      scal[2] = NRB1 * tot_t + NRB2 * KS2 * tsplit + NRB2 * KSX * nspl;
      scal[3] = tot_p;
      scal[4] = tsplit;
      scal[5] = nspl > 0 ? tiles[tsplit].y : tot_p;  // first pair with KSX partials
    }
  }
  if constexpr (EARLY) consumer_sync(); else __syncthreads();
    for (int i = tid; i < M * 8; i += blockDim.x) {
      const int e = lid[i];
      if (e >= 0) {
        const int t = i >> 3;
        const u32 word = bm[e * MW + (t >> 5)];
        const int p = off[e] + wpre[e * MW + (t >> 5)] + __popc(word & ((1u << (t & 31)) - 1u));
        pair_tok[p] = (u16)t;
      }
    }

}


template <int NT, int MW>
__device__ __forceinline__ void parallel_route_count(const Params& P) {
  constexpr int MAXP=MW*32*8, MAXT=32+MAXP/(8*NT);
  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, expert=blockIdx.x;
  u32* bm=reinterpret_cast<u32*>(P.routes+512);
  int* prefix=reinterpret_cast<int*>(bm+32*MW);
  int4* tiles=reinterpret_cast<int4*>(prefix+32*MW);
  u16* pairs=reinterpret_cast<u16*>(tiles+MAXT);
  signed char* lid=reinterpret_cast<signed char*>(pairs+MAXP);
  for (int i=blockIdx.x*256+tid;i<P.M*8;i+=32*256) {
    int e=P.ids[i]-P.local_start;
    lid[i]=(unsigned)e<32u ? (signed char)e : (signed char)-1;
  }
  for (int word=warp;word<MW;word+=8) {
    const int token=word*32+lane;
    bool hit=false;
    if (token<P.M) {
      const int4 a=reinterpret_cast<const int4*>(P.ids)[token*2];
      const int4 b=reinterpret_cast<const int4*>(P.ids)[token*2+1];
      const int e=expert+P.local_start;
      hit=a.x==e||a.y==e||a.z==e||a.w==e||b.x==e||b.y==e||b.z==e||b.w==e;
    }
    u32 bits=__ballot_sync(0xffffffffu,hit);
    if (lane==0) bm[expert*MW+word]=bits;
  }
  __shared__ int warp_sum[8], carry;
  if (tid==0) carry=0;
  __syncthreads();
  for (int first=0;first<MW;first+=256) {
    const int previous=carry, word=first+tid;
    const int count=word<MW ? __popc(bm[expert*MW+word]) : 0;
    int scan=count;
#pragma unroll
    for (int delta=1;delta<32;delta<<=1) {
      int add=__shfl_up_sync(0xffffffffu,scan,delta);
      if (lane>=delta) scan+=add;
    }
    if (lane==31) warp_sum[warp]=scan;
    __syncthreads();
    if (warp==0) {
      const int count_w=lane<8 ? warp_sum[lane] : 0;
      int total=count_w;
#pragma unroll
      for (int delta=1;delta<8;delta<<=1) {
        int add=__shfl_up_sync(0xffffffffu,total,delta);
        if (lane>=delta) total+=add;
      }
      if (lane<8) warp_sum[lane]=total-count_w;
      if (lane==7) carry=previous+total;
    }
    __syncthreads();
    if (word<MW) prefix[expert*MW+word]=previous+warp_sum[warp]+scan-count;
    __syncthreads();
  }
  if (tid==0) reinterpret_cast<int*>(P.routes)[80+expert]=carry;
}

template <int NT, int MW>
__device__ __forceinline__ void parallel_route_pairs(const Params& P) {
  constexpr int MAXP=MW*32*8, MAXT=32+MAXP/(8*NT);
  const int* off=reinterpret_cast<const int*>(P.routes)+48;
  const u32* bm=reinterpret_cast<const u32*>(P.routes+512);
  const int* prefix=reinterpret_cast<const int*>(bm+32*MW);
  const int4* tiles=reinterpret_cast<const int4*>(prefix+32*MW);
  u16* pairs=reinterpret_cast<u16*>(const_cast<int4*>(tiles)+MAXT);
  const signed char* lid=reinterpret_cast<const signed char*>(pairs+MAXP);
  for (int i=blockIdx.x*blockDim.x+threadIdx.x;i<P.M*8;i+=gridDim.x*blockDim.x) {
    int e=lid[i];
    if (e>=0) {
      int t=i>>3;
      u32 word=bm[e*MW+(t>>5)];
      int p=off[e]+prefix[e*MW+(t>>5)]+__popc(word&((1u<<(t&31))-1u));
      pairs[p]=(u16)t;
    }
  }
}

template <int RW, int NT, int KS2, int MW>
__device__ __forceinline__ void parallel_route_finalize(const Params& P) {
  constexpr int KW=8/RW, RB=32*RW, T=8*NT;
  constexpr int NRB1=8192/RB, NRB2=4096/RB;
  constexpr int KSX=KW==1 ? 4 : KS2;
  constexpr int MAXP=MW*32*8, MAXT=32+MAXP/T;
  constexpr bool EARLY=false;
  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, M=P.M;
  int* scal=reinterpret_cast<int*>(P.routes);
  int* ftile=scal+16;
  int* off=scal+48;
  u32* bm=reinterpret_cast<u32*>(P.routes+512);
  int* wpre=reinterpret_cast<int*>(bm+32*MW);
  int4* tiles=reinterpret_cast<int4*>(wpre+32*MW);
  u16* pair_tok=reinterpret_cast<u16*>(tiles+MAXT);
  signed char* lid=reinterpret_cast<signed char*>(pair_tok+MAXP);
  if (tid<16) scal[tid]=0;
  if (warp == 0) {
    const int c = reinterpret_cast<const int*>(P.routes)[80+lane];
    const int nt = (c + T - 1) / T;
    // Largest experts first (load balance), except with single-group tiles, where every tile is
    // about the same size and index order saves the ranking's latency at startup.
    int rank = lane;
#ifndef MOE_NOSORT
    if constexpr (NT > 1) {
      rank = 0;
      for (int j = 0; j < 32; ++j) {
        const int cj = __shfl_sync(0xffffffffu, c, j);
        rank += (cj > c) || (cj == c && j < lane);
      }
    }
#endif
    // pairs and tiles before this expert in rank order: an exclusive scan in rank space (the
    // counts move to their rank through shared memory)
    int2* rk = reinterpret_cast<int2*>(tiles);  // tiles[] (>= 32 entries) is written only after this
    rk[rank] = make_int2(c, nt);
    __syncwarp();
    const int2 mine = rk[lane];  // lane = rank here
    int sc = mine.x, st = mine.y;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int a = __shfl_up_sync(0xffffffffu, sc, d), b = __shfl_up_sync(0xffffffffu, st, d);
      if (lane >= d) { sc += a; st += b; }
    }
    __syncwarp();
    rk[lane] = make_int2(sc - mine.x, st - mine.y);
    __syncwarp();
    const int2 ex = rk[rank];
    const int o = ex.x, to = ex.y;
    __syncwarp();
    ftile[lane] = to;
    off[lane] = o;
    for (int k = 0; k < nt; ++k) tiles[to + k] = make_int4(lane, o + k * T, min(T, c - k * T), 0);
    int tot_t = nt, tot_p = c;
    for (int s = 16; s > 0; s >>= 1) {
      tot_t += __shfl_xor_sync(0xffffffffu, tot_t, s);
      tot_p += __shfl_xor_sync(0xffffffffu, tot_p, s);
    }
    __syncwarp();
    if (lane == 0) {
#ifndef MOE_SPLW
#define MOE_SPLW 1.0f
#endif
      // about MOE_SPLW waves of down items, taken from the last (smallest) tiles, split KSX ways
      const int nspl = KSX > KS2 ? min(tot_t, (int)ceilf(MOE_SPLW * P.route_sms / (NRB2 * KS2))) : 0;
      const int tsplit = tot_t - nspl;
      scal[0] = tot_t;
      scal[1] = NRB1 * tot_t;
      scal[2] = NRB1 * tot_t + NRB2 * KS2 * tsplit + NRB2 * KSX * nspl;
      scal[3] = tot_p;
      scal[4] = tsplit;
      scal[5] = nspl > 0 ? tiles[tsplit].y : tot_p;  // first pair with KSX partials
    }
  }
}


template <int RW, int NT, int MW>
__device__ __forceinline__ void external_gather(const Params& P) {
  constexpr int KW=8/RW, KST=256*KW, T=8*NT, XP=KST*2+16;
  constexpr int WGMIN=(KW==1&&NT>=8) ? MOE_WGMIN : 99;
  constexpr bool HASWG=WGMIN<99;
  constexpr int MAXP=MW*32*8, MAXT=32+MAXP/T;
  const int tid=threadIdx.x;
  const int* scal=reinterpret_cast<const int*>(P.routes);
  const int4* tiles=reinterpret_cast<const int4*>(P.routes+512+32*MW*8);
  const u16* pair_tok=reinterpret_cast<const u16*>(tiles+MAXT);
      const int np_all = scal[3];
      const int pb = (int)((long long)np_all * blockIdx.x / gridDim.x);
      const int pe = (int)((long long)np_all * (blockIdx.x + 1) / gridDim.x);
      // Rows in groups of 4: every load of the group is issued before any store, so the rows'
      // HBM latencies overlap (the compiler cannot reorder loads of x past stores to xs).
      int tile = 0;
      for (int pg = pb; pg < pe; pg += 4) {
        uint4 val[4][2];
        int tl[4], ri[4];
        bool wg[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const int p = pg + r;
          tl[r] = 0; ri[r] = 0; wg[r] = false;
          if (p < pe) {
            while (tile + 1 < scal[0] && tiles[tile + 1].y <= p) ++tile;
            tl[r] = tile; ri[r] = p - tiles[tile].y;
            wg[r] = HASWG && (tiles[tile].z + 7) / 8 >= WGMIN;
            const u16* srow = P.x + (size_t)pair_tok[p] * 4096;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const int ch = tid + 256 * h;
              if (wg[r]) {
                // wgmma B layout: per 64-k chunk, row i's 128 bytes in the weights' k order,
                // 16-byte groups swizzled by i % 8
                const int c64 = ch >> 3, pc = ch & 7, lc = pc ^ (ri[r] & 7);
                u32 v[4];
#pragma unroll
                for (int e2 = 0; e2 < 4; ++e2) {
                  const int qp = lc * 8 + e2 * 2;
                  const int jj = qp >> 4, pp = qp & 15;
                  const int k = c64 * 64 + 16 * ((pp & 7) >> 1) + 4 * jj + ((pp >> 3) << 1);
                  v[e2] = __ldg(reinterpret_cast<const unsigned int*>(srow + k));
                }
                val[r][h] = make_uint4(v[0], v[1], v[2], v[3]);
              } else {
                val[r][h] = __ldg(reinterpret_cast<const uint4*>(srow) + ch);
              }
            }
          }
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          if (pg + r < pe) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const int ch = tid + 256 * h;
              if (wg[r]) {
                const int c64 = ch >> 3, pc = ch & 7;
                u16* dtile = P.xs + (size_t)tl[r] * (16 / KW) * T * XP / 2;
                *reinterpret_cast<uint4*>(dtile + ((size_t)c64 * T + ri[r]) * 64 + pc * 8) = val[r][h];
              } else {
                unsigned char* dst = reinterpret_cast<unsigned char*>(P.xs) + ((size_t)tl[r] * (16 / KW) * T + ri[r]) * XP;
                const int st = ch / (KST / 8), w = ch - st * (KST / 8);
                *reinterpret_cast<uint4*>(dst + (size_t)st * T * XP + w * 16) = val[r][h];
              }
            }
          }
        }
      }
}

template <int RW, int NT, int KS2, int NS, int MW>
__device__ __forceinline__ void moe_body(const Params& P) {
  constexpr int KW = 8 / RW;
  constexpr int RB = 32 * RW;
  constexpr int RGS = RB / 16;
  constexpr int KST = 256 * KW;
  constexpr int T = 8 * NT;
  constexpr int XP = KST * 2 + 16;
  constexpr int WB = RGS * 2048 * KW;
  constexpr int SB = RGS * 256 * KW;
  // Tiles with at least WGMIN 8-token groups run on wgmma (needs every warp on the same k, KW = 1);
  // their activations are 128B-swizzled K-major; smaller tiles keep mma.sync and row-major rows.
  constexpr int WGMIN = (KW == 1 && NT >= 8) ? MOE_WGMIN : 99;
  constexpr bool HASWG = WGMIN < 99;
  // mma.sync covers at most 4 token groups per call: larger tiles must take the wgmma path
  static_assert(!HASWG || (WGMIN >= 3 && WGMIN <= 5), "token groups 5..8 need wgmma");
  constexpr int XB = HASWG ? (T * XP > T * 512 ? T * XP : T * 512) : T * XP;
  constexpr int EPI_T = T > 64 ? 64 : T;
  constexpr int EPI_BYTES = KW * EPI_T * (RB + 4) * 4;
  constexpr int STAGE_RAW = WB + SB + XB > EPI_BYTES ? WB + SB + XB : EPI_BYTES;
  constexpr int STAGE = HASWG ? (STAGE_RAW + 1023) / 1024 * 1024 : (STAGE_RAW + 127) / 128 * 128;
  constexpr int RED = (KW - 1) * RW * 1024 * NT;
  constexpr int NRB1 = 8192 / RB, NRB2 = 4096 / RB;
  constexpr int ST1 = 16 / KW;
  constexpr int ST2 = 16 / (KW * KS2);
  constexpr int MAXP = MW * 32 * 8;
  constexpr int MAXT = 32 + MAXP / T;
  constexpr int WP = RGS * 2048 / 4096 > 0 ? RGS * 2048 / 4096 : 1;  // 4 KB weight pieces per k chunk
  constexpr int WPB = RGS * 2048 / WP;
  static_assert(ST2 >= 1, "k split");
  // the last tiles' down items: split 4 ways in k when every warp shares the k range
  constexpr int KSX = KW == 1 ? 4 : KS2;
  constexpr int KSM = KSX > KS2 ? KSX : KS2;  // partial buffers
  // KW = 1 configs (many tiles): a token is combined once its own tiles are done; otherwise after
  // the whole queue (one counter)
  constexpr bool TILEDONE = KW <= 2;
  // single-group tiles (small batches): full-k down items leave sums of squares for a fast combine
#ifndef MOE_NO_SSQ
  constexpr bool SSQ = NT == 1;
#else
  constexpr bool SSQ = false;
#endif

  constexpr int NSIG = 4;
  extern __shared__ __align__(128) unsigned char smem[];
  static_assert(NS <= 5, "the header holds at most 5 ring stages");
  u64* epi = reinterpret_cast<u64*>(smem);                      // [NSIG] item outputs stored
  u64* sfree = reinterpret_cast<u64*>(smem + 32);               // [NSIG] signal slot free
  int2* sinfo = reinterpret_cast<int2*>(smem + 64);             // [NSIG] (phase, tile)
  u64* rdy = reinterpret_cast<u64*>(smem + 96);                 // pair list (and gather) done
  u64* rtd = reinterpret_cast<u64*>(smem + 104);                // routing tables (tiles, scal) done
  int* scal = reinterpret_cast<int*>(smem + 128);               // [16] ntiles, n1, total, P
  int* ftile = reinterpret_cast<int*>(smem + 192);              // [32] first tile of each expert
  int* off = reinterpret_cast<int*>(smem + 320);                // [32]
  float* gsm = reinterpret_cast<float*>(smem + 448);            // [64] gate/up then down global scales x 2^8
  u64* bars = reinterpret_cast<u64*>(smem + 704);               // full[NS], empty[NS]
  StageInfo* info = reinterpret_cast<StageInfo*>(smem + 704 + 16 * NS);  // [NS]
  u32* bm = reinterpret_cast<u32*>(P.routes + 512);                // [32][MW] token bitmaps per expert
  int* wpre = reinterpret_cast<int*>(bm + 32 * MW);             // [32][MW] exclusive popcount prefix
  int4* tiles = reinterpret_cast<int4*>(wpre + 32 * MW);        // [MAXT] (expert, first pair, tokens, -)
  u16* pair_tok = reinterpret_cast<u16*>(tiles + MAXT);         // [MAXP]
  signed char* lid = reinterpret_cast<signed char*>(pair_tok + MAXP);  // [MAXP] local expert of (t, slot)
  constexpr int ROUTE_BYTES = (32 * MW * 8 + MAXT * 16 + MAXP * 3 + 127) / 128 * 128;
  unsigned char* stages = smem + 1024;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int M = P.M;
  if (tid == 0) MOE_TR(0, 0, 0);
#ifdef MOE_TIMING
  u32* tdbg = reinterpret_cast<u32*>(P.ctr) + 12288 + blockIdx.x * 16;
  if (tid == 0) { tdbg[0] = gtimer(); for (int k = 0; k < 8; ++k) tdbg[4 + k] = 0; tdbg[11] = ctimer(); }
#endif

  // ------------------------------ routing (every CTA) ------------------------------
#ifdef MOE_NO_ROUTE
  if (tid == 0) {
    for (int s = 0; s < NS; ++s) { mbar_init(smem_addr(&bars[s]), 1); mbar_init(smem_addr(&bars[NS + s]), 8); }
    scal[0] = 0; scal[1] = 0; scal[2] = 0; scal[3] = 0; scal[4] = 0; scal[5] = 0;
  }
  __syncthreads();
  if (M > 0) goto route_done;
#endif
  if (tid < 64) gsm[tid] = tid < 32 ? P.gs13[tid] * 256.f : P.gs2[tid - 32] * 256.f;
  if (tid == 0) {
    for (int s = 0; s < NS; ++s) {
      mbar_init(smem_addr(&bars[s]), 1);
      mbar_init(smem_addr(&bars[NS + s]), 8);
    }
    for (int s = 0; s < NSIG; ++s) {
      mbar_init(smem_addr(&epi[s]), 8);
      mbar_init(smem_addr(&sfree[s]), 1);
    }
    mbar_init(smem_addr(rdy), 1);
    mbar_init(smem_addr(rtd), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  // With the early start (bs 1 config), routing runs on warps 0-7 (named barriers) and the
  // producer warpgroup does not wait for it; otherwise every warp takes part.
#ifndef MOE_NO_EARLY_ALL
  constexpr bool EARLY = NT == 1 && MW == 1;
#else
  constexpr bool EARLY = NT == 1 && KW == 4 && MW == 1;
#endif
  if (warp < 8 || !EARLY) {
  if (tid < 6) scal[tid] = reinterpret_cast<const int*>(P.routes)[tid];
  if (tid < 32) {
    ftile[tid] = reinterpret_cast<const int*>(P.routes)[16+tid];
    off[tid] = reinterpret_cast<const int*>(P.routes)[48+tid];
  }
  if constexpr (EARLY) consumer_sync(); else __syncthreads();
  if (tid == 0) mbar_arrive(smem_addr(rtd));  // the producer reads the tables after this
  }
  if (tid == 0) MOE_TR(0, 11, 0);
#ifdef MOE_TIMING
  if (tid == 0) reinterpret_cast<u32*>(P.ctr)[14420 + blockIdx.x * 4 + 0] = gtimer();
#endif
#ifndef MOE_GATHER_MIN
#define MOE_GATHER_MIN 4
#endif
  constexpr bool GATHER = NT >= MOE_GATHER_MIN;  // gate/up activations gathered tile-major first
  // The rest of routing (pair list, gather) runs on the consumer warps: the producer warpgroup
  // leaves now, so the first item's weights stream meanwhile; rdy releases the activation copies.
  if (warp < 8) {
    consumer_sync();
    if (tid == 0) MOE_TR(0, 12, 0);
#ifdef MOE_TIMING
    if (tid == 0) reinterpret_cast<u32*>(P.ctr)[14420 + blockIdx.x * 4 + 3] = gtimer();
#endif
    if constexpr (GATHER && MW < 128) {
      // This CTA's share of the pairs: their activation rows, gathered into the gate/up GEMM's
      // stage-major layout [tile][stage][T rows][XP bytes]; then count this CTA done.
      const int np_all = scal[3];
      const int pb = (int)((long long)np_all * blockIdx.x / gridDim.x);
      const int pe = (int)((long long)np_all * (blockIdx.x + 1) / gridDim.x);
      // Rows in groups of 4: every load of the group is issued before any store, so the rows'
      // HBM latencies overlap (the compiler cannot reorder loads of x past stores to xs).
      int tile = 0;
      for (int pg = pb; pg < pe; pg += 4) {
        uint4 val[4][2];
        int tl[4], ri[4];
        bool wg[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const int p = pg + r;
          tl[r] = 0; ri[r] = 0; wg[r] = false;
          if (p < pe) {
            while (tile + 1 < scal[0] && tiles[tile + 1].y <= p) ++tile;
            tl[r] = tile; ri[r] = p - tiles[tile].y;
            wg[r] = HASWG && (tiles[tile].z + 7) / 8 >= WGMIN;
            const u16* srow = P.x + (size_t)pair_tok[p] * 4096;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const int ch = tid + 256 * h;
              if (wg[r]) {
                // wgmma B layout: per 64-k chunk, row i's 128 bytes in the weights' k order,
                // 16-byte groups swizzled by i % 8
                const int c64 = ch >> 3, pc = ch & 7, lc = pc ^ (ri[r] & 7);
                u32 v[4];
#pragma unroll
                for (int e2 = 0; e2 < 4; ++e2) {
                  const int qp = lc * 8 + e2 * 2;
                  const int jj = qp >> 4, pp = qp & 15;
                  const int k = c64 * 64 + 16 * ((pp & 7) >> 1) + 4 * jj + ((pp >> 3) << 1);
                  v[e2] = __ldg(reinterpret_cast<const unsigned int*>(srow + k));
                }
                val[r][h] = make_uint4(v[0], v[1], v[2], v[3]);
              } else {
                val[r][h] = __ldg(reinterpret_cast<const uint4*>(srow) + ch);
              }
            }
          }
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          if (pg + r < pe) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const int ch = tid + 256 * h;
              if (wg[r]) {
                const int c64 = ch >> 3, pc = ch & 7;
                u16* dtile = P.xs + (size_t)tl[r] * (16 / KW) * T * XP / 2;
                *reinterpret_cast<uint4*>(dtile + ((size_t)c64 * T + ri[r]) * 64 + pc * 8) = val[r][h];
              } else {
                unsigned char* dst = reinterpret_cast<unsigned char*>(P.xs) + ((size_t)tl[r] * (16 / KW) * T + ri[r]) * XP;
                const int st = ch / (KST / 8), w = ch - st * (KST / 8);
                *reinterpret_cast<uint4*>(dst + (size_t)st * T * XP + w * 16) = val[r][h];
              }
            }
          }
        }
      }
      consumer_sync();
      // cumulative release: the CTA's gathered rows (ordered by the barrier) before the count
      if (tid == 0) red_release_add(P.ctr + 3, 1);
    }
    if (tid == 0) mbar_arrive(smem_addr(rdy));
  }
#ifdef MOE_NO_ROUTE
route_done:
#endif
#if defined(MOE_NO_WORK)
  const int ntiles = scal[0], n1 = 0, total = 0, tsplit = scal[4], psplit = scal[5];
#elif defined(MOE_ONLY_G1)
  const int ntiles = scal[0], n1 = scal[1], total = scal[1], tsplit = scal[4], psplit = scal[5];
#else
  int ntiles = 0, n1 = 0, total = 0, tsplit = 0, psplit = 0;
  if (warp < 8 || !EARLY) { ntiles = scal[0]; n1 = scal[1]; total = scal[2]; tsplit = scal[4]; psplit = scal[5]; }
#endif
#ifdef MOE_TIMING
  if (tid == 0) tdbg[1] = gtimer();
#endif
  int* head = P.ctr;
  int* done2 = P.ctr + 1;
  int* cnt1 = P.ctr + 8;
  int* cnt2 = P.ctr + 65536;  // bounded away from all large-prefill gate counters

  // Registers move from the producer warpgroup (warps 8-11: producer, signaler, two idle) to the
  // two consumer warpgroups. The consumer code is reachable only after the increase.
  if (warp >= 8) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 " MOE_STR(MOE_PREG) ";" ::: "memory");
#ifdef MOE_TRACE
    u32 pst[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const u32 pgt = gtimer();
    pst[0] = (u32)clock64();
#endif
    if (warp >= 10) return;
  if (warp == 8) {
    // ------------------------------ producer ------------------------------
    const u64 pol_w = policy_evict_first();
    const u64 pol_x = policy_evict_last();
    const u32 stage0 = smem_addr(stages), bars0 = smem_addr(bars);
#ifndef MOE_NO_TMAP
    // a stage's weights and scales: one tensor-map copy each (lanes 0 and 1)
    const u64 tmap0 = reinterpret_cast<u64>(P.tmap) + (lane & 1) * 128;
    if (lane < 4) asm volatile("prefetch.tensormap [%0];" :: "l"(reinterpret_cast<u64>(P.tmap) + lane * 128) : "memory");
#endif
#ifdef MOE_TRACE
    pst[1] = (u32)clock64();
#endif
    int it = 0;
    int item = 0;
#ifdef MOE_TIMING
    u32 p_empty = 0, p_dep = 0, p_claim = 0, p_poll = 0;
    const u32 p_t0 = ctimer();
#endif
    bool gathered = (GATHER && MW >= 128), routed = false;
    // The first wave is static (CTA b takes item b); the queue hands out items from gridDim on.
    item = blockIdx.x;
    int early = 0;
#ifdef MOE_TRACE
    u32 est[16];
    for (int k = 0; k < 16; ++k) est[k] = 0;
    est[12] = gtimer();
#endif
#ifndef MOE_NO_EARLY
#ifdef MOE_GEARLY2
    if constexpr (GATHER) {
      // Gathering configs: the first item's first stages' weights go out once this CTA's own
      // gather is released (that release waits for every load this SM has in flight), before the
      // wait for every CTA's gather; their activation copies follow it (in the loop).
      if (item < n1) {
        if (lane == 0) mbar_sleep(smem_addr(rdy), 0);
        __syncwarp();
        routed = true;
        const int tile = item / NRB1, rb = item - tile * NRB1;
        const int4 tl = tiles[tile];
        early = NS < ST1 ? NS : ST1;
        const bool wgt = HASWG && (tl.z + 7) / 8 >= WGMIN;
        const unsigned char* esrc = reinterpret_cast<const unsigned char*>(P.w13 + (size_t)tl.x * (8192 * 512));
        const unsigned char* essrc = P.s13 + (size_t)tl.x * (8192 * 256);
        for (int st = 0; st < early; ++st) {
          const u32 full = bars0 + st * 8;
          if (lane == 0) {
            int4* fi = reinterpret_cast<int4*>(&info[st]);
            fi[0] = make_int4(item, 1, tile, rb);
            fi[1] = make_int4(0, st, st == ST1 - 1, tl.z);
            fi[2] = make_int4(tl.y, tl.x, 0, 0);
            mbar_arrive_tx(full, WB + SB + (wgt ? 4 * ((tl.z + 7) & ~7) * 128 : tl.z * XP));
          }
          __syncwarp();
          if (lane < KW * 2) {
            const int kw = lane >> 1;
            const size_t blk = (size_t)(st * KW + kw) * 512 + rb * RGS;
            if (lane & 1) bulk_g2s(stage0 + st * STAGE + WB + kw * RGS * 256, essrc + blk * 256, RGS * 256, full, pol_w);
            else bulk_g2s(stage0 + st * STAGE + kw * RGS * 2048, esrc + blk * 2048, RGS * 2048, full, pol_w);
          }
        }
      }
    }
#endif
    if constexpr (EARLY && !GATHER && !HASWG) {
      // bs 1 config: this warp works out its first item from the token ids
      // itself (same per-expert counts, index-order tiles and offsets as the routing) and issues
      // its first stages while warps 0-7 are still routing.
      u32 em = 0;
      if (lane < M) {
        const int4 v0 = reinterpret_cast<const int4*>(P.ids)[lane * 2], v1 = reinterpret_cast<const int4*>(P.ids)[lane * 2 + 1];
        const int ev[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int e = ev[j] - P.local_start;
          em |= (unsigned)e < 32u ? 1u << e : 0u;
        }
      }
      u32 mybm = 0;
#pragma unroll
      for (int e = 0; e < 32; ++e) {
        const u32 b = __ballot_sync(0xffffffffu, (em >> e) & 1u);
        if (lane == e) mybm = b;
      }
      const int c = __popc(mybm), ntl = (c + T - 1) / T;
      int sc = c, stt = ntl;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const int a = __shfl_up_sync(0xffffffffu, sc, d), b = __shfl_up_sync(0xffffffffu, stt, d);
        if (lane >= d) { sc += a; stt += b; }
      }
      const int tot_t = __shfl_sync(0xffffffffu, stt, 31);
      const int ft_e = stt - ntl, off_e = sc - c;
      if (item < NRB1 * tot_t) {
        const int tile = item / NRB1, rb = item - tile * NRB1;
        const u32 own = __ballot_sync(0xffffffffu, ntl > 0 && ft_e <= tile && tile < ft_e + ntl);
        const int e = __ffs(own) - 1;
        const int c_e = __shfl_sync(0xffffffffu, c, e), ft = __shfl_sync(0xffffffffu, ft_e, e);
        const int p0 = __shfl_sync(0xffffffffu, off_e, e) + (tile - ft) * T;
        const u32 bme = __shfl_sync(0xffffffffu, mybm, e);
        const int kt = tile - ft, n = min(T, c_e - kt * T);
        const int r = (lane < M && ((bme >> lane) & 1u)) ? __popc(bme & ((1u << lane) - 1u)) - kt * T : -1;
        early = NS < ST1 ? NS : ST1;
        const unsigned char* esrc = reinterpret_cast<const unsigned char*>(P.w13 + (size_t)e * (8192 * 512));
        const unsigned char* essrc = P.s13 + (size_t)e * (8192 * 256);
        for (int st = 0; st < early; ++st) {
          const u32 full = bars0 + st * 8;
          if (lane == 0) {
            int4* fi = reinterpret_cast<int4*>(&info[st]);
            fi[0] = make_int4(item, 1, tile, rb);
            fi[1] = make_int4(0, st, st == ST1 - 1, n);
            fi[2] = make_int4(p0, e, 0, 0);
            mbar_arrive_tx(full, WB + SB + n * KST * 2);
          }
          __syncwarp();
#ifdef MOE_TRACE
          __syncwarp();
          if (st < 4) est[st * 3] = gtimer();
#endif
#ifndef MOE_NO_TMAP
          if (lane < 2) tma_4d(stage0 + st * STAGE + (lane ? WB : 0), tmap0, 0, rb * RGS, st * KW, e, full, pol_w);
#else
          if (lane < KW * 2) {
            const int kw = lane >> 1;
            const size_t blk = (size_t)(st * KW + kw) * 512 + rb * RGS;
            if (lane & 1) bulk_g2s(stage0 + st * STAGE + WB + kw * RGS * 256, essrc + blk * 256, RGS * 256, full, pol_w);
            else bulk_g2s(stage0 + st * STAGE + kw * RGS * 2048, esrc + blk * 2048, RGS * 2048, full, pol_w);
          }
#endif
#ifdef MOE_TRACE
          __syncwarp();
          if (st < 4) est[st * 3 + 1] = gtimer();
#endif
          if (r >= 0 && r < n)
            bulk_g2s(stage0 + st * STAGE + WB + SB + r * XP, P.x + (size_t)lane * 4096 + st * KW * 256, KST * 2, full, pol_x);
#ifdef MOE_TRACE
          __syncwarp();
          if (st < 4) est[st * 3 + 2] = gtimer();
#endif
        }
      }
    }
#endif
    if constexpr (EARLY) {
#ifdef MOE_TRACE
      est[13] = gtimer();
#endif
      if (lane == 0) mbar_sleep(smem_addr(rtd), 0);
      __syncwarp();
#ifdef MOE_TRACE
      est[14] = gtimer();
#endif
      ntiles = scal[0]; n1 = scal[1]; total = scal[2]; tsplit = scal[4]; psplit = scal[5];
    }
#ifndef MOE_PD
#define MOE_PD 0
#endif
    // L2 prefetch of the weights and scales MOE_PD stages ahead, into the next item when this one
    // ends sooner: HBM keeps streaming while the shared-memory ring is held by consumers.
    auto item_src = [&](int itm, const unsigned char*& w, const unsigned char*& sc, int& nrg, int& rbo, int& kc, int& ns) {
      int ph2, tl2, ks2, kss2;
      decode_item2(itm, ntiles, tsplit, NRB1, NRB2, KS2, KSX, ph2, tl2, rbo, ks2, kss2);
      if (ph2 == 1) { kc = 0; ns = ST1; }
      else { kc = ks2 * (16 / kss2); ns = 16 / (KW * kss2); }
      const int e2 = tiles[tl2].x;
      nrg = ph2 == 1 ? 512 : 256;
      w = reinterpret_cast<const unsigned char*>(ph2 == 1 ? P.w13 + (size_t)e2 * (8192 * 512) : P.w2 + (size_t)e2 * (4096 * 512));
      sc = ph2 == 1 ? P.s13 + (size_t)e2 * (8192 * 256) : P.s2 + (size_t)e2 * (4096 * 256);
    };
    auto prefetch_stage = [&](const unsigned char* w, const unsigned char* sc, int nrg, int rbo, int kc0) {
      for (int pc = lane; pc < KW * 2; pc += 32) {
        const int kw = pc >> 1;
        const size_t blk = (size_t)(kc0 + kw) * nrg + rbo * RGS;
        if (pc & 1) prefetch_l2(sc + blk * 256, RGS * 256);
        else prefetch_l2(w + blk * 2048, RGS * 2048);
      }
    };
    for (;;) {
      if (item >= total) {
        const int s = it % NS;
        const u32 ph = (it / NS) & 1;
        if (lane == 0) {
          mbar_sleep(smem_addr(&bars[NS + s]), ph ^ 1);
          info[s].item = -1;
          mbar_arrive(smem_addr(&bars[s]));
        }
        break;
      }
      int next;
#ifdef MOE_TRACE
      if (it == 0) pst[2] = (u32)clock64();
      if (lane == 0) MOE_TR(1, 4, item);
#endif
      // The next item, claimed ahead so its latency hides behind this item. Inline PTX: a plain
      // atomicAdd here is warp-aggregated by the compiler, whose shuffle of the result made the
      // producer wait for the round trip before issuing this item's stages.
      asm volatile("{ .reg .pred p; setp.eq.u32 p, %1, 0; @p atom.global.add.u32 %0, [%2], 1; }"
                   : "=r"(next) : "r"(lane), "l"(head) : "memory");
#ifdef MOE_TRACE
      if (it == 0) pst[3] = (u32)clock64();
#endif
      int phase, tile, rb, ks, kss;
      if constexpr (KSX == KS2) {
        int idx;
        decode_item(item, ntiles, NRB1, NRB2 * KS2, phase, tile, idx);
        if (phase == 1) { rb = idx; ks = 0; kss = 1; } else { rb = idx / KS2; ks = idx - rb * KS2; kss = KS2; }
      } else {
        decode_item2(item, ntiles, tsplit, NRB1, NRB2, KS2, KSX, phase, tile, rb, ks, kss);
      }
      const int nst = phase == 1 ? ST1 : 16 / (KW * kss);
#ifdef MOE_TRACE
      if (it == 0) pst[4] = (u32)clock64();
#endif
      const int4 tl = tiles[tile];
      const int e = tl.x, p0 = tl.y, n = tl.z;
      const int nrg_tot = phase == 1 ? 512 : 256;
      const unsigned char* wsrc = reinterpret_cast<const unsigned char*>(
          phase == 1 ? P.w13 + (size_t)e * (8192 * 512) : P.w2 + (size_t)e * (4096 * 512));
      const unsigned char* ssrc = phase == 1 ? P.s13 + (size_t)e * (8192 * 256) : P.s2 + (size_t)e * (4096 * 256);
      const int kc_item = phase == 1 ? 0 : ks * (16 / kss);
      // activation rows this lane copies (r = lane, lane + 32)
      const u16* xrow0 = nullptr;  // set at the item's first stage, once the pair list is ready
      const u16* xrow1 = nullptr;
      bool nx_known = false;
      const unsigned char* nw = nullptr;
      const unsigned char* nsc = nullptr;
      int nnrg = 0, nrbo = 0, nkc = 0, nns = 0;
      u32 rmask = 0;
      // This lane's piece of every stage (the weights or scales of k chunk lane / 2): a source
      // advanced by one stage's k per stage and a fixed offset in the slot. Computed once per item
      // so a stage costs the producer (which shares a scheduler with two consumer warps) a few adds.
      const int pkw = lane >> 1;
      const bool pscl = lane & 1;
      const unsigned char* csrc = pscl ? ssrc + ((size_t)(kc_item + pkw) * nrg_tot + rb * RGS) * 256
                                       : wsrc + ((size_t)(kc_item + pkw) * nrg_tot + rb * RGS) * 2048;
      const size_t cstep = (size_t)KW * nrg_tot * (pscl ? 256 : 2048);
      const u32 cdst = pscl ? WB + pkw * RGS * 256 : pkw * RGS * 2048;
      const u32 cbytes = pscl ? RGS * 256 : RGS * 2048;
      for (int st = 0; st < nst; ++st, ++it) {
        if (it < early) {  // weights issued before the routing tables (or the gather) were ready
          if constexpr (GATHER) {
            if (!gathered) {
              if (lane == 0) {
                while (ld_acquire(P.ctr + 3) < (int)gridDim.x) {}
                asm volatile("fence.proxy.async.global;" ::: "memory");
              }
              __syncwarp();
              gathered = true;
            }
            const int se = it % NS;
            const u32 fulle = bars0 + se * 8, sbe = stage0 + se * STAGE;
            const unsigned char* xre = reinterpret_cast<const unsigned char*>(P.xs) + (size_t)tile * (16 / KW) * T * XP;
            if (HASWG && (n + 7) / 8 >= WGMIN) {
              if (lane < 4)
                bulk_g2s(sbe + WB + SB + lane * T * 128, xre + ((size_t)(st * KW * 4 + lane) * T) * 128, ((n + 7) & ~7) * 128, fulle, pol_x);
            } else if (lane == 0) {
              bulk_g2s(sbe + WB + SB, xre + (size_t)st * T * XP, n * XP, fulle, pol_x);
            }
          }
          csrc += cstep;
          continue;
        }
        const int s = it % NS;
        const u32 ph = (it / NS) & 1;
        unsigned char* sbase = stages + s * STAGE;
        const u32 full = bars0 + s * 8;
        const int kc0 = kc_item + st * KW;
#if MOE_PD > 0
        if (st == 0 && item < total) {
          // this item's first stages were not prefetched when it was claimed ahead late: do them now
        }
        {
          const int sp = st + MOE_PD;
          if (sp < nst) {
            prefetch_stage(wsrc, ssrc, nrg_tot, rb, kc_item + sp * KW);
          } else {
            if (!nx_known) {
              const int nxt = (int)gridDim.x + __shfl_sync(0xffffffffu, next, 0);
              nx_known = true;
              if (nxt < total) item_src(nxt, nw, nsc, nnrg, nrbo, nkc, nns);
              else nns = 0;
            }
            if (sp - nst < nns) prefetch_stage(nw, nsc, nnrg, nrbo, nkc + (sp - nst) * KW);
          }
        }
#endif
        if (lane == 0) {
#ifdef MOE_TIMING
          const u32 tp0 = ctimer();
#endif
          mbar_sleep(smem_addr(&bars[NS + s]), ph ^ 1);
#ifdef MOE_TIMING
          p_empty += ctimer() - tp0;
#endif
#ifdef MOE_TRACE
          if (it == 0) pst[5] = (u32)clock64();
          MOE_TR(1, 5, item * 16 + st);
#endif
          int4* fi = reinterpret_cast<int4*>(&info[s]);
          fi[0] = make_int4(item, phase, tile, rb);
          fi[1] = make_int4(ks, st, st == nst - 1, n);
          fi[2] = make_int4(p0, e, 0, 0);
#ifdef MOE_NO_LOAD
          mbar_arrive(full);
#else
#ifdef MOE_NO_X
          mbar_arrive_tx(full, WB + SB);
#else
          const bool wgt = HASWG && (n + 7) / 8 >= WGMIN;
          mbar_arrive_tx(full, WB + SB + (wgt ? 4 * ((n + 7) & ~7) * 128 : (phase == 2 || GATHER ? n * XP : n * KST * 2)));
#endif
#endif
        }
        __syncwarp();
#ifdef MOE_NO_LOAD
        continue;
#endif
#ifndef MOE_NO_TMAP
        if (lane < 2) tma_4d(stage0 + s * STAGE + (lane ? WB : 0), tmap0 + (phase == 2 ? 256 : 0), 0, rb * RGS, kc0, e, full, pol_w);
#else
        if (lane < KW * 2) bulk_g2s(stage0 + s * STAGE + cdst, csrc, cbytes, full, pol_w);
        csrc += cstep;
#endif
#ifdef MOE_TRACE
        __syncwarp();
        if (lane == 0) MOE_TR(1, 24, item * 16 + st);
#endif
#ifdef MOE_TRACE
        if (it == 0) pst[6] = (u32)clock64();
#endif
        if (phase == 2) {
          // chunks of the intermediate this stage reads: known ready (rmask) or polled
          u32 need = ((1u << KW) - 1u) << kc0;
          if ((rmask & need) != need) {
#ifdef MOE_TIMING
            const u32 tq0 = ctimer();
#endif
            const bool ok = lane < 16 && ld_acquire(&cnt1[tile * 16 + lane]) >= 512 / RB;
            rmask = __ballot_sync(0xffffffffu, ok);
#ifdef MOE_TIMING
            p_poll += ctimer() - tq0;
#endif
            if (lane == 0) asm volatile("fence.proxy.async.global;" ::: "memory");
          }
          if (lane == 0 && (rmask & need) != need) {
#ifdef MOE_TIMING
            const u32 td0 = ctimer();
#endif
            MOE_TR(1, 6, item);
            for (int kw = 0; kw < KW; ++kw)
              while (ld_acquire(&cnt1[tile * 16 + kc0 + kw]) < 512 / RB) {}
            MOE_TR(1, 7, item);
#ifdef MOE_TIMING
            p_dep += ctimer() - td0;
#endif
            asm volatile("fence.proxy.async.global;" ::: "memory");
          }
          __syncwarp();
        }
#ifndef MOE_NO_X
        if (phase == 2 || GATHER) {
          if (phase == 1 && !gathered) {
            if (lane == 0) {
              while (ld_acquire(P.ctr + 3) < (int)gridDim.x) {}
              asm volatile("fence.proxy.async.global;" ::: "memory");
            }
            __syncwarp();
            gathered = true;
          }
          const unsigned char* xreg = reinterpret_cast<const unsigned char*>(phase == 2 ? P.inter : P.xs) +
                                      (size_t)tile * (16 / KW) * T * XP;
          if (HASWG && (n + 7) / 8 >= WGMIN) {
            if (lane < 4)  // chunk lane of this stage's 4 x 64 k
              bulk_g2s(smem_addr(sbase + WB + SB + lane * T * 128), xreg + ((size_t)(kc0 * 4 + lane) * T) * 128,
                       ((n + 7) & ~7) * 128, full, pol_x);
          } else if (lane == 0) {
            bulk_g2s(smem_addr(sbase + WB + SB), xreg + (size_t)(kc0 / KW) * T * XP, n * XP, full, pol_x);
          }
        } else {
          if (st == 0 || (early > 0 && it == early)) {
            if (!routed) {
              if (lane == 0) mbar_sleep(smem_addr(rdy), 0);
              __syncwarp();
              routed = true;
            }
            if (lane < n) xrow0 = P.x + (size_t)pair_tok[p0 + lane] * 4096;
            if (lane + 32 < n) xrow1 = P.x + (size_t)pair_tok[p0 + lane + 32] * 4096;
          }
          if (xrow0) bulk_g2s(smem_addr(sbase + WB + SB + lane * XP), xrow0 + kc0 * 256, KST * 2, full, pol_x);
          if (xrow1) bulk_g2s(smem_addr(sbase + WB + SB + (lane + 32) * XP), xrow1 + kc0 * 256, KST * 2, full, pol_x);
        }
#endif
      }
#ifdef MOE_TIMING
      const u32 tn0 = ctimer();
#endif
      item = (int)gridDim.x + __shfl_sync(0xffffffffu, next, 0);
#ifdef MOE_TIMING
      if (item >= 0) p_claim += ctimer() - tn0;
#endif
    }
#ifdef MOE_TRACE
    if (lane == 0) {
      int* eb = P.ctr + 32768 + 160 * 512 + 160 * 192 + 160 * 80 + blockIdx.x * 16;
      for (int k = 0; k < 15; ++k) eb[k] = (int)est[k];
    }
    if (lane == 0) {
      int* b = P.ctr + 32768 + blockIdx.x * 512 + 256 + 192;
      b[0] = (int)pgt;
      for (int k = 1; k < 7; ++k) b[k] = (int)(pst[k] - pst[0]);
    }
#endif
#ifdef MOE_TIMING
    if (lane == 0) {
      tdbg[7] = p_empty; tdbg[8] = p_dep;
      u32* pt = reinterpret_cast<u32*>(P.ctr) + 14420 + blockIdx.x * 4;
      pt[1] = p_poll;
    }
#endif
    return;
  }

  if (warp == 9) {
    // ------------------------------ signaler ------------------------------
    // Publishes finished items (gate/up: per tile; down: total) once their outputs are stored,
    // off the consumers' critical path.
    if (lane == 0) {
      for (int q = 0;; ++q) {
        const int j = q % NSIG;
        mbar_sleep(smem_addr(&epi[j]), (q / NSIG) & 1);
        const int2 si = sinfo[j];
        mbar_arrive(smem_addr(&sfree[j]));
        if (si.x < 0) break;
        // The release is cumulative: the consumers' stores, seen through the CTA-scope barrier
        // above, are visible to whoever acquires the counter.
#ifdef MOE_SIGFENCE
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
#endif
        red_release_add(si.x == 1 ? &cnt1[si.y] : (TILEDONE ? &cnt2[si.y] : done2), 1);
      }
    }
    return;
  }
  return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 " MOE_STR(MOE_CREG) ";" ::: "memory");
  if (tid == 0) MOE_TR(0, 15, 0);

  // ------------------------------ consumers ------------------------------
  const int kw = warp % KW, rw = warp / KW;
  int q_item = 0;
  auto post_signal = [&](int phase, int tile, int rb) {
    const int j = q_item % NSIG;
    if (warp == 0 && lane == 0) {
      mbar_wait(smem_addr(&sfree[j]), ((q_item / NSIG) & 1) ^ 1);
      sinfo[j] = make_int2(phase, phase == 1 ? tile * 16 + rb * RB / 512 : tile);
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(smem_addr(&epi[j]));
    ++q_item;
  };
  const int g = lane >> 2, tig = lane & 3;
#ifdef MOE_TIMING
  u32 c_wait = 0, c_epi = 0, c_items = 0, c_comp = 0, c_stages = 0, c_red = 0, c_store = 0, c_sig = 0, wcomp = 0;
#endif
  float acc[2][NT][4], acc2[2][NT][4];
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int b = 0; b < NT; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) { acc[a][b][c] = 0.f; acc2[a][b][c] = 0.f; }
  int it = 0;
  for (;;) {
    int s, ntl;
    u32 ph;
    StageInfo f;
#ifdef MOE_TIMING
    u32 tw0;
#endif
    // Stage info read and dispatched once per item, except for KW = 1 with a k split, where the
    // per-stage loop measures faster.
    // Stage info read and dispatched once per item (every config)
    if constexpr (true) {
    s = it % NS;
    ph = (it / NS) & 1;
#ifdef MOE_TIMING
    tw0 = ctimer();
#endif
    mbar_wait(smem_addr(&bars[s]), ph);
#ifdef MOE_TIMING
    c_wait += ctimer() - tw0;
    if (it == 0 && tid == 0) tdbg[15] = gtimer();  // first stage's data in shared memory
#endif
    f = info[s];
    if (f.item < 0) {
      post_signal(-1, 0, 0);
      break;
    }
    if (tid == 0) MOE_TR(0, 1, f.item);
    ntl = (f.n + 7) >> 3;
    const int nst = f.phase == 1 ? ST1 : 16 / (KW * (f.tile >= tsplit ? KSX : KS2));
#ifdef MOE_PI_MARK
    if (tid == 0) atomicAdd(P.ctr + 13000, 1);
#endif
#ifdef MOE_TIMING
#define MOE_T(x) x
#else
#define MOE_T(x)
#endif
    // All of the item's stages with the token-group count fixed at compile time: the stage info
    // is read and dispatched once per item. K = NTL for mma.sync, 100 + NTL for wgmma. The
    // item's last slot stays held for the epilogue's staging.
// Multi-group tiles: the loop waits at its top, so the (fully unrolled) stage body is emitted once
// (the bottom-tested form below is inverted by the compiler, which duplicates the body). Single-
// group tiles keep the bottom-tested form, which measures faster there.
#define MOE_RUN_ITEM(K)                                                                                  \
    if constexpr (NT == 1) { MOE_RUN_ITEM_BOT(K) } else { MOE_RUN_ITEM_TOP(K) }
#define MOE_RUN_ITEM_BOT(K)                                                                              \
    for (int st = 0;;) {                                                                                 \
      MOE_T(const u32 tc0 = ctimer();)                                                                   \
      const unsigned char* sbase = stages + s * STAGE;                                                   \
      const unsigned char* wbase = sbase + (kw * RGS + rw * 2) * 2048 + lane * 16;                       \
      const unsigned char* sclb = sbase + WB + (kw * RGS + rw * 2) * 256 + lane * 8;                     \
      if constexpr (MOE_NOC) {                                                                           \
      } else if constexpr ((K) >= 100) {                                                                 \
        if constexpr (HASWG) stage_compute_wg<NT, (K) - 100>(acc, wbase, sclb, smem_addr(sbase + WB + SB), T); \
      } else {                                                                                           \
        stage_compute<NT, (K)>(acc, acc2, wbase, sclb, sbase + WB + SB + g * XP + kw * 512 + tig * 32, XP); \
      }                                                                                                  \
      __syncwarp();                                                                                      \
      MOE_T(c_comp += ctimer() - tc0; c_stages += 1; if (lane == 0) wcomp += ctimer() - tc0;)            \
      if (++st == nst) break;                                                                            \
      if (lane == 0) mbar_arrive(smem_addr(&bars[NS + s]));                                              \
      if (++s == NS) { s = 0; ph ^= 1; }                                                                 \
      MOE_T(tw0 = ctimer();)                                                                             \
      mbar_wait(smem_addr(&bars[s]), ph);                                                                \
      MOE_T(c_wait += ctimer() - tw0;)                                                                   \
    }
#define MOE_RUN_ITEM_TOP(K)                                                                              \
    for (int st = 0; st < nst; ++st) {                                                                   \
      if (st > 0) {  /* the previous stage's release and this stage's wait */                           \
        if (lane == 0) mbar_arrive(smem_addr(&bars[NS + s]));                                            \
        if (++s == NS) { s = 0; ph ^= 1; }                                                               \
        MOE_T(tw0 = ctimer();)                                                                           \
        mbar_wait(smem_addr(&bars[s]), ph);                                                              \
        MOE_T(c_wait += ctimer() - tw0;)                                                                 \
      }                                                                                                  \
      MOE_T(const u32 tc0 = ctimer();)                                                                   \
      const unsigned char* sbase = stages + s * STAGE;                                                   \
      const unsigned char* wbase = sbase + (kw * RGS + rw * 2) * 2048 + lane * 16;                       \
      const unsigned char* sclb = sbase + WB + (kw * RGS + rw * 2) * 256 + lane * 8;                     \
      if constexpr (MOE_NOC) {                                                                           \
      } else if constexpr ((K) >= 100) {                                                                 \
        if constexpr (HASWG) stage_compute_wg<NT, (K) - 100>(acc, wbase, sclb, smem_addr(sbase + WB + SB), T); \
      } else {                                                                                           \
        stage_compute<NT, (K)>(acc, acc2, wbase, sclb, sbase + WB + SB + g * XP + kw * 512 + tig * 32, XP); \
      }                                                                                                  \
      __syncwarp();                                                                                      \
      MOE_T(c_comp += ctimer() - tc0; c_stages += 1; if (lane == 0) wcomp += ctimer() - tc0;)            \
    }
    if (HASWG && ntl >= WGMIN) {
      if constexpr (HASWG) {
        switch (ntl) {
          case 3: MOE_RUN_ITEM(103) break;
          case 4: MOE_RUN_ITEM(104) break;
          case 5: if constexpr (NT >= 5) { MOE_RUN_ITEM(105) } break;
          case 6: if constexpr (NT >= 6) { MOE_RUN_ITEM(106) } break;
          case 7: if constexpr (NT >= 7) { MOE_RUN_ITEM(107) } break;
          case 8: if constexpr (NT >= 8) { MOE_RUN_ITEM(108) } break;
          case 9: if constexpr (NT >= 9) { MOE_RUN_ITEM(109) } break;
          case 10: if constexpr (NT >= 10) { MOE_RUN_ITEM(110) } break;
          case 11: if constexpr (NT >= 11) { MOE_RUN_ITEM(111) } break;
          case 12: if constexpr (NT >= 12) { MOE_RUN_ITEM(112) } break;
          case 13: if constexpr (NT >= 13) { MOE_RUN_ITEM(113) } break;
          case 14: if constexpr (NT >= 14) { MOE_RUN_ITEM(114) } break;
          case 15: if constexpr (NT >= 15) { MOE_RUN_ITEM(115) } break;
          default: if constexpr (NT >= 16) { MOE_RUN_ITEM(116) } break;
        }
      }
    } else {
      switch (ntl) {
        case 1: MOE_RUN_ITEM(1) break;
        case 2: if constexpr (NT >= 2) { MOE_RUN_ITEM(2) } break;
        case 3: if constexpr (NT >= 3) { MOE_RUN_ITEM(3) } break;
        default: if constexpr (NT >= 4) { MOE_RUN_ITEM((NT < 4 ? NT : 4)) } break;
      }
    }
#undef MOE_RUN_ITEM
#undef MOE_RUN_ITEM_BOT
#undef MOE_RUN_ITEM_TOP
    it += nst;
    } else {
    s = it % NS;
    ph = (it / NS) & 1;
#ifdef MOE_TIMING
    tw0 = ctimer();
#endif
    mbar_wait(smem_addr(&bars[s]), ph);
#ifdef MOE_TIMING
    c_wait += ctimer() - tw0;
    if (it == 0 && tid == 0) tdbg[15] = gtimer();  // first stage's data in shared memory
#endif
    f = info[s];
    if (f.item < 0) {
      post_signal(-1, 0, 0);
      break;
    }
    if (tid == 0 && f.st == 0) MOE_TR(0, 1, f.item);
    ntl = (f.n + 7) >> 3;
#ifdef MOE_TIMING
    const u32 tc0 = ctimer();
#endif
    const unsigned char* sbase = stages + s * STAGE;
    const unsigned char* wbase = sbase + (kw * RGS + rw * 2) * 2048 + lane * 16;
    const unsigned char* sclb = sbase + WB + (kw * RGS + rw * 2) * 256 + lane * 8;
    const unsigned char* xbase = sbase + WB + SB + g * XP + kw * 512 + tig * 32;
    if (MOE_NOC) {
    } else if (HASWG && ntl >= WGMIN) {
      if constexpr (HASWG) {
        const u32 xaddr = smem_addr(sbase + WB + SB);
        switch (ntl) {
          case 3: stage_compute_wg<NT, 3>(acc, wbase, sclb, xaddr, T); break;
          case 4: stage_compute_wg<NT, 4>(acc, wbase, sclb, xaddr, T); break;
          case 5: if constexpr (NT >= 5) stage_compute_wg<NT, 5>(acc, wbase, sclb, xaddr, T); break;
          case 6: if constexpr (NT >= 6) stage_compute_wg<NT, 6>(acc, wbase, sclb, xaddr, T); break;
          case 7: if constexpr (NT >= 7) stage_compute_wg<NT, 7>(acc, wbase, sclb, xaddr, T); break;
          case 8: if constexpr (NT >= 8) stage_compute_wg<NT, 8>(acc, wbase, sclb, xaddr, T); break;
          case 9: if constexpr (NT >= 9) stage_compute_wg<NT, 9>(acc, wbase, sclb, xaddr, T); break;
          case 10: if constexpr (NT >= 10) stage_compute_wg<NT, 10>(acc, wbase, sclb, xaddr, T); break;
          case 11: if constexpr (NT >= 11) stage_compute_wg<NT, 11>(acc, wbase, sclb, xaddr, T); break;
          case 12: if constexpr (NT >= 12) stage_compute_wg<NT, 12>(acc, wbase, sclb, xaddr, T); break;
          case 13: if constexpr (NT >= 13) stage_compute_wg<NT, 13>(acc, wbase, sclb, xaddr, T); break;
          case 14: if constexpr (NT >= 14) stage_compute_wg<NT, 14>(acc, wbase, sclb, xaddr, T); break;
          case 15: if constexpr (NT >= 15) stage_compute_wg<NT, 15>(acc, wbase, sclb, xaddr, T); break;
          default: if constexpr (NT >= 16) stage_compute_wg<NT, 16>(acc, wbase, sclb, xaddr, T); break;
        }
      }
    } else {
      switch (ntl) {
        case 1: stage_compute<NT, 1>(acc, acc2, wbase, sclb, xbase, XP); break;
        case 2: if constexpr (NT >= 2) stage_compute<NT, 2>(acc, acc2, wbase, sclb, xbase, XP); break;
        case 3: if constexpr (NT >= 3) stage_compute<NT, 3>(acc, acc2, wbase, sclb, xbase, XP); break;
        default: if constexpr (NT >= 4) stage_compute<NT, (NT < 4 ? NT : 4)>(acc, acc2, wbase, sclb, xbase, XP); break;
      }
    }
    __syncwarp();
#ifdef MOE_TIMING
    c_comp += ctimer() - tc0; c_stages += 1;
    if (lane == 0) wcomp += ctimer() - tc0;
#endif
    if (lane == 0 && !f.last) mbar_arrive(smem_addr(&bars[NS + s]));
    ++it;
    if (!f.last) continue;
    }

    // ------------------------------ item epilogue ------------------------------
    // The item's last stage slot stays held. Each warp stages its fp32 accumulators there
    // (token-major, one copy per k-warp); then all 256 consumer threads sum the k-warps, apply
    // SwiGLU or the global scale and write 16-byte coalesced stores, and the slot is released.
#ifdef MOE_SPLITACC
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
      for (int b = 0; b < NT; ++b)
#pragma unroll
        for (int c = 0; c < 4; ++c) { acc[a][b][c] += acc2[a][b][c]; acc2[a][b][c] = 0.f; }
#endif
#ifdef MOE_TIMING
    const u32 te0 = ctimer();
#endif
    if (tid == 0) MOE_TR(0, 2, f.item);
    {
      constexpr int EPI_NT = EPI_T / 8;
      static_assert(T % EPI_T == 0, "whole epilogue slices");
#pragma unroll
      for (int tilepart = 0; tilepart < T / EPI_T; ++tilepart) {
      const int token_base = tilepart * EPI_T;
      constexpr int SP = RB + 4;  // floats per staged token row; the +4 keeps fragment stores conflict-free
      static_assert(KW * EPI_T * SP * 4 <= STAGE, "epilogue staging must fit in one stage slot");
      float* stg = reinterpret_cast<float*>(stages + s * STAGE);
      consumer_sync();  // every warp is done reading the slot
#pragma unroll
      for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < NT; ++b)
#pragma unroll
          for (int c = 0; c < 4; ++c)
            if (b / EPI_NT == tilepart) stg[(kw * EPI_T + (b % EPI_NT) * 8 + 2 * tig + (c & 1)) * SP + 16 * (2 * rw + a) + g + 8 * (c >> 1)] = acc[a][b][c];
      consumer_sync();
#ifdef MOE_TIMING
      const u32 te2 = ctimer();
      c_red += te2 - te0;
#endif
      const int nn = max(0, min(EPI_T, f.n - token_base));
      if (f.phase == 1) {
        // gate rows 16 j + 0..7 and up rows 16 j + 8..15 of the item give intermediate 8 j + 0..7
        const float gsc = gsm[f.e];
        auto swiglu2 = [&](int tok, int row) -> u32 {  // rows row, row + 1 (gate) and +8 (up)
          float g0 = 0.f, g1 = 0.f, u0 = 0.f, u1 = 0.f;
#pragma unroll
          for (int k2 = 0; k2 < KW; ++k2) {
            const float* r = stg + (k2 * EPI_T + tok) * SP + row;
            const float2 gv = *reinterpret_cast<const float2*>(r), uv = *reinterpret_cast<const float2*>(r + 8);
            g0 += gv.x; g1 += gv.y; u0 += uv.x; u1 += uv.y;
          }
          g0 *= gsc; g1 *= gsc; u0 *= gsc; u1 *= gsc;
          return (u32)f2bf(__fdividef(g0, 1.f + __expf(-g0)) * u0) | ((u32)f2bf(__fdividef(g1, 1.f + __expf(-g1)) * u1) << 16);
        };
        if (HASWG && ntl >= WGMIN) {
          // the down GEMM runs this tile on wgmma: per 64-k chunk and token, 8 swizzled 16-byte
          // groups; group pc holds positions 8 lc .. 8 lc + 7 (lc = pc ^ token % 8), i.e. k pairs
          // 16 e + 4 (lc / 2) + 2 (lc % 2) + {0, 1}
          u16* dt = P.inter + (size_t)f.tile * (16 / KW) * T * XP / 2;
          for (int o = tid; o < nn * (RB / 16); o += 256) {
            const int pc = o & 7, rest = o >> 3;
            const int c64l = rest / nn, tok = rest - c64l * nn;
            const int lc = pc ^ (tok & 7);
            u32 h[4];
#pragma unroll
            for (int e2 = 0; e2 < 4; ++e2) {
              const int il = 64 * c64l + 16 * e2 + 4 * (lc >> 1) + 2 * (lc & 1);
              h[e2] = swiglu2(tok, 16 * (il >> 3) + (il & 7));
            }
            const int c64g = (RB / 128) * f.rb + c64l;
            *reinterpret_cast<uint4*>(dt + ((size_t)c64g * T + tok + token_base) * 64 + pc * 8) = make_uint4(h[0], h[1], h[2], h[3]);
          }
        } else {
          // [tile][stage of KST k][T rows][XP bytes]: the down GEMM's stage arrives in one copy
          for (int o = tid; o < nn * (RB / 16); o += 256) {
            const int tok = o / (RB / 16), j8 = o - tok * (RB / 16);
            u32 h[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) h[q] = swiglu2(tok, 16 * j8 + 2 * q);
            const int ig = (RB / 2) * f.rb + 8 * j8;
            *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(P.inter) +
                (((size_t)f.tile * (16 / KW) + ig / KST) * T + tok + token_base) * XP + (ig % KST) * 2) = make_uint4(h[0], h[1], h[2], h[3]);
          }
        }
      } else {
        const float gsc = gsm[32 + f.e];
        float* yb = P.ypart + (size_t)f.ks * P.pmax * 4096 + (size_t)(f.p0 + token_base) * 4096 + f.rb * RB;
if constexpr (SSQ) {
        // An item over the whole k range holds its rows' final y: it also leaves, per token and
        // 64-row slot, the sum of squares of y * (1 + w), so the combine needs no row reduction.
        // A half-warp covers one slot (RB / 4 is a multiple of 16), reduced by shuffles.
        const bool full_k = (f.tile >= tsplit ? KSX : KS2) == 1;
        const int lim = nn * (RB / 4);
        for (int o0 = 0; o0 < lim; o0 += 256) {
          const int o = o0 + tid;
          const bool live = o < lim;
          const int tok = o / (RB / 4), r4 = o - tok * (RB / 4);
          float sq = 0.f;
          if (live) {
            float4 v = *reinterpret_cast<const float4*>(stg + tok * SP + 4 * r4);
#pragma unroll
            for (int k2 = 1; k2 < KW; ++k2) {
              const float4 w = *reinterpret_cast<const float4*>(stg + (k2 * EPI_T + tok) * SP + 4 * r4);
              v.x += w.x; v.y += w.y; v.z += w.z; v.w += w.w;
            }
            v.x *= gsc; v.y *= gsc; v.z *= gsc; v.w *= gsc;
            if (full_k) {
              v = bf_round4(v);
              st_bf4(P.yh + (size_t)(f.p0 + tok + token_base) * 4096 + f.rb * RB + 4 * r4, v);
            } else {
              *reinterpret_cast<float4*>(yb + (size_t)tok * 4096 + 4 * r4) = v;
            }
            if (full_k) {
              const float4 w = *reinterpret_cast<const float4*>(P.nw1 + f.rb * RB + 4 * r4);
              const float a = v.x * w.x, b = v.y * w.y, c = v.z * w.z, d = v.w * w.w;
              sq = a * a + b * b + c * c + d * d;
            }
          }
          if (full_k) {
#pragma unroll
            for (int s2 = 8; s2 > 0; s2 >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, s2);
            if (live && (lane & 15) == 0) P.ssq[(size_t)(f.p0 + tok + token_base) * 64 + (f.rb * RB + 4 * r4) / 64] = sq;
          }
        }
} else {
        const bool full_k = (f.tile >= tsplit ? KSX : KS2) == 1;
        for (int o = tid; o < nn * (RB / 4); o += 256) {
          const int tok = o / (RB / 4), r4 = o - tok * (RB / 4);
          float4 v = *reinterpret_cast<const float4*>(stg + tok * SP + 4 * r4);
#pragma unroll
          for (int k2 = 1; k2 < KW; ++k2) {
            const float4 w = *reinterpret_cast<const float4*>(stg + (k2 * EPI_T + tok) * SP + 4 * r4);
            v.x += w.x; v.y += w.y; v.z += w.z; v.w += w.w;
          }
          v.x *= gsc; v.y *= gsc; v.z *= gsc; v.w *= gsc;
          if (full_k) st_bf4(P.yh + (size_t)(f.p0 + tok + token_base) * 4096 + f.rb * RB + 4 * r4, v);
          else *reinterpret_cast<float4*>(yb + (size_t)tok * 4096 + 4 * r4) = v;
        }
}
      }
#ifdef MOE_TIMING
      c_store += ctimer() - te2;
#endif
    }
      }  // epilogue token slices

    __syncwarp();
    if (lane == 0) mbar_arrive(smem_addr(&bars[NS + s]));  // staging read: the slot is free
#ifdef MOE_TIMING
    const u32 te3 = ctimer();
#endif
    post_signal(f.phase, f.tile, f.rb);
#ifdef MOE_TIMING
    c_sig += ctimer() - te3;
#endif
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
      for (int b = 0; b < NT; ++b)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;
#ifdef MOE_TIMING
    c_epi += ctimer() - te0; c_items += 1;
#endif
    if (tid == 0) MOE_TR(0, 3, f.item);
  }

#ifdef MOE_TIMING
  if (tid == 0) {
    tdbg[2] = gtimer(); tdbg[11] = ctimer() - tdbg[11];
    tdbg[4] = c_wait; tdbg[5] = c_epi; tdbg[6] = c_items; tdbg[9] = c_comp; tdbg[10] = c_stages;
    tdbg[12] = c_red; tdbg[13] = c_store; tdbg[14] = c_sig;
  }
  if (lane == 0) {
    reinterpret_cast<u32*>(P.ctr)[15000 + blockIdx.x * 8 + warp] = wcomp;
  }
#endif
  // ------------------------------ combine ------------------------------
#ifdef MOE_NO_COMBINE
  if (M > 0) goto done_combine;
#endif
  {
  // A token is combined as soon as the down items of its tiles are done, not after the last
  // item of the whole queue: most of the combine overlaps the other CTAs' remaining work.
  float* rbuf = reinterpret_cast<float*>(stages);  // [2][4][8] partial sums
  if (tid == 0) MOE_TR(0, 8, 0);
  if constexpr (!TILEDONE) {
#ifndef MOE_NO_WORK
    if (tid == 0) {
      while (ld_acquire(done2) < total - n1) {}
    }
#endif
    consumer_sync();
    if (tid == 0) MOE_TR(0, 20, 0);
  }
  // a token whose pairs all come from full-k items has its sums of squares ready (fast path)
  auto token_fast = [&](int t) -> bool {
    bool fast = true;
#pragma unroll
    for (int sl = 0; sl < 8; ++sl) {
      const int e = lid[t * 8 + sl];
      if (e >= 0) {
        const u32 word = bm[e * MW + (t >> 5)];
        const int pr = wpre[e * MW + (t >> 5)] + __popc(word & ((1u << (t & 31)) - 1u));
        const int tl = ftile[e] + pr / T;
        fast = fast && (tl >= tsplit ? KSX : KS2) == 1;
      }
    }
    return fast;
  };
  // Combine work is claimed from a counter (tokens here; units below), so the CTAs that finish
  // their items first take it and each token is combined as soon as its tiles are done, instead
  // of waiting for a fixed CTA that may itself be among the last to finish.
#ifndef MOE_NO_DYNCOMB
  constexpr bool DYN = !SSQ;
#else
  constexpr bool DYN = false;
#endif
  auto claim = [&](int* ctr_word, int static_first, int static_step, int& prev) -> int {
    if constexpr (!DYN) { prev = prev < 0 ? static_first : prev + static_step; return prev; }
    if (tid == 0) scal[8] = atomicAdd(ctr_word, 1);
    consumer_sync();
    const int v = scal[8];
    consumer_sync();
    return v;
  };
  int tprev = -1;
  for (int t = SSQ ? blockIdx.x : claim(P.ctr + 4, blockIdx.x, gridDim.x, tprev); t < M;
       t = SSQ ? t + (int)gridDim.x : claim(P.ctr + 4, blockIdx.x, gridDim.x, tprev)) {
    if constexpr (SSQ) { if (token_fast(t)) continue; }
    int plist[8];
    float wl[8];
    int np = 0;
#pragma unroll
    for (int sl = 0; sl < 8; ++sl) {
      const int e = lid[t * 8 + sl];
      plist[sl] = 0;
      wl[sl] = 0.f;
      if (e >= 0) {
        const u32 word = bm[e * MW + (t >> 5)];
        const int pr = wpre[e * MW + (t >> 5)] + __popc(word & ((1u << (t & 31)) - 1u));  // rank in the expert
        plist[np] = off[e] + pr;
        wl[np] = P.topk_w[t * 8 + sl];
        if (TILEDONE && tid == np * 32) {  // one warp per pair: the polls run side by side
          const int tl = ftile[e] + pr / T;
          const int want = NRB2 * (tl >= tsplit ? KSX : KS2);
#ifndef MOE_NO_WORK
          while (ld_acquire(&cnt2[tl]) < want) {}
#endif
        }
        ++np;
      }
    }
    if constexpr (TILEDONE) consumer_sync();
    float accv[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) accv[k] = 0.f;
    // (full-k one-k-warp configs, BF16 rows: two pairs per barrier beats pair by pair)
    if constexpr (KW > 1 || KS2 == 1) {
    // Pairs in groups of CG: their loads fly together and one barrier reduces all their sums of
    // squares (rbuf [2][4][8], alternating groups).
    constexpr int CG = (KSM > 2 || KW == 1) ? 2 : 4;
    int rnd = 0;
    for (int q0 = 0; q0 < np; q0 += CG, ++rnd) {
      float v[CG][16];
      float ss[CG];
#pragma unroll
      for (int gq = 0; gq < CG; ++gq) {
        ss[gq] = 0.f;
        int p = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) if (j == q0 + gq) p = plist[j];
        const bool live = q0 + gq < np;
        const int kp = p >= psplit ? KSX : KS2;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int n = (tid + 256 * k) * 4;
          float4 y = make_float4(0.f, 0.f, 0.f, 0.f);
          if (live && kp == 1) y = ld_bf4(P.yh + (size_t)p * 4096 + n);
#pragma unroll
          for (int k2 = 0; k2 < KSM; ++k2)
            if (live && kp > 1 && k2 < kp) {
              const float4 u = __ldcg(reinterpret_cast<const float4*>(P.ypart + ((size_t)k2 * P.pmax + p) * 4096 + n));
              y.x += u.x; y.y += u.y; y.z += u.z; y.w += u.w;
            }
          const float4 w = *reinterpret_cast<const float4*>(P.nw1 + n);
          v[gq][4 * k + 0] = y.x * w.x; v[gq][4 * k + 1] = y.y * w.y; v[gq][4 * k + 2] = y.z * w.z; v[gq][4 * k + 3] = y.w * w.w;
          ss[gq] += v[gq][4 * k] * v[gq][4 * k] + v[gq][4 * k + 1] * v[gq][4 * k + 1] +
                    v[gq][4 * k + 2] * v[gq][4 * k + 2] + v[gq][4 * k + 3] * v[gq][4 * k + 3];
        }
      }
#pragma unroll
      for (int s2 = 16; s2 > 0; s2 >>= 1)
#pragma unroll
        for (int gq = 0; gq < CG; ++gq) ss[gq] += __shfl_xor_sync(0xffffffffu, ss[gq], s2);
      if (lane == 0)
#pragma unroll
        for (int gq = 0; gq < CG; ++gq) rbuf[(rnd & 1) * 32 + gq * 8 + warp] = ss[gq];
      consumer_sync();
#pragma unroll
      for (int gq = 0; gq < CG; ++gq) {
        if (q0 + gq < np) {
          float tot = 0.f;
#pragma unroll
          for (int w = 0; w < 8; ++w) tot += rbuf[(rnd & 1) * 32 + gq * 8 + w];
          float gw = 0.f;
#pragma unroll
          for (int j = 0; j < 8; ++j) if (j == q0 + gq) gw = wl[j];
          const float r = rsqrtf(tot * (1.f / 4096.f) + 1e-8f) * gw;
#pragma unroll
          for (int k = 0; k < 16; ++k) accv[k] += r * v[gq][k];
        }
      }
    }
    } else {
    // KW = 1 configs: pair by pair, the next pair's loads in flight during this pair's reduction
    float4 cur[4 * KSM];
    auto load_pair = [&](int p, float4* dst) {
      const int kp = p >= psplit ? KSX : KS2;  // partials of this pair
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int n = (tid + 256 * k) * 4;
#pragma unroll
        for (int k2 = 0; k2 < KSM; ++k2)
          dst[k * KSM + k2] = kp == 1 ? (k2 == 0 ? ld_bf4(P.yh + (size_t)p * 4096 + n) : make_float4(0.f, 0.f, 0.f, 0.f))
                              : k2 < kp ? __ldcg(reinterpret_cast<const float4*>(P.ypart + ((size_t)k2 * P.pmax + p) * 4096 + n))
                                        : make_float4(0.f, 0.f, 0.f, 0.f);
      }
    };
    if (np > 0) load_pair(plist[0], cur);
    for (int q = 0; q < np; ++q) {
      float v[16];
      float ss = 0.f;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        float4 y = cur[k * KSM];
#pragma unroll
        for (int k2 = 1; k2 < KSM; ++k2) {
          const float4 u = cur[k * KSM + k2];
          y.x += u.x; y.y += u.y; y.z += u.z; y.w += u.w;
        }
        const float4 w = *reinterpret_cast<const float4*>(P.nw1 + (tid + 256 * k) * 4);
        v[4 * k + 0] = y.x * w.x; v[4 * k + 1] = y.y * w.y; v[4 * k + 2] = y.z * w.z; v[4 * k + 3] = y.w * w.w;
        ss += v[4 * k] * v[4 * k] + v[4 * k + 1] * v[4 * k + 1] + v[4 * k + 2] * v[4 * k + 2] + v[4 * k + 3] * v[4 * k + 3];
      }
      int pn = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) if (j == q + 1) pn = plist[j];
      if (q + 1 < np) load_pair(pn, cur);  // next pair's loads fly during this pair's reduction
#pragma unroll
      for (int s2 = 16; s2 > 0; s2 >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, s2);
      if (lane == 0) rbuf[(q & 1) * 8 + warp] = ss;
      consumer_sync();
      float tot = 0.f;
#pragma unroll
      for (int w = 0; w < 8; ++w) tot += rbuf[(q & 1) * 8 + w];
      float gw = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) if (j == q) gw = wl[j];
      const float r = rsqrtf(tot * (1.f / 4096.f) + 1e-8f) * gw;
#pragma unroll
      for (int k = 0; k < 16; ++k) accv[k] += r * v[k];
    }
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int n = (tid + 256 * k) * 4;
      const u32 lo = (u32)f2bf(accv[4 * k]) | ((u32)f2bf(accv[4 * k + 1]) << 16);
      const u32 hi = (u32)f2bf(accv[4 * k + 2]) | ((u32)f2bf(accv[4 * k + 3]) << 16);
      *reinterpret_cast<uint2*>(P.out + (size_t)t * 4096 + n) = make_uint2(lo, hi);
    }
  }
  if constexpr (SSQ) {
  // Fast tokens: an elementwise pass in (token, 1024-column) units spread over every CTA. Each
  // warp sums a pair's 64 slot sums itself, so no barrier is needed beyond the tiles' polls.
  int uprev = -1;
  for (int u = claim(P.ctr + 5, blockIdx.x, gridDim.x, uprev); u < M * 4; u = claim(P.ctr + 5, blockIdx.x, gridDim.x, uprev)) {
    const int t = u >> 2, c = u & 3;
    if (!token_fast(t)) continue;
    int plist[8];
    float wl[8];
    int np = 0;
#pragma unroll
    for (int sl = 0; sl < 8; ++sl) {
      const int e = lid[t * 8 + sl];
      plist[sl] = 0;
      wl[sl] = 0.f;
      if (e >= 0) {
        const u32 word = bm[e * MW + (t >> 5)];
        const int pr = wpre[e * MW + (t >> 5)] + __popc(word & ((1u << (t & 31)) - 1u));
        plist[np] = off[e] + pr;
        wl[np] = P.topk_w[t * 8 + sl];
        if (TILEDONE && tid == np * 32) {
          const int tl = ftile[e] + pr / T;
          const int want = NRB2 * (tl >= tsplit ? KSX : KS2);
#ifndef MOE_NO_WORK
          while (ld_acquire(&cnt2[tl]) < want) {}
#endif
        }
        ++np;
      }
    }
    if constexpr (TILEDONE) consumer_sync();
    const int n = c * 1024 + tid * 4;
    float4 yv[8];
    float sv[8][2];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      if (q < np) {
        yv[q] = ld_bf4(P.yh + (size_t)plist[q] * 4096 + n);
        sv[q][0] = __ldcg(P.ssq + (size_t)plist[q] * 64 + lane);
        sv[q][1] = __ldcg(P.ssq + (size_t)plist[q] * 64 + 32 + lane);
      }
    }
    const float4 w = *reinterpret_cast<const float4*>(P.nw1 + n);
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      if (q < np) {
        float ss = sv[q][0] + sv[q][1];
#pragma unroll
        for (int s2 = 16; s2 > 0; s2 >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, s2);
        const float r = rsqrtf(ss * (1.f / 4096.f) + 1e-8f) * wl[q];
        a.x += r * (yv[q].x * w.x); a.y += r * (yv[q].y * w.y); a.z += r * (yv[q].z * w.z); a.w += r * (yv[q].w * w.w);
      }
    }
    const u32 lo = (u32)f2bf(a.x) | ((u32)f2bf(a.y) << 16);
    const u32 hi = (u32)f2bf(a.z) | ((u32)f2bf(a.w) << 16);
    *reinterpret_cast<uint2*>(P.out + (size_t)t * 4096 + n) = make_uint2(lo, hi);
  }
  }
  }
#ifdef MOE_NO_COMBINE
done_combine:
#endif
  if (tid == 0) MOE_TR(0, 9, 0);
#ifdef MOE_TIMING
  if (tid == 0) tdbg[3] = gtimer();
#endif
  // ------------------------------ reset for the next call ------------------------------
  // One acquire-release arrival per CTA: its uses of the counters happen before it, and the last
  // CTA's resets (ordered after its arrival by the barrier) happen after everyone's; kernel
  // completion publishes them to the next call. The last CTA's 256 consumer threads zero the
  // counters together: one thread's serial stores sat on the kernel's critical path.
  if (tid == 0) {
    int prev;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;" : "=r"(prev) : "l"(P.ctr + 2) : "memory");
    scal[9] = prev == (int)gridDim.x - 1;
  }
  consumer_sync();
  if (scal[9]) {
    for (int i = tid; i < ntiles * 16; i += 256) cnt1[i] = 0;
    for (int i = tid; i < ntiles; i += 256) cnt2[i] = 0;
    if (tid < 6) P.ctr[tid] = 0;  // queue head, down items done, CTAs done, gathered, combine claims
  }
}

// ------------------------------ one token (M = 1) ------------------------------
// Every local expert of the token is split into 128 units of 32 intermediate values. A unit runs
// its 64 gate/up rows (4 stages, the token's row resident in shared memory), then those 32 values'
// share of the down projection (2 stages from a slice-major copy of w2), and adds its partial down
// output into yacc. No CTA waits for another one until the last one combines: under a streaming
// load a cross-CTA signal takes microseconds to be seen.
struct Params1 {
  const u16* x; const int* ids; const float* topk_w;
  const float* gs13; const float* gs2; const float* nw1;
  const unsigned char* w2f; const unsigned char* s2f;
  u16* out; int* ctr; float* yacc; const unsigned char* tmap;
  int local_start;
  int* trc;  // MOE_TRACE1: per CTA 16 globaltimer stamps
  const unsigned char* w13; const unsigned char* s13;
};
#ifdef MOE_TRACE1
#define TR1(k) do { if (tid == 0) P.trc[blockIdx.x * 16 + (k)] = (int)gtimer(); } while (0)
#else
#define TR1(k)
#endif
DEVI void red_v4(float* p, float4 v) {
#ifdef MOE_NO_RED  // timing only: the down partials are dropped
  if (v.x == 1.2345f) *p = v.y;
#else
  asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
               :: "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
#endif
}
#ifndef MOE_YR
#define MOE_YR 1
#endif
template <int NS>
__device__ __forceinline__ void moe1_body(const Params1& P) {
  constexpr int YR = MOE_YR;  // yacc replicas: fewer adds queue on any one line
  constexpr int STAGE = 36864, WB = 32768;
  extern __shared__ __align__(128) unsigned char smem[];
  u64* bars = reinterpret_cast<u64*>(smem);            // full[8], empty[8], x
  float* red = reinterpret_cast<float*>(smem + 256);   // [4 k-warps][64 rows] gate/up partials
  u32* inter = reinterpret_cast<u32*>(smem + 1280);    // [16] the unit's 32 intermediate values (bf16 pairs)
  int* flag = reinterpret_cast<int*>(smem + 1344);
  float* rsum = reinterpret_cast<float*>(smem + 1536); // [2][8][4] combine sums of squares
  float* cw = reinterpret_cast<float*>(smem + 1792);   // [8] per local expert: top-k weight, [8] down scale
  float* ystg = reinterpret_cast<float*>(smem + 2048); // [8 warps][256] down rows of a stage
  unsigned char* xres = smem + 10240;                  // the token's row
  unsigned char* stages = smem + 18432;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  // the token's local experts: every warp derives them from the ids (no barrier on a global load)
  const int4 v0 = reinterpret_cast<const int4*>(P.ids)[0], v1 = reinterpret_cast<const int4*>(P.ids)[1];
  if (tid == 0) {
    for (int s = 0; s < NS; ++s) { mbar_init(smem_addr(&bars[s]), 1); mbar_init(smem_addr(&bars[8 + s]), 8); }
    mbar_init(smem_addr(&bars[16]), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  u64 elist = 0;   // 5 bits per local expert, in slot order
  u32 slist = 0;   // 3 bits per local expert: its top-k slot
  int ne = 0;
  {
    const int ev[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int e = ev[j] - P.local_start;
      if ((unsigned)e < 32u) { elist |= (u64)e << (5 * ne); slist |= (u32)j << (3 * ne); ++ne; }
    }
  }
  const int nu = ne * 128;
  TR1(0);
  if (warp == 8) {
    // ------------------------------ producer ------------------------------
    const u64 pol_w = policy_evict_first();
    const u32 stage0 = smem_addr(stages), bars0 = smem_addr(bars);
    const u64 tmap0 = reinterpret_cast<u64>(P.tmap) + (lane & 1) * 128;
    if (lane == 0) {
      mbar_arrive_tx(smem_addr(&bars[16]), 8192);
      bulk_g2s(smem_addr(xres), P.x, 8192, smem_addr(&bars[16]), policy_evict_last());
    }
    int it = 0;
    for (int u = blockIdx.x; u < nu; u += gridDim.x) {
      const int q = u >> 7, j = u & 127;
      const int e = (int)((elist >> (5 * q)) & 31);
      const unsigned char* w2u = P.w2f + (size_t)(e * 128 + j) * 65536;
      const unsigned char* s2u = P.s2f + (size_t)(e * 128 + j) * 8192;
      for (int st = 0; st < 6; ++st, ++it) {
        const int s = it % NS;
        const u32 ph = (it / NS) & 1;
        const u32 full = bars0 + s * 8;
        if (lane == 0) {
          mbar_sleep(bars0 + (8 + s) * 8, ph ^ 1);
          mbar_arrive_tx(full, STAGE);
        }
        __syncwarp();
        const u32 dst = stage0 + s * STAGE;
        if (st < 4) {
#ifndef MOE_FUSED_TMAP
          // 4 k chunks x (weights, scales): plain bulk copies (tensor-map copies measure slower)
          if (lane < 8) {
            const int kc = st * 4 + (lane >> 1);
            const size_t blk = (size_t)kc * 512 + j * 4;
            if (lane & 1) bulk_g2s(dst + WB + (lane >> 1) * 1024, P.s13 + (size_t)e * (8192 * 256) + blk * 256, 1024, full, pol_w);
            else bulk_g2s(dst + (lane >> 1) * 8192, P.w13 + (size_t)e * (8192 * 2048) + blk * 2048, 8192, full, pol_w);
          }
#else
          if (lane < 2) tma_4d(dst + (lane ? WB : 0), tmap0, 0, j * 4, st * 4, e, full, pol_w);
#endif
        } else if (lane == 0) {
          bulk_g2s(dst, w2u + (st - 4) * 32768, 32768, full, pol_w);
        } else if (lane == 1) {
          bulk_g2s(dst + WB, s2u + (st - 4) * 4096, 4096, full, pol_w);
        }
      }
    }
    return;
  }
  if (warp > 8) return;
  // ------------------------------ consumers ------------------------------
  const int kw = warp & 3, rw = warp >> 2;
  const int g = lane >> 2, tig = lane & 3;

  mbar_wait(smem_addr(&bars[16]), 0);
  TR1(1);
  int it = 0;
  int un = 0;
  for (int u = blockIdx.x; u < nu; u += gridDim.x, ++un) {
    const int q = u >> 7;
    const int e = (int)((elist >> (5 * q)) & 31);
    const float gs1 = P.gs13[e] * 256.f;
    float acc[2][1][4], acc2[2][1][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
      for (int c = 0; c < 4; ++c) { acc[a][0][c] = 0.f; acc2[a][0][c] = 0.f; }
    // gate/up: 64 rows (4 row groups; this warp 2 of them and 256 k of each 1024-k stage)
#pragma unroll 1
    for (int st = 0; st < 4; ++st, ++it) {
      const int s = it % NS;
      mbar_wait(smem_addr(&bars[s]), (it / NS) & 1);
      const unsigned char* sbase = stages + s * STAGE;
      stage_compute<1, 1>(acc, acc2, sbase + (kw * 4 + rw * 2) * 2048 + lane * 16, sbase + WB + (kw * 4 + rw * 2) * 256 + lane * 8,
                          xres + (st * 4 + kw) * 512 + tig * 32, 0);
      __syncwarp();
      if (lane == 0) mbar_arrive(smem_addr(&bars[8 + s]));
    }
    if (un < 4) TR1(2 + 3 * un);
    // reduce the k-warps' partials (the token is column 0: lanes tig == 0), then SwiGLU
    if (tig == 0) {
#pragma unroll
      for (int a = 0; a < 2; ++a) {
        red[kw * 64 + 16 * (2 * rw + a) + g] = acc[a][0][0];
        red[kw * 64 + 16 * (2 * rw + a) + g + 8] = acc[a][0][2];
      }
    }
    consumer_sync();
    if (tid < 16) {
      // values 2 tid, 2 tid + 1: group q' = tid / 4 (gate rows 16 q' + i, up rows 16 q' + 8 + i)
      u32 h2[2];
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const int v = 2 * tid + k, rg = 16 * (v >> 3) + (v & 7);
        float gt = 0.f, up = 0.f;
#pragma unroll
        for (int k2 = 0; k2 < 4; ++k2) { gt += red[k2 * 64 + rg]; up += red[k2 * 64 + rg + 8]; }
        gt *= gs1; up *= gs1;
        h2[k] = f2bf(__fdividef(gt, 1.f + __expf(-gt)) * up);
      }
      inter[tid] = h2[0] | (h2[1] << 16);
    }
    consumer_sync();
    if (un < 4) TR1(3 + 3 * un);
    // B fragments of the down GEMM's two k16 steps (every column the token: all equal)
    const u32 b00 = inter[tig], b01 = inter[tig + 4], b10 = inter[8 + tig], b11 = inter[12 + tig];
    // down: 2 stages of 2048 rows x 32 k; this warp 256 rows (8 row-group pairs) of each
#pragma unroll 1
    for (int h = 0; h < 2; ++h, ++it) {
      const int s = it % NS;
      mbar_wait(smem_addr(&bars[s]), (it / NS) & 1);
      const unsigned char* sbase = stages + s * STAGE;
      float* yw = ystg + warp * 256;
#pragma unroll
      for (int rp = 0; rp < 8; ++rp) {
        const uint4 qv = *reinterpret_cast<const uint4*>(sbase + (warp * 8 + rp) * 512 + lane * 16);
        const uint2 sv = *reinterpret_cast<const uint2*>(sbase + WB + (warp * 8 + rp) * 64 + g * 8);
#pragma unroll
        for (int r2 = 0; r2 < 2; ++r2) {
          const u32 sw = r2 ? sv.y : sv.x;
          const u32 w0 = r2 ? qv.z : qv.x, w1 = r2 ? qv.w : qv.y;
          float c[4] = {0.f, 0.f, 0.f, 0.f};
          u32 a0, a1, a2, a3;
          dequant(w0, scale_of<0, true>(sw), scale_of<1, true>(sw), a0, a1, a2, a3);
          mma16816(c, a0, a1, a2, a3, b00, b01);
          dequant(w1, scale_of<2, true>(sw), scale_of<3, true>(sw), a0, a1, a2, a3);
          mma16816(c, a0, a1, a2, a3, b10, b11);
          if (tig == 0) { yw[(rp * 2 + r2) * 16 + g] = c[0]; yw[(rp * 2 + r2) * 16 + g + 8] = c[2]; }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(smem_addr(&bars[8 + s]));
      float* ya = P.yacc + (q * YR + (blockIdx.x & (YR - 1))) * 4096 + h * 2048 + warp * 256;
      red_v4(ya + lane * 4, *reinterpret_cast<const float4*>(yw + lane * 4));
      red_v4(ya + 128 + lane * 4, *reinterpret_cast<const float4*>(yw + 128 + lane * 4));
      __syncwarp();
    }
    if (un < 4) TR1(4 + 3 * un);
  }
  // ------------------------------ combine (the last CTA) ------------------------------
  // what the combine reads besides yacc: loads in flight during the arrival's round trip
  float4 wv[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) wv[k] = *reinterpret_cast<const float4*>(P.nw1 + (tid + 256 * k) * 4);
  if (tid < ne) {
    cw[tid] = P.topk_w[(slist >> (3 * tid)) & 7];
    cw[8 + tid] = P.gs2[(int)((elist >> (5 * tid)) & 31)] * 256.f;
  }
  consumer_sync();  // the CTA's adds before thread 0's release (cumulative)
  if (tid == 0) {
    int prev;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;" : "=r"(prev) : "l"(P.ctr) : "memory");
    *flag = prev == (int)gridDim.x - 1;
  }
  TR1(14);
  consumer_sync();
  if (!*flag) return;
  TR1(10);
  // Experts in groups of 4: a group's rows are loaded together and one barrier reduces all their
  // sums of squares (rsum [2][8 warps][4], alternating groups).
  float accv[16];
#pragma unroll
  for (int k = 0; k < 16; ++k) accv[k] = 0.f;
  for (int q0 = 0, rnd = 0; q0 < ne; q0 += 4, ++rnd) {
    float v[4][16], ss[4];
#pragma unroll
    for (int gq = 0; gq < 4; ++gq) {
      const int q = q0 + gq;
      ss[gq] = 0.f;
      const bool live = q < ne;
      const float gsc = live ? cw[8 + q] : 0.f;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        // (no store to yacc until every load of the group is issued: a store behind a load of the
        // same address waits for it, and every later load behind the store)
        float4 y = make_float4(0.f, 0.f, 0.f, 0.f);
        if (live) {
#pragma unroll
          for (int r = 0; r < YR; ++r) {
            const float4 u = __ldcg(reinterpret_cast<const float4*>(P.yacc + (q * YR + r) * 4096 + (tid + 256 * k) * 4));
            y.x += u.x; y.y += u.y; y.z += u.z; y.w += u.w;
          }
        }
        v[gq][4 * k] = y.x * gsc * wv[k].x; v[gq][4 * k + 1] = y.y * gsc * wv[k].y;
        v[gq][4 * k + 2] = y.z * gsc * wv[k].z; v[gq][4 * k + 3] = y.w * gsc * wv[k].w;
#pragma unroll
        for (int i = 0; i < 4; ++i) ss[gq] += v[gq][4 * k + i] * v[gq][4 * k + i];
      }
    }
    if (rnd == 0) TR1(11);
    // zero this group's rows for the next call
#pragma unroll
    for (int gq = 0; gq < 4; ++gq)
      if (q0 + gq < ne)
#pragma unroll
        for (int r = 0; r < YR; ++r)
#pragma unroll
          for (int k = 0; k < 4; ++k)
            __stcg(reinterpret_cast<float4*>(P.yacc + ((q0 + gq) * YR + r) * 4096 + (tid + 256 * k) * 4), make_float4(0.f, 0.f, 0.f, 0.f));
#pragma unroll
    for (int s2 = 16; s2 > 0; s2 >>= 1)
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) ss[gq] += __shfl_xor_sync(0xffffffffu, ss[gq], s2);
    if (lane == 0)
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) rsum[(rnd & 1) * 32 + warp * 4 + gq] = ss[gq];
    consumer_sync();
    if (rnd == 0) TR1(12);
#pragma unroll
    for (int gq = 0; gq < 4; ++gq) {
      const int q = q0 + gq;
      if (q < ne) {
        float tot = 0.f;
#pragma unroll
        for (int w = 0; w < 8; ++w) tot += rsum[(rnd & 1) * 32 + w * 4 + gq];
        const float r = rsqrtf(tot * (1.f / 4096.f) + 1e-8f) * cw[q];
#pragma unroll
        for (int k = 0; k < 16; ++k) accv[k] += r * v[gq][k];
      }
    }
  }
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int n = (tid + 256 * k) * 4;
    const u32 lo = (u32)f2bf(accv[4 * k]) | ((u32)f2bf(accv[4 * k + 1]) << 16);
    const u32 hi = (u32)f2bf(accv[4 * k + 2]) | ((u32)f2bf(accv[4 * k + 3]) << 16);
    *reinterpret_cast<uint2*>(P.out + n) = make_uint2(lo, hi);
  }
  if (tid == 0) *P.ctr = 0;  // for the next call (ordered before it by the kernel boundary)
  TR1(15);
}
extern "C" __global__ void __launch_bounds__(288, 1) moe1_3(Params1 p) { moe1_body<3>(p); }
extern "C" __global__ void __launch_bounds__(288, 1) moe1_4(Params1 p) { moe1_body<4>(p); }
extern "C" __global__ void __launch_bounds__(288, 1) moe1_5(Params1 p) { moe1_body<5>(p); }

#define INST(RW, NT, KS2, NS, MW) \
  extern "C" __global__ void __launch_bounds__(256, 2) count_##RW##_##NT##_##KS2##_##NS##_##MW(Params p) { parallel_route_count<NT, MW>(p); } \
  extern "C" __global__ void __launch_bounds__(32, 1) finalize_##RW##_##NT##_##KS2##_##NS##_##MW(Params p) { parallel_route_finalize<RW, NT, KS2, MW>(p); } \
  extern "C" __global__ void __launch_bounds__(256, 2) pairs_##RW##_##NT##_##KS2##_##NS##_##MW(Params p) { parallel_route_pairs<NT, MW>(p); } \
  extern "C" __global__ void __launch_bounds__(256, 3) gather_##RW##_##NT##_##KS2##_##NS##_##MW(Params p) { external_gather<RW, NT, MW>(p); } \
  extern "C" __global__ void __launch_bounds__(384, 1) route_##RW##_##NT##_##KS2##_##NS##_##MW(Params p) { \
    global_route_body<RW, NT, KS2, MW>(p); \
  } \
  extern "C" __global__ void __launch_bounds__(384, 1) moe_##RW##_##NT##_##KS2##_##NS##_##MW(Params p) { \
    moe_body<RW, NT, KS2, NS, MW>(p); \
  }

"""


def _has_wg(rw, nt):
    return rw == 8 and nt >= 8


def _stage_bytes(rw, nt):
    kw = 8 // rw
    rgs = 2 * rw
    wb, sb = rgs * 2048 * kw, rgs * 256 * kw
    xb = 8 * nt * (256 * kw * 2 + 16)
    if _has_wg(rw, nt):  # wgmma tiles: 1 KB aligned stages, room for either activation layout
        return (max(wb + sb + max(xb, 8 * nt * 512), kw * min(8 * nt, 64) * (32 * rw + 4) * 4) + 1023) // 1024 * 1024
    return (wb + sb + xb + 127) // 128 * 128


def _smem_bytes(rw, nt, ns, mw):
    kw = 8 // rw
    t = 8 * nt
    maxp = mw * 32 * 8
    maxt = 32 + maxp // t
    route = (32 * mw * 8 + maxt * 16 + maxp * 3 + 127) // 128 * 128
    red = (kw - 1) * rw * 1024 * nt
    return 1024 + ns * _stage_bytes(rw, nt)


# (largest M, (RW, NT, KS2)) per batch-size bucket; stages are as many as shared memory holds (<= 4).
BUCKETS = [
    (1, (2, 1, 1)),
    (16, (4, 1, 1)),
    (32, (8, 4, 2)),
    (128, (8, 4, 1)),
    (MAX_M, (8, 16, 1)),
]


def _mw(M):
    return 1 << ((max(1, (M + 31) // 32) - 1).bit_length())


def _kernel_cfg(M):
    for bound, (rw, nt, ks2) in BUCKETS:
        if M <= bound:
            break
    mw = _mw(M)
    ns = NS_MAX
    while ns > 2 and _smem_bytes(rw, nt, ns, mw) > SMEM_MAX:
        ns -= 1
    return rw, nt, ks2, ns, mw


def _all_cfgs():
    out = set()
    for M in (1, 8, 16, 32, 33, 64, 65, 128, 129, 256, 257, 512, 513, 1024, 1025, 2048, 2049, 4096, 4097, 8192, 8193, MAX_M):
        out.add(_kernel_cfg(M))
    return sorted(out)


class _Cuda:
    """NVRTC-compiled module and driver-API launches on torch's current stream."""

    def __init__(self, src):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"moe.cu", 0, [], [])
        self._check(err, "nvrtcCreateProgram")
        major, minor = torch.cuda.get_device_capability()
        arch = f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode()
        opts = [arch, b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, size = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * size
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        self._check(err, "nvrtcGetCUBINSize")
        cubin = b" " * size
        (err,) = nvrtc.nvrtcGetCUBIN(prog, cubin)
        self._check(err, "nvrtcGetCUBIN")
        err, self.module = driver.cuModuleLoadData(cubin)
        self._check(err, "cuModuleLoadData")
        self.funcs = {}

    @staticmethod
    def _check(err, what):
        if int(err) != 0:
            raise RuntimeError(f"{what} failed: {err}")

    def func(self, name, smem=0):
        if name not in self.funcs:
            d = self.driver
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            self._check(err, f"cuModuleGetFunction({name})")
            if smem > 48 * 1024:
                (err,) = d.cuFuncSetAttribute(
                    f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
                self._check(err, "cuFuncSetAttribute")
            self.funcs[name] = f
        return self.funcs[name]

    def launch(self, name, grid, block, smem, params):
        f = self.func(name, smem)
        stream = self.driver.CUstream(torch.cuda.current_stream().cuda_stream)
        argv = (ctypes.c_void_p * 1)(ctypes.addressof(params))
        (err,) = self.driver.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        self._check(err, f"launch {name}")


class _ParamsC(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_void_p), ("ids", ctypes.c_void_p), ("topk_w", ctypes.c_void_p),
        ("M", ctypes.c_int), ("local_start", ctypes.c_int),
        ("w13", ctypes.c_void_p), ("s13", ctypes.c_void_p), ("gs13", ctypes.c_void_p),
        ("w2", ctypes.c_void_p), ("s2", ctypes.c_void_p), ("gs2", ctypes.c_void_p), ("nw1", ctypes.c_void_p),
        ("inter", ctypes.c_void_p), ("ypart", ctypes.c_void_p), ("out", ctypes.c_void_p),
        ("ctr", ctypes.c_void_p), ("pmax", ctypes.c_int), ("xs", ctypes.c_void_p),
        ("ssq", ctypes.c_void_p), ("tmap", ctypes.c_void_p), ("yh", ctypes.c_void_p),
        ("routes", ctypes.c_void_p), ("route_sms", ctypes.c_int),
    ]


class _Params1C(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_void_p), ("ids", ctypes.c_void_p), ("topk_w", ctypes.c_void_p),
        ("gs13", ctypes.c_void_p), ("gs2", ctypes.c_void_p), ("nw1", ctypes.c_void_p),
        ("w2f", ctypes.c_void_p), ("s2f", ctypes.c_void_p),
        ("out", ctypes.c_void_p), ("ctr", ctypes.c_void_p), ("yacc", ctypes.c_void_p), ("tmap", ctypes.c_void_p),
        ("local_start", ctypes.c_int), ("trc", ctypes.c_void_p), ("w13", ctypes.c_void_p), ("s13", ctypes.c_void_p),
    ]


NS1 = 5
SMEM1 = 18432 + NS1 * 36864


_POS = [  # per A register (o) and half (h): bit positions of (sign, m2, m1, m0) in the packed word
    [(15, 8, 7, 6), (31, 24, 23, 22)],     # q & M
    [(11, 4, 3, 2), (27, 20, 19, 18)],     # rotl(q, 4) & M
    [(5, 30, 29, 28), (21, 14, 13, 12)],   # rotl(q, 10) & M
    [(9, 26, 1, 0), (25, 10, 17, 16)],     # rotl(q, 6) & 0x80C080C0 | rotl(q, 14) & 0x01000100
]


def _gate_up_rows(device):
    """Row order of the repacked w13: group j = gate rows 8j..8j+7, then up rows 8j..8j+7."""
    j = torch.arange(512, device=device).view(512, 1)
    r = torch.arange(16, device=device).view(1, 16)
    return torch.where(r < 8, 8 * j + r, 4096 + 8 * j + (r - 8)).flatten()


def _repack(packed, sf, rows=None):
    """packed uint8 [R, 2048], sf fp8 [R, 256] -> (words int32 [R*512], scales uint8 [R*256])."""
    R = packed.shape[0]
    codes = torch.stack((packed & 15, packed >> 4), -1).reshape(R, 4096)
    sb = sf.view(torch.uint8)
    # A zero block scale makes its 16 values zero whatever their codes: store zero codes.
    codes = codes.masked_fill((sb == 0).repeat_interleave(16, dim=1), 0)
    if rows is not None:
        codes, sb = codes[rows], sb[rows]
    # rows: rg, rowhalf, g ; k: kc, kk, tig, jj, pairsel, h
    c = codes.view(R // 16, 2, 8, 16, 4, 4, 4, 2, 2).permute(3, 0, 4, 2, 5, 6, 7, 1, 8)
    c = c.reshape(-1, 8).to(torch.int64)
    q = torch.zeros(c.shape[0], dtype=torch.int64, device=c.device)
    for v in range(8):
        o, h = v // 2, v % 2  # v = (pairsel*2 + rowhalf)*2 + h, o = pairsel*2 + rowhalf
        sgn, m2, m1, m0 = _POS[o][h]
        cv = c[:, v]
        q |= ((cv >> 3) & 1) << sgn
        q |= ((cv >> 2) & 1) << m2
        q |= ((cv >> 1) & 1) << m1
        q |= (cv & 1) << m0
    words = q.view(torch.int32).view(-1, 2)[:, 0].contiguous()
    # scales: rg, rowhalf, g ; block: kc, kk, tig -> kc, rg, g, tig, kk, rowhalf
    s = sb.reshape(R // 16, 2, 8, 16, 4, 4).permute(3, 0, 2, 5, 4, 1).contiguous().flatten()
    return words, s


def _codes_to_words(c):
    """c int64 [..., 8] codes in (pairsel, rowhalf, h) order -> int32 words (bit positions _POS)."""
    c = c.reshape(-1, 8)
    q = torch.zeros(c.shape[0], dtype=torch.int64, device=c.device)
    for v in range(8):
        o, h = v // 2, v % 2
        sgn, m2, m1, m0 = _POS[o][h]
        cv = c[:, v]
        q |= ((cv >> 3) & 1) << sgn
        q |= ((cv >> 2) & 1) << m2
        q |= ((cv >> 1) & 1) << m1
        q |= (cv & 1) << m0
    return q.view(torch.int32).view(-1, 2)[:, 0].contiguous()


def _repack_w2_slices(packed, sf):
    """w2 of one expert for the one-token kernel: per 32-column slice j of the intermediate,
    [row-group pair rp 128][lane 32][rg % 2][k16 step 2] words in the mma's natural k order
    (k = 32 j + 16 s + 8 pairsel + 2 tig + h), and scales [rp][g][rg % 2][s][rowhalf]."""
    R = packed.shape[0]
    codes = torch.stack((packed & 15, packed >> 4), -1).reshape(R, 4096)
    sb = sf.view(torch.uint8)
    codes = codes.masked_fill((sb == 0).repeat_interleave(16, dim=1), 0)
    # rows: rp, r2, rowhalf, g ; k: j, s, pairsel, tig, h
    c = codes.view(128, 2, 2, 8, 128, 2, 2, 4, 2).permute(4, 0, 3, 7, 1, 5, 6, 2, 8).to(torch.int64)
    words = _codes_to_words(c)
    sc = sb.reshape(128, 2, 2, 8, 128, 2).permute(4, 0, 3, 1, 5, 2).contiguous().flatten()
    return words, sc


_MOD = None
_FAST = False


def _module(fast_scales=False):
    global _MOD
    if _MOD is None:
        src = ("#define MOE_FAST_SCALES\n" if fast_scales else "") + CUDA_SRC
        src += "".join(f"INST({rw}, {nt}, {ks2}, {ns}, {mw})\n" for rw, nt, ks2, ns, mw in _all_cfgs())
        _MOD = _Cuda(src)
    return _MOD


def _fast_scales_ok(*scales):
    """True when every scale byte is 0 or a positive normal E4M3 below NaN (0x08..0x7E)."""
    for sc in scales:
        b = sc.view(torch.uint8)
        if not bool((((b >= 0x08) & (b <= 0x7E)) | (b == 0)).all()):
            return False
    return True


def _tensor_maps(w13, s13, w2, s2):
    """Per row-warp count RW: tiled tensor maps (w13, s13, w2, s2) whose box is one stage's weights
    (RGS = 2 RW row groups x KW = 8 / RW k chunks of 2 KB) or scales (256 B blocks), [4][128] bytes."""
    from cuda.bindings import driver as d
    import struct
    out = {}
    for rw in (2, 4, 8):
        rgs, kw = 2 * rw, 8 // rw
        maps = []
        for t, nrg, blk in ((w13, 512, 2048), (s13, 512, 256), (w2, 256, 2048), (s2, 256, 256)):
            err, tm = d.cuTensorMapEncodeTiled(
                d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT64, d.cuuint32_t(4), t.data_ptr(),
                [d.cuuint64_t(v) for v in (blk // 8, nrg, 16, t.shape[0])],
                [d.cuuint64_t(v) for v in (blk, nrg * blk, 16 * nrg * blk)],
                [d.cuuint32_t(v) for v in (blk // 8, rgs, kw, 1)], [d.cuuint32_t(1)] * 4,
                d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
                d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
            if int(err) != 0:
                raise RuntimeError(f"cuTensorMapEncodeTiled failed: {err}")
            maps.append(b"".join(struct.pack("<Q", int(v)) for v in tm.opaque))
        out[rw] = b"".join(maps)
    order = sorted(out)
    buf = torch.frombuffer(bytearray(b"".join(out[rw] for rw in order)), dtype=torch.uint8).to(w13.device)
    return buf, {rw: i * 512 for i, rw in enumerate(order)}


def _name(cfg):
    return "moe_" + "_".join(str(v) for v in cfg)


def _native_expert_prepare(weights):
    dev = weights["w13"].device
    start, count = 0, NLOCAL
    rows13 = _gate_up_rows(dev)
    w13p = torch.empty((count, 8192 * 512), dtype=torch.int32, device=dev)
    s13p = torch.empty((count, 8192 * 256), dtype=torch.uint8, device=dev)
    w2p = torch.empty((count, 4096 * 512), dtype=torch.int32, device=dev)
    s2p = torch.empty((count, 4096 * 256), dtype=torch.uint8, device=dev)
    for e in range(count):
        w13p[e], s13p[e] = _repack(weights["w13"][start + e], weights["w13_sf"][start + e], rows13)
        w2p[e], s2p[e] = _repack(weights["w2"][start + e], weights["w2_sf"][start + e])
    tmaps, tmap_off = _tensor_maps(w13p, s13p, w2p, s2p)
    w2f = torch.empty((count, 4096 * 512), dtype=torch.int32, device=dev)
    s2f = torch.empty((count, 4096 * 256), dtype=torch.uint8, device=dev)
    for e in range(count):
        w2f[e], s2f[e] = _repack_w2_slices(weights["w2"][start + e], weights["w2_sf"][start + e])
    fast = _fast_scales_ok(s13p, s2p)
    global _FAST
    _FAST = fast
    mod = _module(fast)
    for cfg in _all_cfgs():
        mod.func(_name(cfg), _smem_bytes(*cfg[:2], cfg[3], cfg[4]))
        mod.func(_name(cfg).replace("moe_", "route_", 1))
        if cfg[4] >= 128:
            for prefix in ("count_", "finalize_", "pairs_", "gather_"):
                mod.func(_name(cfg).replace("moe_", prefix, 1))
    mod.func(f"moe1_{NS1}", 18432 + NS1 * 36864)
    torch.cuda.synchronize()
    return {
        "start": start, "count": count, "sms": torch.cuda.get_device_properties(dev).multi_processor_count,
        "w13": w13p, "s13": s13p, "gs13": weights["w13_gs"][start:start + count].float().contiguous(),
        "w2": w2p, "s2": s2p, "gs2": weights["w2_gs"][start:start + count].float().contiguous(),
        "nw1": (1.0 + weights["post_norm_w"].float()).contiguous(),
        "ctr": torch.zeros(CTR_WORDS, dtype=torch.int32, device=dev),
        "tmaps": tmaps, "tmap_off": tmap_off,
        "w2f": w2f, "s2f": s2f, "yacc": torch.zeros(8 * 8 * 4096, dtype=torch.float32, device=dev),
        "ctr1": torch.zeros(4, dtype=torch.int32, device=dev),
        "trc": torch.zeros(160 * 16, dtype=torch.int32, device=dev),
        "raw": weights,
    }


# Follows the OKC problem's reference implementation (muse_moe_w4a16/tools/reference.py, `ref_kernel`).
def _torch_path(data, state):
    """Batches past MAX_M: the computation in plain PyTorch, one expert at a time."""
    from nvfp4 import dequantize
    w = state["raw"]
    x = data["hidden_states"]
    ids = data["topk_ids"].long() - state["start"]
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    nw1 = state["nw1"]
    for e in range(state["count"]):
        tok, slot = (ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        ge = state["start"] + e
        h = x[tok].float() @ dequantize(w["w13"][ge], w["w13_sf"][ge], w["w13_gs"][ge]).T
        inter = (torch.nn.functional.silu(h[:, :I]) * h[:, I:]).to(torch.bfloat16).float()
        y = (inter @ dequantize(w["w2"][ge], w["w2_sf"][ge], w["w2_gs"][ge]).T) * nw1
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-8)
        out.index_add_(0, tok, y * data["topk_weights"][tok, slot].unsqueeze(1))
    return out.to(torch.bfloat16)



@triton.jit
def _native_finish(IDS, TW, NW, ROUTES, YH, YPART, OUT,
                   MW:tl.constexpr, PMAX:tl.constexpr, KS2:tl.constexpr,
                   KSX:tl.constexpr, LOCAL_START:tl.constexpr):
    token=tl.program_id(0)
    dims=tl.arange(0,4096)
    norm_weight=tl.load(NW+dims)
    psplit=tl.load(ROUTES.to(tl.pointer_type(tl.int32))+5)
    before=(tl.full((),1,tl.uint32) << (token%32))-1
    acc=tl.full((4096,),0,tl.float32)
    for slot in range(8):
        expert=tl.load(IDS+token*8+slot)-LOCAL_START
        if (expert>=0) & (expert<32):
            bitmap=tl.load((ROUTES+512).to(tl.pointer_type(tl.uint32))+expert*MW+token//32)
            bits=bitmap & before
            rank=tl.inline_asm_elementwise("popc.b32 $0, $1;",constraints="=r,r",args=[bits],
                                          dtype=tl.int32,is_pure=True,pack=1)
            prefix=tl.load((ROUTES+512+32*MW*4).to(tl.pointer_type(tl.int32))+expert*MW+token//32)
            offset=tl.load((ROUTES+192).to(tl.pointer_type(tl.int32))+expert)
            pair=offset+prefix+rank
            parts=tl.where(pair>=psplit,KSX,KS2)
            if parts==1:
                y=tl.load(YH+pair*4096+dims).to(tl.float32)
            else:
                y=tl.full((4096,),0,tl.float32)
                for part in range(max(KS2,KSX)):
                    if part<parts:y+=tl.load(YPART+(part*PMAX+pair)*4096+dims)
            value=y*norm_weight
            inv=tl.rsqrt(tl.sum(value*value,0)*(1.0/4096)+1.0e-8)
            weight=tl.load(TW+token*8+slot)
            acc+=value*(inv*weight)
    tl.store(OUT+token*4096+dims,acc)


def _launch_routes(mod, cfg, p):
    rw, nt, ks2, ns, mw = cfg
    name = _name(cfg)
    if mw >= 128:
        mod.launch(name.replace("moe_", "count_", 1), 32, 256, 0, p)
        mod.launch(name.replace("moe_", "finalize_", 1), 1, 32, 0, p)
        mod.launch(name.replace("moe_", "pairs_", 1), (p.M * 8 + 255) // 256, 256, 0, p)
    else:
        mod.launch(name.replace("moe_", "route_", 1), 1, 384, 0, p)


def _native_expert_kernel(data, state):
    cfg = data["config"]
    x = data["hidden_states"].contiguous()
    M = x.shape[0]
    if M > MAX_M or cfg["local_start"] != state["start"] or cfg["local_count"] != state["count"]:
        return _torch_path(data, state)
    ids = data["topk_ids"].to(torch.int32).contiguous()
    tw = data["topk_weights"].to(torch.float32).contiguous()
    if M == 1 and ONE_TOKEN:
        out = torch.empty((1, H), dtype=torch.bfloat16, device=x.device)
        p1 = _Params1C(x.data_ptr(), ids.data_ptr(), tw.data_ptr(), state["gs13"].data_ptr(), state["gs2"].data_ptr(),
                       state["nw1"].data_ptr(), state["w2f"].data_ptr(), state["s2f"].data_ptr(), out.data_ptr(),
                       state["ctr1"].data_ptr(), state["yacc"].data_ptr(), state["tmaps"].data_ptr() + state["tmap_off"][2],
                       state["start"], state["trc"].data_ptr(), state["w13"].data_ptr(), state["s13"].data_ptr())
        _module().launch(f"moe1_{NS1}", state["sms"], 288, 18432 + NS1 * 36864, p1)
        return out
    kc = _kernel_cfg(M)
    rw, nt, ks2, ns, mw = kc
    pmax = M * TOPK
    dev = x.device
    T = 8 * nt
    xp = 256 * (8 // rw) * 2 + 16
    tiles_max = NLOCAL + (pmax + T - 1) // T
    inter = torch.empty(tiles_max * (16 // (8 // rw)) * T * xp // 2, dtype=torch.bfloat16, device=dev)
    xs = torch.empty_like(inter) if nt >= 4 else inter
    ypart = torch.empty((max(ks2, 4 if rw == 8 else ks2), pmax, H), dtype=torch.float32, device=dev)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    ssq = torch.empty(pmax * 64, dtype=torch.float32, device=dev)
    yh = torch.empty((pmax, H), dtype=torch.bfloat16, device=dev)
    max_pairs = mw * 32 * TOPK
    max_tiles = NLOCAL + max_pairs // T
    route_bytes = 512 + 32 * mw * 8 + max_tiles * 16 + max_pairs * 3
    routes = torch.empty(route_bytes, dtype=torch.uint8, device=dev)
    p = _ParamsC(x.data_ptr(), ids.data_ptr(), tw.data_ptr(), M, state["start"],
                 state["w13"].data_ptr(), state["s13"].data_ptr(), state["gs13"].data_ptr(),
                 state["w2"].data_ptr(), state["s2"].data_ptr(), state["gs2"].data_ptr(), state["nw1"].data_ptr(),
                 inter.data_ptr(), ypart.data_ptr(), out.data_ptr(), state["ctr"].data_ptr(), pmax, xs.data_ptr(),
                 ssq.data_ptr(), state["tmaps"].data_ptr() + state["tmap_off"][rw], yh.data_ptr(),
                 routes.data_ptr(), state["sms"])
    _launch_routes(_module(), kc, p)
    if nt >= 4 and mw >= 128:
        _module().launch(_name(kc).replace("moe_", "gather_", 1), state["sms"] * 4, 256, 0, p)
    _module().launch(_name(kc), state["sms"], 384, _smem_bytes(rw, nt, ns, mw), p)
    _native_finish[(M,)](ids, tw, state["nw1"], routes, yh, ypart, out,
                          mw, pmax, ks2, 4 if rw==8 else ks2, state["start"], num_warps=8)
    return out

"""Native packed experts with per-call global routing and up to 16384 tokens per chunk."""


def _native_prepare_without_lt(weights):
    native_weights = {name: weights[name].clone()
                      for name in ('w13', 'w13_sf', 'w13_gs', 'w2', 'w2_sf', 'w2_gs')}
    native_weights['post_norm_w'] = weights['post_expert_norm'].clone()
    native = _native_expert_prepare(native_weights)
    state = _sglang_prepare(weights)
    state['native_experts'] = native
    return state


def _experts(x, topk_weights, topk_ids, state):
    outputs = []
    for first in range(0, x.shape[0], MAX_M):
        end = min(first + MAX_M, x.shape[0])
        data = {'hidden_states': x[first:end], 'topk_ids': topk_ids[first:end],
                'topk_weights': topk_weights[first:end],
                'config': {'local_start': 0, 'local_count': 32}}
        outputs.append(_native_expert_kernel(data, state['native_experts']))
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)

"""cuBLASLt BF16 GEMM plans with FP32 computation and explicit algorithm selection.

Plans contain shape metadata only. Algorithm bytes are obtained from the loaded
library; they are never transferred between cuBLAS versions.
"""
import ctypes as ct
import torch


class LtAlgo(ct.Structure):
    _fields_ = [('data', ct.c_uint64 * 8)]


class LtHeuristic(ct.Structure):
    _fields_ = [('algo', LtAlgo), ('workspaceSize', ct.c_size_t),
                ('state', ct.c_int), ('wavesCount', ct.c_float),
                ('reserved', ct.c_int * 4)]


class LtLibrary:
    def __init__(self):
        self.lib = None
        for name in ('libcublasLt.so.13', 'libcublasLt.so.12'):
            try:
                self.lib = ct.CDLL(name)
                break
            except OSError:
                continue
        if self.lib is None:
            raise RuntimeError('No loaded CUDA 12/13 cuBLASLt library found')
        v, i, s = ct.c_void_p, ct.c_int, ct.c_size_t
        signatures = {
            'cublasLtCreate': [v],
            'cublasLtGetVersion': [],
            'cublasLtMatmulDescCreate': [v, i, i],
            'cublasLtMatmulDescSetAttribute': [v, i, v, s],
            'cublasLtMatrixLayoutCreate': [v, i, ct.c_uint64, ct.c_uint64, ct.c_int64],
            'cublasLtMatmulPreferenceCreate': [v],
            'cublasLtMatmulPreferenceSetAttribute': [v, i, v, s],
            'cublasLtMatmulAlgoGetHeuristic': [v, v, v, v, v, v, v, i, v, v],
            'cublasLtMatmulAlgoConfigGetAttribute': [v, i, v, s, v],
            'cublasLtMatmul': [v] * 14 + [s, v],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes = args
            fn.restype = s if name == 'cublasLtGetVersion' else i
        self.handle = v()
        self.check(self.lib.cublasLtCreate(ct.byref(self.handle)), 'create')
        self.version = self.lib.cublasLtGetVersion()

    @staticmethod
    def check(status, operation):
        if status != 0:
            raise RuntimeError(f'cuBLASLt {operation}: status {status}')


class LtMatmul:
    def __init__(self, library, m, n, k, workspace, *, weight_transposed=False, count=32):
        self.library, self.workspace = library, workspace
        self.m, self.n, self.k = m, n, k
        self.weight_transposed = weight_transposed
        lib, check = library.lib, library.check
        self.desc, self.a, self.b, self.c, self.pref = [ct.c_void_p() for _ in range(5)]
        # CUBLAS_COMPUTE_32F=68; CUDA_R_32F=0; CUDA_R_16BF=14.
        check(lib.cublasLtMatmulDescCreate(ct.byref(self.desc), 68, 0), 'descriptor')
        trans_a = ct.c_int(0 if weight_transposed else 1)
        check(lib.cublasLtMatmulDescSetAttribute(self.desc, 3, ct.byref(trans_a), ct.sizeof(trans_a)), 'transpose A')
        ar, ac = (n, k) if weight_transposed else (k, n)
        check(lib.cublasLtMatrixLayoutCreate(ct.byref(self.a), 14, ar, ac, ar), 'A layout')
        check(lib.cublasLtMatrixLayoutCreate(ct.byref(self.b), 14, k, m, k), 'B layout')
        check(lib.cublasLtMatrixLayoutCreate(ct.byref(self.c), 14, n, m, n), 'C layout')
        check(lib.cublasLtMatmulPreferenceCreate(ct.byref(self.pref)), 'preference')
        size = ct.c_size_t(workspace.numel())
        check(lib.cublasLtMatmulPreferenceSetAttribute(self.pref, 1, ct.byref(size), ct.sizeof(size)), 'workspace preference')
        reduction_mask = ct.c_uint32(2)  # Split reductions must keep FP32 intermediates.
        check(lib.cublasLtMatmulPreferenceSetAttribute(self.pref, 3, ct.byref(reduction_mask),
                                                       ct.sizeof(reduction_mask)), 'reduction precision')
        results = (LtHeuristic * count)()
        returned = ct.c_int()
        check(lib.cublasLtMatmulAlgoGetHeuristic(library.handle, self.desc, self.a, self.b,
                                                self.c, self.c, self.pref, count,
                                                results, ct.byref(returned)), 'heuristics')
        self.algorithms = []
        self.info = []
        for j in range(returned.value):
            result = results[j]
            if result.state != 0 or result.workspaceSize > workspace.numel():
                continue
            algo = LtAlgo.from_buffer_copy(result.algo)
            attrs = {}
            for key, attr in [('id', 0), ('tile', 1), ('split_k', 2), ('reduction', 3),
                               ('swizzle', 4), ('custom', 5), ('stages', 6)]:
                value, written = ct.c_int(), ct.c_size_t()
                status = lib.cublasLtMatmulAlgoConfigGetAttribute(ct.byref(algo), attr, ct.byref(value),
                                                                 ct.sizeof(value), ct.byref(written))
                attrs[key] = value.value if status == 0 else None
            if attrs['reduction'] not in (0, 2):
                continue
            self.algorithms.append(algo)
            self.info.append(dict(heuristic_index=j, workspace_bytes=result.workspaceSize,
                                  waves=result.wavesCount, **attrs))
        if not self.algorithms:
            raise RuntimeError('No supported cuBLASLt algorithms')
        self.alpha, self.beta = ct.c_float(1.0), ct.c_float(0.0)

    def run(self, x, weight, index=0):
        if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise TypeError('BF16 operands required')
        if not x.is_contiguous() or not weight.is_contiguous():
            raise ValueError('Contiguous operands required')
        out = torch.empty((self.m, self.n), device=x.device, dtype=torch.bfloat16)
        status = self.library.lib.cublasLtMatmul(
            self.library.handle, self.desc, ct.byref(self.alpha),
            weight.data_ptr(), self.a, x.data_ptr(), self.b, ct.byref(self.beta),
            out.data_ptr(), self.c, out.data_ptr(), self.c, ct.byref(self.algorithms[index]),
            self.workspace.data_ptr(), self.workspace.numel(), torch.cuda.current_stream(x.device).cuda_stream)
        self.library.check(status, 'matmul')
        return out

"""Measured large-prefill cuBLASLt choices, using algorithms from the loaded library."""

_PREFILL_LT_PLANS = {}


def prepare(weights):
    state = _native_prepare_without_lt(weights)
    library = LtLibrary()
    workspace = torch.empty(64 * 1024 * 1024, device=weights['pooler'].device, dtype=torch.uint8)
    plans = {}
    for name, tile, custom in (('pooler', 23, 4), ('unpooler', 201, 2)):
        weight = state['dense'][name]
        n, k = weight.shape
        plan = LtMatmul(library, 16384, n, k, workspace, count=32)
        matches = [i for i, info in enumerate(plan.info)
                   if info['id'] == 66 and info['tile'] == tile and info['custom'] == custom
                   and info['split_k'] == 1 and info['reduction'] == 0
                   and info['swizzle'] == 0 and info['stages'] == 35]
        if matches:
            plans[(16384, n, k, weight.data_ptr())] = (plan, matches[0])
    # Keys contain shape and immutable weight identity only. No input-dependent
    # values or computed outputs are retained between invocations.
    _PREFILL_LT_PLANS.update(plans)
    state['prefill_lt_plans'] = plans
    state['prefill_lt_library'] = library
    state['prefill_lt_workspace'] = workspace
    return state


def _layer_dense(x, w):
    selected = _PREFILL_LT_PLANS.get((x.shape[0], w.shape[0], x.shape[1], w.data_ptr()))
    if selected is not None and x.is_contiguous() and w.is_contiguous():
        plan, index = selected
        return plan.run(x, w, index)
    return _native_layer_dense_without_lt(x, w)
