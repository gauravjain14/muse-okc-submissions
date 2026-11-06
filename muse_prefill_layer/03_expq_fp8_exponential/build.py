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
    use_att = not cfg["window"] or cfg["tokens"] >= ATT_SLIDING_MIN
    if use_att:
        state["att"].fork(data)
    qkv = _gemm(data["attn_input"], dense["qkv_gate"])
    if use_att:
        state["att"].pre(data)  # cached-key packing beside the projection
    o = (state["att"](data, qkv) if (not cfg["window"] or tokens >= ATT_SLIDING_MIN) else optimized_attention(data, qkv, custom_window=CUSTOM_WINDOW).reshape(tokens, hq * d))
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
// prefill defaults (cand_moe): wgmma groups kept in flight across stage pairs, a shorter split
// tail, and the original gather
#ifndef MOE_NO_XPIPE_DEFAULT
#define MOE_XPIPE
#endif
#ifndef MOE_SPLW
#define MOE_SPLW 0.5f
#endif
#ifndef MOE_NEW_GATHER
#define MOE_OLD_GATHER
#endif
// SiLU as 0.5 g (1 + tanh(g / 2)): one MUFU op per intermediate (+0.7-1%)
#ifndef MOE_NO_TANH_SILU
#define MOE_TANH_SILU
#endif
// next wgmma group's weight words loaded one group ahead (+1-1.7%); the first item's first
// stages' weights issued before the wait for every CTA's gather (+0.5-1% at M 753)
#ifndef MOE_NO_WG2_PF
#define MOE_WG2_PF
#endif
// gate/up epilogue of wgmma tiles straight from the accumulators (+1-1.5% at M 3012 / 16K)
#ifndef MOE_NO_REPI
#define MOE_REPI
#endif
#ifndef MOE_NO_GEARLY2
#define MOE_GEARLY2
#endif
// the last tiles' down items split this many ways in k (multi-group configs; v130: 2 beat 4)
#ifndef MOE_KSX8
#define MOE_KSX8 2
#endif
#ifndef MOE_COMBINE_IN
#define MOE_NO_COMBINE
#endif

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
// silu(g) * u: with MOE_TANH_SILU as 0.5 g (1 + tanh(g / 2)) u (one MUFU op instead of two)
DEVI float silu_mul(float g, float u) {
#ifdef MOE_TANH_SILU
  float t;
  const float h = 0.5f * g;
  asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(h));
  return fmaf(h, t, h) * u;
#else
  return __fdividef(g, 1.f + __expf(-g)) * u;
#endif
}
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
DEVI void pin_a(u32 (&a)[2][4][4]) {
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) asm volatile("" : "+r"(a[i][j][k]) :: "memory");
}
template <int NT, int NTL>
DEVI void stage_compute_wg(float (&acc)[2][NT][4], const unsigned char* wbase, const unsigned char* sclb,
                           u32 xaddr, int T, u32 rel_bar, bool last, u32 (&A)[2][2][4][4]) {
#ifndef MOE_WG_QPERKK
  uint4 qw[2][4];
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    qw[0][kk] = *reinterpret_cast<const uint4*>(wbase + kk * 512);
    qw[1][kk] = *reinterpret_cast<const uint4*>(wbase + 2048 + kk * 512);
  }
#else
  // weight words one 64-k chunk ahead (fewer live registers while the previous stage's last
  // group is still in flight)
  uint4 qn0 = *reinterpret_cast<const uint4*>(wbase), qn1 = *reinterpret_cast<const uint4*>(wbase + 2048);
#endif
  const uint2 sv0 = *reinterpret_cast<const uint2*>(sclb);
  const uint2 sv1 = *reinterpret_cast<const uint2*>(sclb + 256);
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
#ifndef MOE_WG_QPERKK
    const uint4 q0 = qw[0][kk], q1 = qw[1][kk];
#else
#ifdef MOE_WG_PF
    const uint4 q0 = qn0, q1 = qn1;
    if (kk < 3) {
      qn0 = *reinterpret_cast<const uint4*>(wbase + (kk + 1) * 512);
      qn1 = *reinterpret_cast<const uint4*>(wbase + 2048 + (kk + 1) * 512);
    }
#else
    const uint4 q0 = kk ? *reinterpret_cast<const uint4*>(wbase + kk * 512) : qn0;
    const uint4 q1 = kk ? *reinterpret_cast<const uint4*>(wbase + 2048 + kk * 512) : qn1;
#endif
#endif
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
#ifdef MOE_XPIPE
    pin_a(A[(kk + 1) & 1]);  // the group that read these registers has retired
#endif
#ifdef MOE_XPIPE
    if (kk == 0 && rel_bar) {  // the previous stage's last group has retired: release its slot
      __syncwarp();
      if ((threadIdx.x & 31) == 0) mbar_arrive(rel_bar);
    }
#endif
  }
#ifdef MOE_XPIPE
  if (!rel_bar && !last) return;  // pipelined stage: the caller retires the last group
  if (!last) return;
#endif
  wg_wait<0>();
  pin_acc<NT>(acc);
}

template <int NT, int NTL>
DEVI void stage_compute_wg(float (&acc)[2][NT][4], const unsigned char* wbase, const unsigned char* sclb,
                           u32 xaddr, int T) {
  u32 A[2][2][4][4];
  stage_compute_wg<NT, NTL>(acc, wbase, sclb, xaddr, T, 0u, true, A);
}

// Pipelined wgmma stage (MOE_XPIPE): one wgmma group per 64-k chunk and row block, so each row
// block's A fragments (16 registers) have their own buffer; the stage's last group stays in flight
// across the stage boundary and the next stage's first wait retires it (then rel_bar releases the
// previous stage's slot).
template <int NT, int NTL>
DEVI void wg_issue4(float (&acc)[NT][4], const u32 (&Ab)[4][4], u64 d0) {
#pragma unroll
  for (int jj = 0; jj < 4; ++jj) {
    if constexpr (NTL == 3) wgmma_n24<NT>(acc, Ab[jj], d0 + 2 * jj);
    else if constexpr (NTL == 4) wgmma_n32<NT>(acc, Ab[jj], d0 + 2 * jj);
    else if constexpr (NTL == 5) wgmma_n40<NT>(acc, Ab[jj], d0 + 2 * jj);
    else if constexpr (NTL == 6) wgmma_n48<NT>(acc, Ab[jj], d0 + 2 * jj);
    else if constexpr (NTL == 7) wgmma_n56<NT>(acc, Ab[jj], d0 + 2 * jj);
    else if constexpr (NTL == 8) wgmma_n64<NT>(acc, Ab[jj], d0 + 2 * jj);
    else if constexpr (NTL == 9) wgmma_n72<NT>(acc, Ab[jj], d0 + 2 * jj);
    else if constexpr (NTL == 10) wgmma_n80<NT>(acc, Ab[jj], d0 + 2 * jj);
    else if constexpr (NTL == 11) wgmma_n88<NT>(acc, Ab[jj], d0 + 2 * jj);
    else if constexpr (NTL == 12) wgmma_n96<NT>(acc, Ab[jj], d0 + 2 * jj);
    else if constexpr (NTL == 13) wgmma_n104<NT>(acc, Ab[jj], d0 + 2 * jj);
    else if constexpr (NTL == 14) wgmma_n112<NT>(acc, Ab[jj], d0 + 2 * jj);
    else if constexpr (NTL == 15) wgmma_n120<NT>(acc, Ab[jj], d0 + 2 * jj);
    else wgmma_n128<NT>(acc, Ab[jj], d0 + 2 * jj);
  }
}
DEVI void pin_a4(u32 (&a)[4][4]) {
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int k = 0; k < 4; ++k) asm volatile("" : "+r"(a[j][k]) :: "memory");
}
template <int NT, int NTL, bool DRAIN>
DEVI void stage_compute_wg2(float (&acc)[2][NT][4], const unsigned char* wbase, const unsigned char* sclb,
                            u32 xaddr, int T, u32 rel_bar, u32 (&A)[2][4][4]) {
  const uint2 sv0 = *reinterpret_cast<const uint2*>(sclb);
  const uint2 sv1 = *reinterpret_cast<const uint2*>(sclb + 256);
#ifdef MOE_WG2_PF
  uint4 qn = *reinterpret_cast<const uint4*>(wbase);
#endif
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    const u64 d0 = wg_desc(xaddr + kk * T * 128);
#pragma unroll
    for (int r = 0; r < 2; ++r) {
#ifdef MOE_WG2_PF
      const uint4 q = qn;  // the next group's words load under this group's dequant
      if (kk * 2 + r < 7) qn = *reinterpret_cast<const uint4*>(wbase + (r ^ 1) * 2048 + (kk + r) * 512);
#else
      const uint4 q = *reinterpret_cast<const uint4*>(wbase + r * 2048 + kk * 512);
#endif
      const u32 sw = r ? (kk < 2 ? sv1.x : sv1.y) : (kk < 2 ? sv0.x : sv0.y);
      constexpr bool SIM = NTL == 1;
      const u32 sg = (kk & 1) ? scale_of<2, SIM>(sw) : scale_of<0, SIM>(sw);
      const u32 sg8 = (kk & 1) ? scale_of<3, SIM>(sw) : scale_of<1, SIM>(sw);
      u32 (&Ab)[4][4] = A[r];
      dequant(q.x, sg, sg8, Ab[0][0], Ab[0][1], Ab[0][2], Ab[0][3]);
      dequant(q.y, sg, sg8, Ab[1][0], Ab[1][1], Ab[1][2], Ab[1][3]);
      dequant(q.z, sg, sg8, Ab[2][0], Ab[2][1], Ab[2][2], Ab[2][3]);
      dequant(q.w, sg, sg8, Ab[3][0], Ab[3][1], Ab[3][2], Ab[3][3]);
      wg_fence();
      wg_issue4<NT, NTL>(acc[r], Ab, d0);
      wg_commit();
      wg_wait<1>();
      pin_a4(A[r ^ 1]);  // its group has retired
      if (kk == 0 && r == 0 && rel_bar) {
        __syncwarp();
        if ((threadIdx.x & 31) == 0) mbar_arrive(rel_bar);
      }
    }
  }
  if constexpr (DRAIN) {
    wg_wait<0>();
    pin_acc<NT>(acc);
    pin_a4(A[0]);
    pin_a4(A[1]);
  }
}

// Three A buffers (MOE_A3): up to three of the warpgroup's groups in flight (wait<2>), so a group's
// dequant has two groups' tensor time to hide in. Group n of the straight-line block (GOFF + 2 kk + r)
// uses buffer n % 3; the previous stage's last group retires at this stage's second wait.
template <int NT, int NTL, int GOFF, bool DRAIN>
DEVI void stage_compute_wg3(float (&acc)[2][NT][4], const unsigned char* wbase, const unsigned char* sclb,
                            u32 xaddr, int T, u32 rel_bar, u32 (&A)[3][4][4]) {
  const uint2 sv0 = *reinterpret_cast<const uint2*>(sclb);
  const uint2 sv1 = *reinterpret_cast<const uint2*>(sclb + 256);
  uint4 qn = *reinterpret_cast<const uint4*>(wbase);
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    const u64 d0 = wg_desc(xaddr + kk * T * 128);
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int n = GOFF + 2 * kk + r;
      const uint4 q = qn;
      if (kk * 2 + r < 7) qn = *reinterpret_cast<const uint4*>(wbase + (r ^ 1) * 2048 + (kk + r) * 512);
      const u32 sw = r ? (kk < 2 ? sv1.x : sv1.y) : (kk < 2 ? sv0.x : sv0.y);
      constexpr bool SIM = NTL == 1;
      const u32 sg = (kk & 1) ? scale_of<2, SIM>(sw) : scale_of<0, SIM>(sw);
      const u32 sg8 = (kk & 1) ? scale_of<3, SIM>(sw) : scale_of<1, SIM>(sw);
      u32 (&Ab)[4][4] = A[n % 3];
      dequant(q.x, sg, sg8, Ab[0][0], Ab[0][1], Ab[0][2], Ab[0][3]);
      dequant(q.y, sg, sg8, Ab[1][0], Ab[1][1], Ab[1][2], Ab[1][3]);
      dequant(q.z, sg, sg8, Ab[2][0], Ab[2][1], Ab[2][2], Ab[2][3]);
      dequant(q.w, sg, sg8, Ab[3][0], Ab[3][1], Ab[3][2], Ab[3][3]);
      wg_fence();
      wg_issue4<NT, NTL>(acc[r], Ab, d0);
      wg_commit();
      wg_wait<2>();
      pin_a4(A[(n + 1) % 3]);  // group n - 2 has retired: its buffer is the next group's
      if (kk == 0 && r == 1 && rel_bar) {
        __syncwarp();
        if ((threadIdx.x & 31) == 0) mbar_arrive(rel_bar);
      }
    }
  }
  if constexpr (DRAIN) {
    wg_wait<0>();
    pin_acc<NT>(acc);
    pin_a4(A[0]);
    pin_a4(A[1]);
    pin_a4(A[2]);
  }
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

// The same stage on the wgmma activation layout (MOE_LMX: every tile of a wgmma build stores its
// rows [4 chunks of 64 k][T rows][128 B], 16-byte groups swizzled by row % 8, k in the weights'
// order): the B fragments by ldmatrix, the same registers as stage_compute's row-major loads.
DEVI void ldsm_x4(u32 a, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
template <int NT, int NTL>
DEVI void stage_compute_lm(float (&acc)[2][NT][4], const unsigned char* wbase, const unsigned char* sclb,
                           u32 xaddr, int T) {
  uint4 qw[2][4];
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    qw[0][kk] = *reinterpret_cast<const uint4*>(wbase + kk * 512);
    qw[1][kk] = *reinterpret_cast<const uint4*>(wbase + 2048 + kk * 512);
  }
  const uint2 sv0 = *reinterpret_cast<const uint2*>(sclb);
  const uint2 sv1 = *reinterpret_cast<const uint2*>(sclb + 256);
  // lane l: row l % 8 of matrix l / 8 (k16 steps 0-1, then 2-3, of a chunk: groups 0-3, 4-7)
  const int l = threadIdx.x & 31, l7 = l & 7, grp = l >> 3;
  const u32 xl = xaddr + l7 * 128, o0 = (u32)((grp ^ l7) << 4), o1 = (u32)(((grp + 4) ^ l7) << 4);
  u32 bx[2][NTL][8];
#pragma unroll
  for (int nt = 0; nt < NTL; ++nt) {
    ldsm_x4(xl + nt * 1024 + o0, bx[0][nt][0], bx[0][nt][1], bx[0][nt][2], bx[0][nt][3]);
    ldsm_x4(xl + nt * 1024 + o1, bx[0][nt][4], bx[0][nt][5], bx[0][nt][6], bx[0][nt][7]);
  }
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    if (kk < 3) {
#pragma unroll
      for (int nt = 0; nt < NTL; ++nt) {
        const u32 a = xl + (kk + 1) * T * 128 + nt * 1024;
        u32 (&b)[8] = bx[(kk + 1) & 1][nt];
        ldsm_x4(a + o0, b[0], b[1], b[2], b[3]);
        ldsm_x4(a + o1, b[4], b[5], b[6], b[7]);
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
      const u32 (&B)[8] = bx[kk & 1][nt];
#pragma unroll
      for (int jj = 0; jj < 4; ++jj)
#pragma unroll
        for (int r = 0; r < 2; ++r)
          mma16816(acc[r][nt], A[r][jj][0], A[r][jj][1], A[r][jj][2], A[r][jj][3], B[2 * jj], B[2 * jj + 1]);
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
  const u16* prenorm;  // non-null: x is the raw pooler output; gathers apply rmsnorm(row) * prenorm
  // bucketed routes (the router kernel or moe_bucket): cur[e] routes of local expert e, bkt[e * M + i]
  // their tokens in arrival order, spos[t * 8 + k] slot k's index among its expert's routes (-1: not local)
  int* cur; u16* bkt; int* spos;
};
DEVI void pdl_wait() {
#ifdef MOE_PDL
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
DEVI void pdl_launch() {
#ifdef MOE_PDL
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
// Routing tables from the bucket counts, in one warp (lane = local expert): experts in rank order
// (count descending, then index; index order for single-group tiles), T-token tiles. rt[4][33] by
// rank: expert, routes, first tile, first pair (entry 32: the totals); scal, ftile and off as before.
template <int RW, int NT, int KS2>
DEVI void bucket_tables(const Params& P, int lane, int* scal, int* ftile, int* off, int* rt) {
  constexpr int KW=8/RW, RB=32*RW, T=8*NT;
  constexpr int NRB1=8192/RB, NRB2=4096/RB;
  constexpr int KSX=KW==1 ? (NT >= 8 ? MOE_KSX8 : 4) : KS2;
  const int c = __ldcg(P.cur + lane);
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
  int* rex = rt; int* rcnt = rt + 33; int* rft = rt + 66; int* roff = rt + 99;
  rex[rank] = lane; rcnt[rank] = c;
  __syncwarp();
  const int cq = rcnt[lane], ntq = (cq + T - 1) / T;
  int sc = cq, st = ntq;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int a = __shfl_up_sync(0xffffffffu, sc, d), b = __shfl_up_sync(0xffffffffu, st, d);
    if (lane >= d) { sc += a; st += b; }
  }
  rft[lane] = st - ntq; roff[lane] = sc - cq;
  if (lane == 31) { rex[32] = 32; rcnt[32] = 0; rft[32] = st; roff[32] = sc; }
  __syncwarp();
  ftile[lane] = rft[rank]; off[lane] = roff[rank];
  const int tot_t = rft[32], tot_p = roff[32];
  if (lane == 0) {
#ifndef MOE_SPLW
#define MOE_SPLW 1.0f
#endif
    // about MOE_SPLW waves of down items, taken from the last (smallest) tiles, split KSX ways
    const int nspl = KSX > KS2 ? min(tot_t, (int)ceilf(MOE_SPLW * P.route_sms / (NRB2 * KS2))) : 0;
    const int tsplit = tot_t - nspl;
    int psplit = tot_p;
    if (nspl > 0) {  // first pair of tile tsplit
      int r = 0;
      while (r < 31 && rft[r + 1] <= tsplit) ++r;
      psplit = roff[r] + (tsplit - rft[r]) * T;
    }
    scal[0] = tot_t;
    scal[1] = NRB1 * tot_t;
    scal[2] = NRB1 * tot_t + NRB2 * KS2 * tsplit + NRB2 * KSX * nspl;
    scal[3] = tot_p;
    scal[4] = tsplit;
    scal[5] = psplit;  // first pair with KSX partials
  }
  __syncwarp();
}
// rank of tile t (largest r with rft[r] <= t; the non-empty ranks come first, strictly increasing)
DEVI int rank_of(const int* v, int x) {
  int lo = 0, hi = 32;
#pragma unroll
  for (int i = 0; i < 5; ++i) { const int mid = (lo + hi) >> 1; if (v[mid] <= x) lo = mid; else hi = mid; }
  return lo;
}
// tile t: (expert, first pair, routes, its first route's index in bkt)
template <int T>
DEVI int4 tile_of(const int* rt, int t, int M) {
  const int r = rank_of(rt + 66, t), k = t - rt[66 + r], e = rt[r];
  return make_int4(e, rt[99 + r] + k * T, min(T, rt[33 + r] - k * T), e * M + k * T);
}
// pair p: tile, row in the tile, the tile's routes, token
template <int T>
DEVI void pair_of(const int* rt, const u16* bkt, int p, int M, int& tile, int& ri, int& n, int& tok) {
  const int r = rank_of(rt + 99, p), i = p - rt[99 + r], k = i / T;
  tile = rt[66 + r] + k; ri = i - k * T; n = min(T, rt[33 + r] - k * T);
  tok = bkt[rt[r] * M + i];
}
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
  constexpr int KSX=KW==1 ? (NT >= 8 ? MOE_KSX8 : 4) : KS2;
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
#ifdef MOE_INTERLEAVE
      // big and small experts alternate in the queue (largest, smallest, 2nd largest, ...): the
      // small tiles' weight streaming overlaps the big tiles' compute instead of trailing it
      {
        const int nz = __popc(__ballot_sync(0xffffffffu, c > 0));
        if (c > 0) rank = rank < (nz + 1) / 2 ? 2 * rank : 2 * (nz - 1 - rank) + 1;
      }
#endif
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
#ifdef MOE_BAL
    // balanced tiles: an expert's pairs split evenly (multiples of 8) over its ceil(c / T) tiles
    { const int tsz = nt ? (((c + nt - 1) / nt + 7) & ~7) : T;
      for (int k = 0; k < nt; ++k) tiles[to + k] = make_int4(lane, o + k * tsz, min(tsz, c - k * tsz), 0); }
#else
    for (int k = 0; k < nt; ++k) tiles[to + k] = make_int4(lane, o + k * T, min(T, c - k * T), 0);
#endif
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
  constexpr int KSX=KW==1 ? (NT >= 8 ? MOE_KSX8 : 4) : KS2;
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
#ifdef MOE_INTERLEAVE
      // big and small experts alternate in the queue (largest, smallest, 2nd largest, ...): the
      // small tiles' weight streaming overlaps the big tiles' compute instead of trailing it
      {
        const int nz = __popc(__ballot_sync(0xffffffffu, c > 0));
        if (c > 0) rank = rank < (nz + 1) / 2 ? 2 * rank : 2 * (nz - 1 - rank) + 1;
      }
#endif
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
#ifdef MOE_BAL
    // balanced tiles: an expert's pairs split evenly (multiples of 8) over its ceil(c / T) tiles
    { const int tsz = nt ? (((c + nt - 1) / nt + 7) & ~7) : T;
      for (int k = 0; k < nt; ++k) tiles[to + k] = make_int4(lane, o + k * tsz, min(tsz, c - k * tsz), 0); }
#else
    for (int k = 0; k < nt; ++k) tiles[to + k] = make_int4(lane, o + k * T, min(T, c - k * T), 0);
#endif
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


// ---- finish: per token, its local routes' post-expert norm, weighted and summed (one warp per
// token; pass 1 gets each route's 1 / rms, pass 2 accumulates 8 columns at a time) ----
struct FinParams {
  const int* ids; const float* tw; const float* nw; const unsigned char* routes;
  const u16* yh; const float* ypart; u16* out;
  int M, mw, pmax, ks2, ksx, local_start;
  const int* spos;
};
DEVI void ld_row8(const FinParams& F, int pair, int parts, int col, float (&v)[8]) {
  if (parts == 1) {
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(F.yh + (size_t)pair * 4096 + col));
    const u32 w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) { v[2 * i] = u2f(w[i] << 16); v[2 * i + 1] = u2f(w[i] & 0xFFFF0000u); }
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = 0.f;
    for (int part = 0; part < parts; ++part) {
      const float* src = F.ypart + ((size_t)part * F.pmax + pair) * 4096 + col;
      const float4 a = __ldg(reinterpret_cast<const float4*>(src)), b = __ldg(reinterpret_cast<const float4*>(src + 4));
      v[0] += a.x; v[1] += a.y; v[2] += a.z; v[3] += a.w; v[4] += b.x; v[5] += b.y; v[6] += b.z; v[7] += b.w;
    }
  }
}
extern "C" __global__ void __launch_bounds__(128) moe_finish(FinParams F) {
  // one 128-thread block per token; thread i owns columns [8i, 8i + 8) + 1024 c, c = 0..3
  const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  __shared__ float red[4][8];
#ifdef MOE_PDL
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  const int* scal = reinterpret_cast<const int*>(F.routes);
  const int psplit = scal[5];
  const int* off = scal + 48;
  const int4 i0 = reinterpret_cast<const int4*>(F.ids)[t * 2], i1 = reinterpret_cast<const int4*>(F.ids)[t * 2 + 1];
  const int ev[8] = {i0.x, i0.y, i0.z, i0.w, i1.x, i1.y, i1.z, i1.w};
  int pr[8], pt[8];
  float ss[8];
#pragma unroll
  for (int slot = 0; slot < 8; ++slot) {
    const int e = ev[slot] - F.local_start;
    pr[slot] = -1; pt[slot] = 1; ss[slot] = 0.f;
    if ((unsigned)e < 32u) {
      pr[slot] = off[e] + F.spos[t * 8 + slot];
      pt[slot] = pr[slot] >= psplit ? F.ksx : F.ks2;
    }
  }
  float nn[4][8];
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int col = c * 1024 + tid * 8;
    const float4 n0 = __ldg(reinterpret_cast<const float4*>(F.nw + col)), n1 = __ldg(reinterpret_cast<const float4*>(F.nw + col + 4));
    nn[c][0] = n0.x; nn[c][1] = n0.y; nn[c][2] = n0.z; nn[c][3] = n0.w; nn[c][4] = n1.x; nn[c][5] = n1.y; nn[c][6] = n1.z; nn[c][7] = n1.w;
  }
#pragma unroll
  for (int slot = 0; slot < 8; ++slot) {
    if (pr[slot] >= 0) {
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        float v[8];
        ld_row8(F, pr[slot], pt[slot], c * 1024 + tid * 8, v);
#pragma unroll
        for (int i = 0; i < 8; ++i) { const float q = v[i] * nn[c][i]; ss[slot] += q * q; }
      }
    }
  }
#pragma unroll
  for (int slot = 0; slot < 8; ++slot) {
#pragma unroll
    for (int o = 16; o; o >>= 1) ss[slot] += __shfl_xor_sync(0xffffffffu, ss[slot], o);
  }
  if (lane < 8) {
    float v = ss[0];
#pragma unroll
    for (int slot = 1; slot < 8; ++slot) v = lane == slot ? ss[slot] : v;
    red[warp][lane] = v;
  }
  __syncthreads();
  float sc[8];
#pragma unroll
  for (int slot = 0; slot < 8; ++slot)
    sc[slot] = pr[slot] >= 0 ? rsqrtf((red[0][slot] + red[1][slot] + red[2][slot] + red[3][slot]) * (1.f / 4096.f) + 1e-8f) * F.tw[t * 8 + slot] : 0.f;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int col = c * 1024 + tid * 8;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
      if (pr[slot] >= 0) {
        float v[8];
        ld_row8(F, pr[slot], pt[slot], col, v);
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] += (v[i] * nn[c][i]) * sc[slot];
      }
    }
    uint4 o;
    o.x = (u32)f2bf(acc[0]) | ((u32)f2bf(acc[1]) << 16);
    o.y = (u32)f2bf(acc[2]) | ((u32)f2bf(acc[3]) << 16);
    o.z = (u32)f2bf(acc[4]) | ((u32)f2bf(acc[5]) << 16);
    o.w = (u32)f2bf(acc[6]) | ((u32)f2bf(acc[7]) << 16);
    *reinterpret_cast<uint4*>(F.out + (size_t)t * 4096 + col) = o;
  }
}

// Pre-expert norm inside the gathers: per row r of a group of 4, 256 threads hold 2 chunks of 8
// values each (wgmma rows: the pairs at k = 64 c64 + 16 e + 2 lc; plain rows: 8 ch .. 8 ch + 7).
// rstd = rsqrt(mean(x^2) + 1e-5) over the row (FP32), then BF16(x * rstd * w) with w the BF16
// (1 + pre_expert_norm), as F.rms_norm rounds once at its output. red: 8 GR floats of shared memory;
// sync: a barrier over the 256 threads.
template <int GR, typename SYNC>
DEVI void gather_prenorm4(const Params& P, uint4 (&val)[GR][2], const bool (&wg)[GR], const int (&ri)[GR],
                          const bool (&live)[GR], float* red, SYNC sync) {
  const int tid = threadIdx.x & 255, lane = tid & 31, warp = tid >> 5;
  float ss[GR];
#pragma unroll
  for (int r = 0; r < GR; ++r) {
    ss[r] = 0.f;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const u32 w4[4] = {val[r][h].x, val[r][h].y, val[r][h].z, val[r][h].w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float a = u2f(w4[i] << 16), b = u2f(w4[i] & 0xFFFF0000u);
        ss[r] += a * a + b * b;
      }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) ss[r] += __shfl_xor_sync(0xffffffffu, ss[r], o);
  }
  sync();  // the previous group's readers of red are done
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < GR; ++r) red[warp * GR + r] = ss[r];
  }
  sync();
#pragma unroll
  for (int r = 0; r < GR; ++r) {
    if (!live[r]) continue;
    float t = 0.f;
#pragma unroll
    for (int wv = 0; wv < 8; ++wv) t += red[wv * GR + r];
    const float rstd = rsqrtf(t * (1.f / 4096.f) + 1e-5f);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int ch = tid + 256 * h;
      u32 w4[4];
      if (wg[r]) {
        const int c64 = ch >> 3, lc = (ch & 7) ^ (ri[r] & 7);
#pragma unroll
        for (int e = 0; e < 4; ++e) w4[e] = __ldg(reinterpret_cast<const unsigned int*>(P.prenorm + c64 * 64 + 16 * e + 2 * lc));
      } else {
        const uint4 q = __ldg(reinterpret_cast<const uint4*>(P.prenorm) + ch);
        w4[0] = q.x; w4[1] = q.y; w4[2] = q.z; w4[3] = q.w;
      }
      u32 x4[4] = {val[r][h].x, val[r][h].y, val[r][h].z, val[r][h].w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float lo = (u2f(x4[i] << 16) * rstd) * u2f(w4[i] << 16);
        const float hi = (u2f(x4[i] & 0xFFFF0000u) * rstd) * u2f(w4[i] & 0xFFFF0000u);
        x4[i] = (u32)f2bf(lo) | ((u32)f2bf(hi) << 16);
      }
      val[r][h] = make_uint4(x4[0], x4[1], x4[2], x4[3]);
    }
  }
}

template <int RW, int NT, int MW>
__device__ __forceinline__ void external_gather(const Params& P) {
  constexpr int KW=8/RW, KST=256*KW, T=8*NT, XP=KST*2+16;
  constexpr int WGMIN=(KW==1&&NT>=8) ? MOE_WGMIN : 99;
  constexpr bool HASWG=WGMIN<99;
#ifndef MOE_NO_LMX
  constexpr bool LMX = HASWG;
#else
  constexpr bool LMX = false;
#endif
  constexpr int MAXP=MW*32*8, MAXT=32+MAXP/T;
  const int tid=threadIdx.x;
  __shared__ int scal[16], ftile[32], off[32], rt[4 * 33];
  if (tid < 32) bucket_tables<RW, NT, 1>(P, tid, scal, ftile, off, rt);
  __syncthreads();
      const int np_all = scal[3];
      const int pb = (int)((long long)np_all * blockIdx.x / gridDim.x);
      const int pe = (int)((long long)np_all * (blockIdx.x + 1) / gridDim.x);
      // Rows in groups of 4: every load of the group is issued before any store, so the rows'
      // HBM latencies overlap (the compiler cannot reorder loads of x past stores to xs).
      for (int pg = pb; pg < pe; pg += 4) {
        uint4 val[4][2];
        int tl[4], ri[4];
        bool wg[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const int p = pg + r;
          tl[r] = 0; ri[r] = 0; wg[r] = false;
          if (p < pe) {
            int nn, tok;
            pair_of<T>(rt, P.bkt, p, P.M, tl[r], ri[r], nn, tok);
            wg[r] = HASWG && (LMX || (nn + 7) / 8 >= WGMIN);
            const u16* srow = P.x + (size_t)tok * 4096;
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
        if (P.prenorm) {
          __shared__ float gred[32];
          const bool live[4] = {pg < pe, pg + 1 < pe, pg + 2 < pe, pg + 3 < pe};
          gather_prenorm4<4>(P, val, wg, ri, live, gred, [] { __syncthreads(); });
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
  // every tile's activations (and intermediate) on the wgmma layout; mma.sync tiles read them by
  // ldmatrix (MOE_LMX)
#ifndef MOE_NO_LMX
  constexpr bool LMX = HASWG;
#else
  constexpr bool LMX = false;
#endif
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
  constexpr int KSX = KW == 1 ? (NT >= 8 ? MOE_KSX8 : 4) : KS2;
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
  int* rt = reinterpret_cast<int*>(smem + 1024 + NS * STAGE);  // [4][33] routing tables by rank

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
  pdl_wait();  // the router's bucketed routes (and the external gather)
  if (warp == 0) {
    bucket_tables<RW, NT, KS2>(P, lane, scal, ftile, off, rt);
    if (blockIdx.x == 0) {  // for the finish
      int* g = reinterpret_cast<int*>(P.routes);
      if (lane < 16) g[lane] = scal[lane];
      g[48 + lane] = off[lane];
    }
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
#ifndef MOE_IGR
#define MOE_IGR 4
#endif
      constexpr int IGR = MOE_IGR;  // rows per group: all of a group's loads go out before its stores
      // This CTA's share of the pairs: their activation rows, gathered into the gate/up GEMM's
      // stage-major layout [tile][stage][T rows][XP bytes]; then count this CTA done.
      const int np_all = scal[3];
      const int pb = (int)((long long)np_all * blockIdx.x / gridDim.x);
      const int pe = (int)((long long)np_all * (blockIdx.x + 1) / gridDim.x);
      // Rows in groups of 4: every load of the group is issued before any store, so the rows'
      // HBM latencies overlap (the compiler cannot reorder loads of x past stores to xs).
      for (int pg = pb; pg < pe; pg += IGR) {
        uint4 val[IGR][2];
        int tl[IGR], ri[IGR];
        bool wg[IGR];
#pragma unroll
        for (int r = 0; r < IGR; ++r) {
          const int p = pg + r;
          tl[r] = 0; ri[r] = 0; wg[r] = false;
          if (p < pe) {
            int nn, tok;
            pair_of<T>(rt, P.bkt, p, M, tl[r], ri[r], nn, tok);
            wg[r] = HASWG && (LMX || (nn + 7) / 8 >= WGMIN);
            const u16* srow = P.x + (size_t)tok * 4096;
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
        if (P.prenorm) {
          bool live[IGR];
#pragma unroll
          for (int r = 0; r < IGR; ++r) live[r] = pg + r < pe;
          gather_prenorm4<IGR>(P, val, wg, ri, live, reinterpret_cast<float*>(stages), [] { consumer_sync(); });
        }
#pragma unroll
        for (int r = 0; r < IGR; ++r) {
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
        const int4 tl = tile_of<T>(rt, tile, M);
        early = NS < ST1 ? NS : ST1;
        const bool wgt = HASWG && (LMX || (tl.z + 7) / 8 >= WGMIN);
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
    if constexpr (false && EARLY && !GATHER && !HASWG) {  // (token-order rows: not with bucketed routes)
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
      const int e2 = tile_of<T>(rt, tl2, M).x;
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
        pdl_launch();  // this CTA's queue is empty: the finish may launch
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
      const int4 tl = tile_of<T>(rt, tile, M);
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
            if (HASWG && (LMX || (n + 7) / 8 >= WGMIN)) {
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
          const bool wgt = HASWG && (LMX || (n + 7) / 8 >= WGMIN);
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
          if (HASWG && (LMX || (n + 7) / 8 >= WGMIN)) {
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
            if (lane < n) xrow0 = P.x + (size_t)P.bkt[tl.w + lane] * 4096;
            if (lane + 32 < n) xrow1 = P.x + (size_t)P.bkt[tl.w + lane + 32] * 4096;
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
#ifndef MOE_XPU4_MW
#define MOE_XPU4_MW 1024
#endif
#ifndef MOE_XPU
#define MOE_XPU 2
#endif
  // stages per pipelined block: 4 for the largest batches' build (fewer drains), else MOE_XPU
  constexpr int XPUV = MW >= MOE_XPU4_MW ? 4 : MOE_XPU;
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
#ifndef MOE_XPIPE
#define MOE_RUN_ITEM(K)                                                                                  \
    if constexpr (NT == 1) { MOE_RUN_ITEM_BOT(K) } else { MOE_RUN_ITEM_TOP(K) }
#else
#define MOE_RUN_ITEM(K)                                                                                  \
    if constexpr (NT == 1) { MOE_RUN_ITEM_BOT(K) } else if constexpr ((K) >= 100) { MOE_RUN_ITEM_WGP(K) } else { MOE_RUN_ITEM_TOP(K) }
#endif
// wgmma items: a stage's last wgmma group stays in flight across the stage boundary; the next
// stage's first wait retires it and releases its slot
#ifdef MOE_XPIPE2
// continuous: every stage boundary keeps one group in flight; the item's end drains
#define MOE_RUN_ITEM_WGP(K)                                                                              \
    u32 Awg[2][4][4];                                                                                    \
    for (int st = 0; st < nst; ++st) {                                                                   \
      u32 rel = 0;                                                                                       \
      if (st > 0) {                                                                                      \
        rel = smem_addr(&bars[NS + s]);                                                                  \
        if (++s == NS) { s = 0; ph ^= 1; }                                                               \
        mbar_wait(smem_addr(&bars[s]), ph);                                                              \
      }                                                                                                  \
      const unsigned char* sbase = stages + s * STAGE;                                                   \
      const unsigned char* wbase = sbase + (kw * RGS + rw * 2) * 2048 + lane * 16;                       \
      const unsigned char* sclb = sbase + WB + (kw * RGS + rw * 2) * 256 + lane * 8;                     \
      if constexpr (MOE_NOC) { if (rel && lane == 0) mbar_arrive(rel); }                                 \
      else if constexpr (HASWG)                                                                          \
        stage_compute_wg2<NT, (K) - 100, false>(acc, wbase, sclb, smem_addr(sbase + WB + SB), T, rel, Awg); \
      __syncwarp();                                                                                      \
    }                                                                                                    \
    if constexpr (!MOE_NOC) { wg_wait<0>(); pin_acc<NT>(acc); pin_a4(Awg[0]); pin_a4(Awg[1]); }
#else
#ifndef MOE_XPU
#define MOE_XPU 2
#endif
// MOE_XPU stages per straight-line block: the in-flight group crosses the block's inner stage
// boundaries; the block's last stage drains
#ifdef MOE_A3
// two stages per straight-line block, three A buffers
#define MOE_RUN_ITEM_WGP(K)                                                                              \
    u32 Awg[3][4][4];                                                                                    \
    for (int st = 0; st < nst; st += 2) {                                                                \
      if (st > 0) {                                                                                      \
        if (lane == 0) mbar_arrive(smem_addr(&bars[NS + s]));                                            \
        if (++s == NS) { s = 0; ph ^= 1; }                                                               \
        MOE_T(tw0 = ctimer();)                                                                           \
        mbar_wait(smem_addr(&bars[s]), ph);                                                              \
        MOE_T(c_wait += ctimer() - tw0;)                                                                 \
      }                                                                                                  \
      {                                                                                                  \
        const unsigned char* sbase = stages + s * STAGE;                                                 \
        if constexpr (!MOE_NOC && HASWG)                                                                 \
          stage_compute_wg3<NT, (K) - 100, 0, false>(acc, sbase + (kw * RGS + rw * 2) * 2048 + lane * 16, \
              sbase + WB + (kw * RGS + rw * 2) * 256 + lane * 8, smem_addr(sbase + WB + SB), T, 0u, Awg); \
      }                                                                                                  \
      __syncwarp();                                                                                      \
      const u32 rel = smem_addr(&bars[NS + s]);                                                          \
      if (++s == NS) { s = 0; ph ^= 1; }                                                                 \
      MOE_T(tw0 = ctimer();)                                                                             \
      mbar_wait(smem_addr(&bars[s]), ph);                                                                \
      MOE_T(c_wait += ctimer() - tw0;)                                                                   \
      {                                                                                                  \
        const unsigned char* sbase = stages + s * STAGE;                                                 \
        if constexpr (MOE_NOC) { if (lane == 0) mbar_arrive(rel); }                                      \
        else if constexpr (HASWG)                                                                        \
          stage_compute_wg3<NT, (K) - 100, 8, true>(acc, sbase + (kw * RGS + rw * 2) * 2048 + lane * 16, \
              sbase + WB + (kw * RGS + rw * 2) * 256 + lane * 8, smem_addr(sbase + WB + SB), T, rel, Awg); \
      }                                                                                                  \
      __syncwarp();                                                                                      \
      MOE_T(c_stages += 2;)                                                                              \
    }
#else
#define MOE_RUN_ITEM_WGP(K)                                                                              \
    u32 Awg[2][4][4];                                                                                    \
    for (int st = 0; st < nst; st += XPUV) {                                                          \
      if (st > 0) {                                                                                      \
        if (lane == 0) mbar_arrive(smem_addr(&bars[NS + s]));                                            \
        if (++s == NS) { s = 0; ph ^= 1; }                                                               \
        MOE_T(tw0 = ctimer();)                                                                           \
        mbar_wait(smem_addr(&bars[s]), ph);                                                              \
        MOE_T(c_wait += ctimer() - tw0;)                                                                 \
      }                                                                                                  \
      _Pragma("unroll")                                                                                  \
      for (int u = 0; u < XPUV; ++u) {                                                                \
        u32 rel = 0;                                                                                     \
        if (u > 0) {                                                                                     \
          rel = smem_addr(&bars[NS + s]);                                                                \
          if (++s == NS) { s = 0; ph ^= 1; }                                                             \
          MOE_T(tw0 = ctimer();)                                                                         \
          mbar_wait(smem_addr(&bars[s]), ph);                                                            \
          MOE_T(c_wait += ctimer() - tw0;)                                                               \
        }                                                                                                \
        const unsigned char* sbase = stages + s * STAGE;                                                 \
        const unsigned char* wbase = sbase + (kw * RGS + rw * 2) * 2048 + lane * 16;                     \
        const unsigned char* sclb = sbase + WB + (kw * RGS + rw * 2) * 256 + lane * 8;                   \
        if constexpr (MOE_NOC) { if (rel && lane == 0) mbar_arrive(rel); }                               \
        else if constexpr (HASWG)                                                                        \
          stage_compute_wg2<NT, (K) - 100, false>(acc, wbase, sclb, smem_addr(sbase + WB + SB), T, rel, Awg); \
        __syncwarp();                                                                                    \
      }                                                                                                  \
      if constexpr (!MOE_NOC) { wg_wait<0>(); pin_acc<NT>(acc); pin_a4(Awg[0]); pin_a4(Awg[1]); }       \
      MOE_T(c_stages += XPUV;)                                                                        \
    }
#endif
#endif
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
      } else if constexpr (LMX) {                                                                        \
        stage_compute_lm<NT, (K)>(acc, wbase, sclb, smem_addr(sbase + WB + SB), T);                      \
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
#undef MOE_RUN_ITEM_WGP
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
#ifdef MOE_NO_EPI
    if (true) {}  // (ablation: no epilogue stores; outputs are garbage)
    else
#endif
#ifdef MOE_REPI
    // gate/up items of wgmma tiles: SwiGLU straight from the accumulators (rows 16 j + g are gate,
    // 16 j + 8 + g up, intermediate 8 j + g), BF16 into the slot in the down GEMM's B layout
    // ([2 chunks of 64][T tokens][64 swizzled positions] = two contiguous 16 KB blocks), then
    // coalesced copies out
    if (HASWG && T == 128 && KW == 1 && f.phase == 1 && ntl >= WGMIN) {
      if constexpr (HASWG && T == 128 && KW == 1) {
        u16* stg16 = reinterpret_cast<u16*>(stages + s * STAGE);
        const float gsc = gsm[f.e];
        consumer_sync();  // every warp is done reading the slot
#pragma unroll
        for (int a = 0; a < 2; ++a) {
          const int j = 2 * rw + a;
          const int c64l = j >> 3;
          const int pos = 2 * (rw & 3) + (g & 1);
#pragma unroll
          for (int b = 0; b < NT; ++b)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              const int tok = 8 * b + 2 * tig + e;
              const float gt = acc[a][b][e] * gsc, up = acc[a][b][e + 2] * gsc;
              const int pc = (4 * a + (g >> 1)) ^ (tok & 7);
              stg16[(c64l * T + tok) * 64 + pc * 8 + pos] = f2bf(silu_mul(gt, up));
            }
        }
        consumer_sync();
        u16* dt = P.inter + (size_t)f.tile * 16 * T * XP / 2;
        const int nrow = (f.n + 7) & ~7;  // the down GEMM reads whole 8-token groups
#pragma unroll
        for (int c64l = 0; c64l < 2; ++c64l) {
          const uint4* src = reinterpret_cast<const uint4*>(stg16 + c64l * T * 64);
          uint4* dst = reinterpret_cast<uint4*>(dt + ((size_t)(2 * f.rb + c64l) * T) * 64);
          for (int o = tid; o < nrow * 8; o += 256) dst[o] = src[o];
        }
      }
    } else if (HASWG && T == 128 && KW == 1 && f.phase == 2 && ntl >= WGMIN && (f.tile >= tsplit ? KSX : KS2) == 1) {
      // whole-k down items of wgmma tiles: y in BF16 staged [token][256 rows] (padded rows), then
      // each token's 512 contiguous bytes of yh written by 32 threads
      if constexpr (HASWG && T == 128 && KW == 1) {
        constexpr int YS = 264;
        u16* stg16 = reinterpret_cast<u16*>(stages + s * STAGE);
        const float gsc = gsm[32 + f.e];
        consumer_sync();
#pragma unroll
        for (int a = 0; a < 2; ++a)
#pragma unroll
          for (int b = 0; b < NT; ++b)
#pragma unroll
            for (int c = 0; c < 4; ++c)
              stg16[(8 * b + 2 * tig + (c & 1)) * YS + 16 * (2 * rw + a) + g + 8 * (c >> 1)] = f2bf(acc[a][b][c] * gsc);
        consumer_sync();
        for (int o = tid; o < f.n * 32; o += 256) {
          const int tok = o >> 5, c16 = o & 31;
          *reinterpret_cast<uint4*>(P.yh + (size_t)(f.p0 + tok) * 4096 + f.rb * RB + c16 * 8) =
              *reinterpret_cast<const uint4*>(stg16 + tok * YS + c16 * 8);
        }
      }
    } else
#endif
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
        if (HASWG && (LMX || ntl >= WGMIN)) {
          // the down GEMM runs this tile on wgmma (or on mma.sync by ldmatrix, MOE_LMX): per 64-k chunk and token, 8 swizzled 16-byte
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
    if (tid < 32) P.cur[tid] = 0;  // bucket counts (every CTA read them at its start)
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

// Bucketing for callers whose router does not bucket (one thread per token, 256 per CTA): per-CTA
// counts in shared memory, one global reservation per expert and CTA, then the routes' slots.
extern "C" __global__ void __launch_bounds__(256) moe_bucket(Params P) {
  __shared__ int hist[32], base[32];
  pdl_wait();
  pdl_launch();
  const int tid = threadIdx.x, t = blockIdx.x * 256 + tid;
  if (tid < 32) hist[tid] = 0;
  __syncthreads();
  int ev[8], lp[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    ev[j] = -1; lp[j] = 0;
    if (t < P.M) {
      const int e = P.ids[t * 8 + j] - P.local_start;
      if ((unsigned)e < 32u) { ev[j] = e; lp[j] = atomicAdd(&hist[e], 1); }
    }
  }
  __syncthreads();
  if (tid < 32 && hist[tid] > 0) base[tid] = atomicAdd(P.cur + tid, hist[tid]);
  __syncthreads();
  if (t < P.M) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int pos = -1;
      if (ev[j] >= 0) { pos = base[ev[j]] + lp[j]; P.bkt[ev[j] * P.M + pos] = (u16)t; }
      P.spos[t * 8 + j] = pos;
    }
  }
}

// programmatic dependent launch: wait for the previous grid's results, then let the next grid
// launch (its own wait still orders it after this grid's completion); no-ops without PDL
DEVI void pdl_entry() {
#ifdef MOE_PDL
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
#define INST(RW, NT, KS2, NS, MW) \
  extern "C" __global__ void __launch_bounds__(256, 3) gather_##RW##_##NT##_##KS2##_##NS##_##MW(Params p) { pdl_entry(); external_gather<RW, NT, MW>(p); } \
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
    return 2048 + ns * _stage_bytes(rw, nt)  # header, ring, routing tables by rank


# (largest M, (RW, NT, KS2)) per batch-size bucket; stages are as many as shared memory holds (<= 4).
BUCKETS = [
    (1, (2, 1, 1)),
    (16, (4, 1, 1)),
    (32, (8, 4, 2)),
    (128, (8, 4, 1)),
    (MAX_M, (8, 16, 1)),
]


def _mw(M):
    m = 1 << ((max(1, (M + 31) // 32) - 1).bit_length())
    if M > 128:  # large batches: three routing-table sizes (fewer kernel builds)
        m = 32 if m <= 32 else 128 if m <= 128 else 512
    return m


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

    def launch(self, name, grid, block, smem, params, pdl=False):
        f = self.func(name, smem)
        stream = self.driver.CUstream(torch.cuda.current_stream().cuda_stream)
        argv = (ctypes.c_void_p * 1)(ctypes.addressof(params))
        if pdl and PDL:
            # programmatic dependent launch: this grid may start while the previous one (ours,
            # which triggers early) finishes; the kernel's griddepcontrol.wait orders its reads
            d = self.driver
            cfg = d.CUlaunchConfig()
            cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
            cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block, 1, 1
            cfg.sharedMemBytes = smem
            cfg.hStream = stream
            at = d.CUlaunchAttribute()
            at.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
            at.value.programmaticStreamSerializationAllowed = 1
            cfg.attrs = [at]
            cfg.numAttrs = 1
            (err,) = d.cuLaunchKernelEx(cfg, f, ctypes.addressof(argv), 0)
        else:
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
        ("routes", ctypes.c_void_p), ("route_sms", ctypes.c_int), ("prenorm", ctypes.c_void_p),
        ("cur", ctypes.c_void_p), ("bkt", ctypes.c_void_p), ("spos", ctypes.c_void_p),
    ]


NATIVE_FINISH_CUDA = False
GATHER_MIN_NT = 4  # MOE_GATHER_MIN: configs whose gate/up activations are gathered
PDL = True  # programmatic dependent launches between the expert-path kernels
COMBINE_IN = False  # True: build with MOE_COMBINE_IN (the in-kernel combine) and skip the finish
KSX8 = 1  # must match MOE_KSX8 in the CUDA source (1: no tail split, +1% at M 3012)


def _ksx(rw, nt, ks2):
    return (KSX8 if nt >= 8 else 4) if rw == 8 else ks2


class _FinParamsC(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in ("ids", "tw", "nw", "routes", "yh", "ypart", "out")] + [
        (n, ctypes.c_int) for n in ("M", "mw", "pmax", "ks2", "ksx", "local_start")] + [("spos", ctypes.c_void_p)]


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
        src = ("#define MOE_FAST_SCALES\n" if fast_scales else "") + f"#define MOE_KSX8 {KSX8}\n" + ("#define MOE_PDL\n" if PDL else "") + ("#define MOE_COMBINE_IN\n" if COMBINE_IN else "") + CUDA_SRC
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
        if cfg[4] >= 128:
            mod.func(_name(cfg).replace("moe_", "gather_", 1))
    mod.func("moe_bucket")
    mod.func("moe_finish")
    mod.func(f"moe1_{NS1}", 18432 + NS1 * 36864)
    torch.cuda.synchronize()
    return {
        "start": start, "count": count, "sms": torch.cuda.get_device_properties(dev).multi_processor_count,
        "w13": w13p, "s13": s13p, "gs13": weights["w13_gs"][start:start + count].float().contiguous(),
        "w2": w2p, "s2": s2p, "gs2": weights["w2_gs"][start:start + count].float().contiguous(),
        "nw1": (1.0 + weights["post_norm_w"].float()).contiguous(),
        "ctr": torch.zeros(CTR_WORDS, dtype=torch.int32, device=dev),
        # bucketed routes: counts (zeroed again by each expert kernel), tokens, slot positions
        "cur": torch.zeros(32, dtype=torch.int32, device=dev),
        "bkt": torch.empty(32 * MAX_M, dtype=torch.int16, device=dev),
        "spos": torch.empty(MAX_M * TOPK, dtype=torch.int32, device=dev),
        "tmaps": tmaps, "tmap_off": tmap_off,
        "w2f": w2f, "s2f": s2f, "yacc": torch.zeros(8 * 8 * 4096, dtype=torch.float32, device=dev),
        "ctr1": torch.zeros(4, dtype=torch.int32, device=dev),
        "trc": torch.zeros(160 * 16, dtype=torch.int32, device=dev),
        "raw": weights,
        # the pre-expert norm's effective weight, BF16 (1 + w) as SGLang freezes it
        "prenorm": (weights["pre_norm_w"] + 1.0).contiguous() if "pre_norm_w" in weights else None,
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



try:  # programmatic dependent launch of the finish (PDL)
    from triton.language.extra.cuda import gdc_wait as _gdc_wait
except Exception:  # noqa: BLE001
    _gdc_wait = None


@triton.jit
def _native_finish(IDS, TW, NW, ROUTES, YH, YPART, OUT, SPOS,
                   MW:tl.constexpr, PMAX:tl.constexpr, KS2:tl.constexpr,
                   KSX:tl.constexpr, LOCAL_START:tl.constexpr, PDLW:tl.constexpr):
    if PDLW:
        _gdc_wait()  # the expert kernel's outputs (it let this grid launch early)
    token=tl.program_id(0)
    dims=tl.arange(0,4096)
    norm_weight=tl.load(NW+dims)
    psplit=tl.load(ROUTES.to(tl.pointer_type(tl.int32))+5)
    acc=tl.full((4096,),0,tl.float32)
    for slot in range(8):
        expert=tl.load(IDS+token*8+slot)-LOCAL_START
        if (expert>=0) & (expert<32):
            offset=tl.load((ROUTES+192).to(tl.pointer_type(tl.int32))+expert)
            pair=offset+tl.load(SPOS+token*8+slot)
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


def _native_expert_kernel(data, state):
    cfg = data["config"]
    x = data["hidden_states"].contiguous()
    M = x.shape[0]
    prenorm = None
    if data.get("prenorm"):
        # raw pooler rows: the gathers apply the pre-expert norm; configs without a gather (and
        # the PyTorch / one-token paths) normalize first
        if state["prenorm"] is not None and M <= MAX_M and M > 1 and _kernel_cfg(M)[1] >= GATHER_MIN_NT:
            prenorm = state["prenorm"]
        else:
            x = F.rms_norm(x, (x.shape[1],), state["prenorm"], 1e-5).to(x.dtype)
            data = dict(data, hidden_states=x)
    if M > MAX_M or cfg["local_start"] != state["start"] or cfg["local_count"] != state["count"]:
        if data.get("bucketed"):
            state["cur"].zero_()
        return _torch_path(data, state)
    ids = data["topk_ids"].to(torch.int32).contiguous()
    tw = data["topk_weights"].to(torch.float32).contiguous()
    if M == 1 and ONE_TOKEN:
        if data.get("bucketed"):
            state["cur"].zero_()
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
    ypart = torch.empty((max(ks2, _ksx(rw, nt, ks2)), pmax, H), dtype=torch.float32, device=dev)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    ssq = torch.empty(pmax * 64, dtype=torch.float32, device=dev)
    yh = torch.empty((pmax, H), dtype=torch.bfloat16, device=dev)
    routes = torch.empty(512, dtype=torch.uint8, device=dev)  # the expert kernel's tables for the finish
    p = _ParamsC(x.data_ptr(), ids.data_ptr(), tw.data_ptr(), M, state["start"],
                 state["w13"].data_ptr(), state["s13"].data_ptr(), state["gs13"].data_ptr(),
                 state["w2"].data_ptr(), state["s2"].data_ptr(), state["gs2"].data_ptr(), state["nw1"].data_ptr(),
                 inter.data_ptr(), ypart.data_ptr(), out.data_ptr(), state["ctr"].data_ptr(), pmax, xs.data_ptr(),
                 ssq.data_ptr(), state["tmaps"].data_ptr() + state["tmap_off"][rw], yh.data_ptr(),
                 routes.data_ptr(), state["sms"], prenorm.data_ptr() if prenorm is not None else None,
                 state["cur"].data_ptr(), state["bkt"].data_ptr(), state["spos"].data_ptr())
    if not data.get("bucketed"):  # (the layer's router kernel buckets as it routes)
        _module().launch("moe_bucket", (M + 255) // 256, 256, 0, p, pdl=True)
    if nt >= 4 and mw >= 128:
        _module().launch(_name(kc).replace("moe_", "gather_", 1), state["sms"] * 4, 256, 0, p, pdl=True)
    _module().launch(_name(kc), state["sms"], 384, _smem_bytes(rw, nt, ns, mw), p, pdl=True)
    if COMBINE_IN:  # the expert kernel combined the tokens itself
        return out
    if NATIVE_FINISH_CUDA:
        fp = _FinParamsC(ids.data_ptr(), tw.data_ptr(), state["nw1"].data_ptr(), routes.data_ptr(), yh.data_ptr(),
                         ypart.data_ptr(), out.data_ptr(), M, mw, pmax, ks2, _ksx(rw, nt, ks2), state["start"],
                         state["spos"].data_ptr())
        _module().launch("moe_finish", M, 128, 0, fp, pdl=True)
    else:
        pdlf = PDL and _gdc_wait is not None
        _native_finish[(M,)](ids, tw, state["nw1"], routes, yh, ypart, out, state["spos"],
                              mw, pmax, ks2, _ksx(rw, nt, ks2), state["start"], pdlf, num_warps=8,
                              **({"launch_pdl": True} if pdlf else {}))
    return out

"""Native packed experts with per-call global routing and up to 16384 tokens per chunk."""


def _native_prepare_without_lt(weights):
    native_weights = {name: weights[name].clone()
                      for name in ('w13', 'w13_sf', 'w13_gs', 'w2', 'w2_sf', 'w2_gs')}
    native_weights['post_norm_w'] = weights['post_expert_norm'].clone()
    native_weights['pre_norm_w'] = weights['pre_expert_norm'].clone()
    native = _native_expert_prepare(native_weights)
    state = _sglang_prepare(weights)
    state['native_experts'] = native
    return state


def _experts(x, topk_weights, topk_ids, state, prenorm=False, bucketed=False):
    """prenorm: x is the raw pooler output; the pre-expert norm runs inside the expert path.
    bucketed: the router kernel already bucketed the local routes (one chunk)."""
    outputs = []
    for first in range(0, x.shape[0], MAX_M):
        end = min(first + MAX_M, x.shape[0])
        data = {'hidden_states': x[first:end], 'topk_ids': topk_ids[first:end],
                'topk_weights': topk_weights[first:end], 'prenorm': prenorm, 'bucketed': bucketed,
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

"""Dense path: measured cuBLASLt algorithms per scored shape, and for up to 4096 tokens the router
folded into the pooler GEMM (one FP32-output GEMM over [pooler; router_hi; router_lo]) with the
pre-expert RMSNorm and the router's hi + lo sum in one pass over its output."""

import ctypes as _ct


class _DAlgo(_ct.Structure):
    _fields_ = [("data", _ct.c_uint64 * 8)]


class _DHeur(_ct.Structure):
    _fields_ = [("algo", _DAlgo), ("ws", _ct.c_size_t), ("state", _ct.c_int), ("waves", _ct.c_float), ("r", _ct.c_int * 4)]


class DenseLt:
    """out[m, n] (+)= x[m, k] @ W^T (W [n, k] BF16), FP32 accumulation, BF16 or FP32 output."""

    def __init__(self, library, m, n, k, workspace, out32=False, beta=0.0, count=64):
        lib = library.lib
        self.library, self.ws, self.m, self.n, self.k, self.out32 = library, workspace, m, n, k, out32
        self.desc, self.a, self.b, self.c, self.pref = [_ct.c_void_p() for _ in range(5)]
        library.check(lib.cublasLtMatmulDescCreate(_ct.byref(self.desc), 68, 0), "descriptor")
        ta = _ct.c_int(1)
        lib.cublasLtMatmulDescSetAttribute(self.desc, 3, _ct.byref(ta), _ct.sizeof(ta))
        library.check(lib.cublasLtMatrixLayoutCreate(_ct.byref(self.a), 14, k, n, k), "A layout")
        library.check(lib.cublasLtMatrixLayoutCreate(_ct.byref(self.b), 14, k, m, k), "B layout")
        library.check(lib.cublasLtMatrixLayoutCreate(_ct.byref(self.c), 0 if out32 else 14, n, m, n), "C layout")
        lib.cublasLtMatmulPreferenceCreate(_ct.byref(self.pref))
        size = _ct.c_size_t(workspace.numel())
        lib.cublasLtMatmulPreferenceSetAttribute(self.pref, 1, _ct.byref(size), _ct.sizeof(size))
        red = _ct.c_uint32(2)  # split reductions keep FP32 intermediates
        lib.cublasLtMatmulPreferenceSetAttribute(self.pref, 3, _ct.byref(red), _ct.sizeof(red))
        res = (_DHeur * count)()
        got = _ct.c_int()
        library.check(lib.cublasLtMatmulAlgoGetHeuristic(library.handle, self.desc, self.a, self.b, self.c, self.c,
                                                        self.pref, count, res, _ct.byref(got)), "heuristics")
        self.algorithms, self.info = [], []
        for j in range(got.value):
            r = res[j]
            if r.state != 0 or r.ws > workspace.numel():
                continue
            algo = _DAlgo.from_buffer_copy(r.algo)
            attrs = {}
            for key, attr in [("id", 0), ("tile", 1), ("split_k", 2), ("reduction", 3), ("swizzle", 4), ("custom", 5), ("stages", 6)]:
                v, w_ = _ct.c_int(), _ct.c_size_t()
                st = lib.cublasLtMatmulAlgoConfigGetAttribute(_ct.byref(algo), attr, _ct.byref(v), _ct.sizeof(v), _ct.byref(w_))
                attrs[key] = v.value if st == 0 else None
            if attrs["reduction"] not in (0, 2):
                continue
            self.algorithms.append(algo)
            self.info.append(attrs)
        if not self.algorithms:
            raise RuntimeError("no cuBLASLt algorithm")
        self.alpha, self.beta = _ct.c_float(1.0), _ct.c_float(beta)

    def find(self, tile, custom):
        for i, a in enumerate(self.info):
            if a["id"] == 66 and a["tile"] == tile and a["custom"] == custom and a["split_k"] == 1:
                return i
        return None

    def run(self, x, weight, index=0, out=None):
        if out is None:
            out = torch.empty((self.m, self.n), device=x.device, dtype=torch.float32 if self.out32 else torch.bfloat16)
        st = self.library.lib.cublasLtMatmul(
            self.library.handle, self.desc, _ct.byref(self.alpha), weight.data_ptr(), self.a, x.data_ptr(), self.b,
            _ct.byref(self.beta), out.data_ptr(), self.c, out.data_ptr(), self.c, _ct.byref(self.algorithms[index]),
            self.ws.data_ptr(), self.ws.numel(), torch.cuda.current_stream(x.device).cuda_stream)
        self.library.check(st, "matmul")
        return out


@triton.jit
def _pooled_norm_route(Y, W, P, LOG, NP: tl.constexpr, NR: tl.constexpr, LD: tl.constexpr, EPS: tl.constexpr,
                       BLOCK: tl.constexpr):
    """Per row of Y [M, NP + 2 NR] FP32 (pooler | router hi | router lo): P = rmsnorm(Y[:, :NP]) * W
    in BF16 and LOG = hi + lo."""
    row = tl.program_id(0).to(tl.int64)
    base = Y + row * LD
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for c in range(0, NP, BLOCK):
        y = tl.load(base + c + offs)
        acc += y * y
    r = tl.rsqrt(tl.sum(acc, 0) / NP + EPS)
    for c in range(0, NP, BLOCK):
        y = tl.load(base + c + offs)
        w = tl.load(W + c + offs)
        tl.store(P + row * NP + c + offs, (y * r * w).to(tl.bfloat16))
    ro = tl.arange(0, NR)
    tl.store(LOG + row * NR + ro, tl.load(base + NP + ro) + tl.load(base + NP + NR + ro))


# (tokens, layer) -> (tile, custom) of the measured fastest cuBLASLt algorithm (H200, cuBLASLt 13.1)
DENSE_ALGOS = {
    (16384, "o"): (201, 2),
    (16384, "qkv_gate"): (24, 7),
    (3012, "qkv_gate"): (496, 7),
}
FUSED_ROUTER_MAX = 4096           # tokens up to which the router rides in the pooler GEMM
FUSED_ALGOS = {753: (497, 5), 3012: (500, 7)}
ROUTER_ALGOS = {16384: (23, 4)}   # the separate router: two N = 256 GEMMs, the second accumulating
ROUTER512_ALGOS = {16384: (24, 7)}  # large prefill: one N = 512 GEMM over [hi; lo], then pnr2

_codex_prepare = prepare


def prepare(weights):
    state = _codex_prepare(weights)
    library, workspace = state["prefill_lt_library"], state["prefill_lt_workspace"]
    dense = state["dense"]
    for (m, name), (tile, custom) in DENSE_ALGOS.items():
        w = dense[name]
        n, k = w.shape
        plan = DenseLt(library, m, n, k, workspace)
        i = plan.find(tile, custom)
        if i is not None:
            _PREFILL_LT_PLANS[(m, n, k, w.data_ptr())] = (plan, i)
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi"], state["router_lo"] = hi.contiguous(), lo.contiguous()
    state["pool_route_w"] = torch.cat([dense["pooler"], hi, lo], 0).contiguous()   # [4608, 8192], lossless
    state["pre_expert_w32"] = (1.0 + weights["pre_expert_norm"].float()).contiguous()
    state["fused_plans"] = {}
    state["router_plans"] = {}
    state["router_hilo"] = torch.cat([hi, lo], 0).contiguous()   # [512, 8192]
    state["router512_plans"] = {}
    for m, (tile, custom) in ROUTER512_ALGOS.items():
        plan = DenseLt(library, m, 512, HIDDEN, workspace, out32=True)
        i = plan.find(tile, custom)
        if i is not None:
            state["router512_plans"][m] = (plan, i)
    for m, (tile, custom) in ROUTER_ALGOS.items():
        p1 = DenseLt(library, m, 256, HIDDEN, workspace, out32=True)
        p2 = DenseLt(library, m, 256, HIDDEN, workspace, out32=True, beta=1.0)
        i1, i2 = p1.find(tile, custom), p2.find(tile, custom)
        if i1 is not None and i2 is not None:
            state["router_plans"][m] = (p1, i1, p2, i2)
    for m in FUSED_ALGOS:
        _fused_plan(state, m)
    # compile the norm pass once, outside any capture
    y = torch.zeros((2, D_EXPERT + 512), dtype=torch.float32, device=router.device)
    pn = torch.empty((2, D_EXPERT), dtype=torch.bfloat16, device=router.device)
    lg = torch.empty((2, 256), dtype=torch.float32, device=router.device)
    _pooled_norm_route[(2,)](y, state["pre_expert_w32"], pn, lg, D_EXPERT, 256, D_EXPERT + 512, RMS_EPS, 1024, num_warps=8)
    torch.cuda.synchronize()
    return state


def _fused_plan(state, m):
    if m not in state["fused_plans"]:
        plan = DenseLt(state["prefill_lt_library"], m, D_EXPERT + 512, HIDDEN, state["prefill_lt_workspace"], out32=True)
        i = plan.find(*FUSED_ALGOS[m]) if m in FUSED_ALGOS else None
        state["fused_plans"][m] = (plan, 0 if i is None else i)
    return state["fused_plans"][m]


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: SGLang contributors <https://github.com/sgl-project/sglang>
# SPDX-License-Identifier: Apache-2.0
# Adapted from the OKC starter's muse_prefill_layer/tools/baseline.py (SGLang's MuseSpark1xDecoderLayer, sgl-project/sglang @ a55ab7d), `kernel_fn`.
# Changes: runs our attention module instead of the QK norm, cache store, FA3 call and output norm gate; everything runs on the main stream (no side stream): small split-router batches use one fused pooler + router GEMM with a fused pre-expert norm and logit kernel, larger ones cuBLASLt router plans or `_layer_router`.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    dense = state["dense"]
    # Attention: MuseSpark1xAttention.forward.
    use_att = not cfg["window"] or cfg["tokens"] >= ATT_SLIDING_MIN
    if use_att:
        state["att"].fork(data)
    qkv = _gemm(data["attn_input"], dense["qkv_gate"])
    if use_att:
        state["att"].pre(data)  # cached-key packing beside the projection
    o = (state["att"](data, qkv) if (not cfg["window"] or tokens >= ATT_SLIDING_MIN) else optimized_attention(data, qkv, custom_window=CUSTOM_WINDOW).reshape(tokens, hq * d))
    attn_output = _gemm(o, dense["o"])  # all-reduce: identity

    # Post-attention boundary, router and MoE block: MuseSpark1xMoE._forward_experts.
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
    if split_router and tokens <= FUSED_ROUTER_MAX:
        # one GEMM: pooler and router (hi, lo) columns, FP32 out; then the pre-expert norm and hi + lo
        plan, index = _fused_plan(state, tokens)
        y = plan.run(ffn_input.contiguous(), state["pool_route_w"], index)
        expert_input = torch.empty((tokens, D_EXPERT), dtype=torch.bfloat16, device=y.device)
        logits = torch.empty((tokens, 256), dtype=torch.float32, device=y.device)
        _pooled_norm_route[(tokens,)](y, state["pre_expert_w32"], expert_input, logits, D_EXPERT, 256,
                                      D_EXPERT + 512, RMS_EPS, 1024, num_warps=8)
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    else:
        expert_input = _gemm(ffn_input, dense["pooler"])
        if split_router and tokens in state["router_plans"]:
            p1, i1, p2, i2 = state["router_plans"][tokens]
            x = ffn_input.contiguous()
            logits = p2.run(x, state["router_lo"], i2, p1.run(x, state["router_hi"], i1))
        elif split_router:
            logits = _layer_router(ffn_input, state)
        else:
            logits = F.linear(router_input, state["router"])
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
        expert_input = state["pre_expert_norm"](expert_input)
    # This GPU's experts start at local_start: as SGLang's expert_map on that EP rank.
    local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
    # all-reduce: identity
    expert_output = _experts(expert_input, topk_weights, local_ids, state)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = _gemm(expert_output, dense["unpooler"])

    # Post-FFN boundary, with the next layer's pre-attention norm.
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
# SPDX-SnippetEnd


"""dense2:
* residual boundaries without the intermediate stream: boundary 1 writes only the FFN input h; boundary 2
  recomputes stream1 = al1 * s + be1 * bf16(rms(a)) from s and a (fewer bytes than storing and reloading it),
  adds the FFN branch, s2 = (al2 al1) s + (al2 be1) bf16(rms(a)) + be2 bf16(rms(bf16(f (1 + w)))), and applies
  the next pre-attention norm. One CTA of 512 threads per row, contiguous warp accesses, streaming loads
  that skip L1 and per-channel vectors kept in L1. Rounding points are FusedPostNormResidualPreNorm's, with
  the (1 + w) weights in FP32.
* up to 4096 tokens, the pre-expert norm of the fused pooler GEMM's output and the routing (sigmoid,
  biased top 8, renormalization: FusedMuseSparkRouting's math) with the local expert ids in one kernel."""

BND3_SRC = r"""
typedef unsigned int u32;
#define DEVI __device__ __forceinline__
DEVI float bf_lo(u32 x) { return __int_as_float(x << 16); }
DEVI float bf_hi(u32 x) { return __int_as_float(x & 0xFFFF0000u); }
DEVI u32 pack_bf(float lo, float hi) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
DEVI float rbf(float x) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(r) : "f"(x)); return __int_as_float(r << 16); }

struct Bnd3P {
  const float* s; const u32* a; const u32* f;
  float* sout; u32* hout;
  const float* v0; const float* v1; const float* v2;   // b1: al1, be1;  b2: A, Bv, C;  s2: al2, be2
  const u32* w0; const u32* w1;                         // b1: wpre;  b2/s2: wpost, wnext (BF16 w)
  const float* v3; const float* v4;                     // b2 only: al1 (unused), be1 (unused)
};

DEVI uint4 ld_stream(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
DEVI uint4 ld_keep(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::evict_last.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
DEVI void st_stream(void* p, uint4 v) {
  asm volatile("st.global.L1::no_allocate.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
DEVI void unpack8(uint4 x, float* v) {
  v[0] = bf_lo(x.x); v[1] = bf_hi(x.x); v[2] = bf_lo(x.y); v[3] = bf_hi(x.y);
  v[4] = bf_lo(x.z); v[5] = bf_hi(x.z); v[6] = bf_lo(x.w); v[7] = bf_hi(x.w);
}
DEVI uint2 ld_stream8(const void* p) {
  uint2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}
DEVI uint2 ld_keep8(const void* p) {
  uint2 r;
  asm volatile("ld.global.nc.L1::evict_last.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}
DEVI void st_stream8(void* p, uint2 v) {
  asm volatile("st.global.L1::no_allocate.v2.u32 [%0], {%1,%2};" :: "l"(p), "r"(v.x), "r"(v.y) : "memory");
}
// thread t owns columns 4t + 2048k, k = 0..3: every warp access is contiguous
DEVI void ldf16(const float* p, int t, float* v, bool keep) {
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const uint4 x = keep ? ld_keep(p + 2048 * k + 4 * t) : ld_stream(p + 2048 * k + 4 * t);
    v[4 * k] = __int_as_float(x.x); v[4 * k + 1] = __int_as_float(x.y); v[4 * k + 2] = __int_as_float(x.z); v[4 * k + 3] = __int_as_float(x.w);
  }
}
DEVI void ldb16(const u32* p, int t, float* v, bool keep) {
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const uint2 x = keep ? ld_keep8(p + 1024 * k + 2 * t) : ld_stream8(p + 1024 * k + 2 * t);
    v[4 * k] = bf_lo(x.x); v[4 * k + 1] = bf_hi(x.x); v[4 * k + 2] = bf_lo(x.y); v[4 * k + 3] = bf_hi(x.y);
  }
}
DEVI void stf16(float* p, int t, const float* v) {
#pragma unroll
  for (int k = 0; k < 4; ++k)
    st_stream(p + 2048 * k + 4 * t, make_uint4(__float_as_int(v[4 * k]), __float_as_int(v[4 * k + 1]), __float_as_int(v[4 * k + 2]), __float_as_int(v[4 * k + 3])));
}
DEVI void stb16(u32* p, int t, const float* v) {
#pragma unroll
  for (int k = 0; k < 4; ++k)
    st_stream8(p + 1024 * k + 2 * t, make_uint2(pack_bf(v[4 * k], v[4 * k + 1]), pack_bf(v[4 * k + 2], v[4 * k + 3])));
}
DEVI float2 csum2(float x, float y, float2* red) {
#pragma unroll
  for (int o = 16; o; o >>= 1) { x += __shfl_xor_sync(0xffffffff, x, o); y += __shfl_xor_sync(0xffffffff, y, o); }
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0) red[w] = make_float2(x, y);
  __syncthreads();
  float2 v = l < 16 ? red[l] : make_float2(0.f, 0.f);
#pragma unroll
  for (int o = 16; o; o >>= 1) { v.x += __shfl_xor_sync(0xffffffff, v.x, o); v.y += __shfl_xor_sync(0xffffffff, v.y, o); }
  return v;
}

// mode 0: b1 (h only), 1: b2 recompute, 2: b1 storing s1, 3: b2 from stored s1
template <int MODE>
DEVI void body(const Bnd3P& p) {
  asm volatile("griddepcontrol.wait;" ::: "memory");  // PDL: the previous grid is complete
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  __shared__ float2 red[2][16];
  const int t = threadIdx.x;
  const size_t row = blockIdx.x;
  float s[16], a[16], f[16];
  ldf16(p.s + row * 8192, t, s, false);
  if (MODE != 3) ldb16(p.a + row * 4096, t, a, false);
  if (MODE == 1 || MODE == 3) ldb16(p.f + row * 4096, t, f, false);
  float y[16];
  float q = 0.f, qf = 0.f;
  if (MODE != 3) {
#pragma unroll
    for (int i = 0; i < 16; ++i) q += a[i] * a[i];
  }
  if (MODE == 1 || MODE == 3) {
    float w[16];
    ldb16(p.w0, t, w, true);
#pragma unroll
    for (int i = 0; i < 16; ++i) { f[i] = rbf(f[i] * (1.f + w[i])); qf += f[i] * f[i]; }
  }
  const float2 tot = csum2(q, qf, red[0]);
  const float ra = rsqrtf(tot.x * (1.f / 8192.f) + 1e-8f), rf = rsqrtf(tot.y * (1.f / 8192.f) + 1e-8f);
  {
    float v0[16], v1[16];
    ldf16(p.v0, t, v0, true);
    ldf16(p.v1, t, v1, true);
    if (MODE == 0 || MODE == 2) {
#pragma unroll
      for (int i = 0; i < 16; ++i) y[i] = v0[i] * s[i] + v1[i] * rbf(a[i] * ra);
    } else if (MODE == 1) {
      float v2[16];
      ldf16(p.v2, t, v2, true);
#pragma unroll
      for (int i = 0; i < 16; ++i) y[i] = v0[i] * s[i] + v1[i] * rbf(a[i] * ra) + v2[i] * rbf(f[i] * rf);
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) y[i] = v0[i] * s[i] + v1[i] * rbf(f[i] * rf);
    }
  }
  if (MODE != 0) stf16(p.sout + row * 8192, t, y);
  float q2 = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) q2 += y[i] * y[i];
  const float r2 = rsqrtf(csum2(q2, 0.f, red[1]).x * (1.f / 8192.f) + 1e-5f);
  float w[16];
  ldb16(MODE == 0 || MODE == 2 ? p.w0 : p.w1, t, w, true);
#pragma unroll
  for (int i = 0; i < 16; ++i) y[i] = y[i] * r2 * (1.f + w[i]);
  stb16(p.hout + row * 4096, t, y);
}

extern "C" __global__ void __launch_bounds__(512, 2) bnd3_b1(Bnd3P p) { body<0>(p); }
extern "C" __global__ void __launch_bounds__(512, 2) bnd3_b2(Bnd3P p) { body<1>(p); }
extern "C" __global__ void __launch_bounds__(512, 2) bnd3_s1(Bnd3P p) { body<2>(p); }
extern "C" __global__ void __launch_bounds__(512, 2) bnd3_s2(Bnd3P p) { body<3>(p); }
"""

ROUTE_SRC = r"""
// (u32, DEVI and pack_bf come from BND3_SRC, compiled first)

struct RouteP {
  const float* y; int ld;               // pnr: fused GEMM output (ld = 4608); route: logits (ld = 256)
  const float* w;                       // pnr: pre-expert (1 + w) FP32 [4096]
  unsigned short* p;                    // pnr: normalized expert input BF16 [T, 4096]
  const float* bias;                    // [256]
  float* tw; int* ti; int* li;          // [T, 8]
  const int* local_start;
  float mult, eps;
  int rows;
  int* cur; unsigned short* bkt; int* spos;  // non-null: bucket the local routes (expert kernel input)
  const unsigned short* xb;             // pnr2: pooled BF16 rows [T, 4096] (y: the hi | lo logits, ld = 512)
};

// top-8 of one row's 256 logits (lane holds lg[8 lane .. 8 lane + 7]); lanes 0..7 store
DEVI void topk8(const RouteP& p, size_t row, const float (&lg)[8], int lane) {
  float sc[8], st[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    sc[i] = 1.f / (1.f + expf(-lg[i] * p.mult));
    st[i] = sc[i] + p.bias[8 * lane + i];
  }
  u32 taken = 0;
  int my_id = 0;
  float my_w = 0.f;
  float sum = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    float bv = __int_as_float(0xff800000u);
    int bi = 1 << 30;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      if (!((taken >> i) & 1) && st[i] > bv) { bv = st[i]; bi = 8 * lane + i; }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffff, bv, o);
      const int oi = __shfl_xor_sync(0xffffffff, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    float w = 0.f;
    if ((bi >> 3) == lane) {
      const int k = bi & 7;
      taken |= 1u << k;
#pragma unroll
      for (int i = 0; i < 8; ++i) if (i == k) w = sc[i];
    }
    w = __shfl_sync(0xffffffff, w, bi >> 3);
    sum += w;
    if (lane == j) { my_id = bi; my_w = w; }
  }
  if (lane < 8) {
    const float inv = 1.f / (sum + 1e-15f);
    p.tw[row * 8 + lane] = my_w * inv;
    p.ti[row * 8 + lane] = my_id;
    p.li[row * 8 + lane] = (my_id - p.local_start[0]) & 255;
    if (p.cur) {
      const int e = my_id - p.local_start[0];
      int pos = -1;
      if ((unsigned)e < 32u) {
        pos = atomicAdd(p.cur + e, 1);
        p.bkt[(size_t)e * p.rows + pos] = (unsigned short)row;
      }
      p.spos[row * 8 + lane] = pos;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) pnr(RouteP p) {
  asm volatile("griddepcontrol.wait;" ::: "memory");  // PDL: the previous grid is complete
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  __shared__ float red[8];
  // the expert path's first kernel (PDL) may start its prologue; it waits for this grid before reading
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const size_t row = blockIdx.x;
  const int t = threadIdx.x, lane = t & 31, wp = t >> 5;
  const float* y = p.y + row * p.ld;
  float v[16];
  float q = 0.f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float4 x = *reinterpret_cast<const float4*>(y + 1024 * k + 4 * t);
    v[4 * k] = x.x; v[4 * k + 1] = x.y; v[4 * k + 2] = x.z; v[4 * k + 3] = x.w;
    q += x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
  }
  float lg[8];
  if (wp == 0) {
    const float4 h0 = *reinterpret_cast<const float4*>(y + 4096 + 8 * lane), h1 = *reinterpret_cast<const float4*>(y + 4096 + 8 * lane + 4);
    const float4 l0 = *reinterpret_cast<const float4*>(y + 4352 + 8 * lane), l1 = *reinterpret_cast<const float4*>(y + 4352 + 8 * lane + 4);
    lg[0] = h0.x + l0.x; lg[1] = h0.y + l0.y; lg[2] = h0.z + l0.z; lg[3] = h0.w + l0.w;
    lg[4] = h1.x + l1.x; lg[5] = h1.y + l1.y; lg[6] = h1.z + l1.z; lg[7] = h1.w + l1.w;
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffff, q, o);
  if (lane == 0) red[wp] = q;
  __syncthreads();
  q = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) q += red[i];
  const float r = rsqrtf(q * (1.f / 4096.f) + p.eps);
  u32* out = reinterpret_cast<u32*>(p.p + row * 4096);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float4 w = *reinterpret_cast<const float4*>(p.w + 1024 * k + 4 * t);
    *reinterpret_cast<uint2*>(out + 512 * k + 2 * t) =
        make_uint2(pack_bf(v[4 * k] * r * w.x, v[4 * k + 1] * r * w.y), pack_bf(v[4 * k + 2] * r * w.z, v[4 * k + 3] * r * w.w));
  }
  if (wp == 0) topk8(p, row, lg, lane);
}

// Large prefill: per token, the pooled BF16 row's pre-expert norm into p.p and top-8 routing of the
// router GEMM's FP32 hi | lo logits (as F.rms_norm on the BF16 pooled output, then the routing).
extern "C" __global__ void __launch_bounds__(256) pnr2(RouteP p) {
  __shared__ float red[8];
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const size_t row = blockIdx.x;
  const int t = threadIdx.x, lane = t & 31, wp = t >> 5;
  const unsigned short* xr = p.xb + row * 4096;
  float v[16];
  float q = 0.f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const uint2 u = *reinterpret_cast<const uint2*>(xr + 1024 * k + 4 * t);
    v[4 * k] = __uint_as_float(u.x << 16); v[4 * k + 1] = __uint_as_float(u.x & 0xFFFF0000u);
    v[4 * k + 2] = __uint_as_float(u.y << 16); v[4 * k + 3] = __uint_as_float(u.y & 0xFFFF0000u);
#pragma unroll
    for (int i = 0; i < 4; ++i) q += v[4 * k + i] * v[4 * k + i];
  }
  float lg[8];
  if (wp == 0) {
    const float* y = p.y + row * p.ld;
    const float4 h0 = *reinterpret_cast<const float4*>(y + 8 * lane), h1 = *reinterpret_cast<const float4*>(y + 8 * lane + 4);
    const float4 l0 = *reinterpret_cast<const float4*>(y + 256 + 8 * lane), l1 = *reinterpret_cast<const float4*>(y + 256 + 8 * lane + 4);
    lg[0] = h0.x + l0.x; lg[1] = h0.y + l0.y; lg[2] = h0.z + l0.z; lg[3] = h0.w + l0.w;
    lg[4] = h1.x + l1.x; lg[5] = h1.y + l1.y; lg[6] = h1.z + l1.z; lg[7] = h1.w + l1.w;
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffff, q, o);
  if (lane == 0) red[wp] = q;
  __syncthreads();
  q = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) q += red[i];
  const float r = rsqrtf(q * (1.f / 4096.f) + p.eps);
  u32* out = reinterpret_cast<u32*>(p.p + row * 4096);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float4 w = *reinterpret_cast<const float4*>(p.w + 1024 * k + 4 * t);
    *reinterpret_cast<uint2*>(out + 512 * k + 2 * t) =
        make_uint2(pack_bf(v[4 * k] * r * w.x, v[4 * k + 1] * r * w.y), pack_bf(v[4 * k + 2] * r * w.z, v[4 * k + 3] * r * w.w));
  }
  if (wp == 0) topk8(p, row, lg, lane);
}

extern "C" __global__ void __launch_bounds__(256) route(RouteP p) {
  const int lane = threadIdx.x & 31;
  const size_t row = (size_t)blockIdx.x * 8 + (threadIdx.x >> 5);
  if (row >= (size_t)p.rows) return;
  const float* y = p.y + row * p.ld;
  const float4 a = *reinterpret_cast<const float4*>(y + 8 * lane), b = *reinterpret_cast<const float4*>(y + 8 * lane + 4);
  const float lg[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  topk8(p, row, lg, lane);
}
"""


class _Bnd3P(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in ("s", "a", "f", "sout", "hout", "v0", "v1", "v2", "w0", "w1", "v3", "v4")]


class _RouteP(ctypes.Structure):
    _fields_ = [("y", ctypes.c_void_p), ("ld", ctypes.c_int), ("w", ctypes.c_void_p), ("p", ctypes.c_void_p),
                ("bias", ctypes.c_void_p), ("tw", ctypes.c_void_p), ("ti", ctypes.c_void_p), ("li", ctypes.c_void_p),
                ("local_start", ctypes.c_void_p), ("mult", ctypes.c_float), ("eps", ctypes.c_float), ("rows", ctypes.c_int),
                ("cur", ctypes.c_void_p), ("bkt", ctypes.c_void_p), ("spos", ctypes.c_void_p), ("xb", ctypes.c_void_p)]


ROUTER_MULT = 0.1767766952966369
_dense1_prepare = prepare


def prepare(weights):
    state = _dense1_prepare(weights)

    # Follows the OKC problem's reference implementation (muse_prefill_layer/tools/reference.py, `residual_coefficients`).
    def coeff(g):
        w = g.float() / 0.3
        beta = torch.sigmoid(w)
        return torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3)).contiguous(), beta.contiguous()
    al1, be1 = coeff(weights["attn_residual_gate"])
    al2, be2 = coeff(weights["ffn_residual_gate"])
    state["bnd"] = dict(al1=al1, be1=be1, A=(al2 * al1).contiguous(), Bv=(al2 * be1).contiguous(), C=be2,
                        wpre=weights["pre_ffn_norm"].contiguous(), wpost=weights["post_ffn_norm"].contiguous(),
                        wnext=weights["next_pre_attn_norm"].contiguous())
    state["bnd_mod"] = _Cuda(BND3_SRC + ROUTE_SRC)
    state["router_bias32"] = weights["router_bias"].float().contiguous()
    return state


def _bnd1(state, s, a):
    bv = state["bnd"]
    h = torch.empty(a.shape, dtype=torch.bfloat16, device=a.device)
    prm = _Bnd3P(s.data_ptr(), a.data_ptr(), 0, 0, h.data_ptr(), bv["al1"].data_ptr(), bv["be1"].data_ptr(), 0,
                 bv["wpre"].data_ptr(), bv["wpre"].data_ptr(), 0, 0)
    state["bnd_mod"].launch("bnd3_b1", a.shape[0], 512, 0, prm, pdl=True)
    return h


def _bnd2(state, s, a, f):
    bv = state["bnd"]
    out = torch.empty(a.shape, dtype=torch.bfloat16, device=a.device)
    s2 = torch.empty(s.shape, dtype=torch.float32, device=a.device)
    prm = _Bnd3P(s.data_ptr(), a.data_ptr(), f.data_ptr(), s2.data_ptr(), out.data_ptr(), bv["A"].data_ptr(),
                 bv["Bv"].data_ptr(), bv["C"].data_ptr(), bv["wpost"].data_ptr(), bv["wnext"].data_ptr(), 0, 0)
    state["bnd_mod"].launch("bnd3_b2", a.shape[0], 512, 0, prm, pdl=True)
    return out, s2





ATT_HD, ATT_HQ, ATT_HKV, ATT_GRP = 64, 16, 2, 8
ATT_QKV_N = 2 * ATT_HQ * ATT_HD + 2 * ATT_HKV * ATT_HD
ATT_KT = 128          # keys per tile
ATT_QB_DEF = 16           # tokens per CTA block (8 per consumer warpgroup; set from ATT_CW)
ATT_NS_DEF = 8            # ring stages
ATT_LOG2E = 1.4426950408889634

ATT_CUDA_SRC = r"""
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned long long u64;
typedef unsigned char u8;
#define DEVI __device__ __forceinline__
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifndef ATT_NS
#define ATT_NS 8
#endif
#ifndef ATT_POLY
#define ATT_POLY 1
#endif
#ifndef ATT_CW
#define ATT_CW 2            // consumer warpgroups: 8 tokens (64 rows) each
#endif
#define ATT_QB (8 * ATT_CW)  // tokens per CTA block
#ifndef ATT_PP
#define ATT_PP 0            // ping-pong: consumer warpgroups issue their wgmmas in turn
#endif
#ifndef ATT_F16S
#define ATT_F16S 0          // QK accumulates in f16 (32 registers per S tile); softmax on f16x2 pairs
#endif
#ifndef ATT_POLYH
#define ATT_POLYH 0         // with ATT_F16S: n8 blocks (of 16) whose exp2 runs as an f16x2 polynomial
#endif
#ifndef ATT_EXPQ
#define ATT_EXPQ 0          // with ATT_F16S: P's FP8 code straight from the f16 score (Schraudolph:
                            // code = round(8 (S - m) + 56), linear mantissa), no MUFU
#endif
#ifndef ATT_LAZY
#define ATT_LAZY 0          // exp with the previous max at once; the tile max (from P) only gates a rare rescale
#endif
#ifndef ATT_SPLIT
#define ATT_SPLIT 0         // QK in two 64-key halves: the first half's softmax runs under the second's QK
#endif
#ifndef ATT_BIAS
#define ATT_BIAS 0          // subtract the running max on the tensor core (a third QK k-step)
#endif
#if ATT_CW == 2
#define ATT_CREG 240
#elif ATT_CW == 4
#define ATT_CREG 112        // 640 threads launch at 96 registers: 4 x 128 x 112 + 128 x 24 fits that pool
#else
#define ATT_CREG 160
#endif

DEVI u32 smem_addr(const void* p) {
  u32 r;
  asm volatile("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(r) : "l"(p));
  return r;
}
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
// the spin loop lives inside the asm: no divergent control flow around the wgmmas
DEVI void mbar_wait(u32 bar, u32 parity) {
  asm volatile("{ .reg .pred P1; WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1; @P1 bra DONE_%=; bra WAIT_%=; DONE_%=: }"
               :: "r"(bar), "r"(parity) : "memory");
}
// arrive by one thread of the warpgroup, without a branch
DEVI void mbar_arrive_if(u32 bar, int pred) {
  asm volatile("{ .reg .pred P1; setp.ne.b32 P1, %1, 0; @P1 mbarrier.arrive.shared::cta.b64 _, [%0]; }"
               :: "r"(bar), "r"(pred) : "memory");
}
DEVI void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
DEVI void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
DEVI void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> DEVI void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
DEVI void named_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }

// K tile: [128 keys][64 B], 64-byte swizzle (16 B chunk c of row r at c ^ ((r >> 1) & 3)), 8-row groups 512 B apart
DEVI u64 desc_k(u32 saddr) {
  return (u64)((saddr & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)(512 >> 4) << 32) | ((u64)2 << 62);
}
// V^T tile: [72 rows (64 d + 8 ones)][128 B = 128 keys, permuted], 128-byte swizzle, 8-row groups 1024 B apart
// bias B tile: [128 keys][32 B], bytes 0, 1 = 1.0, 32-byte swizzle, 8-row groups 256 B apart
DEVI u64 desc_one(u32 saddr) {
  return (u64)((saddr & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)(256 >> 4) << 32) | ((u64)3 << 62);
}
DEVI u64 desc_v(u32 saddr) {
  return (u64)((saddr & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)(1024 >> 4) << 32) | ((u64)1 << 62);
}

#define ACC16(a) "+f"(a[0][0]), "+f"(a[0][1]), "+f"(a[0][2]), "+f"(a[0][3])
// S[16][4] += Q (64 x 32 FP8, registers) * K^T (32 x 128, shared)
DEVI void wgmma_qk(float (&s)[16][4], u32 a0, u32 a1, u32 a2, u32 a3, u64 desc, int accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %68, 0; wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
               "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
               " {%64,%65,%66,%67}, %69, p, 1, 1; }"
               : "+f"(s[0][0]), "+f"(s[0][1]), "+f"(s[0][2]), "+f"(s[0][3]), "+f"(s[1][0]), "+f"(s[1][1]), "+f"(s[1][2]), "+f"(s[1][3]),
                 "+f"(s[2][0]), "+f"(s[2][1]), "+f"(s[2][2]), "+f"(s[2][3]), "+f"(s[3][0]), "+f"(s[3][1]), "+f"(s[3][2]), "+f"(s[3][3]),
                 "+f"(s[4][0]), "+f"(s[4][1]), "+f"(s[4][2]), "+f"(s[4][3]), "+f"(s[5][0]), "+f"(s[5][1]), "+f"(s[5][2]), "+f"(s[5][3]),
                 "+f"(s[6][0]), "+f"(s[6][1]), "+f"(s[6][2]), "+f"(s[6][3]), "+f"(s[7][0]), "+f"(s[7][1]), "+f"(s[7][2]), "+f"(s[7][3]),
                 "+f"(s[8][0]), "+f"(s[8][1]), "+f"(s[8][2]), "+f"(s[8][3]), "+f"(s[9][0]), "+f"(s[9][1]), "+f"(s[9][2]), "+f"(s[9][3]),
                 "+f"(s[10][0]), "+f"(s[10][1]), "+f"(s[10][2]), "+f"(s[10][3]), "+f"(s[11][0]), "+f"(s[11][1]), "+f"(s[11][2]), "+f"(s[11][3]),
                 "+f"(s[12][0]), "+f"(s[12][1]), "+f"(s[12][2]), "+f"(s[12][3]), "+f"(s[13][0]), "+f"(s[13][1]), "+f"(s[13][2]), "+f"(s[13][3]),
                 "+f"(s[14][0]), "+f"(s[14][1]), "+f"(s[14][2]), "+f"(s[14][3]), "+f"(s[15][0]), "+f"(s[15][1]), "+f"(s[15][2]), "+f"(s[15][3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(accumulate), "l"(desc));
}
// S[8][4] (one 64-key half) = Q (64 x 32 FP8, registers) * K^T (32 x 64, shared)
DEVI void wgmma_qk64(float* s, u32 a0, u32 a1, u32 a2, u32 a3, u64 desc, int accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %36, 0; wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
               " {%32,%33,%34,%35}, %37, p, 1, 1; }"
               : "+f"(s[0]), "+f"(s[1]), "+f"(s[2]), "+f"(s[3]), "+f"(s[4]), "+f"(s[5]), "+f"(s[6]), "+f"(s[7]),
                 "+f"(s[8]), "+f"(s[9]), "+f"(s[10]), "+f"(s[11]), "+f"(s[12]), "+f"(s[13]), "+f"(s[14]), "+f"(s[15]),
                 "+f"(s[16]), "+f"(s[17]), "+f"(s[18]), "+f"(s[19]), "+f"(s[20]), "+f"(s[21]), "+f"(s[22]), "+f"(s[23]),
                 "+f"(s[24]), "+f"(s[25]), "+f"(s[26]), "+f"(s[27]), "+f"(s[28]), "+f"(s[29]), "+f"(s[30]), "+f"(s[31])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(accumulate), "l"(desc));
}
// S16[16][2] (+)= Q * K^T with f16 accumulators: [n8][0] = row g (cols 2tq, 2tq+1), [n8][1] = row g+8
DEVI void wgmma_qk_h(u32 (&h)[16][2], u32 a0, u32 a1, u32 a2, u32 a3, u64 desc, int accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %36, 0; wgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
               " {%32,%33,%34,%35}, %37, p, 1, 1; }"
               : "+r"(h[0][0]), "+r"(h[0][1]), "+r"(h[1][0]), "+r"(h[1][1]), "+r"(h[2][0]), "+r"(h[2][1]), "+r"(h[3][0]), "+r"(h[3][1]),
                 "+r"(h[4][0]), "+r"(h[4][1]), "+r"(h[5][0]), "+r"(h[5][1]), "+r"(h[6][0]), "+r"(h[6][1]), "+r"(h[7][0]), "+r"(h[7][1]),
                 "+r"(h[8][0]), "+r"(h[8][1]), "+r"(h[9][0]), "+r"(h[9][1]), "+r"(h[10][0]), "+r"(h[10][1]), "+r"(h[11][0]), "+r"(h[11][1]),
                 "+r"(h[12][0]), "+r"(h[12][1]), "+r"(h[13][0]), "+r"(h[13][1]), "+r"(h[14][0]), "+r"(h[14][1]), "+r"(h[15][0]), "+r"(h[15][1])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(accumulate), "l"(desc));
}
DEVI u32 h2max(u32 a, u32 b) { u32 r; asm("max.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
DEVI u32 h2sub(u32 a, u32 b) { u32 r; asm("sub.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
DEVI u32 h2ex2(u32 a) { u32 r; asm("ex2.approx.f16x2 %0, %1;" : "=r"(r) : "r"(a)); return r; }
DEVI u32 h2mul(u32 a, u32 b) { u32 r; asm("mul.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
DEVI u32 prmt_b(u32 a, u32 b, u32 sel) { u32 r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel)); return r; }
DEVI u32 h2add(u32 a, u32 b) { u32 r; asm("add.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
DEVI u32 h2fma(u32 a, u32 b, u32 c) { u32 r; asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
// 2^x for f16x2 x <= 8 on the FMA pipe: n = round(x) through the 1551 magic (its low 6 bits hold
// 15 + n), 2^f on [-0.5, 0.5] by a degree-2 minimax fit (f16 error 0.3%), 2^n built in the exponent.
// x is clamped at -14 (2^-14 rounds to 0 in FP8 E4M3).
DEVI u32 h2exp_poly(u32 x) {
  x = h2max(x, 0xCB00CB00u);
  const u32 y = h2add(x, 0x660F660Fu);
  const u32 f = h2sub(x, h2sub(y, 0x660F660Fu));
  const u32 pp = h2fma(h2fma(f, 0x33A133A1u, 0x39A139A1u), f, 0x3C003C00u);
  return h2mul(pp, (y << 10) & 0x7C007C00u);
}
DEVI u16 h2_to_e4m3x2(u32 a) { u16 r; asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(r) : "r"(a)); return r; }
DEVI u32 f2_to_h2(float lo, float hi) { u32 r; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
DEVI float h2_hi_f(u32 a) { return __half2float(__ushort_as_half((u16)(a >> 16))); }
DEVI float h2_lo_f(u32 a) { return __half2float(__ushort_as_half((u16)(a & 0xFFFF))); }
// O[9][4] += P (64 x 32 FP8, registers) * V (32 keys x 72: 64 d + 8 ones rows, shared, K-major V^T)
DEVI void wgmma_pv(float (&o)[9][4], u32 a0, u32 a1, u32 a2, u32 a3, u64 desc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35},"
               " {%36,%37,%38,%39}, %40, p, 1, 1; }"
               : "+f"(o[0][0]), "+f"(o[0][1]), "+f"(o[0][2]), "+f"(o[0][3]), "+f"(o[1][0]), "+f"(o[1][1]), "+f"(o[1][2]), "+f"(o[1][3]),
                 "+f"(o[2][0]), "+f"(o[2][1]), "+f"(o[2][2]), "+f"(o[2][3]), "+f"(o[3][0]), "+f"(o[3][1]), "+f"(o[3][2]), "+f"(o[3][3]),
                 "+f"(o[4][0]), "+f"(o[4][1]), "+f"(o[4][2]), "+f"(o[4][3]), "+f"(o[5][0]), "+f"(o[5][1]), "+f"(o[5][2]), "+f"(o[5][3]),
                 "+f"(o[6][0]), "+f"(o[6][1]), "+f"(o[6][2]), "+f"(o[6][3]), "+f"(o[7][0]), "+f"(o[7][1]), "+f"(o[7][2]), "+f"(o[7][3]),
                 "+f"(o[8][0]), "+f"(o[8][1]), "+f"(o[8][2]), "+f"(o[8][3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
}
DEVI u32 pack_e4m3x4(float a, float b, float c, float d) {  // bytes a, b, c, d (low to high)
  u16 lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(b), "f"(a));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(d), "f"(c));
  u32 r;
  asm("mov.b32 %0, {%1, %2};" : "=r"(r) : "h"(lo), "h"(hi));
  return r;
}
template <int NA> DEVI void pin(float (&a)[NA][4]) {
#pragma unroll
  for (int i = 0; i < NA; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) asm volatile("" : "+f"(a[i][j]) :: "memory");
}

#ifdef ATT_NOEXP
DEVI float ex2_mufu(float x) { return x; }   // ablation only: no exp (wrong results)
#else
DEVI float ex2_mufu(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
#endif
// 2^x on the FMA pipe: round-to-nearest split, degree-3 minimax for 2^f on [-0.5, 0.5], exponent add
DEVI float ex2_poly(float x) {
  const float t = x + 12582912.f;
  const float f = x - (t - 12582912.f);
  float p = fmaf(fmaf(fmaf(f, 0.0555041086648216f, 0.2402264923172690f), f, 0.6931471805599453f), f, 1.0f);
  return __int_as_float(__float_as_int(p) + (__float_as_int(t) << 23));
}
// 2^x for x <= 9 feeding an FP8 P: the FMA pipe does the split and the polynomial, IMAD the exponent
// add (no ALU work); x is floored at -32 by an FFMA-free trick: inputs below give a tiny positive value.
DEVI float ex2_poly8(float x) {
  x = fmaxf(x, -32.f);
  const float t = x + 12582912.f;
  const float f = x - (t - 12582912.f);
  const float p = fmaf(fmaf(fmaf(f, 0.0555041086648216f, 0.2402264923172690f), f, 0.6931471805599453f), f, 1.0f);
  int r;
  asm("mad.lo.s32 %0, %1, 8388608, %2;" : "=r"(r) : "r"(__float_as_int(t)), "r"(__float_as_int(p)));
  return __int_as_float(r);
}
DEVI u32 pack_bf16(float lo, float hi) {
  u32 r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
DEVI u16 f2_to_e4m3x2(float lo, float hi) {
  u16 r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}
DEVI float e4m3_to_f(u8 b) {
  u16 in = b;
  u32 r;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(in));
  return __half2float(__ushort_as_half((u16)(r & 0xFFFF)));
}
// -m as two E4M3 bytes (hi, lo) for the bias k-step: returns the byte pair and sets m to the
// value the pair represents, so the offset actually applied is known exactly
DEVI u32 bias_pair(float& m) {
  const u16 h = f2_to_e4m3x2(-m, 0.f) & 0xFF;
  const float hv = e4m3_to_f((u8)h);
  const u16 l = f2_to_e4m3x2(-m - hv, 0.f) & 0xFF;
  m = -(hv + e4m3_to_f((u8)l));
  return (u32)h | ((u32)l << 8);
}

struct Params {
  const __nv_bfloat16* qkv;    // [T][2304]
  u8* q8;                      // [T][16][64] FP8, pre-scaled
  u8* k_cache;                 // flat rows [rows][2][64]
  u8* v_cache;
  const long long* cache_loc;  // [T]
  const int* page_table;       // [B][max_pages]
  const int* seq_lens;         // [B] cached tokens per request
  const float* k_scale;
  const float* v_scale;
  const float* inv_freq;       // [32]
  u8* kp;                      // [B][2][NTA][128][64] FP8 swizzled
  u8* vp;                      // [B][2][NTA][64 d][128 keys] FP8 V^T, keys permuted per 16, 128B-swizzled
  int* ctab;                   // [B][NQB + 1] tile prefix per request
  __nv_bfloat16* out;          // [T][1024]
  float* oacc;                 // [T][16][64]
  float* lacc;                 // [T][16]
  int* cnt;                    // [B][2][NQB][ATT_CW]
  int B, L, window, rope, page, max_pages, nta, nqb;
  float qscale;
  int part;                    // kv_pack: 0 every tile, 1 tiles of cached keys only (+ ctab), 2 the rest
  const float* rope_tab;       // [npos][32] (cos, sin)
  int npos;
  int unit;                    // attn_fwd: whole-unit scheduling (see for_segments)
  int direct_new;              // qkv_post2 also writes the new keys into kp / vp (kv_pack part 3 does the rest)                // sm_scale * qk_scale_factor * log2(e)
  float* hacc; float* hl;      // split-unit handoff scratch [2048][16][64], [2048][16] (never needs zeroing)
};

// ---------------------------------------------------------------------------------------------
// qkv_post: one block per token, one warp per head vector (16 q, 2 k, 2 v), two elements per lane.
extern "C" __global__ void __launch_bounds__(640) qkv_post(Params p) {
  const int t = blockIdx.x, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int b = t / p.L, i = t % p.L;
  const int pos = p.seq_lens[b] + i;
  const __nv_bfloat16* row = p.qkv + (size_t)t * 2304;
  int col;
  if (w < 16) col = w * 64;
  else if (w < 18) col = 1024 + (w - 16) * 64;
  else col = 1152 + (w - 18) * 64;
  __nv_bfloat162 xv = *reinterpret_cast<const __nv_bfloat162*>(row + col + 2 * lane);
  float x0 = __low2float(xv), x1 = __high2float(xv);
  if (w < 18) {
    float ss = x0 * x0 + x1 * x1;
#pragma unroll
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
    const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
    x0 *= r; x1 *= r;
    if (p.rope) {
      const float ang = (float)pos * p.inv_freq[lane];
      float sn, cs;
      sincosf(ang, &sn, &cs);
      const float y0 = x0 * cs - x1 * sn, y1 = x0 * sn + x1 * cs;
      x0 = y0; x1 = y1;
    }
  }
  if (w < 16) {
    const float c = p.qscale * p.k_scale[0];
    u16 q = f2_to_e4m3x2(x0 * c, x1 * c);
    *reinterpret_cast<u16*>(p.q8 + (size_t)t * 1024 + w * 64 + 2 * lane) = q;
  } else {
    const long long rowi = p.cache_loc[t];
    const int h = (w < 18) ? (w - 16) : (w - 18);
    const float inv = 1.f / ((w < 18) ? p.k_scale[0] : p.v_scale[0]);
    u16 q = f2_to_e4m3x2(x0 * inv, x1 * inv);
    u8* dst = ((w < 18) ? p.k_cache : p.v_cache) + rowi * 128 + h * 64 + 2 * lane;
    *reinterpret_cast<u16*>(dst) = q;
  }
}

DEVI int base_tile(const struct Params& p, int L0);
// qkv_post2: one warp per token. Pass ps (0..4) covers head vectors 4 ps .. 4 ps + 3 (16 q, then 2 k,
// 2 v: columns hv * 64), 8 lanes per vector and 8 elements per lane (16-byte loads). RoPE takes
// cos/sin from a table built like SGLang's (FP32 pos * inv_freq), sincosf past its end.
extern "C" __global__ void __launch_bounds__(256) qkv_post2(Params p) {
  // PDL: the projection (previous grid) must be complete; attn_fwd may launch now (its prologue
  // overlaps this grid; it waits for us before reading q8 / the packed tiles)
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int t = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (t >= p.B * p.L) return;
  const int b = t / p.L, i = t % p.L;
  const int pos = p.seq_lens[b] + i;
  const int sub = lane & 7, hvl = lane >> 3;
  const __nv_bfloat16* row = p.qkv + (size_t)t * 2304;
  uint4 raw[5];
#pragma unroll
  for (int ps = 0; ps < 5; ++ps) raw[ps] = *reinterpret_cast<const uint4*>(row + (4 * ps + hvl) * 64 + 8 * sub);
  float cs[4], sn[4];
  if (p.rope) {
    if (pos < p.npos) {
      const float4* tb = reinterpret_cast<const float4*>(p.rope_tab + ((size_t)pos * 32 + 4 * sub) * 2);
      const float4 a = tb[0], c = tb[1];   // (cos, sin) pairs for freqs 4 sub .. 4 sub + 3
      cs[0] = a.x; sn[0] = a.y; cs[1] = a.z; sn[1] = a.w; cs[2] = c.x; sn[2] = c.y; cs[3] = c.z; sn[3] = c.w;
    } else {
#pragma unroll
      for (int k = 0; k < 4; ++k) sincosf((float)pos * p.inv_freq[4 * sub + k], &sn[k], &cs[k]);
    }
  }
  const long long rowi = p.cache_loc[t];
  const float cq = p.qscale * p.k_scale[0], ik = 1.f / p.k_scale[0], iv = 1.f / p.v_scale[0];
#pragma unroll
  for (int ps = 0; ps < 5; ++ps) {
    const int hv = 4 * ps + hvl;
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&raw[ps]);
    float x[8];
#pragma unroll
    for (int k = 0; k < 4; ++k) { x[2 * k] = __low2float(h2[k]); x[2 * k + 1] = __high2float(h2[k]); }
    if (hv < 18) {
      float ss = 0.f;
#pragma unroll
      for (int k = 0; k < 8; ++k) ss += x[k] * x[k];
      ss += __shfl_xor_sync(0xffffffff, ss, 1);
      ss += __shfl_xor_sync(0xffffffff, ss, 2);
      ss += __shfl_xor_sync(0xffffffff, ss, 4);
      const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
      for (int k = 0; k < 8; ++k) x[k] *= r;
      if (p.rope) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float y0 = x[2 * k] * cs[k] - x[2 * k + 1] * sn[k], y1 = x[2 * k] * sn[k] + x[2 * k + 1] * cs[k];
          x[2 * k] = y0; x[2 * k + 1] = y1;
        }
      }
    }
    const float sc = hv < 16 ? cq : (hv < 18 ? ik : iv);
    u32 lo = (u32)f2_to_e4m3x2(x[0] * sc, x[1] * sc) | ((u32)f2_to_e4m3x2(x[2] * sc, x[3] * sc) << 16);
    u32 hi = (u32)f2_to_e4m3x2(x[4] * sc, x[5] * sc) | ((u32)f2_to_e4m3x2(x[6] * sc, x[7] * sc) << 16);
    u8* dst;
    if (hv < 16) dst = p.q8 + (size_t)t * 1024 + hv * 64 + 8 * sub;
    else if (hv < 18) dst = p.k_cache + rowi * 128 + (hv - 16) * 64 + 8 * sub;
    else dst = p.v_cache + rowi * 128 + (hv - 18) * 64 + 8 * sub;
    *reinterpret_cast<uint2*>(dst) = make_uint2(lo, hi);
    if (p.direct_new && hv >= 16) {
      // the packed tiles too: tile and row of this key, relative to the request's first tile
      const int L0 = p.seq_lens[b];
      const int tl = pos / 128 - base_tile(p, L0), kk = pos & 127;
      const size_t tix = (size_t)(b * 2 + ((hv - 16) & 1)) * p.nta + tl;
      if (hv < 18) {
        *reinterpret_cast<uint2*>(p.kp + tix * 8192 + kk * 64 + ((((sub >> 1) ^ ((kk >> 1) & 3))) << 4) + 8 * (sub & 1)) =
            make_uint2(lo, hi);
      } else {
        // V^T byte of each d: tile position 16 un + e carries key 16 un + pi(e); e = pi^-1(kk & 15)
        const int un = kk >> 4, kr = kk & 15;
        const int e = 4 * ((kr & 7) >> 1) + 2 * (kr >> 3) + (kr & 1);
        u8* vb = p.vp + tix * 8192 + e;
        const u32 wv[2] = {lo, hi};
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const int d = 8 * sub + q;
          vb[d * 128 + ((un ^ (d & 7)) << 4)] = (u8)(wv[q >> 2] >> (8 * (q & 3)));
        }
      }
    }
  }
}

// Visible key range [lo, hi) of query block qb of request b (absolute positions), and the
// request's first packed tile.
DEVI void qb_range(const Params& p, int L0, int qb, int& lo, int& hi) {
  const int i0 = qb * ATT_QB, i1 = min(qb * ATT_QB + ATT_QB, p.L);
  hi = L0 + i1;
  lo = p.window ? max(0, L0 + i0 - (p.window - 1)) : 0;
}
DEVI int base_tile(const Params& p, int L0) { return p.window ? max(0, L0 - (p.window - 1)) / 128 : 0; }
// Work order of query blocks: short and long causal blocks alternate (0, n-1, 1, n-2, ...), so every
// CTA's even share of tiles holds about as many blocks (and per-block costs) as any other.
DEVI int qb_of(int i, int nqb) { return (i & 1) ? nqb - 1 - (i >> 1) : (i >> 1); }

// ---------------------------------------------------------------------------------------------
// kv_pack: blocks [0, B*2*nta) pack one (request, head, tile); the last B blocks build ctab.
#ifndef KVP_OLD
#define KVP_OLD 0           // 1: the byte-gather V transpose (8-way bank conflicts)
#endif
// staged V: 128 key rows of 16 words at a 17-word stride; the transpose reads word w of 16 keys
// per thread (lanes: 16 w x 2 units, conflict-free) and turns 4 x 4 byte blocks with PRMT
DEVI u32 prmt(u32 a, u32 b, u32 sel) { u32 r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel)); return r; }
extern "C" __global__ void __launch_bounds__(256) kv_pack(Params p) {
  __shared__ u32 vsw[128 * 17];
  u8 (&vs)[128][68] = *reinterpret_cast<u8 (*)[128][68]>(vsw);
  const int nblk = p.B * 2 * p.nta;
  const int tid = threadIdx.x;
  if ((int)blockIdx.x >= nblk) {
    if (p.part == 2) return;
    const int b = blockIdx.x - nblk;
    const int L0 = p.seq_lens[b];
    const int bt = base_tile(p, L0);
    // prefix over query blocks (sequential per 256-wide chunk with a block scan)
    __shared__ int part[256];
    int carry = 0;
    for (int q0 = 0; q0 < p.nqb; q0 += 256) {
      const int qi = q0 + tid;
      int n = 0;
      if (qi < p.nqb) {
        int lo, hi;
        qb_range(p, L0, qb_of(qi, p.nqb), lo, hi);
        n = (hi + 127) / 128 - lo / 128;
      }
      part[tid] = n;
      __syncthreads();
      for (int o = 1; o < 256; o <<= 1) {
        int v = tid >= o ? part[tid - o] : 0;
        __syncthreads();
        part[tid] += v;
        __syncthreads();
      }
      if (qi < p.nqb) p.ctab[b * (p.nqb + 1) + qi + 1] = carry + part[tid];
      carry += part[255];
      __syncthreads();
    }
    if (tid == 0) p.ctab[b * (p.nqb + 1)] = 0;
    (void)bt;
    return;
  }
  const int tile = blockIdx.x % p.nta, h = (blockIdx.x / p.nta) & 1, b = blockIdx.x / (2 * p.nta);
  const int L0 = p.seq_lens[b];
  const int total = L0 + p.L;
  const int lo_req = p.window ? max(0, L0 - (p.window - 1)) : 0;
  const int k0 = (base_tile(p, L0) + tile) * 128;
  u8* kd = p.kp + ((size_t)(b * 2 + h) * p.nta + tile) * 8192;
  u8* vd = p.vp + ((size_t)(b * 2 + h) * p.nta + tile) * 8192;
  if (k0 >= total) return;
  // part 1 runs beside the QKV projection: only tiles of keys already in the cache
  if ((p.part == 1 && k0 + 128 > L0) || (p.part == 2 && k0 + 128 <= L0)) return;
  if (p.part == 3 && k0 >= L0 && k0 + 128 <= total) return;   // all new keys: qkv_post2 packs them
  const bool mixed = p.part == 3 && k0 + 128 > L0;            // some keys here are new: skip their bytes
  // K: 128 keys x 4 chunks of 16 B; V into shared memory
  for (int u = tid; u < 512; u += 256) {
    const int kk = u >> 2, c = u & 3, pos = k0 + kk;
    uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
    if (pos < total && pos >= lo_req) {
      const long long rowi = (long long)p.page_table[b * p.max_pages + pos / p.page] * p.page + pos % p.page;
      kv = *reinterpret_cast<const uint4*>(p.k_cache + rowi * 128 + h * 64 + c * 16);
      vv = *reinterpret_cast<const uint4*>(p.v_cache + rowi * 128 + h * 64 + c * 16);
    }
    if (!(mixed && pos >= L0 && pos < total)) *reinterpret_cast<uint4*>(kd + kk * 64 + ((c ^ ((kk >> 1) & 3)) << 4)) = kv;
    u32* vrow = vsw + kk * 17 + c * 4;
    vrow[0] = vv.x; vrow[1] = vv.y; vrow[2] = vv.z; vrow[3] = vv.w;
  }
  __syncthreads();
#if !KVP_OLD
  if (tid < 128) {
    const int w = tid & 15, un = tid >> 4;
    u32 x[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) x[e] = vsw[(16 * un + e) * 17 + w];
    // out row d = 4 w + j, word q: bytes j of keys 2q, 2q+1, 2q+8, 2q+9 (pi of e = 4 q + 0..3)
    u32 o[4][4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const u32 t0 = prmt(x[2 * q], x[2 * q + 1], 0x5140), t1 = prmt(x[2 * q], x[2 * q + 1], 0x7362);
      const u32 t2 = prmt(x[2 * q + 8], x[2 * q + 9], 0x5140), t3 = prmt(x[2 * q + 8], x[2 * q + 9], 0x7362);
      o[0][q] = prmt(t0, t2, 0x5410); o[1][q] = prmt(t0, t2, 0x7632);
      o[2][q] = prmt(t1, t3, 0x5410); o[3][q] = prmt(t1, t3, 0x7632);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int d = 4 * w + j;
      u8* dst = vd + d * 128 + ((un ^ (d & 7)) << 4);
      if (!mixed) {
        *reinterpret_cast<uint4*>(dst) = make_uint4(o[j][0], o[j][1], o[j][2], o[j][3]);
      } else {
#pragma unroll
        for (int e = 0; e < 16; ++e) {
          const int pos = k0 + 16 * un + 2 * (e >> 2) + (e & 1) + 8 * ((e >> 1) & 1);
          if (!(pos >= L0 && pos < total)) dst[e] = (u8)(o[j][e >> 2] >> (8 * (e & 3)));
        }
      }
    }
  }
  return;
#endif
  // V^T: row d, 16-byte unit un holds tile positions 16 un + e, e = 0..15, which carry key
  // 16 un + pi(e), pi(e) = 2 (e >> 2) + (e & 1) + 8 ((e >> 1) & 1): the order a thread's FP8 P
  // fragment takes the S accumulator's columns in. Unit un sits at (un ^ (d & 7)) (128B swizzle).
  for (int u = tid; u < 512; u += 256) {
    const int d = u >> 3, un = u & 7;
    u32 w[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      u32 x = 0;
#pragma unroll
      for (int e2 = 0; e2 < 4; ++e2) {
        const int e = q * 4 + e2;
        const int key = 16 * un + 2 * (e >> 2) + (e & 1) + 8 * ((e >> 1) & 1);
        x |= (u32)vs[key][d] << (8 * e2);
      }
      w[q] = x;
    }
    u8* dst = vd + d * 128 + ((un ^ (d & 7)) << 4);
    if (!mixed) {
      *reinterpret_cast<uint4*>(dst) = make_uint4(w[0], w[1], w[2], w[3]);
    } else {
#pragma unroll
      for (int e = 0; e < 16; ++e) {
        const int pos = k0 + 16 * un + 2 * (e >> 2) + (e & 1) + 8 * ((e >> 1) & 1);
        if (!(pos >= L0 && pos < total)) dst[e] = (u8)(w[e >> 2] >> (8 * (e & 3)));
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// attn_fwd
struct Seg { int b, h, qb, qi, t0, n, full; };

// The CTA's work in request b: tiles [s, e) of the request's (head-major, block, tile) order.
DEVI bool seg_at(const Params& p, int b, int x, int e, Seg& sg) {
  const int* ct = p.ctab + b * (p.nqb + 1);
  const int wh = ct[p.nqb];
  sg.b = b;
  sg.h = x >= wh;
  const int xr = x - sg.h * wh;
  // binary search: last qb with ct[qb] <= xr
  int lo = 0, hi = p.nqb;
  while (hi - lo > 1) {
    const int mid = (lo + hi) >> 1;
    if (ct[mid] <= xr) lo = mid; else hi = mid;
  }
  sg.qb = qb_of(lo, p.nqb);
  sg.qi = lo;
  const int u = xr - ct[lo], nt = ct[lo + 1] - ct[lo];
  sg.t0 = u;
  sg.n = min(e - x, nt - u);
  sg.full = (u == 0 && sg.n == nt);
  return true;
}

DEVI void named_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory"); }

// The CTA's segments, in order. Stream-K (default): each request's tiles split evenly over the CTAs.
// Unit mode (p.unit): whole (request, kv head, query block) units dealt round-robin, no partial
// sums, for short blocks of equal length (sliding layers).
template <class F> DEVI void for_segments(const Params& p, int cta, int G, F&& f) {
  if (p.unit) {
    // p.unit = S: each unit's tiles split into S near-equal parts (partial sums when S > 1)
    const int S = p.unit, U = p.B * 2 * p.nqb * S;
    for (int us = cta; us < U; us += G) {
      Seg sg;
      const int u = us / S, part = us % S;
      const int b = u / (2 * p.nqb), r = u % (2 * p.nqb);
      const int* ct = p.ctab + b * (p.nqb + 1);
      sg.b = b; sg.h = r / p.nqb; sg.qi = r % p.nqb; sg.qb = qb_of(sg.qi, p.nqb);
      const int nt = ct[sg.qi + 1] - ct[sg.qi];
      sg.t0 = nt * part / S; sg.n = nt * (part + 1) / S - sg.t0; sg.full = sg.n == nt;
      if (sg.n > 0) f(sg);
    }
    return;
  }
  for (int b = 0; b < p.B; ++b) {
    const int W = 2 * p.ctab[b * (p.nqb + 1) + p.nqb];
    const int s = (int)((long long)W * cta / G), e = (int)((long long)W * (cta + 1) / G);
    for (int x = s; x < e;) {
      Seg sg;
      seg_at(p, b, x, e, sg);
      x += sg.n;
      f(sg);
    }
  }
}

extern "C" __global__ void __launch_bounds__(128 * (1 + ATT_CW), 1) attn_fwd(Params p) {
  extern __shared__ __align__(1024) u8 smem[];
  // layout: stages [NS][K 8 KB | V^T 8 KB + 1 KB of ones rows] at 1024-aligned offsets, then barriers
  const u32 sbase = smem_addr(smem);
  const u32 bars = sbase + ATT_NS * 18432;
  volatile int* flag = reinterpret_cast<volatile int*>(smem + ATT_NS * 18432 + 16 * ATT_NS);
  const int tid = threadIdx.x, wg = tid >> 7, lt = tid & 127;
  if (tid == 0) {
    for (int s = 0; s < ATT_NS; ++s) {
      mbar_init(bars + 16 * s, 1);
      mbar_init(bars + 16 * s + 8, ATT_CW);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // the ones rows (V^T rows 64..71) of every stage: FP8 1.0 = 0x38
  for (int i = tid; i < ATT_NS * 256; i += blockDim.x)
    reinterpret_cast<u32*>(smem + (i >> 8) * 18432 + 16384)[i & 255] = 0x38383838u;
#if ATT_BIAS
  // bias tile: row r holds 1.0, 1.0 at k = 0, 1; 32B swizzle moves rows 4..7 of each group to chunk 1
  const u32 ones_off = (ATT_NS * 18432 + 16 * ATT_NS + 64 + 1023) & ~1023u;
  for (int i = tid; i < 1024; i += blockDim.x) {
    const int r = i >> 3, w = i & 7;   // 8 words per 32-byte row
    reinterpret_cast<u32*>(smem + ones_off)[i] = (w == 4 * ((r >> 2) & 1)) ? 0x3838u : 0u;
  }
#endif
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  asm volatile("griddepcontrol.wait;" ::: "memory");   // PDL: qkv_post2's q8 / packed keys complete
  const int G = gridDim.x, cta = blockIdx.x;

  if (wg == 0) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 24;" ::: "memory");
    if (tid != 0) return;
    int it = 0;
    for_segments(p, cta, G, [&](const Seg& sg) {
      const int b = sg.b;
      const int L0 = p.seq_lens[b];
        int lo, hi;
        qb_range(p, L0, sg.qb, lo, hi);
        const int tfirst = lo / 128 - base_tile(p, L0) + sg.t0;
        for (int j = 0; j < sg.n; ++j, ++it) {
          const int st = it % ATT_NS;
          mbar_wait(bars + 16 * st + 8, ((it / ATT_NS) & 1) ^ 1);
          const u32 full = bars + 16 * st;
          mbar_arrive_tx(full, 16384);
          const size_t tix = (size_t)(b * 2 + sg.h) * p.nta + tfirst + j;
          bulk_g2s(sbase + st * 18432, p.kp + tix * 8192, 8192, full);
          bulk_g2s(sbase + st * 18432 + 8192, p.vp + tix * 8192, 8192, full);
        }
    });
    return;
  }

  // consumers: warpgroup c = wg - 1 handles tokens qb*16 + c*8 + (0..7); thread rows g, g+8
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(ATT_CREG) : "memory");
  const int c = wg - 1, warp = lt >> 5, lane = lt & 31;
  if (ATT_PP && c == ATT_CW - 1) named_arrive(8, 256);
  const int g = lane >> 2, tq = lane & 3;
  // stage 0 descriptors; a stage's are these plus its offset (18432 B = 1152 descriptor units)
  const u64 dk0 = desc_k(sbase), dv0 = desc_v(sbase + 8192);
#if ATT_BIAS
  const u64 d_one = desc_one(sbase + ((ATT_NS * 18432 + 16 * ATT_NS + 64 + 1023) & ~1023u));
#endif
  const int tok_a = c * 8 + 2 * warp, tok_b = tok_a + 1;   // tokens of rows g and g+8 (within the block)
  const int head = g;                                      // head within the KV group
  int it = 0;
  for_segments(p, cta, G, [&](const Seg& sg) {
      const int b = sg.b;
      const int L0 = p.seq_lens[b];
      int lo, hi;
      qb_range(p, L0, sg.qb, lo, hi);
      const int bt = base_tile(p, L0);
      const int ia = sg.qb * ATT_QB + tok_a, ib = sg.qb * ATT_QB + tok_b;   // token index in the request
      const bool va = ia < p.L, vb = ib < p.L;
      const int pa = L0 + ia, pb = L0 + ib;                          // query positions
      const int hg = sg.h * 8 + head;                                // global q head
      // Q fragments: rows (tok_a, head), (tok_b, head); k = 32 ks + {4tq..4tq+3, 16+4tq..}
      u32 qa[2][4];
      {
        const u8* ra = p.q8 + ((size_t)(b * p.L + ia) * 16 + hg) * 64;
        const u8* rb = p.q8 + ((size_t)(b * p.L + ib) * 16 + hg) * 64;
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
          qa[ks][0] = va ? *reinterpret_cast<const u32*>(ra + 32 * ks + 4 * tq) : 0u;
          qa[ks][1] = vb ? *reinterpret_cast<const u32*>(rb + 32 * ks + 4 * tq) : 0u;
          qa[ks][2] = va ? *reinterpret_cast<const u32*>(ra + 32 * ks + 16 + 4 * tq) : 0u;
          qa[ks][3] = vb ? *reinterpret_cast<const u32*>(rb + 32 * ks + 16 + 4 * tq) : 0u;
        }
      }
      float o[9][4];
#pragma unroll
      for (int j = 0; j < 9; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
      float la = 0.f, lb = 0.f;
#if ATT_BIAS
      // S arrives as S - m (the bias k-step); m starts at 0 and is set from the first finite max
      float ma = 0.f, mb = 0.f;
      bool seta = false, setb = false;
      u32 bia = 0u, bib = 0u;              // A fragment of -m (tq == 0 lanes only)
#else
      float ma = -1e30f, mb = -1e30f;     // P = 2^(S - m): m trails the running row max by 8
#endif
      float sca = 1.f, scb = 1.f;         // O rescale owed for the latest max change
      bool resc = false;
      const int tfirst = lo / 128 - bt + sg.t0;
      // rows' visible key windows (absolute positions)
      const int wa_lo = p.window ? pa - (p.window - 1) : 0, wb_lo = p.window ? pb - (p.window - 1) : 0;
      const int it0 = it, n = sg.n;
      it += n;
      const int pmin = L0 + sg.qb * ATT_QB, pmax = L0 + min(sg.qb * ATT_QB + ATT_QB - 1, p.L - 1);
      u32 pf[4][4];
      auto issue_qk = [&](float (&sacc)[16][4], int jt) {
        const u32 itj = (u32)(it0 + jt), st = itj % ATT_NS;
        mbar_wait(bars + 16 * st, (itj / ATT_NS) & 1);
        const u64 dk = dk0 + (u64)(st * (18432 >> 4));
        wg_fence();
        wgmma_qk(sacc, qa[0][0], qa[0][1], qa[0][2], qa[0][3], dk, 0);
        wgmma_qk(sacc, qa[1][0], qa[1][1], qa[1][2], qa[1][3], dk + 2, 1);
#if ATT_BIAS
        wgmma_qk(sacc, bia, bib, 0u, 0u, d_one, 1);
#endif
        wg_commit();
      };
      auto issue_pv = [&](int jt) {
        const u64 dv = dv0 + (u64)(((u32)(it0 + jt) % ATT_NS) * (18432 >> 4));
#pragma unroll
        for (int kc = 0; kc < 4; ++kc)
          wgmma_pv(o, pf[kc][0], pf[kc][1], pf[kc][2], pf[kc][3], dv + 2 * kc);
        wg_commit();
      };
      // split QK: half A (keys 0..63), half B (keys 64..127), one commit group each
      auto issue_qk2 = [&](float (&sacc)[16][4], int jt) {
        const u32 itj = (u32)(it0 + jt), st = itj % ATT_NS;
        mbar_wait(bars + 16 * st, (itj / ATT_NS) & 1);
        const u64 dk = dk0 + (u64)(st * (18432 >> 4));
        wg_fence();
        wgmma_qk64(&sacc[0][0], qa[0][0], qa[0][1], qa[0][2], qa[0][3], dk, 0);
        wgmma_qk64(&sacc[0][0], qa[1][0], qa[1][1], qa[1][2], qa[1][3], dk + 2, 1);
        wg_commit();
        wgmma_qk64(&sacc[8][0], qa[0][0], qa[0][1], qa[0][2], qa[0][3], dk + 256, 0);
        wgmma_qk64(&sacc[8][0], qa[1][0], qa[1][1], qa[1][2], qa[1][3], dk + 258, 1);
        wg_commit();
      };
      // one 64-key half: mask, max, running-max update (rescaling half A's P if half B moves it), exp
      auto softmax_half = [&](float (&sacc)[16][4], int jt, const int hf) {
        const int k0 = (bt + tfirst + jt) * 128 + 64 * hf;
        const bool masked = (k0 + 63 > pmin) || (p.window && k0 < pmax - (p.window - 1));
        if (hf == 0) { sca = 1.f; scb = 1.f; }
        if (masked) {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const int n8 = 8 * hf + i, kc = k0 + i * 8 + 2 * tq;
            if (!(kc <= pa && kc >= wa_lo)) sacc[n8][0] = -__int_as_float(0x7f800000);
            if (!(kc + 1 <= pa && kc + 1 >= wa_lo)) sacc[n8][1] = -__int_as_float(0x7f800000);
            if (!(kc <= pb && kc >= wb_lo)) sacc[n8][2] = -__int_as_float(0x7f800000);
            if (!(kc + 1 <= pb && kc + 1 >= wb_lo)) sacc[n8][3] = -__int_as_float(0x7f800000);
          }
        }
        float ma2c[2], mb2c[2];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          ma2c[k] = fmaxf(sacc[8 * hf + k][0], sacc[8 * hf + k][1]);
          mb2c[k] = fmaxf(sacc[8 * hf + k][2], sacc[8 * hf + k][3]);
        }
#pragma unroll
        for (int i = 2; i < 8; ++i) {
          ma2c[i & 1] = fmaxf(ma2c[i & 1], fmaxf(sacc[8 * hf + i][0], sacc[8 * hf + i][1]));
          mb2c[i & 1] = fmaxf(mb2c[i & 1], fmaxf(sacc[8 * hf + i][2], sacc[8 * hf + i][3]));
        }
        float ta = fmaxf(ma2c[0], ma2c[1]), tb = fmaxf(mb2c[0], mb2c[1]);
        ta = fmaxf(ta, __shfl_xor_sync(0xffffffff, ta, 1)); ta = fmaxf(ta, __shfl_xor_sync(0xffffffff, ta, 2));
        tb = fmaxf(tb, __shfl_xor_sync(0xffffffff, tb, 1)); tb = fmaxf(tb, __shfl_xor_sync(0xffffffff, tb, 2));
        const bool na = ta > ma + 8.f, nb = tb > mb + 8.f;
        if (__any_sync(0xffffffff, na || nb)) {
          const float ma2 = na ? ta - 8.f : ma, mb2 = nb ? tb - 8.f : mb;
          const float fa = ex2_mufu(ma - ma2), fb = ex2_mufu(mb - mb2);
          if (hf == 1) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { sacc[i][0] *= fa; sacc[i][1] *= fa; sacc[i][2] *= fb; sacc[i][3] *= fb; }
          }
          sca *= fa; scb *= fb;
          ma = ma2; mb = mb2;
          resc = true;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int n8 = 8 * hf + i;
          sacc[n8][0] = ex2_mufu(sacc[n8][0] - ma); sacc[n8][1] = ex2_mufu(sacc[n8][1] - ma);
          sacc[n8][2] = ex2_mufu(sacc[n8][2] - mb); sacc[n8][3] = ex2_mufu(sacc[n8][3] - mb);
        }
      };
      // ---- f16 scores (ATT_F16S): S accumulates in f16 pairs; max, subtract, exp and the FP8 cast run on pairs
      auto issue_qk_h = [&](u32 (&hs)[16][2], int jt) {
        const u32 itj = (u32)(it0 + jt), st = itj % ATT_NS;
        mbar_wait(bars + 16 * st, (itj / ATT_NS) & 1);
        const u64 dk = dk0 + (u64)(st * (18432 >> 4));
        wg_fence();
        wgmma_qk_h(hs, qa[0][0], qa[0][1], qa[0][2], qa[0][3], dk, 0);
        wgmma_qk_h(hs, qa[1][0], qa[1][1], qa[1][2], qa[1][3], dk + 2, 1);
        wg_commit();
      };
      auto softmax_h = [&](u32 (&hs)[16][2], int jt) {
        const int k0 = (bt + tfirst + jt) * 128;
        const bool masked = (k0 + 127 > pmin) || (p.window && k0 < pmax - (p.window - 1));
        if (masked) {
#pragma unroll
          for (int n8 = 0; n8 < 16; ++n8) {
            const int kc = k0 + n8 * 8 + 2 * tq;
            const u32 m0 = ((kc <= pa && kc >= wa_lo) ? 0x0000FFFFu : 0u) | ((kc + 1 <= pa && kc + 1 >= wa_lo) ? 0xFFFF0000u : 0u);
            const u32 m1 = ((kc <= pb && kc >= wb_lo) ? 0x0000FFFFu : 0u) | ((kc + 1 <= pb && kc + 1 >= wb_lo) ? 0xFFFF0000u : 0u);
            hs[n8][0] = (hs[n8][0] & m0) | (0xFC00FC00u & ~m0);
            hs[n8][1] = (hs[n8][1] & m1) | (0xFC00FC00u & ~m1);
          }
        }
        u32 xa[4], xb[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) { xa[k] = hs[k][0]; xb[k] = hs[k][1]; }
#pragma unroll
        for (int n8 = 4; n8 < 16; ++n8) { xa[n8 & 3] = h2max(xa[n8 & 3], hs[n8][0]); xb[n8 & 3] = h2max(xb[n8 & 3], hs[n8][1]); }
        const u32 pa2 = h2max(h2max(xa[0], xa[1]), h2max(xa[2], xa[3])), pb2 = h2max(h2max(xb[0], xb[1]), h2max(xb[2], xb[3]));
        float ta = fmaxf(h2_lo_f(pa2), h2_hi_f(pa2)), tb = fmaxf(h2_lo_f(pb2), h2_hi_f(pb2));
        ta = fmaxf(ta, __shfl_xor_sync(0xffffffff, ta, 1)); ta = fmaxf(ta, __shfl_xor_sync(0xffffffff, ta, 2));
        tb = fmaxf(tb, __shfl_xor_sync(0xffffffff, tb, 1)); tb = fmaxf(tb, __shfl_xor_sync(0xffffffff, tb, 2));
        const bool na = ta > ma + 8.f, nb = tb > mb + 8.f;
        resc = __any_sync(0xffffffff, na || nb);
        if (resc) {
          float ma2 = na ? ta - 8.f : ma, mb2 = nb ? tb - 8.f : mb;
#if ATT_EXPQ
          ma2 = rintf(ma2 * 8.f) * 0.125f; mb2 = rintf(mb2 * 8.f) * 0.125f;   // 8 m integer: the code offset is exact
#endif
          sca = ex2_mufu(ma - ma2); scb = ex2_mufu(mb - mb2);
          ma = ma2; mb = mb2;
        }
#if ATT_EXPQ
        {
          // y = 8 S + (1080 - 8 m) lands in [1024, 2048) (ulp 1) for kept scores: its low byte is the
          // E4M3 code of 2^(S - m) with a linear mantissa; below 1024 (S - m < -7, masked) -> code 0
          const float ca = 1080.f - 8.f * fminf(fmaxf(ma, -1000.f), 1000.f), cb = 1080.f - 8.f * fminf(fmaxf(mb, -1000.f), 1000.f);
          const u32 cha = f2_to_h2(ca, ca), chb = f2_to_h2(cb, cb);
#pragma unroll
          for (int n8 = 0; n8 < 16; ++n8) {
            hs[n8][0] = h2max(h2fma(hs[n8][0], 0x48004800u, cha), 0x64006400u);
            hs[n8][1] = h2max(h2fma(hs[n8][1], 0x48004800u, chb), 0x64006400u);
          }
        }
        if (false) {
#endif
        const float mca = fmaxf(ma, -60000.f), mcb = fmaxf(mb, -60000.f);
        const u32 mha = f2_to_h2(mca, mca), mhb = f2_to_h2(mcb, mcb);
#pragma unroll
        for (int n8 = 0; n8 < 16; ++n8) {
          const u32 x0 = h2sub(hs[n8][0], mha), x1 = h2sub(hs[n8][1], mhb);
          if ((n8 * ATT_POLYH) % 16 < ATT_POLYH) { hs[n8][0] = h2exp_poly(x0); hs[n8][1] = h2exp_poly(x1); }
          else { hs[n8][0] = h2ex2(x0); hs[n8][1] = h2ex2(x1); }
        }
#if ATT_EXPQ
        }
#endif
      };
      auto to_p_h = [&](u32 (&hs)[16][2]) {
#if ATT_EXPQ
#pragma unroll
        for (int kc = 0; kc < 4; ++kc) {   // the codes are the low bytes of each f16 half
          pf[kc][0] = prmt_b(hs[4 * kc][0], hs[4 * kc + 1][0], 0x6420);
          pf[kc][1] = prmt_b(hs[4 * kc][1], hs[4 * kc + 1][1], 0x6420);
          pf[kc][2] = prmt_b(hs[4 * kc + 2][0], hs[4 * kc + 3][0], 0x6420);
          pf[kc][3] = prmt_b(hs[4 * kc + 2][1], hs[4 * kc + 3][1], 0x6420);
        }
        return;
#endif
#pragma unroll
        for (int kc = 0; kc < 4; ++kc) {
          pf[kc][0] = (u32)h2_to_e4m3x2(hs[4 * kc][0]) | ((u32)h2_to_e4m3x2(hs[4 * kc + 1][0]) << 16);
          pf[kc][1] = (u32)h2_to_e4m3x2(hs[4 * kc][1]) | ((u32)h2_to_e4m3x2(hs[4 * kc + 1][1]) << 16);
          pf[kc][2] = (u32)h2_to_e4m3x2(hs[4 * kc + 2][0]) | ((u32)h2_to_e4m3x2(hs[4 * kc + 3][0]) << 16);
          pf[kc][3] = (u32)h2_to_e4m3x2(hs[4 * kc + 2][1]) | ((u32)h2_to_e4m3x2(hs[4 * kc + 3][1]) << 16);
        }
      };
      auto pin_h = [&](u32 (&hs)[16][2]) {
#pragma unroll
        for (int i = 0; i < 16; ++i) { asm volatile("" : "+r"(hs[i][0]) :: "memory"); asm volatile("" : "+r"(hs[i][1]) :: "memory"); }
      };
      // lazy form (tiles after the first): P = 2^(S - m) with the current m straight away, then the
      // tile max of P; if a row's max passed 2^8 its P and O move to m' = m + log2(max P) - 8
      auto softmax_lazy = [&](float (&sacc)[16][4], int jt) {
        const int k0 = (bt + tfirst + jt) * 128;
        const bool masked = (k0 + 127 > pmin) || (p.window && k0 < pmax - (p.window - 1));
        if (masked) {
#pragma unroll
          for (int n8 = 0; n8 < 16; ++n8) {
            const int kc = k0 + n8 * 8 + 2 * tq;
            if (!(kc <= pa && kc >= wa_lo)) sacc[n8][0] = -__int_as_float(0x7f800000);
            if (!(kc + 1 <= pa && kc + 1 >= wa_lo)) sacc[n8][1] = -__int_as_float(0x7f800000);
            if (!(kc <= pb && kc >= wb_lo)) sacc[n8][2] = -__int_as_float(0x7f800000);
            if (!(kc + 1 <= pb && kc + 1 >= wb_lo)) sacc[n8][3] = -__int_as_float(0x7f800000);
          }
        }
        float ma4[4], mb4[4];
#pragma unroll
        for (int n8 = 0; n8 < 16; ++n8) {
          sacc[n8][0] = ex2_mufu(sacc[n8][0] - ma); sacc[n8][1] = ex2_mufu(sacc[n8][1] - ma);
          sacc[n8][2] = ex2_mufu(sacc[n8][2] - mb); sacc[n8][3] = ex2_mufu(sacc[n8][3] - mb);
          const float xa = fmaxf(sacc[n8][0], sacc[n8][1]), xb = fmaxf(sacc[n8][2], sacc[n8][3]);
          if (n8 < 4) { ma4[n8] = xa; mb4[n8] = xb; }
          else { ma4[n8 & 3] = fmaxf(ma4[n8 & 3], xa); mb4[n8 & 3] = fmaxf(mb4[n8 & 3], xb); }
        }
        float ta = fmaxf(fmaxf(ma4[0], ma4[1]), fmaxf(ma4[2], ma4[3]));
        float tb = fmaxf(fmaxf(mb4[0], mb4[1]), fmaxf(mb4[2], mb4[3]));
        ta = fmaxf(ta, __shfl_xor_sync(0xffffffff, ta, 1)); ta = fmaxf(ta, __shfl_xor_sync(0xffffffff, ta, 2));
        tb = fmaxf(tb, __shfl_xor_sync(0xffffffff, tb, 1)); tb = fmaxf(tb, __shfl_xor_sync(0xffffffff, tb, 2));
        const bool na = ta > 256.f, nb = tb > 256.f;
        if (__any_sync(0xffffffff, na || nb)) {
          float lga, lgb;
          asm("lg2.approx.f32 %0, %1;" : "=f"(lga) : "f"(ta));
          asm("lg2.approx.f32 %0, %1;" : "=f"(lgb) : "f"(tb));
          const float ma2 = na ? ma + lga - 8.f : ma, mb2 = nb ? mb + lgb - 8.f : mb;
          const float fa = ex2_mufu(ma - ma2), fb = ex2_mufu(mb - mb2);
#pragma unroll
          for (int n8 = 0; n8 < 16; ++n8) { sacc[n8][0] *= fa; sacc[n8][1] *= fa; sacc[n8][2] *= fb; sacc[n8][3] *= fb; }
          sca = fa; scb = fb;
          ma = ma2; mb = mb2;
          resc = true;
        }
      };
      // S -> P = 2^(S - m) in place (masked to 0), with the running max update
      auto softmax = [&](float (&sacc)[16][4], int jt) {
        const int k0 = (bt + tfirst + jt) * 128;
        const bool masked = (k0 + 127 > pmin) || (p.window && k0 < pmax - (p.window - 1));
        if (masked) {
#pragma unroll
          for (int n8 = 0; n8 < 16; ++n8) {
            const int kc = k0 + n8 * 8 + 2 * tq;
            if (!(kc <= pa && kc >= wa_lo)) sacc[n8][0] = -__int_as_float(0x7f800000);
            if (!(kc + 1 <= pa && kc + 1 >= wa_lo)) sacc[n8][1] = -__int_as_float(0x7f800000);
            if (!(kc <= pb && kc >= wb_lo)) sacc[n8][2] = -__int_as_float(0x7f800000);
            if (!(kc + 1 <= pb && kc + 1 >= wb_lo)) sacc[n8][3] = -__int_as_float(0x7f800000);
          }
        }
#ifdef ATT_NOMAX
        // ablation only (wrong results): a fixed offset instead of the running max
        if (jt == 0 && ma < -1e29f) { ma = 0.f; mb = 0.f; }
        if (false)
#endif
        {
        // row maxima: 4 independent chains per row (a single chain was 32 dependent FMNMX)
        float ma4[4], mb4[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          ma4[k] = fmaxf(sacc[k][0], sacc[k][1]);
          mb4[k] = fmaxf(sacc[k][2], sacc[k][3]);
        }
#pragma unroll
        for (int n8 = 4; n8 < 16; ++n8) {
          ma4[n8 & 3] = fmaxf(ma4[n8 & 3], fmaxf(sacc[n8][0], sacc[n8][1]));
          mb4[n8 & 3] = fmaxf(mb4[n8 & 3], fmaxf(sacc[n8][2], sacc[n8][3]));
        }
        float ta = fmaxf(fmaxf(ma4[0], ma4[1]), fmaxf(ma4[2], ma4[3]));
        float tb = fmaxf(fmaxf(mb4[0], mb4[1]), fmaxf(mb4[2], mb4[3]));
        ta = fmaxf(ta, __shfl_xor_sync(0xffffffff, ta, 1)); ta = fmaxf(ta, __shfl_xor_sync(0xffffffff, ta, 2));
        tb = fmaxf(tb, __shfl_xor_sync(0xffffffff, tb, 1)); tb = fmaxf(tb, __shfl_xor_sync(0xffffffff, tb, 2));
#if ATT_BIAS
        // ta, tb are maxima of S - m. Move m when the max passes 8, or first sees a finite value.
        const bool na = (ta > 8.f) || (!seta && ta > -1e30f), nb = (tb > 8.f) || (!setb && tb > -1e30f);
        resc = __any_sync(0xffffffff, na || nb);
        if (resc) {
          float ma2 = ma + (na ? ta - 8.f : 0.f), mb2 = mb + (nb ? tb - 8.f : 0.f);
          const u32 pa2 = bias_pair(ma2), pb2 = bias_pair(mb2);
          const float da = ma2 - ma, db = mb2 - mb;   // exact: both are representable offsets
          sca = ex2_mufu(-da); scb = ex2_mufu(-db);
          ma = ma2; mb = mb2;
          seta = seta || na; setb = setb || nb;
          bia = tq == 0 ? pa2 : 0u; bib = tq == 0 ? pb2 : 0u;
#pragma unroll
          for (int n8 = 0; n8 < 16; ++n8) { sacc[n8][0] -= da; sacc[n8][1] -= da; sacc[n8][2] -= db; sacc[n8][3] -= db; }
        }
#else
        const bool na = ta > ma + 8.f, nb = tb > mb + 8.f;
        resc = __any_sync(0xffffffff, na || nb);
        if (resc) {
          const float ma2 = na ? ta - 8.f : ma, mb2 = nb ? tb - 8.f : mb;
          sca = ex2_mufu(ma - ma2); scb = ex2_mufu(mb - mb2);
          ma = ma2; mb = mb2;
        }
#endif
        }
#if ATT_BIAS
        const float ea = 0.f, eb = 0.f;    // already offset
#else
        const float ea = ma, eb = mb;
#endif
        if (masked || !ATT_POLY) {
#pragma unroll
          for (int n8 = 0; n8 < 16; ++n8) {
            sacc[n8][0] = ex2_mufu(sacc[n8][0] - ea); sacc[n8][1] = ex2_mufu(sacc[n8][1] - ea);
            sacc[n8][2] = ex2_mufu(sacc[n8][2] - eb); sacc[n8][3] = ex2_mufu(sacc[n8][3] - eb);
          }
        } else {
#pragma unroll
          for (int n8 = 0; n8 < 16; ++n8) {
            if (ATT_POLY == 2 ? (n8 & 7) == 7 : ATT_POLY == 3 ? (n8 % 3) == 2 : (n8 & 3) == 3) {
              sacc[n8][0] = ex2_poly8(sacc[n8][0] - ma); sacc[n8][1] = ex2_poly8(sacc[n8][1] - ma);
              sacc[n8][2] = ex2_poly8(sacc[n8][2] - mb); sacc[n8][3] = ex2_poly8(sacc[n8][3] - mb);
            } else {
              sacc[n8][0] = ex2_mufu(sacc[n8][0] - ma); sacc[n8][1] = ex2_mufu(sacc[n8][1] - ma);
              sacc[n8][2] = ex2_mufu(sacc[n8][2] - mb); sacc[n8][3] = ex2_mufu(sacc[n8][3] - mb);
            }
          }
        }
      };
      // O was accumulated at the previous max: rescale once its PV has retired
      auto rescale_o = [&]() {
        if (resc) {
#pragma unroll
          for (int j = 0; j < 9; ++j) { o[j][0] *= sca; o[j][1] *= sca; o[j][2] *= scb; o[j][3] *= scb; }
          resc = false;
        }
      };
      // FP8 A fragment of k32 chunk kc: tile positions 32 kc + {4tq..4tq+3} -> columns {2tq, 2tq+1, 8+2tq, 9+2tq}
      auto to_p = [&](float (&sacc)[16][4]) {
#pragma unroll
        for (int kc = 0; kc < 4; ++kc) {
          pf[kc][0] = pack_e4m3x4(sacc[4 * kc][0], sacc[4 * kc][1], sacc[4 * kc + 1][0], sacc[4 * kc + 1][1]);
          pf[kc][1] = pack_e4m3x4(sacc[4 * kc][2], sacc[4 * kc][3], sacc[4 * kc + 1][2], sacc[4 * kc + 1][3]);
          pf[kc][2] = pack_e4m3x4(sacc[4 * kc + 2][0], sacc[4 * kc + 2][1], sacc[4 * kc + 3][0], sacc[4 * kc + 3][1]);
          pf[kc][3] = pack_e4m3x4(sacc[4 * kc + 2][2], sacc[4 * kc + 2][3], sacc[4 * kc + 3][2], sacc[4 * kc + 3][3]);
        }
      };
      auto pin_p = [&]() {
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int k = 0; k < 4; ++k) asm volatile("" : "+r"(pf[i][k]) :: "memory");
      };
#ifdef ATT_NOCOMP
      for (int j = 0; j < n; ++j) {
        mbar_wait(bars + 16 * ((it0 + j) % ATT_NS), ((it0 + j) / ATT_NS) & 1);
        mbar_arrive_if(bars + 16 * ((it0 + j) % ATT_NS) + 8, lt == 0);
      }
      la = lb = 1.f;
      if (false)
#endif
      // ping-pong: warpgroup c issues after c-1 (round robin over ATT_CW), named barriers 8 + c
      auto turn_begin = [&]() { if (ATT_PP) named_sync(8 + c, 256); };
      auto turn_end = [&]() { if (ATT_PP) named_arrive(8 + (c + 1) % ATT_CW, 256); };
      {
#if ATT_F16S == 2
        // double-buffered f16 scores: QK(j + 1) is in flight during softmax(j); exactly n QK turns
        // per segment as in the single-buffer loops. PV(n - 1) is left to the common tail below.
        u32 h0[16][2], h1[16][2];
        auto step = [&](u32 (&cur)[16][2], u32 (&nxt)[16][2], int j) {
          wg_fence();
          issue_pv(j);
          wg_wait<1>();
          pin_h(nxt);
          const bool more = j + 2 < n;
          if (more) {
            turn_begin();
            issue_qk_h(cur, j + 2);
            turn_end();
          }
          softmax_h(nxt, j + 1);
          if (more) wg_wait<1>(); else wg_wait<0>();
          pin(o);
          pin_p();
          mbar_arrive_if(bars + 16 * ((u32)(it0 + j) % ATT_NS) + 8, lt == 0);
          rescale_o();
          to_p_h(nxt);
        };
        turn_begin();
        issue_qk_h(h0, 0);
        turn_end();
        if (n > 1) {
          turn_begin();
          issue_qk_h(h1, 1);
          turn_end();
        }
        if (n > 1) wg_wait<1>(); else wg_wait<0>();
        pin_h(h0);
        softmax_h(h0, 0);
        rescale_o();
        to_p_h(h0);
        for (int j = 0; j < n - 1; j += 2) {
          step(h0, h1, j);
          if (j + 1 < n - 1) step(h1, h0, j + 1);
        }
        if (false) {
#elif ATT_F16S
        u32 hs[16][2];
        turn_begin();
        issue_qk_h(hs, 0);
        turn_end();
        wg_wait<0>();
        pin_h(hs);
        softmax_h(hs, 0);
        rescale_o();
        to_p_h(hs);
        for (int j = 1; j < n; ++j) {
          turn_begin();
          issue_qk_h(hs, j);
          if (ATT_PP == 2) turn_end();
          wg_fence();
          issue_pv(j - 1);
          if (ATT_PP != 2) turn_end();
          wg_wait<1>();
          pin_h(hs);
          softmax_h(hs, j);
          wg_wait<0>();
          pin(o);
          pin_p();
          mbar_arrive_if(bars + 16 * ((u32)(it0 + j - 1) % ATT_NS) + 8, lt == 0);
          rescale_o();
          to_p_h(hs);
        }
        if (false) {
#endif
        float sacc[16][4];
#if ATT_SPLIT
        // tile 0
        turn_begin();
        issue_qk2(sacc, 0);
        turn_end();
        wg_wait<1>();
        pin(sacc);
        softmax_half(sacc, 0, 0);
        wg_wait<0>();
        pin(sacc);
        softmax_half(sacc, 0, 1);
        rescale_o();
        to_p(sacc);
        // tile j: QK(j) halves | PV(j-1) | softmax(j, A) under QK(j, B) | softmax(j, B) | P(j)
        for (int j = 1; j < n; ++j) {
          turn_begin();
          issue_qk2(sacc, j);
          if (ATT_PP == 2) turn_end();
          wg_fence();
          issue_pv(j - 1);
          if (ATT_PP != 2) turn_end();
          wg_wait<2>();
          pin(sacc);
          softmax_half(sacc, j, 0);
          wg_wait<1>();
          pin(sacc);
          softmax_half(sacc, j, 1);
          wg_wait<0>();
          pin(o);
          pin_p();
          mbar_arrive_if(bars + 16 * ((u32)(it0 + j - 1) % ATT_NS) + 8, lt == 0);
          rescale_o();
          to_p(sacc);
        }
#else
        // tile 0
        turn_begin();
        issue_qk(sacc, 0);
        turn_end();
        wg_wait<0>();
        pin(sacc);
        softmax(sacc, 0);
        rescale_o();
        to_p(sacc);
        // tile j: QK(j) | PV(j-1) on P(j-1) | softmax(j) under PV(j-1) | P(j)
        for (int j = 1; j < n; ++j) {
          turn_begin();
          issue_qk(sacc, j);
          if (ATT_PP == 2) turn_end();   // PP 2: the turn covers only the QK issue
          wg_fence();
          issue_pv(j - 1);
          if (ATT_PP != 2) turn_end();
          wg_wait<1>();
          pin(sacc);
#if ATT_LAZY
          if (ma > -1e29f && mb > -1e29f) softmax_lazy(sacc, j); else softmax(sacc, j);
#else
          softmax(sacc, j);
#endif
          wg_wait<0>();
          pin(o);
          pin_p();
          mbar_arrive_if(bars + 16 * ((u32)(it0 + j - 1) % ATT_NS) + 8, lt == 0);
          rescale_o();
          to_p(sacc);
        }
#endif
#if ATT_F16S
        }
#endif
        if (ATT_PP != 2) turn_begin();
        wg_fence();
        issue_pv(n - 1);
        if (ATT_PP != 2) turn_end();
        wg_wait<0>();
        pin(o);
        pin_p();
        mbar_arrive_if(bars + 16 * ((u32)(it0 + n - 1) % ATT_NS) + 8, lt == 0);
      }
      // row sums: the ones rows (columns 64..71 of O)
      la = o[8][0]; lb = o[8][2];
      const size_t ta = (size_t)(b * p.L + ia), tb = (size_t)(b * p.L + ib);
      bool fin = sg.full != 0;
#ifdef ATT_NOCOMB
      fin = true;   // ablation: partial units write their own output, no combine (wrong output)
#endif
      if (!fin && p.unit == 2) {
        // split unit (every half has its own resident CTA): the low half hands its 2^m-scaled O and
        // row sums over with plain stores and a flag; the high half adds them and finishes
        const float fa = ex2_mufu(ma), fb = ex2_mufu(mb);
#pragma unroll
        for (int j = 0; j < 8; ++j) { o[j][0] *= fa; o[j][1] *= fa; o[j][2] *= fb; o[j][3] *= fb; }
        la *= fa; lb *= fb;
        float* oa = p.hacc + (ta * 16 + hg) * 64;
        float* ob = p.hacc + (tb * 16 + hg) * 64;
        int* cn = p.cnt + ((b * 2 + sg.h) * p.nqb + sg.qb) * ATT_CW + c;
        if (sg.t0 == 0) {
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const int d = j * 8 + 2 * tq;
            if (va) __stcg(reinterpret_cast<float2*>(oa + d), make_float2(o[j][0], o[j][1]));
            if (vb) __stcg(reinterpret_cast<float2*>(ob + d), make_float2(o[j][2], o[j][3]));
          }
          if (tq == 0) {
            if (va) __stcg(p.hl + ta * 16 + hg, la);
            if (vb) __stcg(p.hl + tb * 16 + hg, lb);
          }
          __threadfence();
          named_sync(1 + c, 128);
          if (lt == 0) asm volatile("st.release.gpu.global.b32 [%0], %1;" :: "l"(cn), "r"(1) : "memory");
        } else {
          if (lt == 0) {
            int v = 0;
            do { asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(cn) : "memory"); } while (v == 0);
            *cn = 0;
          }
          named_sync(1 + c, 128);
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const int d = j * 8 + 2 * tq;
            if (va) { const float2 v = __ldcg(reinterpret_cast<const float2*>(oa + d)); o[j][0] += v.x; o[j][1] += v.y; }
            if (vb) { const float2 v = __ldcg(reinterpret_cast<const float2*>(ob + d)); o[j][2] += v.x; o[j][3] += v.y; }
          }
          // every lane of a row group reads the row sums (tq == 0 wrote them)
          if (va) la += __ldcg(p.hl + ta * 16 + hg);
          if (vb) lb += __ldcg(p.hl + tb * 16 + hg);
          fin = true;
        }
      } else if (!fin) {
        // partial sums meet at an absolute scale, 2^m (QK-norm keeps |S| far below 2^7)
        const float fa = ex2_mufu(ma), fb = ex2_mufu(mb);
#pragma unroll
        for (int j = 0; j < 8; ++j) { o[j][0] *= fa; o[j][1] *= fa; o[j][2] *= fb; o[j][3] *= fb; }
        la *= fa; lb *= fb;
        float* oa = p.oacc + (ta * 16 + hg) * 64;
        float* ob = p.oacc + (tb * 16 + hg) * 64;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int d = j * 8 + 2 * tq;
          if (va) asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(oa + d), "f"(o[j][0]), "f"(o[j][1]) : "memory");
          if (vb) asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(ob + d), "f"(o[j][2]), "f"(o[j][3]) : "memory");
        }
        if (tq == 0) {
          if (va) atomicAdd(p.lacc + ta * 16 + hg, la);
          if (vb) atomicAdd(p.lacc + tb * 16 + hg, lb);
        }
        __threadfence();
        named_sync(1 + c, 128);
        if (lt == 0) {
          int* cn = p.cnt + ((b * 2 + sg.h) * p.nqb + sg.qb) * ATT_CW + c;
          const int nt = p.ctab[b * (p.nqb + 1) + sg.qi + 1] - p.ctab[b * (p.nqb + 1) + sg.qi];
          const int old = atomicAdd(cn, sg.n);
          const int last = (old + sg.n == nt);
          if (last) *cn = 0;
          flag[c] = last;
        }
        named_sync(1 + c, 128);
        fin = flag[c] != 0;
        if (fin) {
          __threadfence();
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const int d = j * 8 + 2 * tq;
            if (va) {
              float2 v = __ldcg(reinterpret_cast<const float2*>(oa + d));
              o[j][0] = v.x; o[j][1] = v.y;
              *reinterpret_cast<float2*>(oa + d) = make_float2(0.f, 0.f);
            }
            if (vb) {
              float2 v = __ldcg(reinterpret_cast<const float2*>(ob + d));
              o[j][2] = v.x; o[j][3] = v.y;
              *reinterpret_cast<float2*>(ob + d) = make_float2(0.f, 0.f);
            }
          }
          if (va) la = __ldcg(p.lacc + ta * 16 + hg);
          if (vb) lb = __ldcg(p.lacc + tb * 16 + hg);
          __syncwarp();
          if (tq == 0) {
            if (va) p.lacc[ta * 16 + hg] = 0.f;
            if (vb) p.lacc[tb * 16 + hg] = 0.f;
          }
        }
      }
      if (fin) {
        const float vsc = p.v_scale[0];
        const float ra = vsc / la, rb = vsc / lb;
        float sa = 0.f, sb = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          o[j][0] *= ra; o[j][1] *= ra; o[j][2] *= rb; o[j][3] *= rb;
          sa += o[j][0] * o[j][0] + o[j][1] * o[j][1];
          sb += o[j][2] * o[j][2] + o[j][3] * o[j][3];
        }
        sa += __shfl_xor_sync(0xffffffff, sa, 1); sa += __shfl_xor_sync(0xffffffff, sa, 2);
        sb += __shfl_xor_sync(0xffffffff, sb, 1); sb += __shfl_xor_sync(0xffffffff, sb, 2);
        const float na = rsqrtf(sa * (1.f / 64.f) + 1e-5f), nb = rsqrtf(sb * (1.f / 64.f) + 1e-5f);
        const __nv_bfloat16* gra = p.qkv + ta * 2304 + 1280 + hg * 64;
        const __nv_bfloat16* grb = p.qkv + tb * 2304 + 1280 + hg * 64;
        __nv_bfloat16* oua = p.out + ta * 1024 + hg * 64;
        __nv_bfloat16* oub = p.out + tb * 1024 + hg * 64;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int d = j * 8 + 2 * tq;
          if (va) {
            const __nv_bfloat162 gt = *reinterpret_cast<const __nv_bfloat162*>(gra + d);
            const float g0 = 1.f / (1.f + __expf(-__low2float(gt))), g1 = 1.f / (1.f + __expf(-__high2float(gt)));
            *reinterpret_cast<__nv_bfloat162*>(oua + d) = __floats2bfloat162_rn(o[j][0] * na * g0, o[j][1] * na * g1);
          }
          if (vb) {
            const __nv_bfloat162 gt = *reinterpret_cast<const __nv_bfloat162*>(grb + d);
            const float g0 = 1.f / (1.f + __expf(-__low2float(gt))), g1 = 1.f / (1.f + __expf(-__high2float(gt)));
            *reinterpret_cast<__nv_bfloat162*>(oub + d) = __floats2bfloat162_rn(o[j][2] * nb * g0, o[j][3] * nb * g1);
          }
        }
      }
    });
}
"""


class _AttCuda:
    """NVRTC-compiled module and driver-API launches on torch's current stream."""

    def __init__(self, src, defines=()):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        src = "".join(f"#define {d}\n" for d in defines) + src
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"attn.cu", 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        import os
        inc = [f"-I{p}".encode() for p in ("/usr/local/cuda/include",) if os.path.isdir(p)]
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17",
                b"-default-device"] + inc
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, size = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * size
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        (err,) = nvrtc.nvrtcGetCUBIN(prog, cubin)
        self.cubin = cubin
        err, self.module = driver.cuModuleLoadData(cubin)
        if int(err):
            raise RuntimeError(f"cuModuleLoadData failed: {err}")
        self.funcs = {}

    def func(self, name, smem=0):
        if name not in self.funcs:
            d = self.driver
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            if smem > 48 * 1024:
                d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
            self.funcs[name] = f
        return self.funcs[name]

    def launch(self, name, grid, block, smem, params, pdl=False):
        f = self.func(name, smem)
        stream = self.driver.CUstream(torch.cuda.current_stream().cuda_stream)
        argv = (ctypes.c_void_p * 1)(ctypes.addressof(params))
        if pdl and PDL:
            # programmatic dependent launch: the grid may start while the previous one finishes;
            # the kernel's griddepcontrol.wait orders its dependent reads
            d = self.driver
            cfg = d.CUlaunchConfig()
            cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
            cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block, 1, 1
            cfg.sharedMemBytes = smem
            cfg.hStream = stream
            at = d.CUlaunchAttribute()
            at.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
            at.value.programmaticStreamSerializationAllowed = 1
            cfg.attrs = [at]
            cfg.numAttrs = 1
            (err,) = d.cuLaunchKernelEx(cfg, f, ctypes.addressof(argv), 0)
        else:
            (err,) = self.driver.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        if int(err):
            raise RuntimeError(f"launch {name} failed: {err}")


class _AttP(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in (
        "qkv", "q8", "k_cache", "v_cache", "cache_loc", "page_table", "seq_lens", "k_scale", "v_scale",
        "inv_freq", "kp", "vp", "ctab", "out", "oacc", "lacc", "cnt")] + [
        (n, ctypes.c_int) for n in ("B", "L", "window", "rope", "page", "max_pages", "nta", "nqb")] + [
        ("qscale", ctypes.c_float), ("part", ctypes.c_int), ("rope_tab", ctypes.c_void_p), ("npos", ctypes.c_int), ("unit", ctypes.c_int), ("direct_new", ctypes.c_int),
        ("hacc", ctypes.c_void_p), ("hl", ctypes.c_void_p)]


ATT_MAX_T = 16384
QKV_POST2 = True
UNIT_SLIDING = True
PDL = True           # programmatic dependent launch for qkv_post2 and attn_fwd
UNIT_SPLIT_MAX = 2   # sliding units split up to this many ways while the grid has room
DIRECT_NEW = True    # after pre(): qkv_post2 packs the new keys, no tail kv_pack
DIRECT_NEW_MAX = 1024  # ... for blocks up to this many tokens (the byte scatter loses on larger ones)
INLINE_PACK_MIN = 0  # full layers with at least this many cache slots pack before the projection (0: never)
INLINE_PACK_MAX_T = 4096


class PrefillAttention:
    """State for the prefill attention path (built once in prepare)."""

    def __init__(self, device, defines=(), grid=None):
        self.mod = _AttCuda(ATT_CUDA_SRC, defines)
        self.ns = ATT_NS_DEF
        self.cw = 2
        for d in defines:
            if d.startswith("ATT_NS "):
                self.ns = int(d.split()[1])
            if d.startswith("ATT_CW "):
                self.cw = int(d.split()[1])
        self.qb = 8 * self.cw
        self.smem = self.ns * 18432 + 16 * self.ns + 64 + 1024 + 4096   # + the bias tile (ATT_BIAS)
        self.grid = grid or torch.cuda.get_device_properties(device).multi_processor_count
        self.oacc = torch.zeros((ATT_MAX_T, ATT_HQ, ATT_HD), dtype=torch.float32, device=device)
        self.lacc = torch.zeros((ATT_MAX_T, ATT_HQ), dtype=torch.float32, device=device)
        self.cnt = torch.zeros((4 * 2 * 1024 * 4,), dtype=torch.int32, device=device)
        self.hacc = torch.empty((2048, ATT_HQ, ATT_HD), dtype=torch.float32, device=device)
        self.hl = torch.empty((2048, ATT_HQ), dtype=torch.float32, device=device)
        theta = 500_000.0
        self.inv_freq = (1.0 / (theta ** (torch.arange(0, ATT_HD, 2, device=device).float() / ATT_HD))).contiguous()
        # RoPE table as SGLang builds it (FP32 positions * inv_freq, then cos, sin), interleaved pairs
        self.npos = 1 << 17
        freqs = torch.arange(self.npos, device=device, dtype=torch.float32)[:, None] * self.inv_freq[None, :]
        self.rope_tab = torch.stack((freqs.cos(), freqs.sin()), -1).contiguous()   # [npos, 32, 2]
        del freqs
        self.bufs = {}

    def _bufs(self, cfg, max_pages, page, device):
        B, L = cfg["batch"], cfg["q_len"]
        window = cfg["window"]
        if window:
            nta = (window - 1 + L + 127) // 128 + 1
        else:
            nta = (max_pages * page + 127) // 128 + 1
        key = (B, L, window, nta)
        if key not in self.bufs:
            T = B * L
            nqb = (L + self.qb - 1) // self.qb
            self.bufs[key] = dict(
                q8=torch.empty((T, ATT_HQ, ATT_HD), dtype=torch.uint8, device=device),
                kp=torch.empty((B, ATT_HKV, nta, ATT_KT, ATT_HD), dtype=torch.uint8, device=device),
                vp=torch.empty((B, ATT_HKV, nta, ATT_HD, ATT_KT), dtype=torch.uint8, device=device),
                ctab=torch.empty((B, nqb + 1), dtype=torch.int32, device=device),
                out=torch.empty((T, ATT_HQ * ATT_HD), dtype=torch.bfloat16, device=device),
                nta=nta, nqb=nqb)
        return self.bufs[key]

    def _direct(self, B, L):
        return DIRECT_NEW and B * L <= DIRECT_NEW_MAX

    def fork(self, data=None):
        """Mark the point the cached-key packing may start from (call before the QKV projection).
        Full layers with a long cached context pack right here instead, on the main stream: the
        projection leaves the side stream too few SMs, so its packing would end on the critical path."""
        self.inline = False
        if data is not None and INLINE_PACK_MIN > 0:
            cfg = data["config"]
            if not cfg["window"] and cfg["batch"] * cfg["q_len"] <= INLINE_PACK_MAX_T:
                ctx = data["page_table"].shape[1] * cfg["page_size"]
                if ctx >= INLINE_PACK_MIN:
                    prm, bf = self._params(data, None)
                    prm.part = 3 if self._direct(prm.B, prm.L) else 1
                    self.mod.launch("kv_pack", prm.B * 2 * bf["nta"] + prm.B, 256, 0, prm)
                    self.inline = True
                    self.fork_ev = None
                    return
        self.side = getattr(self, "side", None) or torch.cuda.Stream()
        self.fork_ev = torch.cuda.Event()
        self.fork_ev.record(torch.cuda.current_stream())

    def pre(self, data):
        """Pack the cached keys' tiles on a side stream. Enqueued after the QKV projection but
        depending only on the fork point, so the projection's CTAs take the SMs first and the
        packing fills its tail."""
        if getattr(self, "inline", False):
            self.pre_done = True
            return
        if getattr(self, "fork_ev", None) is None:
            self.fork()
        self.side.wait_event(self.fork_ev)
        self.fork_ev = None
        with torch.cuda.stream(self.side):
            prm, bf = self._params(data, None)
            prm.part = 3 if self._direct(prm.B, prm.L) else 1
            self.mod.launch("kv_pack", prm.B * 2 * bf["nta"] + prm.B, 256, 0, prm)
        self.pre_done = True

    def __call__(self, data, qkv):
        """qkv [T, 2304] BF16 (the projection) -> gated attention output [T, 1024] BF16; writes the
        new cache rows. After pre(data), only the tiles holding new keys are packed here."""
        prm, bf = self._params(data, qkv)
        T = prm.B * prm.L
        split = getattr(self, "pre_done", False)
        self.pre_done = False
        only = getattr(self, "only", None)   # timing breakdowns: "post", "pack" or "attn" alone
        if getattr(self, "only_attn", False):
            only = "attn"
        direct = split and self._direct(prm.B, prm.L)
        prm.direct_new = 1 if direct else 0
        if only in (None, "post"):
            if QKV_POST2:
                self.mod.launch("qkv_post2", (T + 7) // 8, 256, 0, prm, pdl=True)
            else:
                self.mod.launch("qkv_post", T, 640, 0, prm)
        if only in (None, "pack") and not direct:
            prm.part = 2 if split else 0
            self.mod.launch("kv_pack", prm.B * 2 * bf["nta"] + prm.B, 256, 0, prm)
        if split:
            if not getattr(self, "inline", False):
                torch.cuda.current_stream().wait_stream(self.side)
            self.inline = False
        if only in (None, "attn"):
            grid = min(self.grid, prm.B * 2 * bf["nqb"] * prm.unit) if prm.unit else self.grid
            self.mod.launch("attn_fwd", grid, 128 * (1 + self.cw), self.smem, prm, pdl=True)
        return bf["out"]

    def _params(self, data, qkv):
        cfg = data["config"]
        B, L = cfg["batch"], cfg["q_len"]
        T = B * L
        assert T <= ATT_MAX_T and B * 2 * ((L + self.qb - 1) // self.qb) * self.cw <= self.cnt.numel()
        page = cfg["page_size"]
        pt = data["page_table"]
        bf = self._bufs(cfg, pt.shape[1], page, data["k_cache"].device)
        prm = _AttP()
        prm.qkv, prm.q8 = (qkv.data_ptr() if qkv is not None else 0), bf["q8"].data_ptr()
        prm.k_cache, prm.v_cache = data["k_cache"].data_ptr(), data["v_cache"].data_ptr()
        prm.cache_loc, prm.page_table = data["cache_loc"].data_ptr(), pt.data_ptr()
        prm.seq_lens = data["seq_lens"].data_ptr()
        prm.k_scale, prm.v_scale = data["k_scale"].data_ptr(), data["v_scale"].data_ptr()
        prm.inv_freq = self.inv_freq.data_ptr()
        prm.kp, prm.vp, prm.ctab, prm.out = bf["kp"].data_ptr(), bf["vp"].data_ptr(), bf["ctab"].data_ptr(), bf["out"].data_ptr()
        prm.oacc, prm.lacc, prm.cnt = self.oacc.data_ptr(), self.lacc.data_ptr(), self.cnt.data_ptr()
        prm.hacc, prm.hl = self.hacc.data_ptr(), self.hl.data_ptr()
        prm.B, prm.L, prm.window, prm.rope = B, L, cfg["window"], 1 if cfg["rope_theta"] else 0
        prm.page, prm.max_pages, prm.nta, prm.nqb = page, pt.shape[1], bf["nta"], bf["nqb"]
        prm.qscale = float(cfg["sm_scale"] * cfg["qk_scale_factor"] * ATT_LOG2E)
        assert cfg["rope_theta"] in (0.0, 500_000.0)
        prm.part = 0
        prm.rope_tab, prm.npos = self.rope_tab.data_ptr(), self.npos
        # sliding blocks all see about the same 2K keys: whole units, no partial sums
        prm.unit = 0
        if cfg["window"] and UNIT_SLIDING:
            units = B * 2 * ((L + self.qb - 1) // self.qb)
            prm.unit = max(1, min(UNIT_SPLIT_MAX, self.grid // units)) if UNIT_SPLIT_MAX > 1 else 1
            if prm.unit > 1 and T > self.hacc.shape[0]:
                prm.unit = 1
        return prm, bf


# ---------------------------------------------------------------------------------------------
# Prefill attention (own kernels): replaces FA3 + the output norm/gate. QK-norm/RoPE/FP8 cache store,
# KV packing and a wgmma attention kernel with the output norm and gate fused.
ATT_DEFINES = ['ATT_CW 3', 'ATT_POLY 0', 'ATT_PP 2', 'ATT_F16S 1', 'ATT_EXPQ 1']
ATT_SLIDING_MIN = 0   # sliding layers this long take the new kernels too
_att_base_prepare = prepare


def prepare(weights):
    state = _att_base_prepare(weights)
    state["att"] = PrefillAttention(weights["qkv_gate"].device, ATT_DEFINES)
    torch.cuda.synchronize()
    return state

# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: SGLang contributors <https://github.com/sgl-project/sglang>
# SPDX-License-Identifier: Apache-2.0
# Adapted from the OKC starter's muse_prefill_layer/tools/baseline.py (SGLang's MuseSpark1xDecoderLayer, sgl-project/sglang @ a55ab7d), `kernel_fn`.
# Changes: keeps the baseline's step order and its fallback path (pooler GEMM, router, routing, pre-expert norm, unpooler, both boundaries); adds our attention, our fused boundary kernels and pooler/router GEMM plans whose fused routing kernels bucket routes for our expert path.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    dense = state["dense"]
    # Attention: MuseSpark1xAttention.forward.
    use_att = not cfg["window"] or cfg["tokens"] >= ATT_SLIDING_MIN
    if use_att:
        state["att"].fork(data)
    qkv = _gemm(data["attn_input"], dense["qkv_gate"])
    if use_att:
        state["att"].pre(data)  # cached-key packing beside the projection
    o = (state["att"](data, qkv) if (not cfg["window"] or tokens >= ATT_SLIDING_MIN) else optimized_attention(data, qkv, custom_window=CUSTOM_WINDOW).reshape(tokens, hq * d))
    attn_output = _gemm(o, dense["o"])  # all-reduce: identity

    # Post-attention boundary, router and MoE block: MuseSpark1xMoE._forward_experts.
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    s0 = data["stream"]
    fast_bnd = split_router and s0.is_contiguous() and attn_output.is_contiguous()
    if fast_bnd:
        ffn_input = _bnd1(state, s0, attn_output)
        stream = router_input = None
    else:
        ffn_input, stream, router_input = state["attn_boundary"](
            s0, attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
    dev = ffn_input.device
    bucketed = False
    if split_router and tokens <= FUSED_ROUTER_MAX:
        # one GEMM: pooler and router (hi, lo) columns, FP32 out; then one pass: pre-expert norm + routing
        plan, index = _fused_plan(state, tokens)
        y = plan.run(ffn_input.contiguous(), state["pool_route_w"], index)
        expert_input = torch.empty((tokens, D_EXPERT), dtype=torch.bfloat16, device=dev)
        topk_weights = torch.empty((tokens, TOP_K), dtype=torch.float32, device=dev)
        topk_ids = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
        local_ids = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
        nx = state["native_experts"]
        prm = _RouteP(y.data_ptr(), D_EXPERT + 512, state["pre_expert_w32"].data_ptr(), expert_input.data_ptr(),
                      state["router_bias32"].data_ptr(), topk_weights.data_ptr(), topk_ids.data_ptr(),
                      local_ids.data_ptr(), data["local_start"].data_ptr(), ROUTER_MULT, RMS_EPS, tokens,
                      nx["cur"].data_ptr(), nx["bkt"].data_ptr(), nx["spos"].data_ptr())
        state["bnd_mod"].launch("pnr", tokens, 256, 0, prm, pdl=True)
        bucketed = True
    elif split_router and tokens in state["router512_plans"] and tokens <= MAX_M:
        # one router GEMM (hi | lo), then one pass: pre-expert norm of the pooled rows + routing + buckets
        pooled = _gemm(ffn_input, dense["pooler"])
        plan, index = state["router512_plans"][tokens]
        lg = plan.run(ffn_input.contiguous(), state["router_hilo"], index)
        expert_input = torch.empty((tokens, D_EXPERT), dtype=torch.bfloat16, device=dev)
        topk_weights = torch.empty((tokens, TOP_K), dtype=torch.float32, device=dev)
        topk_ids = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
        local_ids = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
        nx = state["native_experts"]
        prm = _RouteP(lg.data_ptr(), 512, state["pre_expert_w32"].data_ptr(), expert_input.data_ptr(),
                      state["router_bias32"].data_ptr(), topk_weights.data_ptr(), topk_ids.data_ptr(),
                      local_ids.data_ptr(), data["local_start"].data_ptr(), ROUTER_MULT, RMS_EPS, tokens,
                      nx["cur"].data_ptr(), nx["bkt"].data_ptr(), nx["spos"].data_ptr(), pooled.data_ptr())
        state["bnd_mod"].launch("pnr2", tokens, 256, 0, prm)
        bucketed = True
    else:
        expert_input = _gemm(ffn_input, dense["pooler"])
        if split_router and tokens in state["router_plans"]:
            p1, i1, p2, i2 = state["router_plans"][tokens]
            x = ffn_input.contiguous()
            logits = p2.run(x, state["router_lo"], i2, p1.run(x, state["router_hi"], i1))
        elif split_router:
            logits = _layer_router(ffn_input, state)
        else:
            logits = F.linear(router_input, state["router"])
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
        local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
        expert_input = state["pre_expert_norm"](expert_input)
    # all-reduce: identity
    expert_output = _experts(expert_input, topk_weights, local_ids, state, bucketed=bucketed)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = _gemm(expert_output, dense["unpooler"])

    # Post-FFN boundary, with the next layer's pre-attention norm.
    if fast_bnd and ffn_output.is_contiguous():
        next_attn_input, stream = _bnd2(state, s0, attn_output, ffn_output)
    else:
        if stream is None:
            _, stream, _ = state["attn_boundary"](s0, attn_output, branch_dtype=torch.bfloat16, router_dtype=None)
        next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
# SPDX-SnippetEnd
