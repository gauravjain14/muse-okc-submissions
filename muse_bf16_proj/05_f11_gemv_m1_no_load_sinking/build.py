# okc: problem=muse_bf16_proj
# Third-party code: each segment copied or adapted from another project is marked in place with SPDX-SnippetBegin /
# SPDX-SnippetEnd comments that give its source, copyright, license and our changes. Full license texts are in
# LICENSES/ and the third-party notices in NOTICE, both at the repository root.

"""muse_bf16_proj top-3 build (claude/t3/build_t3.py): int_v7 base + t3 agent kernels."""

import torch

# ======================================================================== base: int_v7

"""muse_bf16_proj: Codex v2 paths + K1s (12-bit lossless weights, M <= 16) + fast router. Built by claude/build_sub.py."""

# ======================================================================== component: Codex v2
# Independent BF16 projection candidate, revision 2.
import torch


# Source module: proj_triton
"""Fresh BF16 projection candidates; all reductions accumulate in FP32."""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _b7__cx_pr_gemv(X, W, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
         BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0)*BN + tl.arange(0, BN)
    token = tl.program_id(1)
    kk = tl.arange(0, BK)
    acc = tl.full((BN, BK), 0, tl.float32)
    for base in range(tr.cdiv(K, BK)):
        k = base*BK + kk
        x = tl.load(X + token*K + k, k<K, 0).to(tl.float32)
        w = tl.load(W + rows[:,None]*K + k[None,:], (rows[:,None]<N)&(k[None,:]<K), 0).to(tl.float32)
        acc += w*x[None,:]
    out = tl.sum(acc,1)
    tl.store(Y + token*N + rows, out, rows<N)

@tr.jit
def _b7__cx_pr_matmul_transpose(X,W,Y,P,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                     BN:tl.constexpr,BM:tl.constexpr,BK:tl.constexpr,SK:tl.constexpr):
    # W @ X.T, putting token dimension on the narrow tensor-core axis.
    rn=tl.program_id(0)*BN+tl.arange(0,BN)
    rm=tl.program_id(1)*BM+tl.arange(0,BM)
    rk=tl.arange(0,BK)
    split=tl.program_id(2)
    chunk:tl.constexpr=tr.cdiv(K,SK*BK)*BK
    acc=tl.zeros((BN,BM),tl.float32)
    for i in range(tr.cdiv(K,SK*BK)):
        k=split*chunk+i*BK+rk
        a=tl.load(W+rn[:,None]*K+k[None,:],(rn[:,None]<N)&(k[None,:]<K),0)
        b=tl.load(X+rm[None,:]*K+k[:,None],(rm[None,:]<M)&(k[:,None]<K),0)
        acc=tl.dot(a,b,acc)
    if SK==1:
        tl.store(Y+rm[None,:]*N+rn[:,None],acc,(rn[:,None]<N)&(rm[None,:]<M))
    else:
        tl.store(P+split*M*N+rm[None,:]*N+rn[:,None],acc,(rn[:,None]<N)&(rm[None,:]<M))

@tr.jit
def _b7__cx_pr_reduce_split(P,Y,SIZE:tl.constexpr,SK:tl.constexpr,B:tl.constexpr):
    ix=tl.program_id(0)*B+tl.arange(0,B)
    out=tl.full((B,),0,tl.float32)
    for s in range(SK):
        out += tl.load(P+s*SIZE+ix,ix<SIZE,0)
    tl.store(Y+ix,out,ix<SIZE)

def _b7__cx_pr_run(x,w,n,config):
    m,k=x.shape
    y=torch.empty((m,n),dtype=torch.bfloat16,device=x.device)
    if config[0]=='gemv':
        _,bn,bk,nw=config
        _b7__cx_pr_gemv[(tr.cdiv(n,bn),m)](x,w,y,m,n,k,bn,bk,num_warps=nw)
    else:
        _,bn,bm,bk,sk,nw,ns=config
        p=torch.empty((sk,m,n),dtype=torch.float32,device=x.device) if sk>1 else y
        _b7__cx_pr_matmul_transpose[(tr.cdiv(n,bn),tr.cdiv(m,bm),sk)](x,w,y,p,m,n,k,bn,bm,bk,sk,num_warps=nw,num_stages=ns)
        if sk>1:
            _b7__cx_pr_reduce_split[(tr.cdiv(m*n,256),)](p,y,m*n,sk,256)
    return y

def _b7__cx_pr_configs(m,n,k,quick=False):
    if m==1:
        return [('gemv',bn,bk,nw) for bn,bk,nw in [(4,512,4),(4,1024,4),(8,512,4),(8,1024,8)]]
    if quick:
        return [('mm',64,16 if m<=32 else 64,64,sk,4,3) for sk in [1,4]]
    opts=[]
    for bm in ([16 if m<=16 else 32] if m<=32 else [32,64]):
        for bn in ([16,32,64] if m<=32 else [32,64,128]):
            for bk in [64,128,256]:
                for sk in ([1,2,4,8] if m<=32 else [1,2,4]):
                    if sk*bk>k:continue
                    opts.append(('mm',bn,bm,bk,sk,4,3))
    return opts


def _b7__cx_pr_configs_wide(m,n,k,quick=False):
    # First pooler NCU profile identified weight-load dependency stalls.
    # More K in registers / one-iteration partitions test additional ILP.
    if m==1:
        return [('gemv',bn,bk,nw) for bn in (1,2,4,8)
                for bk in (1024,2048,4096,8192) if bk<=k
                for nw in (4,8) if bn*bk/nw<=4096]
    bm=16 if m<=16 else 32
    return [('mm',bn,bm,bk,max(1,k//bk),nw,1)
            for bn in (16,32,64) for bk in (256,512,1024)
            for nw in (4,8) if (bn+bm)*bk*2<=160*1024]


# Source module: router_candidates
"""Fresh H200 router candidates for the public muse_bf16_proj specification.

API:
    state = prepare_router(weight_fp32)  # untimed, weight-only preparation
    output_fp32 = run_router(x_bf16, state, config_tuple)

Configuration tuples:
    ("simt", block_n, block_k, split_k, num_warps)
    ("dot", block_m, block_n, block_k, split_k, num_warps,
     num_stages, swap_ab)

The SIMT path retains the original FP32 weights. The dot path uses the
explicitly permitted BF16 hi + lo representation and FP32 accumulation.
Both paths mask arbitrary M/N/K boundaries. Split-K uses a separate FP32
workspace and deterministic reduction; every call overwrites its workspace.
This module performs no benchmarking, profiling, or remote submissions.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _b7__cx_rt__router_simt(
    X, W, Y, PARTIAL,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    SX0: tl.constexpr, SX1: tl.constexpr,
    SW0: tl.constexpr, SW1: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    ns = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row = tl.program_id(1)
    split = tl.program_id(2)
    local_k = tl.arange(0, BLOCK_K)
    accum = tl.full((BLOCK_N, BLOCK_K), 0, tl.float32)
    for step in range(tl.cdiv(K, BLOCK_K * SPLIT_K)):
        ks = (step * SPLIT_K + split) * BLOCK_K + local_k
        x = tl.load(X + row * SX0 + ks * SX1, ks < K, 0).to(tl.float32)
        w = tl.load(
            W + ns[:, None] * SW0 + ks[None, :] * SW1,
            (ns[:, None] < N) & (ks[None, :] < K),
            0,
        )
        accum = tl.fma(w, x[None, :], accum)
    result = tl.sum(accum, axis=1)
    if SPLIT_K == 1:
        tl.store(Y + row * N + ns, result, ns < N)
    else:
        tl.store(PARTIAL + split * M * N + row * N + ns, result, ns < N)


@triton.jit
def _b7__cx_rt__router_dot(
    X, HI, LO, Y, PARTIAL,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    SX0: tl.constexpr, SX1: tl.constexpr,
    SW0: tl.constexpr, SW1: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
    SWAP_AB: tl.constexpr,
):
    tile = tl.program_id(0)
    split = tl.program_id(1)
    n_tiles = tl.cdiv(N, BLOCK_N)
    ms = (tile // n_tiles) * BLOCK_M + tl.arange(0, BLOCK_M)
    ns = (tile % n_tiles) * BLOCK_N + tl.arange(0, BLOCK_N)
    local_k = tl.arange(0, BLOCK_K)

    if SWAP_AB:
        accum_hi = tl.full((BLOCK_N, BLOCK_M), 0, tl.float32)
        accum_lo = tl.full((BLOCK_N, BLOCK_M), 0, tl.float32)
    else:
        accum_hi = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
        accum_lo = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)

    for step in range(tl.cdiv(K, BLOCK_K * SPLIT_K)):
        ks = (step * SPLIT_K + split) * BLOCK_K + local_k
        if SWAP_AB:
            w_offsets = ns[:, None] * SW0 + ks[None, :] * SW1
            w_mask = (ns[:, None] < N) & (ks[None, :] < K)
            hi = tl.load(HI + w_offsets, w_mask, 0)
            lo = tl.load(LO + w_offsets, w_mask, 0)
            x = tl.load(
                X + ms[None, :] * SX0 + ks[:, None] * SX1,
                (ms[None, :] < M) & (ks[:, None] < K),
                0,
            )
            accum_hi = tl.dot(hi, x, accum_hi)
            accum_lo = tl.dot(lo, x, accum_lo)
        else:
            x = tl.load(
                X + ms[:, None] * SX0 + ks[None, :] * SX1,
                (ms[:, None] < M) & (ks[None, :] < K),
                0,
            )
            w_offsets = ns[None, :] * SW0 + ks[:, None] * SW1
            w_mask = (ns[None, :] < N) & (ks[:, None] < K)
            hi = tl.load(HI + w_offsets, w_mask, 0)
            lo = tl.load(LO + w_offsets, w_mask, 0)
            accum_hi = tl.dot(x, hi, accum_hi)
            accum_lo = tl.dot(x, lo, accum_lo)

    result = accum_hi + accum_lo
    if SWAP_AB:
        offsets = ms[None, :] * N + ns[:, None]
        mask = (ms[None, :] < M) & (ns[:, None] < N)
    else:
        offsets = ms[:, None] * N + ns[None, :]
        mask = (ms[:, None] < M) & (ns[None, :] < N)
    if SPLIT_K == 1:
        tl.store(Y + offsets, result, mask)
    else:
        tl.store(PARTIAL + split * M * N + offsets, result, mask)


@triton.jit
def _b7__cx_rt__router_reduce(
    PARTIAL, Y,
    ELEMENTS: tl.constexpr, SPLIT_K: tl.constexpr,
    BLOCK_SPLIT: tl.constexpr, BLOCK: tl.constexpr,
):
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, BLOCK_SPLIT)
    partial = tl.load(
        PARTIAL + splits[:, None] * ELEMENTS + indices[None, :],
        (splits[:, None] < SPLIT_K) & (indices[None, :] < ELEMENTS),
        0,
    )
    tl.store(Y + indices, tl.sum(partial, axis=0), indices < ELEMENTS)


def _b7__cx_rt_prepare_router(w):
    """Keep exact FP32 weights and prepare the permitted BF16 hi/lo pair."""
    if w.ndim != 2 or w.dtype != torch.float32:
        raise ValueError("router weights must be a two-dimensional FP32 tensor")
    w = w.contiguous()
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)
    return {"w": w, "hi": hi, "lo": lo, "n": w.shape[0], "k": w.shape[1]}


def _b7__cx_rt__check_power_of_two(value, name, minimum=1):
    if not isinstance(value, int) or value < minimum or value & (value - 1):
        raise ValueError(f"{name} must be a power of two >= {minimum}")


def _b7__cx_rt_run_router(x, state, config_tuple):
    """Compute [M, N] FP32 router logits; configuration depends only on shape."""
    if x.ndim != 2 or x.dtype != torch.bfloat16:
        raise ValueError("router activations must be a two-dimensional BF16 tensor")
    m, k = x.shape
    n = state["n"]
    if k != state["k"] or x.device != state["w"].device:
        raise ValueError("router activation shape/device does not match prepared weights")
    if not config_tuple:
        raise ValueError("router config tuple must not be empty")
    kind = config_tuple[0]
    if kind == "simt" and len(config_tuple) == 5:
        _, bn, bk, splits, warps = config_tuple
        _b7__cx_rt__check_power_of_two(bn, "block_n")
        _b7__cx_rt__check_power_of_two(bk, "block_k")
    elif kind == "dot" and len(config_tuple) == 8:
        _, bm, bn, bk, splits, warps, stages, swap = config_tuple
        _b7__cx_rt__check_power_of_two(bm, "block_m", 16)
        _b7__cx_rt__check_power_of_two(bn, "block_n", 16)
        _b7__cx_rt__check_power_of_two(bk, "block_k", 16)
        if not isinstance(stages, int) or stages < 1:
            raise ValueError("num_stages must be a positive integer")
    else:
        raise ValueError(f"invalid router config: {config_tuple!r}")
    if not isinstance(splits, int) or splits < 1:
        raise ValueError("split_k must be a positive integer")
    _b7__cx_rt__check_power_of_two(warps, "num_warps")

    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    if m == 0 or n == 0:
        return out
    partial = (
        torch.empty((splits, m, n), dtype=torch.float32, device=x.device)
        if splits > 1 else out
    )
    if kind == "simt":
        w = state["w"]
        _b7__cx_rt__router_simt[(triton.cdiv(n, bn), m, splits)](
            x, w, out, partial, m, n, k,
            x.stride(0), x.stride(1), w.stride(0), w.stride(1),
            BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=splits,
            num_warps=warps,
        )
    else:
        hi, lo = state["hi"], state["lo"]
        _b7__cx_rt__router_dot[(triton.cdiv(m, bm) * triton.cdiv(n, bn), splits)](
            x, hi, lo, out, partial, m, n, k,
            x.stride(0), x.stride(1), hi.stride(0), hi.stride(1),
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
            SPLIT_K=splits, SWAP_AB=bool(swap),
            num_warps=warps, num_stages=stages,
        )
    if splits > 1:
        _b7__cx_rt__router_reduce[(triton.cdiv(m * n, 256),)](
            partial, out, m * n, SPLIT_K=splits,
            BLOCK_SPLIT=triton.next_power_of_2(splits), BLOCK=256,
            num_warps=4,
        )
    return out


def _b7__cx_rt_router_configs(m):
    """Full sweep including longer K tiles; the quick sweep stays unchanged."""
    simt = [
        ("simt", 1, 256, 1, 4),
        ("simt", 2, 256, 1, 4),
        ("simt", 4, 512, 1, 4),
        ("simt", 4, 512, 4, 4),
        ("simt", 8, 512, 4, 4),
    ] if m <= 16 else []
    if m <= 16:
        # Single-kernel variants reduce K-loop trips without a partial/reduce
        # launch. Keep both warp counts to measure the larger tiles' occupancy.
        simt += [
            ("simt", bn, bk, 1, warps)
            for bk in (1024, 2048, 4096, 8192)
            for bn in (1, 2, 4)
            for warps in (4, 8)
        ]
    dot = [
        ("dot", 16, 32, 64, 4, 4, 2, False),
        ("dot", 16, 32, 64, 8, 4, 2, False),
        ("dot", 16, 32, 64, 16, 4, 2, False),
        ("dot", 16, 64, 64, 8, 4, 2, True),
        ("dot", 16, 64, 128, 8, 4, 3, True),
        ("dot", 32, 32, 64, 4, 4, 3, False),
        ("dot", 32, 64, 64, 4, 4, 3, True),
    ]
    if m >= 32:
        dot += [
            ("dot", 32, 64, 64, 1, 4, 3, False),
            ("dot", 32, 64, 64, 2, 4, 3, False),
            ("dot", 32, 64, 64, 4, 4, 3, False),
            ("dot", 64, 64, 64, 4, 4, 3, False),
        ]
    if m >= 16:
        bm = 16 if m < 32 else 32
        dot += [
            ("dot", bm, bn, bk, splits, 4, 2, swap)
            for bn in (32, 64)
            for bk in (128, 256)
            for splits in (8, 16, 32)
            for swap in (False, True)
        ]
    return simt + dot


def _b7__cx_rt_configs(m, n, k, quick=False):
    """Benchmark-compatible shape API; quick sweeps contain at most four tuples."""
    if not quick:
        return _b7__cx_rt_router_configs(m)
    if m == 1:
        return [
            ("simt", 1, 256, 1, 4),
            ("simt", 4, 512, 4, 4),
            ("dot", 16, 32, 64, 8, 4, 2, False),
            ("dot", 16, 64, 64, 8, 4, 2, True),
        ]
    if m <= 16:
        return [
            ("simt", 2, 256, 1, 4),
            ("dot", 16, 32, 64, 4, 4, 2, False),
            ("dot", 16, 32, 64, 8, 4, 2, False),
            ("dot", 16, 64, 64, 8, 4, 2, True),
        ]
    return [
        ("dot", 32, 64, 64, 1, 4, 3, False),
        ("dot", 32, 64, 64, 2, 4, 3, False),
        ("dot", 32, 64, 64, 4, 4, 3, False),
        ("dot", 32, 64, 64, 4, 4, 3, True),
    ]


# Source module: tinygemm_candidate
"""Hopper BF16 TinyGEMM candidate adapted from NVIDIA TensorRT-LLM.

Upstream Apache-2.0 sources (original downloads retained in upstream/tinygemm2):
https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/kernels/tinygemm2
Original kernel SHA256: 1891378515dbbc60167531dfab9243cd647ff367daf75fa13b0132f29792a3d8
Original launcher SHA256: 762f7d546b14cd8d64ea5afea9ce44ab01b31d3af7e54552217a9ab295dfdc4d

Changes: remove zero bias and PDL; expose pipeline depth and optional token-tile
interleaving; use the Hopper shared::cluster TMA encoding and an inline PTX
proxy fence; use a checked PyTorch launcher.
BF16 inputs, FP32 MMA accumulation,
BF16 output. No weight repacking or input-dependent preparation. The CUDA source
is embedded so the eventual single Python submission needs no external files.

Config: ("tiny", stages, interleave), stages in {4, 8, 16}, STAGE_UNROLL=4.
run(x, w, n, config) returns BF16 [x.shape[0], n]. The native layout requires
row-contiguous inputs and K divisible by 64, covering all public projection K's.
"""

from functools import lru_cache

from torch.utils.cpp_extension import load_inline


_b7__cx_ti__CUDA_SOURCE = r"""
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: 2025 NVIDIA CORPORATION <https://github.com/NVIDIA/TensorRT-LLM>
// SPDX-License-Identifier: Apache-2.0
// Adapted from NVIDIA/TensorRT-LLM, cpp/tensorrt_llm/kernels/tinygemm2/tinygemm2_kernel.cuh, `tinygemm_kernel` and its helper functions.
// Changes: removed the bias input and the PDL grid-dependency wait and launch-completion trigger, added an INTERLEAVE template flag that swaps the channel-tile and token-tile grid axes, switched both TMA loads to the shared::cluster destination form, replaced the cuda::ptx proxy fence with inline PTX, and dropped the cuda::ptx / cde namespace aliases.
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
/*
 * Copyright (c) 2025-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cuda_bf16.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>

#include "cuda_pipeline.h"
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/std/utility>
#include <cuda_runtime.h>

using barrier = cuda::barrier<cuda::thread_scope_block>;

#define gpuErrChk(ans)                                                                                                 \
    {                                                                                                                  \
        gpuAssert((ans), __FILE__, __LINE__);                                                                          \
    }

inline void gpuAssert(cudaError_t code, char const* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
__device__ uint64_t gclock64()
{
    unsigned long long int rv;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(rv));
    return rv;
}

__device__ void ldmatrix(__nv_bfloat16 rv[2], uint32_t smem_ptr)
{
    int dst;
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(dst) : "r"(smem_ptr));
    int* rvi = reinterpret_cast<int*>(&rv[0]);
    rvi[0] = dst;
}

__device__ void ldmatrix2(__nv_bfloat16 rv[4], uint32_t smem_ptr)
{
    int x, y;
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(x), "=r"(y) : "r"(smem_ptr));

    int* rvi = reinterpret_cast<int*>(&rv[0]);
    rvi[0] = x;
    rvi[1] = y;
}

__device__ void ldmatrix4(__nv_bfloat16 rv[8], uint32_t smem_ptr)
{
    int x, y, z, w;
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                 : "r"(smem_ptr));
    int* rvi = reinterpret_cast<int*>(&rv[0]);
    rvi[0] = x;
    rvi[1] = y;
    rvi[2] = z;
    rvi[3] = w;
}

__device__ void HMMA_1688(float d[4], __nv_bfloat16 a[4], __nv_bfloat16 b[2], float c[4])
{

    uint32_t const* A = reinterpret_cast<uint32_t const*>(&a[0]);
    uint32_t const* B = reinterpret_cast<uint32_t const*>(&b[0]);
    float const* C = reinterpret_cast<float const*>(&c[0]);
    float* D = reinterpret_cast<float*>(&d[0]);

    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B[0]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
}

__device__ void HMMA_16816(float d[4], __nv_bfloat16 a[8], __nv_bfloat16 b[4], float c[4])
{
    uint32_t const* A = reinterpret_cast<uint32_t const*>(&a[0]);
    uint32_t const* B = reinterpret_cast<uint32_t const*>(&b[0]);
    float const* C = reinterpret_cast<float const*>(&c[0]);
    float* D = reinterpret_cast<float*>(&d[0]);

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
}

__device__ void bar_wait(uint32_t bar_ptr, int phase)
{
    asm volatile(
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n" ::"r"(bar_ptr),
        "r"(phase));
}

__device__ bool bar_try_wait(uint32_t bar_ptr, int phase)
{
    uint32_t success;
#ifdef INTERNAL
    asm volatile(".pragma \"set knob DontInsertYield\";\n" : : : "memory");
#endif
    asm volatile(
        "{\n\t"
        ".reg .pred P1; \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
        "selp.b32 %0, 1, 0, P1; \n\t"
        "}"
        : "=r"(success)
        : "r"(bar_ptr), "r"(phase));
    return success;
}

__device__ uint32_t elect_one_sync()
{
    uint32_t pred = 0;
    uint32_t laneid = 0;
    asm volatile(
        "{\n"
        ".reg .b32 %%rx;\n"
        ".reg .pred %%px;\n"
        "     elect.sync %%rx|%%px, %2;\n"
        "@%%px mov.s32 %1, 1;\n"
        "     mov.s32 %0, %%rx;\n"
        "}\n"
        : "+r"(laneid), "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}
#endif

struct Profile
{
    uint64_t start;
    uint64_t weight_load_start;
    uint64_t act_load_start;
    uint64_t compute_start;
    uint64_t complete;
};

template <int WARP_TILE_M, int TILE_M, int TILE_N, int TILE_K, int STAGES, int STAGE_UNROLL, bool PROFILE, bool INTERLEAVE>
__global__ __launch_bounds__(384, 1) void tinygemm_kernel(__nv_bfloat16* output, __nv_bfloat16* weights,
    __nv_bfloat16* activations, int M, int N, int K,
    const __grid_constant__ CUtensorMap weight_map, const __grid_constant__ CUtensorMap activation_map,
    Profile* profile = nullptr)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

    if (PROFILE && threadIdx.x == 0 && blockIdx.y == 0)
        profile[blockIdx.x].start = gclock64();

    extern __shared__ __align__(128) char smem[];

    __nv_bfloat16* sh_weights = (__nv_bfloat16*) &smem[0];
    __nv_bfloat16* sh_activations
        = (__nv_bfloat16*) &smem[STAGES * STAGE_UNROLL * TILE_M * TILE_K * sizeof(__nv_bfloat16)];

#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar_wt_ready[STAGES];
    __shared__ barrier bar_act_ready[STAGES];
    __shared__ barrier bar_data_consumed[STAGES];

    __shared__ float4 reduction_buffer[128];


    if (threadIdx.x == 0)
    {
        for (int i = 0; i < STAGES; i++)
        {
            init(&bar_wt_ready[i], 1);
            init(&bar_act_ready[i], 1);
            init(&bar_data_consumed[i], 32);
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("prefetch.tensormap [%0];" : : "l"(reinterpret_cast<uint64_t>(&weight_map)) : "memory");
        asm volatile("prefetch.tensormap [%0];" : : "l"(reinterpret_cast<uint64_t>(&activation_map)) : "memory");
    }
    __syncthreads();

    // int warp_id = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int phase = 0;

    int mib = (INTERLEAVE ? blockIdx.y : blockIdx.x) * TILE_M;
    int ni = (INTERLEAVE ? blockIdx.x : blockIdx.y) * TILE_N;

    float accum[4];
    for (int i = 0; i < 4; i++)
        accum[i] = 0.f;

    int const K_LOOPS_DMA = (K + 4 * TILE_K * STAGE_UNROLL - 1) / (4 * (TILE_K * STAGE_UNROLL));
    int const K_LOOPS_COMPUTE = K_LOOPS_DMA;

    // Data loading thread
    if (warp_id >= 4 && elect_one_sync())
    {
        int stage = warp_id % 4;

        bool weight_warp = warp_id < 8;

        for (int ki = 0; ki < K_LOOPS_DMA; ki++)
        {

            int k = (ki * 4 + (warp_id % 4)) * TILE_K * STAGE_UNROLL;

            uint64_t desc_ptr_wt = reinterpret_cast<uint64_t>(&weight_map);
            uint64_t desc_ptr_act = reinterpret_cast<uint64_t>(&activation_map);

            uint32_t bar_ptr_wt = __cvta_generic_to_shared(&bar_wt_ready[stage]);
            uint32_t bar_ptr_act = __cvta_generic_to_shared(&bar_act_ready[stage]);
            int bytes_wt = TILE_M * TILE_K * sizeof(__nv_bfloat16);
            int bytes_act = TILE_N * TILE_K * sizeof(__nv_bfloat16);

            bar_wait(__cvta_generic_to_shared(&bar_data_consumed[stage]), phase ^ 1);

            if (weight_warp)
                asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                             :
                             : "r"(bar_ptr_wt), "r"(STAGE_UNROLL * bytes_wt));
            if (!weight_warp)
                asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                             :
                             : "r"(bar_ptr_act), "r"(STAGE_UNROLL * bytes_act));

            if (PROFILE && blockIdx.y == 0 && ki == 0 && weight_warp)
                profile[blockIdx.x].weight_load_start = gclock64();
            if (PROFILE && blockIdx.y == 0 && ki == 0 && !weight_warp)
                profile[blockIdx.x].act_load_start = gclock64();

            for (int i = 0; i < STAGE_UNROLL; i++)
            {
                uint32_t smem_ptr_wt
                    = __cvta_generic_to_shared(&sh_weights[(stage * STAGE_UNROLL + i) * TILE_M * TILE_K]);
                uint32_t crd0 = k + i * TILE_K;
                uint32_t crd1 = mib;
                if (weight_warp)
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3,%4}], "
                        "[%2];"
                        :
                        : "r"(smem_ptr_wt), "l"(desc_ptr_wt), "r"(bar_ptr_wt), "r"(crd0), "r"(crd1)
                        : "memory");

                uint32_t smem_ptr_act
                    = __cvta_generic_to_shared(&sh_activations[(stage * STAGE_UNROLL + i) * TILE_N * TILE_K]);
                crd0 = k + i * TILE_K;
                crd1 = ni;
                if (!weight_warp)
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3,%4}], "
                        "[%2];"
                        :
                        : "r"(smem_ptr_act), "l"(desc_ptr_act), "r"(bar_ptr_act), "r"(crd0), "r"(crd1)
                        : "memory");
            }

            stage += 4;
            if (stage >= STAGES)
            {
                stage = warp_id % 4;
                phase ^= 1;
            }
        }
        // Wait for pending loads to be consumed before exiting, to avoid race
        for (int i = 0; i < (STAGES / 4) - 1; i++)
        {
            bar_wait(__cvta_generic_to_shared(&bar_data_consumed[stage]), phase ^ 1);
            stage += 4;
            if (stage >= STAGES)
            {
                stage = warp_id % 4;
                phase ^= 1;
            }
        }
    }
    // Compute threads
    else if (warp_id < 4)
    {


        int stage = warp_id;

        int phase = 0;
        int lane_id_div8 = lane_id / 8;
        int lane_id_mod8 = lane_id % 8;

        int lane_row_offset_wt = (lane_id_div8 % 2) ? 8 : 0;
        int lane_col_offset_wt = (lane_id_div8 / 2) ? 1 : 0;

        int row_wt = lane_id_mod8 + lane_row_offset_wt;
        int row_act = lane_id_mod8;

        int row_offset_wt = (reinterpret_cast<uintptr_t>(sh_weights) / 128) % 8;
        // int row_offset_act = (reinterpret_cast <uintptr_t>(ptr_act)/128)%8;
        // assert(row_offset_wt==row_offset_act);
        int row_offset_act = row_offset_wt;

        uint32_t bar_ptr_wt = __cvta_generic_to_shared(&bar_wt_ready[stage]);
        uint32_t bar_ptr_act = __cvta_generic_to_shared(&bar_act_ready[stage]);

        bool weight_ready = bar_try_wait(bar_ptr_wt, phase);
        bool act_ready = bar_try_wait(bar_ptr_act, phase);

#pragma unroll 2
        for (int ki = 0; ki < K_LOOPS_COMPUTE; ki++)
        {

            int next_stage = stage + 4;
            int next_phase = phase;
            if (next_stage >= STAGES)
            {
                next_stage = warp_id;
                next_phase ^= 1;
            }

            while (!weight_ready || !act_ready)
            {
                weight_ready = bar_try_wait(__cvta_generic_to_shared(&bar_wt_ready[stage]), phase);
                act_ready = bar_try_wait(__cvta_generic_to_shared(&bar_act_ready[stage]), phase);
            }

            if (PROFILE && blockIdx.y == 0 && threadIdx.x == 0 && ki == 0)
                profile[blockIdx.x].compute_start = gclock64();

            if (ki + 1 < K_LOOPS_COMPUTE)
            {
                weight_ready = bar_try_wait(__cvta_generic_to_shared(&bar_wt_ready[next_stage]), next_phase);
                act_ready = bar_try_wait(__cvta_generic_to_shared(&bar_act_ready[next_stage]), next_phase);
            }

#pragma unroll
            for (int su = 0; su < STAGE_UNROLL; su++)
            {
                __nv_bfloat16* ptr_weights = &sh_weights[(stage * STAGE_UNROLL + su) * TILE_M * TILE_K];
                __nv_bfloat16* ptr_act = &sh_activations[(stage * STAGE_UNROLL + su) * TILE_N * TILE_K];

#pragma unroll
                for (int kii = 0; kii < TILE_K / 16; kii++)
                {

                    __nv_bfloat16 a[8];
                    __nv_bfloat16 b[4];

                    int col = 2 * kii + lane_col_offset_wt;
                    int col_sw = ((row_wt + row_offset_wt) % 8) ^ col;

                    ldmatrix4(a, __cvta_generic_to_shared(&ptr_weights[row_wt * TILE_K + col_sw * 8]));

                    col = 2 * kii + lane_id_div8;
                    col_sw = ((row_act + row_offset_act) % 8) ^ col;

                    ldmatrix2(b, __cvta_generic_to_shared(&ptr_act[row_act * TILE_K + 8 * col_sw]));

                    HMMA_16816(accum, a, b, accum);
#ifdef DEBUG
                    printf("Thread %d: Row: %d, col: %d, col_sw: %d row offset: %d\n", threadIdx.x, row_act, col,
                        col_sw, row_offset_act);
                    printf("Thread %d: a: %f %f %f %f, b: %f %f accum: %f %f %f %f\n", threadIdx.x,
                        __bfloat162float(a[0]), __bfloat162float(a[1]), __bfloat162float(a[2]), __bfloat162float(a[3]),
                        __bfloat162float(b[0]), __bfloat162float(b[1]), accum[0], accum[1], accum[2], accum[3]);
#endif
                }
            }

            uint32_t bar_c = __cvta_generic_to_shared(&bar_data_consumed[stage]);
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" : : "r"(bar_c));

            stage = next_stage;
            phase = next_phase;
        }

        float4 accum4;
        accum4.x = accum[0];
        accum4.y = accum[1];
        accum4.z = accum[2];
        accum4.w = accum[3];
        reduction_buffer[threadIdx.x] = accum4;

        __syncthreads();


        if (warp_id == 0)
        {

            int mi = mib + warp_id * WARP_TILE_M;
            int tm = mi + lane_id / 4;
            int tn = ni + 2 * (lane_id % 4);

            float4 accum1 = reduction_buffer[32 + threadIdx.x];
            float4 accum2 = reduction_buffer[64 + threadIdx.x];
            float4 accum3 = reduction_buffer[96 + threadIdx.x];

            accum[0] = accum[0] + accum1.x + accum2.x + accum3.x;
            accum[1] = accum[1] + accum1.y + accum2.y + accum3.y;
            accum[2] = accum[2] + accum1.z + accum2.z + accum3.z;
            accum[3] = accum[3] + accum1.w + accum2.w + accum3.w;


            if (tn < N && tm < M)
                output[tn * M + tm] = __float2bfloat16(accum[0]);
            if (tn + 1 < N && tm < M)
                output[(tn + 1) * M + tm] = __float2bfloat16(accum[1]);
            if (tn < N && tm + 8 < M)
                output[tn * M + tm + 8] = __float2bfloat16(accum[2]);
            if (tn + 1 < N && tm + 8 < M)
                output[(tn + 1) * M + tm + 8] = __float2bfloat16(accum[3]);

            if (PROFILE && blockIdx.y == 0 && threadIdx.x == 0)
                profile[blockIdx.x].complete = gclock64();
        }
    }
#endif // end if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}
// SPDX-SnippetEnd

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: 2025 NVIDIA CORPORATION <https://github.com/NVIDIA/TensorRT-LLM>
// SPDX-License-Identifier: Apache-2.0
// Adapted from NVIDIA/TensorRT-LLM, cpp/tensorrt_llm/kernels/tinygemm2/tinygemm2_cuda.cu, `launch_tinygemm2`.
// Changes: templated on STAGES and INTERLEAVE, allocates the BF16 output itself, checks the tensor-map encodes with TORCH_CHECK, drops the bias argument and the programmatic-stream-serialization (PDL) launch attribute, and launches with <<<>>> on the current PyTorch stream.
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
// Derived host launcher: no TensorRT dependency and no PDL dependency edges.
template <int STAGES, bool INTERLEAVE>
at::Tensor muse_tiny_launch(const at::Tensor& x, const at::Tensor& w, int64_t n)
{
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 8;
    constexpr int TILE_K = 64;
    constexpr int UNROLL = 4;
    constexpr int SMEM = STAGES * UNROLL * (TILE_M + TILE_N) * TILE_K * sizeof(__nv_bfloat16);
    const int m = static_cast<int>(x.size(0));
    const int k = static_cast<int>(x.size(1));
    auto y = at::empty({m, n}, x.options());
    if (m == 0 || n == 0) return y;

    CUtensorMap weight_map{};
    CUtensorMap activation_map{};
    uint64_t sizes[2] = {static_cast<uint64_t>(k), static_cast<uint64_t>(n)};
    uint64_t strides[1] = {static_cast<uint64_t>(k) * sizeof(__nv_bfloat16)};
    uint32_t boxes[2] = {TILE_K, TILE_M};
    uint32_t element_strides[2] = {1, 1};
    CUresult result = cuTensorMapEncodeTiled(
        &weight_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, w.data_ptr(),
        sizes, strides, boxes, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(result == CUDA_SUCCESS, "weight tensor map failed: ", static_cast<int>(result));
    sizes[1] = static_cast<uint64_t>(m);
    boxes[1] = TILE_N;
    result = cuTensorMapEncodeTiled(
        &activation_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, x.data_ptr(),
        sizes, strides, boxes, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(result == CUDA_SUCCESS, "activation tensor map failed: ", static_cast<int>(result));

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        tinygemm_kernel<16, TILE_M, TILE_N, TILE_K, STAGES, UNROLL, false, INTERLEAVE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    const unsigned channel_tiles = (n + TILE_M - 1) / TILE_M;
    const unsigned token_tiles = (m + TILE_N - 1) / TILE_N;
    const dim3 grid = INTERLEAVE ? dim3(token_tiles, channel_tiles) : dim3(channel_tiles, token_tiles);
    const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    tinygemm_kernel<16, TILE_M, TILE_N, TILE_K, STAGES, UNROLL, false, INTERLEAVE>
        <<<grid, 384, SMEM, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(w.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(x.data_ptr()),
            static_cast<int>(n), m, k, weight_map, activation_map, nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
// SPDX-SnippetEnd

at::Tensor tinygemm_forward(at::Tensor x, at::Tensor w, int64_t n, int64_t stages, bool interleave)
{
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(x.device() == w.device(), "x and w must share a device");
    TORCH_CHECK(x.dim() == 2 && w.dim() == 2, "rank-two tensors required");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16,
                "BF16 inputs and weights required");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "row-contiguous tensors required");
    TORCH_CHECK(x.size(1) == w.size(1) && x.size(1) % 64 == 0 && x.size(1) > 0,
                "matching K, divisible by 64, required for this TMA layout");
    TORCH_CHECK(n >= 0 && n <= w.size(0), "invalid output channel count");
    const c10::cuda::CUDAGuard guard(x.device());
    if (stages == 4) {
        return interleave ? muse_tiny_launch<4, true>(x, w, n) : muse_tiny_launch<4, false>(x, w, n);
    }
    if (stages == 8) {
        return interleave ? muse_tiny_launch<8, true>(x, w, n) : muse_tiny_launch<8, false>(x, w, n);
    }
    if (stages == 16) {
        return interleave ? muse_tiny_launch<16, true>(x, w, n) : muse_tiny_launch<16, false>(x, w, n);
    }
    TORCH_CHECK(false, "stages must be 4, 8, or 16");
}

"""

_b7__cx_ti__CPP_SOURCE = r"""
#include <torch/extension.h>
at::Tensor tinygemm_forward(at::Tensor x, at::Tensor w, int64_t n, int64_t stages, bool interleave);
"""


@lru_cache(maxsize=1)
def _b7__cx_ti__extension():
    return load_inline(
        name="muse_hopper_tinygemm_v2",
        cpp_sources=_b7__cx_ti__CPP_SOURCE,
        cuda_sources=_b7__cx_ti__CUDA_SOURCE,
        functions=["tinygemm_forward"],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=[
            "-O3", "-std=c++20", "-lineinfo",
            "-gencode=arch=compute_90,code=sm_90",
        ],
        extra_ldflags=["-lcuda"],
        with_cuda=True,
        verbose=False,
    )


def _b7__cx_ti_run(x, w, n, config):
    if len(config) != 3 or config[0] != "tiny" or config[1] not in (4, 8, 16):
        raise ValueError(f"invalid TinyGEMM configuration: {config!r}")
    return _b7__cx_ti__extension().tinygemm_forward(x, w, int(n), int(config[1]), bool(config[2]))


def _b7__cx_ti_configs(m, n, k, quick=False):
    if k % 64:
        return []
    if quick:
        return [("tiny", 4, False), ("tiny", 8, False),
                ("tiny", 16, False), ("tiny", 8, True)]
    orders = (False, True) if m > 8 else (False,)
    return [("tiny", stages, interleave)
            for stages in (4, 8, 16) for interleave in orders]


# Design based on the Triton persistent-matmul tutorial's host TensorDescriptor setup and grouped tile order (triton-lang/triton python/tutorials/09-persistent-matmul.py, MIT); code rewritten, no lines copied.
# Source module: proj_tma
"""Fresh Hopper BF16 projection using Triton host TensorDescriptors.

API: run(x, w, n, cfg), configs(m, n, k, quick=False), and optional prepare(w).
The config tuple is ("tma", BN, BM, BK, SPLIT_K, num_warps, num_stages), with
BN output channels and BM tokens. Output is BF16 [M,N], from FP32 accumulation
of the original BF16 weights and activations. This module excludes the router.

The mainloop computes W[N,K] @ X[M,K].T: both TMA descriptors have contiguous
K tiles, and tl.dot(a, b.T) has the large output-channel axis first. Host
descriptors follow the local official Triton tutorial at
upstream/triton/09-persistent-matmul.py. That tutorial disables host-descriptor
warp specialization on Hopper, so this candidate uses software pipelining.
BK128/256 and stage-count sweeps target more concurrent transfers with fewer
mainloop iterations. These are candidates, not measured performance claims.

TMA supplies zero values outside descriptor bounds. The epilogue masks every
M/N tail. Non-TMA-compatible layouts (including odd contiguous K, unaligned
views, and non-unit inner strides) use masked pointer loads in the same
matmul, with no copies or input-derived caching. Split-K writes FP32 partials
and reduces them in a fixed order in a second kernel, rounding to BF16 once.

prepare(w) retains only the original weights and host descriptors keyed by
output width and tile size; using a raw tensor is equally valid. X descriptors
are recreated for each call. No autotuning, GPU probing, or synchronization
occurs in this module. Warm the requested configurations before graph capture.

Additional local design references read:
  kernel-design-agents/skills/KernelWiki/wiki/hardware/tma.md (hw-tma)
  kernel-design-agents/skills/KernelWiki/wiki/techniques/pipeline-stages.md
    (technique-pipeline-stages; source-reported design guidance)
"""

from dataclasses import dataclass, field

import torch
import triton as tr
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@tr.jit
def _b7__cx_tm_tma_matmul(
    WD, XD, Y, P,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    WS0: tl.constexpr, WS1: tl.constexpr,
    XS0: tl.constexpr, XS1: tl.constexpr,
    BN: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
    SK: tl.constexpr, STAGES: tl.constexpr, USE_TMA: tl.constexpr,
):
    tile = tl.program_id(0)
    split = tl.program_id(1)
    n_tiles: tl.constexpr = tr.cdiv(N, BN)
    m_tiles: tl.constexpr = tr.cdiv(M, BM)
    # Group nearby weight rows across token tiles for L2 reuse. Use the local
    # group index so a partial final group also covers each tile exactly once.
    group: tl.constexpr = 8
    group_id = tile // (group * m_tiles)
    first_n = group_id * group
    group_n = tl.minimum(n_tiles - first_n, group)
    local_tile = tile % (group * m_tiles)
    pid_n = first_n + local_tile % group_n
    pid_m = local_tile // group_n
    start_n = pid_n * BN
    start_m = pid_m * BM
    iterations: tl.constexpr = tr.cdiv(K, SK * BK)
    split_start = split * iterations * BK
    acc = tl.zeros((BN, BM), tl.float32)

    for ki in tl.range(0, iterations, num_stages=STAGES):
        start_k = split_start + ki * BK
        if USE_TMA:
            a = WD.load([start_n, start_k])
            b = XD.load([start_m, start_k])
        else:
            rn = start_n + tl.arange(0, BN)
            rm = start_m + tl.arange(0, BM)
            rk = start_k + tl.arange(0, BK)
            a = tl.load(WD + rn[:, None] * WS0 + rk[None, :] * WS1,
                        (rn[:, None] < N) & (rk[None, :] < K), 0)
            b = tl.load(XD + rm[:, None] * XS0 + rk[None, :] * XS1,
                        (rm[:, None] < M) & (rk[None, :] < K), 0)
        acc = tl.dot(a, b.T, acc)

    rn = start_n + tl.arange(0, BN)
    rm = start_m + tl.arange(0, BM)
    offsets = rm[None, :] * N + rn[:, None]
    mask = (rn[:, None] < N) & (rm[None, :] < M)
    if SK == 1:
        tl.store(Y + offsets, acc.to(tl.bfloat16), mask)
    else:
        tl.store(P + split * M * N + offsets, acc, mask)


@tr.jit
def _b7__cx_tm_tma_reduce_split(P, Y, SIZE: tl.constexpr, SK: tl.constexpr, B: tl.constexpr):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    total = tl.full((B,), 0, tl.float32)
    for split in tl.static_range(0, SK):
        total += tl.load(P + split * SIZE + idx, idx < SIZE, 0)
    tl.store(Y + idx, total.to(tl.bfloat16), idx < SIZE)


def _b7__cx_tm__can_describe(tensor):
    # TMA requires a 16-byte-aligned base/outer byte stride and unit inner
    # element stride. Logical dimensions need not be tile multiples.
    return (
        tensor.shape[0] > 0
        and tensor.shape[1] > 0
        and tensor.stride(1) == 1
        and tensor.stride(0) > 0
        and (tensor.stride(0) * tensor.element_size()) % 16 == 0
        and tensor.data_ptr() % 16 == 0
    )


@dataclass
class _b7__cx_tm_PreparedWeights:
    tensor: torch.Tensor
    descriptors: dict = field(default_factory=dict)

    def descriptor(self, n, bn, bk):
        key = (n, bn, bk)
        if key not in self.descriptors:
            self.descriptors[key] = TensorDescriptor.from_tensor(
                self.tensor[:n], [bn, bk]
            )
        return self.descriptors[key]


def _b7__cx_tm_prepare(w):
    """Retain BF16 weights and optionally reuse their host TMA descriptors."""
    if not isinstance(w, torch.Tensor) or w.dtype != torch.bfloat16 or w.ndim != 2:
        raise ValueError("prepare expects a two-dimensional BF16 weight tensor")
    return _b7__cx_tm_PreparedWeights(w)


def _b7__cx_tm_run(x, w, n, cfg=None):
    """Return BF16 x @ w[:n].T; w may be a tensor or PreparedWeights."""
    prepared = w if isinstance(w, _b7__cx_tm_PreparedWeights) else None
    weight = prepared.tensor if prepared is not None else w
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("x and w must be two-dimensional with matching K")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("proj_tma requires BF16 x and w; it does not handle the router")
    if not x.is_cuda or weight.device != x.device:
        raise ValueError("x and w must reside on the same CUDA device")
    m, k = x.shape
    n = int(n)
    if not 0 <= n <= weight.shape[0]:
        raise ValueError("n must select an existing prefix of the weight rows")
    y = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if m == 0 or n == 0:
        return y
    if cfg is None:
        cfg = (_b7__cx_tm_configs(m, n, k, quick=True)[0] if k > 0
               else ("tma", 64, 16, 128, 1, 4, 3))
    kind, bn, bm, bk, sk, warps, stages = cfg
    if kind != "tma":
        raise ValueError("expected ('tma', BN, BM, BK, SK, warps, stages)")
    if bn not in (16, 32, 64, 128, 256) or bm not in (16, 32, 64, 128, 256):
        raise ValueError("BN/BM must be supported power-of-two dot tile dimensions")
    if bk not in (32, 64, 128, 256) or sk not in (1, 2, 4, 8, 16):
        raise ValueError("BK/SK are outside the supported tuning range")
    if warps not in (4, 8) or stages not in (1, 2, 3, 4, 5):
        raise ValueError("unsupported warp or pipeline-stage count")

    partial = torch.empty((sk, m, n), dtype=torch.float32, device=x.device) if sk > 1 else y
    use_tma = _b7__cx_tm__can_describe(x) and _b7__cx_tm__can_describe(weight)
    if use_tma:
        wd = (prepared.descriptor(n, bn, bk) if prepared is not None
              else TensorDescriptor.from_tensor(weight[:n], [bn, bk]))
        xd = TensorDescriptor.from_tensor(x, [bm, bk])
    else:
        wd, xd = weight, x
    _b7__cx_tm_tma_matmul[(tr.cdiv(n, bn) * tr.cdiv(m, bm), sk)](
        wd, xd, y, partial, m, n, k,
        weight.stride(0), weight.stride(1), x.stride(0), x.stride(1),
        bn, bm, bk, sk, stages, use_tma,
        num_warps=warps, num_stages=stages,
    )
    if sk > 1:
        _b7__cx_tm_tma_reduce_split[(tr.cdiv(m * n, 256),)](
            partial, y, m * n, sk, 256, num_warps=4
        )
    return y


def _b7__cx_tm_configs(m, n, k, quick=False):
    """Return a bounded sweep; quick has at most four representative configs."""
    if m < 1 or n < 1 or k < 1:
        return []
    bm = 16 if m <= 16 else (32 if m <= 32 else 64)
    if m <= 32:
        initial = [
            ("tma", 64, bm, 128, 1, 4, 3),
            ("tma", 64, bm, 256, 2, 4, 2),
            ("tma", 64, bm, 128, 4, 4, 4),
            ("tma", 32, bm, 128, 4, 4, 3),
        ]
        # (BN, BM, BK, warps, stages); vary stages at fixed tile size as
        # well as larger K tiles. No variant needs >192 KiB operand buffers.
        shapes = [
            (32, bm, 128, 4, 3),
            (64, bm, 64, 4, 3),
            (64, bm, 128, 4, 2),
            (64, bm, 128, 4, 3),
            (64, bm, 128, 4, 4),
            (64, bm, 256, 4, 2),
            (64, bm, 256, 4, 3),
            (128, bm, 128, 4, 2),
            (128, bm, 128, 8, 3),
        ]
        splits = (1, 2, 4, 8)
    else:
        initial = [
            ("tma", 64, 64, 128, 1, 4, 3),
            ("tma", 64, 32, 128, 1, 4, 3),
            ("tma", 64, 64, 256, 1, 4, 2),
            ("tma", 64, 64, 128, 2, 4, 3),
        ]
        shapes = [
            (bn, block_m, bk, warps, stages)
            for block_m in (32, 64)
            for bn, bk, warps, stages in (
                (64, 64, 4, 3),
                (64, 128, 4, 2),
                (64, 128, 4, 3),
                (64, 128, 4, 4),
                (64, 256, 4, 2),
                (128, 128, 8, 3),
            )
        ]
        splits = (1, 2, 4)
    candidates = initial if quick else initial + [
        ("tma", bn, block_m, bk, split, warps, stages)
        for bn, block_m, bk, warps, stages in shapes
        for split in splits
    ]
    result = []
    for candidate in candidates:
        _, bn, block_m, bk, split, _, stages = candidate
        # A split with no K tile wastes a CTA. Keep SK1 valid for short K
        # because descriptor zero-filling supplies the remaining tile values.
        if split > 1 and split * bk > k:
            continue
        if 2 * (bn + block_m) * bk * stages > 192 * 1024:
            continue
        if candidate not in result:
            result.append(candidate)
    return result


def _b7__cx_tm_configs_large(m,n,k,quick=False):
    # m256 profile: ~71 MB DRAM in both kernels, but repeated weight reads
    # through L2 and only 44% tensor activity. Larger token tiles test reuse.
    return [('tma',64,bm,bk,sk,warps,stages)
            for bm in (128,256) if bm<=max(128,m)
            for bk,warps,stages in ((64,4,3),(128,4,2),(128,4,3),(128,8,3))
            for sk in (1,2,4)
            if (64+bm)*bk*2*stages<=192*1024]


# Source module: proj_packed
"""Lossless BF16 tile repacking to make each weight tile contiguous in HBM.

Weight values and precision are unchanged. prepare stores [N/BN,K/BK,BN,BK]
layouts; the timed kernel uses a 4-D TMA load per contiguous weight tile.
This tests the memory-layout hypothesis from the first pooler profile.
"""
import torch
import triton as tr
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@tr.jit
def _b7__cx_pa_packed_mm(WD, XD, Y, P, M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
              BN:tl.constexpr,BM:tl.constexpr,BK:tl.constexpr,SK:tl.constexpr,
              STAGES:tl.constexpr):
    pn=tl.program_id(0)
    pm=tl.program_id(1)
    split=tl.program_id(2)
    loops:tl.constexpr=tr.cdiv(K,SK*BK)
    acc=tl.zeros((BN,BM),tl.float32)
    for ki in tl.range(0,loops,num_stages=STAGES):
        kb=split*loops+ki
        w=WD.load([pn,kb,0,0]).reshape((BN,BK))
        x=XD.load([pm*BM,kb*BK])
        acc=tl.dot(w,x.T,acc)
    rn=pn*BN+tl.arange(0,BN)
    rm=pm*BM+tl.arange(0,BM)
    off=rm[None,:]*N+rn[:,None]
    mask=(rm[None,:]<M)&(rn[:,None]<N)
    if SK==1:
        tl.store(Y+off,acc.to(tl.bfloat16),mask)
    else:
        tl.store(P+split*M*N+off,acc,mask)


@tr.jit
def _b7__cx_pa_packed_reduce(P,Y,SIZE:tl.constexpr,SK:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    acc=tl.zeros((B,),tl.float32)
    for split in tl.static_range(SK):
        acc+=tl.load(P+split*SIZE+i,i<SIZE,0)
    tl.store(Y+i,acc.to(tl.bfloat16),i<SIZE)


_b7__cx_pa_PACKINGS=((16,128),(32,128),(64,64),(64,128),(64,256),(128,128))


def _b7__cx_pa_prepare(w):
    n,k=w.shape
    state={'weight':w,'packed':{}}
    for bn,bk in _b7__cx_pa_PACKINGS:
        npad,kpad=tr.cdiv(n,bn)*bn,tr.cdiv(k,bk)*bk
        source=w if (n,k)==(npad,kpad) else torch.nn.functional.pad(w,(0,kpad-k,0,npad-n))
        packed=source.view(npad//bn,bn,kpad//bk,bk).permute(0,2,1,3).contiguous()
        state['packed'][(bn,bk)]=(packed,TensorDescriptor.from_tensor(packed,[1,1,bn,bk]))
    return state


def _b7__cx_pa_run(x,state,n,cfg):
    _,bn,bm,bk,sk,nw,ns=cfg
    m,k=x.shape
    y=torch.empty((m,n),dtype=torch.bfloat16,device=x.device)
    partial=torch.empty((sk,m,n),dtype=torch.float32,device=x.device) if sk>1 else y
    wd=state['packed'][(bn,bk)][1]
    xd=TensorDescriptor.from_tensor(x,[bm,bk])
    _b7__cx_pa_packed_mm[(tr.cdiv(n,bn),tr.cdiv(m,bm),sk)](wd,xd,y,partial,m,n,k,bn,bm,bk,sk,ns,num_warps=nw,num_stages=ns)
    if sk>1:_b7__cx_pa_packed_reduce[(tr.cdiv(m*n,256),)](partial,y,m*n,sk,256)
    return y


def _b7__cx_pa_configs(m,n,k,quick=False):
    bm=16 if m<=16 else 32 if m<=32 else 64
    initial=[('packed',64,bm,128,4,4,3),('packed',32,bm,128,4,4,3),
             ('packed',64,bm,256,1,4,3),('packed',64,bm,128,1,4,4)]
    if quick:return initial
    return list(dict.fromkeys(initial+[
        ('packed',bn,bm,bk,sk,4,ns)
        for bn,bk in _b7__cx_pa_PACKINGS for sk in (1,2,4,8) for ns in (2,4)
        if sk*bk<=k and (bn+bm)*bk*2*ns<200*1024
    ]))


# Source module: router_interleaved
"""Router experiment: one FP32 dot accumulator over an interleaved hi/lo K.

API:
    state = prepare_router(w_fp32)
    y_fp32 = run_router(x_bf16, state, config)
    configs(m, n, k, quick=False)

Config: (kind, layout, BM, BN, BK, split_k, num_warps, num_stages).
``layout`` is "pair" ([hi0,lo0,hi1,lo1,...]) or "block128"
([hi[0:128],lo[0:128],hi[128:256],lo[128:256],...]). BK is measured
on the expanded axis. The "simt" kind uses original FP32 weights for M=1.

Precision: hi=BF16(w), lo=BF16(w-FP32(hi)), exactly the permitted two-BF16
representation. Both halves participate in every projection. It approximates
FP32 weights as the baseline does; it is not a lossless encoding of FP32.
Interleaving changes summation order but retains FP32 accumulation and output.
Split boundaries keep whole hi+lo groups together. Split-K reduction has a
fixed FP32 summation tree. Masks cover arbitrary M/N/K, including block padding.

Tradeoff: one tl.dot accumulator stream replaces two separate hi/lo streams,
but the expanded K has twice as many terms, so total tensor-core arithmetic
is unchanged. Each packed layout uses 4*N*K bytes (plus block-tail padding),
equal to the combined BF16 hi/lo payload. X is indexed twice per original K;
there is no activation packing launch, but shared-memory/register traffic for
activation tiles can increase. prepare retains both layouts plus original
FP32 weights for comparisons; state['stats'] reports the actual allocation.

SK1 uses one kernel launch; other variants use a separate deterministic FP32
reduction. All tuning choices are experimental until externally benchmarked.
This module launches no work on import and performs no autotuning/profiling.
"""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _b7__cx_ri__pack_router(W, PAIR, BLOCKED,
                 N: tl.constexpr, K: tl.constexpr, PAD_K: tl.constexpr,
                 B: tl.constexpr):
    index = tl.program_id(0) * B + tl.arange(0, B)
    row = index // PAD_K
    col = index % PAD_K
    valid = (row < N) & (col < K)
    weight = tl.load(W + row * K + col, valid, 0)
    hi = weight.to(tl.bfloat16)
    lo = (weight - hi.to(tl.float32)).to(tl.bfloat16)
    pair_offset = row * (2 * K) + 2 * col
    tl.store(PAIR + pair_offset, hi, valid)
    tl.store(PAIR + pair_offset + 1, lo, valid)
    block_offset = row * (2 * PAD_K) + (col // 128) * 256 + col % 128
    # Also initialize padded positions: zero activations times uninitialized
    # NaNs would otherwise contaminate the FP32 dot accumulator.
    tl.store(BLOCKED + block_offset, hi, row < N)
    tl.store(BLOCKED + block_offset + 128, lo, row < N)


def _b7__cx_ri_prepare_router(w):
    """Prepare both permitted hi+lo layouts, retaining exact FP32 for SIMT."""
    if w.ndim != 2 or w.dtype != torch.float32 or not w.is_cuda:
        raise ValueError("router weights must be a two-dimensional CUDA FP32 tensor")
    w = w.contiguous()
    n, k = w.shape
    padded_k = tr.cdiv(k, 128) * 128
    pair = torch.empty((n, 2 * k), dtype=torch.bfloat16, device=w.device)
    blocked = torch.empty((n, 2 * padded_k), dtype=torch.bfloat16, device=w.device)
    if n and k:
        _b7__cx_ri__pack_router[(tr.cdiv(n * padded_k, 256),)](w, pair, blocked, n, k, padded_k, 256)
    return {
        "w": w, "pair": pair, "block128": blocked,
        "n": n, "k": k, "padded_k": padded_k,
        "stats": {
            "raw_bytes": w.numel() * w.element_size(),
            "pair_bytes": pair.numel() * pair.element_size(),
            "block128_bytes": blocked.numel() * blocked.element_size(),
            "total_weight_bytes": (w.numel() * w.element_size()
                                   + pair.numel() * pair.element_size()
                                   + blocked.numel() * blocked.element_size()),
        },
    }


@tr.jit
def _b7__cx_ri_router_interleaved_dot(
    X, PACKED, Y, PARTIAL,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, PACKED_K: tl.constexpr,
    XS0: tl.constexpr, XS1: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    SK: tl.constexpr, GROUP: tl.constexpr,
):
    tile = tl.program_id(0)
    split = tl.program_id(1)
    # Consecutive token tiles share weight rows in L2.
    m_tiles: tl.constexpr = tr.cdiv(M, BM)
    rn = (tile // m_tiles) * BN + tl.arange(0, BN)
    rm = (tile % m_tiles) * BM + tl.arange(0, BM)
    rk = tl.arange(0, BK)
    quantum: tl.constexpr = max(BK, 2 * GROUP)
    split_span: tl.constexpr = tr.cdiv(PACKED_K, SK * quantum) * quantum
    iterations: tl.constexpr = split_span // BK
    acc = tl.zeros((BN, BM), tl.float32)
    for iteration in range(iterations):
        q = split * split_span + iteration * BK + rk
        # This maps both halves to the same original activation, including
        # full 128-element blocks without an activation duplication kernel.
        original_k = (q // (2 * GROUP)) * GROUP + q % GROUP
        a = tl.load(PACKED + rn[:, None] * PACKED_K + q[None, :],
                    (rn[:, None] < N) & (q[None, :] < PACKED_K), 0)
        b = tl.load(X + rm[:, None] * XS0 + original_k[None, :] * XS1,
                    (rm[:, None] < M) & (original_k[None, :] < K)
                    & (q[None, :] < PACKED_K), 0)
        acc = tl.dot(a, b.T, acc)
    offset = rm[None, :] * N + rn[:, None]
    mask = (rn[:, None] < N) & (rm[None, :] < M)
    if SK == 1:
        tl.store(Y + offset, acc, mask)
    else:
        tl.store(PARTIAL + split * M * N + offset, acc, mask)


@tr.jit
def _b7__cx_ri_router_interleaved_simt(
    X, W, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    XS0: tl.constexpr, XS1: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    token = tl.program_id(1)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BN, BK), tl.float32)
    for block in range(tr.cdiv(K, BK)):
        kk = block * BK + rk
        weight = tl.load(W + rows[:, None] * K + kk[None, :],
                         (rows[:, None] < N) & (kk[None, :] < K), 0)
        x = tl.load(X + token * XS0 + kk * XS1, kk < K, 0).to(tl.float32)
        acc = tl.fma(weight, x[None, :], acc)
    tl.store(Y + token * N + rows, tl.sum(acc, axis=1), rows < N)


@tr.jit
def _b7__cx_ri_router_interleaved_reduce(PARTIAL, Y, SIZE: tl.constexpr,
                              SK: tl.constexpr, BLOCK_SPLIT: tl.constexpr,
                              BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    split = tl.arange(0, BLOCK_SPLIT)
    value = tl.load(PARTIAL + split[:, None] * SIZE + index[None, :],
                    (split[:, None] < SK) & (index[None, :] < SIZE), 0)
    tl.store(Y + index, tl.sum(value, axis=0), index < SIZE)


def _b7__cx_ri_run_router(x, state, cfg):
    """Return FP32 router logits from original FP32 or both BF16 halves."""
    if x.ndim != 2 or x.dtype != torch.bfloat16:
        raise ValueError("router activations must be a two-dimensional BF16 tensor")
    m, k = x.shape
    n = state["n"]
    if k != state["k"] or x.device != state["w"].device:
        raise ValueError("activation shape/device does not match the prepared weights")
    kind, layout, bm, bn, bk, sk, warps, stages = cfg
    if warps not in (4, 8) or stages not in (1, 2, 3, 4):
        raise ValueError("invalid warp or pipeline-stage count")
    if sk not in (1, 2, 4, 8, 16, 32):
        raise ValueError("split_k must be 1, 2, 4, 8, 16, or 32")
    output = torch.empty((m, n), dtype=torch.float32, device=x.device)
    if not m or not n:
        return output
    if kind == "simt":
        if layout != "fp32" or sk != 1 or bn not in (1, 2, 4, 8):
            raise ValueError("SIMT requires FP32 weights, SK1, and BN1/2/4/8")
        if bk not in (256, 512, 1024, 2048, 4096, 8192):
            raise ValueError("unsupported SIMT K tile")
        _b7__cx_ri_router_interleaved_simt[(tr.cdiv(n, bn), m)](
            x, state["w"], output, m, n, k, x.stride(0), x.stride(1), bn, bk,
            num_warps=warps, num_stages=stages,
        )
    elif kind == "interleaved":
        if layout not in ("pair", "block128"):
            raise ValueError("layout must be pair or block128")
        if bm not in (16, 32, 64, 128) or bn not in (16, 32, 64):
            raise ValueError("invalid tensor-core tile dimensions")
        if bk not in (128, 256, 512):
            raise ValueError("expanded BK must be 128, 256, or 512")
        packed = state[layout]
        group = 1 if layout == "pair" else 128
        partial = torch.empty((sk, m, n), dtype=torch.float32, device=x.device) if sk > 1 else output
        _b7__cx_ri_router_interleaved_dot[(tr.cdiv(n, bn) * tr.cdiv(m, bm), sk)](
            x, packed, output, partial, m, n, k, packed.shape[1],
            x.stride(0), x.stride(1), bm, bn, bk, sk, group,
            num_warps=warps, num_stages=stages,
        )
        if sk > 1:
            _b7__cx_ri_router_interleaved_reduce[(tr.cdiv(m * n, 256),)](
                partial, output, m * n, sk, tr.next_power_of_2(sk), 256,
                num_warps=4,
            )
    else:
        raise ValueError("router kind must be simt or interleaved")
    return output


def _b7__cx_ri_configs(m, n, k, quick=False):
    """At most four quick tuples; full sweep covers two layouts and SK8/16/32."""
    if m < 1 or n < 1 or k < 1:
        return []
    if m == 1:
        initial = [("simt", "fp32", 1, bn, bk, 1, 4, 1)
                   for bn, bk in ((1, 512), (1, 2048), (2, 1024), (4, 2048))]
        if quick:
            return initial
        return initial + [("simt", "fp32", 1, bn, bk, 1, warps, 1)
                          for bn, bk, warps in ((1, 8192, 4), (1, 8192, 8),
                                               (2, 4096, 4), (4, 8192, 8))]
    bm = 16 if m <= 16 else 32
    if m <= 32:
        initial = [
            ("interleaved", "pair", bm, 16, 256, 16, 4, 2),
            ("interleaved", "pair", bm, 32, 512, 16, 4, 2),
            ("interleaved", "block128", bm, 32, 256, 16, 4, 2),
            ("interleaved", "block128", bm, 64, 512, 32, 4, 2),
        ]
    else:
        initial = [
            ("interleaved", "pair", bm, 16, 256, 8, 4, 2),
            ("interleaved", "pair", bm, 32, 512, 8, 4, 2),
            ("interleaved", "block128", bm, 32, 256, 8, 4, 2),
            ("interleaved", "block128", 16, 16, 512, 1, 4, 2),
        ]
    candidates = list(initial)
    if not quick:
        candidates += [
            ("interleaved", layout, bm, bn, bk, sk, 4, 2)
            for layout in ("pair", "block128")
            for bn in (16, 32, 64)
            for bk in (128, 256, 512)
            for sk in (8, 16, 32)
        ]
        if m > 32:
            # Enough output tiles can fill the GPU without global split-K.
            candidates += [
                ("interleaved", layout, 16, bn, bk, 1, 4, 2)
                for layout in ("pair", "block128")
                for bn in (16, 32)
                for bk in (256, 512)
            ]
            candidates += [
                ("interleaved", layout, 32, 32, bk, 2, 4, 2)
                for layout in ("pair", "block128") for bk in (256, 512)
            ]
    result = []
    for cfg in candidates:
        _, layout, _, _, bk, sk, _, _ = cfg
        group = 1 if layout == "pair" else 128
        packed_k = 2 * ((k + group - 1) // group) * group
        if sk > 1 and sk * max(bk, 2 * group) > packed_k:
            continue
        if cfg not in result:
            result.append(cfg)
    if not result:
        result = [("interleaved", "pair", bm, 16, 128, 1, 4, 2)]
    return result


# Source module: cublaslt_candidate
"""Development-only cuBLASLt heuristic enumeration and static algorithm runner.

BF16 weights/activations/output and FP32 compute. The column-major output [N,M]
shares the row-major [M,N] output storage. TN uses original W[N,K]; NN uses
untimed contiguous W.T[K,Nfull], with leading dimension Nfull for sliced N.

configs() queries heuristics without timing or running a GEMM. It returns only
JSON-compatible numeric settings, sufficient to reconstruct each algorithm via
the public cuBLASLt API. run() performs no heuristic search or timing. For a final
submission, retain measured static tuples and omit calls to configs().

Tuple fields:
  ("lt", "tn"/"nn", algorithm_id, tile_id, split_k, reduction_scheme,
   cta_swizzle, custom_option, stages_id, inner_shape_id, cluster_shape_id,
   workspace_bytes, cublaslt_version)

Reference: https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul
"""

from functools import lru_cache

import torch
from torch.utils.cpp_extension import load_inline


_b7__cx_lt_MAX_WORKSPACE_BYTES = 256 * 1024 * 1024

_b7__cx_lt__CPP_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

constexpr size_t MAX_WORKSPACE = 256ull * 1024 * 1024;

static void lt_check(cublasStatus_t status, const char* operation)
{
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, operation, " failed with cuBLAS status ", static_cast<int>(status));
}

static const cublasLtMatmulAlgoConfigAttributes_t config_attributes[9] = {
    CUBLASLT_ALGO_CONFIG_ID,
    CUBLASLT_ALGO_CONFIG_TILE_ID,
    CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
    CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
    CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
    CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
    CUBLASLT_ALGO_CONFIG_STAGES_ID,
    CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,
    CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
};

static int64_t lt_get_attribute(const cublasLtMatmulAlgo_t& algo, int index)
{
    size_t written = 0;
    if (index >= 7) {
        uint16_t value = 0;
        lt_check(cublasLtMatmulAlgoConfigGetAttribute(&algo, config_attributes[index],
                 &value, sizeof(value), &written), "get 16-bit algorithm attribute");
        return value;
    }
    uint32_t value = 0;
    lt_check(cublasLtMatmulAlgoConfigGetAttribute(&algo, config_attributes[index],
             &value, sizeof(value), &written), "get 32-bit algorithm attribute");
    return value;
}

static void lt_set_attributes(cublasLtMatmulAlgo_t& algo, const std::vector<int64_t>& settings)
{
    TORCH_CHECK(settings.size() == 9, "nine algorithm settings required");
    for (int i = 1; i < 9; ++i) {
        if (i >= 7) {
            TORCH_CHECK(settings[i] >= 0 && settings[i] <= UINT16_MAX, "invalid 16-bit algorithm setting");
            const uint16_t value = static_cast<uint16_t>(settings[i]);
            lt_check(cublasLtMatmulAlgoConfigSetAttribute(&algo, config_attributes[i],
                     &value, sizeof(value)), "set 16-bit algorithm attribute");
        } else {
            const uint32_t value = static_cast<uint32_t>(settings[i]);
            lt_check(cublasLtMatmulAlgoConfigSetAttribute(&algo, config_attributes[i],
                     &value, sizeof(value)), "set 32-bit algorithm attribute");
        }
    }
}

struct LtDescriptors {
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a = nullptr;
    cublasLtMatrixLayout_t b = nullptr;
    cublasLtMatrixLayout_t output = nullptr;

    LtDescriptors(int64_t m, int64_t n, int64_t k, int64_t lda, bool nn)
    {
        lt_check(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F), "create matmul descriptor");
        const cublasOperation_t trans_a = nn ? CUBLAS_OP_N : CUBLAS_OP_T;
        const cublasOperation_t trans_b = CUBLAS_OP_N;
        lt_check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA,
                 &trans_a, sizeof(trans_a)), "set transpose A");
        lt_check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB,
                 &trans_b, sizeof(trans_b)), "set transpose B");
        // TN: stored A[k,n] -> op(A)[n,k]. NN: stored A[n,k].
        lt_check(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, nn ? n : k, nn ? k : n, lda),
                 "create weight layout");
        lt_check(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, k, m, k), "create activation layout");
        lt_check(cublasLtMatrixLayoutCreate(&output, CUDA_R_16BF, n, m, n), "create output layout");
    }

    ~LtDescriptors()
    {
        if (output) cublasLtMatrixLayoutDestroy(output);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (operation) cublasLtMatmulDescDestroy(operation);
    }
    LtDescriptors(const LtDescriptors&) = delete;
    LtDescriptors& operator=(const LtDescriptors&) = delete;
};

struct LtPlan : LtDescriptors {
    cublasLtMatmulAlgo_t algorithm{};
    size_t workspace_bytes = 0;

    LtPlan(int64_t m, int64_t n, int64_t k, int64_t lda, bool nn,
           const std::vector<int64_t>& settings, int64_t version)
        : LtDescriptors(m, n, k, lda, nn)
    {
        TORCH_CHECK(version == static_cast<int64_t>(cublasLtGetVersion()),
                    "static algorithm was selected with another cuBLASLt version");
        TORCH_CHECK(settings.size() == 9, "nine algorithm settings required");
        // Exclude reductions that store intermediate sums in the BF16 output type.
        TORCH_CHECK(settings[3] == CUBLASLT_REDUCTION_SCHEME_NONE ||
                    settings[3] == CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE,
                    "FP32 split-K reductions required");
        const auto handle = at::cuda::getCurrentCUDABlasLtHandle();
        lt_check(cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,
                 CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
                 static_cast<int>(settings[0]), &algorithm), "initialize static algorithm");
        lt_set_attributes(algorithm, settings);
        cublasLtMatmulHeuristicResult_t result{};
        lt_check(cublasLtMatmulAlgoCheck(handle, operation, a, b, output, output,
                 &algorithm, &result), "check static algorithm");
        lt_check(result.state, "static algorithm state");
        workspace_bytes = result.workspaceSize;
        TORCH_CHECK(workspace_bytes <= MAX_WORKSPACE, "algorithm exceeds workspace limit");
    }
};

static std::shared_ptr<LtPlan> lt_plan(int64_t m, int64_t n, int64_t k, int64_t lda,
                                     bool nn, const std::vector<int64_t>& settings,
                                     int64_t version, int64_t device)
{
    static std::mutex mutex;
    static std::map<std::vector<int64_t>, std::shared_ptr<LtPlan>> cache;
    std::vector<int64_t> key{m, n, k, lda, nn, version, device};
    key.insert(key.end(), settings.begin(), settings.end());
    std::lock_guard<std::mutex> lock(mutex);
    auto found = cache.find(key);
    if (found != cache.end()) return found->second;
    auto plan = std::make_shared<LtPlan>(m, n, k, lda, nn, settings, version);
    cache.emplace(std::move(key), plan);
    return plan;
}

std::vector<std::vector<int64_t>> lt_heuristics(
    int64_t m, int64_t n, int64_t k, bool nn, int64_t count, int64_t device)
{
    const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
    TORCH_CHECK(m > 0 && n > 0 && k > 0 && count > 0, "positive shape/count required");
    const auto handle = at::cuda::getCurrentCUDABlasLtHandle();
    LtDescriptors desc(m, n, k, nn ? n : k, nn);
    cublasLtMatmulPreference_t preference = nullptr;
    lt_check(cublasLtMatmulPreferenceCreate(&preference), "create heuristic preference");
    const size_t workspace = MAX_WORKSPACE;
    lt_check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
             &workspace, sizeof(workspace)), "set workspace preference");
    const uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
    lt_check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
             &reduction, sizeof(reduction)), "set FP32 reduction preference");
    const uint32_t alignment = 256;
    for (const auto attr : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                           CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                           CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                           CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES}) {
        lt_check(cublasLtMatmulPreferenceSetAttribute(preference, attr, &alignment, sizeof(alignment)),
                 "set alignment preference");
    }
    std::vector<cublasLtMatmulHeuristicResult_t> results(count);
    int returned = 0;
    const auto status = cublasLtMatmulAlgoGetHeuristic(handle, desc.operation, desc.a, desc.b,
                        desc.output, desc.output, preference, static_cast<int>(count), results.data(), &returned);
    cublasLtMatmulPreferenceDestroy(preference);
    lt_check(status, "query heuristics");
    std::vector<std::vector<int64_t>> output;
    for (int i = 0; i < returned; ++i) {
        if (results[i].state != CUBLAS_STATUS_SUCCESS || results[i].workspaceSize > MAX_WORKSPACE) continue;
        std::vector<int64_t> row;
        for (int attr = 0; attr < 9; ++attr) row.push_back(lt_get_attribute(results[i].algo, attr));
        if (row[3] != CUBLASLT_REDUCTION_SCHEME_NONE && row[3] != CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) continue;
        row.push_back(static_cast<int64_t>(results[i].workspaceSize));
        row.push_back(static_cast<int64_t>(cublasLtGetVersion()));
        output.push_back(std::move(row));
    }
    return output;
}

int64_t lt_workspace(int64_t m, int64_t n, int64_t k, int64_t lda, bool nn,
                     std::vector<int64_t> settings, int64_t version, int64_t device)
{
    const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
    return static_cast<int64_t>(lt_plan(m, n, k, lda, nn, settings, version, device)->workspace_bytes);
}

at::Tensor lt_run(at::Tensor x, at::Tensor weight, int64_t n, bool nn,
                  std::vector<int64_t> settings, int64_t version, at::Tensor workspace)
{
    TORCH_CHECK(x.is_cuda() && weight.is_cuda() && workspace.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(x.device() == weight.device() && x.device() == workspace.device(), "matching devices required");
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2, "rank-two matrices required");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16,
                "BF16 activation and weight tensors required");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && workspace.is_contiguous(),
                "contiguous tensors required");
    TORCH_CHECK(workspace.scalar_type() == at::kByte, "byte workspace required");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 256 == 0 &&
                reinterpret_cast<uintptr_t>(weight.data_ptr()) % 256 == 0,
                "this heuristic set requires 256-byte aligned tensor bases");
    const int64_t m = x.size(0), k = x.size(1);
    TORCH_CHECK(nn ? (weight.size(0) == k && weight.size(1) >= n)
                   : (weight.size(1) == k && weight.size(0) >= n), "incompatible weight shape");
    const int64_t lda = nn ? weight.size(1) : k;
    const c10::cuda::CUDAGuard guard(x.device());
    auto y = at::empty({m, n}, x.options());
    if (m == 0 || n == 0) return y;
    auto plan = lt_plan(m, n, k, lda, nn, settings, version, x.get_device());
    TORCH_CHECK(static_cast<size_t>(workspace.numel()) >= plan->workspace_bytes, "workspace too small");
    const float alpha = 1.0f, beta = 0.0f;
    const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
    lt_check(cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(), plan->operation,
             &alpha, weight.data_ptr(), plan->a, x.data_ptr(), plan->b,
             &beta, y.data_ptr(), plan->output, y.data_ptr(), plan->output,
             &plan->algorithm, plan->workspace_bytes ? workspace.data_ptr() : nullptr,
             plan->workspace_bytes, stream.stream()), "execute static cuBLASLt algorithm");
    return y;
}
"""


@lru_cache(maxsize=1)
def _b7__cx_lt__extension():
    return load_inline(
        name="muse_cublaslt_static_v1",
        cpp_sources=_b7__cx_lt__CPP_SOURCE,
        functions=["lt_heuristics", "lt_workspace", "lt_run"],
        extra_cflags=["-O3", "-std=c++20"],
        extra_ldflags=["-lcublasLt", "-lcublas"],
        with_cuda=True,
        verbose=False,
    )


def _b7__cx_lt_prepare(w):
    if w.ndim != 2 or w.dtype != torch.bfloat16:
        raise ValueError("cuBLASLt candidate expects rank-two BF16 weights")
    w = w.contiguous()
    if w.data_ptr() % 256:
        w = w.clone()
    return {"w": w, "wt": w.t().contiguous(), "n": w.shape[0], "k": w.shape[1]}


@lru_cache(maxsize=None)
def _b7__cx_lt__heuristics(m, n, k, nn, count, device):
    return tuple(tuple(row) for row in _b7__cx_lt__extension().lt_heuristics(m, n, k, nn, count, device))


def _b7__cx_lt_configs(m, n, k, quick=False):
    if m <= 0 or n <= 0 or k <= 0:
        return []
    count = 2 if quick else 32
    device = torch.cuda.current_device()
    result = []
    for layout in ("tn", "nn"):
        for row in _b7__cx_lt__heuristics(m, n, k, layout == "nn", count, device):
            candidate = ("lt", layout, *row)
            if candidate not in result:
                result.append(candidate)
    return result


def _b7__cx_lt_describe_config(cfg):
    fields = ("kind", "layout", "algorithm_id", "tile_id", "split_k",
              "reduction_scheme", "cta_swizzle", "custom_option", "stages_id",
              "inner_shape_id", "cluster_shape_id", "workspace_bytes", "cublaslt_version")
    if len(cfg) != len(fields):
        raise ValueError("invalid cuBLASLt configuration length")
    return dict(zip(fields, cfg))


def _b7__cx_lt_run(x, state, n, cfg):
    if len(cfg) != 13 or cfg[0] != "lt" or cfg[1] not in ("tn", "nn"):
        raise ValueError(f"invalid cuBLASLt configuration: {cfg!r}")
    if x.ndim != 2 or x.dtype != torch.bfloat16 or x.shape[1] != state["k"]:
        raise ValueError("incompatible BF16 activation tensor")
    if not 0 <= n <= state["n"]:
        raise ValueError("output width exceeds prepared weights")
    m, k = x.shape
    if m == 0 or n == 0:
        return torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    nn = cfg[1] == "nn"
    weight = state["wt"] if nn else state["w"]
    lda = weight.shape[1] if nn else k
    settings = tuple(int(value) for value in cfg[2:11])
    version = int(cfg[12])
    extension = _b7__cx_lt__extension()
    required = extension.lt_workspace(m, n, k, lda, nn, settings, version, x.device.index)
    # A distinct allocation per invocation is captured in the CUDA graph. No
    # shared mutable workspace can race across separately captured invocations.
    workspace = torch.empty((required,), dtype=torch.uint8, device=x.device)
    return extension.lt_run(x, weight, n, nn, settings, version, workspace)


_b7__cx__CHOICES = {'8/qkv_gate': ('tiny', ['tiny', 8, False]), '8/o': ('packed', ['packed', 64, 16, 128, 1, 4, 4]), '8/pooler': ('tiny', ['tiny', 8, False]), '8/unpooler': ('tiny', ['tiny', 16, False]), '16/qkv_gate': ('tma', ['tma', 64, 16, 128, 4, 4, 4]), '16/o': ('triton', ['mm', 64, 16, 256, 1, 4, 3]), '16/pooler': ('tma', ['tma', 64, 16, 128, 4, 4, 4]), '16/unpooler': ('torch_transpose', ['nn']), '32/qkv_gate': ('tma', ['tma', 64, 32, 128, 4, 4, 4]), '32/o': ('packed', ['packed', 64, 32, 128, 1, 4, 4]), '32/pooler': ('tma', ['tma', 64, 32, 128, 4, 4, 4]), '32/unpooler': ('torch_transpose', ['nn']), '128/qkv_gate': ('lt', ['lt', 'nn', 66, 17, 3, 2, 0, 0, 35, 0, 4, 3538960, 130101]), '128/o': ('torch_transpose', ['nn']), '128/pooler': ('torch', []), '128/unpooler': ('lt', ['lt', 'nn', 66, 18, 1, 0, 0, 3, 35, 0, 5, 0, 130101]), '256/qkv_gate': ('lt', ['lt', 'tn', 66, 184, 4, 2, 0, 2, 35, 0, 5, 9437200, 130101]), '256/o': ('torch', []), '256/pooler': ('torch_transpose', ['nn']), '256/unpooler': ('torch', []), '1/router': ('router_interleaved', ['simt', 'fp32', 1, 1, 8192, 1, 4, 1]), '8/router': ('router_interleaved', ['interleaved', 'block128', 16, 64, 256, 32, 4, 2]), '16/router': ('router', ['dot', 16, 32, 256, 32, 4, 2, False]), '32/router': ('router', ['dot', 32, 64, 256, 32, 4, 2, False]), '128/router': ('router_interleaved', ['interleaved', 'block128', 32, 64, 128, 32, 4, 2]), '256/router': ('router_interleaved', ['interleaved', 'block128', 32, 64, 128, 16, 4, 2]), '1/qkv_gate': ('triton', ['gemv', 1, 8192, 8]), '1/o': ('triton', ['gemv', 2, 1024, 4]), '1/pooler': ('triton', ['gemv', 1, 4096, 8]), '1/unpooler': ('triton', ['gemv', 1, 4096, 8])}


def _b7__cx_prepare(weights):
    _b7__cx_ti__extension()
    _b7__cx_lt__extension()
    return {
        'weights': weights,
        'lt': {name:_b7__cx_lt_prepare(w) for name,w in weights.items() if name!='router'},
        'router': _b7__cx_rt_prepare_router(weights['router']),
        'router_interleaved': _b7__cx_ri_prepare_router(weights['router']),
        'packed': {'o':_b7__cx_pa_prepare(weights['o'])},
        'transposed': {name:w.T.contiguous() for name,w in weights.items() if name!='router'},
    }




def _b7__cx_kernel_fn(data,state):
    x=data['x']
    layer=data['config']['layer']
    n=data['config']['n']
    m=x.shape[0]
    if m==1:bucket=1
    elif m<=8:bucket=8
    elif m<=16:bucket=16
    elif m<=32:bucket=32
    elif m<=192:bucket=128
    else:bucket=256
    family,cfg=_b7__cx__CHOICES[str(bucket)+'/'+layer]
    if layer=='router':
        if family=='router_interleaved':return _b7__cx_ri_run_router(x,state['router_interleaved'],cfg)
        return _b7__cx_rt_run_router(x,state['router'],cfg)
    w=state['weights'][layer]
    if m>384:return torch.mm(x,w[:n].T)
    if family=='triton':return _b7__cx_pr_run(x,w,n,cfg)
    if family=='cuda':return cu_run(x,w,n,cfg)
    if family=='tiny':return _b7__cx_ti_run(x,w,n,cfg)
    if family=='tma':return _b7__cx_tm_run(x,w,n,cfg)
    if family=='packed':return _b7__cx_pa_run(x,state['packed'][layer],n,cfg)
    if family=='lt':return _b7__cx_lt_run(x,state['lt'][layer],n,cfg)
    if family=='torch_transpose':return torch.mm(x,state['transposed'][layer][:,:n])
    return torch.mm(x,w[:n].T)

# ======================================================================== component: NVRTC helper
"""NVRTC compile + driver-API launch on torch's current stream (graph-capturable)."""
import ctypes
import os
import struct

import torch


class _b7__cu_Module:
    def __init__(self, src, name="k.cu", opts=()):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.cuda.init()
        torch.zeros(1, device="cuda")  # make sure the primary context exists
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        self._check(err, "nvrtcCreateProgram")
        major, minor = torch.cuda.get_device_capability()
        arch = f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode()
        o = [arch, b"-std=c++17", b"--use_fast_math"] + [x.encode() for x in opts]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(o), o)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        self.log = log.decode(errors="replace").strip("\x00 \n")
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + self.log)
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        self._check(err, "nvrtcGetCUBINSize")
        cubin = b" " * size
        (err,) = nvrtc.nvrtcGetCUBIN(prog, cubin)
        self._check(err, "nvrtcGetCUBIN")
        self.cubin = cubin
        err, self.module = driver.cuModuleLoadData(cubin)
        self._check(err, "cuModuleLoadData")
        self.funcs = {}

    @staticmethod
    def _check(err, what):
        if int(err) != 0:
            raise RuntimeError(f"{what} failed: {err}")

    def func(self, name, smem=0):
        key = (name, smem)
        if key not in self.funcs:
            d = self.driver
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            self._check(err, f"cuModuleGetFunction({name})")
            if smem > 48 * 1024:
                (err,) = d.cuFuncSetAttribute(
                    f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
                self._check(err, "cuFuncSetAttribute")
            self.funcs[key] = f
        return self.funcs[key]

    def attrs(self, name):
        d = self.driver
        f = self.func(name)
        out = {}
        for k in ("NUM_REGS", "LOCAL_SIZE_BYTES", "SHARED_SIZE_BYTES", "MAX_THREADS_PER_BLOCK"):
            err, v = d.cuFuncGetAttribute(getattr(d.CUfunction_attribute, "CU_FUNC_ATTRIBUTE_" + k), f)
            out[k] = v
        return out

    def launch(self, name, grid, block, smem, args):
        """args: packed bytes of the kernel's single struct parameter (see pack())."""
        f = self.func(name, smem)
        raw = ctypes.create_string_buffer(len(args) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, args, len(args))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = self.driver.CUstream(torch.cuda.current_stream().cuda_stream)
        gx, gy, gz = grid if isinstance(grid, tuple) else (grid, 1, 1)
        (err,) = self.driver.cuLaunchKernel(f, gx, gy, gz, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        self._check(err, f"launch {name}")


def _b7__cu_pack(*vals):
    """Pack kernel params: tensors/ints as 8-byte words (kernels take a struct of u64/i64/pointers)."""
    out = b""
    for v in vals:
        if isinstance(v, torch.Tensor):
            out += struct.pack("Q", v.data_ptr())
        elif isinstance(v, float):
            out += struct.pack("d", v)
        else:
            out += struct.pack("q", int(v))
    return out


# ======================================================================== component: K1s
"""K1r: register-streaming skinny projection for M <= 16, 12-bit lossless weights (or BF16).

No smem ring: every consumer warp streams its own contiguous slice of weight tiles from global straight into
registers (D k32-steps in flight), decodes, and runs mma.sync m16n8k16 (A = x from smem, B = W rows, 8-row groups).
x for the CTA's k range is resident in smem (one 3D TMA copy, 128B swizzle, TOKB rows).
CTA (cluster q, rank r) owns row groups [q*G/Q, (q+1)*G/Q) and k range [r*KC, r*KC+KC); warp (wr, wk) takes groups
[wr*GW, wr*GW+GW) x k32 steps [wk*NKT, wk*NKT+NKT). Per warp and k32 step the global block is
  COMP: [32 lanes x 8*GW B: low bf16 bytes of its GW tiles][32 lanes x 4*GW B: exponent codes]  (12 B/lane/tile)
  BF16: [32 lanes x 16*GW B]
Raw tiles (an exponent outside the window) keep exact BF16: bytes 0-7 in the low slot, 8-11 in the code slot, 12-15
in the CTA overflow area (smem, 128 B per raw tile, bulk-copied at start). Per-warp header [u64 raw mask over its
tiles (step-major), u32 overflow base]. Partials -> padded smem slices -> sum -> (cluster: push partner's half) -> out.
"""
import struct

import torch

_b7__k1_SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) P {
  TMap tmx;
  const u8* w; const u8* ovf; const uint4* hdr; u16* out;
  long long M, N, G, Q;
  u32 base2, ovstride, ovoff, pad;
  unsigned long long* trace;
};
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define TR(i) do { if (p.trace) p.trace[blockIdx.x * 16 + (i)] = gtime(); } while (0)
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u32 b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect_arrive(u32 b, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(u32 b, int ph) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}\n"
               :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma3d(u32 dst, const void* tmap, int c0, int c1, int c2, u32 bar) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
               :: "r"(dst), "l"(tmap), "r"(c0), "r"(c1), "r"(c2), "r"(bar) : "memory");
}
__device__ __forceinline__ void ldsm_x4(u32* a, u32 addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr));
}
__device__ __forceinline__ u32 lds32(u32 addr) { u32 v; asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(addr)); return v; }
__device__ __forceinline__ uint4 ldg128(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ uint2 ldg64(const void* p) {
  uint2 v; asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p)); return v;
}
__device__ __forceinline__ u32 ldg32(const void* p) {
  u32 v; asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v;
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 s) { u32 r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s)); return r; }
__device__ __forceinline__ void mma16816(float* c, const u32* a, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ u32 cluster_rank() { u32 r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ void st_dsmem4(u32 addr, u32 rank, float4 v) {
  u32 ra; asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(addr), "r"(rank));
  asm volatile("st.shared::cluster.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) {
  u32 r; asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ uint4 decode12(uint2 lo, u32 nw, u32 B4) {
  const u32 hiA = ((nw << 4) & 0x80808080u) | ((nw & 0x07070707u) + B4);
  const u32 hiB = (nw & 0x80808080u) | (((nw >> 4) & 0x07070707u) + B4);
  uint4 o;
  o.x = prmt(lo.x, hiA, 0x5140); o.y = prmt(lo.x, hiA, 0x7362);
  o.z = prmt(lo.y, hiB, 0x5140); o.w = prmt(lo.y, hiB, 0x7362);
  return o;
}

// per-lane payload of one k32 step for GW tiles
template <int GW, int COMP> struct Step { uint2 lo[COMP ? GW : 1]; u32 cd[COMP ? GW : 1]; uint4 bf[COMP ? 1 : GW]; };

template <int GW, int COMP>
__device__ __forceinline__ void load_step(Step<GW, COMP>& s, const u8* blk, int lane, int gc) {
  if (COMP) {
    const u8* p = blk;
#pragma unroll
    for (int j = 0; j + 1 < GW; j += 2) {
      if (j < gc) { const uint4 v = ldg128(p + lane * 16); s.lo[j] = make_uint2(v.x, v.y); s.lo[j + 1] = make_uint2(v.z, v.w); }
      p += 512;
    }
    if (GW & 1) { if (GW - 1 < gc) s.lo[GW - 1] = ldg64(p + lane * 8); p += 256; }
#pragma unroll
    for (int j = 0; j + 3 < GW; j += 4) {
      if (j < gc) { const uint4 v = ldg128(p + lane * 16); s.cd[j] = v.x; s.cd[j + 1] = v.y; s.cd[j + 2] = v.z; s.cd[j + 3] = v.w; }
      p += 512;
    }
    constexpr int J0 = GW & ~3;
    if ((GW & 3) >= 2) { if (J0 < gc) { const uint2 v = ldg64(p + lane * 8); s.cd[J0] = v.x; s.cd[J0 + 1] = v.y; } p += 256; }
    if (GW & 1) { if (GW - 1 < gc) s.cd[GW - 1] = ldg32(p + lane * 4); }
  } else {
#pragma unroll
    for (int j = 0; j < GW; ++j) if (j < gc) s.bf[j] = ldg128(blk + j * 512 + lane * 16);
  }
}

template <int C, int TOKB, int GW, int WR, int WK, int NKT, int D, int KT, int COMP>
__device__ __forceinline__ void k1s_body(const P& p) {
  constexpr int MT = (TOKB + 15) / 16;
  constexpr int TOKP = MT * 16;
  constexpr int NCW = WR * WK;
  constexpr int KC = KT * 32 / C;
  constexpr int RG = WR * GW;
  constexpr int RS = RG * 8 + 8;
  constexpr int SB = COMP ? 32 * 12 * GW : 32 * 16 * GW;   // global bytes per warp step
  static_assert(NKT * WK * 32 * C == KT * 32, "K split");
  static_assert(WK % 2 == 0, "x chunks are whole 64-k boxes");
  static_assert(NKT + 1 <= 128, "barriers");
  extern __shared__ __align__(1024) unsigned char sm[];
  const u32 bar0 = smem_u32(sm);                    // [0] overflow, [1 + i] x chunk i
  const u32 xs = smem_u32(sm) + 1024;
  const u32 ovs = smem_u32(sm) + p.ovoff;

  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const u32 r = (C > 1) ? cluster_rank() : 0;
  const int cta = blockIdx.x;
  const int q = cta / C;
  const int M = (int)p.M;
  const int g0 = (int)((long long)q * p.G / p.Q), g1 = (int)((long long)(q + 1) * p.G / p.Q);
  const int ng = g1 - g0;
  const int wr = wid / WK, wk = wid % WK;
  const int gcount = ng - wr * GW;
  // warp stream: CTA region = (g0*KT + r*ng*(KC/32)) tiles; warp (wr, wk) block = (wr*WK + wk) * NKT steps
  // CTA regions are padded to RG groups: cta * NCW * NKT steps
  const u8* ws = p.w + ((size_t)cta * NCW + (size_t)(wr * WK + wk)) * NKT * SB;

  if (threadIdx.x == 0) {
    TR(0);
    for (int i = 0; i <= NKT; ++i) mbar_init(bar0 + 8 * i, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    // x chunk i = k32 steps [i*WK, i*WK+WK) = WK/2 boxes of 64 k
#pragma unroll 1
    for (int i = 0; i < NKT; ++i) {
      mbar_expect_arrive(bar0 + 8 * (1 + i), TOKB * WK * 64);
      tma3d(xs + i * (WK / 2) * TOKB * 128, &p.tmx, 0, 0, (int)r * (KC / 64) + i * (WK / 2), bar0 + 8 * (1 + i));
    }
    if (COMP && p.ovstride) {
      mbar_expect_arrive(bar0, p.ovstride);
      bulk_g2s(ovs, p.ovf + (size_t)cta * p.ovstride, p.ovstride, bar0);
    }
  }
  __syncthreads();
  Step<GW, COMP> buf[D];
#pragma unroll
  for (int d = 0; d < D && d < NKT; ++d) load_step<GW, COMP>(buf[d], ws + (size_t)d * SB, lane, gcount);
  uint4 hd = make_uint4(0, 0, 0, 0);
  if (COMP) hd = p.hdr[(size_t)cta * NCW + wid];
  const unsigned long long rawm = ((unsigned long long)hd.y << 32) | hd.x;
  const u32 ovbase = hd.z;
  bool ovready = false;

  float acc[GW][MT][4];
#pragma unroll
  for (int j = 0; j < GW; ++j)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[j][mt][c] = 0.f;
  u32 xrow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) xrow[mt] = (u32)((mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * 128);
  const u32 xsw = (u32)(lane & 7);   // row & 7 for every mt
  const u32 base2 = p.base2;
  if (threadIdx.x == 0) TR(1);

#pragma unroll
  for (int i = 0; i < NKT; ++i) {
    const int slot = i % D;
    const int kt = i * WK + wk;                  // k32 step within the CTA range (interleaved across warps)
    mbar_wait(bar0 + 8 * (1 + i), 0);
    u32 a[MT][2][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int stp = 0; stp < 2; ++stp) {
        const u32 ch = 4 * (kt & 1) + 2 * stp + (lane >> 4);
        ldsm_x4(a[mt][stp], xs + (kt >> 1) * TOKB * 128 + xrow[mt] + ((ch ^ xsw) << 4));
      }
    uint4 bt[GW];
#pragma unroll
    for (int j = 0; j < GW; ++j) {
      if (COMP) {
        const int ti = i * GW + j;
        if ((rawm >> ti) & 1ull) {
          if (!ovready) { mbar_wait(bar0, 0); ovready = true; }
          const int ov = __popcll(rawm & ((1ull << ti) - 1ull));
          bt[j] = make_uint4(buf[slot].lo[j].x, buf[slot].lo[j].y, buf[slot].cd[j], lds32(ovs + (ovbase + ov) * 128 + lane * 4));
        } else {
          bt[j] = decode12(buf[slot].lo[j], buf[slot].cd[j], base2);
        }
      } else {
        bt[j] = buf[slot].bf[j];
      }
    }
    if (i + D < NKT) load_step<GW, COMP>(buf[slot], ws + (size_t)(i + D) * SB, lane, gcount);
#pragma unroll
    for (int j = 0; j < GW; ++j)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        mma16816(acc[j][mt], a[mt][0], bt[j].x, bt[j].y);
        mma16816(acc[j][mt], a[mt][1], bt[j].z, bt[j].w);
      }
  }
  if (threadIdx.x == 0) TR(2);
  __syncthreads();                                // everyone is done with x
  if (threadIdx.x == 0) TR(3);
  if (C > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");  // my x area is free for pushes
  float* part = (float*)(sm + 1024);
  float* red2 = part + WK * TOKP * RS;
  float* recv = red2 + TOKP * RS;
#pragma unroll
  for (int j = 0; j < GW; ++j) {
    if (j < gcount) {
      const int n = (wr * GW + j) * 8 + 2 * (lane & 3);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int t0 = mt * 16 + (lane >> 2);
        *(float2*)&part[(wk * TOKP + t0) * RS + n] = make_float2(acc[j][mt][0], acc[j][mt][1]);
        *(float2*)&part[(wk * TOKP + t0 + 8) * RS + n] = make_float2(acc[j][mt][2], acc[j][mt][3]);
      }
    }
  }
  if (threadIdx.x == 0) TR(7);
  __syncthreads();
  if (threadIdx.x == 0) TR(6);
  const int nq = ng * 2;
  if (C > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
  const int qa = (int)(r * nq / C), qb = (int)((r + 1) * nq / C);
  for (int e = threadIdx.x; e < M * nq; e += blockDim.x) {
    const int t = e / nq, qd = e % nq, n = qd * 4;
    float4 s = *(const float4*)&part[t * RS + n];
#pragma unroll
    for (int w = 1; w < WK; ++w) {
      const float4 v = *(const float4*)&part[(w * TOKP + t) * RS + n];
      s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
    if (C == 1) {
      uint2 o; o.x = pack_bf2(s.x, s.y); o.y = pack_bf2(s.z, s.w);
      *(uint2*)(p.out + (size_t)t * p.N + (size_t)g0 * 8 + n) = o;
    } else if (qd < qa || qd >= qb) {
      u32 owner = 0;
#pragma unroll
      for (int c2 = 1; c2 < C; ++c2) if (qd >= (c2 * nq) / C) owner = (u32)c2;
      st_dsmem4(smem_u32(&recv[t * RS + n]), owner, s);
    } else {
      *(float4*)&red2[t * RS + n] = s;
    }
  }
  if (C > 1) {
    cluster_sync_all();
    if (threadIdx.x == 0) TR(4);
    const int nmy = qb - qa;
    for (int e = threadIdx.x; e < M * nmy; e += blockDim.x) {
      const int t = e / nmy, n = (qa + e % nmy) * 4;
      float4 s = *(const float4*)&red2[t * RS + n];
      const float4 v = *(const float4*)&recv[t * RS + n];
      s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
      uint2 o; o.x = pack_bf2(s.x, s.y); o.y = pack_bf2(s.z, s.w);
      *(uint2*)(p.out + (size_t)t * p.N + (size_t)g0 * 8 + n) = o;
    }
  } else if (threadIdx.x == 0) TR(4);
  if (threadIdx.x == 0) TR(5);
}
"""

_b7__k1_LAYER_K = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "pooler1": 8192, "unpooler": 4096, "unpslice": 4096}
# layer -> (C, Q, GW, WR, WK, D)
_b7__k1_CFG = {
    "qkv_gate": (2, 66, 5, 1, 16, 3),
    "pooler": (2, 66, 4, 2, 8, 4),
    "pooler1": (1, 132, 4, 1, 16, 4),
    "o": (1, 132, 4, 2, 8, 4),
    "unpooler": (1, 132, 4, 2, 8, 4),
    "unpslice": (1, 128, 1, 1, 16, 4),
}
_b7__k1_VARIANT = {("pooler", 8): "pooler1"}   # (layer, tokb) -> config key
_b7__k1_OVMAX = 8192
_b7__k1_TOKBS = (8, 16)
_b7__k1_SMEM_MAX = 227 * 1024


def _b7__k1_kname(layer, tokb, comp):
    return f"k1s_{layer}_{tokb}_{comp}"


def _b7__k1__dims(layer, tokb, comp):
    c, q, gw, wr, wk, d = _b7__k1_CFG[layer]
    kc = _b7__k1_LAYER_K[layer] // c
    tokp = (tokb + 15) // 16 * 16
    xb = tokb * kc * 2
    if xb > 200 * 1024:
        return dict(ovoff=0, smem=10**9, nkt=kc // 32 // wk)
    epi = (wk + 2) * tokp * (gw * wr * 8 + 8) * 4
    ovoff = 1024 + (max(xb + 1024, epi) + 1023) // 1024 * 1024
    smem = max(ovoff + (_b7__k1_OVMAX if comp else 0), 120 * 1024)  # one CTA per SM
    return dict(ovoff=ovoff, smem=smem, nkt=kc // 32 // wk)


def _b7__k1_source(layers=None, tokbs=(8, 16)):
    s = _b7__k1_SRC
    for layer, (c, q, gw, wr, wk, d) in _b7__k1_CFG.items():
        if layers and layer not in layers:
            continue
        kt = _b7__k1_LAYER_K[layer] // 32
        for tokb in tokbs:
            for comp in (0, 1):
                dm = _b7__k1__dims(layer, tokb, comp)
                if dm["smem"] > _b7__k1_SMEM_MAX:
                    continue
                nt = 32 * wr * wk
                cl = f"__cluster_dims__({c},1,1) " if c > 1 else ""
                s += (f'extern "C" __global__ void {cl}__launch_bounds__({nt},1) {_b7__k1_kname(layer, tokb, comp)}'
                      f'(const __grid_constant__ P p) {{ k1s_body<{c},{tokb},{gw},{wr},{wk},{dm["nkt"]},{d},{kt},{comp}>(p); }}\n')
    return s


def _b7__k1__frag(t, n, k):
    return t.contiguous().view(n // 8, 8, k // 32, 2, 2, 4, 2).permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 32, 32, 8)


def _b7__k1_compress(w):
    u = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e = u >> 15, (u >> 7) & 0xFF
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    best, H = -1, 0
    for h in range(0, 121):
        inside = int(cs[2 * h + 16] - cs[2 * h])
        if inside > best:
            best, H = inside, h
    idx = (e >> 1) - H
    esc = (idx < 0) | (idx > 7)
    code = torch.where(esc, torch.zeros_like(e), (s << 3) | idx)
    lo = torch.where(esc, torch.zeros_like(u), u & 0xFF)
    return code, lo, H, esc


def _b7__k1_prepare_layer(w, layer, comp):
    c, q, gw, wr, wk, d = _b7__k1_CFG[layer]
    n, k = w.shape
    G, KT = n // 8, k // 32
    kc = k // c
    nkt = kc // 32 // wk
    dev = w.device
    raw16 = _b7__k1__frag(w, n, k).contiguous().view(G, KT, 32, 8)        # bf16 [G, KT, lane, 8]
    if comp:
        code, lo, H, esc = _b7__k1_compress(w)
        cl = _b7__k1__frag(lo, n, k).to(torch.uint8)                     # [G, KT, 32, 8]
        cc = _b7__k1__frag(code, n, k).to(torch.int32)
        cw = (cc[..., 0:4] | (cc[..., 4:8] << 4)).to(torch.uint8)  # [G, KT, 32, 4]
        rawt = _b7__k1__frag(esc.to(torch.int32), n, k).reshape(G, KT, 256).any(-1)   # [G, KT]
        rb = raw16.view(torch.uint8).view(G, KT, 32, 16)
        cl = torch.where(rawt[..., None, None], rb[..., 0:8], cl)
        cw = torch.where(rawt[..., None, None], rb[..., 8:12], cw)
        ov = rb[..., 12:16]                                      # [G, KT, 32, 4]
    parts, hdrs, ovs = [], [], []
    ovstride = 0
    hdr_all = torch.zeros((q * c, wr * wk, 4), dtype=torch.int64)
    for qq in range(q):
        g0, g1 = qq * G // q, (qq + 1) * G // q
        ng = g1 - g0
        # pad groups to wr*gw for the per-warp layout
        pad = wr * gw - ng
        def padg(t):
            if pad <= 0:
                return t
            return torch.cat([t, torch.zeros((pad,) + tuple(t.shape[1:]), dtype=t.dtype, device=t.device)], 0)
        if comp:
            TL = padg(cl[g0:g1]).view(wr, gw, c, nkt, wk, 32, 8).permute(2, 0, 4, 3, 1, 5, 6)   # [c,wr,wk,nkt,gw,32,8]
            TC = padg(cw[g0:g1]).view(wr, gw, c, nkt, wk, 32, 4).permute(2, 0, 4, 3, 1, 5, 6)   # [c,wr,wk,nkt,gw,32,4]
            subs = []
            for pp in range(gw // 2):
                subs.append(torch.cat([TL[..., 2 * pp, :, :], TL[..., 2 * pp + 1, :, :]], -1))
            if gw & 1:
                subs.append(TL[..., gw - 1, :, :])
            for qd in range(gw // 4):
                subs.append(torch.cat([TC[..., 4 * qd + i, :, :] for i in range(4)], -1))
            j0 = gw & ~3
            if (gw & 3) >= 2:
                subs.append(torch.cat([TC[..., j0, :, :], TC[..., j0 + 1, :, :]], -1))
            if gw & 1:
                subs.append(TC[..., gw - 1, :, :])
            blk = torch.cat([t_.reshape(c, wr, wk, nkt, -1) for t_ in subs], -1)   # [c, wr, wk, nkt, 32*12*gw]
            R = padg(rawt[g0:g1].to(torch.uint8)).view(wr, gw, c, nkt, wk).permute(2, 0, 4, 3, 1).reshape(c, wr * wk, nkt * gw)
            O = padg(ov[g0:g1]).view(wr, gw, c, nkt, wk, 32, 4).permute(2, 0, 4, 3, 1, 5, 6).reshape(c, wr * wk, nkt * gw, 128)
            for rr in range(c):
                cta = qq * c + rr
                obytes = []
                base = 0
                for wi in range(wr * wk):
                    bits = R[rr, wi].cpu()
                    assert nkt * gw <= 64
                    m = 0
                    for ti in range(nkt * gw):
                        if bits[ti]:
                            m |= 1 << ti
                            obytes.append(O[rr, wi, ti])
                    hdr_all[cta, wi, 0] = m & 0xFFFFFFFF
                    hdr_all[cta, wi, 1] = m >> 32
                    hdr_all[cta, wi, 2] = base
                    base += int(bits.sum())
                ovs.append(torch.stack(obytes).view(-1) if obytes else torch.zeros(0, dtype=torch.uint8, device=dev))
                ovstride = max(ovstride, base * 128)
        else:
            B = padg(raw16[g0:g1].contiguous().view(torch.uint8).view(g1 - g0, KT, 32, 16))
            blk = B.view(wr, gw, c, nkt, wk, 32, 16).permute(2, 0, 4, 3, 1, 5, 6).reshape(c, wr, wk, nkt, 32 * gw * 16)
        # only the first ng groups' bytes are real; keep the padded layout only when ng == wr*gw
        if pad > 0:
            # drop padding: the kernel indexes by ng-sized regions; rebuild with true ng (warps beyond ng read zeros)
            pass
        parts.append(blk.reshape(-1))
    if comp:
        ovstride = (ovstride + 127) // 128 * 128
        if ovstride > _b7__k1_OVMAX:
            raise RuntimeError(f"{layer}: overflow {ovstride} > OVMAX")
        ovf_all = torch.zeros((q * c, max(ovstride, 128)), dtype=torch.uint8, device=dev)
        for i, ob in enumerate(ovs):
            if ob.numel():
                ovf_all[i, :ob.numel()] = ob
        hdr = hdr_all.to(torch.int32).view(-1).to(dev)
        return {"w": torch.cat(parts), "ovf": ovf_all.view(-1), "hdr": hdr, "ovstride": ovstride,
                "base2": H * 0x01010101, "n_raw": int(rawt.sum()), "raw_frac": float(rawt.float().mean()), "base": 2 * H}
    z = torch.zeros(128, dtype=torch.uint8, device=dev)
    return {"w": torch.cat(parts), "ovf": z, "hdr": torch.zeros(4 * q * c * wr * wk, dtype=torch.int32, device=dev),
            "ovstride": 0, "base2": 0, "n_raw": 0, "base": None}


_b7__k1__TM_CACHE = {}


def _b7__k1_x_tmap3(x, m, k, tokb, nb):
    key = (x.data_ptr(), m, k, tokb, nb)
    if key in _b7__k1__TM_CACHE:
        return _b7__k1__TM_CACHE[key]
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(3), x.data_ptr(),
        [d.cuuint64_t(64), d.cuuint64_t(m), d.cuuint64_t(k // 64)], [d.cuuint64_t(k * 2), d.cuuint64_t(128)],
        [d.cuuint32_t(64), d.cuuint32_t(tokb), d.cuuint32_t(nb)], [d.cuuint32_t(1)] * 3,
        d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    if int(err) != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed: {err}")
    b = b"".join(struct.pack("<Q", int(v)) for v in tm.opaque)
    if len(_b7__k1__TM_CACHE) > 64:
        _b7__k1__TM_CACHE.clear()
    _b7__k1__TM_CACHE[key] = b
    return b


_b7__k1_TRACE = None


def _b7__k1_cfg_key(layer, m, n):
    lname = "unpslice" if (layer == "unpooler" and n == 1024) else layer
    tokb = 8 if m <= 8 else 16
    return _b7__k1_VARIANT.get((lname, tokb), lname), tokb


def _b7__k1_launch(mod, layer, ent, x, out, m, n, comp):
    lname, tokb = _b7__k1_cfg_key(layer, m, n)
    c, q, gw, wr, wk, d = _b7__k1_CFG[lname]
    k = _b7__k1_LAYER_K[lname]
    kc = k // c
    dm = _b7__k1__dims(lname, tokb, comp)
    tm = _b7__k1_x_tmap3(x, m, k, tokb, wk // 2)
    args = tm + struct.pack("<QQQQqqqqIIIIQ", ent["w"].data_ptr(), ent["ovf"].data_ptr(), ent["hdr"].data_ptr(),
                            out.data_ptr(), m, n, n // 8, q, ent["base2"], ent["ovstride"], dm["ovoff"], 0,
                            _b7__k1_TRACE.data_ptr() if _b7__k1_TRACE is not None else 0)
    args += b"\0" * (256 - len(args))
    mod.launch(_b7__k1_kname(lname, tokb, comp), q * c, 32 * wr * wk, dm["smem"], args)


# ======================================================================== component: K1t (M=32)
"""K1r: register-streaming skinny projection for M <= 16, 12-bit lossless weights (or BF16).

No smem ring: every consumer warp streams its own contiguous slice of weight tiles from global straight into
registers (D k32-steps in flight), decodes, and runs mma.sync m16n8k16 (A = x from smem, B = W rows, 8-row groups).
x for the CTA's k range is resident in smem (one 3D TMA copy, 128B swizzle, TOKB rows).
CTA (cluster q, rank r) owns row groups [q*G/Q, (q+1)*G/Q) and k range [r*KC, r*KC+KC); warp (wr, wk) takes groups
[wr*GW, wr*GW+GW) x k32 steps [wk*NKT, wk*NKT+NKT). Per warp and k32 step the global block is
  COMP: [32 lanes x 8*GW B: low bf16 bytes of its GW tiles][32 lanes x 4*GW B: exponent codes]  (12 B/lane/tile)
  BF16: [32 lanes x 16*GW B]
Raw tiles (an exponent outside the window) keep exact BF16: bytes 0-7 in the low slot, 8-11 in the code slot, 12-15
in the CTA overflow area (smem, 128 B per raw tile, bulk-copied at start). Per-warp header [u64 raw mask over its
tiles (step-major), u32 overflow base]. Partials -> padded smem slices -> sum -> (cluster: push partner's half) -> out.
"""
import struct

import torch

_b7__k1t_SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) P {
  TMap tmx;
  const u8* w; const u8* ovf; const uint4* hdr; u16* out;
  long long M, N, G, Q;
  u32 base2, ovstride, ovoff, rcvoff;
  unsigned long long* trace;
};
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define TR(i) do { if (p.trace) p.trace[blockIdx.x * 16 + (i)] = gtime(); } while (0)
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u32 b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect_arrive(u32 b, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(u32 b, int ph) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}\n"
               :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma3d(u32 dst, const void* tmap, int c0, int c1, int c2, u32 bar) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
               :: "r"(dst), "l"(tmap), "r"(c0), "r"(c1), "r"(c2), "r"(bar) : "memory");
}
__device__ __forceinline__ void ldsm_x4(u32* a, u32 addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr));
}
__device__ __forceinline__ u32 lds32(u32 addr) { u32 v; asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(addr)); return v; }
__device__ __forceinline__ uint4 ldg128(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ uint2 ldg64(const void* p) {
  uint2 v; asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p)); return v;
}
__device__ __forceinline__ u32 ldg32(const void* p) {
  u32 v; asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v;
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 s) { u32 r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s)); return r; }
__device__ __forceinline__ void mma16816(float* c, const u32* a, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ u32 cluster_rank() { u32 r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ void st_async4(u32 addr, u32 bar, u32 rank, float4 v) {
  u32 ra, rb;
  asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(addr), "r"(rank));
  asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rb) : "r"(bar), "r"(rank));
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];"
               :: "r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rb) : "memory");
}
__device__ __forceinline__ void st_dsmem4(u32 addr, u32 rank, float4 v) {
  u32 ra; asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(addr), "r"(rank));
  asm volatile("st.shared::cluster.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) {
  u32 r; asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ uint4 decode12(uint2 lo, u32 nw, u32 B4) {
  const u32 hiA = ((nw << 4) & 0x80808080u) | ((nw & 0x07070707u) + B4);
  const u32 hiB = (nw & 0x80808080u) | (((nw >> 4) & 0x07070707u) + B4);
  uint4 o;
  o.x = prmt(lo.x, hiA, 0x5140); o.y = prmt(lo.x, hiA, 0x7362);
  o.z = prmt(lo.y, hiB, 0x5140); o.w = prmt(lo.y, hiB, 0x7362);
  return o;
}

// per-lane payload of one k32 step for GW tiles
template <int GW, int COMP> struct Step { uint2 lo[COMP ? GW : 1]; u32 cd[COMP ? GW : 1]; uint4 bf[COMP ? 1 : GW]; };

template <int GW, int COMP>
__device__ __forceinline__ void load_step(Step<GW, COMP>& s, const u8* blk, int lane, int gc) {
  if (COMP) {
    const u8* p = blk;
#pragma unroll
    for (int j = 0; j + 1 < GW; j += 2) {
      if (j < gc) { const uint4 v = ldg128(p + lane * 16); s.lo[j] = make_uint2(v.x, v.y); s.lo[j + 1] = make_uint2(v.z, v.w); }
      p += 512;
    }
    if (GW & 1) { if (GW - 1 < gc) s.lo[GW - 1] = ldg64(p + lane * 8); p += 256; }
#pragma unroll
    for (int j = 0; j + 3 < GW; j += 4) {
      if (j < gc) { const uint4 v = ldg128(p + lane * 16); s.cd[j] = v.x; s.cd[j + 1] = v.y; s.cd[j + 2] = v.z; s.cd[j + 3] = v.w; }
      p += 512;
    }
    constexpr int J0 = GW & ~3;
    if ((GW & 3) >= 2) { if (J0 < gc) { const uint2 v = ldg64(p + lane * 8); s.cd[J0] = v.x; s.cd[J0 + 1] = v.y; } p += 256; }
    if (GW & 1) { if (GW - 1 < gc) s.cd[GW - 1] = ldg32(p + lane * 4); }
  } else {
#pragma unroll
    for (int j = 0; j < GW; ++j) if (j < gc) s.bf[j] = ldg128(blk + j * 512 + lane * 16);
  }
}

// HALF layout: per step [GW tiles][32 lanes][8 B lo] then [GW tiles][32 lanes][4 B codes]; tile hj loads lanes < 16 only
template <int GW, int COMP>
__device__ __forceinline__ void load_step_h(Step<GW, COMP>& s, const u8* blk, int lane, int gc, int hj) {
#pragma unroll
  for (int j = 0; j < GW; ++j) {
    if (j < gc && (j != hj || lane < 16)) {
      if (COMP) { s.lo[j] = ldg64(blk + j * 256 + lane * 8); s.cd[j] = ldg32(blk + GW * 256 + j * 128 + lane * 4); }
      else s.bf[j] = ldg128(blk + j * 512 + lane * 16);
    }
  }
}

template <int C, int TOKB, int GW, int WR, int WK, int NKT, int D, int KT, int COMP, int R, int PW, int HALF>
__device__ __forceinline__ void k1t_body(const P& p) {
  constexpr int MT = (TOKB + 15) / 16;
  constexpr int TOKP = MT * 16;
  constexpr int NCW = WR * WK;
  constexpr int KC = KT * 32 / C;
  constexpr int RG = WR * GW;
  constexpr int RS = RG * 8 + 8;
  constexpr int SB = COMP ? 32 * 12 * GW : 32 * 16 * GW;   // global bytes per warp step
  static_assert(NKT * WK * 32 * C == KT * 32, "K split");
  static_assert(WK % 2 == 0, "x chunks are whole 64-k boxes");
  static_assert(8 * (1 + 2 * R) + 4 * R <= 1016, "barrier area");   // [1016] = split-K receive barrier
  extern __shared__ __align__(1024) unsigned char sm[];
  // barriers: [0] overflow, [1 + j] x slot j full, [1 + R + j] x slot j empty (NCW consumer warps)
  const u32 bar0 = smem_u32(sm);
  constexpr u32 CHB = TOKB * WK * 64;               // x chunk bytes (WK k32 steps)
  const u32 xs = smem_u32(sm) + 1024;
  const u32 ovs = smem_u32(sm) + p.ovoff;

  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const u32 r = (C > 1) ? cluster_rank() : 0;
  const int cta = blockIdx.x;
  const int q = cta / C;
  const int M = (int)p.M;
  // rows of this cluster: HALF -> 4-row units [u0, u1) (last 8-row group may be half), else 8-row groups
  const int u0 = (int)((long long)q * p.G / p.Q), u1 = (int)((long long)(q + 1) * p.G / p.Q);
  const int r0 = HALF ? 4 * u0 : 8 * u0;
  const int nrows = HALF ? 4 * (u1 - u0) : 8 * (u1 - u0);
  const int ng = (nrows + 7) / 8;
  const int g0 = r0 / 8;   // (only used when !HALF)
  const int wr = wid / WK, wk = wid % WK;
  const int gcount = ng - wr * GW;
  const int hj = (HALF && (nrows & 4)) ? (ng - 1 - wr * GW) : -1;   // this warp's half group index (or -1/out of range)
  // warp stream: CTA region = (g0*KT + r*ng*(KC/32)) tiles; warp (wr, wk) block = (wr*WK + wk) * NKT steps
  // CTA regions are padded to RG groups: cta * NCW * NKT steps
  const u8* ws = p.w + ((size_t)cta * NCW + (size_t)(wr * WK + wk)) * NKT * SB;

  if ((PW ? (wid == NCW) : (wid == 0)) && lane == 0) {
    TR(0);
    mbar_init(bar0, 1);
    for (int j = 0; j < R; ++j) { mbar_init(bar0 + 8 * (1 + j), 1); *(u32*)(sm + 8 * (1 + 2 * R) + 4 * j) = 0u; }
    if (C > 1) {
      // split-K partials from the other ranks of the cluster land in a dedicated buffer (not aliasing x):
      // (C-1) ranks x M tokens x my quads x 16 B
      const int nq0 = nrows / 4;
      const int nmy0 = (int)((r + 1) * nq0 / C) - (int)(r * nq0 / C);
      mbar_init(bar0 + 1016, 1);
      mbar_expect_arrive(bar0 + 1016, (u32)((C - 1) * M * nmy0 * 16));
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#pragma unroll 1
    for (int i = 0; i < R && i < NKT; ++i) {
      mbar_expect_arrive(bar0 + 8 * (1 + i), CHB);
      tma3d(xs + i * CHB, &p.tmx, 0, 0, (int)r * (KC / 64) + i * (WK / 2), bar0 + 8 * (1 + i));
    }
    if (COMP && p.ovstride) {
      mbar_expect_arrive(bar0, p.ovstride);
      bulk_g2s(ovs, p.ovf + (size_t)cta * p.ovstride, p.ovstride, bar0);
    }
  }
  __syncthreads();
  if (C > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");  // receive barriers initialised
  float acc[GW][MT][4];
#pragma unroll
  for (int j = 0; j < GW; ++j)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[j][mt][c] = 0.f;
  if (PW && wid == NCW) {
    if (lane == 0) {
#pragma unroll 1
      for (int i = R; i < NKT; ++i) {
        const int j = i % R;
        mbar_wait(bar0 + 8 * (1 + R + j), ((i / R) - 1) & 1);
        mbar_expect_arrive(bar0 + 8 * (1 + j), CHB);
        tma3d(xs + j * CHB, &p.tmx, 0, 0, (int)r * (KC / 64) + i * (WK / 2), bar0 + 8 * (1 + j));
      }
    }
    __syncwarp();
  } else {
  Step<GW, COMP> buf[D];
#pragma unroll
  for (int d = 0; d < D && d < NKT; ++d) if (HALF) load_step_h<GW, COMP>(buf[d], ws + (size_t)d * SB, lane, gcount, hj); else load_step<GW, COMP>(buf[d], ws + (size_t)d * SB, lane, gcount);
  uint4 hd = make_uint4(0, 0, 0, 0), hd2 = make_uint4(0, 0, 0, 0);
  if (COMP) { hd = p.hdr[((size_t)cta * NCW + wid) * 2]; hd2 = p.hdr[((size_t)cta * NCW + wid) * 2 + 1]; }
  constexpr int MW = (NKT * GW + 31) / 32;                                   // mask words actually used
  const u32 rw_[4] = {hd.x, MW > 1 ? hd.y : 0u, MW > 2 ? hd.z : 0u, MW > 3 ? hd.w : 0u};
  const u32 ovbase = hd2.x;
  bool ovready = false;
  u32 xrow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) xrow[mt] = (u32)((mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * 128);
  const u32 xsw = (u32)(lane & 7);   // row & 7 for every mt
  const u32 base2 = p.base2;
  if (threadIdx.x == 0) TR(1);

#pragma unroll
  for (int i = 0; i < NKT; ++i) {
    const int slot = i % D;
    const int kt = i * WK + wk;                  // k32 step within the CTA range (interleaved across warps)
    const u32 xsl = xs + (i % R) * CHB;          // x slot of chunk i
    mbar_wait(bar0 + 8 * (1 + i % R), (i / R) & 1);
    u32 a[MT][2][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int stp = 0; stp < 2; ++stp) {
        const u32 ch = 4 * (kt & 1) + 2 * stp + (lane >> 4);
        ldsm_x4(a[mt][stp], xsl + (wk >> 1) * TOKB * 128 + xrow[mt] + ((ch ^ xsw) << 4));
      }
    if (R < NKT && i + R < NKT) {
      // the last warp to finish reading chunk i refills its slot with chunk i + R
      __syncwarp();
      if (lane == 0) {
        u32* cnt = (u32*)(sm + 8 * (1 + 2 * R) + 4 * (i % R));
        __threadfence_block();
        const u32 old = atomicAdd(cnt, 1u);
        if (old == NCW - 1) {
          *cnt = 0;
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          const int j = i % R, i2 = i + R;
          mbar_expect_arrive(bar0 + 8 * (1 + j), CHB);
          tma3d(xs + j * CHB, &p.tmx, 0, 0, (int)r * (KC / 64) + i2 * (WK / 2), bar0 + 8 * (1 + j));
        }
      }
    }
    uint4 bt[GW];
#pragma unroll
    for (int j = 0; j < GW; ++j) {
      if (COMP) {
        const int ti = i * GW + j;
        const bool israw = ((rw_[ti >> 5] >> (ti & 31)) & 1u) != 0;
        if (israw) {
          if (!ovready) { mbar_wait(bar0, 0); ovready = true; }
          int ov = __popc(rw_[ti >> 5] & ((1u << (ti & 31)) - 1u));
#pragma unroll
          for (int wq = 0; wq < MW; ++wq) if (wq < (ti >> 5)) ov += __popc(rw_[wq]);
          bt[j] = make_uint4(buf[slot].lo[j].x, buf[slot].lo[j].y, buf[slot].cd[j], lds32(ovs + (ovbase + ov) * 128 + lane * 4));
        } else {
          bt[j] = decode12(buf[slot].lo[j], buf[slot].cd[j], base2);
        }
      } else {
        bt[j] = buf[slot].bf[j];
      }
    }
    if (i + D < NKT) { if (HALF) load_step_h<GW, COMP>(buf[slot], ws + (size_t)(i + D) * SB, lane, gcount, hj); else load_step<GW, COMP>(buf[slot], ws + (size_t)(i + D) * SB, lane, gcount); }
#pragma unroll
    for (int j = 0; j < GW; ++j)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        mma16816(acc[j][mt], a[mt][0], bt[j].x, bt[j].y);
        mma16816(acc[j][mt], a[mt][1], bt[j].z, bt[j].w);
      }
  }
  if (threadIdx.x == 0) TR(2);
  // partial stores happen below, after the barrier; keep acc live
  }
  __syncthreads();                                // everyone is done with x
  if (threadIdx.x == 0) TR(3);
  float* part = (float*)(sm + 1024);
  float* red2 = part + WK * TOKP * RS;
  float* recv = (float*)(sm + p.rcvoff);
#pragma unroll
  for (int j = 0; j < GW; ++j) {
    if (j < gcount && wid < NCW && (wr * GW + j) * 8 + 2 * (lane & 3) < nrows) {
      const int n = (wr * GW + j) * 8 + 2 * (lane & 3);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int t0 = mt * 16 + (lane >> 2);
        *(float2*)&part[(wk * TOKP + t0) * RS + n] = make_float2(acc[j][mt][0], acc[j][mt][1]);
        *(float2*)&part[(wk * TOKP + t0 + 8) * RS + n] = make_float2(acc[j][mt][2], acc[j][mt][3]);
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) TR(6);
  const int nq = nrows / 4;
  if (C > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");   // pairs with the arrive at start
  const int qa = (int)(r * nq / C), qb = (int)((r + 1) * nq / C);
  for (int e = threadIdx.x; e < M * nq; e += blockDim.x) {
    const int t = e / nq, qd = e % nq, n = qd * 4;
    float4 s = *(const float4*)&part[t * RS + n];
#pragma unroll
    for (int w = 1; w < WK; ++w) {
      const float4 v = *(const float4*)&part[(w * TOKP + t) * RS + n];
      s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
    if (C == 1) {
      uint2 o; o.x = pack_bf2(s.x, s.y); o.y = pack_bf2(s.z, s.w);
      *(uint2*)(p.out + (size_t)t * p.N + (size_t)r0 + n) = o;
    } else if (qd < qa || qd >= qb) {
      u32 owner = 0;
#pragma unroll
      for (int c2 = 1; c2 < C; ++c2) if (qd >= (c2 * nq) / C) owner = (u32)c2;
      const int sl = (int)r < (int)owner ? (int)r : (int)r - 1;   // one receive slot per source rank
      st_async4(smem_u32(&recv[(sl * TOKP + t) * RS + n]), bar0 + 1016, owner, s);
    } else {
      *(float4*)&red2[t * RS + n] = s;
    }
  }
  if (C > 1) {
    mbar_wait(bar0 + 1016, 0);                    // the other ranks' partials for my quads
    __syncthreads();                              // red2 from all threads
    if (threadIdx.x == 0) TR(4);
    const int nmy = qb - qa;
    for (int e = threadIdx.x; e < M * nmy; e += blockDim.x) {
      const int t = e / nmy, n = (qa + e % nmy) * 4;
      float4 s = *(const float4*)&red2[t * RS + n];
#pragma unroll
      for (int sl = 0; sl < C - 1; ++sl) {
        const float4 v = *(const float4*)&recv[(sl * TOKP + t) * RS + n];
        s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
      }
      uint2 o; o.x = pack_bf2(s.x, s.y); o.y = pack_bf2(s.z, s.w);
      *(uint2*)(p.out + (size_t)t * p.N + (size_t)r0 + n) = o;
    }
  } else if (threadIdx.x == 0) TR(4);
  if (threadIdx.x == 0) TR(5);
}
"""

_b7__k1t_LAYER_K = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "pooler1": 8192, "unpooler": 4096, "unpslice": 4096,
           "qkv32": 8192, "pooler32": 8192, "unpooler32": 4096, "o32": 1024}
# layer -> (C, Q, GW, WR, WK, D)
_b7__k1t_CFG = {
    "qkv_gate": (2, 66, 5, 1, 16, 3),
    "pooler": (2, 66, 4, 2, 8, 4),
    "pooler1": (1, 132, 4, 1, 16, 4),
    "o": (1, 132, 4, 2, 8, 4),
    "unpooler": (1, 132, 4, 2, 8, 4),
    "unpslice": (1, 128, 1, 1, 16, 4),
    "qkv32": (2, 66, 5, 1, 16, 3),
    "pooler32": (2, 66, 4, 2, 8, 3),
    "unpooler32": (1, 132, 4, 2, 8, 3),
    "o32": (1, 132, 4, 2, 8, 3),
}
_b7__k1t_VARIANT = {("pooler", 8): "pooler1", ("qkv_gate", 32): "qkv32", ("pooler", 32): "pooler32",
           ("unpooler", 32): "unpooler32", ("o", 32): "o32"}   # (layer, tokb) -> config key
_b7__k1t_OVMAX = 8192
_b7__k1t_TOKBS = (8, 16, 32)
_b7__k1t_SMEM_MAX = 227 * 1024


def _b7__k1t_kname(layer, tokb, comp):
    return f"k1t_{layer}_{tokb}_{comp}"


def _b7__k1t__dims(layer, tokb, comp):
    c, q, gw, wr, wk, d = _b7__k1t_CFG[layer]
    kc = _b7__k1t_LAYER_K[layer] // c
    nkt = kc // 32 // wk
    tokp = (tokb + 15) // 16 * 16
    chb = tokb * wk * 64
    epi = (wk + 1 + max(c - 1, 1)) * tokp * (gw * wr * 8 + 8) * 4
    ovm = _b7__k1t_OVMAX if comp else 0
    r = nkt
    while r > 2 and 1024 + max(r * chb + 1024, epi) + ovm > _b7__k1t_SMEM_MAX:
        r -= 1
    ovoff = 1024 + (max(r * chb + 1024, epi) + 1023) // 1024 * 1024
    rcvoff = ovoff + ovm
    rcv = (c - 1) * tokp * (gw * wr * 8 + 8) * 4 if c > 1 else 0
    smem = max(rcvoff + rcv, 120 * 1024)  # one CTA per SM
    if smem > _b7__k1t_SMEM_MAX:
        smem = 10**9
    return dict(ovoff=ovoff, smem=smem, nkt=nkt, r=r, rcvoff=rcvoff)


def _b7__k1t__needed():
    out = set()
    for layer in ("qkv_gate", "o", "pooler", "unpooler", "unpslice"):
        for tokb in _b7__k1t_TOKBS:
            if layer == "unpslice" and tokb > 8:
                continue
            out.add((_b7__k1t_VARIANT.get((layer, tokb), layer), tokb))
    return out


def _b7__k1t_source(layers=None, tokbs=None, comps=(1,)):
    s = _b7__k1t_SRC
    for key, tokb in sorted(_b7__k1t__needed()):
        if layers and key not in layers:
            continue
        if tokbs and tokb not in tokbs:
            continue
        c, q, gw, wr, wk, d = _b7__k1t_CFG[key]
        kt = _b7__k1t_LAYER_K[key] // 32
        for comp in comps:
            dm = _b7__k1t__dims(key, tokb, comp)
            if dm["smem"] > _b7__k1t_SMEM_MAX:
                continue
            pw = 0
            nt = 32 * (wr * wk + pw)
            cl = f"__cluster_dims__({c},1,1) " if c > 1 else ""
            s += (f'extern "C" __global__ void {cl}__launch_bounds__({nt},1) {_b7__k1t_kname(key, tokb, comp)}'
                  f'(const __grid_constant__ P p) {{ k1t_body<{c},{tokb},{gw},{wr},{wk},{dm["nkt"]},{d},{kt},{comp},{dm["r"]},{pw},{1 if key in _b7__k1t_HALF_CFGS else 0}>(p); }}\n')
    return s


def _b7__k1t__frag(t, n, k):
    return t.contiguous().view(n // 8, 8, k // 32, 2, 2, 4, 2).permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 32, 32, 8)


def _b7__k1t_compress(w):
    u = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e = u >> 15, (u >> 7) & 0xFF
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    best, H = -1, 0
    for h in range(0, 121):
        inside = int(cs[2 * h + 16] - cs[2 * h])
        if inside > best:
            best, H = inside, h
    idx = (e >> 1) - H
    esc = (idx < 0) | (idx > 7)
    code = torch.where(esc, torch.zeros_like(e), (s << 3) | idx)
    lo = torch.where(esc, torch.zeros_like(u), u & 0xFF)
    return code, lo, H, esc


_b7__k1t_HALF_CFGS = {"qkv_gate", "qkv32"}


def _b7__k1t_prepare_layer(w, layer, comp):
    c, q, gw, wr, wk, d = _b7__k1t_CFG[layer]
    half = layer in _b7__k1t_HALF_CFGS
    unit = 4 if half else 8
    n, k = w.shape
    GU = n // unit
    KT = k // 32
    kc = k // c
    nkt = kc // 32 // wk
    rg = wr * gw
    dev = w.device
    if comp:
        code, lo, H, esc = _b7__k1t_compress(w)
    parts, ovs = [], []
    ovstride = 0
    hdr_all = torch.zeros((q * c, wr * wk, 8), dtype=torch.int64)
    n_raw, n_tiles = 0, 0

    def take(t, r0, nr):
        sub = t[r0:r0 + nr]
        if nr < rg * 8:
            sub = torch.cat([sub, torch.zeros((rg * 8 - nr,) + tuple(sub.shape[1:]), dtype=sub.dtype, device=sub.device)], 0)
        return sub.contiguous()

    for qq in range(q):
        u0, u1 = qq * GU // q, (qq + 1) * GU // q
        r0, nr = u0 * unit, (u1 - u0) * unit
        assert nr <= rg * 8, (layer, nr, rg)
        if comp:
            cl = _b7__k1t__frag(take(lo, r0, nr), rg * 8, k).to(torch.uint8)                  # [rg, KT, 32, 8]
            cc = _b7__k1t__frag(take(code, r0, nr), rg * 8, k).to(torch.int32)
            cw = (cc[..., 0:4] | (cc[..., 4:8] << 4)).to(torch.uint8)               # [rg, KT, 32, 4]
            rawt = _b7__k1t__frag(take(esc.to(torch.int32), r0, nr), rg * 8, k).reshape(rg, KT, 256).any(-1)
            rb = _b7__k1t__frag(take(w, r0, nr), rg * 8, k).contiguous().view(torch.uint8).view(rg, KT, 32, 16)
            cl = torch.where(rawt[..., None, None], rb[..., 0:8], cl)
            cw = torch.where(rawt[..., None, None], rb[..., 8:12], cw)
            ov = rb[..., 12:16]
            n_raw += int(rawt.sum()); n_tiles += rawt.numel()
            TL = cl.view(wr, gw, c, nkt, wk, 32, 8).permute(2, 0, 4, 3, 1, 5, 6)   # [c,wr,wk,nkt,gw,32,8]
            TC = cw.view(wr, gw, c, nkt, wk, 32, 4).permute(2, 0, 4, 3, 1, 5, 6)
            if half:
                blk = torch.cat([TL.reshape(c, wr, wk, nkt, -1), TC.reshape(c, wr, wk, nkt, -1)], -1)
            else:
                subs = []
                for pp in range(gw // 2):
                    subs.append(torch.cat([TL[..., 2 * pp, :, :], TL[..., 2 * pp + 1, :, :]], -1))
                if gw & 1:
                    subs.append(TL[..., gw - 1, :, :])
                for qd in range(gw // 4):
                    subs.append(torch.cat([TC[..., 4 * qd + i, :, :] for i in range(4)], -1))
                j0 = gw & ~3
                if (gw & 3) >= 2:
                    subs.append(torch.cat([TC[..., j0, :, :], TC[..., j0 + 1, :, :]], -1))
                if gw & 1:
                    subs.append(TC[..., gw - 1, :, :])
                blk = torch.cat([t_.reshape(c, wr, wk, nkt, -1) for t_ in subs], -1)
            R = rawt.to(torch.uint8).view(wr, gw, c, nkt, wk).permute(2, 0, 4, 3, 1).reshape(c, wr * wk, nkt * gw).cpu()
            O = ov.view(wr, gw, c, nkt, wk, 32, 4).permute(2, 0, 4, 3, 1, 5, 6).reshape(c, wr * wk, nkt * gw, 128)
            for rr in range(c):
                cta = qq * c + rr
                obytes = []
                base = 0
                for wi in range(wr * wk):
                    bits = R[rr, wi]
                    assert nkt * gw <= 128
                    m = 0
                    idx = bits.nonzero().flatten().tolist()
                    for ti in idx:
                        m |= 1 << ti
                    if idx:
                        obytes.append(O[rr, wi, idx].reshape(-1))
                    for wq in range(4):
                        hdr_all[cta, wi, wq] = (m >> (32 * wq)) & 0xFFFFFFFF
                    hdr_all[cta, wi, 4] = base
                    base += len(idx)
                ovs.append(torch.cat(obytes) if obytes else torch.zeros(0, dtype=torch.uint8, device=dev))
                ovstride = max(ovstride, base * 128)
        else:
            B = _b7__k1t__frag(take(w, r0, nr), rg * 8, k).contiguous().view(torch.uint8).view(rg, KT, 32, 16)
            blk = B.view(wr, gw, c, nkt, wk, 32, 16).permute(2, 0, 4, 3, 1, 5, 6).reshape(c, wr, wk, nkt, 32 * gw * 16)
        parts.append(blk.reshape(-1))
    if comp:
        ovstride = (ovstride + 127) // 128 * 128
        if ovstride > _b7__k1t_OVMAX:
            raise RuntimeError(f"{layer}: overflow {ovstride} > OVMAX")
        ovf_all = torch.zeros((q * c, max(ovstride, 128)), dtype=torch.uint8, device=dev)
        for i, ob in enumerate(ovs):
            if ob.numel():
                ovf_all[i, :ob.numel()] = ob
        hdr = hdr_all.to(torch.int32).view(-1).to(dev)
        return {"w": torch.cat(parts), "ovf": ovf_all.view(-1), "hdr": hdr, "ovstride": ovstride,
                "base2": H * 0x01010101, "n_raw": n_raw, "raw_frac": n_raw / max(n_tiles, 1), "base": 2 * H, "GU": GU}
    z = torch.zeros(128, dtype=torch.uint8, device=dev)
    return {"w": torch.cat(parts), "ovf": z, "hdr": torch.zeros(8 * q * c * wr * wk, dtype=torch.int32, device=dev),
            "ovstride": 0, "base2": 0, "n_raw": 0, "base": None, "GU": GU}


_b7__k1t__TM_CACHE = {}


def _b7__k1t_x_tmap3(x, m, k, tokb, nb):
    key = (x.data_ptr(), m, k, tokb, nb)
    if key in _b7__k1t__TM_CACHE:
        return _b7__k1t__TM_CACHE[key]
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(3), x.data_ptr(),
        [d.cuuint64_t(64), d.cuuint64_t(m), d.cuuint64_t(k // 64)], [d.cuuint64_t(k * 2), d.cuuint64_t(128)],
        [d.cuuint32_t(64), d.cuuint32_t(tokb), d.cuuint32_t(nb)], [d.cuuint32_t(1)] * 3,
        d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    if int(err) != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed: {err}")
    b = b"".join(struct.pack("<Q", int(v)) for v in tm.opaque)
    if len(_b7__k1t__TM_CACHE) > 64:
        _b7__k1t__TM_CACHE.clear()
    _b7__k1t__TM_CACHE[key] = b
    return b


_b7__k1t_TRACE = None


def _b7__k1t_cfg_key(layer, m, n):
    lname = "unpslice" if (layer == "unpooler" and n == 1024) else layer
    tokb = 8 if m <= 8 else (16 if m <= 16 else 32)
    return _b7__k1t_VARIANT.get((lname, tokb), lname), tokb


def _b7__k1t_launch(mod, layer, ent, x, out, m, n, comp):
    lname, tokb = _b7__k1t_cfg_key(layer, m, n)
    c, q, gw, wr, wk, d = _b7__k1t_CFG[lname]
    k = _b7__k1t_LAYER_K[lname]
    kc = k // c
    dm = _b7__k1t__dims(lname, tokb, comp)
    tm = _b7__k1t_x_tmap3(x, m, k, tokb, wk // 2)
    args = tm + struct.pack("<QQQQqqqqIIIIQ", ent["w"].data_ptr(), ent["ovf"].data_ptr(), ent["hdr"].data_ptr(),
                            out.data_ptr(), m, n, ent["GU"], q, ent["base2"], ent["ovstride"], dm["ovoff"], dm["rcvoff"],
                            _b7__k1t_TRACE.data_ptr() if _b7__k1t_TRACE is not None else 0)
    args += b"\0" * (256 - len(args))
    pw = 0
    mod.launch(_b7__k1t_kname(lname, tokb, comp), q * c, 32 * (wr * wk + pw), dm["smem"], args)


# ======================================================================== component: router
"""Router projection: out[M,256] fp32 = x[M,8192] bf16 @ W[256,8192]^T, W fp32 run as a BF16 hi + lo pair, FP32 accumulation.

One kernel per call. CTA = (token tile tt, row tile rt, k split ks). Weights (hi, lo tiles in mma B-fragment order) stream with
cp.async.bulk, x tiles with a 2D TMA (128B swizzle, rows >= M zero-filled), mma.sync m16n8k16 (x = A, W rows = n8 B).
Split-K partials: red.global.add.v4.f32 into a self-resetting fp32 workspace; the last CTA of an output tile (counter)
writes the output and zeroes the workspace + counter for the next call.
"""
import ctypes
import struct

import torch

_b7__rt_SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;

struct __align__(64) TMap { u64 v[16]; };
__device__ __forceinline__ unsigned long long gtime0() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define TRT(i) do { if (p.trc && threadIdx.x == 0) p.trc[(blockIdx.x) * 8 + (i)] = gtime0(); } while (0)
struct __align__(64) P {
  TMap tmx;
  const unsigned char* w;
  float* out;
  float* ws;
  u32* cnt;
  long long M;
  long long NTT;
  const u16* x;            // compressed path: zero CTAs read x for escape corrections
  const int* erow;         // [257] escape row offsets
  const int* ek;           // escape k
  const float* ed;         // escape delta (true hi+lo minus decoded hi+lo)
  long long base2;         // (base << 7) | (base << 23)
  unsigned long long* trc;
};

__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u32 b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect_arrive(u32 b, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(u32 b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void mbar_wait(u32 b, int ph) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}\n"
               :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma2d(u32 dst, const TMap* tm, int c0, int c1, u32 bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma3d(u32 dst, const TMap* tm, int c0, int c1, int c2, u32 bar) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
               :: "r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(c2), "r"(bar) : "memory");
}
__device__ __forceinline__ void ldsm_x4(u32* a, u32 addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr));
}
__device__ __forceinline__ uint4 lds128(u32 addr) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr)); return v;
}
__device__ __forceinline__ void mma16816(float* c, const u32* a, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void red_add_v4(float* p, float4 v) {
  asm volatile("red.global.add.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
__device__ __forceinline__ float4 ldcg4(const float* p) {
  float4 v; asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p) : "memory"); return v;
}

// TT tokens per tile (MT = TT/16), RR rows per tile (G = RR/8 groups), KK k per CTA, KS k per stage, NS ring slots,
// WK x WG consumer warps (k32 tiles x groups). Grid = NTT * (256/RR) * (8192/KK).

// ---- in-kernel output zeroing: Z leading CTAs zero `out` and bump zf (release); adder CTA b waits (acquire) until
//      zf has advanced by Z since its own private generation pc[b], red.adds, then pc[b] += Z. Single kernel, no grid sync.
#define ZCTAS 4
__device__ __forceinline__ u32 ld_acquire_u32(const u32* p) {
  u32 v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void zero_cta(float* out, long long n, u32* zf, int z) {
  float4* o = (float4*)out;
  const long long n4 = n / 4;
  for (long long i = z * (long long)blockDim.x + threadIdx.x; i < n4; i += (long long)ZCTAS * blockDim.x)
    o[i] = make_float4(0.f, 0.f, 0.f, 0.f);
  __syncthreads();
  if (threadIdx.x == 0) { __threadfence(); asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(zf) : "memory"); }
}
__device__ __forceinline__ void zero_cta_esc(const P& p, int z) {
  const int n = (int)p.M * 256;
  for (int e = z * blockDim.x + threadIdx.x; e < n; e += ZCTAS * blockDim.x) {
    const int t = e >> 8, r = e & 255;
    float v = 0.f;
    for (int i = p.erow[r]; i < p.erow[r + 1]; ++i)
      v = fmaf(__uint_as_float((u32)p.x[(size_t)t * 8192 + p.ek[i]] << 16), p.ed[i], v);
    p.out[e] = v;
  }
  __syncthreads();
  if (threadIdx.x == 0) { __threadfence(); asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(p.cnt) : "memory"); }
}
__device__ __forceinline__ u32 prmt(u32 a, u32 sel) { u32 r; asm("prmt.b32 %0, %1, %1, %2;" : "=r"(r) : "r"(a), "r"(sel)); return r; }
// 8 compressed weights (byte plane bytes b, nibble word nw) -> 4 bf16x2 regs; hi uses base2, lo is relative to hi
__device__ __forceinline__ void dec_hi(uint2 b, u32 nw, u32 base2, u32* o) {
  const u32 e0 = (nw << 7) & 0x07800780u, e1 = (nw << 3) & 0x07800780u;
  const u32 e2 = (nw >> 1) & 0x07800780u, e3 = (nw >> 5) & 0x07800780u;
  o[0] = ((prmt(b.x, 0x1100) & 0x807F807Fu) | e0) + base2;
  o[1] = ((prmt(b.x, 0x3322) & 0x807F807Fu) | e1) + base2;
  o[2] = ((prmt(b.y, 0x1100) & 0x807F807Fu) | e2) + base2;
  o[3] = ((prmt(b.y, 0x3322) & 0x807F807Fu) | e3) + base2;
}
__device__ __forceinline__ void dec_lo(uint2 b, u32 nw, const u32* h, u32* o) {
  constexpr u32 E8 = (8u << 7) | (8u << 23);
  const u32 e0 = ((nw << 7) & 0x07800780u) + E8, e1 = ((nw << 3) & 0x07800780u) + E8;
  const u32 e2 = ((nw >> 1) & 0x07800780u) + E8, e3 = ((nw >> 5) & 0x07800780u) + E8;
  o[0] = (prmt(b.x, 0x1100) & 0x807F807Fu) | ((h[0] & 0x7F807F80u) - e0);
  o[1] = (prmt(b.x, 0x3322) & 0x807F807Fu) | ((h[1] & 0x7F807F80u) - e1);
  o[2] = (prmt(b.y, 0x1100) & 0x807F807Fu) | ((h[2] & 0x7F807F80u) - e2);
  o[3] = (prmt(b.y, 0x3322) & 0x807F807Fu) | ((h[3] & 0x7F807F80u) - e3);
}
__device__ __forceinline__ uint2 lds64(u32 addr) {
  uint2 v; asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(addr)); return v;
}
__device__ __forceinline__ void wait_zero(const u32* zf, const u32* pcb) {
  const u32 target = *(volatile const u32*)pcb + ZCTAS;
  while ((int)(ld_acquire_u32(zf) - target) < 0) {}
}
template <int TT, int RR, int KK, int KS, int NS, int WK, int WG, int FL>
__device__ __forceinline__ void rt_body(const P& p) {
  constexpr int MT = TT / 16, G = RR / 8, GW = G / WG;
  constexpr int NR = 256 / RR, S = 8192 / KK, NSTG = KK / KS;
  constexpr int NCW = WK * WG;
  constexpr bool CMP = (FL & 32) != 0;
  constexpr int TB = CMP ? 768 : 1024;
  constexpr int XB = TT * KS * 2, WB = KS * RR * (CMP ? 3 : 4);
  constexpr int SB = ((XB + WB) + 1023) / 1024 * 1024;
  constexpr int KT = KS / 32;          // k32 tiles per stage
  constexpr int KTW = KT / WK;         // per warp per stage
  constexpr int RSTR = RR + 4;         // scratch row stride (floats)
  static_assert(KT % WK == 0 && G % WG == 0, "split");
  static_assert(NCW * TT * RSTR * 4 / WG <= NS * SB, "scratch fits");
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  unsigned char* sm = (unsigned char*)(((size_t)sm_raw + 1023) & ~(size_t)1023);
  const u32 bars = smem_u32(sm);                 // full[NS] at 0, empty[NS] at 8*NS
  const u32 ring = bars + 1024;
  float* scratch = (float*)(sm + 1024);
  __shared__ int s_last;

  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if ((FL & 16) && blockIdx.x < ZCTAS) { zero_cta(p.out, p.M * 256, p.cnt, blockIdx.x); return; }
  const int b = (FL & 16) ? blockIdx.x - ZCTAS : blockIdx.x;
  const int ks = b % S, rt = (b / S) % NR, tt = b / (S * NR);
  const int M = (int)p.M;
  const unsigned char* wsrc = p.w + (size_t)(rt * S + ks) * (RR * KK * (CMP ? 3 : 4));
  const u32 base2 = (u32)p.base2;
  const u32 zgen = (FL & 256) ? *(volatile const u32*)(p.cnt + 64 + b) : 0u;
  __shared__ int esc_e[CMP ? 32 : 1];
  __shared__ float esc_d[CMP ? 32 : 1];
  __shared__ float corr[CMP ? TT * RR : 1];
  __shared__ int esc_n;
  if (CMP)
    for (int i = threadIdx.x; i < TT * RR; i += blockDim.x) corr[i] = 0.f;

  if (wid == NCW && lane == 0) {
    for (int s = 0; s < NS; ++s) { mbar_init(bars + 8 * s, 1); mbar_init(bars + 8 * (NS + s), NCW); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();  // barrier init visible
  if (wid == NCW) {
    if (lane == 0) {
      if (FL & 128) { const unsigned long long t0 = gtime0(); while (gtime0() - t0 < (unsigned long long)p.base2) {} }
#pragma unroll 1
      for (int s = 0; s < NSTG; ++s) {
        const int slot = s % NS;
        if (s >= NS) mbar_wait(bars + 8 * (NS + slot), ((s / NS) - 1) & 1);
        const u32 st = ring + slot * SB;
        mbar_expect_arrive(bars + 8 * slot, XB + WB);
        bulk_g2s(st + XB, wsrc + (size_t)s * WB, WB, bars + 8 * slot);
        if (FL & 64) {
          tma3d(st, &p.tmx, 0, tt * TT, (ks * KK + s * KS) >> 6, bars + 8 * slot);
        } else {
#pragma unroll 1
          for (int bx = 0; bx < KS / 64; ++bx)
            tma2d(st + bx * TT * 128, &p.tmx, ks * KK + s * KS + bx * 64, tt * TT, bars + 8 * slot);
        }
      }
    }
    __syncwarp();
  }

  float acc[GW][MT][4];
#pragma unroll
  for (int g = 0; g < GW; ++g)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[g][mt][c] = 0.f;
  const int wk = wid % WK, wg = wid / WK;
  TRT(0);
  if (CMP && wid == 0) {   // this CTA's escapes; issued after the producer started, used after the main loop
    const int blk = rt * S + ks;
    const int o0 = p.erow[blk], n = min(p.erow[blk + 1] - o0, 32);
    if (lane < n) { esc_e[lane] = p.ek[o0 + lane]; esc_d[lane] = p.ed[o0 + lane]; }
    if (lane == 0) esc_n = n;
  }
  if (wid < NCW) {
    const int rowa = (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll 1
    for (int s = 0; s < NSTG; ++s) {
      const int slot = s % NS;
      mbar_wait(bars + 8 * slot, (s / NS) & 1);
      if (s == 0) TRT(1);
      if (s == NSTG - 1) TRT(2);
      const u32 st = ring + slot * SB;
#pragma unroll
      for (int i = 0; i < KTW; ++i) {
        const int kt = wk * KTW + i;            // k32 tile within stage
        u32 a[MT][2][4];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int kl = kt * 32 + h * 16;
          const u32 xb = st + (kl >> 6) * (TT * 128);
          const int c = ((kl & 63) >> 3) + (lane >> 4);
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            const int row = mt * 16 + rowa;
            ldsm_x4(a[mt][h], xb + row * 128 + ((c ^ (row & 7)) << 4));
          }
        }
        const u32 tb = st + XB + (u32)((kt * G + wg * GW) * TB);
#pragma unroll
        for (int g = 0; g < GW; ++g) {
          uint4 bh, bl;
          if (CMP) {
            const uint4 bb = lds128(tb + g * TB + lane * 16);
            const uint2 nn = lds64(tb + g * TB + 512 + lane * 8);
            u32 h[4], l[4];
            dec_hi(make_uint2(bb.x, bb.y), nn.x, base2, h);
            dec_lo(make_uint2(bb.z, bb.w), nn.y, h, l);
            bh = make_uint4(h[0], h[1], h[2], h[3]);
            bl = make_uint4(l[0], l[1], l[2], l[3]);
          } else {
            bh = lds128(tb + g * 1024 + lane * 16);
            bl = lds128(tb + g * 1024 + 512 + lane * 16);
          }
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            if (FL & 2) { acc[g][mt][0] += __uint_as_float(bh.x ^ bl.y ^ a[mt][0][0]); continue; }
            mma16816(acc[g][mt], a[mt][0], bh.x, bh.y);
            mma16816(acc[g][mt], a[mt][1], bh.z, bh.w);
            mma16816(acc[g][mt], a[mt][0], bl.x, bl.y);
            mma16816(acc[g][mt], a[mt][1], bl.z, bl.w);
          }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(bars + 8 * (NS + slot));
    }
  }
  if (FL & 256) {     // per-warp epilogue: wait for the zero CTAs, red.add fragments directly (no smem reduction)
    if (wid < NCW) {
      const u32 target = zgen + ZCTAS;
      while ((int)(ld_acquire_u32(p.cnt) - target) < 0) {}
#pragma unroll
      for (int g = 0; g < GW; ++g) {
        const int n = rt * RR + (wg * GW + g) * 8 + 2 * (lane & 3);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          const int t = tt * TT + mt * 16 + (lane >> 2);
          if (t < M) asm volatile("red.global.add.v2.f32 [%0], {%1,%2};" :: "l"(&p.out[(size_t)t * 256 + n]), "f"(acc[g][mt][0]), "f"(acc[g][mt][1]) : "memory");
          if (t + 8 < M) asm volatile("red.global.add.v2.f32 [%0], {%1,%2};" :: "l"(&p.out[(size_t)(t + 8) * 256 + n]), "f"(acc[g][mt][2]), "f"(acc[g][mt][3]) : "memory");
        }
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) p.cnt[64 + b] = zgen + ZCTAS;
    return;
  }
  __syncthreads();  // all stages landed and consumed
  TRT(3);
  if (CMP) {          // escape corrections from the x tiles still in the ring: corr[t][r] += x[t][k] * delta
    const int tmx = min(TT, M - tt * TT);
    for (int i = threadIdx.x; i < esc_n * tmx; i += blockDim.x) {
      const int e = i / tmx, t = i % tmx;
      const int rl = esc_e[e] >> 16, kl = esc_e[e] & 0xFFFF;
      const int s_ = kl / KS, kin = kl % KS, bx = kin >> 6, ki = kin & 63;
      const unsigned char* xp = sm + 1024 + (s_ % NS) * SB + bx * (TT * 128) + t * 128 + ((((ki >> 3) ^ (t & 7)) << 4) | ((ki & 7) * 2));
      const float xv = __uint_as_float((u32)(*(const u16*)xp) << 16);
      atomicAdd(&corr[t * RR + rl], xv * esc_d[e]);
    }
    __syncthreads();
  }
  if (wid < NCW) {
    float* sc = scratch + wk * (TT * RSTR);
#pragma unroll
    for (int g = 0; g < GW; ++g) {
      const int n = (wg * GW + g) * 8 + 2 * (lane & 3);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int t = mt * 16 + (lane >> 2);
        *(float2*)&sc[t * RSTR + n] = make_float2(acc[g][mt][0], acc[g][mt][1]);
        *(float2*)&sc[(t + 8) * RSTR + n] = make_float2(acc[g][mt][2], acc[g][mt][3]);
      }
    }
  }
  __syncthreads();
  const int tile = tt * NR + rt;
  float* wst = p.ws + (size_t)tile * (TT * RR);
  const int tmax = min(TT, M - tt * TT);
  if (FL & 28) {
    if (FL & 4) asm volatile("griddepcontrol.wait;" ::: "memory");
    TRT(4);
    if (FL & 16) wait_zero(p.cnt, p.cnt + 64 + b);
    TRT(5);
    for (int e = threadIdx.x; e < tmax * (RR / 4); e += blockDim.x) {
      const int t = e / (RR / 4), n = (e % (RR / 4)) * 4;
      float4 v = *(float4*)&scratch[t * RSTR + n];
#pragma unroll
      for (int w = 1; w < WK; ++w) {
        const float4 u = *(float4*)&scratch[w * (TT * RSTR) + t * RSTR + n];
        v.x += u.x; v.y += u.y; v.z += u.z; v.w += u.w;
      }
      if (CMP) { v.x += corr[t * RR + n]; v.y += corr[t * RR + n + 1]; v.z += corr[t * RR + n + 2]; v.w += corr[t * RR + n + 3]; }
      red_add_v4(&p.out[(size_t)(tt * TT + t) * 256 + rt * RR + n], v);
    }
    if (FL & 16) { __syncthreads(); if (threadIdx.x == 0) p.cnt[64 + b] += ZCTAS; }
    TRT(6);
    return;
  }
  for (int e = threadIdx.x; e < tmax * (RR / 4); e += blockDim.x) {
    const int t = e / (RR / 4), n = (e % (RR / 4)) * 4;
    float4 v = *(float4*)&scratch[t * RSTR + n];
#pragma unroll
    for (int w = 1; w < WK; ++w) {
      const float4 u = *(float4*)&scratch[w * (TT * RSTR) + t * RSTR + n];
      v.x += u.x; v.y += u.y; v.z += u.z; v.w += u.w;
    }
    if (S == 1 || (FL & 1)) *(float4*)&p.out[(size_t)(tt * TT + t) * 256 + rt * RR + n] = v;
    else red_add_v4(wst + t * RR + n, v);
  }
  if (S == 1 || (FL & 1)) return;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) s_last = (atomicAdd(&p.cnt[tile], 1u) == (u32)(S - 1));
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  for (int e = threadIdx.x; e < tmax * (RR / 4); e += blockDim.x) {
    const int t = e / (RR / 4), n = (e % (RR / 4)) * 4;
    const float4 v = ldcg4(wst + t * RR + n);
    *(float4*)&p.out[(size_t)(tt * TT + t) * 256 + rt * RR + n] = v;
    *(float4*)(wst + t * RR + n) = make_float4(0.f, 0.f, 0.f, 0.f);
  }
  if (threadIdx.x == 0) p.cnt[tile] = 0;
}

struct __align__(64) PS { TMap tmx; const float* w; float* out; long long M; };
// w prelaid [128 CTAs][NCH][2 rows][CK] fp32; x chunk c = TMA box [256, CK/256, M] -> smem [M][CK] bf16
template <int M, int NCH>
__device__ __forceinline__ void rs_body(const PS& p) {
  constexpr int CK = 8192 / NCH;
  constexpr int WCB = 2 * CK * 4, XCB = M * CK * 2;
  constexpr int HALF = CK / 2;                 // thread i: k = i*4 + j*HALF... (256 threads, 4 k each per half)
  static_assert(CK == 2048, "256 threads x 8 k");
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  unsigned char* sm = (unsigned char*)(((size_t)sm_raw + 1023) & ~(size_t)1023);
  const u32 bars = smem_u32(sm);
  const u32 wsm = bars + 1024;
  const u32 xsm = wsm + NCH * WCB;
  __shared__ float red[8][2 * M];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  if (tid == 0) {
    for (int c = 0; c < NCH; ++c) mbar_init(bars + 8 * c, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    const float* wsrc = p.w + (size_t)blockIdx.x * (2 * 8192);
#pragma unroll 1
    for (int c = 0; c < NCH; ++c) {
      mbar_expect_arrive(bars + 8 * c, WCB + XCB);
      bulk_g2s(wsm + c * WCB, wsrc + (size_t)c * 2 * CK, WCB, bars + 8 * c);
      tma3d(xsm + c * XCB, &p.tmx, 0, c * (CK / 256), 0, bars + 8 * c);
    }
  }
  __syncthreads();
  float a0[M], a1[M];
#pragma unroll
  for (int t = 0; t < M; ++t) { a0[t] = 0.f; a1[t] = 0.f; }
#pragma unroll 1
  for (int c = 0; c < NCH; ++c) {
    mbar_wait(bars + 8 * c, 0);
    const unsigned char* wc = sm + 1024 + c * WCB;
    const unsigned char* xc = sm + 1024 + NCH * WCB + c * XCB;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int k = j * HALF + tid * 4;
      const float4 w0 = *(const float4*)(wc + k * 4);
      const float4 w1 = *(const float4*)(wc + (CK + k) * 4);
#pragma unroll
      for (int t = 0; t < M; ++t) {
        const uint2 xv = *(const uint2*)(xc + (t * CK + k) * 2);
        const float x0 = __uint_as_float(xv.x << 16), x1 = __uint_as_float(xv.x & 0xffff0000u);
        const float x2 = __uint_as_float(xv.y << 16), x3 = __uint_as_float(xv.y & 0xffff0000u);
        a0[t] = fmaf(x0, w0.x, a0[t]); a0[t] = fmaf(x1, w0.y, a0[t]); a0[t] = fmaf(x2, w0.z, a0[t]); a0[t] = fmaf(x3, w0.w, a0[t]);
        a1[t] = fmaf(x0, w1.x, a1[t]); a1[t] = fmaf(x1, w1.y, a1[t]); a1[t] = fmaf(x2, w1.z, a1[t]); a1[t] = fmaf(x3, w1.w, a1[t]);
      }
    }
  }
#pragma unroll
  for (int t = 0; t < M; ++t) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      a0[t] += __shfl_xor_sync(0xffffffffu, a0[t], o);
      a1[t] += __shfl_xor_sync(0xffffffffu, a1[t], o);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < M; ++t) { red[wid][2 * t] = a0[t]; red[wid][2 * t + 1] = a1[t]; }
  }
  __syncthreads();
  if (tid < 2 * M) {
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) v += red[w][tid];
    p.out[(size_t)(tid >> 1) * 256 + blockIdx.x * 2 + (tid & 1)] = v;
  }
}

// ---- rc: cluster of 8 along K (1024 k each), 15 clusters own 17/18 rows each; DSMEM reduction; M <= 64 ----
struct __align__(64) PC { TMap tmx; const unsigned char* w; float* out; long long M; unsigned long long* trc; };
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define TR(i) do { if (p.trc && threadIdx.x == 0) p.trc[blockIdx.x * 8 + (i)] = gtime(); } while (0)
__device__ __forceinline__ void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ u32 cluster_rank() { u32 r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ float ld_dsmem(u32 addr, u32 rank) {
  u32 ra; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(addr), "r"(rank));
  float v; asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(ra) : "memory"); return v;
}
template <int MT>
__device__ __forceinline__ void rc_body(const PC& p) {
  constexpr int TT = MT * 16, Q = 15, NSTG = 4, KS = 256;
  constexpr int XB = TT * KS * 2, WBM = 18 * KS * 4;
  constexpr int SB = (XB + WBM + 1023) / 1024 * 1024;
  constexpr int RS = 28;
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  unsigned char* sm = (unsigned char*)(((size_t)sm_raw + 1023) & ~(size_t)1023);
  const u32 bars = smem_u32(sm);
  const u32 ring = bars + 1024;
  float* scratch = (float*)(sm + 1024);
  float* part = scratch + 8 * TT * RS;
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const u32 rank = cluster_rank();
  const int q = blockIdx.x >> 3;
  const int r0 = q * 256 / Q, nr = (q + 1) * 256 / Q - r0;
  const int M = (int)p.M;
  const u32 wbytes = (u32)(nr * KS * 4);
  const unsigned char* wsrc = p.w + ((size_t)r0 * 8192 + (size_t)rank * nr * 1024) * 4;
  TR(0);
  if (wid == 8 && lane == 0) {
    for (int s = 0; s < NSTG; ++s) mbar_init(bars + 8 * s, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#pragma unroll 1
    for (int s = 0; s < NSTG; ++s) {
      mbar_expect_arrive(bars + 8 * s, XB + wbytes);
      bulk_g2s(ring + s * SB + XB, wsrc + (size_t)s * wbytes, wbytes, bars + 8 * s);
      tma3d(ring + s * SB, &p.tmx, 0, 0, (int)rank * 16 + s * 4, bars + 8 * s);
    }
  }
  __syncthreads();
  float acc[3][MT][4];
#pragma unroll
  for (int g = 0; g < 3; ++g)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[g][mt][c] = 0.f;
  if (wid < 8) {
    const int rowa = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int ng = (nr + 7) >> 3;
#pragma unroll 1
    for (int s = 0; s < NSTG; ++s) {
      mbar_wait(bars + 8 * s, 0);
      if (s == 0) TR(1);
      if (s == NSTG - 1) TR(2);
      const u32 st = ring + s * SB;
      u32 a[MT][2][4];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int kl = wid * 32 + h * 16;
        const u32 xb = st + (kl >> 6) * (TT * 128);
        const int c = ((kl & 63) >> 3) + (lane >> 4);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          const int t = mt * 16 + rowa;
          ldsm_x4(a[mt][h], xb + t * 128 + ((c ^ (t & 7)) << 4));
        }
      }
      const u32 tb = st + XB + (u32)(wid * nr * 128);
#pragma unroll
      for (int g = 0; g < 3; ++g) {
        if (g < ng) {
          const int rg = min(8, nr - 8 * g);
          uint4 bh = make_uint4(0, 0, 0, 0), bl = make_uint4(0, 0, 0, 0);
          if (lane < 4 * rg) { bh = lds128(tb + g * 1024 + lane * 16); bl = lds128(tb + g * 1024 + rg * 64 + lane * 16); }
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            mma16816(acc[g][mt], a[mt][0], bh.x, bh.y);
            mma16816(acc[g][mt], a[mt][1], bh.z, bh.w);
            mma16816(acc[g][mt], a[mt][0], bl.x, bl.y);
            mma16816(acc[g][mt], a[mt][1], bl.z, bl.w);
          }
        }
      }
    }
  }
  __syncthreads();
  if (wid < 8) {
    float* sc = scratch + wid * (TT * RS);
#pragma unroll
    for (int g = 0; g < 3; ++g) {
      const int n = g * 8 + 2 * (lane & 3);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int t = mt * 16 + (lane >> 2);
        *(float2*)&sc[t * RS + n] = make_float2(acc[g][mt][0], acc[g][mt][1]);
        *(float2*)&sc[(t + 8) * RS + n] = make_float2(acc[g][mt][2], acc[g][mt][3]);
      }
    }
  }
  __syncthreads();
  for (int e = threadIdx.x; e < M * nr; e += blockDim.x) {
    const int t = e / nr, n = e % nr;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) v += scratch[w * (TT * RS) + t * RS + n];
    part[t * 24 + n] = v;
  }
  TR(3);
  cluster_sync_all();
  TR(4);
  const u32 pu = smem_u32(part);
  for (int e = (int)rank + 8 * threadIdx.x; e < M * nr; e += 8 * blockDim.x) {
    const int t = e / nr, n = e % nr;
    float v[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) v[c] = ld_dsmem(pu + (u32)(t * 24 + n) * 4, (u32)c);
    p.out[(size_t)t * 256 + r0 + n] = ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));
  }
  TR(5);
  cluster_sync_all();
  TR(6);
}

struct PZ { float4* out; long long n4; };
extern "C" __global__ void rzero(const PZ p) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < p.n4; i += (long long)gridDim.x * blockDim.x)
    p.out[i] = make_float4(0.f, 0.f, 0.f, 0.f);
}

// ---- rw: wgmma, A = W rows (64 per CTA, hi & lo 128B-swizzled tiles), B = x (TN tokens via TMA), K split S;
//      partial tiles red.add'ed into out (zeroed by rzero, ordered with griddepcontrol.wait) ----
struct __align__(64) PW { TMap tmx; const unsigned char* w; float* out; long long M; long long NTT; unsigned long long* trc; u32* zc;
  const int* eoff; const int* ent; const float* edl; long long base2; };
__device__ __forceinline__ u64 mkdesc(u32 saddr) {
  return (u64)((saddr & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)(1024 >> 4) << 32) | ((u64)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void wgmma_n256(float* d, u64 da, u64 db) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, p, 1, 1, 0, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127]) : "l"(da), "l"(db));
}
__device__ __forceinline__ void wgmma_n128(float* d, u64 da, u64 db) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1, 0, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]) : "l"(da), "l"(db));
}
__device__ __forceinline__ void wgmma_n64(float* d, u64 da, u64 db) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1, 0, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) : "l"(da), "l"(db));
}
__device__ __forceinline__ void wgmma_rs_n128(float* d, const u32* a, u64 db) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
}
__device__ __forceinline__ void wgmma_rs_n64(float* d, const u32* a, u64 db) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
}

__device__ __forceinline__ void bulk_g2s_mc(u32 dst, const void* src, u32 bytes, u32 bar, unsigned short mask) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "h"(mask) : "memory");
}
__device__ __forceinline__ void mbar_arrive_remote(u32 b, u32 rank) {
  u32 ra; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(b), "r"(rank));
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(ra) : "memory");
}
template <int TN, int KK, int NS, int MC, int ZF, int CA = 0, int ZA = 0>
__device__ __forceinline__ void rw_body(const PW& p) {
  constexpr int S = 8192 / KK, NSTG = KK / 64;
  constexpr int AB = CA ? 12288 : 16384, XB = TN * 128, SB = AB + XB;
  constexpr int NACC = TN / 2;
  constexpr int TS = 68;                    // transpose row stride (floats)
  static_assert(TN * TS * 4 <= NS * SB, "transpose fits");
  __shared__ int esc_e[CA ? 32 : 1];
  __shared__ float esc_d[CA ? 32 : 1];
  __shared__ float corrv[CA ? 32 * TN : 1];
  __shared__ int esc_n;
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  unsigned char* sm = (unsigned char*)(((size_t)sm_raw + 1023) & ~(size_t)1023);
  const u32 bars = smem_u32(sm);            // full[NS] at 0, empty[NS] at 8*NS
  const u32 ring = bars + 1024;
  float* tr = (float*)(sm + 1024);
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int NTT = (int)p.NTT, M = (int)p.M;
  if (ZF && blockIdx.x < ZCTAS) { zero_cta(p.out, p.M * 256, p.zc, blockIdx.x); return; }
  const int b = ZF ? blockIdx.x - ZCTAS : blockIdx.x;
  const int ks = MC ? (b / 2) % S : b % S;
  const int tt = MC ? b % 2 : (b / S) % NTT;
  const int rt = MC ? b / (2 * S) : b / (S * NTT);
  const unsigned char* wsrc = p.w + (size_t)(rt * S + ks) * NSTG * AB;
  TR(0);
  if (wid == 4 && lane == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&p.tmx) : "memory");
    for (int s = 0; s < NS; ++s) { mbar_init(bars + 8 * s, 1); mbar_init(bars + 8 * (NS + s), MC ? 8 : 4); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (MC) asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
  if (ZA && wid == 5) {   // zero tokens [ks*TN/S, (ks+1)*TN/S) x 64 rows of this CTA's output tile, then release
    const int tz = TN / S, t0 = tt * TN + ks * tz;
    for (int i = lane; i < tz * 16; i += 32) {
      const int t = t0 + (i >> 4);
      if (t < M) *(float4*)&p.out[(size_t)t * 256 + rt * 64 + (i & 15) * 4] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    __syncwarp();
    if (lane == 0) { __threadfence(); asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(p.zc + tt * 4 + rt) : "memory"); }
  }
  if (wid == 4) {
    if (lane == 0) {
#pragma unroll 1
      for (int s = 0; s < NSTG; ++s) {
        const int slot = s % NS;
        if (s >= NS) mbar_wait(bars + 8 * (NS + slot), ((s / NS) - 1) & 1);
        const u32 st = ring + slot * SB;
        mbar_expect_arrive(bars + 8 * slot, SB);
        if (MC) bulk_g2s_mc(st + tt * 8192, wsrc + (size_t)s * AB + tt * 8192, 8192, bars + 8 * slot, (unsigned short)3);
        else bulk_g2s(st, wsrc + (size_t)s * AB, AB, bars + 8 * slot);
        tma2d(st + AB, &p.tmx, ks * KK + s * 64, tt * TN, bars + 8 * slot);
      }
    }
    __syncwarp();
  }
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = 0.f;
  if (CA && wid == 0) {
    const int blk = rt * S + ks;
    const int o0 = p.eoff[blk], n = min(p.eoff[blk + 1] - o0, 32);
    if (lane < n) { esc_e[lane] = p.ent[o0 + lane]; esc_d[lane] = p.edl[o0 + lane]; }
    if (lane == 0) esc_n = n;
  }
  if (CA && wid < 4) asm volatile("bar.sync 1, 128;" ::: "memory");   // consumer warps see esc_* (not the producer)
  if (CA && wid < 4) {
    const u32 base2 = (u32)p.base2;
    const int tmx = min(TN, M - tt * TN);
    u32 hA[4][4], lA[4][4], hB[4][4], lB[4][4];
    auto go = [&](int s, u32 (&h)[4][4], u32 (&l)[4][4]) {
      const int slot = s % NS;
      mbar_wait(bars + 8 * slot, (s / NS) & 1);
      const u32 st = ring + slot * SB;
      const u32 tq = st + threadIdx.x * 16;
      const uint4 hb01 = lds128(tq), hb23 = lds128(tq + 2048), lb01 = lds128(tq + 4096), lb23 = lds128(tq + 6144);
      const uint4 hn = lds128(tq + 8192), ln = lds128(tq + 10240);
      dec_hi(make_uint2(hb01.x, hb01.y), hn.x, base2, h[0]); dec_lo(make_uint2(lb01.x, lb01.y), ln.x, h[0], l[0]);
      dec_hi(make_uint2(hb01.z, hb01.w), hn.y, base2, h[1]); dec_lo(make_uint2(lb01.z, lb01.w), ln.y, h[1], l[1]);
      dec_hi(make_uint2(hb23.x, hb23.y), hn.z, base2, h[2]); dec_lo(make_uint2(lb23.x, lb23.y), ln.z, h[2], l[2]);
      dec_hi(make_uint2(hb23.z, hb23.w), hn.w, base2, h[3]); dec_lo(make_uint2(lb23.z, lb23.w), ln.w, h[3], l[3]);
      const u64 dx = mkdesc(st + AB);
      wg_fence();
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (TN == 128) { wgmma_rs_n128(acc, h[j], dx + 2 * j); wgmma_rs_n128(acc, l[j], dx + 2 * j); }
        else { wgmma_rs_n64(acc, h[j], dx + 2 * j); wgmma_rs_n64(acc, l[j], dx + 2 * j); }
      }
      wg_commit();
    };
    auto done = [&](int s) {   // wgmma(s) complete: escape corrections from stage s's x tile, then release the slot
      const int slot = s % NS;
      const u32 st = ring + slot * SB;
      for (int i = threadIdx.x; i < esc_n * TN; i += 128) {
        const int e = i / TN, t = i % TN;
        const int en = esc_e[e];
        if ((en >> 16) == s && t < tmx) {
          const int kl = en & 63;
          const u32 xa = st + AB + t * 128 + ((((kl >> 3) ^ (t & 7)) << 4) | ((kl & 7) * 2));
          u32 xv; asm volatile("ld.shared.u16 %0, [%1];" : "=r"(xv) : "r"(xa));
          corrv[e * TN + t] = __uint_as_float(xv << 16) * esc_d[e];
        }
      }
      __syncwarp();
      if (lane == 0 && s + NS < NSTG) mbar_arrive(bars + 8 * (NS + slot));
    };
    go(0, hA, lA);
#pragma unroll 1
    for (int s = 1; s < NSTG; s += 2) {
      go(s, hB, lB); wg_wait<1>(); done(s - 1);
      if (s + 1 < NSTG) { go(s + 1, hA, lA); wg_wait<1>(); done(s); }
    }
    wg_wait<0>();
    done(NSTG - 1);
  }
  if (!CA && wid < 4) {
#pragma unroll 1
    for (int s = 0; s < NSTG; ++s) {
      const int slot = s % NS;
      mbar_wait(bars + 8 * slot, (s / NS) & 1);
      if (s == 0) TR(1);
      if (s == NSTG - 1) TR(2);
      const u32 st = ring + slot * SB;
      const u64 dh = mkdesc(st), dl = mkdesc(st + 8192), dx = mkdesc(st + AB);
      wg_fence();
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (TN == 256) { wgmma_n256(acc, dh + 2 * j, dx + 2 * j); wgmma_n256(acc, dl + 2 * j, dx + 2 * j); }
        else if (TN == 128) { wgmma_n128(acc, dh + 2 * j, dx + 2 * j); wgmma_n128(acc, dl + 2 * j, dx + 2 * j); }
        else { wgmma_n64(acc, dh + 2 * j, dx + 2 * j); wgmma_n64(acc, dl + 2 * j, dx + 2 * j); }
      }
      wg_commit();
      wg_wait<0>();
      __syncwarp();
      if (lane == 0 && s + NS < NSTG) {
        mbar_arrive(bars + 8 * (NS + slot));
        if (MC) mbar_arrive_remote(bars + 8 * (NS + slot), (u32)(tt ^ 1));
      }
    }
    TR(3);
  }
  __syncthreads();
  if (wid < 4) {
    // acc[4i+{0,1,2,3}] = D[16w + l/4 (+8)][8i + 2(l%4) (+1)]; D rows = W rows, cols = tokens -> tr[token][row]
    const int r = 16 * wid + (lane >> 2);
#pragma unroll
    for (int i = 0; i < TN / 8; ++i) {
      const int t = 8 * i + 2 * (lane & 3);
      tr[t * TS + r] = acc[4 * i + 0];
      tr[(t + 1) * TS + r] = acc[4 * i + 1];
      tr[t * TS + r + 8] = acc[4 * i + 2];
      tr[(t + 1) * TS + r + 8] = acc[4 * i + 3];
    }
  }
  __syncthreads();
  if (CA) {
    const int tmx = min(TN, M - tt * TN);
    for (int i = threadIdx.x; i < esc_n * tmx; i += blockDim.x) {
      const int e = i / tmx, t = i % tmx;
      atomicAdd(&tr[t * TS + ((esc_e[e] >> 8) & 63)], corrv[e * TN + t]);
    }
    __syncthreads();
  }
  TR(4);
  if (ZA) {
    const u32 target = *(volatile const u32*)(p.zc + 64 + b) + S;
    while ((int)(ld_acquire_u32(p.zc + tt * 4 + rt) - target) < 0) {}
  } else if (ZF) wait_zero(p.zc, p.zc + 64 + b);
  else asm volatile("griddepcontrol.wait;" ::: "memory");
  TR(5);
  const int tmax = min(TN, M - tt * TN);
  for (int e = threadIdx.x; e < tmax * 16; e += blockDim.x) {
    const int t = e >> 4, r4 = (e & 15) * 4;
    const float4 v = *(float4*)&tr[t * TS + r4];
    red_add_v4(&p.out[(size_t)(tt * TN + t) * 256 + rt * 64 + r4], v);
  }
  __syncthreads();
  if (ZF && threadIdx.x == 0) p.zc[64 + b] += ZCTAS;
  if (ZA && threadIdx.x == 0) p.zc[64 + b] += S;
  TR(6);
}

// ---- rq: M <= 8, no split-K: CTA = 2 rows x full K (128 CTAs), mma.sync hi/lo; x via 3D TMA box (64, 8, 32) swizzled ----
template <int NCH>
__device__ __forceinline__ void rq_body(const PS& p) {
  constexpr int CK = 8192 / NCH;                // 2048
  constexpr int WCB = CK / 32 * 256, XCB = 8 * CK * 2;
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  unsigned char* sm = (unsigned char*)(((size_t)sm_raw + 1023) & ~(size_t)1023);
  const u32 bars = smem_u32(sm);
  const u32 xsm = bars + 1024;                  // NCH x 32 KB (1024-aligned)
  const u32 wsm = xsm + NCH * XCB;
  __shared__ float red[8][32];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const int M = (int)p.M;
  if (tid == 0) {
    for (int c = 0; c < NCH; ++c) mbar_init(bars + 8 * c, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    const unsigned char* wsrc = (const unsigned char*)p.w + (size_t)blockIdx.x * (NCH * WCB);
#pragma unroll 1
    for (int c = 0; c < NCH; ++c) {
      mbar_expect_arrive(bars + 8 * c, WCB + XCB);
      bulk_g2s(wsm + c * WCB, wsrc + (size_t)c * WCB, WCB, bars + 8 * c);
      tma3d(xsm + c * XCB, &p.tmx, 0, 0, c * (CK / 64), bars + 8 * c);
    }
  }
  __syncthreads();
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  const int t = lane & 7;                       // ldmatrix row (rows 8..15 alias 0..7: their outputs are discarded)
#pragma unroll 1
  for (int c = 0; c < NCH; ++c) {
    mbar_wait(bars + 8 * c, 0);
    const u32 xc = xsm + c * XCB, wc = wsm + c * WCB;
#pragma unroll
    for (int i = 0; i < CK / 32 / 8; ++i) {
      const int kt = wid * (CK / 32 / 8) + i;   // k32 tile within chunk
      u32 a[2][4];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int kl = kt * 32 + h * 16;
        const int cc = ((kl & 63) >> 3) + (lane >> 4);
        ldsm_x4(a[h], xc + (kl >> 6) * 1024 + t * 128 + ((cc ^ t) << 4));
      }
      uint4 bh = make_uint4(0, 0, 0, 0), bl = make_uint4(0, 0, 0, 0);
      if (lane < 8) { bh = lds128(wc + kt * 256 + lane * 16); bl = lds128(wc + kt * 256 + 128 + lane * 16); }
      mma16816(acc, a[0], bh.x, bh.y);
      mma16816(acc, a[1], bh.z, bh.w);
      mma16816(acc, a[0], bl.x, bl.y);
      mma16816(acc, a[1], bl.z, bl.w);
    }
  }
  // acc: c0,c1 = (token lane/4, rows 2(lane&3), +1); only rows 0,1 (lane&3 == 0) and tokens < 8 matter
  if ((lane & 3) == 0) { red[wid][2 * (lane >> 2)] = acc[0]; red[wid][2 * (lane >> 2) + 1] = acc[1]; }
  __syncthreads();
  if (tid < 2 * M) {
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) v += red[w][tid];
    p.out[(size_t)(tid >> 1) * 256 + blockIdx.x * 2 + (tid & 1)] = v;
  }
}
"""

# name: (TT, RR, KK, KS, NS, WK, WG, max_M)
_b7__rt_CFGS = {
    "rt16": (16, 16, 1024, 256, 4, 8, 1, 16),
    "rt32": (32, 16, 1024, 256, 4, 8, 1, 32),
    "rt64": (64, 32, 512, 256, 2, 8, 1, 64),
    "rt128": (128, 32, 512, 256, 2, 4, 2, 128),
    "rtbig": (128, 32, 512, 256, 2, 4, 2, 1 << 30),
}


def _b7__rt_smem_bytes(cfg):
    tt, rr, kk, ks, ns, wk, wg, _ = cfg
    sb = (tt * ks * 2 + ks * rr * 4 + 1023) // 1024 * 1024
    return 1024 + 1024 + ns * sb


_b7__rt_RS_NCH = 4
_b7__rt_RS_MAX = 8


def _b7__rt_rs_smem(m):
    return 1024 + 1024 + _b7__rt_RS_NCH * (2 * 2048 * 4 + m * 2048 * 2)


_b7__rt_FULL_SOURCE = False   # True: compile every experimental kernel (ablations); False: only what DISPATCH uses


def _b7__rt_source(cfgs=_b7__rt_CFGS):
    s = _b7__rt_SRC
    if not _b7__rt_FULL_SOURCE:
        for name in ("rt16", "rt32", "rt64"):
            tt, rr, kk, ks, ns, wk, wg, _ = cfgs[name]
            nt = 32 * (wk * wg + 1)
            for fl in (4, 16, 272):
                s += (f'extern "C" __global__ void __launch_bounds__({nt},1) {name}_f{fl}(const __grid_constant__ P p)'
                      f' {{ rt_body<{tt},{rr},{kk},{ks},{ns},{wk},{wg},{fl}>(p); }}\n')
        for name in ("rw64", "rw128b"):
            tn, kk, ns = _b7__rt_RW_CFGS[name]
            s += (f'extern "C" __global__ void __launch_bounds__(160,1) {name}(const __grid_constant__ PW p)'
                  f' {{ rw_body<{tn},{kk},{ns},0,0>(p); }}\n')
            s += (f'extern "C" __global__ void __launch_bounds__(160,1) {name}c(const __grid_constant__ PW p)'
                  f' {{ rw_body<{tn},{kk},{ns},0,0,1>(p); }}\n')
            s += (f'extern "C" __global__ void __launch_bounds__(192,1) {name}za(const __grid_constant__ PW p)'
                  f' {{ rw_body<{tn},{kk},{ns},0,0,0,1>(p); }}\n')
        for m in range(1, 5):
            s += (f'extern "C" __global__ void __launch_bounds__(256,1) rs{m}(const __grid_constant__ PS p)'
                  f' {{ rs_body<{m},{_b7__rt_RS_NCH}>(p); }}\n')
        return s
    for name, (tt, rr, kk, ks, ns, wk, wg, _) in cfgs.items():
        nt = 32 * (wk * wg + 1)
        for fl in (0, 1, 2, 3, 4, 8, 16, 48, 80, 144, 272):
            s += (f'extern "C" __global__ void __launch_bounds__({nt},1) {name}_f{fl}(const __grid_constant__ P p)'
                  f' {{ rt_body<{tt},{rr},{kk},{ks},{ns},{wk},{wg},{fl}>(p); }}\n')
    for name, (tn, kk, ns) in _b7__rt_RW_CFGS.items():
        mc = 1 if name.endswith("mc") else 0
        cd = "__cluster_dims__(2,1,1) " if mc else ""
        s += (f'extern "C" __global__ void {cd}__launch_bounds__(160,1) {name}(const __grid_constant__ PW p)'
              f' {{ rw_body<{tn},{kk},{ns},{mc},0>(p); }}\n')
        if not mc:
            s += (f'extern "C" __global__ void __launch_bounds__(160,1) {name}z(const __grid_constant__ PW p)'
                  f' {{ rw_body<{tn},{kk},{ns},0,1>(p); }}\n')
            s += (f'extern "C" __global__ void __launch_bounds__(160,1) {name}c(const __grid_constant__ PW p)'
                  f' {{ rw_body<{tn},{kk},{ns},0,0,1>(p); }}\n')
            s += (f'extern "C" __global__ void __launch_bounds__(192,1) {name}za(const __grid_constant__ PW p)'
                  f' {{ rw_body<{tn},{kk},{ns},0,0,0,1>(p); }}\n')
    for mt in (1, 2, 4):
        s += (f'extern "C" __global__ void __cluster_dims__(8,1,1) __launch_bounds__(288,1) rc{mt}(const __grid_constant__ PC p)'
              f' {{ rc_body<{mt}>(p); }}\n')
    s += 'extern "C" __global__ void __launch_bounds__(256,1) rq(const __grid_constant__ PS p) { rq_body<4>(p); }\n'
    for m in range(1, _b7__rt_RS_MAX + 1):
        s += (f'extern "C" __global__ void __launch_bounds__(256,1) rs{m}(const __grid_constant__ PS p)'
              f' {{ rs_body<{m},{_b7__rt_RS_NCH}>(p); }}\n')
    return s


class _b7__rt__Mod:
    def __init__(self, src):
        from cuda.bindings import driver, nvrtc
        self.d = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"router.cu", 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err
        self.funcs = {}

    def func(self, name, smem):
        f = self.funcs.get(name)
        if f is None:
            d = self.d
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            assert int(err) == 0, (name, err)
            (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
            assert int(err) == 0, err
            self.funcs[name] = f
        return f

    def launch_pdl(self, name, grid, block, smem, params):
        d = self.d
        f = self.func(name, smem)
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        cfg = d.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block, 1, 1
        cfg.sharedMemBytes = smem
        cfg.hStream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        at = d.CUlaunchAttribute()
        at.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        at.value.programmaticStreamSerializationAllowed = 1
        cfg.attrs = [at]
        cfg.numAttrs = 1
        (err,) = d.cuLaunchKernelEx(cfg, f, ctypes.addressof(argv), 0)
        assert int(err) == 0, err

    def launch(self, name, grid, block, smem, params):
        f = self.func(name, smem)
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = self.d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = self.d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        assert int(err) == 0, err


def _b7__rt__tmap_x(x, m, box_rows):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(2), x.data_ptr(),
        [d.cuuint64_t(8192), d.cuuint64_t(m)], [d.cuuint64_t(8192 * 2)],
        [d.cuuint32_t(64), d.cuuint32_t(box_rows)], [d.cuuint32_t(1), d.cuuint32_t(1)],
        d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


def _b7__rt__tmap_x3(x, m):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(3), x.data_ptr(),
        [d.cuuint64_t(256), d.cuuint64_t(32), d.cuuint64_t(m)], [d.cuuint64_t(512), d.cuuint64_t(8192 * 2)],
        [d.cuuint32_t(256), d.cuuint32_t(8), d.cuuint32_t(m)], [d.cuuint32_t(1)] * 3,
        d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
        d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


_b7__rt_RC_Q = 15
# rw configs: name -> (TN tokens per CTA, KK k per CTA, NS ring slots)
_b7__rt_RW_CFGS = {"rw128a": (128, 256, 4), "rw128b": (128, 512, 6), "rw64": (64, 512, 6), "rw128bmc": (128, 512, 6),
           "rw128cmc": (128, 1024, 6), "rw64mc": (64, 512, 6), "rw256": (256, 256, 4), "rw64n8": (64, 512, 8),
           "rw128b7": (128, 512, 7)}
_b7__rt_RW_SEL = {"small": "rw128a", "big": "rw128b"}   # M <= 128 -> small, else big
_b7__rt_USE_RW = True


def _b7__rt_rw_smem(name):
    tn, kk, ns = _b7__rt_RW_CFGS[name]
    return 1024 + 1024 + ns * (16384 + tn * 128)


def _b7__rt__rw_layout(hi, lo, kk):
    """[4 row tiles][S k splits][KK/64 stages][hi 64x64 swizzled, lo 64x64 swizzled] bf16."""
    s = 8192 // kk
    idx = (torch.arange(8, device=hi.device)[None, :] ^ (torch.arange(64, device=hi.device)[:, None] & 7))
    idx = idx[:, :, None].expand(64, 8, 8)

    def tiles(w):
        t = w.contiguous().view(4, 64, 128, 64).permute(0, 2, 1, 3).contiguous()   # [rt][k64][row][k]
        t = t.view(4, 128, 64, 8, 8)
        return torch.gather(t, 3, idx[None, None].expand(4, 128, 64, 8, 8))      # out[r][j] = T[r][j ^ (r&7)]
    th, tl = tiles(hi), tiles(lo)
    t = torch.stack([th, tl], 2)                                                  # [rt][k64][2][64][8][8]
    return t.view(4, s, kk // 64, 2, 64 * 64).contiguous().view(-1)


def _b7__rt__rwc_layout(c, kk):
    """[4 rt][S][KK/64 stages][6 chunks][128 threads][16 B]: per thread & k16 step j the 8 A-fragment values
    v = (kh, h, p) for rows 16w + 8h + g, k = 16j + 8kh + 2q + p (thread = (w, g, q)); chunks: hi bytes j01, j23,
    lo bytes j01, j23, hi nibble words j0..3, lo nibble words j0..3."""
    s_ = 8192 // kk

    def perm(a):   # [256, 8192] -> [4 rt][128 k64][128 thr][4 j][8 v]
        t = a.view(4, 4, 2, 8, 128, 4, 2, 4, 2)          # rt, w, h, g, k64, j, kh, q, p
        return t.permute(0, 4, 1, 3, 7, 5, 6, 2, 8).contiguous().view(4, 128, 128, 4, 8)

    def nib(cc):
        cc = perm(cc).long()
        w = torch.zeros(cc.shape[:4], dtype=torch.long, device=cc.device)
        for i in range(4):
            w |= cc[..., 2 * i] << (4 * i)
            w |= cc[..., 2 * i + 1] << (16 + 4 * i)
        return (w - ((w >> 31) & 1) * (1 << 32)).to(torch.int32).contiguous().view(torch.uint8).view(4, 128, 128, 16)
    hb = perm(c["hb"]).to(torch.uint8).view(4, 128, 128, 32)
    lb = perm(c["lb"]).to(torch.uint8).view(4, 128, 128, 32)
    ch = [hb[..., :16], hb[..., 16:], lb[..., :16], lb[..., 16:], nib(c["ch"]), nib(c["cl"])]
    t = torch.stack(ch, 2)                                  # [4 rt][128 k64][6][128 thr][16]
    return t.contiguous().view(-1)


def _b7__rt__rwc_escapes(c, kk, dev):
    rows = torch.repeat_interleave(torch.arange(256, device=dev), (c["erow"][1:] - c["erow"][:-1]).long())
    k = c["ek"].long()
    s_ = 8192 // kk
    blk = (rows // 64) * s_ + k // kk
    stage = (k % kk) // 64
    order = torch.argsort(blk * (1 << 20) + stage * 4096 + (rows % 64) * 64 + k % 64)
    cnt = torch.bincount(blk, minlength=4 * s_)
    off = torch.zeros(4 * s_ + 1, dtype=torch.int32, device=dev)
    off[1:] = cnt.cumsum(0).int()
    ent = ((stage << 16) | ((rows % 64) << 8) | (k % 64))[order].int().contiguous()
    return off, ent, c["ed"][order].contiguous(), int(cnt.max()) if cnt.numel() else 0


def _b7__rt__tmap_x2(x, m, box_rows):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(2), x.data_ptr(),
        [d.cuuint64_t(8192), d.cuuint64_t(m)], [d.cuuint64_t(8192 * 2)],
        [d.cuuint32_t(64), d.cuuint32_t(box_rows)], [d.cuuint32_t(1), d.cuuint32_t(1)],
        d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


def _b7__rt_rc_smem(mt):
    tt = 16 * mt
    sb = (tt * 256 * 2 + 18 * 256 * 4 + 1023) // 1024 * 1024
    return 1024 + 1024 + 4 * sb


def _b7__rt__tmap_x3s(x, m, tt):
    """x [m, 8192] as 3D (64 k_inner, m rows, 128 k_outer), box (64, tt, 4), 128B swizzle -> smem [4][tt][128 B]."""
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(3), x.data_ptr(),
        [d.cuuint64_t(64), d.cuuint64_t(m), d.cuuint64_t(128)], [d.cuuint64_t(8192 * 2), d.cuuint64_t(128)],
        [d.cuuint32_t(64), d.cuuint32_t(tt), d.cuuint32_t(4)], [d.cuuint32_t(1)] * 3,
        d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


_b7__rt_RQ_SMEM = 1024 + 1024 + 4 * (32768 + 16384)


def _b7__rt__rq_layout(hi, lo):
    """[128 CTAs (rows 2b, 2b+1)][256 kt][hi: 8 lanes x 16 B][lo: 8 lanes x 16 B] (lanes = row*4 + tid, row < 2)."""
    out = []
    for w in (hi, lo):
        t = w.contiguous().view(128, 2, 256, 2, 2, 4, 2).permute(0, 2, 1, 5, 3, 4, 6).contiguous()  # [b][kt][row][tid][..]
        out.append(t.view(128, 256, 1, 64))
    return torch.cat(out, 2).contiguous().view(-1)


def _b7__rt__tmap_x3q(x, m):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(3), x.data_ptr(),
        [d.cuuint64_t(64), d.cuuint64_t(m), d.cuuint64_t(128)], [d.cuuint64_t(8192 * 2), d.cuuint64_t(128)],
        [d.cuuint32_t(64), d.cuuint32_t(8), d.cuuint32_t(32)], [d.cuuint32_t(1)] * 3,
        d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


def _b7__rt__rc_layout(hi, lo):
    """Per cluster q (rows r0..r0+nr) and rank (k slice of 1024): [32 kt][groups: hi rg x 64 B, lo rg x 64 B]."""
    parts = []
    for q in range(_b7__rt_RC_Q):
        r0, r1 = q * 256 // _b7__rt_RC_Q, (q + 1) * 256 // _b7__rt_RC_Q
        nr = r1 - r0
        cols = []
        for g in range((nr + 7) // 8):
            a, b = r0 + 8 * g, min(r1, r0 + 8 * g + 8)
            rg = b - a
            for w in (hi, lo):
                t = w[a:b].contiguous().view(rg, 256, 2, 2, 4, 2).permute(1, 0, 4, 2, 3, 5).contiguous()
                cols.append(t.view(256, rg * 32))
        blk = torch.cat(cols, 1)                    # [256 kt][nr * 64 bf16]
        parts.append(blk.view(8, 32, nr * 64).contiguous().view(-1))
    return torch.cat(parts)


def _b7__rt__compress_pair(hi, lo):
    """Lossless 24-bit code of the (hi, lo) BF16 pair: per weight a sign+mantissa byte for hi and for lo, a 4-bit hi
    exponent code (e_hi - base) and a 4-bit lo code (e_hi_dec - 8 - e_lo). Out-of-range / zero exponents decode to a
    nearby value; their exact difference goes to an escape list (row, k, delta) applied by the zero CTAs."""
    h = hi.contiguous().view(torch.int16).int() & 0xFFFF
    l = lo.contiguous().view(torch.int16).int() & 0xFFFF
    eh, el = (h >> 7) & 0xFF, (l >> 7) & 0xFF
    hist = torch.bincount(eh.flatten(), minlength=256).double()
    hist[0] = 0
    cs = torch.cat([torch.zeros(1, dtype=torch.float64, device=h.device), hist.cumsum(0)])
    win = cs[16:256] - cs[0:240]                 # coverage of [b, b+15] for b = 0..239
    win[0] = -1
    base = int(win.argmax())
    ch = eh - base
    hesc = (ch < 0) | (ch > 15) | (eh == 0)
    ch = torch.where(hesc, torch.zeros_like(ch), ch)
    ehd = ch + base
    d = ehd - 8 - el
    lesc = (d < 0) | (d > 15) | (el == 0)
    d = torch.where(lesc, torch.zeros_like(d), d)
    eld = ehd - 8 - d
    hb = ((h >> 8) & 0x80) | (h & 0x7F)
    lb = ((l >> 8) & 0x80) | (l & 0x7F)
    hdec = (h & 0x8000) | (ehd << 7) | (h & 0x7F)
    ldec = (l & 0x8000) | (eld << 7) | (l & 0x7F)

    def asbf(v):
        return (v.to(torch.int32) & 0xFFFF).to(torch.int16).view(torch.bfloat16).float()
    esc = hesc | lesc
    rows, ks = esc.nonzero(as_tuple=True)
    delta = ((hi.float() + lo.float()) - (asbf(hdec) + asbf(ldec)))[rows, ks]
    erow = torch.zeros(257, dtype=torch.int32, device=h.device)
    erow[1:] = torch.bincount(rows, minlength=256).cumsum(0).int()
    return dict(hb=hb, lb=lb, ch=ch, cl=d, base=base, erow=erow, ek=ks.int().contiguous(), ed=delta.float().contiguous(),
                nesc=int(rows.numel()))


def _b7__rt__cmp_layout(c, rr, kk, ks):
    """Tiles (8 rows x 32 k): [lane x (hi bytes 8 | lo bytes 8)] 512 B + [lane x (hi nib u32 | lo nib u32)] 256 B."""
    def perm(a):
        return a.view(32, 8, 256, 2, 2, 4, 2).permute(0, 2, 1, 5, 3, 4, 6).contiguous().view(32, 256, 32, 8)
    hb, lb = perm(c["hb"]), perm(c["lb"])
    byt = torch.cat([hb, lb], 3).to(torch.uint8).view(32, 256, 512)

    def nib(cc):
        cc = perm(cc).long()
        w = torch.zeros(cc.shape[:3], dtype=torch.long, device=cc.device)
        for j in range(4):
            w |= cc[..., 2 * j] << (4 * j)
            w |= cc[..., 2 * j + 1] << (16 + 4 * j)
        return w.to(torch.int64).view(-1).to(torch.int32).view(32, 256, 32, 1) if False else\
            (w - ((w >> 31) & 1) * (1 << 32)).to(torch.int32).view(32, 256, 32, 1)
    nb = torch.cat([nib(c["ch"]), nib(c["cl"])], 3).contiguous().view(torch.uint8).view(32, 256, 256)
    t = torch.cat([byt, nb], 2)                                   # [32 g][256 kt][768]
    nr, s, nstg = 256 // rr, 8192 // kk, kk // ks
    t = t.view(nr, rr // 8, s, nstg, ks // 32, 768).permute(0, 2, 3, 4, 1, 5).contiguous()
    return t.view(-1)


def _b7__rt__perm_tiles(w):
    """[256, 8192] bf16 -> [32 groups][256 k32][256 bf16] in mma B-fragment lane order."""
    return w.contiguous().view(32, 8, 256, 2, 2, 4, 2).permute(0, 2, 1, 5, 3, 4, 6).contiguous().view(32, 256, 256)


def _b7__rt__layout(hi_t, lo_t, rr, kk, ks):
    t = torch.stack([hi_t, lo_t], 2)  # [32, 256, 2, 256]
    nr, s, nstg = 256 // rr, 8192 // kk, kk // ks
    t = t.view(nr, rr // 8, s, nstg, ks // 32, 2, 256).permute(0, 2, 3, 4, 1, 5, 6).contiguous()
    return t.view(-1)


def _b7__rt_prepare_router(weights):
    w = weights["router"].float().contiguous()
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)
    hi_t, lo_t = _b7__rt__perm_tiles(hi), _b7__rt__perm_tiles(lo)
    st = {"mod": _b7__rt__Mod(_b7__rt_source()), "lay": {}}
    for name, cfg in _b7__rt_CFGS.items():
        key = (cfg[1], cfg[2], cfg[3])
        if key not in st["lay"]:
            st["lay"][key] = _b7__rt__layout(hi_t, lo_t, *key)
    c = _b7__rt__compress_pair(hi, lo)
    st["cmp"] = c
    st["w_cmp"], st["esc"] = {}, {}
    rows = torch.repeat_interleave(torch.arange(256, device=w.device), (c["erow"][1:] - c["erow"][:-1]).long())
    ks_ = c["ek"].long()
    for name, cfg in _b7__rt_CFGS.items():
        key = (cfg[1], cfg[2], cfg[3])
        if key not in st["w_cmp"]:
            rr, kk = cfg[1], cfg[2]
            st["w_cmp"][key] = _b7__rt__cmp_layout(c, *key)
            blk = (rows // rr) * (8192 // kk) + ks_ // kk
            order = torch.argsort(blk * (1 << 24) + (rows % rr) * 8192 + ks_ % kk)
            nblk = (256 // rr) * (8192 // kk)
            cnt = torch.bincount(blk, minlength=nblk)
            off = torch.zeros(nblk + 1, dtype=torch.int32, device=w.device)
            off[1:] = cnt.cumsum(0).int()
            ent = (((rows % rr) << 16) | (ks_ % kk))[order].int().contiguous()
            st["esc"][key] = (off, ent, c["ed"][order].contiguous(), int(cnt.max()) if cnt.numel() else 0)
    st["w_rc"] = _b7__rt__rc_layout(hi, lo)
    st["w_rq"] = _b7__rt__rq_layout(hi, lo)
    st["w_rw"] = {kk: _b7__rt__rw_layout(hi, lo, kk) for kk in sorted({c[1] for c in _b7__rt_RW_CFGS.values()})}
    st["w_rwc"] = {kk: _b7__rt__rwc_layout(st["cmp"], kk) for kk in (512,)}
    st["rwc_esc"] = {kk: _b7__rt__rwc_escapes(st["cmp"], kk, w.device) for kk in (512,)}
    st["w_rs"] = w.view(128, 2, _b7__rt_RS_NCH, 8192 // _b7__rt_RS_NCH).permute(0, 2, 1, 3).contiguous()
    st["ws"] = torch.zeros(1 << 20, dtype=torch.float32, device=w.device)
    # in-kernel zeroing sync state, one block of 4096 ints per (kernel, grid) key: [0] zf, [64 + b] CTA b's generation.
    # Each key always launches the same grid, so every CTA index takes part in every call of that key.
    st["zkeys"] = {k: i for i, k in enumerate([("rt16", 1), ("rt32", 1), ("rt64", 1)] + [("rw64", n) for n in (1, 2)]
                                              + [("rw128b", n) for n in range(1, 17)] + [("rw128a", n) for n in range(1, 17)])}
    st["zc"] = torch.zeros(len(st["zkeys"]) * 4096, dtype=torch.int32, device=w.device)
    st["cnt"] = torch.zeros(4096, dtype=torch.int32, device=w.device)
    return st


def _b7__rt__cfg_for(m):
    for name, cfg in _b7__rt_CFGS.items():
        if m <= cfg[7]:
            return name, cfg


_b7__rt_USE_RS = False
_b7__rt_USE_RC = True
_b7__rt_USE_RQ = False
_b7__rt_FLAGS = 0
_b7__rt_TRACE = None
_b7__rt_ZMODE = False
_b7__rt_DELAY_NS = 0
_b7__rt_RWC = False
_b7__rt_RWZA = False


# default dispatch (measured 2026-10-27, see NOTES.md); the module-level switches above are for ablations
_b7__rt_DISPATCH = True
_b7__rt_SMALL_FLAGS = None   # None: per-M choice below; an int forces one rt epilogue


def _b7__rt_run_router(x, st):
    global _b7__rt_USE_RS, _b7__rt_USE_RC, _b7__rt_USE_RW, _b7__rt_RW_SEL, _b7__rt_FLAGS, _b7__rt_USE_RQ
    m = x.shape[0]
    if _b7__rt_DISPATCH:
        _b7__rt_USE_RC = False
        _b7__rt_USE_RQ = False
        _b7__rt_USE_RS = m <= 4
        _b7__rt_USE_RW = m > 64
        _b7__rt_RW_SEL = {"small": "rw64", "big": "rw128b"}
        # rt: 16 = in-kernel zero CTAs; 272 = same + per-warp direct red.adds (fewer reds at M <= 8 only)
        _b7__rt_FLAGS = _b7__rt_SMALL_FLAGS if _b7__rt_SMALL_FLAGS is not None else (272 if m <= 8 else 16)
    out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
    if _b7__rt_USE_RW and m > 64:
        name = _b7__rt_RW_SEL["small"] if m <= 128 else _b7__rt_RW_SEL["big"]
        tn, kk, ns = _b7__rt_RW_CFGS[name]
        ntt = (m + tn - 1) // tn
        if name.endswith("mc") and ntt != 2:
            name = {"rw128bmc": "rw128b", "rw128cmc": "rw128b", "rw64mc": "rw64"}[name]
            tn, kk, ns = _b7__rt_RW_CFGS[name]
            ntt = (m + tn - 1) // tn
        zk = st["zkeys"].get((name, ntt))
        zptr = st["zc"].data_ptr() + 4 * 4096 * (zk or 0)
        if _b7__rt_RWZA and zk is not None:
            params = _b7__rt__tmap_x2(x, m, tn) + struct.pack("QqqqQQ", st["w_rw"][kk].data_ptr(), out.data_ptr(), m, ntt,
                                                      _b7__rt_TRACE.data_ptr() if _b7__rt_TRACE is not None else 0, zptr)
            params += b"\0" * ((-len(params)) % 64)
            st["mod"].launch(name + "za", 4 * ntt * (8192 // kk), 192, _b7__rt_rw_smem(name), params)
            return out
        if _b7__rt_RWC and kk in st["w_rwc"] and st["rwc_esc"][kk][3] <= 32:
            c = st["cmp"]
            off, ent, dl, _ = st["rwc_esc"][kk]
            b2 = (c["base"] << 7) | (c["base"] << 23)
            params = _b7__rt__tmap_x2(x, m, tn) + struct.pack("QqqqQQQQQq", st["w_rwc"][kk].data_ptr(), out.data_ptr(), m, ntt,
                                                      0, zptr, off.data_ptr(), ent.data_ptr(), dl.data_ptr(), b2)
            params += b"\0" * ((-len(params)) % 64)
            st["mod"].launch("rzero", 16, 256, 0, struct.pack("Qq", out.data_ptr(), m * 64))
            st["mod"].launch_pdl(name + "c", 4 * ntt * (8192 // kk), 160, _b7__rt_rw_smem(name) + 0, params)
            return out
        params = _b7__rt__tmap_x2(x, m, tn) + struct.pack("QqqqQQ", st["w_rw"][kk].data_ptr(), out.data_ptr(), m, ntt,
                                                  _b7__rt_TRACE.data_ptr() if _b7__rt_TRACE is not None else 0, zptr)
        params += b"\0" * ((-len(params)) % 64)
        if _b7__rt_ZMODE and zk is not None:
            st["mod"].launch(name + "z", 4 + 4 * ntt * (8192 // kk), 160, _b7__rt_rw_smem(name), params)
            return out
        st["mod"].launch("rzero", 16, 256, 0, struct.pack("Qq", out.data_ptr(), m * 64))
        st["mod"].launch_pdl(name, 4 * ntt * (8192 // kk), 160, _b7__rt_rw_smem(name), params)
        return out
    if _b7__rt_USE_RQ and m <= 8:
        params = _b7__rt__tmap_x3q(x, m) + struct.pack("QQq", st["w_rq"].data_ptr(), out.data_ptr(), m)
        params += b"\0" * ((-len(params)) % 64)
        st["mod"].launch("rq", 128, 256, _b7__rt_RQ_SMEM, params)
        return out
    if _b7__rt_USE_RC and m <= 64 and not (_b7__rt_USE_RS and m <= _b7__rt_RS_MAX):
        mt = 1 if m <= 16 else (2 if m <= 32 else 4)
        params = _b7__rt__tmap_x3s(x, m, 16 * mt) + struct.pack("QqqQ", st["w_rc"].data_ptr(), out.data_ptr(), m,
                                                          _b7__rt_TRACE.data_ptr() if _b7__rt_TRACE is not None else 0)
        params += b"\0" * ((-len(params)) % 64)
        st["mod"].launch(f"rc{mt}", 8 * _b7__rt_RC_Q, 288, _b7__rt_rc_smem(mt), params)
        return out
    if _b7__rt_USE_RS and m <= _b7__rt_RS_MAX:
        params = _b7__rt__tmap_x3(x, m) + struct.pack("QQq", st["w_rs"].data_ptr(), out.data_ptr(), m)
        params += b"\0" * ((-len(params)) % 64)
        st["mod"].launch(f"rs{m}", 128, 256, _b7__rt_rs_smem(m), params)
        return out
    name, cfg = _b7__rt__cfg_for(m)
    tt, rr, kk, ks, ns, wk, wg, _ = cfg
    ntt = (m + tt - 1) // tt
    assert ntt * tt * 256 <= st["ws"].numel() and ntt * (256 // rr) <= st["cnt"].numel()
    params = _b7__rt__tmap_x(x, m, tt) + struct.pack("QQQQqq", st["lay"][(rr, kk, ks)].data_ptr(), out.data_ptr(),
                                             st["ws"].data_ptr(), st["cnt"].data_ptr(), m, ntt)
    params += b"\0" * ((-len(params)) % 64)
    grid = ntt * (256 // rr) * (8192 // kk)
    zk = st["zkeys"].get((name, ntt))
    if _b7__rt_FLAGS & 16 and zk is None:
        _b7__rt_FLAGS = 4
    if _b7__rt_FLAGS & 32 and zk is not None and st["esc"][(rr, kk, ks)][3] <= 32:
        c = st["cmp"]
        off, ent, dl, _ = st["esc"][(rr, kk, ks)]
        b2 = (c["base"] << 7) | (c["base"] << 23)
        p2 = _b7__rt__tmap_x(x, m, tt) + struct.pack("QQQQqqQQQQqQ", st["w_cmp"][(rr, kk, ks)].data_ptr(), out.data_ptr(),
                                             st["ws"].data_ptr(), st["zc"].data_ptr() + 4 * 4096 * zk, m, ntt,
                                             x.data_ptr(), off.data_ptr(), ent.data_ptr(), dl.data_ptr(), b2,
                                             _b7__rt_TRACE.data_ptr() if _b7__rt_TRACE is not None else 0)
        p2 += b"\0" * ((-len(p2)) % 64)
        st["mod"].launch(f"{name}_f48", 4 + grid, 32 * (wk * wg + 1), _b7__rt_smem_bytes(cfg), p2)
        return out
    _b7__rt_FLAGS &= ~32
    if _b7__rt_FLAGS & 16:
        tmb = _b7__rt__tmap_x3s(x, m, tt) if _b7__rt_FLAGS & 64 else _b7__rt__tmap_x(x, m, tt)
        p2 = tmb + struct.pack("QQQQqqQQQQqQ", st["lay"][(rr, kk, ks)].data_ptr(), out.data_ptr(),
                                             st["ws"].data_ptr(), st["zc"].data_ptr() + 4 * 4096 * zk, m, ntt,
                                             0, 0, 0, 0, _b7__rt_DELAY_NS, _b7__rt_TRACE.data_ptr() if _b7__rt_TRACE is not None else 0)
        p2 += b"\0" * ((-len(p2)) % 64)
        st["mod"].launch(f"{name}_f{_b7__rt_FLAGS & 464}", 4 + grid, 32 * (wk * wg + 1), _b7__rt_smem_bytes(cfg), p2)
        return out
    if _b7__rt_FLAGS & 8:   # ablation: red.add into out with no zeroing / ordering (wrong values, timing only)
        st["mod"].launch(f"{name}_f8", grid, 32 * (wk * wg + 1), _b7__rt_smem_bytes(cfg), params)
        return out
    if _b7__rt_FLAGS & 4:
        st["mod"].launch("rzero", 16, 256, 0, struct.pack("Qq", out.data_ptr(), m * 64))
        st["mod"].launch_pdl(f"{name}_f{_b7__rt_FLAGS}", grid, 32 * (wk * wg + 1), _b7__rt_smem_bytes(cfg), params)
        return out
    st["mod"].launch(f"{name}_f{_b7__rt_FLAGS}", grid, 32 * (wk * wg + 1), _b7__rt_smem_bytes(cfg), params)
    return out


# ======================================================================== component: GV3 (M=1)
"""GV3: many-CTA GEMV for M == 1 on 12-bit lossless weights (Codex's GEMV structure, compressed).

One CTA per R rows, full K, thousands of CTAs: the hardware overlaps each CTA's start and reduction with the
others' streaming. Row blob = [lo bytes: K][codes: K/2][escape slots: E x u32], so a row's escapes arrive with its
weights (GV lost on serial cold loads of an offset table and then the escape list).
  lo/codes: pair p (weights 16p..16p+15) = lo bytes [16p, 16p+16), codes [8p, 8p+8) (decode12 order, see gv.py)
  escape slot: (k << 16) | true bf16 bits; the escaped weight is stored as code 0 / lo 0, which decodes to
  2^(2H-127) ("wdec"); out += x[k] * (true - wdec). Unused slots hold (0, wdec) and add 0.
x is read straight from global (L1-cached: the CTAs on an SM share it).
"""
import struct

import torch

_b7__gv_SRC = r"""
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
struct P {
  const u8* w; const u16* x; u16* out;
  long long N;
  u32 B4, E, RB, wdec;
  u32 M, pad;
};
__device__ __forceinline__ uint4 ldg128(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ uint2 ldg64(const void* p) {
  uint2 v; asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p)); return v;
}
__device__ __forceinline__ u32 ldg32(const void* p) {
  u32 v; asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v;
}
__device__ __forceinline__ uint4 ldx128(const void* p) {
  uint4 v; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ u16 ldx16(const void* p) {
  u16 v; asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(v) : "l"(p)); return v;
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 s) { u32 r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s)); return r; }
__device__ __forceinline__ uint4 decode12(uint2 lo, u32 nw, u32 B4) {
  const u32 hiA = ((nw << 4) & 0x80808080u) | ((nw & 0x07070707u) + B4);
  const u32 hiB = (nw & 0x80808080u) | (((nw >> 4) & 0x07070707u) + B4);
  uint4 o;
  o.x = prmt(lo.x, hiA, 0x5140); o.y = prmt(lo.x, hiA, 0x7362);
  o.z = prmt(lo.y, hiB, 0x5140); o.w = prmt(lo.y, hiB, 0x7362);
  return o;
}
__device__ __forceinline__ float blo(u32 v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bhi(u32 v) { return __uint_as_float(v & 0xFFFF0000u); }

template <int R, int NT, int K>
__device__ __forceinline__ void gv3_body(const P& p) {
  constexpr int NP = K / 16;            // weight pairs (16 weights) per row
  constexpr int PPT = NP / NT;          // pairs per thread
  static_assert(NP % NT == 0, "pairs divide threads");
  constexpr int NW = NT / 32;
  __shared__ float red[NW][R];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const long long row0 = (long long)blockIdx.x * R;
  const u8* wb = p.w + row0 * p.RB;
  uint4 lo[R][PPT];
  uint2 cd[R][PPT];
#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int j = 0; j < PPT; ++j) {
      const int pr = j * NT + tid;
      lo[r][j] = ldg128(wb + r * p.RB + pr * 16);
      cd[r][j] = ldg64(wb + r * p.RB + K + pr * 8);
    }
  }
  const int ne = R * (int)p.E;
  const bool hasesc = (wid == 0) && (lane < ne);
  const int er = hasesc ? lane / (int)p.E : 0;
  u32 es = 0;
  if (hasesc) es = ldg32(wb + er * p.RB + K + K / 2 + (lane - er * (int)p.E) * 4);
  uint4 xa[PPT], xb[PPT];
#pragma unroll
  for (int j = 0; j < PPT; ++j) {
    const int pr = j * NT + tid;
    xa[j] = ldx128(p.x + pr * 16);
    xb[j] = ldx128(p.x + pr * 16 + 8);
  }
  float acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = 0.f;
#pragma unroll
  for (int j = 0; j < PPT; ++j) {
    const float xf[16] = {blo(xa[j].x), bhi(xa[j].x), blo(xa[j].y), bhi(xa[j].y), blo(xa[j].z), bhi(xa[j].z), blo(xa[j].w), bhi(xa[j].w),
                          blo(xb[j].x), bhi(xb[j].x), blo(xb[j].y), bhi(xb[j].y), blo(xb[j].z), bhi(xb[j].z), blo(xb[j].w), bhi(xb[j].w)};
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const uint4 w0 = decode12(make_uint2(lo[r][j].x, lo[r][j].y), cd[r][j].x, p.B4);
      const uint4 w1 = decode12(make_uint2(lo[r][j].z, lo[r][j].w), cd[r][j].y, p.B4);
      float a = acc[r];
      a = fmaf(blo(w0.x), xf[0], a); a = fmaf(bhi(w0.x), xf[1], a); a = fmaf(blo(w0.y), xf[2], a); a = fmaf(bhi(w0.y), xf[3], a);
      a = fmaf(blo(w0.z), xf[4], a); a = fmaf(bhi(w0.z), xf[5], a); a = fmaf(blo(w0.w), xf[6], a); a = fmaf(bhi(w0.w), xf[7], a);
      a = fmaf(blo(w1.x), xf[8], a); a = fmaf(bhi(w1.x), xf[9], a); a = fmaf(blo(w1.y), xf[10], a); a = fmaf(bhi(w1.y), xf[11], a);
      a = fmaf(blo(w1.z), xf[12], a); a = fmaf(bhi(w1.z), xf[13], a); a = fmaf(blo(w1.w), xf[14], a); a = fmaf(bhi(w1.w), xf[15], a);
      acc[r] = a;
    }
  }
  if (hasesc) {
    const float xv = __uint_as_float(((u32)ldx16(p.x + (es >> 16))) << 16);
    const float c = xv * (__uint_as_float(es << 16) - __uint_as_float(p.wdec));
#pragma unroll
    for (int r = 0; r < R; ++r) if (r == er) acc[r] += c;
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    float v = acc[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (NW == 1) {
      if (lane == 0) { u16 ob; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(ob) : "f"(v)); p.out[row0 + r] = ob; }
    } else if (lane == 0) {
      red[wid][r] = v;
    }
  }
  if (NW > 1) {
    __syncthreads();
    if (tid < R) {
      float s = 0.f;
#pragma unroll
      for (int w = 0; w < NW; ++w) s += red[w][tid];
      u16 ob; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(ob) : "f"(s));
      p.out[row0 + tid] = ob;
    }
  }
}

// M tokens (M <= MT), FP32 FMA. W decoded once per pair to FP32, reused across tokens; x per token from L1.
template <int R, int NT, int K, int MT>
__device__ __forceinline__ void gv3m_body(const P& p) {
  constexpr int NP = K / 16, PPT = NP / NT;
  static_assert(NP % NT == 0, "pairs divide threads");
  constexpr int NW = NT / 32;
  __shared__ float red[NW][R * MT];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const int M = (int)p.M;
  const long long row0 = (long long)blockIdx.x * R;
  const u8* wb = p.w + row0 * p.RB;
  uint4 lo[R][PPT];
  uint2 cd[R][PPT];
#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int j = 0; j < PPT; ++j) {
      const int pr = j * NT + tid;
      lo[r][j] = ldg128(wb + r * p.RB + pr * 16);
      cd[r][j] = ldg64(wb + r * p.RB + K + pr * 8);
    }
  }
  const int ne = R * (int)p.E;
  const bool hasesc = (wid == 0) && (lane < ne);
  const int er = hasesc ? lane / (int)p.E : 0;
  u32 es = 0;
  if (hasesc) es = ldg32(wb + er * p.RB + K + K / 2 + (lane - er * (int)p.E) * 4);
  float acc[R][MT];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int t = 0; t < MT; ++t) acc[r][t] = 0.f;
#pragma unroll
  for (int j = 0; j < PPT; ++j) {
    const int pr = j * NT + tid;
    float wf[R][16];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const uint4 w0 = decode12(make_uint2(lo[r][j].x, lo[r][j].y), cd[r][j].x, p.B4);
      const uint4 w1 = decode12(make_uint2(lo[r][j].z, lo[r][j].w), cd[r][j].y, p.B4);
      wf[r][0] = blo(w0.x); wf[r][1] = bhi(w0.x); wf[r][2] = blo(w0.y); wf[r][3] = bhi(w0.y);
      wf[r][4] = blo(w0.z); wf[r][5] = bhi(w0.z); wf[r][6] = blo(w0.w); wf[r][7] = bhi(w0.w);
      wf[r][8] = blo(w1.x); wf[r][9] = bhi(w1.x); wf[r][10] = blo(w1.y); wf[r][11] = bhi(w1.y);
      wf[r][12] = blo(w1.z); wf[r][13] = bhi(w1.z); wf[r][14] = blo(w1.w); wf[r][15] = bhi(w1.w);
    }
#pragma unroll
    for (int t = 0; t < MT; ++t) {
      if (t < M) {
        const uint4 xa = ldx128(p.x + (size_t)t * K + pr * 16);
        const uint4 xb = ldx128(p.x + (size_t)t * K + pr * 16 + 8);
        const float xf[16] = {blo(xa.x), bhi(xa.x), blo(xa.y), bhi(xa.y), blo(xa.z), bhi(xa.z), blo(xa.w), bhi(xa.w),
                              blo(xb.x), bhi(xb.x), blo(xb.y), bhi(xb.y), blo(xb.z), bhi(xb.z), blo(xb.w), bhi(xb.w)};
#pragma unroll
        for (int r = 0; r < R; ++r) {
          float a = acc[r][t];
#pragma unroll
          for (int i = 0; i < 16; ++i) a = fmaf(wf[r][i], xf[i], a);
          acc[r][t] = a;
        }
      }
    }
  }
  if (hasesc) {
    const float dl = __uint_as_float(es << 16) - __uint_as_float(p.wdec);
#pragma unroll
    for (int t = 0; t < MT; ++t) {
      if (t < M) {
        const float xv = __uint_as_float(((u32)ldx16(p.x + (size_t)t * K + (es >> 16))) << 16);
#pragma unroll
        for (int r = 0; r < R; ++r) if (r == er) acc[r][t] = fmaf(xv, dl, acc[r][t]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int t = 0; t < MT; ++t) {
      float v = acc[r][t];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) red[wid][r * MT + t] = v;
    }
  }
  __syncthreads();
  if (tid < R * M) {
    const int r = tid / M, t = tid - r * M;
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) s += red[w][r * MT + t];
    u16 ob; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(ob) : "f"(s));
    p.out[(size_t)t * p.N + row0 + r] = ob;
  }
}
"""

_b7__gv_LAYER_K = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpslice": 4096}
# layer -> (R rows per CTA, threads) default
_b7__gv_CFG = {"qkv_gate": (1, 256), "o": (4, 64), "pooler": (1, 256), "unpslice": (1, 128)}


def _b7__gv_kname(layer, r, nt, mb=0):
    return f"gv3_{layer}_{r}_{nt}" + (f"_{mb}" if mb else "")


def _b7__gv_mkname(layer, r, nt):
    return f"gv3m_{layer}_{r}_{nt}"


_b7__gv_MT = 8


def _b7__gv_source(cfgs=None, mcfgs=()):
    """cfgs: iterable of (layer, R, NT) for M == 1 (default CFG); mcfgs: same for the M <= 8 variant"""
    cfgs = [(l, r, nt) for l, (r, nt) in _b7__gv_CFG.items()] if cfgs is None else cfgs
    s = _b7__gv_SRC
    for c in sorted(set(tuple(c) + (0,) * (4 - len(c)) for c in cfgs)):
        layer, r, nt, mb = c
        lb = f"{nt}, {mb}" if mb else f"{nt}"
        s += (f'extern "C" __global__ void __launch_bounds__({lb}) {_b7__gv_kname(layer, r, nt, mb)}(const __grid_constant__ P p) '
              f'{{ gv3_body<{r},{nt},{_b7__gv_LAYER_K[layer]}>(p); }}\n')
    for layer, r, nt in sorted(set(mcfgs)):
        s += (f'extern "C" __global__ void __launch_bounds__({nt}) {_b7__gv_mkname(layer, r, nt)}(const __grid_constant__ P p) '
              f'{{ gv3m_body<{r},{nt},{_b7__gv_LAYER_K[layer]},{_b7__gv_MT}>(p); }}\n')
    return s


def _b7__gv_compress(w):
    u = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e = u >> 15, (u >> 7) & 0xFF
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    best, H = -1, 0
    for h in range(0, 121):
        inside = int(cs[2 * h + 16] - cs[2 * h])
        if inside > best:
            best, H = inside, h
    idx = (e >> 1) - H
    esc = (idx < 0) | (idx > 7)
    code = torch.where(esc, torch.zeros_like(e), (s << 3) | idx)
    lo = torch.where(esc, torch.zeros_like(u), u & 0xFF)
    return code, lo, H, esc, u


def _b7__gv_prepare_layer(w, emax=64):
    """-> dict, or None if some row has more than emax escapes (caller keeps its BF16 path)"""
    n, k = w.shape
    code, lo, H, esc, u = _b7__gv_compress(w)
    cnt = esc.sum(1)
    mx = int(cnt.max())
    if mx > emax:
        return None
    E = max(4, (mx + 3) // 4 * 4)
    nch = k // 8
    lo8 = lo.to(torch.uint8).view(n, k)
    cc = code.to(torch.int32).view(n, nch, 8)
    cw = (cc[..., 0:4] | (cc[..., 4:8] << 4)).to(torch.uint8).view(n, k // 2)
    wdec = H << 8                                                         # bf16 bits of a decoded escape
    slots = torch.full((n, E), wdec, dtype=torch.int64, device=w.device)
    nn, kk = esc.nonzero(as_tuple=True)                                   # row-major order
    if len(nn):
        first = torch.zeros(n + 1, dtype=torch.int64, device=w.device)
        first[1:] = torch.cumsum(cnt, 0)
        rank = torch.arange(len(nn), device=w.device) - first[nn]
        slots[nn, rank] = (kk.to(torch.int64) << 16) | u[nn, kk].to(torch.int64)
    s32 = slots.to(torch.int32)                                           # k < 2^15: always < 2^31
    blob = torch.cat([lo8, cw, s32.contiguous().view(torch.uint8).view(n, 4 * E)], 1).contiguous()
    rb = blob.shape[1]
    assert rb % 16 == 0
    return {"w": blob.view(-1), "B4": H * 0x01010101, "E": E, "RB": rb, "wdec": wdec << 16, "n": n, "k": k,
            "n_esc": int(len(nn)), "emax": mx}


def _b7__gv_launch(mod, lname, ent, x, out, cfg=None):
    cfg = tuple(cfg or _b7__gv_CFG[lname])
    r, nt = cfg[0], cfg[1]
    mb = cfg[2] if len(cfg) > 2 else 0
    n = out.shape[-1]
    assert r * ent["E"] <= 32 and n % r == 0
    m = x.shape[0]
    args = struct.pack("<QQQqIIIIII", ent["w"].data_ptr(), x.data_ptr(), out.data_ptr(), n, ent["B4"], ent["E"],
                       ent["RB"], ent["wdec"], m, 0)
    if m == 1:
        mod.launch(_b7__gv_kname(lname, r, nt, mb), n // r, nt, 0, args)
    else:
        assert m <= _b7__gv_MT
        mod.launch(_b7__gv_mkname(lname, r, nt), n // r, nt, 0, args)


# ======================================================================== component: GM (o, M<=8)
"""GM: many-CTA tensor-core GEMV for small M (<= 32) on 12-bit lossless weights, for layers whose x fits in L1 (o).

CTA = one 16-row tile x full K; NW warps split K; mma.m16n8k16 with W as A (16 rows) and x as B (n8 token tiles),
x fragments read from global (L1-cached: every CTA on an SM reads the same x). Everything a warp streams is
issued at its start; the hardware overlaps CTAs (GV3's structure, with tensor cores for the token dimension).

k order inside a 32-k block (two mma k16 steps): lane t takes k = 8t..8t+7; step 0 uses 8t..8t+3, step 1 uses
8t+4..8t+7 (a consistent permutation of k for A and B). So one 16 B load of x[tok][32p + 8t ..] feeds both steps,
and the A fragment for (step, lane g/t) is the 8 weights
  [W[g][k0], W[g][k0+1], W[g+8][k0], W[g+8][k0+1], W[g][k0+2], W[g][k0+3], W[g+8][k0+2], W[g+8][k0+3]]
(k0 = 32p + 8t + 4*step) -> decode12 order (o.x..o.w = a0..a3).
Tile blob: lo [S/2 step pairs][32 lanes][16 B], codes [S/4 step quads][32 lanes][16 B], slots [16 rows][E] u32.
Escapes (out-of-window exponents) as in GV3: stored as code 0 / lo 0 (decodes to wdec), corrected after the
reduction from per-row (k, true bits) slots.
"""
import struct

import torch


_b7__gm_SRC = r"""
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
struct P {
  const u8* w; const u16* x; u16* out;
  long long N;
  u32 B4, E, TB, wdec;
  u32 M, pad;
};
__device__ __forceinline__ uint4 ldg128(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ uint4 ldx128(const void* p) {
  uint4 v; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ u32 ldx32(const void* p) {
  u32 v; asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v;
}
__device__ __forceinline__ u16 ldx16(const void* p) {
  u16 v; asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(v) : "l"(p)); return v;
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 s) { u32 r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s)); return r; }
__device__ __forceinline__ uint4 decode12(uint2 lo, u32 nw, u32 B4) {
  const u32 hiA = ((nw << 4) & 0x80808080u) | ((nw & 0x07070707u) + B4);
  const u32 hiB = (nw & 0x80808080u) | (((nw >> 4) & 0x07070707u) + B4);
  uint4 o;
  o.x = prmt(lo.x, hiA, 0x5140); o.y = prmt(lo.x, hiA, 0x7362);
  o.z = prmt(lo.y, hiB, 0x5140); o.w = prmt(lo.y, hiB, 0x7362);
  return o;
}
__device__ __forceinline__ void mma16816(float* d, const uint4 a, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

// NW warps split the S = K/16 steps of one 16-row tile; NT8 n8 token tiles (M <= 8*NT8)
template <int NW, int K, int NT8, int EMAX>   // EMAX * 16 <= NW * 32
__device__ __forceinline__ void gm_body(const P& p) {
  constexpr int S = K / 16, SW = S / NW;          // steps per warp
  static_assert(SW % 4 == 0, "steps per warp: multiple of 4");
  constexpr int NPAIR = SW / 2, NQUAD = SW / 4;
  __shared__ float red[NW][16][8 * NT8 + 1];
  __shared__ u32 slot[16 * EMAX];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int M = (int)p.M;
  const u8* tb = p.w + (size_t)blockIdx.x * p.TB;
  const u8* lob = tb + (size_t)(wid * NPAIR) * 512 + lane * 16;
  const u8* cdb = tb + (size_t)(S / 2) * 512 + (size_t)(wid * NQUAD) * 512 + lane * 16;
  uint4 lo[NPAIR], cd[NQUAD];
#pragma unroll
  for (int i = 0; i < NPAIR; ++i) lo[i] = ldg128(lob + i * 512);
#pragma unroll
  for (int i = 0; i < NQUAD; ++i) cd[i] = ldg128(cdb + i * 512);
  const int ns = 16 * (int)p.E;
  const u8* sb = tb + (size_t)(S / 2) * 512 + (size_t)(S / 4) * 512;
  // escape slots: into registers now (no wait), into smem after the main loop
  u32 myslot = 0;
  if (tid < ns) myslot = ldx32(sb + tid * 4);
  float acc[NT8][4];
#pragma unroll
  for (int n = 0; n < NT8; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
  const int kw = wid * SW * 16;                    // this warp's first k
  uint4 xb[NPAIR][NT8];                            // all of this warp's x fragments, issued up front
#pragma unroll
  for (int pi = 0; pi < NPAIR; ++pi) {
#pragma unroll
    for (int n = 0; n < NT8; ++n) {
      const int tok = n * 8 + g;
      xb[pi][n] = tok < M ? ldx128(p.x + (size_t)tok * K + kw + pi * 32 + t * 8) : make_uint4(0, 0, 0, 0);
    }
  }
#pragma unroll
  for (int pi = 0; pi < NPAIR; ++pi) {
    // codes: quad q holds steps 4q..4q+3 -> u32 per step; pair pi = steps 2pi, 2pi+1 -> quad pi/2, words 2*(pi&1), +1
    const uint4 cq = cd[pi >> 1];
    const u32 cs0 = (pi & 1) ? cq.z : cq.x, cs1 = (pi & 1) ? cq.w : cq.y;
    const uint4 a0 = decode12(make_uint2(lo[pi].x, lo[pi].y), cs0, p.B4);
    const uint4 a1 = decode12(make_uint2(lo[pi].z, lo[pi].w), cs1, p.B4);
#pragma unroll
    for (int n = 0; n < NT8; ++n) {
      mma16816(acc[n], a0, xb[pi][n].x, xb[pi][n].y);
      mma16816(acc[n], a1, xb[pi][n].z, xb[pi][n].w);
    }
  }
  if (tid < ns) slot[tid] = myslot;
#pragma unroll
  for (int n = 0; n < NT8; ++n) {
    red[wid][g][n * 8 + 2 * t] = acc[n][0];
    red[wid][g][n * 8 + 2 * t + 1] = acc[n][1];
    red[wid][g + 8][n * 8 + 2 * t] = acc[n][2];
    red[wid][g + 8][n * 8 + 2 * t + 1] = acc[n][3];
  }
  __syncthreads();
  const float wdec = __uint_as_float(p.wdec);
  const long long row0 = (long long)blockIdx.x * 16;
  for (int i = tid; i < 16 * M; i += NW * 32) {
    const int tok = i >> 4, r = i & 15;
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) s += red[w][r][tok];
    for (int e = 0; e < (int)p.E; ++e) {
      const u32 es = slot[r * p.E + e];
      const float dl = __uint_as_float(es << 16) - wdec;
      if (dl != 0.f) s = fmaf(__uint_as_float(((u32)ldx16(p.x + (size_t)tok * K + (es >> 16))) << 16), dl, s);
    }
    u16 ob; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(ob) : "f"(s));
    p.out[(size_t)tok * p.N + row0 + r] = ob;
  }
}
"""

_b7__gm_LAYER_K = {"o": 1024}
_b7__gm_EMAX = 16


def _b7__gm_kname(layer, nw, nt8, mb=0):
    return f"gm_{layer}_{nw}_{nt8}_{mb}"


def _b7__gm_source(cfgs):
    """cfgs: iterable of (layer, NW, NT8[, min blocks per SM])"""
    s = _b7__gm_SRC
    for c in sorted(set(tuple(c) + (0,) * (4 - len(c)) for c in cfgs)):
        layer, nw, nt8, mb = c
        lb = f"{nw * 32}, {mb}" if mb else f"{nw * 32}"
        s += (f'extern "C" __global__ void __launch_bounds__({lb}) {_b7__gm_kname(layer, nw, nt8, mb)}(const __grid_constant__ P p) '
              f'{{ gm_body<{nw},{_b7__gm_LAYER_K[layer]},{nt8},{min(_b7__gm_EMAX, nw * 2)}>(p); }}\n')
    return s


def _b7__gm_prepare_layer(w):
    """-> dict, or None if some row has more than EMAX escapes"""
    n, k = w.shape
    assert n % 16 == 0 and k % 64 == 0
    code, lo, H, esc, u = _b7__gv_compress(w)
    cnt = esc.sum(1)
    mx = int(cnt.max())
    if mx > _b7__gm_EMAX:
        return None
    E = max(1, mx)
    S = k // 16
    T = n // 16
    # weight order per (tile, step, lane): 8 weights [g k0, g k0+1, g+8 k0, g+8 k0+1, g k0+2, g k0+3, g+8 k0+2, g+8 k0+3]
    # k0 = 32*(step//2) + 8*t + 4*(step%2)
    dev = w.device
    st = torch.arange(S, device=dev)
    ln = torch.arange(32, device=dev)
    g, t = ln // 4, ln % 4
    k0 = (32 * (st // 2))[:, None] + (8 * t)[None, :] + (4 * (st % 2))[:, None]          # [S, 32]
    rr = torch.stack([g, g, g + 8, g + 8, g, g, g + 8, g + 8], 1)                           # [32, 8]
    kk = k0[:, :, None] + torch.tensor([0, 1, 0, 1, 2, 3, 2, 3], device=dev)[None, None, :]  # [S, 32, 8]
    rows = torch.arange(T, device=dev)[:, None, None, None] * 16 + rr[None, None, :, :]      # [T, 1, 32, 8]
    rows = rows.expand(T, S, 32, 8)
    kidx = kk[None].expand(T, S, 32, 8)
    lo_g = lo[rows, kidx].to(torch.uint8)                                                  # [T, S, 32, 8]
    cc = code[rows, kidx].to(torch.int32)
    cw = (cc[..., 0:4] | (cc[..., 4:8] << 4)).to(torch.uint8)                               # [T, S, 32, 4]
    # lo: [T][S/2 pairs][32][16 B] = steps 2p, 2p+1 for the lane
    lo_p = lo_g.view(T, S // 2, 2, 32, 8).permute(0, 1, 3, 2, 4).reshape(T, S // 2 * 512)
    cd_q = cw.view(T, S // 4, 4, 32, 4).permute(0, 1, 3, 2, 4).reshape(T, S // 4 * 512)
    wdec = H << 8
    slots = torch.full((n, E), wdec, dtype=torch.int64, device=dev)
    nn, kq = esc.nonzero(as_tuple=True)
    if len(nn):
        first = torch.zeros(n + 1, dtype=torch.int64, device=dev)
        first[1:] = torch.cumsum(cnt, 0)
        rank = torch.arange(len(nn), device=dev) - first[nn]
        slots[nn, rank] = (kq.to(torch.int64) << 16) | u[nn, kq].to(torch.int64)
    sl = slots.to(torch.int32).view(T, 16 * E).contiguous().view(torch.uint8).view(T, 64 * E)
    blob = torch.cat([lo_p, cd_q, sl], 1)
    pad = (-blob.shape[1]) % 128
    if pad:
        blob = torch.cat([blob, torch.zeros((T, pad), dtype=torch.uint8, device=dev)], 1)
    blob = blob.contiguous()
    return {"w": blob.view(-1), "B4": H * 0x01010101, "E": E, "TB": blob.shape[1], "wdec": wdec << 16, "n": n,
            "k": k, "n_esc": int(len(nn)), "emax": mx}


def _b7__gm_launch(mod, layer, ent, x, out, nw=4, mb=0):
    m, n = x.shape[0], out.shape[-1]
    nt8 = (m + 7) // 8
    assert nt8 <= 4 and n % 16 == 0 and 16 * ent["E"] <= nw * 32 and ent["E"] <= min(_b7__gm_EMAX, nw * 2)
    args = struct.pack("<QQQqIIIIII", ent["w"].data_ptr(), x.data_ptr(), out.data_ptr(), n, ent["B4"], ent["E"],
                       ent["TB"], ent["wdec"], m, 0)
    mod.launch(_b7__gm_kname(layer, nw, nt8, mb), n // 16, nw * 32, 0, args)




# ======================================================================== dispatch
# Measured choices (claude/out/*): K1s = 12-bit lossless weights, register streaming, M <= 16.
_b7__K1_USE = {
    "qkv_gate": lambda m: False,      # K1t C=4 below
    "pooler": lambda m: 1 <= m <= 16,
    "unpooler": lambda m: 9 <= m <= 16,
    "o": lambda m: False,
}
# library GEMMs that beat Codex's picks (claude/out/lib3.log): flashinfer cuDNN backend, plain torch.mm
_b7__LIB = {}
# exhaustive cuBLASLt search winners (claude/lt_tune.py, claude/lt_128_256.json): (m, layer) -> (layout, settings)
_b7__LT = {(128, 'qkv_gate'): ('tn', [66, 92, 1, 0, 0, 0, 35, 0, 5, 0, 130101]), (128, 'o'): ('tn', [66, 18, 1, 0, 0, 0, 35, 0, 5, 0, 130101]), (128, 'pooler'): ('nn', [66, 16, 1, 0, 0, 1, 35, 0, 2, 0, 130101]), (128, 'unpooler'): ('nn', [66, 18, 1, 0, 0, 0, 35, 0, 5, 0, 130101]), (256, 'qkv_gate'): ('tn', [66, 124, 1, 0, 0, 0, 35, 0, 2, 0, 130101]), (256, 'o'): ('nn', [66, 21, 1, 0, 0, 0, 35, 0, 5, 0, 130101]), (256, 'pooler'): ('nn', [66, 18, 1, 0, 0, 0, 35, 0, 2, 0, 130101]), (256, 'unpooler'): ('nn', [66, 21, 1, 0, 0, 0, 35, 0, 2, 0, 130101])}
# GV3: many-CTA 12-bit GEMV at M == 1 (claude/gv3.py, claude/out/gv3s.log); per layer, configs in preference order
# (rows per CTA x threads); escape lanes need rows * E <= 32
_b7__GV_PREF = {"qkv_gate": [(2, 512), (2, 256), (1, 256)], "o": [(8, 64), (4, 64), (2, 64), (1, 64)],
            "pooler": [(2, 512), (1, 512)], "unpslice": [(1, 256, 8), (1, 256)]}
# GM: many-CTA tensor-core 12-bit GEMV for o at 2 <= M <= 8 (claude/gm.py, claude/out/gm3.log): (NW warps, min blocks)
_b7__GM_O = (4, 6)
# qkv_gate at M <= 16: K1t, 4-CTA clusters (120 SMs), 8-row groups (claude/out/q4.log)
_b7__k1t_CFG["qkv_gate"] = (4, 30, 5, 2, 8, 3)
_b7__k1t_HALF_CFGS.discard("qkv_gate")
_b7__K1T_USE = {
    "qkv_gate": lambda m: 2 <= m <= 16,
    "pooler": lambda m: 17 <= m <= 32,
    "unpooler": lambda m: 17 <= m <= 32,
}


def _b7_prepare(weights):
    st = {"cx": _b7__cx_prepare(weights), "rt": _b7__rt_prepare_router(weights)}
    k1 = {}
    for name in ("pooler", "unpooler"):
        k1[name] = _b7__k1_prepare_layer(weights[name], name, 1)
    k1["pooler1"] = _b7__k1_prepare_layer(weights["pooler"], "pooler1", 1)
    st["k1"] = k1
    st["k1mod"] = _b7__cu_Module(_b7__k1_source(layers=("pooler", "pooler1", "unpooler")), "k1s.cu")
    k1t = {}
    for v, base in (("pooler32", "pooler"), ("unpooler32", "unpooler"), ("qkv_gate", "qkv_gate")):
        k1t[v] = _b7__k1t_prepare_layer(weights[base], v, 1)
    st["k1t"] = k1t
    st["k1tmod"] = _b7__cu_Module(_b7__k1t_source(layers=("pooler32", "unpooler32", "qkv_gate"), tokbs=(8, 16, 32)), "k1t.cu")
    gv, gcfg = {}, {}
    for lname, src in (("qkv_gate", "qkv_gate"), ("o", "o"), ("pooler", "pooler"), ("unpslice", "unpooler")):
        w = weights[src][:1024].contiguous() if lname == "unpslice" else weights[src]
        ent = _b7__gv_prepare_layer(w)
        if ent is None:
            continue
        for c in _b7__GV_PREF[lname]:
            if c[0] * ent["E"] <= 32:
                gv[lname], gcfg[lname] = ent, tuple(c)
                break
    st["gv"], st["gvcfg"] = gv, gcfg
    st["gvmod"] = _b7__cu_Module(_b7__gv_source([(l,) + c for l, c in gcfg.items()]), "gv3.cu") if gcfg else None
    ent = _b7__gm_prepare_layer(weights["o"])
    nw, mb = _b7__GM_O
    if ent is not None and 16 * ent["E"] <= nw * 32 and ent["E"] <= min(_b7__gm_EMAX, nw * 2):
        st["gm_o"] = ent
        st["gmmod"] = _b7__cu_Module(_b7__gm_source([("o", nw, 1, mb)]), "gm.cu")
    else:
        st["gm_o"] = None
    st["wt"] = {k: weights[k].t() for k in ("qkv_gate", "o", "pooler", "unpooler")}
    for (layer, m), be in _b7__LIB.items():   # build/cached library plans outside the timed calls
        xw = torch.randn((m, weights[layer].shape[1]), device=weights[layer].device).to(torch.bfloat16)
        for _ in range(2):
            _b7__lib_mm(xw, st, layer, weights[layer].shape[0], be)
    torch.cuda.synchronize()
    return st


def _b7__lib_mm(x, st, layer, n, be):
    import flashinfer
    return flashinfer.mm_bf16(x, st["wt"][layer][:, :n], backend=be)


def _b7_kernel_fn(data, state):
    x, cfg = data["x"], data["config"]
    layer, m, n = cfg["layer"], cfg["m"], cfg["n"]
    if layer == "router":
        return _b7__rt_run_router(x, state["rt"])
    if m == 1:
        lname = "unpslice" if layer == "unpooler" else layer
        ent = state["gv"].get(lname)
        if ent is not None and n == ent["n"] and x.is_contiguous() and x.data_ptr() % 16 == 0:
            out = torch.empty((1, n), dtype=torch.bfloat16, device=x.device)
            _b7__gv_launch(state["gvmod"], lname, ent, x, out, state["gvcfg"][lname])
            return out
    if layer == "o" and 2 <= m <= 8 and state["gm_o"] is not None and x.is_contiguous() and x.data_ptr() % 16 == 0:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _b7__gm_launch(state["gmmod"], "o", state["gm_o"], x, out, *_b7__GM_O)
        return out
    be = _b7__LIB.get((layer, m))
    if be is not None:
        return _b7__lib_mm(x, state, layer, n, be)
    lt = _b7__LT.get((m, layer))
    if lt is not None:
        return _b7__cx_lt_run(x, state["cx"]["lt"][layer], n, ("lt", lt[0], *lt[1]))
    if _b7__K1_USE[layer](m) and x.is_contiguous() and x.data_ptr() % 16 == 0:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _b7__k1_launch(state["k1mod"], layer, state["k1"][_b7__k1_cfg_key(layer, m, n)[0]], x, out, m, n, 1)
        return out
    if layer in _b7__K1T_USE and _b7__K1T_USE[layer](m) and x.is_contiguous() and x.data_ptr() % 16 == 0:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _b7__k1t_launch(state["k1tmod"], layer, state["k1t"][_b7__k1t_cfg_key(layer, m, n)[0]], x, out, m, n, 1)
        return out
    return _b7__cx_kernel_fn(data, state["cx"])


# ======================================================================== module G: G/gemv1.py
"""G / gemv1: M == 1 GEMV for qkv_gate, o, pooler and the unpooler slice (claude/t3/G, 2026-11-02).

    state = prepare_gemv1(weights)           # weights: the problem's dict (BF16 [N, K] per layer)
    out   = run_gemv1(x, config, state)      # x [1, K] bf16, config {"m": 1, "n", "k", "layer"} -> [1, n] bf16
    run_gemv1 falls back to torch for anything it does not handle (m != 1, router, misaligned x).

Two lossless weight formats (both store each weight as a "fake" bf16 = true * 2^(F - E_lo); the kernel accumulates
x * fake in FP32 - products exact, same rounding as the true weights - and multiplies each output by 2^(E_lo - F)):
  F11 (qkv_gate, pooler, unpooler slice): 8-exponent window, lo byte + 3-bit code (s, h) per weight, 32 weights ->
       8 lo words + 3 code words; escapes (~1.3%) in a per-row slot list (k << 16 | true bf16), see NOTES.md.
  F12 (o): 16-exponent window, lo byte + 4-bit nibble (s, h), escapes ~1e-4 in per-row slots.
Kernels:
  g311 (F11, FMA): CTA = 1 row x 256 threads (K/32), thread = one 32-weight group (k = c*K/4 + 8j + i), every load
       issued up front (min-blocks set so ptxas does not sink loads), escapes from the row's slot list.
  tc   (F12, tensor cores): CTA = 16-row tile x 8 warps (K slices), A = decoded weights, B column 0 = x, weight pointer
       baked into the kernel as a constant (the param bank is a cold miss after the judge's L2 flush).
"""
import ctypes
import struct

import torch

_g1_SRC = r"""
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
struct P {
  const u8* w; const u16* x; u16* out;
  u32 E, RB;
  float scale, invscale;
};
struct PX { const u16* x; u16* out; };
template <unsigned long long WA, unsigned EE, unsigned RBB, unsigned SC, unsigned ISC>
__device__ __forceinline__ P mkp(const PX& q) {
  P p; unsigned long long a; asm("mov.u64 %0, %1;" : "=l"(a) : "n"(WA)); p.w = (const u8*)a;
  p.x = q.x; p.out = q.out; p.E = EE; p.RB = RBB; p.scale = __uint_as_float(SC); p.invscale = __uint_as_float(ISC);
  return p;
}
__device__ __forceinline__ uint4 ldg128(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ u32 ldg32(const void* p) {
  u32 v; asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v;
}
__device__ __forceinline__ uint4 ldx128(const void* p) {
  uint4 v; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ u16 ldx16(const void* p) {
  u16 v; asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(v) : "l"(p)); return v;
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 s) { u32 r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s)); return r; }
__device__ __forceinline__ u32 rotr(u32 x, int s) { return __funnelshift_r(x, x, s); }
__device__ __forceinline__ float blo(u32 v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bhi(u32 v) { return __uint_as_float(v & 0xFFFF0000u); }
__device__ __forceinline__ u16 f2bf(float v) { u16 o; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(o) : "f"(v)); return o; }
__device__ __forceinline__ void mma16816(float* d, const uint4 a, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

// ------------------------------------------------------------------ F12 decode: 8 weights from 2 lo words + 1 nibble word
// hiA = (nw & 0x87878787) | 0x38383838 ; hiB = (rotr(nw, 4) & 0x87878787) | 0x38383838 ; 4 PRMT (3 ALU + 4 PRMT)
__device__ __forceinline__ u32 hi12(u32 x, u32 M87) {
  u32 r; asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(x), "r"(M87), "n"(0x38383838)); return r;
}
__device__ __forceinline__ uint4 decode12f(u32 L0, u32 L1, u32 nw, u32 M87) {
  const u32 hiA = hi12(nw, M87), hiB = hi12(rotr(nw, 4), M87);
  uint4 o;
  o.x = prmt(L0, hiA, 0x5140); o.y = prmt(L0, hiA, 0x7362);
  o.z = prmt(L1, hiB, 0x5140); o.w = prmt(L1, hiB, 0x7362);
  return o;
}
__device__ __forceinline__ float esc_delta12(u32 es, float invscale) {   // true * 2^(112-E_lo) - 2^-15
  return fmaf(__uint_as_float(es << 16), invscale, -__uint_as_float(0x38000000u));
}

// ------------------------------------------------------------------ F11 decode: 32 weights from 8 lo words + 3 code words
// X0=W0 X1=rotr(W0,3) X2=W1 X3=rotr(W1,3) X4=W2 X5=rotr(W2,3) X6=sel80(rotr(W0,6),rotr(W1,5)) X7=sel80(rotr(W0,7),rotr(W2,5))
// hi_g = (X_g & 0x83838383) | 0x3C3C3C3C ; bf16x2 = prmt(lo_g, hi_g, 0x5140 / 0x7362)   (33 ALU per 32 weights)
__device__ __forceinline__ u32 hi11(u32 x, u32 M83) {
  u32 r; asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(x), "r"(M83), "n"(0x3C3C3C3C)); return r;
}
__device__ __forceinline__ u32 sel80(u32 a, u32 b, u32 M80) {
  u32 r; asm("lop3.b32 %0, %1, %2, %3, 0xE4;" : "=r"(r) : "r"(a), "r"(b), "r"(M80)); return r;
}
__device__ __forceinline__ void dec11_hi(u32 W0, u32 W1, u32 W2, u32* h, u32 M83, u32 M80) {
  h[0] = hi11(W0, M83);          h[1] = hi11(rotr(W0, 3), M83);
  h[2] = hi11(W1, M83);          h[3] = hi11(rotr(W1, 3), M83);
  h[4] = hi11(W2, M83);          h[5] = hi11(rotr(W2, 3), M83);
  h[6] = hi11(sel80(rotr(W0, 6), rotr(W1, 5), M80), M83);
  h[7] = hi11(sel80(rotr(W0, 7), rotr(W2, 5), M80), M83);
}
__device__ __forceinline__ float esc_delta11(u32 es, float invscale) {   // true * 2^(120-E_lo) - 2^-7
  return fmaf(__uint_as_float(es << 16), invscale, -__uint_as_float(0x3C000000u));
}

// ------------------------------------------------------------------ TC: F12, tensor cores (o)
// CTA = one 16-row tile; warp w = K slice w (SW = K/16/NW mma steps); lane (g, t) A fragment per step:
// [W[g][k0], W[g][k0+1], W[g+8][k0], W[g+8][k0+1], W[g][k0+2], W[g][k0+3], W[g+8][k0+2], W[g+8][k0+3]],
// k0 = 32*(step/2) + 8t + 4*(step%2) (the same k permutation for x = B column 0, loaded by lanes g == 0).
// Tile blob: lo [S/2 pairs][32 lanes][16 B] | nibble words [S/4 quads][32 lanes][16 B] | slots [16 rows][E] u32.
template <int NW, int K, int EMAXT>
__device__ __forceinline__ void tc_body(const P& p) {
  constexpr int S = K / 16, SW = S / NW;
  static_assert(SW % 4 == 0, "steps per warp: multiple of 4");
  constexpr int NPAIR = SW / 2, NQUAD = SW / 4;
  __shared__ float red[NW][16];
  __shared__ float escc[16 * EMAXT];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const u32 M87 = 0x87878787u;
  const u8* tb = p.w + (size_t)blockIdx.x * p.RB;
  uint4 lo[NPAIR], cd[NQUAD];
#pragma unroll
  for (int i = 0; i < NPAIR; ++i) lo[i] = ldg128(tb + (size_t)(wid * NPAIR + i) * 512 + lane * 16);
#pragma unroll
  for (int i = 0; i < NQUAD; ++i) cd[i] = ldg128(tb + (size_t)(S / 2) * 512 + (size_t)(wid * NQUAD + i) * 512 + lane * 16);
  const int E = (int)p.E, ns = 16 * E;
  u32 es = 0;
  if (tid < ns) es = ldg32(tb + (size_t)(S / 2 + S / 4) * 512 + tid * 4);
  const int kw = wid * SW * 16;
  uint4 xb[NPAIR];
#pragma unroll
  for (int i = 0; i < NPAIR; ++i) xb[i] = (g == 0) ? ldx128(p.x + kw + i * 32 + t * 8) : make_uint4(0, 0, 0, 0);
  float ecv = 0.f;            // escape correction issued before the mma loop; filler slots (delta 0) skip the gather
  if (tid < ns) {
    const float dl = esc_delta12(es, p.invscale);
    if (dl != 0.f) ecv = __uint_as_float(((u32)ldx16(p.x + (es >> 16))) << 16) * dl;
  }
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int i = 0; i < NPAIR; ++i) {
    const uint4 cq = cd[i >> 1];
    const u32 c0 = (i & 1) ? cq.z : cq.x, c1 = (i & 1) ? cq.w : cq.y;
    mma16816(acc, decode12f(lo[i].x, lo[i].y, c0, M87), xb[i].x, xb[i].y);
    mma16816(acc, decode12f(lo[i].z, lo[i].w, c1, M87), xb[i].z, xb[i].w);
  }
  if (tid < ns) escc[tid] = ecv;
  if (t == 0) { red[wid][g] = acc[0]; red[wid][g + 8] = acc[2]; }
  __syncthreads();
  if (tid < 16) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) s += red[w][tid];
    for (int e = 0; e < E; ++e) s += escc[tid * E + e];
    p.out[(size_t)blockIdx.x * 16 + tid] = f2bf(s * p.scale);
  }
}

// ------------------------------------------------------------------ G311: F11, FMA, one row per CTA (qkv, pooler, unpslice)
// NT = K/32 threads; thread j takes k = c*K/4 + 8j + i (c = 0..3, i = 0..7) -> group weight 8c + i, so every warp load
// (lo halves, code planes, x quarters) is one contiguous block. Row blob: lo [2][K/32][16 B] | W0[K/32] W1 W2 (u32 planes)
// | slots [E] u32 ((k << 16) | true bf16; unused slots carry delta 0).
template <int R, int K, int SPT>
__device__ __forceinline__ void g311_body(const P& p) {
  // (this exact form keeps every load ahead of the first FFMA under ptxas 12.1 and 13.0 at 40 registers; a variant that
  //  folded the escape FMA into the main chain had 2 loads sunk below the FMAs - see NOTES.md, nvrtc_sass.py)
  constexpr int NT = K / 32, NW = NT / 32;
  constexpr int RB = K + K / 32 * 12;          // per-row bytes before the slot pool
  __shared__ float red[NW][R];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const u32 M83 = 0x83838383u, M80 = 0x80808080u;
  const long long row0 = (long long)blockIdx.x * R;
  const u8* cb = p.w + (size_t)blockIdx.x * p.RB;
  uint4 lo[R][2];
  u32 W[R][3];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const u8* rb = cb + r * RB;
    lo[r][0] = ldg128(rb + tid * 16);
    lo[r][1] = ldg128(rb + K / 2 + tid * 16);
#pragma unroll
    for (int c = 0; c < 3; ++c) W[r][c] = ldg32(rb + K + c * (K / 8) + tid * 4);
  }
  const int E = (int)p.E;
  u32 es[SPT];
#pragma unroll
  for (int j = 0; j < SPT; ++j) {
    const int i = tid + j * NT;
    es[j] = (i < E) ? ldg32(cb + R * RB + i * 4) : 0u;
  }
  uint4 xq[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) xq[i] = ldx128(p.x + i * (K / 4) + tid * 8);
  float acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = 0.f;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    u32 h[8];
    dec11_hi(W[r][0], W[r][1], W[r][2], h, M83, M80);
    const u32 lw[8] = {lo[r][0].x, lo[r][0].y, lo[r][0].z, lo[r][0].w, lo[r][1].x, lo[r][1].y, lo[r][1].z, lo[r][1].w};
    float a = acc[r];
#pragma unroll
    for (int gq = 0; gq < 8; ++gq) {
      const u32 w01 = prmt(lw[gq], h[gq], 0x5140), w23 = prmt(lw[gq], h[gq], 0x7362);
      const uint4 xv = xq[gq >> 1];
      const u32 x01 = (gq & 1) ? xv.z : xv.x, x23 = (gq & 1) ? xv.w : xv.y;
      a = fmaf(blo(w01), blo(x01), a); a = fmaf(bhi(w01), bhi(x01), a);
      a = fmaf(blo(w23), blo(x23), a); a = fmaf(bhi(w23), bhi(x23), a);
    }
    acc[r] = a;
  }
#pragma unroll
  for (int j = 0; j < SPT; ++j) {
    const int i = tid + j * NT;
    if (i < E) {
      const u32 pos = es[j] >> 16;
      const float c = __uint_as_float(((u32)ldx16(p.x + (pos & (K - 1)))) << 16) * esc_delta11(es[j], p.invscale);
      const int er = pos >> 13;
#pragma unroll
      for (int r = 0; r < R; ++r) if (r == er) acc[r] += c;
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    float v = acc[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[wid][r] = v;
  }
  __syncthreads();
  if (tid < R) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) s += red[w][tid];
    p.out[row0 + tid] = f2bf(s * p.scale);
  }
}

// ------------------------------------------------------------------ TCM: F12, tensor cores, 2 <= M <= 8*NT8 (unpooler slice)
// CTA = RT 16-row tiles x NW warps (K slices); B fragments (x^T, 8 tokens per n-tile) loaded per step pair from global
// (L1), one pair ahead; per-CTA reduction in smem laid out [warp][token][row] (conflict-free row-contiguous reads).

// XS = 1: x staged in smem (MT x (K+8)); XS = 0: B fragments loaded from global (L1-cached) one step pair ahead.
template <int RT, int NW, int K, int NT8, int EMAXT, int XS>
__device__ __forceinline__ void tcm_body(const P& p, int M, long long N) {
  constexpr int S = K / 16, SW = S / NW;
  static_assert(SW % 4 == 0, "steps per warp: multiple of 4");
  constexpr int NPAIR = SW / 2, NQUAD = SW / 4, NT = NW * 32, XP = K + 8, MT = NT8 * 8;
  extern __shared__ __align__(16) unsigned char smraw[];
  u16* xs = (u16*)smraw;                                     // [MT][XP] when XS
  float* red = (float*)(smraw + (XS == 1 ? MT * XP * 2 : 0));     // [NW][MT][RT*16]
  float* escc = red + NW * RT * 16 * MT;                     // [RT*16*EMAXT][MT]
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const u32 M87 = 0x87878787u;
  const u8* tb0 = p.w + (size_t)blockIdx.x * RT * p.RB;
  uint4 lo[RT][NPAIR], cd[RT][NQUAD];
#pragma unroll
  for (int rt = 0; rt < RT; ++rt) {
    const u8* tb = tb0 + (size_t)rt * p.RB;
#pragma unroll
    for (int i = 0; i < NPAIR; ++i) lo[rt][i] = ldg128(tb + (size_t)(wid * NPAIR + i) * 512 + lane * 16);
#pragma unroll
    for (int i = 0; i < NQUAD; ++i) cd[rt][i] = ldg128(tb + (size_t)(S / 2) * 512 + (size_t)(wid * NQUAD + i) * 512 + lane * 16);
  }
  const int E = (int)p.E, ns = RT * 16 * E;
  u32 es = 0;
  if (tid < ns) {
    const int r = tid / E;
    es = ldg32(tb0 + (size_t)(r >> 4) * p.RB + (size_t)(S / 2 + S / 4) * 512 + ((r & 15) * E + tid - r * E) * 4);
  }
  const int kw = wid * SW * 16;
  // B fragments of step pair i for n-tile n: x[tok = 8n + g][kw + 32i + 8t .. +7] (16 B: steps 0 and 1)
  uint4 xb[NT8], xn[NT8];
  uint4 xall[XS == 5 ? NPAIR : 1][NT8];                     // XS == 5: every B fragment loaded up front
  if (XS == 5) {
#pragma unroll
    for (int i = 0; i < NPAIR; ++i)
#pragma unroll
      for (int n = 0; n < NT8; ++n) {
        const int tok = n * 8 + g;
        xall[i][n] = tok < M ? ldx128(p.x + (size_t)tok * K + kw + i * 32 + t * 8) : make_uint4(0, 0, 0, 0);
      }
  } else if (XS == 1) {
    for (int c = tid; c < MT * (K / 8); c += NT) {
      const int tok = c / (K / 8), kc = c - tok * (K / 8);
      const uint4 v = tok < M ? ldx128(p.x + (size_t)tok * K + kc * 8) : make_uint4(0, 0, 0, 0);
      *(uint4*)(xs + tok * XP + kc * 8) = v;
    }
    __syncthreads();
  } else {
#pragma unroll
    for (int n = 0; n < NT8; ++n) {
      const int tok = n * 8 + g;
      xb[n] = tok < M ? ldx128(p.x + (size_t)tok * K + kw + t * 8) : make_uint4(0, 0, 0, 0);
    }
  }
  float acc[RT][NT8][4];
#pragma unroll
  for (int rt = 0; rt < RT; ++rt)
#pragma unroll
    for (int n = 0; n < NT8; ++n) acc[rt][n][0] = acc[rt][n][1] = acc[rt][n][2] = acc[rt][n][3] = 0.f;
#pragma unroll
  for (int i = 0; i < NPAIR; ++i) {
    if (XS == 5) {
#pragma unroll
      for (int n = 0; n < NT8; ++n) xb[n] = xall[XS == 5 ? i : 0][n];
    } else if (XS) {
#pragma unroll
      for (int n = 0; n < NT8; ++n) xb[n] = *(const uint4*)(xs + (n * 8 + g) * XP + kw + i * 32 + t * 8);
    } else if (i + 1 < NPAIR) {
#pragma unroll
      for (int n = 0; n < NT8; ++n) {
        const int tok = n * 8 + g;
        xn[n] = tok < M ? ldx128(p.x + (size_t)tok * K + kw + (i + 1) * 32 + t * 8) : make_uint4(0, 0, 0, 0);
      }
    }
#pragma unroll
    for (int rt = 0; rt < RT; ++rt) {
      const uint4 cq = cd[rt][i >> 1];
      const u32 c0 = (i & 1) ? cq.z : cq.x, c1 = (i & 1) ? cq.w : cq.y;
      const uint4 a0 = decode12f(lo[rt][i].x, lo[rt][i].y, c0, M87);
      const uint4 a1 = decode12f(lo[rt][i].z, lo[rt][i].w, c1, M87);
#pragma unroll
      for (int n = 0; n < NT8; ++n) { mma16816(acc[rt][n], a0, xb[n].x, xb[n].y); mma16816(acc[rt][n], a1, xb[n].z, xb[n].w); }
    }
    if (XS == 0 && i + 1 < NPAIR) {
#pragma unroll
      for (int n = 0; n < NT8; ++n) xb[n] = xn[n];
    }
  }
  // escapes: slot thread -> correction for every token (filler slots have delta 0 and skip the x reads)
  if (tid < ns) {
    const float dl = esc_delta12(es, p.invscale);
    const int k = es >> 16;
    for (int tok = 0; tok < M; ++tok) {
      float c = 0.f;
      if (dl != 0.f) c = __uint_as_float(((u32)(XS == 1 ? xs[tok * XP + k] : ldx16(p.x + (size_t)tok * K + k))) << 16) * dl;
      escc[((tid % E) * MT + tok) * (RT * 16) + tid / E] = c;
    }
  }
#pragma unroll
  for (int rt = 0; rt < RT; ++rt)
#pragma unroll
    for (int n = 0; n < NT8; ++n) {
      float* rr = red + ((size_t)wid * MT + n * 8 + 2 * t) * (RT * 16) + rt * 16;   // [w][tok][row]
      rr[g] = acc[rt][n][0]; rr[RT * 16 + g] = acc[rt][n][1];
      rr[g + 8] = acc[rt][n][2]; rr[RT * 16 + g + 8] = acc[rt][n][3];
    }
  __syncthreads();
  for (int o = tid; o < RT * 16 * M; o += NT) {
    const int tok = o / (RT * 16), r = o - tok * (RT * 16);
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) s += red[((size_t)w * MT + tok) * (RT * 16) + r];
    for (int e = 0; e < E; ++e) s += escc[((size_t)e * MT + tok) * (RT * 16) + r];
    p.out[(size_t)tok * N + (size_t)blockIdx.x * RT * 16 + r] = f2bf(s * p.scale);
  }
}
struct PM { const u16* x; u16* out; long long N; int M; int pad; };

"""

_g1_FAKE12, _g1_FAKE11 = 112, 120
_g1_LAYER_K = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpslice": 4096}
# layer -> (kernel, min blocks per SM, bake weight pointer). Measured on the H200 sandbox (NOTES.md).
_g1_CFG = {"qkv_gate": ("g311", 6, False), "pooler": ("g311", 6, False), "unpslice": ("g311", 12, False),
       "o": ("tc", 4, True)}
_g1_TC_NW = 8
# small M (2..8) for the unpooler slice: (RT, NW, min blocks); 64 CTAs x 1024 threads, all x B fragments loaded up front
_g1_SMALLM = {"unpslice": (1, 32, 1)}


# ------------------------------------------------------------------------------------------- NVRTC module
class _g1__Mod:
    def __init__(self, src, name="gemv1.cu"):
        from cuda.bindings import driver, nvrtc
        self.d = driver
        torch.cuda.init()
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17", b"--use_fast_math"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, n = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        _, n = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * n
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        if int(err):
            raise RuntimeError(f"cuModuleLoadData: {err}")
        self.funcs = {}

    def launch(self, name, grid, block, args, smem=0):
        f = self.funcs.get(name)
        if f is None:
            err, f = self.d.cuModuleGetFunction(self.module, name.encode())
            if int(err):
                raise RuntimeError(f"cuModuleGetFunction({name}): {err}")
            self.funcs[name] = f
        raw = ctypes.create_string_buffer(len(args) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, args, len(args))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = self.d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = self.d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        if int(err):
            raise RuntimeError(f"launch {name}: {err}")


# ------------------------------------------------------------------------------------------- formats
def _g1__window(e, width):
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    return max(range(1, 256 - width), key=lambda q: int(cs[q + width] - cs[q]))


def _g1_compress11(w):
    """F11 planes: lo (fake low byte), s, h (2-bit), esc, u (true bits), elo (window start)"""
    u = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    elo = _g1__window(e, 8)
    c = e - elo
    esc = (c < 0) | (c > 7)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m), "s": torch.where(esc, torch.zeros_like(s), s),
            "h": c >> 1, "esc": esc, "u": u, "elo": elo}


def _g1_compress12(w):
    """F12 planes: lo (fake low byte), nib (s << 3 | h, 3-bit h), esc, u, elo"""
    u = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    elo = _g1__window(e, 16)
    c = e - elo
    esc = (c < 0) | (c > 15)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m),
            "nib": torch.where(esc, torch.zeros_like(u), (s << 3) | (c >> 1)), "esc": esc, "u": u, "elo": elo}


def _g1_pack_code32(s, h):
    """F11 code words: s, h [..., 32] (weight 4g + b = byte lane b of hi word g) -> int32 [..., 3] (W0, W1, W2)
      W0 byte b: bit7 s(0,b) | bits0-1 h(0,b) | bits3-4 h(1,b) ; byte (b+1)%4: bit2 s(1,b), bit5 s(6,b), bit6 s(7,b)
      W1 byte b: bit7 s(2,b) | bits0-1 h(2,b) | bits3-4 h(3,b) | bits5-6 h(6,b) ; byte (b+1)%4: bit2 s(3,b)
      W2 byte b: bit7 s(4,b) | bits0-1 h(4,b) | bits3-4 h(5,b) | bits5-6 h(7,b) ; byte (b+1)%4: bit2 s(5,b)"""
    sh = s.to(torch.int64).view(*s.shape[:-1], 8, 4)
    hh = h.to(torch.int64).view(*h.shape[:-1], 8, 4)
    W = [torch.zeros(s.shape[:-1], dtype=torch.int64, device=s.device) for _ in range(3)]
    for b in range(4):
        nb = (b + 1) % 4
        W[0] |= (sh[..., 0, b] << 7 | hh[..., 0, b] | hh[..., 1, b] << 3) << (8 * b)
        W[0] |= (sh[..., 1, b] << 2 | sh[..., 6, b] << 5 | sh[..., 7, b] << 6) << (8 * nb)
        W[1] |= (sh[..., 2, b] << 7 | hh[..., 2, b] | hh[..., 3, b] << 3 | hh[..., 6, b] << 5) << (8 * b)
        W[1] |= (sh[..., 3, b] << 2) << (8 * nb)
        W[2] |= (sh[..., 4, b] << 7 | hh[..., 4, b] | hh[..., 5, b] << 3 | hh[..., 7, b] << 5) << (8 * b)
        W[2] |= (sh[..., 5, b] << 2) << (8 * nb)
    out = torch.stack(W, -1)
    return torch.where(out >= 2 ** 31, out - 2 ** 32, out).to(torch.int32)


def _g1_pack_nib8(nib):
    """F12 nibble word: nib [..., 8] (s << 3 | h) -> int32 [...]: byte b = h_b | s_b << 7 | h_{4+b} << 4 | s_{4+(b+3)%4} << 3"""
    nib = nib.to(torch.int64)
    s, h = nib >> 3, nib & 7
    word = torch.zeros(nib.shape[:-1], dtype=torch.int64, device=nib.device)
    for b in range(4):
        word |= (h[..., b] | (s[..., b] << 7) | (h[..., 4 + b] << 4) | (s[..., 4 + (b + 3) % 4] << 3)) << (8 * b)
    return torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)


def _g1__row_slots(f, E, fake):
    """per-row escape slots [n, E] int32: (k << 16) | true bits; unused slots hold 2^(elo - 127) (delta 0)"""
    esc, u = f["esc"], f["u"]
    n = esc.shape[0]
    cnt = esc.sum(1)
    slots = torch.full((n, E), f["elo"] << 7, dtype=torch.int64, device=esc.device)
    nn, kk = esc.nonzero(as_tuple=True)
    if len(nn):
        first = torch.zeros(n + 1, dtype=torch.int64, device=esc.device)
        first[1:] = torch.cumsum(cnt, 0)
        rank = torch.arange(len(nn), device=esc.device) - first[nn]
        slots[nn, rank] = (kk.to(torch.int64) << 16) | u[nn, kk].to(torch.int64)
    return torch.where(slots >= 2 ** 31, slots - 2 ** 32, slots).to(torch.int32)


def _g1_pack_g311(w, f=None):
    """F11 row blobs for g311 -> dict(w, E, RB, elo, ...)"""
    n, k = w.shape
    f = f or _g1_compress11(w)
    G = k // 32
    perm = lambda a: a.view(n, 4, G, 8).permute(0, 2, 1, 3).reshape(n, G, 32)   # thread j, chunk c -> 8c + i
    words = _g1_pack_code32(perm(f["s"]), perm(f["h"]))                                # [n, G, 3]
    planes = words.permute(0, 2, 1).contiguous().view(torch.uint8).view(n, 12 * G)
    lo = perm(f["lo"]).view(n, G, 2, 16).permute(0, 2, 1, 3).reshape(n, k).to(torch.uint8)
    E = max(4, (int(f["esc"].sum(1).max()) + 3) // 4 * 4)
    sl = _g1__row_slots(f, E, _g1_FAKE11)
    blob = torch.cat([lo, planes, sl.contiguous().view(torch.uint8).view(n, 4 * E)], 1)
    pad = (-blob.shape[1]) % 16
    if pad:
        blob = torch.cat([blob, torch.zeros((n, pad), dtype=torch.uint8, device=w.device)], 1)
    blob = blob.contiguous()
    return {"w": blob.view(-1), "E": E, "RB": blob.shape[1], "n": n, "k": k, "elo": f["elo"], "fake": _g1_FAKE11,
            "spt": (E + G - 1) // G, "n_esc": int(f["esc"].sum())}


def _g1_pack_tc(w, f=None, align=128):
    """F12 16-row tile blobs for tc -> dict"""
    n, k = w.shape
    assert n % 16 == 0 and k % 64 == 0
    f = f or _g1_compress12(w)
    E = max(1, int(f["esc"].sum(1).max()))
    S, T = k // 16, n // 16
    dev = w.device
    st = torch.arange(S, device=dev)
    ln = torch.arange(32, device=dev)
    g, t = ln // 4, ln % 4
    k0 = (32 * (st // 2))[:, None] + (8 * t)[None, :] + (4 * (st % 2))[:, None]
    rr = torch.stack([g, g, g + 8, g + 8, g, g, g + 8, g + 8], 1)
    kk = k0[:, :, None] + torch.tensor([0, 1, 0, 1, 2, 3, 2, 3], device=dev)[None, None, :]
    rows = (torch.arange(T, device=dev)[:, None, None, None] * 16 + rr[None, None, :, :]).expand(T, S, 32, 8)
    kidx = kk[None].expand(T, S, 32, 8)
    lo_g = f["lo"][rows, kidx].to(torch.uint8)
    cw = _g1_pack_nib8(f["nib"][rows, kidx])
    lo_p = lo_g.view(T, S // 2, 2, 32, 8).permute(0, 1, 3, 2, 4).reshape(T, S // 2 * 512)
    cd_q = cw.contiguous().view(torch.uint8).view(T, S // 4, 4, 32, 4).permute(0, 1, 3, 2, 4).reshape(T, S // 4 * 512)
    sl = _g1__row_slots(f, E, _g1_FAKE12).view(T, 16 * E).contiguous().view(torch.uint8).view(T, 64 * E)
    blob = torch.cat([lo_p, cd_q, sl], 1)
    pad = (-blob.shape[1]) % align
    if pad:
        blob = torch.cat([blob, torch.zeros((T, pad), dtype=torch.uint8, device=dev)], 1)
    blob = blob.contiguous()
    return {"w": blob.view(-1), "E": E, "RB": blob.shape[1], "n": n, "k": k, "elo": f["elo"], "fake": _g1_FAKE12,
            "n_esc": int(f["esc"].sum())}


def _g1__f32bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def _g1__kernel(lname, ent, kind, mb, bake):
    K = ent["k"]
    name = f"g1_{lname}_{kind}"
    if kind == "g311":
        nt, body = K // 32, f"g311_body<1,{K},{ent['spt']}>"
    else:
        nt, body = 32 * _g1_TC_NW, f"tc_body<{_g1_TC_NW},{K},{ent['E']}>"
    lb = f"{nt}, {mb}" if mb else f"{nt}"
    if bake:
        sc = 2.0 ** (ent["elo"] - ent["fake"])
        tpl = f"{ent['w'].data_ptr()}ull,{ent['E']}u,{ent['RB']}u,{_g1__f32bits(sc)}u,{_g1__f32bits(1.0 / sc)}u"
        src = (f'extern "C" __global__ void __launch_bounds__({lb}) {name}(const __grid_constant__ PX q) '
               f'{{ const P p = mkp<{tpl}>(q); {body}(p); }}\n')
    else:
        src = f'extern "C" __global__ void __launch_bounds__({lb}) {name}(const __grid_constant__ P p) {{ {body}(p); }}\n'
    return name, nt, src


def _g1_prepare_gemv1(weights, cfg=None):
    cfg = dict(_g1_CFG if cfg is None else cfg)
    srcw = {"qkv_gate": weights["qkv_gate"], "o": weights["o"], "pooler": weights["pooler"],
            "unpslice": weights["unpooler"][:1024]}
    st = {"layers": {}, "w": {k: v for k, v in weights.items()}}
    src = _g1_SRC
    for lname, (kind, mb, bake) in cfg.items():
        w = srcw[lname].contiguous()
        ent = _g1_pack_g311(w) if kind == "g311" else _g1_pack_tc(w)
        if kind == "g311" and ent["spt"] > 4:      # pathological escape count: leave the layer to the caller
            continue
        if kind == "tc" and ent["E"] > 16:
            continue
        name, nt, ksrc = _g1__kernel(lname, ent, kind, mb, bake)
        src += ksrc
        sc = 2.0 ** (ent["elo"] - ent["fake"])
        ent.update(kind=kind, name=name, nt=nt, bake=bake, scale=sc,
                   grid=ent["n"] if kind == "g311" else ent["n"] // 16)
        st["layers"][lname] = ent
    st["small"] = {}
    for lname, (rt, nw, mb) in _g1_SMALLM.items():
        ent = st["layers"].get(lname)
        if ent is not None and ent["kind"] == "tc":
            e12 = ent
        else:
            e12 = _g1_pack_tc(srcw[lname].contiguous())
            if e12["E"] > 16:
                continue
        name = f"g1m_{lname}"
        sc = 2.0 ** (e12["elo"] - e12["fake"])
        tpl = f"{e12['w'].data_ptr()}ull,{e12['E']}u,{e12['RB']}u,{_g1__f32bits(sc)}u,{_g1__f32bits(1.0 / sc)}u"
        lb = f"{nw * 32}, {mb}" if mb else f"{nw * 32}"
        src += (f'extern "C" __global__ void __launch_bounds__({lb}) {name}(const __grid_constant__ PM q) '
                f'{{ const PX px = {{q.x, q.out}}; const P p = mkp<{tpl}>(px); tcm_body<{rt},{nw},{e12["k"]},1,{e12["E"]},5>(p, q.M, q.N); }}\n')
        smem = nw * rt * 16 * 8 * 4 + rt * 16 * e12["E"] * 8 * 4
        st["small"][lname] = {"ent": e12, "name": name, "grid": e12["n"] // (16 * rt), "nt": nw * 32, "smem": smem}
    st["mod"] = _g1__Mod(src) if (st["layers"] or st["small"]) else None
    return st


def _g1_run_gemv1(x, config, state):
    """-> [m, n] (bf16; router fp32). Kernels for m == 1 on qkv_gate / o / pooler / the unpooler slice; anything else
    (other m, router, misaligned x, a layer whose escape count was pathological) falls back to torch."""
    layer, m, n = config["layer"], config["m"], config["n"]
    lname = "unpslice" if layer == "unpooler" else layer
    sm = state["small"].get(lname) if 2 <= m <= 8 else None
    if sm is not None and n == sm["ent"]["n"] and x.is_contiguous() and x.data_ptr() % 16 == 0:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        state["mod"].launch(sm["name"], sm["grid"], sm["nt"], struct.pack("<QQqii", x.data_ptr(), out.data_ptr(), n, m, 0),
                            smem=sm["smem"])
        return out
    ent = state["layers"].get(lname) if m == 1 else None
    if ent is None or n != ent["n"] or not x.is_contiguous() or x.data_ptr() % 16:
        w = state["w"][layer]
        if layer == "router":
            return torch.nn.functional.linear(x.float(), w)
        return torch.mm(x, w[:n].t())
    out = torch.empty((1, n), dtype=torch.bfloat16, device=x.device)
    if ent["bake"]:
        args = struct.pack("<QQ", x.data_ptr(), out.data_ptr())
    else:
        args = struct.pack("<QQQIIff", ent["w"].data_ptr(), x.data_ptr(), out.data_ptr(), ent["E"], ent["RB"],
                           ent["scale"], 1.0 / ent["scale"])
    state["mod"].launch(ent["name"], ent["grid"], ent["nt"], args)
    return out


# ------------------------------------------------------------------------------------------- round-trip test
def _g1_decode11_ref(lo, words):
    """exact torch emulation of the device F11 decode: lo [..., 32], words [..., 3] -> fake bf16 bits [..., 32]"""
    W = [(words[..., i].to(torch.int64) & 0xFFFFFFFF) for i in range(3)]
    rot = lambda v, r: ((v >> r) | (v << (32 - r))) & 0xFFFFFFFF
    sel = lambda a, b: (a & 0x80808080) | (b & 0x7F7F7F7F)
    X = [W[0], rot(W[0], 3), W[1], rot(W[1], 3), W[2], rot(W[2], 3), sel(rot(W[0], 6), rot(W[1], 5)), sel(rot(W[0], 7), rot(W[2], 5))]
    out = []
    for g in range(8):
        hw = (X[g] & 0x83838383) | 0x3C3C3C3C
        for b in range(4):
            out.append((((hw >> (8 * b)) & 0xFF) << 8) | lo[..., 4 * g + b].to(torch.int64))
    return torch.stack(out, -1)


def _g1_decode12_ref(lo, word):
    """exact torch emulation of the device F12 decode: lo [..., 8], word [...] -> fake bf16 bits [..., 8]"""
    wd = word.to(torch.int64) & 0xFFFFFFFF
    out = []
    for i in range(8):
        src = wd if i < 4 else (((wd >> 4) | (wd << 28)) & 0xFFFFFFFF)
        hi = (((src >> (8 * (i % 4))) & 0xFF) & 0x87) | 0x38
        out.append((hi << 8) | lo[..., i].to(torch.int64))
    return torch.stack(out, -1)


def _g1_roundtrip(w, fmt):
    """compress -> pack -> exact device-decode emulation -> true bits (+ escapes from the slots); True if bit-exact"""
    n, k = w.shape
    u = w.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
    if fmt == 11:
        f = _g1_compress11(w)
        words = _g1_pack_code32(f["s"].view(n, k // 32, 32), f["h"].view(n, k // 32, 32))
        fake = _g1_decode11_ref(f["lo"].view(n, k // 32, 32), words).view(n, k)
        shift = f["elo"] - _g1_FAKE11
    else:
        f = _g1_compress12(w)
        word = _g1_pack_nib8(f["nib"].view(n, k // 8, 8))
        fake = _g1_decode12_ref(f["lo"].view(n, k // 8, 8), word).view(n, k)
        shift = f["elo"] - _g1_FAKE12
    true = (fake & 0x807F) | ((((fake >> 7) & 0xFF) + shift) << 7)
    E = max(1, int(f["esc"].sum(1).max()))
    slots = _g1__row_slots(f, E, 0).to(torch.int64) & 0xFFFFFFFF                      # escapes come back from the slots
    rows = torch.arange(n, device=w.device)[:, None].expand(n, E)
    real = (slots & 0xFFFF) != (f["elo"] << 7) if True else None
    kk = slots >> 16
    true[rows[real], kk[real]] = slots[real] & 0xFFFF
    return bool((true == u).all()), int(f["esc"].sum())

# ======================================================================== dispatch
def prepare(weights):
    st = {'b7': _b7_prepare(weights)}
    st['G'] = _g1_prepare_gemv1(weights)
    torch.cuda.synchronize()
    return st


def kernel_fn(data, state):
    x, cfg = data['x'], data['config']
    layer, m = cfg['layer'], cfg['m']
    if layer != 'router' and (m == 1 or (layer == 'unpooler' and m <= 8)):
        return _g1_run_gemv1(x, cfg, state['G'])
    return _b7_kernel_fn(data, state['b7'])
