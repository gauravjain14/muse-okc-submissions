# okc: problem=muse_bf16_proj
# Third-party code: each segment copied or adapted from another project is marked in place with SPDX-SnippetBegin /
# SPDX-SnippetEnd comments that give its source, copyright, license and our changes. Full license texts are in
# LICENSES/ and the third-party notices in NOTICE, both at the repository root.
# Independent BF16 projection candidate, revision 2.
import torch


# Source module: proj_triton
"""Fresh BF16 projection candidates; all reductions accumulate in FP32."""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def pr_gemv(X, W, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
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
def pr_matmul_transpose(X,W,Y,P,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
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
def pr_reduce_split(P,Y,SIZE:tl.constexpr,SK:tl.constexpr,B:tl.constexpr):
    ix=tl.program_id(0)*B+tl.arange(0,B)
    out=tl.full((B,),0,tl.float32)
    for s in range(SK):
        out += tl.load(P+s*SIZE+ix,ix<SIZE,0)
    tl.store(Y+ix,out,ix<SIZE)

def pr_run(x,w,n,config):
    m,k=x.shape
    y=torch.empty((m,n),dtype=torch.bfloat16,device=x.device)
    if config[0]=='gemv':
        _,bn,bk,nw=config
        pr_gemv[(tr.cdiv(n,bn),m)](x,w,y,m,n,k,bn,bk,num_warps=nw)
    else:
        _,bn,bm,bk,sk,nw,ns=config
        p=torch.empty((sk,m,n),dtype=torch.float32,device=x.device) if sk>1 else y
        pr_matmul_transpose[(tr.cdiv(n,bn),tr.cdiv(m,bm),sk)](x,w,y,p,m,n,k,bn,bm,bk,sk,num_warps=nw,num_stages=ns)
        if sk>1:
            pr_reduce_split[(tr.cdiv(m*n,256),)](p,y,m*n,sk,256)
    return y

def pr_configs(m,n,k,quick=False):
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


def pr_configs_wide(m,n,k,quick=False):
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
def rt__router_simt(
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
def rt__router_dot(
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
def rt__router_reduce(
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


def rt_prepare_router(w):
    """Keep exact FP32 weights and prepare the permitted BF16 hi/lo pair."""
    if w.ndim != 2 or w.dtype != torch.float32:
        raise ValueError("router weights must be a two-dimensional FP32 tensor")
    w = w.contiguous()
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)
    return {"w": w, "hi": hi, "lo": lo, "n": w.shape[0], "k": w.shape[1]}


def rt__check_power_of_two(value, name, minimum=1):
    if not isinstance(value, int) or value < minimum or value & (value - 1):
        raise ValueError(f"{name} must be a power of two >= {minimum}")


def rt_run_router(x, state, config_tuple):
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
        rt__check_power_of_two(bn, "block_n")
        rt__check_power_of_two(bk, "block_k")
    elif kind == "dot" and len(config_tuple) == 8:
        _, bm, bn, bk, splits, warps, stages, swap = config_tuple
        rt__check_power_of_two(bm, "block_m", 16)
        rt__check_power_of_two(bn, "block_n", 16)
        rt__check_power_of_two(bk, "block_k", 16)
        if not isinstance(stages, int) or stages < 1:
            raise ValueError("num_stages must be a positive integer")
    else:
        raise ValueError(f"invalid router config: {config_tuple!r}")
    if not isinstance(splits, int) or splits < 1:
        raise ValueError("split_k must be a positive integer")
    rt__check_power_of_two(warps, "num_warps")

    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    if m == 0 or n == 0:
        return out
    partial = (
        torch.empty((splits, m, n), dtype=torch.float32, device=x.device)
        if splits > 1 else out
    )
    if kind == "simt":
        w = state["w"]
        rt__router_simt[(triton.cdiv(n, bn), m, splits)](
            x, w, out, partial, m, n, k,
            x.stride(0), x.stride(1), w.stride(0), w.stride(1),
            BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=splits,
            num_warps=warps,
        )
    else:
        hi, lo = state["hi"], state["lo"]
        rt__router_dot[(triton.cdiv(m, bm) * triton.cdiv(n, bn), splits)](
            x, hi, lo, out, partial, m, n, k,
            x.stride(0), x.stride(1), hi.stride(0), hi.stride(1),
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
            SPLIT_K=splits, SWAP_AB=bool(swap),
            num_warps=warps, num_stages=stages,
        )
    if splits > 1:
        rt__router_reduce[(triton.cdiv(m * n, 256),)](
            partial, out, m * n, SPLIT_K=splits,
            BLOCK_SPLIT=triton.next_power_of_2(splits), BLOCK=256,
            num_warps=4,
        )
    return out


def rt_router_configs(m):
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


def rt_configs(m, n, k, quick=False):
    """Benchmark-compatible shape API; quick sweeps contain at most four tuples."""
    if not quick:
        return rt_router_configs(m)
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


ti__CUDA_SOURCE = r"""
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

ti__CPP_SOURCE = r"""
#include <torch/extension.h>
at::Tensor tinygemm_forward(at::Tensor x, at::Tensor w, int64_t n, int64_t stages, bool interleave);
"""


@lru_cache(maxsize=1)
def ti__extension():
    return load_inline(
        name="muse_hopper_tinygemm_v2",
        cpp_sources=ti__CPP_SOURCE,
        cuda_sources=ti__CUDA_SOURCE,
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


def ti_run(x, w, n, config):
    if len(config) != 3 or config[0] != "tiny" or config[1] not in (4, 8, 16):
        raise ValueError(f"invalid TinyGEMM configuration: {config!r}")
    return ti__extension().tinygemm_forward(x, w, int(n), int(config[1]), bool(config[2]))


def ti_configs(m, n, k, quick=False):
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
def tm_tma_matmul(
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
def tm_tma_reduce_split(P, Y, SIZE: tl.constexpr, SK: tl.constexpr, B: tl.constexpr):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    total = tl.full((B,), 0, tl.float32)
    for split in tl.static_range(0, SK):
        total += tl.load(P + split * SIZE + idx, idx < SIZE, 0)
    tl.store(Y + idx, total.to(tl.bfloat16), idx < SIZE)


def tm__can_describe(tensor):
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
class tm_PreparedWeights:
    tensor: torch.Tensor
    descriptors: dict = field(default_factory=dict)

    def descriptor(self, n, bn, bk):
        key = (n, bn, bk)
        if key not in self.descriptors:
            self.descriptors[key] = TensorDescriptor.from_tensor(
                self.tensor[:n], [bn, bk]
            )
        return self.descriptors[key]


def tm_prepare(w):
    """Retain BF16 weights and optionally reuse their host TMA descriptors."""
    if not isinstance(w, torch.Tensor) or w.dtype != torch.bfloat16 or w.ndim != 2:
        raise ValueError("prepare expects a two-dimensional BF16 weight tensor")
    return tm_PreparedWeights(w)


def tm_run(x, w, n, cfg=None):
    """Return BF16 x @ w[:n].T; w may be a tensor or PreparedWeights."""
    prepared = w if isinstance(w, tm_PreparedWeights) else None
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
        cfg = (tm_configs(m, n, k, quick=True)[0] if k > 0
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
    use_tma = tm__can_describe(x) and tm__can_describe(weight)
    if use_tma:
        wd = (prepared.descriptor(n, bn, bk) if prepared is not None
              else TensorDescriptor.from_tensor(weight[:n], [bn, bk]))
        xd = TensorDescriptor.from_tensor(x, [bm, bk])
    else:
        wd, xd = weight, x
    tm_tma_matmul[(tr.cdiv(n, bn) * tr.cdiv(m, bm), sk)](
        wd, xd, y, partial, m, n, k,
        weight.stride(0), weight.stride(1), x.stride(0), x.stride(1),
        bn, bm, bk, sk, stages, use_tma,
        num_warps=warps, num_stages=stages,
    )
    if sk > 1:
        tm_tma_reduce_split[(tr.cdiv(m * n, 256),)](
            partial, y, m * n, sk, 256, num_warps=4
        )
    return y


def tm_configs(m, n, k, quick=False):
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


def tm_configs_large(m,n,k,quick=False):
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
def pa_packed_mm(WD, XD, Y, P, M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
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
def pa_packed_reduce(P,Y,SIZE:tl.constexpr,SK:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    acc=tl.zeros((B,),tl.float32)
    for split in tl.static_range(SK):
        acc+=tl.load(P+split*SIZE+i,i<SIZE,0)
    tl.store(Y+i,acc.to(tl.bfloat16),i<SIZE)


pa_PACKINGS=((16,128),(32,128),(64,64),(64,128),(64,256),(128,128))


def pa_prepare(w):
    n,k=w.shape
    state={'weight':w,'packed':{}}
    for bn,bk in pa_PACKINGS:
        npad,kpad=tr.cdiv(n,bn)*bn,tr.cdiv(k,bk)*bk
        source=w if (n,k)==(npad,kpad) else torch.nn.functional.pad(w,(0,kpad-k,0,npad-n))
        packed=source.view(npad//bn,bn,kpad//bk,bk).permute(0,2,1,3).contiguous()
        state['packed'][(bn,bk)]=(packed,TensorDescriptor.from_tensor(packed,[1,1,bn,bk]))
    return state


def pa_run(x,state,n,cfg):
    _,bn,bm,bk,sk,nw,ns=cfg
    m,k=x.shape
    y=torch.empty((m,n),dtype=torch.bfloat16,device=x.device)
    partial=torch.empty((sk,m,n),dtype=torch.float32,device=x.device) if sk>1 else y
    wd=state['packed'][(bn,bk)][1]
    xd=TensorDescriptor.from_tensor(x,[bm,bk])
    pa_packed_mm[(tr.cdiv(n,bn),tr.cdiv(m,bm),sk)](wd,xd,y,partial,m,n,k,bn,bm,bk,sk,ns,num_warps=nw,num_stages=ns)
    if sk>1:pa_packed_reduce[(tr.cdiv(m*n,256),)](partial,y,m*n,sk,256)
    return y


def pa_configs(m,n,k,quick=False):
    bm=16 if m<=16 else 32 if m<=32 else 64
    initial=[('packed',64,bm,128,4,4,3),('packed',32,bm,128,4,4,3),
             ('packed',64,bm,256,1,4,3),('packed',64,bm,128,1,4,4)]
    if quick:return initial
    return list(dict.fromkeys(initial+[
        ('packed',bn,bm,bk,sk,4,ns)
        for bn,bk in pa_PACKINGS for sk in (1,2,4,8) for ns in (2,4)
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
def ri__pack_router(W, PAIR, BLOCKED,
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


def ri_prepare_router(w):
    """Prepare both permitted hi+lo layouts, retaining exact FP32 for SIMT."""
    if w.ndim != 2 or w.dtype != torch.float32 or not w.is_cuda:
        raise ValueError("router weights must be a two-dimensional CUDA FP32 tensor")
    w = w.contiguous()
    n, k = w.shape
    padded_k = tr.cdiv(k, 128) * 128
    pair = torch.empty((n, 2 * k), dtype=torch.bfloat16, device=w.device)
    blocked = torch.empty((n, 2 * padded_k), dtype=torch.bfloat16, device=w.device)
    if n and k:
        ri__pack_router[(tr.cdiv(n * padded_k, 256),)](w, pair, blocked, n, k, padded_k, 256)
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
def ri_router_interleaved_dot(
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
def ri_router_interleaved_simt(
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
def ri_router_interleaved_reduce(PARTIAL, Y, SIZE: tl.constexpr,
                              SK: tl.constexpr, BLOCK_SPLIT: tl.constexpr,
                              BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    split = tl.arange(0, BLOCK_SPLIT)
    value = tl.load(PARTIAL + split[:, None] * SIZE + index[None, :],
                    (split[:, None] < SK) & (index[None, :] < SIZE), 0)
    tl.store(Y + index, tl.sum(value, axis=0), index < SIZE)


def ri_run_router(x, state, cfg):
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
        ri_router_interleaved_simt[(tr.cdiv(n, bn), m)](
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
        ri_router_interleaved_dot[(tr.cdiv(n, bn) * tr.cdiv(m, bm), sk)](
            x, packed, output, partial, m, n, k, packed.shape[1],
            x.stride(0), x.stride(1), bm, bn, bk, sk, group,
            num_warps=warps, num_stages=stages,
        )
        if sk > 1:
            ri_router_interleaved_reduce[(tr.cdiv(m * n, 256),)](
                partial, output, m * n, sk, tr.next_power_of_2(sk), 256,
                num_warps=4,
            )
    else:
        raise ValueError("router kind must be simt or interleaved")
    return output


def ri_configs(m, n, k, quick=False):
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


lt_MAX_WORKSPACE_BYTES = 256 * 1024 * 1024

lt__CPP_SOURCE = r"""
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
def lt__extension():
    return load_inline(
        name="muse_cublaslt_static_v1",
        cpp_sources=lt__CPP_SOURCE,
        functions=["lt_heuristics", "lt_workspace", "lt_run"],
        extra_cflags=["-O3", "-std=c++20"],
        extra_ldflags=["-lcublasLt", "-lcublas"],
        with_cuda=True,
        verbose=False,
    )


def lt_prepare(w):
    if w.ndim != 2 or w.dtype != torch.bfloat16:
        raise ValueError("cuBLASLt candidate expects rank-two BF16 weights")
    w = w.contiguous()
    if w.data_ptr() % 256:
        w = w.clone()
    return {"w": w, "wt": w.t().contiguous(), "n": w.shape[0], "k": w.shape[1]}


@lru_cache(maxsize=None)
def lt__heuristics(m, n, k, nn, count, device):
    return tuple(tuple(row) for row in lt__extension().lt_heuristics(m, n, k, nn, count, device))


def lt_configs(m, n, k, quick=False):
    if m <= 0 or n <= 0 or k <= 0:
        return []
    count = 2 if quick else 32
    device = torch.cuda.current_device()
    result = []
    for layout in ("tn", "nn"):
        for row in lt__heuristics(m, n, k, layout == "nn", count, device):
            candidate = ("lt", layout, *row)
            if candidate not in result:
                result.append(candidate)
    return result


def lt_describe_config(cfg):
    fields = ("kind", "layout", "algorithm_id", "tile_id", "split_k",
              "reduction_scheme", "cta_swizzle", "custom_option", "stages_id",
              "inner_shape_id", "cluster_shape_id", "workspace_bytes", "cublaslt_version")
    if len(cfg) != len(fields):
        raise ValueError("invalid cuBLASLt configuration length")
    return dict(zip(fields, cfg))


def lt_run(x, state, n, cfg):
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
    extension = lt__extension()
    required = extension.lt_workspace(m, n, k, lda, nn, settings, version, x.device.index)
    # A distinct allocation per invocation is captured in the CUDA graph. No
    # shared mutable workspace can race across separately captured invocations.
    workspace = torch.empty((required,), dtype=torch.uint8, device=x.device)
    return extension.lt_run(x, weight, n, nn, settings, version, workspace)


_CHOICES = {'8/qkv_gate': ('tiny', ['tiny', 8, False]), '8/o': ('packed', ['packed', 64, 16, 128, 1, 4, 4]), '8/pooler': ('tiny', ['tiny', 8, False]), '8/unpooler': ('tiny', ['tiny', 16, False]), '16/qkv_gate': ('tma', ['tma', 64, 16, 128, 4, 4, 4]), '16/o': ('triton', ['mm', 64, 16, 256, 1, 4, 3]), '16/pooler': ('tma', ['tma', 64, 16, 128, 4, 4, 4]), '16/unpooler': ('torch_transpose', ['nn']), '32/qkv_gate': ('tma', ['tma', 64, 32, 128, 4, 4, 4]), '32/o': ('packed', ['packed', 64, 32, 128, 1, 4, 4]), '32/pooler': ('tma', ['tma', 64, 32, 128, 4, 4, 4]), '32/unpooler': ('torch_transpose', ['nn']), '128/qkv_gate': ('lt', ['lt', 'nn', 66, 17, 3, 2, 0, 0, 35, 0, 4, 3538960, 130101]), '128/o': ('torch_transpose', ['nn']), '128/pooler': ('torch', []), '128/unpooler': ('lt', ['lt', 'nn', 66, 18, 1, 0, 0, 3, 35, 0, 5, 0, 130101]), '256/qkv_gate': ('lt', ['lt', 'tn', 66, 184, 4, 2, 0, 2, 35, 0, 5, 9437200, 130101]), '256/o': ('torch', []), '256/pooler': ('torch_transpose', ['nn']), '256/unpooler': ('torch', []), '1/router': ('router_interleaved', ['simt', 'fp32', 1, 1, 8192, 1, 4, 1]), '8/router': ('router_interleaved', ['interleaved', 'block128', 16, 64, 256, 32, 4, 2]), '16/router': ('router', ['dot', 16, 32, 256, 32, 4, 2, False]), '32/router': ('router', ['dot', 32, 64, 256, 32, 4, 2, False]), '128/router': ('router_interleaved', ['interleaved', 'block128', 32, 64, 128, 32, 4, 2]), '256/router': ('router_interleaved', ['interleaved', 'block128', 32, 64, 128, 16, 4, 2]), '1/qkv_gate': ('triton', ['gemv', 1, 8192, 8]), '1/o': ('triton', ['gemv', 2, 1024, 4]), '1/pooler': ('triton', ['gemv', 1, 4096, 8]), '1/unpooler': ('triton', ['gemv', 1, 4096, 8])}


def prepare(weights):
    ti__extension()
    lt__extension()
    return {
        'weights': weights,
        'lt': {name:lt_prepare(w) for name,w in weights.items() if name!='router'},
        'router': rt_prepare_router(weights['router']),
        'router_interleaved': ri_prepare_router(weights['router']),
        'packed': {'o':pa_prepare(weights['o'])},
        'transposed': {name:w.T.contiguous() for name,w in weights.items() if name!='router'},
    }




def kernel_fn(data,state):
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
    family,cfg=_CHOICES[str(bucket)+'/'+layer]
    if layer=='router':
        if family=='router_interleaved':return ri_run_router(x,state['router_interleaved'],cfg)
        return rt_run_router(x,state['router'],cfg)
    w=state['weights'][layer]
    if m>384:return torch.mm(x,w[:n].T)
    if family=='triton':return pr_run(x,w,n,cfg)
    if family=='cuda':return cu_run(x,w,n,cfg)
    if family=='tiny':return ti_run(x,w,n,cfg)
    if family=='tma':return tm_run(x,w,n,cfg)
    if family=='packed':return pa_run(x,state['packed'][layer],n,cfg)
    if family=='lt':return lt_run(x,state['lt'][layer],n,cfg)
    if family=='torch_transpose':return torch.mm(x,state['transposed'][layer][:,:n])
    return torch.mm(x,w[:n].T)
