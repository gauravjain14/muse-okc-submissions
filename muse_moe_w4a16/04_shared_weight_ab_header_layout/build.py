# okc: problem=muse_moe_w4a16
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


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


H = 4096
I = 4096
TOPK = 8
NLOCAL = 32
MAX_M = 512          # larger batches take the PyTorch path
ONE_TOKEN = True     # M = 1: the fused one-token kernel

NS_MAX = 4           # ring stages: as many as shared memory holds, at most this
CTR_WORDS = 16384    # workspace counters: [0] queue head, [1] down items done, [2] CTAs done, [3] gathered, [8..] per tile x k chunk
SMEM_MAX = 232448

CUDA_SRC = r"""
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
// consumer item events carry the token groups and phase (MOE_TRN) for dev/trace_ntl.py
#ifdef MOE_TRN
#define MOE_TRI(item) ((item) | (ntl << 16) | ((f.phase == 2) << 23))
#else
#define MOE_TRI(item) (item)
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
        else wgmma_n64<NT>(acc[r], Ab[r][jj], d0 + 2 * jj);
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
  u16* yh;  // [pmax][4096] BF16 down output of pairs whose down items span all of k (as SGLang keeps it)
};
// Constant-bank values used later in a call fetched at its start: a first use behind the weight
// stream waits microseconds for its constant-cache line (the one-token kernel's first stage took
// 3.5 us instead of 1).
DEVI void warm_consts() {
  u32 c0, c1, c2;
  asm volatile("ld.const.u32 %0, [c_shl];" : "=r"(c0));
  asm volatile("ld.const.u32 %0, [c_shl+4];" : "=r"(c1));
  asm volatile("ld.const.u32 %0, [c_s16];" : "=r"(c2));
  asm volatile("" :: "r"(c0), "r"(c1), "r"(c2));
}
#define WARM_P(f) asm volatile("" :: "l"((u64)(f)))
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
#define MOE_PREG 64
#endif
#ifndef MOE_CREG
#define MOE_CREG 216
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
  constexpr int STAGE = HASWG ? (WB + SB + XB + 1023) / 1024 * 1024 : (WB + SB + XB + 127) / 128 * 128;
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
#ifndef MOE_SSQ_NT
#define MOE_SSQ_NT 1
#endif
  constexpr bool SSQ = NT <= MOE_SSQ_NT;
#else
  constexpr bool SSQ = false;
#endif

  constexpr int NSIG = 4;
  extern __shared__ __align__(128) unsigned char smem[];
#ifndef MOE_NEWHDR
  // (the ring's barriers and stage records first: bs 8-16 measured 1-2% faster with this layout
  // than with them at the end of the header, v83 against v85-v103)
  static_assert(NS <= 4, "the header holds at most 4 ring stages");
  u64* bars = reinterpret_cast<u64*>(smem);                     // full[NS], empty[NS]
  u64* epi = reinterpret_cast<u64*>(smem + 64);                 // [NSIG] item outputs stored
  u64* sfree = reinterpret_cast<u64*>(smem + 96);               // [NSIG] signal slot free
  StageInfo* info = reinterpret_cast<StageInfo*>(smem + 128);   // [NS]
  int2* sinfo = reinterpret_cast<int2*>(smem + 320);            // [NSIG] (phase, tile)
  int* scal = reinterpret_cast<int*>(smem + 352);               // [16] ntiles, n1, total, P
  int* ftile = reinterpret_cast<int*>(smem + 416);              // [32] first tile of each expert
  int* off = reinterpret_cast<int*>(smem + 544);                // [32]
  u64* rdy = reinterpret_cast<u64*>(smem + 672);                // pair list (and gather) done
  u64* rtd = reinterpret_cast<u64*>(smem + 688);                // routing tables (tiles, scal) done
  u64* eis = reinterpret_cast<u64*>(smem + 696);                // early start: the first stages issued
  float* gsm = reinterpret_cast<float*>(smem + 768);            // [64] gate/up then down global scales x 2^8
#else
  static_assert(NS <= 5, "the header holds at most 5 ring stages");
  u64* epi = reinterpret_cast<u64*>(smem);                      // [NSIG] item outputs stored
  u64* sfree = reinterpret_cast<u64*>(smem + 32);               // [NSIG] signal slot free
  int2* sinfo = reinterpret_cast<int2*>(smem + 64);             // [NSIG] (phase, tile)
  u64* rdy = reinterpret_cast<u64*>(smem + 96);                 // pair list (and gather) done
  u64* rtd = reinterpret_cast<u64*>(smem + 104);                // routing tables (tiles, scal) done
  u64* eis = reinterpret_cast<u64*>(smem + 112);                // early start: the first stages issued
  int* scal = reinterpret_cast<int*>(smem + 128);               // [16] ntiles, n1, total, P
  int* ftile = reinterpret_cast<int*>(smem + 192);              // [32] first tile of each expert
  int* off = reinterpret_cast<int*>(smem + 320);                // [32]
  float* gsm = reinterpret_cast<float*>(smem + 448);            // [64] gate/up then down global scales x 2^8
  u64* bars = reinterpret_cast<u64*>(smem + 704);               // full[NS], empty[NS]
  StageInfo* info = reinterpret_cast<StageInfo*>(smem + 704 + 16 * NS);  // [NS]
#endif
  u32* bm = reinterpret_cast<u32*>(smem + 1024);                // [32][MW] token bitmaps per expert
  int* wpre = reinterpret_cast<int*>(bm + 32 * MW);             // [32][MW] exclusive popcount prefix
  int4* tiles = reinterpret_cast<int4*>(wpre + 32 * MW);        // [MAXT] (expert, first pair, tokens, -)
  u16* pair_tok = reinterpret_cast<u16*>(tiles + MAXT);         // [MAXP]
  signed char* lid = reinterpret_cast<signed char*>(pair_tok + MAXP);  // [MAXP] local expert of (t, slot)
  constexpr int ROUTE_BYTES = (32 * MW * 8 + MAXT * 16 + MAXP * 3 + 127) / 128 * 128;
  // (and the ring past an 8 KB gap in the two-k-warp configs: v83's layout exactly)
#ifndef MOE_NO_REDPAD
  constexpr int RPAD = (KW - 1) * RW * 1024 * NT;
#else
  constexpr int RPAD = 0;
#endif
  unsigned char* stages = smem + (1024 + ROUTE_BYTES + (2 * RPAD + 127) / 128 * 128 + 1023) / 1024 * 1024;

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
  for (int i = tid; i < 32 * MW; i += blockDim.x) bm[i] = 0u;
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
    mbar_init(smem_addr(eis), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  // With the early start (bs 1 config), routing runs on warps 0-7 (named barriers) and the
  // producer warpgroup does not wait for it; otherwise every warp takes part.
  // (not at bs 8-16: with this header layout their early start measured 1.3-1.5% slower)
#ifdef MOE_EARLY_ALL
  constexpr bool EARLY = NT == 1 && MW == 1;
#else
  constexpr bool EARLY = NT == 1 && KW == 4 && MW == 1;
#endif
  if (warp < 8 || !EARLY) {
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
      const int nspl = KSX > KS2 ? min(tot_t, (int)ceilf(MOE_SPLW * gridDim.x / (NRB2 * KS2))) : 0;
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
    for (int i = tid; i < M * 8; i += 256) {
      const int e = lid[i];
      if (e >= 0) {
        const int t = i >> 3;
        const u32 word = bm[e * MW + (t >> 5)];
        const int p = off[e] + wpre[e * MW + (t >> 5)] + __popc(word & ((1u << (t & 31)) - 1u));
        pair_tok[p] = (u16)t;
      }
    }
    consumer_sync();
    if (tid == 0) MOE_TR(0, 12, 0);
#ifdef MOE_TIMING
    if (tid == 0) reinterpret_cast<u32*>(P.ctr)[14420 + blockIdx.x * 4 + 3] = gtimer();
#endif
    if constexpr (GATHER) {
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
  int* cnt2 = P.ctr + 8192;  // per tile: down items done

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
    bool gathered = false, routed = false;
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
#ifndef MOE_NO_EIS
    if constexpr (EARLY) {  // the consumers start computing only now: their compute starved this issue
      __syncwarp();
      if (lane == 0) mbar_arrive(smem_addr(eis));
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
#ifndef MOE_NO_EIS
  if constexpr (EARLY) mbar_wait(smem_addr(eis), 0);
#endif

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
  u32 c_wait = 0, c_epi = 0, c_items = 0, c_comp = 0, c_stages = 0, c_red = 0, c_store = 0, c_sig = 0, wcomp = 0, c_loop = 0, c_rel = 0, c_istart = 0, c_iend = 0;
  const u32 c_loop0 = ctimer();
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
#ifdef MOE_TIMN  // per (phase, token groups): warp 0's compute and wait cycles, items, epilogue
    const u32 cc0 = c_comp, cw0 = c_wait;
#endif
#ifdef MOE_TIMING
    const u32 tl0 = ctimer();
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
    const u32 tis = ctimer();
    c_wait += tis - tw0;
    if (it == 0 && tid == 0) tdbg[15] = gtimer();  // first stage's data in shared memory
#endif
    f = info[s];
    if (f.item < 0) {
      post_signal(-1, 0, 0);
      break;
    }
    ntl = (f.n + 7) >> 3;
    if (tid == 0) MOE_TR(0, 1, MOE_TRI(f.item));
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
        MOE_T(const u32 tr0 = ctimer();)                                                                 \
        if (lane == 0) mbar_arrive(smem_addr(&bars[NS + s]));                                            \
        if (++s == NS) { s = 0; ph ^= 1; }                                                               \
        MOE_T(tw0 = ctimer(); c_rel += tw0 - tr0;)                                                       \
        mbar_wait(smem_addr(&bars[s]), ph);                                                              \
        MOE_T(c_wait += ctimer() - tw0;)                                                                 \
      }                                                                                                  \
      MOE_T(const u32 tc0 = ctimer(); if (st == 0) c_istart += tc0 - tis;)                               \
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
          default: if constexpr (NT >= 8) { MOE_RUN_ITEM(108) } break;
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
#ifdef MOE_TIMING
    c_iend += ctimer();  // (minus the epilogue start below)
#endif
#ifdef MOE_TIMN
    if (tid == 0) {
      int* tn = P.ctr + 16100 + ((f.phase - 1) * 9 + ntl) * 4;
      atomicAdd(tn, (int)(c_comp - cc0)); atomicAdd(tn + 1, (int)(c_wait - cw0)); atomicAdd(tn + 2, 1);
    }
#endif
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
          default: if constexpr (NT >= 8) stage_compute_wg<NT, 8>(acc, wbase, sclb, xaddr, T); break;
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
    c_iend -= te0;
#endif
    if (tid == 0) MOE_TR(0, 2, MOE_TRI(f.item));
    {
      constexpr int SP = RB + 4;  // floats per staged token row; the +4 keeps fragment stores conflict-free
      static_assert(KW * T * SP * 4 <= STAGE, "epilogue staging must fit in one stage slot");
      float* stg = reinterpret_cast<float*>(stages + s * STAGE);
      consumer_sync();  // every warp is done reading the slot
#pragma unroll
      for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < NT; ++b)
#pragma unroll
          for (int c = 0; c < 4; ++c)
            stg[(kw * T + b * 8 + 2 * tig + (c & 1)) * SP + 16 * (2 * rw + a) + g + 8 * (c >> 1)] = acc[a][b][c];
      consumer_sync();
#ifdef MOE_TIMING
      const u32 te2 = ctimer();
      c_red += te2 - te0;
#endif
      const int nn = f.n;
      if (f.phase == 1) {
        // gate rows 16 j + 0..7 and up rows 16 j + 8..15 of the item give intermediate 8 j + 0..7
        const float gsc = gsm[f.e];
        auto swiglu2 = [&](int tok, int row) -> u32 {  // rows row, row + 1 (gate) and +8 (up)
          float g0 = 0.f, g1 = 0.f, u0 = 0.f, u1 = 0.f;
#pragma unroll
          for (int k2 = 0; k2 < KW; ++k2) {
            const float* r = stg + (k2 * T + tok) * SP + row;
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
            *reinterpret_cast<uint4*>(dt + ((size_t)c64g * T + tok) * 64 + pc * 8) = make_uint4(h[0], h[1], h[2], h[3]);
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
                (((size_t)f.tile * (16 / KW) + ig / KST) * T + tok) * XP + (ig % KST) * 2) = make_uint4(h[0], h[1], h[2], h[3]);
          }
        }
      } else {
        const float gsc = gsm[32 + f.e];
        float* yb = P.ypart + (size_t)f.ks * P.pmax * 4096 + (size_t)f.p0 * 4096 + f.rb * RB;
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
              const float4 w = *reinterpret_cast<const float4*>(stg + (k2 * T + tok) * SP + 4 * r4);
              v.x += w.x; v.y += w.y; v.z += w.z; v.w += w.w;
            }
            v.x *= gsc; v.y *= gsc; v.z *= gsc; v.w *= gsc;
            if (full_k) {
              v = bf_round4(v);
              st_bf4(P.yh + (size_t)(f.p0 + tok) * 4096 + f.rb * RB + 4 * r4, v);
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
            if (live && (lane & 15) == 0) P.ssq[(size_t)(f.p0 + tok) * 64 + (f.rb * RB + 4 * r4) / 64] = sq;
          }
        }
} else {
        const bool full_k = (f.tile >= tsplit ? KSX : KS2) == 1;
        for (int o = tid; o < nn * (RB / 4); o += 256) {
          const int tok = o / (RB / 4), r4 = o - tok * (RB / 4);
          float4 v = *reinterpret_cast<const float4*>(stg + tok * SP + 4 * r4);
#pragma unroll
          for (int k2 = 1; k2 < KW; ++k2) {
            const float4 w = *reinterpret_cast<const float4*>(stg + (k2 * T + tok) * SP + 4 * r4);
            v.x += w.x; v.y += w.y; v.z += w.z; v.w += w.w;
          }
          v.x *= gsc; v.y *= gsc; v.z *= gsc; v.w *= gsc;
          if (full_k) st_bf4(P.yh + (size_t)(f.p0 + tok) * 4096 + f.rb * RB + 4 * r4, v);
          else *reinterpret_cast<float4*>(yb + (size_t)tok * 4096 + 4 * r4) = v;
        }
}
      }
#ifdef MOE_TIMING
      c_store += ctimer() - te2;
#endif
    }
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
    c_loop += ctimer() - tl0;
#endif
#ifdef MOE_TIMN
    if (tid == 0) atomicAdd(P.ctr + 16100 + ((f.phase - 1) * 9 + ntl) * 4 + 3, (int)(ctimer() - te0));
#endif
    if (tid == 0) MOE_TR(0, 3, MOE_TRI(f.item));
  }

#ifdef MOE_TIMING
  if (tid == 0) {
    tdbg[2] = gtimer(); tdbg[11] = ctimer() - tdbg[11];
    tdbg[4] = c_wait; tdbg[5] = c_epi; tdbg[6] = c_items; tdbg[9] = c_comp; tdbg[10] = c_stages;
    tdbg[12] = c_red; tdbg[13] = c_store; tdbg[14] = c_sig;
    reinterpret_cast<u32*>(P.ctr)[16384 + blockIdx.x] = c_loop;  // (timing builds: dev/abl.py sizes ctr)
    reinterpret_cast<u32*>(P.ctr)[16384 + 160 + blockIdx.x] = ctimer() - c_loop0;
    reinterpret_cast<u32*>(P.ctr)[16384 + 320 + blockIdx.x] = c_rel;
    reinterpret_cast<u32*>(P.ctr)[16384 + 480 + blockIdx.x] = c_istart;
    reinterpret_cast<u32*>(P.ctr)[16384 + 640 + blockIdx.x] = 0u - c_iend;
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
  const unsigned char* w13f; const unsigned char* s13f;  // MOE_W13F: per unit and stage, contiguous
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
#ifndef MOE_NO_WARMC
  if (warp == 0) {  // one warp per SM: the constant cache is the SM's
    warm_consts();
    WARM_P(P.x); WARM_P(P.ids); WARM_P(P.topk_w); WARM_P(P.gs13); WARM_P(P.gs2); WARM_P(P.nw1); WARM_P(P.w2f);
    WARM_P(P.s2f); WARM_P(P.out); WARM_P(P.ctr); WARM_P(P.yacc); WARM_P(P.tmap); WARM_P(P.local_start);
    WARM_P(P.trc); WARM_P(P.w13); WARM_P(P.s13);
  }
#endif
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
#ifdef MOE_W13F
          // the unit's gate/up rows stored stage-contiguous: one copy each for weights and scales
          if (lane == 0) bulk_g2s(dst, P.w13f + ((size_t)(e * 128 + j) * 4 + st) * 32768, 32768, full, pol_w);
          else if (lane == 1) bulk_g2s(dst + WB, P.s13f + ((size_t)(e * 128 + j) * 4 + st) * 4096, 4096, full, pol_w);
#elif !defined(MOE_FUSED_TMAP)
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
  // what the combine reads besides yacc, loaded at the start of this CTA's last unit so the loads
  // are back when it ends (cw0, cw1 stored to cw before the arrival)
  float4 wv[4];
  float cw0 = 0.f, cw1 = 0.f;
  bool pre = false;
  for (int u = blockIdx.x; u < nu; u += gridDim.x, ++un) {
    const int q = u >> 7;
    if (u + (int)gridDim.x >= nu) {
#pragma unroll
      for (int k = 0; k < 4; ++k) wv[k] = *reinterpret_cast<const float4*>(P.nw1 + (tid + 256 * k) * 4);
      if (tid < ne) {
        cw0 = P.topk_w[(slist >> (3 * tid)) & 7];
        cw1 = P.gs2[(int)((elist >> (5 * tid)) & 31)] * 256.f;
      }
      pre = true;
    }
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
#ifndef MOE_F_NOCOMP
      stage_compute<1, 1>(acc, acc2, sbase + (kw * 4 + rw * 2) * 2048 + lane * 16, sbase + WB + (kw * 4 + rw * 2) * 256 + lane * 8,
                          xres + (st * 4 + kw) * 512 + tig * 32, 0);
#endif
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
#ifndef MOE_F_NOCOMP
      {
        // the stage's words and scales first (a store to ystg ahead of a load would order it
        // after the store), then every row group's two steps, then the stores
        uint4 qv[8];
        uint2 sv[8];
#pragma unroll
        for (int rp = 0; rp < 8; ++rp) {
          qv[rp] = *reinterpret_cast<const uint4*>(sbase + (warp * 8 + rp) * 512 + lane * 16);
          sv[rp] = *reinterpret_cast<const uint2*>(sbase + WB + (warp * 8 + rp) * 64 + g * 8);
        }
        float co[16][2];
#pragma unroll
        for (int rp = 0; rp < 8; ++rp) {
#pragma unroll
          for (int r2 = 0; r2 < 2; ++r2) {
            const u32 sw = r2 ? sv[rp].y : sv[rp].x;
            const u32 w0 = r2 ? qv[rp].z : qv[rp].x, w1 = r2 ? qv[rp].w : qv[rp].y;
            float c[4] = {0.f, 0.f, 0.f, 0.f};
            u32 a0, a1, a2, a3;
            dequant(w0, scale_of<0, true>(sw), scale_of<1, true>(sw), a0, a1, a2, a3);
            mma16816(c, a0, a1, a2, a3, b00, b01);
            dequant(w1, scale_of<2, true>(sw), scale_of<3, true>(sw), a0, a1, a2, a3);
            mma16816(c, a0, a1, a2, a3, b10, b11);
            co[rp * 2 + r2][0] = c[0];
            co[rp * 2 + r2][1] = c[2];
          }
        }
        if (tig == 0) {
#pragma unroll
          for (int q = 0; q < 16; ++q) { yw[q * 16 + g] = co[q][0]; yw[q * 16 + g + 8] = co[q][1]; }
        }
      }
#endif
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
  if (!pre) {  // (no unit here)
#pragma unroll
    for (int k = 0; k < 4; ++k) wv[k] = *reinterpret_cast<const float4*>(P.nw1 + (tid + 256 * k) * 4);
    if (tid < ne) {
      cw0 = P.topk_w[(slist >> (3 * tid)) & 7];
      cw1 = P.gs2[(int)((elist >> (5 * tid)) & 31)] * 256.f;
    }
  }
  if (tid < ne) { cw[tid] = cw0; cw[8 + tid] = cw1; }
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

#define INST(RW, NT, KS2, NS, MW)                                                                   \
  extern "C" __global__ void __launch_bounds__(384, 1) moe_##RW##_##NT##_##KS2##_##NS##_##MW(Params p) { \
    moe_body<RW, NT, KS2, NS, MW>(p);                                                               \
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
        return (wb + sb + max(xb, 8 * nt * 512) + 1023) // 1024 * 1024
    return (wb + sb + xb + 127) // 128 * 128


def _smem_bytes(rw, nt, ns, mw):
    kw = 8 // rw
    t = 8 * nt
    maxp = mw * 32 * 8
    maxt = 32 + maxp // t
    route = (32 * mw * 8 + maxt * 16 + maxp * 3 + 127) // 128 * 128
    red = 0 if "#define MOE_NO_REDPAD" in CUDA_SRC else (kw - 1) * rw * 1024 * nt
    return (1024 + route + (2 * red + 127) // 128 * 128 + 1023) // 1024 * 1024 + ns * _stage_bytes(rw, nt)


# (largest M, (RW, NT, KS2)) per batch-size bucket; stages are as many as shared memory holds (<= 4).
BUCKETS = [
    (1, (2, 1, 1)),
    (16, (4, 1, 1)),
    (32, (8, 2, 1)),   # (re-swept on shared weights: +1-2% over (8, 4, 2) at bs 20-32)
    (128, (8, 4, 1)),
    (MAX_M, (8, 8, 1)),
]


def _mw(M):
    return 1 if M <= 32 else (2 if M <= 64 else (4 if M <= 128 else (8 if M <= 256 else 16)))


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
    for M in (1, 8, 16, 32, 33, 64, 65, 128, 129, 256, 257, MAX_M):
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
    ]


class _Params1C(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_void_p), ("ids", ctypes.c_void_p), ("topk_w", ctypes.c_void_p),
        ("gs13", ctypes.c_void_p), ("gs2", ctypes.c_void_p), ("nw1", ctypes.c_void_p),
        ("w2f", ctypes.c_void_p), ("s2f", ctypes.c_void_p),
        ("out", ctypes.c_void_p), ("ctr", ctypes.c_void_p), ("yacc", ctypes.c_void_p), ("tmap", ctypes.c_void_p),
        ("local_start", ctypes.c_int), ("trc", ctypes.c_void_p), ("w13", ctypes.c_void_p), ("s13", ctypes.c_void_p),
        ("w13f", ctypes.c_void_p), ("s13f", ctypes.c_void_p),
    ]


NS1 = 4   # one-token kernel ring stages (re-swept on shared weights: 4 is ~2% faster than 5)
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


def prepare(weights):
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
    # one-token kernel's gate/up rows, per 64-row unit j and stage (4 k chunks): [j][st][kc 4][rg 4][2 KB]
    if "#define MOE_W13F" in CUDA_SRC:
        w13f = w13p.view(count, 16, 128, 4, 512).permute(0, 2, 1, 3, 4).contiguous().view(count, -1)
        s13f = s13p.view(count, 16, 128, 4, 256).permute(0, 2, 1, 3, 4).contiguous().view(count, -1)
    else:
        w13f, s13f = w13p, s13p
    fast = _fast_scales_ok(s13p, s2p)
    global _FAST
    _FAST = fast
    mod = _module(fast)
    for cfg in _all_cfgs():
        mod.func(_name(cfg), _smem_bytes(*cfg[:2], cfg[3], cfg[4]))
    mod.func(f"moe1_{NS1}", 18432 + NS1 * 36864)
    torch.cuda.synchronize()
    return {
        "start": start, "count": count, "sms": torch.cuda.get_device_properties(dev).multi_processor_count,
        "w13": w13p, "s13": s13p, "gs13": weights["w13_gs"][start:start + count].float().contiguous(),
        "w2": w2p, "s2": s2p, "gs2": weights["w2_gs"][start:start + count].float().contiguous(),
        "nw1": (1.0 + weights["post_norm_w"].float()).contiguous(),
        "ctr": torch.zeros(CTR_WORDS, dtype=torch.int32, device=dev),
        "tmaps": tmaps, "tmap_off": tmap_off,
        "w2f": w2f, "s2f": s2f, "w13f": w13f, "s13f": s13f, "yacc": torch.zeros(8 * 8 * 4096, dtype=torch.float32, device=dev),
        "ctr1": torch.zeros(4, dtype=torch.int32, device=dev),
        "trc": torch.zeros(3 * 160 * 16, dtype=torch.int32, device=dev),
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


def kernel_fn(data, state):
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
                       state["start"], state["trc"].data_ptr(), state["w13"].data_ptr(), state["s13"].data_ptr(),
                       state["w13f"].data_ptr(), state["s13f"].data_ptr())
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
    p = _ParamsC(x.data_ptr(), ids.data_ptr(), tw.data_ptr(), M, state["start"],
                 state["w13"].data_ptr(), state["s13"].data_ptr(), state["gs13"].data_ptr(),
                 state["w2"].data_ptr(), state["s2"].data_ptr(), state["gs2"].data_ptr(), state["nw1"].data_ptr(),
                 inter.data_ptr(), ypart.data_ptr(), out.data_ptr(), state["ctr"].data_ptr(), pmax, xs.data_ptr(),
                 ssq.data_ptr(), state["tmaps"].data_ptr() + state["tmap_off"][rw], yh.data_ptr())
    _module().launch(_name(kc), state["sms"], 384, _smem_bytes(rw, nt, ns, mw), p)
    return out
