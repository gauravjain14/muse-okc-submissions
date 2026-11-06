# okc: problem=muse_bf16_proj

# Third-party code: each segment copied or adapted from another project is marked in place with SPDX-SnippetBegin /
# SPDX-SnippetEnd comments that give its source, copyright, license and our changes. Full license texts are in
# LICENSES/ and the third-party notices in NOTICE, both at the repository root.
"""muse_bf16_proj build (claude/t3/build_t3.py): base file + t3 agent kernels."""

import torch

# ======================================================================== base: int_v7
"""Fallback for every (layer, m) that no specialised kernel in this file claims: SGLang's own calls.

BF16 projections run F.linear in BF16 (cuBLAS, FP32 accumulation); the router keeps its FP32 weights and runs in
FP32, the reference precision. At up to 8 tokens the unpooler computes only this GPU's slice, the first
config["n"] rows of W.
"""
import torch
import torch.nn.functional as F


def _b7_prepare(weights):
    return {"w": dict(weights)}


def _b7_kernel_fn(data, state):
    x, cfg = data["x"], data["config"]
    w = state["w"][cfg["layer"]][: cfg["n"]]
    if w.dtype == torch.float32:
        return F.linear(x.float(), w)
    return F.linear(x, w)


# ======================================================================== module N: N/small.py
"""N / small: small-m kernels for the decode-1 / verify-1 loads (agent N, claude/t3/N, 2026-11-02).

    state = prepare_small(weights)          # weights: the problem's dict
    out   = run_small(x, config, state)     # -> [m, n] or None (None = let the base handle the shape)

Shapes taken (each beat bo6 in placement-averaged same-run A/Bs, see NOTES.md):
  o (K=1024, N=8192), m == 1 and m == 8 (M compiled in): "ta16" tensor-core tile GEMV on agent G's lossless F12 format:
     512 CTAs x 16 warps; CTA = one 16-row tile, warp = 64-k slice (2 x 16 B low bytes + 1 x 16 B nibble words per
     lane, all loads issued up front), x B fragments per warp (up to 8 tokens = one n8 tile), mma.sync m16n8k16 with
     2 accumulator chains, per-CTA smem reduction, weight pointer baked as an NVRTC constant; m=1: x loads first,
     __launch_bounds__(512, 2); m=8: weights first, __launch_bounds__(512, 2).
     Escapes (exponent outside the 16-exponent window, ~1e-4) come from per-row slots and are applied exactly:
     out = (sum_k x*fake - sum_esc x*2^-15) * 2^(E_lo-112) + sum_esc x*true  (bit-exact one-hot rows, no flush).
Everything else returns None.
"""

import ctypes
import struct

import torch

_ns_HEAD = r"""
#ifndef NK_LDF
#define NK_LDF 0
#endif
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
struct PM { const u16* x; u16* out; unsigned long long* tr; int M; int pad; };
typedef unsigned long long u64;
__device__ __forceinline__ u64 ck() { u64 t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)); return t; }
__shared__ u32 nk_sink[64];
__device__ __forceinline__ u64 ckd(u32 dep) {   // clock64 after dep is available (volatile STS consumes it first)
  *(volatile u32*)&nk_sink[threadIdx.x >> 5] = dep;
  u64 t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory"); return t;
}
__device__ __forceinline__ u64 gtime() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
// trace (ABL & 16): per CTA 64 u64: [0] start, [1] after barrier, [2] end, [3] end clock, [4] end globaltimer (tick edge),
// [8 + 4w + j] per warp w: j=0 first W quad landed, 1 all W landed, 2 x landed, 3 MMA done
#define TRW(w, j, v) if ((ABL & 16) && lane == 0) q.tr[(size_t)blockIdx.x * 192 + 8 + 4 * (w) + (j)] = (v)
__device__ __forceinline__ uint4 ldg128(const void* p) {
#if NK_LDF == 0
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
#elif NK_LDF == 1
  return __ldg(reinterpret_cast<const uint4*>(p));
#else
  uint4 v; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
#endif
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
__device__ __forceinline__ u16 f2bf(float v) { u16 o; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(o) : "f"(v)); return o; }
__device__ __forceinline__ void mma4(float* d, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// F12 decode (agent G): 8 weights from 2 lo words + 1 nibble word -> 4 bf16x2 words (w0w1, w2w3, w4w5, w6w7)
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
__device__ __forceinline__ float bf2f(u16 v) { return __uint_as_float(((u32)v) << 16); }

// ------------------------------------------------------------------------------------------------ bt: W as B (n8 tiles)
// XF: 0 = W loads then x, 1 = x loads first, 2 = first W quad, x, rest of W
template <unsigned long long WA, int TPC, int KC, int K, int E, unsigned RB, unsigned SC, unsigned ISC, int T16, int XF, int NACC, int ABL, int MC = 0>
__device__ __forceinline__ void bt_body(const PM& q) {
  constexpr int NW = TPC * KC, NT = NW * 32, KW = K / KC, NQ = KW / 64, NO = (NQ + 1) / 2;
  constexpr int CB = (NQ + NO) * 512;
  constexpr int MT = T16 ? 16 : 8;
  static_assert(KW % 64 == 0, "bt: KW multiple of 64");
  __shared__ float red[(ABL & 32) ? 1 : KC][TPC][MT][8];
  __shared__ float redt[(ABL & 32) ? TPC : 1][MT][8][KC + 1];
  __shared__ float escs[2 * MT * TPC * 8 * E];   // [A|B][tok][e][tile*8+row] (conflict-free row reads)
#define EIX(slot, tok) escs[((tok) * E + (slot) % E) * (TPC * 8) + (slot) / E]
#define EIXB(slot, tok) escs[MT * TPC * 8 * E + ((tok) * E + (slot) % E) * (TPC * 8) + (slot) / E]
  const u8* wb; asm("mov.u64 %0, %1;" : "=l"(wb) : "n"(WA));
  const float scale = __uint_as_float(SC), invscale = __uint_as_float(ISC);
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, g = lane >> 2, t = lane & 3;
  const int tile = wid % TPC, kc = wid / TPC;
  const u8* tb = wb + (size_t)(blockIdx.x * TPC + tile) * RB + kc * CB;
  const u32 M87 = 0x87878787u;
  u64 tstart = 0, gstart = 0;
  if (ABL & 16) { tstart = ck(); gstart = gtime(); }
  if ((ABL & 256) && wid == NW) {   // escape warp: slots (baked) first, then x gathers; off the MMA warps' path
    constexpr int NS = TPC * 8 * E, SPL = (NS + 31) / 32;
    u32 sl[SPL];
#pragma unroll
    for (int j = 0; j < SPL; ++j) {
      const int i = lane + 32 * j;
      sl[j] = i < NS ? ldg32(wb + (size_t)(blockIdx.x * TPC + i / (8 * E)) * RB + KC * CB + (i % (8 * E)) * 4) : 0u;
    }
    const int Me = MC ? MC : q.M;
#pragma unroll
    for (int j = 0; j < SPL; ++j) {
      const int i = lane + 32 * j;
      if (i < NS) {
        const float dl = esc_delta12(sl[j], invscale);
        const int k = sl[j] >> 16;
        const float tv = __uint_as_float(sl[j] << 16);
        for (int tok = 0; tok < Me; ++tok) {
          const float xv = dl != 0.f ? bf2f(ldx16(q.x + (size_t)tok * K + k)) : 0.f;
          EIX(i, tok) = xv * -3.0517578125e-05f; EIXB(i, tok) = xv * tv;
        }
      }
    }
    __syncthreads();
    return;
  }
  uint4 lo[NQ], nb[NO];
  uint4 xa[NQ][2], xc[T16 ? NQ : 1][2];
  constexpr int ns = (ABL & (1 | 256)) ? 0 : TPC * 8 * E;
  u32 es = 0;
  const u8* sb = wb + (size_t)(blockIdx.x * TPC + tid / (8 * E)) * RB + KC * CB + (tid % (8 * E)) * 4;
  int M = 0;
  auto load_x = [&]() {
    M = MC ? MC : q.M;
    const u16* xr = q.x + (size_t)g * K + kc * KW + 16 * t;
#pragma unroll
    for (int i = 0; i < NQ; ++i) {
      xa[i][0] = (ABL & 4) ? make_uint4(g, t, i, 1) : g < M ? ldx128(xr + 64 * i) : make_uint4(0, 0, 0, 0);
      xa[i][1] = (ABL & 4) ? make_uint4(g, t, i, 2) : g < M ? ldx128(xr + 64 * i + 8) : make_uint4(0, 0, 0, 0);
      if (T16) {
        xc[T16 ? i : 0][0] = g + 8 < M ? ldx128(xr + 8 * K + 64 * i) : make_uint4(0, 0, 0, 0);
        xc[T16 ? i : 0][1] = g + 8 < M ? ldx128(xr + 8 * K + 64 * i + 8) : make_uint4(0, 0, 0, 0);
      }
    }
  };
  if (XF == 1) load_x();
#pragma unroll
  for (int i = 0; i < NQ; ++i) {
    lo[i] = ldg128(tb + i * 512 + lane * 16);
    if (XF == 2 && i == 0) {
      nb[0] = ldg128(tb + NQ * 512 + lane * 16); load_x();
      if (ABL & 512) asm volatile("bar.sync 1, %0;" :: "n"(NT) : "memory");   // interleave issue across warps
    }
  }
#pragma unroll
  for (int i = (XF == 2 ? 1 : 0); i < NO; ++i) nb[i] = ldg128(tb + NQ * 512 + i * 512 + lane * 16);
  if (tid < ns) es = ldg32(sb);
  if (XF == 0) load_x();
  float acc[NACC][4];
#pragma unroll
  for (int a = 0; a < NACC; ++a) acc[a][0] = acc[a][1] = acc[a][2] = acc[a][3] = 0.f;
  if (ABL & 16) {
    TRW(wid, 0, ckd(lo[0].x ^ nb[0].x));
    u32 dd = 0;
#pragma unroll
    for (int i = 0; i < NQ; ++i) dd ^= lo[i].x ^ lo[i].w;
#pragma unroll
    for (int i = 0; i < NO; ++i) dd ^= nb[i].x ^ nb[i].w;
    TRW(wid, 1, ckd(dd));
    u32 dx = 0;
#pragma unroll
    for (int i = 0; i < NQ; ++i) dx ^= xa[i][0].x ^ xa[i][1].w;
    TRW(wid, 2, ckd(dx));
  }
#pragma unroll
  for (int i = 0; i < NQ; ++i) {
    const uint4 nq = nb[i >> 1];
    const u32 na = (i & 1) ? nq.z : nq.x, nbw = (i & 1) ? nq.w : nq.y;
    const uint4 d0 = decode12f(lo[i].x, lo[i].y, na, M87), d1 = decode12f(lo[i].z, lo[i].w, nbw, M87);
    const uint4 x0 = xa[i][0], x1 = xa[i][1];
    const uint4 y0 = T16 ? xc[T16 ? i : 0][0] : make_uint4(0, 0, 0, 0), y1 = T16 ? xc[T16 ? i : 0][1] : make_uint4(0, 0, 0, 0);
    float* c0 = acc[(2 * i) % NACC];
    float* c1 = acc[(2 * i + 1) % NACC];
    if (ABL & 2) {
      c0[0] += __uint_as_float((d0.x ^ d0.y ^ d1.x ^ d1.y) & 0x3fffffff) * __uint_as_float(x0.x ^ x1.y);
      c1[1] += __uint_as_float((d0.z ^ d0.w ^ d1.z ^ d1.w) & 0x3fffffff) * __uint_as_float(x0.z ^ x1.w);
    } else {
    mma4(c0, x0.x, y0.x, x0.y, y0.y, d0.x, d0.y);
    mma4(c1, x0.z, y0.z, x0.w, y0.w, d0.z, d0.w);
    mma4(c0, x1.x, y1.x, x1.y, y1.y, d1.x, d1.y);
    mma4(c1, x1.z, y1.z, x1.w, y1.w, d1.z, d1.w);
    }
  }
#pragma unroll
  for (int a = 1; a < NACC; ++a) { acc[0][0] += acc[a][0]; acc[0][1] += acc[a][1]; acc[0][2] += acc[a][2]; acc[0][3] += acc[a][3]; }
  TRW(wid, 3, ckd(__float_as_uint(acc[0][0])));
  if (tid < ns) {
    const float dl = esc_delta12(es, invscale);
    const int k = es >> 16;
    const float tv = __uint_as_float(es << 16);
    for (int tok = 0; tok < M; ++tok) {
      const float xv = dl != 0.f ? bf2f(ldx16(q.x + (size_t)tok * K + k)) : 0.f;
      EIX(tid, tok) = xv * -3.0517578125e-05f; EIXB(tid, tok) = xv * tv;
    }
  }
  if (ABL & 32) {
    if (g < M) { redt[(ABL & 32) ? tile : 0][g][2 * t][kc] = acc[0][0]; redt[(ABL & 32) ? tile : 0][g][2 * t + 1][kc] = acc[0][1]; }
    if (T16 && g + 8 < M) { redt[(ABL & 32) ? tile : 0][T16 ? g + 8 : 0][2 * t][kc] = acc[0][2]; redt[(ABL & 32) ? tile : 0][T16 ? g + 8 : 0][2 * t + 1][kc] = acc[0][3]; }
  } else {
  if (g < M) { red[(ABL & 32) ? 0 : kc][tile][g][2 * t] = acc[0][0]; red[(ABL & 32) ? 0 : kc][tile][g][2 * t + 1] = acc[0][1]; }
  if (T16 && g + 8 < M) { red[(ABL & 32) ? 0 : kc][tile][T16 ? g + 8 : 0][2 * t] = acc[0][2]; red[(ABL & 32) ? 0 : kc][tile][T16 ? g + 8 : 0][2 * t + 1] = acc[0][3]; }
  }
  __syncthreads();
  if ((ABL & 16) && tid == 0) { q.tr[(size_t)blockIdx.x * 192] = tstart; q.tr[(size_t)blockIdx.x * 192 + 1] = ck(); q.tr[(size_t)blockIdx.x * 192 + 5] = gstart; }
  if (ABL & 32) {   // tree: KC (== 32) lanes per output, shuffle reduction; output o = (tile, row, tok) tile-major
    static_assert(KC == 32 || !(ABL & 32), "tree reduction: KC == 32");
    const int nout = TPC * 8 * M;
    for (int o = wid; o < nout; o += NW) {
      const int tl = TPC == 1 ? 0 : o / (8 * M), rem = TPC == 1 ? o : o - tl * 8 * M;
      const int r = rem & 7, tok = rem >> 3;
      float s = redt[(ABL & 32) ? tl : 0][tok][r][lane];
      if (!(ABL & 1) && lane < E) s += EIX((tl * 8 + r) * E + lane, tok);
      s += __shfl_xor_sync(0xffffffffu, s, 16);
      s += __shfl_xor_sync(0xffffffffu, s, 8);
      s += __shfl_xor_sync(0xffffffffu, s, 4);
      s += __shfl_xor_sync(0xffffffffu, s, 2);
      s += __shfl_xor_sync(0xffffffffu, s, 1);
      if (lane == 0) {
        float sb = 0.f;
        for (int e = 0; e < ((ABL & 1) ? 0 : E); ++e) sb += EIXB((tl * 8 + r) * E + e, tok);
        q.out[(size_t)tok * (gridDim.x * TPC * 8) + (size_t)(blockIdx.x * TPC + tl) * 8 + r] = f2bf(fmaf(s, scale, sb));
      }
    }
  } else
  for (int o = tid; o < TPC * M * 8; o += NT) {
    const int tl = o / (M * 8), rem = o - tl * M * 8, tok = rem >> 3, r = rem & 7;
    float s = 0.f;
    if (ABL & 64) {
      float s4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int c = 0; c < KC; ++c) s4[c & 3] += red[(ABL & 32) ? 0 : c][tl][tok][r];
      s = (s4[0] + s4[1]) + (s4[2] + s4[3]);
    } else
#pragma unroll
    for (int c = 0; c < KC; ++c) s += red[(ABL & 32) ? 0 : c][tl][tok][r];
    float sb = 0.f;
#pragma unroll
    for (int e = 0; e < ((ABL & 1) ? 0 : E); ++e) { s += EIX((tl * 8 + r) * E + e, tok); sb += EIXB((tl * 8 + r) * E + e, tok); }
    q.out[(size_t)tok * (gridDim.x * TPC * 8) + (size_t)(blockIdx.x * TPC + tl) * 8 + r] = f2bf(fmaf(s, scale, sb));
  }
  if ((ABL & 16) && tid == 0) {
    q.tr[(size_t)blockIdx.x * 192 + 2] = ck();
    const u64 ga = gtime(); u64 gb, cb; do { cb = ck(); gb = gtime(); } while (gb == ga);
    q.tr[(size_t)blockIdx.x * 192 + 3] = cb; q.tr[(size_t)blockIdx.x * 192 + 4] = gb;
  }
}
#undef EIX
#undef EIXB

// ------------------------------------------------------------------------------------------------ ta: W as A (16-row tiles)
template <unsigned long long WA, int TPC, int KC, int K, int E, unsigned RB, unsigned SC, unsigned ISC, int NT8, int XF, int NACC, int ABL, int MC = 0>
__device__ __forceinline__ void ta_body(const PM& q) {
  constexpr int NW = TPC * KC, NT = NW * 32, KW = K / KC, SW = KW / 16, NP = SW / 2, NQ4 = SW / 4;
  constexpr int CB = (NP + NQ4) * 512;
  constexpr int MT = NT8 * 8;
  static_assert(SW % 4 == 0, "ta: steps per chunk multiple of 4");
  __shared__ float red[(ABL & 32) ? 1 : KC][TPC][MT][16];
  __shared__ float redt[(ABL & 32) ? TPC : 1][MT][16][KC + 1];
  __shared__ float escs[2 * MT * TPC * 16 * E];
#define EIX(slot, tok) escs[((tok) * E + (slot) % E) * (TPC * 16) + (slot) / E]
#define EIXB(slot, tok) escs[MT * TPC * 16 * E + ((tok) * E + (slot) % E) * (TPC * 16) + (slot) / E]
  const u8* wb; asm("mov.u64 %0, %1;" : "=l"(wb) : "n"(WA));
  const float scale = __uint_as_float(SC), invscale = __uint_as_float(ISC);
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, g = lane >> 2, t = lane & 3;
  const int tile = wid % TPC, kc = wid / TPC;
  const u8* tb = wb + (size_t)(blockIdx.x * TPC + tile) * RB + kc * CB;
  const u32 M87 = 0x87878787u;
  if ((ABL & 256) && wid == NW) {
    constexpr int NS = TPC * 16 * E, SPL = (NS + 31) / 32;
    u32 sl[SPL];
#pragma unroll
    for (int j = 0; j < SPL; ++j) {
      const int i = lane + 32 * j;
      sl[j] = i < NS ? ldg32(wb + (size_t)(blockIdx.x * TPC + i / (16 * E)) * RB + KC * CB + (i % (16 * E)) * 4) : 0u;
    }
    const int Me = MC ? MC : q.M;
#pragma unroll
    for (int j = 0; j < SPL; ++j) {
      const int i = lane + 32 * j;
      if (i < NS) {
        const float dl = esc_delta12(sl[j], invscale);
        const int k = sl[j] >> 16;
        const float tv = __uint_as_float(sl[j] << 16);
        for (int tok = 0; tok < Me; ++tok) {
          const float xv = dl != 0.f ? bf2f(ldx16(q.x + (size_t)tok * K + k)) : 0.f;
          EIX(i, tok) = xv * -3.0517578125e-05f; EIXB(i, tok) = xv * tv;
        }
      }
    }
    __syncthreads();
    return;
  }
  uint4 lo[NP], cd[NQ4], xb[NP][NT8];
  constexpr int ns = (ABL & (1 | 256)) ? 0 : TPC * 16 * E;
  u32 es = 0;
  const u8* sb = wb + (size_t)(blockIdx.x * TPC + tid / (16 * E)) * RB + KC * CB + (tid % (16 * E)) * 4;
  int M = 0;
  u64 tstart = 0, gstart = 0;
  if (ABL & 16) { tstart = ck(); gstart = gtime(); }
  auto load_x = [&]() {
    M = MC ? MC : q.M;
#pragma unroll
    for (int i = 0; i < NP; ++i)
#pragma unroll
      for (int n = 0; n < NT8; ++n) {
        const int tok = n * 8 + g;
        xb[i][n] = (ABL & 4) ? make_uint4(g, t, i, n) : tok < M ? ldx128(q.x + (size_t)tok * K + kc * KW + 32 * i + 8 * t) : make_uint4(0, 0, 0, 0);
      }
  };
  if (XF == 1) load_x();
#pragma unroll
  for (int i = 0; i < NP; ++i) lo[i] = ldg128(tb + i * 512 + lane * 16);
#pragma unroll
  for (int i = 0; i < NQ4; ++i) cd[i] = ldg128(tb + NP * 512 + i * 512 + lane * 16);
  if (tid < ns) es = ldg32(sb);
  if (XF != 1) load_x();
  float acc[NACC][NT8][4];
#pragma unroll
  for (int a = 0; a < NACC; ++a)
#pragma unroll
    for (int n = 0; n < NT8; ++n) acc[a][n][0] = acc[a][n][1] = acc[a][n][2] = acc[a][n][3] = 0.f;
  if (ABL & 16) {
    TRW(wid, 0, ckd(lo[0].x ^ cd[0].x));
    u32 dd = 0;
#pragma unroll
    for (int i = 0; i < NP; ++i) dd ^= lo[i].x ^ lo[i].w;
#pragma unroll
    for (int i = 0; i < NQ4; ++i) dd ^= cd[i].x ^ cd[i].w;
    TRW(wid, 1, ckd(dd));
    u32 dx = 0;
#pragma unroll
    for (int i = 0; i < NP; ++i) dx ^= xb[i][0].x ^ xb[i][0].w;
    TRW(wid, 2, ckd(dx));
  }
#pragma unroll
  for (int i = 0; i < NP; ++i) {
    const uint4 cq = cd[i >> 1];
    const u32 c0 = (i & 1) ? cq.z : cq.x, c1 = (i & 1) ? cq.w : cq.y;
    const uint4 a0 = decode12f(lo[i].x, lo[i].y, c0, M87), a1 = decode12f(lo[i].z, lo[i].w, c1, M87);
#pragma unroll
    for (int n = 0; n < NT8; ++n) {
      if (ABL & 2) {
        acc[(2 * i) % NACC][n][0] += __uint_as_float((a0.x ^ a0.y ^ a1.z ^ a1.w) & 0x3fffffff) * __uint_as_float(xb[i][n].x ^ xb[i][n].w);
      } else {
      mma4(acc[(2 * i) % NACC][n], a0.x, a0.y, a0.z, a0.w, xb[i][n].x, xb[i][n].y);
      mma4(acc[(2 * i + 1) % NACC][n], a1.x, a1.y, a1.z, a1.w, xb[i][n].z, xb[i][n].w);
      }
    }
  }
#pragma unroll
  for (int a = 1; a < NACC; ++a)
#pragma unroll
    for (int n = 0; n < NT8; ++n) { acc[0][n][0] += acc[a][n][0]; acc[0][n][1] += acc[a][n][1]; acc[0][n][2] += acc[a][n][2]; acc[0][n][3] += acc[a][n][3]; }
  TRW(wid, 3, ckd(__float_as_uint(acc[0][0][0])));
  if (tid < ns) {
    const float dl = esc_delta12(es, invscale);
    const int k = es >> 16;
    const float tv = __uint_as_float(es << 16);
    for (int tok = 0; tok < M; ++tok) {
      const float xv = dl != 0.f ? bf2f(ldx16(q.x + (size_t)tok * K + k)) : 0.f;
      EIX(tid, tok) = xv * -3.0517578125e-05f; EIXB(tid, tok) = xv * tv;
    }
  }
#pragma unroll
  for (int n = 0; n < NT8; ++n) {
    const int tk = n * 8 + 2 * t;
    if (ABL & 32) {
      if (tk < M) { redt[(ABL & 32) ? tile : 0][tk][g][kc] = acc[0][n][0]; redt[(ABL & 32) ? tile : 0][tk][g + 8][kc] = acc[0][n][2]; }
      if (tk + 1 < M) { redt[(ABL & 32) ? tile : 0][tk + 1][g][kc] = acc[0][n][1]; redt[(ABL & 32) ? tile : 0][tk + 1][g + 8][kc] = acc[0][n][3]; }
    } else {
    if (tk < M) { red[(ABL & 32) ? 0 : kc][tile][tk][g] = acc[0][n][0]; red[(ABL & 32) ? 0 : kc][tile][tk][g + 8] = acc[0][n][2]; }
    if (tk + 1 < M) { red[(ABL & 32) ? 0 : kc][tile][tk + 1][g] = acc[0][n][1]; red[(ABL & 32) ? 0 : kc][tile][tk + 1][g + 8] = acc[0][n][3]; }
    }
  }
  __syncthreads();
  if ((ABL & 16) && tid == 0) { q.tr[(size_t)blockIdx.x * 192] = tstart; q.tr[(size_t)blockIdx.x * 192 + 1] = ck(); q.tr[(size_t)blockIdx.x * 192 + 5] = gstart; }
  if (ABL & 32) {
    constexpr int LG = KC < 32 ? KC : 32, OPW = 32 / LG;
    const int nout = TPC * M * 16, m16 = M * 16;
    for (int base = wid * OPW; base < nout; base += NW * OPW) {
      const int o = base + lane / LG, c = lane % LG;
      const int tl = o / m16, rem = o - tl * m16, tok = rem >> 4, r = rem & 15;
      float s = 0.f;
      if (o < nout) {
#pragma unroll
        for (int cc = c; cc < KC; cc += LG) s += redt[(ABL & 32) ? tl : 0][tok][r][cc];
        for (int e = c; e < ((ABL & 1) ? 0 : E); e += LG) s += EIX((tl * 16 + r) * E + e, tok);
      }
#pragma unroll
      for (int off = LG / 2; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
      if (c == 0 && o < nout) {
        float sb = 0.f;
        for (int e = 0; e < ((ABL & 1) ? 0 : E); ++e) sb += EIXB((tl * 16 + r) * E + e, tok);
        q.out[(size_t)tok * (gridDim.x * TPC * 16) + (size_t)(blockIdx.x * TPC + tl) * 16 + r] = f2bf(fmaf(s, scale, sb));
      }
    }
  } else
  for (int o = tid; o < TPC * M * 16; o += NT) {
    const int tl = o / (M * 16), rem = o - tl * M * 16, tok = rem >> 4, r = rem & 15;
    float s = 0.f;
    if (ABL & 64) {
      float s4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int c = 0; c < KC; ++c) s4[c & 3] += red[(ABL & 32) ? 0 : c][tl][tok][r];
      s = (s4[0] + s4[1]) + (s4[2] + s4[3]);
    } else
#pragma unroll
    for (int c = 0; c < KC; ++c) s += red[(ABL & 32) ? 0 : c][tl][tok][r];
    float sb = 0.f;
#pragma unroll
    for (int e = 0; e < ((ABL & 1) ? 0 : E); ++e) { s += EIX((tl * 16 + r) * E + e, tok); sb += EIXB((tl * 16 + r) * E + e, tok); }
    q.out[(size_t)tok * (gridDim.x * TPC * 16) + (size_t)(blockIdx.x * TPC + tl) * 16 + r] = f2bf(fmaf(s, scale, sb));
  }
  if ((ABL & 16) && tid == 0) {
    q.tr[(size_t)blockIdx.x * 192 + 2] = ck();
    const u64 ga = gtime(); u64 gb, cb; do { cb = ck(); gb = gtime(); } while (gb == ga);
    q.tr[(size_t)blockIdx.x * 192 + 3] = cb; q.tr[(size_t)blockIdx.x * 192 + 4] = gb;
  }
}
#undef EIX
#undef EIXB
"""

_ns_FAKE12 = 112


# ------------------------------------------------------------------------------------------- format (agent G's F12)
def _ns__window(e, width):
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    return max(range(1, 256 - width), key=lambda q: int(cs[q + width] - cs[q]))


def _ns_compress12(w):
    """F12 planes: lo (fake low byte), nib (s << 3 | h, 3-bit h), esc, u (true bits), elo (window start)"""
    u = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    elo = _ns__window(e, 16)
    c = e - elo
    esc = (c < 0) | (c > 15)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m),
            "nib": torch.where(esc, torch.zeros_like(u), (s << 3) | (c >> 1)), "esc": esc, "u": u, "elo": elo}


def _ns_pack_nib8(nib):
    """nib [..., 8] (s << 3 | h) -> int32 [...]: byte b = h_b | s_b << 7 | h_{4+b} << 4 | s_{4+(b+3)%4} << 3"""
    nib = nib.to(torch.int64)
    s, h = nib >> 3, nib & 7
    word = torch.zeros(nib.shape[:-1], dtype=torch.int64, device=nib.device)
    for b in range(4):
        word |= (h[..., b] | (s[..., b] << 7) | (h[..., 4 + b] << 4) | (s[..., 4 + (b + 3) % 4] << 3)) << (8 * b)
    return torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)


def _ns__row_slots(f, E):
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


def _ns__bytes(t):
    return t.contiguous().view(torch.uint8)


def _ns__finish(chunks, slots, T, align=128):
    blob = torch.cat([chunks.reshape(T, -1), slots.reshape(T, -1)], 1)
    pad = (-blob.shape[1]) % align
    if pad:
        blob = torch.cat([blob, torch.zeros((T, pad), dtype=torch.uint8, device=blob.device)], 1)
    return blob.contiguous()


def _ns_pack_bt(w, KC, f=None):
    """8-row tiles, KC k-chunks: per tile [KC][lo NQ x 512 | nib NO x 512] + slots [8][E] -> dict"""
    n, k = w.shape
    T, KW = n // 8, k // KC
    NQ = KW // 64
    NO = (NQ + 1) // 2
    assert n % 8 == 0 and KW % 64 == 0
    f = f or _ns_compress12(w)
    dev = w.device
    E = max(1, int(f["esc"].sum(1).max()))
    ln = torch.arange(32, device=dev)
    g, t = ln // 4, ln % 4
    rows = torch.arange(T, device=dev)[:, None, None, None, None] * 8 + g[None, None, None, :, None]
    kk = (torch.arange(KC, device=dev)[:, None, None, None] * KW + torch.arange(NQ, device=dev)[None, :, None, None] * 64
          + 16 * t[None, None, :, None] + torch.arange(16, device=dev)[None, None, None, :])[None]
    lo = f["lo"][rows, kk].to(torch.uint8)                                    # [T, KC, NQ, 32, 16]
    nw = _ns_pack_nib8(f["nib"][rows, kk].view(T, KC, NQ, 32, 2, 8))              # [T, KC, NQ, 32, 2]
    if NO * 2 != NQ:
        nw = torch.cat([nw, torch.zeros((T, KC, 1, 32, 2), dtype=torch.int32, device=dev)], 2)
    nw = nw.view(T, KC, NO, 2, 32, 2).permute(0, 1, 2, 4, 3, 5).reshape(T, KC, NO, 32, 4)
    chunks = torch.cat([lo.reshape(T, KC, NQ * 512), _ns__bytes(nw).reshape(T, KC, NO * 512)], 2)
    sl = _ns__bytes(_ns__row_slots(f, E).view(T, 8 * E))
    blob = _ns__finish(chunks, sl, T)
    return {"w": blob.view(-1), "E": E, "RB": blob.shape[1], "n": n, "k": k, "elo": f["elo"], "KC": KC,
            "n_esc": int(f["esc"].sum()), "kind": "bt"}


def _ns_pack_ta(w, KC, f=None):
    """16-row tiles (G's tc fragment order), KC k-chunks: per tile [KC][lo NP x 512 | nib NQ4 x 512] + slots [16][E]"""
    n, k = w.shape
    T, KW = n // 16, k // KC
    SW = KW // 16
    NP, NQ4 = SW // 2, SW // 4
    assert n % 16 == 0 and SW % 4 == 0
    f = f or _ns_compress12(w)
    dev = w.device
    E = max(1, int(f["esc"].sum(1).max()))
    ln = torch.arange(32, device=dev)
    g, t = ln // 4, ln % 4
    rr = torch.stack([g, g, g + 8, g + 8, g, g, g + 8, g + 8], 1)                                  # [32, 8]
    ko = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3], device=dev)
    st = torch.arange(SW, device=dev)
    k0 = (torch.arange(KC, device=dev)[:, None, None] * KW + (32 * (st // 2) + 4 * (st % 2))[None, :, None]
          + 8 * t[None, None, :])                                                                    # [KC, SW, 32]
    kk = (k0[..., None] + ko)[None]                                                                  # [1, KC, SW, 32, 8]
    rows = torch.arange(T, device=dev)[:, None, None, None, None] * 16 + rr[None, None, None]       # [T, 1, 1, 32, 8]
    lo = f["lo"][rows, kk].to(torch.uint8)                                                           # [T, KC, SW, 32, 8]
    lo = lo.view(T, KC, NP, 2, 32, 8).permute(0, 1, 2, 4, 3, 5).reshape(T, KC, NP * 512)
    nw = _ns_pack_nib8(f["nib"][rows, kk])                                                               # [T, KC, SW, 32]
    nw = nw.view(T, KC, NQ4, 4, 32).permute(0, 1, 2, 4, 3).reshape(T, KC, NQ4, 32, 4)
    chunks = torch.cat([lo, _ns__bytes(nw).reshape(T, KC, NQ4 * 512)], 2)
    sl = _ns__bytes(_ns__row_slots(f, E).view(T, 16 * E))
    blob = _ns__finish(chunks, sl, T)
    return {"w": blob.view(-1), "E": E, "RB": blob.shape[1], "n": n, "k": k, "elo": f["elo"], "KC": KC,
            "n_esc": int(f["esc"].sum()), "kind": "ta"}


def _ns__f32bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


# ------------------------------------------------------------------------------------------- NVRTC module
class _ns_Mod:
    def __init__(self, src, name="nk.cu"):
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
        self.cubin = cubin
        err, self.module = driver.cuModuleLoadData(cubin)
        if int(err):
            raise RuntimeError(f"cuModuleLoadData: {err}")
        self.funcs = {}

    def func(self, name):
        f = self.funcs.get(name)
        if f is None:
            err, f = self.d.cuModuleGetFunction(self.module, name.encode())
            if int(err):
                raise RuntimeError(f"cuModuleGetFunction({name}): {err}")
            self.funcs[name] = f
        return f

    def attrs(self, name):
        d = self.d
        f = self.func(name)
        out = {}
        for k in ("NUM_REGS", "LOCAL_SIZE_BYTES", "SHARED_SIZE_BYTES"):
            _, v = d.cuFuncGetAttribute(getattr(d.CUfunction_attribute, "CU_FUNC_ATTRIBUTE_" + k), f)
            out[k] = v
        return out

    def launch(self, name, grid, block, args, smem=0):
        f = self.func(name)
        raw = ctypes.create_string_buffer(len(args) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, args, len(args))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = self.d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = self.d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        if int(err):
            raise RuntimeError(f"launch {name}: {err}")


# ------------------------------------------------------------------------------------------- configs
# cfg: (kind, KC, TPC, T16/NT8, XF, NACC, MB)
def _ns_kname(tag, cfg):
    return f"nk_{tag}_" + "_".join(str(v) for v in cfg[1:])


def _ns_kernel_src(tag, ent, cfg):
    kind, KC, TPC, TT, XF, NACC, MB = cfg[:7]
    ABL = cfg[7] if len(cfg) > 7 else 0
    MC = cfg[8] if len(cfg) > 8 else 0
    sc = 2.0 ** (ent["elo"] - _ns_FAKE12)
    nt = TPC * KC * 32 + (32 if ABL & 256 else 0)
    tpl = (f"{ent['w'].data_ptr()}ull,{TPC},{KC},{ent['k']},{ent['E']},{ent['RB']}u,{_ns__f32bits(sc)}u,{_ns__f32bits(1.0 / sc)}u,"
           f"{TT},{XF},{NACC},{ABL},{MC}")
    lb = f"{nt}, {MB}" if MB else f"{nt}"
    body = "bt_body" if kind == "bt" else "ta_body"
    return (f'extern "C" __global__ void __launch_bounds__({lb}) {_ns_kname(tag, cfg)}(const __grid_constant__ PM q) '
            f'{{ {body}<{tpl}>(q); }}\n')


class _ns_NK:
    """variants: {tag: (w [n, k] bf16, cfg)} -> compiled kernels; run(tag, x) -> out [m, n] bf16"""

    def __init__(self, variants, ldf=0):
        self.ents, self.cfgs, packs = {}, {}, {}
        src = f"#define NK_LDF {ldf}\n" + _ns_HEAD
        for tag, (w, cfg) in variants.items():
            kind, KC = cfg[0], cfg[1]
            key = (id(w), kind, KC)
            if key not in packs:
                packs[key] = (_ns_pack_bt if kind == "bt" else _ns_pack_ta)(w.contiguous(), KC)
            ent = packs[key]
            self.ents[tag], self.cfgs[tag] = ent, cfg
            src += _ns_kernel_src(tag, ent, cfg)
        self.src = src
        self.mod = _ns_Mod(src)

    def grid(self, tag):
        ent, cfg = self.ents[tag], self.cfgs[tag]
        rows = (8 if cfg[0] == "bt" else 16) * cfg[2]
        abl = cfg[7] if len(cfg) > 7 else 0
        return ent["n"] // rows, cfg[1] * cfg[2] * 32 + (32 if abl & 256 else 0)

    def run(self, tag, x, out=None, tr=None):
        ent, cfg = self.ents[tag], self.cfgs[tag]
        m = x.shape[0]
        if out is None:
            out = torch.empty((m, ent["n"]), dtype=torch.bfloat16, device=x.device)
        grid, nt = self.grid(tag)
        self.mod.launch(_ns_kname(tag, cfg), grid, nt, struct.pack("<QQQii", x.data_ptr(), out.data_ptr(), tr.data_ptr() if tr is not None else 0, m, 0))
        return out


# ------------------------------------------------------------------------------------------- deliverable API
# (kind, KC, TPC, NT8/T16, XF, NACC, MB[, ABL]) per shape key
# o m=1: x-first, 2 chains, min-blocks 2, M = 1 compiled in; o m=8: W-first, 2 chains, min-blocks 2, M = 8 compiled in
_ns_SMALL_CFG = {"o1": ("ta", 16, 1, 1, 1, 2, 2, 0, 1), "o8": ("ta", 16, 1, 1, 0, 2, 2, 0, 8)}
_ns_SMALL_W = {"o1": lambda w: w["o"], "o8": lambda w: w["o"]}


def _ns_prepare_small(weights):
    st = {"nk": None, "keys": set()}
    try:
        var = {k: (_ns_SMALL_W[k](weights).contiguous(), cfg) for k, cfg in _ns_SMALL_CFG.items()}
        nkm = _ns_NK(var)
        st["keys"] = {k for k in var if nkm.ents[k]["E"] <= 16}   # pathological escape counts: leave to the base
        st["nk"] = nkm
    except Exception as e:  # noqa: BLE001  (never break the submission: the base handles the shape)
        print(f"[small] prepare failed: {e}")
    return st


def _ns__key(layer, m, n):
    if layer == "o" and n == 8192:
        return {1: "o1", 8: "o8"}.get(m)
    return None


def _ns_run_small(x, config, state):
    nkm = state.get("nk")
    if nkm is None:
        return None
    key = _ns__key(config["layer"], config["m"], config["n"])
    if key is None or key not in state["keys"]:
        return None
    if not x.is_contiguous() or x.data_ptr() % 16 or x.dtype != torch.bfloat16:
        return None
    return nkm.run(key, x)


# ======================================================================== module R: R/router3.py
"""Router projection: out[M,256] fp32 = x[M,8192] bf16 @ W[256,8192]^T, W fp32 run as a BF16 hi + lo pair, FP32 accumulation.

One kernel per call. CTA = (token tile tt, row tile rt, k split ks). Weights (hi, lo tiles in mma B-fragment order) stream with
cp.async.bulk, x tiles with a 2D TMA (128B swizzle, rows >= M zero-filled), mma.sync m16n8k16 (x = A, W rows = n8 B).
Split-K partials: red.global.add.v4.f32 into a self-resetting fp32 workspace; the last CTA of an output tile (counter)
writes the output and zeroes the workspace + counter for the next call.
"""
import ctypes
import struct

import torch

_r3_SRC = r"""
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
template <int TT, int RR, int KK, int KS, int NS, int WK, int WG, int FL, u64 WA = 0, int ESB = 0, u32 B2I = 0>
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
  u64 wa = 0; if (WA) asm("mov.u64 %0, %1;" : "=l"(wa) : "n"(WA));
  const unsigned char* wblk = (WA ? (const unsigned char*)wa : p.w) + (size_t)(rt * S + ks) * (RR * KK * (CMP ? 3 : 4) + ESB);
  const unsigned char* wsrc = wblk + ESB;   // ESB: escape record (int n; n x (row << 16 | k, delta)) before the stages
  const u32 base2 = B2I ? B2I : (u32)p.base2;
  const u32 zgen = (FL & 256) ? *(volatile const u32*)(p.cnt + 64 + b) : 0u;
  __shared__ int esc_e[CMP ? 32 : 1];
  __shared__ float esc_d[CMP ? 32 : 1];
  __shared__ float corr[CMP ? TT * RR : 1];
  __shared__ int esc_n;
  if (CMP)
    for (int i = threadIdx.x; i < TT * RR; i += blockDim.x) corr[i] = 0.f;

  if (wid == NCW && lane == 0) {
    if (FL & 512) asm volatile("prefetch.tensormap [%0];" :: "l"(&p.tmx) : "memory");
    for (int s = 0; s < NS; ++s) { mbar_init(bars + 8 * s, 1); mbar_init(bars + 8 * (NS + s), NCW); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (FL & 512) {   // W stage 0 right away (pointer compiled in), x of stage 0 after it in the producer loop
      mbar_expect_arrive(bars, XB + WB + (ESB ? ESB : 0));
      bulk_g2s(ring + XB, wsrc, WB, bars);
    }
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
        if (!((FL & 512) && s == 0)) {
          mbar_expect_arrive(bars + 8 * slot, XB + WB + (ESB && s == 0 ? ESB : 0));
          bulk_g2s(st + XB, wsrc + (size_t)s * WB, WB, bars + 8 * slot);
        }
        if (ESB && s == 0) bulk_g2s(bars + 512, wblk, ESB, bars);
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
  if (CMP && !ESB && wid == 0) {   // this CTA's escapes; issued after the producer started, used after the main loop
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
    if (CMP && ESB && wid == 0) {   // escape corrections (record in smem since stage 0; x tiles still in the ring)
      const int* er = (const int*)(sm + 512);
      const int n = er[0], tmx = min(TT, M - tt * TT);
      for (int i = lane; i < n * tmx; i += 32) {
        const int e = i / tmx, t = i % tmx;
        const int rl = er[4 + 2 * e] >> 16, kl = er[4 + 2 * e] & 0xFFFF;
        const int s_ = kl / KS, kin = kl % KS, bx = kin >> 6, ki = kin & 63;
        const unsigned char* xp = sm + 1024 + (s_ % NS) * SB + bx * (TT * 128) + t * 128 + ((((ki >> 3) ^ (t & 7)) << 4) | ((ki & 7) * 2));
        const float xv = __uint_as_float((u32)(*(const u16*)xp) << 16);
        asm volatile("red.global.add.f32 [%0], %1;" :: "l"(&p.out[(size_t)(tt * TT + t) * 256 + rt * RR + rl]), "f"(xv * __int_as_float(er[5 + 2 * e])) : "memory");
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) p.cnt[64 + b] = zgen + ZCTAS;
    return;
  }
  __syncthreads();  // all stages landed and consumed
  TRT(3);
  if (CMP && ESB) {
    const int* er = (const int*)(sm + 512);
    if (threadIdx.x < 32) {
      const int n = min(er[0], 32);
      if ((int)threadIdx.x < n) { esc_e[threadIdx.x] = er[4 + 2 * threadIdx.x]; esc_d[threadIdx.x] = __int_as_float(er[5 + 2 * threadIdx.x]); }
      if (threadIdx.x == 0) esc_n = n;
    }
    __syncthreads();
  }
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
template <int M, int NCH, u64 WA = 0>
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
    const float* wsrc = (WA ? (const float*)WA : p.w) + (size_t)blockIdx.x * (2 * 8192);
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

// w prelaid [128 CTAs][NCH][2 rows][CK] fp32; x chunk c = TMA box [256, CK/256, M] -> smem [M][CK] bf16
template <int M, int NCH, u64 WA = 0>
__device__ __forceinline__ void rs1r_body(const PS& p) {   // 1 row per CTA (256 CTAs, 2 per SM)
  constexpr int CK = 8192 / NCH;
  constexpr int WCB = CK * 4, XCB = M * CK * 2;
  constexpr int HALF = CK / 2;                 // thread i: k = i*4 + j*HALF... (256 threads, 4 k each per half)
  static_assert(CK == 2048, "256 threads x 8 k");
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  unsigned char* sm = (unsigned char*)(((size_t)sm_raw + 1023) & ~(size_t)1023);
  const u32 bars = smem_u32(sm);
  const u32 wsm = bars + 1024;
  const u32 xsm = wsm + NCH * WCB;
  __shared__ float red[8][M];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  if (tid == 0) {
    for (int c = 0; c < NCH; ++c) mbar_init(bars + 8 * c, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    u64 wa = 0; if (WA) asm("mov.u64 %0, %1;" : "=l"(wa) : "n"(WA));
    const float* wsrc = (WA ? (const float*)wa : p.w) + (size_t)blockIdx.x * 8192;
#pragma unroll 1
    for (int c = 0; c < NCH; ++c) {
      mbar_expect_arrive(bars + 8 * c, WCB + XCB);
      bulk_g2s(wsm + c * WCB, wsrc + (size_t)c * CK, WCB, bars + 8 * c);
      tma3d(xsm + c * XCB, &p.tmx, 0, c * (CK / 256), 0, bars + 8 * c);
    }
  }
  __syncthreads();
  float a0[M], a1[M];
#pragma unroll
  for (int t = 0; t < M; ++t) { a0[t] = 0.f; a1[t] = 0.f; }   // a1: second accumulator of the same row (ILP)
#pragma unroll 1
  for (int c = 0; c < NCH; ++c) {
    mbar_wait(bars + 8 * c, 0);
    const unsigned char* wc = sm + 1024 + c * WCB;
    const unsigned char* xc = sm + 1024 + NCH * WCB + c * XCB;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int k = j * HALF + tid * 4;
      const float4 w0 = *(const float4*)(wc + k * 4);
#pragma unroll
      for (int t = 0; t < M; ++t) {
        const uint2 xv = *(const uint2*)(xc + (t * CK + k) * 2);
        const float x0 = __uint_as_float(xv.x << 16), x1 = __uint_as_float(xv.x & 0xffff0000u);
        const float x2 = __uint_as_float(xv.y << 16), x3 = __uint_as_float(xv.y & 0xffff0000u);
        a0[t] = fmaf(x0, w0.x, a0[t]); a1[t] = fmaf(x1, w0.y, a1[t]); a0[t] = fmaf(x2, w0.z, a0[t]); a1[t] = fmaf(x3, w0.w, a1[t]);
      }
    }
  }
#pragma unroll
  for (int t = 0; t < M; ++t) {
    a0[t] += a1[t];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a0[t] += __shfl_xor_sync(0xffffffffu, a0[t], o);
  }
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < M; ++t) red[wid][t] = a0[t];
  }
  __syncthreads();
  if (tid < M) {
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) v += red[w][tid];
    p.out[(size_t)tid * 256 + blockIdx.x] = v;
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

struct PR { const float4* ws; float4* out; long long n4; long long slab4; long long S; };
template <int S>
__device__ __forceinline__ void rred_body(const PR& p) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const long long i = blockIdx.x * 128LL + threadIdx.x;
  if (i >= p.n4) return;
  float4 u[S];
#pragma unroll
  for (int s = 0; s < S; ++s) u[s] = __ldcg(p.ws + s * p.slab4 + i);
#pragma unroll
  for (int w = 1; w < S; w <<= 1)
#pragma unroll
    for (int s = 0; s + w < S; s += 2 * w) { u[s].x += u[s + w].x; u[s].y += u[s + w].y; u[s].z += u[s + w].z; u[s].w += u[s + w].w; }
  p.out[i] = u[0];
}
extern "C" __global__ void __launch_bounds__(128) rred16(const PR p) { rred_body<16>(p); }
extern "C" __global__ void __launch_bounds__(128) rred32(const PR p) { rred_body<32>(p); }
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
template <int TN, int KK, int NS, int MC, int ZF, int CA = 0, int ZA = 0, u64 WA = 0, int WS = 0>
__device__ __forceinline__ void rw_body(const PW& p) {
  if (WS == 1) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
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
  static_assert(MC != 2 || NS >= NSTG, "x multicast: all stages resident (no slot reuse across the pair)");
  const int ks = MC == 2 ? (b / 2) % S : MC ? (b / 2) % S : b % S;
  const int tt = MC == 2 ? (b / (2 * S)) % NTT : MC ? b % 2 : (b / S) % NTT;
  const int rt = MC == 2 ? (b / (2 * S * NTT)) * 2 + (b & 1) : MC ? b / (2 * S) : b / (S * NTT);
  const unsigned char* wsrc = (WA ? (const unsigned char*)WA : p.w) + (size_t)(rt * S + ks) * NSTG * AB;
  TR(0);
  if (wid == 4 && lane == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&p.tmx) : "memory");
    for (int s = 0; s < NS; ++s) { mbar_init(bars + 8 * s, 1); mbar_init(bars + 8 * (NS + s), MC == 1 ? 8 : 4); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (MC) asm volatile("barrier.cluster.arrive.relaxed.aligned;\nbarrier.cluster.wait.aligned;" ::: "memory");
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
        if (MC == 1) bulk_g2s_mc(st + tt * 8192, wsrc + (size_t)s * AB + tt * 8192, 8192, bars + 8 * slot, (unsigned short)3);
        else bulk_g2s(st, wsrc + (size_t)s * AB, AB, bars + 8 * slot);
        if (MC == 2) {   // half of the x tile (TN/2 tokens), multicast to both row tiles of the pair
          const int hr = b & 1;
          asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;"
                       :: "r"(st + AB + hr * (TN / 2) * 128), "l"(&p.tmx), "r"(ks * KK + s * 64), "r"(tt * TN + hr * (TN / 2)),
                          "r"(bars + 8 * slot), "h"((unsigned short)3) : "memory");
        } else tma2d(st + AB, &p.tmx, ks * KK + s * 64, tt * TN, bars + 8 * slot);
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
        if (MC == 1) mbar_arrive_remote(bars + 8 * (NS + slot), (u32)(tt ^ 1));
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
  else if (WS != 1) asm volatile("griddepcontrol.wait;" ::: "memory");
  TR(5);
  const int tmax = min(TN, M - tt * TN);
  if (WS == 2) {   // TMA bulk reduction: one cp.reduce.async.bulk (64 floats = 256 B) per token row
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    for (int t = threadIdx.x; t < tmax; t += blockDim.x)
      asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], 256;"
                   :: "l"(&p.out[(size_t)(tt * TN + t) * 256 + rt * 64]), "r"(smem_u32(&tr[t * TS])) : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
  } else if (WS) {   // split-K partial -> workspace slab ks (plain stores); a PDL-launched rred kernel sums the slabs
    float* ws = (float*)p.zc;
    for (int e = threadIdx.x; e < tmax * 16; e += blockDim.x) {
      const int t = e >> 4, r4 = (e & 15) * 4;
      *(float4*)&ws[((size_t)ks * NTT * TN + tt * TN + t) * 256 + rt * 64 + r4] = *(float4*)&tr[t * TS + r4];
    }
  } else
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

_r3_RK_SRC = r"""
// ---- rk: cluster split-K for M <= 64. Cluster q (C CTAs) owns rows [r0, r1); rank j owns k in [j*KC, (j+1)*KC).
//      Producer: W stage 0 (pointer compiled in) at once, then x rows (plain bulk copies into padded smem rows, x pointer
//      from params), then the remaining W stages. mma.sync m16n8k16: A = x (ldmatrix), B = W n8 row groups (hi, lo;
//      CMP: 24-bit codes decoded in registers), separate hi/lo accumulators for ILP. Partials reduced inside the cluster:
//      token t owned by rank t % C, pushed with st.async (mbarrier complete_tx, no fences); owner sums, plain stores.
struct __align__(64) PK { const unsigned char* x; float* out; long long M; unsigned long long* trc; };
__device__ __forceinline__ u32 mapa_u32(u32 a, u32 rank) {
  u32 r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank)); return r;
}
__device__ __forceinline__ void st_async_f32(u32 a, float v, u32 bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];" :: "r"(a), "f"(v), "r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait_cl(u32 b, int ph) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}\n"
               :: "r"(b), "r"(ph) : "memory");
}
#ifndef SPINW
#define SPINW 1
#endif
__device__ __forceinline__ void mbar_spin(u32 b, int ph) {
  if (SPINW) asm volatile("{\n .reg .pred p;\n W: mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}\n" :: "r"(b), "r"(ph) : "memory");
  else mbar_wait(b, ph);
}
__device__ __forceinline__ void mbar_spin_cl(u32 b, int ph) {
  if (SPINW) asm volatile("{\n .reg .pred p;\n W: mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}\n" :: "r"(b), "r"(ph) : "memory");
  else mbar_wait_cl(b, ph);
}
#ifndef RTRACE
#define RTRACE 0
#endif
__device__ __forceinline__ u64 clk() { u64 t = 0; if (RTRACE) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)); return t; }
// FL: 1 skip x, 2 skip compute, 4 skip cluster reduction (rank 0 stores its partial; wrong values), 8 x before W stage 0
template <int C, int Q, int MT, int NSTG, int CMP, u64 WA, int ESB, int NCW, u64 EDUM2, u32 BASE2, int FL = 0>
__device__ __forceinline__ void rk_body(const PK& p) {
  constexpr int KC = 8192 / C, KS = KC / NSTG, KTS = KS / 32;
  constexpr int UPW = KTS * 2 / NCW;                        // k16 units (k32 tile halves) per warp per stage
  static_assert(UPW >= 1 && KTS * 2 == UPW * NCW, "consumer warps x k16 units");
  constexpr int PW = NCW;                                   // producer warp index
  constexpr int TT = MT * 16, XS = KC * 2 + 16;
  constexpr int NRMAX = (256 + Q - 1) / Q, NG = (NRMAX + 7) / 8;
  constexpr int TB = CMP ? 96 : 128;                        // bytes per row per k32 tile
  constexpr int WMAX = ESB + KC / 32 * TB * NRMAX;
  constexpr int SCS = NG * 8 + 4;                           // scratch row stride (floats)
  constexpr int MO = (TT + C - 1) / C;                      // max tokens owned per rank
  constexpr int WOFF = 1024;
  constexpr int XOFF = WOFF + ((WMAX + 1023) / 1024) * 1024;
  constexpr int SOFF = XOFF + TT * XS;                      // slots [C][MO][NRMAX] f32
  constexpr int COFF = SOFF + C * MO * NRMAX * 4;           // corr [TT][NRMAX] f32
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  unsigned char* sm = (unsigned char*)(((size_t)sm_raw + 1023) & ~(size_t)1023);
  const u32 bars = smem_u32(sm);                            // wbar[NSTG], xbar, rbar
  const u32 xbar = bars + 8 * NSTG, rbar = xbar + 8;
  const u32 wsm = bars + WOFF + ESB, xsm = bars + XOFF;   // escape record (CMP) at bars + WOFF
  float* slots = (float*)(sm + SOFF);
  float* corr = (float*)(sm + COFF);
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int q = blockIdx.x / C;
  u32 j; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(j));
  const int r0 = q * 256 / Q, nr = (q + 1) * 256 / Q - r0;
  const u32 stb = (u32)(KTS * TB * nr);                    // W bytes per stage
  u64 wa; asm("mov.u64 %0, %1;" : "=l"(wa) : "n"(WA));   // 64-bit immediate (NVRTC dropped high bits of folded pointer constants)
  const unsigned char* wsrc = (const unsigned char*)wa + ((size_t)r0 * 8192 + (size_t)j * nr * KC) * (TB / 32) + (size_t)(q * C + (int)j) * ESB;
  const u64 c0 = clk();
  u64 g0 = 0; if (RTRACE) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  if (wid == PW) {
    if (lane == 0) {
      for (int s = 0; s < NSTG; ++s) mbar_init(bars + 8 * s, 1);
      mbar_init(xbar, 1);
      mbar_init(rbar, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      if (FL & 32) { for (int s = 0; s < NSTG; ++s) mbar_arrive(bars + 8 * s); }
      else if (!(FL & 8)) { mbar_expect_arrive(bars, ESB + stb); bulk_g2s(wsm - ESB, wsrc, ESB + stb, bars); }
    }
    __syncwarp();
    const int M = (int)p.M;
    const unsigned char* xs = p.x + (size_t)j * KC * 2;
    if (lane == 0) {
      mbar_expect_arrive(xbar, (FL & 33) ? 0u : (u32)(M * KC * 2));
      // this rank owns tokens t = j, j + C, ...: C senders x nr floats each
      const int own = M > (int)j ? (M - 1 - (int)j) / C + 1 : 0;
      if (!(FL & 4)) mbar_expect_arrive(rbar, (u32)(own * C * nr * 4));
      else mbar_arrive(rbar);
    }
    __syncwarp();
    if (!(FL & 33)) for (int t = lane; t < M; t += 32) bulk_g2s(xsm + t * XS, xs + (size_t)t * 16384, KC * 2, xbar);
    if (lane == 0) {
      if (FL & 8) { mbar_expect_arrive(bars, ESB + stb); bulk_g2s(wsm - ESB, wsrc, ESB + stb, bars); }
#pragma unroll 1
      for (int s = 1; s < NSTG; ++s) {
        if (FL & 32) break;
        mbar_expect_arrive(bars + 8 * s, stb);
        bulk_g2s(wsm + s * stb, wsrc + ESB + (size_t)s * stb, stb, bars + 8 * s);
      }
    }
  }
  __syncthreads();
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  const int M = (int)p.M;
  if (CMP && wid == PW) {   // escape record (int n; n x (row_local << 16 | k_local, delta)) arrives with W stage 0
    for (int i = lane; i < TT * NRMAX; i += 32) corr[i] = 0.f;
    mbar_spin(xbar, 0);
    mbar_spin(bars, 0);
    __syncwarp();
    const int* er = (const int*)(sm + WOFF);
    const int e_n = er[0];
    for (int i = lane; i < e_n * M; i += 32) {
      const int e = i / M, t = i % M;
      const int rk = er[4 + 2 * e];
      const float d = __int_as_float(er[5 + 2 * e]);
      const int kl = rk & 0xFFFF, rl = rk >> 16;
      const float xv = __uint_as_float((u32)(*(const unsigned short*)(sm + XOFF + t * XS + kl * 2)) << 16);
      atomicAdd(&corr[t * NRMAX + rl], xv * d);
    }
  }
  float ah[NG][MT][4], al[NG][MT][4];
#pragma unroll
  for (int g = 0; g < NG; ++g)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int c = 0; c < 4; ++c) { ah[g][mt][c] = 0.f; al[g][mt][c] = 0.f; }
  u64 cx = 0, cs0 = 0, cl = 0, cc = 0;
  u64 cit[NSTG + 1];
  if (wid < NCW) {
    mbar_spin(xbar, 0);
    cx = clk();
    const int rowa = lane & 15, cha = lane >> 4;
#pragma unroll 1
    for (int s = 0; s < NSTG; ++s) {
      mbar_spin(bars + 8 * s, 0);
      if (s == 0) cs0 = clk();
      cl = clk();
      if (RTRACE) cit[s] = cl;
      if (FL & 2) continue;
#pragma unroll
      for (int u = 0; u < UPW; ++u) {
        const int unit = wid * UPW + u, ktl = unit >> 1, hh = unit & 1;   // k32 tile within stage, k16 half
        const int kt = s * KTS + ktl;
        const u32 tb = wsm + s * stb + ktl * TB * nr;
        u32 a[MT][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          ldsm_x4(a[mt], xsm + (mt * 16 + rowa) * XS + (kt * 32 + hh * 16 + cha * 8) * 2);
        u32 bh[NG][2], bl[NG][2];
#pragma unroll
        for (int g = 0; g < NG; ++g) {
          const int rg = min(8, nr - 8 * g);
          bh[g][0] = bh[g][1] = bl[g][0] = bl[g][1] = 0u;
          if (lane < 4 * rg) {
            if (CMP) {
              const uint4 bb = lds128(tb + g * 8 * TB + lane * 16);
              const uint2 nn = lds64(tb + g * 8 * TB + rg * 64 + lane * 8);
              const u32 hb = hh ? bb.y : bb.x, lb = hh ? bb.w : bb.z;
              const u32 hn = nn.x >> (8 * hh), ln = nn.y >> (8 * hh);
              bh[g][0] = ((prmt(hb, 0x1100) & 0x807F807Fu) | ((hn << 7) & 0x07800780u)) + BASE2;
              bh[g][1] = ((prmt(hb, 0x3322) & 0x807F807Fu) | ((hn << 3) & 0x07800780u)) + BASE2;
              constexpr u32 E8 = (8u << 7) | (8u << 23);
              bl[g][0] = (prmt(lb, 0x1100) & 0x807F807Fu) | ((bh[g][0] & 0x7F807F80u) - (((ln << 7) & 0x07800780u) + E8));
              bl[g][1] = (prmt(lb, 0x3322) & 0x807F807Fu) | ((bh[g][1] & 0x7F807F80u) - (((ln << 3) & 0x07800780u) + E8));
            } else {
              const uint2 vh = lds64(tb + g * 8 * TB + lane * 16 + hh * 8);
              const uint2 vl = lds64(tb + g * 8 * TB + rg * 64 + lane * 16 + hh * 8);
              bh[g][0] = vh.x; bh[g][1] = vh.y; bl[g][0] = vl.x; bl[g][1] = vl.y;
            }
          }
        }
        if (FL & 64) {
#pragma unroll
          for (int g = 0; g < NG; ++g)
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) { ah[g][mt][0] += __uint_as_float(a[mt][0] ^ bh[g][0] ^ bl[g][1]); al[g][mt][1] += __uint_as_float(a[mt][2] ^ bl[g][0] ^ bh[g][1]); }
        } else {
#pragma unroll
        for (int g = 0; g < NG; ++g)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) { mma16816(ah[g][mt], a[mt], bh[g][0], bh[g][1]); mma16816(al[g][mt], a[mt], bl[g][0], bl[g][1]); }
        }
      }
    }
    if (RTRACE) cit[NSTG] = clk();
  }
  __syncthreads();                                        // W region free -> scratch
  float* sc = (float*)(sm + WOFF);
  if (wid < NCW) {
#pragma unroll
    for (int g = 0; g < NG; ++g)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int t = mt * 16 + (lane >> 2), n = g * 8 + 2 * (lane & 3);
        *(float2*)&sc[(wid * TT + t) * SCS + n] = make_float2(ah[g][mt][0] + al[g][mt][0], ah[g][mt][1] + al[g][mt][1]);
        *(float2*)&sc[(wid * TT + t + 8) * SCS + n] = make_float2(ah[g][mt][2] + al[g][mt][2], ah[g][mt][3] + al[g][mt][3]);
      }
  }
  __syncthreads();
  cc = clk();
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");   // every rank's rbar initialized
  const u64 cq = clk();
  for (int e = threadIdx.x; e < M * nr; e += blockDim.x) {
    const int t = e / nr, n = e % nr;
    float v = CMP ? corr[t * NRMAX + n] : 0.f;
#pragma unroll
    for (int w = 0; w < NCW; ++w) v += sc[(w * TT + t) * SCS + n];
    const u32 o = (u32)(t % C);
    st_async_f32(mapa_u32(smem_u32(&slots[((int)j * MO + t / C) * NRMAX + n]), o), v, mapa_u32(rbar, o));
  }
  const u64 cpu = clk();
  mbar_spin_cl(rbar, 0);
  const u64 cr = clk();
  for (int e = threadIdx.x; e < MO * nr; e += blockDim.x) {
    const int to = e / nr, n = e % nr, t = to * C + (int)j;
    if (t < M) {
      float v = 0.f;
#pragma unroll
      for (int i = 0; i < C; ++i) v += slots[(i * MO + to) * NRMAX + n];
      p.out[(size_t)t * 256 + r0 + n] = v;
    }
  }
  const u64 cst = clk();
  if (RTRACE && p.trc && threadIdx.x == 0) {
    u64 g1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
    unsigned long long* tr = p.trc + blockIdx.x * 16;
    tr[0] = g0; tr[1] = c0; tr[2] = cx; tr[3] = cs0; tr[4] = cl; tr[5] = cc; tr[6] = g1; tr[7] = clk();
    for (int i = 0; i <= NSTG && i < 4; ++i) tr[8 + i] = cit[i];
    tr[12] = cq; tr[13] = cpu; tr[14] = cst; tr[15] = cr;
  }
}
"""

_r3_R1_SRC = r"""
// ---- r1: M <= 2, no split-K. CTA b = rows 2b, 2b+1 x full K (128 CTAs), 8 consumer warps + 1 producer warp.
//      NS1 stages of KST k; stage s = W part (CMP: [256 groups x 16 B bytes][256 x 8 B nibbles]; exact: [256 x 32 B fp32])
//      + x chunk s (M rows x KST bf16, one bulk copy per row). CTA b walks the stages starting at s = b % NS1 so the 128
//      CTAs do not all hit the same x lines at once (L2 hot spot). CMP: escape record (ESB) bulk-copied first.
struct __align__(64) P1 { const unsigned char* x; float* out; long long M; unsigned long long* trc; };
__device__ __forceinline__ float bf_lo(u32 v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf_hi(u32 v) { return __uint_as_float(v & 0xffff0000u); }
template <int M, int NS1, int CMP, int ESB, u64 WA, u32 BASE2, int PRE = 1>
__device__ __forceinline__ void r1_body(const P1& p) {
  constexpr int KST = 8192 / NS1;                 // k per stage
  constexpr int GPS = 2 * KST / 8;                // groups per stage
  static_assert(GPS == 256, "one group per consumer thread per stage");
  constexpr int STB = GPS * (CMP ? 24 : 32);
  constexpr int BLK = ESB + NS1 * STB;
  constexpr int XST = M * KST * 2;                // x bytes per stage
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  unsigned char* sm = (unsigned char*)(((size_t)sm_raw + 1023) & ~(size_t)1023);
  const u32 bars = smem_u32(sm);                  // bar[NS1], ebar
  const u32 ebar = bars + 8 * NS1;
  const u32 wsm = bars + 1024;                    // [ESB][W stages]
  const u32 xsm = wsm + ((BLK + 1023) / 1024) * 1024;   // [stage][M][KST] bf16
  __shared__ float red[8][M];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  u64 wa; asm("mov.u64 %0, %1;" : "=l"(wa) : "n"(WA));
  const unsigned char* wsrc = (const unsigned char*)wa + (size_t)blockIdx.x * BLK;
  const int s0 = 0;
  const u64 c0 = clk();
  u64 cit[NS1 + 2];
  if (wid == 8 && lane == 0) {
    for (int s = 0; s <= NS1; ++s) mbar_init(bars + 8 * s, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    // W stages 0..PRE-1 before touching the (cold) params, then x chunk s right after W stage s
#pragma unroll
    for (int s = 0; s < PRE; ++s) {
      mbar_expect_arrive(bars + 8 * s, STB + XST);
      bulk_g2s(wsm + ESB + s * STB, wsrc + ESB + (size_t)s * STB, STB, bars + 8 * s);
    }
    if (CMP) { mbar_expect_arrive(ebar, ESB); bulk_g2s(wsm, wsrc, ESB, ebar); }
    const unsigned char* xg = p.x;
#pragma unroll 1
    for (int s = 0; s < NS1; ++s) {
      if (s >= PRE) {
        mbar_expect_arrive(bars + 8 * s, STB + XST);
        bulk_g2s(wsm + ESB + s * STB, wsrc + ESB + (size_t)s * STB, STB, bars + 8 * s);
      }
#pragma unroll
      for (int t = 0; t < M; ++t)
        bulk_g2s(xsm + s * XST + t * KST * 2, xg + ((size_t)t * 8192 + s * KST) * 2, KST * 2, bars + 8 * s);
    }
  }
  __syncthreads();
  float acc[M][2];
#pragma unroll
  for (int t = 0; t < M; ++t) { acc[t][0] = 0.f; acc[t][1] = 0.f; }
  const int k8 = tid & 127;
  if (wid < 8) {
    if (RTRACE) cit[0] = clk();
#pragma unroll 1
    for (int i = 0; i < NS1; ++i) {
      const int s = (s0 + i) % NS1;
      mbar_spin(bars + 8 * s, 0);
      cit[1 + i] = clk();
      const u32 st = wsm + ESB + s * STB;
      const u32 xs = xsm + s * XST + k8 * 16;
      if (CMP) {
        const uint4 bb = lds128(st + tid * 16);
        const uint2 nn = lds64(st + GPS * 16 + tid * 8);
        u32 h[4], l[4];
        dec_hi(make_uint2(bb.x, bb.y), nn.x, BASE2, h);
        dec_lo(make_uint2(bb.z, bb.w), nn.y, h, l);
#pragma unroll
        for (int t = 0; t < M; ++t) {
          const uint4 xv = lds128(xs + t * KST * 2);
          const u32 xw[4] = {xv.x, xv.y, xv.z, xv.w};
          float a0 = acc[t][0], a1 = acc[t][1];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float x0 = bf_lo(xw[q]), x1 = bf_hi(xw[q]);
            a0 = fmaf(x0, bf_lo(h[q]), a0); a1 = fmaf(x0, bf_lo(l[q]), a1);
            a0 = fmaf(x1, bf_hi(h[q]), a0); a1 = fmaf(x1, bf_hi(l[q]), a1);
          }
          acc[t][0] = a0; acc[t][1] = a1;
        }
      } else {
        const float4 w0 = *(const float4*)(sm + (st - bars) + tid * 32);
        const float4 w1 = *(const float4*)(sm + (st - bars) + tid * 32 + 16);
#pragma unroll
        for (int t = 0; t < M; ++t) {
          const uint4 xv = lds128(xs + t * KST * 2);
          float a0 = acc[t][0], a1 = acc[t][1];
          a0 = fmaf(bf_lo(xv.x), w0.x, a0); a1 = fmaf(bf_hi(xv.x), w0.y, a1);
          a0 = fmaf(bf_lo(xv.y), w0.z, a0); a1 = fmaf(bf_hi(xv.y), w0.w, a1);
          a0 = fmaf(bf_lo(xv.z), w1.x, a0); a1 = fmaf(bf_hi(xv.z), w1.y, a1);
          a0 = fmaf(bf_lo(xv.w), w1.z, a0); a1 = fmaf(bf_hi(xv.w), w1.w, a1);
          acc[t][0] = a0; acc[t][1] = a1;
        }
      }
    }
#pragma unroll
    for (int t = 0; t < M; ++t) {
      float v = acc[t][0] + acc[t][1];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) red[wid][t] = v;
    }
  }
  __syncthreads();
  if (tid < 2 * M) {
    const int r = tid / M, t = tid % M;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < 4; ++w) v += red[r * 4 + w][t];
    if (CMP) {   // escape record: int n; then n x (int (row << 16 | k), float delta) at +16
      mbar_spin(ebar, 0);
      const int* er = (const int*)(sm + 1024);
      const int n = er[0];
      for (int e = 0; e < n; ++e) {
        const int rk = er[4 + 2 * e];
        if ((rk >> 16) == r) {
          const int kk = rk & 0xFFFF;
          const unsigned char* xp = sm + (xsm - bars) + (kk / KST) * XST + (t * KST + kk % KST) * 2;
          v = fmaf(__uint_as_float((u32)(*(const unsigned short*)xp) << 16), __int_as_float(er[5 + 2 * e]), v);
        }
      }
    }
    p.out[(size_t)t * 256 + blockIdx.x * 2 + r] = v;
  }
  if (RTRACE && p.trc && tid == 0) {
    cit[NS1 + 1] = clk();
    for (int i = 0; i < NS1 + 2 && i < 16; ++i) p.trc[blockIdx.x * 16 + i] = cit[i] - c0;
  }
}
"""

# name: (TT, RR, KK, KS, NS, WK, WG, max_M)
_r3_CFGS = {
    "rt16": (16, 16, 1024, 256, 4, 8, 1, 16),
    "rt32": (32, 16, 1024, 256, 4, 8, 1, 32),
    "rt64": (64, 32, 512, 256, 2, 8, 1, 64),
    "rt128": (128, 32, 512, 256, 2, 4, 2, 128),
    "rtbig": (128, 32, 512, 256, 2, 4, 2, 1 << 30),
}


def _r3_smem_bytes(cfg):
    tt, rr, kk, ks, ns, wk, wg, _ = cfg
    sb = (tt * ks * 2 + ks * rr * 4 + 1023) // 1024 * 1024
    return 1024 + 1024 + ns * sb


_r3_RS_NCH = 4
_r3_RS_MAX = 8


def _r3_rs_smem(m):
    return 1024 + 1024 + _r3_RS_NCH * (2 * 2048 * 4 + m * 2048 * 2)


_r3_FULL_SOURCE = False   # True: compile every experimental kernel (ablations); False: only what DISPATCH uses


def _r3_source(cfgs=_r3_CFGS, addr=None):
    s = _r3_SRC
    if not _r3_FULL_SOURCE:
        for name in ("rt16", "rt32", "rt64"):
            tt, rr, kk, ks, ns, wk, wg, _ = cfgs[name]
            nt = 32 * (wk * wg + 1)
            for fl in (528, 784):
                s += (f'extern "C" __global__ void __launch_bounds__({nt},1) {name}_f{fl}_i(const __grid_constant__ P p)'
                      f' {{ rt_body<{tt},{rr},{kk},{ks},{ns},{wk},{wg},{fl},{addr["rt"][(rr, kk, ks)]}ull>(p); }}\n')
            for fl in (4, 16, 272):
                s += (f'extern "C" __global__ void __launch_bounds__({nt},1) {name}_f{fl}(const __grid_constant__ P p)'
                      f' {{ rt_body<{tt},{rr},{kk},{ks},{ns},{wk},{wg},{fl}>(p); }}\n')
                if addr:
                    s += (f'extern "C" __global__ void __launch_bounds__({nt},1) {name}_f{fl}_i(const __grid_constant__ P p)'
                          f' {{ rt_body<{tt},{rr},{kk},{ks},{ns},{wk},{wg},{fl},{addr["rt"][(rr, kk, ks)]}ull>(p); }}\n')
            if addr and "cmpe" in addr:
                wa, esb = addr["cmpe"][(rr, kk, ks)]
                for fl in (48, 304):
                    s += (f'extern "C" __global__ void __launch_bounds__({nt},1) {name}_f{fl}_ie(const __grid_constant__ P p)'
                          f' {{ rt_body<{tt},{rr},{kk},{ks},{ns},{wk},{wg},{fl},{wa}ull,{esb},{addr["b2"]}u>(p); }}\n')
        for name in ("rw64", "rw128b"):
            tn, kk, ns = _r3_RW_CFGS[name]
            s += (f'extern "C" __global__ void __launch_bounds__(160,1) {name}(const __grid_constant__ PW p)'
                  f' {{ rw_body<{tn},{kk},{ns},0,0>(p); }}\n')
            if addr:
                s += (f'extern "C" __global__ void __launch_bounds__(160,1) {name}_i(const __grid_constant__ PW p)'
                      f' {{ rw_body<{tn},{kk},{ns},0,0,0,0,{addr["rw"][kk]}ull>(p); }}\n')
        for name in ("rw64", "rw128b", "rw128a"):
            tn, kk, ns = _r3_RW_CFGS[name]
            s += (f'extern "C" __global__ void __launch_bounds__(160,1) {name}_ws(const __grid_constant__ PW p)'
                  f' {{ rw_body<{tn},{kk},{ns},0,0,0,0,{addr["rw"][kk]}ull,1>(p); }}\n')
        for name in ("rw64", "rw128b"):
            tn, kk, ns = _r3_RW_CFGS[name]
            s += (f'extern "C" __global__ void __launch_bounds__(160,1) {name}_br(const __grid_constant__ PW p)'
                  f' {{ rw_body<{tn},{kk},{ns},0,0,0,0,{addr["rw"][kk]}ull,2>(p); }}\n')
        for name in ("rw128a", "rw256"):
            tn, kk, ns = _r3_RW_CFGS[name]
            s += (f'extern "C" __global__ void __cluster_dims__(2,1,1) __launch_bounds__(160,1) {name}_x(const __grid_constant__ PW p)'
                  f' {{ rw_body<{tn},{kk},{ns},2,0,0,0,{addr["rw"][kk]}ull>(p); }}\n')
        for name in ("rw64mc", "rw128bmc"):
            tn, kk, ns = _r3_RW_CFGS[name]
            s += (f'extern "C" __global__ void __cluster_dims__(2,1,1) __launch_bounds__(160,1) {name}_i(const __grid_constant__ PW p)'
                  f' {{ rw_body<{tn},{kk},{ns},1,0,0,0,{addr["rw"][kk]}ull>(p); }}\n')
        for m in (1, 2):
            s += (f'extern "C" __global__ void __launch_bounds__(256,2) rs1r{m}_i(const __grid_constant__ PS p)'
                  f' {{ rs1r_body<{m},{_r3_RS_NCH},{addr["w"]}ull>(p); }}\n')
        for m in range(1, 5):
            s += (f'extern "C" __global__ void __launch_bounds__(256,1) rs{m}(const __grid_constant__ PS p)'
                  f' {{ rs_body<{m},{_r3_RS_NCH}>(p); }}\n')
            if addr:
                s += (f'extern "C" __global__ void __launch_bounds__(256,1) rs{m}_i(const __grid_constant__ PS p)'
                      f' {{ rs_body<{m},{_r3_RS_NCH},{addr["rs"]}ull>(p); }}\n')
        return s
    for name, (tt, rr, kk, ks, ns, wk, wg, _) in cfgs.items():
        nt = 32 * (wk * wg + 1)
        for fl in (0, 1, 2, 3, 4, 8, 16, 48, 80, 144, 272):
            s += (f'extern "C" __global__ void __launch_bounds__({nt},1) {name}_f{fl}(const __grid_constant__ P p)'
                  f' {{ rt_body<{tt},{rr},{kk},{ks},{ns},{wk},{wg},{fl}>(p); }}\n')
    for name, (tn, kk, ns) in _r3_RW_CFGS.items():
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
    for m in range(1, _r3_RS_MAX + 1):
        s += (f'extern "C" __global__ void __launch_bounds__(256,1) rs{m}(const __grid_constant__ PS p)'
              f' {{ rs_body<{m},{_r3_RS_NCH}>(p); }}\n')
    return s


class _r3__Mod:
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


def _r3__tmap_x(x, m, box_rows):
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


def _r3__tmap_x3(x, m):
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


_r3_RC_Q = 15
# rw configs: name -> (TN tokens per CTA, KK k per CTA, NS ring slots)
_r3_RW_CFGS = {"rw128a": (128, 256, 4), "rw128b": (128, 512, 6), "rw64": (64, 512, 6), "rw128bmc": (128, 512, 6),
           "rw128cmc": (128, 1024, 6), "rw64mc": (64, 512, 6), "rw256": (256, 256, 4), "rw64n8": (64, 512, 8),
           "rw128b7": (128, 512, 7)}
_r3_RW_SEL = {"small": "rw128a", "big": "rw128b"}   # M <= 128 -> small, else big
_r3_USE_RW = True


def _r3_rw_smem(name):
    tn, kk, ns = _r3_RW_CFGS[name]
    return 1024 + 1024 + ns * (16384 + tn * 128)


def _r3__rw_layout(hi, lo, kk):
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


def _r3__rwc_layout(c, kk):
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


def _r3__rwc_escapes(c, kk, dev):
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


def _r3__tmap_x2(x, m, box_rows):
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


def _r3_rc_smem(mt):
    tt = 16 * mt
    sb = (tt * 256 * 2 + 18 * 256 * 4 + 1023) // 1024 * 1024
    return 1024 + 1024 + 4 * sb


def _r3__tmap_x3s(x, m, tt):
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


_r3_RQ_SMEM = 1024 + 1024 + 4 * (32768 + 16384)


def _r3__rq_layout(hi, lo):
    """[128 CTAs (rows 2b, 2b+1)][256 kt][hi: 8 lanes x 16 B][lo: 8 lanes x 16 B] (lanes = row*4 + tid, row < 2)."""
    out = []
    for w in (hi, lo):
        t = w.contiguous().view(128, 2, 256, 2, 2, 4, 2).permute(0, 2, 1, 5, 3, 4, 6).contiguous()  # [b][kt][row][tid][..]
        out.append(t.view(128, 256, 1, 64))
    return torch.cat(out, 2).contiguous().view(-1)


def _r3__tmap_x3q(x, m):
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


def _r3__rc_layout(hi, lo):
    """Per cluster q (rows r0..r0+nr) and rank (k slice of 1024): [32 kt][groups: hi rg x 64 B, lo rg x 64 B]."""
    parts = []
    for q in range(_r3_RC_Q):
        r0, r1 = q * 256 // _r3_RC_Q, (q + 1) * 256 // _r3_RC_Q
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


def _r3__compress_pair(hi, lo):
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


def _r3__cmp_layout(c, rr, kk, ks):
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



def _r3__cmp_layout_esc(c, key, esc):
    """_cmp_layout blocks with an escape record (int n, pad, n x (row_local << 16 | k_local, delta)) prefixed per block."""
    rr, kk, ks = key
    lay = _r3__cmp_layout(c, rr, kk, ks).view(-1)
    nblk = (256 // rr) * (8192 // kk)
    off, ent, dl, emax = esc
    esb = (16 + 8 * max(emax, 1) + 15) // 16 * 16
    rec = torch.zeros(nblk, esb // 4, dtype=torch.int32, device=lay.device)
    for bi in range(nblk):
        o0, o1 = int(off[bi]), int(off[bi + 1])
        rec[bi, 0] = o1 - o0
        rec[bi, 4:4 + 2 * (o1 - o0):2] = ent[o0:o1]
        rec[bi, 5:5 + 2 * (o1 - o0):2] = dl[o0:o1].contiguous().view(torch.int32)
    blk = torch.cat([rec.view(torch.uint8).view(nblk, esb), lay.view(nblk, -1)], 1).contiguous()
    return blk.view(-1), esb


def _r3__perm_tiles(w):
    """[256, 8192] bf16 -> [32 groups][256 k32][256 bf16] in mma B-fragment lane order."""
    return w.contiguous().view(32, 8, 256, 2, 2, 4, 2).permute(0, 2, 1, 5, 3, 4, 6).contiguous().view(32, 256, 256)


def _r3__layout(hi_t, lo_t, rr, kk, ks):
    t = torch.stack([hi_t, lo_t], 2)  # [32, 256, 2, 256]
    nr, s, nstg = 256 // rr, 8192 // kk, kk // ks
    t = t.view(nr, rr // 8, s, nstg, ks // 32, 2, 256).permute(0, 2, 3, 4, 1, 5, 6).contiguous()
    return t.view(-1)



# ---------------------------------------------------------------- rk (cluster split-K, M <= 64)
_r3_RK_Q = {8: 15, 4: 30, 2: 64}          # clusters per cluster size (120 SMs for C = 4/8; 128 CTAs for C = 2)
_r3_RK_NSTG = {8: 4, 4: 4, 2: 8}
_r3_RK_KERNELS = [(8, mt, cmp) for mt in (1, 2) for cmp in (0, 1)] + [(4, mt, cmp) for mt in (1, 2) for cmp in (0, 1)]\
    + [(2, 1, cmp) for cmp in (0, 1)]


_r3_RK_NCW16 = int(__import__('os').environ.get('RK_NCW16', '0'))


def _r3_rk_ncw(c, mt):
    return 16 if (_r3_RK_NCW16 and mt <= 2 and (8192 // c // _r3_RK_NSTG[c]) // 32 * 2 == 16) else 8


def _r3_rk_name(c, mt, cmp):
    return f"rk_c{c}_m{mt}_{'c' if cmp else 'u'}"


def _r3_rk_smem(c, mt, cmp, esb=528):
    q = _r3_RK_Q[c]
    kc = 8192 // c
    tt, xs = mt * 16, kc * 2 + 16
    nrmax = (256 + q - 1) // q
    ng = (nrmax + 7) // 8
    tb = 96 if cmp else 128
    wmax = (esb if cmp else 0) + kc // 32 * tb * nrmax
    scs = ng * 8 + 4
    mo = (tt + c - 1) // c
    xoff = 1024 + (wmax + 1023) // 1024 * 1024
    return xoff + max(tt * xs, 0) + c * mo * nrmax * 4 + tt * nrmax * 4 + 1024 + max(0, _r3_rk_ncw(c, mt) * tt * scs * 4 - wmax)


def _r3__rk_rows(q_, qq):
    return qq * 256 // q_, (qq + 1) * 256 // q_


def _r3__lanes(a):
    """[256, 8192] -> [256 row, 256 kt, 4 q, 8 values (h, half, pair)] in mma B-fragment lane order."""
    return a.view(256, 256, 2, 2, 4, 2).permute(0, 1, 4, 2, 3, 5).reshape(256, 256, 4, 8)


def _r3__rk_assemble(per_row, c):
    """per_row: list of 2 tensors [256 row, 256 kt, 4 q, X] (uint8 bytes per lane) -> blocks in rk order."""
    q_ = _r3_RK_Q[c]
    kt_c = 256 // c
    out = []
    for qq in range(q_):
        r0, r1 = _r3__rk_rows(q_, qq)
        nr = r1 - r0
        for j in range(c):
            parts = []
            for g in range((nr + 7) // 8):
                a, b = r0 + 8 * g, min(r1, r0 + 8 * g + 8)
                for t in per_row:
                    parts.append(t[a:b, j * kt_c:(j + 1) * kt_c].permute(1, 0, 2, 3).reshape(kt_c, -1))
            out.append(torch.cat(parts, 1).reshape(-1))
    return torch.cat(out).contiguous()


def _r3__rk_layout_u(hi, lo, c):
    h = _r3__lanes(hi.contiguous().view(torch.int16)).contiguous().view(torch.uint8).view(256, 256, 4, 16)
    l_ = _r3__lanes(lo.contiguous().view(torch.int16)).contiguous().view(torch.uint8).view(256, 256, 4, 16)
    return _r3__rk_assemble([h, l_], c)


def _r3__rk_layout_c(cm, c, esb=0):
    hb, lb = _r3__lanes(cm["hb"]), _r3__lanes(cm["lb"])
    byt = torch.cat([hb, lb], -1).to(torch.uint8).contiguous()            # [row][kt][q][16]

    def nib(cc):
        cc = _r3__lanes(cc).long()
        w = torch.zeros(cc.shape[:3], dtype=torch.long, device=cc.device)
        for i in range(4):
            w |= cc[..., 2 * i] << (4 * i)
            w |= cc[..., 2 * i + 1] << (16 + 4 * i)
        return (w - ((w >> 31) & 1) * (1 << 32)).to(torch.int32)
    nb = torch.stack([nib(cm["ch"]), nib(cm["cl"])], -1).contiguous().view(torch.uint8).view(256, 256, 4, 8)
    # per group: all lanes' bytes, then all lanes' nibble words
    q_ = _r3_RK_Q[c]
    kt_c = 256 // c
    off, ent, dl, _ = _r3__rk_escapes(cm, c, cm["hb"].device)
    out = []
    for qq in range(q_):
        r0, r1 = _r3__rk_rows(q_, qq)
        nr = r1 - r0
        for j in range(c):
            bi = qq * c + j
            rec = torch.zeros(esb // 4, dtype=torch.int32, device=byt.device)
            o0, o1 = int(off[bi]), int(off[bi + 1])
            rec[0] = o1 - o0
            rec[4:4 + 2 * (o1 - o0):2] = ent[o0:o1]
            rec[5:5 + 2 * (o1 - o0):2] = dl[o0:o1].view(torch.int32)
            parts = []
            for g in range((nr + 7) // 8):
                a, b = r0 + 8 * g, min(r1, r0 + 8 * g + 8)
                parts.append(byt[a:b, j * kt_c:(j + 1) * kt_c].permute(1, 0, 2, 3).reshape(kt_c, -1))
                parts.append(nb[a:b, j * kt_c:(j + 1) * kt_c].permute(1, 0, 2, 3).reshape(kt_c, -1))
            out.append(rec.view(torch.uint8))
            out.append(torch.cat(parts, 1).reshape(-1))
    return torch.cat(out).contiguous()


def _r3__rk_escapes(cm, c, dev):
    q_ = _r3_RK_Q[c]
    kc = 8192 // c
    rows = torch.repeat_interleave(torch.arange(256, device=dev), (cm["erow"][1:] - cm["erow"][:-1]).long())
    k = cm["ek"].long()
    r0s = torch.tensor([_r3__rk_rows(q_, qq)[0] for qq in range(q_)], device=dev)
    qq = torch.searchsorted(r0s, rows, right=True) - 1
    blk = qq * c + k // kc
    order = torch.argsort(blk * (1 << 24) + (rows - r0s[qq]) * 8192 + k % kc)
    cnt = torch.bincount(blk, minlength=q_ * c)
    off = torch.zeros(q_ * c + 1, dtype=torch.int32, device=dev)
    off[1:] = cnt.cumsum(0).int()
    ent = (((rows - r0s[qq]) << 16) | (k % kc))[order].int().contiguous()
    return off, ent, cm["ed"][order].contiguous(), int(cnt.max()) if cnt.numel() else 0


_r3_RK_FL = [(8, 1, 0, 32), (8, 1, 0, 96), (8, 1, 0, 34)]


def _r3_rk_source(st):
    s = _r3_RK_SRC
    b2 = (st["cmp"]["base"] << 7) | (st["cmp"]["base"] << 23)
    for c, mt, cmp in _r3_RK_KERNELS:
        wa = st["rk_w"][(c, cmp)].data_ptr()
        e = (st["rk_esb"][c] if cmp else 0, 0, 0)
        nw = _r3_rk_ncw(c, mt)
        s += (f'extern "C" __global__ void __cluster_dims__({c},1,1) __launch_bounds__({32 * nw + 32},1) {_r3_rk_name(c, mt, cmp)}'
              f'(const __grid_constant__ PK p) {{ rk_body<{c},{_r3_RK_Q[c]},{mt},{_r3_RK_NSTG[c]},{cmp},{wa}ull,{e[0]},'
              f'{nw},{e[2]}ull,{b2}u>(p); }}\n')
    for c, mt, cmp, fl in _r3_RK_FL:
        wa = st["rk_w"][(c, cmp)].data_ptr()
        e = (st["rk_esb"][c] if cmp else 0, 0, 0)
        nw = _r3_rk_ncw(c, mt)
        s += (f'extern "C" __global__ void __cluster_dims__({c},1,1) __launch_bounds__({32 * nw + 32},1) {_r3_rk_name(c, mt, cmp)}_f{fl}'
              f'(const __grid_constant__ PK p) {{ rk_body<{c},{_r3_RK_Q[c]},{mt},{_r3_RK_NSTG[c]},{cmp},{wa}ull,{e[0]},'
              f'{nw},{e[2]}ull,{b2}u,{fl}>(p); }}\n')
    return s


def _r3__rk_prepare(st, hi, lo):
    st["rk_w"], st["rk_esc"], st["rk_esb"] = {}, {}, {}
    for c in sorted({k[0] for k in _r3_RK_KERNELS}):
        st["rk_w"][(c, 0)] = _r3__rk_layout_u(hi, lo, c)
        st["rk_esc"][c] = _r3__rk_escapes(st["cmp"], c, hi.device)
        st["rk_esb"][c] = (16 + 8 * max(st["rk_esc"][c][3], 1) + 15) // 16 * 16
        st["rk_w"][(c, 1)] = _r3__rk_layout_c(st["cmp"], c, st["rk_esb"][c])


_r3_RK_MAXM = 32
_r3_RK_SEL = None    # None: default per-M choice; else (C, cmp)


def _r3_rk_choice(m, st):
    sel = st.get("rk_sel", _r3_RK_SEL)
    if sel is not None:
        return sel
    return (8, 1)


def _r3_run_rk(x, st, m, c, cmp, out):
    mt = (m + 15) // 16
    params = struct.pack("QQqQ", x.data_ptr(), out.data_ptr(), m, _r3_TRACE.data_ptr() if _r3_TRACE is not None else 0)
    params += b"\0" * ((-len(params)) % 64)
    fl = st.get("rk_fl", 0)
    st["mod"].launch(_r3_rk_name(c, mt, cmp) + (f"_f{fl}" if fl else ""), _r3_RK_Q[c] * c, 32 * _r3_rk_ncw(c, mt) + 32, _r3_rk_smem(c, mt, cmp, st['rk_esb'][c]), params)
    return out


# ---------------------------------------------------------------- r1 (M <= 2, no split, compressed, SIMT)
_r3_R1_NS = 8
_r3_R1_PRE = int(__import__('os').environ.get('R1_PRE', '1'))


def _r3__r1_layout(cm, dev):
    """[128 CTAs][ESB escape record][4 stages][512 groups x 16 B bytes][512 groups x 8 B nibbles] (group = row_l*256 + k8)."""
    def grp(a):   # [256, 8192] -> [128 cta][8 stage][256 grp][8 v]   (grp = row_l * 128 + k8)
        return a.view(128, 2, 8, 128, 8).permute(0, 2, 1, 3, 4).reshape(128, 8, 256, 8)
    byt = torch.cat([grp(cm["hb"]), grp(cm["lb"])], -1).to(torch.uint8)                # [128][4][512][16]

    def nib(cc):
        cc = grp(cc).long()
        w = torch.zeros(cc.shape[:3], dtype=torch.long, device=cc.device)
        for i in range(4):
            w |= cc[..., 2 * i] << (4 * i)
            w |= cc[..., 2 * i + 1] << (16 + 4 * i)
        return (w - ((w >> 31) & 1) * (1 << 32)).to(torch.int32)
    nb = torch.stack([nib(cm["ch"]), nib(cm["cl"])], -1).contiguous().view(torch.uint8).view(128, 8, 256, 8)
    stg = torch.cat([byt.reshape(128, 8, 4096), nb.reshape(128, 8, 2048)], -1).reshape(128, -1)
    rows = torch.repeat_interleave(torch.arange(256, device=dev), (cm["erow"][1:] - cm["erow"][:-1]).long())
    k = cm["ek"].long()
    cta = rows // 2
    cnt = torch.bincount(cta, minlength=128)
    emax = int(cnt.max()) if cnt.numel() else 0
    esb = (16 + 8 * max(emax, 1) + 15) // 16 * 16
    rec = torch.zeros(128, esb // 4, dtype=torch.int32, device=dev)
    rec[:, 0] = cnt.int()
    order = torch.argsort(cta * (1 << 24) + (rows % 2) * 8192 + k)
    cs = torch.zeros(129, dtype=torch.long, device=dev)
    cs[1:] = cnt.cumsum(0)
    for b in torch.nonzero(cnt).flatten().tolist():
        idx = order[cs[b]:cs[b + 1]]
        n = idx.numel()
        rec[b, 4:4 + 2 * n:2] = (((rows[idx] % 2) << 16) | k[idx]).int()
        rec[b, 5:5 + 2 * n:2] = cm["ed"][idx].contiguous().view(torch.int32)
    blk = torch.cat([rec.view(torch.uint8).view(128, esb), stg], 1).contiguous()
    return blk.view(-1), esb


def _r3__r1f_layout(w):
    """exact FP32: [128 CTAs][8 stages][256 groups (row_l*128 + k8) x 8 fp32]."""
    return w.view(128, 2, 8, 128, 8).permute(0, 2, 1, 3, 4).contiguous().view(-1)


def _r3_r1_smem(m, esb, cmp=1):
    blk = esb + _r3_R1_NS * (6144 if cmp else 8192)
    return 1024 + 1024 + (blk + 1023) // 1024 * 1024 + m * 16384


def _r3_r1_source(st):
    b2 = (st["cmp"]["base"] << 7) | (st["cmp"]["base"] << 23)
    s = _r3_R1_SRC
    for m in (1, 2):
        s += (f'extern "C" __global__ void __launch_bounds__(288,1) r1_m{m}(const __grid_constant__ P1 p)'
              f' {{ r1_body<{m},{_r3_R1_NS},1,{st["r1_esb"]},{st["r1_w"].data_ptr()}ull,{b2}u,{_r3_R1_PRE}>(p); }}\n')
        s += (f'extern "C" __global__ void __launch_bounds__(288,1) r1f_m{m}(const __grid_constant__ P1 p)'
              f' {{ r1_body<{m},{_r3_R1_NS},0,0,{st["r1f_w"].data_ptr()}ull,{b2}u,{_r3_R1_PRE}>(p); }}\n')
    return s


def _r3_run_r1(x, st, m, out):
    params = struct.pack("QQqQ", x.data_ptr(), out.data_ptr(), m, _r3_TRACE.data_ptr() if _r3_TRACE is not None else 0)
    params += b"\0" * ((-len(params)) % 64)
    if st.get("r1_exact"):
        st["mod"].launch(f"r1f_m{m}", 128, 288, _r3_r1_smem(m, 0, 0), params)
    else:
        st["mod"].launch(f"r1_m{m}", 128, 288, _r3_r1_smem(m, st["r1_esb"]), params)
    return out

def _r3_prepare_router(weights):
    w = weights["router"].float().contiguous()
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)
    hi_t, lo_t = _r3__perm_tiles(hi), _r3__perm_tiles(lo)
    st = {"lay": {}}
    for name, cfg in _r3_CFGS.items():
        key = (cfg[1], cfg[2], cfg[3])
        if key not in st["lay"]:
            st["lay"][key] = _r3__layout(hi_t, lo_t, *key)
    c = _r3__compress_pair(hi, lo)
    st["cmp"] = c
    st["w_cmp"], st["esc"] = {}, {}
    rows = torch.repeat_interleave(torch.arange(256, device=w.device), (c["erow"][1:] - c["erow"][:-1]).long())
    ks_ = c["ek"].long()
    for name, cfg in _r3_CFGS.items():
        key = (cfg[1], cfg[2], cfg[3])
        if key not in st["w_cmp"]:
            rr, kk = cfg[1], cfg[2]
            st["w_cmp"][key] = _r3__cmp_layout(c, *key)
            blk = (rows // rr) * (8192 // kk) + ks_ // kk
            order = torch.argsort(blk * (1 << 24) + (rows % rr) * 8192 + ks_ % kk)
            nblk = (256 // rr) * (8192 // kk)
            cnt = torch.bincount(blk, minlength=nblk)
            off = torch.zeros(nblk + 1, dtype=torch.int32, device=w.device)
            off[1:] = cnt.cumsum(0).int()
            ent = (((rows % rr) << 16) | (ks_ % kk))[order].int().contiguous()
            st["esc"][key] = (off, ent, c["ed"][order].contiguous(), int(cnt.max()) if cnt.numel() else 0)
    st["w_rc"] = _r3__rc_layout(hi, lo)
    st["w_rq"] = _r3__rq_layout(hi, lo)
    st["w_rw"] = {kk: _r3__rw_layout(hi, lo, kk) for kk in sorted({c[1] for c in _r3_RW_CFGS.values()})}
    st["w_rwc"] = {kk: _r3__rwc_layout(st["cmp"], kk) for kk in (512,)}
    st["rwc_esc"] = {kk: _r3__rwc_escapes(st["cmp"], kk, w.device) for kk in (512,)}
    st["w_rs"] = w.view(128, 2, _r3_RS_NCH, 8192 // _r3_RS_NCH).permute(0, 2, 1, 3).contiguous()
    st["w_fp32"] = w
    addr = {"rt": {k: v.data_ptr() for k, v in st["lay"].items()}, "rw": {k: v.data_ptr() for k, v in st["w_rw"].items()},
            "rs": st["w_rs"].data_ptr(), "w": w.data_ptr()}
    st["w_cmpe"] = {k: _r3__cmp_layout_esc(c, k, st["esc"][k]) for k in st["w_cmp"]}
    addr["cmpe"] = {k: (v[0].data_ptr(), v[1]) for k, v in st["w_cmpe"].items()}
    addr["b2"] = (c["base"] << 7) | (c["base"] << 23)
    _r3__rk_prepare(st, hi, lo)
    st["r1_w"], st["r1_esb"] = _r3__r1_layout(st["cmp"], w.device)
    st["r1f_w"] = _r3__r1f_layout(w)
    st["mod"] = _r3__Mod(f"#define SPINW {_r3_SPINW}\n#define RTRACE {_r3_RTRACE}\n" + _r3_source(addr=addr) + _r3_rk_source(st) + _r3_r1_source(st))
    st["imm"] = _r3_IMM
    st["ws"] = torch.zeros(1 << 20, dtype=torch.float32, device=w.device)
    # in-kernel zeroing sync state, one block of 4096 ints per (kernel, grid) key: [0] zf, [64 + b] CTA b's generation.
    # Each key always launches the same grid, so every CTA index takes part in every call of that key.
    st["zkeys"] = {k: i for i, k in enumerate([("rt16", 1), ("rt32", 1), ("rt64", 1)] + [("rw64", n) for n in (1, 2)]
                                              + [("rw128b", n) for n in range(1, 17)] + [("rw128a", n) for n in range(1, 17)])}
    st["zc"] = torch.zeros(len(st["zkeys"]) * 4096, dtype=torch.int32, device=w.device)
    st["cnt"] = torch.zeros(4096, dtype=torch.int32, device=w.device)
    st["rws"] = torch.zeros(32 * 1024 * 256, dtype=torch.float32, device=w.device)   # split-K partial slabs (32 MB)
    return st


def _r3__cfg_for(m):
    for name, cfg in _r3_CFGS.items():
        if m <= cfg[7]:
            return name, cfg


_r3_USE_RS = False
_r3_USE_RC = True
_r3_USE_RQ = False
_r3_FLAGS = 0
_r3_TRACE = None
_r3_ZMODE = False
_r3_DELAY_NS = 0
_r3_RWC = False
_r3_RWZA = False


# default dispatch (measured 2026-10-27, see NOTES.md); the module-level switches above are for ablations
_r3_DISPATCH = True
_r3_USE_RK = False   # rk (cluster split-K) measured slower or equal at every M (NOTES.md); kept for ablations
_r3_USE_R1 = False   # r1 (M<=2, 24-bit compressed SIMT) measured slower (decode-bound); kept for ablations
_r3_RW_WS = False
_r3_RW_BR = False
_r3_SPINW = int(__import__('os').environ.get('SPINW', '1'))
_r3_RTRACE = int(__import__('os').environ.get('RTRACE', '0'))
_r3_IMM = "_i"   # "_i": weight pointers compiled into the kernels (no cold param fetch before the weight stream); "": from params
_r3_SMALL_FLAGS = None   # None: per-M choice below; an int forces one rt epilogue


def _r3_run_router(x, st):
    global _r3_USE_RS, _r3_USE_RC, _r3_USE_RW, _r3_RW_SEL, _r3_FLAGS, _r3_USE_RQ
    m = x.shape[0]
    if _r3_DISPATCH:
        _r3_USE_RC = False
        _r3_USE_RQ = False
        _r3_USE_RS = m <= 4
        _r3_USE_RW = m > 64
        _r3_RW_SEL = st.get("rw_sel", {"small": "rw64", "big": "rw128b"})
        # rt: 16 = in-kernel zero CTAs; 272 = same + per-warp direct red.adds (fewer reds at M <= 8 only)
        _r3_FLAGS = _r3_SMALL_FLAGS if _r3_SMALL_FLAGS is not None else (272 if m <= 8 else 16)
    out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
    if st.get("use_r1", _r3_USE_R1) and m <= 2:
        return _r3_run_r1(x, st, m, out)
    if _r3_USE_RK and m <= _r3_RK_MAXM:
        c, cmp = _r3_rk_choice(m, st)
        if c and (c, (m + 15) // 16, cmp) in _r3_RK_KERNELS:
            return _r3_run_rk(x, st, m, c, cmp, out)
    if _r3_USE_RW and m > 64:
        name = _r3_RW_SEL["small"] if m <= 128 else _r3_RW_SEL["big"]
        tn, kk, ns = _r3_RW_CFGS[name]
        ntt = (m + tn - 1) // tn
        if name.endswith("mc") and ntt != 2:
            name = {"rw128bmc": "rw128b", "rw128cmc": "rw128b", "rw64mc": "rw64"}[name]
            tn, kk, ns = _r3_RW_CFGS[name]
            ntt = (m + tn - 1) // tn
        zk = st["zkeys"].get((name, ntt))
        zptr = st["zc"].data_ptr() + 4 * 4096 * (zk or 0)
        if _r3_RWZA and zk is not None:
            params = _r3__tmap_x2(x, m, tn) + struct.pack("QqqqQQ", st["w_rw"][kk].data_ptr(), out.data_ptr(), m, ntt,
                                                      _r3_TRACE.data_ptr() if _r3_TRACE is not None else 0, zptr)
            params += b"\0" * ((-len(params)) % 64)
            st["mod"].launch(name + "za", 4 * ntt * (8192 // kk), 192, _r3_rw_smem(name), params)
            return out
        if _r3_RWC and kk in st["w_rwc"] and st["rwc_esc"][kk][3] <= 32:
            c = st["cmp"]
            off, ent, dl, _ = st["rwc_esc"][kk]
            b2 = (c["base"] << 7) | (c["base"] << 23)
            params = _r3__tmap_x2(x, m, tn) + struct.pack("QqqqQQQQQq", st["w_rwc"][kk].data_ptr(), out.data_ptr(), m, ntt,
                                                      0, zptr, off.data_ptr(), ent.data_ptr(), dl.data_ptr(), b2)
            params += b"\0" * ((-len(params)) % 64)
            st["mod"].launch("rzero", 16, 256, 0, struct.pack("Qq", out.data_ptr(), m * 64))
            st["mod"].launch_pdl(name + "c", 4 * ntt * (8192 // kk), 160, _r3_rw_smem(name) + 0, params)
            return out
        params = _r3__tmap_x2(x, m, tn) + struct.pack("QqqqQQ", st["w_rw"][kk].data_ptr(), out.data_ptr(), m, ntt,
                                                  _r3_TRACE.data_ptr() if _r3_TRACE is not None else 0, zptr)
        params += b"\0" * ((-len(params)) % 64)
        if _r3_ZMODE and zk is not None:
            st["mod"].launch(name + "z", 4 + 4 * ntt * (8192 // kk), 160, _r3_rw_smem(name), params)
            return out
        if st.get("rw_xmc"):
            name = "rw128a" if m <= 128 else "rw256"
            tn, kk, ns = _r3_RW_CFGS[name]
            ntt = (m + tn - 1) // tn
            params = _r3__tmap_x2(x, m, tn // 2) + struct.pack("QqqqQQ", st["w_rw"][kk].data_ptr(), out.data_ptr(), m, ntt, 0, 0)
            params += b"\0" * ((-len(params)) % 64)
            st["mod"].launch("rzero", 16, 256, 0, struct.pack("Qq", out.data_ptr(), m * 64))
            st["mod"].launch_pdl(name + "_x", 4 * ntt * (8192 // kk), 160, _r3_rw_smem(name), params)
            return out
        if st.get("rw_ws", _r3_RW_WS):
            mp = ntt * tn
            s_ = 8192 // kk
            assert s_ * mp * 256 <= st["rws"].numel()
            params = _r3__tmap_x2(x, m, tn) + struct.pack("QqqqQQ", st["w_rw"][kk].data_ptr(), out.data_ptr(), m, ntt, 0,
                                                      st["rws"].data_ptr())
            params += b"\0" * ((-len(params)) % 64)
            st["mod"].launch(name + "_ws", 4 * ntt * s_, 160, _r3_rw_smem(name), params)
            st["mod"].launch_pdl(f"rred{s_}", (m * 64 + 127) // 128, 128, 0,
                                 struct.pack("QQqqq", st["rws"].data_ptr(), out.data_ptr(), m * 64, mp * 64, s_))
            return out
        st["mod"].launch("rzero", 16, 256, 0, struct.pack("Qq", out.data_ptr(), m * 64))
        st["mod"].launch_pdl(name + ("_br" if st.get("rw_br", _r3_RW_BR) else st["imm"]), 4 * ntt * (8192 // kk), 160, _r3_rw_smem(name), params)
        return out
    if _r3_USE_RQ and m <= 8:
        params = _r3__tmap_x3q(x, m) + struct.pack("QQq", st["w_rq"].data_ptr(), out.data_ptr(), m)
        params += b"\0" * ((-len(params)) % 64)
        st["mod"].launch("rq", 128, 256, _r3_RQ_SMEM, params)
        return out
    if _r3_USE_RC and m <= 64 and not (_r3_USE_RS and m <= _r3_RS_MAX):
        mt = 1 if m <= 16 else (2 if m <= 32 else 4)
        params = _r3__tmap_x3s(x, m, 16 * mt) + struct.pack("QqqQ", st["w_rc"].data_ptr(), out.data_ptr(), m,
                                                          _r3_TRACE.data_ptr() if _r3_TRACE is not None else 0)
        params += b"\0" * ((-len(params)) % 64)
        st["mod"].launch(f"rc{mt}", 8 * _r3_RC_Q, 288, _r3_rc_smem(mt), params)
        return out
    if _r3_USE_RS and m <= _r3_RS_MAX:
        params = _r3__tmap_x3(x, m) + struct.pack("QQq", st["w_rs"].data_ptr(), out.data_ptr(), m)
        params += b"\0" * ((-len(params)) % 64)
        if st.get("rs1r") and m <= 2:
            st["mod"].launch(f"rs1r{m}_i", 256, 256, 1024 + 1024 + _r3_RS_NCH * (2048 * 4 + m * 2048 * 2), params)
            return out
        st["mod"].launch(f"rs{m}{st['imm']}", 128, 256, _r3_rs_smem(m), params)
        return out
    name, cfg = _r3__cfg_for(m)
    tt, rr, kk, ks, ns, wk, wg, _ = cfg
    ntt = (m + tt - 1) // tt
    assert ntt * tt * 256 <= st["ws"].numel() and ntt * (256 // rr) <= st["cnt"].numel()
    params = _r3__tmap_x(x, m, tt) + struct.pack("QQQQqq", st["lay"][(rr, kk, ks)].data_ptr(), out.data_ptr(),
                                             st["ws"].data_ptr(), st["cnt"].data_ptr(), m, ntt)
    params += b"\0" * ((-len(params)) % 64)
    grid = ntt * (256 // rr) * (8192 // kk)
    zk = st["zkeys"].get((name, ntt))
    if _r3_FLAGS & 16 and zk is None:
        _r3_FLAGS = 4
    if st.get("rt_cmp") and zk is not None and name in ("rt16", "rt32", "rt64"):
        fl = 304 if m <= 8 else 48
        p2 = _r3__tmap_x(x, m, tt) + struct.pack("QQQQqqQQQQqQ", 0, out.data_ptr(), st["ws"].data_ptr(),
                                             st["zc"].data_ptr() + 4 * 4096 * zk, m, ntt, 0, 0, 0, 0, 0,
                                             _r3_TRACE.data_ptr() if _r3_TRACE is not None else 0)
        p2 += b"\0" * ((-len(p2)) % 64)
        st["mod"].launch(f"{name}_f{fl}_ie", 4 + grid, 32 * (wk * wg + 1), _r3_smem_bytes(cfg), p2)
        return out
    if _r3_FLAGS & 32 and zk is not None and st["esc"][(rr, kk, ks)][3] <= 32:
        c = st["cmp"]
        off, ent, dl, _ = st["esc"][(rr, kk, ks)]
        b2 = (c["base"] << 7) | (c["base"] << 23)
        p2 = _r3__tmap_x(x, m, tt) + struct.pack("QQQQqqQQQQqQ", st["w_cmp"][(rr, kk, ks)].data_ptr(), out.data_ptr(),
                                             st["ws"].data_ptr(), st["zc"].data_ptr() + 4 * 4096 * zk, m, ntt,
                                             x.data_ptr(), off.data_ptr(), ent.data_ptr(), dl.data_ptr(), b2,
                                             _r3_TRACE.data_ptr() if _r3_TRACE is not None else 0)
        p2 += b"\0" * ((-len(p2)) % 64)
        st["mod"].launch(f"{name}_f48", 4 + grid, 32 * (wk * wg + 1), _r3_smem_bytes(cfg), p2)
        return out
    _r3_FLAGS &= ~32
    if _r3_FLAGS & 16:
        tmb = _r3__tmap_x3s(x, m, tt) if _r3_FLAGS & 64 else _r3__tmap_x(x, m, tt)
        p2 = tmb + struct.pack("QQQQqqQQQQqQ", st["lay"][(rr, kk, ks)].data_ptr(), out.data_ptr(),
                                             st["ws"].data_ptr(), st["zc"].data_ptr() + 4 * 4096 * zk, m, ntt,
                                             0, 0, 0, 0, _r3_DELAY_NS, _r3_TRACE.data_ptr() if _r3_TRACE is not None else 0)
        p2 += b"\0" * ((-len(p2)) % 64)
        fl_ = (_r3_FLAGS & 464) | (512 if st.get("rt_pre") and st["imm"] else 0)
        st["mod"].launch(f"{name}_f{fl_}{st['imm']}", 4 + grid, 32 * (wk * wg + 1), _r3_smem_bytes(cfg), p2)
        return out
    if _r3_FLAGS & 8:   # ablation: red.add into out with no zeroing / ordering (wrong values, timing only)
        st["mod"].launch(f"{name}_f8", grid, 32 * (wk * wg + 1), _r3_smem_bytes(cfg), params)
        return out
    if _r3_FLAGS & 4:
        st["mod"].launch("rzero", 16, 256, 0, struct.pack("Qq", out.data_ptr(), m * 64))
        st["mod"].launch_pdl(f"{name}_f{_r3_FLAGS}", grid, 32 * (wk * wg + 1), _r3_smem_bytes(cfg), params)
        return out
    st["mod"].launch(f"{name}_f{_r3_FLAGS}", grid, 32 * (wk * wg + 1), _r3_smem_bytes(cfg), params)
    return out


# ---------------------------------------------------------------- deliverable API (agent R, 2026-11-02)
def _r3_prepare_router3(weights):
    """weights: dict with "router" [256, 8192] fp32 (or the fp32 tensor itself). Untimed: layouts + one NVRTC module."""
    w = weights["router"] if isinstance(weights, dict) else weights
    return _r3_prepare_router({"router": w})


def _r3_run_router3(x, config, state):
    """x [M, 8192] bf16 contiguous -> out [M, 256] fp32. config unused (shape-only dispatch inside)."""
    return _r3_run_router(x, state)


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


# ======================================================================== module Q2: Q2/g128q.py
"""g128q (agent Q2): generated by mkg128q.py from qg.py + the prepare_g128q / run_g128q API (end of file).
Self-contained (torch + cuda.bindings NVRTC). Derived from agent K's kg.py <- agent V's vg.py.

qg (agent Q2, derived from agent K's kg.py <- agent V's vg.py): BF16 SS-wgmma GEMM, built for m <= MMAX (128 here).
RED 5 (Q2): S-way asymmetric split-K (S - 1 helpers publish fp32 partials + flags, the finisher's producer bulk-copies them
into freed ring slots, consumers add after the loop, direct register epilogue).
 out[M, N] = x[M, K] @ W[N, K]^T.

Non-swapped orientation: A = x tile (tokens, 64 per warpgroup), B = W tile (BN rows of W = output columns), both
TMA-loaded with 128B swizzle into an NS-stage ring (one full + one empty mbarrier per stage). 1 producer warpgroup
(thread 0 issues all TMA; setmaxnreg 40) + 2 consumer warpgroups (setmaxnreg 232).
cfg = (TM, BN, LAY, S, NS):
  TM  tokens per CTA tile (64 or 128)
  BN  W rows (output columns) per CTA tile
  LAY 0: WG g owns tokens [64g, 64g + 64) x all BN columns (TM = 128)
      1: both WGs on the same 64 tokens, WG g owns columns [g BN/2, (g + 1) BN/2)            (TM = 64)
      2: both WGs on the same 64 tokens x BN columns, WG g takes the k-blocks kb = g mod 2;
         the two partials are added through smem at the end                                   (TM = 64)
  S   K splits = cluster size (1 or 2). S = 2: the two CTAs of a cluster take the two K halves of the same tile;
      the partial of the WG the partner finalises is bulk-copied smem -> DSMEM (complete_tx on the partner's
      mbarrier), the receiving WG adds it, converts to BF16 and stores (reduce-scatter, no global partials).
  NS  ring stages.
Epilogue: BF16 via stmatrix into a padded smem tile, then 16-B coalesced global stores (rows >= M masked).
Compiled with NVRTC at prepare(); W tensor maps are built once, x maps are cached per x pointer.

Agent K's additions, cfg = (TM, BN, LAY, S, NS, MC, RED, MAP, PERM, PF, PFX, EPI, PRO, PRI, LASTD, WPK, NT2, EPF, XLD):
  EPI   2: epilogue straight from the accumulators (4x4 quad transpose of bf16x2 words -> STG.128), 1: TMA store
  PRO   1: fast prologue (first two stages issued before the other barrier inits)
  PRI   cap on the stages in flight at the start (0 = all NS)
  LASTD tail priority: warpgroup 1 runs its last LASTD k-blocks after warpgroup 0 finished (named barrier 5); both
        solo phases are issued back to back; warpgroup 0's stores overlap warpgroup 1's last MMAs (LAY 0/1, S = 1)
  WPK   1/2: W prepacked (pack_w) into contiguous 128B-swizzled stage images, one 1D bulk copy per stage
        (2: packed-W address baked into the code)
  NT2   two BN-column sub-tiles per CTA (sub-tile 0 stored by the producer warpgroup's warps 1-3)   [not used]
  EPF   early prefetch.global.L2 of the first stages                                               [not used]
  XLD   x of the first XLD stages by LDG -> STS from all threads                                    [not used]
  PF/PFX tensor L2 prefetch of later stages                                                         [not used]
"""
import ctypes
import os
import struct

import torch

_q2_SMEM_MAX = 227 * 1024


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: 2026 Nathen Fernandes <https://github.com/NathenFernandes/muse-bf16-proj>
# SPDX-License-Identifier: MIT
# Adapted from NathenFernandes/muse-bf16-proj, `_mma` wgmma wrapper generator (via alygg77/okc-muse-spark-kernels).
# Changes: emits a predicate-register (setp) form of the scale-d operand; renamed.
# License text: LICENSES/MIT.txt; notices: NOTICE (repository root).
def _q2__wg_ss(n):
    nr = n // 2
    regs = ", ".join(f"%{i}" for i in range(nr))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nr))
    return (f'__device__ __forceinline__ void wg_ss_n{n}(float* d, u64 da, u64 db) {{\n'
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{nr}, %{nr + 1}, p, 1, 1, 0, 0;\\n}}\\n"\n'
            f'    : {outs} : "l"(da), "l"(db));\n}}\n')
# SPDX-SnippetEnd


_q2_HEAD = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) P { TMap tmx; TMap tmw; TMap tmo; u16* out; unsigned long long* trace; uint4* ws; unsigned* flags; const unsigned char* wp; const unsigned char* xp; int M; int MT; int NH; int PERM;
                         unsigned short perm[136]; };
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#ifdef VG_TRACE
#define TR(i) do { if (p.trace) { p.trace[blockIdx.x * 16 + (i)] = gtime(); p.trace[4096 + blockIdx.x * 16 + (i)] = clock64(); } } while (0)
#else
#define TR(i) do { } while (0)
#endif
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
__device__ __forceinline__ void tma2d(u32 dst, const void* tm, int c0, int c1, u32 bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma2d_h(u32 dst, const void* tm, int c0, int c1, u32 bar, u64 pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;"
               :: "r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void tma2d_mc(u32 dst, const void* tm, int c0, int c1, u32 bar, u16 mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;"
               :: "r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(bar), "h"(mask) : "memory");
}
__device__ __forceinline__ void mbar_arrive_cl(u32 remote) {
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(remote) : "memory");
}
__device__ __forceinline__ uint4 lds128(u32 a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }
__device__ __forceinline__ void sts128(u32 a, uint4 v) { asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
__device__ __forceinline__ u64 mkdesc(u32 saddr) {
  return (u64)((saddr & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)(1024 >> 4) << 32) | ((u64)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void fence_acc(float* a) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(a[i]) :: "memory");
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b)); return r; }
"""

_q2_BODY = r"""
#ifndef VG_BACKOFF
#define VG_BACKOFF 200
#endif
#ifdef VG_TMASTORE
#define VG_TMASTORE_ON 1
#else
#define VG_TMASTORE_ON 0
#endif
#ifdef VG_PUBBULK
#define VG_PUBBULK_ON 1
#else
#define VG_PUBBULK_ON 0
#endif
#ifdef VG_NOSETMAX
#define VG_NOSETMAX_ON 1
#else
#define VG_NOSETMAX_ON 0
#endif
#ifdef VG_NORED
#define VG_NORED_ON 1
#else
#define VG_NORED_ON 0
#endif
template <int WN> __device__ __forceinline__ void WGSS(float* d, u64 da, u64 db);
@WGSS_SPECS@
// stmatrix the 64 x WN accumulator tile of this warpgroup as BF16 into a row-major smem tile (row stride STR bytes)
template <int WN, int STR>
__device__ __forceinline__ void stage_bf16(const float* acc, u32 T, int lane, int wq) {
  constexpr int NG = WN / 8;
  const int mi = lane >> 3, rr = (mi & 1) * 8 + (lane & 7);
#pragma unroll
  for (int c = 0; c < NG; c += 2) {
    if (c + 1 < NG) {
      const u32 r0 = pack_bf2(acc[4 * c], acc[4 * c + 1]), r1 = pack_bf2(acc[4 * c + 2], acc[4 * c + 3]);
      const u32 r2 = pack_bf2(acc[4 * c + 4], acc[4 * c + 5]), r3 = pack_bf2(acc[4 * c + 6], acc[4 * c + 7]);
      const u32 a = T + (16 * wq + rr) * STR + (8 * (c + (mi >> 1))) * 2;
      asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
                   :: "r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
    } else {
      const u32 r0 = pack_bf2(acc[4 * c], acc[4 * c + 1]), r1 = pack_bf2(acc[4 * c + 2], acc[4 * c + 3]);
      const u32 a = T + (16 * wq + rr) * STR + (8 * c) * 2;
      asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};" :: "r"(a), "r"(r0), "r"(r1) : "memory");
    }
  }
}
// EPI 2: BF16 stores straight from the wgmma accumulators (no smem staging): per group of 4 n8-blocks a 4 x 4 transpose
// of bf16x2 words inside each lane quad gives every lane 16 contiguous bytes of one row -> STG.128 (8 rows x 64 B per
// warp instruction, full sectors)
template <int WN, int N>
__device__ __forceinline__ void store_direct(const float* acc, u16* out, int tok, int col, int M, int lane, int wq) {
  const int q = lane & 3, r0 = 16 * wq + (lane >> 2);
#pragma unroll
  for (int m4 = 0; m4 < WN / 32; ++m4) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {             // h = 0: row r0, h = 1: row r0 + 8
      u32 w[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) w[i] = pack_bf2(acc[4 * (4 * m4 + i) + 2 * h], acc[4 * (4 * m4 + i) + 2 * h + 1]);
      {   // stage k = 2
        const bool hi = q & 2;
        const u32 s0 = hi ? w[0] : w[2], s1 = hi ? w[1] : w[3];
        const u32 x0 = __shfl_xor_sync(0xffffffffu, s0, 2), x1 = __shfl_xor_sync(0xffffffffu, s1, 2);
        if (hi) { w[0] = x0; w[1] = x1; } else { w[2] = x0; w[3] = x1; }
      }
      {   // stage k = 1
        const bool od = q & 1;
        const u32 s0 = od ? w[0] : w[1], s1 = od ? w[2] : w[3];
        const u32 x0 = __shfl_xor_sync(0xffffffffu, s0, 1), x1 = __shfl_xor_sync(0xffffffffu, s1, 1);
        if (od) { w[0] = x0; w[2] = x1; } else { w[1] = x0; w[3] = x1; }
      }
      const int row = tok + r0 + 8 * h;
      if (row < M)
        asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(out + (size_t)row * N + col + 8 * (4 * m4 + q)),
                     "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]) : "memory");
    }
  }
  // (Q2) remaining n8 blocks when WN % 32 != 0: 4-byte stores (lane q holds columns 2q, 2q + 1 of rows r0, r0 + 8)
#pragma unroll
  for (int j = (WN / 32) * 4; j < WN / 8; ++j) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = tok + r0 + 8 * h;
      if (row < M)
        asm volatile("st.global.u32 [%0], %1;" :: "l"(out + (size_t)row * N + col + 8 * j + 2 * q),
                     "r"(pack_bf2(acc[4 * j + 2 * h], acc[4 * j + 2 * h + 1])) : "memory");
    }
  }
}
// 64 rows x WN columns of BF16 from smem tile T to out (row tok, column col); 128 threads (ct)
template <int WN, int STR, int N>
__device__ __forceinline__ void store_tile(u32 T, u16* out, int tok, int col, int M, int ct) {
  constexpr int CH = WN / 8;                       // 16-B chunks per row
#pragma unroll 4
  for (int e = ct; e < 64 * CH; e += 128) {
    const int row = e / CH, c = e % CH;
    if (tok + row < M) {
      const uint4 v = lds128(T + row * STR + c * 16);
      asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(out + (size_t)(tok + row) * N + col + c * 8),
                   "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
    }
  }
}

template <int TM, int BN, int LAY, int S, int NS, int NKB, int N, int MC, int RED, int MAP, int PERMC, int MTC, int NHC, int PF, int PFX, int EPI, int PRO, int PRI, int LASTD, int WPK, int NKBT, class WPBT, int NT2, int EPF, int XLD>
__device__ __forceinline__ void vg_body(const P& p) {
  constexpr bool FASTPRO_C = PRO == 1 && MC == 0 && !(S > 1 && RED == 0);
  static_assert(EPF == 0 || WPK > 0, "EPF: packed W");
  static_assert(NT2 == 0 || (S == 1 && LAY == 0 && MC == 0 && RED == 0), "NT2: S = 1, LAY 0");
  static_assert(WPK == 0 || MC == 0, "packed W: no multicast");
  constexpr int WN = LAY == 1 ? BN / 2 : BN;       // wgmma N per warpgroup
  constexpr int NACC = WN / 2;
  constexpr int XB = TM * 128, WB = BN * 128, SB = XB + WB;
  constexpr int NCW = (LAY == 2 ? 4 : 8) * (MC ? 2 : 1);   // consumer warps releasing one stage (both CTAs if MC)
  static_assert(!(MC && S > 1 && !VG_NORED_ON), "multicast pairs use the cluster: S = 1");
  constexpr int PB = 128 * NACC * 4;               // fp32 partial of one warpgroup (fragment order)
  constexpr int STR = WN * 2 + 16;                 // bf16 staging row stride
  static_assert(XB % 1024 == 0 && WB % 1024 == 0, "1024-B stage alignment");
  static_assert(LAY != 0 || TM == 128, "LAY 0: 2 x 64 tokens");
  static_assert(LAY == 0 || TM == 64, "LAY 1/2: 64 tokens");
  static_assert(!(LAY == 2 && S > 1 && RED < 6), "k-split layout: S = 1 (or RED >= 6)");
  static_assert(WN % 8 == 0 && WN <= 256, "wgmma N");
  static_assert(NS * SB >= PB + 2 * 64 * STR, "epilogue staging fits in the ring");
  static_assert(RED != 4 || (SB >= PB && SB >= 64 * STR && NS >= 4), "RED 4: partial halves / staging in ring slots");
  static_assert(RED != 5 || (SB >= PB && 2 * (S - 1) <= NS - 1 && EPI == 2), "RED 5: partial halves in freed ring slots");
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  const u32 bars = (smem_u32(sm_raw) + 1023) & ~1023u;
  const u32 full = bars, empty = bars + 8 * NS, rbar = bars + 16 * NS;
  const u32 ring = bars + 1024;
  const u32 rbuf = ring + NS * SB;                  // RED 0: partner partial (DSMEM); RED 2: helper partial (both WGs)
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31;
  constexpr int CLN = (S > 1 && RED == 0 && !VG_NORED_ON) ? S : (MC || RED >= 8) ? 2 : 1;   // cluster size of this launch
  // optional unit permutation (GPC-aware tile placement measured at prepare): unit = CTA or cluster
  // MTC / NHC: token tiles and tiles per k-split, compile-time (no parameter load before the first TMA)
  const int bid = !PERMC ? (int)blockIdx.x
                : CLN == 1 ? (int)p.perm[blockIdx.x] : CLN * (int)p.perm[blockIdx.x / CLN] + (int)(blockIdx.x % CLN);
  // RED 2 (asymmetric split-K): r = 0 helpers (bids [0, NH)) take k-blocks [NA, NKBT), r = 1 finishers [0, NA)
  // RED 8: cluster pair = (rank 0 helper, rank 1 finisher) of one tile
  const int r = S > 1 ? (RED >= 8 ? (bid & 1) : (MAP == 1 || RED >= 2) ? bid / NHC : bid % S) : 0;
  const int rest = RED >= 8 ? (bid >> 1) : (MAP == 1 || RED >= 2) ? bid % NHC : bid / S;
  // RED 5 (S-way asymmetric): r = 0 .. S - 2 helpers take NH5 k-blocks each, r = S - 1 the finisher takes NA5 = NKBT -
  // (S - 1) NH5 (MAP = extra k-blocks of the finisher); finisher k-blocks [0, NA5), helper r [NA5 + r NH5, + NH5)
  // RED 6 (LAY 2 only): the same k-ranges; the two warpgroups' partials are combined in smem first, the helper publishes
  // one 64 x WN fp32 partial, the finisher's producer pulls it into a dedicated buffer after the ring (rbuf)
  // RED 7 (LAY 2, S = 2): RED 6 without flags / fences / pulls: the helper stores its partial with bit 0 of every fp32
  // word = launch epoch parity (per-tile epoch counter in p.flags, read at kernel start); the finisher's warpgroup 0 polls
  // the data itself (ld.global.cg) until every word carries this launch's parity, adds, stores, then bumps the epoch
  // RED 8 (LAY 2, S = 2, cluster pair): the helper pushes its combined partial into the finisher's rbuf with st.async
  // (complete_tx on the finisher's rbar): no global memory, no fence, no polling
  // RED 9 (LAY 2, S = 2, cluster pair, symmetric, MAP 0): DSMEM reduce-scatter: rank r finalises rows [32 r, 32 r + 32)
  // of the 64-token tile; warps of warpgroup 0 holding the partner's rows push them (9 KB) into the partner's rbuf
  constexpr bool R56 = RED == 5 || RED == 6 || RED == 7 || RED >= 8;
  constexpr int NH5 = R56 ? (NKBT - MAP) / S : 1;
  constexpr int NA5 = R56 ? NKBT - (S - 1) * NH5 : 1;
  static_assert(RED != 5 || (S >= 2 && NH5 >= 1 && LAY != 2), "RED 5");
  static_assert(RED != 6 || (S >= 2 && NH5 >= 2 && LAY == 2 && EPI == 2 && MC == 0), "RED 6");
  static_assert(RED != 7 || (S == 2 && NH5 >= 2 && LAY == 2 && EPI == 2 && MC == 0), "RED 7");
  static_assert(RED < 8 || (S == 2 && NH5 >= 2 && LAY == 2 && EPI == 2 && MC == 0), "RED 8 / 9");
  const bool fin5 = R56 && r == S - 1;
  constexpr int NA = RED >= 2 ? NKB + MAP : NKB;   // RED 2: NKB = half the k-blocks, MAP = extra k-blocks of finishers
  const int nkb = R56 ? (fin5 ? NA5 : NH5) : RED >= 2 ? (r ? NA : 2 * NKB - NA) : NKB;
  const int c = MC ? (bid & 1) : 0;                // multicast pair rank
  const int base = bid >> 1;
  constexpr int NTL = NHC / MTC;                     // column tiles
  const int mt = MC == 1 ? 2 * (base % (MTC >> 1)) + c : MC == 2 ? base % MTC : (MAP == 2 && RED < 2) ? rest / NTL : rest % MTC;
  const int nt = MC == 1 ? base / (MTC >> 1) : MC == 2 ? 2 * (base / MTC) + c : (MAP == 2 && RED < 2) ? rest % NTL : rest / MTC;
  const int ntc = NT2 ? 2 * nt : nt;               // NT2: the CTA owns column tiles ntc, ntc + 1 (processed in turn)
  const int tok0 = mt * TM, col0 = ntc * BN;
  const int nkbt = NT2 ? 2 * nkb : nkb;             // stages streamed by this CTA
  const int kb0 = R56 ? (fin5 ? 0 : NA5 + r * NH5) : RED >= 2 ? (r ? 0 : NA) : r * NKB;
  static_assert(XLD == 0 || (FASTPRO_C && XLD <= 2 && NT2 == 0), "XLD: fast prologue, <= 2 stages");
  constexpr int XLDN = (XLD * TM * 8 + 383) / 384;     // 16-B chunks per thread
  uint4 xv[XLDN > 0 ? XLDN : 1];
  if constexpr (XLD > 0) {
#pragma unroll
    for (int i = 0; i < XLDN; ++i) {
      const int e = tid + 384 * i;
      xv[i] = make_uint4(0u, 0u, 0u, 0u);
      if (e < XLD * TM * 8) {
        const int st = e / (TM * 8), rr = (e / 8) % TM, cc = e % 8;
        if (tok0 + rr < p.M)
          asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(xv[i].x), "=r"(xv[i].y), "=r"(xv[i].z), "=r"(xv[i].w)
                       : "l"(p.xp + ((size_t)(tok0 + rr) * NKBT + kb0 + st) * 128 + cc * 16));
      }
    }
  }
  if constexpr (EPF > 0) {
    // early L2 prefetch (LSU prefetch.global.L2, first thing in the kernel, before the barrier init and the stage
    // copies): the CTA's first NS W stages (contiguous in the packed W) and, EPF 2, its x rows' first NS k-blocks
    if (tid < 128) {
      u64 wb;
      if constexpr (WPK == 2) wb = WPBT::get(); else wb = (u64)p.wp;
      const u64 w0 = wb + ((u64)ntc * NKBT + (u64)kb0) * WB;
#pragma unroll 1
      for (int l = tid; l < NS * WB / 128; l += 128)
        asm volatile("prefetch.global.L2 [%0];" :: "l"(w0 + (u64)l * 128) : "memory");
      if (EPF == 2) {
        constexpr int LPR = NS;                         // lines per x row (one 128 B line per k-block)
#pragma unroll 1
        for (int l = tid; l < TM * LPR; l += 128) {
          const int row = l / LPR, kk = l % LPR;
          if (tok0 + row < p.M)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(p.xp + ((size_t)(tok0 + row) * NKBT + kb0 + kk) * 128) : "memory");
        }
      }
    }
  }
  // stage loads: own tiles always; with MC the shared tile of k-block kb is issued by pair rank kb & 1 and
  // multicast into both CTAs (same smem offset, complete_tx on each CTA's full barrier)
#ifdef VG_WPOL
  u64 polw; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(polw));
#endif
#ifdef VG_XPOL
  u64 polx; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(polx));
#endif
  auto issue_own = [&](int slot, int sidx, bool withx = true) {
    const int tsub = NT2 ? (sidx >= nkb) : 0, kb = sidx - tsub * nkb;
#ifndef VG_NOX
    if (withx)
#ifdef VG_XPOL
    if (MC != 2) tma2d_h(ring + slot * SB, &p.tmx, (kb0 + kb) * 64, tok0, full + 8 * slot, polx);
#else
    if (MC != 2) tma2d(ring + slot * SB, &p.tmx, (kb0 + kb) * 64, tok0, full + 8 * slot);
#endif
#endif
#ifndef VG_NOW
    if constexpr (WPK > 0) {
      // W prepacked at prepare(): [N / BN][K / 64][BN rows][128 B, 128B-swizzled] -> one contiguous 1D bulk copy per stage
      // (DRAM-page friendly, no tensor-map descriptor fetch on the critical path)
      u64 wb;
      if constexpr (WPK >= 2) wb = WPBT::get();     // baked at NVRTC time (literal in the asm text)
      else wb = (u64)p.wp;
      // WPK 3 (Q2): stage-major images [K / 64][N / BN][BN rows][128 B]: at any time the CTAs read one contiguous region
      const u64 wimg = WPK == 3 ? ((u64)(kb0 + kb) * (u64)(N / BN) + (u64)(ntc + tsub)) : ((u64)(ntc + tsub) * NKBT + (u64)(kb0 + kb));
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   :: "r"(ring + slot * SB + XB), "l"(wb + wimg * WB), "r"(WB), "r"(full + 8 * slot) : "memory");
    } else {
#ifdef VG_WPOL
    if (MC != 1) tma2d_h(ring + slot * SB + XB, &p.tmw, (kb0 + kb) * 64, col0 + tsub * BN, full + 8 * slot, polw);
#else
    if (MC != 1) tma2d(ring + slot * SB + XB, &p.tmw, (kb0 + kb) * 64, col0 + tsub * BN, full + 8 * slot);
#endif
    }
#endif
  };
  // L2 prefetch of k-block kb (W, optionally x) PF k-blocks ahead of the ring: DRAM latency leaves the ring's
  // critical path (the ring then only waits for L2 hits)
  auto pf = [&](int kb) {
    if (PF > 0 && kb < nkb) {
      if (MC != 1) asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"(&p.tmw), "r"((kb0 + kb) * 64), "r"(col0) : "memory");
      if (PFX && MC != 2) asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"(&p.tmx), "r"((kb0 + kb) * 64), "r"(tok0) : "memory");
    }
  };
  auto issue_shared = [&](int slot, int kb) {
    if (MC && (kb & 1) == c) {
      if (MC == 1) tma2d_mc(ring + slot * SB + XB, &p.tmw, (kb0 + kb) * 64, col0, full + 8 * slot, (u16)3);
      else tma2d_mc(ring + slot * SB, &p.tmx, (kb0 + kb) * 64, tok0, full + 8 * slot, (u16)3);
    }
  };
#if defined(VG_NOX)
  constexpr u32 EXP = WB;
#elif defined(VG_NOW)
  constexpr u32 EXP = XB;
#else
  constexpr u32 EXP = SB;
#endif
  const int NPRO = NS < nkbt ? NS : nkbt;
  static_assert(!(RED == 3 && WN > 128), "RED 3 polls from the producer warpgroup: needs its full register file");

  constexpr bool FASTPRO = PRO == 1 && MC == 0 && !(S > 1 && RED == 0);
  constexpr int LASTDE = (S == 1 && (LAY == 0 || LAY == 1) && !(VG_TMASTORE_ON || EPI == 1) && MC == 0) ? LASTD : 0;
  static_assert(LASTDE == 0 || LASTDE <= NS - 2, "LASTD <= NS - 2");
#ifdef KG_NOSOLO
  constexpr bool SOLOE = false;
#else
  constexpr bool SOLOE = LASTDE > 0;
#endif
  if (tid == 0 && !FASTPRO) {
    TR(0);
#ifdef VG_TRACE
    if (p.trace) { unsigned sid; asm volatile("mov.u32 %0, %%smid;" : "=r"(sid)); p.trace[8192 + blockIdx.x] = sid; }
#endif
#ifdef VG_PFTM
    asm volatile("prefetch.tensormap [%0];" :: "l"(&p.tmx) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&p.tmw) : "memory");
#endif
#pragma unroll 1
    for (int s = 0; s < NS; ++s) { mbar_init(full + 8 * s, 1); mbar_init(empty + 8 * s, NCW); }
    mbar_init(rbar, RED == 3 ? 128 : 1);
    mbar_init(rbar + 8, 1);
    if (S > 1 && RED == 0) mbar_expect_arrive(rbar, PB);
    if (RED >= 8) mbar_expect_arrive(rbar, RED == 9 ? PB / 2 : PB);
#ifdef KG_CTAFENCE
    if (CLN == 1 && !(S > 1 && RED == 0)) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    else asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#else
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
    TR(10);
#pragma unroll 1
    for (int s = 0; s < NPRO; ++s) {
      mbar_expect_arrive(full + 8 * s, EXP);
      issue_own(s, s);
      if (s == 0) TR(13);
    }
    TR(12);
    if (PF > 0) {
#pragma unroll 1
      for (int q = 0; q < PF; ++q) pf(NPRO + q);
    }
  }
  if (tid == 0 && FASTPRO) {
    // fast prologue: the first two stages' full barriers, fence, their loads; then the rest of the barriers
    TR(0);
#ifdef VG_TRACE
    if (p.trace) { unsigned sid; asm volatile("mov.u32 %0, %%smid;" : "=r"(sid)); p.trace[8192 + blockIdx.x] = sid; }
#endif
    mbar_init(full, 1);
    mbar_init(full + 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    // XLD stages: x is loaded by all threads with LDG -> STS (below), only W is copied here
    mbar_expect_arrive(full, XLD > 0 ? (u32)WB : EXP);
    issue_own(0, 0, XLD < 1);
    TR(13);
    mbar_expect_arrive(full + 8, XLD > 1 ? (u32)WB : EXP);
    issue_own(1, 1, XLD < 2);
#pragma unroll
    for (int s = 2; s < NS; ++s) mbar_init(full + 8 * s, 1);
#pragma unroll
    for (int s = 0; s < NS; ++s) mbar_init(empty + 8 * s, NCW);
    mbar_init(rbar, RED == 3 ? 128 : 1);
    mbar_init(rbar + 8, 1);
    if (RED >= 8) mbar_expect_arrive(rbar, RED == 9 ? PB / 2 : PB);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    TR(10);
    constexpr int PRIE = PRI > 0 ? PRI : NS;      // stages in flight at the start
#pragma unroll 1
    for (int s = 2; s < NPRO && s < PRIE; ++s) {
      mbar_expect_arrive(full + 8 * s, EXP);
      issue_own(s, s);
    }
    TR(12);
    if (PF > 0) {
#pragma unroll 1
      for (int q = 0; q < PF; ++q) pf(NPRO + q);
    }
  }
  if constexpr (XLD > 0) {
    // x of the first XLD stages through the LSU (no tensor-map descriptor fetch on the critical path): LDG.128 by all
    // 384 threads (issued first, see xv), STS into the 128B-swizzled slot layout, proxy fence, then the CTA barrier
#pragma unroll
    for (int i = 0; i < XLDN; ++i) {
      const int e = tid + 384 * i;
      if (e < XLD * TM * 8) {
        const int st = e / (TM * 8), rr = (e / 8) % TM, cc = e % 8;
        sts128(ring + st * SB + rr * 128 + ((cc ^ (rr & 7)) << 4), xv[i]);
      }
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
#ifdef VG_TRACE
  if (tid == 128) TR(11);
#endif
  if (S > 1 && RED == 0 && !VG_NORED_ON) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  if (RED >= 8) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");   // rbar armed before any push
  if (MC) {   // the partner's barriers are initialised before any multicast / remote arrive
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
  }
  u32 rempty = 0;
  if (MC) asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rempty) : "r"(empty), "r"(c ^ 1));
  if (wg == 0) {
    if constexpr (RED != 3 && !VG_NOSETMAX_ON) asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    if (tid == 0) {
#pragma unroll 1
      for (int s = 0; s < NPRO; ++s) issue_shared(s, s);
      if constexpr (FASTPRO && PRI > 0) {
        // ramp: stage s is issued once stage s - PRI has landed (at most PRI stages in flight at the start)
#pragma unroll 1
        for (int s = PRI; s < NPRO; ++s) {
          mbar_wait(full + 8 * (s - PRI), 0);
          mbar_expect_arrive(full + 8 * s, EXP);
          issue_own(s, s);
        }
      }
      int slot = 0, ph = 0;
#pragma unroll 1
      for (int kb = NS; kb < nkbt; ++kb) {
        mbar_wait(empty + 8 * slot, ph);
        mbar_expect_arrive(full + 8 * slot, EXP);
        issue_own(slot, kb);
        issue_shared(slot, kb);
        pf(kb + PF);
        if (++slot == NS) { slot = 0; ph ^= 1; }
      }
      if (RED == 3 && r == 1) mbar_arrive(rbar + 8);
      if (RED == 6 && fin5) {
        mbar_expect_arrive(rbar, (S - 1) * PB);
#pragma unroll 1
        for (int h = 0; h < S - 1; ++h) {
          unsigned* fl = p.flags + rest * (S - 1) + h;
          unsigned f;
          asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(f) : "l"(fl) : "memory");
          while (f == 0u) {
            __nanosleep(VG_BACKOFF);
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(f) : "l"(fl) : "memory");
          }
          asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(fl), "r"(0u) : "memory");
          asm volatile("fence.proxy.async.global;" ::: "memory");
          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                       :: "r"(rbuf + h * PB), "l"(p.ws + ((size_t)rest * (S - 1) + h) * (PB / 16)), "r"(PB), "r"(rbar) : "memory");
        }
        TR(12);
      }
      if (RED == 5 && fin5) {
        // helper h's two warpgroup partials land in the ring slots that k-blocks nkb + 2h, nkb + 2h + 1 would use
        // (freed in order by the consumers); one expect for all of them on rbar
        mbar_expect_arrive(rbar, 2 * (S - 1) * PB);
#pragma unroll 1
        for (int h = 0; h < S - 1; ++h) {
          u32 dst[2];
#pragma unroll
          for (int q = 0; q < 2; ++q) {
            mbar_wait(empty + 8 * slot, ph);
            dst[q] = ring + slot * SB;
            if (++slot == NS) { slot = 0; ph ^= 1; }
          }
          unsigned* fl = p.flags + rest * (S - 1) + h;
          unsigned f;
          asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(f) : "l"(fl) : "memory");
          while (f == 0u) {
            __nanosleep(VG_BACKOFF);
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(f) : "l"(fl) : "memory");
          }
          asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(fl), "r"(0u) : "memory");
          asm volatile("fence.proxy.async.global;" ::: "memory");
#pragma unroll
          for (int q = 0; q < 2; ++q)
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                         :: "r"(dst[q]), "l"(p.ws + ((size_t)rest * (S - 1) + h) * (2 * PB / 16) + q * (PB / 16)), "r"(PB), "r"(rbar) : "memory");
        }
        TR(12);
      }
      if (RED == 4 && r == 1) {
        // the helper partial lands in the two ring slots that k-blocks nkb, nkb + 1 would use (freed in order)
        u32 dst[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          mbar_wait(empty + 8 * slot, ph);
          dst[h] = ring + slot * SB;
          if (++slot == NS) { slot = 0; ph ^= 1; }
        }
        unsigned* fl = p.flags + rest;
        unsigned f;
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(f) : "l"(fl) : "memory");
        while (f == 0u) {
          __nanosleep(VG_BACKOFF);
          asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(f) : "l"(fl) : "memory");
        }
        asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(fl), "r"(0u) : "memory");
        asm volatile("fence.proxy.async.global;" ::: "memory");
        mbar_expect_arrive(rbar, 2 * PB);
#pragma unroll
        for (int h = 0; h < 2; ++h)
          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                       :: "r"(dst[h]), "l"(p.ws + (size_t)rest * (2 * PB / 16) + h * (PB / 16)), "r"(PB), "r"(rbar) : "memory");
        TR(12);
      }
      if (RED == 2 && r == 1) {
        unsigned* fl = p.flags + rest;
        unsigned f;
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(f) : "l"(fl) : "memory");
        while (f == 0u) {
          __nanosleep(VG_BACKOFF);
          asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(f) : "l"(fl) : "memory");
        }
        asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(fl), "r"(0u) : "memory");
        asm volatile("fence.proxy.async.global;" ::: "memory");
        mbar_expect_arrive(rbar, 2 * PB);
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     :: "r"(rbuf), "l"(p.ws + (size_t)rest * (2 * PB / 16)), "r"(2 * PB), "r"(rbar) : "memory");
        TR(12);
      }
    }
    if (RED == 3 && r == 1) {
      // fetch the helper's partial: 16-B cp.async per chunk into rbuf; a chunk still holding the sentinel (not yet
      // written by the helper) is fetched again; then restore the sentinels and release the consumers (rbar)
      mbar_wait(rbar + 8, 0);                          // thread 0 has issued its last TMA
      __syncwarp();
      constexpr int NCH = 2 * PB / 16 / 128;           // 16-B chunks per producer thread
      const int pt = tid;                              // 0..127
      const uint4* src = p.ws + (size_t)rest * (2 * PB / 16);
      u32 pend = (NCH >= 32) ? 0xFFFFFFFFu : ((1u << NCH) - 1u);
      bool first = true;
      while (pend) {
        if (!first) __nanosleep(VG_BACKOFF);
        first = false;
#pragma unroll
        for (int i = 0; i < NCH; ++i)
          if (pend & (1u << i))
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(rbuf + (i * 128 + pt) * 16), "l"(src + i * 128 + pt) : "memory");
        asm volatile("cp.async.wait_all;" ::: "memory");
#pragma unroll
        for (int i = 0; i < NCH; ++i)
          if (pend & (1u << i)) {
            const uint4 v = lds128(rbuf + (i * 128 + pt) * 16);
            if (v.x != 0xFFFFFFFFu && v.y != 0xFFFFFFFFu && v.z != 0xFFFFFFFFu && v.w != 0xFFFFFFFFu) pend &= ~(1u << i);
          }
      }
      if (pt == 0) TR(12);
      mbar_arrive(rbar);
#pragma unroll
      for (int i = 0; i < NCH; ++i)
        asm volatile("st.global.cg.v4.u32 [%0], {%1,%1,%1,%1};" :: "l"(src + i * 128 + pt), "r"(0xFFFFFFFFu) : "memory");
    }
    if (NT2 && tid >= 32) {
      // tile 0's epilogue stores, overlapped with tile 1's main loop: the consumers stage tile 0 as BF16 in the
      // staging buffer (after the ring) and arrive on named barrier 6; warps 1..3 of this warpgroup store it
      asm volatile("bar.sync 6, 352;" ::: "memory");
      constexpr int CH0 = BN / 8, STR0 = BN * 2 + 16;
      const u32 S0 = ring + NS * SB;
#pragma unroll 2
      for (int e = tid - 32; e < TM * CH0; e += 96) {
        const int row = e / CH0, cc = e % CH0;
        if (tok0 + row < p.M) {
          const uint4 v = lds128(S0 + row * STR0 + cc * 16);
          asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(p.out + (size_t)(tok0 + row) * N + col0 + cc * 8),
                       "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
        }
      }
    }
    if (MC) {
      __syncwarp();
      asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
      asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    }
    if (RED >= 8) { __syncwarp(); asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }
    return;
  }
  if constexpr (RED != 3 && !VG_NOSETMAX_ON) asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
  const int g = __shfl_sync(0xffffffffu, wg, 0) - 1, ct = tid & 127;
  const int wq = __shfl_sync(0xffffffffu, (tid >> 5) & 3, 0);
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = 0.f;
  const u32 xoff = LAY == 0 ? g * 8192 : 0;
  const u32 woff = LAY == 1 ? g * (WN * 128) : 0;
  const u64 da0 = mkdesc(ring + xoff), db0 = mkdesc(ring + XB + woff);
  u32 ep7 = 0;
  if (RED == 7 && g == 0) asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(ep7) : "l"(p.flags + rest) : "memory");
  if (ct == 0) TR(1 + g);
  if constexpr (LAY != 2) {
#ifdef VG_TRACE
    mbar_wait(full, 0);
    if (ct == 0 && g == 0) TR(14);
#endif
    int slot = 0, ph = 0, pslot = 0;
#pragma unroll 1
    for (int kb = 0; kb < nkbt; ++kb) {
      // tail priority: warpgroup 1 yields the tensor pipe for the last LASTD k-blocks until warpgroup 0 is done,
      // so that warpgroup 0's epilogue overlaps warpgroup 1's last MMAs
      if (LASTDE > 0 && g == 1 && kb == nkbt - LASTDE) asm volatile("bar.sync 5, 256;" ::: "memory");
      mbar_wait(full + 8 * slot, ph);
      const u64 so = (u64)(slot * (SB >> 4));
      wg_fence();
#ifndef VG_NOMMA
#pragma unroll
      for (int j = 0; j < 4; ++j) WGSS<WN>(acc, da0 + so + 2 * j, db0 + so + 2 * j);
#endif
      wg_commit();
      // tail priority solo phase: each warpgroup runs its last LASTD k-blocks alone on the tensor pipe; issue them
      // back to back (no wait, no slot release: the producer has issued every stage by then, LASTD <= NS - 2)
      const bool solo = SOLOE && kb >= nkbt - LASTDE;
      if (!solo) {
        wg_wait<1>(); fence_acc<NACC>(acc);
        if (kb > 0) { __syncwarp(); if (lane == 0) { mbar_arrive(empty + 8 * pslot); if (MC) mbar_arrive_cl(rempty + 8 * pslot); } }
      }
      pslot = slot;
      if (++slot == NS) { slot = 0; ph ^= 1; }
      if (NT2 && kb == nkb - 1) {
        // end of tile 0: stage it as BF16 for the store warps, restart the accumulators for tile 1
        wg_wait<0>(); fence_acc<NACC>(acc);
        stage_bf16<WN, BN * 2 + 16>(acc, ring + NS * SB + g * (64 * (BN * 2 + 16)), lane, wq);
        asm volatile("bar.arrive 6, 352;" ::: "memory");
#ifdef VG_TRACE
        if (ct == 0 && g == 0) TR(9);
#endif
#pragma unroll
        for (int i = 0; i < NACC; ++i) acc[i] = 0.f;
      }
    }
    wg_wait<0>(); fence_acc<NACC>(acc);
    if (LASTDE > 0 && g == 0) asm volatile("bar.arrive 5, 256;" ::: "memory");
  } else {
    int slot = g, ph = 0, pslot = -1;
    if (slot >= NS) { slot -= NS; ph ^= 1; }
#pragma unroll 1
    for (int kb = g; kb < nkb; kb += 2) {
      mbar_wait(full + 8 * slot, ph);
      const u64 so = (u64)(slot * (SB >> 4));
      wg_fence();
#ifndef VG_NOMMA
#pragma unroll
      for (int j = 0; j < 4; ++j) WGSS<WN>(acc, da0 + so + 2 * j, db0 + so + 2 * j);
#endif
      wg_commit();
      wg_wait<1>(); fence_acc<NACC>(acc);
      if (pslot >= 0) { __syncwarp(); if (lane == 0) { mbar_arrive(empty + 8 * pslot); if (MC) mbar_arrive_cl(rempty + 8 * pslot); } }
      pslot = slot;
      slot += 2; if (slot >= NS) { slot -= NS; ph ^= 1; }
    }
    wg_wait<0>(); fence_acc<NACC>(acc);
  }
  if (ct == 0) TR(3 + g);
  // (LAY 2 always: warpgroup 1 writes its partial into ring slots 0-1 next, which warpgroup 0 may still be reading)
  if (LASTDE == 0 && (!(EPI == 2 && S == 1) || LAY == 2) && RED < 5) asm volatile("bar.sync 1, 256;" ::: "memory");      // both warpgroups are done with the ring
#ifdef VG_TRACE
  if (S == 1 && ct == 0 && g == 0) TR(5);
#endif
  // bf16 staging tile of this warpgroup; with LASTD: ring slots of k-blocks nkb - LASTD - 2 + g (free: both warpgroups
  // are past them and the producer never refills them)
  const u32 T = LASTDE > 0 ? ring + (u32)((nkbt - LASTDE - 2 + g) % NS) * SB : ring + PB + g * (64 * STR);
  const int tokw = tok0 + (LAY == 0 ? 64 * g : 0);
  const int colw = col0 + (NT2 ? BN : 0) + (LAY == 1 ? WN * g : 0);
  if constexpr (LAY == 2 && RED == 9) {
    constexpr int NQ = NACC / 4;
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");   // the partner's rbar is armed (long done)
    asm volatile("bar.sync 1, 256;" ::: "memory");
    if (g == 1) {
#pragma unroll
      for (int i = 0; i < NQ; ++i)
        sts128(ring + (i * 128 + ct) * 16, make_uint4(__float_as_uint(acc[4 * i]), __float_as_uint(acc[4 * i + 1]),
                                                      __float_as_uint(acc[4 * i + 2]), __float_as_uint(acc[4 * i + 3])));
    }
    asm volatile("bar.sync 1, 256;" ::: "memory");
    if (g == 0) {
#pragma unroll
      for (int i = 0; i < NQ; ++i) {
        const uint4 v = lds128(ring + (i * 128 + ct) * 16);
        acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
        acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
      }
      const int slot64 = (wq & 1) * 32 + lane;           // position of this lane among the 64 threads of a row half
      if ((wq >> 1) != r) {   // rows of the partner's half: push into its rbuf (complete_tx on its rbar)
        u32 rdst, rmb;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rdst) : "r"(rbuf + slot64 * 16), "r"(r ^ 1));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rmb) : "r"(rbar), "r"(r ^ 1));
#pragma unroll
        for (int i = 0; i < NQ; ++i)
          asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1,%2,%3,%4}, [%5];"
                       :: "r"(rdst + i * 64 * 16), "r"(__float_as_uint(acc[4 * i])), "r"(__float_as_uint(acc[4 * i + 1])),
                          "r"(__float_as_uint(acc[4 * i + 2])), "r"(__float_as_uint(acc[4 * i + 3])), "r"(rmb) : "memory");
        if (lane == 0 && (wq & 1) == 0) TR(9);
      } else {                // my half: wait for the partner's rows, add, store
        mbar_wait(rbar, 0);
        if (lane == 0 && (wq & 1) == 0) TR(5);
#pragma unroll
        for (int i = 0; i < NQ; ++i) {
          const uint4 v = lds128(rbuf + (i * 64 + slot64) * 16);
          acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
          acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
        }
        store_direct<WN, N>(acc, p.out, tokw, colw, p.M, lane, wq);
      }
    }
  } else if constexpr (LAY == 2 && RED == 8) {
    constexpr int NQ = NACC / 4;
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");   // pairs the arrive after the prologue (long done)
    asm volatile("bar.sync 1, 256;" ::: "memory");          // both warpgroups are done reading the ring
    if (g == 1) {
#pragma unroll
      for (int i = 0; i < NQ; ++i)
        sts128(ring + (i * 128 + ct) * 16, make_uint4(__float_as_uint(acc[4 * i]), __float_as_uint(acc[4 * i + 1]),
                                                      __float_as_uint(acc[4 * i + 2]), __float_as_uint(acc[4 * i + 3])));
    }
    asm volatile("bar.sync 1, 256;" ::: "memory");
    if (g == 0) {
#pragma unroll
      for (int i = 0; i < NQ; ++i) {
        const uint4 v = lds128(ring + (i * 128 + ct) * 16);
        acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
        acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
      }
      if (!fin5) {   // helper: push the combined partial into the finisher's rbuf (DSMEM), complete_tx on its rbar
        u32 rdst, rmb;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rdst) : "r"(rbuf + ct * 16), "r"(1));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rmb) : "r"(rbar), "r"(1));
#pragma unroll
        for (int i = 0; i < NQ; ++i)
          asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1,%2,%3,%4}, [%5];"
                       :: "r"(rdst + i * 128 * 16), "r"(__float_as_uint(acc[4 * i])), "r"(__float_as_uint(acc[4 * i + 1])),
                          "r"(__float_as_uint(acc[4 * i + 2])), "r"(__float_as_uint(acc[4 * i + 3])), "r"(rmb) : "memory");
        if (ct == 0) TR(9);
      } else {
        mbar_wait(rbar, 0);
        if (ct == 0) TR(5);
#pragma unroll
        for (int i = 0; i < NQ; ++i) {
          const uint4 v = lds128(rbuf + (i * 128 + ct) * 16);
          acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
          acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
        }
        store_direct<WN, N>(acc, p.out, tokw, colw, p.M, lane, wq);
      }
    }
  } else if constexpr (LAY == 2 && RED == 7) {
    constexpr int NQ = NACC / 4;
    asm volatile("bar.sync 1, 256;" ::: "memory");          // both warpgroups are done reading the ring
    if (g == 1) {
#pragma unroll
      for (int i = 0; i < NQ; ++i)
        sts128(ring + (i * 128 + ct) * 16, make_uint4(__float_as_uint(acc[4 * i]), __float_as_uint(acc[4 * i + 1]),
                                                      __float_as_uint(acc[4 * i + 2]), __float_as_uint(acc[4 * i + 3])));
    }
    asm volatile("bar.sync 1, 256;" ::: "memory");
    if (g == 0) {
#pragma unroll
      for (int i = 0; i < NQ; ++i) {
        const uint4 v = lds128(ring + (i * 128 + ct) * 16);
        acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
        acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
      }
      const u32 par = ep7 & 1u;
      uint4* wsl = p.ws + (size_t)rest * (PB / 16) + ct;
      if (!fin5) {   // helper: tagged partial (bit 0 := epoch parity, <= 1 ulp), no fence, no flag
#pragma unroll
        for (int i = 0; i < NQ; ++i)
          asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(wsl + i * 128),
                       "r"((__float_as_uint(acc[4 * i]) & ~1u) | par), "r"((__float_as_uint(acc[4 * i + 1]) & ~1u) | par),
                       "r"((__float_as_uint(acc[4 * i + 2]) & ~1u) | par), "r"((__float_as_uint(acc[4 * i + 3]) & ~1u) | par)
                       : "memory");
        if (ct == 0) TR(9);
      } else {
        uint4 v[NQ];
        u32 pend = (1u << NQ) - 1u;
        bool first = true;
        int npoll = 0;
        while (pend) {
          if (!first) __nanosleep(64);
          first = false;
          ++npoll;
#pragma unroll
          for (int i = 0; i < NQ; ++i)
            if (pend & (1u << i))
              asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v[i].x), "=r"(v[i].y), "=r"(v[i].z), "=r"(v[i].w)
                           : "l"(wsl + i * 128) : "memory");
#pragma unroll
          for (int i = 0; i < NQ; ++i)
            if ((pend & (1u << i)) && ((v[i].x & 1u) == par) && ((v[i].y & 1u) == par) && ((v[i].z & 1u) == par) &&
                ((v[i].w & 1u) == par)) pend &= ~(1u << i);
        }
        if (ct == 0) { TR(5); if (p.trace) p.trace[blockIdx.x * 16 + 15] = npoll; }
#pragma unroll
        for (int i = 0; i < NQ; ++i) {
          acc[4 * i] += __uint_as_float(v[i].x); acc[4 * i + 1] += __uint_as_float(v[i].y);
          acc[4 * i + 2] += __uint_as_float(v[i].z); acc[4 * i + 3] += __uint_as_float(v[i].w);
        }
        store_direct<WN, N>(acc, p.out, tokw, colw, p.M, lane, wq);
        // every thread of the warpgroup has read this launch's data: bump the tile epoch for the next launch
        asm volatile("bar.sync 2, 128;" ::: "memory");
        if (ct == 0) asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(p.flags + rest), "r"(ep7 + 1u) : "memory");
      }
    }
  } else if constexpr (LAY == 2 && RED == 6) {
    constexpr int NQ = NACC / 4;
    asm volatile("bar.sync 1, 256;" ::: "memory");          // both warpgroups are done reading the ring
    if (g == 1) {
#pragma unroll
      for (int i = 0; i < NQ; ++i)
        sts128(ring + (i * 128 + ct) * 16, make_uint4(__float_as_uint(acc[4 * i]), __float_as_uint(acc[4 * i + 1]),
                                                      __float_as_uint(acc[4 * i + 2]), __float_as_uint(acc[4 * i + 3])));
    }
    asm volatile("bar.sync 1, 256;" ::: "memory");
    if (g == 0) {
#pragma unroll
      for (int i = 0; i < NQ; ++i) {
        const uint4 v = lds128(ring + (i * 128 + ct) * 16);
        acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
        acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
      }
      if (!fin5) {   // helper r: publish the CTA partial (fragment order), one release flag
        uint4* wsl = p.ws + ((size_t)rest * (S - 1) + r) * (PB / 16) + ct;
#pragma unroll
        for (int i = 0; i < NQ; ++i)
          asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(wsl + i * 128), "r"(__float_as_uint(acc[4 * i])),
                       "r"(__float_as_uint(acc[4 * i + 1])), "r"(__float_as_uint(acc[4 * i + 2])), "r"(__float_as_uint(acc[4 * i + 3]))
                       : "memory");
        asm volatile("bar.sync 2, 128;" ::: "memory");
        if (ct == 0) {
          asm volatile("fence.acq_rel.gpu;" ::: "memory");
          asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(p.flags + rest * (S - 1) + r), "r"(1u) : "memory");
          TR(9);
        }
      } else {
        mbar_wait(rbar, 0);
        if (ct == 0) TR(5);
#pragma unroll 1
        for (int h = 0; h < S - 1; ++h) {
#pragma unroll
          for (int i = 0; i < NQ; ++i) {
            const uint4 v = lds128(rbuf + h * PB + (i * 128 + ct) * 16);
            acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
            acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
          }
        }
        store_direct<WN, N>(acc, p.out, tokw, colw, p.M, lane, wq);
      }
    }
  } else if constexpr (LAY == 2) {
    if (g == 1) {
#pragma unroll
      for (int i = 0; i < NACC / 4; ++i)
        sts128(ring + (i * 128 + ct) * 16, make_uint4(__float_as_uint(acc[4 * i]), __float_as_uint(acc[4 * i + 1]),
                                                      __float_as_uint(acc[4 * i + 2]), __float_as_uint(acc[4 * i + 3])));
    }
    asm volatile("bar.sync 1, 256;" ::: "memory");
    if (g == 0) {
#pragma unroll
      for (int i = 0; i < NACC / 4; ++i) {
        const uint4 v = lds128(ring + (i * 128 + ct) * 16);
        acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
        acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
      }
      if constexpr (EPI == 2) {
        store_direct<WN, N>(acc, p.out, tokw, colw, p.M, lane, wq);
      } else {
        stage_bf16<WN, STR>(acc, T, lane, wq);
        asm volatile("bar.sync 2, 128;" ::: "memory");
        store_tile<WN, STR, N>(T, p.out, tokw, colw, p.M, ct);
      }
    }
  } else if constexpr ((S == 1 || VG_NORED_ON) && (VG_TMASTORE_ON || EPI == 1) && WN % 64 == 0) {
    // BF16 tile -> 128B-swizzled [64 rows][64 cols] sub-tiles (stmatrix, conflict-free) -> TMA tensor store
    // (rows >= M are clipped by the tensor map); the CTA only waits for the smem reads of the store
    constexpr int NSUB = WN / 64;
    const u32 T0 = (ring + g * NSUB * 8192 + 1023) & ~1023u;
    const int mi = lane >> 3, rr = (mi & 1) * 8 + (lane & 7), row = 16 * wq + rr;
#pragma unroll
    for (int c = 0; c < WN / 8; c += 2) {
      const int col0 = 8 * (c + (mi >> 1));
      const u32 a = T0 + (col0 >> 6) * 8192 + row * 128 + ((((col0 & 63) >> 3) ^ (row & 7)) << 4);
      const u32 r0 = pack_bf2(acc[4 * c], acc[4 * c + 1]), r1 = pack_bf2(acc[4 * c + 2], acc[4 * c + 3]);
      const u32 r2 = pack_bf2(acc[4 * c + 4], acc[4 * c + 5]), r3 = pack_bf2(acc[4 * c + 6], acc[4 * c + 7]);
      asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
                   :: "r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("bar.sync %0, 128;" :: "r"(2 + g) : "memory");
    if (ct == 0) {
#pragma unroll
      for (int q = 0; q < NSUB; ++q)
        asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                     :: "l"(&p.tmo), "r"(colw + 64 * q), "r"(tokw), "r"(T0 + q * 8192) : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
    }
  } else if constexpr ((S == 1 || VG_NORED_ON) && EPI == 2) {
    store_direct<WN, N>(acc, p.out, tokw, colw, p.M, lane, wq);
  } else if constexpr (S == 1 || VG_NORED_ON) {
    stage_bf16<WN, STR>(acc, T, lane, wq);
    asm volatile("bar.sync %0, 128;" :: "r"(2 + g) : "memory");
#ifdef VG_TRACE
    if (ct == 0 && g == 0) TR(6);
#endif
    store_tile<WN, STR, N>(T, p.out, tokw, colw, p.M, ct);
  } else if constexpr (RED == 5) {
    constexpr int NQ = NACC / 4;
    if (!fin5) {   // helper r: publish my partial (both warpgroups, fragment order), one release flag
      uint4* wsl = p.ws + ((size_t)rest * (S - 1) + r) * (2 * PB / 16) + (size_t)g * NQ * 128 + ct;
#pragma unroll
      for (int i = 0; i < NQ; ++i)
        asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(wsl + i * 128), "r"(__float_as_uint(acc[4 * i])),
                     "r"(__float_as_uint(acc[4 * i + 1])), "r"(__float_as_uint(acc[4 * i + 2])), "r"(__float_as_uint(acc[4 * i + 3]))
                     : "memory");
      asm volatile("bar.sync 1, 256;" ::: "memory");
      if (g == 0 && ct == 0) {
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(p.flags + rest * (S - 1) + r), "r"(1u) : "memory");
      }
      if (ct == 0) TR(9);
    } else {
      mbar_wait(rbar, 0);
      if (ct == 0) TR(5 + g);
#pragma unroll 1
      for (int h = 0; h < S - 1; ++h) {
        const u32 rsrc = ring + (u32)((nkb + 2 * h + g) % NS) * SB;
#pragma unroll
        for (int i = 0; i < NQ; ++i) {
          const uint4 v = lds128(rsrc + (i * 128 + ct) * 16);
          acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
          acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
        }
      }
      store_direct<WN, N>(acc, p.out, tokw, colw, p.M, lane, wq);
    }
  } else if constexpr (RED == 2 || RED == 3 || RED == 4) {
    constexpr int NQ = NACC / 4;
    if (r == 0 && VG_PUBBULK_ON && RED == 4) {   // helper: stage the partial in smem, one bulk store, flag
#pragma unroll
      for (int i = 0; i < NQ; ++i)
        sts128(ring + ((g * NQ + i) * 128 + ct) * 16, make_uint4(__float_as_uint(acc[4 * i]), __float_as_uint(acc[4 * i + 1]),
                                                                __float_as_uint(acc[4 * i + 2]), __float_as_uint(acc[4 * i + 3])));
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      asm volatile("bar.sync 1, 256;" ::: "memory");
      if (g == 0 && ct == 0) {
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                     :: "l"(p.ws + (size_t)rest * (2 * PB / 16)), "r"(ring), "r"(2 * PB) : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(p.flags + rest), "r"(1u) : "memory");
        TR(9);
      }
    } else if (r == 0) {   // helper: publish my partial (both WGs) for the finisher, then exit
      uint4* slot = p.ws + (size_t)rest * (2 * PB / 16) + (size_t)g * NQ * 128 + ct;
#pragma unroll
      for (int i = 0; i < NQ; ++i)
        asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(slot + i * 128), "r"(__float_as_uint(acc[4 * i])),
                     "r"(__float_as_uint(acc[4 * i + 1])), "r"(__float_as_uint(acc[4 * i + 2])), "r"(__float_as_uint(acc[4 * i + 3]))
                     : "memory");
      if constexpr (RED == 2 || RED == 4) {
#ifdef VG_FENCE_ALL
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
#endif
        asm volatile("bar.sync 1, 256;" ::: "memory");
        if (g == 0 && ct == 0) {
          asm volatile("fence.acq_rel.gpu;" ::: "memory");   // cumulative: covers the CTA's stores seen through bar.sync
          asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(p.flags + rest), "r"(1u) : "memory");
        }
      }
      if (ct == 0) TR(9);
    } else {
      mbar_wait(rbar, 0);
      if (ct == 0) TR(5 + g);
      const u32 rsrc = RED == 4 ? ring + ((nkb + g) % NS) * SB : rbuf + g * NQ * 128 * 16;
      const u32 T4 = RED == 4 ? ring + ((nkb + 2 + g) % NS) * SB : T;
#pragma unroll
      for (int i = 0; i < NQ; ++i) {
        const uint4 v = lds128(rsrc + (i * 128 + ct) * 16);
        acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
        acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
      }
      stage_bf16<WN, STR>(acc, T4, lane, wq);
      asm volatile("bar.sync %0, 128;" :: "r"(2 + g) : "memory");
      store_tile<WN, STR, N>(T4, p.out, tokw, colw, p.M, ct);
    }
  } else if constexpr (RED == 1) {
    // global exchange without flags: slots hold the sentinel 0xFFFFFFFF (never a finite partial) until written;
    // the receiver polls until no word of its slot is the sentinel, then restores the sentinel for the next call
    constexpr int NQ = NACC / 4;
    uint4* slot = p.ws + (size_t)(rest * 2 + g) * NQ * 128 + ct;
    if (g != r) {
#pragma unroll
      for (int i = 0; i < NQ; ++i)
        asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(slot + i * 128), "r"(__float_as_uint(acc[4 * i])),
                     "r"(__float_as_uint(acc[4 * i + 1])), "r"(__float_as_uint(acc[4 * i + 2])), "r"(__float_as_uint(acc[4 * i + 3]))
                     : "memory");
      if (ct == 0) TR(9);
    } else {
      if (ct == 0) TR(10);
      int npoll = 0;
      uint4 v[NQ];
      u32 pend = (1u << NQ) - 1u;
      bool first = true;
      while (pend) {
        if (!first) __nanosleep(VG_BACKOFF);
        first = false;
        ++npoll;
#pragma unroll
        for (int i = 0; i < NQ; ++i)
          if (pend & (1u << i))
            asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v[i].x), "=r"(v[i].y), "=r"(v[i].z), "=r"(v[i].w)
                         : "l"(slot + i * 128) : "memory");
#pragma unroll
        for (int i = 0; i < NQ; ++i)
          if ((pend & (1u << i)) && v[i].x != 0xFFFFFFFFu && v[i].y != 0xFFFFFFFFu && v[i].z != 0xFFFFFFFFu &&
              v[i].w != 0xFFFFFFFFu) pend &= ~(1u << i);
      }
      if (ct == 0) { TR(5 + g); if (p.trace) p.trace[blockIdx.x * 16 + 11] = npoll; }
#pragma unroll
      for (int i = 0; i < NQ; ++i) {
        asm volatile("st.global.cg.v4.u32 [%0], {%1,%1,%1,%1};" :: "l"(slot + i * 128), "r"(0xFFFFFFFFu) : "memory");
        acc[4 * i] += __uint_as_float(v[i].x); acc[4 * i + 1] += __uint_as_float(v[i].y);
        acc[4 * i + 2] += __uint_as_float(v[i].z); acc[4 * i + 3] += __uint_as_float(v[i].w);
      }
      stage_bf16<WN, STR>(acc, T, lane, wq);
      asm volatile("bar.sync %0, 128;" :: "r"(2 + g) : "memory");
      store_tile<WN, STR, N>(T, p.out, tokw, colw, p.M, ct);
    }
  } else {
    if (g != r) {   // sender: my partial goes to the partner, whose warpgroup g finalises this part
#pragma unroll
      for (int i = 0; i < NACC / 4; ++i)
        sts128(ring + (i * 128 + ct) * 16, make_uint4(__float_as_uint(acc[4 * i]), __float_as_uint(acc[4 * i + 1]),
                                                      __float_as_uint(acc[4 * i + 2]), __float_as_uint(acc[4 * i + 3])));
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      asm volatile("bar.sync %0, 128;" :: "r"(2 + g) : "memory");
      asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
      if (ct == 0) {
        u32 rdst, rmb;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rdst) : "r"(rbuf), "r"(1 - r));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rmb) : "r"(rbar), "r"(1 - r));
        asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     :: "r"(rdst), "r"(ring), "r"(PB), "r"(rmb) : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
      }
    } else {
      asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
      mbar_wait(rbar, 0);
      if (ct == 0) TR(5 + g);
#pragma unroll
      for (int i = 0; i < NACC / 4; ++i) {
        const uint4 v = lds128(rbuf + (i * 128 + ct) * 16);
        acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
        acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
      }
      stage_bf16<WN, STR>(acc, T, lane, wq);
      asm volatile("bar.sync %0, 128;" :: "r"(2 + g) : "memory");
      store_tile<WN, STR, N>(T, p.out, tokw, colw, p.M, ct);
    }
  }
  if (ct == 0) TR(7 + g);
  if (MC) {
    asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
  }
}
"""


_q2_PROBE_SRC = r"""
__device__ __forceinline__ void vg_probe_body(int* out) {
  unsigned sid; asm volatile("mov.u32 %0, %%smid;" : "=r"(sid));
  if (threadIdx.x == 0) out[blockIdx.x] = (int)sid;
  const unsigned long long t0 = gtime();
  while (gtime() - t0 < 30000ull) { }
}
extern "C" __global__ void __launch_bounds__(384, 1) vg_probe1(int* out) { vg_probe_body(out); }
extern "C" __global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(384, 1) vg_probe2(int* out) { vg_probe_body(out); }
extern "C" __global__ void __launch_bounds__(128, 1) vg_probex(int* out) { vg_probe_body(out); }
"""


def _q2__specs(wns):
    return "".join(f"template <> __device__ __forceinline__ void WGSS<{n}>(float* d, u64 da, u64 db) {{ wg_ss_n{n}(d, da, db); }}\n"
                   for n in sorted(wns))


class _q2_Module:
    def __init__(self, src, name="vg.cu", opts=()):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        o = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        o += [x.encode() for x in opts]
        if os.environ.get("CU_LINEINFO"):
            o.append(b"-lineinfo")
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(o), o)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        self.log = log.decode(errors="replace").strip("\x00 \n")
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + self.log)
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        self.cubin = cubin
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err
        self.funcs = {}

    def func(self, name, smem):
        key = (name, smem)
        f = self.funcs.get(key)
        if f is None:
            d = self.driver
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            assert int(err) == 0, (name, err)
            if smem > 48 * 1024:
                (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
                assert int(err) == 0, (name, smem, err)
            self.funcs[key] = f
        return f

    def attrs(self, name, smem=0):
        d = self.driver
        f = self.func(name, smem)
        out = {}
        for k in ("NUM_REGS", "LOCAL_SIZE_BYTES"):
            err, v = d.cuFuncGetAttribute(getattr(d.CUfunction_attribute, "CU_FUNC_ATTRIBUTE_" + k), f)
            out[k] = v
        return out

    def launch(self, name, grid, block, smem, params):
        d = self.driver
        f = self.func(name, smem)
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        assert int(err) == 0, (name, err)

    def max_active_clusters(self, name, smem, block, csize, grid):
        d = self.driver
        f = self.func(name, smem)
        cfg = d.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block, 1, 1
        cfg.sharedMemBytes = smem
        at = d.CUlaunchAttribute()
        at.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = csize, 1, 1
        cfg.attrs = [at]
        cfg.numAttrs = 1
        err, ncl = d.cuOccupancyMaxActiveClusters(f, cfg)
        return int(ncl) if int(err) == 0 else 0


def _q2__tmap2d(ptr, rows, cols, box_rows, box_cols=64):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(2), ptr,
        [d.cuuint64_t(cols), d.cuuint64_t(rows)], [d.cuuint64_t(cols * 2)], [d.cuuint32_t(box_cols), d.cuuint32_t(box_rows)],
        [d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


_q2_NCFG = 19


def _q2_norm(cfg):
    """(TM, BN, LAY, S, NS[, MC, RED, MAP, PERM, PF, PFX, EPI, PRO, PRI, LASTD, WPK, NT2, EPF, XLD])"""
    cfg = tuple(cfg)
    return cfg + (0,) * (_q2_NCFG - len(cfg))


def _q2_kname(layer, cfg):
    return f"kg_{layer}_" + "_".join(map(str, _q2_norm(cfg)))


def _q2_smem_bytes(cfg):
    TM, BN, LAY, S, NS, MC, RED, MAP, PERM, PF, PFX, EPI, PRO, PRI, LASTD, WPK, NT2, EPF, XLD = _q2_norm(cfg)
    WN = BN // 2 if LAY == 1 else BN
    return (2048 + NS * (TM * 128 + BN * 128) + (128 * (WN // 2) * 4 * (1 if RED == 0 else 2) if S > 1 and RED in (0, 2, 3) else 0)  # RED 4: receive in the ring
            + ((S - 1) * 128 * (WN // 2) * 4 if S > 1 and RED in (6, 8, 9) else 0)
            + (TM * (BN * 2 + 16) if NT2 else 0))


def _q2_build(specs, defines="", wpb=None, mmax=128):
    wpb = wpb or {}
    """specs: list of (layer, n, k, cfg). -> (Module, {(layer, cfg): entry})"""
    wns = set()
    src_k = ""
    ents = {}
    for layer, n, k, cfg in specs:
        cfg = _q2_norm(cfg)
        TM, BN, LAY, S, NS, MC, RED, MAP, PERM, PF, PFX, EPI, PRO, PRI, LASTD, WPK, NT2, EPF, XLD = cfg
        WN = BN // 2 if LAY == 1 else BN
        wns.add(WN)
        nkb = k // 64 // S
        assert (k % (64 * S) == 0 or (RED in (5, 6, 7, 8, 9) and k % 64 == 0)) and n % BN == 0
        kn = _q2_kname(layer, cfg)
        MTC = mmax // TM                      # grid is built for m <= mmax (same tile count as m = mmax)
        if MC == 1:
            MTC += MTC & 1
        NHC = (n // BN // (2 if NT2 else 1)) * MTC
        CL = S if (S > 1 and RED == 0) else (2 if (MC or RED >= 8) else 1)
        assert RED != 2 or (S == 2 and LAY in (0, 1))
        if "VG_NORED" in defines:
            CL = 2 if MC else 1
        if "VG_DUMMYCL" in defines:
            CL = 2
        cl = f"__cluster_dims__({CL}, 1, 1) " if CL > 1 else ""
        _wa = wpb.get((layer, cfg), 0)
        # 64-bit literal + a runtime zero ptxas cannot fold (%dynamic_smem_size >> 31): a folded 64-bit immediate added
        # to an address lost its high half in some instantiations (ptxas bug seen 2026-11-02)
        src_k += (f'struct WPB_{kn} {{ static __device__ __forceinline__ u64 get() {{ u64 r; '
                  f'asm volatile("{{ .reg .u32 z; .reg .u64 zz; mov.u32 z, %%dynamic_smem_size; shr.u32 z, z, 31; cvt.u64.u32 zz, z; '
                  f'mov.u64 %0, {_wa}; add.u64 %0, %0, zz; }}" : "=l"(r)); return r; }} }};\n')
        src_k += (f'extern "C" __global__ void {cl}__launch_bounds__(384, 1) {kn}(const __grid_constant__ P p) '
                  f'{{ vg_body<{TM}, {BN}, {LAY}, {S}, {NS}, {nkb}, {n}, {MC}, {RED}, {MAP}, {PERM}, {MTC}, {NHC}, {PF}, {PFX}, {EPI}, {PRO}, {PRI}, {LASTD}, {WPK}, {k // 64}, WPB_{kn}, {NT2}, {EPF}, {XLD}>(p); }}\n')
        ents[(layer, cfg)] = {"kn": kn, "n": n, "k": k, "cfg": cfg, "smem": _q2_smem_bytes(cfg)}
    src = defines + _q2_HEAD + "".join(_q2__wg_ss(n) for n in sorted(wns)) + _q2_BODY.replace("@WGSS_SPECS@", _q2__specs(wns)) + src_k + _q2_PROBE_SRC
    if os.environ.get("VG_DUMP"):
        open(os.environ["VG_DUMP"], "w").write(src)
        return None, ents
    mod = _q2_Module(src)
    return mod, ents


def _q2_pack_w(w, bn, stage_major=False):
    """[N, K] bf16 -> [N / bn][K / 64][bn rows][8 x 16 B chunks, chunk c of row r stored at c ^ (r & 7)] (the TMA 128B
    swizzle image of every (column tile, k-block) stage, contiguous)."""
    n, k = w.shape
    t = w.reshape(n // bn, bn, k // 64, 8, 8).permute(2, 0, 1, 3, 4) if stage_major else\
        w.reshape(n // bn, bn, k // 64, 8, 8).permute(0, 2, 1, 3, 4)
    r = torch.arange(bn, device=w.device)
    c = torch.arange(8, device=w.device)
    idx = (c[None, :] ^ (r[:, None] & 7))[None, None, :, :, None].expand(*t.shape)
    return t.gather(3, idx).contiguous()


class _q2_VG:
    """Dev wrapper: weights dict + list of (layer, cfg)."""

    def __init__(self, weights, lcfgs, defines="", mmax=128):
        self.mmax = mmax
        specs = []
        packed, wpb = {}, {}
        for layer, cfg in lcfgs:
            w = weights[layer]
            cfg = _q2_norm(cfg)
            specs.append((layer, w.shape[0], w.shape[1], cfg))
            if cfg[15] > 0:
                key = (layer, cfg[1], cfg[15] == 3)
                if key not in packed:
                    packed[key] = _q2_pack_w(w.contiguous(), cfg[1], cfg[15] == 3)
                if cfg[15] >= 2:
                    wpb[(layer, cfg)] = packed[key].data_ptr()
        self.packed = packed
        self.mod, self.ents = _q2_build(specs, defines, wpb, mmax)
        for (layer, cfg), e in self.ents.items():
            if cfg[15] > 0:
                e["wp"] = packed[(layer, cfg[1], cfg[15] == 3)]
        self.w = {}
        for (layer, cfg), e in self.ents.items():
            w = weights[layer].contiguous()
            self.w[layer] = w
            e["tmw"] = _q2__tmap2d(w.data_ptr(), e["n"], e["k"], cfg[1])
            assert e["smem"] <= _q2_SMEM_MAX, (layer, cfg, e["smem"])
            CL = cfg[3] if (cfg[3] > 1 and cfg[6] == 0) else (2 if (cfg[5] or cfg[6] >= 8) else 1)
            if cfg[3] > 1 and cfg[6] in (1, 2, 3, 4, 5, 6, 7):
                WN = cfg[1] // 2 if cfg[2] == 1 else cfg[1]
                ntile = (e["n"] // cfg[1]) * (mmax // cfg[0])
                nh = cfg[3] - 1 if cfg[6] in (5, 6, 7) else 1
                nwgp = 1 if cfg[6] in (6, 7) else 2
                e["ws"] = torch.full((ntile * nh * nwgp * 128 * WN // 2,), -1, dtype=torch.int32, device=w.device)
                e["flags"] = torch.zeros(ntile * nh, dtype=torch.int32, device=w.device)
            if CL > 1 and "VG_NORED" not in defines:
                ncta = (e["n"] // cfg[1]) * (mmax // cfg[0]) * cfg[3]
                e["clusters"] = self.mod.max_active_clusters(e["kn"], e["smem"], 384, CL, ncta)
        for (layer, cfg), e in self.ents.items():
            if cfg[8]:
                e["perm"] = self.gpc_perm(e)
        self.xm = {}
        self.trace = None

    def _probe(self, name, grid, smem, cluster=0):
        out = torch.full((grid,), -1, dtype=torch.int32, device="cuda")
        args = struct.pack("<Q", out.data_ptr())
        if cluster:
            import ctypes
            d = self.mod.driver
            f = self.mod.func(name, smem)
            if cluster > 8:
                d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1)
            cfg = d.CUlaunchConfig()
            cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
            cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = 128, 1, 1
            cfg.sharedMemBytes = smem
            cfg.hStream = d.CUstream(torch.cuda.current_stream().cuda_stream)
            at = d.CUlaunchAttribute()
            at.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = cluster, 1, 1
            cfg.attrs = [at]
            cfg.numAttrs = 1
            raw = ctypes.create_string_buffer(args)
            argv = (ctypes.c_void_p * 1)(ctypes.addressof(raw))
            (err,) = d.cuLaunchKernelEx(cfg, f, ctypes.addressof(argv), 0)
            if int(err) != 0:
                return None
        else:
            self.mod.launch(name, grid, 384 if name != "vg_probex" else 128, smem, args)
        torch.cuda.synchronize()
        return out.tolist()

    def gpc_groups(self):
        """SM -> locality group (GPC: clusters of 16 when the GPU allows them, else clusters of 8)."""
        if getattr(self, "_grp", None) is not None:
            return self._grp
        nsm = torch.cuda.get_device_properties(0).multi_processor_count
        grp = {}
        for cs in (16, 8):
            n = (nsm // cs) * cs
            o = self._probe("vg_probex", n, 120 * 1024, cluster=cs)
            if o is None or min(o) < 0:
                continue
            for i in range(0, n, cs):
                key = min(o[i:i + cs])
                for sm in o[i:i + cs]:
                    grp.setdefault(sm, ("c%d" % cs, key))
        for sm in range(nsm):
            grp.setdefault(sm, ("s", sm))
        self._grp = grp
        return grp

    def gpc_perm(self, e):
        """Physical unit -> logical unit so that the units each GPC actually runs get consecutive logical tiles
        (mt fastest: compact tile blocks per GPC). Helpers / finishers (RED >= 2) are permuted within their class."""
        TM, BN, LAY, S, NS, MC, RED, MAP, PERM, PF, PFX, EPI, PRO, PRI, LASTD, WPK, NT2, EPF, XLD = e["cfg"]
        CL = S if (S > 1 and RED == 0) else (2 if (MC or RED >= 8) else 1)
        ncta = (e["n"] // BN) * (self.mmax // TM) * S
        units = ncta // CL
        if units > 136:
            return None
        o = self._probe("vg_probe2" if CL == 2 else "vg_probe1", ncta, e["smem"])
        if o is None or min(o) < 0 or (CL not in (1, 2)):
            return None
        grp = self.gpc_groups()
        first = {}
        for u in range(units):
            first.setdefault(grp[o[u * CL]], u)
        cls = (lambda u: u // (units // (S if RED in (5, 6, 7) else 2))) if (RED >= 2 and CL == 1) else (lambda u: 0)
        perm = [0] * units
        for c in sorted(set(cls(u) for u in range(units))):
            us = [u for u in range(units) if cls(u) == c]
            order = sorted(us, key=lambda u: (first[grp[o[u * CL]]], u))
            for i, u in enumerate(order):
                perm[u] = us[i]
        return perm

    def run(self, x, layer, cfg, out=None):
        e = self.ents[(layer, _q2_norm(cfg))]
        TM, BN, LAY, S, NS, MC, RED, MAP, PERM, PF, PFX, EPI, PRO, PRI, LASTD, WPK, NT2, EPF, XLD = e["cfg"]
        m, k = x.shape
        key = (x.data_ptr(), m, k, TM)
        tmx = self.xm.get(key)
        if tmx is None:
            tmx = self.xm[key] = _q2__tmap2d(x.data_ptr(), m, k, TM)
        if out is None:
            out = torch.empty((m, e["n"]), dtype=torch.bfloat16, device=x.device)
        tmo = _q2__tmap2d(out.data_ptr(), m, e["n"], 64, 64) if EPI else bytes(128)
        assert 0 < m <= self.mmax, m
        mt = self.mmax // TM                  # compile-time grid (MTC); token tiles past m load zeros, stores are masked
        if MC == 1:
            mt += mt & 1
        ncta = (e["n"] // BN // (2 if NT2 else 1)) * mt * S
        ws = e.get("ws")
        fl = e.get("flags")
        perm = e.get("perm")
        CL = S if (S > 1 and RED == 0) else (2 if (MC or RED >= 8) else 1)
        if perm is not None and len(perm) * CL != (e["n"] // BN) * mt * S:
            perm = None                                    # the permutation was measured for the full grid
        args = tmx + e["tmw"] + tmo + struct.pack("<QQQQQQiiii", out.data_ptr(), self.trace.data_ptr() if self.trace is not None else 0,
                                            ws.data_ptr() if ws is not None else 0, fl.data_ptr() if fl is not None else 0,
                                            e["wp"].data_ptr() if "wp" in e else 0, x.data_ptr(),
                                            m, mt, (e["n"] // BN) * mt, 1 if perm else 0)
        args += struct.pack("<136H", *((perm or []) + [0] * (136 - len(perm or []))))
        self.mod.launch(e["kn"], ncta, 384, e["smem"], args)
        return out


# ============================================================================================ deliverable API (agent Q2)
# 97 <= m <= 128 (verify-16 load); layers without a config here return None (the base keeps them).
# cfg = (TM, BN, LAY, S, NS, MC, RED, MAP, PERM, PF, PFX, EPI, PRO, PRI, LASTD, WPK, NT2), see the qg docstring.
_q2_G128Q_CFG = {
    "o": (64, 128, 1, 1, 9, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 2),
    "qkv_gate": (64, 72, 2, 2, 12, 0, 9, 0, 1, 0, 0, 2, 1, 0, 0, 2),
    "pooler": (64, 64, 2, 1, 14, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 2),
    "unpooler": (64, 128, 1, 1, 9, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 2),
}
_q2_G128Q_M = (97, 128)


def _q2_prepare_g128q(weights, cfgs=None):
    """-> state; state["ok"] False (route disabled, run_g128q returns None) on any failure."""
    import torch.nn.functional as _F
    st = {"ok": False, "cfg": {}}
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            return st
        if torch.cuda.get_device_properties(0).multi_processor_count < 128:
            return st
        lc = [(l, _q2_norm(c)) for l, c in (_q2_G128Q_CFG if cfgs is None else cfgs).items() if l in weights]
        if not lc:
            return st
        V = _q2_VG(weights, lc, "", 128)
        g = torch.Generator(device="cuda").manual_seed(11)
        for layer, cfg in lc:
            W = weights[layer]
            for m in (128, 100, 128):
                x = torch.randn(m, W.shape[1], device="cuda", generator=g).to(torch.bfloat16)
                x[3] = 0
                x[3, 5] = 8.0                          # one-hot probe row: must reproduce 8 * W[:, 5] exactly
                x[m - 1] = 0
                x[m - 1, W.shape[1] - 7] = -16.0
                out = V.run(x, layer, cfg)
                ref = _F.linear(x, W)
                torch.cuda.synchronize()
                err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                if (not err < 5e-3 or not torch.equal(out[3], (8.0 * W[:, 5].float()).to(torch.bfloat16))
                        or not torch.equal(out[m - 1], (-16.0 * W[:, W.shape[1] - 7].float()).to(torch.bfloat16))):
                    raise RuntimeError(f"g128q selftest {layer} m={m}: rel err {err}")
            st["cfg"][layer] = cfg
        st["V"] = V
        st["nk"] = {l: tuple(weights[l].shape) for l in st["cfg"]}
        st["ok"] = True
    except Exception as e:  # noqa: BLE001 - any failure: the base handles every shape
        import sys
        print(f"[g128q] disabled: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        st = {"ok": False, "cfg": {}}
    return st


def _q2_run_g128q(x, config, st):
    """out [m, n] bf16 for the configured layers at 97 <= m <= 128, else None (the base handles it)."""
    if not st.get("ok"):
        return None
    layer, m = config["layer"], config["m"]
    cfg = st["cfg"].get(layer)
    if cfg is None or not (_q2_G128Q_M[0] <= m <= _q2_G128Q_M[1]):
        return None
    n, k = st["nk"][layer]
    if (config["n"] != n or x.dim() != 2 or x.shape[0] != m or x.shape[1] != k or not x.is_contiguous()
            or x.data_ptr() % 16 or x.dtype != torch.bfloat16 or not x.is_cuda):
        return None
    return st["V"].run(x, layer, cfg)


# ======================================================================== module Q3: Q3/g256q.py
"""g256q (agent Q3): generated by mkg256q.py from p3.py + the prepare_g256q / run_g256q API (end of file).
Self-contained (torch + cuda.bindings NVRTC). Derived from agent Q2's qg.py <- agent K's kg.py <- agent V's vg.py.

p3 (agent Q3): qkv_gate (W [2304, 8192] BF16) at 193 <= m <= 256: BF16 SS-wgmma, 128 tok x 72 col tiles, 2-way
split-K as a CLUSTER PAIR with a PHASED k schedule so that the split-K exchange happens in the middle of the kernel.

Derived from agent Q2's qg.py <- agent K's kg.py <- agent V's vg.py (packed 128B-swizzled W stage images by one bulk
copy each, baked W address, fast prologue, direct register epilogue, cluster-pair st.async push, GPC-aware cluster
placement measured at prepare).

Tile (mt, nt) = tokens [128 mt, +128) x W rows [72 nt, +72), k-blocks kb = 0..127 (64 k each). Pair rank r finalises
token half r (tokens 64 r .. 64 r + 63 of the tile); consumer warpgroup g of either CTA owns token half g in phase 1.
  rank 0: phase 1 kb [0, A) both halves (x 128 rows + W), phase 2 kb [A, 128 - A) half 0 only (x 64 rows + W)
  rank 1: phase 1 kb [128 - A, 128) both halves,            phase 2 kb [A, 128 - A) half 1 only
At the end of phase 1 the warpgroup holding the partner's half (g = 1 - r, the "pusher") pushes its 64 x 72 fp32
partial (18 KB) into the partner's smem with st.async (complete_tx on the partner's rbar) and then shares phase 2 with
the finisher warpgroup (alternate stages, k-split); the transfer overlaps phase 2. Epilogue: the pusher's phase-2
partial goes through smem to the finisher, which adds it and the partner's partial and stores BF16 from registers.
A = 64 is the plain symmetric reduce-scatter (no phase 2, exchange on the tail).
Every CTA does the same tensor work (128 x 72 x 64 k-blocks); phase 2 re-reads W k-blocks [A, 128 - A) in both CTAs of
a pair at the same time (L2 hits).

1 producer warpgroup (thread 0 issues every copy; setmaxnreg 40) + 2 consumer warpgroups (setmaxnreg 232).
cfg = (NS, A, PERM, EPS, DBG):
  NS    ring stages (25 KB each: x 128 rows 16 KB + W 72 rows 9 KB)
  A     phase-1 k-blocks per CTA (64 = symmetric)
  PERM  1: GPC-aware cluster permutation measured at prepare, baked into the code
  EPS   epilogue split: 0 finisher stores all 72 columns; 1: both warpgroups store (40 / 32 columns)
  DBG   1: skip the tensor work (isolation)
"""
import ctypes
import os
import struct

import torch

_q3_SMEM_MAX = 227 * 1024
_q3_N_Q, _q3_K_Q = 2304, 8192
_q3_BN = 72


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: 2026 Nathen Fernandes <https://github.com/NathenFernandes/muse-bf16-proj>
# SPDX-License-Identifier: MIT
# Adapted from NathenFernandes/muse-bf16-proj, `_mma` wgmma wrapper generator (via alygg77/okc-muse-spark-kernels).
# Changes: emits a predicate-register (setp) form of the scale-d operand; renamed.
# License text: LICENSES/MIT.txt; notices: NOTICE (repository root).
def _q3__wg_ss(n):
    nr = n // 2
    regs = ", ".join(f"%{i}" for i in range(nr))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nr))
    return (f'__device__ __forceinline__ void wg_ss_n{n}(float* d, u64 da, u64 db) {{\n'
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{nr}, %{nr + 1}, p, 1, 1, 0, 0;\\n}}\\n"\n'
            f'    : {outs} : "l"(da), "l"(db));\n}}\n')
# SPDX-SnippetEnd


_q3_HEAD = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) P { TMap tmx; TMap tmx64; TMap tmx32; u16* out; unsigned long long* trace; const unsigned char* xp; int M; };
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#ifdef VG_TRACE
#define TR(i) do { if (p.trace) { p.trace[blockIdx.x * 32 + (i)] = gtime(); p.trace[8192 + blockIdx.x * 32 + (i)] = clock64(); } } while (0)
#define TRV(i, v) do { if (p.trace) { p.trace[16640 + blockIdx.x * 32 + (i)] = (v); } } while (0)
#else
#define TR(i) do { } while (0)
#define TRV(i, v) do { } while (0)
#endif
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u32 b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect_arrive(u32 b, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(u32 b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void mbar_arrive_n(u32 b, u32 n) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_arrive_cl(u32 remote) {
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(remote) : "memory");
}
__device__ __forceinline__ void mbar_wait(u32 b, int ph) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}\n"
               :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void tma2d(u32 dst, const void* tm, int c0, int c1, u32 bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ uint4 lds128(u32 a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }
__device__ __forceinline__ void sts128(u32 a, uint4 v) { asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
__device__ __forceinline__ u64 mkdesc(u32 saddr) {
  return (u64)((saddr & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)(1024 >> 4) << 32) | ((u64)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void fence_acc(float* a) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(a[i]) :: "memory");
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b)); return r; }
"""

_q3_BODY = r"""
// BF16 stores straight from the wgmma accumulators (agent K's EPI 2): per group of 4 n8-blocks a 4 x 4 transpose of
// bf16x2 words inside each lane quad gives every lane 16 contiguous bytes of one row -> STG.128; n8 blocks [J0, J1)
template <int J0, int J1, int N>
__device__ __forceinline__ void store_cols(const float* acc, u16* out, int tok, int col, int M, int lane, int wq) {
  const int q = lane & 3, r0 = 16 * wq + (lane >> 2);
  constexpr int NG = (J1 - J0) / 4;
#pragma unroll
  for (int m4 = 0; m4 < NG; ++m4) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      u32 w[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) w[i] = pack_bf2(acc[4 * (J0 + 4 * m4 + i) + 2 * h], acc[4 * (J0 + 4 * m4 + i) + 2 * h + 1]);
      {
        const bool hi = q & 2;
        const u32 s0 = hi ? w[0] : w[2], s1 = hi ? w[1] : w[3];
        const u32 x0 = __shfl_xor_sync(0xffffffffu, s0, 2), x1 = __shfl_xor_sync(0xffffffffu, s1, 2);
        if (hi) { w[0] = x0; w[1] = x1; } else { w[2] = x0; w[3] = x1; }
      }
      {
        const bool od = q & 1;
        const u32 s0 = od ? w[0] : w[1], s1 = od ? w[2] : w[3];
        const u32 x0 = __shfl_xor_sync(0xffffffffu, s0, 1), x1 = __shfl_xor_sync(0xffffffffu, s1, 1);
        if (od) { w[0] = x0; w[2] = x1; } else { w[1] = x0; w[3] = x1; }
      }
      const int row = tok + r0 + 8 * h;
      if (row < M)
        asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(out + (size_t)row * N + col + 8 * (J0 + 4 * m4 + q)),
                     "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]) : "memory");
    }
  }
#pragma unroll
  for (int j = J0 + NG * 4; j < J1; ++j) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = tok + r0 + 8 * h;
      if (row < M)
        asm volatile("st.global.u32 [%0], %1;" :: "l"(out + (size_t)row * N + col + 8 * j + 2 * q),
                     "r"(pack_bf2(acc[4 * j + 2 * h], acc[4 * j + 2 * h + 1])) : "memory");
    }
  }
}

template <int NS, int A, int N, int NKBT, int MTC, int PERMC, int EPS, int DBG, int RB, int W2, int PB2, int PF, int PFX, int LPF, int LPX, class WPBT, class PRMT>
__device__ __forceinline__ void ph_body(const P& p) {
  constexpr int WN = 72, NACC = 36, NQ = 9;
  constexpr int XB = 128 * 128, XH = 64 * 128, WB = 72 * 128, SB = XB + WB;
  constexpr int NST = NKBT - A, P2 = NKBT - 2 * A;
  constexpr int PB = 128 * NACC * 4;
  static_assert(P2 >= 0 && P2 % 2 == 0 && A >= 1, "A");
  static_assert(NS >= 2 && NS <= NST, "NS");
  static_assert(SB % 1024 == 0 && XH % 1024 == 0, "stage alignment");
  // RB 1: the partner pushes into my ring slot of stage NST - NS once I have consumed it (ready handshake); that
  // stage must be a phase-1 stage (no circular wait between the two pushers)
  static_assert(RB == 0 || (NST - NS < A && NS >= 3), "RB 1: P2 <= NS - 2");
  constexpr int RS = (NST - NS) % NS;
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  const u32 bars = (smem_u32(sm_raw) + 1023) & ~1023u;
  const u32 full = bars, empty = bars + 8 * NS, rbar = bars + 16 * NS, rdy = bars + 16 * NS + 8;
  const u32 prog = bars + 16 * NS + 16;                                   // LPF: last stage issued by the producer
  const u32 ring = bars + 1024;
  const u32 rbuf = RB ? ring + RS * SB : ring + NS * SB;                 // partner's partial lands here
  const u32 sbuf = ring + NS * SB + (RB ? 0 : PB);                        // PB2: my push staging
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31;
  const int r = (int)(blockIdx.x & 1);
  const int cid = PERMC ? PRMT::get((int)(blockIdx.x >> 1)) : (int)(blockIdx.x >> 1);
  const int mt = cid % MTC, nt = cid / MTC;
  const int tok0 = mt * 128, col0 = nt * WN;
  // stage j -> k-block (rank 1 runs its phase-1 k-blocks [NKBT - A, NKBT) first, then phase 2 [A, NKBT - A))
  auto issue = [&](int slot, int j) {
    const int kb = (r == 0 || j >= A) ? j : NKBT - A + j;
    const u32 fb = full + 8 * slot;
#ifdef VG_HALF
    if (nt >= VG_HALF) { mbar_expect_arrive(fb, 0); return; }   // diagnostic: only CTAs with nt < VG_HALF stream
#endif
#if defined(VG_NOX)
    constexpr u32 EX1 = WB, EX2 = WB;
#elif defined(VG_NOW)
    constexpr u32 EX1 = XB, EX2 = XH;
#else
    constexpr u32 EX1 = XB + WB, EX2 = XH + WB;
#endif
    if (j < A) {
      mbar_expect_arrive(fb, EX1);
#ifndef VG_NOX
      tma2d(ring + slot * SB, &p.tmx, kb * 64, tok0, fb);
#endif
    } else {
      mbar_expect_arrive(fb, EX2);
#ifndef VG_NOX
      tma2d(ring + slot * SB + r * XH, &p.tmx64, kb * 64, tok0 + 64 * r, fb);
#endif
    }
#ifndef VG_NOW
    const u64 wb = WPBT::get();
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(ring + slot * SB + XB), "l"(wb + ((u64)nt * NKBT + (u64)kb) * WB), "r"(WB), "r"(fb) : "memory");
#endif
  };
  // PF: L2 prefetch of the W image of stage j + PF when stage j is issued (steady state only: the startup burst is
  // not enlarged); PFX: also x (one CTA in 8 of the column tiles prefetches the shared x tile)
  auto prefetch = [&](int j) {
    if (j < NST) {
      const int kb = (r == 0 || j >= A) ? j : NKBT - A + j;
      if (PF > 0 && mt == 0) {
        const u64 wb = WPBT::get();
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(wb + ((u64)nt * NKBT + (u64)kb) * WB), "r"(WB) : "memory");
      }
      if (PFX > 0 && (nt & 7) == 0) {
        if (j < A) asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"(&p.tmx), "r"(kb * 64), "r"(tok0) : "memory");
        else asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"(&p.tmx64), "r"(kb * 64), "r"(tok0 + 64 * r) : "memory");
      }
    }
  };
  if (tid == 0) {
    // fast prologue: stages 0-1 first, then the other barriers and stages
#ifdef VG_TRACE
    // (trace build: prologue timestamps in registers, written after the prologue: no cold param read up front)
    unsigned long long tlg[4], tlc[4];
#define TRL(k) do { tlg[k] = gtime(); tlc[k] = clock64(); } while (0)
#else
#define TRL(k) do { } while (0)
#endif
    TRL(0);
    mbar_init(full, 1);
    mbar_init(full + 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    issue(0, 0);
    TRL(1);
    issue(1, 1);
#pragma unroll
    for (int s = 2; s < NS; ++s) mbar_init(full + 8 * s, 1);
#pragma unroll
    for (int s = 0; s < NS; ++s) mbar_init(empty + 8 * s, 8);
    mbar_init(rbar, 1);
    mbar_expect_arrive(rbar, PB);
    mbar_init(rdy, 1);
    asm volatile("st.shared.u32 [%0], %1;" :: "r"(prog), "r"(0) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    TRL(2);
#pragma unroll 1
    for (int s = 2; s < NS; ++s) issue(s, s);
    TRL(3);
#ifdef VG_TRACE
    if (p.trace) {
      unsigned sid; asm volatile("mov.u32 %0, %%smid;" : "=r"(sid)); p.trace[16384 + blockIdx.x] = sid;
      const int ix[4] = {0, 13, 10, 12};
#pragma unroll
      for (int k = 0; k < 4; ++k) { p.trace[blockIdx.x * 32 + ix[k]] = tlg[k]; p.trace[8192 + blockIdx.x * 32 + ix[k]] = tlc[k]; }
    }
#endif
  }
  __syncthreads();
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");   // rbar armed before any push
  if (wg == 0) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    if (tid == 0) {
      int slot = 0, ph = 0;
#pragma unroll 1
      for (int j = NS; j < NST; ++j) {
        mbar_wait(empty + 8 * slot, ph);
        issue(slot, j);
        if (PF > 0) prefetch(j + PF);
        if (LPF > 0) asm volatile("st.volatile.shared.u32 [%0], %1;" :: "r"(prog), "r"(j) : "memory");
        if (++slot == NS) { slot = 0; ph ^= 1; }
      }
      TR(11);
      if constexpr (RB == 1) {
        // my slot RS is free once stage NST - NS is consumed: tell the partner's pusher
        mbar_wait(empty + 8 * RS, ((NST - NS) / NS) & 1);
        u32 rr;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rr) : "r"(rdy), "r"(r ^ 1));
        mbar_arrive_cl(rr);
        TR(17);
      }
    }
    if (LPF > 0 && tid >= 32) {
      // LSU L2 prefetch from the idle producer warps 1-3 (not the TMA unit): when the producer has issued stage j,
      // prefetch the W image (72 lines) of stage j + LPF (and, LPX, the 128 x rows of that k-block in one CTA of
      // every LPX column tiles); starts at the first steady-state issue (the startup burst is not enlarged)
      const int pt = tid - 32;
      const u64 wb = WPBT::get();
      const unsigned char* xp = p.xp;
      int nextp = NS + LPF;
#pragma unroll 1
      while (nextp < NST) {
        int pr;
        asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(pr) : "r"(prog) : "memory");
        if (nextp > pr + LPF) { __nanosleep(64); continue; }
        const int t = nextp;
        const int kb = (r == 0 || t >= A) ? t : NKBT - A + t;
        if (pt < WB / 128)
          asm volatile("prefetch.global.L2 [%0];" :: "l"(wb + ((u64)nt * NKBT + (u64)kb) * WB + (u64)pt * 128) : "memory");
        if (LPX > 0 && (nt % (LPX > 0 ? LPX : 1)) == 0) {
          const int rows = t < A ? 128 : 64, rb = t < A ? 0 : 64 * r;
#pragma unroll
          for (int rr = pt; rr < 128; rr += 96)
            if (rr < rows && tok0 + rb + rr < p.M)
              asm volatile("prefetch.global.L2 [%0];" :: "l"(xp + ((size_t)(tok0 + rb + rr) * NKBT + kb) * 128) : "memory");
        }
        ++nextp;
      }
    }
    __syncwarp();
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
  const int g = __shfl_sync(0xffffffffu, wg, 0) - 1, ct = tid & 127;
  const int wq = __shfl_sync(0xffffffffu, (tid >> 5) & 3, 0);
  const bool fin = g == r;
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = 0.f;
  const u64 db0 = mkdesc(ring + XB), da1 = mkdesc(ring + g * XH), da2 = mkdesc(ring + r * XH);
  if (ct == 0) TR(1 + g);
  // phase 1: every stage, x half g
  int slot = 0, ph = 0, pslot = -1;
  u32 pcnt = 1;
#ifdef VG_TRACE
  long long sw1 = 0, sw2 = 0, tw;
#define WAITF(a, h, s) do { tw = clock64(); mbar_wait(a, h); s += clock64() - tw; } while (0)
#else
#define WAITF(a, h, s) mbar_wait(a, h)
#endif
#pragma unroll 1
  for (int j = 0; j < A; ++j) {
    WAITF(full + 8 * slot, ph, sw1);
#ifdef VG_TRACE
    if (j == 0 && ct == 0 && g == 0) TR(14);
#endif
    const u64 so = (u64)(slot * (SB >> 4));
    wg_fence();
    if (!DBG) {
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) wg_ss_n72(acc, da1 + so + 2 * jj, db0 + so + 2 * jj);
    }
    wg_commit();
    wg_wait<1>(); fence_acc<NACC>(acc);
    if (pslot >= 0) { __syncwarp(); if (lane == 0) mbar_arrive(empty + 8 * pslot); }
    pslot = slot;
    if (++slot == NS) { slot = 0; ph ^= 1; }
  }
  if (!fin) {
    // pusher: my half belongs to the partner: push the 64 x 72 fp32 partial into its rbuf (DSMEM, complete_tx)
    wg_wait<0>(); fence_acc<NACC>(acc);
    __syncwarp();
    if (lane == 0) mbar_arrive(empty + 8 * pslot);
    pslot = -1;
    if (ct == 0) TR(3 + g);
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    if constexpr (PB2 == 0) {
      if constexpr (RB == 1) mbar_wait(rdy, 0);
      u32 rdst, rmb;
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rdst) : "r"(rbuf + ct * 16), "r"(r ^ 1));
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rmb) : "r"(rbar), "r"(r ^ 1));
#pragma unroll
      for (int i = 0; i < NQ; ++i)
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1,%2,%3,%4}, [%5];"
                     :: "r"(rdst + i * 128 * 16), "r"(__float_as_uint(acc[4 * i])), "r"(__float_as_uint(acc[4 * i + 1])),
                        "r"(__float_as_uint(acc[4 * i + 2])), "r"(__float_as_uint(acc[4 * i + 3])), "r"(rmb) : "memory");
    } else {
      // stage the partial in my smem, one bulk copy smem -> partner smem (complete_tx on its rbar): the warpgroup
      // does not wait for the DSMEM transfer
#pragma unroll
      for (int i = 0; i < NQ; ++i)
        sts128(sbuf + (i * 128 + ct) * 16, make_uint4(__float_as_uint(acc[4 * i]), __float_as_uint(acc[4 * i + 1]),
                                                      __float_as_uint(acc[4 * i + 2]), __float_as_uint(acc[4 * i + 3])));
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      asm volatile("bar.sync 3, 128;" ::: "memory");
      if (ct == 0) {
        if constexpr (RB == 1) mbar_wait(rdy, 0);
        u32 rdst, rmb;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rdst) : "r"(rbuf), "r"(r ^ 1));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rmb) : "r"(rbar), "r"(r ^ 1));
        asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     :: "r"(rdst), "r"(sbuf), "r"(PB), "r"(rmb) : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      }
    }
    if (ct == 0) TR(9);
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] = 0.f;
    // skip my stage (the finisher takes stage A, I take A + 1, ...)
    if (++slot == NS) { slot = 0; ph ^= 1; }
  } else {
    if (ct == 0) TR(3 + g);
  }
  if constexpr (P2 > 0) {
    // phase 2: stages j = A .. NST - 1 alternate between the warpgroups, x half r (the finisher's tokens)
#pragma unroll 1
    for (int j = A + (fin ? 0 : 1); j < NST; j += 2) {
      WAITF(full + 8 * slot, ph, sw2);
      const u64 so = (u64)(slot * (SB >> 4));
      wg_fence();
      if (!DBG) {
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) wg_ss_n72(acc, da2 + so + 2 * jj, db0 + so + 2 * jj);
      }
      wg_commit();
      if constexpr (W2 == 0) {
        wg_wait<1>(); fence_acc<NACC>(acc);
        if (pslot >= 0) { __syncwarp(); if (lane == 0) mbar_arrive_n(empty + 8 * pslot, pcnt); }
        pslot = slot;
        pcnt = 2;
      } else {
        // release my stage as soon as its MMAs are done (the other warpgroup keeps the tensor pipe busy)
        wg_wait<0>(); fence_acc<NACC>(acc);
        __syncwarp();
        if (lane == 0) {
          if (pslot >= 0) mbar_arrive_n(empty + 8 * pslot, pcnt);
          mbar_arrive_n(empty + 8 * slot, 2);
        }
        pslot = -1;
      }
      slot += 2; if (slot >= NS) { slot -= NS; ph ^= 1; }
    }
  }
  wg_wait<0>(); fence_acc<NACC>(acc);
  if (ct == 0) { TR(5 + g); TRV(2 * g, sw1); TRV(2 * g + 1, sw2); }
  constexpr u32 TP = (u32)(((NST - 1) % NS) * SB);     // the pusher's last stage slot (P2 > 0): free after its wait
  constexpr u32 TF = (u32)(((NST - 2 + NS) % NS) * SB); // the finisher's last stage slot (P2 > 0)
  if (!fin) {
    if constexpr (P2 > 0) {
      if constexpr (EPS == 0) {
#pragma unroll
        for (int i = 0; i < NQ; ++i)
          sts128(ring + TP + (i * 128 + ct) * 16, make_uint4(__float_as_uint(acc[4 * i]), __float_as_uint(acc[4 * i + 1]),
                                                           __float_as_uint(acc[4 * i + 2]), __float_as_uint(acc[4 * i + 3])));
        asm volatile("bar.arrive 1, 256;" ::: "memory");
      } else {
        // EPS 1: I finalise n8 blocks 5..8 (32 columns): give the finisher my blocks 0..4, take its blocks 5..8
#pragma unroll
        for (int i = 0; i < 5; ++i)
          sts128(ring + TP + (i * 128 + ct) * 16, make_uint4(__float_as_uint(acc[4 * i]), __float_as_uint(acc[4 * i + 1]),
                                                           __float_as_uint(acc[4 * i + 2]), __float_as_uint(acc[4 * i + 3])));
        asm volatile("bar.sync 1, 256;" ::: "memory");
#pragma unroll
        for (int i = 5; i < NQ; ++i) {
          const uint4 v = lds128(ring + TF + (i * 128 + ct) * 16);
          acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
          acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
        }
        mbar_wait(rbar, 0);
#pragma unroll
        for (int i = 5; i < NQ; ++i) {
          const uint4 v = lds128(rbuf + (i * 128 + ct) * 16);
          acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
          acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
        }
        store_cols<5, 9, N>(acc, p.out, tok0 + 64 * r, col0, p.M, lane, wq);
      }
    }
    if (ct == 0) TR(7 + g);
    if (PB2 && ct == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
    return;
  }
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
  if constexpr (P2 > 0 && EPS == 0) {
    asm volatile("bar.sync 1, 256;" ::: "memory");
#pragma unroll
    for (int i = 0; i < NQ; ++i) {
      const uint4 v = lds128(ring + TP + (i * 128 + ct) * 16);
      acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
      acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
    }
  }
  if constexpr (P2 > 0 && EPS == 1) {
    // my blocks 5..8 -> the pusher (it stores them) through my last stage's slot TF; its blocks 0..4 -> me (TP)
#pragma unroll
    for (int i = 5; i < NQ; ++i)
      sts128(ring + TF + (i * 128 + ct) * 16, make_uint4(__float_as_uint(acc[4 * i]), __float_as_uint(acc[4 * i + 1]),
                                                       __float_as_uint(acc[4 * i + 2]), __float_as_uint(acc[4 * i + 3])));
    asm volatile("bar.sync 1, 256;" ::: "memory");
#pragma unroll
    for (int i = 0; i < 5; ++i) {
      const uint4 v = lds128(ring + TP + (i * 128 + ct) * 16);
      acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
      acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
    }
  }
  mbar_wait(rbar, 0);
  if (ct == 0) TR(15);
  constexpr int JE = (P2 > 0 && EPS == 1) ? 5 : NQ;
#pragma unroll
  for (int i = 0; i < JE; ++i) {
    const uint4 v = lds128(rbuf + (i * 128 + ct) * 16);
    acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
    acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
  }
  store_cols<0, JE, N>(acc, p.out, tok0 + 64 * r, col0, p.M, lane, wq);
  if (ct == 0) TR(7 + g);
}

// ============================================================================================ mc_body (x multicast)
// Symmetric 2-way split-K (k halves kh = 0 / 1) as before, plus TMA multicast of x among the MCN CTAs of the cluster
// that compute different column tiles of the same token tile and k half. Cluster = 2 * MCN CTAs, rank q: kh = q & 1,
// c = q >> 1 (column sub-tile). Split-K partner q ^ 1 (DSMEM push, ring-slot rbuf with ready handshake: RB 1).
// Each CTA issues 128 / MCN x rows of every stage with multicast to its group (same smem offset in every member);
// slot release: every consumer warp arrives on its own empty barrier and on each group member's (count 8 MCN), except
// for the last NS stages (no refills; local arrive with count MCN instead: no remote traffic near the exit).
template <int NB>
__device__ __forceinline__ void wgss(float* d, u64 da, u64 db) {
  if constexpr (NB == 9) wg_ss_n72(d, da, db);
  else if constexpr (NB == 7) wg_ss_n56(d, da, db);
  else if constexpr (NB == 6) wg_ss_n48(d, da, db);
  else if constexpr (NB == 5) wg_ss_n40(d, da, db);
  else if constexpr (NB == 4) wg_ss_n32(d, da, db);
  else if constexpr (NB == 3) wg_ss_n24(d, da, db);
  else if constexpr (NB == 2) wg_ss_n16(d, da, db);
  else wg_ss_n8(d, da, db);
}
// E > 0 (MCN 1): the pusher warpgroup runs its last E stages in two column passes (n8 blocks [0, JC), then [JC, 9))
// and pushes each chunk as soon as it is final: only the second (smaller) chunk's DSMEM transfer is on the tail; the
// finisher stores the first chunk's columns while the second chunk is in flight
// LD > 0 (pusher first): the finisher warpgroup waits (named barrier 5) before its last LD stages until the pusher has
// finished all of its stages; the pusher's push then overlaps the finisher's solo stages. Solo stages are issued back
// to back (no wait, no release: the last NS stages are never refilled), agent K's LASTD idea with the roles chosen so
// the DSMEM transfer, not the store, is hidden.
// Q2 > 0 (MCN 1): solo phase 2. Rank r runs Q2 extra stages of 64-token x (its own half, k-blocks [64 - Q2/2,
// 64 + Q2/2)) after its 64 - Q2/2 full stages; only the finisher warpgroup computes them (back to back, no release:
// they are among the last NS stages), while the pusher's DSMEM push (issued right after the full stages) is in flight.
// The tail after the last stage lands is then only that stage's MMAs + add + store.
template <int NS, int MCN, int N, int NKBT, int MTC, int PERMC, int E, int JC, int LD, int Q2, class WPBT, class PRMT>
__device__ __forceinline__ void mc_body(const P& p) {
  constexpr int WN = 72, NACC = 36, NQ = 9, CL = 2 * MCN;
  constexpr int XB = 128 * 128, XH = 64 * 128, WB = 72 * 128, SB = XB + WB, XR = 128 / MCN, XP = XB / MCN;
  constexpr int NA = NKBT / 2 - Q2 / 2;            // full stages (both token halves)
  constexpr int NST = NA + Q2;                     // stages per CTA
  static_assert(Q2 == 0 || (MCN == 1 && E == 0 && LD == 0 && Q2 % 2 == 0 && Q2 < NS), "Q2: solo phase 2");
  constexpr int PB = 128 * NACC * 4;
  constexpr int RS = (NST - NS) % NS;
  static_assert(NS >= 3 && NS <= NST && XP % 1024 == 0, "cfg");
  static_assert(E == 0 || (MCN == 1 && E <= NS - 1 && JC >= 1 && JC <= 8), "E: two-pass pusher tail");
  static_assert(LD == 0 || (E == 0 && LD <= NS - 1 && LD >= 2), "LD: pusher first");
  constexpr u32 C1 = (u32)(E > 0 ? JC : NQ) * 2048, C2 = (u32)(NQ - (E > 0 ? JC : NQ)) * 2048;   // chunk bytes
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  const u32 bars = (smem_u32(sm_raw) + 1023) & ~1023u;
  const u32 full = bars, empty = bars + 8 * NS, rbar = bars + 16 * NS, rdy = bars + 16 * NS + 8, rbar2 = bars + 16 * NS + 24;
  const u32 ring = bars + 1024;
  const u32 rbuf = ring + RS * SB;
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31;
  const int q = (int)(blockIdx.x % CL), r = q & 1, c = q >> 1;
  // RQ (Q2 >= 4): the partner's partial lands in the unused x half (1 - rank) of the slots of my phase-2 stages
  // NA, NA + 1, NA + 2 (8 KB each, 4 + 4 + 1 chunks of 2 KB); free as soon as my producer has issued those stages
  // (their previous occupants are consumed), i.e. long before the push: the ready handshake never waits
  constexpr bool RQ = Q2 >= 4;
  auto rbq = [&](int i, int s) -> u32 {          // chunk i in the smem of the CTA with rank s
    return ring + (u32)(((NA + i / 4) % NS) * SB) + (u32)((1 - s) * XH) + (u32)((i % 4) * 2048);
  };
  const int cid = PERMC ? PRMT::get((int)(blockIdx.x / CL)) : (int)(blockIdx.x / CL);
  const int mt = cid % MTC, nt = (cid / MTC) * MCN + c;
  const int tok0 = mt * 128, col0 = nt * WN;
  const u16 gmask = (u16)((r ? 0xAAAA : 0x5555) & ((1 << CL) - 1));
  // stage j -> k-block: rank 0: j; rank 1: full stages [NKBT - NA, NKBT), then phase 2 [NA, NKBT - NA)
  auto kbof = [&](int j) { return Q2 == 0 ? r * NST + j : ((r == 0 || j >= NA) ? j : NKBT - NA + j); };
  auto issue_w = [&](int slot, int j) {
    const int kb = kbof(j);
    const u32 fb = full + 8 * slot;
    mbar_expect_arrive(fb, (Q2 > 0 && j >= NA) ? XH + WB : XB + WB);
    const u64 wb = WPBT::get();
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(ring + slot * SB + XB), "l"(wb + ((u64)nt * NKBT + (u64)kb) * WB), "r"(WB), "r"(fb) : "memory");
  };
  auto issue_x = [&](int slot, int j) {
    const int kb = kbof(j);
    const u32 fb = full + 8 * slot;
    if constexpr (MCN == 1) {
      if (Q2 > 0 && j >= NA) tma2d(ring + slot * SB + r * XH, &p.tmx64, kb * 64, tok0 + 64 * r, fb);
      else tma2d(ring + slot * SB, &p.tmx, kb * 64, tok0, fb);
    } else {
      asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;"
                   :: "r"(ring + slot * SB + c * XP), "l"(MCN == 2 ? (const void*)&p.tmx64 : (const void*)&p.tmx32),
                      "r"(kb * 64), "r"(tok0 + c * XR), "r"(fb), "h"(gmask) : "memory");
    }
  };
#ifdef VG_TRACE
  unsigned long long tlg[4], tlc[4];
#endif
  if (tid == 0) {
    TRL(0);
#pragma unroll
    for (int s = 0; s < NS; ++s) { mbar_init(full + 8 * s, 1); mbar_init(empty + 8 * s, 8 * MCN); }
    mbar_init(rbar, 1);
    mbar_expect_arrive(rbar, C1);
    mbar_init(rdy, 1);
    if constexpr (E > 0) { mbar_init(rbar2, 1); mbar_expect_arrive(rbar2, C2); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if constexpr (MCN == 1) {
      issue_w(0, 0); issue_x(0, 0);
      TRL(1);
#pragma unroll 1
      for (int s = 1; s < NS; ++s) { issue_w(s, s); issue_x(s, s); }
      TRL(2);
    } else {
      // local W of the whole prologue first (own barriers only), x multicasts after the cluster barrier below
#pragma unroll 1
      for (int s = 0; s < NS; ++s) issue_w(s, s);
      TRL(1);
    }
  }
  __syncthreads();
  if constexpr (MCN == 1) {
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  } else {
    // every member's barriers are initialised before any multicast / remote arrive
    asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    if (tid == 0) {
#pragma unroll 1
      for (int s = 0; s < NS; ++s) issue_x(s, s);
      TRL(2);
    }
  }
  if (wg == 0) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    if (tid == 0) {
#ifdef VG_TRACE
      if (p.trace) {
        unsigned sid; asm volatile("mov.u32 %0, %%smid;" : "=r"(sid)); p.trace[16384 + blockIdx.x] = sid;
        const int ix[3] = {0, 13, 12};
#pragma unroll
        for (int k = 0; k < 3; ++k) { p.trace[blockIdx.x * 32 + ix[k]] = tlg[k]; p.trace[8192 + blockIdx.x * 32 + ix[k]] = tlc[k]; }
      }
#endif
      int slot = 0, ph = 0;
#pragma unroll 1
      for (int j = NS; j < NST; ++j) {
        mbar_wait(empty + 8 * slot, ph);
        issue_w(slot, j);
        issue_x(slot, j);
        if (RQ && j == NA + 2) {
          u32 rr;
          asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rr) : "r"(rdy), "r"(q ^ 1));
          mbar_arrive_cl(rr);
          TR(17);
        }
        if (++slot == NS) { slot = 0; ph ^= 1; }
      }
      TR(11);
      if constexpr (!RQ) {
        // my slot RS is free once stage NST - NS is consumed: tell the split-K partner's pusher
        mbar_wait(empty + 8 * RS, ((NST - NS) / NS) & 1);
        u32 rr;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rr) : "r"(rdy), "r"(q ^ 1));
        mbar_arrive_cl(rr);
        TR(17);
      }
    }
    __syncwarp();
    if constexpr (MCN == 1) asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
  const int g = __shfl_sync(0xffffffffu, wg, 0) - 1, ct = tid & 127;
  const int wq = __shfl_sync(0xffffffffu, (tid >> 5) & 3, 0);
  const bool fin = g == r;
  u32 rempty[MCN > 1 ? MCN - 1 : 1];
  if constexpr (MCN > 1) {
#pragma unroll
    for (int i = 0; i < MCN - 1; ++i) {
      const int peer = 2 * ((c + 1 + i) % MCN) + r;
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rempty[i]) : "r"(empty), "r"(peer));
    }
  }
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = 0.f;
  const u64 db0 = mkdesc(ring + XB), da1 = mkdesc(ring + g * 8192);
  if (ct == 0) TR(1 + g);
  int slot = 0, ph = 0, pslot = -1;
#ifdef VG_TRACE
  long long sw1 = 0, tw;
#endif
  auto release = [&](int s, int j) {
    __syncwarp();
    if (lane == 0) {
      if (MCN == 1 || j < NST - NS) {
        mbar_arrive(empty + 8 * s);
        if constexpr (MCN > 1) {
#pragma unroll
          for (int i = 0; i < MCN - 1; ++i) mbar_arrive_cl(rempty[i] + 8 * s);
        }
      } else {
        mbar_arrive_n(empty + 8 * s, MCN);
      }
    }
  };
  const int nj = (E > 0 && !fin) ? NST - E : NA;
#pragma unroll 1
  for (int j = 0; j < nj; ++j) {
    if (LD > 0 && j == NST - LD && fin) asm volatile("bar.sync 5, 256;" ::: "memory");
#ifdef VG_TRACE
    tw = clock64();
#endif
    mbar_wait(full + 8 * slot, ph);
#ifdef VG_TRACE
    sw1 += clock64() - tw;
    if (j == 0 && ct == 0 && g == 0) TR(14);
#endif
    const u64 so = (u64)(slot * (SB >> 4));
    wg_fence();
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) wg_ss_n72(acc, da1 + so + 2 * jj, db0 + so + 2 * jj);
    wg_commit();
    if (LD == 0 || j <= NST - LD) {
      wg_wait<1>(); fence_acc<NACC>(acc);
      if (pslot >= 0) release(pslot, j - 1);
    }
    pslot = slot;
    if (++slot == NS) { slot = 0; ph ^= 1; }
  }
  wg_wait<0>(); fence_acc<NACC>(acc);
  if (LD == 0) release(pslot, nj - 1);
  if (LD > 0 && !fin) asm volatile("bar.arrive 5, 256;" ::: "memory");
  if constexpr (Q2 > 0) {
    // solo phase 2 (finisher only; the pusher runs zero iterations of the same code). Q2C 2: two independent
    // accumulator chains (k16 steps 0, 2 -> acc, 1, 3 -> acc2): a lone warpgroup with one chain is latency-bound
    const int jend = fin ? NST : NA;
#ifndef Q2C
#define Q2C 2
#endif
    float acc2[NACC];
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc2[i] = 0.f;
#pragma unroll 1
    for (int j = NA; j < jend; ++j) {
      mbar_wait(full + 8 * slot, ph);
      const u64 so = (u64)(slot * (SB >> 4));
      wg_fence();
      if (Q2C == 2) {
#pragma unroll
        for (int jj = 0; jj < 4; jj += 2) {
          wg_ss_n72(acc, da1 + so + 2 * jj, db0 + so + 2 * jj);
          wg_ss_n72(acc2, da1 + so + 2 * jj + 2, db0 + so + 2 * jj + 2);
        }
      } else {
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) wg_ss_n72(acc, da1 + so + 2 * jj, db0 + so + 2 * jj);
      }
      wg_commit();
      wg_wait<2>(); fence_acc<NACC>(acc); fence_acc<NACC>(acc2);
      if (++slot == NS) { slot = 0; ph ^= 1; }
    }
    wg_wait<0>(); fence_acc<NACC>(acc); fence_acc<NACC>(acc2);
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] += acc2[i];
  }
  if (ct == 0) { TR(3 + g); TR(5 + g); TRV(2 * g, sw1); }
  constexpr int J1 = E > 0 ? JC : NQ;
  if (!fin) {
    // pusher: my token half belongs to the partner: 64 x 72 fp32 partial into its ring slot RS (DSMEM, complete_tx)
    if constexpr (E > 0) {
      // pass 1: columns [0, 8 JC) of the last E stages (no releases: these slots are never refilled). Separate
      // accumulators per wgmma shape (one register array in two shapes serialises every wgmma, ptxas C7516)
      float accA[4 * JC];
#pragma unroll
      for (int i = 0; i < 4 * JC; ++i) accA[i] = 0.f;
      int s1 = slot, p1 = ph;
#pragma unroll 1
      for (int j = NST - E; j < NST; ++j) {
        mbar_wait(full + 8 * s1, p1);
        const u64 so = (u64)(s1 * (SB >> 4));
        wg_fence();
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) wgss<JC>(accA, da1 + so + 2 * jj, db0 + so + 2 * jj);
        wg_commit();
        wg_wait<1>(); fence_acc<4 * JC>(accA);
        if (++s1 == NS) { s1 = 0; p1 ^= 1; }
      }
      wg_wait<0>(); fence_acc<4 * JC>(accA);
#pragma unroll
      for (int i = 0; i < 4 * JC; ++i) acc[i] += accA[i];
    }
    if constexpr (MCN == 1) asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    mbar_wait(rdy, 0);
    u32 rdst, rmb;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rdst) : "r"(rbuf + ct * 16), "r"(q ^ 1));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rmb) : "r"(rbar), "r"(q ^ 1));
    if constexpr (RQ) {
      u32 rd[3];
#pragma unroll
      for (int k = 0; k < 3; ++k) asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rd[k]) : "r"(rbq(4 * k, q ^ 1) + ct * 16), "r"(q ^ 1));
#pragma unroll
      for (int i = 0; i < NQ; ++i)
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1,%2,%3,%4}, [%5];"
                     :: "r"(rd[i / 4] + (i % 4) * 2048), "r"(__float_as_uint(acc[4 * i])), "r"(__float_as_uint(acc[4 * i + 1])),
                        "r"(__float_as_uint(acc[4 * i + 2])), "r"(__float_as_uint(acc[4 * i + 3])), "r"(rmb) : "memory");
    } else {
#pragma unroll
    for (int i = 0; i < J1; ++i)
      asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1,%2,%3,%4}, [%5];"
                   :: "r"(rdst + i * 128 * 16), "r"(__float_as_uint(acc[4 * i])), "r"(__float_as_uint(acc[4 * i + 1])),
                      "r"(__float_as_uint(acc[4 * i + 2])), "r"(__float_as_uint(acc[4 * i + 3])), "r"(rmb) : "memory");
    }
    if (ct == 0) TR(9);
    if constexpr (E > 0) {
      // pass 2: columns [8 JC, 72) of the last E stages, then the second chunk
      float accB[4 * (NQ - JC)];
#pragma unroll
      for (int i = 0; i < 4 * (NQ - JC); ++i) accB[i] = 0.f;
      int s1 = slot, p1 = ph;
#pragma unroll 1
      for (int j = NST - E; j < NST; ++j) {
        const u64 so = (u64)(s1 * (SB >> 4));
        wg_fence();
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) wgss<NQ - JC>(accB, da1 + so + 2 * jj, db0 + so + 2 * jj + 64 * JC);
        wg_commit();
        wg_wait<1>(); fence_acc<4 * (NQ - JC)>(accB);
        if (++s1 == NS) { s1 = 0; p1 ^= 1; }
      }
      wg_wait<0>(); fence_acc<4 * (NQ - JC)>(accB);
#pragma unroll
      for (int i = 0; i < 4 * (NQ - JC); ++i) acc[4 * JC + i] += accB[i];
      u32 rmb2;
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rmb2) : "r"(rbar2), "r"(q ^ 1));
#pragma unroll
      for (int i = J1; i < NQ; ++i)
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1,%2,%3,%4}, [%5];"
                     :: "r"(rdst + i * 128 * 16), "r"(__float_as_uint(acc[4 * i])), "r"(__float_as_uint(acc[4 * i + 1])),
                        "r"(__float_as_uint(acc[4 * i + 2])), "r"(__float_as_uint(acc[4 * i + 3])), "r"(rmb2) : "memory");
    }
    if (ct == 0) TR(7 + g);
    return;
  }
  if constexpr (MCN == 1) asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
  mbar_wait(rbar, 0);
  if (ct == 0) TR(15);
#pragma unroll
  for (int i = 0; i < J1; ++i) {
    const uint4 v = lds128(RQ ? rbq(i, r) + ct * 16 : rbuf + (i * 128 + ct) * 16);
    acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
    acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
  }
  store_cols<0, J1, N>(acc, p.out, tok0 + 64 * r, col0, p.M, lane, wq);
  if constexpr (E > 0) {
    mbar_wait(rbar2, 0);
#pragma unroll
    for (int i = J1; i < NQ; ++i) {
      const uint4 v = lds128(rbuf + (i * 128 + ct) * 16);
      acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
      acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
    }
    store_cols<J1, NQ, N>(acc, p.out, tok0 + 64 * r, col0, p.M, lane, wq);
  }
  if (ct == 0) TR(7 + g);
}
"""

_q3_PROBE_SRC = r"""
__device__ __forceinline__ void vg_probe_body(int* out) {
  unsigned sid; asm volatile("mov.u32 %0, %%smid;" : "=r"(sid));
  if (threadIdx.x == 0) out[blockIdx.x] = (int)sid;
  const unsigned long long t0 = gtime();
  while (gtime() - t0 < 30000ull) { }
}
extern "C" __global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(384, 1) vg_probe2(int* out) { vg_probe_body(out); }
extern "C" __global__ void __cluster_dims__(4, 1, 1) __launch_bounds__(384, 1) vg_probe4(int* out) { vg_probe_body(out); }
extern "C" __global__ void __cluster_dims__(8, 1, 1) __launch_bounds__(384, 1) vg_probe8(int* out) { vg_probe_body(out); }
extern "C" __global__ void __launch_bounds__(128, 1) vg_probex(int* out) { vg_probe_body(out); }
"""


class _q3_Module:
    def __init__(self, src, name="p3.cu", opts=()):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        o = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        o += [x.encode() for x in opts]
        if os.environ.get("CU_LINEINFO"):
            o.append(b"-lineinfo")
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(o), o)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        self.log = log.decode(errors="replace").strip("\x00 \n")
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + self.log)
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        self.cubin = cubin
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err
        self.funcs = {}

    def func(self, name, smem):
        key = (name, smem)
        f = self.funcs.get(key)
        if f is None:
            d = self.driver
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            assert int(err) == 0, (name, err)
            if smem > 48 * 1024:
                (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
                assert int(err) == 0, (name, smem, err)
            self.funcs[key] = f
        return f

    def attrs(self, name, smem=0):
        d = self.driver
        f = self.func(name, smem)
        out = {}
        for k in ("NUM_REGS", "LOCAL_SIZE_BYTES"):
            err, v = d.cuFuncGetAttribute(getattr(d.CUfunction_attribute, "CU_FUNC_ATTRIBUTE_" + k), f)
            out[k] = v
        return out

    def launch(self, name, grid, block, smem, params):
        d = self.driver
        f = self.func(name, smem)
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        assert int(err) == 0, (name, err)

    def launch_cluster(self, name, grid, block, smem, args, cluster):
        d = self.driver
        f = self.func(name, smem)
        if cluster > 8:
            d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1)
        cfg = d.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block, 1, 1
        cfg.sharedMemBytes = smem
        cfg.hStream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        at = d.CUlaunchAttribute()
        at.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = cluster, 1, 1
        cfg.attrs = [at]
        cfg.numAttrs = 1
        raw = ctypes.create_string_buffer(args)
        argv = (ctypes.c_void_p * 1)(ctypes.addressof(raw))
        (err,) = d.cuLaunchKernelEx(cfg, f, ctypes.addressof(argv), 0)
        return int(err) == 0


def _q3__tmap2d(ptr, rows, cols, box_rows, box_cols=64):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(2), ptr,
        [d.cuuint64_t(cols), d.cuuint64_t(rows)], [d.cuuint64_t(cols * 2)], [d.cuuint32_t(box_cols), d.cuuint32_t(box_rows)],
        [d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


def _q3_pack_w(w, bn):
    """[N, K] bf16 -> [N / bn][K / 64][bn rows][8 x 16 B chunks, chunk c of row r stored at c ^ (r & 7)] (the TMA 128B
    swizzle image of every (column tile, k-block) stage, contiguous) (agent K's pack_w)."""
    n, k = w.shape
    t = w.reshape(n // bn, bn, k // 64, 8, 8).permute(0, 2, 1, 3, 4)
    r = torch.arange(bn, device=w.device)
    c = torch.arange(8, device=w.device)
    idx = (c[None, :] ^ (r[:, None] & 7))[None, None, :, :, None].expand(*t.shape)
    return t.gather(3, idx).contiguous()


_q3_NCFG = 17


def _q3_norm(cfg):
    """(NS, A, PERM, EPS, DBG, RB, W2, PB2, PF, PFX, MCN, LPF, LPX, E, JC, LD, Q2); MCN > 0: mc_body (x multicast among MCN column tiles,
    symmetric split-K, ring-slot rbuf; only NS / PERM / MCN are used)"""
    cfg = tuple(cfg)
    return cfg + (0,) * (_q3_NCFG - len(cfg))


def _q3_kname(cfg):
    return "ph_qkv_" + "_".join(map(str, _q3_norm(cfg)))


def _q3_clsize(cfg):
    return 2 * _q3_norm(cfg)[10] if _q3_norm(cfg)[10] else 2


def _q3_smem_bytes(cfg):
    NS, A, PERM, EPS, DBG, RB, W2, PB2, PF, PFX, MCN, LPF, LPX, E, JC, LD, Q2 = _q3_norm(cfg)
    if MCN:
        return 2048 + NS * (128 * 128 + _q3_BN * 128)
    return 2048 + NS * (128 * 128 + _q3_BN * 128) + 128 * 36 * 4 * ((RB == 0) + (PB2 == 1))


def _q3__u64_const(name, val):
    # 64-bit literal + a runtime zero ptxas cannot fold (agent K: a folded 64-bit immediate lost its high half)
    return (f'struct {name} {{ static __device__ __forceinline__ u64 get() {{ u64 r; '
            f'asm volatile("{{ .reg .u32 z; .reg .u64 zz; mov.u32 z, %%dynamic_smem_size; shr.u32 z, z, 31; cvt.u64.u32 zz, z; '
            f'mov.u64 %0, {val}; add.u64 %0, %0, zz; }}" : "=l"(r)); return r; }} }};\n')


def _q3__perm_struct(name, perm):
    """perm (list of < 256 ints) baked as 64-bit immediates: 8 entries per word, selected by a compare chain."""
    if perm is None:
        return f"struct {name} {{ static __device__ __forceinline__ int get(int c) {{ return c; }} }};\n"
    words = []
    for i in range(0, len(perm), 8):
        v = 0
        for j, e in enumerate(perm[i:i + 8]):
            v |= (e & 0xFF) << (8 * j)
        words.append(v)
    s = f"struct {name} {{ static __device__ __forceinline__ int get(int c) {{ const int q = c >> 3; u64 w = {words[0]}ull;\n"
    for i in range(1, len(words)):
        s += f"  w = q == {i} ? {words[i]}ull : w;\n"
    s += "  return (int)((w >> (8 * (c & 7))) & 0xFF); } };\n"
    return s


class _q3_PH:
    """Dev wrapper: weights dict + list of cfgs (qkv_gate only)."""

    def __init__(self, weights, cfgs, defines="", mmax=256):
        assert mmax == 256
        w = weights["qkv_gate"].contiguous()
        assert tuple(w.shape) == (_q3_N_Q, _q3_K_Q)
        self.w = w
        self.wp = _q3_pack_w(w, _q3_BN)
        self.cfgs = [_q3_norm(c) for c in cfgs]
        self.mt = 2
        self.ncta = (_q3_N_Q // _q3_BN) * self.mt * 2
        # placement probe (same smem / threads / cluster dims as the kernels)
        self.pmod = _q3_Module(_q3_HEAD + _q3_PROBE_SRC, "p3probe.cu")
        self.perm = {}
        for c in self.cfgs:
            if c[2]:
                key = (_q3_smem_bytes(c), _q3_clsize(c))
                if key not in self.perm:
                    self.perm[key] = self.gpc_perm(*key)
        wns = set(range(8, 80, 8))
        src = defines + _q3_HEAD + "".join(_q3__wg_ss(n) for n in sorted(wns)) + _q3_BODY
        src += _q3__u64_const("WPB", self.wp.data_ptr())
        self.ents = {}
        for c in self.cfgs:
            NS, A, PERM, EPS, DBG, RB, W2, PB2, PF, PFX, MCN, LPF, LPX, E, JC, LD, Q2 = c
            kn = _q3_kname(c)
            prm = self.perm.get((_q3_smem_bytes(c), _q3_clsize(c))) if PERM else None
            src += _q3__perm_struct(f"PRM_{kn}", prm)
            if MCN:
                src += (f'extern "C" __global__ void __cluster_dims__({2 * MCN}, 1, 1) __launch_bounds__(384, 1) {kn}(const __grid_constant__ P p) '
                        f'{{ mc_body<{NS}, {MCN}, {_q3_N_Q}, {_q3_K_Q // 64}, {self.mt}, {1 if prm else 0}, {E}, {JC or 5}, {LD}, {Q2}, WPB, PRM_{kn}>(p); }}\n')
            else:
                src += (f'extern "C" __global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(384, 1) {kn}(const __grid_constant__ P p) '
                        f'{{ ph_body<{NS}, {A}, {_q3_N_Q}, {_q3_K_Q // 64}, {self.mt}, {1 if prm else 0}, {EPS}, {DBG}, {RB}, {W2}, {PB2}, {PF}, {PFX}, {LPF}, {LPX}, WPB, PRM_{kn}>(p); }}\n')
            self.ents[c] = {"kn": kn, "smem": _q3_smem_bytes(c), "perm": prm}
            assert _q3_smem_bytes(c) <= _q3_SMEM_MAX, (c, _q3_smem_bytes(c))
        self.src = src
        if os.environ.get("VG_DUMP"):
            open(os.environ["VG_DUMP"], "w").write(src)
        self.mod = _q3_Module(src)
        self.xm = {}
        self.trace = None

    def _probe(self, name, grid, smem, cluster):
        out = torch.full((grid,), -1, dtype=torch.int32, device="cuda")
        args = struct.pack("<Q", out.data_ptr())
        ok = self.pmod.launch_cluster(name, grid, 128 if name == "vg_probex" else 384, smem, args, cluster)
        torch.cuda.synchronize()
        return out.tolist() if ok else None

    def gpc_groups(self):
        """SM -> locality group (GPC: clusters of 16 when the GPU allows them, else clusters of 8)."""
        if getattr(self, "_grp", None) is not None:
            return self._grp
        nsm = torch.cuda.get_device_properties(0).multi_processor_count
        grp = {}
        for cs in (16, 8):
            n = (nsm // cs) * cs
            o = self._probe("vg_probex", n, 120 * 1024, cs)
            if o is None or min(o) < 0:
                continue
            for i in range(0, n, cs):
                key = min(o[i:i + cs])
                for sm in o[i:i + cs]:
                    grp.setdefault(sm, ("c%d" % cs, key))
        for sm in range(nsm):
            grp.setdefault(sm, ("s", sm))
        self._grp = grp
        return grp

    def gpc_perm(self, smem, cl=2):
        """Physical cluster -> logical tile unit so that the clusters each GPC actually runs get consecutive units
        (mt fastest: 2 token tiles x a few column tiles per GPC) (agent Q2 / V)."""
        units = self.ncta // cl
        o = self._probe(f"vg_probe{cl}", self.ncta, smem, cl)
        if o is None or min(o) < 0:
            return None
        self.probe_sm = getattr(self, "probe_sm", {})
        self.probe_sm[(smem, cl)] = o
        grp = self.gpc_groups()
        first = {}
        for u in range(units):
            first.setdefault(grp[o[u * cl]], u)
        order = sorted(range(units), key=lambda u: (first[grp[o[u * cl]]], u))
        perm = [0] * units
        for i, u in enumerate(order):
            perm[u] = i
        return perm

    def run(self, x, cfg, out=None):
        c = _q3_norm(cfg)
        e = self.ents[c]
        m, k = x.shape
        key = (x.data_ptr(), m, k)
        tm = self.xm.get(key)
        if tm is None:
            tm = self.xm[key] = _q3__tmap2d(x.data_ptr(), m, k, 128) + _q3__tmap2d(x.data_ptr(), m, k, 64) + _q3__tmap2d(x.data_ptr(), m, k, 32)
        if out is None:
            out = torch.empty((m, _q3_N_Q), dtype=torch.bfloat16, device=x.device)
        assert 0 < m <= 256, m
        args = tm + struct.pack("<QQQi", out.data_ptr(), self.trace.data_ptr() if self.trace is not None else 0, x.data_ptr(), m)
        self.mod.launch(e["kn"], self.ncta, 384, e["smem"], args)
        return out


# ============================================================================================ deliverable API (agent Q3)
# qkv_gate (W [2304, 8192]) at G256Q_M[0] <= m <= 256 (verify-32 load); everything else returns None (base handles it).
# cfg = (NS, A, PERM, EPS, DBG, RB, W2, PB2, PF, PFX, MCN, LPF, LPX, E, JC, LD, Q2): mc_body, 9 stages, GPC-aware
# cluster placement, solo phase 2 of Q2 stages (see the docstrings of p3 / mc_body).
_q3_G256Q_CFG = (9, 64, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 6)
_q3_G256Q_M = (240, 256)
_q3__G256Q_XMAX = 64          # cached x tensor maps (per x pointer); cleared beyond this


def _q3_prepare_g256q(weights, cfg=None):
    """-> state; state["ok"] False (route disabled, run_g256q returns None) on any failure."""
    import torch.nn.functional as _F
    st = {"ok": False}
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            return st
        if torch.cuda.get_device_properties(0).multi_processor_count < 128:
            return st
        W = weights.get("qkv_gate") if hasattr(weights, "get") else None
        if W is None or tuple(W.shape) != (_q3_N_Q, _q3_K_Q) or W.dtype != torch.bfloat16 or not W.is_cuda:
            return st
        c = _q3_norm(_q3_G256Q_CFG if cfg is None else cfg)
        V = _q3_PH(weights, [c])
        g = torch.Generator(device="cuda").manual_seed(13)
        for m in (256, 255, _q3_G256Q_M[0], 256):
            x = torch.randn(m, _q3_K_Q, device="cuda", generator=g).to(torch.bfloat16)
            x[3] = 0
            x[3, 5] = 8.0                          # one-hot probe rows: must reproduce s * W[:, k] exactly
            x[m - 1] = 0
            x[m - 1, _q3_K_Q - 7] = -16.0
            x[130] = 0
            x[130, 4100] = 2.0 ** -3              # second token tile, second k half
            out = V.run(x, c, out=torch.full((m, _q3_N_Q), float("nan"), dtype=torch.bfloat16, device="cuda"))
            ref = _F.linear(x, W)
            torch.cuda.synchronize()
            err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
            ok = (err < 5e-3 and torch.equal(out[3], (8.0 * W[:, 5].float()).to(torch.bfloat16))
                  and torch.equal(out[m - 1], (-16.0 * W[:, _q3_K_Q - 7].float()).to(torch.bfloat16))
                  and torch.equal(out[130], (2.0 ** -3 * W[:, 4100].float()).to(torch.bfloat16)))
            if not ok:
                raise RuntimeError(f"g256q selftest m={m}: rel err {err}")
        V.xm.clear()
        st = {"ok": True, "V": V, "cfg": c}
    except Exception as e:  # noqa: BLE001 - any failure: the base handles every shape
        import sys
        print(f"[g256q] disabled: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        st = {"ok": False}
    return st


def _q3_run_g256q(x, config, st):
    """out [m, 2304] bf16 for qkv_gate at G256Q_M[0] <= m <= 256, else None (the base handles it)."""
    if not st.get("ok"):
        return None
    m = config["m"]
    if config["layer"] != "qkv_gate" or not (_q3_G256Q_M[0] <= m <= _q3_G256Q_M[1]) or config["n"] != _q3_N_Q:
        return None
    if (x.dim() != 2 or x.shape[0] != m or x.shape[1] != _q3_K_Q or not x.is_contiguous() or x.dtype != torch.bfloat16
            or not x.is_cuda or x.data_ptr() % 16):
        return None
    V = st["V"]
    if len(V.xm) > _q3__G256Q_XMAX:
        V.xm.clear()
    return V.run(x, st["cfg"])


# ======================================================================== module P5: P5/pu5.py
"""pu5 (agent P5): pooler W[4096, 8192] / unpooler W[8192, 4096] at m = 16 / 32 (generated from pk.py by mkpu5.py).

    state = prepare_pu5(weights)            # never raises; < 5 s
    out   = run_pu5(x, config, state)       # [M, N] bf16, or None where the base (bo12) keeps the shape

Owned shapes: see OWN below (pooler / unpooler only, exact m; every other shape returns None).
Kernel derived from vjeux/okc-2026 `sk2` (via alygg77's public OKC submission, Apache-2.0), agent X's NVRTC port sk3,
agent F's sk11 (agent G's lossless F12 fake-exponent format, register-preloaded escape lists) and agent H's skh. P5:
ring sized to a per-SM in-flight sweet spot (~100-120 KB incl. x), x of stage s issued with W of stage s + 1 (XLAG,
agent V2), W / escape-list pointers and M / N / scale baked as NVRTC constants.
"""

import ctypes
import os
import struct

import torch

_p5_SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) P { TMap xmap; const u8* W; const u16* x; u16* out; const u8* etab; u64* trc;
                         float scale; int M; int N; int K; int pad0; int pad1; };

template <u64 A> __device__ __forceinline__ const u8* cptr() { const u8* p; asm("mov.u64 %0, %1;" : "=l"(p) : "n"(A)); return p; }
__device__ __forceinline__ u64 gtime() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ u64 ctime() { u64 t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory"); return t; }
__device__ __forceinline__ void ldsm_x4(u32 addr, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 sel) {
  u32 d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel)); return d;
}
__device__ __forceinline__ u32 rotr(u32 x, int s) { return __funnelshift_r(x, x, s); }
// agent G's F12 decode: 8 weights from 2 lo words + 1 nibble word
__device__ __forceinline__ void decode12(u32 L0, u32 L1, u32 nw, u32* r) {
  u32 hA, hB;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(hA) : "r"(nw), "r"(0x87878787u), "n"(0x38383838));
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(hB) : "r"(rotr(nw, 4)), "r"(0x87878787u), "n"(0x38383838));
  r[0] = prmt(L0, hA, 0x5140); r[1] = prmt(L0, hA, 0x7362);
  r[2] = prmt(L1, hB, 0x5140); r[3] = prmt(L1, hB, 0x7362);
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a)); return r; }
__device__ __forceinline__ void mbar_init(u32 bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `mbar_test` / `mbar_wait` / `mbar_wait_sleep` (MBAR_HELPERS) (via alygg77/okc-muse-spark-kernels).
// Changes: renamed types (u32); mbar_wait written on one line; vjeux's esc_delta, which sits between mbar_wait and mbar_wait_sleep upstream, left out.
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
__device__ __forceinline__ bool mbar_test(u32 bar, u32 parity) {
  u32 ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(u32 bar, u32 parity) { while (!mbar_test(bar, parity)) {} }
__device__ __forceinline__ void mbar_wait_sleep(u32 bar, u32 parity) {
  u32 ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(0x989680u) : "memory");
  } while (!ok);
}
// SPDX-SnippetEnd
__device__ __forceinline__ void mbar_arrive(u32 bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(u32 bar, u32 bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\n}\n" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_load_3d(u32 dst, const TMap* map, u32 bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void sts32(u32 a, u32 v) { asm volatile("st.shared.u32 [%0], %1;" :: "r"(a), "r"(v) : "memory"); }
__device__ __forceinline__ u32 lds32(u32 a) { u32 v; asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ uint2 lds64(u32 a) {
  uint2 v; asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a) : "memory"); return v;
}
__device__ __forceinline__ u32 lds16(u32 a) { u16 v; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a)); return (u32)v; }
__device__ __forceinline__ void stg64(u64* p, u64 v) { asm volatile("st.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory"); }

// (P5) pipelined consumer helpers: raw unit operands (W lo/nibble words + x ldmatrix fragments) and unit compute
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_kernel` consumer loop (shared-memory loads, ldmatrix of x, decode and mma steps) (via alygg77/okc-muse-spark-kernels).
// Changes: split out of sk2's kernel body into a load helper (the 768-byte tile-unit ld.shared loads and the swizzled ldmatrix of the x fragments) and a compute helper; sk2's decode8 replaced by our decode12, the accumulator set picked by a template unit index, and sk2's uncompressed and hi/lo (router) paths dropped.
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
template <int NT, int MT>
__device__ __forceinline__ void pk_ld_unit(u32 wb, u32 xt, const int* trow, int chi, int sw0, int lane,
                                           uint4 (&L)[NT], uint2 (&N)[NT], u32 (&a)[4][MT][4]) {
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(L[nt].x), "=r"(L[nt].y), "=r"(L[nt].z), "=r"(L[nt].w) : "r"(wb + nt * 768u + lane * 16));
    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(N[nt].x), "=r"(N[nt].y) : "r"(wb + nt * 768u + 512 + lane * 8));
  }
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
      ldsm_x4(xt + trow[mt] * 128 + (((2 * j + chi) ^ ((sw0 + trow[mt]) & 7)) << 4), a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
}
template <int NT, int MT, int NACC, int UU>
__device__ __forceinline__ void pk_mma_unit(float (&acc)[NACC][MT][NT][4], const uint4 (&L)[NT], const uint2 (&N)[NT],
                                            const u32 (&a)[4][MT][4]) {
  u32 b[NT][8];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    decode12(L[nt].x, L[nt].y, N[nt].x, &b[nt][0]);
    decode12(L[nt].z, L[nt].w, N[nt].y, &b[nt][4]);
  }
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
        mma_bf16(acc[(UU * 4 + j) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], b[nt][2 * j], b[nt][2 * j + 1]);
}
// SPDX-SnippetEnd

// trace layout (u64 words): [0, 4096) per CTA 16 words; LAND + cta * 64 + st: stage landing (observer);
// REL + (cta * 64 + st) * 8 + warp: consumer slot release; ISS + cta * 64 + st: producer W issue
#define TR_LAND 4096
#define TR_ISS (4096 + 16384)
#define TR_REL (4096 + 32768)

// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_kernel` (via alygg77/okc-muse-spark-kernels).
// Changes: kept sk2's structure (swapped operands: the x tile is the A operand, loaded by ldmatrix from a TMA-loaded 128B-swizzled tile; each CTA owns R rows x the whole K; a producer warp refills a ring of stages); replaced its 12-bit weight decode (8-exponent window) with our lossless fake-exponent formats and its escape-warp corrections with our per-CTA escape lists and exact corrections, and changed the ring sizing, the producer's issue order and the constants (baked in by NVRTC).
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, int NACC, int NST, u64 WB, u64 EB, int LC,
          int XLAG, int MM, int NN, u32 SCB, int FL>
__device__ __forceinline__ void pk_body(const P& p) {
  constexpr int NW = NWR * WSK;
  constexpr bool ESC1 = (FL & 131072) != 0;
  constexpr int NE = (FL & 128) ? 0 : ESC1 ? 1 : NWR * NT;
  constexpr bool STR = (FL & 4096) != 0;
  constexpr bool TRC = (FL & 2) != 0 || STR;
  constexpr bool PIPE = (FL & 33554432) != 0;    // (P5) software-pipelined consumer (1 << 25)
  constexpr bool EARLYW = (FL & 67108864) != 0;  // (P5) W(0) issued after full0's init only (1 << 26)
  constexpr bool OUT8 = (FL & 134217728) != 0;   // (P5) 8 rows per output iteration, 16-B stores (1 << 27)
  constexpr int NC = NW * 32;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int T = UPW * NT;
  static_assert(ESC1 || NE == 0 || NWR * NT <= 4, "at most 4 escape tiles");
  constexpr int WBY = NW * T * 768;
  constexpr int ELB = ESC1 ? 16 + LC * 8 : 16 + NWR * NT * LC * 4;
  constexpr int LSK = UPS == 1 ? 6 : UPS == 2 ? 7 : UPS == 4 ? 8 : UPS == 8 ? 9 : 10;
  constexpr int XT = TOKR * 128;
  constexpr int XBY = UPS * XT;
  constexpr int XB = (FL & 1073741824) ? 0 : XBY;
  constexpr int WBB = (FL & 536870912) ? 0 : WBY;
  constexpr int SST = (XBY + WBY + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);
  static_assert(XLAG >= 0 && XLAG < NSTAGE, "XLAG");
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const u32 sraw = (u32)__cvta_generic_to_shared(smem_raw);
  const u32 sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const u32 bars = sbase + NSTAGE * SST;
  const u32 full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8;
  const u32 etab_s = pbar + 16;
  float* corr = reinterpret_cast<float*>(gbase + (etab_s - sbase) + ELB);
  float* red = reinterpret_cast<float*>(gbase);
  const u64 tstart = TRC ? gtime() : 0ull;
  const u64 cstart = TRC ? ctime() : 0ull;
  const u8* Wp = WB ? cptr<WB>() : p.W;
  const u8* Ep = EB ? cptr<EB>() : p.etab;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = (int)blockIdx.x, ncta = (int)gridDim.x;
  const u8* wsrc = Wp + (size_t)cta * WBY;
  const size_t wstride = (size_t)ncta * WBY;
  u64* trc = p.trc;

  if (tid == 0) {
    u64 ta = 0, tb = 0;
    if constexpr (EARLYW) {
      mbar_init(full0, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      if constexpr (TRC) ta = gtime();
      mbar_expect_tx(full0, XB + WBB);
      if constexpr (WBB > 0) bulk_g2s(sbase + XBY, wsrc, WBY, full0);
      if constexpr (TRC) tb = gtime();
      asm volatile("prefetch.tensormap [%0];" :: "l"(&p.xmap) : "memory");
      for (int i = 1; i < NSTAGE; ++i) mbar_init(full0 + 8 * i, 1);
      for (int i = 0; i < NSTAGE; ++i) mbar_init(empty0 + 8 * i, NW + NE);
      mbar_init(pbar, 1);
      mbar_init(cbar, NE > 0 ? NE : 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    } else {
      asm volatile("prefetch.tensormap [%0];" :: "l"(&p.xmap) : "memory");
      for (int i = 0; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + NE); }
      mbar_init(pbar, 1);
      mbar_init(cbar, NE > 0 ? NE : 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      if constexpr (TRC) ta = gtime();
      mbar_expect_tx(full0, XB + WBB);
      if constexpr (WBB > 0) bulk_g2s(sbase + XBY, wsrc, WBY, full0);
      if constexpr (TRC) tb = gtime();
    }
    if constexpr (XLAG == 0 && XB > 0) tma_load_3d(sbase, &p.xmap, full0, 0, 0, 0);
    if constexpr (NE > 0) {
      mbar_expect_tx(pbar, ELB);
      bulk_g2s(etab_s, Ep + (size_t)cta * ELB, ELB, pbar);
    }
    if constexpr (TRC) { const u64 tc = gtime(); stg64(trc + cta * 16 + 8, ta); stg64(trc + cta * 16 + 9, tb); stg64(trc + cta * 16 + 10, tc); }
    if constexpr (STR) stg64(trc + TR_ISS + cta * 64, tb);
  }
  __syncthreads();

  if (warp == NW) {   // ---------------- producer
    if (lane == 0) {
      for (int st = 1; st < NST + XLAG; ++st) {
        if (st < NST) {
          const int slot = st % NSTAGE;
          if (st >= NSTAGE) {
            if constexpr ((FL & 32) != 0) mbar_wait(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
            else mbar_wait_sleep(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
          }
          if constexpr (STR) stg64(trc + TR_ISS + cta * 64 + st, gtime());
          const u32 s0 = sbase + slot * SST;
          mbar_expect_tx(full0 + 8 * slot, XB + WBB);
          if constexpr (WBB > 0) bulk_g2s(s0 + XBY, wsrc + (size_t)st * wstride, WBY, full0 + 8 * slot);
          if constexpr (XLAG == 0 && XB > 0) tma_load_3d(s0, &p.xmap, full0 + 8 * slot, 0, 0, st * UPS);
        }
        if constexpr (XLAG > 0 && XB > 0) {
          const int sx = st - XLAG;
          if (sx >= 0 && sx < NST) tma_load_3d(sbase + (sx % NSTAGE) * SST, &p.xmap, full0 + 8 * (sx % NSTAGE), 0, 0, sx * UPS);
        }
      }
    }
    __syncwarp();
    return;
  }

  if (STR && warp == NW + 1 + NE) {   // ---------------- observer (trace only): landing time of every stage
    if (lane == 0) {
      for (int st = 0; st < NST; ++st) {
        mbar_wait_sleep(full0 + 8 * (st % NSTAGE), (st / NSTAGE) & 1);
        stg64(trc + TR_LAND + cta * 64 + st, gtime());
      }
    }
    __syncwarp();
    return;
  }

  if (ESC1 && NE > 0 && warp == NW + 1) {   // ---------------- (H) single escape warp, stage-sorted list
    for (int i = lane; i < 2 * TOKS * RP; i += 32) corr[i] = 0.f;
    mbar_wait(pbar, 0);
    const u32 corr_s = etab_s + ELB;
    const int cnt = (int)lds32(etab_s);
    int ptr = 0;
    uint2 nx = cnt > 0 ? lds64(etab_s + 16) : make_uint2(0xFFFFFFFFu, 0u);
    const float nd = -__uint_as_float(0x38000000u);   // -d = -2^-15 (fake domain): every F12 escape decodes to d
    const int tok = lane;
    for (int st = 0; st < NST; ++st) {
      const int slot = st % NSTAGE;
      mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
      const u32 s0 = sbase + slot * SST;
      while ((nx.x >> 16) == (u32)st) {
        const u32 kl = nx.x & (u32)(UPS * 64 - 1), kc = kl & 63u, u = kl >> 6;
        const int row = (int)((nx.x >> 9) & 127u);
        const float val = __uint_as_float(nx.y);
        float x = 0.f;
        if (tok < TOKR)
          x = __uint_as_float(lds16(s0 + u * XT + (u32)tok * 128 + ((((kc >> 3) ^ (u32)(tok & 7))) << 4) + (kc & 7u) * 2) << 16);
        if (tok < TOKS) {
          const u32 c1 = corr_s + (u32)(tok * RP + row) * 4u, c2 = corr_s + (u32)((TOKS + tok) * RP + row) * 4u;
          const float v1 = __uint_as_float(lds32(c1)), v2 = __uint_as_float(lds32(c2));
          sts32(c1, __float_as_uint(fmaf(nd, x, v1)));
          sts32(c2, __float_as_uint(fmaf(val, x, v2)));
        }
        ++ptr;
        nx = ptr < cnt ? lds64(etab_s + 16 + (u32)ptr * 8u) : make_uint2(0xFFFFFFFFu, 0u);
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(cbar);
    return;
  }

  if (!ESC1 && NE > 0 && warp > NW) {   // ---------------- (F) per-tile escape warps (rare escapes, lane = token)
    const int et = warp - NW - 1;
    const u32* el = reinterpret_cast<const u32*>(gbase + (etab_s - sbase));
    mbar_wait(pbar, 0);
    const int cnt = (int)el[et];
    const u32 ea = lane < cnt ? el[4 + et * LC + lane] : 0u;
    const u32 eb2 = (LC > 32 && lane + 32 < cnt) ? el[4 + et * LC + 32 + lane] : 0u;
    const int sa = lane < cnt ? (int)((ea & 0x1FFFu) >> LSK) : -1;
    const int sb = (LC > 32 && lane + 32 < cnt) ? (int)((eb2 & 0x1FFFu) >> LSK) : -1;
    float E1r[8], E2r[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) { E1r[r] = 0.f; E2r[r] = 0.f; }
    const float nd = -__uint_as_float(0x38000000u);
    const int tok = lane;
    for (int st = 0; st < NST; ++st) {
      const int slot = st % NSTAGE;
      mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
      u32 m0 = __ballot_sync(0xffffffffu, sa == st);
      u32 m1 = LC > 32 ? __ballot_sync(0xffffffffu, sb == st) : 0u;
      if (m0 | m1) {
        const u32 s0 = sbase + slot * SST;
        while (m0 | m1) {
          u32 e;
          if (m0) { const int i = __ffs(m0) - 1; m0 &= m0 - 1; e = __shfl_sync(0xffffffffu, ea, i); }
          else { const int i = __ffs(m1) - 1; m1 &= m1 - 1; e = __shfl_sync(0xffffffffu, eb2, i); }
          const u32 kl = e & (u32)(UPS * 64 - 1), kc = kl & 63u, u = kl >> 6;
          const int row = (int)((e >> 13) & 7u);
          const float val = __uint_as_float(e & 0xFFFF0000u);
          float x = 0.f;
          if (tok < TOKR)
            x = __uint_as_float(lds16(s0 + u * XT + (u32)tok * 128 + ((((kc >> 3) ^ (u32)(tok & 7))) << 4) + (kc & 7u) * 2) << 16);
#pragma unroll
          for (int r = 0; r < 8; ++r)
            if (row == r) { E1r[r] = fmaf(nd, x, E1r[r]); E2r[r] = fmaf(val, x, E2r[r]); }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
    if (tok < TOKS) {
#pragma unroll
      for (int r = 0; r < 8; r += 4) {
        *reinterpret_cast<float4*>(corr + tok * RP + 8 * et + r) = make_float4(E1r[r], E1r[r + 1], E1r[r + 2], E1r[r + 3]);
        *reinterpret_cast<float4*>(corr + (TOKS + tok) * RP + 8 * et + r) = make_float4(E2r[r], E2r[r + 1], E2r[r + 2], E2r[r + 3]);
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(cbar);
    return;
  }

  // ---------------- consumers
  const int wl = warp;
  const int wr = wl / WSK, wk = wl % WSK;
  const int g = lane >> 2, t4 = lane & 3;
  const int q8 = lane >> 3;
  const int chi = q8 >> 1;
  int trow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) trow[mt] = TOKR >= 16 ? mt * 16 + (lane & 7) + 8 * (q8 & 1) : (lane & 7);
  float acc[NACC][MT][NT][4];
#pragma unroll
  for (int c = 0; c < NACC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;
  u64 tfirst = 0;
  if constexpr (PIPE && (FL & 16) == 0) {
    // (P5) the loads of unit q + 1 (next unit of this stage, or unit 0 of the next stage after waiting for it) are
    // issued before the decode + HMMA of unit q; operand buffers A / B alternate by unit parity (no register copies)
    constexpr int SU = (UPW % 2 == 0) ? 1 : 2;
    static_assert(NST % SU == 0, "PIPE with UPW odd needs an even stage count");
    uint4 LA[NT], LB[NT];
    uint2 NA[NT], NB[NT];
    u32 aA[4][MT][4], aB[4][MT][4];
    const u32 wofs = XBY + (u32)wl * (T * 768);
    mbar_wait(full0, 0);
    if (TRC) tfirst = gtime();
    pk_ld_unit<NT, MT>(sbase + wofs, sbase + (wk * UPW) * XT, trow, chi, (wk * UPW) * TOKR, lane, LA, NA, aA);
    for (int st0 = 0; st0 < NST; st0 += SU) {
#pragma unroll
      for (int k = 0; k < SU * UPW; ++k) {
        const int st = st0 + k / UPW;
        const int uu = k % UPW;
        const int slot = st % NSTAGE;
        const u32 s0 = sbase + slot * SST;
#define PK_PIPE_STEP(LC_, NC_, AC_, LN_, NN_, AN_) do { \
          if (uu + 1 < UPW) { \
            pk_ld_unit<NT, MT>(s0 + wofs + (u32)((uu + 1) * NT) * 768u, s0 + (wk * UPW + uu + 1) * XT, trow, chi, (wk * UPW + uu + 1) * TOKR, lane, LN_, NN_, AN_); \
          } else { \
            __syncwarp(); \
            if (lane == 0) mbar_arrive(empty0 + 8 * slot); \
            if (STR && lane == 0) stg64(trc + TR_REL + (cta * 64 + st) * 8 + warp, gtime()); \
            if (st + 1 < NST) { \
              const int s1 = (st + 1) % NSTAGE; \
              mbar_wait(full0 + 8 * s1, ((st + 1) / NSTAGE) & 1); \
              const u32 b1 = sbase + s1 * SST; \
              pk_ld_unit<NT, MT>(b1 + wofs, b1 + (wk * UPW) * XT, trow, chi, (wk * UPW) * TOKR, lane, LN_, NN_, AN_); \
            } \
          } \
          if (uu == 0) pk_mma_unit<NT, MT, NACC, 0>(acc, LC_, NC_, AC_); \
          else if (uu == 1) pk_mma_unit<NT, MT, NACC, 1>(acc, LC_, NC_, AC_); \
          else if (uu == 2) pk_mma_unit<NT, MT, NACC, 2>(acc, LC_, NC_, AC_); \
          else pk_mma_unit<NT, MT, NACC, 3>(acc, LC_, NC_, AC_); \
        } while (0)
        if (k & 1) PK_PIPE_STEP(LB, NB, aB, LA, NA, aA);
        else PK_PIPE_STEP(LA, NA, aA, LB, NB, aB);
#undef PK_PIPE_STEP
      }
    }
  } else
  for (int st = 0; st < NST; ++st) {
    const int slot = st % NSTAGE;
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    if (TRC && st == 0) tfirst = gtime();
    const u32 s0 = sbase + slot * SST;
    if constexpr ((FL & 16) != 0) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      if (STR && lane == 0) stg64(trc + TR_REL + (cta * 64 + st) * 8 + warp, gtime());
      continue;
    }
    const u32 wb = s0 + XBY + (u32)wl * (T * 768);
    u32 b[T][8];
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const int ui = wk * UPW + uu;
      const u32 xt = s0 + ui * XT;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        uint4 L;
        uint2 N;
        const u32 tb = wb + (u32)(uu * NT + nt) * 768u;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(L.x), "=r"(L.y), "=r"(L.z), "=r"(L.w) : "r"(tb + lane * 16));
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(N.x), "=r"(N.y) : "r"(tb + 512 + lane * 8));
        decode12(L.x, L.y, N.x, &b[uu * NT + nt][0]);
        decode12(L.z, L.w, N.y, &b[uu * NT + nt][4]);
      }
      u32 a[4][MT][4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          ldsm_x4(xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * TOKR + trow[mt]) & 7)) << 4),
                  a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      if (UPW == 1 && uu == 0) {
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt)
            mma_bf16(acc[(uu * 4 + j) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3],
                     b[uu * NT + nt][2 * j], b[uu * NT + nt][2 * j + 1]);
    }
    if (UPW > 1) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
    if (STR && lane == 0) stg64(trc + TR_REL + (cta * 64 + st) * 8 + warp, gtime());
  }
  if (TRC && warp == 0 && lane == 0) {
    stg64(trc + cta * 16 + 0, tstart);
    stg64(trc + cta * 16 + 1, tfirst);
    stg64(trc + cta * 16 + 2, gtime());
    u32 smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    stg64(trc + cta * 16 + 6, smid);
  }
  if constexpr (NE > 0) mbar_wait(cbar, 0);
  if (TRC && warp == 0 && lane == 0) stg64(trc + cta * 16 + 3, gtime());
  named_bar(1, NC);
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int row = (wr * NT + nt) * 8 + 2 * t4;
      const int tok = mt * 16 + g;
      float* r0 = red + (wk * TOKS + tok) * RP + row;
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = acc[0][mt][nt][e];
#pragma unroll
        for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
      }
      *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
    }
  named_bar(1, NC);
  if (TRC && tid == 0) stg64(trc + cta * 16 + 11, gtime());
  constexpr int R4 = R / 4;
  const float sc = SCB ? __uint_as_float(SCB) : p.scale;
  const int M = MM ? MM : p.M;
  const int N = NN ? NN : p.N;
  if constexpr (OUT8) {
    constexpr int R8 = R / 8;
    for (int idx = tid; idx < R8 * M; idx += NC) {
      const int tt = idx / R8, row = (idx - tt * R8) * 8;
      float4 v0 = *reinterpret_cast<const float4*>(red + tt * RP + row);
      float4 v1 = *reinterpret_cast<const float4*>(red + tt * RP + row + 4);
#pragma unroll
      for (int w = 1; w < WSK; ++w) {
        const float4 p0 = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
        const float4 p1 = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row + 4);
        v0.x += p0.x; v0.y += p0.y; v0.z += p0.z; v0.w += p0.w;
        v1.x += p1.x; v1.y += p1.y; v1.z += p1.z; v1.w += p1.w;
      }
      if constexpr (NE > 0) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float4 e0 = *reinterpret_cast<const float4*>(corr + (h * TOKS + tt) * RP + row);
          const float4 e1 = *reinterpret_cast<const float4*>(corr + (h * TOKS + tt) * RP + row + 4);
          v0.x += e0.x; v0.y += e0.y; v0.z += e0.z; v0.w += e0.w;
          v1.x += e1.x; v1.y += e1.y; v1.z += e1.z; v1.w += e1.w;
        }
      }
      u16* o = p.out + (size_t)tt * N + cta * R + row;
      uint4 pk;
      pk.x = pack_bf2(v0.x * sc, v0.y * sc);
      pk.y = pack_bf2(v0.z * sc, v0.w * sc);
      pk.z = pack_bf2(v1.x * sc, v1.y * sc);
      pk.w = pack_bf2(v1.z * sc, v1.w * sc);
      *reinterpret_cast<uint4*>(o) = pk;
    }
  } else
  for (int idx = tid; idx < R4 * M; idx += NC) {
    const int tt = idx / R4, row = (idx - tt * R4) * 4;
    float4 v = *reinterpret_cast<const float4*>(red + tt * RP + row);
#pragma unroll
    for (int w = 1; w < WSK; ++w) {
      const float4 pp = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
      v.x += pp.x; v.y += pp.y; v.z += pp.z; v.w += pp.w;
    }
    if constexpr (NE > 0) {
      const float4 e1 = *reinterpret_cast<const float4*>(corr + tt * RP + row);
      v.x += e1.x; v.y += e1.y; v.z += e1.z; v.w += e1.w;
      const float4 e2 = *reinterpret_cast<const float4*>(corr + (TOKS + tt) * RP + row);
      v.x += e2.x; v.y += e2.y; v.z += e2.z; v.w += e2.w;
    }
    u16* o = p.out + (size_t)tt * N + cta * R + row;
    uint2 pk;
    pk.x = pack_bf2(v.x * sc, v.y * sc);
    pk.y = pack_bf2(v.z * sc, v.w * sc);
    *reinterpret_cast<uint2*>(o) = pk;
  }
  if (TRC && tid == 0) { stg64(trc + cta * 16 + 4, gtime()); stg64(trc + cta * 16 + 7, ctime() - cstart); }
}
// SPDX-SnippetEnd
"""

# ------------------------------------------------------------------------------------------------- configuration
# (R, NT, NWR, WSK, UPW, NACC, cap KB, fl, XLAG, BAKE)
_p5_F12 = 1 << 20               # (kept for parity with sk11 cfg tuples; P5 is F12-only, the bit is ignored)
_p5_FLAG_ESC1 = 1 << 17
_p5_FLAG_STR = 4096
_p5_FLAG_TRC = 2
_p5_CFG = {('pooler', 16): (32, 2, 2, 2, 2, 1, 120, 0, 1, 7), ('pooler', 32): (32, 2, 2, 2, 2, 1, 176, 0, 1, 7), ('unpooler', 16): (32, 2, 2, 2, 1, 1, 50, 0, 1, 7)}
_p5_OWN = set(_p5_CFG)
_p5_K_OF = {"pooler": 8192, "unpooler": 4096}
_p5_N_OF = {"pooler": 4096, "unpooler": 8192}
_p5_SMEM_LIMIT = 232_448
_p5_FAKE12 = 112
_p5_TRC_WORDS = 4096 + 32768 + 16384 * 8


def _p5__xlag(cfg):
    return cfg[8] if len(cfg) > 8 else 0


def _p5__bake(cfg):
    return cfg[9] if len(cfg) > 9 else 0


def _p5__geom(cfg, layer, m, lc):
    R, NT, NWR, WSK, UPW, NACC, cap, fl = cfg[:8]
    tokr = 16 if m <= 16 else 32
    mt = 2 if tokr == 32 else 1
    ups = WSK * UPW
    nw = NWR * WSK
    t = UPW * NT
    wby = nw * t * 768
    xby = ups * tokr * 128
    nst = _p5_K_OF[layer] // (ups * 64)
    toks = 16 * mt
    rp = R + ((40 - (R % 32)) % 32)
    esc1 = bool(fl & _p5_FLAG_ESC1)
    elb = 16 + lc * 8 if esc1 else 16 + (R // 8) * lc * 4
    ne = 0 if fl & 128 else 1 if esc1 else R // 8
    nthr = (nw + 1 + ne + (1 if fl & _p5_FLAG_STR else 0)) * 32
    return dict(tokr=tokr, mt=mt, ups=ups, nw=nw, t=t, wby=wby, xby=xby, nst=nst, toks=toks, rp=rp, ne=ne, elb=elb,
                lc=lc, esc1=esc1, nthr=nthr)


def _p5__nstage(cfg, g):
    sst = (g["xby"] + g["wby"] + 1023) // 1024 * 1024
    extra = 1024 + 16 * 16 + 16 + g["elb"] + 2 * g["toks"] * g["rp"] * 4 + 64
    ring = min(cfg[6] * 1024, _p5_SMEM_LIMIT - extra)
    return max(2, min(16, g["nst"], ring // sst)), sst


def _p5__smem(g, nstage, sst):
    return 1024 + nstage * sst + 16 * nstage + 16 + g["elb"] + 2 * g["toks"] * g["rp"] * 4


# ------------------------------------------------------------------------------------------------- weight packing
def _p5__window(e, width):
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    return max(range(1, 256 - width), key=lambda q: int(cs[q + width] - cs[q]))


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Copied from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_tileunits` (via alygg77/okc-muse-spark-kernels).
# Changes: renamed; docstring shortened.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def _p5__tileunits(w):
    """W [N, K] -> [N/8, K/64, 32 lanes, 16] in mma.m16n8k16 B-fragment order (sk2_tileunits)."""
    n, k = w.shape
    t = w.view(n // 8, 8, k // 64, 4, 2, 4, 2)
    return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)
# SPDX-SnippetEnd


def _p5_compress12(w):
    """F12 planes in tile-unit order [N/8, K/64, 32, 16] (agent G's fake-exponent 12-bit: 16-exponent window)."""
    tu = _p5__tileunits(w.contiguous())
    u = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    elo = _p5__window(e, 16)
    c = e - elo
    esc = (c < 0) | (c > 15)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m),
            "nib": torch.where(esc, torch.zeros_like(u), (s << 3) | (c >> 1)), "esc": esc, "u": u, "elo": elo}


def _p5_pack_nib8(nib):
    """agent G's F12 nibble word: nib [..., 8] (s << 3 | h) -> int32."""
    nib = nib.to(torch.int64)
    s, h = nib >> 3, nib & 7
    word = torch.zeros(nib.shape[:-1], dtype=torch.int64, device=nib.device)
    for b in range(4):
        word |= (h[..., b] | (s[..., b] << 7) | (h[..., 4 + b] << 4) | (s[..., 4 + (b + 3) % 4] << 3)) << (8 * b)
    return torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_stream` (via alygg77/okc-muse-spark-kernels).
# Changes: gathers our separate low-byte and 4-bit-code planes per tile unit (codes repacked into nibble words) and concatenates them, instead of indexing sk2's pre-packed 768-byte tile units; the stage / CTA / warp / unit index arithmetic is sk2's.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def _p5__wstream12(f, n_rows, R, NT, NWR, WSK, UPW):
    """[stage][cta][warp][T tile-units: 512 lo bytes (lane x 16) | 256 B nibble words (lane x 8)]"""
    dev = f["lo"].device
    U = f["lo"].shape[1]
    ups, nw, T = WSK * UPW, NWR * WSK, UPW * NT
    S, C = U // ups, n_rows // R
    ar = lambda v: torch.arange(v, device=dev)
    s = ar(S).view(-1, 1, 1, 1, 1)
    c = ar(C).view(1, -1, 1, 1, 1)
    w = ar(nw).view(1, 1, -1, 1, 1)
    uu = ar(UPW).view(1, 1, 1, -1, 1)
    nt = ar(NT).view(1, 1, 1, 1, -1)
    tile = (c * (R // 8) + (w // WSK) * NT + nt).expand(S, C, nw, UPW, NT)
    unit = (s * ups + (w % WSK) * UPW + uu).expand(S, C, nw, UPW, NT)
    lo = f["lo"][tile, unit].reshape(S, C, nw, T, 512).to(torch.uint8)
    nw8 = _p5_pack_nib8(f["nib"][tile, unit].reshape(S, C, nw, T, 32, 2, 8))
    nb = nw8.contiguous().view(torch.uint8).reshape(S, C, nw, T, 256)
    return torch.cat([lo, nb], -1).contiguous().view(-1)
# SPDX-SnippetEnd


def _p5__esc_common(f):
    idx = f["esc"].nonzero()
    tile, unit, lane, i = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    row = tile * 8 + (lane >> 2)
    k = unit * 64 + 16 * (i >> 2) + 8 * ((i >> 1) & 1) + 2 * (lane & 3) + (i & 1)
    t = (f["u"][idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]].to(torch.int32) << 16).view(torch.float32)
    fv = t * (2.0 ** (_p5_FAKE12 - f["elo"]))
    fb16 = fv.to(torch.bfloat16)
    if not bool((torch.isfinite(fv) & (fb16.float() == fv) & (fb16.float() * (2.0 ** (f["elo"] - _p5_FAKE12)) == t)).all()):
        return None
    return row, k, fb16.view(torch.int16).to(torch.int64) & 0xFFFF


def _p5__elist12(f, n_rows, R, UPS):
    """(F) Per CTA: [4 x u32 per-tile counts][NE tiles x LC u32 entries k | row << 13 | fake_bits << 16]."""
    r = _p5__esc_common(f)
    if r is None:
        return None
    row, k, fb = r
    dev = row.device
    C, NE = n_rows // R, R // 8
    cta, et, rit = row // R, (row % R) // 8, row % 8
    key = (cta * NE + et).long()
    cnt = torch.bincount(key, minlength=C * NE)
    cmax = int(cnt.max()) if cnt.numel() else 0
    if cmax > 64 or (k.numel() and int(k.max()) >= 8192) or NE > 4:
        return None
    lc = 32 if cmax <= 32 else 64
    o = torch.argsort(key * 8192 + k)
    key, ent = key[o], (k | (rit << 13)).long()[o] | (fb[o] << 16)
    st0 = torch.cumsum(cnt, 0) - cnt
    j = torch.arange(key.numel(), device=dev) - st0[key]
    buf = torch.zeros(C, 4 + NE * lc, dtype=torch.int64, device=dev)
    buf[:, :NE] = cnt.view(C, NE)
    buf.view(-1)[(key // NE) * (4 + NE * lc) + 4 + (key % NE) * lc + j] = ent
    buf = torch.where(buf >= 2 ** 31, buf - 2 ** 32, buf).to(torch.int32)
    return {"buf": buf.contiguous(), "lc": lc, "n": int(cnt.sum()), "cmax": cmax}


def _p5__elist1(f, n_rows, R, UPS):
    """(H) ESC1 list per CTA: [count, 0, 0, 0] then uint2 entries sorted by stage:
    x = kl | row << 9 | stage << 16 (kl = k within the stage, row within the CTA), y = fake bf16 bits << 16."""
    r = _p5__esc_common(f)
    if r is None:
        return None
    row, k, fb = r
    dev = row.device
    C = n_rows // R
    s_, kl = (k // (UPS * 64)).long(), (k % (UPS * 64)).long()
    cta, rl = (row // R).long(), (row % R).long()
    if R > 128 or (s_.numel() and int(s_.max()) >= 0xFFFF):
        return None
    cnt = torch.bincount(cta, minlength=C)
    cmax = int(cnt.max()) if cnt.numel() else 0
    if cmax > 2048:
        return None
    cap = max(2, (cmax + 1) // 2 * 2)
    o = torch.argsort(((cta * 65536 + s_) * 128 + rl) * 512 + kl)
    cta, ex, ey = cta[o], (kl | (rl << 9) | (s_ << 16))[o], (fb << 16)[o]
    st0 = torch.cumsum(cnt, 0) - cnt
    j = torch.arange(cta.numel(), device=dev) - st0[cta]
    buf = torch.zeros(C, 2 + cap, 2, dtype=torch.int64, device=dev)
    buf[:, 0, 0] = cnt
    buf[cta, 2 + j, 0] = ex
    buf[cta, 2 + j, 1] = ey
    buf = torch.where(buf >= 2 ** 31, buf - 2 ** 32, buf).to(torch.int32)
    return {"buf": buf.contiguous(), "lc": cap, "n": int(cnt.sum()), "cmax": cmax}


# ------------------------------------------------------------------------------------------------- NVRTC module
class _p5__Module:
    def __init__(self, src, name="pu5.cu", opts=()):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        o = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        o += [x.encode() for x in opts]
        if os.environ.get("CU_LINEINFO"):
            o.append(b"-lineinfo")
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(o), o)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        self.log = log.decode(errors="replace").strip("\x00 \n")
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + self.log)
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        self.cubin = cubin
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err
        self.funcs = {}

    def func(self, name, smem):
        key = (name, smem)
        f = self.funcs.get(key)
        if f is None:
            d = self.driver
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            assert int(err) == 0, (name, err)
            if smem > 48 * 1024:
                (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
                assert int(err) == 0, (name, smem, err)
            self.funcs[key] = f
        return f

    def attrs(self, name, smem=0):
        d = self.driver
        f = self.func(name, smem)
        out = {}
        for k in ("NUM_REGS", "LOCAL_SIZE_BYTES"):
            err, v = d.cuFuncGetAttribute(getattr(d.CUfunction_attribute, "CU_FUNC_ATTRIBUTE_" + k), f)
            out[k] = v
        return out

    def launch(self, f, grid, block, smem, params):
        d = self.driver
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        assert int(err) == 0, err


def _p5__tmap_x3d(ptr, m, k, box_rows, units):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(3), ptr,
        [d.cuuint64_t(64), d.cuuint64_t(m), d.cuuint64_t(k // 64)], [d.cuuint64_t(k * 2), d.cuuint64_t(128)],
        [d.cuuint32_t(64), d.cuuint32_t(box_rows), d.cuuint32_t(units)],
        [d.cuuint32_t(1), d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


_p5__TM = {}


def _p5__xmap(x, m, k, box_rows, units):
    key = (x.data_ptr(), m, k, box_rows, units)
    b = _p5__TM.get(key)
    if b is None:
        if len(_p5__TM) > 512:
            _p5__TM.clear()
        b = _p5__TM[key] = _p5__tmap_x3d(x.data_ptr(), m, k, box_rows, units)
    return b


# ------------------------------------------------------------------------------------------------- prepare / run
def _p5_prepare_pu5(weights, cfgs=None, verbose=False, variants=None, selftest=True):
    """cfgs: {(layer, m): cfg} (default CFG restricted to OWN); variants: extra {(layer, m, tag): cfg}. Never raises."""
    try:
        st = _p5__prepare_impl(weights, cfgs, verbose, variants)
    except Exception as ex:   # noqa: BLE001
        print(f"[pu5] prepare failed, base keeps every shape: {type(ex).__name__}: {ex}", flush=True)
        return {"ent": {}}
    if selftest:
        for (layer, m, tag), e in st["ent"].items():
            if not e["ok"]:
                continue
            try:
                gen = torch.Generator(device="cuda").manual_seed(1234 + m)
                x = torch.randn((m, e["k"]), generator=gen, device="cuda").to(torch.bfloat16)
                out = _p5_run_pu5(x, {"m": m, "n": e["n"], "k": e["k"], "layer": layer}, st, tag=tag or "__self")
                ref = x.float() @ weights[layer][: e["n"]].float().T
                err = float((out.float() - ref).norm() / ref.norm())
                e["ok"] = err < 5e-3
                if not e["ok"]:
                    print(f"[pu5] self-test failed for {(layer, m, tag)}: rel err {err:.3e}", flush=True)
            except Exception as ex:   # noqa: BLE001
                e["ok"] = False
                print(f"[pu5] self-test error for {(layer, m, tag)}: {ex}", flush=True)
        torch.cuda.synchronize()
    return st


def _p5__prepare_impl(weights, cfgs=None, verbose=False, variants=None):
    cfgs = dict({k: v for k, v in _p5_CFG.items() if k in _p5_OWN} if cfgs is None else cfgs)
    inst = [(layer, m, "", cfg) for (layer, m), cfg in cfgs.items()]
    if variants:
        inst += [(layer, m, tag, cfg) for (layer, m, tag), cfg in variants.items()]
    st = {"ent": {}, "keep": []}
    if any(c[7] & (_p5_FLAG_TRC | _p5_FLAG_STR) for _, _, _, c in inst):
        st["trc_buf"] = torch.zeros(_p5_TRC_WORDS, dtype=torch.int64, device="cuda")
        st["trc"] = st["trc_buf"].data_ptr()
    comps, wss, ess = {}, {}, {}
    for layer, m, tag, cfg in inst:
        R, NT, NWR, WSK, UPW, NACC, cap, fl = cfg[:8]
        assert R == NWR * NT * 8 and m in (16, 32), cfg
        n_rows = _p5_N_OF[layer]
        if layer not in comps:
            comps[layer] = _p5_compress12(weights[layer][:n_rows])
        f = comps[layer]
        esc1 = bool(fl & _p5_FLAG_ESC1)
        if ((R // 8 > 4 and not esc1) or WSK * UPW * 64 > 1024 or n_rows % R or (_p5_K_OF[layer] // (WSK * UPW * 64)) > 64
                or (esc1 and WSK * UPW * 64 > 512)):
            print(f"[pu5] {(layer, m, tag)} cfg {cfg} unsupported", flush=True)
            continue
        wkey = (layer, R, NT, NWR, WSK, UPW)
        if wkey not in wss:
            wss[wkey] = _p5__wstream12(f, n_rows, R, NT, NWR, WSK, UPW)
        ws = wss[wkey]
        ekey = (layer, R, WSK * UPW, esc1)
        if ekey not in ess:
            ess[ekey] = _p5__elist1(f, n_rows, R, WSK * UPW) if esc1 else _p5__elist12(f, n_rows, R, WSK * UPW)
        es = ess[ekey]
        if es is None:
            print(f"[pu5] {(layer, m, tag)}: escapes do not fit: left to the base", flush=True)
            continue
        g = _p5__geom(cfg, layer, m, es["lc"])
        nstage, sst = _p5__nstage(cfg, g)
        smem = _p5__smem(g, nstage, sst)
        if smem > _p5_SMEM_LIMIT or _p5__xlag(cfg) >= nstage:
            print(f"[pu5] {(layer, m, tag)} needs {smem} B smem / xlag {_p5__xlag(cfg)} vs nstage {nstage}: skipped", flush=True)
            continue
        ncta = n_rows // R
        scale = 2.0 ** (f["elo"] - _p5_FAKE12)
        st["ent"][(layer, m, tag)] = dict(cfg=cfg, ws=ws, es=es, n=n_rows, k=_p5_K_OF[layer], g=g, nthr=g["nthr"],
                                          ncta=ncta, smem=smem, kn=f"pu5_{layer}_{m}{tag}", nstage=nstage, sst=sst,
                                          ok=True, scale=scale)
    src = _p5_SRC
    for (layer, m, tag), e in st["ent"].items():
        R, NT, NWR, WSK, UPW, NACC, cap, fl = e["cfg"][:8]
        g = e["g"]
        bk = _p5__bake(e["cfg"])
        wb = f"{e['ws'].data_ptr()}ull" if bk & 1 else "0ull"
        eb = f"{e['es']['buf'].data_ptr()}ull" if bk & 2 else "0ull"
        mm, nn = (m, e["n"]) if bk & 4 else (0, 0)
        scb = struct.unpack("<I", struct.pack("<f", e["scale"]))[0] if bk & 4 else 0
        kfl = fl & ~_p5_F12
        src += (f'extern "C" __global__ void __launch_bounds__({e["nthr"]}, 1) {e["kn"]}(const __grid_constant__ P p) '
                f'{{ pk_body<{g["mt"]}, {g["tokr"]}, {NT}, {NWR}, {WSK}, {UPW}, {e["nstage"]}, {NACC}, {g["nst"]}, '
                f'{wb}, {eb}, {g["lc"]}, {_p5__xlag(e["cfg"])}, {mm}, {nn}, {scb}u, {kfl}>(p); }}\n')
    st["src"] = src
    st["mod"] = _p5__Module(src)
    for key, e in st["ent"].items():
        e["f"] = st["mod"].func(e["kn"], e["smem"])
        if verbose:
            a = st["mod"].attrs(e["kn"], e["smem"])
            print(f"[pu5] {key} cfg {e['cfg']} nstage {e['nstage']} sst {e['sst']} smem {e['smem']} thr {e['nthr']} "
                  f"ncta {e['ncta']} regs {a['NUM_REGS']} local {a['LOCAL_SIZE_BYTES']} esc {e['es']['n']} "
                  f"cmax {e['es']['cmax']} lc {e['g']['lc']}", flush=True)
    torch.cuda.synchronize()
    return st


def _p5_run_pu5(x, config, state, tag=""):
    m, layer = config["m"], config["layer"]
    if (layer, m) not in _p5_OWN and not tag:
        return None
    e = state["ent"].get((layer, m, "" if tag == "__self" else tag))
    if e is None or not e["ok"] or config["n"] != e["n"] or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    n, k = e["n"], e["k"]
    g = e["g"]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    tm = _p5__xmap(x, m, k, g["tokr"], g["ups"])
    params = tm + struct.pack("<QQQQQfiiiii", e["ws"].data_ptr(), x.data_ptr(), out.data_ptr(), e["es"]["buf"].data_ptr(),
                              state.get("trc", 0), e["scale"], m, n, k, 0, 0)
    state["mod"].launch(e["f"], e["ncta"], e["nthr"], e["smem"], params)
    return out


# ======================================================================== module H: H/skh.py
"""skh (agent H): agent F's sk11 ring kernel (claude/t3/F/sk11.py) + H options; shipped: unpooler m = 32 only.

    state = prepare_skh(weights)            # never raises (any failure -> run_skh returns None for that shape)
    out   = run_skh(x, config, state)       # -> [M, N] bf16, or None where the base (bo7b) keeps the shape

Shipped (H_CFG): unpooler m = 32 with R = 64 rows per CTA (128 CTAs, one per SM: x is read once per SM instead of
twice), one escape warp with a per-CTA stage-sorted escape list and shared-memory corrections (ESC1, lifts sk11's
4-tile limit), baked W pointer. Same-run bench vs bo7b: 13.07-13.18 vs 13.37-13.53 us local (claude/t3/H/NOTES.md).
Everything else in this file is behind cfg flags / extra cfg fields and documented (mostly as dead ends) in NOTES.md.

Kernel structure derived from vjeux/okc-2026 `sk2` (via alygg77's public OKC submission, Apache-2.0), agent X's NVRTC
port sk3 (claude/t3/X/sk3.py) and agent F's sk11 (F12 format by agent G): swapped operands (A = x tile 16 tokens x 16 k
via ldmatrix from a TMA-loaded 128B-swizzled tile, B = 8 weight rows x 16 k decoded from the 12-bit weight stream), each
CTA owns R rows x the whole K, a producer warp refills a ring of stages (x tile + this CTA's weights).
"""
import ctypes
import math
import os
import struct

import torch

# ----------------------------------------------------------------------------------------------------------- CUDA
_hh_SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) P { TMap xmap; const u8* W; const u16* x; u16* out; const u8* E; const uint2* etab; u64* trc;
                         float scale; int M; int N; int K; int pad0; int pad1; TMap wmap; };

// baked pointer + a non-foldable runtime zero (agent K: ptxas may otherwise drop the high 32 bits of the constant)
template <int V> struct ICst { static constexpr int value = V; };
#ifndef SKH_RTZERO
#define SKH_RTZERO 0
#endif
template <u64 A> __device__ __forceinline__ const u8* cptr() {
  const u8* p; asm("mov.u64 %0, %1;" : "=l"(p) : "n"(A));
  if constexpr (SKH_RTZERO) { u32 z; asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(z)); p += (z >> 31); }
  return p;
}
__device__ __forceinline__ u64 gtime() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ u64 ctime() { u64 t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory"); return t; }
__device__ __forceinline__ void ldsm_x4(u32 addr, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 sel) {
  u32 d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel)); return d;
}
__device__ __forceinline__ u32 rotr(u32 x, int s) { return __funnelshift_r(x, x, s); }
__device__ __forceinline__ u32 hi11(u32 x) {
  u32 r; asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(x), "r"(0x83838383u), "n"(0x3C3C3C3C)); return r;
}
__device__ __forceinline__ u32 sel80(u32 a, u32 b) {
  u32 r; asm("lop3.b32 %0, %1, %2, %3, 0xE4;" : "=r"(r) : "r"(a), "r"(b), "r"(0x80808080u)); return r;
}
// F11 decode (agent G's layout): 8 lo words (L0 = tile-unit A, L1 = tile-unit B) + 3 code words -> 2 x 8 bf16x2
template <int NP>
__device__ __forceinline__ void decode11p(const uint4& L0, const uint4& L1, u32 W0, u32 W1, u32 W2, u32* bA, u32* bB, const u32* pe) {
  u32 h[8];
  h[0] = hi11(W0); h[1] = hi11(rotr(W0, 3)); h[2] = hi11(W1); h[3] = hi11(rotr(W1, 3));
  h[4] = hi11(W2); h[5] = hi11(rotr(W2, 3));
  h[6] = hi11(sel80(rotr(W0, 6), rotr(W1, 5))); h[7] = hi11(sel80(rotr(W0, 7), rotr(W2, 5)));
  // exponent patches: entry = pos (5 bits: weight 4g + b of the group) | mag << 5 (hi byte -= mag), 0 = none
#pragma unroll
  for (int r = 0; r < NP; ++r) {
    const u32 pos = pe[r] & 31u, v = ((pe[r] >> 5) & 63u) << ((pos & 3u) * 8u), gq = pos >> 2;
#pragma unroll
    for (int q = 0; q < 8; ++q) if (gq == (u32)q) h[q] -= v;
  }
  const u32 h0 = h[0], h1 = h[1], h2 = h[2], h3 = h[3], h4 = h[4], h5 = h[5], h6 = h[6], h7 = h[7];
  bA[0] = prmt(L0.x, h0, 0x5140); bA[1] = prmt(L0.x, h0, 0x7362);
  bA[2] = prmt(L0.y, h1, 0x5140); bA[3] = prmt(L0.y, h1, 0x7362);
  bA[4] = prmt(L0.z, h2, 0x5140); bA[5] = prmt(L0.z, h2, 0x7362);
  bA[6] = prmt(L0.w, h3, 0x5140); bA[7] = prmt(L0.w, h3, 0x7362);
  bB[0] = prmt(L1.x, h4, 0x5140); bB[1] = prmt(L1.x, h4, 0x7362);
  bB[2] = prmt(L1.y, h5, 0x5140); bB[3] = prmt(L1.y, h5, 0x7362);
  bB[4] = prmt(L1.z, h6, 0x5140); bB[5] = prmt(L1.z, h6, 0x7362);
  bB[6] = prmt(L1.w, h7, 0x5140); bB[7] = prmt(L1.w, h7, 0x7362);
}
__device__ __forceinline__ void decode11(const uint4& L0, const uint4& L1, u32 W0, u32 W1, u32 W2, u32* bA, u32* bB) {
  decode11p<0>(L0, L1, W0, W1, W2, bA, bB, nullptr);
}
// agent G's F12 decode: 8 weights from 2 lo words + 1 nibble word (byte b = h_b | s_b << 7 | h_{4+b} << 4 | s_{4+b} at bit 3 of byte b-1)
__device__ __forceinline__ void decode12(u32 L0, u32 L1, u32 nw, u32* r) {
  u32 hA, hB;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(hA) : "r"(nw), "r"(0x87878787u), "n"(0x38383838));
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(hB) : "r"(rotr(nw, 4)), "r"(0x87878787u), "n"(0x38383838));
  r[0] = prmt(L0, hA, 0x5140); r[1] = prmt(L0, hA, 0x7362);
  r[2] = prmt(L1, hB, 0x5140); r[3] = prmt(L1, hB, 0x7362);
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a)); return r; }
__device__ __forceinline__ void mbar_init(u32 bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `mbar_test` / `mbar_wait` / `mbar_wait_sleep` (MBAR_HELPERS) (via alygg77/okc-muse-spark-kernels).
// Changes: renamed types (u32); mbar_wait written on one line; vjeux's esc_delta, which sits between mbar_wait and mbar_wait_sleep upstream, left out.
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
__device__ __forceinline__ bool mbar_test(u32 bar, u32 parity) {
  u32 ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(u32 bar, u32 parity) { while (!mbar_test(bar, parity)) {} }
__device__ __forceinline__ void mbar_wait_sleep(u32 bar, u32 parity) {
  u32 ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(0x989680u) : "memory");
  } while (!ok);
}
// SPDX-SnippetEnd
__device__ __forceinline__ void mbar_arrive(u32 bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(u32 bar, u32 bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\n}\n" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_expect_only(u32 bar, u32 bytes) {
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ u64 pol_first() { u64 q; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(q)); return q; }
__device__ __forceinline__ u64 pol_last() { u64 q; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(q)); return q; }
__device__ __forceinline__ void bulk_g2s_h(u32 dst, const void* src, u32 bytes, u32 bar, u64 pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void tma_load_3d_h(u32 dst, const TMap* map, u32 bar, int c0, int c1, int c2, u64 pol) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4, %5}], [%2], %6;"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2), "l"(pol) : "memory");
}
__device__ __forceinline__ void prefetch_l2(const void* src, u32 bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(src), "r"(bytes) : "memory");
}
// (H) W / x copies with optional L2 cache hints (FL 1 << 24: W evict_first, 1 << 25: x evict_last)
template <int FL> __device__ __forceinline__ void w_copy(u32 dst, const void* src, u32 bytes, u32 bar) {
  if constexpr ((FL & 16777216) != 0) bulk_g2s_h(dst, src, bytes, bar, pol_first()); else bulk_g2s(dst, src, bytes, bar);
}
// (H) FL 1 << 29: W stage block as a TMA 3D tile of 1 KB pieces interleaved across CTAs ([stage][piece][cta][1 KB])
#define W_COPY(DST_, ST_, BAR_) do { if constexpr (XP == 8) { /* (H) ablation: no W traffic */ } else if constexpr ((FL & 536870912) != 0) tma_load_3d(DST_, &p.wmap, BAR_, 0, cta, (ST_) * (WBY / 1024)); \
    else if (HF > 0 && (ST_) >= NSTN) w_copy<FL>(DST_, wsrc_h + (size_t)((ST_) - NSTN) * (wstride / 2), WBY / 2, BAR_); \
    else w_copy<FL>(DST_, wsrc + (size_t)(ST_) * wstride, WBY, BAR_); } while (0)
__device__ __forceinline__ void tma_load_3d(u32 dst, const TMap* map, u32 bar, int c0, int c1, int c2);
template <int FL> __device__ __forceinline__ void x_copy1(u32 dst, const TMap* map, u32 bar, int c2);
// (H) XS > 1: the stage's x box is fetched as XS TMA loads of UPS / XS units (tensor-map box = UPS / XS units)
template <int FL, int XS, int UPS, int XT> __device__ __forceinline__ void x_copy(u32 dst, const TMap* map, u32 bar, int c2) {
#pragma unroll
  for (int i = 0; i < XS; ++i) x_copy1<FL>(dst + i * (UPS / XS) * XT, map, bar, c2 + i * (UPS / XS));
}
template <int FL> __device__ __forceinline__ void x_copy1(u32 dst, const TMap* map, u32 bar, int c2) {
  if constexpr ((FL & 1073741824) != 0) return;   // (H) ablation: no x traffic (wrong results)
  if constexpr ((FL & (1 << 20)) != 0) return;    // (H) internal: XL > 0 (x via cp.async loader warps)
  if constexpr ((FL & 33554432) != 0) tma_load_3d_h(dst, map, bar, 0, 0, c2, pol_last()); else tma_load_3d(dst, map, bar, 0, 0, c2);
}
__device__ __forceinline__ void tma_load_3d(u32 dst, const TMap* map, u32 bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ u32 atom_add_ar(u32 addr, u32 v) {
  u32 old; asm volatile("atom.acq_rel.cta.shared::cta.add.u32 %0, [%1], %2;" : "=r"(old) : "r"(addr), "r"(v) : "memory"); return old;
}
// relaxed variant (H flag 1 << 21): every arriving warp has consumed its smem reads (MMA/FMA operands) before lane 0
// arrives, and the refill is issued only after the last atomic returns
__device__ __forceinline__ u32 atom_add_rx(u32 addr, u32 v) {
  u32 old; asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(old) : "r"(addr), "r"(v) : "memory"); return old;
}
__device__ __forceinline__ u32 mapa_s(u32 a, u32 rank) {
  u32 r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank)); return r;
}
__device__ __forceinline__ void st_async4(u32 raddr, float4 v, u32 rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               :: "r"(raddr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");
}
__device__ __forceinline__ void mbar_wait_cl(u32 bar, u32 parity) {
  u32 ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  } while (!ok);
}
__device__ __forceinline__ void sts32(u32 a, u32 v) { asm volatile("st.shared.u32 [%0], %1;" :: "r"(a), "r"(v) : "memory"); }
__device__ __forceinline__ u32 lds32(u32 a) { u32 v; asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ uint2 lds64(u32 a) {
  uint2 v; asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a) : "memory"); return v;
}
__device__ __forceinline__ u32 lds16(u32 a) { u16 v; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a)); return (u32)v; }

// escape-mma B fragments for lane (g, t) from its 4 entries (slots 2t, 2t+1, 2t+8, 2t+9): B1 = -d, B2 = true value
// (fake bf16) where the entry's row (bits 9-12; 8 = invalid) equals g, else 0
__device__ __forceinline__ void esc_b(const u32* e, int g, u32 ND, u32& b1a, u32& b1b, u32& b2a, u32& b2b) {
  bool mm[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) mm[q] = ((e[q] >> 9) & 15u) == (u32)g;
  b1a = (mm[0] ? ND : 0u) | (mm[1] ? (ND << 16) : 0u);
  b1b = (mm[2] ? ND : 0u) | (mm[3] ? (ND << 16) : 0u);
  b2a = (mm[0] ? (e[0] >> 16) : 0u) | (mm[1] ? (e[1] & 0xFFFF0000u) : 0u);
  b2b = (mm[2] ? (e[2] >> 16) : 0u) | (mm[3] ? (e[3] & 0xFFFF0000u) : 0u);
}

__device__ __forceinline__ bool mbar_testwait(u32 bar, u32 parity) {
  u32 ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
// escape warps: spin (default) or sleep (FL 256)
template <int FL> __device__ __forceinline__ void wait_e(u32 bar, u32 par) {
  if constexpr ((FL & 256) != 0) mbar_wait_sleep(bar, par); else mbar_wait(bar, par);
}
template <int FL> __device__ __forceinline__ void wait_s(u32 bar, u32 par) {
  if constexpr ((FL & 32) != 0) mbar_wait(bar, par); else mbar_wait_sleep(bar, par);
}

// MT token tiles of 16 (TOKR = x box rows = M); NT n8 tiles per consumer warp; NWR row warps; WSK k warps; UPW 64-k
// units per warp per stage; NSTAGE ring slots; NACC accumulator sets; NST stages; WB baked W pointer (0 = params);
// EBC escape-area capacity per slot; CIB fixed size of the first NSTAGE escape blocks; FL flags:
//   2 trace, 16 consumers skip decode+mma (ablation), 32 spin waits for producer, 64 escape warps skip work (ablation,
//   wrong results; 512 / 1024 / 2048: escape warps skip the x gathers / the B build + mma / the entry loads),
//   wrong results), 128 no escape warps at all (ablation, wrong results), 256 escape warps sleep-wait, 4096 per-stage trace (CTA 0), 8192 consumers sleep-wait, 16384 escape-warp clock64 trace, 32768 escape warps off the empty barriers (timing only),
//   65536 consumer exponent-patch timing probe
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_kernel` (via alygg77/okc-muse-spark-kernels).
// Changes: kept sk2's structure (swapped operands: the x tile is the A operand, loaded by ldmatrix from a TMA-loaded 128B-swizzled tile; each CTA owns R rows x the whole K; a producer warp refills a ring of stages); replaced its 12-bit weight decode (8-exponent window) with our lossless fake-exponent formats and its escape-warp corrections with our per-CTA escape lists and exact corrections, and changed the ring sizing, the producer's issue order and the constants (baked in by NVRTC).
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, int NACC, int NST, u64 WB, int EBC, int CIB, int NG, int FMT, int LC, int FL, int SK, int PM, int RG, int XS, int XP, int HF, int IF, int XL>
__device__ __forceinline__ void sk11_body(const P& p) {
  constexpr int NW = NWR * WSK;
  constexpr bool ESC1 = (FL & 131072) != 0;   // (H) one escape warp, per-CTA stage-sorted list, smem corrections
  constexpr bool LR = (FL & 262144) != 0;     // (H) last arriver of a slot issues its refill (no producer wake-up)
  constexpr bool HTR = (FL & 524288) != 0;    // (H) per-warp per-stage trace (CTA 0 and CTA ncta/2)
  static_assert(!LR || ESC1, "LR needs ESC1 (the escape warp does the initial fill)");
  static_assert(!ESC1 || FMT == 12, "ESC1 is F12 only");
  constexpr int NE = (FL & 128) ? 0 : ESC1 ? 1 : NWR * NT;
  constexpr int PFD = (FL >> 26) & 7;         // (H) W L2-prefetch distance in stages beyond the ring (0 = off)
  constexpr int NARR = NW + ((FL & 32768) ? 0 : NE);   // arrivals per stage on a slot
  // (H) XL > 0: x tiles are fetched by XL x-loader warps with cp.async (LSU path, 16 B per lane, 128B-swizzled like the
  // TMA box) and arrive on the full barrier via cp.async.mbarrier.arrive.noinc; the TMA unit then only carries W
  static_assert(XL == 0 || (ESC1 && !LR && SK == 1 && HF == 0 && RG == 0 && XS == 1), "XL");
  constexpr int EW0 = LR ? NW : NW + 1 + XL;            // first escape warp
  constexpr int NC = NW * 32;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int T = UPW * NT;
  static_assert(FMT == 12 || T % 2 == 0, "F11: tile-units per warp-stage must be even");
  static_assert(ESC1 || NWR * NT <= 4, "at most 4 escape tiles (header holds 4 counts)");
  constexpr int TB = FMT == 12 ? 768 : 704;
  constexpr int WBY = NW * T * TB;
  constexpr int ELB = FMT == 12 ? (ESC1 ? 16 + LC * 8 : 16 + NWR * NT * LC * 4) : NST * 16;   // escape lists (F12) / stage table (F11)
  constexpr int LSK = UPS == 1 ? 6 : UPS == 2 ? 7 : UPS == 4 ? 8 : 9;
  constexpr int XT = TOKR * 128;
  constexpr int LXU = TOKR == 8 ? 10 : TOKR == 16 ? 11 : 12;
  constexpr int XBY = UPS * XT;
  constexpr int XBYE = ((FL & 1073741824) || XP == 9 || XL > 0) ? 0 : XBY;   // x bytes expected per stage (TMA)
  constexpr int SST = (XBY + WBY + EBC + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);
  constexpr int NSTP = NST;           // table entries (uint4 per stage)
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const u32 sraw = (u32)__cvta_generic_to_shared(smem_raw);
  const u32 sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const u32 bars = sbase + NSTAGE * SST;
  const u32 full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8;
  const u32 rbar = pbar + 16;                               // (H) split-K receive barrier (SK > 1)
  const u32 etab_s = pbar + 32;
  const u32 xbar_d = rbar;   // (H) XP 9 ablation: x loads complete on this never-waited barrier (rbar unused when SK = 1)
  const uint4* etab_p = reinterpret_cast<const uint4*>(gbase + (etab_s - sbase));
  float* corr = reinterpret_cast<float*>(gbase + (etab_s - sbase) + ELB);
  float* red = reinterpret_cast<float*>(gbase);
  constexpr int RS = R / SK;                                    // (H) output rows per split-K rank
  static_assert(SK == 1 || (RS % 4 == 0 && ESC1), "split-K: R / SK multiple of 4, ESC1 escapes");
  const u32 recv_s = etab_s + ELB + 2 * TOKS * RP * 4;          // (H) [SK-1][M][RS] fp32 partials from the partners
  const u64 tstart = (FL & (2 | 524288)) ? gtime() : 0ull;
  const u8* Wp = WB ? cptr<WB>() : p.W;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // (H) PM > 0: spread placement: of every 16 consecutive blockIdx (one GPC under the hardware block scheduler) only
  // the first PM work; the rest exit at once
  if (PM > 0 && (int)(blockIdx.x & 15u) >= PM) return;
  const int cta = PM > 0 ? (int)(blockIdx.x >> 4) * PM + (int)(blockIdx.x & 15u) : (int)blockIdx.x;
  const int ncta = PM > 0 ? (int)(gridDim.x >> 4) * PM : (int)gridDim.x;
  const int kr = SK > 1 ? cta % SK : 0, rb = SK > 1 ? cta / SK : cta;   // split-K rank / row block
  const int xu0 = kr * NST * UPS;                                          // first x unit of this CTA's K range
  // (H) RG > 1: x-stage rotation: CTA group cta % RG processes its stages in the order (t + rot) % NST so that the CTAs
  // do not all read the same x box at the same time; the W stream is re-laid out time-major ([t][cta]).
  const int rot = RG > 1 ? (cta % RG) * (NST / RG) : 0;
#define XST(T_) (RG > 1 ? (((T_) + rot) % NST) : RG < 0 ? ((T_) % (-RG)) : (T_))   // RG < 0: ablation, x of stage t % -RG
  const u8* wsrc = Wp + (size_t)cta * WBY;
  const size_t wstride = (size_t)ncta * WBY;
  // (H) HF > 0: the last HF stages are split into 2 HF half stages (UPW / 2 units per warp, half the W bytes), so the
  // compute left after the final landing is halved. Half-stage W blocks follow the NSTN normal stages.
  constexpr int NSTN = NST - 2 * HF;
  static_assert(HF == 0 || (UPW % 2 == 0 && FMT == 12 && SK == 1 && RG == 0 && ESC1 && !(FL & 536870912) && NSTN >= 1), "HF");
  const u8* wsrc_h = Wp + (size_t)NSTN * ncta * WBY + (size_t)cta * (WBY / 2);
#define XU(S_) ((HF > 0 && (S_) >= NSTN) ? NSTN * UPS + ((S_) - NSTN) * (UPS / 2) : (S_) * UPS)
#define WBYS(S_) (XP == 8 ? 0 : (HF > 0 && (S_) >= NSTN) ? WBY / 2 + XBYE / 2 - XBYE : WBY)   // (+ x bytes of a half stage - XBYE)
#define XMAP(S_) ((HF > 0 && (S_) >= NSTN) ? &p.wmap : &p.xmap)                 // half stages: x map with UPS / 2 units

  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&p.xmap) : "memory");
    if constexpr ((FL & 536870912) != 0) asm volatile("prefetch.tensormap [%0];" :: "l"(&p.wmap) : "memory");
    for (int i = 0; i < NSTAGE; ++i) {
      mbar_init(full0 + 8 * i, 1 + 32 * XL);
      if constexpr (LR) sts32(empty0 + 8 * i, 0u); else mbar_init(empty0 + 8 * i, NARR);
    }
    mbar_init(pbar, 1);
    mbar_init(cbar, NE > 0 ? NE : 1);
    if constexpr (SK > 1 || XP == 9) { mbar_init(rbar, 1); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if constexpr (SK > 1) mbar_expect_tx(rbar, (u32)((SK - 1) * p.M * RS * 4));
    mbar_expect_tx(full0, XBYE + (XP == 8 ? 0 : WBY) + CIB);
    W_COPY(sbase + XBY, 0, full0);
    if constexpr (XP == 9) { mbar_expect_only(xbar_d, XBY); x_copy<FL, XS, UPS, XT>(sbase, &p.xmap, xbar_d, xu0 + XU(XST(0))); }
    else x_copy<FL, XS, UPS, XT>(sbase, &p.xmap, full0, xu0 + XU(XST(0)));
    if constexpr (CIB > 0) bulk_g2s(sbase + XBY + WBY, p.E + (size_t)cta * CIB, CIB, full0);
    mbar_expect_tx(pbar, ELB);
    bulk_g2s(etab_s, reinterpret_cast<const u8*>(p.etab) + (size_t)cta * ELB, ELB, pbar);
    // (H) XP 1: CTA c < M prefetches x row c (this CTA's K range) into L2 (x is cold after the flush and every CTA
    // reads it); XP 2: CTA c < M * 2 prefetches half a row
    if constexpr (XP == 1) {
      if (cta < p.M) prefetch_l2(reinterpret_cast<const u8*>(p.x) + ((size_t)cta * p.K + (size_t)xu0 * 64) * 2, NST * UPS * 128);
    } else if constexpr (XP == 2) {
      if (cta < 2 * p.M) prefetch_l2(reinterpret_cast<const u8*>(p.x) + ((size_t)(cta >> 1) * p.K + (size_t)xu0 * 64 + (cta & 1) * NST * UPS * 32) * 2, NST * UPS * 64);
    }
  }
  __syncthreads();
  if constexpr (SK > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

  const int tsel = HTR ? (cta == 0 ? 0 : cta == ncta / 2 ? 1 : -1) : -1;
  if (HTR && tsel >= 0 && tid == 0) { p.trc[16384 + 512 + tsel] = tstart; p.trc[16384 + 520 + tsel] = ctime(); }
  // refill of stage s into its slot (W + x tile; F12 only in LR mode)
#define SKH_ISSUE(S_) do { const int s_ = (S_); const int sl_ = s_ % NSTAGE; const u32 d_ = sbase + sl_ * SST; \
    if (HTR && tsel >= 0) p.trc[16384 + tsel * 128 + s_] = gtime(); \
    mbar_expect_tx(full0 + 8 * sl_, XBYE + WBYS(s_)); \
    W_COPY(d_ + XBY, s_, full0 + 8 * sl_); \
    x_copy<FL, XS, (HF > 0 ? 1 : UPS), XT>(d_, XMAP(s_), full0 + 8 * sl_, xu0 + XU(XST(s_))); } while (0)
  // slot release by lane 0 after __syncwarp: mbarrier arrive, or (LR) counter + refill by the last arriver
#define SKH_RELEASE(ST_) do { const int st_ = (ST_); const int sl_ = st_ % NSTAGE; \
    if constexpr (LR) { \
      const u32 old_ = (FL & 2097152) ? atom_add_rx(empty0 + 8 * sl_, 1u) : atom_add_ar(empty0 + 8 * sl_, 1u); \
      if (old_ == (u32)((st_ / NSTAGE + 1) * NARR - 1) && st_ + NSTAGE < NST) SKH_ISSUE(st_ + NSTAGE); \
    } else mbar_arrive(empty0 + 8 * sl_); } while (0)

  // (H) IF: the initial ring fill (stages 1 .. NSTAGE - 1) is issued in parallel by lane 0 of warps 0, 1, ... instead of
  // serially by the producer thread (each bulk / TMA issue costs the issuing thread ~60-130 cycles)
  if constexpr (IF > 0 && !LR) {
    constexpr int NWT = NW + 1 + NE;
    if (lane == 0)
      for (int s1 = 1 + warp; s1 < NSTAGE && s1 < NST; s1 += NWT) SKH_ISSUE(s1);
  }
  if (!LR && warp == NW) {   // ---------------- producer
    if (lane == 0) {
      if constexpr (PFD > 0)   // (H) stages NSTAGE .. NSTAGE + PFD - 1 to L2 right away
        for (int s1 = NSTAGE; s1 < NSTAGE + PFD - 1 && s1 < NST; ++s1) prefetch_l2(wsrc + (size_t)s1 * wstride, WBY);
      for (int st = (IF > 0 ? NSTAGE : 1); st < NST; ++st) {
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) wait_s<FL>(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
        const u8* esrc = p.E;
        u32 eby = 0;
        if constexpr (FMT == 11) {
          if (st == NSTAGE) wait_s<FL>(pbar, 0);
          if (st < NSTAGE) { esrc = p.E + ((size_t)st * ncta + cta) * CIB; eby = CIB; }
          else { const uint4 e4 = etab_p[st]; esrc = p.E + (size_t)e4.x * 16; eby = e4.y; }
        }
        const u32 s0 = sbase + slot * SST;
        if ((FL & 4096) && cta == 0) p.trc[32768 + st * 8 + 0] = gtime();
        mbar_expect_tx(full0 + 8 * slot, XBYE + WBYS(st) + eby);
        W_COPY(s0 + XBY, st, full0 + 8 * slot);
        if constexpr (XP == 9) { mbar_expect_only(xbar_d, XBY); x_copy<FL, XS, UPS, XT>(s0, &p.xmap, xbar_d, xu0 + XU(XST(st))); }
        else x_copy<FL, XS, (HF > 0 ? 1 : UPS), XT>(s0, XMAP(st), full0 + 8 * slot, xu0 + XU(XST(st)));
        if (eby) bulk_g2s(s0 + XBY + WBY, esrc, eby, full0 + 8 * slot);
        if constexpr (PFD > 0) {   // (H) L2 prefetch of the W block PFD stages beyond this refill
          if (st >= NSTAGE - 1 && st + PFD < NST) prefetch_l2(wsrc + (size_t)(st + PFD) * wstride, WBY);
        }
      }
    }
    __syncwarp();
    return;
  }

  if (XL > 0 && warp > NW && warp <= NW + XL) {   // ---------------- (H) x-loader warps (cp.async, LSU path)
    // XL warps x 4 rows of 128 B per iteration; all offsets but the per-lane base are compile-time immediates
    constexpr int KK = NST * UPS * 64 * SK;     // K
    constexpr int RPS = UPS * TOKR;             // 128-B rows of a stage's x tile
    constexpr int RPI = XL * 4;                 // rows per loader iteration (all loader warps)
    static_assert(XL == 0 || (RPI % 8 == 0 && TOKR % RPI == 0), "XL must be even with TOKR % (4 XL) == 0");
    const int xw = warp - NW - 1;
    const int row0 = xw * 4 + (lane >> 3), c = lane & 7;
    const int M = p.M;
    const u32 dofs = (u32)(row0 * 128) + ((u32)(c ^ (row0 & 7)) << 4);
    const u8* xb = reinterpret_cast<const u8*>(p.x) + ((size_t)row0 * KK + (size_t)xu0 * 64 + c * 8) * 2;
    for (int st = 0; st < NST; ++st) {
      const int slot = st % NSTAGE;
      if (st >= NSTAGE) wait_s<FL>(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
      const u32 s0 = sbase + slot * SST + dofs;
      const u8* sb = xb + (size_t)st * UPS * 128;
#pragma unroll
      for (int j = 0; j < RPS / RPI; ++j) {
        constexpr int dummy = 0; (void)dummy;
        const int rr = j * RPI, toff = rr % TOKR, u = rr / TOKR;
        const bool ok = row0 + toff < M;
        const u8* src = ok ? sb + ((size_t)toff * KK + u * 64) * 2 : reinterpret_cast<const u8*>(p.x);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(s0 + (u32)(u * XT + toff * 128)), "l"(src), "r"(ok ? 16 : 0) : "memory");
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(full0 + 8 * slot) : "memory");
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    return;
  }

  if (ESC1 && warp == EW0) {   // ---------------- (H) single escape warp (LR: also the initial ring fill)
    if constexpr (LR) {
      if (lane == 0)
        for (int s1 = 1; s1 < NSTAGE && s1 < NST; ++s1) SKH_ISSUE(s1);
      __syncwarp();
    }
    for (int i = lane; i < 2 * TOKS * RP; i += 32) corr[i] = 0.f;
    wait_e<FL>(pbar, 0);
    const u32 corr_s = etab_s + ELB;
    const int cnt = (int)lds32(etab_s);
    int ptr = 0;
    uint2 nx = cnt > 0 ? lds64(etab_s + 16) : make_uint2(0xFFFFFFFFu, 0u);
    const float nd = -__uint_as_float(0x38000000u);   // -d = -2^-15 (fake domain): every F12 escape decodes to d
    const int tok = lane;
    for (int st = 0; st < NST; ++st) {
      const int slot = st % NSTAGE;
      wait_e<FL>(full0 + 8 * slot, (st / NSTAGE) & 1);
      if (HTR && tsel >= 0 && lane == 0) p.trc[32768 + ((tsel * 128 + st) * 16 + warp) * 2] = gtime();
      if (!(FL & 64)) {
        const u32 s0 = sbase + slot * SST;
        while ((nx.x >> 16) == (u32)st) {
          const u32 kl = nx.x & (u32)(UPS * 64 - 1), kc = kl & 63u, u = kl >> 6;
          const int row = (int)((nx.x >> 9) & 127u);
          const float val = __uint_as_float(nx.y);
          float x = 0.f;
          if (tok < TOKR)
            x = __uint_as_float(lds16(s0 + u * XT + (u32)tok * 128 + ((((kc >> 3) ^ (u32)(tok & 7))) << 4) + (kc & 7u) * 2) << 16);
          if (tok < TOKS) {
            const u32 c1 = corr_s + (u32)(tok * RP + row) * 4u, c2 = corr_s + (u32)((TOKS + tok) * RP + row) * 4u;
            const float v1 = __uint_as_float(lds32(c1)), v2 = __uint_as_float(lds32(c2));
            sts32(c1, __float_as_uint(fmaf(nd, x, v1)));
            sts32(c2, __float_as_uint(fmaf(val, x, v2)));
          }
          ++ptr;
          nx = ptr < cnt ? lds64(etab_s + 16 + (u32)ptr * 8u) : make_uint2(0xFFFFFFFFu, 0u);
        }
      }
      __syncwarp();
      if (lane == 0) SKH_RELEASE(st);
      if (HTR && tsel >= 0 && lane == 0) { p.trc[32768 + ((tsel * 128 + st) * 16 + warp) * 2 + 1] = gtime(); p.trc[40960 + ((tsel * 128 + st) * 16 + warp)] = ctime(); }
    }
    __syncwarp();
    if ((FL & 2) && lane == 0) p.trc[blockIdx.x * 8 + 7] = gtime();
    if (lane == 0) mbar_arrive(cbar);
    return;
  }

  if (FMT == 12 && !ESC1 && warp > NW) {   // ---------------- F12 escape warps (rare escapes, lane = token)
    const int et = warp - NW - 1;
    const u32* el = reinterpret_cast<const u32*>(gbase + (etab_s - sbase));
    wait_e<FL>(pbar, 0);
    const int cnt = (int)el[et];
    const u32 ea = lane < cnt ? el[4 + et * LC + lane] : 0u;
    const u32 eb2 = (LC > 32 && lane + 32 < cnt) ? el[4 + et * LC + 32 + lane] : 0u;
    const int sa = lane < cnt ? (int)((ea & 0x1FFFu) >> LSK) : -1;
    const int sb = (LC > 32 && lane + 32 < cnt) ? (int)((eb2 & 0x1FFFu) >> LSK) : -1;
    float E1r[8], E2r[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) { E1r[r] = 0.f; E2r[r] = 0.f; }
    const float nd = -__uint_as_float(0x38000000u);   // -d = -2^-15 (fake domain): every F12 escape decodes to d
    const int tok = lane;
    for (int st = 0; st < NST; ++st) {
      const int slot = st % NSTAGE;
      wait_e<FL>(full0 + 8 * slot, (st / NSTAGE) & 1);
      u32 m0 = __ballot_sync(0xffffffffu, sa == st);
      u32 m1 = LC > 32 ? __ballot_sync(0xffffffffu, sb == st) : 0u;
      if ((m0 | m1) && !(FL & 64)) {
        const u32 s0 = sbase + slot * SST;
        while (m0 | m1) {
          u32 e;
          if (m0) { const int i = __ffs(m0) - 1; m0 &= m0 - 1; e = __shfl_sync(0xffffffffu, ea, i); }
          else { const int i = __ffs(m1) - 1; m1 &= m1 - 1; e = __shfl_sync(0xffffffffu, eb2, i); }
          const u32 kl = e & (u32)(UPS * 64 - 1), kc = kl & 63u, u = kl >> 6;
          const int row = (int)((e >> 13) & 7u);
          const float val = __uint_as_float(e & 0xFFFF0000u);
          float x = 0.f;
          if (tok < TOKR)
            x = __uint_as_float(lds16(s0 + u * XT + (u32)tok * 128 + ((((kc >> 3) ^ (u32)(tok & 7))) << 4) + (kc & 7u) * 2) << 16);
#pragma unroll
          for (int r = 0; r < 8; ++r)
            if (row == r) { E1r[r] = fmaf(nd, x, E1r[r]); E2r[r] = fmaf(val, x, E2r[r]); }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
    if (tok < TOKS) {
#pragma unroll
      for (int r = 0; r < 8; r += 4) {
        *reinterpret_cast<float4*>(corr + tok * RP + 8 * et + r) = make_float4(E1r[r], E1r[r + 1], E1r[r + 2], E1r[r + 3]);
        *reinterpret_cast<float4*>(corr + (TOKS + tok) * RP + 8 * et + r) = make_float4(E2r[r], E2r[r + 1], E2r[r + 2], E2r[r + 3]);
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(cbar);
    return;
  }

  if (FMT == 11 && warp > NW) {    // ---------------- F11 escape warps: tile et = rows 8 et .. 8 et + 7 of this CTA
    const int et = warp - NW - 1;
    const int g = lane >> 2, t = lane & 3;
    float E1[MT][4], E2[MT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int e = 0; e < 4; ++e) { E1[mt][e] = 0.f; E2[mt][e] = 0.f; }
    const u32 ND = 0xBC00u;   // -d = -2^-7 (fake domain), the decoded value of every escape
    // per stage: this tile's escape count n and entry offset (tile lists are padded to 4 entries) from the table
    auto cnt_of = [&](int st, int& n, u32& off) {
      const uint4 v = etab_p[st];
      const u32 c0 = v.z & 0xFFFFu, c1 = v.z >> 16, c2 = v.w & 0xFFFFu, c3 = v.w >> 16;
      const u32 p0 = (c0 + 3u) & ~3u, p1 = (c1 + 3u) & ~3u, p2 = (c2 + 3u) & ~3u;
      n = (int)(et == 0 ? c0 : et == 1 ? c1 : et == 2 ? c2 : c3);
      off = et == 0 ? 0u : et == 1 ? p0 : et == 2 ? p0 + p1 : p0 + p1 + p2;
    };
    // entries of group G for lane t: 4 consecutive u32 (slots 2t, 2t+1, 2t+8, 2t+9); invalid -> row 8 (no match)
    auto load_ent = [&](int st, u32 (*e)[4]) {
      int n; u32 off;
      cnt_of(st, n, off);
      const u32 eb = sbase + (st % NSTAGE) * SST + XBY + WBY + off * 4u + (u32)t * 16u;
#pragma unroll
      for (int G = 0; G < NG; ++G) {
        if (16 * G < n) {
          if constexpr ((FL & 2048) != 0) {
#pragma unroll
            for (int q = 0; q < 4; ++q) e[G][q] = ((u32)(G * 37 + q * 11 + t) & 0x1FFu) | 0x3C000000u;
          } else {
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(e[G][0]), "=r"(e[G][1]), "=r"(e[G][2]), "=r"(e[G][3]) : "r"(eb + (u32)G * 64u));
          }
#pragma unroll
          for (int q = 0; q < 4; ++q) if (16 * G + 4 * t + q >= n) e[G][q] = 8u << 9;
        }
      }
    };
    u32 ec[NG][4], en[NG][4];
    wait_e<FL>(pbar, 0);
    wait_e<FL>(full0, 0);
    if constexpr ((FL & 64) == 0) load_ent(0, ec);
    for (int st = 0; st < NST; ++st) {
      const int slot = st % NSTAGE;
      if (st > 0 || (FL & 64)) wait_e<FL>(full0 + 8 * slot, (st / NSTAGE) & 1);
      if ((FL & 4096) && cta == 0 && et == 0 && lane == 0) p.trc[32768 + st * 8 + 3] = gtime();
      if ((FL & 16384) && cta == 0 && et == 0 && lane == 0) p.trc[40960 + st * 8 + 0] = ctime();
      const u32 s0 = sbase + slot * SST;
      if constexpr ((FL & 64) != 0) {
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
        continue;
      }
      int n; u32 off;
      cnt_of(st, n, off);
      if ((FL & 16384) && cta == 0 && et == 0) { __syncwarp(); if (lane == 0) p.trc[40960 + st * 8 + 1] = ctime() + (u64)(n & 0); }
      const u32 xl = s0 + (u32)g * 128;
      // gathers for this stage (entries already in registers)
      u32 aa[NG][MT][4];
#pragma unroll
      for (int G = 0; G < NG; ++G) {
        if (16 * G < n) {
          u32 ad[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) ad[q] = xl + ((((ec[G][q] & 63u) << 1) | (((ec[G][q] >> 6) & 7u) << LXU)) ^ ((u32)g << 4));
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            const u32 o = (u32)mt * 2048u;
            if constexpr ((FL & 512) != 0) {
              aa[G][mt][0] = ad[0] + o; aa[G][mt][1] = ad[1] + o; aa[G][mt][2] = ad[2] + o; aa[G][mt][3] = ad[3] + o;
              continue;
            }
            aa[G][mt][0] = lds16(ad[0] + o) | (lds16(ad[1] + o) << 16);
            aa[G][mt][2] = lds16(ad[2] + o) | (lds16(ad[3] + o) << 16);
            if constexpr (TOKR >= 16) {
              aa[G][mt][1] = lds16(ad[0] + o + 1024u) | (lds16(ad[1] + o + 1024u) << 16);
              aa[G][mt][3] = lds16(ad[2] + o + 1024u) | (lds16(ad[3] + o + 1024u) << 16);
            } else {
              aa[G][mt][1] = 0u; aa[G][mt][3] = 0u;
            }
          }
        }
      }
      if ((FL & 16384) && cta == 0 && et == 0) {
        u32 acc = 0;
#pragma unroll
        for (int G = 0; G < NG; ++G) if (16 * G < n) acc ^= aa[G][0][0] ^ aa[G][MT - 1][3];
        __syncwarp();
        if (lane == 0) p.trc[40960 + st * 8 + 2] = ctime() + (u64)(acc & 0u);
      }
      // prefetch the next stage's entries if it has landed (non-blocking test)
      bool hn = false;
      if (st + 1 < NST) {
        const int s1 = (st + 1) % NSTAGE;
        if (mbar_testwait(full0 + 8 * s1, ((st + 1) / NSTAGE) & 1)) { load_ent(st + 1, en); hn = true; }
      }
      // overflow groups (rare): entries + gathers + mma while the slot is still held
      for (int G0 = 16 * NG; G0 < n; G0 += 16) {
        u32 e[4], ad[4];
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(e[0]), "=r"(e[1]), "=r"(e[2]), "=r"(e[3])
                     : "r"(s0 + XBY + WBY + off * 4u + (u32)(G0 + 4 * t) * 4u));
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          if (G0 + 4 * t + q >= n) e[q] = 8u << 9;
          ad[q] = xl + ((((e[q] & 63u) << 1) | (((e[q] >> 6) & 7u) << LXU)) ^ ((u32)g << 4));
        }
        u32 b1a, b1b, b2a, b2b;
        esc_b(e, g, ND, b1a, b1b, b2a, b2b);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          const u32 o = (u32)mt * 2048u;
          const u32 a0 = lds16(ad[0] + o) | (lds16(ad[1] + o) << 16);
          const u32 a2 = lds16(ad[2] + o) | (lds16(ad[3] + o) << 16);
          u32 a1 = 0u, a3 = 0u;
          if constexpr (TOKR >= 16) {
            a1 = lds16(ad[0] + o + 1024u) | (lds16(ad[1] + o + 1024u) << 16);
            a3 = lds16(ad[2] + o + 1024u) | (lds16(ad[3] + o + 1024u) << 16);
          }
          mma_bf16(E1[mt], a0, a1, a2, a3, b1a, b1b);
          mma_bf16(E2[mt], a0, a1, a2, a3, b2a, b2b);
        }
      }
      if ((FL & 16384) && cta == 0 && et == 0 && lane == 0) p.trc[40960 + st * 8 + 3] = ctime();
      __syncwarp();
      if (lane == 0 && !(FL & 32768)) mbar_arrive(empty0 + 8 * slot);
      if ((FL & 4096) && cta == 0 && et == 0 && lane == 0) p.trc[32768 + st * 8 + 4] = gtime();
      if ((FL & 16384) && cta == 0 && et == 0 && lane == 0) p.trc[40960 + st * 8 + 4] = ctime();
#pragma unroll
      for (int G = 0; G < NG; ++G) {
        if (16 * G < n) {
          if constexpr ((FL & 1024) != 0) {
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) E1[mt][0] += __uint_as_float((aa[G][mt][0] ^ aa[G][mt][1] ^ aa[G][mt][2] ^ aa[G][mt][3] ^ ec[G][0] ^ ec[G][1] ^ ec[G][2] ^ ec[G][3]) & 0x3FFFFFu);
          } else {
            u32 b1a, b1b, b2a, b2b;
            esc_b(ec[G], g, ND, b1a, b1b, b2a, b2b);
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
              mma_bf16(E1[mt], aa[G][mt][0], aa[G][mt][1], aa[G][mt][2], aa[G][mt][3], b1a, b1b);
              mma_bf16(E2[mt], aa[G][mt][0], aa[G][mt][1], aa[G][mt][2], aa[G][mt][3], b2a, b2b);
            }
          }
        }
      }
      if ((FL & 4096) && cta == 0 && et == 0 && lane == 0) p.trc[32768 + st * 8 + 5] = gtime();
      if ((FL & 16384) && cta == 0 && et == 0) {
        float acc = 0.f;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) acc += E1[mt][0] + E2[mt][3];
        __syncwarp();
        if (lane == 0) p.trc[40960 + st * 8 + 5] = ctime() + (u64)(acc != acc);
      }
      if (st + 1 < NST) {
        if (!hn) {
          wait_e<FL>(full0 + 8 * ((st + 1) % NSTAGE), ((st + 1) / NSTAGE) & 1);
          load_ent(st + 1, en);
        }
#pragma unroll
        for (int G = 0; G < NG; ++G)
#pragma unroll
          for (int q = 0; q < 4; ++q) ec[G][q] = en[G][q];
      }
    }
    if ((FL & 2) && et == 0 && lane == 0) p.trc[blockIdx.x * 8 + 7] = gtime();
    float* c1p = corr;
    float* c2p = corr + TOKS * RP;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      const int tok = mt * 16 + g, row = 8 * et + 2 * t;
      *reinterpret_cast<float2*>(c1p + tok * RP + row) = make_float2(E1[mt][0], E1[mt][1]);
      *reinterpret_cast<float2*>(c1p + (tok + 8) * RP + row) = make_float2(E1[mt][2], E1[mt][3]);
      *reinterpret_cast<float2*>(c2p + tok * RP + row) = make_float2(E2[mt][0], E2[mt][1]);
      *reinterpret_cast<float2*>(c2p + (tok + 8) * RP + row) = make_float2(E2[mt][2], E2[mt][3]);
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(cbar);
    return;
  }

  // ---------------- consumers
  const int wl = warp;
  const int wr = wl / WSK, wk = wl % WSK;
  const int g = lane >> 2, t4 = lane & 3;
  const int q8 = lane >> 3;
  const int chi = q8 >> 1;
  int trow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) trow[mt] = TOKR >= 16 ? mt * 16 + (lane & 7) + 8 * (q8 & 1) : (lane & 7);
  float acc[NACC][MT][NT][4];
#pragma unroll
  for (int c = 0; c < NACC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;
  u64 tfirst = 0;
  for (int st = 0; st < NST; ++st) {
    const int slot = st % NSTAGE;
    if constexpr ((FL & 8192) != 0) mbar_wait_sleep(full0 + 8 * slot, (st / NSTAGE) & 1);
    else mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    if ((FL & 2) && st == 0) tfirst = gtime();
    if (HTR && tsel >= 0 && lane == 0) p.trc[32768 + ((tsel * 128 + st) * 16 + warp) * 2] = gtime();
    if ((FL & 4096) && cta == 0 && warp == 0 && lane == 0) p.trc[32768 + st * 8 + 1] = gtime();
    const u32 s0 = sbase + slot * SST;
    if constexpr ((FL & 16) != 0) {
      __syncwarp();
      if (lane == 0) SKH_RELEASE(st);
      continue;
    }
    if (HF > 0 && st >= NSTN) {   // (H) half stage: UPW / 2 units per warp
    {   // stage body, UP = (UPW >= 2 ? UPW / 2 : 1)
        constexpr int UP = (UPW >= 2 ? UPW / 2 : 1);
        constexpr int TU = UP * NT;
        const u32 wb = s0 + XBY + (u32)wl * (TU * TB);
        u32 b[TU][8];
    #pragma unroll
        for (int uu = 0; uu < UP; ++uu) {
          const int ui = wk * UP + uu;
          const u32 xt = s0 + ui * XT;
          if constexpr (FMT == 12) {
    #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
              uint4 L;
              uint2 N;
              const u32 tb = wb + (u32)(uu * NT + nt) * 768u;
              asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(L.x), "=r"(L.y), "=r"(L.z), "=r"(L.w) : "r"(tb + lane * 16));
              asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(N.x), "=r"(N.y) : "r"(tb + 512 + lane * 8));
              decode12(L.x, L.y, N.x, &b[uu * NT + nt][0]);
              decode12(L.z, L.w, N.y, &b[uu * NT + nt][4]);
            }
          }
    #pragma unroll
          for (int j = 0; j < (FMT == 12 ? 0 : TU / 2); ++j) {
            if ((2 * j) / NT != uu) continue;
            uint4 L0, L1;
            uint2 C01;
            u32 C2;
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(L0.x), "=r"(L0.y), "=r"(L0.z), "=r"(L0.w) : "r"(wb + (2 * j) * 512 + lane * 16));
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(L1.x), "=r"(L1.y), "=r"(L1.z), "=r"(L1.w) : "r"(wb + (2 * j + 1) * 512 + lane * 16));
            asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(C01.x), "=r"(C01.y) : "r"(wb + TU * 512 + j * 384 + lane * 8));
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(C2) : "r"(wb + TU * 512 + j * 384 + 256 + lane * 4));
            if constexpr ((FL & 65536) != 0) {   // timing probe of the exponent-patch path (wrong results)
              u32 pe[2];
              const u32 eb0 = s0 + XBY + WBY;
              pe[0] = lds16(eb0 + ((lane * 7 + j * 64) & 255) * 2);
              pe[1] = lds16(eb0 + ((lane * 5 + 37 + j * 64) & 255) * 2);
              pe[0] = (lane & 3) ? (pe[0] & 0x7FFu) : 0u;
              pe[1] = (lane & 7) == 1 ? (pe[1] & 0x7FFu) : 0u;
              decode11p<2>(L0, L1, C01.x, C01.y, C2, b[2 * j], b[2 * j + 1], pe);
            } else {
              decode11(L0, L1, C01.x, C01.y, C2, b[2 * j], b[2 * j + 1]);
            }
          }
          u32 a[4][MT][4];
    #pragma unroll
          for (int j = 0; j < 4; ++j)
    #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
              if constexpr ((FL & 8388608) != 0) {   // (H) ablation: no x LDSM
                a[j][mt][0] = a[j][mt][1] = a[j][mt][2] = a[j][mt][3] = 0x3F803F80u ^ (u32)(st + j);
              } else {
                ldsm_x4(xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * TOKR + trow[mt]) & 7)) << 4),
                        a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
              }
            }
          if (UP == 1 && uu == 0) {
            __syncwarp();
            if (lane == 0) SKH_RELEASE(st);
            if (HTR && tsel >= 0 && lane == 0) { p.trc[32768 + ((tsel * 128 + st) * 16 + warp) * 2 + 1] = gtime(); p.trc[40960 + ((tsel * 128 + st) * 16 + warp)] = ctime(); }
          }
          if constexpr ((FL & 4194304) != 0) {   // (H) ablation: no HMMA (loads + decode kept alive by one XOR fold)
            u32 h = 0;
    #pragma unroll
            for (int j = 0; j < 4; ++j)
    #pragma unroll
              for (int mt = 0; mt < MT; ++mt) h ^= a[j][mt][0] ^ a[j][mt][1] ^ a[j][mt][2] ^ a[j][mt][3];
    #pragma unroll
            for (int nt = 0; nt < NT; ++nt)
    #pragma unroll
              for (int q = 0; q < 8; ++q) h ^= b[uu * NT + nt][q];
            acc[0][0][0][0] += __uint_as_float(h & 0x007FFFFFu);
          } else {
    #pragma unroll
          for (int j = 0; j < 4; ++j)
    #pragma unroll
            for (int mt = 0; mt < MT; ++mt)
    #pragma unroll
              for (int nt = 0; nt < NT; ++nt)
                mma_bf16(acc[(uu * 4 + j) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3],
                         b[uu * NT + nt][2 * j], b[uu * NT + nt][2 * j + 1]);
          }
        }
        if (UP > 1) {
          __syncwarp();
          if (lane == 0) SKH_RELEASE(st);
          if (HTR && tsel >= 0 && lane == 0) { p.trc[32768 + ((tsel * 128 + st) * 16 + warp) * 2 + 1] = gtime(); p.trc[40960 + ((tsel * 128 + st) * 16 + warp)] = ctime(); }
        }
      }
    } else
    {   // stage body, UP = UPW
      constexpr int UP = UPW;
      constexpr int TU = UP * NT;
      const u32 wb = s0 + XBY + (u32)wl * (TU * TB);
      u32 b[TU][8];
  #pragma unroll
      for (int uu = 0; uu < UP; ++uu) {
        const int ui = wk * UP + uu;
        const u32 xt = s0 + ui * XT;
        if constexpr (FMT == 12) {
  #pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            uint4 L;
            uint2 N;
            const u32 tb = wb + (u32)(uu * NT + nt) * 768u;
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(L.x), "=r"(L.y), "=r"(L.z), "=r"(L.w) : "r"(tb + lane * 16));
            asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(N.x), "=r"(N.y) : "r"(tb + 512 + lane * 8));
            decode12(L.x, L.y, N.x, &b[uu * NT + nt][0]);
            decode12(L.z, L.w, N.y, &b[uu * NT + nt][4]);
          }
        }
  #pragma unroll
        for (int j = 0; j < (FMT == 12 ? 0 : TU / 2); ++j) {
          if ((2 * j) / NT != uu) continue;
          uint4 L0, L1;
          uint2 C01;
          u32 C2;
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(L0.x), "=r"(L0.y), "=r"(L0.z), "=r"(L0.w) : "r"(wb + (2 * j) * 512 + lane * 16));
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(L1.x), "=r"(L1.y), "=r"(L1.z), "=r"(L1.w) : "r"(wb + (2 * j + 1) * 512 + lane * 16));
          asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(C01.x), "=r"(C01.y) : "r"(wb + TU * 512 + j * 384 + lane * 8));
          asm volatile("ld.shared.u32 %0, [%1];" : "=r"(C2) : "r"(wb + TU * 512 + j * 384 + 256 + lane * 4));
          if constexpr ((FL & 65536) != 0) {   // timing probe of the exponent-patch path (wrong results)
            u32 pe[2];
            const u32 eb0 = s0 + XBY + WBY;
            pe[0] = lds16(eb0 + ((lane * 7 + j * 64) & 255) * 2);
            pe[1] = lds16(eb0 + ((lane * 5 + 37 + j * 64) & 255) * 2);
            pe[0] = (lane & 3) ? (pe[0] & 0x7FFu) : 0u;
            pe[1] = (lane & 7) == 1 ? (pe[1] & 0x7FFu) : 0u;
            decode11p<2>(L0, L1, C01.x, C01.y, C2, b[2 * j], b[2 * j + 1], pe);
          } else {
            decode11(L0, L1, C01.x, C01.y, C2, b[2 * j], b[2 * j + 1]);
          }
        }
        u32 a[4][MT][4];
  #pragma unroll
        for (int j = 0; j < 4; ++j)
  #pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            if constexpr ((FL & 8388608) != 0) {   // (H) ablation: no x LDSM
              a[j][mt][0] = a[j][mt][1] = a[j][mt][2] = a[j][mt][3] = 0x3F803F80u ^ (u32)(st + j);
            } else {
              ldsm_x4(xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * TOKR + trow[mt]) & 7)) << 4),
                      a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
            }
          }
        if (UP == 1 && uu == 0) {
          __syncwarp();
          if (lane == 0) SKH_RELEASE(st);
          if (HTR && tsel >= 0 && lane == 0) { p.trc[32768 + ((tsel * 128 + st) * 16 + warp) * 2 + 1] = gtime(); p.trc[40960 + ((tsel * 128 + st) * 16 + warp)] = ctime(); }
        }
        if constexpr ((FL & 4194304) != 0) {   // (H) ablation: no HMMA (loads + decode kept alive by one XOR fold)
          u32 h = 0;
  #pragma unroll
          for (int j = 0; j < 4; ++j)
  #pragma unroll
            for (int mt = 0; mt < MT; ++mt) h ^= a[j][mt][0] ^ a[j][mt][1] ^ a[j][mt][2] ^ a[j][mt][3];
  #pragma unroll
          for (int nt = 0; nt < NT; ++nt)
  #pragma unroll
            for (int q = 0; q < 8; ++q) h ^= b[uu * NT + nt][q];
          acc[0][0][0][0] += __uint_as_float(h & 0x007FFFFFu);
        } else {
  #pragma unroll
        for (int j = 0; j < 4; ++j)
  #pragma unroll
          for (int mt = 0; mt < MT; ++mt)
  #pragma unroll
            for (int nt = 0; nt < NT; ++nt)
              mma_bf16(acc[(uu * 4 + j) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3],
                       b[uu * NT + nt][2 * j], b[uu * NT + nt][2 * j + 1]);
        }
      }
      if (UP > 1) {
        __syncwarp();
        if (lane == 0) SKH_RELEASE(st);
        if (HTR && tsel >= 0 && lane == 0) { p.trc[32768 + ((tsel * 128 + st) * 16 + warp) * 2 + 1] = gtime(); p.trc[40960 + ((tsel * 128 + st) * 16 + warp)] = ctime(); }
      }
    }
    if ((FL & 4096) && cta == 0 && warp == 0 && lane == 0) p.trc[32768 + st * 8 + 2] = gtime();
  }
  if ((FL & 2) && warp == 0 && lane == 0) {
    p.trc[blockIdx.x * 8 + 0] = tstart;
    p.trc[blockIdx.x * 8 + 1] = tfirst;
    p.trc[blockIdx.x * 8 + 2] = gtime();
    u32 smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    p.trc[blockIdx.x * 8 + 6] = smid;
  }
  if (HTR && tsel >= 0 && lane == 0) p.trc[16384 + 1024 + tsel * 64 + warp] = gtime();
  if constexpr (NE > 0) mbar_wait(cbar, 0);
  if ((FL & 2) && warp == 0 && lane == 0) p.trc[blockIdx.x * 8 + 5] = gtime();
  named_bar(1, NC);
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int row = (wr * NT + nt) * 8 + 2 * t4;
      const int tok = mt * 16 + g;
      float* r0 = red + (wk * TOKS + tok) * RP + row;
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = acc[0][mt][nt][e];
#pragma unroll
        for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
      }
      *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
    }
  named_bar(1, NC);
  constexpr int R4 = R / 4;
  const float sc = p.scale;
  const int M = p.M;
  if constexpr (SK > 1) {   // (H) split-K: own partial (+ escape corrections), reduce-scatter over the cluster
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");   // partners' rbar initialised (long done by now)
    for (int idx = tid; idx < R4 * M; idx += NC) {
      const int tt = idx / R4, row = (idx - tt * R4) * 4;
      float4 v = *reinterpret_cast<const float4*>(red + tt * RP + row);
#pragma unroll
      for (int w = 1; w < WSK; ++w) {
        const float4 pp = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
        v.x += pp.x; v.y += pp.y; v.z += pp.z; v.w += pp.w;
      }
      if constexpr (NE > 0) {
        const float4 e1 = *reinterpret_cast<const float4*>(corr + tt * RP + row);
        v.x += e1.x; v.y += e1.y; v.z += e1.z; v.w += e1.w;
        const float4 e2 = *reinterpret_cast<const float4*>(corr + (TOKS + tt) * RP + row);
        v.x += e2.x; v.y += e2.y; v.z += e2.z; v.w += e2.w;
      }
      const int dst = row / RS;
      if (dst != kr) {
        const int sl = kr < dst ? kr : kr - 1;
        st_async4(mapa_s(recv_s + (u32)(((sl * M + tt) * RS + (row - dst * RS)) * 4), (u32)dst), v, mapa_s(rbar, (u32)dst));
      } else {
        *reinterpret_cast<float4*>(red + tt * RP + row) = v;
      }
    }
    named_bar(1, NC);
    mbar_wait_cl(rbar, 0);
    const float* rcv = reinterpret_cast<const float*>(gbase + (recv_s - sbase));
    constexpr int RS4 = RS / 4;
    for (int idx = tid; idx < RS4 * M; idx += NC) {
      const int tt = idx / RS4, rr = (idx - tt * RS4) * 4, row = kr * RS + rr;
      float4 v = *reinterpret_cast<const float4*>(red + tt * RP + row);
#pragma unroll
      for (int sl = 0; sl < SK - 1; ++sl) {
        const float4 pp = *reinterpret_cast<const float4*>(rcv + (sl * M + tt) * RS + rr);
        v.x += pp.x; v.y += pp.y; v.z += pp.z; v.w += pp.w;
      }
      u16* o = p.out + (size_t)tt * p.N + rb * R + row;
      uint2 pk;
      pk.x = pack_bf2(v.x * sc, v.y * sc);
      pk.y = pack_bf2(v.z * sc, v.w * sc);
      *reinterpret_cast<uint2*>(o) = pk;
    }
  } else
  for (int idx = tid; idx < R4 * M; idx += NC) {
    const int tt = idx / R4, row = (idx - tt * R4) * 4;
    float4 v = *reinterpret_cast<const float4*>(red + tt * RP + row);
#pragma unroll
    for (int w = 1; w < WSK; ++w) {
      const float4 pp = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
      v.x += pp.x; v.y += pp.y; v.z += pp.z; v.w += pp.w;
    }
    if constexpr (NE > 0) {
      const float4 e1 = *reinterpret_cast<const float4*>(corr + tt * RP + row);
      v.x += e1.x; v.y += e1.y; v.z += e1.z; v.w += e1.w;
      const float4 e2 = *reinterpret_cast<const float4*>(corr + (TOKS + tt) * RP + row);
      v.x += e2.x; v.y += e2.y; v.z += e2.z; v.w += e2.w;
    }
    u16* o = p.out + (size_t)tt * p.N + cta * R + row;
    uint2 pk;
    pk.x = pack_bf2(v.x * sc, v.y * sc);
    pk.y = pack_bf2(v.z * sc, v.w * sc);
    *reinterpret_cast<uint2*>(o) = pk;
  }
  if ((FL & 2) && tid == 0) p.trc[blockIdx.x * 8 + 4] = gtime();
  if (HTR && tsel >= 0 && tid == 0) p.trc[16384 + 1024 + tsel * 64 + 32] = gtime();
  (void)wsrc;
#undef SKH_ISSUE
#undef SKH_RELEASE
#undef XST
#undef XU
#undef WBYS
#undef XMAP
}
// SPDX-SnippetEnd
"""

# ------------------------------------------------------------------------------------------------- configuration
# (R, NT, NWR, WSK, UPW, NACC, ring cap KB, flags) -- R = NWR*NT*8 rows per CTA; flags: 4 = baked W pointer + kernel FL
# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-SnippetCopyrightText: 2026 alygg77 <https://github.com/alygg77/okc-muse-spark-kernels>
# SPDX-License-Identifier: Apache-2.0
# Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `SK2_CONFIGS` / `SK2_PICK` tile configurations, with alygg77's `u32_r96` unpooler entry (alygg77/okc-muse-spark-kernels, kernels/muse_bf16_proj.py).
# Changes: sk3's (layer, m) entries of those tables with the ring cap and a flags word as columns; the pooler m = 8 row changed and baked-pointer flags set on three rows.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
_hh_BASE_CFG = {   # sk3's configs (vjeux/alygg sk2 tables) for the shapes this module may own
    ("qkv_gate", 8): (24, 3, 1, 4, 2, 2, 144, 0),
    ("qkv_gate", 16): (24, 3, 1, 4, 2, 2, 200, 0),
    ("qkv_gate", 32): (24, 3, 1, 4, 2, 1, 208, 0),
    ("pooler", 8): (32, 2, 2, 2, 4, 2, 144, 0),
    ("pooler", 16): (32, 2, 2, 2, 4, 1, 192, 0),
    ("pooler", 32): (32, 2, 2, 2, 2, 1, 144, 0),
    ("unpooler", 16): (32, 2, 2, 2, 2, 1, 72, 4),
    ("unpooler", 32): (32, 2, 2, 2, 2, 1, 96, 4),
    ("o", 8): (32, 4, 1, 4, 1, 2, 88, 0),
    ("o", 16): (32, 2, 2, 1, 4, 2, 68, 4),
    ("o", 32): (32, 2, 2, 2, 2, 1, 100, 0),
}
# SPDX-SnippetEnd
_hh_F12 = 1 << 20
# shipped configs: F12 stream + register-preloaded rare-escape lists (iteration 5+); same-run wins vs bo5 in NOTES.md
_hh_CFG = {k: v[:7] + (v[7] | _hh_F12,) for k, v in _hh_BASE_CFG.items()}
_hh_CFG[("qkv_gate", 32)] = (24, 3, 1, 4, 2, 1, 208, _hh_F12 | 4)
_hh_CFG[("pooler", 16)] = (32, 2, 2, 2, 4, 1, 192, _hh_F12)
_hh_CFG[("unpooler", 32)] = (32, 2, 2, 2, 2, 1, 96, _hh_F12 | 4)
_hh_CFG[("o", 8)] = (32, 4, 1, 4, 1, 2, 88, _hh_F12 | 4)
_hh_CFG[("o", 16)] = (32, 4, 1, 4, 1, 2, 88, _hh_F12)
_hh_CFG[("o", 32)] = (32, 2, 2, 2, 2, 1, 100, _hh_F12 | 4)
# same-run wins vs bo5 on both sandboxes (bench.py, iteration 5); o m8 stays with bo5 (Nathen v18o) and unpooler m32
# with bo5's sk3 (wins standalone by ~0.05, loses by 0.2-0.3 in both integrated runs)
_hh_OWN = {("qkv_gate", 8), ("qkv_gate", 16), ("qkv_gate", 32), ("pooler", 8), ("pooler", 16), ("pooler", 32),
       ("unpooler", 16), ("o", 16), ("o", 32)}
_hh_K_OF = {"pooler": 8192, "unpooler": 4096, "o": 1024, "qkv_gate": 8192}
_hh_N_OF = {"pooler": 4096, "unpooler": 8192, "o": 8192, "qkv_gate": 2304}
_hh_FLAG_BAKE = 4
_hh_NGCAP = 4                   # escape groups (16 escapes of one 8-row tile) gathered up front per stage
_hh_SMEM_LIMIT = 232_448
_hh_FAKE = 120                  # F11 fake exponent bias (agent G)
_hh_FAKE12 = 112                # F12 fake exponent bias (agent G)
_hh_TB = 704
_hh_FLAG_F12 = 1 << 20          # cfg flags bit: 12-bit F12 stream + rare-escape lists (else F11)
_hh_FLAG_ESC1 = 1 << 17         # (H) kernel flag: one escape warp, per-CTA stage-sorted escape list, smem corrections
_hh_FLAG_LR = 1 << 18           # (H) kernel flag: the last warp to release a slot issues its refill (needs ESC1)
_hh_FLAG_HTR = 1 << 19          # (H) kernel flag: per-warp per-stage globaltimer trace of CTA 0 and CTA ncta/2
_hh_FLAG_WTMA = 1 << 29         # (H) kernel flag: W stage blocks loaded as TMA tiles of 1 KB pieces interleaved across CTAs
_hh_ESC1_CAP = 2048             # max escapes per CTA for ESC1 (more -> the shape is left to the base)

# (H) shipped shapes: same-run wins vs bo7b (NOTES.md); everything else -> None (bo7b keeps it)
_hh_H_CFG = {
    ("unpooler", 32): (64, 4, 2, 2, 2, 1, 160, _hh_F12 | _hh_FLAG_ESC1 | _hh_FLAG_BAKE),
}
_hh_CFG.update(_hh_H_CFG)
_hh_OWN = set(_hh_H_CFG)
if os.environ.get("SKH_OWN") is not None:   # experiments: e.g. "pooler:32,unpooler:32" (cfg from CFG)
    _hh_OWN = set()
    for _t in filter(None, os.environ["SKH_OWN"].split(",")):
        _l, _m = _t.split(":")
        _hh_OWN.add((_l, int(_m)))


def _hh__sk(cfg):
    return cfg[8] if len(cfg) > 8 else 1


def _hh__pm(cfg):
    return cfg[9] if len(cfg) > 9 else 0


def _hh__rg(cfg):
    return cfg[10] if len(cfg) > 10 else 0


def _hh__xs(cfg):
    return cfg[11] if len(cfg) > 11 else 1


def _hh__xp(cfg):
    return cfg[12] if len(cfg) > 12 else 0


def _hh__hf(cfg):
    return cfg[13] if len(cfg) > 13 else 0


def _hh__if(cfg):
    return cfg[15] if len(cfg) > 15 else 0


def _hh__xl(cfg):
    return cfg[16] if len(cfg) > 16 else 0


def _hh__rot_of(c, rg, nst):
    return (c % rg) * (nst // rg) if rg > 1 else 0


def _hh__geom(cfg, layer, m, lc=32):
    R, NT, NWR, WSK, UPW, NACC, cap, fl = cfg[:8]
    sk = _hh__sk(cfg)
    tokr = 8 if m <= 8 else 16 if m <= 16 else 32
    mt = 2 if tokr == 32 else 1
    ups = WSK * UPW
    nw = NWR * WSK
    t = UPW * NT
    f12 = bool(fl & _hh_FLAG_F12)
    wby = nw * t * (768 if f12 else _hh_TB)
    xby = ups * tokr * 128
    nst = _hh_K_OF[layer] // (ups * 64) // sk + _hh__hf(cfg)
    toks = 16 * mt
    rp = R + ((40 - (R % 32)) % 32)
    esc1 = bool(fl & _hh_FLAG_ESC1)
    elb = (16 + lc * 8 if esc1 else 16 + (R // 8) * lc * 4) if f12 else nst * 16
    ne = 0 if fl & 128 else 1 if esc1 else R // 8
    nthr = (nw + (0 if fl & _hh_FLAG_LR else 1) + ne + _hh__xl(cfg)) * 32
    return dict(tokr=tokr, mt=mt, ups=ups, nw=nw, t=t, wby=wby, xby=xby, nst=nst, toks=toks, rp=rp,
                nstp=nst, ne=ne, elb=elb, f12=f12, lc=lc, esc1=esc1, nthr=nthr, sk=sk,
                xtra=16 + ((sk - 1) * toks * (R // sk) * 4 if sk > 1 else 0))


def _hh__nstage(cfg, g, ebc):
    sst = (g["xby"] + g["wby"] + ebc + 1023) // 1024 * 1024
    extra = 1024 + 16 * 16 + 16 + g["elb"] + 2 * g["toks"] * g["rp"] * 4 + 64 + g["xtra"]
    ring = min(cfg[6] * 1024, _hh_SMEM_LIMIT - extra)
    return max(2, min(16, g["nst"], ring // sst)), sst


def _hh__smem(g, nstage, sst):
    return 1024 + nstage * sst + 16 * nstage + 16 + g["elb"] + 2 * g["toks"] * g["rp"] * 4 + g["xtra"]


# ------------------------------------------------------------------------------------------------- weight packing
def _hh__window(e, width):
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    return max(range(1, 256 - width), key=lambda q: int(cs[q + width] - cs[q]))


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Copied from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_tileunits` (via alygg77/okc-muse-spark-kernels).
# Changes: renamed; docstring shortened.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def _hh__tileunits(w):
    """W [N, K] -> [N/8, K/64, 32 lanes, 16] in mma.m16n8k16 B-fragment order (sk2_tileunits)."""
    n, k = w.shape
    t = w.view(n // 8, 8, k // 64, 4, 2, 4, 2)
    return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)
# SPDX-SnippetEnd


def _hh_pack_code32(s, h):
    """agent G's F11 code words: s, h [..., 32] (weight 4g + b = byte lane b of hi word g) -> int32 [..., 3]"""
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


def _hh_compress11(w):
    """F11 planes in tile-unit order [N/8, K/64, 32, 16]: lo, s, h, esc mask, true bits u, elo."""
    tu = _hh__tileunits(w.contiguous())
    u = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    elo = _hh__window(e, 8)
    c = e - elo
    esc = (c < 0) | (c > 7)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m), "s": torch.where(esc, torch.zeros_like(s), s),
            "h": c >> 1, "esc": esc, "u": u, "elo": elo}


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_stream` (via alygg77/okc-muse-spark-kernels).
# Changes: gathers our 11-bit planes per tile unit (low bytes, plus sign / exponent codes packed three words per 32 weights) instead of sk2's pre-packed 768-byte tile units; the stage / CTA / warp / unit index arithmetic is sk2's.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def _hh__wstream(f, n_rows, R, NT, NWR, WSK, UPW):
    """[stage][cta][warp][T tile-units lo (512 B each) | T/2 code groups (W0W1 lane x 8 | W2 lane x 4)]"""
    dev = f["lo"].device
    U = f["lo"].shape[1]
    ups = WSK * UPW
    nw = NWR * WSK
    T = UPW * NT
    S, C = U // ups, n_rows // R
    ar = lambda v: torch.arange(v, device=dev)
    s = ar(S).view(-1, 1, 1, 1, 1)
    c = ar(C).view(1, -1, 1, 1, 1)
    w = ar(nw).view(1, 1, -1, 1, 1)
    uu = ar(UPW).view(1, 1, 1, -1, 1)
    nt = ar(NT).view(1, 1, 1, 1, -1)
    tile = (c * (R // 8) + (w // WSK) * NT + nt).expand(S, C, nw, UPW, NT)
    unit = (s * ups + (w % WSK) * UPW + uu).expand(S, C, nw, UPW, NT)
    lo = f["lo"][tile, unit].reshape(S, C, nw, T, 32, 16).to(torch.uint8)
    sg = f["s"][tile, unit].reshape(S, C, nw, T // 2, 2, 32, 16).permute(0, 1, 2, 3, 5, 4, 6).reshape(S, C, nw, T // 2, 32, 32)
    hg = f["h"][tile, unit].reshape(S, C, nw, T // 2, 2, 32, 16).permute(0, 1, 2, 3, 5, 4, 6).reshape(S, C, nw, T // 2, 32, 32)
    words = _hh_pack_code32(sg, hg)                                            # [S, C, nw, T/2, 32, 3]
    w01 = words[..., 0:2].contiguous().view(torch.uint8).reshape(S, C, nw, T // 2, 256)
    w2 = words[..., 2].contiguous().view(torch.uint8).reshape(S, C, nw, T // 2, 128)
    codes = torch.cat([w01, w2], -1).reshape(S, C, nw, T // 2 * 384)
    blob = torch.cat([lo.reshape(S, C, nw, T * 512), codes], -1)
    return blob.contiguous().view(-1)
# SPDX-SnippetEnd


def _hh__fake_bits(u, elo):
    """true bf16 bits (int32) -> fake bf16 bits (true * 2^(FAKE - elo)), or None if not exactly representable."""
    t = (u.to(torch.int32) << 16).view(torch.float32)
    f = t * (2.0 ** (_hh_FAKE - elo))
    fb = f.to(torch.bfloat16)
    ok = torch.isfinite(f) & (fb.float() == f) & ((fb.float() * (2.0 ** (elo - _hh_FAKE))) == t)
    if not bool(ok.all()):
        return None
    return fb.view(torch.int16).to(torch.int32) & 0xFFFF


def _hh__estream(f, n_rows, R, UPS, nstage):
    """Escape blocks per (stage, cta): per 8-row tile a list of 4-byte entries kl | row << 9 | fake << 16, padded to a
    multiple of 4 entries (lane t of the escape warp reads entries 16G + 4t .. +3 as one 16-B load = slots 2t, 2t+1,
    2t+8, 2t+9 of group G); within a tile, groups of 4 consecutive entries come from one bank bucket (kl >> 1) & 3 so
    the 4 lanes t of one gather read 4 different buckets (conflict-free). Blocks for stage < nstage have a fixed
    capacity CIB at (s * C + c) * CIB; later blocks follow. Per-CTA table etab[c][s] = (offset / 16, bytes,
    cnt0 | cnt1 << 16, cnt2 | cnt3 << 16)."""
    dev = f["u"].device
    idx = f["esc"].nonzero()
    tile, unit, lane, i = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    row = tile * 8 + (lane >> 2)
    k = unit * 64 + 16 * (i >> 2) + 8 * ((i >> 1) & 1) + 2 * (lane & 3) + (i & 1)
    fb = _hh__fake_bits(f["u"][idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]], f["elo"])
    if fb is None:
        return None
    K = f["u"].shape[1] * 64
    S, C, NE = K // (UPS * 64), n_rows // R, R // 8
    cta, et, rit = row // R, (row % R) // 8, row % 8
    s, kl = k // (UPS * 64), k % (UPS * 64)
    ent = (kl | (rit << 9)).to(torch.int64) | (fb.to(torch.int64) << 16)
    bk = (s * C + cta) * NE + et                               # (stage, cta, tile) key
    b = (kl >> 1) & 3
    o1 = torch.argsort(((bk * 4 + b) << 12) | kl)
    bk, b, ent = bk[o1], b[o1], ent[o1]
    gk = bk * 4 + b
    gcnt = torch.bincount(gk, minlength=S * C * NE * 4)
    gst = torch.cumsum(gcnt, 0) - gcnt
    rank = torch.arange(gk.numel(), device=dev) - gst[gk]
    key = (rank // 4) * 16 + b * 4 + (rank % 4)
    o2 = torch.argsort((bk << 32) | key)
    bk, ent = bk[o2], ent[o2]
    cnt = torch.bincount(bk, minlength=S * C * NE).view(S, C, NE)
    if int(cnt.max()) > 65535 or NE > 4:
        return None
    pcnt = (cnt + 3) // 4 * 4                                   # padded per-tile lists
    bbytes = 4 * pcnt.sum(-1)                                   # [S, C], multiple of 16
    ns0 = min(nstage, S)
    cib = int(bbytes[:ns0].max())
    off = torch.zeros(S, C, dtype=torch.long, device=dev)
    off[:ns0] = (torch.arange(ns0, device=dev).view(-1, 1) * C + torch.arange(C, device=dev).view(1, -1)) * cib
    base = ns0 * C * cib
    if S > ns0:
        vb = bbytes[ns0:].reshape(-1)
        off[ns0:] = (base + torch.cumsum(vb, 0) - vb).view(S - ns0, C)
        total = base + int(vb.sum())
    else:
        total = base
    ebc = max(cib, int(bbytes.max()), 16)
    buf = torch.full((max(total, 16) // 4,), 8 << 9, dtype=torch.int64, device=dev)   # padding entries: row 8
    tst = torch.cumsum(cnt.view(-1), 0) - cnt.view(-1)          # first sorted index of each (s, c, et)
    ppre = torch.cumsum(pcnt, -1) - pcnt                         # padded entries before tile et in the block
    pos = (off.view(-1, 1) // 4 + ppre.view(S * C, NE)).view(-1)  # word offset of each (s, c, et) list
    j = torch.arange(bk.numel(), device=dev) - tst[bk]
    buf[pos[bk] + j] = ent
    buf = torch.where(buf >= 2 ** 31, buf - 2 ** 32, buf).to(torch.int32)
    c4 = torch.zeros(S, C, 4, dtype=torch.int64, device=dev)
    c4[..., :NE] = cnt
    etab = torch.zeros(C, S, 4, dtype=torch.int64, device=dev)
    etab[..., 0] = (off // 16).t()
    etab[..., 1] = bbytes.t()
    etab[..., 2] = (c4[..., 0] | (c4[..., 1] << 16)).t()
    etab[..., 3] = (c4[..., 2] | (c4[..., 3] << 16)).t()
    etab = torch.where(etab >= 2 ** 31, etab - 2 ** 32, etab).to(torch.int32)
    cm = cnt.view(-1).float()
    return {"buf": buf.contiguous(), "etab": etab.contiguous(), "cib": cib, "ebc": ebc, "n": int(cnt.sum()),
            "cmax": int(cnt.max()), "cq": float(torch.quantile(cm[:min(cm.numel(), 1 << 24)], 0.999))}


def _hh_compress12(w):
    """F12 planes in tile-unit order [N/8, K/64, 32, 16] (agent G's fake-exponent 12-bit: 16-exponent window)."""
    tu = _hh__tileunits(w.contiguous())
    u = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    elo = _hh__window(e, 16)
    c = e - elo
    esc = (c < 0) | (c > 15)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m),
            "nib": torch.where(esc, torch.zeros_like(u), (s << 3) | (c >> 1)), "esc": esc, "u": u, "elo": elo}


def _hh_pack_nib8(nib):
    """agent G's F12 nibble word: nib [..., 8] (s << 3 | h) -> int32: byte b = h_b | s_b << 7 | h_{4+b} << 4 | s_{4+(b+3)%4} << 3"""
    nib = nib.to(torch.int64)
    s, h = nib >> 3, nib & 7
    word = torch.zeros(nib.shape[:-1], dtype=torch.int64, device=nib.device)
    for b in range(4):
        word |= (h[..., b] | (s[..., b] << 7) | (h[..., 4 + b] << 4) | (s[..., 4 + (b + 3) % 4] << 3)) << (8 * b)
    return torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)


def _hh__wstream12(f, n_rows, R, NT, NWR, WSK, UPW, SK=1, HF=0):
    if HF:   # (H) NSTN normal stages, then 2 HF half stages (UPW / 2 units per warp)
        U = f["lo"].shape[1]
        ups = WSK * UPW
        nstn = U // ups - HF
        b0 = _hh__wstream12_1(f, n_rows, R, NT, NWR, WSK, UPW).view(U // ups, -1)[:nstn].reshape(-1)
        u0s = [nstn * ups + h * (ups // 2) for h in range(2 * HF)]
        b1 = _hh__wstream12_1(f, n_rows, R, NT, NWR, WSK, UPW // 2, u0s)
        return torch.cat([b0, b1]).contiguous()
    b = _hh__wstream12_1(f, n_rows, R, NT, NWR, WSK, UPW)
    if SK == 1:
        return b
    U = f["lo"].shape[1]
    S, C = U // (WSK * UPW), n_rows // R
    # (H) split-K: CTA rb * SK + kr streams row block rb at global stage kr * S / SK + ls; layout [ls][rb][kr][...]
    return b.view(SK, S // SK, C, -1).permute(1, 2, 0, 3).contiguous().view(-1)


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_stream` (via alygg77/okc-muse-spark-kernels).
# Changes: gathers our separate low-byte and 4-bit-code planes per tile unit (codes repacked into nibble words) and concatenates them, instead of indexing sk2's pre-packed 768-byte tile units; the stage / CTA / warp / unit index arithmetic is sk2's, and optionally starts each stage at an explicit unit (half stages).
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def _hh__wstream12_1(f, n_rows, R, NT, NWR, WSK, UPW, u0s=None):
    """[stage][cta][warp][T tile-units: 512 lo bytes (lane x 16) | 256 B nibble words (lane x 8)]
    u0s (H): explicit first unit per stage (half stages), else stage s starts at unit s * WSK * UPW."""
    dev = f["lo"].device
    U = f["lo"].shape[1]
    ups, nw, T = WSK * UPW, NWR * WSK, UPW * NT
    S, C = (U // ups if u0s is None else len(u0s)), n_rows // R
    ar = lambda v: torch.arange(v, device=dev)
    s = ar(S).view(-1, 1, 1, 1, 1)
    if u0s is not None:
        s = torch.tensor(u0s, device=dev).view(-1, 1, 1, 1, 1)
        ups = 1
    c = ar(C).view(1, -1, 1, 1, 1)
    w = ar(nw).view(1, 1, -1, 1, 1)
    uu = ar(UPW).view(1, 1, 1, -1, 1)
    nt = ar(NT).view(1, 1, 1, 1, -1)
    tile = (c * (R // 8) + (w // WSK) * NT + nt).expand(S, C, nw, UPW, NT)
    unit = (s * ups + (w % WSK) * UPW + uu).expand(S, C, nw, UPW, NT)
    lo = f["lo"][tile, unit].reshape(S, C, nw, T, 512).to(torch.uint8)
    nw8 = _hh_pack_nib8(f["nib"][tile, unit].reshape(S, C, nw, T, 32, 2, 8))          # [S, C, nw, T, 32, 2]
    nb = nw8.contiguous().view(torch.uint8).reshape(S, C, nw, T, 256)
    return torch.cat([lo, nb], -1).contiguous().view(-1)
# SPDX-SnippetEnd


def _hh__elist12(f, n_rows, R, UPS):
    """Per CTA: [4 x u32 per-tile counts][NE tiles x LC u32 entries k | row << 13 | fake_bits << 16] (LC = 32 or 64)."""
    dev = f["u"].device
    idx = f["esc"].nonzero()
    tile, unit, lane, i = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    row = tile * 8 + (lane >> 2)
    k = unit * 64 + 16 * (i >> 2) + 8 * ((i >> 1) & 1) + 2 * (lane & 3) + (i & 1)
    t = (f["u"][idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]].to(torch.int32) << 16).view(torch.float32)
    fv = t * (2.0 ** (_hh_FAKE12 - f["elo"]))
    fb16 = fv.to(torch.bfloat16)
    if not bool((torch.isfinite(fv) & (fb16.float() == fv) & (fb16.float() * (2.0 ** (f["elo"] - _hh_FAKE12)) == t)).all()):
        return None
    fb = fb16.view(torch.int16).to(torch.int64) & 0xFFFF
    C, NE = n_rows // R, R // 8
    cta, et, rit = row // R, (row % R) // 8, row % 8
    key = (cta * NE + et).long()
    cnt = torch.bincount(key, minlength=C * NE)
    cmax = int(cnt.max()) if cnt.numel() else 0
    if cmax > 64 or k.numel() and int(k.max()) >= 8192:
        return None
    lc = 32 if cmax <= 32 else 64
    o = torch.argsort(key * 8192 + k)
    key, ent = key[o], (k | (rit << 13)).long()[o] | (fb[o] << 16)
    st0 = torch.cumsum(cnt, 0) - cnt
    j = torch.arange(key.numel(), device=dev) - st0[key]
    buf = torch.zeros(C, 4 + NE * lc, dtype=torch.int64, device=dev)
    buf[:, :NE] = cnt.view(C, NE)
    buf.view(-1)[(key // NE) * (4 + NE * lc) + 4 + (key % NE) * lc + j] = ent
    buf = torch.where(buf >= 2 ** 31, buf - 2 ** 32, buf).to(torch.int32)
    return {"buf": buf.contiguous(), "lc": lc, "n": int(cnt.sum()), "cmax": cmax}


def _hh__elist1(f, n_rows, R, UPS, SK=1, RG=0, HF=0):
    """(H) ESC1 list per CTA: [count, 0, 0, 0] then uint2 entries sorted by stage:
    x = kl | row << 9 | stage << 16 (kl = k within the stage, row within the CTA), y = fake bf16 bits << 16."""
    dev = f["u"].device
    idx = f["esc"].nonzero()
    tile, unit, lane, i = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    row = tile * 8 + (lane >> 2)
    k = unit * 64 + 16 * (i >> 2) + 8 * ((i >> 1) & 1) + 2 * (lane & 3) + (i & 1)
    t = (f["u"][idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]].to(torch.int32) << 16).view(torch.float32)
    fv = t * (2.0 ** (_hh_FAKE12 - f["elo"]))
    fb16 = fv.to(torch.bfloat16)
    if not bool((torch.isfinite(fv) & (fb16.float() == fv) & (fb16.float() * (2.0 ** (f["elo"] - _hh_FAKE12)) == t)).all()):
        return None
    fb = fb16.view(torch.int16).to(torch.int64) & 0xFFFF
    nstk = f["u"].shape[1] // UPS // SK
    C = n_rows // R * SK
    s_, kl = (k // (UPS * 64)).long(), (k % (UPS * 64)).long()
    cta, rl = (row // R).long() * SK + s_ // nstk, (row % R).long()
    s_ = s_ % nstk
    if RG > 1:   # time index t with s = (t + rot) % nstk
        s_ = (s_ - (cta % RG) * (nstk // RG)) % nstk
    if HF:   # (H) half stages: units >= nstn * UPS go to stage nstn + (u - nstn * UPS) // (UPS / 2)
        nstn = nstk - HF
        u = (k // 64).long()
        hs = (u - nstn * UPS).clamp(min=0) // (UPS // 2)
        inh = u >= nstn * UPS
        s_ = torch.where(inh, nstn + hs, s_)
        kl = torch.where(inh, k.long() - (nstn * UPS + hs * (UPS // 2)) * 64, kl)
    if R > 128 or (s_.numel() and int(s_.max()) >= 0xFFFF):
        return None
    cnt = torch.bincount(cta, minlength=C)
    cmax = int(cnt.max()) if cnt.numel() else 0
    if cmax > _hh_ESC1_CAP:
        return None
    cap = max(2, (cmax + 1) // 2 * 2)
    o = torch.argsort(((cta * 65536 + s_) * 128 + rl) * 512 + kl)
    cta, ex, ey = cta[o], (kl | (rl << 9) | (s_ << 16))[o], (fb << 16)[o]
    st0 = torch.cumsum(cnt, 0) - cnt
    j = torch.arange(cta.numel(), device=dev) - st0[cta]
    buf = torch.zeros(C, 2 + cap, 2, dtype=torch.int64, device=dev)
    buf[:, 0, 0] = cnt
    buf[cta, 2 + j, 0] = ex
    buf[cta, 2 + j, 1] = ey
    buf = torch.where(buf >= 2 ** 31, buf - 2 ** 32, buf).to(torch.int32)
    return {"buf": buf.contiguous(), "lc": cap, "n": int(cnt.sum()), "cmax": cmax}


# ------------------------------------------------------------------------------------------------- NVRTC module
class _hh__Module:
    def __init__(self, src, name="skh.cu", opts=()):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        o = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        o += [x.encode() for x in opts]
        if os.environ.get("CU_LINEINFO"):
            o.append(b"-lineinfo")
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(o), o)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        self.log = log.decode(errors="replace").strip("\x00 \n")
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + self.log)
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        self.cubin = cubin
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err
        self.funcs = {}

    def func(self, name, smem):
        key = (name, smem)
        f = self.funcs.get(key)
        if f is None:
            d = self.driver
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            assert int(err) == 0, (name, err)
            if smem > 48 * 1024:
                (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
                assert int(err) == 0, (name, smem, err)
            self.funcs[key] = f
        return f

    def attrs(self, name, smem=0):
        d = self.driver
        f = self.func(name, smem)
        out = {}
        for k in ("NUM_REGS", "LOCAL_SIZE_BYTES"):
            err, v = d.cuFuncGetAttribute(getattr(d.CUfunction_attribute, "CU_FUNC_ATTRIBUTE_" + k), f)
            out[k] = v
        return out

    def launch(self, f, grid, block, smem, params):
        d = self.driver
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        assert int(err) == 0, err


def _hh__tmap_w3d(ptr, ncta, nrows, nch):
    """(H) W stream [stage * nch + j][cta][1 KB] as uint32 [nrows][ncta][256], box (256, 1, nch)."""
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT32, d.cuuint32_t(3), ptr,
        [d.cuuint64_t(256), d.cuuint64_t(ncta), d.cuuint64_t(nrows)], [d.cuuint64_t(1024), d.cuuint64_t(1024 * ncta)],
        [d.cuuint32_t(256), d.cuuint32_t(1), d.cuuint32_t(nch)],
        [d.cuuint32_t(1), d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


def _hh__tmap_x3d(ptr, m, k, box_rows, units):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(3), ptr,
        [d.cuuint64_t(64), d.cuuint64_t(m), d.cuuint64_t(k // 64)], [d.cuuint64_t(k * 2), d.cuuint64_t(128)],
        [d.cuuint32_t(64), d.cuuint32_t(box_rows), d.cuuint32_t(units)],
        [d.cuuint32_t(1), d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, getattr(d.CUtensorMapL2promotion, "CU_TENSOR_MAP_L2_PROMOTION_" + _hh_XPROMO),
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


_hh_XPROMO = os.environ.get("SKH_XPROMO", "L2_256B")   # (H) experiment: NONE / L2_64B / L2_128B / L2_256B
_hh__TM = {}


def _hh__xmap(x, m, k, box_rows, units):
    key = (x.data_ptr(), m, k, box_rows, units)
    b = _hh__TM.get(key)
    if b is None:
        if len(_hh__TM) > 512:
            _hh__TM.clear()
        b = _hh__TM[key] = _hh__tmap_x3d(x.data_ptr(), m, k, box_rows, units)
    return b


# ------------------------------------------------------------------------------------------------- prepare / run
def _hh_prepare_skh(weights, cfgs=None, verbose=False, variants=None, selftest=True):
    """cfgs: {(layer, m): cfg} (default: CFG restricted to OWN). variants: extra {(layer, m, tag): cfg} instances.
    Never raises: any failure (packing, NVRTC, smem, self-test) leaves the shape to the base (run_sk11 -> None)."""
    try:
        st = _hh__prepare_impl(weights, cfgs, verbose, variants)
    except Exception as ex:   # noqa: BLE001
        print(f"[skh] prepare failed, base keeps every shape: {type(ex).__name__}: {ex}", flush=True)
        return {"ent": {}}
    if selftest:
        for (layer, m, tag), e in st["ent"].items():
            if not e["ok"]:
                continue
            try:
                g = torch.Generator(device="cuda").manual_seed(1234 + m)
                x = torch.randn((m, e["k"]), generator=g, device="cuda").to(torch.bfloat16)
                out = _hh_run_skh(x, {"m": m, "n": e["n"], "k": e["k"], "layer": layer}, st, tag=tag or "__self")
                ref = x.float() @ weights[layer][: e["n"]].float().T
                err = float((out.float() - ref).norm() / ref.norm())
                e["ok"] = err < 5e-3
                if not e["ok"]:
                    print(f"[skh] self-test failed for {(layer, m, tag)}: rel err {err:.3e}", flush=True)
            except Exception as ex:   # noqa: BLE001
                e["ok"] = False
                print(f"[skh] self-test error for {(layer, m, tag)}: {ex}", flush=True)
        torch.cuda.synchronize()
    return st


def _hh__prepare_impl(weights, cfgs=None, verbose=False, variants=None):
    cfgs = dict({k: v for k, v in _hh_CFG.items() if k in _hh_OWN} if cfgs is None else cfgs)
    inst = [(layer, m, "", cfg) for (layer, m), cfg in cfgs.items()]
    if variants:
        inst += [(layer, m, tag, cfg) for (layer, m, tag), cfg in variants.items()]
    st = {"ent": {}}
    if any(c[-1] & (2 | 4096 | 16384 | _hh_FLAG_HTR) for c in list(cfgs.values()) + list((variants or {}).values())):
        st["trc_buf"] = torch.zeros(65536, dtype=torch.int64, device="cuda")
        st["trc"] = st["trc_buf"].data_ptr()
    comps, wss, ess = {}, {}, {}
    for layer, m, tag, cfg in inst:
        R, NT, NWR, WSK, UPW, NACC, cap, fl = cfg[:8]
        SK = _hh__sk(cfg)
        assert R == NWR * NT * 8 and m in (8, 16, 32)
        n_rows = _hh_N_OF[layer]
        f12 = bool(fl & _hh_FLAG_F12)
        ckey = (layer, f12)
        if ckey not in comps:
            comps[ckey] = (_hh_compress12 if f12 else _hh_compress11)(weights[layer][:n_rows])
        f = comps[ckey]
        esc1 = bool(fl & _hh_FLAG_ESC1)
        if (R // 8 > 4 and not esc1) or WSK * UPW * 64 > 512 or n_rows % R or (not f12 and (UPW * NT) % 2):
            continue
        if SK > 1 and (not esc1 or (_hh_K_OF[layer] // (WSK * UPW * 64)) % SK or (R // SK) % 4):
            continue
        RG = _hh__rg(cfg)
        if RG < 0:
            RG = 0   # ablation (x always from the first -RG stages): W stream / escape list unchanged
        if RG > 1 and (SK > 1 or not esc1 or bool(fl & _hh_FLAG_WTMA)):
            continue
        wt = bool(fl & _hh_FLAG_WTMA)
        if wt and (not f12 or (NWR * WSK * UPW * NT * 768) % 1024 or (NWR * WSK * UPW * NT * 768) // 1024 > 256):
            continue
        HF = _hh__hf(cfg)
        if HF and (SK > 1 or RG or wt or not esc1 or UPW % 2 or _hh__xp(cfg) or _hh__xs(cfg) > 1):
            continue
        wkey = (layer, f12, R, NT, NWR, WSK, UPW, SK, wt, RG, HF, cfg[14] if len(cfg) > 14 else 0)   # cfg[14]: clone id (separate W blob)
        if wkey not in wss:
            b = _hh__wstream12(f, n_rows, R, NT, NWR, WSK, UPW, SK, HF) if f12 else _hh__wstream(f, n_rows, R, NT, NWR, WSK, UPW)
            if RG > 1:   # [s][c] -> [t][c] with s = (t + rot_c) % NST
                nc_, nst_ = n_rows // R, _hh_K_OF[layer] // (WSK * UPW * 64)
                bv = b.view(nst_, nc_, -1)
                tt = torch.arange(nst_, device=b.device).view(-1, 1)
                cc = torch.arange(nc_, device=b.device).view(1, -1)
                rot = (cc % RG) * (nst_ // RG)
                b = bv[(tt + rot) % nst_, cc.expand(nst_, -1)].contiguous().view(-1)
            if wt:   # [stage][cta][nch][1 KB] -> [stage][nch][cta][1 KB]
                nc_ = n_rows // R * SK
                nch = NWR * WSK * UPW * NT * 768 // 1024
                b = b.view(-1, nc_, nch, 1024).permute(0, 2, 1, 3).contiguous().view(-1)
            wss[wkey] = b
        ws = wss[wkey]
        if f12:
            ekey = (layer, True, R, WSK * UPW, esc1, SK, RG, HF)
            if ekey not in ess:
                ess[ekey] = _hh__elist1(f, n_rows, R, WSK * UPW, SK, RG, HF) if esc1 else _hh__elist12(f, n_rows, R, WSK * UPW)
            es = ess[ekey]
            if es is None:
                print(f"[skh] {(layer, m, tag)}: F12 escapes do not fit (count / fake domain): left to the base", flush=True)
                continue
            g = _hh__geom(cfg, layer, m, es["lc"])
            es = dict(es, ebc=0, cib=0, etab=es["buf"], cq=0.0)
            nstage, sst = _hh__nstage(cfg, g, 0)
        else:
            g = _hh__geom(cfg, layer, m)
            # escape capacity depends on nstage and nstage on the capacity: iterate once
            nstage, sst = _hh__nstage(cfg, g, 1024)
            ekey = (layer, False, R, g["ups"], nstage)
            if ekey not in ess:
                ess[ekey] = _hh__estream(f, n_rows, R, g["ups"], nstage)
            es = ess[ekey]
            if es is None:
                print(f"[skh] {(layer, m, tag)}: escapes not representable in the fake domain: left to the base", flush=True)
                continue
            nst2, sst = _hh__nstage(cfg, g, es["ebc"])
            if nst2 != nstage:
                nstage = nst2
                ekey = (layer, False, R, g["ups"], nstage)
                if ekey not in ess:
                    ess[ekey] = _hh__estream(f, n_rows, R, g["ups"], nstage)
                es = ess[ekey]
                nst3, sst = _hh__nstage(cfg, g, es["ebc"])
                if nst3 != nstage:
                    continue
        smem = _hh__smem(g, nstage, sst)
        if smem > _hh_SMEM_LIMIT:
            print(f"[skh] {(layer, m, tag)} needs {smem} B smem: left to the base", flush=True)
            continue
        nthr = g["nthr"]
        ng = 1 if f12 else (int(os.environ.get("SK11_NG", "0")) or min(_hh_NGCAP, max(1, math.ceil(es["cq"] / 16))))
        ncta = n_rows // R * SK
        PM = _hh__pm(cfg)
        if PM and (ncta % PM or ncta // PM * 16 > 132 or SK > 1):
            continue
        grid = ncta // PM * 16 if PM else ncta
        kn = f"skh_{layer}_{m}{tag}"
        wmap = bytes(128)
        if wt:
            nch = g["wby"] // 1024
            wmap = bytes(128) if os.environ.get("SKH_NOTMAP") else _hh__tmap_w3d(ws.data_ptr(), ncta, g["nst"] * nch, nch)
        st["ent"][(layer, m, tag)] = dict(cfg=cfg, ws=ws, es=es, elo=f["elo"], n=n_rows, k=_hh_K_OF[layer], g=g, nthr=nthr, wmap=wmap, grid=grid,
                                          ncta=ncta, smem=smem, kn=kn, nstage=nstage, sst=sst, ok=True, ng=ng,
                                          scale=2.0 ** (f["elo"] - (_hh_FAKE12 if f12 else _hh_FAKE)))
    src = _hh_SRC
    for (layer, m, tag), e in st["ent"].items():
        R, NT, NWR, WSK, UPW, NACC, cap, fl = e["cfg"][:8]
        g = e["g"]
        wb = f"{e['ws'].data_ptr()}ull" if fl & _hh_FLAG_BAKE else "0ull"
        cl_attr = f"__cluster_dims__({g['sk']}, 1, 1) " if g["sk"] > 1 else ""
        src += (f'extern "C" __global__ void {cl_attr}__launch_bounds__({e["nthr"]}, 1) {e["kn"]}(const __grid_constant__ P p) '
                f'{{ sk11_body<{g["mt"]}, {g["tokr"]}, {NT}, {NWR}, {WSK}, {UPW}, {e["nstage"]}, {NACC}, {g["nst"]}, '
                f'{wb}, {e["es"]["ebc"]}, {e["es"]["cib"]}, {e["ng"]}, {12 if g["f12"] else 11}, {g["lc"]}, '
                f'{(fl & ~(_hh_FLAG_BAKE | _hh_FLAG_F12)) | ((1 << 20) if _hh__xl(e["cfg"]) else 0)}, {g["sk"]}, {_hh__pm(e["cfg"])}, {_hh__rg(e["cfg"])}, {_hh__xs(e["cfg"])}, {_hh__xp(e["cfg"])}, {_hh__hf(e["cfg"])}, {_hh__if(e["cfg"])}, {_hh__xl(e["cfg"])}>(p); }}\n')
    if os.environ.get("SKH_RTZERO"):
        src = "#define SKH_RTZERO 1\n" + src
    st["src"] = src
    st["mod"] = _hh__Module(src)
    for key, e in st["ent"].items():
        e["f"] = st["mod"].func(e["kn"], e["smem"])
        if verbose:
            a = st["mod"].attrs(e["kn"], e["smem"])
            g = e["g"]
            wbits = 8.0 * (e["ws"].numel() + e["es"]["buf"].numel() * 4) / (e["n"] * e["k"])
            if e["g"]["f12"]:
                wbits = 8.0 * e["ws"].numel() / (e["n"] * e["k"])
            print(f"[skh] {key} cfg {e['cfg']} nstage {e['nstage']} sst {e['sst']} smem {e['smem']} thr {e['nthr']} "
                  f"ncta {e['ncta']} regs {a['NUM_REGS']} local {a['LOCAL_SIZE_BYTES']} esc {e['es']['n']} "
                  f"cib {e['es']['cib']} ebc {e['es']['ebc']} cmax {e['es']['cmax']} ng {e['ng']} lc {e['g']['lc']} bits/w {wbits:.3f}", flush=True)
    torch.cuda.synchronize()
    return st


def _hh_run_skh(x, config, state, tag=""):
    m, layer = config["m"], config["layer"]
    if (layer, m) not in _hh_OWN and not tag:
        return None
    e = state["ent"].get((layer, m, "" if tag == "__self" else tag))
    if e is None or not e["ok"] or config["n"] != e["n"] or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    n, k = e["n"], e["k"]
    g = e["g"]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    tm = _hh__xmap(x, m, k, g["tokr"], g["ups"] // _hh__xs(e["cfg"]))
    params = tm + struct.pack("<QQQQQQfiiiii", e["ws"].data_ptr(), x.data_ptr(), out.data_ptr(), e["es"]["buf"].data_ptr(),
                              e["es"]["etab"].data_ptr(), state.get("trc", 0), e["scale"], m, n, k, 0, 0) + bytes(56) +\
        (_hh__xmap(x, m, k, g["tokr"], g["ups"] // 2) if _hh__hf(e["cfg"]) else e["wmap"])
    state["mod"].launch(e["f"], e["grid"], e["nthr"], e["smem"], params)
    return out


# ======================================================================== module V2: V2/qkvb.py
"""qkvb (agent V2): agent F's sk11 ring kernel for qkv_gate W[2304, 8192] with V2's x-lag producer.

    state = prepare_qkvb(weights)           # never raises (any failure -> run_qkvb returns None for that shape)
    out   = run_qkvb(x, config, state)      # -> [M, N] bf16, or None where the base (bo10) keeps the shape

Derived from agent F's sk11 (claude/t3/F/sk11.py; itself derived from vjeux/okc-2026 `sk2` via alygg77's public OKC
submission, Apache-2.0, and agent X's NVRTC port sk3), F12 path only (agent G's lossless 12-bit fake-exponent format,
escapes in per-(CTA, 8-row tile) register lists). Generated from claude/t3/V2/skv.py by mkqkvb.py.
Shipped (CFG): qkv_gate m = 8 and m = 16 with F's geometry (R = 24, 96 CTAs, 5-stage ring) , XLAG = 1 (the producer
issues the x tile of stage s together with the W block of stage s + 1 (x lags W by one stage in the per-SM TMA queue) and baked
W / escape-list pointers + M / N / scale constants.
Same-run vs bo10's sk11 in claude/t3/V2/NOTES.md. Every other shape returns None (bo10 keeps it). The kernel source
keeps V2's research options (flags) for reference; the shipped configs use only XLAG.
"""
import ctypes
import os
import struct

import torch

_q8_SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) P { TMap xmap; const u8* W; const u16* x; u16* out; const u32* el; u64* trc; float* part; u32* cnt;
                         float scale; int M; int N; int K; int pad0; int pad1; };

template <u64 A> __device__ __forceinline__ const u8* cptr() { const u8* p; asm("mov.u64 %0, %1;" : "=l"(p) : "n"(A)); return p; }
__device__ __forceinline__ u64 gtime() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ u64 ctime() { u64 t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory"); return t; }
__device__ __forceinline__ void ldsm_x4(u32 addr, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 sel) {
  u32 d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel)); return d;
}
__device__ __forceinline__ u32 rotr(u32 x, int s) { return __funnelshift_r(x, x, s); }
// agent G's F12 decode: 8 weights from 2 lo words + 1 nibble word
__device__ __forceinline__ void decode12(u32 L0, u32 L1, u32 nw, u32* r) {
  u32 hA, hB;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(hA) : "r"(nw), "r"(0x87878787u), "n"(0x38383838));
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(hB) : "r"(rotr(nw, 4)), "r"(0x87878787u), "n"(0x38383838));
  r[0] = prmt(L0, hA, 0x5140); r[1] = prmt(L0, hA, 0x7362);
  r[2] = prmt(L1, hB, 0x5140); r[3] = prmt(L1, hB, 0x7362);
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a)); return r; }
__device__ __forceinline__ void mbar_init(u32 bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `mbar_test` / `mbar_wait` / `mbar_wait_sleep` (MBAR_HELPERS) (via alygg77/okc-muse-spark-kernels).
// Changes: renamed types (u32); mbar_wait written on one line; vjeux's esc_delta, which sits between mbar_wait and mbar_wait_sleep upstream, left out.
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
__device__ __forceinline__ bool mbar_test(u32 bar, u32 parity) {
  u32 ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(u32 bar, u32 parity) { while (!mbar_test(bar, parity)) {} }
__device__ __forceinline__ void mbar_wait_sleep(u32 bar, u32 parity) {
  u32 ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(0x989680u) : "memory");
  } while (!ok);
}
// SPDX-SnippetEnd
__device__ __forceinline__ void mbar_arrive(u32 bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(u32 bar, u32 bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\n}\n" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_load_3d(u32 dst, const TMap* map, u32 bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ u32 lds16(u32 a) { u16 v; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a)); return (u32)v; }

// one (warp, unit) of a stage: raw W words of NT tile-units (partial last tile: clamped lane) + x A-fragments
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_kernel` consumer loop (shared-memory loads, ldmatrix of x, decode and mma steps) (via alygg77/okc-muse-spark-kernels).
// Changes: split out of sk2's kernel body into a load helper (the 768-byte tile-unit ld.shared loads and the swizzled ldmatrix of the x fragments) and a compute helper, with a partial last tile (PR rows: clamped lane and its own code offset TPN); sk2's decode8 replaced by our decode12, and sk2's uncompressed and hi/lo (router) paths dropped.
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
template <int MT, int TOKR, int NT, int PR, int TPN>
__device__ __forceinline__ void ld_unit(u32 wbu, u32 xt, u32 lane, u32 lp, int ui, const int* trow, int chi,
                                        uint4 (&L)[NT], uint2 (&N)[NT], u32 (&a)[4][MT][4]) {
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const bool part = PR < 8 && nt == NT - 1;
    const u32 ll = part ? lp : lane;
    const u32 tb = wbu + (u32)nt * 768u;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(L[nt].x), "=r"(L[nt].y), "=r"(L[nt].z), "=r"(L[nt].w) : "r"(tb + ll * 16));
    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(N[nt].x), "=r"(N[nt].y) : "r"(tb + (part ? TPN : 512) + ll * 8));
  }
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
      ldsm_x4(xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * TOKR + trow[mt]) & 7)) << 4),
              a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
}
// decode NT raw tile-units and run the unit's HMMAs
template <int MT, int NT, int NACC>
__device__ __forceinline__ void mma_unit(int uu, const uint4 (&L)[NT], const uint2 (&N)[NT], const u32 (&a)[4][MT][4],
                                         float (&acc)[NACC][MT][NT][4]) {
  u32 b[NT][8];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    decode12(L[nt].x, L[nt].y, N[nt].x, &b[nt][0]);
    decode12(L[nt].z, L[nt].w, N[nt].y, &b[nt][4]);
  }
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
        mma_bf16(acc[(uu * 4 + j) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], b[nt][2 * j], b[nt][2 * j + 1]);
}
// SPDX-SnippetEnd

// MT token tiles of 16 (TOKR = x box rows); NT n8 tiles per CTA (the last one holds PR real rows); WSK k warps; UPW
// 64-k units per warp per stage; NSTAGE ring slots; NACC accumulator sets; NST stages per CTA; WB baked W pointer (0 =
// params); LC escape-list capacity per tile; R real rows per CTA; KS CTAs per row block (split-K); FL flags:
//   1 x issued before W, 2 per-CTA trace, 16 consumers skip decode+mma (ablation), 32 spin waits for producer,
//   128 no escape warps (ablation, wrong results), 1<<29 no W copies (ablation), 1<<30 no x copies (ablation),
//   bits 8-9 initial fill depth D (stage st < NSTAGE is issued only after stage st - D landed; 0 = all at once),
//   8192 software-pipelined consumer (needs even UPW), 1024 params read late (old behaviour: p.out / scale / M / N loaded in the epilogue), 2048 no output stores
//   (ablation), 16384 no epilogue (ablation), 32768 split-K: single acq_rel RMW instead of per-thread fences,
//   bits 22-23 XLAG: x copy of stage st issued with the W copy of stage st + XLAG,
//   65536 escape warps read x from global memory at kernel start (not on the ring's empty barriers), 4096 output pointer staged through shared memory at kernel start (ptxas cannot sink it)
// EB / PB / CB baked escape-list / split-K scratch / counter pointers (0 = params); MM, NN, SCB (scale bits) baked
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_kernel` (via alygg77/okc-muse-spark-kernels).
// Changes: kept sk2's structure (swapped operands: the x tile is the A operand, loaded by ldmatrix from a TMA-loaded 128B-swizzled tile; each CTA owns R rows x the whole K; a producer warp refills a ring of stages); replaced its 12-bit weight decode (8-exponent window) with our lossless fake-exponent formats and its escape-warp corrections with our per-CTA escape lists and exact corrections, and changed the ring sizing, the producer's issue order and the constants (baked in by NVRTC).
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
template <int MT, int TOKR, int NT, int PR, int WSK, int UPW, int NSTAGE, int NACC, int NST, u64 WB, int LC, int R, int KS, int FL,
          u64 EB, int MM, int NN, u32 SCB, u64 PB, u64 CB>
__device__ __forceinline__ void skv_body(const P& p) {
  constexpr int NW = WSK;
  constexpr int NE = (FL & 128) ? 0 : NT;
  constexpr int NC = NW * 32;
  constexpr int RV = NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int T = UPW * NT;
  constexpr int TPB = PR == 8 ? 768 : 96 * PR;      // bytes of the last tile-unit (partial tile: lanes < 4 PR only)
  constexpr int TPN = PR == 8 ? 512 : 64 * PR;      // nibble-word offset inside the last tile-unit
  constexpr int TUW = 768 * (NT - 1) + TPB;         // bytes per (warp, unit)
  constexpr int WBY = NW * UPW * TUW;
  constexpr int HDR = NT <= 4 ? 4 : 8;
  constexpr int ELB = 4 * HDR + NT * LC * 4;
  constexpr int LSK = UPS == 1 ? 6 : UPS == 2 ? 7 : UPS == 4 ? 8 : UPS == 8 ? 9 : 10;
  constexpr int XT = TOKR * 128;
  constexpr int XBY = UPS * XT;
  constexpr int XB = (FL & (1 << 30)) ? 0 : XBY;
  constexpr int WBB = (FL & (1 << 29)) ? 0 : WBY;
  constexpr int SST = (XBY + WBY + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = RV + ((40 - (RV % 32)) % 32);
  static_assert(PR >= 1 && PR <= 8, "PR");
  static_assert(R == 8 * (NT - 1) + PR, "R");
  static_assert(R % 2 == 0, "R even");
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const u32 sraw = (u32)__cvta_generic_to_shared(smem_raw);
  const u32 sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const u32 bars = sbase + NSTAGE * SST;
  const u32 full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8;
  const u32 oslot = pbar + 16, etab_s = pbar + 32;
  float* corr = reinterpret_cast<float*>(gbase + (etab_s - sbase) + ELB);
  float* red = reinterpret_cast<float*>(gbase);
  const u64 tstart = (FL & 2) ? gtime() : 0ull;
  const u64 cstart = (FL & 2) ? ctime() : 0ull;
  const u8* Wp = WB ? cptr<WB>() : p.W;
  const u8* Ep = EB ? cptr<EB>() : reinterpret_cast<const u8*>(p.el);
  constexpr int DI = (FL >> 8) & 3;
  constexpr int XLAG = (FL >> 22) & 3;   // x copies issued XLAG stages after their W copy (x lags W in issue order)
  static_assert(XLAG < NSTAGE, "XLAG < NSTAGE");
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = (int)blockIdx.x, ncta = (int)gridDim.x;
  const int ks = KS > 1 ? cta % KS : 0;
  const u8* wsrc = Wp + (size_t)cta * WBY;
  const size_t wstride = (size_t)ncta * WBY;
  const int xu0 = ks * NST * UPS;

  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&p.xmap) : "memory");
    for (int i = 0; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + ((FL & 65536) ? 0 : NE)); }
    mbar_init(pbar, 1);
    mbar_init(cbar, NE > 0 ? NE : 1);
    if constexpr ((FL & 4096) != 0) {
      u16* o = p.out;
      asm volatile("st.shared.u64 [%0], %1;" :: "r"(oslot), "l"(o) : "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_tx(full0, XB + WBB);
    if constexpr ((FL & 1) && XB && XLAG == 0) tma_load_3d(sbase, &p.xmap, full0, 0, 0, xu0);
    if constexpr (WBB > 0) bulk_g2s(sbase + XBY, wsrc, WBY, full0);
    if constexpr (!(FL & 1) && XB && XLAG == 0) tma_load_3d(sbase, &p.xmap, full0, 0, 0, xu0);
    mbar_expect_tx(pbar, ELB);
    bulk_g2s(etab_s, Ep + (size_t)cta * ELB, ELB, pbar);
  }
  __syncthreads();

  if (warp == NW) {   // ---------------- producer
    if (lane == 0) {
      if constexpr (XLAG > 0 && XB > 0) {
        for (int st = 1; st < NST + XLAG; ++st) {
          if (st < NST) {
            const int slot = st % NSTAGE;
            if (st >= NSTAGE) {
              if constexpr ((FL & 32) != 0) mbar_wait(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
              else mbar_wait_sleep(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
            }
            const u32 s0 = sbase + slot * SST;
            mbar_expect_tx(full0 + 8 * slot, XB + WBB);
            if constexpr (WBB > 0) bulk_g2s(s0 + XBY, wsrc + (size_t)st * wstride, WBY, full0 + 8 * slot);
          }
          const int sx = st - XLAG;
          if (sx >= 0 && sx < NST) {
            const int xs = sx % NSTAGE;
            tma_load_3d(sbase + xs * SST, &p.xmap, full0 + 8 * xs, 0, 0, xu0 + sx * UPS);
          }
        }
      } else
      for (int st = 1; st < NST; ++st) {
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) {
          if constexpr ((FL & 32) != 0) mbar_wait(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
          else mbar_wait_sleep(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
        } else if (DI > 0 && st >= DI) {
          mbar_wait(full0 + 8 * (st - DI), 0);
        }
        const u32 s0 = sbase + slot * SST;
        mbar_expect_tx(full0 + 8 * slot, XB + WBB);
        if constexpr ((FL & 1) && XB) tma_load_3d(s0, &p.xmap, full0 + 8 * slot, 0, 0, xu0 + st * UPS);
        if constexpr (WBB > 0) bulk_g2s(s0 + XBY, wsrc + (size_t)st * wstride, WBY, full0 + 8 * slot);
        if constexpr (!(FL & 1) && XB) tma_load_3d(s0, &p.xmap, full0 + 8 * slot, 0, 0, xu0 + st * UPS);
      }
    }
    __syncwarp();
    return;
  }

  if ((FL & 65536) && warp > NW) {   // ---------------- escape warps, global-x mode (off the ring)
    // corrections from x read straight from global memory at kernel start (lane = token), in list order (= the
    // per-stage order of the ring mode), so the results are bit-identical; no per-stage wake-ups / slot holding
    const int et = warp - NW - 1;
    const u32* el = reinterpret_cast<const u32*>(gbase + (etab_s - sbase));
    const u16* xg = p.x;
    mbar_wait_sleep(pbar, 0);
    const int cnt = (int)el[et];
    float E1r[8], E2r[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) { E1r[r] = 0.f; E2r[r] = 0.f; }
    const float nd = -__uint_as_float(0x38000000u);
    const int tok = lane;
    const int kb = ks * NST * UPS * 64;
    for (int i0 = 0; i0 < cnt; i0 += 16) {
      u32 ev[16];
      float xv[16];
#pragma unroll
      for (int q = 0; q < 16; ++q) {
        ev[q] = (i0 + q < cnt) ? el[HDR + et * LC + i0 + q] : 0u;
        u16 h = 0;
        if (i0 + q < cnt && tok < MM && !(FL & (1 << 17))) {
          const u16* a = xg + (size_t)tok * (KS * NST * UPS * 64) + kb + (ev[q] & 0x1FFFu);
          if constexpr ((FL & (1 << 19)) != 0) asm volatile("ld.global.u16 %0, [%1];" : "=h"(h) : "l"(a));
          else asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(h) : "l"(a));
        }
        xv[q] = __uint_as_float((u32)h << 16);
      }
#pragma unroll
      for (int q = 0; q < 16; ++q) {
        if (i0 + q < cnt) {
          const int row = (int)((ev[q] >> 13) & 7u);
          const float val = __uint_as_float(ev[q] & 0xFFFF0000u);
#pragma unroll
          for (int r = 0; r < 8; ++r)
            if (row == r) { E1r[r] = fmaf(nd, xv[q], E1r[r]); E2r[r] = fmaf(val, xv[q], E2r[r]); }
        }
      }
    }
    if (tok < TOKS && !(FL & (1 << 18))) {
#pragma unroll
      for (int r = 0; r < 8; r += 4) {
        *reinterpret_cast<float4*>(corr + tok * RP + 8 * et + r) = make_float4(E1r[r], E1r[r + 1], E1r[r + 2], E1r[r + 3]);
        *reinterpret_cast<float4*>(corr + (TOKS + tok) * RP + 8 * et + r) = make_float4(E2r[r], E2r[r + 1], E2r[r + 2], E2r[r + 3]);
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(cbar);
    return;
  }

  if (warp > NW) {   // ---------------- escape warps (rare escapes, lane = token): tile et of this CTA
    const int et = warp - NW - 1;
    const u32* el = reinterpret_cast<const u32*>(gbase + (etab_s - sbase));
    mbar_wait_sleep(pbar, 0);
    const int cnt = (int)el[et];
    const u32 ea = lane < cnt ? el[HDR + et * LC + lane] : 0u;
    const u32 eb2 = (LC > 32 && lane + 32 < cnt) ? el[HDR + et * LC + 32 + lane] : 0u;
    const int sa = lane < cnt ? (int)((ea & 0x1FFFu) >> LSK) : -1;
    const int sb = (LC > 32 && lane + 32 < cnt) ? (int)((eb2 & 0x1FFFu) >> LSK) : -1;
    float E1r[8], E2r[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) { E1r[r] = 0.f; E2r[r] = 0.f; }
    const float nd = -__uint_as_float(0x38000000u);   // -d = -2^-15 (fake domain): every F12 escape decodes to d
    const int tok = lane;
    for (int st = 0; st < NST; ++st) {
      const int slot = st % NSTAGE;
      mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
      u32 m0 = __ballot_sync(0xffffffffu, sa == st);
      u32 m1 = LC > 32 ? __ballot_sync(0xffffffffu, sb == st) : 0u;
      if (m0 | m1) {
        const u32 s0 = sbase + slot * SST;
        while (m0 | m1) {
          u32 e;
          if (m0) { const int i = __ffs(m0) - 1; m0 &= m0 - 1; e = __shfl_sync(0xffffffffu, ea, i); }
          else { const int i = __ffs(m1) - 1; m1 &= m1 - 1; e = __shfl_sync(0xffffffffu, eb2, i); }
          const u32 kl = e & (u32)(UPS * 64 - 1), kc = kl & 63u, u = kl >> 6;
          const int row = (int)((e >> 13) & 7u);
          const float val = __uint_as_float(e & 0xFFFF0000u);
          float x = 0.f;
          if (tok < TOKR)
            x = __uint_as_float(lds16(s0 + u * XT + (u32)tok * 128 + ((((kc >> 3) ^ (u32)(tok & 7))) << 4) + (kc & 7u) * 2) << 16);
#pragma unroll
          for (int r = 0; r < 8; ++r)
            if (row == r) { E1r[r] = fmaf(nd, x, E1r[r]); E2r[r] = fmaf(val, x, E2r[r]); }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
    if (tok < TOKS) {
#pragma unroll
      for (int r = 0; r < 8; r += 4) {
        *reinterpret_cast<float4*>(corr + tok * RP + 8 * et + r) = make_float4(E1r[r], E1r[r + 1], E1r[r + 2], E1r[r + 3]);
        *reinterpret_cast<float4*>(corr + (TOKS + tok) * RP + 8 * et + r) = make_float4(E2r[r], E2r[r + 1], E2r[r + 2], E2r[r + 3]);
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(cbar);
    return;
  }

  // ---------------- consumers (warp wk = k slice)
  const int wk = warp;
  u16* outp = nullptr;
  if constexpr ((FL & 1024) == 0 && (FL & 4096) == 0) asm volatile("mov.b64 %0, %1;" : "=l"(outp) : "l"(p.out));
  const int g = lane >> 2, t4 = lane & 3;
  const int q8 = lane >> 3;
  const int chi = q8 >> 1;
  const u32 lp = (u32)(lane < 4 * PR ? lane : 4 * PR - 1);   // lane used for the partial tile's loads
  int trow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) trow[mt] = TOKR >= 16 ? mt * 16 + (lane & 7) + 8 * (q8 & 1) : (lane & 7);
  float acc[NACC][MT][NT][4];
#pragma unroll
  for (int c = 0; c < NACC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;
  u64 tfirst = 0;
  if constexpr ((FL & 8192) != 0 && (FL & 16) == 0) {
    // software-pipelined consumer: unit u+1's loads are issued before unit u's decode + HMMA; the next stage's first
    // unit is prefetched when its full barrier already tests complete; a slot is released as soon as its last loads
    // are issued (in-order shared-memory pipe, as sk11's UPW == 1 path)
    static_assert(UPW % 2 == 0, "pipelined consumer needs an even UPW");
    uint4 Lb[2][NT];
    uint2 Nb[2][NT];
    u32 Ab[2][4][MT][4];
    mbar_wait(full0, 0);
    if (FL & 2) tfirst = gtime();
    {
      const u32 wb0 = sbase + XBY + (u32)wk * (UPW * TUW);
      ld_unit<MT, TOKR, NT, PR, TPN>(wb0, sbase + (wk * UPW) * XT, (u32)lane, lp, wk * UPW, trow, chi, Lb[0], Nb[0], Ab[0]);
    }
    for (int st = 0; st < NST; ++st) {
      const int slot = st % NSTAGE;
      const u32 s0 = sbase + slot * SST;
      const u32 wb = s0 + XBY + (u32)wk * (UPW * TUW);
#pragma unroll
      for (int uu = 0; uu < UPW; ++uu) {
        const int cb = uu & 1, nb = cb ^ 1;
        bool pre = true;
        if (uu + 1 < UPW) {
          const int ui = wk * UPW + uu + 1;
          ld_unit<MT, TOKR, NT, PR, TPN>(wb + (u32)(uu + 1) * TUW, s0 + ui * XT, (u32)lane, lp, ui, trow, chi, Lb[nb], Nb[nb], Ab[nb]);
        } else {
          __syncwarp();
          if (lane == 0) mbar_arrive(empty0 + 8 * slot);
          if (st + 1 < NST) {
            const int s1 = (st + 1) % NSTAGE;
            pre = mbar_test(full0 + 8 * s1, ((st + 1) / NSTAGE) & 1);
            if (pre) {
              const u32 n0 = sbase + s1 * SST;
              ld_unit<MT, TOKR, NT, PR, TPN>(n0 + XBY + (u32)wk * (UPW * TUW), n0 + (wk * UPW) * XT, (u32)lane, lp, wk * UPW,
                                             trow, chi, Lb[nb], Nb[nb], Ab[nb]);
            }
          }
        }
        mma_unit<MT, NT, NACC>(uu, Lb[cb], Nb[cb], Ab[cb], acc);
        if (uu + 1 == UPW && st + 1 < NST && !pre) {
          const int s1 = (st + 1) % NSTAGE;
          mbar_wait(full0 + 8 * s1, ((st + 1) / NSTAGE) & 1);
          const u32 n0 = sbase + s1 * SST;
          ld_unit<MT, TOKR, NT, PR, TPN>(n0 + XBY + (u32)wk * (UPW * TUW), n0 + (wk * UPW) * XT, (u32)lane, lp, wk * UPW,
                                         trow, chi, Lb[nb], Nb[nb], Ab[nb]);
        }
      }
    }
  } else
  for (int st = 0; st < NST; ++st) {
    const int slot = st % NSTAGE;
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    if ((FL & 2) && st == 0) tfirst = gtime();
    const u32 s0 = sbase + slot * SST;
    if constexpr ((FL & 16) != 0) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      continue;
    }
    const u32 wb = s0 + XBY + (u32)wk * (UPW * TUW);
    u32 b[T][8];
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const int ui = wk * UPW + uu;
      const u32 xt = s0 + ui * XT;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const bool part = PR < 8 && nt == NT - 1;
        const u32 ll = part ? lp : (u32)lane;
        uint4 L;
        uint2 N;
        const u32 tb = wb + (u32)uu * TUW + (u32)nt * 768u;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(L.x), "=r"(L.y), "=r"(L.z), "=r"(L.w) : "r"(tb + ll * 16));
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(N.x), "=r"(N.y) : "r"(tb + (part ? TPN : 512) + ll * 8));
        decode12(L.x, L.y, N.x, &b[uu * NT + nt][0]);
        decode12(L.z, L.w, N.y, &b[uu * NT + nt][4]);
      }
      u32 a[4][MT][4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          ldsm_x4(xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * TOKR + trow[mt]) & 7)) << 4),
                  a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      if (UPW == 1 && uu == 0) {
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt)
            mma_bf16(acc[(uu * 4 + j) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3],
                     b[uu * NT + nt][2 * j], b[uu * NT + nt][2 * j + 1]);
    }
    if (UPW > 1) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
  }
  if ((FL & 2) && warp == 0 && lane == 0) {
    p.trc[blockIdx.x * 8 + 0] = tstart;
    p.trc[blockIdx.x * 8 + 1] = tfirst;
    p.trc[blockIdx.x * 8 + 2] = gtime();
    u32 smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    p.trc[blockIdx.x * 8 + 6] = smid;
    p.trc[4096 + blockIdx.x * 2 + 0] = cstart;
    p.trc[4096 + blockIdx.x * 2 + 1] = ctime();
  }
  if constexpr ((FL & 16384) != 0) {   // ablation: no epilogue at all (keep acc alive)
    float sacc = 0.f;
#pragma unroll
    for (int c = 0; c < NACC; ++c)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) sacc += acc[c][mt][nt][0] + acc[c][mt][nt][3];
    if (sacc == 1.2345f) p.trc[0] = 1;
    return;
  }
  if constexpr (NE > 0) mbar_wait(cbar, 0);
  if ((FL & 2) && warp == 0 && lane == 0) p.trc[blockIdx.x * 8 + 3] = gtime();
  named_bar(1, NC);
  if ((FL & 2) && warp == 0 && lane == 0) p.trc[blockIdx.x * 8 + 5] = gtime();
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int row = nt * 8 + 2 * t4;
      const int tok = mt * 16 + g;
      float* r0 = red + (wk * TOKS + tok) * RP + row;
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = acc[0][mt][nt][e];
#pragma unroll
        for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
      }
      *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
    }
  named_bar(1, NC);
  if ((FL & 2) && warp == 0 && lane == 0) p.trc[blockIdx.x * 8 + 7] = gtime();
  constexpr int R2 = R / 2;
  float sc;
  int M, N;
  if constexpr ((FL & 1024) != 0) { sc = p.scale; M = p.M; N = p.N; outp = p.out; }
  else { sc = __uint_as_float(SCB); M = MM; N = NN; }
  if constexpr ((FL & 4096) != 0) asm volatile("ld.shared.u64 %0, [%1];" : "=l"(outp) : "r"(oslot) : "memory");
  const int rb = KS > 1 ? cta / KS : cta;
  if constexpr (KS == 1 && R % 4 == 0) {
    constexpr int R4 = R / 4;
    for (int idx = tid; idx < R4 * M; idx += NC) {
      const int tt = idx / R4, row = (idx - tt * R4) * 4;
      float4 v = *reinterpret_cast<const float4*>(red + tt * RP + row);
#pragma unroll
      for (int w = 1; w < WSK; ++w) {
        const float4 pp = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
        v.x += pp.x; v.y += pp.y; v.z += pp.z; v.w += pp.w;
      }
      if constexpr (NE > 0) {
        const float4 e1 = *reinterpret_cast<const float4*>(corr + tt * RP + row);
        v.x += e1.x; v.y += e1.y; v.z += e1.z; v.w += e1.w;
        const float4 e2 = *reinterpret_cast<const float4*>(corr + (TOKS + tt) * RP + row);
        v.x += e2.x; v.y += e2.y; v.z += e2.z; v.w += e2.w;
      }
      uint2 pk;
      pk.x = pack_bf2(v.x * sc, v.y * sc);
      pk.y = pack_bf2(v.z * sc, v.w * sc);
      if constexpr ((FL & 2048) == 0) *reinterpret_cast<uint2*>(outp + (size_t)tt * N + rb * R + row) = pk;
      else if (v.x == 1.2345f) outp[0] = 0;
    }
  } else if constexpr (KS == 1) {
    for (int idx = tid; idx < R2 * M; idx += NC) {
      const int tt = idx / R2, row = (idx - tt * R2) * 2;
      float2 v = *reinterpret_cast<const float2*>(red + tt * RP + row);
#pragma unroll
      for (int w = 1; w < WSK; ++w) {
        const float2 pp = *reinterpret_cast<const float2*>(red + (w * TOKS + tt) * RP + row);
        v.x += pp.x; v.y += pp.y;
      }
      if constexpr (NE > 0) {
        const float2 e1 = *reinterpret_cast<const float2*>(corr + tt * RP + row);
        v.x += e1.x; v.y += e1.y;
        const float2 e2 = *reinterpret_cast<const float2*>(corr + (TOKS + tt) * RP + row);
        v.x += e2.x; v.y += e2.y;
      }
      if constexpr ((FL & 2048) == 0)
        *reinterpret_cast<u32*>(outp + (size_t)tt * N + rb * R + row) = pack_bf2(v.x * sc, v.y * sc);
      else if (v.x == 1.2345f) outp[0] = 0;
    }
  } else {
    // split-K: own partial -> scratch part[cta][tok][R]; the last arriver of the row block sums the KS partials in
    // fixed order (deterministic) and writes bf16; it also resets the counter for the next launch
    float* const part = PB ? reinterpret_cast<float*>(const_cast<u8*>(cptr<PB>())) : p.part;
    u32* const cntp = CB ? reinterpret_cast<u32*>(const_cast<u8*>(cptr<CB>())) : p.cnt;
    float* mine = part + (size_t)cta * (TOKS * R);
    for (int idx = tid; idx < R2 * M; idx += NC) {
      const int tt = idx / R2, row = (idx - tt * R2) * 2;
      float2 v = *reinterpret_cast<const float2*>(red + tt * RP + row);
#pragma unroll
      for (int w = 1; w < WSK; ++w) {
        const float2 pp = *reinterpret_cast<const float2*>(red + (w * TOKS + tt) * RP + row);
        v.x += pp.x; v.y += pp.y;
      }
      if constexpr (NE > 0) {
        const float2 e1 = *reinterpret_cast<const float2*>(corr + tt * RP + row);
        v.x += e1.x; v.y += e1.y;
        const float2 e2 = *reinterpret_cast<const float2*>(corr + (TOKS + tt) * RP + row);
        v.x += e2.x; v.y += e2.y;
      }
      *reinterpret_cast<float2*>(red + tt * RP + row) = v;       // keep own partial in smem
      __stcg(reinterpret_cast<float2*>(mine + tt * R + row), v);
    }
    if constexpr ((FL & 32768) == 0) __threadfence();
    named_bar(1, NC);
    __shared__ u32 s_last;
    if (tid == 0) {
      u32 prev;
      if constexpr ((FL & 32768) != 0)   // one thread: CTA barrier + gpu-scope acq_rel RMW (cumulative release)
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(prev) : "l"(cntp + rb) : "memory");
      else
        prev = atomicAdd(cntp + rb, 1u);
      s_last = prev == (u32)(KS - 1);
      if (s_last) cntp[rb] = 0u;
    }
    named_bar(1, NC);
    if (s_last) {
      if constexpr ((FL & 32768) == 0) __threadfence();
      for (int idx = tid; idx < R2 * M; idx += NC) {
        const int tt = idx / R2, row = (idx - tt * R2) * 2;
        float2 v = make_float2(0.f, 0.f);
#pragma unroll
        for (int q = 0; q < KS; ++q) {
          float2 pp;
          if (q == ks) pp = *reinterpret_cast<const float2*>(red + tt * RP + row);
          else pp = __ldcg(reinterpret_cast<const float2*>(part + (size_t)(rb * KS + q) * (TOKS * R) + tt * R + row));
          v.x += pp.x; v.y += pp.y;
        }
        *reinterpret_cast<u32*>(outp + (size_t)tt * N + rb * R + row) = pack_bf2(v.x * sc, v.y * sc);
      }
    }
  }
  if ((FL & 2) && tid == 0) p.trc[blockIdx.x * 8 + 4] = gtime();
}
// SPDX-SnippetEnd
"""

# ------------------------------------------------------------------------------------------------- configuration
# cfg = (R, NT, WSK, UPW, NACC, ring cap KB, flags, KS); PR = R - 8 (NT - 1) rows in the last tile; flags: 4 = baked W
# pointer, other bits = kernel FL
_q8_XLAG1 = 1 << 22
_q8_CFG = {   # (R, NT, WSK, UPW, NACC, ring cap KB, flags, KS)
    ("qkv_gate", 8): (24, 3, 4, 2, 2, 144, _q8_XLAG1 | 4, 1),    # 5 stages, baked pointers
    ("qkv_gate", 16): (24, 3, 4, 2, 2, 200, _q8_XLAG1 | 4, 1),   # 5 stages, baked pointers
}
_q8_OWN = set(_q8_CFG)
if os.environ.get("QKVB_OWN") is not None:   # experiments: e.g. "qkv_gate:16"
    _q8_OWN = set()
    for _t in filter(None, os.environ["QKVB_OWN"].split(",")):
        _l, _m = _t.split(":")
        _q8_OWN.add((_l, int(_m)))
_q8_K_OF = {"pooler": 8192, "unpooler": 4096, "o": 1024, "qkv_gate": 8192}
_q8_N_OF = {"pooler": 4096, "unpooler": 8192, "o": 8192, "qkv_gate": 2304}
_q8_FLAG_BAKE = 4
_q8_SMEM_LIMIT = 232_448
_q8_FAKE12 = 112


def _q8__hdr(nt):
    return 4 if nt <= 4 else 8


def _q8__geom(cfg, layer, m, lc=32):
    R, NT, WSK, UPW, NACC, cap, fl, KS = cfg
    PR = R - 8 * (NT - 1)
    tokr = 8 if m <= 8 else 16 if m <= 16 else 32
    mt = 2 if tokr == 32 else 1
    ups = WSK * UPW
    tpb = 768 if PR == 8 else 96 * PR
    tuw = 768 * (NT - 1) + tpb
    wby = WSK * UPW * tuw
    xby = ups * tokr * 128
    nst = _q8_K_OF[layer] // KS // (ups * 64)
    toks = 16 * mt
    rv = NT * 8
    rp = rv + ((40 - (rv % 32)) % 32)
    elb = 4 * _q8__hdr(NT) + NT * lc * 4
    return dict(tokr=tokr, mt=mt, ups=ups, wby=wby, xby=xby, nst=nst, toks=toks, rp=rp, rv=rv, PR=PR, tuw=tuw,
                ne=NT, elb=elb, lc=lc)


def _q8__nstage(cfg, g):
    sst = (g["xby"] + g["wby"] + 1023) // 1024 * 1024
    extra = 1024 + 16 * 16 + 16 + g["elb"] + 2 * g["toks"] * g["rp"] * 4 + 64
    ring = min(cfg[5] * 1024, _q8_SMEM_LIMIT - extra)
    return max(2, min(16, g["nst"], ring // sst)), sst


def _q8__smem(g, nstage, sst):
    return 1024 + nstage * sst + 16 * nstage + 32 + g["elb"] + 2 * g["toks"] * g["rp"] * 4


# ------------------------------------------------------------------------------------------------- weight packing
def _q8__window(e, width):
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    return max(range(1, 256 - width), key=lambda q: int(cs[q + width] - cs[q]))


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Copied from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_tileunits` (via alygg77/okc-muse-spark-kernels).
# Changes: renamed; docstring shortened.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def _q8__tileunits(w):
    """W [N, K] -> [N/8, K/64, 32 lanes, 16] in mma.m16n8k16 B-fragment order (sk2_tileunits)."""
    n, k = w.shape
    t = w.view(n // 8, 8, k // 64, 4, 2, 4, 2)
    return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)
# SPDX-SnippetEnd


def _q8_compress12(w, elo=None):
    """F12 planes in tile-unit order [N/8, K/64, 32, 16] (agent G's fake-exponent 12-bit: 16-exponent window)."""
    tu = _q8__tileunits(w.contiguous())
    u = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    if elo is None:
        elo = _q8__window(e, 16)
    c = e - elo
    esc = (c < 0) | (c > 15)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m),
            "nib": torch.where(esc, torch.zeros_like(u), (s << 3) | (c >> 1)), "esc": esc, "u": u, "elo": elo}


def _q8_pack_nib8(nib):
    """agent G's F12 nibble word: nib [..., 8] (s << 3 | h) -> int32"""
    nib = nib.to(torch.int64)
    s, h = nib >> 3, nib & 7
    word = torch.zeros(nib.shape[:-1], dtype=torch.int64, device=nib.device)
    for b in range(4):
        word |= (h[..., b] | (s[..., b] << 7) | (h[..., 4 + b] << 4) | (s[..., 4 + (b + 3) % 4] << 3)) << (8 * b)
    return torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)


def _q8__pad_rows(w, R, NT):
    """[N, K] -> [N/R * 8 NT, K]: row block rb (R real rows) occupies rows rb * 8 NT .. + R, the rest zero."""
    n, k = w.shape
    nrb = n // R
    wp = torch.zeros(nrb, 8 * NT, k, dtype=w.dtype, device=w.device)
    wp[:, :R] = w.view(nrb, R, k)
    return wp.view(nrb * 8 * NT, k)


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_stream` (via alygg77/okc-muse-spark-kernels).
# Changes: gathers our separate low-byte and 4-bit-code planes per tile unit (codes repacked into nibble words) instead of sk2's pre-packed 768-byte tile units, with one row block per CTA and every warp a k warp (no row warps), split over KS CTAs along K, with a partial last tile of PR rows.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def _q8__wstream(f, nrb, NT, PR, WSK, UPW, KS):
    """[stage][cta = rb * KS + ks][warp][uu][nt tile-units: 512 lo | 256 nibble words; last tile 64 PR lo | 32 PR nib]"""
    dev = f["lo"].device
    U = f["lo"].shape[1]
    ups = WSK * UPW
    S = U // KS // ups
    C = nrb * KS
    ar = lambda v: torch.arange(v, device=dev)
    s = ar(S).view(-1, 1, 1, 1, 1)
    c = ar(C).view(1, -1, 1, 1, 1)
    w = ar(WSK).view(1, 1, -1, 1, 1)
    uu = ar(UPW).view(1, 1, 1, -1, 1)
    nt = ar(NT).view(1, 1, 1, 1, -1)
    tile = ((c // KS) * NT + nt).expand(S, C, WSK, UPW, NT)
    unit = ((c % KS) * (S * ups) + s * ups + w * UPW + uu).expand(S, C, WSK, UPW, NT)
    lo = f["lo"][tile, unit].to(torch.uint8)                                          # [S, C, W, UPW, NT, 32, 16]
    nb = _q8_pack_nib8(f["nib"][tile, unit].reshape(S, C, WSK, UPW, NT, 32, 2, 8))         # [S, C, W, UPW, NT, 32, 2]
    nb = nb.contiguous().view(torch.uint8).reshape(S, C, WSK, UPW, NT, 32, 8)
    parts = []
    for t in range(NT):
        nl = 32 if (t < NT - 1 or PR == 8) else 4 * PR
        parts.append(lo[:, :, :, :, t, :nl].reshape(S, C, WSK, UPW, nl * 16))
        parts.append(nb[:, :, :, :, t, :nl].reshape(S, C, WSK, UPW, nl * 8))
    return torch.cat(parts, -1).contiguous().view(-1)
# SPDX-SnippetEnd


def _q8__elist(f, nrb, R, NT, KS, K):
    """Per CTA: [HDR u32 per-tile counts][NT tiles x LC u32 entries k_local | row << 13 | fake_bits << 16]."""
    dev = f["u"].device
    idx = f["esc"].nonzero()
    tile, unit, lane, i = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    g = lane >> 2
    et = tile % NT
    keep = et * 8 + g < R                       # padding rows (zeros) are not escapes
    idx, tile, unit, lane, i, g, et = idx[keep], tile[keep], unit[keep], lane[keep], i[keep], g[keep], et[keep]
    rb = tile // NT
    k = unit * 64 + 16 * (i >> 2) + 8 * ((i >> 1) & 1) + 2 * (lane & 3) + (i & 1)
    t = (f["u"][idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]].to(torch.int32) << 16).view(torch.float32)
    fv = t * (2.0 ** (_q8_FAKE12 - f["elo"]))
    fb16 = fv.to(torch.bfloat16)
    if not bool((torch.isfinite(fv) & (fb16.float() == fv) & (fb16.float() * (2.0 ** (f["elo"] - _q8_FAKE12)) == t)).all()):
        return None
    fb = fb16.view(torch.int16).to(torch.int64) & 0xFFFF
    kp = K // KS
    ks, kl = k // kp, k % kp
    C = nrb * KS
    cta = rb * KS + ks
    key = (cta * NT + et).long()
    cnt = torch.bincount(key, minlength=C * NT)
    cmax = int(cnt.max()) if cnt.numel() else 0
    HDR = _q8__hdr(NT)
    if cmax > 64 or NT > HDR:
        return None
    lc = 32 if cmax <= 32 else 64
    o = torch.argsort(key * 8192 + kl)
    key, ent = key[o], (kl | (g << 13)).long()[o] | (fb[o] << 16)
    st0 = torch.cumsum(cnt, 0) - cnt
    j = torch.arange(key.numel(), device=dev) - st0[key]
    row = HDR + NT * lc
    buf = torch.zeros(C, row, dtype=torch.int64, device=dev)
    buf[:, :NT] = cnt.view(C, NT)
    buf.view(-1)[(key // NT) * row + HDR + (key % NT) * lc + j] = ent
    buf = torch.where(buf >= 2 ** 31, buf - 2 ** 32, buf).to(torch.int32)
    return {"buf": buf.contiguous(), "lc": lc, "n": int(cnt.sum()), "cmax": cmax}


# ------------------------------------------------------------------------------------------------- NVRTC module
class _q8__Module:
    def __init__(self, src, name="qkvb.cu", opts=()):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        o = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        o += [x.encode() for x in opts]
        if os.environ.get("CU_LINEINFO"):
            o.append(b"-lineinfo")
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(o), o)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        self.log = log.decode(errors="replace").strip("\x00 \n")
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + self.log)
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        self.cubin = cubin
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err
        self.funcs = {}

    def func(self, name, smem):
        key = (name, smem)
        f = self.funcs.get(key)
        if f is None:
            d = self.driver
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            assert int(err) == 0, (name, err)
            if smem > 48 * 1024:
                (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
                assert int(err) == 0, (name, smem, err)
            self.funcs[key] = f
        return f

    def attrs(self, name, smem=0):
        d = self.driver
        f = self.func(name, smem)
        out = {}
        for k in ("NUM_REGS", "LOCAL_SIZE_BYTES"):
            err, v = d.cuFuncGetAttribute(getattr(d.CUfunction_attribute, "CU_FUNC_ATTRIBUTE_" + k), f)
            out[k] = v
        return out

    def launch(self, f, grid, block, smem, params):
        d = self.driver
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        assert int(err) == 0, err


def _q8__tmap_x3d(ptr, m, k, box_rows, units):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(3), ptr,
        [d.cuuint64_t(64), d.cuuint64_t(m), d.cuuint64_t(k // 64)], [d.cuuint64_t(k * 2), d.cuuint64_t(128)],
        [d.cuuint32_t(64), d.cuuint32_t(box_rows), d.cuuint32_t(units)],
        [d.cuuint32_t(1), d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


_q8__TM = {}


def _q8__xmap(x, m, k, box_rows, units):
    key = (x.data_ptr(), m, k, box_rows, units)
    b = _q8__TM.get(key)
    if b is None:
        if len(_q8__TM) > 512:
            _q8__TM.clear()
        b = _q8__TM[key] = _q8__tmap_x3d(x.data_ptr(), m, k, box_rows, units)
    return b


# ------------------------------------------------------------------------------------------------- prepare / run
def _q8_prepare_qkvb(weights, cfgs=None, verbose=False, variants=None, selftest=True):
    """cfgs: {(layer, m): cfg} (default: CFG restricted to OWN). variants: extra {(layer, m, tag): cfg} instances.
    Never raises: any failure (packing, NVRTC, smem, self-test) leaves the shape to the base (run_qkvb -> None)."""
    try:
        st = _q8__prepare_impl(weights, cfgs, verbose, variants)
    except Exception as ex:   # noqa: BLE001
        print(f"[qkvb] prepare failed, base keeps every shape: {type(ex).__name__}: {ex}", flush=True)
        return {"ent": {}}
    if selftest:
        for (layer, m, tag), e in st["ent"].items():
            if not e["ok"]:
                continue
            try:
                g = torch.Generator(device="cuda").manual_seed(1234 + m)
                x = torch.randn((m, e["k"]), generator=g, device="cuda").to(torch.bfloat16)
                out = _q8_run_qkvb(x, {"m": m, "n": e["n"], "k": e["k"], "layer": layer}, st, tag=tag or "__self")
                ref = x.float() @ weights[layer][: e["n"]].float().T
                err = float((out.float() - ref).norm() / ref.norm())
                e["ok"] = err < 5e-3
                if not e["ok"]:
                    print(f"[qkvb] self-test failed for {(layer, m, tag)}: rel err {err:.3e}", flush=True)
            except Exception as ex:   # noqa: BLE001
                e["ok"] = False
                print(f"[qkvb] self-test error for {(layer, m, tag)}: {ex}", flush=True)
        torch.cuda.synchronize()
    return st


def _q8__prepare_impl(weights, cfgs=None, verbose=False, variants=None):
    cfgs = dict({k: v for k, v in _q8_CFG.items() if k in _q8_OWN} if cfgs is None else cfgs)
    inst = [(layer, m, "", cfg) for (layer, m), cfg in cfgs.items()]
    if variants:
        inst += [(layer, m, tag, cfg) for (layer, m, tag), cfg in variants.items()]
    st = {"ent": {}}
    if any(c[6] & 2 for _, _, _, c in inst):
        st["trc_buf"] = torch.zeros(65536, dtype=torch.int64, device="cuda")
        st["trc"] = st["trc_buf"].data_ptr()
    comps, wss, ess, elos = {}, {}, {}, {}
    for layer, m, tag, cfg in inst:
        R, NT, WSK, UPW, NACC, cap, fl, KS = cfg
        PR = R - 8 * (NT - 1)
        n_rows, K = _q8_N_OF[layer], _q8_K_OF[layer]
        if not (1 <= PR <= 8) or R % 2 or n_rows % R or m not in (8, 16, 32) or WSK * UPW * 64 > 1024:
            print(f"[qkvb] bad cfg {cfg} for {(layer, m)}", flush=True)
            continue
        nrb = n_rows // R
        if layer not in elos:
            elos[layer] = _q8_compress12(weights[layer][:n_rows])["elo"]
        ckey = (layer, R, NT)
        if ckey not in comps:
            comps[ckey] = _q8_compress12(_q8__pad_rows(weights[layer][:n_rows], R, NT), elo=elos[layer])
        f = comps[ckey]
        if K % (KS * WSK * UPW * 64):
            continue
        wkey = (layer, R, NT, WSK, UPW, KS)
        if wkey not in wss:
            wss[wkey] = _q8__wstream(f, nrb, NT, PR, WSK, UPW, KS)
        ws = wss[wkey]
        ekey = (layer, R, NT, KS)
        if ekey not in ess:
            ess[ekey] = _q8__elist(f, nrb, R, NT, KS, K)
        es = ess[ekey]
        if es is None:
            print(f"[qkvb] {(layer, m, tag)}: escapes do not fit: left to the base", flush=True)
            continue
        g = _q8__geom(cfg, layer, m, es["lc"])
        nstage, sst = _q8__nstage(cfg, g)
        smem = _q8__smem(g, nstage, sst)
        if smem > _q8_SMEM_LIMIT:
            print(f"[qkvb] {(layer, m, tag)} needs {smem} B smem: left to the base", flush=True)
            continue
        nthr = (WSK + 1 + (0 if fl & 128 else NT)) * 32
        ncta = nrb * KS
        kn = f"qkvb_{layer}_{m}{tag}"
        ent = dict(cfg=cfg, ws=ws, es=es, elo=f["elo"], n=n_rows, k=K, g=g, nthr=nthr, ncta=ncta, smem=smem, kn=kn,
                   nstage=nstage, sst=sst, ok=True, scale=2.0 ** (f["elo"] - _q8_FAKE12), nrb=nrb)
        if KS > 1:
            ent["part"] = torch.zeros(ncta * g["toks"] * R, dtype=torch.float32, device="cuda")
            ent["cnt"] = torch.zeros(nrb, dtype=torch.int32, device="cuda")
        st["ent"][(layer, m, tag)] = ent
    src = _q8_SRC
    for (layer, m, tag), e in st["ent"].items():
        R, NT, WSK, UPW, NACC, cap, fl, KS = e["cfg"]
        g = e["g"]
        bake = bool(fl & _q8_FLAG_BAKE)
        wb = f"{e['ws'].data_ptr()}ull" if bake else "0ull"
        eb = f"{e['es']['buf'].data_ptr()}ull" if bake else "0ull"
        pb = f"{e['part'].data_ptr()}ull" if (bake and 'part' in e) else "0ull"
        cb = f"{e['cnt'].data_ptr()}ull" if (bake and 'cnt' in e) else "0ull"
        scb = struct.unpack("<I", struct.pack("<f", e["scale"]))[0]
        src += (f'extern "C" __global__ void __launch_bounds__({e["nthr"]}, 1) {e["kn"]}(const __grid_constant__ P p) '
                f'{{ skv_body<{g["mt"]}, {g["tokr"]}, {NT}, {g["PR"]}, {WSK}, {UPW}, {e["nstage"]}, {NACC}, {g["nst"]}, '
                f'{wb}, {g["lc"]}, {R}, {KS}, {fl & ~_q8_FLAG_BAKE}, {eb}, {m}, {e["n"]}, {scb}u, {pb}, {cb}>(p); }}\n')
    st["src"] = src
    if not st["ent"]:
        return st
    st["mod"] = _q8__Module(src)
    for key, e in st["ent"].items():
        e["f"] = st["mod"].func(e["kn"], e["smem"])
        if verbose:
            a = st["mod"].attrs(e["kn"], e["smem"])
            wbits = 8.0 * e["ws"].numel() / (e["n"] * e["k"])
            print(f"[qkvb] {key} cfg {e['cfg']} nstage {e['nstage']} sst {e['sst']} smem {e['smem']} thr {e['nthr']} "
                  f"ncta {e['ncta']} regs {a['NUM_REGS']} local {a['LOCAL_SIZE_BYTES']} esc {e['es']['n']} "
                  f"cmax {e['es']['cmax']} lc {e['g']['lc']} bits/w {wbits:.3f}", flush=True)
    torch.cuda.synchronize()
    return st


def _q8_run_qkvb(x, config, state, tag=""):
    m, layer = config["m"], config["layer"]
    if (layer, m) not in _q8_OWN and not tag:
        return None
    e = state["ent"].get((layer, m, "" if tag == "__self" else tag))
    if e is None or not e["ok"] or config["n"] != e["n"] or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    n, k = e["n"], e["k"]
    g = e["g"]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    tm = _q8__xmap(x, m, k, g["tokr"], g["ups"])
    part = e["part"].data_ptr() if "part" in e else 0
    cnt = e["cnt"].data_ptr() if "cnt" in e else 0
    params = tm + struct.pack("<QQQQQQQfiiiii", e["ws"].data_ptr(), x.data_ptr(), out.data_ptr(), e["es"]["buf"].data_ptr(),
                              state.get("trc", 0), part, cnt, e["scale"], m, n, k, 0, 0)
    state["mod"].launch(e["f"], e["ncta"], e["nthr"], e["smem"], params)
    return out


# ======================================================================== module F: F/sk11.py
"""sk11 (agent F): skinny GEMM for qkv_gate / pooler / unpooler (full) / o at m = 8 / 16 / 32.

    state = prepare_sk11(weights)           # never raises (any failure -> run_sk11 returns None for that shape)
    out   = run_sk11(x, config, state)      # -> [M, N] bf16, or None where the base (bo5) keeps the shape

Kernel structure derived from vjeux/okc-2026 `sk2` (via alygg77's public OKC submission, Apache-2.0) and agent X's
NVRTC port sk3 (claude/t3/X/sk3.py): swapped operands (A = x tile 16 tokens x 16 k via ldmatrix from a TMA-loaded
128B-swizzled tile, B = 8 weight rows x 16 k decoded from the weight stream), each CTA owns R rows x the whole K, a
producer warp refills a ring of stages (x tile + this CTA's weights [+ escape block]).

Shipped path (flag F12, NOTES.md iteration 5): agent G's lossless F12 fake-exponent 12-bit format (lo byte + nibble,
16-exponent window, escapes ~7e-5) with escapes in per-(CTA, 8-row tile) lists loaded once at kernel start; one escape
warp per tile keeps its list in registers and per stage only votes on it (no shared-memory work on the slot-release path
unless an escape falls in that stage), corrections E1 = -d * x and E2 = true * x in fp32 registers, output
((partials + E1) + E2) * 2^(E_lo - 112): bit-exact for one-hot rows.
Research path (F11, agent G's 11-bit code + 1.3% escapes corrected by tensor-core escape warps): exact but slower
(NOTES.md iterations 1-4); kept behind the format flag for reference.
"""
import ctypes
import math
import os
import struct

import torch

# ----------------------------------------------------------------------------------------------------------- CUDA
_f1_SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) P { TMap xmap; const u8* W; const u16* x; u16* out; const u8* E; const uint2* etab; u64* trc;
                         float scale; int M; int N; int K; int pad0; int pad1; };

template <u64 A> __device__ __forceinline__ const u8* cptr() { const u8* p; asm("mov.u64 %0, %1;" : "=l"(p) : "n"(A)); return p; }
__device__ __forceinline__ u64 gtime() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ u64 ctime() { u64 t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory"); return t; }
__device__ __forceinline__ void ldsm_x4(u32 addr, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 sel) {
  u32 d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel)); return d;
}
__device__ __forceinline__ u32 rotr(u32 x, int s) { return __funnelshift_r(x, x, s); }
__device__ __forceinline__ u32 hi11(u32 x) {
  u32 r; asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(x), "r"(0x83838383u), "n"(0x3C3C3C3C)); return r;
}
__device__ __forceinline__ u32 sel80(u32 a, u32 b) {
  u32 r; asm("lop3.b32 %0, %1, %2, %3, 0xE4;" : "=r"(r) : "r"(a), "r"(b), "r"(0x80808080u)); return r;
}
// F11 decode (agent G's layout): 8 lo words (L0 = tile-unit A, L1 = tile-unit B) + 3 code words -> 2 x 8 bf16x2
template <int NP>
__device__ __forceinline__ void decode11p(const uint4& L0, const uint4& L1, u32 W0, u32 W1, u32 W2, u32* bA, u32* bB, const u32* pe) {
  u32 h[8];
  h[0] = hi11(W0); h[1] = hi11(rotr(W0, 3)); h[2] = hi11(W1); h[3] = hi11(rotr(W1, 3));
  h[4] = hi11(W2); h[5] = hi11(rotr(W2, 3));
  h[6] = hi11(sel80(rotr(W0, 6), rotr(W1, 5))); h[7] = hi11(sel80(rotr(W0, 7), rotr(W2, 5)));
  // exponent patches: entry = pos (5 bits: weight 4g + b of the group) | mag << 5 (hi byte -= mag), 0 = none
#pragma unroll
  for (int r = 0; r < NP; ++r) {
    const u32 pos = pe[r] & 31u, v = ((pe[r] >> 5) & 63u) << ((pos & 3u) * 8u), gq = pos >> 2;
#pragma unroll
    for (int q = 0; q < 8; ++q) if (gq == (u32)q) h[q] -= v;
  }
  const u32 h0 = h[0], h1 = h[1], h2 = h[2], h3 = h[3], h4 = h[4], h5 = h[5], h6 = h[6], h7 = h[7];
  bA[0] = prmt(L0.x, h0, 0x5140); bA[1] = prmt(L0.x, h0, 0x7362);
  bA[2] = prmt(L0.y, h1, 0x5140); bA[3] = prmt(L0.y, h1, 0x7362);
  bA[4] = prmt(L0.z, h2, 0x5140); bA[5] = prmt(L0.z, h2, 0x7362);
  bA[6] = prmt(L0.w, h3, 0x5140); bA[7] = prmt(L0.w, h3, 0x7362);
  bB[0] = prmt(L1.x, h4, 0x5140); bB[1] = prmt(L1.x, h4, 0x7362);
  bB[2] = prmt(L1.y, h5, 0x5140); bB[3] = prmt(L1.y, h5, 0x7362);
  bB[4] = prmt(L1.z, h6, 0x5140); bB[5] = prmt(L1.z, h6, 0x7362);
  bB[6] = prmt(L1.w, h7, 0x5140); bB[7] = prmt(L1.w, h7, 0x7362);
}
__device__ __forceinline__ void decode11(const uint4& L0, const uint4& L1, u32 W0, u32 W1, u32 W2, u32* bA, u32* bB) {
  decode11p<0>(L0, L1, W0, W1, W2, bA, bB, nullptr);
}
// agent G's F12 decode: 8 weights from 2 lo words + 1 nibble word (byte b = h_b | s_b << 7 | h_{4+b} << 4 | s_{4+b} at bit 3 of byte b-1)
__device__ __forceinline__ void decode12(u32 L0, u32 L1, u32 nw, u32* r) {
  u32 hA, hB;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(hA) : "r"(nw), "r"(0x87878787u), "n"(0x38383838));
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(hB) : "r"(rotr(nw, 4)), "r"(0x87878787u), "n"(0x38383838));
  r[0] = prmt(L0, hA, 0x5140); r[1] = prmt(L0, hA, 0x7362);
  r[2] = prmt(L1, hB, 0x5140); r[3] = prmt(L1, hB, 0x7362);
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a)); return r; }
__device__ __forceinline__ void mbar_init(u32 bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `mbar_test` / `mbar_wait` / `mbar_wait_sleep` (MBAR_HELPERS) (via alygg77/okc-muse-spark-kernels).
// Changes: renamed types (u32); mbar_wait written on one line; vjeux's esc_delta, which sits between mbar_wait and mbar_wait_sleep upstream, left out.
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
__device__ __forceinline__ bool mbar_test(u32 bar, u32 parity) {
  u32 ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(u32 bar, u32 parity) { while (!mbar_test(bar, parity)) {} }
__device__ __forceinline__ void mbar_wait_sleep(u32 bar, u32 parity) {
  u32 ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(0x989680u) : "memory");
  } while (!ok);
}
// SPDX-SnippetEnd
__device__ __forceinline__ void mbar_arrive(u32 bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(u32 bar, u32 bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\n}\n" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_load_3d(u32 dst, const TMap* map, u32 bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ u32 lds32(u32 a) { u32 v; asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a)); return v; }
__device__ __forceinline__ u32 lds16(u32 a) { u16 v; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a)); return (u32)v; }

// escape-mma B fragments for lane (g, t) from its 4 entries (slots 2t, 2t+1, 2t+8, 2t+9): B1 = -d, B2 = true value
// (fake bf16) where the entry's row (bits 9-12; 8 = invalid) equals g, else 0
__device__ __forceinline__ void esc_b(const u32* e, int g, u32 ND, u32& b1a, u32& b1b, u32& b2a, u32& b2b) {
  bool mm[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) mm[q] = ((e[q] >> 9) & 15u) == (u32)g;
  b1a = (mm[0] ? ND : 0u) | (mm[1] ? (ND << 16) : 0u);
  b1b = (mm[2] ? ND : 0u) | (mm[3] ? (ND << 16) : 0u);
  b2a = (mm[0] ? (e[0] >> 16) : 0u) | (mm[1] ? (e[1] & 0xFFFF0000u) : 0u);
  b2b = (mm[2] ? (e[2] >> 16) : 0u) | (mm[3] ? (e[3] & 0xFFFF0000u) : 0u);
}

__device__ __forceinline__ bool mbar_testwait(u32 bar, u32 parity) {
  u32 ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
// escape warps: spin (default) or sleep (FL 256)
template <int FL> __device__ __forceinline__ void wait_e(u32 bar, u32 par) {
  if constexpr ((FL & 256) != 0) mbar_wait_sleep(bar, par); else mbar_wait(bar, par);
}
template <int FL> __device__ __forceinline__ void wait_s(u32 bar, u32 par) {
  if constexpr ((FL & 32) != 0) mbar_wait(bar, par); else mbar_wait_sleep(bar, par);
}

// MT token tiles of 16 (TOKR = x box rows = M); NT n8 tiles per consumer warp; NWR row warps; WSK k warps; UPW 64-k
// units per warp per stage; NSTAGE ring slots; NACC accumulator sets; NST stages; WB baked W pointer (0 = params);
// EBC escape-area capacity per slot; CIB fixed size of the first NSTAGE escape blocks; FL flags:
//   2 trace, 16 consumers skip decode+mma (ablation), 32 spin waits for producer, 64 escape warps skip work (ablation,
//   wrong results; 512 / 1024 / 2048: escape warps skip the x gathers / the B build + mma / the entry loads),
//   wrong results), 128 no escape warps at all (ablation, wrong results), 256 escape warps sleep-wait, 4096 per-stage trace (CTA 0), 8192 consumers sleep-wait, 16384 escape-warp clock64 trace, 32768 escape warps off the empty barriers (timing only),
//   65536 consumer exponent-patch timing probe
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_kernel` (via alygg77/okc-muse-spark-kernels).
// Changes: kept sk2's structure (swapped operands: the x tile is the A operand, loaded by ldmatrix from a TMA-loaded 128B-swizzled tile; each CTA owns R rows x the whole K; a producer warp refills a ring of stages); replaced its 12-bit weight decode (8-exponent window) with our lossless fake-exponent formats and its escape-warp corrections with our per-CTA escape lists and exact corrections, and changed the ring sizing, the producer's issue order and the constants (baked in by NVRTC).
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, int NACC, int NST, u64 WB, int EBC, int CIB, int NG, int FMT, int LC, int FL>
__device__ __forceinline__ void sk11_body(const P& p) {
  constexpr int NW = NWR * WSK;
  constexpr int NE = (FL & 128) ? 0 : NWR * NT;
  constexpr int NC = NW * 32;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int T = UPW * NT;
  static_assert(FMT == 12 || T % 2 == 0, "F11: tile-units per warp-stage must be even");
  static_assert(NWR * NT <= 4, "at most 4 escape tiles (header holds 4 counts)");
  constexpr int TB = FMT == 12 ? 768 : 704;
  constexpr int WBY = NW * T * TB;
  constexpr int ELB = FMT == 12 ? 16 + NWR * NT * LC * 4 : NST * 16;   // escape lists (F12) / stage table (F11)
  constexpr int LSK = UPS == 1 ? 6 : UPS == 2 ? 7 : UPS == 4 ? 8 : 9;
  constexpr int XT = TOKR * 128;
  constexpr int LXU = TOKR == 8 ? 10 : TOKR == 16 ? 11 : 12;
  constexpr int XBY = UPS * XT;
  constexpr int SST = (XBY + WBY + EBC + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);
  constexpr int NSTP = NST;           // table entries (uint4 per stage)
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const u32 sraw = (u32)__cvta_generic_to_shared(smem_raw);
  const u32 sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const u32 bars = sbase + NSTAGE * SST;
  const u32 full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8;
  const u32 etab_s = pbar + 16;
  const uint4* etab_p = reinterpret_cast<const uint4*>(gbase + (etab_s - sbase));
  float* corr = reinterpret_cast<float*>(gbase + (etab_s - sbase) + ELB);
  float* red = reinterpret_cast<float*>(gbase);
  const u64 tstart = (FL & 2) ? gtime() : 0ull;
  const u8* Wp = WB ? cptr<WB>() : p.W;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = (int)blockIdx.x, ncta = (int)gridDim.x;
  const u8* wsrc = Wp + (size_t)cta * WBY;
  const size_t wstride = (size_t)ncta * WBY;

  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&p.xmap) : "memory");
    for (int i = 0; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + ((FL & 32768) ? 0 : NE)); }
    mbar_init(pbar, 1);
    mbar_init(cbar, NE > 0 ? NE : 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_tx(full0, XBY + WBY + CIB);
    bulk_g2s(sbase + XBY, wsrc, WBY, full0);
    tma_load_3d(sbase, &p.xmap, full0, 0, 0, 0);
    if constexpr (CIB > 0) bulk_g2s(sbase + XBY + WBY, p.E + (size_t)cta * CIB, CIB, full0);
    mbar_expect_tx(pbar, ELB);
    bulk_g2s(etab_s, reinterpret_cast<const u8*>(p.etab) + (size_t)cta * ELB, ELB, pbar);
  }
  __syncthreads();

  if (warp == NW) {   // ---------------- producer
    if (lane == 0) {
      for (int st = 1; st < NST; ++st) {
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) wait_s<FL>(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
        const u8* esrc = p.E;
        u32 eby = 0;
        if constexpr (FMT == 11) {
          if (st == NSTAGE) wait_s<FL>(pbar, 0);
          if (st < NSTAGE) { esrc = p.E + ((size_t)st * ncta + cta) * CIB; eby = CIB; }
          else { const uint4 e4 = etab_p[st]; esrc = p.E + (size_t)e4.x * 16; eby = e4.y; }
        }
        const u32 s0 = sbase + slot * SST;
        if ((FL & 4096) && cta == 0) p.trc[32768 + st * 8 + 0] = gtime();
        mbar_expect_tx(full0 + 8 * slot, XBY + WBY + eby);
        bulk_g2s(s0 + XBY, wsrc + (size_t)st * wstride, WBY, full0 + 8 * slot);
        tma_load_3d(s0, &p.xmap, full0 + 8 * slot, 0, 0, st * UPS);
        if (eby) bulk_g2s(s0 + XBY + WBY, esrc, eby, full0 + 8 * slot);
      }
    }
    __syncwarp();
    return;
  }

  if (FMT == 12 && warp > NW) {   // ---------------- F12 escape warps (rare escapes, lane = token)
    const int et = warp - NW - 1;
    const u32* el = reinterpret_cast<const u32*>(gbase + (etab_s - sbase));
    wait_e<FL>(pbar, 0);
    const int cnt = (int)el[et];
    const u32 ea = lane < cnt ? el[4 + et * LC + lane] : 0u;
    const u32 eb2 = (LC > 32 && lane + 32 < cnt) ? el[4 + et * LC + 32 + lane] : 0u;
    const int sa = lane < cnt ? (int)((ea & 0x1FFFu) >> LSK) : -1;
    const int sb = (LC > 32 && lane + 32 < cnt) ? (int)((eb2 & 0x1FFFu) >> LSK) : -1;
    float E1r[8], E2r[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) { E1r[r] = 0.f; E2r[r] = 0.f; }
    const float nd = -__uint_as_float(0x38000000u);   // -d = -2^-15 (fake domain): every F12 escape decodes to d
    const int tok = lane;
    for (int st = 0; st < NST; ++st) {
      const int slot = st % NSTAGE;
      wait_e<FL>(full0 + 8 * slot, (st / NSTAGE) & 1);
      u32 m0 = __ballot_sync(0xffffffffu, sa == st);
      u32 m1 = LC > 32 ? __ballot_sync(0xffffffffu, sb == st) : 0u;
      if ((m0 | m1) && !(FL & 64)) {
        const u32 s0 = sbase + slot * SST;
        while (m0 | m1) {
          u32 e;
          if (m0) { const int i = __ffs(m0) - 1; m0 &= m0 - 1; e = __shfl_sync(0xffffffffu, ea, i); }
          else { const int i = __ffs(m1) - 1; m1 &= m1 - 1; e = __shfl_sync(0xffffffffu, eb2, i); }
          const u32 kl = e & (u32)(UPS * 64 - 1), kc = kl & 63u, u = kl >> 6;
          const int row = (int)((e >> 13) & 7u);
          const float val = __uint_as_float(e & 0xFFFF0000u);
          float x = 0.f;
          if (tok < TOKR)
            x = __uint_as_float(lds16(s0 + u * XT + (u32)tok * 128 + ((((kc >> 3) ^ (u32)(tok & 7))) << 4) + (kc & 7u) * 2) << 16);
#pragma unroll
          for (int r = 0; r < 8; ++r)
            if (row == r) { E1r[r] = fmaf(nd, x, E1r[r]); E2r[r] = fmaf(val, x, E2r[r]); }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
    if (tok < TOKS) {
#pragma unroll
      for (int r = 0; r < 8; r += 4) {
        *reinterpret_cast<float4*>(corr + tok * RP + 8 * et + r) = make_float4(E1r[r], E1r[r + 1], E1r[r + 2], E1r[r + 3]);
        *reinterpret_cast<float4*>(corr + (TOKS + tok) * RP + 8 * et + r) = make_float4(E2r[r], E2r[r + 1], E2r[r + 2], E2r[r + 3]);
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(cbar);
    return;
  }

  if (FMT == 11 && warp > NW) {    // ---------------- F11 escape warps: tile et = rows 8 et .. 8 et + 7 of this CTA
    const int et = warp - NW - 1;
    const int g = lane >> 2, t = lane & 3;
    float E1[MT][4], E2[MT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int e = 0; e < 4; ++e) { E1[mt][e] = 0.f; E2[mt][e] = 0.f; }
    const u32 ND = 0xBC00u;   // -d = -2^-7 (fake domain), the decoded value of every escape
    // per stage: this tile's escape count n and entry offset (tile lists are padded to 4 entries) from the table
    auto cnt_of = [&](int st, int& n, u32& off) {
      const uint4 v = etab_p[st];
      const u32 c0 = v.z & 0xFFFFu, c1 = v.z >> 16, c2 = v.w & 0xFFFFu, c3 = v.w >> 16;
      const u32 p0 = (c0 + 3u) & ~3u, p1 = (c1 + 3u) & ~3u, p2 = (c2 + 3u) & ~3u;
      n = (int)(et == 0 ? c0 : et == 1 ? c1 : et == 2 ? c2 : c3);
      off = et == 0 ? 0u : et == 1 ? p0 : et == 2 ? p0 + p1 : p0 + p1 + p2;
    };
    // entries of group G for lane t: 4 consecutive u32 (slots 2t, 2t+1, 2t+8, 2t+9); invalid -> row 8 (no match)
    auto load_ent = [&](int st, u32 (*e)[4]) {
      int n; u32 off;
      cnt_of(st, n, off);
      const u32 eb = sbase + (st % NSTAGE) * SST + XBY + WBY + off * 4u + (u32)t * 16u;
#pragma unroll
      for (int G = 0; G < NG; ++G) {
        if (16 * G < n) {
          if constexpr ((FL & 2048) != 0) {
#pragma unroll
            for (int q = 0; q < 4; ++q) e[G][q] = ((u32)(G * 37 + q * 11 + t) & 0x1FFu) | 0x3C000000u;
          } else {
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(e[G][0]), "=r"(e[G][1]), "=r"(e[G][2]), "=r"(e[G][3]) : "r"(eb + (u32)G * 64u));
          }
#pragma unroll
          for (int q = 0; q < 4; ++q) if (16 * G + 4 * t + q >= n) e[G][q] = 8u << 9;
        }
      }
    };
    u32 ec[NG][4], en[NG][4];
    wait_e<FL>(pbar, 0);
    wait_e<FL>(full0, 0);
    if constexpr ((FL & 64) == 0) load_ent(0, ec);
    for (int st = 0; st < NST; ++st) {
      const int slot = st % NSTAGE;
      if (st > 0 || (FL & 64)) wait_e<FL>(full0 + 8 * slot, (st / NSTAGE) & 1);
      if ((FL & 4096) && cta == 0 && et == 0 && lane == 0) p.trc[32768 + st * 8 + 3] = gtime();
      if ((FL & 16384) && cta == 0 && et == 0 && lane == 0) p.trc[40960 + st * 8 + 0] = ctime();
      const u32 s0 = sbase + slot * SST;
      if constexpr ((FL & 64) != 0) {
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
        continue;
      }
      int n; u32 off;
      cnt_of(st, n, off);
      if ((FL & 16384) && cta == 0 && et == 0) { __syncwarp(); if (lane == 0) p.trc[40960 + st * 8 + 1] = ctime() + (u64)(n & 0); }
      const u32 xl = s0 + (u32)g * 128;
      // gathers for this stage (entries already in registers)
      u32 aa[NG][MT][4];
#pragma unroll
      for (int G = 0; G < NG; ++G) {
        if (16 * G < n) {
          u32 ad[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) ad[q] = xl + ((((ec[G][q] & 63u) << 1) | (((ec[G][q] >> 6) & 7u) << LXU)) ^ ((u32)g << 4));
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            const u32 o = (u32)mt * 2048u;
            if constexpr ((FL & 512) != 0) {
              aa[G][mt][0] = ad[0] + o; aa[G][mt][1] = ad[1] + o; aa[G][mt][2] = ad[2] + o; aa[G][mt][3] = ad[3] + o;
              continue;
            }
            aa[G][mt][0] = lds16(ad[0] + o) | (lds16(ad[1] + o) << 16);
            aa[G][mt][2] = lds16(ad[2] + o) | (lds16(ad[3] + o) << 16);
            if constexpr (TOKR >= 16) {
              aa[G][mt][1] = lds16(ad[0] + o + 1024u) | (lds16(ad[1] + o + 1024u) << 16);
              aa[G][mt][3] = lds16(ad[2] + o + 1024u) | (lds16(ad[3] + o + 1024u) << 16);
            } else {
              aa[G][mt][1] = 0u; aa[G][mt][3] = 0u;
            }
          }
        }
      }
      if ((FL & 16384) && cta == 0 && et == 0) {
        u32 acc = 0;
#pragma unroll
        for (int G = 0; G < NG; ++G) if (16 * G < n) acc ^= aa[G][0][0] ^ aa[G][MT - 1][3];
        __syncwarp();
        if (lane == 0) p.trc[40960 + st * 8 + 2] = ctime() + (u64)(acc & 0u);
      }
      // prefetch the next stage's entries if it has landed (non-blocking test)
      bool hn = false;
      if (st + 1 < NST) {
        const int s1 = (st + 1) % NSTAGE;
        if (mbar_testwait(full0 + 8 * s1, ((st + 1) / NSTAGE) & 1)) { load_ent(st + 1, en); hn = true; }
      }
      // overflow groups (rare): entries + gathers + mma while the slot is still held
      for (int G0 = 16 * NG; G0 < n; G0 += 16) {
        u32 e[4], ad[4];
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(e[0]), "=r"(e[1]), "=r"(e[2]), "=r"(e[3])
                     : "r"(s0 + XBY + WBY + off * 4u + (u32)(G0 + 4 * t) * 4u));
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          if (G0 + 4 * t + q >= n) e[q] = 8u << 9;
          ad[q] = xl + ((((e[q] & 63u) << 1) | (((e[q] >> 6) & 7u) << LXU)) ^ ((u32)g << 4));
        }
        u32 b1a, b1b, b2a, b2b;
        esc_b(e, g, ND, b1a, b1b, b2a, b2b);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          const u32 o = (u32)mt * 2048u;
          const u32 a0 = lds16(ad[0] + o) | (lds16(ad[1] + o) << 16);
          const u32 a2 = lds16(ad[2] + o) | (lds16(ad[3] + o) << 16);
          u32 a1 = 0u, a3 = 0u;
          if constexpr (TOKR >= 16) {
            a1 = lds16(ad[0] + o + 1024u) | (lds16(ad[1] + o + 1024u) << 16);
            a3 = lds16(ad[2] + o + 1024u) | (lds16(ad[3] + o + 1024u) << 16);
          }
          mma_bf16(E1[mt], a0, a1, a2, a3, b1a, b1b);
          mma_bf16(E2[mt], a0, a1, a2, a3, b2a, b2b);
        }
      }
      if ((FL & 16384) && cta == 0 && et == 0 && lane == 0) p.trc[40960 + st * 8 + 3] = ctime();
      __syncwarp();
      if (lane == 0 && !(FL & 32768)) mbar_arrive(empty0 + 8 * slot);
      if ((FL & 4096) && cta == 0 && et == 0 && lane == 0) p.trc[32768 + st * 8 + 4] = gtime();
      if ((FL & 16384) && cta == 0 && et == 0 && lane == 0) p.trc[40960 + st * 8 + 4] = ctime();
#pragma unroll
      for (int G = 0; G < NG; ++G) {
        if (16 * G < n) {
          if constexpr ((FL & 1024) != 0) {
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) E1[mt][0] += __uint_as_float((aa[G][mt][0] ^ aa[G][mt][1] ^ aa[G][mt][2] ^ aa[G][mt][3] ^ ec[G][0] ^ ec[G][1] ^ ec[G][2] ^ ec[G][3]) & 0x3FFFFFu);
          } else {
            u32 b1a, b1b, b2a, b2b;
            esc_b(ec[G], g, ND, b1a, b1b, b2a, b2b);
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
              mma_bf16(E1[mt], aa[G][mt][0], aa[G][mt][1], aa[G][mt][2], aa[G][mt][3], b1a, b1b);
              mma_bf16(E2[mt], aa[G][mt][0], aa[G][mt][1], aa[G][mt][2], aa[G][mt][3], b2a, b2b);
            }
          }
        }
      }
      if ((FL & 4096) && cta == 0 && et == 0 && lane == 0) p.trc[32768 + st * 8 + 5] = gtime();
      if ((FL & 16384) && cta == 0 && et == 0) {
        float acc = 0.f;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) acc += E1[mt][0] + E2[mt][3];
        __syncwarp();
        if (lane == 0) p.trc[40960 + st * 8 + 5] = ctime() + (u64)(acc != acc);
      }
      if (st + 1 < NST) {
        if (!hn) {
          wait_e<FL>(full0 + 8 * ((st + 1) % NSTAGE), ((st + 1) / NSTAGE) & 1);
          load_ent(st + 1, en);
        }
#pragma unroll
        for (int G = 0; G < NG; ++G)
#pragma unroll
          for (int q = 0; q < 4; ++q) ec[G][q] = en[G][q];
      }
    }
    if ((FL & 2) && et == 0 && lane == 0) p.trc[blockIdx.x * 8 + 7] = gtime();
    float* c1p = corr;
    float* c2p = corr + TOKS * RP;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      const int tok = mt * 16 + g, row = 8 * et + 2 * t;
      *reinterpret_cast<float2*>(c1p + tok * RP + row) = make_float2(E1[mt][0], E1[mt][1]);
      *reinterpret_cast<float2*>(c1p + (tok + 8) * RP + row) = make_float2(E1[mt][2], E1[mt][3]);
      *reinterpret_cast<float2*>(c2p + tok * RP + row) = make_float2(E2[mt][0], E2[mt][1]);
      *reinterpret_cast<float2*>(c2p + (tok + 8) * RP + row) = make_float2(E2[mt][2], E2[mt][3]);
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(cbar);
    return;
  }

  // ---------------- consumers
  const int wl = warp;
  const int wr = wl / WSK, wk = wl % WSK;
  const int g = lane >> 2, t4 = lane & 3;
  const int q8 = lane >> 3;
  const int chi = q8 >> 1;
  int trow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) trow[mt] = TOKR >= 16 ? mt * 16 + (lane & 7) + 8 * (q8 & 1) : (lane & 7);
  float acc[NACC][MT][NT][4];
#pragma unroll
  for (int c = 0; c < NACC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;
  u64 tfirst = 0;
  for (int st = 0; st < NST; ++st) {
    const int slot = st % NSTAGE;
    if constexpr ((FL & 8192) != 0) mbar_wait_sleep(full0 + 8 * slot, (st / NSTAGE) & 1);
    else mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    if ((FL & 2) && st == 0) tfirst = gtime();
    if ((FL & 4096) && cta == 0 && warp == 0 && lane == 0) p.trc[32768 + st * 8 + 1] = gtime();
    const u32 s0 = sbase + slot * SST;
    if constexpr ((FL & 16) != 0) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      continue;
    }
    const u32 wb = s0 + XBY + (u32)wl * (T * TB);
    u32 b[T][8];
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const int ui = wk * UPW + uu;
      const u32 xt = s0 + ui * XT;
      if constexpr (FMT == 12) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          uint4 L;
          uint2 N;
          const u32 tb = wb + (u32)(uu * NT + nt) * 768u;
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(L.x), "=r"(L.y), "=r"(L.z), "=r"(L.w) : "r"(tb + lane * 16));
          asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(N.x), "=r"(N.y) : "r"(tb + 512 + lane * 8));
          decode12(L.x, L.y, N.x, &b[uu * NT + nt][0]);
          decode12(L.z, L.w, N.y, &b[uu * NT + nt][4]);
        }
      }
#pragma unroll
      for (int j = 0; j < (FMT == 12 ? 0 : T / 2); ++j) {
        if ((2 * j) / NT != uu) continue;
        uint4 L0, L1;
        uint2 C01;
        u32 C2;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(L0.x), "=r"(L0.y), "=r"(L0.z), "=r"(L0.w) : "r"(wb + (2 * j) * 512 + lane * 16));
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(L1.x), "=r"(L1.y), "=r"(L1.z), "=r"(L1.w) : "r"(wb + (2 * j + 1) * 512 + lane * 16));
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(C01.x), "=r"(C01.y) : "r"(wb + T * 512 + j * 384 + lane * 8));
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(C2) : "r"(wb + T * 512 + j * 384 + 256 + lane * 4));
        if constexpr ((FL & 65536) != 0) {   // timing probe of the exponent-patch path (wrong results)
          u32 pe[2];
          const u32 eb0 = s0 + XBY + WBY;
          pe[0] = lds16(eb0 + ((lane * 7 + j * 64) & 255) * 2);
          pe[1] = lds16(eb0 + ((lane * 5 + 37 + j * 64) & 255) * 2);
          pe[0] = (lane & 3) ? (pe[0] & 0x7FFu) : 0u;
          pe[1] = (lane & 7) == 1 ? (pe[1] & 0x7FFu) : 0u;
          decode11p<2>(L0, L1, C01.x, C01.y, C2, b[2 * j], b[2 * j + 1], pe);
        } else {
          decode11(L0, L1, C01.x, C01.y, C2, b[2 * j], b[2 * j + 1]);
        }
      }
      u32 a[4][MT][4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          ldsm_x4(xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * TOKR + trow[mt]) & 7)) << 4),
                  a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      if (UPW == 1 && uu == 0) {
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt)
            mma_bf16(acc[(uu * 4 + j) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3],
                     b[uu * NT + nt][2 * j], b[uu * NT + nt][2 * j + 1]);
    }
    if (UPW > 1) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
    if ((FL & 4096) && cta == 0 && warp == 0 && lane == 0) p.trc[32768 + st * 8 + 2] = gtime();
  }
  if ((FL & 2) && warp == 0 && lane == 0) {
    p.trc[blockIdx.x * 8 + 0] = tstart;
    p.trc[blockIdx.x * 8 + 1] = tfirst;
    p.trc[blockIdx.x * 8 + 2] = gtime();
    u32 smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    p.trc[blockIdx.x * 8 + 6] = smid;
  }
  if constexpr (NE > 0) mbar_wait(cbar, 0);
  if ((FL & 2) && warp == 0 && lane == 0) p.trc[blockIdx.x * 8 + 5] = gtime();
  named_bar(1, NC);
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int row = (wr * NT + nt) * 8 + 2 * t4;
      const int tok = mt * 16 + g;
      float* r0 = red + (wk * TOKS + tok) * RP + row;
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = acc[0][mt][nt][e];
#pragma unroll
        for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
      }
      *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
    }
  named_bar(1, NC);
  constexpr int R4 = R / 4;
  const float sc = p.scale;
  const int M = p.M;
  for (int idx = tid; idx < R4 * M; idx += NC) {
    const int tt = idx / R4, row = (idx - tt * R4) * 4;
    float4 v = *reinterpret_cast<const float4*>(red + tt * RP + row);
#pragma unroll
    for (int w = 1; w < WSK; ++w) {
      const float4 pp = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
      v.x += pp.x; v.y += pp.y; v.z += pp.z; v.w += pp.w;
    }
    if constexpr (NE > 0) {
      const float4 e1 = *reinterpret_cast<const float4*>(corr + tt * RP + row);
      v.x += e1.x; v.y += e1.y; v.z += e1.z; v.w += e1.w;
      const float4 e2 = *reinterpret_cast<const float4*>(corr + (TOKS + tt) * RP + row);
      v.x += e2.x; v.y += e2.y; v.z += e2.z; v.w += e2.w;
    }
    u16* o = p.out + (size_t)tt * p.N + cta * R + row;
    uint2 pk;
    pk.x = pack_bf2(v.x * sc, v.y * sc);
    pk.y = pack_bf2(v.z * sc, v.w * sc);
    *reinterpret_cast<uint2*>(o) = pk;
  }
  if ((FL & 2) && tid == 0) p.trc[blockIdx.x * 8 + 4] = gtime();
}
// SPDX-SnippetEnd
"""

# ------------------------------------------------------------------------------------------------- configuration
# (R, NT, NWR, WSK, UPW, NACC, ring cap KB, flags) -- R = NWR*NT*8 rows per CTA; flags: 4 = baked W pointer + kernel FL
# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-SnippetCopyrightText: 2026 alygg77 <https://github.com/alygg77/okc-muse-spark-kernels>
# SPDX-License-Identifier: Apache-2.0
# Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `SK2_CONFIGS` / `SK2_PICK` tile configurations, with alygg77's `u32_r96` unpooler entry (alygg77/okc-muse-spark-kernels, kernels/muse_bf16_proj.py).
# Changes: sk3's (layer, m) entries of those tables with the ring cap and a flags word as columns.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
_f1_BASE_CFG = {   # sk3's configs (vjeux/alygg sk2 tables) for the shapes this module may own
    ("qkv_gate", 8): (24, 3, 1, 4, 2, 2, 144, 0),
    ("qkv_gate", 16): (24, 3, 1, 4, 2, 2, 200, 0),
    ("qkv_gate", 32): (24, 3, 1, 4, 2, 1, 208, 0),
    ("pooler", 8): (32, 2, 2, 2, 4, 2, 144, 0),
    ("pooler", 16): (32, 2, 2, 2, 4, 1, 192, 0),
    ("pooler", 32): (32, 2, 2, 2, 2, 1, 144, 0),
    ("unpooler", 16): (32, 2, 2, 2, 2, 1, 72, 4),
    ("unpooler", 32): (32, 2, 2, 2, 2, 1, 96, 4),
    ("o", 8): (32, 4, 1, 4, 1, 2, 88, 0),
    ("o", 16): (32, 2, 2, 1, 4, 2, 68, 4),
    ("o", 32): (32, 2, 2, 2, 2, 1, 100, 0),
}
# SPDX-SnippetEnd
_f1_F12 = 1 << 20
# shipped configs: F12 stream + register-preloaded rare-escape lists (iteration 5+); same-run wins vs bo5 in NOTES.md
_f1_CFG = {k: v[:7] + (v[7] | _f1_F12,) for k, v in _f1_BASE_CFG.items()}
_f1_CFG[("qkv_gate", 32)] = (24, 3, 1, 4, 2, 1, 208, _f1_F12 | 4)
_f1_CFG[("pooler", 16)] = (32, 2, 2, 2, 4, 1, 192, _f1_F12)
_f1_CFG[("unpooler", 32)] = (32, 2, 2, 2, 2, 1, 96, _f1_F12 | 4)
_f1_CFG[("o", 8)] = (32, 4, 1, 4, 1, 2, 88, _f1_F12 | 4)
_f1_CFG[("o", 16)] = (32, 4, 1, 4, 1, 2, 88, _f1_F12)
_f1_CFG[("o", 32)] = (32, 2, 2, 2, 2, 1, 100, _f1_F12 | 4)
# same-run wins vs bo5 on both sandboxes (bench.py, iteration 5); o m8 stays with bo5 (Nathen v18o) and unpooler m32
# with bo5's sk3 (wins standalone by ~0.05, loses by 0.2-0.3 in both integrated runs)
_f1_OWN = {("qkv_gate", 8), ("qkv_gate", 16), ("qkv_gate", 32), ("pooler", 8), ("pooler", 16), ("pooler", 32),
       ("unpooler", 16), ("o", 16), ("o", 32)}
if os.environ.get("SK11_OWN"):   # experiments: e.g. "pooler:32,unpooler:32"
    _f1_OWN = set()
    for _t in filter(None, os.environ["SK11_OWN"].split(",")):
        _l, _m = _t.split(":")
        _f1_OWN.add((_l, int(_m)))
_f1_K_OF = {"pooler": 8192, "unpooler": 4096, "o": 1024, "qkv_gate": 8192}
_f1_N_OF = {"pooler": 4096, "unpooler": 8192, "o": 8192, "qkv_gate": 2304}
_f1_FLAG_BAKE = 4
_f1_NGCAP = 4                   # escape groups (16 escapes of one 8-row tile) gathered up front per stage
_f1_SMEM_LIMIT = 232_448
_f1_FAKE = 120                  # F11 fake exponent bias (agent G)
_f1_FAKE12 = 112                # F12 fake exponent bias (agent G)
_f1_TB = 704
_f1_FLAG_F12 = 1 << 20          # cfg flags bit: 12-bit F12 stream + rare-escape lists (else F11)


def _f1__geom(cfg, layer, m, lc=32):
    R, NT, NWR, WSK, UPW, NACC, cap, fl = cfg
    tokr = 8 if m <= 8 else 16 if m <= 16 else 32
    mt = 2 if tokr == 32 else 1
    ups = WSK * UPW
    nw = NWR * WSK
    t = UPW * NT
    f12 = bool(fl & _f1_FLAG_F12)
    wby = nw * t * (768 if f12 else _f1_TB)
    xby = ups * tokr * 128
    nst = _f1_K_OF[layer] // (ups * 64)
    toks = 16 * mt
    rp = R + ((40 - (R % 32)) % 32)
    elb = 16 + (R // 8) * lc * 4 if f12 else nst * 16
    return dict(tokr=tokr, mt=mt, ups=ups, nw=nw, t=t, wby=wby, xby=xby, nst=nst, toks=toks, rp=rp,
                nstp=nst, ne=(R // 8), elb=elb, f12=f12, lc=lc)


def _f1__nstage(cfg, g, ebc):
    sst = (g["xby"] + g["wby"] + ebc + 1023) // 1024 * 1024
    extra = 1024 + 16 * 16 + 16 + g["elb"] + 2 * g["toks"] * g["rp"] * 4 + 64
    ring = min(cfg[6] * 1024, _f1_SMEM_LIMIT - extra)
    return max(2, min(16, g["nst"], ring // sst)), sst


def _f1__smem(g, nstage, sst):
    return 1024 + nstage * sst + 16 * nstage + 16 + g["elb"] + 2 * g["toks"] * g["rp"] * 4


# ------------------------------------------------------------------------------------------------- weight packing
def _f1__window(e, width):
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    return max(range(1, 256 - width), key=lambda q: int(cs[q + width] - cs[q]))


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Copied from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_tileunits` (via alygg77/okc-muse-spark-kernels).
# Changes: renamed; docstring shortened.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def _f1__tileunits(w):
    """W [N, K] -> [N/8, K/64, 32 lanes, 16] in mma.m16n8k16 B-fragment order (sk2_tileunits)."""
    n, k = w.shape
    t = w.view(n // 8, 8, k // 64, 4, 2, 4, 2)
    return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)
# SPDX-SnippetEnd


def _f1_pack_code32(s, h):
    """agent G's F11 code words: s, h [..., 32] (weight 4g + b = byte lane b of hi word g) -> int32 [..., 3]"""
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


def _f1_compress11(w):
    """F11 planes in tile-unit order [N/8, K/64, 32, 16]: lo, s, h, esc mask, true bits u, elo."""
    tu = _f1__tileunits(w.contiguous())
    u = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    elo = _f1__window(e, 8)
    c = e - elo
    esc = (c < 0) | (c > 7)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m), "s": torch.where(esc, torch.zeros_like(s), s),
            "h": c >> 1, "esc": esc, "u": u, "elo": elo}


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_stream` (via alygg77/okc-muse-spark-kernels).
# Changes: gathers our 11-bit planes per tile unit (low bytes, plus sign / exponent codes packed three words per 32 weights) instead of sk2's pre-packed 768-byte tile units; the stage / CTA / warp / unit index arithmetic is sk2's.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def _f1__wstream(f, n_rows, R, NT, NWR, WSK, UPW):
    """[stage][cta][warp][T tile-units lo (512 B each) | T/2 code groups (W0W1 lane x 8 | W2 lane x 4)]"""
    dev = f["lo"].device
    U = f["lo"].shape[1]
    ups = WSK * UPW
    nw = NWR * WSK
    T = UPW * NT
    S, C = U // ups, n_rows // R
    ar = lambda v: torch.arange(v, device=dev)
    s = ar(S).view(-1, 1, 1, 1, 1)
    c = ar(C).view(1, -1, 1, 1, 1)
    w = ar(nw).view(1, 1, -1, 1, 1)
    uu = ar(UPW).view(1, 1, 1, -1, 1)
    nt = ar(NT).view(1, 1, 1, 1, -1)
    tile = (c * (R // 8) + (w // WSK) * NT + nt).expand(S, C, nw, UPW, NT)
    unit = (s * ups + (w % WSK) * UPW + uu).expand(S, C, nw, UPW, NT)
    lo = f["lo"][tile, unit].reshape(S, C, nw, T, 32, 16).to(torch.uint8)
    sg = f["s"][tile, unit].reshape(S, C, nw, T // 2, 2, 32, 16).permute(0, 1, 2, 3, 5, 4, 6).reshape(S, C, nw, T // 2, 32, 32)
    hg = f["h"][tile, unit].reshape(S, C, nw, T // 2, 2, 32, 16).permute(0, 1, 2, 3, 5, 4, 6).reshape(S, C, nw, T // 2, 32, 32)
    words = _f1_pack_code32(sg, hg)                                            # [S, C, nw, T/2, 32, 3]
    w01 = words[..., 0:2].contiguous().view(torch.uint8).reshape(S, C, nw, T // 2, 256)
    w2 = words[..., 2].contiguous().view(torch.uint8).reshape(S, C, nw, T // 2, 128)
    codes = torch.cat([w01, w2], -1).reshape(S, C, nw, T // 2 * 384)
    blob = torch.cat([lo.reshape(S, C, nw, T * 512), codes], -1)
    return blob.contiguous().view(-1)
# SPDX-SnippetEnd


def _f1__fake_bits(u, elo):
    """true bf16 bits (int32) -> fake bf16 bits (true * 2^(FAKE - elo)), or None if not exactly representable."""
    t = (u.to(torch.int32) << 16).view(torch.float32)
    f = t * (2.0 ** (_f1_FAKE - elo))
    fb = f.to(torch.bfloat16)
    ok = torch.isfinite(f) & (fb.float() == f) & ((fb.float() * (2.0 ** (elo - _f1_FAKE))) == t)
    if not bool(ok.all()):
        return None
    return fb.view(torch.int16).to(torch.int32) & 0xFFFF


def _f1__estream(f, n_rows, R, UPS, nstage):
    """Escape blocks per (stage, cta): per 8-row tile a list of 4-byte entries kl | row << 9 | fake << 16, padded to a
    multiple of 4 entries (lane t of the escape warp reads entries 16G + 4t .. +3 as one 16-B load = slots 2t, 2t+1,
    2t+8, 2t+9 of group G); within a tile, groups of 4 consecutive entries come from one bank bucket (kl >> 1) & 3 so
    the 4 lanes t of one gather read 4 different buckets (conflict-free). Blocks for stage < nstage have a fixed
    capacity CIB at (s * C + c) * CIB; later blocks follow. Per-CTA table etab[c][s] = (offset / 16, bytes,
    cnt0 | cnt1 << 16, cnt2 | cnt3 << 16)."""
    dev = f["u"].device
    idx = f["esc"].nonzero()
    tile, unit, lane, i = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    row = tile * 8 + (lane >> 2)
    k = unit * 64 + 16 * (i >> 2) + 8 * ((i >> 1) & 1) + 2 * (lane & 3) + (i & 1)
    fb = _f1__fake_bits(f["u"][idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]], f["elo"])
    if fb is None:
        return None
    K = f["u"].shape[1] * 64
    S, C, NE = K // (UPS * 64), n_rows // R, R // 8
    cta, et, rit = row // R, (row % R) // 8, row % 8
    s, kl = k // (UPS * 64), k % (UPS * 64)
    ent = (kl | (rit << 9)).to(torch.int64) | (fb.to(torch.int64) << 16)
    bk = (s * C + cta) * NE + et                               # (stage, cta, tile) key
    b = (kl >> 1) & 3
    o1 = torch.argsort(((bk * 4 + b) << 12) | kl)
    bk, b, ent = bk[o1], b[o1], ent[o1]
    gk = bk * 4 + b
    gcnt = torch.bincount(gk, minlength=S * C * NE * 4)
    gst = torch.cumsum(gcnt, 0) - gcnt
    rank = torch.arange(gk.numel(), device=dev) - gst[gk]
    key = (rank // 4) * 16 + b * 4 + (rank % 4)
    o2 = torch.argsort((bk << 32) | key)
    bk, ent = bk[o2], ent[o2]
    cnt = torch.bincount(bk, minlength=S * C * NE).view(S, C, NE)
    if int(cnt.max()) > 65535 or NE > 4:
        return None
    pcnt = (cnt + 3) // 4 * 4                                   # padded per-tile lists
    bbytes = 4 * pcnt.sum(-1)                                   # [S, C], multiple of 16
    ns0 = min(nstage, S)
    cib = int(bbytes[:ns0].max())
    off = torch.zeros(S, C, dtype=torch.long, device=dev)
    off[:ns0] = (torch.arange(ns0, device=dev).view(-1, 1) * C + torch.arange(C, device=dev).view(1, -1)) * cib
    base = ns0 * C * cib
    if S > ns0:
        vb = bbytes[ns0:].reshape(-1)
        off[ns0:] = (base + torch.cumsum(vb, 0) - vb).view(S - ns0, C)
        total = base + int(vb.sum())
    else:
        total = base
    ebc = max(cib, int(bbytes.max()), 16)
    buf = torch.full((max(total, 16) // 4,), 8 << 9, dtype=torch.int64, device=dev)   # padding entries: row 8
    tst = torch.cumsum(cnt.view(-1), 0) - cnt.view(-1)          # first sorted index of each (s, c, et)
    ppre = torch.cumsum(pcnt, -1) - pcnt                         # padded entries before tile et in the block
    pos = (off.view(-1, 1) // 4 + ppre.view(S * C, NE)).view(-1)  # word offset of each (s, c, et) list
    j = torch.arange(bk.numel(), device=dev) - tst[bk]
    buf[pos[bk] + j] = ent
    buf = torch.where(buf >= 2 ** 31, buf - 2 ** 32, buf).to(torch.int32)
    c4 = torch.zeros(S, C, 4, dtype=torch.int64, device=dev)
    c4[..., :NE] = cnt
    etab = torch.zeros(C, S, 4, dtype=torch.int64, device=dev)
    etab[..., 0] = (off // 16).t()
    etab[..., 1] = bbytes.t()
    etab[..., 2] = (c4[..., 0] | (c4[..., 1] << 16)).t()
    etab[..., 3] = (c4[..., 2] | (c4[..., 3] << 16)).t()
    etab = torch.where(etab >= 2 ** 31, etab - 2 ** 32, etab).to(torch.int32)
    cm = cnt.view(-1).float()
    return {"buf": buf.contiguous(), "etab": etab.contiguous(), "cib": cib, "ebc": ebc, "n": int(cnt.sum()),
            "cmax": int(cnt.max()), "cq": float(torch.quantile(cm[:min(cm.numel(), 1 << 24)], 0.999))}


def _f1_compress12(w):
    """F12 planes in tile-unit order [N/8, K/64, 32, 16] (agent G's fake-exponent 12-bit: 16-exponent window)."""
    tu = _f1__tileunits(w.contiguous())
    u = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    elo = _f1__window(e, 16)
    c = e - elo
    esc = (c < 0) | (c > 15)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m),
            "nib": torch.where(esc, torch.zeros_like(u), (s << 3) | (c >> 1)), "esc": esc, "u": u, "elo": elo}


def _f1_pack_nib8(nib):
    """agent G's F12 nibble word: nib [..., 8] (s << 3 | h) -> int32: byte b = h_b | s_b << 7 | h_{4+b} << 4 | s_{4+(b+3)%4} << 3"""
    nib = nib.to(torch.int64)
    s, h = nib >> 3, nib & 7
    word = torch.zeros(nib.shape[:-1], dtype=torch.int64, device=nib.device)
    for b in range(4):
        word |= (h[..., b] | (s[..., b] << 7) | (h[..., 4 + b] << 4) | (s[..., 4 + (b + 3) % 4] << 3)) << (8 * b)
    return torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_stream` (via alygg77/okc-muse-spark-kernels).
# Changes: gathers our separate low-byte and 4-bit-code planes per tile unit (codes repacked into nibble words) and concatenates them, instead of indexing sk2's pre-packed 768-byte tile units; the stage / CTA / warp / unit index arithmetic is sk2's.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def _f1__wstream12(f, n_rows, R, NT, NWR, WSK, UPW):
    """[stage][cta][warp][T tile-units: 512 lo bytes (lane x 16) | 256 B nibble words (lane x 8)]"""
    dev = f["lo"].device
    U = f["lo"].shape[1]
    ups, nw, T = WSK * UPW, NWR * WSK, UPW * NT
    S, C = U // ups, n_rows // R
    ar = lambda v: torch.arange(v, device=dev)
    s = ar(S).view(-1, 1, 1, 1, 1)
    c = ar(C).view(1, -1, 1, 1, 1)
    w = ar(nw).view(1, 1, -1, 1, 1)
    uu = ar(UPW).view(1, 1, 1, -1, 1)
    nt = ar(NT).view(1, 1, 1, 1, -1)
    tile = (c * (R // 8) + (w // WSK) * NT + nt).expand(S, C, nw, UPW, NT)
    unit = (s * ups + (w % WSK) * UPW + uu).expand(S, C, nw, UPW, NT)
    lo = f["lo"][tile, unit].reshape(S, C, nw, T, 512).to(torch.uint8)
    nw8 = _f1_pack_nib8(f["nib"][tile, unit].reshape(S, C, nw, T, 32, 2, 8))          # [S, C, nw, T, 32, 2]
    nb = nw8.contiguous().view(torch.uint8).reshape(S, C, nw, T, 256)
    return torch.cat([lo, nb], -1).contiguous().view(-1)
# SPDX-SnippetEnd


def _f1__elist12(f, n_rows, R, UPS):
    """Per CTA: [4 x u32 per-tile counts][NE tiles x LC u32 entries k | row << 13 | fake_bits << 16] (LC = 32 or 64)."""
    dev = f["u"].device
    idx = f["esc"].nonzero()
    tile, unit, lane, i = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    row = tile * 8 + (lane >> 2)
    k = unit * 64 + 16 * (i >> 2) + 8 * ((i >> 1) & 1) + 2 * (lane & 3) + (i & 1)
    t = (f["u"][idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]].to(torch.int32) << 16).view(torch.float32)
    fv = t * (2.0 ** (_f1_FAKE12 - f["elo"]))
    fb16 = fv.to(torch.bfloat16)
    if not bool((torch.isfinite(fv) & (fb16.float() == fv) & (fb16.float() * (2.0 ** (f["elo"] - _f1_FAKE12)) == t)).all()):
        return None
    fb = fb16.view(torch.int16).to(torch.int64) & 0xFFFF
    C, NE = n_rows // R, R // 8
    cta, et, rit = row // R, (row % R) // 8, row % 8
    key = (cta * NE + et).long()
    cnt = torch.bincount(key, minlength=C * NE)
    cmax = int(cnt.max()) if cnt.numel() else 0
    if cmax > 64 or k.numel() and int(k.max()) >= 8192:
        return None
    lc = 32 if cmax <= 32 else 64
    o = torch.argsort(key * 8192 + k)
    key, ent = key[o], (k | (rit << 13)).long()[o] | (fb[o] << 16)
    st0 = torch.cumsum(cnt, 0) - cnt
    j = torch.arange(key.numel(), device=dev) - st0[key]
    buf = torch.zeros(C, 4 + NE * lc, dtype=torch.int64, device=dev)
    buf[:, :NE] = cnt.view(C, NE)
    buf.view(-1)[(key // NE) * (4 + NE * lc) + 4 + (key % NE) * lc + j] = ent
    buf = torch.where(buf >= 2 ** 31, buf - 2 ** 32, buf).to(torch.int32)
    return {"buf": buf.contiguous(), "lc": lc, "n": int(cnt.sum()), "cmax": cmax}


# ------------------------------------------------------------------------------------------------- NVRTC module
class _f1__Module:
    def __init__(self, src, name="sk11.cu", opts=()):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        o = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        o += [x.encode() for x in opts]
        if os.environ.get("CU_LINEINFO"):
            o.append(b"-lineinfo")
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(o), o)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        self.log = log.decode(errors="replace").strip("\x00 \n")
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + self.log)
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        self.cubin = cubin
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err
        self.funcs = {}

    def func(self, name, smem):
        key = (name, smem)
        f = self.funcs.get(key)
        if f is None:
            d = self.driver
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            assert int(err) == 0, (name, err)
            if smem > 48 * 1024:
                (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
                assert int(err) == 0, (name, smem, err)
            self.funcs[key] = f
        return f

    def attrs(self, name, smem=0):
        d = self.driver
        f = self.func(name, smem)
        out = {}
        for k in ("NUM_REGS", "LOCAL_SIZE_BYTES"):
            err, v = d.cuFuncGetAttribute(getattr(d.CUfunction_attribute, "CU_FUNC_ATTRIBUTE_" + k), f)
            out[k] = v
        return out

    def launch(self, f, grid, block, smem, params):
        d = self.driver
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        assert int(err) == 0, err


def _f1__tmap_x3d(ptr, m, k, box_rows, units):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(3), ptr,
        [d.cuuint64_t(64), d.cuuint64_t(m), d.cuuint64_t(k // 64)], [d.cuuint64_t(k * 2), d.cuuint64_t(128)],
        [d.cuuint32_t(64), d.cuuint32_t(box_rows), d.cuuint32_t(units)],
        [d.cuuint32_t(1), d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


_f1__TM = {}


def _f1__xmap(x, m, k, box_rows, units):
    key = (x.data_ptr(), m, k, box_rows, units)
    b = _f1__TM.get(key)
    if b is None:
        if len(_f1__TM) > 512:
            _f1__TM.clear()
        b = _f1__TM[key] = _f1__tmap_x3d(x.data_ptr(), m, k, box_rows, units)
    return b


# ------------------------------------------------------------------------------------------------- prepare / run
def _f1_prepare_sk11(weights, cfgs=None, verbose=False, variants=None, selftest=True):
    """cfgs: {(layer, m): cfg} (default: CFG restricted to OWN). variants: extra {(layer, m, tag): cfg} instances.
    Never raises: any failure (packing, NVRTC, smem, self-test) leaves the shape to the base (run_sk11 -> None)."""
    try:
        st = _f1__prepare_impl(weights, cfgs, verbose, variants)
    except Exception as ex:   # noqa: BLE001
        print(f"[sk11] prepare failed, base keeps every shape: {type(ex).__name__}: {ex}", flush=True)
        return {"ent": {}}
    if selftest:
        for (layer, m, tag), e in st["ent"].items():
            if not e["ok"]:
                continue
            try:
                g = torch.Generator(device="cuda").manual_seed(1234 + m)
                x = torch.randn((m, e["k"]), generator=g, device="cuda").to(torch.bfloat16)
                out = _f1_run_sk11(x, {"m": m, "n": e["n"], "k": e["k"], "layer": layer}, st, tag=tag or "__self")
                ref = x.float() @ weights[layer][: e["n"]].float().T
                err = float((out.float() - ref).norm() / ref.norm())
                e["ok"] = err < 5e-3
                if not e["ok"]:
                    print(f"[sk11] self-test failed for {(layer, m, tag)}: rel err {err:.3e}", flush=True)
            except Exception as ex:   # noqa: BLE001
                e["ok"] = False
                print(f"[sk11] self-test error for {(layer, m, tag)}: {ex}", flush=True)
        torch.cuda.synchronize()
    return st


def _f1__prepare_impl(weights, cfgs=None, verbose=False, variants=None):
    cfgs = dict({k: v for k, v in _f1_CFG.items() if k in _f1_OWN} if cfgs is None else cfgs)
    inst = [(layer, m, "", cfg) for (layer, m), cfg in cfgs.items()]
    if variants:
        inst += [(layer, m, tag, cfg) for (layer, m, tag), cfg in variants.items()]
    st = {"ent": {}}
    if any(c[-1] & 2 for c in list(cfgs.values()) + list((variants or {}).values())):
        st["trc_buf"] = torch.zeros(65536, dtype=torch.int64, device="cuda")
        st["trc"] = st["trc_buf"].data_ptr()
    comps, wss, ess = {}, {}, {}
    for layer, m, tag, cfg in inst:
        R, NT, NWR, WSK, UPW, NACC, cap, fl = cfg
        assert R == NWR * NT * 8 and m in (8, 16, 32)
        n_rows = _f1_N_OF[layer]
        f12 = bool(fl & _f1_FLAG_F12)
        ckey = (layer, f12)
        if ckey not in comps:
            comps[ckey] = (_f1_compress12 if f12 else _f1_compress11)(weights[layer][:n_rows])
        f = comps[ckey]
        if R // 8 > 4 or WSK * UPW * 64 > 512 or n_rows % R or (not f12 and (UPW * NT) % 2):
            continue
        wkey = (layer, f12, R, NT, NWR, WSK, UPW)
        if wkey not in wss:
            wss[wkey] = (_f1__wstream12 if f12 else _f1__wstream)(f, n_rows, R, NT, NWR, WSK, UPW)
        ws = wss[wkey]
        if f12:
            ekey = (layer, True, R, WSK * UPW)
            if ekey not in ess:
                ess[ekey] = _f1__elist12(f, n_rows, R, WSK * UPW)
            es = ess[ekey]
            if es is None:
                print(f"[sk11] {(layer, m, tag)}: F12 escapes do not fit (count / fake domain): left to the base", flush=True)
                continue
            g = _f1__geom(cfg, layer, m, es["lc"])
            es = dict(es, ebc=0, cib=0, etab=es["buf"], cq=0.0)
            nstage, sst = _f1__nstage(cfg, g, 0)
        else:
            g = _f1__geom(cfg, layer, m)
            # escape capacity depends on nstage and nstage on the capacity: iterate once
            nstage, sst = _f1__nstage(cfg, g, 1024)
            ekey = (layer, False, R, g["ups"], nstage)
            if ekey not in ess:
                ess[ekey] = _f1__estream(f, n_rows, R, g["ups"], nstage)
            es = ess[ekey]
            if es is None:
                print(f"[sk11] {(layer, m, tag)}: escapes not representable in the fake domain: left to the base", flush=True)
                continue
            nst2, sst = _f1__nstage(cfg, g, es["ebc"])
            if nst2 != nstage:
                nstage = nst2
                ekey = (layer, False, R, g["ups"], nstage)
                if ekey not in ess:
                    ess[ekey] = _f1__estream(f, n_rows, R, g["ups"], nstage)
                es = ess[ekey]
                nst3, sst = _f1__nstage(cfg, g, es["ebc"])
                if nst3 != nstage:
                    continue
        smem = _f1__smem(g, nstage, sst)
        if smem > _f1_SMEM_LIMIT:
            print(f"[sk11] {(layer, m, tag)} needs {smem} B smem: left to the base", flush=True)
            continue
        nthr = (g["nw"] + 1 + (0 if fl & 128 else g["ne"])) * 32
        ng = 1 if f12 else (int(os.environ.get("SK11_NG", "0")) or min(_f1_NGCAP, max(1, math.ceil(es["cq"] / 16))))
        ncta = n_rows // R
        kn = f"sk11_{layer}_{m}{tag}"
        st["ent"][(layer, m, tag)] = dict(cfg=cfg, ws=ws, es=es, elo=f["elo"], n=n_rows, k=_f1_K_OF[layer], g=g, nthr=nthr,
                                          ncta=ncta, smem=smem, kn=kn, nstage=nstage, sst=sst, ok=True, ng=ng,
                                          scale=2.0 ** (f["elo"] - (_f1_FAKE12 if f12 else _f1_FAKE)))
    src = _f1_SRC
    for (layer, m, tag), e in st["ent"].items():
        R, NT, NWR, WSK, UPW, NACC, cap, fl = e["cfg"]
        g = e["g"]
        wb = f"{e['ws'].data_ptr()}ull" if fl & _f1_FLAG_BAKE else "0ull"
        src += (f'extern "C" __global__ void __launch_bounds__({e["nthr"]}, 1) {e["kn"]}(const __grid_constant__ P p) '
                f'{{ sk11_body<{g["mt"]}, {g["tokr"]}, {NT}, {NWR}, {WSK}, {UPW}, {e["nstage"]}, {NACC}, {g["nst"]}, '
                f'{wb}, {e["es"]["ebc"]}, {e["es"]["cib"]}, {e["ng"]}, {12 if g["f12"] else 11}, {g["lc"]}, '
                f'{fl & ~(_f1_FLAG_BAKE | _f1_FLAG_F12)}>(p); }}\n')
    st["src"] = src
    st["mod"] = _f1__Module(src)
    for key, e in st["ent"].items():
        e["f"] = st["mod"].func(e["kn"], e["smem"])
        if verbose:
            a = st["mod"].attrs(e["kn"], e["smem"])
            g = e["g"]
            wbits = 8.0 * (e["ws"].numel() + e["es"]["buf"].numel() * 4) / (e["n"] * e["k"])
            if e["g"]["f12"]:
                wbits = 8.0 * e["ws"].numel() / (e["n"] * e["k"])
            print(f"[sk11] {key} cfg {e['cfg']} nstage {e['nstage']} sst {e['sst']} smem {e['smem']} thr {e['nthr']} "
                  f"ncta {e['ncta']} regs {a['NUM_REGS']} local {a['LOCAL_SIZE_BYTES']} esc {e['es']['n']} "
                  f"cib {e['es']['cib']} ebc {e['es']['ebc']} cmax {e['es']['cmax']} ng {e['ng']} lc {e['g']['lc']} bits/w {wbits:.3f}", flush=True)
    torch.cuda.synchronize()
    return st


def _f1_run_sk11(x, config, state, tag=""):
    m, layer = config["m"], config["layer"]
    if (layer, m) not in _f1_OWN and not tag:
        return None
    e = state["ent"].get((layer, m, "" if tag == "__self" else tag))
    if e is None or not e["ok"] or config["n"] != e["n"] or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    n, k = e["n"], e["k"]
    g = e["g"]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    tm = _f1__xmap(x, m, k, g["tokr"], g["ups"])
    params = tm + struct.pack("<QQQQQQfiiiii", e["ws"].data_ptr(), x.data_ptr(), out.data_ptr(), e["es"]["buf"].data_ptr(),
                              e["es"]["etab"].data_ptr(), state.get("trc", 0), e["scale"], m, n, k, 0, 0)
    state["mod"].launch(e["f"], e["ncta"], e["nthr"], e["smem"], params)
    return out


# ======================================================================== module X: X/sk3.py
"""sk3 (agent X): skinny GEMM for qkv_gate / pooler / unpooler (full) / o at m = 8 / 16 / 32 on 12-bit lossless weights.

    state = prepare_sk3(weights)            # weights: the problem's dict; never raises
    out   = run_sk3(x, config, state)       # -> [M, N] bf16, or None where the base (bo1) is kept

Owned shapes (same-run wins vs bo1, see NOTES.md iteration 6): pooler m32, unpooler m32, qkv_gate m32, unpooler m16,
o m16, qkv_gate m8. Everything else returns None.

Kernel derived from vjeux/okc-2026 `sk2` (via alygg77's public OKC submission), Apache-2.0: swapped operands
(A = x tile 16 tokens x 16 k via ldmatrix from a TMA-loaded 128B-swizzled tile, B = 8 weight rows x 16 k decoded from
the 12-bit stream), each CTA owns R rows x the whole K, a producer warp refills a ring of stages (x tiles + this CTA's
12-bit W), escape warps add the exact corrections for out-of-window weights. Ported to NVRTC + driver API by agent X,
with: CF4 (corrections kept per escape warp in a [token][RP] layout -> float4, bank-conflict-free reads in the output
loop) and a baked (compile-time) weight-stream pointer for unpooler 16/32 and o16. Experimental switches that lost
(kept for A/B, all measured in NOTES.md): x multicast (MC), split-K clusters (CL, CLR, EW), inline escapes (EIN),
one-escape-warp-per-slot (ESC1), stage-parallel consumers (SPL), early slot release (EARLY), stage rotation (ROT),
x-first issue order, global-x escape warps (batched variant: do not use).
"""
import ctypes
import os
import struct

import torch

# ----------------------------------------------------------------------------------------------------------- CUDA
_x3_SRC_HEAD = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) P { TMap xmap; const u8* W; const u16* x; u16* out; const int2* ptab; u64* trc;
                         int pmax; u32 base4; int M; int N; int K; int pad; };

template <u64 A> __device__ __forceinline__ const u8* cptr() { const u8* p; asm("mov.u64 %0, %1;" : "=l"(p) : "n"(A)); return p; }
__device__ __forceinline__ u64 gtime() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

__device__ __forceinline__ void ldsm_x4(u32 addr, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 sel) {
  u32 d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel)); return d;
}
__device__ __forceinline__ u32 lop3_sel(u32 a, u32 b, u32 c) {  // c ? b : a, bitwise
  u32 d; asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d;
}
// 8 weights: low bytes of weights 0..3 in lw0, 4..7 in lw1, codes in nw (low nibble of byte b -> weight b, high nibble
// -> weight 4+b; code = s<<3 | o). Output 4 bf16x2 (w0,w1) (w2,w3) (w4,w5) (w6,w7).
__device__ __forceinline__ void decode8(u32 lw0, u32 lw1, u32 nw, u32 base4, u32* r) {
  u32 ha = lop3_sel(nw & 0x07070707u, nw << 4, 0x80808080u) + base4;
  u32 hb = lop3_sel((nw >> 4) & 0x07070707u, nw, 0x80808080u) + base4;
  r[0] = prmt(lw0, ha, 0x5140); r[1] = prmt(lw0, ha, 0x7362);
  r[2] = prmt(lw1, hb, 0x5140); r[3] = prmt(lw1, hb, 0x7362);
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a)); return r; }

__device__ __forceinline__ void mbar_init(u32 bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `mbar_test` / `mbar_wait` / `mbar_wait_sleep` / `mbar_wait_cluster` (MBAR_HELPERS) (via alygg77/okc-muse-spark-kernels).
// Changes: renamed types (u32); mbar_wait written on one line; the helpers vjeux keeps between them (esc_delta, mbar_arrive, named_bar, mbar_expect_tx, L2 policies, bulk copies) left out.
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
__device__ __forceinline__ bool mbar_test(u32 bar, u32 parity) {
  u32 ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(u32 bar, u32 parity) { while (!mbar_test(bar, parity)) {} }
__device__ __forceinline__ void mbar_wait_sleep(u32 bar, u32 parity) {
  u32 ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(0x989680u) : "memory");
  } while (!ok);
}
__device__ __forceinline__ void mbar_wait_cluster(u32 bar, u32 parity) {
  u32 ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n"
                 " selp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  } while (!ok);
}
// SPDX-SnippetEnd
__device__ __forceinline__ void mbar_arrive(u32 bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}
// arrive on the mbarrier at the same offset in cluster CTA `rank` (CUTLASS ClusterBarrier::arrive(cta_id) form)
__device__ __forceinline__ void mbar_arrive_remote(u32 bar, u32 rank) {
  asm volatile("{\n .reg .b32 ra;\n mapa.shared::cluster.u32 ra, %0, %1;\n mbarrier.arrive.shared::cluster.b64 _, [ra];\n}\n"
               :: "r"(bar), "r"(rank) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(u32 bar, u32 bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\n}\n" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_load_3d(u32 dst, const TMap* map, u32 bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void tma_load_3d_mc(u32 dst, const TMap* map, u32 bar, int c0, int c1, int c2, u16 mask) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
               " [%0], [%1, {%3, %4, %5}], [%2], %6;"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2), "h"(mask) : "memory");
}
// escape correction from the stored original BF16 weight t: its 12-bit decode keeps the sign and low byte and clamps the
// exponent into [base, base + 7]; the exact difference is formed here in FP32
__device__ __forceinline__ float esc_delta(u32 t, u32 bq) {
  const u32 hi = (t >> 8) & 0xFFu, e71 = hi & 0x7Fu;
  const u32 dhi = (hi & 0x80u) | (e71 < bq ? bq : bq + 7u);
  return __uint_as_float(t << 16) - __uint_as_float(((dhi << 8) | (t & 0xFFu)) << 16);
}
__device__ __forceinline__ float bf2f(u16 v) { return __uint_as_float(((u32)v) << 16); }
"""

_x3_SRC_BODY = r"""
// MT token tiles of 16; TOKR rows per x box; NT n8 tiles per warp; NWR row warps; WSK k warps; UPW 64-k units per warp
// per stage; NSTAGE ring stages; NACC accumulator sets; CL CTAs per cluster split K; MC CTAs per cluster share x
// (multicast); PK: x box has exactly M rows; ESCG: escape warps read x from global (do not hold ring slots);
// WB: baked W pointer (0 = from params); FL: flags (1 x before W, 2 trace)
template <int FL> __device__ __forceinline__ void wait_s(u32 bar, u32 par) {
  if constexpr ((FL & 32) != 0) mbar_wait(bar, par); else mbar_wait_sleep(bar, par);
}
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-SnippetCopyrightText: 2026 alygg77 <https://github.com/alygg77/okc-muse-spark-kernels>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_kernel` (via alygg77/okc-muse-spark-kernels), with alygg77's escape-warp loop from the same file's `_escape_source` (alygg77/okc-muse-spark-kernels, kernels/muse_bf16_proj.py).
// Changes: ported from the torch extension to NVRTC (configuration as template parameters, optional baked W pointer); kept sk2's swapped-operand ring kernel and its 12-bit decode, and added x multicast across a cluster, alygg77's escape-warp loop that reads x from global, further escape modes (batched global x loads, inline corrections in the consumer warps, a float4 correction layout), split consumer sets and trace hooks.
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, int NACC, int CL, int MC, bool PK, int EM,
          u64 WB, int FL>
__device__ __forceinline__ void sk3_body(const P& p) {
  // EM: 0 escape warps on the ring (sk2), 1 escape warps reading x from global (bo1 m<=8 variant), 2 inline: each
  // consumer warp applies its own escapes (per-warp, stage-tagged list; exact fp32 deltas) from the x tile in smem
  constexpr bool ESCG = EM == 1;
  constexpr bool EIN = EM == 2;
  constexpr int NW = NWR * WSK;                 // consumer warps per stage set
  constexpr int SPL = 1 << ((FL >> 13) & 3);    // consumer sets taking turns on stages (latency hiding)
  constexpr int NWT = NW * SPL;
  constexpr bool EARLY = (FL & 32768) != 0;     // load every unit's smem first, release the slot, then decode + mma
  constexpr int NC = NWT * 32;
  static_assert(CL == 1 || SPL == 1, "stage-parallel consumers need CL == 1");
  static_assert(NSTAGE % SPL == 0, "each slot must belong to one consumer set");
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = 768;
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XT = TOKR * 128;
  constexpr int XBY = UPS * XT;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);
  constexpr int CP = R + 1;
  // ESC1: one escape warp per ring slot (only it holds/arrives on that slot), private float4-layout correction copies
  // merged by the escape warps before they signal cbar
  constexpr bool ESC1 = (FL & 65536) != 0 && EM == 0;
  static_assert(!ESC1 || CL == 1, "ESC1 needs CL == 1");
  constexpr int NESC = EIN ? 0 : ESC1 ? NSTAGE : (NSTAGE >= 4 ? 4 : NSTAGE);
  constexpr int NEA = ESCG ? 0 : ESC1 ? 1 : NESC;   // escape warps arriving on the empty barriers
  constexpr int CS = CL * MC;                   // cluster size
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const u32 sraw = (u32)__cvta_generic_to_shared(smem_raw);
  const u32 sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const u32 bars = sbase + NSTAGE * SST;
  const u32 full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8, qbar = pbar + 16;
  float* corr = reinterpret_cast<float*>(gbase + NSTAGE * SST + 16 * NSTAGE + 32);
  constexpr bool CF4 = ESC1 || (FL & 262144) != 0;   // corrections [copy][token][RP]: float4 reads in the output
  constexpr int CSTR = CF4 ? RP : CP;           // correction row stride per token
  constexpr int CSZ = CF4 ? TOKS * RP : ((TOKS * CP + 3) & ~3);
  float* peer = corr + NESC * CSZ;
  int2* ptab_s = reinterpret_cast<int2*>(peer + (CL > 1 ? (CL - 1) * WSK * TOKS * RP : 0));
  const int pmax = p.pmax;
  const int M = p.M;
  const u32 zrow = (u32)__cvta_generic_to_shared(ptab_s + pmax);
  const int Mb = PK ? M : TOKR;
  const u32 XU = (u32)Mb * 128u;
  const u32 XBYr = (FL & 8) ? 0u : UPS * XU;
  const u64 tstart = (FL & 2) ? gtime() : 0ull;
  float* red = reinterpret_cast<float*>(gbase);
  const u8* Wp = WB ? cptr<WB>() : p.W;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // cluster = MC row blocks x CL k parts; blockIdx.x = ((rbp * CL + crank) * MC + mrank) with rbp the row-block pair
  const int mrank = MC > 1 ? (int)(blockIdx.x % MC) : 0;
  const int crank = CL > 1 ? (int)((blockIdx.x / MC) % CL) : 0;
  const int cta = (int)(blockIdx.x / CS) * MC + mrank;      // row block
  const int nrb = (int)(gridDim.x / CS) * MC;
  const int K = p.K;
  const int tst = K / (UPS * 64);
  const int gs0 = (crank * tst) / CL;
  const int nst = ((crank + 1) * tst) / CL - gs0;
  // ROT: each CTA walks its stages starting at a CTA-dependent stage (wrapping), so the CTAs do not all read the same
  // x box (and W stage region) at the same moment (L2 hot spot); stage index sr(st) = (st + rot) % nst
  constexpr bool ROT = (FL & 2097152) != 0;
  // CLR (CL > 1): ranks > 0 reduce their k-warp slices + corrections locally, then send one [M][R] fp32 slice with
  // 16-byte st.async; rank 0 adds one peer slice per rank
  constexpr bool CLR = CL > 1 && (FL & 8388608) != 0;
  static_assert(!ROT || (CL == 1 && MC == 1), "ROT needs CL == MC == 1");
  const int rot = ROT ? (int)(((long long)cta * nst) / nrb) : 0;
  const u8* wsrc = Wp + (size_t)cta * WBY;
  const size_t wstride = (size_t)nrb * WBY;
  // rank of this CTA inside the cluster and of its multicast partners: crank * MC + m
  const u32 myrank = (u32)(crank * MC + mrank);
  const u16 mcmask = (u16)(((1u << MC) - 1u) << (crank * MC));

  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&p.xmap) : "memory");
    mbar_init(full0, 1);
    if constexpr (CS > 1) {
      mbar_init(empty0, MC * (NW + NEA));
      for (int i = 1; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, MC * (NW + NEA)); }
      mbar_init(pbar, 1);
      mbar_init(cbar, EIN ? 1 : 32 * NESC);
      mbar_init(qbar, 1);
      if (CL > 1 && crank == 0) mbar_expect_tx(qbar, CLR ? (u32)((CL - 1) * p.M * R * 4) : (CL - 1) * NC * MT * NT * 16);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if constexpr (MC == 1) {
      mbar_expect_tx(full0, XBYr + WBY);
      if ((FL & 1) && !(FL & 8)) tma_load_3d(sbase, &p.xmap, full0, 0, 0, (gs0 + rot) * UPS);
      bulk_g2s(sbase + XBY, wsrc + (size_t)(gs0 + rot) * wstride, WBY, full0);
      if (!(FL & 1) && !(FL & 8)) tma_load_3d(sbase, &p.xmap, full0, 0, 0, (gs0 + rot) * UPS);
    }
    {   // this CTA's escape table
      if constexpr (CS == 1) {
        mbar_init(pbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      }
      mbar_expect_tx(pbar, pmax * 8);
      bulk_g2s((u32)__cvta_generic_to_shared(ptab_s), p.ptab + (size_t)cta * pmax, pmax * 8, pbar);
    }
    if constexpr (CS == 1) {
      mbar_init(empty0, NW + NEA);
      for (int i = 1; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + NEA); }
      mbar_init(cbar, EIN ? 1 : 32 * NESC);
      mbar_init(qbar, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
  }
  if (tid >= 32 && tid < 64) reinterpret_cast<u32*>(ptab_s + pmax)[tid - 32] = 0u;
  __syncthreads();
  // EW (CL > 1): the one cluster barrier is passed right here (rank 0's qbar is initialized before ranks > 0 can
  // signal it); no cluster barrier at the end of the kernel
  constexpr bool EW = CL > 1 && MC == 1 && (FL & 4194304) != 0;
  if constexpr (CS > 1) {
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    // multicast: the partner's barriers must be initialized before our x copies signal them
    if constexpr (MC > 1 || EW) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
  }

  if (warp == NWT) {
    if (lane == 0) {
      for (int st = (MC > 1 ? 0 : 1); st < nst; ++st) {
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) wait_s<FL>(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
        const u32 s0 = sbase + slot * SST;
        mbar_expect_tx(full0 + 8 * slot, XBYr + WBY);
        if constexpr (MC > 1) {
          const bool mine = (st % MC) == mrank && !(FL & 8);
          if ((FL & 1) && mine) tma_load_3d_mc(s0, &p.xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS, mcmask);
          bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);
          if (!(FL & 1) && mine) tma_load_3d_mc(s0, &p.xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS, mcmask);
        } else {
          const int sr = ROT ? (st + rot) % nst : st;
          if ((FL & 1) && !(FL & 8)) tma_load_3d(s0, &p.xmap, full0 + 8 * slot, 0, 0, (gs0 + sr) * UPS);
          bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + sr) * wstride, WBY, full0 + 8 * slot);
          if (!(FL & 1) && !(FL & 8)) tma_load_3d(s0, &p.xmap, full0 + 8 * slot, 0, 0, (gs0 + sr) * UPS);
        }
      }
    }
    __syncwarp();
    if constexpr (CS > 1 && MC == 1 && !EW) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if constexpr (MC > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }
  if (warp >= NWT + 1) {
    const int ew = warp - (NWT + 1);
    float* cr = corr + ew * CSZ;
    for (int i = lane; i < CSZ; i += 32) cr[i] = 0.f;
    wait_s<FL>(pbar, 0);
    const int cnt = ptab_s[0].x;
    constexpr int KST = UPS * 64;
    const u32 bq = p.base4 & 0xFFu;
    if constexpr (ESCG && (FL & 131072) == 0) {
      // bo1's variant (alygg): escape warps stride over the table, x from global, one entry at a time
      for (int pe = 1 + ew; pe <= cnt; pe += NESC) {
        const int2 e = ptab_s[pe];
        const int kk = e.x & 0xFFFF;
        if (lane < M && kk >= gs0 * KST && kk < (gs0 + nst) * KST) {
          const float v = bf2f(p.x[(size_t)lane * K + kk]);
          cr[lane * CSTR + (e.x >> 16)] += esc_delta((u32)e.y, bq) * v;
        }
      }
    } else if constexpr (ESCG) {
      // all of this warp's x loads go out back to back (one DRAM round trip at kernel start), then the corrections;
      // the escape warps never touch the ring (NEA = 0) and are done long before the stream ends
      constexpr int EPW = 16;
      const int klo = gs0 * KST, khi = (gs0 + nst) * KST;
      for (int b0 = 1 + ew; b0 <= ((FL & 128) ? 0 : cnt); b0 += NESC * EPW) {
        u16 xv[EPW];
        int ent[EPW];
#pragma unroll
        for (int j = 0; j < EPW; ++j) {
          const int pe = b0 + j * NESC;
          ent[j] = 0; xv[j] = 0;
          if (pe <= cnt) {
            const int kk = ptab_s[pe].x & 0xFFFF;
            if (lane < M && kk >= klo && kk < khi) {
              ent[j] = pe;
              xv[j] = __ldg(reinterpret_cast<const unsigned short*>(p.x) + (size_t)lane * K + kk);
            }
          }
        }
#pragma unroll
        for (int j = 0; j < EPW; ++j)
          if (ent[j]) {
            const int2 e = ptab_s[ent[j]];
            cr[lane * CSTR + (e.x >> 16)] += esc_delta((u32)e.y, bq) * bf2f(xv[j]);
          }
      }
    } else if constexpr (ESC1) {
      int pe = 1;
      const int slot = ew;
      for (int st = ew; st < nst; st += NSTAGE) {
        const int kbeg = (gs0 + st) * KST, kend = kbeg + KST;
        wait_s<FL>(full0 + 8 * slot, (st / NSTAGE) & 1);
        while (pe <= cnt && (ptab_s[pe].x & 0xFFFF) < kbeg) ++pe;
        const unsigned char* xs = gbase + slot * SST;
        while (pe <= cnt) {
          const int2 e = ptab_s[pe];
          const int kk = e.x & 0xFFFF;
          if (kk >= kend) break;
          if (lane < M) {
            const int kl = kk - kbeg, u = kl >> 6, kc = kl & 63;
            const u16 v = *reinterpret_cast<const u16*>(
                xs + u * XU + lane * 128 + (((kc >> 3) ^ ((u * Mb + lane) & 7)) << 4) + (kc & 7) * 2);
            cr[lane * CSTR + (e.x >> 16)] += esc_delta((u32)e.y, bq) * bf2f(v);
          }
          ++pe;
        }
        __syncwarp();
        if (lane == 0) {
          mbar_arrive(empty0 + 8 * slot);
          if constexpr (MC > 1) {
#pragma unroll
            for (int r = 1; r < MC; ++r) mbar_arrive_remote(empty0 + 8 * slot, (u32)(crank * MC + (mrank + r) % MC));
          }
        }
      }
      // merge the NESC private copies into copy 0 (float4), escape warps only
      named_bar(2, NESC * 32);
      for (int i = (ew * 32 + lane) * 4; i < CSZ; i += NESC * 128) {
        float4 v = *reinterpret_cast<const float4*>(corr + i);
#pragma unroll 4
        for (int c = 1; c < NESC; ++c) {
          const float4 q = *reinterpret_cast<const float4*>(corr + c * CSZ + i);
          v.x += q.x; v.y += q.y; v.z += q.z; v.w += q.w;
        }
        *reinterpret_cast<float4*>(corr + i) = v;
      }
    } else {
      int pe = 1;
      if constexpr (CL > 1) {
        // skip (warp-parallel) the entries of the other k parts: the table is k-sorted, the ballot mask is a prefix
        const int k0 = gs0 * KST;
        while (true) {
          const int idx = pe + lane;
          const int kk = idx <= cnt ? (ptab_s[idx].x & 0xFFFF) : 0x7FFFFFFF;
          const unsigned bm = __ballot_sync(0xffffffffu, kk < k0);
          pe += __popc(bm);
          if (bm != 0xffffffffu) break;
        }
      }
      for (int st = 0; st < nst; ++st) {
        const int sr = ROT ? (st + rot) % nst : st;
        if (ROT && sr == 0) pe = 1;   // wrapped around to the first k range: rescan the k-ordered table from the start
        const int kbeg = (gs0 + sr) * KST, kend = kbeg + KST;
        const int slot = st % NSTAGE;
        wait_s<FL>(full0 + 8 * slot, (st / NSTAGE) & 1);
        if ((st % NESC) != ew) {
          while (pe <= cnt && (ptab_s[pe].x & 0xFFFF) < kend) ++pe;
          __syncwarp();
          if (lane == 0) {
            mbar_arrive(empty0 + 8 * slot);
            if constexpr (MC > 1) {
#pragma unroll
              for (int r = 1; r < MC; ++r) mbar_arrive_remote(empty0 + 8 * slot, (u32)(crank * MC + (mrank + r) % MC));
            }
          }
          continue;
        }
        const unsigned char* xs = gbase + slot * SST;
        while (pe <= cnt) {
          const int2 e = ptab_s[pe];
          const int kk = e.x & 0xFFFF;
          if (kk >= kend) break;
          if (lane < M && kk >= kbeg) {
            const int kl = kk - kbeg, u = kl >> 6, kc = kl & 63;
            const u16 v = *reinterpret_cast<const u16*>(
                xs + u * XU + lane * 128 + (((kc >> 3) ^ ((u * Mb + lane) & 7)) << 4) + (kc & 7) * 2);
            cr[lane * CSTR + (e.x >> 16)] += esc_delta((u32)e.y, bq) * bf2f(v);
          }
          ++pe;
        }
        __syncwarp();
        if (lane == 0) {
          mbar_arrive(empty0 + 8 * slot);
          if constexpr (MC > 1) {
#pragma unroll
            for (int r = 1; r < MC; ++r) mbar_arrive_remote(empty0 + 8 * slot, (u32)(crank * MC + (mrank + r) % MC));
          }
        }
      }
    }
    mbar_arrive(cbar);
    __syncwarp();
    if constexpr (CS > 1 && MC == 1 && !EW) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if constexpr (MC > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }

  // ---------------- consumers
  const int cset = warp / NW, wl = warp % NW;
  const int wr = wl / WSK, wk = wl % WSK;
  const int g = lane >> 2, t4 = lane & 3;
  const int q = lane >> 3;
  const int chi = q >> 1;
  int trow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) trow[mt] = TOKR >= 16 ? mt * 16 + (lane & 7) + 8 * (q & 1) : (lane & 7);
  float acc[NACC][MT][NT][4];
#pragma unroll
  for (int c = 0; c < NACC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;

  u64 tfirst = 0;
  float eacc[EIN ? MT : 1][EIN ? NT : 1][4];   // escape corrections: separate from the mma accumulators (no wait on HMMA)
#pragma unroll
  for (int mt = 0; mt < (EIN ? MT : 1); ++mt)
#pragma unroll
    for (int nt = 0; nt < (EIN ? NT : 1); ++nt) eacc[mt][nt][0] = eacc[mt][nt][1] = eacc[mt][nt][2] = eacc[mt][nt][3] = 0.f;
  int pe = 0, pend = 0, nxt = 1 << 30;
  if constexpr (EIN) {
    mbar_wait(pbar, 0);
    const int2 h = ptab_s[1 + wl];
    pe = h.x; pend = h.x + h.y;
    while (pe < pend && (ptab_s[pe].x >> 16) < gs0) ++pe;
    nxt = pe < pend ? (ptab_s[pe].x >> 16) : (1 << 30);
  }
  auto do_esc = [&](int st, int slot, bool hesc) {
    const u32 s0 = sbase + slot * SST;
    if constexpr (EIN) {
      if (hesc) {
        do {
          const int2 e = ptab_s[pe];
          const int ex = e.x;
          const int euu = (ex >> 13) & 7, nte = (ex >> 9) & 15, c = (ex >> 6) & 7, kc = ex & 63;
          if (!(FL & 1024) && (c >> 1) == t4) {
            const float d = __int_as_float(e.y);
            const int eu = wk * UPW + euu;
            const u32 xu = s0 + eu * XU;
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
              float xv[2];
#pragma unroll
              for (int h = 0; h < 2; ++h) {
                const int tok = mt * 16 + g + 8 * h;
                u16 v = 0;
                if ((FL & 4096) == 0 && (!PK || tok < M))
                  asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v)
                               : "r"(xu + tok * 128 + (((kc >> 3) ^ ((eu * Mb + tok) & 7)) << 4) + (kc & 7) * 2));
                xv[h] = bf2f(v);
              }
#pragma unroll
              for (int nt = 0; nt < NT; ++nt)
                if (nt == nte) {
                  if (c & 1) { eacc[mt][nt][1] = fmaf(d, xv[0], eacc[mt][nt][1]); eacc[mt][nt][3] = fmaf(d, xv[1], eacc[mt][nt][3]); }
                  else       { eacc[mt][nt][0] = fmaf(d, xv[0], eacc[mt][nt][0]); eacc[mt][nt][2] = fmaf(d, xv[1], eacc[mt][nt][2]); }
                }
            }
          }
          ++pe;
          nxt = pe < pend ? (ptab_s[pe].x >> 16) : (1 << 30);
        } while (nxt == gs0 + st);
      }
    }
  };
  for (int st = cset; st < nst; st += SPL) {
    const int slot = st % NSTAGE;
    if constexpr (EIN && SPL > 1) {
      while (nxt < gs0 + st) { ++pe; nxt = pe < pend ? (ptab_s[pe].x >> 16) : (1 << 30); }
    }
    const bool hesc = EIN && nxt == gs0 + st;
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    if ((FL & 2) && st == 0) tfirst = gtime();
    if ((FL & 2) && warp == 0 && lane == 0 && st + SPL >= nst) p.trc[blockIdx.x * 8 + 2] = gtime();
    const u32 s0 = sbase + slot * SST;
    if constexpr ((FL & 16) != 0) {
      __syncwarp();
      if (lane == 0) {
        mbar_arrive(empty0 + 8 * slot);
        if constexpr (MC > 1) {
#pragma unroll
          for (int r = 1; r < MC; ++r) mbar_arrive_remote(empty0 + 8 * slot, (u32)(crank * MC + (mrank + r) % MC));
        }
      }
      continue;
    }
    if constexpr (EARLY) {
      uint4 lw[UPW][NT];
      uint2 cw[UPW][NT];
      u32 a[UPW][4][MT][4];
#pragma unroll
      for (int uu = 0; uu < UPW; ++uu) {
        const int ui = wk * UPW + uu;
        const u32 xt = s0 + ui * XU;
        const u32 wb = s0 + XBY + (wl * UPW + uu) * NT * TU;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(lw[uu][nt].x), "=r"(lw[uu][nt].y), "=r"(lw[uu][nt].z), "=r"(lw[uu][nt].w) : "r"(wb + nt * TU + lane * 16));
          asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(cw[uu][nt].x), "=r"(cw[uu][nt].y) : "r"(wb + nt * TU + 512 + lane * 8));
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt)
            ldsm_x4((PK && trow[mt] >= M) ? zrow
                        : xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * Mb + trow[mt]) & 7)) << 4),
                    a[uu][j][mt][0], a[uu][j][mt][1], a[uu][j][mt][2], a[uu][j][mt][3]);
      }
      do_esc(st, slot, hesc);
      __syncwarp();
      if (lane == 0) {
        mbar_arrive(empty0 + 8 * slot);
        if constexpr (MC > 1) {
#pragma unroll
          for (int r = 1; r < MC; ++r) mbar_arrive_remote(empty0 + 8 * slot, (u32)(crank * MC + (mrank + r) % MC));
        }
      }
#pragma unroll
      for (int uu = 0; uu < UPW; ++uu) {
        u32 b[NT][8];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          decode8(lw[uu][nt].x, lw[uu][nt].y, cw[uu][nt].x, p.base4, &b[nt][0]);
          decode8(lw[uu][nt].z, lw[uu][nt].w, cw[uu][nt].y, p.base4, &b[nt][4]);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
              mma_bf16(acc[(uu * 4 + j) % NACC][mt][nt], a[uu][j][mt][0], a[uu][j][mt][1], a[uu][j][mt][2], a[uu][j][mt][3], b[nt][2 * j], b[nt][2 * j + 1]);
      }
      continue;
    }
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const int ui = wk * UPW + uu;
      const u32 xt = s0 + ui * XU;
      const u32 wb = s0 + XBY + (wl * UPW + uu) * NT * TU;
      uint4 lw[NT];
      uint2 cw[NT];
      u32 b[NT][8];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(lw[nt].x), "=r"(lw[nt].y), "=r"(lw[nt].z), "=r"(lw[nt].w) : "r"(wb + nt * TU + lane * 16));
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(cw[nt].x), "=r"(cw[nt].y) : "r"(wb + nt * TU + 512 + lane * 8));
      }
      u32 a[4][MT][4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          ldsm_x4((PK && trow[mt] >= M) ? zrow
                      : xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * Mb + trow[mt]) & 7)) << 4),
                  a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      if (UPW == 1 && uu == 0 && !hesc) {
        __syncwarp();
        if (lane == 0) {
          mbar_arrive(empty0 + 8 * slot);
          if constexpr (MC > 1) {
#pragma unroll
            for (int r = 1; r < MC; ++r) mbar_arrive_remote(empty0 + 8 * slot, (u32)(crank * MC + (mrank + r) % MC));
          }
        }
      }
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        decode8(lw[nt].x, lw[nt].y, cw[nt].x, p.base4, &b[nt][0]);
        decode8(lw[nt].z, lw[nt].w, cw[nt].y, p.base4, &b[nt][4]);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt)
            mma_bf16(acc[(uu * 4 + j) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], b[nt][2 * j], b[nt][2 * j + 1]);
    }
    if constexpr (!EARLY) do_esc(st, slot, hesc);
    if (UPW > 1 || hesc) {
      __syncwarp();
      if (lane == 0) {
        mbar_arrive(empty0 + 8 * slot);
        if constexpr (MC > 1) {
#pragma unroll
          for (int r = 1; r < MC; ++r) mbar_arrive_remote(empty0 + 8 * slot, (u32)(crank * MC + (mrank + r) % MC));
        }
      }
    }
  }

  if constexpr (EIN) {
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[0][mt][nt][e] += eacc[mt][nt][e];
  }
  // ---------------- epilogue: K-warps reduce through shared memory (the ring is free now)
  if ((FL & 2) && warp == 0 && lane == 0) {
    p.trc[blockIdx.x * 8 + 0] = tstart;
    p.trc[blockIdx.x * 8 + 1] = tfirst;
    p.trc[blockIdx.x * 8 + 3] = gtime();
    u32 smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    p.trc[blockIdx.x * 8 + 6] = smid;
  }
  if constexpr (!EIN) mbar_wait(cbar, 0);
  if ((FL & 2) && warp == 0 && lane == 0) p.trc[blockIdx.x * 8 + 5] = gtime();
  if (CLR && crank > 0) {
    named_bar(1, NC);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int row = (wr * NT + nt) * 8 + 2 * t4;
        const int tok = mt * 16 + g;
        float* r0 = red + ((cset * WSK + wk) * TOKS + tok) * RP + row;
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          v[e] = acc[0][mt][nt][e];
#pragma unroll
          for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
        }
        *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
        *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
      }
    named_bar(1, NC);
    if constexpr (MC == 1 && !EW) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    const u32 peer_l = (u32)__cvta_generic_to_shared(peer + (crank - 1) * TOKS * RP);
    const u32 r0rank = (u32)mrank;
    u32 peer_r, qbar_r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(peer_r) : "r"(peer_l), "r"(r0rank));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(qbar_r) : "r"(qbar), "r"(r0rank));
    constexpr int R4c = R / 4;
    for (int idx = tid; idx < R4c * M; idx += NC) {
      const int tt = idx / R4c, row = (idx - tt * R4c) * 4;
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int c = 0; c < NESC; ++c) {
        if constexpr (CF4) {
          const float4 q = *reinterpret_cast<const float4*>(corr + c * CSZ + tt * RP + row);
          v.x += q.x; v.y += q.y; v.z += q.z; v.w += q.w;
        } else {
          v.x += corr[c * CSZ + tt * CP + row]; v.y += corr[c * CSZ + tt * CP + row + 1];
          v.z += corr[c * CSZ + tt * CP + row + 2]; v.w += corr[c * CSZ + tt * CP + row + 3];
        }
      }
#pragma unroll
      for (int w = 0; w < WSK; ++w) {
        const float4 q = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
        v.x += q.x; v.y += q.y; v.z += q.z; v.w += q.w;
      }
      asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                   :: "r"(peer_r + (u32)((tt * RP + row) * 4)), "r"(__float_as_uint(v.x)), "r"(__float_as_uint(v.y)),
                      "r"(__float_as_uint(v.z)), "r"(__float_as_uint(v.w)), "r"(qbar_r) : "memory");
    }
    if ((FL & 2) && warp == 0 && lane == 0) p.trc[blockIdx.x * 8 + 4] = gtime();
    return;
  }
  if (CL > 1 && crank > 0) {
    if constexpr (MC == 1 && !EW) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    const u32 peer_l = (u32)__cvta_generic_to_shared(peer + (crank - 1) * WSK * TOKS * RP);
    const u32 r0rank = (u32)mrank;   // rank 0 of this row block's k-split group: crank 0, same mrank
    u32 peer_r, qbar_r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(peer_r) : "r"(peer_l), "r"(r0rank));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(qbar_r) : "r"(qbar), "r"(r0rank));
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int row = (wr * NT + nt) * 8 + 2 * t4;
        const int tok = mt * 16 + g;
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          v[e] = acc[0][mt][nt][e];
#pragma unroll
          for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
        }
        if (!EIN && wk == 0) {
#pragma unroll
          for (int c = 0; c < NESC; ++c) {
            v[0] += corr[c * CSZ + tok * CSTR + row];
            v[1] += corr[c * CSZ + tok * CSTR + row + 1];
            if (tok + 8 < TOKS) {
              v[2] += corr[c * CSZ + (tok + 8) * CSTR + row];
              v[3] += corr[c * CSZ + (tok + 8) * CSTR + row + 1];
            }
          }
        }
        const u32 a0 = peer_r + (u32)(((wk * TOKS + tok) * RP + row) * 4);
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                     :: "r"(a0), "r"(__float_as_uint(v[0])), "r"(__float_as_uint(v[1])), "r"(qbar_r) : "memory");
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                     :: "r"(a0 + 8 * RP * 4), "r"(__float_as_uint(v[2])), "r"(__float_as_uint(v[3])), "r"(qbar_r) : "memory");
      }
    if ((FL & 2) && warp == 0 && lane == 0) p.trc[blockIdx.x * 8 + 4] = gtime();
    if constexpr (MC > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }
  named_bar(1, NC);
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int row = (wr * NT + nt) * 8 + 2 * t4;
      const int tok = mt * 16 + g;
      float* r0 = red + ((cset * WSK + wk) * TOKS + tok) * RP + row;
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = acc[0][mt][nt][e];
#pragma unroll
        for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
      }
      *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
    }
  named_bar(1, NC);
  if ((FL & 2) && warp == 0 && lane == 0) p.trc[blockIdx.x * 8 + 7] = gtime();
  if constexpr (CL > 1) {
    if constexpr (MC == 1 && !EW) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    mbar_wait_cluster(qbar, 0);
    if ((FL & 2) && warp == 0 && lane == 0) p.trc[blockIdx.x * 8 + 6] = gtime();
  }
  constexpr int R4 = R / 4;
  for (int idx = tid; idx < R4 * M; idx += NC) {
    const int tt = idx / R4, row = (idx - tt * R4) * 4;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if constexpr (ESC1) {
      v = *reinterpret_cast<const float4*>(corr + tt * RP + row);
    } else if constexpr (CF4) {
#pragma unroll
      for (int c = 0; c < NESC; ++c) {
        const float4 q = *reinterpret_cast<const float4*>(corr + c * CSZ + tt * RP + row);
        v.x += q.x; v.y += q.y; v.z += q.z; v.w += q.w;
      }
    } else if (CL == 1 || crank == 0) {
#pragma unroll
      for (int c = 0; c < NESC; ++c) {
        v.x += corr[c * CSZ + tt * CP + row];
        v.y += corr[c * CSZ + tt * CP + row + 1];
        v.z += corr[c * CSZ + tt * CP + row + 2];
        v.w += corr[c * CSZ + tt * CP + row + 3];
      }
    }
#pragma unroll
    for (int w = 0; w < SPL * WSK; ++w) {
      const float4 pp = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
      v.x += pp.x; v.y += pp.y; v.z += pp.z; v.w += pp.w;
#pragma unroll
      for (int c = 0; c < CL - 1; ++c) {
        if (CLR && w > 0) break;
        const float4 qq = *reinterpret_cast<const float4*>(peer + ((CLR ? c : c * WSK + w) * TOKS + tt) * RP + row);
        v.x += qq.x; v.y += qq.y; v.z += qq.z; v.w += qq.w;
      }
    }
    u16* o = p.out + (size_t)tt * p.N + cta * R + row;
    if (ptab_s[0].y != 0)
      o = p.out + (size_t)tt * p.N + reinterpret_cast<const int*>(ptab_s + ptab_s[0].y)[row];
    uint2 pk;
    pk.x = pack_bf2(v.x, v.y);
    pk.y = pack_bf2(v.z, v.w);
    *reinterpret_cast<uint2*>(o) = pk;
  }
  if ((FL & 2) && tid == 0) p.trc[blockIdx.x * 8 + 4] = gtime();
  if constexpr (MC > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
// SPDX-SnippetEnd
"""

# ------------------------------------------------------------------------------------------------- configuration
# (R, NT, NWR, WSK, UPW, NACC, CL, MC, ring cap KB, flags)  -- R = NWR*NT*8 rows per CTA (per k-split group)
# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-SnippetCopyrightText: 2026 alygg77 <https://github.com/alygg77/okc-muse-spark-kernels>
# SPDX-License-Identifier: Apache-2.0
# Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `SK2_CONFIGS` / `SK2_PICK` tile configurations, with alygg77's `u32_r96` unpooler entry (alygg77/okc-muse-spark-kernels, kernels/muse_bf16_proj.py).
# Changes: the (layer, m) entries that bo1 picks from those tables, re-keyed by (layer, m), with CL / MC, the ring cap and a flags word as columns (flag 64 selects alygg77's global-x escape warps for qkv_gate m <= 8).
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
_x3_BASE_CFG = {   # bo1's sk2 configs (vjeux/alygg tables) for the shapes this module may own
    ("qkv_gate", 8): (24, 3, 1, 4, 2, 2, 1, 1, 144, 64),
    ("qkv_gate", 16): (24, 3, 1, 4, 2, 2, 1, 1, 200, 0),
    ("qkv_gate", 32): (24, 3, 1, 4, 2, 1, 1, 1, 208, 0),
    ("pooler", 8): (16, 1, 2, 2, 4, 1, 1, 1, 72, 0),
    ("pooler", 16): (32, 2, 2, 2, 4, 1, 1, 1, 192, 0),
    ("pooler", 32): (32, 2, 2, 2, 2, 1, 1, 1, 144, 0),
    ("unpooler", 16): (32, 2, 2, 2, 2, 1, 1, 1, 72, 0),
    ("unpooler", 32): (32, 2, 2, 2, 2, 1, 1, 1, 96, 0),
    ("o", 8): (32, 4, 1, 4, 1, 2, 1, 1, 88, 0),
    ("o", 16): (32, 2, 2, 1, 4, 2, 1, 1, 68, 0),
    ("o", 32): (32, 2, 2, 2, 2, 1, 1, 1, 100, 0),
}
# SPDX-SnippetEnd
_x3_FLAG_CF4 = 262144           # flags bit: corrections in [copy][token][RP] layout, float4 reads in the output loop
_x3_CFG = {k: v[:9] + (v[9] | _x3_FLAG_CF4,) for k, v in _x3_BASE_CFG.items()}   # sk2 configs + float4 corrections
for _k in (("unpooler", 16), ("unpooler", 32), ("o", 16)):   # baked W pointer: -0.17 / -0.24 / -0.05 us same run
    _x3_CFG[_k] = _x3_CFG[_k][:9] + (_x3_CFG[_k][9] | 4,)   # 4 = FLAG_BAKE
# shapes run_sk3 handles (others -> None, base kept): same-run wins vs bo1 (bench1: pooler32 -0.37, unpooler32 -0.32,
# qkv32 -0.25, unpooler16 -0.24 us local)
_x3_OWN = {("pooler", 32), ("unpooler", 32), ("qkv_gate", 32), ("unpooler", 16), ("o", 16), ("qkv_gate", 8)}
for _t in filter(None, os.environ.get("SK3_OWN_EXTRA", "").split(",")):   # experiments: e.g. "o:16,o:32"
    _l, _m = _t.split(":")
    _x3_OWN.add((_l, int(_m)))
for _t in filter(None, os.environ.get("SK3_BAKE_EXTRA", "").split(",")):
    _l, _m = _t.split(":")
    _x3_CFG[(_l, int(_m))] = _x3_CFG[(_l, int(_m))][:9] + (_x3_CFG[(_l, int(_m))][9] | 4,)
_x3_BALANCE = {"qkv_gate"}      # escape-balanced row order (as bo1)
_x3_ESCG = set()   # (bo1's global-x escape warps for qkv m <= 8 are selected by flag 64 in its config)
_x3_K_OF = {"pooler": 8192, "unpooler": 4096, "o": 1024, "qkv_gate": 8192}
_x3_N_OF = {"pooler": 4096, "unpooler": 8192, "o": 8192, "qkv_gate": 2304}
_x3_FLAG_BAKE = 4               # flags bit: bake the W pointer
_x3_FLAG_EIN = 512              # flags bit: escapes inline in the consumer warps (no escape warps)
_x3_SMEM_BUDGET = 232_448
_x3_EXPONENT_WINDOW = 8


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `_sk2_geom` (via alygg77/okc-muse-spark-kernels).
# Changes: takes (cfg, layer, m) with the ring cap in the config instead of the config name, fixes the escape-warp share at 4, and adds stage-count rules for split consumer sets and for one escape warp per stage; returns a dict of the derived sizes.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def _x3__geom(cfg, layer, m):
    R, NT, NWR, WSK, UPW, NACC, CL, MC, cap, fl = cfg
    tokr = 8 if m <= 8 else 16 if m <= 16 else 32
    mt = 2 if tokr == 32 else 1
    ups = WSK * UPW
    nw = NWR * WSK
    wby = nw * UPW * NT * 768
    xby = ups * tokr * 128
    sst = (xby + wby + 1023) // 1024 * 1024
    nst = -(-(_x3_K_OF[layer] // (ups * 64)) // CL)
    toks = 16 * mt
    cp = R + 1
    rp = R + ((40 - (R % 32)) % 32)
    extra = 1024 + 16 * 16 + 32 + 4 * ((toks * cp + 3) // 4 * 4) * 4 + ((CL - 1) * WSK * toks * rp * 4 if CL > 1 else 0) + 64 * 8 + 128
    ring = min(cap * 1024, _x3_SMEM_BUDGET - extra)
    nstage = max(2, min(16, nst, ring // sst))
    spl = 1 << ((fl >> 13) & 3)
    if spl > 1:   # each ring slot must always belong to the same consumer set (phase parity safety)
        nstage = max(spl, nstage - nstage % spl)
    if fl & 65536:   # ESC1: nstage escape warps, each with a [toks][rp] correction copy
        while nstage > 2 and (1024 + nstage * sst + 16 * nstage + 32 + nstage * toks * rp * 4 + 64 * 8 + 128) > _x3_SMEM_BUDGET:
            nstage -= 1
    return dict(tokr=tokr, mt=mt, nstage=nstage, sst=sst, toks=toks, cp=cp, rp=rp, nw=nw)
# SPDX-SnippetEnd


_x3_FLAG_ESCG = 64              # flags bit: escape warps load their x values from global, all at kernel start


def _x3__em(cfg, layer, m):
    return 2 if cfg[9] & _x3_FLAG_EIN else 1 if (cfg[9] & _x3_FLAG_ESCG or (layer, m) in _x3_ESCG) else 0


_x3_FLAG_ESC1 = 65536           # flags bit: one escape warp per ring slot + merged float4 corrections


def _x3__nesc(cfg, layer, m):
    g = _x3__geom(cfg, layer, m)
    if _x3__em(cfg, layer, m) == 2:
        return 0
    if cfg[9] & _x3_FLAG_ESC1 and _x3__em(cfg, layer, m) == 0:
        return g["nstage"]
    return 4 if g["nstage"] >= 4 else g["nstage"]


def _x3__smem(cfg, layer, m, pmax):
    R, NT, NWR, WSK, UPW, NACC, CL, MC, cap, fl = cfg
    g = _x3__geom(cfg, layer, m)
    nesc = _x3__nesc(cfg, layer, m)
    em = _x3__em(cfg, layer, m)
    cf4 = (fl & _x3_FLAG_ESC1 and em == 0) or (fl & 262144 and em != 2)
    csz = g["toks"] * g["rp"] if cf4 else (g["toks"] * g["cp"] + 3) // 4 * 4
    return (1024 + g["nstage"] * g["sst"] + 16 * g["nstage"] + 32 + nesc * csz * 4
            + ((CL - 1) * WSK * g["toks"] * g["rp"] * 4 if CL > 1 else 0) + pmax * 8 + 128)


# ------------------------------------------------------------------------------------------------- weight packing
# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `_exponent_window`, `sk2_tileunits`, `sk2_compress`, `sk2_stream`, `sk2_balance_perm` and `sk2_table` (via alygg77/okc-muse-spark-kernels).
# Changes: renamed, docstrings and comments trimmed, some module constants inlined; vjeux's `_escapes` inlined, and sk2_compress no longer rejects a matrix with too many escapes.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def _x3__exponent_window(e71):
    hist = torch.bincount(e71.flatten(), minlength=128)
    win = torch.stack([hist[b:b + _x3_EXPONENT_WINDOW].sum() for b in range(128 - _x3_EXPONENT_WINDOW + 1)])
    base = int(win.argmax())
    return base, (e71 - base).clamp(0, _x3_EXPONENT_WINDOW - 1)


def _x3__tileunits(w):
    """W [N, K] -> [N/8, K/64, 32 lanes, 16] in mma.m16n8k16 B-fragment order (sk2_tileunits)."""
    n, k = w.shape
    t = w.view(n // 8, 8, k // 64, 4, 2, 4, 2)
    return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)


def _x3__compress(w):
    """sk2_compress: lossless 12-bit tile units [N/8, K/64, 768] + escapes (rows, cols, original bits)."""
    n, k = w.shape
    tu = _x3__tileunits(w)
    bits = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi = bits >> 8
    low = bits & 0xFF
    e71 = hi & 0x7F
    sign = hi >> 7
    base, o = _x3__exponent_window(e71)
    code = (sign << 3) | o
    esc = (e71 < base) | (e71 > base + _x3_EXPONENT_WINDOW - 1)
    T, U = n // 8, k // 64
    c8 = code.to(torch.uint8)
    w0 = c8[..., 0:4] | (c8[..., 4:8] << 4)
    w1 = c8[..., 8:12] | (c8[..., 12:16] << 4)
    codes = torch.cat([w0, w1], dim=-1)
    packed = torch.cat([low.to(torch.uint8).reshape(T, U, 512), codes.reshape(T, U, 256)], dim=-1).contiguous()
    idx = esc.nonzero()
    dec_bits = ((((sign << 7) | (base + o)) << 8) | low)[esc]
    wdec = dec_bits.to(torch.int16).view(torch.bfloat16).float()
    wtrue = tu[esc].float()
    if not bool(((wdec + (wtrue - wdec)) == wtrue).all()):
        return None
    lane, v = idx[:, 2], idx[:, 3]
    rows = idx[:, 0] * 8 + (lane >> 2)
    cols = idx[:, 1] * 64 + 16 * (v >> 2) + 8 * ((v >> 1) & 1) + 2 * (lane & 3) + (v & 1)
    wbits = (tu[esc].contiguous().view(torch.int16).to(torch.int32) & 0xFFFF).contiguous()
    return {"packed": packed, "base4": base * 0x01010101, "rows": rows, "cols": cols, "wbits": wbits}


def _x3__stream(tub, n_cfg, R, NT, NWR, WSK, UPW):
    """Stage-major stream [stage][rowblock][warp][unit][tile][768] (sk2_stream)."""
    dev = tub.device
    U = tub.shape[1]
    ups = WSK * UPW
    ar = lambda v: torch.arange(v, device=dev)
    s = ar(U // ups).view(-1, 1, 1, 1, 1)
    c = ar(n_cfg // R).view(1, -1, 1, 1, 1)
    w = ar(NWR * WSK).view(1, 1, -1, 1, 1)
    uu = ar(UPW).view(1, 1, 1, -1, 1)
    nt = ar(NT).view(1, 1, 1, 1, -1)
    tile = c * (R // 8) + (w // WSK) * NT + nt
    unit = s * ups + (w % WSK) * UPW + uu
    return tub[tile, unit].contiguous()


def _x3__balance_perm(w, n_rows, R):
    wl = w[:n_rows]
    bits = wl.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    e71 = (bits >> 8) & 0x7F
    base, _ = _x3__exponent_window(e71)
    cnt = ((e71 < base) | (e71 > base + _x3_EXPONENT_WINDOW - 1)).sum(1)
    G4 = 4
    tcnt = cnt.view(n_rows // G4, G4).sum(1)
    order = torch.argsort(tcnt, descending=True, stable=True)
    TB = R // G4
    nb = n_rows // R
    i = torch.arange(n_rows // G4, device=w.device)
    rnd, pos = i // nb, i % nb
    blk = torch.where(rnd % 2 == 0, pos, nb - 1 - pos)
    tperm = torch.empty(n_rows // G4, dtype=torch.long, device=w.device)
    tperm[blk * TB + rnd] = order
    return (tperm.view(-1, 1) * G4 + torch.arange(G4, device=w.device).view(1, G4)).reshape(-1)


def _x3__table(comp, n_cfg, R):
    dev = comp["rows"].device
    rows, cols, delta = comp["rows"], comp["cols"], comp["wbits"]
    keep = rows < n_cfg
    rows, cols, delta = rows[keep], cols[keep], delta[keep]
    ncta = n_cfg // R
    cta = rows // R
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max().item()) + 1 if rows.numel() else 1
    pmax += pmax % 2
    order = torch.argsort(cta.long() * 65536 + cols.long())
    cta, rows, cols, delta = cta[order], rows[order], cols[order], delta[order]
    start = torch.zeros(ncta + 1, dtype=torch.long, device=dev)
    start[1:] = torch.cumsum(counts, 0)
    slot = torch.arange(cta.numel(), device=dev) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=dev)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, slot, 0] = (((rows % R) << 16) | cols).to(torch.int32)
    tab[cta, slot, 1] = delta.to(torch.int32)
    return tab.contiguous(), pmax
# SPDX-SnippetEnd


def _x3__table_ein(comp, n_cfg, R, NT, NWR, WSK, UPW, perm=None):
    """Per row block: [0] = (0, perm offset), [1 + w] = (offset, count) of warp w's escapes, then the entries in stage
    order: x = gst << 16 | uu << 13 | nt << 9 | c << 6 | kc, y = float bits of the exact fp32 delta w_true - w_dec."""
    dev = comp["rows"].device
    rows, cols, wb = comp["rows"], comp["cols"], comp["wbits"]
    keep = rows < n_cfg
    rows, cols, wb = rows[keep].long(), cols[keep].long(), wb[keep].long()
    bq = comp["base4"] & 0xFF
    hi = (wb >> 8) & 0xFF
    e71 = hi & 0x7F
    dhi = (hi & 0x80) | torch.where(e71 < bq, torch.full_like(e71, bq), torch.full_like(e71, bq + 7))
    wdec = (((dhi << 8) | (wb & 0xFF)).to(torch.int32) << 16).view(torch.float32)
    wtrue = (wb.to(torch.int32) << 16).view(torch.float32)
    delta = wtrue - wdec
    assert bool(((wdec + delta) == wtrue).all())
    NW = NWR * WSK
    ups = WSK * UPW
    ncta = n_cfg // R
    cta, rl = rows // R, rows % R
    wr, nt, c = rl // (NT * 8), (rl // 8) % NT, rl % 8
    gu = cols // 64
    gst, eu, kc = gu // ups, gu % ups, cols % 64
    wk, uu = eu // UPW, eu % UPW
    warp = wr * WSK + wk
    key = (cta * NW + warp) * 65536 + gst
    order = torch.argsort(key)
    cta, warp, gst, uu, nt, c, kc, delta = [t[order] for t in (cta, warp, gst, uu, nt, c, kc, delta)]
    grp = cta * NW + warp
    cnt = torch.bincount(grp, minlength=ncta * NW).view(ncta, NW)
    tot = cnt.sum(1)
    hdr = 1 + NW
    nent = int((hdr + tot).max().item())
    pmax = nent + (R // 2 if perm is not None else 0)
    pmax += pmax % 2
    off = hdr + torch.cumsum(cnt, 1) - cnt
    start = torch.zeros(ncta * NW + 1, dtype=torch.long, device=dev)
    start[1:] = torch.cumsum(cnt.view(-1), 0)
    rank = torch.arange(grp.numel(), device=dev) - start[grp]
    pos = off.view(-1)[grp] + rank
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=dev)
    tab[:, 1:hdr, 0] = off.to(torch.int32)
    tab[:, 1:hdr, 1] = cnt.to(torch.int32)
    tab[cta, pos, 0] = ((gst << 16) | (uu << 13) | (nt << 9) | (c << 6) | kc).to(torch.int32)
    tab[cta, pos, 1] = delta.contiguous().view(torch.int32)
    if perm is not None:
        tab[:, nent:nent + R // 2] = perm.to(torch.int32).view(ncta, R // 2, 2)
        tab[:, 0, 1] = nent
    return tab.contiguous(), pmax


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Copied from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_table_add_perm` (via alygg77/okc-muse-spark-kernels).
# Changes: renamed; docstring removed.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def _x3__table_add_perm(tab, pmax, perm, R):
    ncta = tab.shape[0]
    pm2 = pmax + R // 2
    pm2 += pm2 % 2
    out = torch.zeros(ncta, pm2, 2, dtype=torch.int32, device=tab.device)
    out[:, :pmax] = tab
    out[:, pmax:pmax + R // 2] = perm.to(torch.int32).view(ncta, R // 2, 2)
    out[:, 0, 1] = pmax
    return out.contiguous(), pm2
# SPDX-SnippetEnd


# ------------------------------------------------------------------------------------------------- NVRTC module
class _x3__Module:
    def __init__(self, src, name="sk3.cu", opts=()):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        o = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        o += [x.encode() for x in opts]
        if os.environ.get("CU_LINEINFO"):
            o.append(b"-lineinfo")
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(o), o)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        self.log = log.decode(errors="replace").strip("\x00 \n")
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + self.log)
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        self.cubin = cubin
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err
        self.funcs = {}

    def func(self, name, smem):
        key = (name, smem)
        f = self.funcs.get(key)
        if f is None:
            d = self.driver
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            assert int(err) == 0, (name, err)
            if smem > 48 * 1024:
                (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
                assert int(err) == 0, (name, smem, err)
            self.funcs[key] = f
        return f

    def attrs(self, name, smem=0):
        d = self.driver
        f = self.func(name, smem)
        out = {}
        for k in ("NUM_REGS", "LOCAL_SIZE_BYTES"):
            err, v = d.cuFuncGetAttribute(getattr(d.CUfunction_attribute, "CU_FUNC_ATTRIBUTE_" + k), f)
            out[k] = v
        return out

    def launch(self, f, grid, block, smem, params):
        d = self.driver
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        assert int(err) == 0, err

    def max_active_clusters(self, f, smem, block, csize, grid):
        d = self.driver
        cfg = d.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block, 1, 1
        cfg.sharedMemBytes = smem
        at = d.CUlaunchAttribute()
        at.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = csize, 1, 1
        cfg.attrs = [at]
        cfg.numAttrs = 1
        err, ncl = d.cuOccupancyMaxActiveClusters(f, cfg)
        return int(ncl) if int(err) == 0 else 0


def _x3__tmap_x3d(ptr, m, k, box_rows, units):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(3), ptr,
        [d.cuuint64_t(64), d.cuuint64_t(m), d.cuuint64_t(k // 64)], [d.cuuint64_t(k * 2), d.cuuint64_t(128)],
        [d.cuuint32_t(64), d.cuuint32_t(box_rows), d.cuuint32_t(units)],
        [d.cuuint32_t(1), d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


_x3__TM = {}


def _x3__xmap(x, m, k, box_rows, units):
    key = (x.data_ptr(), m, k, box_rows, units)
    b = _x3__TM.get(key)
    if b is None:
        if len(_x3__TM) > 512:
            _x3__TM.clear()
        b = _x3__TM[key] = _x3__tmap_x3d(x.data_ptr(), m, k, box_rows, units)
    return b


# ------------------------------------------------------------------------------------------------- prepare / run
def _x3__kname(layer, m, tag=""):
    return f"sk3_{layer}_{m}{tag}"


_x3_SMEM_LIMIT = 232_448


def _x3_prepare_sk3(weights, cfgs=None, verbose=False, variants=None, selftest=True):
    """cfgs: {(layer, m): cfg} (default: CFG restricted to OWN). variants: extra {(layer, m, tag): cfg} instances.
    Never raises: any failure (compression, NVRTC, smem, self-test) leaves the shape to the base (run_sk3 -> None)."""
    try:
        st = _x3__prepare_impl(weights, cfgs, verbose, variants)
    except Exception as ex:   # noqa: BLE001
        print(f"[sk3] prepare failed, base keeps every shape: {type(ex).__name__}: {ex}", flush=True)
        return {"ent": {}, "keep": []}
    if selftest:
        for (layer, m, tag), e in st["ent"].items():
            if not e["ok"]:
                continue
            try:
                g = torch.Generator(device="cuda").manual_seed(1234 + m)
                x = torch.randn((m, e["k"]), generator=g, device="cuda").to(torch.bfloat16)
                out = _x3_run_sk3(x, {"m": m, "n": e["n"], "k": e["k"], "layer": layer}, st, tag=tag or "__self")
                ref = x.float() @ weights[layer][: e["n"]].float().T
                err = float((out.float() - ref).norm() / ref.norm())
                e["ok"] = err < 5e-3   # BF16 output rounding alone gives ~1.7e-3
                if not e["ok"]:
                    print(f"[sk3] self-test failed for {(layer, m, tag)}: rel err {err:.3e}", flush=True)
            except Exception as ex:   # noqa: BLE001
                e["ok"] = False
                print(f"[sk3] self-test error for {(layer, m, tag)}: {ex}", flush=True)
        torch.cuda.synchronize()
    return st


def _x3__prepare_impl(weights, cfgs=None, verbose=False, variants=None):
    cfgs = dict({k: v for k, v in _x3_CFG.items() if k in _x3_OWN} if cfgs is None else cfgs)
    inst = [(layer, m, "", cfg) for (layer, m), cfg in cfgs.items()]
    if variants:
        inst += [(layer, m, tag, cfg) for (layer, m, tag), cfg in variants.items()]
    st = {"ent": {}, "keep": []}
    comps = {}
    shared = {}
    for layer, m, tag, cfg in inst:
        R, NT, NWR, WSK, UPW, NACC, CL, MC, cap, fl = cfg
        assert R == NWR * NT * 8
        w = weights[layer]
        n_rows = _x3_N_OF[layer]
        perm = None
        if layer in _x3_BALANCE:
            pkey = ("bal", layer, R)
            if pkey not in comps:
                pr = _x3__balance_perm(w, n_rows, R)
                comps[pkey] = (pr, _x3__compress(w[:n_rows][pr].contiguous()))
            perm, comp = comps[pkey]
            if comp is None:
                perm = None
        if perm is None:
            if layer not in comps:
                comps[layer] = _x3__compress(w.contiguous())
            comp = comps[layer]
        if comp is None:
            continue
        ein = bool(fl & _x3_FLAG_EIN)
        lkey = (layer, R, NT, NWR, WSK, UPW, perm is not None, ein)
        if lkey not in shared:
            if ein:
                tab, pmax = _x3__table_ein(comp, n_rows, R, NT, NWR, WSK, UPW, perm)
            else:
                tab, pmax = _x3__table(comp, n_rows, R)
                if perm is not None:
                    tab, pmax = _x3__table_add_perm(tab, pmax, perm, R)
            stream = _x3__stream(comp["packed"], n_rows, R, NT, NWR, WSK, UPW)
            assert stream.data_ptr() % 128 == 0 and tab.data_ptr() % 16 == 0 and pmax * 8 % 16 == 0
            shared[lkey] = (stream, tab, pmax)
        stream, tab, pmax = shared[lkey]
        g = _x3__geom(cfg, layer, m)
        spl = 1 << ((fl >> 13) & 3)
        nthr = (g["nw"] * spl + 1 + _x3__nesc(cfg, layer, m)) * 32
        assert spl * WSK * g["toks"] * g["rp"] * 4 <= g["nstage"] * g["sst"], "red buffer must fit in the ring"
        ncta = (n_rows // R) * CL
        smem = _x3__smem(cfg, layer, m, pmax)
        if smem > _x3_SMEM_LIMIT:
            print(f"[sk3] {(layer, m, tag)} needs {smem} B smem (escape table pmax {pmax}): left to the base", flush=True)
            continue
        kn = _x3__kname(layer, m, tag)
        st["ent"][(layer, m, tag)] = dict(cfg=cfg, stream=stream, tab=tab, pmax=pmax, b4=comp["base4"], n=n_rows,
                                          k=_x3_K_OF[layer], g=g, nthr=nthr, ncta=ncta, smem=smem, kn=kn, ok=True)
    src = _x3_SRC_HEAD + _x3_SRC_BODY
    for (layer, m, tag), e in st["ent"].items():
        R, NT, NWR, WSK, UPW, NACC, CL, MC, cap, fl = e["cfg"]
        g = e["g"]
        pk = "true" if m < g["tokr"] or g["tokr"] == 8 else "false"
        escg = str(_x3__em(e["cfg"], layer, m))
        wb = f"{e['stream'].data_ptr()}ull" if fl & _x3_FLAG_BAKE else "0ull"
        cs = CL * MC
        if fl & 1048576:   # timing probe: cluster-of-2 launch attribute with no cluster interaction at all
            cs = 2
        cl_attr = f"__cluster_dims__({cs}, 1, 1) " if cs > 1 else ""
        src += (f'extern "C" __global__ void {cl_attr}__launch_bounds__({e["nthr"]}, 1) {e["kn"]}(const __grid_constant__ P p) '
                f'{{ sk3_body<{g["mt"]}, {g["tokr"]}, {NT}, {NWR}, {WSK}, {UPW}, {g["nstage"]}, {NACC}, {CL}, {MC}, {pk}, {escg}, '
                f'{wb}, {fl & ~(_x3_FLAG_BAKE | _x3_FLAG_EIN | _x3_FLAG_ESCG | 1048576)}>(p); }}\n')
    st["src"] = src
    st["mod"] = _x3__Module(src)
    for key, e in st["ent"].items():
        e["f"] = st["mod"].func(e["kn"], e["smem"])
        cs = e["cfg"][6] * e["cfg"][7] * (2 if e["cfg"][9] & 1048576 else 1)
        if cs > 1:
            e["ncta"] = (e["n"] // e["cfg"][0]) * e["cfg"][6]
            ncl = st["mod"].max_active_clusters(e["f"], e["smem"], e["nthr"], cs, e["ncta"])
            e["clusters"] = ncl
            e["ok"] = ncl >= e["ncta"] // cs
        if verbose:
            a = st["mod"].attrs(e["kn"], e["smem"])
            print(f"[sk3] {key} cfg {e['cfg']} nstage {e['g']['nstage']} smem {e['smem']} thr {e['nthr']} "
                  f"ncta {e['ncta']} regs {a['NUM_REGS']} local {a['LOCAL_SIZE_BYTES']} ok {e['ok']}"
                  + (f" clusters {e.get('clusters')}" if cs > 1 else ""), flush=True)
    torch.cuda.synchronize()
    return st


def _x3_run_sk3(x, config, state, tag=""):
    m, layer = config["m"], config["layer"]
    if (layer, m) not in _x3_OWN and not tag:
        return None
    e = state["ent"].get((layer, m, "" if tag == "__self" else tag))
    if e is None or not e["ok"] or config["n"] != e["n"] or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    n, k = e["n"], e["k"]
    g = e["g"]
    R, NT, NWR, WSK, UPW, NACC, CL, MC, cap, fl = e["cfg"]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    pk = m < g["tokr"] or g["tokr"] == 8
    tm = _x3__xmap(x, m, k, m if pk else g["tokr"], WSK * UPW)
    params = tm + struct.pack("<QQQQQiIiiii", e["stream"].data_ptr(), x.data_ptr(), out.data_ptr(), e["tab"].data_ptr(),
                              state.get("trc", 0),
                              e["pmax"], e["b4"], m, n, k, 0)
    state["mod"].launch(e["f"], e["ncta"], e["nthr"], e["smem"], params)
    return out


# ======================================================================== module R4: R4/rt4.py
# Epilogue pattern (partials staged [token][row] in shared memory, line-coalesced red.add.v4.f32) based on vjeux's rk router GEMM (vjeux/okc-2026, Apache-2.0, as shipped in alygg77/okc-muse-spark-kernels); code rewritten.
"""rt4 (agent R4, claude/t3/R4/rt4.py): router out[M,256] fp32 = x[M,8192] bf16 @ (hi + lo)[256,8192]^T, FP32
accumulation, for 97 <= M <= 128 and 193 <= M <= 256 (run_rt4 returns None elsewhere: the base handles it).

Generated by claude/t3/R4/mk_rt4.py from the lab module r4lab.py. Kernel r4 (sm_90a, NVRTC):
- split-K SS-wgmma GEMM; A = x tile (64 tokens, TMA 2D, 128B swizzle), B = [hi 64 rows ; lo 64 rows] stacked in the
  wgmma N dimension (one m64n128k16 per k16 step: x is read once for hi and lo; hi + lo partials are added in
  registers at the end, which is exact for one-hot rows since hi + lo is exact in fp32);
- W (hi/lo of the FP32 router weight, lossless BF16 pair) prepacked at prepare into 128B-swizzled stage images
  (one contiguous 16 KB 1D bulk copy per stage; idea from agent K's kg.py), packed-W address baked into the code;
- tile 64 W rows x 64 tokens x K/S (S = 16 at M <= 128, 8 at M <= 256): grid 4 x ceil(M/64) x S = 128 CTAs + 4
  zeroing CTAs (blockIdx 0-3: clear out, release counter; compute CTAs wait on it before their adds: agent Y's ya
  scheme; no co-residency requirement);
- fast prologue: stage 0's barrier and copies are issued before the other barrier inits;
- epilogue: partial staged [token][row] in smem, line-coalesced red.add.v4.f32 into out (alygg77's rk epilogue
  pattern, Apache-2.0).
The adds into out are order-dependent (run-to-run differences ~1e-7 relative, inside the router tolerance).
"""
import ctypes
import os
import struct

import torch

_r4_HEAD = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) P { TMap tmx; const unsigned char* wp; float* out; int* zcnt; u64* trc; int M; int ntt; int pad0; int pad1;
                         unsigned char perm[136]; };
__device__ __forceinline__ u64 gtime() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u32 b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect(u32 b, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(u32 b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void mbar_wait(u32 b, int ph) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}\n"
               :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void tma2d(u32 dst, const void* tm, int c0, int c1, u32 bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ u64 mkdesc(u32 saddr) {   // K-major, 128 B rows, 128B swizzle, SBO 1024
  return (u64)((saddr & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)(1024 >> 4) << 32) | ((u64)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void red4(float* a, float x, float y, float z, float w) {
  asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
}
#define TRK(i) do { if (TRC) p.trc[blockIdx.x * 16 + (i)] = gtime(); } while (0)
"""


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: 2026 Nathen Fernandes <https://github.com/NathenFernandes/muse-bf16-proj>
# SPDX-License-Identifier: MIT
# Copied from NathenFernandes/muse-bf16-proj, `_mma` wgmma wrapper generator (via alygg77/okc-muse-spark-kernels).
# Changes: renamed (generated function, parameters, u64 type) and the emitted text laid out over several lines; the wgmma instruction is unchanged.
# License text: LICENSES/MIT.txt; notices: NOTICE (repository root).
def _r4__wg_ss(n):
    nr = n // 2
    regs = ", ".join(f"%{i}" for i in range(nr))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nr))
    return (f'__device__ __forceinline__ void wg_ss_n{n}(float* d, u64 da, u64 db) {{\n'
            f'  asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{nr}, %{nr + 1}, 1, 1, 1, 0, 0;\\n"\n'
            f'    : {outs} : "l"(da), "l"(db));\n}}\n')
# SPDX-SnippetEnd


_r4_BODY = r"""
template <int N> __device__ __forceinline__ void WGSS(float* d, u64 da, u64 db);
template <> __device__ __forceinline__ void WGSS<128>(float* d, u64 da, u64 db) { wg_ss_n128(d, da, db); }
template <> __device__ __forceinline__ void WGSS<256>(float* d, u64 da, u64 db) { wg_ss_n256(d, da, db); }
struct WPARAM { static __device__ __forceinline__ const unsigned char* get(const P& p) { return p.wp; } };

// R: W rows per CTA (64 / 128); NWG: consumer warpgroups (64 tokens each); S: K splits (NKB = 128 / S k-blocks of 64);
// NS: ring stages; CL: 1, or 2 = cluster pair of adjacent K splits reduced through DSMEM (reduce-scatter, st.async);
// ZC: 0 = every CTA's helper warp clears a slice of out + grid counter (all CTAs resident), 4 = four zeroing CTAs
// (blockIdx 0..3) + generation counters (agent Y's ya scheme); WA: W base address provider; TRC: globaltimer trace
template <int R, int NWG, int S, int NS, int CL, int ZC, class WA, int TRC, int ZP, int EPI, int PERM, int IO>
__device__ __forceinline__ void r4_body(const P& p) {
  constexpr int NRT = 256 / R, NKB = 128 / S, TOK = 64 * NWG;
  constexpr u32 XB = TOK * 128, WB0 = 2 * R * 128, WB = (IO & 16) ? WB0 * 3 / 4 : WB0, SB = XB + WB;
  constexpr int NPRO = NS < NKB ? NS : NKB;
  constexpr int NJ = R / 8;                         // n8 blocks of combined (hi + lo) results per thread
  // DSMEM push per thread: NWG = 2: the whole warpgroup partial (NJ v4); NWG = 1: half the warps push (NJ v4)
  constexpr u32 PB = (NWG == 2 ? 128 : 64) * NJ * 16;
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  const u32 base = (smem_u32(sm_raw) + 1023u) & ~1023u;
  const u32 full = base, empty = base + 8 * NS, zbar = base + 16 * NS, rbar = zbar + 8;
  const u32 ring = base + 1024;
  const u32 rbuf = ring + NS * (XB + WB0);
  // EPI 1: per-warpgroup staging [64 tokens][R + 4] fp32; with one consumer warpgroup and no cluster the ring is idle
  // once the warpgroup has waited for its last k-block (every copy has landed), so the staging aliases ring slot 0
  const u32 ebuf = (NWG == 1 && CL == 1) ? ring : rbuf + (CL == 2 ? PB : 0);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ncomp = (int)gridDim.x - ZC;
  // compute CTA index (ZC: zeroing CTAs first (ZP 0) or last (ZP 1))
  const int cb = ZP ? ((int)blockIdx.x < ncomp ? (int)blockIdx.x : -1 - ((int)blockIdx.x - ncomp)) : (int)blockIdx.x - ZC;
  if (ZC && cb < 0) {   // zeroing CTA: clear 1/ZC of out, publish with a release add on counter A (idle SM: fast)
    constexpr int ZD = ZC ? ZC : 1;
    const int zi = -1 - cb;
    const int n4 = p.M * 64, per = (n4 + ZD - 1) / ZD;
    const int z0 = (ZP ? zi : (int)blockIdx.x) * per, z1 = min(n4, z0 + per);
    for (int i = z0 + tid; i < z1; i += blockDim.x) reinterpret_cast<float4*>(p.out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncthreads();
    if (tid == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(p.zcnt) : "memory");
    return;
  }
  const int ntt = p.ntt;
  // CL 2: pair = two adjacent K splits (DSMEM reduce); CL 3: pair = two adjacent token tiles (W multicast)
  // PERM: logical unit (CTA or pair) from a prepare-time GPC-aware permutation of the launch order
  const int rank = CL >= 2 ? (cb & 1) : 0;
  const int u0 = CL >= 2 ? (cb >> 1) : cb;
  const int rest = PERM ? (int)p.perm[u0] : u0;
  const int ntu = CL == 3 ? (ntt >> 1) : ntt;
  const int rt = rest % NRT, ttu = (rest / NRT) % ntu, spu = rest / (NRT * ntu);
  const int tt = CL == 3 ? 2 * ttu + rank : ttu;
  const int sp = CL == 2 ? 2 * spu + rank : spu;
  const int kb0 = sp * NKB;
  const unsigned char* wsrc = WA::get(p) + ((size_t)rt * 128 + kb0) * WB;
  if ((IO & 7) == 6 && warp == 0) {
    // fast prologue + parallel issue: lane 0 does stage 0 first, then the other barriers; lanes 1.. issue stage = lane
    if (lane == 0) {
      TRK(0);
      mbar_init(full, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      mbar_expect(full, SB);
      bulk_g2s(ring + XB, wsrc, WB, full);
      tma2d(ring, &p.tmx, kb0 * 64, tt * TOK, full);
#pragma unroll 1
      for (int s = 1; s < NS; ++s) mbar_init(full + 8 * s, 1);
#pragma unroll 1
      for (int s = 0; s < NS; ++s) mbar_init(empty + 8 * s, NWG);
      mbar_init(zbar, 1);
      mbar_init(rbar, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      if (CL == 2) mbar_expect(rbar, PB);
    }
    __syncwarp();
    if (lane >= 1 && lane < NPRO) {
      const int s = lane;
      mbar_expect(full + 8 * s, SB);
      bulk_g2s(ring + s * SB + XB, wsrc + (size_t)s * WB, WB, full + 8 * s);
      tma2d(ring + s * SB, &p.tmx, (kb0 + s) * 64, tt * TOK, full + 8 * s);
    }
    __syncwarp();
    if (lane == 0) TRK(10);
  }
  if (tid == 0 && (IO & 7) >= 3 && (IO & 7) <= 4) {
    // fast prologue: stage 0's barrier + copies first (IO 3: W then x, IO 4: x then W), then everything else
    TRK(0);
    mbar_init(full, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect(full, SB);
    if ((IO & 7) == 4) tma2d(ring, &p.tmx, kb0 * 64, tt * TOK, full);
    bulk_g2s(ring + XB, wsrc, WB, full);
    if ((IO & 7) == 3) tma2d(ring, &p.tmx, kb0 * 64, tt * TOK, full);
#pragma unroll 1
    for (int s = 1; s < NS; ++s) mbar_init(full + 8 * s, 1);
#pragma unroll 1
    for (int s = 0; s < NS; ++s) mbar_init(empty + 8 * s, NWG);
    mbar_init(zbar, 1);
    mbar_init(rbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#pragma unroll 1
    for (int s = 1; s < NPRO; ++s) {
      mbar_expect(full + 8 * s, SB);
      if ((IO & 7) == 4) tma2d(ring + s * SB, &p.tmx, (kb0 + s) * 64, tt * TOK, full + 8 * s);
      bulk_g2s(ring + s * SB + XB, wsrc + (size_t)s * WB, WB, full + 8 * s);
      if ((IO & 7) == 3) tma2d(ring + s * SB, &p.tmx, (kb0 + s) * 64, tt * TOK, full + 8 * s);
    }
    if (CL == 2) mbar_expect(rbar, PB);
    TRK(10);
  }
  if (tid == 0 && (IO & 7) < 3) {
    TRK(0);
    if (TRC) { unsigned sid; asm volatile("mov.u32 %0, %%smid;" : "=r"(sid)); p.trc[blockIdx.x * 16 + 15] = sid; }
#pragma unroll 1
    for (int s = 0; s < NS; ++s) { mbar_init(full + 8 * s, 1); mbar_init(empty + 8 * s, NWG); }
    mbar_init(zbar, 1);
    mbar_init(rbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (IO == 1 || IO == 2) {   // IO 1: per stage x then W; IO 2: every W stage, then every x stage
#pragma unroll 1
      for (int s = 0; s < NPRO; ++s) {
        mbar_expect(full + 8 * s, SB);
        if (IO == 1) tma2d(ring + s * SB, &p.tmx, (kb0 + s) * 64, tt * TOK, full + 8 * s);
        bulk_g2s(ring + s * SB + XB, wsrc + (size_t)s * WB, WB, full + 8 * s);
      }
      if (IO == 2) {
#pragma unroll 1
        for (int s = 0; s < NPRO; ++s) tma2d(ring + s * SB, &p.tmx, (kb0 + s) * 64, tt * TOK, full + 8 * s);
      }
    } else if (CL == 3) {   // own x tiles first (no partner dependency), then the W halves multicast after the cluster barrier
#pragma unroll 1
      for (int s = 0; s < NPRO; ++s) {
        mbar_expect(full + 8 * s, SB);
        tma2d(ring + s * SB, &p.tmx, (kb0 + s) * 64, tt * TOK, full + 8 * s);
      }
    } else {
#pragma unroll 1
    for (int s = 0; s < NPRO; ++s) {
      mbar_expect(full + 8 * s, SB);
      bulk_g2s(ring + s * SB + XB, wsrc + (size_t)s * WB, WB, full + 8 * s);
      tma2d(ring + s * SB, &p.tmx, (kb0 + s) * 64, tt * TOK, full + 8 * s);
    }
    }
    if (CL == 2) mbar_expect(rbar, PB);
    TRK(10);
  }
  __syncthreads();
  if (CL >= 2) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");   // my barriers are initialised
  if (CL == 3) asm volatile("barrier.cluster.wait.aligned;" ::: "memory");             // ... and the partner's
  if (CL == 3 && tid == 0) {
    static_assert(CL != 3 || NKB <= NS, "CL 3: every stage resident (no cross-CTA slot reuse)");
    // rank 0 copies the hi half (R rows), rank 1 the lo half of each W stage image into both CTAs
#pragma unroll 1
    for (int s = 0; s < NPRO; ++s)
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;"
                   :: "r"(ring + s * SB + XB + rank * (WB / 2)), "l"(wsrc + (size_t)s * WB + rank * (WB / 2)), "r"(WB / 2),
                      "r"(full + 8 * s), "h"((unsigned short)3) : "memory");
    TRK(11);
  }
  if (warp == 4 * NWG) {   // producer: the rest of the ring
    if (lane == 0 && NKB > NS) {
#pragma unroll 1
      for (int i = NS; i < NKB; ++i) {
        const int s = i % NS;
        mbar_wait(empty + 8 * s, ((i / NS) - 1) & 1);
        mbar_expect(full + 8 * s, SB);
        bulk_g2s(ring + s * SB + XB, wsrc + (size_t)i * WB, WB, full + 8 * s);
        tma2d(ring + s * SB, &p.tmx, (kb0 + i) * 64, tt * TOK, full + 8 * s);
      }
      TRK(11);
    }
    return;
  }
  if (warp == 4 * NWG + 1) {   // helper: zero barrier
    if (ZC) {   // generation from counter B (relaxed atomic), wait for counter A >= ZC (gen + 1)
      if (lane == 0) {
        int v;
        asm volatile("atom.relaxed.gpu.global.add.s32 %0, [%1], 1;" : "=r"(v) : "l"(p.zcnt + 256) : "memory");
        const int target = (v / ncomp + 1) * ZC;
        TRK(8);
        const u64 tz = gtime();
        for (;;) {
          int a;
          asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(a) : "l"(p.zcnt) : "memory");
          if (a - target >= 0) break;
          if (gtime() - tz > 200000000ull) __trap();
          __nanosleep(32);
        }
        TRK(9);
        mbar_arrive(zbar);
      }
      return;
    }
    int target = 0;
    if (lane == 0) {
      int v;
      asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p.zcnt) : "memory");
      target = (v / (int)gridDim.x + 1) * (int)gridDim.x;
    }
    const int n4 = p.M * 64;
    const int per = (n4 + (int)gridDim.x - 1) / (int)gridDim.x;
    const int z0 = cb * per, z1 = min(n4, z0 + per);
    for (int i = z0 + lane; i < z1; i += 32) reinterpret_cast<float4*>(p.out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncwarp();
    if (lane == 0) {
      asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(p.zcnt) : "memory");
      TRK(8);
      const u64 tz = gtime();
      for (;;) {
        int v;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p.zcnt) : "memory");
        if (v - target >= 0) break;
        if (gtime() - tz > 200000000ull) __trap();   // 0.2 s: not all CTAs resident -> fail loudly, never hang
        __nanosleep(64);
      }
      TRK(9);
      mbar_arrive(zbar);
    }
    return;
  }
  // consumer warpgroup g: tokens [64 g, 64 g + 64) of the token tile, all 2R stacked hi|lo rows
  const int g = warp >> 2, ct = tid & 127, wq = warp & 3;
  float d[R];
#pragma unroll
  for (int i = 0; i < R; ++i) d[i] = 0.f;
#pragma unroll 1
  for (int i = 0; i < NKB; ++i) {
    const int s = i % NS;
    mbar_wait(full + 8 * s, (i / NS) & 1);
    if (TRC && ct == 0 && g == 0) { if (i == 0) TRK(1); if (i == NKB - 1) TRK(2); }
    wg_fence();
    const u32 a = ring + s * SB + g * 8192, b = ring + s * SB + XB;
#pragma unroll
    for (int k16 = 0; k16 < 4; ++k16) WGSS<2 * R>(d, mkdesc(a + 32 * k16), mkdesc(b + 32 * k16));
    wg_commit();
    wg_wait<1>();
    if (i > 0 && ct == 0) mbar_arrive(empty + 8 * ((i - 1) % NS));
  }
  wg_wait<0>();
#pragma unroll
  for (int i = 0; i < R; ++i) asm volatile("" : "+f"(d[i]) :: "memory");
  if (TRC && ct == 0 && g == 0) TRK(3);
  // hi + lo: n8 block j (hi rows 8j..) and j + R/8 (lo rows 8j..) are held by the same thread
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) d[4 * j + e] += d[4 * (j + NJ) + e];
  if (CL == 2) {
    // reduce-scatter with the partner (same tile, the other K half): NWG = 2: rank r finalises warpgroup r's 64 tokens,
    // the other warpgroup pushes its whole partial; NWG = 1: rank r finalises rows [32 r, 32 r + 32) (warps 2r, 2r+1)
    const bool push = NWG == 2 ? (g != rank) : ((wq >> 1) != rank);
    const int slot = NWG == 2 ? ct : ((wq & 1) * 32 + lane);
    constexpr int NSLOT = NWG == 2 ? 128 : 64;
    if (push) {
      asm volatile("barrier.cluster.wait.aligned;" ::: "memory");   // the partner's rbar is armed (long done)
      u32 rdst, rmb;
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rdst) : "r"(rbuf + slot * 16), "r"(rank ^ 1));
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rmb) : "r"(rbar), "r"(rank ^ 1));
#pragma unroll
      for (int j = 0; j < NJ; ++j)
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1,%2,%3,%4}, [%5];"
                     :: "r"(rdst + j * NSLOT * 16), "r"(__float_as_uint(d[4 * j])), "r"(__float_as_uint(d[4 * j + 1])),
                        "r"(__float_as_uint(d[4 * j + 2])), "r"(__float_as_uint(d[4 * j + 3])), "r"(rmb) : "memory");
      if (TRC && ct == 0) TRK(6);
      return;
    }
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    mbar_wait(rbar, 0);
    if (TRC && (ct & 63) == 0) TRK(7);
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      float4 v;
      asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                   : "r"(rbuf + (j * NSLOT + slot) * 16));
      d[4 * j] += v.x; d[4 * j + 1] += v.y; d[4 * j + 2] += v.z; d[4 * j + 3] += v.w;
    }
  }
  if (EPI == 2) {
    // per-warp staging: warp wq stages its own 16 token rows [16][R + 4] fp32 and red.adds them line-coalesced
    // (2 rows x R*4 B per instruction); no warpgroup barrier
    constexpr int TP = R + 4, C4 = R / 4, RPI = 32 / C4;   // rows per red instruction
    const u32 T = ebuf + (u32)(g * 4 + wq) * (16 * TP * 4);
    const int rl = lane >> 2, q = lane & 3;
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const int cl = 8 * j + 2 * q;
      asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "r"(T + (rl * TP + cl) * 4), "f"(d[4 * j]), "f"(d[4 * j + 1]) : "memory");
      asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "r"(T + ((rl + 8) * TP + cl) * 4), "f"(d[4 * j + 2]), "f"(d[4 * j + 3]) : "memory");
    }
    __syncwarp();
    mbar_wait(zbar, 0);
    if (TRC && (ct & 63) == 0) TRK(4);
    const int rot = IO >= 8 ? (sp * (16 / S > 0 ? 16 / S : 1)) & 15 : 0;
#pragma unroll
    for (int it = 0; it < 16 / RPI; ++it) {
      const int tk = (it * RPI + lane / C4 + rot) & 15, c4 = lane % C4;
      const int tok = tt * TOK + 64 * g + 16 * wq + tk;
      float4 v;
      asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(T + (tk * TP + 4 * c4) * 4));
      if (tok < p.M) red4(p.out + (size_t)tok * 256 + rt * R + 4 * c4, v.x, v.y, v.z, v.w);
    }
  } else if (EPI == 1) {
    // rk-style: stage [64 tokens][R rows] fp32 in smem, then each warp red.adds 2 full token rows (R * 4 B each) per
    // instruction (line-coalesced) instead of 16 rows x 32 B
    constexpr int TP = R + 4;
    const u32 T = ebuf + (u32)g * (64 * TP * 4);
    const int rl = 16 * wq + (lane >> 2), q = lane & 3;
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const int cl = 8 * j + 2 * q;
      asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "r"(T + (rl * TP + cl) * 4), "f"(d[4 * j]), "f"(d[4 * j + 1]) : "memory");
      asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "r"(T + ((rl + 8) * TP + cl) * 4), "f"(d[4 * j + 2]), "f"(d[4 * j + 3]) : "memory");
    }
    asm volatile("bar.sync %0, 128;" :: "r"(1 + g) : "memory");
    mbar_wait(zbar, 0);
    if (TRC && (ct & 63) == 0) TRK(4);
    constexpr int C4 = R / 4;                       // float4 per token row
    // IO >= 8: rotate the token-row order by the K split (spreads the S same-address adds of a tile over time)
    const int rot = IO >= 8 ? (sp * (64 / S)) & 63 : 0;
#pragma unroll 4
    for (int idx = ct; idx < 64 * C4; idx += 128) {
      const int tk = ((idx / C4) + rot) & 63, c4 = idx % C4;
      const int tok = tt * TOK + 64 * g + tk;
      float4 v;
      asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(T + (tk * TP + 4 * c4) * 4));
      if (tok < p.M) red4(p.out + (size_t)tok * 256 + rt * R + 4 * c4, v.x, v.y, v.z, v.w);
    }
  } else {
  mbar_wait(zbar, 0);
  if (TRC && (ct & 63) == 0) TRK(4);
  const int q = lane & 3, odd = q & 1;
  const int tok = tt * TOK + 64 * g + 16 * wq + (lane >> 2) + 8 * odd;
  float* orow = p.out + (size_t)tok * 256 + rt * R + 2 * (q & 2);
  const bool ok = tok < p.M;
#pragma unroll
  for (int j = 0; j < NJ; ++j) {
    // lane q holds (row r: cols 8j+2q, +1) in d0,d1 and (row r+8: same cols) in d2,d3; pair (q, q^1) swaps so the even
    // lane owns row r cols 8j + 2q .. +3 and the odd lane row r + 8 cols 8j + 2(q-1) .. +3
    const float s0 = odd ? d[4 * j] : d[4 * j + 2], s1 = odd ? d[4 * j + 1] : d[4 * j + 3];
    const float r0 = __shfl_xor_sync(0xffffffffu, s0, 1), r1 = __shfl_xor_sync(0xffffffffu, s1, 1);
    float v0, v1, v2, v3;
    if (odd) { v0 = r0; v1 = r1; v2 = d[4 * j + 2]; v3 = d[4 * j + 3]; }
    else { v0 = d[4 * j]; v1 = d[4 * j + 1]; v2 = r0; v3 = r1; }
    if (ok) red4(orow + 8 * j, v0, v1, v2, v3);
  }
  }
  if (TRC && (ct & 63) == 0) TRK(5);
}

// ---------------------------------------------------------------------------------------------------------------------
// r4p (phase-ordered, NWG = 2, 128 tokens x R rows x K/S): every stage is issued at kernel start (NKB small); the CTA's
// stream is ordered [W kb + x(tokens 0-63) kb] for kb = 0..NKB-1, then [x(tokens 64-127) kb]: warpgroup 0 finishes
// when ~3/4 of the CTA's bytes have landed and its coalesced red.adds overlap the rest of the stream; only warpgroup 1's
// 64 x R partial is exposed after the last byte.
template <int R, int S, int ZC, class WA, int TRC, int ZP>
__device__ __forceinline__ void r4p_body(const P& p) {
  constexpr int NRT = 256 / R, NKB = 128 / S, TOK = 128;
  constexpr u32 XH = 64 * 128, WB = 2 * R * 128, SA = XH + WB;
  constexpr int NJ = R / 8, TP = R + 4;
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  const u32 base = (smem_u32(sm_raw) + 1023u) & ~1023u;
  const u32 full = base, zbar = base + 16 * NKB;
  const u32 ringA = base + 1024, ringB = ringA + NKB * SA, ebuf = ringB + NKB * XH;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ncomp = (int)gridDim.x - ZC;
  const int cb = ZP ? ((int)blockIdx.x < ncomp ? (int)blockIdx.x : -1 - ((int)blockIdx.x - ncomp)) : (int)blockIdx.x - ZC;
  if (cb < 0) {
    const int zi = -1 - cb;
    const int n4 = p.M * 64, per = (n4 + ZC - 1) / ZC;
    const int z0 = (ZP ? zi : (int)blockIdx.x) * per, z1 = min(n4, z0 + per);
    for (int i = z0 + tid; i < z1; i += blockDim.x) reinterpret_cast<float4*>(p.out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncthreads();
    if (tid == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(p.zcnt) : "memory");
    return;
  }
  const int ntt = p.ntt;
  const int rt = cb % NRT, tt = (cb / NRT) % ntt, sp = cb / (NRT * ntt);
  const int kb0 = sp * NKB;
  const unsigned char* wsrc = WA::get(p) + ((size_t)rt * 128 + kb0) * WB;
  if (tid == 0) {
    TRK(0);
#pragma unroll
    for (int s = 0; s < 2 * NKB; ++s) mbar_init(full + 8 * s, 1);
    mbar_init(zbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#pragma unroll
    for (int s = 0; s < NKB; ++s) {
      mbar_expect(full + 8 * s, SA);
      bulk_g2s(ringA + s * SA + XH, wsrc + (size_t)s * WB, WB, full + 8 * s);
      tma2d(ringA + s * SA, &p.tmx, (kb0 + s) * 64, tt * TOK, full + 8 * s);
    }
#pragma unroll
    for (int s = 0; s < NKB; ++s) {
      mbar_expect(full + 8 * (NKB + s), XH);
      tma2d(ringB + s * XH, &p.tmx, (kb0 + s) * 64, tt * TOK + 64, full + 8 * (NKB + s));
    }
    TRK(10);
  }
  __syncthreads();
  if (warp == 8) {   // helper: wait for the zeroing CTAs (generation counters)
    if (lane == 0) {
      int v;
      asm volatile("atom.relaxed.gpu.global.add.s32 %0, [%1], 1;" : "=r"(v) : "l"(p.zcnt + 256) : "memory");
      const int target = (v / ncomp + 1) * ZC;
      const u64 tz = gtime();
      for (;;) {
        int a;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(a) : "l"(p.zcnt) : "memory");
        if (a - target >= 0) break;
        if (gtime() - tz > 200000000ull) __trap();
        __nanosleep(32);
      }
      TRK(9);
      mbar_arrive(zbar);
    }
    return;
  }
  if (warp > 8) return;
  const int g = warp >> 2, ct = tid & 127, wq = warp & 3;
  float d[R];
#pragma unroll
  for (int i = 0; i < R; ++i) d[i] = 0.f;
#pragma unroll
  for (int i = 0; i < NKB; ++i) {
    mbar_wait(full + 8 * i, 0);
    if (g) mbar_wait(full + 8 * (NKB + i), 0);
    if (TRC && ct == 0) { if (i == 0) TRK(1 + 6 * g); if (i == NKB - 1) TRK(2 + 6 * g); }
    wg_fence();
    const u32 a = g ? ringB + i * XH : ringA + i * SA, b = ringA + i * SA + XH;
#pragma unroll
    for (int k16 = 0; k16 < 4; ++k16) WGSS<2 * R>(d, mkdesc(a + 32 * k16), mkdesc(b + 32 * k16));
    wg_commit();
  }
  wg_wait<0>();
#pragma unroll
  for (int i = 0; i < R; ++i) asm volatile("" : "+f"(d[i]) :: "memory");
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) d[4 * j + e] += d[4 * (j + NJ) + e];
  if (TRC && ct == 0) TRK(3 + 10 * g);
  const u32 T = ebuf + (u32)g * (64 * TP * 4);
  const int rl = 16 * wq + (lane >> 2), q = lane & 3;
#pragma unroll
  for (int j = 0; j < NJ; ++j) {
    const int cl = 8 * j + 2 * q;
    asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "r"(T + (rl * TP + cl) * 4), "f"(d[4 * j]), "f"(d[4 * j + 1]) : "memory");
    asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "r"(T + ((rl + 8) * TP + cl) * 4), "f"(d[4 * j + 2]), "f"(d[4 * j + 3]) : "memory");
  }
  asm volatile("bar.sync %0, 128;" :: "r"(1 + g) : "memory");
  mbar_wait(zbar, 0);
  if (TRC && ct == 0) TRK(4 + g);
  constexpr int C4 = R / 4;
#pragma unroll 4
  for (int idx = ct; idx < 64 * C4; idx += 128) {
    const int tk = idx / C4, c4 = idx % C4;
    const int tok = tt * TOK + 64 * g + tk;
    float4 v;
    asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(T + (tk * TP + 4 * c4) * 4));
    if (tok < p.M) red4(p.out + (size_t)tok * 256 + rt * R + 4 * c4, v.x, v.y, v.z, v.w);
  }
  if (TRC && ct == 0) TRK(6 + 6 * g);
}

// ---------------------------------------------------------------------------------------------------------------------
// r4c: r4 (64 W rows x 64 tokens, one consumer warpgroup, 4 zeroing CTAs first, coalesced red epilogue) with the W
// stages in agent Y's lossless 24-bit hi/lo code format (per value: hi low byte, lo low byte, 4-bit hi code
// sign | e7 - YB, 4-bit lo code sign | (hi e7 - 4 - lo e7)); a stage image = 6 x 128 x 16 B code segments (12 KB: thread
// t's 16-B segment i at (i * 128 + t) * 16) + a 256-B escape trailer (u32 count, then (u16 pos, u16 hi, u16 lo, u16 0)).
// The consumer warpgroup decodes stage i into a double-buffered 128B-swizzled [hi 64 rows ; lo 64 rows] BF16 tile while
// stage i-1's wgmma runs, patches the escapes (exact hi / lo bits) in smem, then issues stage i's wgmma.
// RAWL: the CTA's last k-block comes as a raw 16 KB BF16 image into a third tile (no decode after the last byte).
__device__ __forceinline__ u32 prmt2(u32 a, u32 b, u32 sel) { u32 r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel)); return r; }
template <u32 B4>
__device__ __forceinline__ void dec_pair(u32 hl0, u32 hl1, u32 hn, u32 ll0, u32 ll1, u32 ln, uint4& ah, uint4& al) {
  // agent Y's decoder (yr.py): SIMD byte ops on the 4-bit codes + PRMT with the low bytes
  const u32 e0 = (hn & 0x07070707u) + B4, e1 = ((hn >> 4) & 0x07070707u) + B4;
  const u32 h0 = e0 | ((hn << 4) & 0x80808080u), h1 = e1 | (hn & 0x80808080u);
  ah.x = prmt2(hl0, h0, 0x5140); ah.y = prmt2(hl0, h0, 0x7362); ah.z = prmt2(hl1, h1, 0x5140); ah.w = prmt2(hl1, h1, 0x7362);
  const u32 g0 = (e0 - 0x04040404u - (ln & 0x07070707u)) | ((ln << 4) & 0x80808080u);
  const u32 g1 = (e1 - 0x04040404u - ((ln >> 4) & 0x07070707u)) | (ln & 0x80808080u);
  al.x = prmt2(ll0, g0, 0x5140); al.y = prmt2(ll0, g0, 0x7362); al.z = prmt2(ll1, g1, 0x5140); al.w = prmt2(ll1, g1, 0x7362);
}
__device__ __forceinline__ uint4 lds128u(u32 a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }
__device__ __forceinline__ void sts128u(u32 a, uint4 v) { asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
template <int S, int NS, class WA, int TRC, int RAWL, u32 B4, int DX = 0>
__device__ __forceinline__ void r4c_body(const P& p) {
  // 2 decoder warpgroups (256 threads; warpgroup 0 also issues the wgmma), 3 decoded BF16 tiles, producer + helper warps
  constexpr int NKB = 128 / S, R = 64, NJ = 8, TOK = 64;
  constexpr u32 XB = 8192, CB = 12288 + 256, SB = 21504, DBB = 16384;   // x tile, code image, ring slot, BF16 tile
  constexpr int NPRO = NS < NKB ? NS : NKB;
  static_assert(RAWL == 0, "r4c v2: RAWL not supported");
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  const u32 base = (smem_u32(sm_raw) + 1023u) & ~1023u;
  const u32 full = base, empty = base + 8 * NS, zbar = base + 16 * NS;
  const u32 ring = base + 1024, dbuf = ring + NS * SB;   // dbuf: 3 decoded tiles
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cb = (int)blockIdx.x - 4;
  if (cb < 0) {   // zeroing CTA
    const int n4 = p.M * 64, per = (n4 + 3) / 4;
    const int z0 = (int)blockIdx.x * per, z1 = min(n4, z0 + per);
    for (int i = z0 + tid; i < z1; i += blockDim.x) reinterpret_cast<float4*>(p.out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncthreads();
    if (tid == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(p.zcnt) : "memory");
    return;
  }
  const int ntt = p.ntt;
  const int rt = cb & 3, tt = (cb >> 2) % ntt, sp = (cb >> 2) / ntt;
  const int kb0 = sp * NKB;
  const unsigned char* wsrc = WA::get(p) + ((size_t)rt * 128 + kb0) * CB;
  if (tid == 0) {   // fast prologue: stage 0 first
    TRK(0);
    mbar_init(full, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect(full, XB + CB);
    bulk_g2s(ring + XB, wsrc, CB, full);
    tma2d(ring, &p.tmx, kb0 * 64, tt * TOK, full);
#pragma unroll 1
    for (int s = 1; s < NS; ++s) mbar_init(full + 8 * s, 1);
#pragma unroll 1
    for (int s = 0; s < NS; ++s) mbar_init(empty + 8 * s, 1);
    mbar_init(zbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#pragma unroll 1
    for (int s = 1; s < NPRO; ++s) {
      mbar_expect(full + 8 * s, XB + CB);
      bulk_g2s(ring + s * SB + XB, wsrc + (size_t)s * CB, CB, full + 8 * s);
      tma2d(ring + s * SB, &p.tmx, (kb0 + s) * 64, tt * TOK, full + 8 * s);
    }
    TRK(10);
  }
  __syncthreads();
  if (warp == 8) {   // producer: the rest of the ring
    if (lane == 0 && NKB > NS) {
#pragma unroll 1
      for (int i = NS; i < NKB; ++i) {
        const int s = i % NS;
        mbar_wait(empty + 8 * s, ((i / NS) - 1) & 1);
        mbar_expect(full + 8 * s, XB + CB);
        bulk_g2s(ring + s * SB + XB, wsrc + (size_t)i * CB, CB, full + 8 * s);
        tma2d(ring + s * SB, &p.tmx, (kb0 + i) * 64, tt * TOK, full + 8 * s);
      }
    }
    return;
  }
  if (warp == 9) {   // helper: wait for the zeroing CTAs (generation counters)
    if (lane == 0) {
      const int ncomp = (int)gridDim.x - 4;
      int v;
      asm volatile("atom.relaxed.gpu.global.add.s32 %0, [%1], 1;" : "=r"(v) : "l"(p.zcnt + 256) : "memory");
      const int target = (v / ncomp + 1) * 4;
      const u64 tz = gtime();
      for (;;) {
        int a;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(a) : "l"(p.zcnt) : "memory");
        if (a - target >= 0) break;
        if (gtime() - tz > 200000000ull) __trap();
        __nanosleep(32);
      }
      TRK(9);
      mbar_arrive(zbar);
    }
    return;
  }
  const int dt = tid;                               // decoder thread 0..255: row dt >> 2, k-chunks 2 (dt & 3), +1
  const int drow = dt >> 2, dkc = (dt & 3) * 2;
  const int ct = tid & 127, wq = warp & 3;
  // decode k-block i (stage slot s) into tile (i % 3): 2 chunks of 8 hi + 8 lo values per thread, escapes patched
  auto decode = [&](int i, u32 code, u32 btile) {
    if (DX == 2) return;
    const uint4 hv = lds128u(code + (0 * 256 + dt) * 16), lv = lds128u(code + (1 * 256 + dt) * 16);
    const uint4 nv = lds128u(code + (2 * 256 + dt) * 16);
    u32 flg;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(flg) : "r"(code + 12288 + (dt >> 5) * 4));
    const u32 rowa = btile + drow * 128, rowb = btile + (64 + drow) * 128, sw = (u32)(drow & 7);
    uint4 ah, al;
    dec_pair<B4>(hv.x, hv.y, nv.x, lv.x, lv.y, nv.z, ah, al);
    sts128u(rowa + (((dkc + 0) ^ sw) << 4), ah); sts128u(rowb + (((dkc + 0) ^ sw) << 4), al);
    dec_pair<B4>(hv.z, hv.w, nv.y, lv.z, lv.w, nv.w, ah, al);
    sts128u(rowa + (((dkc + 1) ^ sw) << 4), ah); sts128u(rowb + (((dkc + 1) ^ sw) << 4), al);
    if ((flg >> (dt & 31)) & 1u) {   // this thread's chunks hold escapes: exact hi / lo bits from the trailer
      u32 ne;
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(ne) : "r"(code + 12288 + 32));
#pragma unroll 1
      for (u32 e = 0; e < ne; ++e) {
        uint2 ent;
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(ent.x), "=r"(ent.y) : "r"(code + 12288 + 40 + 8 * e));
        const u32 pos = ent.x & 0xFFFFu, row = pos >> 6, k = pos & 63;
        if ((int)((row << 2) | (k >> 4)) == dt) {
          const u32 off = (((k >> 3) ^ (row & 7)) << 4) + (k & 7) * 2;
          asm volatile("st.shared.u16 [%0], %1;" :: "r"(btile + row * 128 + off), "h"((unsigned short)(ent.x >> 16)) : "memory");
          asm volatile("st.shared.u16 [%0], %1;" :: "r"(btile + (64 + row) * 128 + off), "h"((unsigned short)(ent.y & 0xFFFFu)) : "memory");
        }
      }
    }
    if (DX != 1) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  };
  if (warp >= 4) {   // decoder-only warpgroup
#pragma unroll 1
    for (int i = 0; i < NKB; ++i) {
      const int s = i % NS;
      mbar_wait(full + 8 * s, (i / NS) & 1);
      decode(i, ring + s * SB + XB, dbuf + (u32)(i % 3) * DBB);
      asm volatile("bar.sync 1, 256;" ::: "memory");
    }
    return;
  }
  float d[R];
#pragma unroll
  for (int i = 0; i < R; ++i) d[i] = 0.f;
#pragma unroll 1
  for (int i = 0; i < NKB; ++i) {
    const int s = i % NS;
    mbar_wait(full + 8 * s, (i / NS) & 1);
    if (TRC && tid == 0) { if (i == 0) TRK(1); if (i == NKB - 1) TRK(2); }
    const u32 slot = ring + s * SB;
    const u32 btile = dbuf + (u32)(i % 3) * DBB;
    decode(i, slot + XB, btile);
    asm volatile("bar.sync 1, 256;" ::: "memory");
    wg_fence();
#pragma unroll
    for (int k16 = 0; k16 < 4; ++k16) WGSS<128>(d, mkdesc(slot + 32 * k16), mkdesc(btile + 32 * k16));
    wg_commit();
    wg_wait<1>();
    if (i > 0 && ct == 0) mbar_arrive(empty + 8 * ((i - 1) % NS));
  }
  wg_wait<0>();
#pragma unroll
  for (int i = 0; i < R; ++i) asm volatile("" : "+f"(d[i]) :: "memory");
  if (TRC && ct == 0) TRK(3);
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) d[4 * j + e] += d[4 * (j + NJ) + e];
  // coalesced red epilogue; staging in ring slot 0 (every copy has landed, every wgmma is done, the decoder warpgroup
  // has left its loop after its last decode)
  constexpr int TP = R + 4, C4 = R / 4;
  const u32 T = ring;
  static_assert(64 * TP * 4 <= SB, "staging fits in a ring slot");
  const int rl = 16 * wq + (lane >> 2), q = lane & 3;
#pragma unroll
  for (int j = 0; j < NJ; ++j) {
    const int cl = 8 * j + 2 * q;
    asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "r"(T + (rl * TP + cl) * 4), "f"(d[4 * j]), "f"(d[4 * j + 1]) : "memory");
    asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "r"(T + ((rl + 8) * TP + cl) * 4), "f"(d[4 * j + 2]), "f"(d[4 * j + 3]) : "memory");
  }
  asm volatile("bar.sync 2, 128;" ::: "memory");
  mbar_wait(zbar, 0);
  if (TRC && ct == 0) TRK(4);
#pragma unroll 4
  for (int idx = ct; idx < 64 * C4; idx += 128) {
    const int tk = idx / C4, c4 = idx % C4;
    const int tok = tt * TOK + tk;
    float4 v;
    asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(T + (tk * TP + 4 * c4) * 4));
    if (tok < p.M) red4(p.out + (size_t)tok * 256 + rt * R + 4 * c4, v.x, v.y, v.z, v.w);
  }
  if (TRC && ct == 0) TRK(5);
}
"""


# cfg = (R, NWG, S, NS, CL, ZC, WPB, TRC, ZP, EPI, PH, PERM, IO), or ("c", S, NS, RAWL, TRC) for the 24-bit kernel
_r4_YB = 54   # agent Y's hi e7 window [54, 61] = exponents 108..123


def _r4_kname(cf):
    if cf[0] == "c":
        return "r4c_" + "_".join(str(v) for v in cf[1:])
    return "r4_" + "_".join(str(v) for v in cf)


def _r4_kthreads(cf):
    if cf[0] == "c":
        return 320
    return 128 * cf[1] + 64


def _r4_xcf(cf):
    return tuple(cf) + (0,) * (13 - len(cf))


def _r4_ksmem(cf):
    if cf[0] == "c":
        _, S, NS, RAWL, TRC = cf[:5]
        return 1024 + NS * 21504 + 3 * 16384 + 1024
    R, NWG, S, NS, CL, ZC, WPB, TRC, ZP, EPI, PH, PERM, IO = _r4_xcf(cf)
    if PH:
        nkb = 128 // S
        return 1024 + nkb * (2 * 8192 + 2 * R * 128) + 2 * 64 * (R + 4) * 4 + 1024
    pb = ((128 if NWG == 2 else 64) * (R // 8) * 16) if CL == 2 else 0
    eb = NWG * 64 * (R + 4) * 4 if (EPI >= 1 and not (NWG == 1 and CL == 1)) else 0
    return 1024 + NS * (64 * NWG * 128 + 2 * R * 128) + pb + eb + 1024


def _r4_source(cfgs, waddr):
    """waddr: {R: packed-W device address} for WPB configs (address baked into the code: no parameter load before the
    first W copy; agent R/K's trick incl. a non-foldable runtime zero so ptxas keeps the high half)."""
    s = _r4_HEAD + _r4__wg_ss(128) + _r4__wg_ss(256) + _r4_BODY
    for cf in cfgs:
        if cf[0] == "c":
            _, S, NS, RAWL, TRC = cf[:5]
            DX = cf[5] if len(cf) > 5 else 0
            wa = f"WA_{_r4_kname(cf)}"
            s += (f'struct {wa} {{ static __device__ __forceinline__ const unsigned char* get(const P& p) {{ u64 v; u32 z;\n'
                  f'  asm("mov.u64 %0, {waddr["c"]};" : "=l"(v)); asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(z));\n'
                  f'  return (const unsigned char*)(v + (z >> 31)); }} }};\n')
            s += (f'extern "C" __global__ void __launch_bounds__(320, 1) {_r4_kname(cf)}(const __grid_constant__ P p) '
                  f'{{ r4c_body<{S}, {NS}, {wa}, {TRC}, {RAWL}, {_r4_YB * 0x01010101}u, {DX}>(p); }}\n')
            continue
        R, NWG, S, NS, CL, ZC, WPB, TRC, ZP, EPI, PH, PERM, IO = _r4_xcf(cf)
        wa = "WPARAM"
        if WPB:
            wa = f"WA_{_r4_kname(cf)}"
            s += (f'struct {wa} {{ static __device__ __forceinline__ const unsigned char* get(const P& p) {{ u64 v; u32 z;\n'
                  f'  asm("mov.u64 %0, {waddr[R]};" : "=l"(v)); asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(z));\n'
                  f'  return (const unsigned char*)(v + (z >> 31)); }} }};\n')
        s += (f'extern "C" __global__ void __launch_bounds__({_r4_kthreads(cf)}, 1) {_r4_kname(cf)}(const __grid_constant__ P p) '
              + (f'{{ r4p_body<{R}, {S}, {ZC}, {wa}, {TRC}, {ZP}>(p); }}\n' if PH else
                 f'{{ r4_body<{R}, {NWG}, {S}, {NS}, {CL}, {ZC}, {wa}, {TRC}, {ZP}, {EPI}, {PERM}, {IO}>(p); }}\n'))
    return s


class _r4_Mod:
    """NVRTC module + cuLaunchKernelEx (cluster dims) on torch's current stream (copied from agent Y's yr.Mod)."""

    def __init__(self, src, name="r4.cu"):
        from cuda.bindings import driver, nvrtc
        self.d = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        if os.environ.get("R4_LINEINFO"):
            opts.append(b"-lineinfo")
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        self.log = log.decode(errors="replace").strip("\x00 \n")
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + self.log)
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        self.cubin = cubin
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
            (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1)
            assert int(err) == 0, err
            self.funcs[name] = f
        return f

    def attrs(self, name, smem):
        d = self.d
        f = self.func(name, smem)
        out = {}
        for k in ("NUM_REGS", "LOCAL_SIZE_BYTES"):
            err, v = d.cuFuncGetAttribute(getattr(d.CUfunction_attribute, "CU_FUNC_ATTRIBUTE_" + k), f)
            out[k] = v
        return out

    def _cfg(self, grid, block, smem, cluster, stream=None):
        d = self.d
        cfg = d.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block, 1, 1
        cfg.sharedMemBytes = smem
        if stream is not None:
            cfg.hStream = stream
        attrs = []
        if cluster > 1:
            at = d.CUlaunchAttribute()
            at.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = cluster, 1, 1
            attrs.append(at)
        cfg.attrs = attrs
        cfg.numAttrs = len(attrs)
        return cfg

    def occupancy(self, name, block, smem):
        f = self.func(name, smem)
        err, nb = self.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(f, block, smem)
        return int(nb) if int(err) == 0 else 0

    def max_clusters(self, name, grid, block, smem, cluster):
        f = self.func(name, smem)
        err, n = self.d.cuOccupancyMaxActiveClusters(f, self._cfg(grid, block, smem, cluster))
        return int(n) if int(err) == 0 else -int(err)

    def launch(self, name, grid, block, smem, cluster, params):
        d = self.d
        f = self.func(name, smem)
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        cfg = self._cfg(grid, block, smem, cluster, d.CUstream(torch.cuda.current_stream().cuda_stream))
        (err,) = d.cuLaunchKernelEx(cfg, f, ctypes.addressof(argv), 0)
        assert int(err) == 0, (name, err)


def _r4_tmap(ptr, rows, cols, box_rows, box_cols=64):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(2), ptr,
        [d.cuuint64_t(cols), d.cuuint64_t(rows)], [d.cuuint64_t(cols * 2)], [d.cuuint32_t(box_cols), d.cuuint32_t(box_rows)],
        [d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


def _r4_pack_w(hi, lo, R):
    """[256/R][128 k-blocks][hi R rows ; lo R rows][64 k] bf16, 128B swizzle applied per 128-B row (16-B chunk c of row r
    stored at chunk c ^ (r & 7)): one contiguous stage image per (row tile, k-block)."""
    nrt = 256 // R
    h = hi.view(nrt, R, 128, 64).permute(0, 2, 1, 3)
    l = lo.view(nrt, R, 128, 64).permute(0, 2, 1, 3)
    img = torch.cat([h, l], dim=2).contiguous().view(nrt, 128, 2 * R, 8, 8)
    r = torch.arange(2 * R, device=hi.device).view(2 * R, 1)
    c = torch.arange(8, device=hi.device).view(1, 8)
    idx = (c ^ (r & 7)).view(1, 1, 2 * R, 8, 1).expand(nrt, 128, 2 * R, 8, 8)
    return torch.gather(img, 3, idx).contiguous()


_r4_DEF_CFGS = [(64, 2, 32, 4, 1, 0, 0, 0), (64, 2, 16, 6, 1, 0, 0, 0)]


def _r4_grid_of(cf, m):
    if cf[0] == "c":
        return 4 * ((m + 63) // 64) * cf[1] + 4
    R, NWG, S, NS, CL, ZC = cf[:6]
    return (256 // R) * ((m + 64 * NWG - 1) // (64 * NWG)) * S + ZC


_r4_PROBE_SRC = r"""
__device__ __forceinline__ void r4_probe_body(int* out) {
  unsigned sid; asm volatile("mov.u32 %0, %%smid;" : "=r"(sid));
  if (threadIdx.x == 0) out[blockIdx.x] = (int)sid;
  const unsigned long long t0 = gtime();
  while (gtime() - t0 < 30000ull) { }
}
extern "C" __global__ void __launch_bounds__(128, 1) r4_probe128(int* out) { r4_probe_body(out); }
extern "C" __global__ void __launch_bounds__(192, 1) r4_probe192(int* out) { r4_probe_body(out); }
extern "C" __global__ void __launch_bounds__(320, 1) r4_probe320(int* out) { r4_probe_body(out); }
extern "C" __global__ void __launch_bounds__(576, 1) r4_probe576(int* out) { r4_probe_body(out); }
"""


def _r4__probe(st, block, grid, smem, cluster):
    out = torch.full((grid,), -1, dtype=torch.int32, device="cuda")
    try:
        st["mod"].launch(f"r4_probe{block}", grid, block, smem, cluster, struct.pack("<Q", out.data_ptr()))
        torch.cuda.synchronize()
    except AssertionError:
        return None
    o = out.tolist()
    return None if min(o) < 0 else o


def _r4_gpc_groups(st):
    """SM -> locality group (GPC), from cluster-16 (else cluster-8) probe launches (agent Q2's method)."""
    if st.get("grp") is not None:
        return st["grp"]
    nsm = st["nsm"]
    grp = {}
    for cs in (16, 8):
        n = (nsm // cs) * cs
        o = _r4__probe(st, 128, n, 120 * 1024, cs)
        if o is None:
            continue
        for i in range(0, n, cs):
            key = min(o[i:i + cs])
            for sm in o[i:i + cs]:
                grp.setdefault(sm, ("c%d" % cs, key))
    for sm in range(nsm):
        grp.setdefault(sm, ("s", sm))
    st["grp"] = grp
    return grp


def _r4_make_perm(st, cf, m):
    """Launch-order unit -> logical unit so that the units each GPC runs get consecutive logical units (rt fastest:
    the units of one GPC share x / W tiles). None if the placement could not be measured."""
    R, NWG, S, NS, CL, ZC, WPB, TRC, ZP = _r4_xcf(cf)[:9]
    grid = _r4_grid_of(cf, m)
    cl = 2 if CL >= 2 else 1
    o = _r4__probe(st, _r4_kthreads(cf), grid, _r4_ksmem(cf), cl)
    if o is None:
        return None
    grp = _r4_gpc_groups(st)
    ncomp = grid - ZC
    first = ZC if (ZC and not ZP) else 0
    units = ncomp // cl
    if units > 136:
        return None
    gfirst = {}
    for u in range(units):
        gfirst.setdefault(grp[o[first + u * cl]], u)
    order = sorted(range(units), key=lambda u: (gfirst[grp[o[first + u * cl]]], u))
    perm = [0] * units
    for i, u in enumerate(order):
        perm[u] = i
    return perm


def _r4_pack_c(hi, lo, maxe=27):
    """24-bit pair code images [4 row tiles][128 k-blocks][12288 B codes + 256 B escape trailer] followed by the raw
    BF16 stage images [4][128][16 KB] (pack_w, R = 64; used for the last k-block with RAWL). None if a stage has more
    than maxe escapes. Escapes (hi exponent outside [108, 123], lo == 0, lo exponent delta outside [0, 7]) carry the exact
    hi / lo bits in the trailer and are patched after decoding, so the decoded tiles equal the raw BF16 pair."""
    dev = hi.device
    h16 = hi.contiguous().view(torch.int16).int() & 0xFFFF
    l16 = lo.contiguous().view(torch.int16).int() & 0xFFFF
    he7 = (h16 >> 8) & 0x7F
    le7 = (l16 >> 8) & 0x7F
    dl = he7 - 4 - le7
    hbad = (he7 < _r4_YB) | (he7 > _r4_YB + 7)
    esc = hbad | ((l16 & 0x7FFF) == 0) | (dl < 0) | (dl > 7)
    z = torch.zeros_like(h16)
    hn = torch.where(esc, z, ((h16 >> 15) << 3) | (he7 - _r4_YB).clamp(0, 7))
    hl = torch.where(esc, z, h16 & 0xFF)
    lnb = torch.where(esc, z, ((l16 >> 15) << 3) | dl.clamp(0, 7))
    ll = torch.where(esc, z, l16 & 0xFF)
    # emulated decode == raw for every non-escape
    dh = ((hn >> 3) << 15) | (((hn & 7) + _r4_YB) << 8) | hl
    dlo = ((lnb >> 3) << 15) | (((hn & 7) + _r4_YB - 4 - (lnb & 7)) << 8) | ll
    assert bool(((dh == h16) | esc).all()) and bool(((dlo == l16) | esc).all())

    def thr(a):   # [256, 8192] -> [4 rt][128 kb][256 t][2 p][8 e] with t = 4 row + (kc >> 1), p = kc & 1
        return a.view(4, 64, 128, 8, 8).permute(0, 2, 1, 3, 4).reshape(4, 128, 256, 2, 8)
    hl_t, ll_t = thr(hl).to(torch.uint8), thr(ll).to(torch.uint8)
    hn_t, ln_t = thr(hn), thr(lnb)
    hnw = (hn_t[..., 0:4] | (hn_t[..., 4:8] << 4)).to(torch.uint8)          # [4][128][256][2 p][4 B]
    lnw = (ln_t[..., 0:4] | (ln_t[..., 4:8] << 4)).to(torch.uint8)
    nib = torch.cat([hnw, lnw], dim=3)                                      # [4][128][256][4][4 B]: hn p0, hn p1, ln p0, ln p1
    seg = lambda a: a.reshape(4, 128, 256, -1, 16).permute(0, 1, 3, 2, 4)   # [4][128][nseg][256 t][16 B]
    codes = torch.cat([seg(hl_t), seg(ll_t), seg(nib)], dim=2).reshape(4, 128, 12288)
    trailer = torch.zeros((4, 128, 256), dtype=torch.uint8)
    nz = esc.nonzero().cpu().tolist()
    groups = {}
    hc, lc = h16.cpu(), l16.cpu()
    for row, k in nz:
        groups.setdefault((row // 64, k // 64), []).append(((row % 64) * 64 + (k % 64), int(hc[row, k]), int(lc[row, k])))
    if any(len(v) > maxe for v in groups.values()):
        return None
    for (rt, kb), ents in groups.items():
        flags = [0] * 8
        for (pos, hb, lb) in ents:
            t = ((pos >> 6) << 2) | ((pos & 63) >> 4)
            flags[t >> 5] |= 1 << (t & 31)
        words = [f - (1 << 32) if f >= (1 << 31) else f for f in flags] + [len(ents), 0]
        words += [w for (pos, hb, lb) in ents for w in ((hb << 16) | pos, lb)]
        trailer[rt, kb, :4 * len(words)] = torch.tensor(words, dtype=torch.int64).to(torch.int32).view(torch.uint8)
    img = torch.cat([codes.cpu(), trailer], dim=2).to(dev).contiguous().view(-1)
    raw = _r4_pack_w(hi, lo, 64).contiguous().view(torch.uint8).view(-1)
    return torch.cat([img, raw]).contiguous(), len(nz), max((len(v) for v in groups.values()), default=0)


_r4_DEF_CFGS = [(64, 1, 16, 8, 1, 4, 1, 0, 0, 1)]


def _r4_prepare_r4(weights, cfgs=_r4_DEF_CFGS, perm_ms=None):
    """perm_ms: {cfg: m} for configs with PERM (the permutation is measured for that m's grid)."""
    w = weights["router"].contiguous()
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)
    st = {"cfgs": list(cfgs), "wp": {}, "xm": {}, "perm": {}}
    for R in sorted({c[0] for c in cfgs}, key=str):
        if R == "c":
            pk = _r4_pack_c(hi, lo)
            if pk is None:
                raise RuntimeError("r4c: a stage has too many escapes")
            st["wp"]["c"], st["nesc"], st["maxesc"] = pk
        else:
            st["wp"][R] = _r4_pack_w(hi, lo, R)
    st["mod"] = _r4_Mod(_r4_HEAD + _r4_PROBE_SRC + _r4_source(cfgs, {R: t.data_ptr() for R, t in st["wp"].items()})[len(_r4_HEAD):], "r4.cu")
    st["zc"] = {z: torch.zeros(1024, dtype=torch.int32, device=w.device) for z in (0, 4)}
    st["trc"] = torch.zeros(1024 * 16, dtype=torch.int64, device=w.device)
    st["nsm"] = torch.cuda.get_device_properties(w.device).multi_processor_count
    st["attrs"] = {}
    for cf in cfgs:
        name, smem, nt = _r4_kname(cf), _r4_ksmem(cf), _r4_kthreads(cf)
        occ = st["mod"].occupancy(name, nt, smem)
        st["attrs"][cf] = (st["mod"].attrs(name, smem), occ)
        if cf[0] != "c" and _r4_xcf(cf)[11]:
            for m in (perm_ms or {}).get(cf, [128, 256]):
                st["perm"][(cf, _r4_grid_of(cf, m))] = _r4_make_perm(st, cf, m)
    torch.cuda.synchronize()
    return st


def _r4_run_r4(x, st, cf, out=None):
    if cf[0] == "c":
        _, S, NS, RAWL, TRC = cf[:5]
        m = x.shape[0]
        ntt = (m + 63) // 64
        grid = 4 * ntt * S + 4
        assert grid <= st["nsm"], grid
        if out is None:
            out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
        key = (x.data_ptr(), m, 64, 0)
        mx = st["xm"].get(key)
        if mx is None:
            if len(st["xm"]) > 64:
                st["xm"].clear()
            mx = st["xm"][key] = _r4_tmap(x.data_ptr(), m, 8192, 64)
        zc = st["zc"][4][grid:]
        params = mx + struct.pack("<QQQQiiii", st["wp"]["c"].data_ptr(), out.data_ptr(), zc.data_ptr(), st["trc"].data_ptr(),
                                  m, ntt, 0, 0) + b"\0" * 136
        st["mod"].launch(_r4_kname(cf), grid, 320, _r4_ksmem(cf), 1, params)
        return out
    R, NWG, S, NS, CL, ZC, WPB, TRC, ZP, EPI, PH, PERM, IO = _r4_xcf(cf)
    m = x.shape[0]
    tok = 64 * NWG
    ntt = (m + tok - 1) // tok
    grid = _r4_grid_of(cf, m)
    assert grid <= st["nsm"], grid
    perm = b"\0" * 136
    if PERM:
        pl = st["perm"].get((cf, grid))
        if pl is None:
            pl = list(range((grid - ZC) // (2 if CL >= 2 else 1)))
        perm = bytes(pl) + b"\0" * (136 - len(pl))
    if out is None:
        out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
    key = (x.data_ptr(), m, tok, PH)
    mx = st["xm"].get(key)
    if mx is None:
        if len(st["xm"]) > 64:
            st["xm"].clear()
        mx = st["xm"][key] = _r4_tmap(x.data_ptr(), m, 8192, 64 if PH else tok)
    zc = st["zc"][ZC][grid:]
    params = mx + struct.pack("<QQQQiiii", st["wp"][R].data_ptr(), out.data_ptr(), zc.data_ptr(), st["trc"].data_ptr(),
                              m, ntt, 0, 0) + perm
    st["mod"].launch(_r4_kname(cf), grid, _r4_kthreads(cf), _r4_ksmem(cf), 2 if CL >= 2 else 1, params)
    return out



# ------------------------------------------------------------------ deliverable API
_r4_RT4_ROUTES = [((97, 128), (64, 1, 16, 8, 1, 4, 1, 0, 0, 1, 0, 0, 3)),
              ((193, 256), (64, 1, 8, 8, 1, 4, 1, 0, 0, 1, 0, 0, 3))]


def _r4__selftest(st, w):
    """Dense + one-hot rows at the route boundaries vs an FP64 reference; NaN-prefilled outputs."""
    g = torch.Generator(device=w.device)
    g.manual_seed(1234)
    hi = w.to(torch.bfloat16)
    pair = hi.double() + (w - hi.float()).to(torch.bfloat16).double()
    for (lo_m, hi_m), cf in _r4_RT4_ROUTES:
        for m in (lo_m, hi_m):
            x = torch.randn((m, 8192), generator=g, device=w.device).to(torch.bfloat16)
            x[0].zero_()
            x[0, 4097] = 1.0
            out = torch.full((m, 256), float("nan"), device=w.device)
            _r4_run_r4(x, st, cf, out=out)
            torch.cuda.synchronize()
            ref = x.double() @ pair.t()
            err = float((out.double() - ref).norm() / ref.norm())
            if not (err < 1e-5) or not torch.equal(out[0], pair[:, 4097].float()):
                return False
    return True


def _r4_prepare_rt4(weights):
    w = weights.get("router")
    if w is None or tuple(w.shape) != (256, 8192) or w.dtype != torch.float32 or not w.is_cuda:
        return {"ok": False}
    if torch.cuda.get_device_capability(w.device) != (9, 0):
        return {"ok": False}
    try:
        st = _r4_prepare_r4(weights, sorted({cf for _, cf in _r4_RT4_ROUTES}))
        if st["nsm"] < 132:
            return {"ok": False}
        st["ok"] = _r4__selftest(st, w.contiguous())
    except Exception:  # noqa: BLE001 -- any failure disables the route (the base handles the router)
        return {"ok": False}
    torch.cuda.synchronize()
    return st


def _r4_run_rt4(x, config, state):
    if not state.get("ok") or config.get("layer") != "router" or config.get("n") != 256:
        return None
    if x.dim() != 2 or x.shape[1] != 8192 or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    m = x.shape[0]
    for (lo_m, hi_m), cf in _r4_RT4_ROUTES:
        if lo_m <= m <= hi_m:
            return _r4_run_r4(x, state, cf)
    return None


# ======================================================================== module Y: Y/rt3.py
# Design based on Nathen Fernandes' small-M router (NathenFernandes/muse-bf16-proj, MIT, via alygg77); code rewritten, no lines copied.
"""rt3 (agent Y, claude/t3/Y/rt3.py): router out[M,256] fp32 = x[M,8192] bf16 @ (hi + lo)[256,8192]^T, FP32 accumulation.

Generated by claude/t3/Y/mk_rt3.py from the lab modules yr.py / yk.py. Routes (None = let the base handle it):
- 2 <= m <= 8: `ys` -- derived from Nathen Fernandes' small router (muse-bf16-proj, MIT, via alygg77): every lane LDGs
  its whole hi/lo A-fragment share + x B fragments up front, mma.sync m16n8k16, split-K over a cluster of 8 CTAs
  reduced through DSMEM; here 32 rows x 1024 k per CTA (64 CTAs), W pointer compiled in, x issued first.
- 9..32: `ya` -- the same ys main loop (16 rows x 1024 k per CTA, 128 CTAs) without clusters: split-K partials are
  red.add'ed into out, which 4 extra CTAs (no loads on their SMs, so the release fence is fast) clear in-kernel;
  compute CTAs wait on a release counter (generation from a relaxed atomic) before their adds; the zeroing CTAs are
  blockIdx 0-3 (dispatched first), so no co-residency is required.
- 97..128: `rk` -- NVRTC port of alygg77's rk router GEMM (Apache-2.0, built on vjeux/okc-2026,
  Apache-2.0): TMA ring + SS-wgmma m64n64k16 on [hi | lo], split-K with in-kernel zeroing + red.add; K split 16 at
  m=128 (grid 128; bo1 uses 12).
"""
import ctypes
import struct

import torch

_yr_HEAD = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
struct __align__(16) P { const uint4* x; float* out; long long m; u64* trc; };
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ u64 gtime() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void mbar_init(u32 b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_expect(u32 b, u32 n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_wait_cl(u32 b, int ph) {
  asm volatile("{ .reg .pred p; W: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1; @!p bra W; }" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ u32 mapa(u32 a, int r) { u32 b; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(b) : "r"(a), "r"(r)); return b; }
__device__ __forceinline__ void st_async(u32 d, float4 v, u32 b) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];"
               :: "r"(d), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(b) : "memory");
}
__device__ __forceinline__ uint4 ldg_nc(const uint4* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint4 ldg_x(const uint4* p) {
  uint4 v; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ void mma16816(float* c, uint4 a, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ u32 prmt2(u32 a, u32 b, u32 sel) { u32 r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel)); return r; }
// 24-bit pair format: per 8-value A fragment: hi low bytes (uint2), hi nibbles (s | e7-base), lo low bytes, lo nibbles
// (s | delta, lo e7 = hi e7 - 4 - delta). Byte b of a nibble word holds values b (low nibble) and b+4 (high nibble).
template <u32 B4>
__device__ __forceinline__ void dec_pair(u32 hl0, u32 hl1, u32 hn, u32 ll0, u32 ll1, u32 ln, uint4& ah, uint4& al) {
  const u32 e0 = (hn & 0x07070707u) + B4, e1 = ((hn >> 4) & 0x07070707u) + B4;
  const u32 h0 = e0 | ((hn << 4) & 0x80808080u), h1 = e1 | (hn & 0x80808080u);
  ah.x = prmt2(hl0, h0, 0x5140); ah.y = prmt2(hl0, h0, 0x7362); ah.z = prmt2(hl1, h1, 0x5140); ah.w = prmt2(hl1, h1, 0x7362);
  const u32 g0 = (e0 - 0x04040404u - (ln & 0x07070707u)) | ((ln << 4) & 0x80808080u);
  const u32 g1 = (e1 - 0x04040404u - ((ln >> 4) & 0x07070707u)) | (ln & 0x80808080u);
  al.x = prmt2(ll0, g0, 0x5140); al.y = prmt2(ll0, g0, 0x7362); al.z = prmt2(ll1, g1, 0x5140); al.w = prmt2(ll1, g1, 0x7362);
}
__device__ __forceinline__ u32 comp4(uint4 v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }
__device__ __forceinline__ void cp_async16(u32 dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ uint4 lds128(u32 a) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v;
}
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }
// AS mode (RT=1): W slot j of a k-unit belongs to commit group 0 (k16 steps 0-1) or 1 (steps 2-3)
template <int CMP> __device__ __forceinline__ constexpr int as_grp(int j) { return CMP ? ((j == 1 || j == 3) ? 1 : 0) : (j >= 4 ? 1 : 0); }
__device__ __forceinline__ void f4add(float4& a, float4 b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
#define TR(i) do { if (TRC && threadIdx.x == 0) { if ((i) == 0) { p.trc[blockIdx.x * 16] = gtime(); c0_ = clock64(); } else p.trc[blockIdx.x * 16 + (i)] = clock64() - c0_; } } while (0)
#define TRE(i) do { if (TRC && lane == 0) p.trc[blockIdx.x * 16 + (i)] = clock64() - c0e_; } while (0)

// RT row tiles of 16 per CTA, S = cluster size = k splits, WK warps, MT token tiles of 8, XF: x loads first
// one escape weight (compile-time position, value delta = d1 + d2): acc[row][tok] += x[tok][k] * delta, x via shuffles
template <int R, int G, int H, int KUI, int KT, int KJ, int MT, class C, class X>
__device__ __forceinline__ void esc1(C& c, const X& xx, int lane, float d1, float d2) {
#pragma unroll
  for (int n = 0; n < MT; ++n) {
    const uint4 v4 = xx[KUI][n][KJ >> 3];
    constexpr int WI = (KJ >> 1) & 3;
    const u32 wv = WI == 0 ? v4.x : WI == 1 ? v4.y : WI == 2 ? v4.z : v4.w;
    const float xs = __uint_as_float((KJ & 1) ? (wv & 0xFFFF0000u) : (wv << 16));
    const float x0 = __shfl_sync(0xffffffffu, xs, ((lane & 3) * 2) * 4 + KT);
    const float x1 = __shfl_sync(0xffffffffu, xs, ((lane & 3) * 2 + 1) * 4 + KT);
    if ((lane >> 2) == G) { c[R][0][n][H * 2] += x0 * d1 + x0 * d2; c[R][0][n][H * 2 + 1] += x1 * d1 + x1 * d2; }
  }
}
struct NoEsc { static constexpr bool gen = false; template <class C, class X> __device__ static void apply(C&, const X&, int, int) {}
  static constexpr bool on = false; __device__ static int4 ent(int) { return make_int4(0, 0, 0, 0); } __device__ static int off(int) { return 0; } };
template <int RT, int S, int WK, int MT, int XF, int TRC, int CMP, u32 B4, int EMAX = 0, class EC = NoEsc>
__device__ __forceinline__ void ys_body(const P& p, const uint4* __restrict__ wp, const int4* __restrict__ ep = nullptr) {
  constexpr int KU = 128 / (S * WK);
  constexpr int NJ = CMP ? 6 * RT : 8 * RT;
  constexpr int NP = 4 * RT;
  constexpr int NI = RT * MT * 32;
  constexpr int OWN = NI / S;
  static_assert(NI % S == 0, "owner split");
  extern __shared__ __align__(16) unsigned char sm[];
  float4* part = (float4*)sm;
  float4* recv = part + WK * NI;
  const u32 rb = smem_u32(recv + S * OWN);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int rank = blockIdx.x % S, rtile = blockIdx.x / S;
  const int m = (int)p.m;
  long long c0_ = 0;
  TR(0);
  if (TRC && tid == 0) { u32 sid; asm volatile("mov.u32 %0, %%smid;" : "=r"(sid)); p.trc[blockIdx.x * 16 + 15] = sid; }
  long long c0e_ = clock64();
  uint4 w[KU][NJ];
  uint4 xx[KU][MT][2];
  auto ldx = [&]() {
#pragma unroll
    for (int k = 0; k < KU; ++k)
#pragma unroll
      for (int n = 0; n < MT; ++n) {
        const int tok = min(n * 8 + (lane >> 2), m - 1);
        const uint4* q = p.x + (size_t)tok * 1024 + ((rank * WK + warp) * KU + k) * 8 + (lane & 3) * 2;
        xx[k][n][0] = ldg_x(q); xx[k][n][1] = ldg_x(q + 1);
      }
  };
  auto ldw = [&]() {
#pragma unroll
    for (int k = 0; k < KU; ++k)
#pragma unroll
      for (int j = 0; j < NJ; ++j)
        w[k][j] = ldg_nc(wp + ((size_t)(rtile * 128 + (rank * WK + warp) * KU + k) * NJ + j) * 32 + lane);
  };
  int4 ent = make_int4(0, 0, 0, 0);
  if (EMAX > 0 && lane < EMAX) ent = __ldg(ep + ((size_t)blockIdx.x * WK + warp) * EMAX + lane);   // this warp's escapes
  if (XF & 1) { ldx(); ldw(); } else { ldw(); ldx(); }
  if (TRC && warp < WK) {   // trace: time when this thread's loads have landed
    u32 acc = 0;
#pragma unroll
    for (int k = 0; k < KU; ++k) {
#pragma unroll
      for (int j = 0; j < NJ; ++j) acc ^= w[k][j].x;
#pragma unroll
      for (int n = 0; n < MT; ++n) acc ^= xx[k][n][0].x ^ xx[k][n][1].y;
    }
    if (acc == 0x12345679u) p.trc[0] = 0;
    TR(5);
  }
  if (tid == 0) {
    mbar_init(rb, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid == 0) mbar_expect(rb, (S - 1) * OWN * 16);
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  {
  float c[RT][2][MT][4];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int n = 0; n < MT; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) c[r][h][n][e] = 0.f;
  if (EC::gen) EC::apply(c, xx, warp, lane);   // generated per-CTA escape code (compile-time positions)
  if (EMAX > 0 || EC::on) {   // escapes first (needs x only; overlaps the W stream): acc += x[tok][k] * (d1 + d2)
    int e0 = 0, e1 = EMAX;
    if (EC::on) { e0 = EC::off(blockIdx.x * WK + warp); e1 = EC::off(blockIdx.x * WK + warp + 1); }
#pragma unroll 1
    for (int e = e0; e < e1; ++e) {
      int ex, ez, ew;
      if (EC::on) { const int4 q = EC::ent(e); ex = q.x; ez = q.z; ew = q.w; }
      else {
        ex = __shfl_sync(0xffffffffu, ent.x, e);
        ez = __shfl_sync(0xffffffffu, ent.z, e); ew = __shfl_sync(0xffffffffu, ent.w, e);
      }
      if ((ez | ew) == 0) break;
      const int rr = ex & 0xFF, kq = ex >> 8;
      const int ku = kq >> 6, kt = (kq >> 4) & 3, kj = kq & 15;
      const float d1 = __int_as_float(ez), d2 = __int_as_float(ew);
      const int r = rr >> 4, g = rr & 7, half = (rr >> 3) & 1;
#pragma unroll
      for (int n = 0; n < MT; ++n) {
        uint4 lo4 = xx[0][n][0], hi4 = xx[0][n][1];
#pragma unroll
        for (int k = 1; k < KU; ++k) if (ku == k) { lo4 = xx[k][n][0]; hi4 = xx[k][n][1]; }
        const uint4 v4 = (kj & 8) ? hi4 : lo4;
        const int wi = (kj >> 1) & 3;
        const u32 wv = wi == 0 ? v4.x : wi == 1 ? v4.y : wi == 2 ? v4.z : v4.w;
        const float xs = __uint_as_float((kj & 1) ? (wv & 0xFFFF0000u) : (wv << 16));
        const float x0 = __shfl_sync(0xffffffffu, xs, ((lane & 3) * 2) * 4 + kt);
        const float x1 = __shfl_sync(0xffffffffu, xs, ((lane & 3) * 2 + 1) * 4 + kt);
        if ((lane >> 2) == g) {
          const float v0 = x0 * d1 + x0 * d2, v1 = x1 * d1 + x1 * d2;
#pragma unroll
          for (int r2 = 0; r2 < RT; ++r2)
            if (r2 == r) {
              if (half) { c[r2][0][n][2] += v0; c[r2][0][n][3] += v1; }
              else { c[r2][0][n][0] += v0; c[r2][0][n][1] += v1; }
            }
        }
      }
    }
  }
#pragma unroll
  for (int k = 0; k < KU; ++k) {
    if (CMP) {
#pragma unroll
      for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          const int pp = q * RT + r;
          const uint4 HL = w[k][pp >> 1], LL = w[k][NP / 2 + (pp >> 1)];
          const u32 hn = comp4(w[k][NP + (pp >> 2)], pp & 3), lnn = comp4(w[k][NP + NP / 4 + (pp >> 2)], pp & 3);
          uint4 ah, al;
          if (pp & 1) dec_pair<B4>(HL.z, HL.w, hn, LL.z, LL.w, lnn, ah, al);
          else dec_pair<B4>(HL.x, HL.y, hn, LL.x, LL.y, lnn, ah, al);
#pragma unroll
          for (int n = 0; n < MT; ++n) {
            const u32 b0 = (q < 2) ? (q == 0 ? xx[k][n][0].x : xx[k][n][0].z) : (q == 2 ? xx[k][n][1].x : xx[k][n][1].z);
            const u32 b1 = (q < 2) ? (q == 0 ? xx[k][n][0].y : xx[k][n][0].w) : (q == 2 ? xx[k][n][1].y : xx[k][n][1].w);
            mma16816(c[r][0][n], ah, b0, b1);
            mma16816(c[r][1][n], al, b0, b1);
          }
        }
    } else if (XF & 2) {   // W-arrival order: each W fragment's MMAs as soon as it lands
#pragma unroll
      for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int r = 0; r < RT; ++r)
#pragma unroll
            for (int n = 0; n < MT; ++n) {
              const u32 b0 = (q < 2) ? (q == 0 ? xx[k][n][0].x : xx[k][n][0].z) : (q == 2 ? xx[k][n][1].x : xx[k][n][1].z);
              const u32 b1 = (q < 2) ? (q == 0 ? xx[k][n][0].y : xx[k][n][0].w) : (q == 2 ? xx[k][n][1].y : xx[k][n][1].w);
              mma16816(c[r][h][n], w[k][(q * 2 + h) * RT + r], b0, b1);
            }
    } else {
#pragma unroll
      for (int n = 0; n < MT; ++n) {
        const u32 b[8] = {xx[k][n][0].x, xx[k][n][0].y, xx[k][n][0].z, xx[k][n][0].w,
                          xx[k][n][1].x, xx[k][n][1].y, xx[k][n][1].z, xx[k][n][1].w};
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int r = 0; r < RT; ++r) mma16816(c[r][h][n], w[k][(q * 2 + h) * RT + r], b[2 * q], b[2 * q + 1]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int n = 0; n < MT; ++n)
      part[warp * NI + (r * MT + n) * 32 + lane] =
          make_float4(c[r][0][n][0] + c[r][1][n][0], c[r][0][n][1] + c[r][1][n][1],
                      c[r][0][n][2] + c[r][1][n][2], c[r][0][n][3] + c[r][1][n][3]);
  }
  TR(1);
  __syncthreads();
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
  TR(2);
  for (int v = tid; v < NI && warp < WK; v += WK * 32) {
    float4 s = part[v];
#pragma unroll
    for (int q = 1; q < WK; ++q) f4add(s, part[q * NI + v]);
    const int dst = v / OWN, loc = v % OWN;
    if (dst == rank) recv[rank * OWN + loc] = s;
    else st_async(mapa(smem_u32(recv + rank * OWN + loc), dst), s, mapa(rb, dst));
  }
  __syncthreads();
  mbar_wait_cl(rb, 0);
  TR(3);
  for (int t = tid; t < OWN && warp < WK; t += WK * 32) {
    float4 s = recv[t];
#pragma unroll
    for (int q = 1; q < S; ++q) f4add(s, recv[q * OWN + t]);
    const int v = rank * OWN + t;
    const int r = v / (MT * 32), n = (v / 32) % MT, ln = v % 32;
    const int row = rtile * 16 * RT + r * 16 + (ln >> 2), tok = n * 8 + (ln & 3) * 2;
    if (tok < m) { p.out[tok * 256 + row] = s.x; p.out[tok * 256 + row + 8] = s.z; }
    if (tok + 1 < m) { p.out[(tok + 1) * 256 + row] = s.y; p.out[(tok + 1) * 256 + row + 8] = s.w; }
  }
  TR(4);
}
// ---------------------------------------------------------------------------- ya: ys without clusters
// Same LDG-all-up-front + mma.sync main loop (raw hi/lo), split-K over S independent CTAs (grid <= #SMs, all resident):
// out is cleared in-kernel by an extra helper warp (slice + release counter, rk-style grid counter, spin, mbarrier),
// partials are summed over the CTA's warps in smem and red.add.v4'd into out.
struct __align__(16) PA { const uint4* x; float* out; long long m; u64* trc; int* zcnt; };
// escapes in the kernel parameter block (constant bank, no dependent global load): per CTA [eo[c], eo[c+1]) entries
// (row_in_rtile | k << 5, fp32 delta); a weight whose delta needs > 24 bits gets two entries.
struct __align__(16) PAE { const uint4* x; float* out; long long m; u64* trc; int* zcnt; unsigned short eo[132]; uint2 ent[472]; };
__device__ __forceinline__ void mbar_wait_cta(u32 b, int ph) {
  asm volatile("{ .reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W; }" :: "r"(b), "r"(ph) : "memory");
}
#define TA(i) do { if (TRA && lane == 0 && cb >= 0) p.trc[cb * 16 + (i)] = gtime(); } while (0)
template <int RT, int S, int WK, int MT, int XF, int CMP = 0, u32 B4 = 0, int EMAX = 0, class PT = PA, int TRA = 0>
__device__ __forceinline__ void ya_body(const PT& p, const uint4* __restrict__ wp, const int4* __restrict__ ep = nullptr) {
  constexpr int KU = 128 / (S * WK);
  constexpr int NJ = CMP ? 6 * RT : 8 * RT;
  constexpr int NP = 4 * RT;
  constexpr int NI = RT * MT * 32;
  constexpr int RR = 16 * RT;                  // rows per CTA
  constexpr int TP = RR + 4;                   // staging [MT*8 tokens][RR rows] fp32
  extern __shared__ __align__(16) unsigned char sm[];
  float4* part = (float4*)sm;
  float* T = (float*)(part + WK * NI);
  const u32 zb = smem_u32(T + MT * 8 * TP);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  constexpr int ZC = (XF & 16) ? 4 : 0;   // dedicated zeroing CTAs: blockIdx 0..ZC-1 (dispatched first, no load flood)
  const int cb = (int)blockIdx.x - ZC;    // compute CTA index
  const int rank = cb % S, rtile = cb / S;
  const int m = (int)p.m;
  if (tid == 0) { mbar_init(zb, 1); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  if (warp == 0) TA(0);
  __syncthreads();
  const int ncomp = (int)gridDim.x - ZC;
  if (ZC && cb < 0) {   // zero CTA: clear 1/ZC of out, publish with a release add on counter A
    const int zi = blockIdx.x;
    const int n4 = m * 64, per = (n4 + ZC - 1) / ZC;
    const int z0 = zi * per, z1 = min(n4, z0 + per);
    for (int i = z0 + tid; i < z1; i += blockDim.x) reinterpret_cast<float4*>(p.out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncthreads();
    if (tid == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(p.zcnt) : "memory");
    return;
  }
  if (ZC && warp == WK) {   // helper: generation from counter B (relaxed atomic), wait for counter A >= ZC*(gen+1)
    float zev[EMAX < 0 ? MT * 8 : 1];
    uint2 zpe = make_uint2(0, 0);
    if constexpr (EMAX < 0) {   // param-block escapes (lane = entry): x[tok][k] loads issued before the wait
      const int o0 = p.eo[cb], ne = p.eo[cb + 1] - o0;
      if (lane < ne) zpe = p.ent[o0 + lane];
      const unsigned short* xs = (const unsigned short*)p.x + (int)(zpe.x >> 5);
#pragma unroll
      for (int t = 0; t < MT * 8; ++t) zev[t] = __uint_as_float((u32)xs[(size_t)min(t, m - 1) * 8192] << 16);
    }
    if (lane == 0) {
      int v;
      asm volatile("atom.relaxed.gpu.global.add.s32 %0, [%1], 1;" : "=r"(v) : "l"(p.zcnt + 256) : "memory");
      const int target = (v / ncomp + 1) * ZC;
      const u64 tz = gtime();
      for (;;) {
        int a;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(a) : "l"(p.zcnt) : "memory");
        if (a - target >= 0) break;
        if (gtime() - tz > 200000000ull) __trap();
        __nanosleep(32);
      }
    }
    __syncwarp();
    if constexpr (EMAX < 0) {
      if (zpe.y) {
        const float d = __uint_as_float(zpe.y);
        float* o = p.out + rtile * 16 * RT + (zpe.x & 31);
#pragma unroll
        for (int t = 0; t < MT * 8; ++t)
          if (t < m) asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;" :: "l"(o + (size_t)t * 256), "f"(zev[t] * d) : "memory");
      }
    }
    TA(9);
    if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(zb) : "memory");
    return;
  }
  if (warp == WK) {   // helper: clear this CTA's slice of out, count it, wait for every CTA's count, release the adds
    int4 ent = make_int4(0, 0, 0, 0);
    if (EMAX > 0 && lane < EMAX) ent = __ldg(ep + (size_t)cb * EMAX + lane);   // escapes of this CTA (24-bit format)
    int target = 0;
    if (lane == 0) {
      int v;
      asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p.zcnt) : "memory");
      target = (v / (int)gridDim.x + 1) * (int)gridDim.x;
    }
    const int n4 = m * 64;
    const int per = (n4 + (int)gridDim.x - 1) / (int)gridDim.x;
    const int z0 = blockIdx.x * per, z1 = min(n4, z0 + per);
    for (int i = z0 + lane; i < z1; i += 32) reinterpret_cast<float4*>(p.out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncwarp();
    if (XF & 4) asm volatile("bar.arrive 2, %0;" :: "r"(WK * 32 + 32) : "memory");   // zero stores issued: release the loads
    float ev[EMAX > 0 ? EMAX : EMAX < 0 ? MT * 8 : 1];
    uint2 pe = make_uint2(0, 0);
    if constexpr (EMAX < 0) {   // param-block escapes: lane = entry, loads of x[tok][k] for all tokens issued now
      const int o0 = p.eo[cb], ne = p.eo[cb + 1] - o0;
      if (lane < ne) pe = p.ent[o0 + lane];
      const int k = (int)(pe.x >> 5);
      const unsigned short* xs = (const unsigned short*)p.x + k;
#pragma unroll
      for (int t = 0; t < MT * 8; ++t) ev[t] = __uint_as_float((u32)xs[(size_t)min(t, m - 1) * 8192] << 16);
    }
    if (EMAX > 0) {   // x[tok = lane][k] * (d1 + d2) for every escape (lanes = tokens), loads issued before the wait
#pragma unroll
      for (int e = 0; e < EMAX; ++e) {
        const int k = __shfl_sync(0xffffffffu, ent.y, e);
        const float d1 = __int_as_float(__shfl_sync(0xffffffffu, ent.z, e)), d2 = __int_as_float(__shfl_sync(0xffffffffu, ent.w, e));
        const unsigned short xb = ((const unsigned short*)p.x)[(size_t)min(lane, m - 1) * 8192 + k];
        const float xv = __uint_as_float((u32)xb << 16);
        ev[e] = xv * d1 + xv * d2;
      }
    }
    if (lane == 0) {
      asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(p.zcnt) : "memory");
      TA(8);
      const u64 tz = gtime();
      for (;;) {
        int v;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p.zcnt) : "memory");
        if (v - target >= 0) break;
        if (gtime() - tz > 200000000ull) __trap();   // not all CTAs resident: fail loudly, never hang
        __nanosleep(32);
      }
    }
    __syncwarp();
    if (EMAX > 0) {   // out is clear (acquired by lane 0, ordered for the warp by __syncwarp): add the corrections
#pragma unroll
      for (int e = 0; e < EMAX; ++e) {
        const int rowl = __shfl_sync(0xffffffffu, ent.x, e);
        const int valid = __shfl_sync(0xffffffffu, ent.z | ent.w, e);
        if (valid && lane < m)
          asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;" :: "l"(p.out + (size_t)lane * 256 + rtile * 16 * RT + rowl), "f"(ev[e]) : "memory");
      }
    }
    if constexpr (EMAX < 0) {
      if (pe.y) {
        const float d = __uint_as_float(pe.y);
        float* o = p.out + rtile * 16 * RT + (pe.x & 31);
#pragma unroll
        for (int t = 0; t < MT * 8; ++t)
          if (t < m) asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;" :: "l"(o + (size_t)t * 256), "f"(ev[t] * d) : "memory");
      }
    }
    TA(9);
    if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(zb) : "memory");
    return;
  }
  uint4 w[KU][NJ];
  uint4 xx[KU][MT][2];
  auto ldx = [&]() {
#pragma unroll
    for (int k = 0; k < KU; ++k)
#pragma unroll
      for (int n = 0; n < MT; ++n) {
        const int tok = min(n * 8 + (lane >> 2), m - 1);
        const uint4* q = p.x + (size_t)tok * 1024 + ((rank * WK + warp) * KU + k) * 8 + (lane & 3) * 2;
        xx[k][n][0] = ldg_x(q); xx[k][n][1] = ldg_x(q + 1);
      }
  };
  constexpr bool AS = (XF & 8) != 0;   // W via cp.async into smem, one commit group per (k-unit, k16 half)
  static_assert(!AS || RT == 1, "AS mode: RT == 1");
  const u32 wsm = (smem_u32(T + MT * 8 * TP) + 16 + 15) & ~15u;   // [WK][KU][NJ][32] uint4
  auto ldw = [&]() {
    if constexpr (AS) {
#pragma unroll
      for (int k = 0; k < KU; ++k)
#pragma unroll
        for (int g = 0; g < 2; ++g) {
#pragma unroll
          for (int j = 0; j < NJ; ++j)
            if (as_grp<CMP>(j) == g)
              cp_async16(wsm + (((warp * KU + k) * NJ + j) * 32 + lane) * 16,
                         wp + ((size_t)(rtile * 128 + (rank * WK + warp) * KU + k) * NJ + j) * 32 + lane);
          asm volatile("cp.async.commit_group;" ::: "memory");
        }
    } else if (XF & 64) {   // two batches: k-unit 1 is issued when k-unit 0's last load has landed (double buffering)
#pragma unroll
      for (int j = 0; j < NJ; ++j)
        w[0][j] = ldg_nc(wp + ((size_t)(rtile * 128 + (rank * WK + warp) * KU) * NJ + j) * 32 + lane);
      u32 z;
      asm volatile("and.b32 %0, %1, 0;" : "=r"(z) : "r"(w[0][NJ - 1].w));
      const uint4* wp2 = wp + z;
#pragma unroll
      for (int k = 1; k < KU; ++k)
#pragma unroll
        for (int j = 0; j < NJ; ++j)
          w[k][j] = ldg_nc(wp2 + ((size_t)(rtile * 128 + (rank * WK + warp) * KU + k) * NJ + j) * 32 + lane);
    } else {
#pragma unroll
      for (int k = 0; k < KU; ++k)
#pragma unroll
        for (int j = 0; j < NJ; ++j)
          w[k][j] = ldg_nc(wp + ((size_t)(rtile * 128 + (rank * WK + warp) * KU + k) * NJ + j) * 32 + lane);
    }
  };
  auto as_fetch = [&](int k, int g) {   // wait for commit group (k, g), pull its slots into registers
    if constexpr (AS) {
      const int left = 2 * KU - 1 - (2 * k + g);
      if (left == 0) cp_async_wait<0>(); else if (left == 1) cp_async_wait<1>(); else if (left == 2) cp_async_wait<2>();
      else if (left == 3) cp_async_wait<3>(); else if (left == 4) cp_async_wait<4>(); else if (left == 5) cp_async_wait<5>();
      else if (left == 6) cp_async_wait<6>(); else cp_async_wait<7>();
#pragma unroll
      for (int j = 0; j < NJ; ++j)
        if (as_grp<CMP>(j) == g) w[k][j] = lds128(wsm + (((warp * KU + k) * NJ + j) * 32 + lane) * 16);
    }
  };
  if (XF & 4) asm volatile("bar.sync 2, %0;" :: "r"(WK * 32 + 32) : "memory");   // after the helper's zero stores
  if (XF & 1) { ldx(); ldw(); } else { ldw(); ldx(); }
  if (TRA && !(XF & 8)) {   // trace: this lane's loads landed
    u32 a = 0;
#pragma unroll
    for (int k = 0; k < KU; ++k) {
#pragma unroll
      for (int j = 0; j < NJ; ++j) a ^= w[k][j].x;
#pragma unroll
      for (int n = 0; n < MT; ++n) a ^= xx[k][n][0].x ^ xx[k][n][1].w;
    }
    if (a == 0x7654321u) p.trc[1] = 0;
    if (warp == 0) TA(1);
  }
  float c[RT][2][MT][4];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int n = 0; n < MT; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) c[r][h][n][e] = 0.f;
#pragma unroll
  for (int k = 0; k < KU; ++k) {
    if (CMP) {
#pragma unroll
      for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          if (AS && r == 0 && (q & 1) == 0) as_fetch(k, q >> 1);
          const int pp = q * RT + r;
          const uint4 HL = w[k][pp >> 1], LL = w[k][NP / 2 + (pp >> 1)];
          const u32 hn = comp4(w[k][NP + (pp >> 2)], pp & 3), lnn = comp4(w[k][NP + NP / 4 + (pp >> 2)], pp & 3);
          uint4 ah, al;
          if (pp & 1) dec_pair<B4>(HL.z, HL.w, hn, LL.z, LL.w, lnn, ah, al);
          else dec_pair<B4>(HL.x, HL.y, hn, LL.x, LL.y, lnn, ah, al);
#pragma unroll
          for (int n = 0; n < MT; ++n) {
            const u32 b0 = (q < 2) ? (q == 0 ? xx[k][n][0].x : xx[k][n][0].z) : (q == 2 ? xx[k][n][1].x : xx[k][n][1].z);
            const u32 b1 = (q < 2) ? (q == 0 ? xx[k][n][0].y : xx[k][n][0].w) : (q == 2 ? xx[k][n][1].y : xx[k][n][1].w);
            mma16816(c[r][0][n], ah, b0, b1);
            mma16816(c[r][1][n], al, b0, b1);
          }
        }
    } else if (AS) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        if ((q & 1) == 0) as_fetch(k, q >> 1);
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int n = 0; n < MT; ++n) {
            const u32 b0 = (q < 2) ? (q == 0 ? xx[k][n][0].x : xx[k][n][0].z) : (q == 2 ? xx[k][n][1].x : xx[k][n][1].z);
            const u32 b1 = (q < 2) ? (q == 0 ? xx[k][n][0].y : xx[k][n][0].w) : (q == 2 ? xx[k][n][1].y : xx[k][n][1].w);
            mma16816(c[0][h][n], w[k][q * 2 + h], b0, b1);
          }
      }
    } else {
#pragma unroll
      for (int n = 0; n < MT; ++n) {
        const u32 b[8] = {xx[k][n][0].x, xx[k][n][0].y, xx[k][n][0].z, xx[k][n][0].w,
                          xx[k][n][1].x, xx[k][n][1].y, xx[k][n][1].z, xx[k][n][1].w};
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int r = 0; r < RT; ++r) mma16816(c[r][h][n], w[k][(q * 2 + h) * RT + r], b[2 * q], b[2 * q + 1]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int n = 0; n < MT; ++n)
      part[warp * NI + (r * MT + n) * 32 + lane] =
          make_float4(c[r][0][n][0] + c[r][1][n][0], c[r][0][n][1] + c[r][1][n][1],
                      c[r][0][n][2] + c[r][1][n][2], c[r][0][n][3] + c[r][1][n][3]);
  if (warp == 0) TA(2);
  asm volatile("bar.sync 1, %0;" :: "r"(WK * 32) : "memory");
  if (warp == 0) TA(3);
  for (int v = tid; v < NI; v += WK * 32) {   // sum over warps, scatter into the [token][row] staging tile
    float4 s4 = part[v];
#pragma unroll
    for (int q = 1; q < WK; ++q) f4add(s4, part[q * NI + v]);
    const int r = v / (MT * 32), n = (v / 32) % MT, ln = v % 32;
    const int row = r * 16 + (ln >> 2), tok = n * 8 + (ln & 3) * 2;
    T[tok * TP + row] = s4.x; T[(tok + 1) * TP + row] = s4.y; T[tok * TP + row + 8] = s4.z; T[(tok + 1) * TP + row + 8] = s4.w;
  }
  asm volatile("bar.sync 1, %0;" :: "r"(WK * 32) : "memory");
  if (warp == 0) TA(4);
  mbar_wait_cta(zb, 0);
  if (warp == 0) TA(5);
  for (int i = tid; i < MT * 8 * (RR / 4); i += WK * 32) {
    const int tok = i / (RR / 4), q4 = i % (RR / 4);
    if (tok < m) {
      const float4 v = *reinterpret_cast<const float4*>(T + tok * TP + q4 * 4);
      asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
                   :: "l"(p.out + (size_t)tok * 256 + rtile * RR + q4 * 4), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
    }
  }
  if (warp == 0) TA(6);
}
"""


_yr_YB = 54   # hi e7 window [54, 61] = exponents 108..123


def _yr_ya_name(rt, s, mt, xf, cmp=0, tra=0):
    return f"ya_{rt}_{s}_{mt}_{xf}_{cmp}" + ("_t" if tra else "")


def _yr_ya_smem(rt, s, mt, wk=8, xf=0, cmp=0):
    wk = 16 if xf & 32 else wk
    ni = rt * mt * 32
    base = wk * ni * 16 + mt * 8 * (16 * rt + 4) * 4 + 16
    if xf & 8:
        ku = 128 // (s * wk)
        base += 16 + wk * ku * (6 if cmp else 8) * rt * 32 * 16
    return base


def _yr_ya_source(cfgs, waddr, eaddr=None, emax=16):
    """cfgs: (rt, s, mt, xf[, cmp]); waddr[(rt, cmp)] (or waddr[rt] for raw), eaddr[(rt, s)] escape tables (cmp)."""
    src = _yr_HEAD
    for cf in cfgs:
        rt, s, mt, xf = cf[:4]
        cmp = cf[4] if len(cf) > 4 else 0
        tra = cf[5] if len(cf) > 5 else 0
        wa = waddr[(rt, cmp)] if (rt, cmp) in waddr else waddr[rt]
        pt = "PAE" if cmp == 2 else "PA"
        wk = 16 if xf & 32 else 8
        src += (f'extern "C" __global__ void __launch_bounds__({(wk + 1) * 32}, 1) {_yr_ya_name(rt, s, mt, xf, cmp, tra)}(const __grid_constant__ {pt} p) {{\n'
                f'  const uint4* wp; asm("mov.u64 %0, %1;" : "=l"(wp) : "n"({wa}ull));\n')
        if tra:
            src += f'  ya_body<{rt}, {s}, {wk}, {mt}, {xf}, 0, 0u, 0, PA, 1>(p, wp);\n}}\n'
        elif cmp == 2:
            src += f'  ya_body<{rt}, {s}, {wk}, {mt}, {xf}, 1, {_yr_YB * 0x01010101}u, -1, PAE>(p, wp);\n}}\n'
        elif cmp:
            src += (f'  const int4* ep; asm("mov.u64 %0, %1;" : "=l"(ep) : "n"({eaddr[(rt, s)]}ull));\n'
                    f'  ya_body<{rt}, {s}, {wk}, {mt}, {xf}, 1, {_yr_YB * 0x01010101}u, {emax}>(p, wp, ep);\n}}\n')
        else:
            src += f'  ya_body<{rt}, {s}, {wk}, {mt}, {xf}>(p, wp);\n}}\n'
    return src


def _yr_ya_param_escapes(hi, lo, codes, rt, s, maxent=472, maxcta=32):
    """Param-block escape list for ya (rt, s): bytes of (eo[132] u16, ent[472] (u32 row|k<<5, f32 delta)) or None.
    delta = (hi + lo) - (hi_dec + lo_dec) exactly: split into fp32 pieces (usually one) -> one entry per piece."""
    hn, hl, lnb, ll, delta = codes
    nz = delta.nonzero().tolist()
    kc = 8192 // s
    nrt = 256 // (16 * rt)
    per = {}
    for row, k in nz:
        rem = float(delta[row, k])
        rtile, rr = divmod(row, 16 * rt)
        c = rtile * s + k // kc
        for _ in range(4):
            if rem == 0.0:
                break
            piece = float(torch.tensor(rem, dtype=torch.float64).float())
            per.setdefault(c, []).append(((rr | (k << 5)), piece))
            rem = rem - piece
        if rem != 0.0:
            return None
    if any(len(v) > maxcta for v in per.values()):
        return None
    eo, ents = [0], []
    for c in range(nrt * s):
        ents += per.get(c, [])
        eo.append(len(ents))
    if len(ents) > maxent or nrt * s + 1 > 132:
        return None
    eo += [len(ents)] * (132 - len(eo))
    b = struct.pack("<132H", *eo)
    b += b"".join(struct.pack("<If", rk, d) for rk, d in ents) + b"\0" * (8 * (maxent - len(ents)))
    return b


def _yr_ya_escape_table(delta, rt, s, emax=16):
    """Per CTA (rtile*S + rank): int4 (row in rtile, k, d1 bits, d2 bits); None if a CTA has more than emax escapes."""
    nz = delta.nonzero()
    kc = 8192 // s
    nrt = 256 // (16 * rt)
    ents = {}
    for row, k in nz.tolist():
        dv = float(delta[row, k])
        d1 = torch.tensor(dv, dtype=torch.float64).float()
        d2 = (torch.tensor(dv, dtype=torch.float64) - d1.double()).float()
        rtile, rr = divmod(row, 16 * rt)
        ents.setdefault(rtile * s + k // kc, []).append((rr, k, int(d1.view(torch.int32)), int(d2.view(torch.int32))))
    if any(len(v) > emax for v in ents.values()):
        return None
    tab = torch.zeros((nrt * s, emax, 4), dtype=torch.int32)
    for c, lst in ents.items():
        tab[c, :len(lst)] = torch.tensor(lst, dtype=torch.int64).to(torch.int32)
    return tab


def _yr_ys_name(rt, s, mt, xf, trc=0, cmp=0):
    return f"ys_{rt}_{s}_{mt}_{xf}_{trc}_{cmp}"


def _yr_ys_smem(rt, s, mt, wk=8, esc=False):
    ni = rt * mt * 32
    ku = 128 // (s * wk)
    return wk * ni * 16 + s * (ni // s) * 16 + 16


def _yr_ys_source(cfgs, waddr, eaddr=None, emax=None):
    src = _yr_HEAD
    eaddr = eaddr or {}
    emax = emax or {}
    for cf in cfgs:
        (rt, s, mt, xf, trc, cmp) = cf
        em = emax.get(cf, 0) if cmp else 0
        nt = 256
        src += (f'extern "C" __global__ void __launch_bounds__({nt}, 1) {_yr_ys_name(rt, s, mt, xf, trc, cmp)}(const __grid_constant__ P p) {{\n'
                f'  const uint4* wp; asm("mov.u64 %0, %1;" : "=l"(wp) : "n"({waddr[(rt, cmp)]}ull));\n')
        if em == "gen":
            nm = _yr_ys_name(rt, s, mt, xf, trc, cmp)
            src = src[:src.rindex('extern "C"')] + _yr_eg_source(nm, eaddr[cf], rt, s, mt) + src[src.rindex('extern "C"'):]
            src += f'  ys_body<{rt}, {s}, 8, {mt}, {xf}, {trc}, {cmp}, {_yr_YB * 0x01010101}u, 0, EG_{nm}>(p, wp);\n}}\n'
        elif em == "const":
            nm = _yr_ys_name(rt, s, mt, xf, trc, cmp)
            ents, offs = eaddr[cf]
            pre = (f'__constant__ int4 ce_{nm}[{max(1, len(ents))}] = {{' + ",".join("{%d,%d,%d,%d}" % tuple(e) for e in ents) + '};\n'
                   f'__constant__ unsigned short co_{nm}[{len(offs)}] = {{' + ",".join(str(o) for o in offs) + '};\n'
                   f'struct EC_{nm} {{ static constexpr bool on = true, gen = false; template <class C, class X> __device__ static void apply(C&, const X&, int, int) {{}} __device__ static int4 ent(int i) {{ return ce_{nm}[i]; }} '
                   f'__device__ static int off(int i) {{ return co_{nm}[i]; }} }};\n')
            src = src[:src.rindex('extern "C"')] + pre + src[src.rindex('extern "C"'):]
            src += f'  ys_body<{rt}, {s}, 8, {mt}, {xf}, {trc}, {cmp}, {_yr_YB * 0x01010101}u, 0, EC_{nm}>(p, wp);\n}}\n'
        elif em:
            src += (f'  const int4* ep; asm("mov.u64 %0, %1;" : "=l"(ep) : "n"({eaddr[cf]}ull));\n'
                    f'  ys_body<{rt}, {s}, 8, {mt}, {xf}, {trc}, {cmp}, {_yr_YB * 0x01010101}u, {em}>(p, wp, ep);\n}}\n')
        else:
            src += f'  ys_body<{rt}, {s}, 8, {mt}, {xf}, {trc}, {cmp}, {_yr_YB * 0x01010101}u>(p, wp);\n}}\n'
    return src


def _yr_ys_decode_codes(hn, hl, lnb, ll, base=_yr_YB):
    """Emulate the kernel decode on per-weight codes -> (hi_dec, lo_dec) as float64 tensors."""
    he7 = (hn & 7) + base
    hbits = ((hn >> 3) << 15) | (he7 << 8) | hl
    le7 = he7 - 4 - (lnb & 7)
    lbits = ((lnb >> 3) << 15) | (le7 << 8) | ll

    def tof(b):
        b = b.to(torch.int32)
        b = torch.where(b >= 32768, b - 65536, b).to(torch.int16)
        return b.view(torch.bfloat16).double()
    return tof(hbits), tof(lbits)


def _yr_ys_codes(hi, lo, base=_yr_YB):
    h16 = hi.contiguous().view(torch.int16).int() & 0xFFFF
    l16 = lo.contiguous().view(torch.int16).int() & 0xFFFF
    he7 = (h16 >> 8) & 0x7F
    le7 = (l16 >> 8) & 0x7F
    dl = he7 - 4 - le7
    hbad = (he7 < base) | (he7 > base + 7)
    lbad = hbad | ((l16 & 0x7FFF) == 0) | (dl < 0) | (dl > 7)
    hn = torch.where(hbad, torch.zeros_like(h16), ((h16 >> 15) << 3) | (he7 - base).clamp(0, 7))
    hl = torch.where(hbad, torch.zeros_like(h16), h16 & 0xFF)
    lnb = torch.where(lbad, torch.full_like(l16, 7), ((l16 >> 15) << 3) | dl.clamp(0, 7))
    ll = torch.where(lbad, torch.zeros_like(l16), l16 & 0xFF)
    hd, ld = _yr_ys_decode_codes(hn, hl, lnb, ll, base)
    delta = (hi.double() + lo.double()) - (hd + ld)
    return hn, hl, lnb, ll, delta


def _yr_eg_source(nm, ents_by_cta, rt, s, mt):
    """Generated escape code: switch (blockIdx.x) -> per warp esc1<...> calls with literal positions and deltas."""
    lines = [f"struct EG_{nm} {{ static constexpr bool gen = true, on = false;",
             "  __device__ static int4 ent(int) { return make_int4(0, 0, 0, 0); } __device__ static int off(int) { return 0; }",
             "  template <class C, class X> __device__ static void apply(C& c, const X& xx, int warp, int lane) {",
             "    switch (blockIdx.x) {"]
    for cta in sorted(ents_by_cta):
        lines.append(f"    case {cta}:")
        for (warp, rr, kq, d1, d2) in ents_by_cta[cta]:
            r, g, half = rr >> 4, rr & 7, (rr >> 3) & 1
            ku, kt, kj = kq >> 6, (kq >> 4) & 3, kq & 15
            lines.append(f"      if (warp == {warp}) esc1<{r}, {g}, {half}, {ku}, {kt}, {kj}, {mt}>(c, xx, lane, "
                         f"__int_as_float({d1 & 0xFFFFFFFF}u), __int_as_float({d2 & 0xFFFFFFFF}u));")
        lines.append("      break;")
    lines += ["    default: break;", "    }", "  }", "};", ""]
    return "\n".join(lines)


def _yr_ys_escape_gen(delta, rt, s, mt, wk=8):
    """Escapes grouped per CTA: {cta: [(warp, row_in_rtile, k_in_warp, d1 bits, d2 bits)]}."""
    nz = delta.nonzero()
    kc = 8192 // s
    kw = kc // wk
    out = {}
    for row, k in nz.tolist():
        dv = float(delta[row, k])
        d1 = torch.tensor(dv, dtype=torch.float64).float()
        d2 = (torch.tensor(dv, dtype=torch.float64) - d1.double()).float()
        rtile, rr = divmod(row, 16 * rt)
        rank, kl = divmod(k, kc)
        warp, kq = divmod(kl, kw)
        out.setdefault(rtile * s + rank, []).append((warp, rr, kq, int(d1.view(torch.int32)), int(d2.view(torch.int32))))
    return out


def _yr_ys_escape_const(delta, rt, s, mt, wk=8):
    """Flat per-warp escape lists for the __constant__ path: (ents [[x,0,d1,d2]...], offs [nwarps+1])."""
    tab, mx = _yr_ys_escape_entries(delta, rt, s, mt, 1 << 20, wk, shrink=True)
    nz = (tab[:, :, 2] != 0) | (tab[:, :, 3] != 0)
    ents, offs = [], [0]
    for wi in range(tab.shape[0]):
        n = int(nz[wi].sum())
        ents += tab[wi, :n].tolist()
        offs.append(len(ents))
    return ents, offs


def _yr_ys_escape_entries(delta, rt, s, mt, emax, wk=8, shrink=False):
    """Per (CTA, warp) list of int4 (row_in_rtile | k_in_warp << 8, 0, d1, d2); None if a warp needs > emax."""
    nz = delta.nonzero()
    nrt = 256 // (16 * rt)
    kc = 8192 // s
    kw = kc // wk
    ents = {}
    for row, k in nz.tolist():
        dv = float(delta[row, k])
        d1 = torch.tensor(dv, dtype=torch.float64).float()
        d2 = (torch.tensor(dv, dtype=torch.float64) - d1.double()).float()
        rtile, rr = divmod(row, 16 * rt)
        rank, kl = divmod(k, kc)
        warp, kq = divmod(kl, kw)
        ents.setdefault((rtile * s + rank) * wk + warp, []).append((rr | (kq << 8), 0, int(d1.view(torch.int32)), int(d2.view(torch.int32))))
    if shrink:
        emax = max((len(v) for v in ents.values()), default=1)
    tab = torch.zeros((nrt * s * wk, emax, 4), dtype=torch.int32)
    for c, lst in ents.items():
        if len(lst) > emax:
            return None, max(len(v) for v in ents.values())
        tab[c, :len(lst)] = torch.tensor(lst, dtype=torch.int64).to(torch.int32)
    return tab, max((len(v) for v in ents.values()), default=0)


def _yr_ys_pack_c(hi, lo, rt, base=_yr_YB, codes=None):
    """24-bit pair format in the ys fragment order. Returns (packed int32 [NRT,128,6RT,32,4], delta [256,8192] f64)."""
    nrt = 256 // (16 * rt)
    np_ = 4 * rt
    hn, hl, lnb, ll, delta = codes if codes is not None else _yr_ys_codes(hi, lo, base)

    def frag(v):   # [256, 8192] -> [NRT, 128 u, NP (q, r), 32 lanes, 8 e]
        v = v.view(nrt, rt, 2, 8, 128, 4, 4, 2, 2)                 # rt r hh g u t q p i
        v = v.permute(0, 4, 6, 1, 3, 5, 7, 2, 8)                   # rt u q r g t p hh i
        return v.reshape(nrt, 128, np_, 32, 8)

    def bytes2(v):  # [..., 8] bytes -> [..., 2] u32 words (e0..e3, e4..e7)
        v = v.view(*v.shape[:-1], 2, 4)
        return v[..., 0] | (v[..., 1] << 8) | (v[..., 2] << 16) | (v[..., 3] << 24)

    def nibw(v):    # [..., 8] nibbles -> [...] u32: byte b = nib[b] | nib[b+4] << 4
        b = v[..., :4] | (v[..., 4:] << 4)
        return b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24)

    HLw, LLw = bytes2(frag(hl)), bytes2(frag(ll))              # [nrt,128,NP,32,2]
    HNw, LNw = nibw(frag(hn)), nibw(frag(lnb))                 # [nrt,128,NP,32]

    def lslots(v):  # pairs (2j, 2j+1) -> slot j: [nrt,128,NP/2,32,4]
        return v.view(nrt, 128, np_ // 2, 2, 32, 2).permute(0, 1, 2, 4, 3, 5).reshape(nrt, 128, np_ // 2, 32, 4)

    def nslots(v):
        return v.view(nrt, 128, np_ // 4, 4, 32).permute(0, 1, 2, 4, 3)

    out = torch.cat([lslots(HLw), lslots(LLw), nslots(HNw), nslots(LNw)], dim=2)
    out = out.to(torch.int64)
    out = torch.where(out >= 2 ** 31, out - 2 ** 32, out).to(torch.int32).contiguous()
    return out, delta


def _yr_ys_pack(hi, lo, rt):
    """[NRT, 128 units, 4 q, 2 h, RT, 32 lanes, 8] bf16: lane (g, t), step q holds rows g / g+8, k = u*64 + 16t + 4q + 0..3."""
    nrt = 256 // (16 * rt)
    parts = []
    for w in (hi, lo):
        v = w.contiguous().view(torch.int16).view(nrt, rt, 2, 8, 128, 4, 4, 2, 2)  # rt r hh g u t q p i
        parts.append(v.permute(0, 4, 6, 1, 3, 5, 7, 2, 8))                          # rt u q r g t p hh i
    return torch.stack(parts, dim=3).contiguous()                                    # rt u q h r g t p hh i


class _yr_Mod:
    def __init__(self, src, name="yr.cu"):
        from cuda.bindings import driver, nvrtc
        self.d = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        self.log = log.decode(errors="replace").strip("\x00 \n")
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + self.log)
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        self.cubin = cubin
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
            (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1)
            assert int(err) == 0, err
            self.funcs[name] = f
        return f

    def attrs(self, name, smem):
        d = self.d
        f = self.func(name, smem)
        out = {}
        for k in ("NUM_REGS", "LOCAL_SIZE_BYTES"):
            err, v = d.cuFuncGetAttribute(getattr(d.CUfunction_attribute, "CU_FUNC_ATTRIBUTE_" + k), f)
            out[k] = v
        return out

    def _cfg(self, grid, block, smem, cluster, stream=None):
        d = self.d
        cfg = d.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block, 1, 1
        cfg.sharedMemBytes = smem
        if stream is not None:
            cfg.hStream = stream
        attrs = []
        if cluster > 1:
            at = d.CUlaunchAttribute()
            at.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = cluster, 1, 1
            attrs.append(at)
        cfg.attrs = attrs
        cfg.numAttrs = len(attrs)
        return cfg

    def max_clusters(self, name, grid, block, smem, cluster):
        f = self.func(name, smem)
        err, n = self.d.cuOccupancyMaxActiveClusters(f, self._cfg(grid, block, smem, cluster))
        return int(n) if int(err) == 0 else -int(err)

    def launch(self, name, grid, block, smem, cluster, params):
        d = self.d
        f = self.func(name, smem)
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        cfg = self._cfg(grid, block, smem, cluster, d.CUstream(torch.cuda.current_stream().cuda_stream))
        (err,) = d.cuLaunchKernelEx(cfg, f, ctypes.addressof(argv), 0)
        assert int(err) == 0, (name, err)




# ------------------------------------------------------------------ rk port
_yr_RK_SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
struct __align__(64) TMap { u64 v[16]; };
struct __align__(64) P { TMap mw; TMap mx; float* out; int* zcnt; u64* trc; int M; int S; int ntt; int pad; TMap mo; };
__device__ __forceinline__ u64 gtime() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `rk_kernel` and its helpers (RK_SRC) (via alygg77/okc-muse-spark-kernels).
// Changes: ported from a torch extension to NVRTC (one parameter struct, ring depth QS a template parameter) and extended with template options: multicast of the W tiles across a cluster of token tiles (MC), a 128-token tile (BN), four appended zeroing CTAs (RD=2) or one TMA reduce-add of the tile (RD=1), DSMEM pre-reduction of K-split pairs (PR), trace hooks, and a 0.2 s instead of 2 s clear timeout.
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
__device__ __forceinline__ u64 rk_desc(u32 addr) {
  return (u64)((addr & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)64 << 32) | (1ull << 62);
}
__device__ __forceinline__ void rk_mbar_init(u32 bar, u32 cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(cnt)); }
__device__ __forceinline__ void rk_mbar_expect(u32 bar, u32 bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory"); }
__device__ __forceinline__ void rk_mbar_arrive(u32 bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory"); }
__device__ __forceinline__ void rk_mbar_wait(u32 bar, u32 ph) {
  asm volatile("{\n.reg .pred P1;\nRKW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n@!P1 bra RKW_%=;\n}\n" :: "r"(bar), "r"(ph) : "memory");
}
__device__ __forceinline__ void rk_tma2d(u32 dst, const TMap* map, u32 bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void rk_wgmma(float d[4][8], u64 da, u64 db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n"
               : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
               : "l"(da), "l"(db));
}
__device__ __forceinline__ void rk_tma2d_mc(u32 dst, const TMap* map, u32 bar, int c0, int c1, unsigned short mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "h"(mask) : "memory");
}
__device__ __forceinline__ u32 rk_mapa(u32 a, int r) { u32 b; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(b) : "r"(a), "r"(r)); return b; }
__device__ __forceinline__ void rk_arrive_remote(u32 a) { asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(a) : "memory"); }
__device__ __forceinline__ void rk_wgmma128(float d[8][8], u64 da, u64 db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1, 1, 1, 0, 0;\n"
               : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
               : "l"(da), "l"(db));
}
#define TRK(i) do { if (TRC) { p.trc[blockIdx.x * 16 + (i)] = gtime(); } } while (0)

template <int QS, int TRC, int MC, int BN, int PR = 0, int RD = 0>
__device__ __forceinline__ void rk_body(const P& p) {
  constexpr int BM = 64, NJ = BN / 16;
  constexpr int W_BYTES = 2 * BM * 128, ST = W_BYTES + BN * 128;
  extern __shared__ __align__(1024) unsigned char rk_smem[];
  const int M = p.M, S = p.S, ntt = p.ntt;
  const u32 sraw = (u32)__cvta_generic_to_shared(rk_smem);
  const u32 sbase = (sraw + 1023u) & ~1023u;
  const u32 fullb = sbase + QS * ST, emptyb = fullb + 8 * QS, zbar = emptyb + 8 * QS;
  int rt, tt, sp;
  if (PR) {   // cluster = a pair of adjacent K splits: partials pre-reduced through DSMEM, half the red.adds
    const int rest = blockIdx.x >> 1;
    rt = rest & 3; tt = (rest >> 2) % ntt; sp = 2 * ((rest >> 2) / ntt) + (blockIdx.x & 1);
  } else if (MC > 1) { tt = blockIdx.x % ntt; rt = (blockIdx.x / ntt) & 3; sp = blockIdx.x / (4 * ntt); }   // cluster = the ntt token tiles
  else { rt = blockIdx.x & 3; tt = (blockIdx.x >> 2) % ntt; sp = (blockIdx.x >> 2) / ntt; }
  const int crank = MC > 1 ? tt : 0;
  const int kb0 = (sp * 128) / S, nk = ((sp + 1) * 128) / S - kb0;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    TRK(0);
    asm volatile("prefetch.tensormap [%0];" :: "l"(&p.mw) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&p.mx) : "memory");
    for (int i = 0; i < QS; ++i) { rk_mbar_init(fullb + 8 * i, 1); rk_mbar_init(emptyb + 8 * i, MC); }
    rk_mbar_init(zbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (MC > 1) {   // peers' barriers must be initialized before any multicast / remote arrive
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
  }
  if (RD == 2) {   // 4 zeroing CTAs appended to the grid: they clear out and publish with a release add on counter A
    const int ncomp = (int)gridDim.x - 4;
    if ((int)blockIdx.x >= ncomp) {
      const int zi = blockIdx.x - ncomp;
      const int n4 = M * 64, per = (n4 + 3) / 4;
      const int z0 = zi * per, z1 = min(n4, z0 + per);
      for (int i = z0 + tid; i < z1; i += blockDim.x) reinterpret_cast<float4*>(p.out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
      __syncthreads();
      if (tid == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(p.zcnt) : "memory");
      return;
    }
    if (warp == 5) {   // generation from counter B (relaxed atomic), wait for A >= 4 * (gen + 1)
      if (lane == 0) {
        int v;
        asm volatile("atom.relaxed.gpu.global.add.s32 %0, [%1], 1;" : "=r"(v) : "l"(p.zcnt + 256) : "memory");
        const int target = (v / ncomp + 1) * 4;
        const u64 tz = gtime();
        for (;;) {
          int a;
          asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(a) : "l"(p.zcnt) : "memory");
          if (a - target >= 0) break;
          if (gtime() - tz > 200000000ull) __trap();
          __nanosleep(32);
        }
        TRK(9);
        rk_mbar_arrive(zbar);
      }
      return;
    }
  }
  if (warp == 5) {
    int target = 0;
    if (lane == 0) {
      int v;
      asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p.zcnt) : "memory");
      target = (v / (int)gridDim.x + 1) * (int)gridDim.x;
    }
    const int n4 = M * 64;
    const int per = (n4 + (int)gridDim.x - 1) / (int)gridDim.x;
    const int z0 = blockIdx.x * per, z1 = min(n4, z0 + per);
    for (int i = z0 + lane; i < z1; i += 32) reinterpret_cast<float4*>(p.out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncwarp();
    if (lane == 0) {
      asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(p.zcnt) : "memory");
      TRK(8);
      const u64 tz = gtime();
      for (;;) {
        int v;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p.zcnt) : "memory");
        if (v - target >= 0) break;
        if (gtime() - tz > 200000000ull) __trap();   // 0.2 s: not all CTAs resident -> fail loudly, never hang
        __nanosleep(64);
      }
      TRK(9);
      rk_mbar_arrive(zbar);
    }
    if (!PR) return;
  }
  else if (warp == 4) {
    if (lane == 0) {
      for (int i = 0; i < nk; ++i) {
        const int s = i % QS;
        if (i >= QS) rk_mbar_wait(emptyb + 8 * s, ((i / QS) - 1) & 1);
        const u32 fb = fullb + 8 * s, st = sbase + s * ST;
        rk_mbar_expect(fb, ST);
        const int kb = kb0 + i;
        if (MC > 1) {   // this CTA's 64/MC rows of the hi and lo tiles, multicast to every token tile of the cluster
          constexpr int RR = BM / MC;
          const unsigned short mask = (unsigned short)((1u << MC) - 1);
          rk_tma2d_mc(st + crank * RR * 128, &p.mw, fb, kb * 64, rt * BM + crank * RR, mask);
          rk_tma2d_mc(st + BM * 128 + crank * RR * 128, &p.mw, fb, 8192 + kb * 64, rt * BM + crank * RR, mask);
        } else {
          rk_tma2d(st, &p.mw, fb, kb * 64, rt * BM);
          rk_tma2d(st + BM * 128, &p.mw, fb, 8192 + kb * 64, rt * BM);
        }
        rk_tma2d(st + W_BYTES, &p.mx, fb, kb * 64, tt * BN);
      }
      TRK(10);
    }
    if (!PR) return;
  } else {
  float d[NJ][8];
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int e = 0; e < 8; ++e) d[j][e] = 0.f;
  for (int i = 0; i < nk; ++i) {
    const int s = i % QS;
    rk_mbar_wait(fullb + 8 * s, (i / QS) & 1);
    if (TRC && tid == 0 && i == 0) TRK(1);
    if (TRC && tid == 0 && i == nk - 1) TRK(2);
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    const u32 a = sbase + s * ST, b = a + W_BYTES;
#pragma unroll
    for (int k16 = 0; k16 < 4; ++k16) {
      if constexpr (BN == 64) {
        rk_wgmma(d, rk_desc(a + 32 * k16), rk_desc(b + 32 * k16));
        rk_wgmma(d, rk_desc(a + BM * 128 + 32 * k16), rk_desc(b + 32 * k16));
      } else {
        rk_wgmma128(d, rk_desc(a + 32 * k16), rk_desc(b + 32 * k16));
        rk_wgmma128(d, rk_desc(a + BM * 128 + 32 * k16), rk_desc(b + 32 * k16));
      }
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
    if (i > 0 && (i - 1) + QS < nk) {   // release slot (i-1) to every producer that multicasts into it
      if (MC > 1) { if (tid < MC) rk_arrive_remote(rk_mapa(emptyb + 8 * ((i - 1) % QS), tid)); }
      else if (tid == 0) rk_mbar_arrive(emptyb + 8 * ((i - 1) % QS));
    }
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int e = 0; e < 8; ++e) asm volatile("" : "+f"(d[j][e]) :: "memory");
  if (TRC && tid == 0) TRK(3);
  constexpr int TP = RD == 1 ? 64 : 68;   // staging [BN tokens][64 rows (+ pad)] fp32 (RD: dense box for the TMA reduce)
  float* T = reinterpret_cast<float*>(rk_smem + (sbase - sraw));
  const int rl = (tid >> 5) * 16 + ((tid & 31) >> 2);
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int cl = j * 16 + h * 8 + (tid & 3) * 2;
      T[cl * TP + rl] = d[j][4 * h]; T[cl * TP + rl + 8] = d[j][4 * h + 2];
      T[(cl + 1) * TP + rl] = d[j][4 * h + 1]; T[(cl + 1) * TP + rl + 8] = d[j][4 * h + 3];
    }
  asm volatile("bar.sync 1, 128;" ::: "memory");
  if (RD == 1) {   // one TMA tensor reduce-add of the whole [BN tokens][64 rows] fp32 tile (rows >= M clipped by the map)
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("bar.sync 1, 128;" ::: "memory");
    rk_mbar_wait(zbar, 0);
    if (TRC && tid == 0) TRK(4);
    if (tid == 0) {
      asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2}], [%3];"
                   :: "l"(&p.mo), "r"(rt * BM), "r"(tt * BN), "r"(sbase) : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
    }
    if (TRC && tid == 0) TRK(5);
  } else if (!PR) {
  rk_mbar_wait(zbar, 0);
  if (TRC && tid == 0) TRK(4);
  for (int idx = tid; idx < BN * 16; idx += 128) {
    const int tk = idx >> 4, qd = idx & 15;
    const int c = tt * BN + tk;
    if (c < M) {
      const float4 v = *reinterpret_cast<const float4*>(T + tk * TP + qd * 4);
      asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
                   :: "l"(p.out + (size_t)c * 256 + rt * BM + qd * 4), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
    }
  }
  if (TRC && tid == 0) TRK(5);
  }
  }   // consumer
  if (PR) {
    __syncwarp();
    asm volatile("barrier.cluster.arrive.release;" ::: "memory");   // partial tiles staged in both CTAs
    asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
    if (warp < 4) {
      constexpr int TP = 68;
      const int rank = blockIdx.x & 1;
      const u32 tl = sbase;
      const u32 tr = rk_mapa(sbase, rank ^ 1);
      rk_mbar_wait(zbar, 0);
      if (TRC && tid == 0) TRK(4);
      for (int idx = tid; idx < BN * 8; idx += 128) {   // this CTA's half of the token rows
        const int tk = rank * (BN / 2) + (idx >> 4), qd = idx & 15;
        const int c = tt * BN + tk;
        if (c < M) {
          const u32 off = (u32)((tk * TP + qd * 4) * 4);
          float4 a, b;
          asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(a.x), "=f"(a.y), "=f"(a.z), "=f"(a.w) : "r"(tl + off));
          asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(b.x), "=f"(b.y), "=f"(b.z), "=f"(b.w) : "r"(tr + off));
          asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
                       :: "l"(p.out + (size_t)c * 256 + rt * BM + qd * 4), "f"(a.x + b.x), "f"(a.y + b.y), "f"(a.z + b.z), "f"(a.w + b.w) : "memory");
        }
      }
      if (TRC && tid == 0) TRK(5);
    }
    __syncwarp();
    asm volatile("barrier.cluster.arrive.release;" ::: "memory");   // the partner has read my tile before I exit
    asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
  }
}
// SPDX-SnippetEnd
"""


def _yr_rk_name(qs, trc, mc=1, bn=64, pr=0, rd=0):
    return f"rk_{qs}_{trc}_{mc}_{bn}_{pr}" + (f"_r{rd}" if rd else "")


def _yr_rk_smem(qs, bn=64):
    return 1024 + qs * (16384 + bn * 128) + 8 * (2 * qs + 1) + 64


def _yr_rk_source(cfgs):
    s = _yr_RK_SRC
    for cf in cfgs:
        qs, trc, mc, bn = cf[:4]
        pr = cf[4] if len(cf) > 4 else 0
        rd = cf[5] if len(cf) > 5 else 0
        s += (f'extern "C" __global__ void __launch_bounds__(192, 1) {_yr_rk_name(qs, trc, mc, bn, pr, rd)}(const __grid_constant__ P p) '
              f'{{ rk_body<{qs}, {trc}, {mc}, {bn}, {pr}, {rd}>(p); }}\n')
    return s


def _yr_rk_tmap(ptr, rows, cols, box_rows=64, box_cols=64):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(2), ptr,
        [d.cuuint64_t(cols), d.cuuint64_t(rows)], [d.cuuint64_t(cols * 2)], [d.cuuint32_t(box_cols), d.cuuint32_t(box_rows)],
        [d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])



# ------------------------------------------------------------------ deliverable API
_yr_RT3_YS = [((2, 8), (2, 8, 1))]           # m range -> ys (RT, S, XF), raw hi/lo, cluster DSMEM reduce
_yr_RT3_YA = [((9, 32), (1, 8, 17))]         # m range -> ya (RT, S, XF), raw hi/lo, no clusters, 4 zeroing CTAs + red.add
_yr_RT3_RK = [((97, 128), 16)]   # m range -> rk K splits (BN 64, QS 6); m=256: no win over bo1's rk (same kernel)


def _yr_prepare_rt3(weights):
    w = weights.get("router")
    if w is None or tuple(w.shape) != (256, 8192) or w.dtype != torch.float32 or not w.is_cuda:
        return {"ok": False}
    if torch.cuda.get_device_capability(w.device) != (9, 0):
        return {"ok": False}
    w = w.contiguous()
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)
    st = {"ok": True, "ys": {}, "rk": {}}
    # ys (small m)
    cfgs = sorted({(rt, s, mt, xf, 0, 0) for (_, (rt, s, xf)) in _yr_RT3_YS for mt in (1,)})
    st["ys_pack"] = {rt: _yr_ys_pack(hi, lo, rt) for rt in sorted({c[0] for c in cfgs})}
    st["ys_mod"] = _yr_Mod(_yr_ys_source(cfgs, {(rt, 0): t.data_ptr() for rt, t in st["ys_pack"].items()}), "rt3_ys.cu")
    for (rt, s, mt, xf, trc, cmp) in cfgs:
        name = _yr_ys_name(rt, s, mt, xf, trc, cmp)
        grid = (256 // (16 * rt)) * s
        smem = _yr_ys_smem(rt, s, mt)
        if st["ys_mod"].max_clusters(name, grid, 256, smem, s) >= grid // s:   # all clusters co-resident
            st["ys"][(rt, s, mt, xf)] = (name, grid, smem)
    st["trc"] = torch.zeros(16, dtype=torch.int64, device=w.device)
    st["zc"] = torch.zeros(1024, dtype=torch.int32, device=w.device)   # grid counters (rk, ya), indexed by grid size
    # ya (m 9..32): every CTA must be resident (in-kernel grid counter) -> check occupancy
    nsm = torch.cuda.get_device_properties(w.device).multi_processor_count
    acfg = sorted({(rt, s, mt, xf) for (_, (rt, s, xf)) in _yr_RT3_YA for mt in (2, 4)})
    st["ya_pack"] = {rt: (st["ys_pack"][rt] if rt in st["ys_pack"] else _yr_ys_pack(hi, lo, rt)) for rt in sorted({c[0] for c in acfg})}
    st["ya_mod"] = _yr_Mod(_yr_ya_source(acfg, {rt: t.data_ptr() for rt, t in st["ya_pack"].items()}), "rt3_ya.cu")
    st["ya"] = {}
    for (rt, s, mt, xf) in acfg:
        name, smem = _yr_ya_name(rt, s, mt, xf), _yr_ya_smem(rt, s, mt)
        grid = (256 // (16 * rt)) * s + (4 if xf & 16 else 0)
        f = st["ya_mod"].func(name, smem)
        err, nb = st["ya_mod"].d.cuOccupancyMaxActiveBlocksPerMultiprocessor(f, 288, smem)
        if int(err) == 0 and int(nb) * nsm >= grid:
            st["ya"][(rt, s, mt, xf)] = (name, grid, smem)
    # rk (m 97..256)
    st["cat"] = torch.cat([hi, lo], dim=1).contiguous()
    st["mw"] = _yr_rk_tmap(st["cat"].data_ptr(), 256, 16384)
    st["rk_mod"] = _yr_Mod(_yr_rk_source([(6, 0, 1, 64, 0)]), "rt3_rk.cu")
    nsm = torch.cuda.get_device_properties(w.device).multi_processor_count
    name, smem = _yr_rk_name(6, 0, 1, 64), _yr_rk_smem(6, 64)
    f = st["rk_mod"].func(name, smem)
    err, nb = st["rk_mod"].d.cuOccupancyMaxActiveBlocksPerMultiprocessor(f, 192, smem)
    st["rk_ok"] = int(err) == 0 and int(nb) >= 1
    st["nsm"] = nsm
    st["xm"] = {}
    torch.cuda.synchronize()
    return st


def _yr__rt3_run_ys(x, m, st):
    for (lo_m, hi_m), (rt, s, xf) in _yr_RT3_YS:
        if lo_m <= m <= hi_m:
            mt = 1 if m <= 8 else 2 if m <= 16 else 4
            e = st["ys"].get((rt, s, mt, xf))
            if e is None:
                return None
            name, grid, smem = e
            out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
            params = struct.pack("<QQqQ", x.data_ptr(), out.data_ptr(), m, st["trc"].data_ptr())
            st["ys_mod"].launch(name, grid, 256, smem, s, params)
            return out
    return None


def _yr__rt3_run_ya(x, m, st):
    for (lo_m, hi_m), (rt, s, xf) in _yr_RT3_YA:
        if lo_m <= m <= hi_m:
            mt = 1 if m <= 8 else 2 if m <= 16 else 4
            e = st["ya"].get((rt, s, mt, xf))
            if e is None:
                return None
            name, grid, smem = e
            out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
            # counters: A = zc[grid] (zeroing CTAs, release), B = zc[grid + 256] (generation)
            params = struct.pack("<QQqQQ", x.data_ptr(), out.data_ptr(), m, st["trc"].data_ptr(), st["zc"][grid:].data_ptr())
            st["ya_mod"].launch(name, grid, 288, smem, 1, params)
            return out
    return None


def _yr__rt3_run_rk(x, m, st):
    if not st.get("rk_ok"):
        return None
    for (lo_m, hi_m), S in _yr_RT3_RK:
        if lo_m <= m <= hi_m:
            ntt = (m + 63) // 64
            grid = 4 * ntt * S
            if grid > st["nsm"]:
                return None
            out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
            key = (x.data_ptr(), m)
            mx = st["xm"].get(key)
            if mx is None:
                if len(st["xm"]) > 64:
                    st["xm"].clear()
                mx = st["xm"][key] = _yr_rk_tmap(x.data_ptr(), m, 8192, box_rows=64)
            zc = st["zc"][grid:]   # one grid counter per grid size
            params = st["mw"] + mx + struct.pack("<QQQiiii", out.data_ptr(), zc.data_ptr(), st["trc"].data_ptr(), m, S, ntt, 0)
            st["rk_mod"].launch(_yr_rk_name(6, 0, 1, 64), grid, 192, _yr_rk_smem(6, 64), 1, params)
            return out
    return None


def _yr_run_rt3(x, config, state):
    if not state.get("ok") or config.get("layer") != "router" or config.get("n") != 256:
        return None
    if x.dim() != 2 or x.shape[1] != 8192 or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    m = x.shape[0]
    if m <= 8:
        return _yr__rt3_run_ys(x, m, state)
    if m <= 32:
        return _yr__rt3_run_ya(x, m, state)
    if m > 64:
        return _yr__rt3_run_rk(x, m, state)
    return None


# ======================================================================== module K: K/g256k.py
"""g256k (agent K): generated by mkg256k.py from kg.py + the prepare_g256k / run_g256k API (end of file).
Self-contained (torch + cuda.bindings NVRTC). Derived from agent V's vg.py.

kg (agent K, derived from agent V's vg.py): BF16 SS-wgmma GEMM for the m = 256 (verify-32) shapes. out[M, N] = x[M, K] @ W[N, K]^T.

Non-swapped orientation: A = x tile (tokens, 64 per warpgroup), B = W tile (BN rows of W = output columns), both
TMA-loaded with 128B swizzle into an NS-stage ring (one full + one empty mbarrier per stage). 1 producer warpgroup
(thread 0 issues all TMA; setmaxnreg 40) + 2 consumer warpgroups (setmaxnreg 232).
cfg = (TM, BN, LAY, S, NS):
  TM  tokens per CTA tile (64 or 128)
  BN  W rows (output columns) per CTA tile
  LAY 0: WG g owns tokens [64g, 64g + 64) x all BN columns (TM = 128)
      1: both WGs on the same 64 tokens, WG g owns columns [g BN/2, (g + 1) BN/2)            (TM = 64)
      2: both WGs on the same 64 tokens x BN columns, WG g takes the k-blocks kb = g mod 2;
         the two partials are added through smem at the end                                   (TM = 64)
  S   K splits = cluster size (1 or 2). S = 2: the two CTAs of a cluster take the two K halves of the same tile;
      the partial of the WG the partner finalises is bulk-copied smem -> DSMEM (complete_tx on the partner's
      mbarrier), the receiving WG adds it, converts to BF16 and stores (reduce-scatter, no global partials).
  NS  ring stages.
Epilogue: BF16 via stmatrix into a padded smem tile, then 16-B coalesced global stores (rows >= M masked).
Compiled with NVRTC at prepare(); W tensor maps are built once, x maps are cached per x pointer.

Agent K's additions, cfg = (TM, BN, LAY, S, NS, MC, RED, MAP, PERM, PF, PFX, EPI, PRO, PRI, LASTD, WPK, NT2, EPF, XLD):
  EPI   2: epilogue straight from the accumulators (4x4 quad transpose of bf16x2 words -> STG.128), 1: TMA store
  PRO   1: fast prologue (first two stages issued before the other barrier inits)
  PRI   cap on the stages in flight at the start (0 = all NS)
  LASTD tail priority: warpgroup 1 runs its last LASTD k-blocks after warpgroup 0 finished (named barrier 5); both
        solo phases are issued back to back; warpgroup 0's stores overlap warpgroup 1's last MMAs (LAY 0/1, S = 1)
  WPK   1/2: W prepacked (pack_w) into contiguous 128B-swizzled stage images, one 1D bulk copy per stage
        (2: packed-W address baked into the code)
  NT2   two BN-column sub-tiles per CTA (sub-tile 0 stored by the producer warpgroup's warps 1-3)   [not used]
  EPF   early prefetch.global.L2 of the first stages                                               [not used]
  XLD   x of the first XLD stages by LDG -> STS from all threads                                    [not used]
  PF/PFX tensor L2 prefetch of later stages                                                         [not used]
"""
import ctypes
import os
import struct

import torch

_kk_SMEM_MAX = 227 * 1024


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: 2026 Nathen Fernandes <https://github.com/NathenFernandes/muse-bf16-proj>
# SPDX-License-Identifier: MIT
# Adapted from NathenFernandes/muse-bf16-proj, `_mma` wgmma wrapper generator (via alygg77/okc-muse-spark-kernels).
# Changes: emits a predicate-register (setp) form of the scale-d operand; renamed.
# License text: LICENSES/MIT.txt; notices: NOTICE (repository root).
def _kk__wg_ss(n):
    nr = n // 2
    regs = ", ".join(f"%{i}" for i in range(nr))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nr))
    return (f'__device__ __forceinline__ void wg_ss_n{n}(float* d, u64 da, u64 db) {{\n'
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{nr}, %{nr + 1}, p, 1, 1, 0, 0;\\n}}\\n"\n'
            f'    : {outs} : "l"(da), "l"(db));\n}}\n')
# SPDX-SnippetEnd


_kk_HEAD = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) P { TMap tmx; TMap tmw; TMap tmo; u16* out; unsigned long long* trace; uint4* ws; unsigned* flags; const unsigned char* wp; const unsigned char* xp; int M; int MT; int NH; int PERM;
                         unsigned short perm[136]; };
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#ifdef VG_TRACE
#define TR(i) do { if (p.trace) { p.trace[blockIdx.x * 16 + (i)] = gtime(); p.trace[4096 + blockIdx.x * 16 + (i)] = clock64(); } } while (0)
#else
#define TR(i) do { } while (0)
#endif
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
__device__ __forceinline__ void tma2d(u32 dst, const void* tm, int c0, int c1, u32 bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma2d_h(u32 dst, const void* tm, int c0, int c1, u32 bar, u64 pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;"
               :: "r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void tma2d_mc(u32 dst, const void* tm, int c0, int c1, u32 bar, u16 mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;"
               :: "r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(bar), "h"(mask) : "memory");
}
__device__ __forceinline__ void mbar_arrive_cl(u32 remote) {
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(remote) : "memory");
}
__device__ __forceinline__ uint4 lds128(u32 a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }
__device__ __forceinline__ void sts128(u32 a, uint4 v) { asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
__device__ __forceinline__ u64 mkdesc(u32 saddr) {
  return (u64)((saddr & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)(1024 >> 4) << 32) | ((u64)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void fence_acc(float* a) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(a[i]) :: "memory");
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b)); return r; }
"""

_kk_BODY = r"""
#ifndef VG_BACKOFF
#define VG_BACKOFF 200
#endif
#ifdef VG_TMASTORE
#define VG_TMASTORE_ON 1
#else
#define VG_TMASTORE_ON 0
#endif
#ifdef VG_PUBBULK
#define VG_PUBBULK_ON 1
#else
#define VG_PUBBULK_ON 0
#endif
#ifdef VG_NOSETMAX
#define VG_NOSETMAX_ON 1
#else
#define VG_NOSETMAX_ON 0
#endif
#ifdef VG_NORED
#define VG_NORED_ON 1
#else
#define VG_NORED_ON 0
#endif
template <int WN> __device__ __forceinline__ void WGSS(float* d, u64 da, u64 db);
@WGSS_SPECS@
// stmatrix the 64 x WN accumulator tile of this warpgroup as BF16 into a row-major smem tile (row stride STR bytes)
template <int WN, int STR>
__device__ __forceinline__ void stage_bf16(const float* acc, u32 T, int lane, int wq) {
  constexpr int NG = WN / 8;
  const int mi = lane >> 3, rr = (mi & 1) * 8 + (lane & 7);
#pragma unroll
  for (int c = 0; c < NG; c += 2) {
    if (c + 1 < NG) {
      const u32 r0 = pack_bf2(acc[4 * c], acc[4 * c + 1]), r1 = pack_bf2(acc[4 * c + 2], acc[4 * c + 3]);
      const u32 r2 = pack_bf2(acc[4 * c + 4], acc[4 * c + 5]), r3 = pack_bf2(acc[4 * c + 6], acc[4 * c + 7]);
      const u32 a = T + (16 * wq + rr) * STR + (8 * (c + (mi >> 1))) * 2;
      asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
                   :: "r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
    } else {
      const u32 r0 = pack_bf2(acc[4 * c], acc[4 * c + 1]), r1 = pack_bf2(acc[4 * c + 2], acc[4 * c + 3]);
      const u32 a = T + (16 * wq + rr) * STR + (8 * c) * 2;
      asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};" :: "r"(a), "r"(r0), "r"(r1) : "memory");
    }
  }
}
// EPI 2: BF16 stores straight from the wgmma accumulators (no smem staging): per group of 4 n8-blocks a 4 x 4 transpose
// of bf16x2 words inside each lane quad gives every lane 16 contiguous bytes of one row -> STG.128 (8 rows x 64 B per
// warp instruction, full sectors)
template <int WN, int N>
__device__ __forceinline__ void store_direct(const float* acc, u16* out, int tok, int col, int M, int lane, int wq) {
  const int q = lane & 3, r0 = 16 * wq + (lane >> 2);
#pragma unroll
  for (int m4 = 0; m4 < WN / 32; ++m4) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {             // h = 0: row r0, h = 1: row r0 + 8
      u32 w[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) w[i] = pack_bf2(acc[4 * (4 * m4 + i) + 2 * h], acc[4 * (4 * m4 + i) + 2 * h + 1]);
      {   // stage k = 2
        const bool hi = q & 2;
        const u32 s0 = hi ? w[0] : w[2], s1 = hi ? w[1] : w[3];
        const u32 x0 = __shfl_xor_sync(0xffffffffu, s0, 2), x1 = __shfl_xor_sync(0xffffffffu, s1, 2);
        if (hi) { w[0] = x0; w[1] = x1; } else { w[2] = x0; w[3] = x1; }
      }
      {   // stage k = 1
        const bool od = q & 1;
        const u32 s0 = od ? w[0] : w[1], s1 = od ? w[2] : w[3];
        const u32 x0 = __shfl_xor_sync(0xffffffffu, s0, 1), x1 = __shfl_xor_sync(0xffffffffu, s1, 1);
        if (od) { w[0] = x0; w[2] = x1; } else { w[1] = x0; w[3] = x1; }
      }
      const int row = tok + r0 + 8 * h;
      if (row < M)
        asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(out + (size_t)row * N + col + 8 * (4 * m4 + q)),
                     "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]) : "memory");
    }
  }
}
// 64 rows x WN columns of BF16 from smem tile T to out (row tok, column col); 128 threads (ct)
template <int WN, int STR, int N>
__device__ __forceinline__ void store_tile(u32 T, u16* out, int tok, int col, int M, int ct) {
  constexpr int CH = WN / 8;                       // 16-B chunks per row
#pragma unroll 4
  for (int e = ct; e < 64 * CH; e += 128) {
    const int row = e / CH, c = e % CH;
    if (tok + row < M) {
      const uint4 v = lds128(T + row * STR + c * 16);
      asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(out + (size_t)(tok + row) * N + col + c * 8),
                   "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
    }
  }
}

template <int TM, int BN, int LAY, int S, int NS, int NKB, int N, int MC, int RED, int MAP, int PERMC, int MTC, int NHC, int PF, int PFX, int EPI, int PRO, int PRI, int LASTD, int WPK, int NKBT, class WPBT, int NT2, int EPF, int XLD>
__device__ __forceinline__ void vg_body(const P& p) {
  constexpr bool FASTPRO_C = PRO == 1 && MC == 0 && !(S > 1 && RED == 0);
  static_assert(EPF == 0 || WPK > 0, "EPF: packed W");
  static_assert(NT2 == 0 || (S == 1 && LAY == 0 && MC == 0 && RED == 0), "NT2: S = 1, LAY 0");
  static_assert(WPK == 0 || MC == 0, "packed W: no multicast");
  constexpr int WN = LAY == 1 ? BN / 2 : BN;       // wgmma N per warpgroup
  constexpr int NACC = WN / 2;
  constexpr int XB = TM * 128, WB = BN * 128, SB = XB + WB;
  constexpr int NCW = (LAY == 2 ? 4 : 8) * (MC ? 2 : 1);   // consumer warps releasing one stage (both CTAs if MC)
  static_assert(!(MC && S > 1 && !VG_NORED_ON), "multicast pairs use the cluster: S = 1");
  constexpr int PB = 128 * NACC * 4;               // fp32 partial of one warpgroup (fragment order)
  constexpr int STR = WN * 2 + 16;                 // bf16 staging row stride
  static_assert(XB % 1024 == 0 && WB % 1024 == 0, "1024-B stage alignment");
  static_assert(LAY != 0 || TM == 128, "LAY 0: 2 x 64 tokens");
  static_assert(LAY == 0 || TM == 64, "LAY 1/2: 64 tokens");
  static_assert(!(LAY == 2 && S > 1), "k-split layout: S = 1");
  static_assert(WN % 8 == 0 && WN <= 256, "wgmma N");
  static_assert(NS * SB >= PB + 2 * 64 * STR, "epilogue staging fits in the ring");
  static_assert(RED != 4 || (SB >= PB && SB >= 64 * STR && NS >= 4), "RED 4: partial halves / staging in ring slots");
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  const u32 bars = (smem_u32(sm_raw) + 1023) & ~1023u;
  const u32 full = bars, empty = bars + 8 * NS, rbar = bars + 16 * NS;
  const u32 ring = bars + 1024;
  const u32 rbuf = ring + NS * SB;                  // RED 0: partner partial (DSMEM); RED 2: helper partial (both WGs)
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31;
  constexpr int CLN = (S > 1 && RED == 0 && !VG_NORED_ON) ? S : (MC ? 2 : 1);   // cluster size of this launch
  // optional unit permutation (GPC-aware tile placement measured at prepare): unit = CTA or cluster
  // MTC / NHC: token tiles and tiles per k-split, compile-time (no parameter load before the first TMA)
  const int bid = !PERMC ? (int)blockIdx.x
                : CLN == 1 ? (int)p.perm[blockIdx.x] : CLN * (int)p.perm[blockIdx.x / CLN] + (int)(blockIdx.x % CLN);
  // RED 2 (asymmetric split-K): r = 0 helpers (bids [0, NH)) take k-blocks [NA, NKBT), r = 1 finishers [0, NA)
  const int r = S > 1 ? ((MAP == 1 || RED >= 2) ? bid / NHC : bid % S) : 0;
  const int rest = (MAP == 1 || RED >= 2) ? bid % NHC : bid / S;
  constexpr int NA = RED >= 2 ? NKB + MAP : NKB;   // RED 2: NKB = half the k-blocks, MAP = extra k-blocks of finishers
  const int nkb = RED >= 2 ? (r ? NA : 2 * NKB - NA) : NKB;
  const int c = MC ? (bid & 1) : 0;                // multicast pair rank
  const int base = bid >> 1;
  constexpr int NTL = NHC / MTC;                     // column tiles
  const int mt = MC == 1 ? 2 * (base % (MTC >> 1)) + c : MC == 2 ? base % MTC : (MAP == 2 && RED < 2) ? rest / NTL : rest % MTC;
  const int nt = MC == 1 ? base / (MTC >> 1) : MC == 2 ? 2 * (base / MTC) + c : (MAP == 2 && RED < 2) ? rest % NTL : rest / MTC;
  const int ntc = NT2 ? 2 * nt : nt;               // NT2: the CTA owns column tiles ntc, ntc + 1 (processed in turn)
  const int tok0 = mt * TM, col0 = ntc * BN;
  const int nkbt = NT2 ? 2 * nkb : nkb;             // stages streamed by this CTA
  const int kb0 = RED >= 2 ? (r ? 0 : NA) : r * NKB;
  static_assert(XLD == 0 || (FASTPRO_C && XLD <= 2 && NT2 == 0), "XLD: fast prologue, <= 2 stages");
  constexpr int XLDN = (XLD * TM * 8 + 383) / 384;     // 16-B chunks per thread
  uint4 xv[XLDN > 0 ? XLDN : 1];
  if constexpr (XLD > 0) {
#pragma unroll
    for (int i = 0; i < XLDN; ++i) {
      const int e = tid + 384 * i;
      xv[i] = make_uint4(0u, 0u, 0u, 0u);
      if (e < XLD * TM * 8) {
        const int st = e / (TM * 8), rr = (e / 8) % TM, cc = e % 8;
        if (tok0 + rr < p.M)
          asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(xv[i].x), "=r"(xv[i].y), "=r"(xv[i].z), "=r"(xv[i].w)
                       : "l"(p.xp + ((size_t)(tok0 + rr) * NKBT + kb0 + st) * 128 + cc * 16));
      }
    }
  }
  if constexpr (EPF > 0) {
    // early L2 prefetch (LSU prefetch.global.L2, first thing in the kernel, before the barrier init and the stage
    // copies): the CTA's first NS W stages (contiguous in the packed W) and, EPF 2, its x rows' first NS k-blocks
    if (tid < 128) {
      u64 wb;
      if constexpr (WPK == 2) wb = WPBT::get(); else wb = (u64)p.wp;
      const u64 w0 = wb + ((u64)ntc * NKBT + (u64)kb0) * WB;
#pragma unroll 1
      for (int l = tid; l < NS * WB / 128; l += 128)
        asm volatile("prefetch.global.L2 [%0];" :: "l"(w0 + (u64)l * 128) : "memory");
      if (EPF == 2) {
        constexpr int LPR = NS;                         // lines per x row (one 128 B line per k-block)
#pragma unroll 1
        for (int l = tid; l < TM * LPR; l += 128) {
          const int row = l / LPR, kk = l % LPR;
          if (tok0 + row < p.M)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(p.xp + ((size_t)(tok0 + row) * NKBT + kb0 + kk) * 128) : "memory");
        }
      }
    }
  }
  // stage loads: own tiles always; with MC the shared tile of k-block kb is issued by pair rank kb & 1 and
  // multicast into both CTAs (same smem offset, complete_tx on each CTA's full barrier)
#ifdef VG_WPOL
  u64 polw; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(polw));
#endif
#ifdef VG_XPOL
  u64 polx; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(polx));
#endif
  auto issue_own = [&](int slot, int sidx, bool withx = true) {
    const int tsub = NT2 ? (sidx >= nkb) : 0, kb = sidx - tsub * nkb;
#ifndef VG_NOX
    if (withx)
#ifdef VG_XPOL
    if (MC != 2) tma2d_h(ring + slot * SB, &p.tmx, (kb0 + kb) * 64, tok0, full + 8 * slot, polx);
#else
    if (MC != 2) tma2d(ring + slot * SB, &p.tmx, (kb0 + kb) * 64, tok0, full + 8 * slot);
#endif
#endif
#ifndef VG_NOW
    if constexpr (WPK > 0) {
      // W prepacked at prepare(): [N / BN][K / 64][BN rows][128 B, 128B-swizzled] -> one contiguous 1D bulk copy per stage
      // (DRAM-page friendly, no tensor-map descriptor fetch on the critical path)
      u64 wb;
      if constexpr (WPK == 2) wb = WPBT::get();     // baked at NVRTC time (literal in the asm text)
      else wb = (u64)p.wp;
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   :: "r"(ring + slot * SB + XB), "l"(wb + ((u64)(ntc + tsub) * NKBT + (u64)(kb0 + kb)) * WB), "r"(WB), "r"(full + 8 * slot) : "memory");
    } else {
#ifdef VG_WPOL
    if (MC != 1) tma2d_h(ring + slot * SB + XB, &p.tmw, (kb0 + kb) * 64, col0 + tsub * BN, full + 8 * slot, polw);
#else
    if (MC != 1) tma2d(ring + slot * SB + XB, &p.tmw, (kb0 + kb) * 64, col0 + tsub * BN, full + 8 * slot);
#endif
    }
#endif
  };
  // L2 prefetch of k-block kb (W, optionally x) PF k-blocks ahead of the ring: DRAM latency leaves the ring's
  // critical path (the ring then only waits for L2 hits)
  auto pf = [&](int kb) {
    if (PF > 0 && kb < nkb) {
      if (MC != 1) asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"(&p.tmw), "r"((kb0 + kb) * 64), "r"(col0) : "memory");
      if (PFX && MC != 2) asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"(&p.tmx), "r"((kb0 + kb) * 64), "r"(tok0) : "memory");
    }
  };
  auto issue_shared = [&](int slot, int kb) {
    if (MC && (kb & 1) == c) {
      if (MC == 1) tma2d_mc(ring + slot * SB + XB, &p.tmw, (kb0 + kb) * 64, col0, full + 8 * slot, (u16)3);
      else tma2d_mc(ring + slot * SB, &p.tmx, (kb0 + kb) * 64, tok0, full + 8 * slot, (u16)3);
    }
  };
#if defined(VG_NOX)
  constexpr u32 EXP = WB;
#elif defined(VG_NOW)
  constexpr u32 EXP = XB;
#else
  constexpr u32 EXP = SB;
#endif
  const int NPRO = NS < nkbt ? NS : nkbt;
  static_assert(!(RED == 3 && WN > 128), "RED 3 polls from the producer warpgroup: needs its full register file");

  constexpr bool FASTPRO = PRO == 1 && MC == 0 && !(S > 1 && RED == 0);
  constexpr int LASTDE = (S == 1 && (LAY == 0 || LAY == 1) && !(VG_TMASTORE_ON || EPI == 1) && MC == 0) ? LASTD : 0;
  static_assert(LASTDE == 0 || LASTDE <= NS - 2, "LASTD <= NS - 2");
#ifdef KG_NOSOLO
  constexpr bool SOLOE = false;
#else
  constexpr bool SOLOE = LASTDE > 0;
#endif
  if (tid == 0 && !FASTPRO) {
    TR(0);
#ifdef VG_TRACE
    if (p.trace) { unsigned sid; asm volatile("mov.u32 %0, %%smid;" : "=r"(sid)); p.trace[8192 + blockIdx.x] = sid; }
#endif
#ifdef VG_PFTM
    asm volatile("prefetch.tensormap [%0];" :: "l"(&p.tmx) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&p.tmw) : "memory");
#endif
#pragma unroll 1
    for (int s = 0; s < NS; ++s) { mbar_init(full + 8 * s, 1); mbar_init(empty + 8 * s, NCW); }
    mbar_init(rbar, RED == 3 ? 128 : 1);
    mbar_init(rbar + 8, 1);
    if (S > 1 && RED == 0) mbar_expect_arrive(rbar, PB);
#ifdef KG_CTAFENCE
    if (CLN == 1 && !(S > 1 && RED == 0)) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    else asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#else
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
    TR(10);
#pragma unroll 1
    for (int s = 0; s < NPRO; ++s) {
      mbar_expect_arrive(full + 8 * s, EXP);
      issue_own(s, s);
      if (s == 0) TR(13);
    }
    TR(12);
    if (PF > 0) {
#pragma unroll 1
      for (int q = 0; q < PF; ++q) pf(NPRO + q);
    }
  }
  if (tid == 0 && FASTPRO) {
    // fast prologue: the first two stages' full barriers, fence, their loads; then the rest of the barriers
    TR(0);
#ifdef VG_TRACE
    if (p.trace) { unsigned sid; asm volatile("mov.u32 %0, %%smid;" : "=r"(sid)); p.trace[8192 + blockIdx.x] = sid; }
#endif
    mbar_init(full, 1);
    mbar_init(full + 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    // XLD stages: x is loaded by all threads with LDG -> STS (below), only W is copied here
    mbar_expect_arrive(full, XLD > 0 ? (u32)WB : EXP);
    issue_own(0, 0, XLD < 1);
    TR(13);
    mbar_expect_arrive(full + 8, XLD > 1 ? (u32)WB : EXP);
    issue_own(1, 1, XLD < 2);
#pragma unroll
    for (int s = 2; s < NS; ++s) mbar_init(full + 8 * s, 1);
#pragma unroll
    for (int s = 0; s < NS; ++s) mbar_init(empty + 8 * s, NCW);
    mbar_init(rbar, RED == 3 ? 128 : 1);
    mbar_init(rbar + 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    TR(10);
    constexpr int PRIE = PRI > 0 ? PRI : NS;      // stages in flight at the start
#pragma unroll 1
    for (int s = 2; s < NPRO && s < PRIE; ++s) {
      mbar_expect_arrive(full + 8 * s, EXP);
      issue_own(s, s);
    }
    TR(12);
    if (PF > 0) {
#pragma unroll 1
      for (int q = 0; q < PF; ++q) pf(NPRO + q);
    }
  }
  if constexpr (XLD > 0) {
    // x of the first XLD stages through the LSU (no tensor-map descriptor fetch on the critical path): LDG.128 by all
    // 384 threads (issued first, see xv), STS into the 128B-swizzled slot layout, proxy fence, then the CTA barrier
#pragma unroll
    for (int i = 0; i < XLDN; ++i) {
      const int e = tid + 384 * i;
      if (e < XLD * TM * 8) {
        const int st = e / (TM * 8), rr = (e / 8) % TM, cc = e % 8;
        sts128(ring + st * SB + rr * 128 + ((cc ^ (rr & 7)) << 4), xv[i]);
      }
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
#ifdef VG_TRACE
  if (tid == 128) TR(11);
#endif
  if (S > 1 && RED == 0 && !VG_NORED_ON) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  if (MC) {   // the partner's barriers are initialised before any multicast / remote arrive
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
  }
  u32 rempty = 0;
  if (MC) asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rempty) : "r"(empty), "r"(c ^ 1));
  if (wg == 0) {
    if constexpr (RED != 3 && !VG_NOSETMAX_ON) asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    if (tid == 0) {
#pragma unroll 1
      for (int s = 0; s < NPRO; ++s) issue_shared(s, s);
      if constexpr (FASTPRO && PRI > 0) {
        // ramp: stage s is issued once stage s - PRI has landed (at most PRI stages in flight at the start)
#pragma unroll 1
        for (int s = PRI; s < NPRO; ++s) {
          mbar_wait(full + 8 * (s - PRI), 0);
          mbar_expect_arrive(full + 8 * s, EXP);
          issue_own(s, s);
        }
      }
      int slot = 0, ph = 0;
#pragma unroll 1
      for (int kb = NS; kb < nkbt; ++kb) {
        mbar_wait(empty + 8 * slot, ph);
        mbar_expect_arrive(full + 8 * slot, EXP);
        issue_own(slot, kb);
        issue_shared(slot, kb);
        pf(kb + PF);
        if (++slot == NS) { slot = 0; ph ^= 1; }
      }
      if (RED == 3 && r == 1) mbar_arrive(rbar + 8);
      if (RED == 4 && r == 1) {
        // the helper partial lands in the two ring slots that k-blocks nkb, nkb + 1 would use (freed in order)
        u32 dst[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          mbar_wait(empty + 8 * slot, ph);
          dst[h] = ring + slot * SB;
          if (++slot == NS) { slot = 0; ph ^= 1; }
        }
        unsigned* fl = p.flags + rest;
        unsigned f;
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(f) : "l"(fl) : "memory");
        while (f == 0u) {
          __nanosleep(VG_BACKOFF);
          asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(f) : "l"(fl) : "memory");
        }
        asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(fl), "r"(0u) : "memory");
        asm volatile("fence.proxy.async.global;" ::: "memory");
        mbar_expect_arrive(rbar, 2 * PB);
#pragma unroll
        for (int h = 0; h < 2; ++h)
          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                       :: "r"(dst[h]), "l"(p.ws + (size_t)rest * (2 * PB / 16) + h * (PB / 16)), "r"(PB), "r"(rbar) : "memory");
        TR(12);
      }
      if (RED == 2 && r == 1) {
        unsigned* fl = p.flags + rest;
        unsigned f;
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(f) : "l"(fl) : "memory");
        while (f == 0u) {
          __nanosleep(VG_BACKOFF);
          asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(f) : "l"(fl) : "memory");
        }
        asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(fl), "r"(0u) : "memory");
        asm volatile("fence.proxy.async.global;" ::: "memory");
        mbar_expect_arrive(rbar, 2 * PB);
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     :: "r"(rbuf), "l"(p.ws + (size_t)rest * (2 * PB / 16)), "r"(2 * PB), "r"(rbar) : "memory");
        TR(12);
      }
    }
    if (RED == 3 && r == 1) {
      // fetch the helper's partial: 16-B cp.async per chunk into rbuf; a chunk still holding the sentinel (not yet
      // written by the helper) is fetched again; then restore the sentinels and release the consumers (rbar)
      mbar_wait(rbar + 8, 0);                          // thread 0 has issued its last TMA
      __syncwarp();
      constexpr int NCH = 2 * PB / 16 / 128;           // 16-B chunks per producer thread
      const int pt = tid;                              // 0..127
      const uint4* src = p.ws + (size_t)rest * (2 * PB / 16);
      u32 pend = (NCH >= 32) ? 0xFFFFFFFFu : ((1u << NCH) - 1u);
      bool first = true;
      while (pend) {
        if (!first) __nanosleep(VG_BACKOFF);
        first = false;
#pragma unroll
        for (int i = 0; i < NCH; ++i)
          if (pend & (1u << i))
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(rbuf + (i * 128 + pt) * 16), "l"(src + i * 128 + pt) : "memory");
        asm volatile("cp.async.wait_all;" ::: "memory");
#pragma unroll
        for (int i = 0; i < NCH; ++i)
          if (pend & (1u << i)) {
            const uint4 v = lds128(rbuf + (i * 128 + pt) * 16);
            if (v.x != 0xFFFFFFFFu && v.y != 0xFFFFFFFFu && v.z != 0xFFFFFFFFu && v.w != 0xFFFFFFFFu) pend &= ~(1u << i);
          }
      }
      if (pt == 0) TR(12);
      mbar_arrive(rbar);
#pragma unroll
      for (int i = 0; i < NCH; ++i)
        asm volatile("st.global.cg.v4.u32 [%0], {%1,%1,%1,%1};" :: "l"(src + i * 128 + pt), "r"(0xFFFFFFFFu) : "memory");
    }
    if (NT2 && tid >= 32) {
      // tile 0's epilogue stores, overlapped with tile 1's main loop: the consumers stage tile 0 as BF16 in the
      // staging buffer (after the ring) and arrive on named barrier 6; warps 1..3 of this warpgroup store it
      asm volatile("bar.sync 6, 352;" ::: "memory");
      constexpr int CH0 = BN / 8, STR0 = BN * 2 + 16;
      const u32 S0 = ring + NS * SB;
#pragma unroll 2
      for (int e = tid - 32; e < TM * CH0; e += 96) {
        const int row = e / CH0, cc = e % CH0;
        if (tok0 + row < p.M) {
          const uint4 v = lds128(S0 + row * STR0 + cc * 16);
          asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(p.out + (size_t)(tok0 + row) * N + col0 + cc * 8),
                       "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
        }
      }
    }
    if (MC) {
      __syncwarp();
      asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
      asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    }
    return;
  }
  if constexpr (RED != 3 && !VG_NOSETMAX_ON) asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
  const int g = __shfl_sync(0xffffffffu, wg, 0) - 1, ct = tid & 127;
  const int wq = __shfl_sync(0xffffffffu, (tid >> 5) & 3, 0);
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = 0.f;
  const u32 xoff = LAY == 0 ? g * 8192 : 0;
  const u32 woff = LAY == 1 ? g * (WN * 128) : 0;
  const u64 da0 = mkdesc(ring + xoff), db0 = mkdesc(ring + XB + woff);
  if (ct == 0) TR(1 + g);
  if constexpr (LAY != 2) {
#ifdef VG_TRACE
    mbar_wait(full, 0);
    if (ct == 0 && g == 0) TR(14);
#endif
    int slot = 0, ph = 0, pslot = 0;
#pragma unroll 1
    for (int kb = 0; kb < nkbt; ++kb) {
      // tail priority: warpgroup 1 yields the tensor pipe for the last LASTD k-blocks until warpgroup 0 is done,
      // so that warpgroup 0's epilogue overlaps warpgroup 1's last MMAs
      if (LASTDE > 0 && g == 1 && kb == nkbt - LASTDE) asm volatile("bar.sync 5, 256;" ::: "memory");
      mbar_wait(full + 8 * slot, ph);
      const u64 so = (u64)(slot * (SB >> 4));
      wg_fence();
#ifndef VG_NOMMA
#pragma unroll
      for (int j = 0; j < 4; ++j) WGSS<WN>(acc, da0 + so + 2 * j, db0 + so + 2 * j);
#endif
      wg_commit();
      // tail priority solo phase: each warpgroup runs its last LASTD k-blocks alone on the tensor pipe; issue them
      // back to back (no wait, no slot release: the producer has issued every stage by then, LASTD <= NS - 2)
      const bool solo = SOLOE && kb >= nkbt - LASTDE;
      if (!solo) {
        wg_wait<1>(); fence_acc<NACC>(acc);
        if (kb > 0) { __syncwarp(); if (lane == 0) { mbar_arrive(empty + 8 * pslot); if (MC) mbar_arrive_cl(rempty + 8 * pslot); } }
      }
      pslot = slot;
      if (++slot == NS) { slot = 0; ph ^= 1; }
      if (NT2 && kb == nkb - 1) {
        // end of tile 0: stage it as BF16 for the store warps, restart the accumulators for tile 1
        wg_wait<0>(); fence_acc<NACC>(acc);
        stage_bf16<WN, BN * 2 + 16>(acc, ring + NS * SB + g * (64 * (BN * 2 + 16)), lane, wq);
        asm volatile("bar.arrive 6, 352;" ::: "memory");
#ifdef VG_TRACE
        if (ct == 0 && g == 0) TR(9);
#endif
#pragma unroll
        for (int i = 0; i < NACC; ++i) acc[i] = 0.f;
      }
    }
    wg_wait<0>(); fence_acc<NACC>(acc);
    if (LASTDE > 0 && g == 0) asm volatile("bar.arrive 5, 256;" ::: "memory");
  } else {
    int slot = g, ph = 0, pslot = -1;
    if (slot >= NS) { slot -= NS; ph ^= 1; }
#pragma unroll 1
    for (int kb = g; kb < NKB; kb += 2) {
      mbar_wait(full + 8 * slot, ph);
      const u64 so = (u64)(slot * (SB >> 4));
      wg_fence();
#ifndef VG_NOMMA
#pragma unroll
      for (int j = 0; j < 4; ++j) WGSS<WN>(acc, da0 + so + 2 * j, db0 + so + 2 * j);
#endif
      wg_commit();
      wg_wait<1>(); fence_acc<NACC>(acc);
      if (pslot >= 0) { __syncwarp(); if (lane == 0) { mbar_arrive(empty + 8 * pslot); if (MC) mbar_arrive_cl(rempty + 8 * pslot); } }
      pslot = slot;
      slot += 2; if (slot >= NS) { slot -= NS; ph ^= 1; }
    }
    wg_wait<0>(); fence_acc<NACC>(acc);
  }
  if (ct == 0) TR(3 + g);
  if (LASTDE == 0 && !(EPI == 2 && S == 1)) asm volatile("bar.sync 1, 256;" ::: "memory");      // both warpgroups are done with the ring
#ifdef VG_TRACE
  if (S == 1 && ct == 0 && g == 0) TR(5);
#endif
  // bf16 staging tile of this warpgroup; with LASTD: ring slots of k-blocks nkb - LASTD - 2 + g (free: both warpgroups
  // are past them and the producer never refills them)
  const u32 T = LASTDE > 0 ? ring + (u32)((nkbt - LASTDE - 2 + g) % NS) * SB : ring + PB + g * (64 * STR);
  const int tokw = tok0 + (LAY == 0 ? 64 * g : 0);
  const int colw = col0 + (NT2 ? BN : 0) + (LAY == 1 ? WN * g : 0);
  if constexpr (LAY == 2) {
    if (g == 1) {
#pragma unroll
      for (int i = 0; i < NACC / 4; ++i)
        sts128(ring + (i * 128 + ct) * 16, make_uint4(__float_as_uint(acc[4 * i]), __float_as_uint(acc[4 * i + 1]),
                                                      __float_as_uint(acc[4 * i + 2]), __float_as_uint(acc[4 * i + 3])));
    }
    asm volatile("bar.sync 1, 256;" ::: "memory");
    if (g == 0) {
#pragma unroll
      for (int i = 0; i < NACC / 4; ++i) {
        const uint4 v = lds128(ring + (i * 128 + ct) * 16);
        acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
        acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
      }
      stage_bf16<WN, STR>(acc, T, lane, wq);
      asm volatile("bar.sync 2, 128;" ::: "memory");
      store_tile<WN, STR, N>(T, p.out, tokw, colw, p.M, ct);
    }
  } else if constexpr ((S == 1 || VG_NORED_ON) && (VG_TMASTORE_ON || EPI == 1) && WN % 64 == 0) {
    // BF16 tile -> 128B-swizzled [64 rows][64 cols] sub-tiles (stmatrix, conflict-free) -> TMA tensor store
    // (rows >= M are clipped by the tensor map); the CTA only waits for the smem reads of the store
    constexpr int NSUB = WN / 64;
    const u32 T0 = (ring + g * NSUB * 8192 + 1023) & ~1023u;
    const int mi = lane >> 3, rr = (mi & 1) * 8 + (lane & 7), row = 16 * wq + rr;
#pragma unroll
    for (int c = 0; c < WN / 8; c += 2) {
      const int col0 = 8 * (c + (mi >> 1));
      const u32 a = T0 + (col0 >> 6) * 8192 + row * 128 + ((((col0 & 63) >> 3) ^ (row & 7)) << 4);
      const u32 r0 = pack_bf2(acc[4 * c], acc[4 * c + 1]), r1 = pack_bf2(acc[4 * c + 2], acc[4 * c + 3]);
      const u32 r2 = pack_bf2(acc[4 * c + 4], acc[4 * c + 5]), r3 = pack_bf2(acc[4 * c + 6], acc[4 * c + 7]);
      asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
                   :: "r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("bar.sync %0, 128;" :: "r"(2 + g) : "memory");
    if (ct == 0) {
#pragma unroll
      for (int q = 0; q < NSUB; ++q)
        asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                     :: "l"(&p.tmo), "r"(colw + 64 * q), "r"(tokw), "r"(T0 + q * 8192) : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
    }
  } else if constexpr ((S == 1 || VG_NORED_ON) && EPI == 2) {
    store_direct<WN, N>(acc, p.out, tokw, colw, p.M, lane, wq);
  } else if constexpr (S == 1 || VG_NORED_ON) {
    stage_bf16<WN, STR>(acc, T, lane, wq);
    asm volatile("bar.sync %0, 128;" :: "r"(2 + g) : "memory");
#ifdef VG_TRACE
    if (ct == 0 && g == 0) TR(6);
#endif
    store_tile<WN, STR, N>(T, p.out, tokw, colw, p.M, ct);
  } else if constexpr (RED == 2 || RED == 3 || RED == 4) {
    constexpr int NQ = NACC / 4;
    if (r == 0 && VG_PUBBULK_ON && RED == 4) {   // helper: stage the partial in smem, one bulk store, flag
#pragma unroll
      for (int i = 0; i < NQ; ++i)
        sts128(ring + ((g * NQ + i) * 128 + ct) * 16, make_uint4(__float_as_uint(acc[4 * i]), __float_as_uint(acc[4 * i + 1]),
                                                                __float_as_uint(acc[4 * i + 2]), __float_as_uint(acc[4 * i + 3])));
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      asm volatile("bar.sync 1, 256;" ::: "memory");
      if (g == 0 && ct == 0) {
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                     :: "l"(p.ws + (size_t)rest * (2 * PB / 16)), "r"(ring), "r"(2 * PB) : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(p.flags + rest), "r"(1u) : "memory");
        TR(9);
      }
    } else if (r == 0) {   // helper: publish my partial (both WGs) for the finisher, then exit
      uint4* slot = p.ws + (size_t)rest * (2 * PB / 16) + (size_t)g * NQ * 128 + ct;
#pragma unroll
      for (int i = 0; i < NQ; ++i)
        asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(slot + i * 128), "r"(__float_as_uint(acc[4 * i])),
                     "r"(__float_as_uint(acc[4 * i + 1])), "r"(__float_as_uint(acc[4 * i + 2])), "r"(__float_as_uint(acc[4 * i + 3]))
                     : "memory");
      if constexpr (RED == 2 || RED == 4) {
#ifdef VG_FENCE_ALL
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
#endif
        asm volatile("bar.sync 1, 256;" ::: "memory");
        if (g == 0 && ct == 0) {
          asm volatile("fence.acq_rel.gpu;" ::: "memory");   // cumulative: covers the CTA's stores seen through bar.sync
          asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(p.flags + rest), "r"(1u) : "memory");
        }
      }
      if (ct == 0) TR(9);
    } else {
      mbar_wait(rbar, 0);
      if (ct == 0) TR(5 + g);
      const u32 rsrc = RED == 4 ? ring + ((nkb + g) % NS) * SB : rbuf + g * NQ * 128 * 16;
      const u32 T4 = RED == 4 ? ring + ((nkb + 2 + g) % NS) * SB : T;
#pragma unroll
      for (int i = 0; i < NQ; ++i) {
        const uint4 v = lds128(rsrc + (i * 128 + ct) * 16);
        acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
        acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
      }
      stage_bf16<WN, STR>(acc, T4, lane, wq);
      asm volatile("bar.sync %0, 128;" :: "r"(2 + g) : "memory");
      store_tile<WN, STR, N>(T4, p.out, tokw, colw, p.M, ct);
    }
  } else if constexpr (RED == 1) {
    // global exchange without flags: slots hold the sentinel 0xFFFFFFFF (never a finite partial) until written;
    // the receiver polls until no word of its slot is the sentinel, then restores the sentinel for the next call
    constexpr int NQ = NACC / 4;
    uint4* slot = p.ws + (size_t)(rest * 2 + g) * NQ * 128 + ct;
    if (g != r) {
#pragma unroll
      for (int i = 0; i < NQ; ++i)
        asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(slot + i * 128), "r"(__float_as_uint(acc[4 * i])),
                     "r"(__float_as_uint(acc[4 * i + 1])), "r"(__float_as_uint(acc[4 * i + 2])), "r"(__float_as_uint(acc[4 * i + 3]))
                     : "memory");
      if (ct == 0) TR(9);
    } else {
      if (ct == 0) TR(10);
      int npoll = 0;
      uint4 v[NQ];
      u32 pend = (1u << NQ) - 1u;
      bool first = true;
      while (pend) {
        if (!first) __nanosleep(VG_BACKOFF);
        first = false;
        ++npoll;
#pragma unroll
        for (int i = 0; i < NQ; ++i)
          if (pend & (1u << i))
            asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v[i].x), "=r"(v[i].y), "=r"(v[i].z), "=r"(v[i].w)
                         : "l"(slot + i * 128) : "memory");
#pragma unroll
        for (int i = 0; i < NQ; ++i)
          if ((pend & (1u << i)) && v[i].x != 0xFFFFFFFFu && v[i].y != 0xFFFFFFFFu && v[i].z != 0xFFFFFFFFu &&
              v[i].w != 0xFFFFFFFFu) pend &= ~(1u << i);
      }
      if (ct == 0) { TR(5 + g); if (p.trace) p.trace[blockIdx.x * 16 + 11] = npoll; }
#pragma unroll
      for (int i = 0; i < NQ; ++i) {
        asm volatile("st.global.cg.v4.u32 [%0], {%1,%1,%1,%1};" :: "l"(slot + i * 128), "r"(0xFFFFFFFFu) : "memory");
        acc[4 * i] += __uint_as_float(v[i].x); acc[4 * i + 1] += __uint_as_float(v[i].y);
        acc[4 * i + 2] += __uint_as_float(v[i].z); acc[4 * i + 3] += __uint_as_float(v[i].w);
      }
      stage_bf16<WN, STR>(acc, T, lane, wq);
      asm volatile("bar.sync %0, 128;" :: "r"(2 + g) : "memory");
      store_tile<WN, STR, N>(T, p.out, tokw, colw, p.M, ct);
    }
  } else {
    if (g != r) {   // sender: my partial goes to the partner, whose warpgroup g finalises this part
#pragma unroll
      for (int i = 0; i < NACC / 4; ++i)
        sts128(ring + (i * 128 + ct) * 16, make_uint4(__float_as_uint(acc[4 * i]), __float_as_uint(acc[4 * i + 1]),
                                                      __float_as_uint(acc[4 * i + 2]), __float_as_uint(acc[4 * i + 3])));
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      asm volatile("bar.sync %0, 128;" :: "r"(2 + g) : "memory");
      asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
      if (ct == 0) {
        u32 rdst, rmb;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rdst) : "r"(rbuf), "r"(1 - r));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rmb) : "r"(rbar), "r"(1 - r));
        asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     :: "r"(rdst), "r"(ring), "r"(PB), "r"(rmb) : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
      }
    } else {
      asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
      mbar_wait(rbar, 0);
      if (ct == 0) TR(5 + g);
#pragma unroll
      for (int i = 0; i < NACC / 4; ++i) {
        const uint4 v = lds128(rbuf + (i * 128 + ct) * 16);
        acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
        acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
      }
      stage_bf16<WN, STR>(acc, T, lane, wq);
      asm volatile("bar.sync %0, 128;" :: "r"(2 + g) : "memory");
      store_tile<WN, STR, N>(T, p.out, tokw, colw, p.M, ct);
    }
  }
  if (ct == 0) TR(7 + g);
  if (MC) {
    asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
  }
}
"""


_kk_PROBE_SRC = r"""
__device__ __forceinline__ void vg_probe_body(int* out) {
  unsigned sid; asm volatile("mov.u32 %0, %%smid;" : "=r"(sid));
  if (threadIdx.x == 0) out[blockIdx.x] = (int)sid;
  const unsigned long long t0 = gtime();
  while (gtime() - t0 < 30000ull) { }
}
extern "C" __global__ void __launch_bounds__(384, 1) vg_probe1(int* out) { vg_probe_body(out); }
extern "C" __global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(384, 1) vg_probe2(int* out) { vg_probe_body(out); }
extern "C" __global__ void __launch_bounds__(128, 1) vg_probex(int* out) { vg_probe_body(out); }
"""


def _kk__specs(wns):
    return "".join(f"template <> __device__ __forceinline__ void WGSS<{n}>(float* d, u64 da, u64 db) {{ wg_ss_n{n}(d, da, db); }}\n"
                   for n in sorted(wns))


class _kk_Module:
    def __init__(self, src, name="vg.cu", opts=()):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        o = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        o += [x.encode() for x in opts]
        if os.environ.get("CU_LINEINFO"):
            o.append(b"-lineinfo")
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(o), o)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        self.log = log.decode(errors="replace").strip("\x00 \n")
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + self.log)
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        self.cubin = cubin
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err
        self.funcs = {}

    def func(self, name, smem):
        key = (name, smem)
        f = self.funcs.get(key)
        if f is None:
            d = self.driver
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            assert int(err) == 0, (name, err)
            if smem > 48 * 1024:
                (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
                assert int(err) == 0, (name, smem, err)
            self.funcs[key] = f
        return f

    def attrs(self, name, smem=0):
        d = self.driver
        f = self.func(name, smem)
        out = {}
        for k in ("NUM_REGS", "LOCAL_SIZE_BYTES"):
            err, v = d.cuFuncGetAttribute(getattr(d.CUfunction_attribute, "CU_FUNC_ATTRIBUTE_" + k), f)
            out[k] = v
        return out

    def launch(self, name, grid, block, smem, params):
        d = self.driver
        f = self.func(name, smem)
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        assert int(err) == 0, (name, err)

    def max_active_clusters(self, name, smem, block, csize, grid):
        d = self.driver
        f = self.func(name, smem)
        cfg = d.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block, 1, 1
        cfg.sharedMemBytes = smem
        at = d.CUlaunchAttribute()
        at.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = csize, 1, 1
        cfg.attrs = [at]
        cfg.numAttrs = 1
        err, ncl = d.cuOccupancyMaxActiveClusters(f, cfg)
        return int(ncl) if int(err) == 0 else 0


def _kk__tmap2d(ptr, rows, cols, box_rows, box_cols=64):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(2), ptr,
        [d.cuuint64_t(cols), d.cuuint64_t(rows)], [d.cuuint64_t(cols * 2)], [d.cuuint32_t(box_cols), d.cuuint32_t(box_rows)],
        [d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


_kk_NCFG = 19


def _kk_norm(cfg):
    """(TM, BN, LAY, S, NS[, MC, RED, MAP, PERM, PF, PFX, EPI, PRO, PRI, LASTD, WPK, NT2, EPF, XLD])"""
    cfg = tuple(cfg)
    return cfg + (0,) * (_kk_NCFG - len(cfg))


def _kk_kname(layer, cfg):
    return f"kg_{layer}_" + "_".join(map(str, _kk_norm(cfg)))


def _kk_smem_bytes(cfg):
    TM, BN, LAY, S, NS, MC, RED, MAP, PERM, PF, PFX, EPI, PRO, PRI, LASTD, WPK, NT2, EPF, XLD = _kk_norm(cfg)
    WN = BN // 2 if LAY == 1 else BN
    return (2048 + NS * (TM * 128 + BN * 128) + (128 * (WN // 2) * 4 * (1 if RED == 0 else 2) if S > 1 and RED in (0, 2, 3) else 0)  # RED 4: receive in the ring
            + (TM * (BN * 2 + 16) if NT2 else 0))


def _kk_build(specs, defines="", wpb=None):
    wpb = wpb or {}
    """specs: list of (layer, n, k, cfg). -> (Module, {(layer, cfg): entry})"""
    wns = set()
    src_k = ""
    ents = {}
    for layer, n, k, cfg in specs:
        cfg = _kk_norm(cfg)
        TM, BN, LAY, S, NS, MC, RED, MAP, PERM, PF, PFX, EPI, PRO, PRI, LASTD, WPK, NT2, EPF, XLD = cfg
        WN = BN // 2 if LAY == 1 else BN
        wns.add(WN)
        nkb = k // 64 // S
        assert k % (64 * S) == 0 and n % BN == 0
        kn = _kk_kname(layer, cfg)
        MTC = 256 // TM                       # grid is built for 193 <= m <= 256 (same tile count as m = 256)
        if MC == 1:
            MTC += MTC & 1
        NHC = (n // BN // (2 if NT2 else 1)) * MTC
        CL = S if (S > 1 and RED == 0) else (2 if MC else 1)
        assert RED != 2 or (S == 2 and LAY in (0, 1))
        if "VG_NORED" in defines:
            CL = 2 if MC else 1
        if "VG_DUMMYCL" in defines:
            CL = 2
        cl = f"__cluster_dims__({CL}, 1, 1) " if CL > 1 else ""
        _wa = wpb.get((layer, cfg), 0)
        # 64-bit literal + a runtime zero ptxas cannot fold (%dynamic_smem_size >> 31): a folded 64-bit immediate added
        # to an address lost its high half in some instantiations (ptxas bug seen 2026-11-02)
        src_k += (f'struct WPB_{kn} {{ static __device__ __forceinline__ u64 get() {{ u64 r; '
                  f'asm volatile("{{ .reg .u32 z; .reg .u64 zz; mov.u32 z, %%dynamic_smem_size; shr.u32 z, z, 31; cvt.u64.u32 zz, z; '
                  f'mov.u64 %0, {_wa}; add.u64 %0, %0, zz; }}" : "=l"(r)); return r; }} }};\n')
        src_k += (f'extern "C" __global__ void {cl}__launch_bounds__(384, 1) {kn}(const __grid_constant__ P p) '
                  f'{{ vg_body<{TM}, {BN}, {LAY}, {S}, {NS}, {nkb}, {n}, {MC}, {RED}, {MAP}, {PERM}, {MTC}, {NHC}, {PF}, {PFX}, {EPI}, {PRO}, {PRI}, {LASTD}, {WPK}, {k // 64}, WPB_{kn}, {NT2}, {EPF}, {XLD}>(p); }}\n')
        ents[(layer, cfg)] = {"kn": kn, "n": n, "k": k, "cfg": cfg, "smem": _kk_smem_bytes(cfg)}
    src = defines + _kk_HEAD + "".join(_kk__wg_ss(n) for n in sorted(wns)) + _kk_BODY.replace("@WGSS_SPECS@", _kk__specs(wns)) + src_k + _kk_PROBE_SRC
    if os.environ.get("VG_DUMP"):
        open(os.environ["VG_DUMP"], "w").write(src)
        return None, ents
    mod = _kk_Module(src)
    return mod, ents


def _kk_pack_w(w, bn):
    """[N, K] bf16 -> [N / bn][K / 64][bn rows][8 x 16 B chunks, chunk c of row r stored at c ^ (r & 7)] (the TMA 128B
    swizzle image of every (column tile, k-block) stage, contiguous)."""
    n, k = w.shape
    t = w.reshape(n // bn, bn, k // 64, 8, 8).permute(0, 2, 1, 3, 4)
    r = torch.arange(bn, device=w.device)
    c = torch.arange(8, device=w.device)
    idx = (c[None, :] ^ (r[:, None] & 7))[None, None, :, :, None].expand(n // bn, k // 64, bn, 8, 8)
    return t.gather(3, idx).contiguous()


class _kk_VG:
    """Dev wrapper: weights dict + list of (layer, cfg)."""

    def __init__(self, weights, lcfgs, defines=""):
        specs = []
        packed, wpb = {}, {}
        for layer, cfg in lcfgs:
            w = weights[layer]
            cfg = _kk_norm(cfg)
            specs.append((layer, w.shape[0], w.shape[1], cfg))
            if cfg[15] > 0:
                key = (layer, cfg[1])
                if key not in packed:
                    packed[key] = _kk_pack_w(w.contiguous(), cfg[1])
                if cfg[15] == 2:
                    wpb[(layer, cfg)] = packed[key].data_ptr()
        self.packed = packed
        self.mod, self.ents = _kk_build(specs, defines, wpb)
        for (layer, cfg), e in self.ents.items():
            if cfg[15] > 0:
                e["wp"] = packed[(layer, cfg[1])]
        self.w = {}
        for (layer, cfg), e in self.ents.items():
            w = weights[layer].contiguous()
            self.w[layer] = w
            e["tmw"] = _kk__tmap2d(w.data_ptr(), e["n"], e["k"], cfg[1])
            assert e["smem"] <= _kk_SMEM_MAX, (layer, cfg, e["smem"])
            CL = cfg[3] if (cfg[3] > 1 and cfg[6] == 0) else (2 if cfg[5] else 1)
            if cfg[3] > 1 and cfg[6] in (1, 2, 3, 4):
                WN = cfg[1] // 2 if cfg[2] == 1 else cfg[1]
                ntile = (e["n"] // cfg[1]) * (256 // cfg[0])
                e["ws"] = torch.full((ntile * 2 * 128 * WN // 2,), -1, dtype=torch.int32, device=w.device)
                e["flags"] = torch.zeros(ntile, dtype=torch.int32, device=w.device)
            if CL > 1 and "VG_NORED" not in defines:
                ncta = (e["n"] // cfg[1]) * (256 // cfg[0]) * cfg[3]
                e["clusters"] = self.mod.max_active_clusters(e["kn"], e["smem"], 384, CL, ncta)
        for (layer, cfg), e in self.ents.items():
            if cfg[8]:
                e["perm"] = self.gpc_perm(e)
        self.xm = {}
        self.trace = None

    def _probe(self, name, grid, smem, cluster=0):
        out = torch.full((grid,), -1, dtype=torch.int32, device="cuda")
        args = struct.pack("<Q", out.data_ptr())
        if cluster:
            import ctypes
            d = self.mod.driver
            f = self.mod.func(name, smem)
            if cluster > 8:
                d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1)
            cfg = d.CUlaunchConfig()
            cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
            cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = 128, 1, 1
            cfg.sharedMemBytes = smem
            cfg.hStream = d.CUstream(torch.cuda.current_stream().cuda_stream)
            at = d.CUlaunchAttribute()
            at.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = cluster, 1, 1
            cfg.attrs = [at]
            cfg.numAttrs = 1
            raw = ctypes.create_string_buffer(args)
            argv = (ctypes.c_void_p * 1)(ctypes.addressof(raw))
            (err,) = d.cuLaunchKernelEx(cfg, f, ctypes.addressof(argv), 0)
            if int(err) != 0:
                return None
        else:
            self.mod.launch(name, grid, 384 if name != "vg_probex" else 128, smem, args)
        torch.cuda.synchronize()
        return out.tolist()

    def gpc_groups(self):
        """SM -> locality group (GPC: clusters of 16 when the GPU allows them, else clusters of 8)."""
        if getattr(self, "_grp", None) is not None:
            return self._grp
        nsm = torch.cuda.get_device_properties(0).multi_processor_count
        grp = {}
        for cs in (16, 8):
            n = (nsm // cs) * cs
            o = self._probe("vg_probex", n, 120 * 1024, cluster=cs)
            if o is None or min(o) < 0:
                continue
            for i in range(0, n, cs):
                key = min(o[i:i + cs])
                for sm in o[i:i + cs]:
                    grp.setdefault(sm, ("c%d" % cs, key))
        for sm in range(nsm):
            grp.setdefault(sm, ("s", sm))
        self._grp = grp
        return grp

    def gpc_perm(self, e):
        """Physical unit -> logical unit so that the units each GPC actually runs get consecutive logical tiles
        (mt fastest: compact tile blocks per GPC). Helpers / finishers (RED >= 2) are permuted within their class."""
        TM, BN, LAY, S, NS, MC, RED, MAP, PERM, PF, PFX, EPI, PRO, PRI, LASTD, WPK, NT2, EPF, XLD = e["cfg"]
        CL = S if (S > 1 and RED == 0) else (2 if MC else 1)
        ncta = (e["n"] // BN) * (256 // TM) * S
        units = ncta // CL
        if units > 136:
            return None
        o = self._probe("vg_probe2" if CL == 2 else "vg_probe1", ncta, e["smem"])
        if o is None or min(o) < 0 or (CL not in (1, 2)):
            return None
        grp = self.gpc_groups()
        first = {}
        for u in range(units):
            first.setdefault(grp[o[u * CL]], u)
        cls = (lambda u: u // (units // 2)) if (RED >= 2 and CL == 1) else (lambda u: 0)
        perm = [0] * units
        for c in sorted(set(cls(u) for u in range(units))):
            us = [u for u in range(units) if cls(u) == c]
            order = sorted(us, key=lambda u: (first[grp[o[u * CL]]], u))
            for i, u in enumerate(order):
                perm[u] = us[i]
        return perm

    def run(self, x, layer, cfg, out=None):
        e = self.ents[(layer, _kk_norm(cfg))]
        TM, BN, LAY, S, NS, MC, RED, MAP, PERM, PF, PFX, EPI, PRO, PRI, LASTD, WPK, NT2, EPF, XLD = e["cfg"]
        m, k = x.shape
        key = (x.data_ptr(), m, k, TM)
        tmx = self.xm.get(key)
        if tmx is None:
            tmx = self.xm[key] = _kk__tmap2d(x.data_ptr(), m, k, TM)
        if out is None:
            out = torch.empty((m, e["n"]), dtype=torch.bfloat16, device=x.device)
        tmo = _kk__tmap2d(out.data_ptr(), m, e["n"], 64, 64) if EPI else bytes(128)
        assert 0 < m <= 256, m
        mt = 256 // TM                        # compile-time grid (MTC); token tiles past m load zeros, stores are masked
        if MC == 1:
            mt += mt & 1
        ncta = (e["n"] // BN // (2 if NT2 else 1)) * mt * S
        ws = e.get("ws")
        fl = e.get("flags")
        perm = e.get("perm")
        CL = S if (S > 1 and RED == 0) else (2 if MC else 1)
        if perm is not None and len(perm) * CL != (e["n"] // BN) * mt * S:
            perm = None                                    # the permutation was measured for the m = 256 grid
        args = tmx + e["tmw"] + tmo + struct.pack("<QQQQQQiiii", out.data_ptr(), self.trace.data_ptr() if self.trace is not None else 0,
                                            ws.data_ptr() if ws is not None else 0, fl.data_ptr() if fl is not None else 0,
                                            e["wp"].data_ptr() if "wp" in e else 0, x.data_ptr(),
                                            m, mt, (e["n"] // BN) * mt, 1 if perm else 0)
        args += struct.pack("<136H", *((perm or []) + [0] * (136 - len(perm or []))))
        self.mod.launch(e["kn"], ncta, 384, e["smem"], args)
        return out


# ============================================================================================ deliverable API (agent K)
# m in [193, 256] (verify-32 load) for o / pooler / unpooler; qkv_gate stays with agent V's g256 (bo6).
# cfg = (TM, BN, LAY, S, NS, MC, RED, MAP, PERM, PF, PFX, EPI, PRO, PRI, LASTD, WPK, NT2), see the kg docstring:
#   o, unpooler: 128 x 128 tiles (warpgroup g: tokens 64g.. x 128 cols), 6 stages, fast prologue, tail priority 3
#             (solo phase issued back to back), W prepacked into contiguous 16 KB swizzled stage images (1D bulk
#             copies, baked address), direct register -> global epilogue
#   pooler:   64 x 128 tiles (warpgroup g: 64 tokens x cols 64g..), 9 stages, fast prologue, packed W, direct epilogue
_kk_G256K_CFG = {
    "o": (128, 128, 0, 1, 6, 0, 0, 0, 0, 0, 0, 2, 1, 0, 3, 2),
    "unpooler": (128, 128, 0, 1, 6, 0, 0, 0, 0, 0, 0, 2, 1, 0, 3, 2),
    "pooler": (64, 128, 1, 1, 9, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 2),
}
_kk_G256K_M = (193, 256)


def _kk_prepare_g256k(weights, cfgs=None):
    """-> state; state["ok"] False (route disabled, run_g256k returns None) on any failure."""
    import torch.nn.functional as _F
    st = {"ok": False, "cfg": {}}
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            return st
        if torch.cuda.get_device_properties(0).multi_processor_count < 128:
            return st
        lc = [(l, _kk_norm(c)) for l, c in (cfgs or _kk_G256K_CFG).items() if l in weights]
        if not lc:
            return st
        V = _kk_VG(weights, lc)
        g = torch.Generator(device="cuda").manual_seed(11)
        for layer, cfg in lc:
            W = weights[layer]
            for m in (256, 200):
                x = torch.randn(m, W.shape[1], device="cuda", generator=g).to(torch.bfloat16)
                x[3] = 0
                x[3, 5] = 8.0                          # one-hot probe row: must reproduce 8 * W[:, 5] exactly
                out = V.run(x, layer, cfg)
                ref = _F.linear(x, W)
                torch.cuda.synchronize()
                err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                if not err < 5e-3 or not torch.equal(out[3], (8.0 * W[:, 5].float()).to(torch.bfloat16)):
                    raise RuntimeError(f"g256k selftest {layer} m={m}: rel err {err}")
            st["cfg"][layer] = cfg
        st["V"] = V
        st["nk"] = {l: tuple(weights[l].shape) for l in st["cfg"]}
        st["ok"] = True
    except Exception as e:  # noqa: BLE001 - any failure: the base handles every shape
        import sys
        print(f"[g256k] disabled: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        st = {"ok": False, "cfg": {}}
    return st


def _kk_run_g256k(x, config, st):
    """out [m, n] bf16 for o / pooler / unpooler at 193 <= m <= 256, else None (the base handles it)."""
    if not st.get("ok"):
        return None
    layer, m = config["layer"], config["m"]
    cfg = st["cfg"].get(layer)
    if cfg is None or not (_kk_G256K_M[0] <= m <= _kk_G256K_M[1]):
        return None
    n, k = st["nk"][layer]
    if (config["n"] != n or x.dim() != 2 or x.shape[0] != m or x.shape[1] != k or not x.is_contiguous()
            or x.data_ptr() % 16 or x.dtype != torch.bfloat16 or not x.is_cuda):
        return None
    return st["V"].run(x, layer, cfg)

# ======================================================================== dispatch
def prepare(weights):
    st = {'b7': _b7_prepare(weights)}
    st['N'] = _ns_prepare_small(weights)
    st['R'] = _r3_prepare_router3(weights)
    st['G'] = _g1_prepare_gemv1(weights)
    st['Q2'] = _q2_prepare_g128q(weights)
    st['Q3'] = _q3_prepare_g256q(weights)
    st['P5'] = _p5_prepare_pu5(weights)
    st['H'] = _hh_prepare_skh(weights)
    st['V2'] = _q8_prepare_qkvb(weights)
    st['F'] = _f1_prepare_sk11(weights)
    st['X'] = _x3_prepare_sk3(weights)
    st['R4'] = _r4_prepare_rt4(weights)
    st['Y'] = _yr_prepare_rt3(weights)
    st['K'] = _kk_prepare_g256k(weights)
    torch.cuda.synchronize()
    return st


def kernel_fn(data, state):
    x, cfg = data['x'], data['config']
    layer, m = cfg['layer'], cfg['m']
    if layer == 'o' and m in (1, 8):
        _o = _ns_run_small(x, cfg, state['N'])
        if _o is not None:
            return _o
    if layer == 'router' and m == 1:
        _o = _r3_run_router3(x, cfg, state['R'])
        if _o is not None:
            return _o
    if layer in ('qkv_gate', 'pooler', 'o') and m == 1 or (layer == 'unpooler' and m <= 8):
        _o = _g1_run_gemv1(x, cfg, state['G'])
        if _o is not None:
            return _o
    if layer in ('qkv_gate', 'o', 'pooler', 'unpooler') and 97 <= m <= 128:
        _o = _q2_run_g128q(x, cfg, state['Q2'])
        if _o is not None:
            return _o
    if layer == 'qkv_gate' and 240 <= m <= 256:
        _o = _q3_run_g256q(x, cfg, state['Q3'])
        if _o is not None:
            return _o
    if layer in ('pooler', 'unpooler') and 9 <= m <= 32:
        _o = _p5_run_pu5(x, cfg, state['P5'])
        if _o is not None:
            return _o
    if layer == 'unpooler' and m == 32:
        _o = _hh_run_skh(x, cfg, state['H'])
        if _o is not None:
            return _o
    if layer == 'qkv_gate' and m in (8, 16):
        _o = _q8_run_qkvb(x, cfg, state['V2'])
        if _o is not None:
            return _o
    if (layer in ('qkv_gate', 'pooler') and 2 <= m <= 32) or (layer == 'o' and 17 <= m <= 32):
        _o = _f1_run_sk11(x, cfg, state['F'])
        if _o is not None:
            return _o
    if layer != 'router' and 2 <= m <= 32 and not (layer == 'unpooler' and m <= 8):
        _o = _x3_run_sk3(x, cfg, state['X'])
        if _o is not None:
            return _o
    if layer == 'router' and (97 <= m <= 128 or 193 <= m <= 256):
        _o = _r4_run_rt4(x, cfg, state['R4'])
        if _o is not None:
            return _o
    if layer == 'router':
        _o = _yr_run_rt3(x, cfg, state['Y'])
        if _o is not None:
            return _o
    if layer in ('o', 'pooler', 'unpooler') and 193 <= m <= 256:
        _o = _kk_run_g256k(x, cfg, state['K'])
        if _o is not None:
            return _o
    return _b7_kernel_fn(data, state['b7'])
